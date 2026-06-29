#!/usr/bin/env python3
"""Stage-0/1 from-scratch REINFORCE on the policy head (no value head, no search).

Plan: ~/.claude/plans/functional-seeking-iverson.md

The smallest test of "can a reward move this policy at all?":
  - fresh `MyModel` (NO checkpoint loaded -- start from scratch, no inherited bias);
  - act = one forward pass, SAMPLE softmax(policy/tau) over single-select policy
    nodes; `auto_resolve` everything else (exactly eval_mymodel.greedy_agent's split);
  - opponent = a FIXED `random_agent` on the other seat (Lucario deck.csv both sides,
    so it is *our policy* vs *random*, an asymmetric/stationary task);
  - reward R = prize differential (opp_remaining - our_remaining)/6 at game end
    (the same quantity tcg_model._rollout_value uses; denser than binary win/loss);
  - loss = -(R - baseline)*logpi(a) - beta*H(pi)  [REINFORCE + MC baseline + entropy];
  - the value head/loss is DROPPED -- the value scalar is ignored, so it never trains
    (the encoder trunk still learns via the decoder's cross-attention).

Stage 0 (verify-only smoke -- runs end-to-end, finite loss, ~50% greedy vs random):
    python train_reinforce.py --games 12 --iterations 1 --eval-games 30

Stage 1 (tiny mechanism test):
    python train_reinforce.py --games 300 --iterations 4 --eval-games 200 --out-prefix rl

Official go/no-go (greedy, deterministic) is eval_mymodel.py on the saved checkpoint:
    TORCHDYNAMO_DISABLE=1 python eval_mymodel.py --ckpt out/rl3.pth \
        --agents greedy --opponents random --games 200
"""

# === RISK NOTES (this script empirically REGRESSED play; kept as PPO prior-art) ===
# Both tiny runs moved the weights but toward WORSE play (memory:
# "reinforce-stage1-regresses"). The plumbing is correct; the *method* is the risk:
#   1. GAMEABLE REWARD: prize-diff (opp-our)/6 is a Goodhart proxy. vs teacher (beta=0)
#      it collapsed to a degenerate short-game prize-race -- proxy meanR rose
#      (-0.49 -> -0.26) while greedy-vs-teacher win% fell 16% -> 0% and move-agreement
#      fell BELOW the random floor. Prefer a terminal +1/-1/0 reward.
#   2. ENTROPY BRACKET: plain REINFORCE sits between FLATTEN (beta too high: when
#      advantages ~0 the entropy bonus dominates and pushes pi toward uniform -- vs
#      random it went greedy 83% -> 62%) and COLLAPSE (beta=0 + gameable reward). No
#      stable middle without a trust region (NASHPG's KL-to-reference; see plan.md).
#   3. DEAD SIGNAL vs random: fresh-init greedy already beats random_agent ~83%, so the
#      reward is saturated (meanR ~ 0) and there is ~nothing to learn. Use the rule-based
#      teacher / a frozen past-self as the live signal, not vs-random.
#   4. FLAT CREDIT + SCALAR BASELINE: every decision in a game shares the one terminal
#      return minus a single global mean -- weak, high-variance credit assignment.
#   5. COVERAGE: only single-select policy nodes are learned; multi-select / NUMBER
#      nodes are auto_resolved and never trained.
# PPO-PORTING TRAPS if reusing this skeleton (plan.md risk register R2/R4):
#   - act_sample stores neither old_logp NOR V_old -- PPO must capture both at sampling.
#   - tau!=1 is self-consistent for REINFORCE here, but a PPO port must sample at tau=1
#     (on-policy) or the importance ratio is wrong.

import argparse
import os
import sys
import random

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

import tcg_model  # noqa: E402
from tcg_model import (  # noqa: E402
    MyModel, random_agent, get_encoder_input, get_decoder_input, eval_nn,
    LearnInput, progress,
)
# Reuse the eval split (single-select policy node vs auto-resolved) and the
# official greedy-vs-random measurement -- no new eval code needed.
from eval_mymodel import is_policy_select, auto_resolve, greedy_agent, play  # noqa: E402
from eval import wilson  # noqa: E402
import rulebased_lucario  # noqa: E402  (the Lucario teacher, a fixed asymmetric opponent)
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def act_sample(obs_dict, model, deck, tau):
    """Our actor during data collection: SAMPLE on single-select policy nodes,
    auto-resolve everything else. Returns (selected_indices, record_or_None) where
    record = (sv_enc, sv_dec, n_opt, sampled_index) for learned decisions only."""
    o = to_observation_class(obs_dict)
    sel = o.select
    if not is_policy_select(sel):
        return auto_resolve(sel), None
    n = len(sel.option)
    sv_enc = get_encoder_input(o, deck)
    sv_dec = get_decoder_input(o, [[i] for i in range(n)])
    # RISK: value head intentionally unused -> never calibrates; a PPO port MUST capture
    # V_old here AND the chosen action's old_logp (this skeleton stores neither).
    _value, policy = eval_nn(sv_enc, sv_dec, model)  # value ignored -- policy only
    probs = torch.softmax(torch.tensor(policy) / tau, dim=0)
    a = int(torch.multinomial(probs, 1).item())
    return [a], (sv_enc, sv_dec, n, a)


def collect_game(deck, model, tau, seat, opp_fn):
    """Play one battle: our sampling actor on `seat`, fixed opponent on the other.
    Returns (records, R, won, drew)."""
    # RISK: the cg battle is a PROCESS-GLOBAL singleton and there is no try/finally --
    # an exception between here and battle_finish() leaks the live battle and corrupts
    # every later game in this process. Wrap in try/finally to harden.
    obs, start = battle_start(deck, deck)
    if start.errorPlayer >= 0:
        raise ValueError("deck rejected by engine")
    recs = []
    while obs["current"]["result"] < 0:
        if obs["current"]["yourIndex"] == seat:
            sel, rec = act_sample(obs, model, deck, tau)
            if rec is not None:
                recs.append(rec)
            obs = battle_select(sel)
        else:
            obs = battle_select(opp_fn(obs))
    battle_finish()
    cur = obs["current"]
    our = len(cur["players"][seat]["prize"])
    opp = len(cur["players"][1 - seat]["prize"])
    # RISK: prize-diff is a GAMEABLE proxy (Goodhart) -- it can climb while win% falls
    # (RISK NOTES #1). A terminal +1/-1/0 reward is the safer target.
    R = (opp - our) / 6.0  # prize differential from our seat's view; win => positive
    return recs, R, cur["result"] == seat, cur["result"] == 2


def train_iter(records, baseline, model, opt, tau, beta, batch_size, device):
    """One pass of minibatched REINFORCE over the collected decisions.
    Each record = (sv_enc, sv_dec, n_opt, sampled_index, R). The decoder needs a
    uniform action-bag width per batch, so we pad to W = max(n_opt) with EMPTY bags
    (zero embedding) and mask those columns out of the softmax."""
    random.shuffle(records)
    model.train()
    total_loss, nb = 0.0, 0
    for s in range(0, len(records), batch_size):
        mb = records[s:s + batch_size]
        B = len(mb)
        W = max(r[2] for r in mb)
        enc, dec = LearnInput(), LearnInput()
        for (sv_enc, sv_dec, n, _a, _R) in mb:
            enc.add(sv_enc)
            dec.add(sv_dec)                       # n real action-bags
            for _ in range(W - n):                # pad to W with empty (zero) bags
                dec.offset.append(len(dec.index))
        _v, p = model(
            torch.tensor(enc.index, dtype=torch.int32, device=device),
            torch.tensor(enc.value, dtype=torch.float32, device=device),
            torch.tensor(enc.offset, dtype=torch.int32, device=device),
            torch.tensor(dec.index, dtype=torch.int32, device=device),
            torch.tensor(dec.value, dtype=torch.float32, device=device),
            torch.tensor(dec.offset, dtype=torch.int32, device=device))
        # p: [B, W]. Mask padded columns, then softmax over the real options.
        # Use a large FINITE negative (not -inf): -inf into log_softmax NaNs the
        # backward pass (0*inf). tanh keeps p in [-1,1], so -1e9 is decisively masking.
        n_opts = torch.tensor([r[2] for r in mb], device=device)
        valid = torch.arange(W, device=device).unsqueeze(0) < n_opts.unsqueeze(1)
        logits = (p / tau).masked_fill(~valid, -1e9)
        logp = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        acts = torch.tensor([r[3] for r in mb], device=device)
        chosen_logp = logp[torch.arange(B, device=device), acts]
        ent = -(probs * logp).masked_fill(~valid, 0.0).sum(dim=1)  # avoid 0*-inf=nan
        adv = torch.tensor([r[4] - baseline for r in mb], dtype=torch.float32, device=device)
        loss = -(adv * chosen_logp).mean() - beta * ent.mean()
        opt.zero_grad()
        loss.backward()
        # RISK: no grad clipping -- REINFORCE grads (adv*logp) can spike; consider
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5).
        opt.step()
        total_loss += float(loss.item())
        nb += 1
    return total_loss / max(nb, 1)


def eval_greedy(model, deck, n_games, seed, opp_fn):
    """Official go/no-go signal: deterministic greedy policy vs `opp_fn`, Wilson CI.
    Saves/restores the global RNG so it can't perturb training collection."""
    st = random.getstate()
    model.eval()
    with torch.inference_mode():
        w, l, d = play(lambda obs: greedy_agent(obs, deck, model), opp_fn, deck, n_games, seed)
    random.setstate(st)
    p, lo, hi = wilson(w, w + l)
    return f"{w}/{l}/{d}  win%={100 * p:.1f} (95% CI {100 * lo:.1f}-{100 * hi:.1f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200, help="self-play games per iteration")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.5, help="sampling temperature for softmax(policy/tau)")
    ap.add_argument("--beta", type=float, default=0.01, help="entropy bonus coefficient")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-games", type=int, default=0, help="inline greedy-vs-opponent eval per iter (0=off)")
    # RISK: opponent=random is a SATURATED signal (fresh greedy already ~83%); 'teacher'
    # gives a live reward (RISK NOTES #3).
    ap.add_argument("--opponent", default="random", choices=["random", "teacher"],
                    help="fixed opponent for collection AND eval (random_agent or the Lucario teacher)")
    ap.add_argument("--out-prefix", default="rl", help="checkpoints -> out/<prefix><iter>.pth")
    ap.add_argument("--deck", default=os.path.join(_REPO_ROOT, "deck.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", default=None, help="optional warm-start checkpoint (default: from scratch)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    deck = [int(x) for x in open(args.deck).read().split()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opp_fn = random_agent if args.opponent == "random" else rulebased_lucario.agent
    model = MyModel(128, 2, 256, 1, 1).to(device)  # FROM SCRATCH unless --load given
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print(f"warm-started from {args.load}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(os.path.join(_REPO_ROOT, "out"), exist_ok=True)
    print(f"from-scratch REINFORCE | deck={os.path.basename(args.deck)} | vs {args.opponent} | "
          f"games/iter={args.games} iters={args.iterations} tau={args.tau} beta={args.beta} "
          f"lr={args.lr} device={device.type}", flush=True)

    # Pre-training baseline for a clean A/B: the SAME fresh-init net before any update.
    init_path = os.path.join(_REPO_ROOT, "out", f"{args.out_prefix}_init.pth")
    torch.save(model.state_dict(), init_path)
    if args.eval_games:
        print(f"  [baseline] init greedy vs {args.opponent}: "
              f"{eval_greedy(model, deck, args.eval_games, args.seed, opp_fn)} -> {init_path}", flush=True)

    for it in range(args.iterations):
        model.eval()
        records, Rs, wins, draws = [], [], 0, 0
        with torch.no_grad():
            for g in progress(args.games, f"iter {it} collect"):
                recs, R, won, drew = collect_game(deck, model, args.tau, g % 2, opp_fn)
                for rec in recs:
                    records.append(rec + (R,))
                Rs.append(R)
                wins += won
                draws += drew
        # RISK: one GLOBAL scalar baseline + the same terminal R on every decision =
        # flat, high-variance credit assignment vs a state-dependent V(s) critic (#4).
        baseline = sum(r[4] for r in records) / max(len(records), 1)
        loss = train_iter(records, baseline, model, opt, args.tau, args.beta, args.batch_size, device)

        path = os.path.join(_REPO_ROOT, "out", f"{args.out_prefix}{it}.pth")
        torch.save(model.state_dict(), path)
        line = (f"iter {it}: games={args.games} decisions={len(records)} "
                f"meanR={sum(Rs) / len(Rs):+.3f} sampling_win%={100 * wins / args.games:4.1f} "
                f"baseline={baseline:+.3f} loss={loss:.4f} -> {path}")

        if args.eval_games:
            line += (f"\n  [eval] greedy vs {args.opponent}: "
                     f"{eval_greedy(model, deck, args.eval_games, args.seed, opp_fn)}")
        print(line, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
