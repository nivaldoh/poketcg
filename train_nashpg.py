#!/usr/bin/env python3
"""NASHPG self-play for MyModel (model-free PPO + a KL anchor to a frozen, periodically
reset reference policy rho). Implements plan.md; supersedes train_reinforce.py.

The mechanism (two knobs over plain PPO):
  1. add `alpha * KL(pi_theta || rho)` to the loss over visited info sets (a trust region);
  2. every OUTER iteration, reset `rho <- theta` (chase the actual, unexploitable equilibrium).
Plain self-play PG cycles/collapses (our `reinforce-stage1-regresses` memo); the KL-to-rho
+ reset *is* the trust region that memo said was missing.

v1 scope (locked): single-select policy nodes only (multi-select/NUMBER stay heuristic via
`auto_resolve`); from-scratch `MyModel(128,2,256,1,1)`; `sample_deck` mirror; terminal
+1/-1/0 reward only (prize-diff is Goodhart-gameable -- confirmed regressed in train_reinforce).
Goal = validate the learning mechanism locally on a tiny budget, NOT win Kaggle.

Smoke (plumbing -- the smallest verifiable signal):
    python train_nashpg.py --outer 2 --inner 2 --games 4 --eval-games 20
  => every game reaches result>=0, every update prints a FINITE loss + diagnostics,
     win-rates print with Wilson CIs. (Prefix TORCHDYNAMO_DISABLE=1 if dynamo trips.)

The bare-policy progress signal is win% vs a FROZEN past-self in the mirror with CI
separation -- NOT "beats random" (saturated ~83% even untrained) and NOT MCTS-on-top
(eval.make_model confounds policy vs search). See R11.
"""

# === RISK REGISTER (encoded inline; full text in plan.md) ===
# R1 (conceptual, BINDING constraint): the sample_deck mirror is symmetric, so the expected
#    outcome is 50/50 and returns are +-1 with HIGH variance. The value baseline only cuts
#    that variance once V predicts mid-game advantage -- which it learns slowly from the same
#    noisy +-1 returns (chicken-and-egg, worse from-scratch). At locally-feasible G the
#    gradient SNR may be too low to demonstrate improvement at all. The KL anchor fixes
#    flatten/collapse (the trust-region problem) but NOT this variance/credit problem.
#    => "loss moved" is necessary-not-sufficient; trust win% vs frozen past-self w/ CI gap.
# R10 (feasibility): T*K*G games on a Python rules engine + a serial PROCESS-GLOBAL battle
#    ~ hours at ~1 game/s. Measure games/sec (printed each outer iter) before sizing real runs.
import argparse
import json
import math
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

from tcg_model import (  # noqa: E402
    MyModel, get_encoder_input, get_decoder_input, eval_nn, LearnInput,
    sample_deck, progress,
)
# sampled_agent + LOGIT_SCALE are the SINGLE SOURCE of the no-search bare-policy agent and the
# logits ≡ decoder_out × 10 convention (shared with eval.py via --agent-kind sampled). Importing
# LOGIT_SCALE here keeps act-time / update-time / eval-time distributions identical (PPO ratio).
from eval_mymodel import is_policy_select, auto_resolve, sampled_agent, LOGIT_SCALE  # noqa: E402
from eval import Agent, play_match, wilson, make_rulebased  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def load_state(model, path, device):
    """Load a checkpoint's weights into `model` (shared by --load resume and the --br-vs target)."""
    model.load_state_dict(torch.load(path, map_location=device))


class Record:
    """One learned (single-select) decision. `sv_dec` is STORED so the act-time (batch=1)
    candidate set is byte-identical to the update-time (padded, batched) one -- safe because
    decoder candidates are mutually independent (R12: DecoderLayer cross-attends to the
    encoder only, no self-attention among candidate words; re-verify if that ever changes)."""
    __slots__ = ("sv_enc", "sv_dec", "n", "a", "old_logp", "V_old", "seat", "adv", "ret")

    def __init__(self, sv_enc, sv_dec, n, a, old_logp, V_old):
        self.sv_enc, self.sv_dec, self.n, self.a = sv_enc, sv_dec, n, a
        self.old_logp, self.V_old = old_logp, V_old   # captured AT sampling (R4)
        self.seat = self.adv = self.ret = None        # filled at game end


def act_collect(obs_dict, model, deck):
    """Sampling actor for collection. Returns (selected_indices, Record|None).
    Single-select policy nodes are sampled at tau=1 (on-policy -- R3); everything else is
    auto-resolved and produces no record (so it is never trained -- v1 coverage limit)."""
    o = to_observation_class(obs_dict)
    sel = o.select
    if not is_policy_select(sel):
        return auto_resolve(sel), None
    n = len(sel.option)
    sv_enc = get_encoder_input(o, deck)
    sv_dec = get_decoder_input(o, [[i] for i in range(n)])  # one word per option -> policy len == n
    value, policy = eval_nn(sv_enc, sv_dec, model)          # batch=1; no padding, so no masking here
    logits = torch.tensor(policy) * LOGIT_SCALE
    logp = torch.log_softmax(logits, dim=0)
    # R3: SAMPLE from pi at tau=1 (the policy being optimized). A sharpening temperature here
    # would make the importance ratio wrong (silently off-policy). Temperature is test-time only.
    a = int(torch.multinomial(torch.softmax(logits, dim=0), 1).item())
    # R4: old_logp and V_old must come from THIS masked forward, under the exact distribution
    # we sampled from -- recomputing later makes ratio≈1 always and PPO degenerates.
    return [a], Record(sv_enc, sv_dec, n, a, float(logp[a]), value)


def compute_gae(seat_recs, G, lam):
    """In-place GAE over ONE seat's own ordered decisions (gamma=1, reward 0 except a
    terminal G, V(terminal)=0). lam=1 telescopes to MC (adv = G - V_old, ret = G) -- the
    v1 smoke default; lam=0.95 is the real-run variance reducer.

    Each seat's records are a temporal subsequence, so we bootstrap each decision off the
    SAME seat's NEXT decision value -- the opponent's intervening moves are 'environment'.
    This is the original NashPG's per-current_player carry, made trivial by serial per-seat
    lists (no interleaved-env scan needed)."""
    next_v, adv = 0.0, 0.0
    for i in range(len(seat_recs) - 1, -1, -1):
        rec = seat_recs[i]
        r = G if i == len(seat_recs) - 1 else 0.0
        delta = r + next_v - rec.V_old           # gamma=1; V(terminal)=0 on the last step
        adv = delta + lam * adv                  # gamma*lam, gamma=1
        rec.adv, rec.ret = adv, adv + rec.V_old
        next_v = rec.V_old


def collect_game(model, deck, lam):
    """Play one self-play game: the SAME theta pilots both seats; both seats emit records.
    Returns the flat record list (adv/ret already filled). G is seat-relative."""
    # R7-note: the cg battle is a PROCESS-GLOBAL singleton -- a leaked live battle corrupts
    # every later game in this process, so battle_finish() MUST run even on error.
    obs, start = battle_start(deck, deck)
    if start.errorPlayer >= 0:
        raise ValueError("deck rejected by engine")
    recs = ([], [])
    try:
        while obs["current"]["result"] < 0:
            yi = obs["current"]["yourIndex"]
            sel, rec = act_collect(obs, model, deck)
            if rec is not None:
                rec.seat = yi
                recs[yi].append(rec)               # per-seat, in temporal order (GAE needs this)
            obs = battle_select(sel)
    finally:
        battle_finish()
    result = obs["current"]["result"]
    # CONCEPTUAL RISK (must stay correct): V is ACTING-SEAT-RELATIVE -- get_encoder_input
    # frames the state by yourIndex -- and G_seat is seat-relative too, so they share a frame
    # => NO sign-flip. The one obligation is pairing each record with ITS OWN seat's terminal
    # G, which the per-seat lists guarantee. (The MCTS value flip is a tree-sharing artifact
    # and does NOT apply to per-player self-play.)
    for seat in (0, 1):
        G = 0.0 if result == 2 else (1.0 if result == seat else -1.0)
        compute_gae(recs[seat], G, lam)
    return recs[0] + recs[1]


def collect_br_game(model, frozen, deck, lam, br_seat):
    """One best-response game (plan.md §6.4): the BR (`model`, seat `br_seat`) samples at tau=1 and
    emits Records; the FROZEN target plays the other seat via sampled_agent (tau=1, NO record). This
    single-seat collection is the reference's `only_use_player0_experience` -- we simply never build
    opponent records. Returns (br_records, G) with G the BR seat's terminal return.

    compute_gae over just this seat's ordered subsequence is correct (no new math, NO sign flip): V
    is acting-seat-relative and G is that-same-seat-relative; the opponent's moves are 'environment'
    -- the exact case collect_game/compute_gae already document."""
    obs, start = battle_start(deck, deck)
    if start.errorPlayer >= 0:
        raise ValueError("deck rejected by engine")
    br_recs = []
    try:
        while obs["current"]["result"] < 0:
            yi = obs["current"]["yourIndex"]
            if yi == br_seat:
                sel, rec = act_collect(obs, model, deck)        # BR: sample tau=1, emit record (R3/R4)
                if rec is not None:
                    rec.seat = yi
                    br_recs.append(rec)
            else:
                sel = sampled_agent(obs, deck, frozen)          # frozen target: tau=1, NO record
            obs = battle_select(sel)
    finally:
        battle_finish()                                         # singleton: must run even on error
    result = obs["current"]["result"]
    G = 0.0 if result == 2 else (1.0 if result == br_seat else -1.0)
    compute_gae(br_recs, G, lam)
    return br_recs, G


def _build_minibatch(mb, device):
    """Pad each item's decoder to W=max(n) candidate bags (EMPTY/zero bags for padding) and
    build the EmbeddingBag tensors + a [B,W] validity mask. PER-ITEM padding is mandatory:
    forward does p.reshape(B,-1,d_model), which assumes a uniform candidate count -- appending
    all reals then all pads would scramble candidates across items (R5)."""
    W = max(r.n for r in mb)
    enc, dec = LearnInput(), LearnInput()
    for r in mb:
        enc.add(r.sv_enc)
        dec.add(r.sv_dec)                          # r.n real candidate bags
        for _ in range(W - r.n):
            dec.offset.append(len(dec.index))      # empty (zero-embedding) padding bag
    t = lambda lst, dt: torch.tensor(lst, dtype=dt, device=device)
    tensors = (t(enc.index, torch.int32), t(enc.value, torch.float32), t(enc.offset, torch.int32),
               t(dec.index, torch.int32), t(dec.value, torch.float32), t(dec.offset, torch.int32))
    n_opts = torch.tensor([r.n for r in mb], device=device)
    valid = torch.arange(W, device=device).unsqueeze(0) < n_opts.unsqueeze(1)  # [B,W]
    return tensors, valid


def ppo_update(buffer, model, ref, opt, args, device):
    """E epochs of masked-minibatch PPO with a forward-KL anchor to the frozen rho (`ref`).
    Returns an averaged-metrics dict for logging (or None if there is nothing to learn)."""
    if not buffer:
        return None
    model.train()
    acc, nb = {}, 0
    for _ in range(args.epochs):
        random.shuffle(buffer)
        for s in range(0, len(buffer), args.batch_size):
            mb = buffer[s:s + args.batch_size]
            B = len(mb)
            tensors, valid = _build_minibatch(mb, device)
            value, p = model(*tensors)
            with torch.no_grad():                          # rho: frozen, never in the optimizer (R13)
                _, p_ref = ref(*tensors)
            logits = (p * LOGIT_SCALE).masked_fill(~valid, -1e9)     # finite, NOT -inf (R6)
            logp = torch.log_softmax(logits, dim=1)
            pi = torch.softmax(logits, dim=1)
            logp_ref = torch.log_softmax((p_ref * LOGIT_SCALE).masked_fill(~valid, -1e9), dim=1)

            idx = torch.arange(B, device=device)
            acts = torch.tensor([r.a for r in mb], device=device)
            old_logp = torch.tensor([r.old_logp for r in mb], device=device)
            ret = torch.tensor([r.ret for r in mb], dtype=torch.float32, device=device)
            adv = torch.tensor([r.adv for r in mb], dtype=torch.float32, device=device)
            # R9 per-minibatch normalize. unbiased=False is load-bearing: the last minibatch can
            # have B==1 (when len(buffer) % batch_size == 1), and unbiased std (dof=n-1=0) returns
            # NaN there -> NaN loss. Population std gives 0 for B==1 -> adv=0 (harmless), no NaN.
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            logp_a = logp[idx, acts]
            ratio = torch.exp(logp_a - old_logp)                    # ≈1 at the first epoch (R4/R12 check)
            L_clip = -torch.min(ratio * adv,
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
            v = value.squeeze(-1)
            if args.clip_vloss:
                # PPO-clipped critic guards the SHARED encoder trunk the policy depends on (R7).
                V_old = torch.tensor([r.V_old for r in mb], dtype=torch.float32, device=device)
                v_clip = V_old + torch.clamp(v - V_old, -args.clip, args.clip)
                L_v = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
            else:
                L_v = ((v - ret) ** 2).mean()                       # v1 smoke: plain MSE
            # masked_fill(~valid, 0) avoids any 0·(-1e9) term polluting the entropy/KL sums (R6).
            H = -(pi * logp).masked_fill(~valid, 0.0).sum(1).mean()
            KL = (pi * (logp - logp_ref)).masked_fill(~valid, 0.0).sum(1).mean()  # forward KL pi||rho
            loss = L_clip + args.vf_coef * L_v - args.ent_coef * H + args.alpha * KL

            opt.zero_grad()
            loss.backward()
            # R7: keep grads bounded into the shared trunk (also the original NashPG's clip).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            assert math.isfinite(loss.item()), "non-finite loss -- smoke contract violated"

            with torch.no_grad():
                # PPO health diagnostics (R3/R4/R1): clip_frac>0 => the clip engages,
                # explained_var rising => the critic is learning, mag_kl>0 => the anchor bites.
                # .item() (not float()) -- L_clip/L_v/H/KL/loss still carry requires_grad here.
                log_ratio = logp_a - old_logp
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()   # Schulman k3
                clip_frac = (ratio.sub(1).abs() > args.clip).float().mean()
                # unbiased=False for the same B==1 reason as the adv normalize: unbiased var
                # (dof=0) is NaN for a singleton minibatch and would poison this update's logged
                # average. Logging-only (inside no_grad) -> training is unaffected either way.
                expl_var = 1.0 - (ret - v).var(unbiased=False) / (ret.var(unbiased=False) + 1e-8)
                step = dict(ppo_loss=L_clip.item(), critic_loss=L_v.item(), entropy=H.item(),
                            mag_kl=KL.item(), approx_kl=approx_kl.item(), clip_frac=clip_frac.item(),
                            explained_var=expl_var.item(), loss=loss.item())
            for k, val in step.items():
                acc[k] = acc.get(k, 0.0) + val
            nb += 1
    return {k: val / nb for k, val in acc.items()} | {"n_decisions": len(buffer)}


def sampled_agent_fn(model, deck):
    """Bind the shared no-search bare-policy agent (eval_mymodel.sampled_agent) to (model, deck)
    -> an obs->indices callable (R11: sampled not argmax, no MCTS confound). Same source as
    eval.py's --agent-kind sampled, so the trainer's eval and the end SPRT use one policy."""
    return lambda obs_dict: sampled_agent(obs_dict, deck, model)


def eval_checkpoint(model, init_model, deck, n_games, tag):
    """Mirror vs FROZEN past-self (init) isolates skill (same deck both sides); vs the rule-based
    teacher is a DECK-CONFOUNDED but live signal. Reports W/L/D + Wilson CIs. Snapshots/restores
    RNG so eval can't perturb training collection."""
    if n_games <= 0:
        return
    rng_py, rng_t = random.getstate(), torch.get_rng_state()
    model.eval()
    cur = Agent(f"nashpg:{tag}", sampled_agent_fn(model, deck), deck)
    with torch.inference_mode():
        for opp in (Agent("init", sampled_agent_fn(init_model, deck), deck), make_rulebased()):
            w, l, d = play_match(cur, opp, n_games)
            p, lo, hi = wilson(w, w + l)
            print(f"  [eval {tag}] vs {opp.name:9s}: {w}/{l}/{d}  "
                  f"win%={100 * p:.1f} (95% CI {100 * lo:.1f}-{100 * hi:.1f})", flush=True)
    random.setstate(rng_py)
    torch.set_rng_state(rng_t)


def run_br(args, device, deck, out_dir):
    """BR-exploitability (plan.md §6.4): train a FRESH best-response vs a FROZEN target with plain
    PPO (alpha=0, NO rho reset, single-seat learning). The BR's average return vs the target is a
    LOWER BOUND on the target's exploitability -- a weak/under-trained BR under-estimates, so read
    the br_return curve as a lower bound. Symmetric sample_deck mirror with terminal +-1/0 => game
    value 0: a Nash target => best BR return ~ 0 (win% ~ 50); a fully exploitable target => -> +1.
    The headline number is a high-n seat-swapped BR-vs-target win% + Wilson CI (no CRN: the engine
    is unseeded, so compare targets via non-overlapping CIs, never point estimates)."""
    args.alpha = 0.0                                          # FORCE plain PPO -- the anchor must NOT bias a BR
    assert args.alpha == 0.0, "BR mode requires alpha=0 (no KL anchor)"
    model = MyModel(128, 2, 256, 1, 1).to(device)            # fresh BR (or --load to resume on Kaggle)
    if args.load:
        load_state(model, args.load, device)
    frozen = MyModel(128, 2, 256, 1, 1).to(device)           # the target whose exploitability we measure
    load_state(frozen, args.br_vs, device)
    frozen.eval()
    for pr in frozen.parameters():
        pr.requires_grad_(False)
    ref = MyModel(128, 2, 256, 1, 1).to(device)              # alpha=0 => logging-only (mag_kl = BR drift from init)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    for pr in ref.parameters():
        pr.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(out_dir, f"{args.out_prefix}_metrics.jsonl")
    open(metrics_path, "w").close()
    target_name = os.path.splitext(os.path.basename(args.br_vs))[0]
    print(f"BR-exploitability | BR vs FROZEN {target_name}"
          f"{' | resumed ' + os.path.basename(args.load) if args.load else ' | fresh'} "
          f"| updates={args.br_updates} games={args.games} epochs={args.epochs} "
          f"| lr={args.lr} ent={args.ent_coef} lam={args.lam} clip_vloss={args.clip_vloss} "
          f"| device={device.type}", flush=True)

    save_every = max(1, args.br_updates // 10)
    for u in range(args.br_updates):
        model.eval()
        buffer, Gs, t0 = [], [], time.time()
        with torch.no_grad():
            for i in progress(args.games, f"br update {u} collect"):
                recs, G = collect_br_game(model, frozen, deck, args.lam, i % 2)   # alternate BR seat
                buffer += recs
                Gs.append(G)
        gps = args.games / max(time.time() - t0, 1e-9)
        br_return = sum(Gs) / len(Gs)                        # on-policy avg terminal return (exploitability proxy)
        m = ppo_update(buffer, model, ref, opt, args, device)
        row = {"update": u, "br_return": round(br_return, 4), "games_per_s": round(gps, 2)} | (m or {})
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        msg = f"br update {u}: br_return={br_return:+.3f}"
        if m is not None:
            msg += (f" loss={m['loss']:+.4f} ppo={m['ppo_loss']:+.4f} H={m['entropy']:.3f} "
                    f"clip_frac={m['clip_frac']:.2f} expl_var={m['explained_var']:+.2f}")
        print(msg + f" ({gps:.1f} games/s)", flush=True)
        if u % save_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"{args.out_prefix}_br_u{u}.pth"))

    final = os.path.join(out_dir, f"{args.out_prefix}_br_final.pth")
    torch.save(model.state_dict(), final)

    # Headline: high-n seat-swapped BR-vs-target win% + Wilson CI (the exploitability lower bound).
    model.eval()
    n = args.final_eval_games
    with torch.inference_mode():
        br_agent = Agent("br", sampled_agent_fn(model, deck), deck)
        tgt_agent = Agent(target_name, sampled_agent_fn(frozen, deck), deck)
        w, l, d = play_match(br_agent, tgt_agent, n)
    p, lo, hi = wilson(w, w + l)
    ret = (w - l) / max(w + l + d, 1)
    print(f"\nEXPLOITABILITY vs {target_name}: BR {w}/{l}/{d} over {n} games | "
          f"win%={100 * p:.1f} (95% CI {100 * lo:.1f}-{100 * hi:.1f}) | return={ret:+.3f}\n"
          f"  => exploitability lower bound ~ max(0, {ret:+.3f})  (Nash~0, fully-exploitable~+1)", flush=True)
    # Machine-readable headline (so a sweep can aggregate without scraping logs). The n>=200 final
    # eval is NOT in the per-update metrics JSONL, so this is the artifact to compare across targets.
    result = {"target": target_name, "seed": args.seed, "n": n, "w": w, "l": l, "d": d,
              "br_win_pct": round(100 * p, 2), "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2),
              "br_return": round(ret, 4), "exploitability_lb": round(max(0.0, ret), 4),
              "br_updates": args.br_updates, "games": args.games}
    result_path = os.path.join(out_dir, f"{args.out_prefix}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"done -> {final} | metrics -> {metrics_path} | result -> {result_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outer", type=int, default=2, help="outer iterations (rho reset per outer)")
    ap.add_argument("--inner", type=int, default=2, help="inner PPO updates per outer iter")
    ap.add_argument("--games", type=int, default=4, help="self-play games per inner update")
    ap.add_argument("--epochs", type=int, default=1, help="PPO epochs per buffer (E small -- tiny buffers, R9)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="AdamW lr (default 3e-4 NASHPG, 1e-4 BR mode)")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO + value clip epsilon")
    # alpha is FIXED at 0.2 -- do NOT anneal; the rho RESET is the mechanism, not a schedule.
    ap.add_argument("--alpha", type=float, default=0.2, help="KL(pi||rho) coef (0 => plain PPO)")
    # ent-coef: SWEEP 0.01-0.05. Original NashPG uses 0.05 WITH the anchor+GAE (a different
    # regime than our 0.01 from plain REINFORCE). Watch entropy + mag_kl; raise if it collapses.
    ap.add_argument("--ent-coef", type=float, default=None, help="entropy coef (default 0.01 NASHPG, 0.1 BR mode)")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="value loss coef (modest -- shared trunk, R7)")
    ap.add_argument("--lam", type=float, default=1.0, help="GAE lambda (1.0=MC smoke; 0.95 real runs)")
    ap.add_argument("--clip-vloss", action="store_true", help="PPO-clipped critic (real runs; default plain MSE)")
    ap.add_argument("--eval-games", type=int, default=20, help="bare-policy eval games per outer (0=off)")
    ap.add_argument("--out-prefix", default="nashpg")
    ap.add_argument("--deck", default=None, help="deck CSV path (default: tcg_model.sample_deck -- the locked mirror)")
    ap.add_argument("--seed", type=int, default=0)
    # --- BR-exploitability mode + Kaggle resume (plan.md §6.4) ---
    ap.add_argument("--br-vs", default=None,
                    help="BR mode: train a fresh best-response vs this FROZEN target ckpt (=> exploitability)")
    ap.add_argument("--load", default=None,
                    help="init theta (and rho, NASHPG mode) from this ckpt (Kaggle resume); default from-scratch")
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "out"),
                    help="output dir for ckpts+metrics (set to /kaggle/working on Kaggle)")
    ap.add_argument("--br-updates", type=int, default=300, help="BR mode: number of PPO updates")
    ap.add_argument("--final-eval-games", type=int, default=200,
                    help="BR mode: high-n seat-swapped BR-vs-target eval (>= eval.TRUST_N)")
    args = ap.parse_args()

    # None-sentinel resolution: BR mode uses the reference recipe (lr 1e-4, ent 0.1); NASHPG keeps
    # its own (lr 3e-4, ent 0.01). An explicit CLI value always wins.
    if args.lr is None:
        args.lr = 1e-4 if args.br_vs else 3e-4
    if args.ent_coef is None:
        args.ent_coef = 0.1 if args.br_vs else 0.01

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train AND eval on the same deck (sample_deck) so eval measures the trained policy on its
    # own deck. (plan.md's collection + eval both center on sample_deck; the "--deck deck.csv"
    # in its CLI line was a train_reinforce copy artifact. CSV override kept for flexibility.)
    deck = sample_deck if args.deck is None else [int(x) for x in open(args.deck).read().split()]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.br_vs:                                         # BR-exploitability mode (plan.md §6.4)
        return run_br(args, device, deck, out_dir)

    model = MyModel(128, 2, 256, 1, 1).to(device)          # from-scratch unless --load (Kaggle resume)
    if args.load:
        load_state(model, args.load, device)
    ref = MyModel(128, 2, 256, 1, 1).to(device)            # rho: frozen reference policy
    ref.load_state_dict(model.state_dict())                # rho starts at theta (loaded or fresh)
    ref.eval()
    for pr in ref.parameters():
        pr.requires_grad_(False)                           # belt-and-suspenders: never trained (R13)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)  # created ONCE; Adam state persists across resets

    metrics_path = os.path.join(out_dir, f"{args.out_prefix}_metrics.jsonl")
    open(metrics_path, "w").close()                        # truncate the metrics log for this run

    # Frozen past-self anchor for the mirror progress signal (the same fresh init, never updated).
    init_path = os.path.join(out_dir, f"{args.out_prefix}_init.pth")
    torch.save(model.state_dict(), init_path)
    init_model = MyModel(128, 2, 256, 1, 1).to(device)
    init_model.load_state_dict(model.state_dict())
    init_model.eval()

    print(f"NASHPG self-play | {'resumed ' + os.path.basename(args.load) if args.load else 'from-scratch'} "
          f"| deck={'sample_deck' if args.deck is None else os.path.basename(args.deck)} "
          f"| outer={args.outer} inner={args.inner} games={args.games} epochs={args.epochs} "
          f"| alpha={args.alpha} ent={args.ent_coef} vf={args.vf_coef} lam={args.lam} "
          f"clip_vloss={args.clip_vloss} | device={device.type}", flush=True)
    eval_checkpoint(model, init_model, deck, args.eval_games, "init")

    for t in range(args.outer):
        for k in range(args.inner):
            model.eval()
            buffer, t0 = [], time.time()
            with torch.no_grad():
                for _ in progress(args.games, f"outer {t} inner {k} collect"):
                    buffer += collect_game(model, deck, args.lam)
            gps = args.games / max(time.time() - t0, 1e-9)     # R10: games/sec to size real runs
            m = ppo_update(buffer, model, ref, opt, args, device)
            if m is None:
                print(f"outer {t} inner {k}: 0 decisions collected -- skipped", flush=True)
                continue
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"outer": t, "inner": k, "games_per_s": round(gps, 2)} | m) + "\n")
            print(f"outer {t} inner {k}: decisions={m['n_decisions']} loss={m['loss']:+.4f} "
                  f"ppo={m['ppo_loss']:+.4f} crit={m['critic_loss']:.4f} H={m['entropy']:.3f} "
                  f"approx_kl={m['approx_kl']:.4f} clip_frac={m['clip_frac']:.2f} "
                  f"mag_kl={m['mag_kl']:.4f} expl_var={m['explained_var']:+.2f} "
                  f"({gps:.1f} games/s)", flush=True)

        ref.load_state_dict(model.state_dict())            # REFERENCE RESET -- once per outer iter
        ckpt = os.path.join(out_dir, f"{args.out_prefix}{t}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"outer {t}: rho<-theta reset; saved {ckpt}", flush=True)
        eval_checkpoint(model, init_model, deck, args.eval_games, f"t{t}")

    final = os.path.join(out_dir, f"{args.out_prefix}_final.pth")
    torch.save(model.state_dict(), final)
    print(f"done -> {final} | metrics -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
