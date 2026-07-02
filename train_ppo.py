#!/usr/bin/env python3
"""Minimal, from-scratch PPO trainer for the Pokémon TCG `cg` agent — the DRIVER.

The reusable implementation (featurization, network, rollout buffer, the `PPO` algorithm) lives
in `ppo_model.py`; this file is just the training driver — the cg battle loop, the opponent
factory, and the train loop + argparse — mirroring how `tcg_model.py` holds the model layer for
the other trainers. See `ppo_model.py`'s docstring for the design rationale (fresh net, the
per-option scoring head, GAE(λ) + TD(λ) targets) and the reference (PPO-PyTorch/PPO.py).

Design for the driver:
  * FIELD TRAINING (default `--opponent rulebased`): the opponent seat is one of the 4 rule-based
    archetype agents (random per game), each playing ITS OWN deck (asymmetric battle_start); the
    PPO seat plays a random `--decks` deck, ONE shared model. This replaced mirror self-play as
    the default after the measured self-play NON-TRANSFER (docs/ppo/ppo_ceiling_report.md: crushes
    random everywhere, loses all 16 deck×opponent cells vs the field). DELIBERATE consequence:
    the training field == eval_ppo.py's field, so that eval is in-distribution — acceptable
    because the project goal is beating exactly this field.
  * `self` (live-policy mirror), `random`, `frozen` remain as curriculum/fallback modes, and
    `--opponent` accepts a comma MIX drawn per game (e.g. `rulebased,random` — the reward-
    starvation curriculum lever from docs/ppo/ppo_failure_investigation.md). Under
    `--opponent self` the collect win-rate is ~50% BY CONSTRUCTION (same policy both seats) —
    read progress from eval_ppo.py instead; vs the rule-based field the collect win-rate IS a
    genuine progress signal.
  * v1 coverage limit: we only learn SINGLE-SELECT policy nodes (`is_policy_select`); every
    other decision is auto-resolved (`auto_resolve`) and emits no training record. Multi-select
    is a documented future extension.

Salvaged from the previous (discarded NASHPG-copy) train_ppo.py: only the cg.api / cg.game
interaction — the `battle_start → battle_select* → battle_finish` driver loop, the
process-global-singleton `try/finally` discipline, and the `yourIndex` seat dispatch.

Smallest build check (argparse + imports resolve):
    python train_ppo.py --help
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: importing tcg_model runs the `CG_LIB_PATH` sys.path insert
# (tcg_model.py, the `_CG_LIB_PATH` block) that makes the vendored `cg` package importable. The
# `cg`/`ppo_model` imports below depend on that having happened — keep this ordering.
from tcg_model import random_agent                                     # noqa: E402
from eval_mymodel import is_policy_select, auto_resolve                # noqa: E402  (frozen opp)
from eval import make_rulebased, make_abomasnow, make_dragapult, make_iono  # noqa: E402  (field opp)
from utils.run_smoketest_battle import load_deck                       # noqa: E402  (canonical deck CSV loader)
from cg.api import to_observation_class                                # noqa: E402  (frozen opp)
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402
from ppo_model import PPO, STATE_DIM, OPTION_DIM                       # noqa: E402


# ============================================================================================
# cg ENGINE PLUMBING — the salvaged interaction. Drives battles, collects the PPO seat's
# trajectory into the buffer, and writes the terminal reward.
# ============================================================================================
# TODO: maybe this is where we could later add a battle visualizer?
def collect_games(ppo, next_opponent, decks, n_games):
    """Play `n_games`, appending the PPO seat's learned decisions to ppo.buffer. Returns per-game
    records (deck_name, opp_name, G) — G is the seat-relative result — so the metrics jsonl can
    carry the per-matchup breakdown (the failure investigation had to reconstruct per-opponent
    trends from checkpoints because only the pooled win-rate was logged). `decks` is a list of
    (name, deck) pairs; the learner picks one at random each game, so a single shared
    buffer/model learns all decks. `next_opponent()` is called once per game and returns
    (opp_fn, opp_deck, opp_name): opp_deck None = MIRROR (the opponent plays the learner's deck —
    the self/random/frozen modes, exactly the old behavior); a real deck = ASYMMETRIC battle (the
    rulebased field, each archetype on its own deck — eval.py's `_play_one` seat pattern). Vs a
    fixed opponent the win-rate is a genuine progress meter; under --opponent self it stays ~50%
    BY CONSTRUCTION (use eval_ppo.py instead there).

    The cg battle is a PROCESS-GLOBAL SINGLETON — a leaked live battle corrupts every later
    game in the process, so battle_finish() MUST run even on error (try/finally). In the local
    battle_start-driven loop `obs.select` is never None (the deck-return path is Kaggle-only)."""
    records = []
    for g in range(n_games):
        ppo_seat = g % 2                      # alternate seats for symmetry
        deck_name, learner_deck = random.choice(decks)  # random learner deck/game; decoupled from seat parity
        opp_fn, opp_deck, opp_name = next_opponent()
        seat_decks = [None, None]
        seat_decks[ppo_seat] = learner_deck
        seat_decks[1 - ppo_seat] = opp_deck if opp_deck is not None else learner_deck
        obs, start = battle_start(seat_decks[0], seat_decks[1])
        if start.errorPlayer >= 0:
            raise ValueError("deck rejected by engine")
        recorded_any = False
        try:
            while obs["current"]["result"] < 0:
                yi = obs["current"]["yourIndex"]
                if yi == ppo_seat:
                    # TODO: PPO taking actions one-by-one? If we had multi-select, probably we would have to collect the whole multi-select decision here.
                    sel, recorded = ppo.select_action(obs)
                    recorded_any = recorded_any or recorded
                else:
                    sel = opp_fn(obs)
                obs = battle_select(sel)
        finally:
            battle_finish()
        result = obs["current"]["result"]
        # G ∈ {+1 win, -1 loss, 0 draw}, seat-relative. RISK (draw): a drawn game's whole episode is
        # all-zero returns, which after advantage-normalization reads as a small NEGATIVE advantage in
        # a win-heavy batch — a minor spurious penalty on draws. Rare; accepted for v1.
        G = 0.0 if result == 2 else (1.0 if result == ppo_seat else -1.0)
        # Terminal-only reward: the PPO seat's records this game are the tail of the buffer (the
        # opponent never appends), so buffer[-1] is this game's last learned decision.
        # COARSE CREDIT (accepted v1): G is attributed to the last POLICY-SELECT, which may precede the
        # real game end by several auto-resolved / opponent steps — no per-step reward shaping.
        if recorded_any:
            ppo.buffer.rewards[-1] = G
            ppo.buffer.is_terminals[-1] = True
        records.append((deck_name, opp_name, G))
    return records


# ============================================================================================
# OPPONENTS — fixed (stationary) opponent factory for v1.
# ============================================================================================
def policy_agent(net, greedy=False):
    """Wrap a (trained / frozen) ActorCritic as an agent_fn(obs_dict) -> list[int]: SAMPLE (or argmax
    if greedy) on single-select policy nodes, auto_resolve everything else — the SAME decision routing
    the PPO seat uses at collection time. Shared by make_opponent('frozen') and eval_ppo.py so the
    frozen-opponent, trained-agent, and eval wrappers can never drift apart."""
    def agent_fn(obs_dict):
        o = to_observation_class(obs_dict)
        sel = o.select
        if not is_policy_select(sel):
            return auto_resolve(sel)
        with torch.no_grad():
            return [net.act_index(o, greedy=greedy)]
    return agent_fn


def make_opponent(kind, ppo, device):
    """Return a per-game provider next_opponent() -> (opp_fn, opp_deck_or_None, opp_name) for the
    non-PPO seat; opp_deck None = mirror (opponent plays the learner's deck). `kind` may be a
    COMMA LIST (e.g. 'rulebased,random'): one entry is drawn uniformly per game, and repeating a
    kind weights it ('rulebased,rulebased,random' = 2:1) — the curriculum lever for reward
    starvation (docs/ppo/ppo_failure_investigation.md, verdict 2: vs dragapult/iono the field alone
    yields ~0.1 wins/update). Default: rulebased."""
    kinds = [k.strip() for k in kind.split(",") if k.strip()]
    if not kinds:
        raise ValueError(f"empty opponent spec: {kind!r}")
    if len(kinds) > 1:
        providers = [make_opponent(k, ppo, device) for k in kinds]
        return lambda: random.choice(providers)()
    kind = kinds[0]
    if kind == "self":
        # SELF-PLAY: opponent = the LIVE current policy (no snapshot). Within a collection
        # round no update runs, so the opponent is a consistent copy of the current policy; across
        # rounds both seats improve in lockstep. policy_agent never touches ppo.buffer, so ONLY the
        # PPO seat records and the PPO ratio (old_logp/V_old captured at act time) stays valid.
        fn = policy_agent(ppo.policy)
        return lambda: (fn, None, "self")
    if kind == "random":
        return lambda: (random_agent, None, "random")
    if kind == "frozen":
        snapshot = copy.deepcopy(ppo.policy).to(device).eval()
        fn = policy_agent(snapshot, greedy=False)     # samples, exactly like the old inline opp()
        return lambda: (fn, None, "frozen")
    if kind == "rulebased":
        # THE FIELD (default): one of the 4 rule-based archetypes per game, each playing ITS OWN
        # deck (asymmetric battle_start — the fix for the measured self-play non-transfer).
        # Reuses eval.py's factories; the modules reset their per-game state on their own turn-0
        # detection, so reuse across games is safe (same pattern as eval's gauntlet).
        pool = [make_rulebased(), make_abomasnow(), make_dragapult(), make_iono()]

        def next_opponent():
            a = random.choice(pool)
            return a.fn, a.deck, a.name               # eval.Agent.name (make_rulebased = "rulebased")
        return next_opponent
    raise ValueError(f"unknown opponent kind: {kind!r}")


# ============================================================================================
# TRAINING LOOP — concrete control flow; learning happens inside the stubbed PPO.update.
# ============================================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    deck_names = [s.strip() for s in args.decks.split(",") if s.strip()]
    decks = [(n, load_deck(os.path.join(_REPO_ROOT, "decks", f"{n}.csv"))) for n in deck_names]
    ppo = PPO(STATE_DIM, OPTION_DIM, args.lr, args.gamma, args.eps_clip, args.k_epochs,
              args.vf_coef, args.ent_coef, device,
              gae_lambda=args.gae_lambda, minibatches=args.minibatches)
    next_opponent = make_opponent(args.opponent, ppo, device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Layer-1 telemetry sink (evaluation_harness_report.md §1a): one flat JSON object per update
    # (driver fields + the full PPO.update() stats dict), truncated at run start — the same
    # convention as the legacy nashpg trainer's {prefix}_metrics.jsonl.
    metrics_path = os.path.join(args.out_dir, f"{args.out_prefix}_metrics.jsonl")
    open(metrics_path, "w").close()

    # Save the UNTRAINED net first so eval_ppo.py can anchor "did training help?" against the fresh-net
    # baseline. That baseline is NOT ~50% vs random: a fresh net still uses the real auto_resolve
    # heuristic on all non-policy nodes, so it starts with an edge — the success test is beating IT.
    ppo.save(os.path.join(args.out_dir, f"{args.out_prefix}_init.pth"))

    print(f"PPO train | opponent={args.opponent} | decks={deck_names} | "
          f"{args.games_per_update} games/update x {args.updates} updates | device={device} | "
          f"metrics={metrics_path}", flush=True)
    for it in range(1, args.updates + 1):
        t0 = time.time()
        records = collect_games(ppo, next_opponent, decks, args.games_per_update)
        n_samples = len(ppo.buffer)                # capture before update() clears the buffer
        stats = ppo.update() or {}
        results = [G for _, _, G in records]
        wins = sum(1 for G in results if G > 0)
        draws = sum(1 for G in results if G == 0)
        n = len(results)
        games_per_s = n / max(time.time() - t0, 1e-9)   # collect+update per the nashpg convention
        with open(metrics_path, "a") as f:
            # per_game [[learner_deck, opponent, G], ...] — the per-matchup breakdown the failure
            # investigation lacked (pooled win-rate masked the dead dragapult/iono columns).
            f.write(json.dumps({"update": it, "games": n, "wins": wins, "draws": draws,
                                "losses": n - wins - draws, "n_samples": n_samples,
                                "games_per_s": round(games_per_s, 2),
                                "per_game": [[d, o, G] for d, o, G in records]} | stats) + "\n")
        # Vs a FIXED opponent (rulebased/random/frozen) winrate is a genuine progress signal;
        # under --opponent self it is ~.5 by construction (mirror) — use eval_ppo.py there. The
        # update diagnostics (value fit / entropy / KL) matter in every mode.
        print(f"[update {it}/{args.updates}] games={n} winrate={wins / n:.3f} samples={n_samples} "
              f"loss={stats.get('loss', float('nan')):.3f} vloss={stats.get('value_loss', float('nan')):.3f} "
              f"ev={stats.get('explained_var', float('nan')):+.2f} "
              f"ev_h={stats.get('ev_true_pre', float('nan')):+.2f} ent={stats.get('entropy', float('nan')):.3f} "
              f"ent_n={stats.get('entropy_norm', float('nan')):.2f} kl={stats.get('approx_kl', float('nan')):.4f} "
              f"clip={stats.get('clip_frac', float('nan')):.2f} ({games_per_s:.1f} g/s)", flush=True)
        # WARN-only tripwires — only unambiguous bug signals (zero false-positive risk), and never
        # abort a run. Everything threshold-y (kl, ev, clip_frac) stays a logged number instead:
        # the repo's own healthy legacy run had approx_kl=0.039 on update 1, so a 0.02-style KL
        # warning would be wallpaper.
        if stats.get("ratio_dev_e0", 0.0) > 1e-4:
            print(f"[WARN] epoch-0 ratio deviates from 1 by {stats['ratio_dev_e0']:.2e} — "
                  f"act/evaluate recompute drift (bug)", flush=True)
        if not math.isfinite(stats.get("loss", 0.0)):
            print("[WARN] non-finite loss", flush=True)
        if it % args.save_every == 0:
            ppo.save(os.path.join(args.out_dir, f"{args.out_prefix}{it}.pth"))
    ppo.save(os.path.join(args.out_dir, f"{args.out_prefix}_final.pth"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # PPO hyperparameters (eps_clip=0.2 per the paper; γ/λ/epochs×minibatches follow the NashPG
    # config for the same terminal-only zero-sum reward shape).
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="discount; default 1.0 = undiscounted (NashPG's setting for terminal-only "
                         "zero-sum: every step of a won game targets +1, the easiest critic target) "
                         "— THEORY-REVIEW FLAG, was 0.99")
    ap.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE(λ); the λ-return TD targets replace pure MC returns (broken-critic fix)")
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--k-epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4,
                    help="shuffled minibatches per epoch (per-minibatch advantage normalization)")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    # Rollout / loop control.
    ap.add_argument("--games-per-update", type=int, default=16,
                    help="games collected before each PPO update")
    ap.add_argument("--updates", type=int, default=100, help="number of PPO updates")
    ap.add_argument("--opponent", default="rulebased",
                    help="non-PPO seat: comma list over {rulebased,self,random,frozen}, drawn "
                         "uniformly per game (repeat a kind to weight it). 'rulebased'=the "
                         "4-archetype field, each on its own deck (default — the measured fix for "
                         "self-play non-transfer); 'self'=live current policy (mirror self-play); "
                         "'random'/'frozen'=curriculum/fallback. E.g. 'rulebased,random' mixes in "
                         "winnable games against the reward starvation measured in "
                         "docs/ppo/ppo_failure_investigation.md")
    ap.add_argument("--decks", default="lucario,abomasnow,dragapult_ex,iono",
                    help="comma list of deck names in decks/<name>.csv the LEARNER cycles through "
                         "(the rulebased opponent always plays its own archetype deck)")
    # Bookkeeping (repo conventions: out/{prefix}*.pth).
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="ppo")
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "out"))
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================================
# ROADMAP (done; each shipped with the smallest verification):
#   1. ppo_model.encode_state / encode_options -> check tensor shapes on a real obs (STATE_DIM/OPTION_DIM).
#   2. (ActorCritic.act/evaluate are already concrete) -> one forward pass on a single collected step.
#   3. ppo_model.PPO.update (MC returns + clipped surrogate) -> loss finite; ratio≈1 on epoch 0.
#   4. train() vs random -> win-rate cleared the fresh-net baseline (eval_ppo.py, n per eval-discipline).
#   5. SELF-PLAY + MULTI-DECK: `--opponent self` (live policy, mirror) across all 4 decks, one
#      shared model. Outcome measured in docs/ppo/ppo_ceiling_report.md: NON-TRANSFER (crushes random,
#      loses all 16 cells vs the rule-based field) + broken critic (explained_var ~0.1).
#   6. FIELD TRAINING + GAE + MINIBATCHES (current): `--opponent rulebased` default (asymmetric
#      decks vs the 4 archetypes — the fix for finding 5's non-transfer), GAE(λ=0.95)/γ=1.0
#      TD(λ) value targets (the broken-critic fix), K epochs × shuffled minibatches with
#      per-minibatch advantage normalization (approx_kl was ~0.001 — policy barely moved).
#   7. CRITIC FIX + HONEST TELEMETRY (current, from docs/ppo/ppo_failure_investigation.md): LINEAR
#      value head (the tanh output saturated near -1 under the ~90%-loss field, holdout EV ~0 —
#      "win-only REINFORCE"), ev_true_pre = holdout EV of act-time V vs true outcome (the gate;
#      in-sample explained_var read ~0.4 while holdout was ~0), per_game [deck,opp,G] records in
#      the metrics jsonl, and --opponent comma-mix (e.g. 'rulebased,random') as the
#      reward-starvation curriculum lever.
#      Future: both-seat recording under self-play (2x data), league/lagged-snapshot opponent
#      mixes, multi-select coverage.
# ============================================================================================
