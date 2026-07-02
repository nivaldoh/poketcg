#!/usr/bin/env python3
"""Minimal, from-scratch PPO trainer for the Pokémon TCG `cg` agent — the DRIVER.

The reusable implementation (featurization, network, rollout buffer, the `PPO` algorithm) lives
in `ppo_model.py`; this file is just the training driver — the cg battle loop, the opponent
factory, and the train loop + argparse — mirroring how `tcg_model.py` holds the model layer for
the other trainers. See `ppo_model.py`'s docstring for the design rationale (fresh net, the
per-option scoring head, Monte-Carlo returns) and the reference (PPO-PyTorch/PPO.py).

Design (locked) for the driver:
  * SELF-PLAY (default `--opponent self`): the opponent seat is the LIVE current policy, and the
    PPO seat learns a MIRROR (same deck both seats) across all 4 game decks (`--decks`) with ONE
    shared model. `random`/`frozen` opponents are kept as a curriculum/fallback because pure
    cold-start self-play can stall/diverge on this simple starter (see the repo notes).
  * v1 coverage limit: we only learn SINGLE-SELECT policy nodes (`is_policy_select`); every
    other decision is auto-resolved (`auto_resolve`) and emits no training record. Multi-select
    is a documented future extension.
  * Mirror self-play win-rate is ~50% BY CONSTRUCTION (same policy both seats) — read progress
    from eval_ppo.py's deck×opponent matrix, not from the collect win-rate.

Salvaged from the previous (discarded NASHPG-copy) train_ppo.py: only the cg.api / cg.game
interaction — the `battle_start → battle_select* → battle_finish` driver loop, the
process-global-singleton `try/finally` discipline, and the `yourIndex` seat dispatch.

Skeleton-stage check (the model's stubs in ppo_model.py still raise; this just builds argparse
and resolves imports):
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
from utils.run_smoketest_battle import load_deck                       # noqa: E402  (canonical deck CSV loader)
from cg.api import to_observation_class                                # noqa: E402  (frozen opp)
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402
from ppo_model import PPO, STATE_DIM, OPTION_DIM                       # noqa: E402


# ============================================================================================
# cg ENGINE PLUMBING — the salvaged interaction. Drives battles, collects the PPO seat's
# trajectory into the buffer, and writes the terminal reward.
# ============================================================================================
# TODO: maybe this is where we could later add a battle visualizer?
def collect_games(ppo, opp_fn, decks, n_games):
    """Play `n_games` of MIRROR self-play (both seats the same deck), appending the PPO seat's
    learned decisions to ppo.buffer. Returns the per-game seat-relative result G (for win-rate
    logging). `decks` is a LIST; each game picks one at random and both seats play it, so a single
    shared buffer/model learns all decks. RISK: with --opponent self the win-rate is ~50% BY
    CONSTRUCTION (same policy both seats) — a sanity check, NOT a progress meter (use eval_ppo.py).

    The cg battle is a PROCESS-GLOBAL SINGLETON — a leaked live battle corrupts every later
    game in the process, so battle_finish() MUST run even on error (try/finally). In the local
    battle_start-driven loop `obs.select` is never None (the deck-return path is Kaggle-only)."""
    results = []
    for g in range(n_games):
        ppo_seat = g % 2                      # alternate seats for symmetry
        deck = random.choice(decks)           # random deck/game (mirror); decoupled from seat parity
        obs, start = battle_start(deck, deck)
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
        results.append(G)
    return results


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
    """Return an opp_fn(obs_dict) -> list[int] for the non-PPO seat. Default: self (self-play)."""
    if kind == "self":
        # SELF-PLAY (default): opponent = the LIVE current policy (no snapshot). Within a collection
        # round no update runs, so the opponent is a consistent copy of the current policy; across
        # rounds both seats improve in lockstep. policy_agent never touches ppo.buffer, so ONLY the
        # PPO seat records and the PPO ratio (old_logp/V_old captured at act time) stays valid.
        return policy_agent(ppo.policy)
    if kind == "random":
        return random_agent
    if kind == "frozen":
        snapshot = copy.deepcopy(ppo.policy).to(device).eval()
        return policy_agent(snapshot, greedy=False)   # samples, exactly like the old inline opp()
    if kind == "rulebased":
        # The rule-based agents are hardcoded to their own (e.g. Lucario) deck, so a rule-based
        # opponent in our same-deck mirror would be deck-confounded — out of v1 scope. When we
        # add it: eval.make_rulebased() gives (fn, deck); drive with asymmetric decks.
        raise NotImplementedError("rulebased opponent: deferred (deck-confounding) — see comment")
    raise ValueError(f"unknown opponent kind: {kind!r}")


# ============================================================================================
# TRAINING LOOP — concrete control flow; learning happens inside the stubbed PPO.update.
# ============================================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    deck_names = [s.strip() for s in args.decks.split(",") if s.strip()]
    decks = [load_deck(os.path.join(_REPO_ROOT, "decks", f"{n}.csv")) for n in deck_names]
    ppo = PPO(STATE_DIM, OPTION_DIM, args.lr, args.gamma, args.eps_clip, args.k_epochs,
              args.vf_coef, args.ent_coef, device)
    opp_fn = make_opponent(args.opponent, ppo, device)
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

    print(f"PPO self-play | opponent={args.opponent} | decks={deck_names} | "
          f"{args.games_per_update} games/update x {args.updates} updates | device={device} | "
          f"metrics={metrics_path}", flush=True)
    for it in range(1, args.updates + 1):
        t0 = time.time()
        results = collect_games(ppo, opp_fn, decks, args.games_per_update)
        n_samples = len(ppo.buffer)                # capture before update() clears the buffer
        stats = ppo.update() or {}
        wins = sum(1 for G in results if G > 0)
        draws = sum(1 for G in results if G == 0)
        n = len(results)
        games_per_s = n / max(time.time() - t0, 1e-9)   # collect+update per the nashpg convention
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"update": it, "games": n, "wins": wins, "draws": draws,
                                "losses": n - wins - draws, "n_samples": n_samples,
                                "games_per_s": round(games_per_s, 2)} | stats) + "\n")
        # winrate is ~.5 under self-play (mirror) — a sanity check, NOT progress; the real signal is
        # the eval matrix (eval_ppo.py) + these update diagnostics (value fit / entropy / KL).
        print(f"[update {it}/{args.updates}] games={n} winrate={wins / n:.3f} samples={n_samples} "
              f"loss={stats.get('loss', float('nan')):.3f} vloss={stats.get('value_loss', float('nan')):.3f} "
              f"ev={stats.get('explained_var', float('nan')):+.2f} ent={stats.get('entropy', float('nan')):.3f} "
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
    # PPO hyperparameters (reference defaults; eps_clip=0.2 per the paper, K=4 vs ref's 80).
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--k-epochs", type=int, default=4)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    # Rollout / loop control.
    ap.add_argument("--games-per-update", type=int, default=16,
                    help="self-play games collected before each PPO update")
    ap.add_argument("--updates", type=int, default=100, help="number of PPO updates")
    ap.add_argument("--opponent", default="self", choices=["self", "random", "frozen", "rulebased"],
                    help="non-PPO seat: 'self'=live current policy (self-play, default); "
                         "'random'/'frozen' kept as curriculum/fallback (self-play can stall)")
    ap.add_argument("--decks", default="lucario,abomasnow,dragapult_ex,iono",
                    help="comma list of deck names in decks/<name>.csv; mirror self-play cycles them")
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
#   5. SELF-PLAY + MULTI-DECK (current): `--opponent self` (live policy, mirror) across all 4 decks,
#      one shared model. Progress is read from eval_ppo.py's deck×opponent matrix vs the rule-based
#      agents (the mirror win-rate here is ~50% by construction). Future: both-seat recording (2x
#      data), lagged-snapshot opponent if cold-start self-play stalls, multi-select coverage.
# ============================================================================================
