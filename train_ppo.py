#!/usr/bin/env python3
"""Minimal, from-scratch PPO trainer for the Pokémon TCG `cg` agent — the DRIVER.

The reusable implementation (featurization, network, rollout buffer, the `PPO` algorithm) lives
in `ppo_model.py`; this file is just the training driver — the cg battle loop, the opponent
factory, and the train loop + argparse — mirroring how `tcg_model.py` holds the model layer for
the other trainers. See `ppo_model.py`'s docstring for the design rationale (fresh net, the
per-option scoring head, Monte-Carlo returns) and the reference (PPO-PyTorch/PPO.py).

Design (locked) for the driver:
  * FIXED opponent (stationary env) for v1 — the PPO seat plays a fixed opponent (default
    random; later: frozen snapshot / rule-based). Matches the single-agent PPO reference and
    sidesteps the self-play divergence the repo has seen on this simple starter.
  * v1 coverage limit: we only learn SINGLE-SELECT policy nodes (`is_policy_select`); every
    other decision is auto-resolved (`auto_resolve`) and emits no training record. Multi-select
    is a documented future extension.

Salvaged from the previous (discarded NASHPG-copy) train_ppo.py: only the cg.api / cg.game
interaction — the `battle_start → battle_select* → battle_finish` driver loop, the
process-global-singleton `try/finally` discipline, and the `yourIndex` seat dispatch.

Skeleton-stage check (the model's stubs in ppo_model.py still raise; this just builds argparse
and resolves imports):
    python train_ppo.py --help
"""

import argparse
import copy
import os
import random
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: importing tcg_model runs the `CG_LIB_PATH` sys.path insert
# (tcg_model.py, the `_CG_LIB_PATH` block) that makes the vendored `cg` package importable. The
# `cg`/`ppo_model` imports below depend on that having happened — keep this ordering.
from tcg_model import random_agent, sample_deck                        # noqa: E402
from eval_mymodel import is_policy_select, auto_resolve                # noqa: E402  (frozen opp)
from cg.api import to_observation_class                                # noqa: E402  (frozen opp)
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402
from ppo_model import PPO, STATE_DIM, OPTION_DIM                       # noqa: E402


# ============================================================================================
# cg ENGINE PLUMBING — the salvaged interaction. Drives battles, collects the PPO seat's
# trajectory into the buffer, and writes the terminal reward.
# ============================================================================================
# TODO: maybe this is where we could later add a battle visualizer?
def collect_games(ppo, opp_fn, deck, n_games):
    """Play `n_games` vs a FIXED opponent, appending the PPO seat's learned decisions to
    ppo.buffer. Returns the per-game seat-relative result G (for win-rate logging).

    The cg battle is a PROCESS-GLOBAL SINGLETON — a leaked live battle corrupts every later
    game in the process, so battle_finish() MUST run even on error (try/finally). In the local
    battle_start-driven loop `obs.select` is never None (the deck-return path is Kaggle-only)."""
    results = []
    for g in range(n_games):
        ppo_seat = g % 2                      # alternate seats for symmetry
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
        G = 0.0 if result == 2 else (1.0 if result == ppo_seat else -1.0)
        # Terminal-only reward: the PPO seat's records this game are the tail of the buffer
        # (the opponent never appends), so buffer[-1] is this game's last learned decision.
        if recorded_any:
            ppo.buffer.rewards[-1] = G
            ppo.buffer.is_terminals[-1] = True
        results.append(G)
    return results


# ============================================================================================
# OPPONENTS — fixed (stationary) opponent factory for v1.
# ============================================================================================
def make_opponent(kind, ppo, device):
    """Return an opp_fn(obs_dict) -> list[int] for the non-PPO seat. v1 default: random."""
    if kind == "random":
        return random_agent
    if kind == "frozen":
        snapshot = copy.deepcopy(ppo.policy).to(device).eval()
        def opp(obs_dict):
            o = to_observation_class(obs_dict)
            sel = o.select
            if not is_policy_select(sel):
                return auto_resolve(sel)
            with torch.no_grad():
                return [snapshot.act(o).action]
        return opp
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

    deck = sample_deck                         # v1: locked mirror deck for both seats
    ppo = PPO(STATE_DIM, OPTION_DIM, args.lr, args.gamma, args.eps_clip, args.k_epochs,
              args.vf_coef, args.ent_coef, device)
    opp_fn = make_opponent(args.opponent, ppo, device)
    os.makedirs(args.out_dir, exist_ok=True)

    for it in range(1, args.updates + 1):
        results = collect_games(ppo, opp_fn, deck, args.games_per_update)
        stats = ppo.update()
        wins = sum(1 for G in results if G > 0)
        n = len(results)
        print(f"[update {it}/{args.updates}] games={n} winrate={wins / n:.3f} "
              f"buffer={len(ppo.buffer)} loss={stats.get('loss') if stats else 'n/a'}")
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
    ap.add_argument("--opponent", default="random", choices=["random", "frozen", "rulebased"])
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
# FILL-IN ROADMAP (one step at a time, smallest verification each):
#   1. ppo_model.encode_state / encode_options -> check tensor shapes on a real obs (STATE_DIM/OPTION_DIM).
#   2. (ActorCritic.act/evaluate are already concrete) -> one forward pass on a single collected step.
#   3. ppo_model.PPO.update (MC returns + clipped surrogate) -> loss finite; ratio≈1 on epoch 0.
#   4. run train() vs random; confirm win-rate climbs above the ~50% random baseline (eval.py,
#      trustworthy n per the eval-discipline skill).
# ============================================================================================
