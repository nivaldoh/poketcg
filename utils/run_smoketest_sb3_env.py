#!/usr/bin/env python3
"""Smoke test for sb3_env.CGEnv: SB3's env_checker plus full masked episodes.

Asserts (per the README's "verifiable results" goal, the smallest end-to-end signal):
  1. `stable_baselines3.common.env_checker.check_env` passes (warnings about unbounded Box
     features are expected and fine; errors are not).
  2. N full episodes driven by uniform-over-`action_masks()` actions each reach a terminal with
     reward in {-1, 0, +1}, never set info["invalid_action"] (masked play can't go out of range),
     and carry the win_mode telemetry dict.
  3. Multi-select fires across all shapes and never emits an invalid battle_select: optional-single
     [0..1], fixed-k, true-variable (min<max, max>1), and single-option decline (len==1) nodes are
     surfaced (ms_optional/ms_fixed/ms_variable/ms_single_option > 0), and the empty-submit decline
     battle_select([]) at minCount==0 is exercised (ms_empty_submit > 0) — via masked play plus two
     scripted micro-drives (always-STOP / always-pick-max). Any invalid list raises out of env.step.
  4. After close(), a fresh raw battle still runs — the process-global cg battle singleton was
     not leaked.

Usage (from the repo root):
    python utils/run_smoketest_sb3_env.py             # check_env + 3 episodes + multi-select drives
    python utils/run_smoketest_sb3_env.py --games 10
"""

import argparse
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

# sb3_env owns the CG_LIB_PATH import-order discipline (it imports tcg_model first).
from sb3_env import CGEnv, N_MAX, STOP_INDEX                  # noqa: E402
from utils.run_smoketest_battle import load_deck, play_one    # noqa: E402


def run_episode(env, rng):
    """One full episode with uniform-over-valid actions. Returns (reward, steps, info)."""
    obs, _ = env.reset()
    steps = 0
    while True:
        valid = np.flatnonzero(env.action_masks())
        assert valid.size >= 1, "empty action mask at a policy node"
        assert obs["action_mask"][valid].all(), "obs mask disagrees with action_masks()"
        obs, r, terminated, truncated, info = env.step(int(rng.choice(valid)))
        steps += 1
        assert not truncated, "CGEnv never truncates"
        assert "invalid_action" not in info, "masked play produced an invalid action"
        if terminated:
            return r, steps, info


def drive_episode(env, choose):
    """One full episode where `choose(mask) -> action index` picks each action. Scripted policies
    (always-STOP / always-pick-max) exercise the multi-select decline and auto-commit paths
    deterministically. Every `choose` return is a live-mask index, so no invalid_action fires; any
    engine-invalid battle_select list would instead raise IndexError out of env.step and fail here."""
    env.reset()
    steps = 0
    while True:
        a = choose(env.action_masks())
        _, r, terminated, truncated, info = env.step(a)
        steps += 1
        assert not truncated, "CGEnv never truncates"
        assert "invalid_action" not in info, "scripted play produced an invalid action"
        if terminated:
            return r, steps


def _choose_stop(mask):
    """Pick STOP whenever legal (exercises empty-submit at minCount==0 + early stop); else the first
    legal real option."""
    if mask[STOP_INDEX]:
        return STOP_INDEX
    return int(np.flatnonzero(mask)[0])


def _choose_max(mask):
    """Never STOP voluntarily: pick the first legal real option to drive fixed-k / optional / variable
    nodes to auto-commit at maxCount. If a node ever had maxCount > option count the real options run
    out before the ceiling, leaving only STOP legal — the engine caps maxCount <= len(option) for the
    current decks (census-verified), so this fallback to STOP is defensive, not a live path."""
    real = np.flatnonzero(mask[:STOP_INDEX])
    return int(real[0]) if real.size else STOP_INDEX


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=3, help="full episodes after check_env")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    deck_names = ["lucario", "abomasnow", "dragapult_ex", "iono"]
    decks = [(n, load_deck(os.path.join(REPO_ROOT, "decks", f"{n}.csv"))) for n in deck_names]

    # 1) SB3's own contract check (steps the env with RAW action-space samples, so it also
    # exercises the invalid-action substitution path).
    from stable_baselines3.common.env_checker import check_env
    env = CGEnv(decks)
    check_env(env)
    print("check_env: OK", flush=True)

    # 2) Full masked episodes (uniform over the live mask, which now includes STOP when legal).
    tally = {1.0: 0, -1.0: 0, 0.0: 0}
    for i in range(args.games):
        r, steps, info = run_episode(env, rng)
        assert r in (-1.0, 0.0, 1.0), f"unexpected terminal reward {r!r}"
        assert isinstance(info.get("win_mode"), dict), "terminal info missing win_mode telemetry"
        tally[r] += 1
        print(f"episode {i + 1}/{args.games}: reward={r:+.0f} decisions={steps} "
              f"deck={info['deck_name']} opp={info['opp_name']} mode={info['win_mode'].get('mode')}",
              flush=True)

    # 3) Scripted multi-select micro-drives (plan multiselect.md §6): always-STOP-when-legal
    # exercises the empty-submit decline (battle_select([]) at minCount==0) + early stop;
    # always-pick-max exercises auto-commit at maxCount. Both submit real battle_select lists — an
    # engine-invalid one raises out of env.step and fails the test.
    for _ in range(max(2, args.games)):
        drive_episode(env, _choose_stop)
        drive_episode(env, _choose_max)

    # Multi-select coverage, accumulated across check_env + masked + scripted drives. The cg RNG is
    # unseedable (no pinned run), so assert on the UNION rather than any single run's node draw.
    assert env.ms_optional > 0, "no optional-single [0..1] multi node ever fired"
    assert env.ms_fixed > 0, "no fixed-k (min==max) multi node ever fired"
    assert env.ms_variable > 0, "no true-variable (min<max, max>1) multi node ever fired"
    assert env.ms_single_option > 0, "no single-option [0..1] decline (len==1) node ever fired"
    assert env.ms_empty_submit > 0, ("battle_select([]) (STOP at minCount==0) never exercised — the "
                                     "empty-submit decline path is unverified")
    assert env.overflows == 0, (f"{env.overflows} learnable nodes exceeded N_MAX-1={N_MAX - 1} — "
                                f"measure and raise N_MAX")
    print(f"multi-select coverage: optional={env.ms_optional} fixed={env.ms_fixed} "
          f"variable={env.ms_variable} single_option={env.ms_single_option} "
          f"empty_submit={env.ms_empty_submit}", flush=True)
    env.close()

    # 4) Singleton not leaked: a raw battle outside the env still completes.
    result = play_one(decks[0][1], decks[0][1])
    assert result in (0, 1, 2)
    print(f"post-close raw battle: result={result} (no singleton leak)", flush=True)

    print(f"OK: check_env + {args.games} masked episodes (W/L/D={tally[1.0]}/{tally[-1.0]}/{tally[0.0]}) "
          f"+ multi-select coverage + singleton discipline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
