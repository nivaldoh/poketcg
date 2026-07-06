#!/usr/bin/env python3
"""Smoke test for sb3_env.CGEnv: SB3's env_checker plus full masked episodes.

Asserts (per the README's "verifiable results" goal, the smallest end-to-end signal):
  1. `stable_baselines3.common.env_checker.check_env` passes (warnings about unbounded Box
     features are expected and fine; errors are not).
  2. N full episodes driven by uniform-over-`action_masks()` actions each reach a terminal with
     reward in {-1, 0, +1}, never set info["invalid_action"] (masked play can't go out of range),
     and carry the win_mode telemetry dict.
  3. After close(), a fresh raw battle still runs — the process-global cg battle singleton was
     not leaked.

Usage (from the repo root):
    python utils/run_smoketest_sb3_env.py             # check_env + 3 episodes
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
from sb3_env import CGEnv, N_MAX                              # noqa: E402
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

    # 2) Full masked episodes.
    tally = {1.0: 0, -1.0: 0, 0.0: 0}
    for i in range(args.games):
        r, steps, info = run_episode(env, rng)
        assert r in (-1.0, 0.0, 1.0), f"unexpected terminal reward {r!r}"
        assert isinstance(info.get("win_mode"), dict), "terminal info missing win_mode telemetry"
        tally[r] += 1
        print(f"episode {i + 1}/{args.games}: reward={r:+.0f} decisions={steps} "
              f"deck={info['deck_name']} opp={info['opp_name']} mode={info['win_mode'].get('mode')}",
              flush=True)
    assert env.overflows == 0, f"{env.overflows} policy nodes exceeded N_MAX={N_MAX} — measure and raise it"
    env.close()

    # 3) Singleton not leaked: a raw battle outside the env still completes.
    result = play_one(decks[0][1], decks[0][1])
    assert result in (0, 1, 2)
    print(f"post-close raw battle: result={result} (no singleton leak)", flush=True)

    print(f"OK: check_env + {args.games} masked episodes (W/L/D={tally[1.0]}/{tally[-1.0]}/{tally[0.0]}) "
          f"+ singleton discipline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
