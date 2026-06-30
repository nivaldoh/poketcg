#!/usr/bin/env python3
"""Local end-to-end battle smoke test.

Plays full Pokemon TCG battles between two trivial random agents using the
vendored `cg` simulator and asserts that every battle reaches a terminal
result (win / loss / draw). This is the smallest verifiable signal that the
engine runs locally, as called for by the README's "verifiable results" goal.

The `cg` SDK is proprietary and not committed to the repo (see .gitignore). It
lives under vendor/cg-lib/ after being downloaded from the Kaggle competition
`pokemon-tcg-ai-battle`. Override its location with the CG_LIB_PATH env var.

Usage (run from the repo root):
    python utils/run_smoketest_battle.py            # one battle
    python utils/run_smoketest_battle.py --games 5  # five battles, prints a win/loss/draw tally
"""

import argparse
import os
import random
import sys

# This script lives in utils/; defaults (vendor dir, deck.csv) are anchored to
# the repo root, one level up, so they resolve no matter the working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Replace the Kaggle `glob('/kaggle/input/**/cg-lib')` path injection with a
# local vendor dir (overridable), then import the simulator.
CG_LIB_PATH = os.environ.get("CG_LIB_PATH", os.path.join(REPO_ROOT, "vendor", "cg-lib"))
sys.path.insert(0, CG_LIB_PATH)

try:
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
except ImportError as e:  # pragma: no cover - setup guidance
    sys.exit(
        f"Could not import the `cg` simulator from {CG_LIB_PATH!r}: {e}\n"
        "Download it from the Kaggle competition 'pokemon-tcg-ai-battle' "
        "(sample_submission/cg/) into vendor/cg-lib/cg/, or set CG_LIB_PATH."
    )


def load_deck(path: str) -> list[int]:
    """Read a 60-card deck (one card ID per line) from a CSV file."""
    with open(path) as f:
        deck = [int(line) for line in f.read().split("\n") if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 card IDs, found {len(deck)}")
    return deck


def random_agent(obs_dict: dict) -> list[int]:
    """Pick `maxCount` distinct legal option indices at random.

    Mirrors the reference agent: in the battle_start-driven loop `obs.select`
    is always populated (the `select is None` deck-return path only occurs in
    the Kaggle submission flow), so we never need to return a deck here.
    """
    obs = to_observation_class(obs_dict)
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def play_one(deck0: list[int], deck1: list[int]) -> int:
    """Run a single battle to completion and return its terminal result.

    Result codes: 0 -> player 0 won, 1 -> player 1 won, 2 -> draw.
    """
    obs, start_data = battle_start(deck0, deck1)
    if start_data.errorPlayer >= 0 or obs is None:
        raise ValueError(
            f"Deck rejected by engine: errorPlayer={start_data.errorPlayer} "
            f"errorType={start_data.errorType}"
        )

    while obs["current"]["result"] < 0:
        obs = battle_select(random_agent(obs))

    battle_finish()
    result = obs["current"]["result"]
    assert result in (0, 1, 2), f"unexpected terminal result {result!r}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1, help="number of battles to play")
    parser.add_argument(
        "--deck",
        default=os.path.join(REPO_ROOT, "deck.csv"),
        help="path to a 60-card deck CSV (used for both players)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the agents")
    args = parser.parse_args()

    random.seed(args.seed)
    deck = load_deck(args.deck)

    tally = {0: 0, 1: 0, 2: 0}
    for i in range(args.games):
        result = play_one(deck, deck)
        tally[result] += 1
        label = "draw" if result == 2 else f"player {result} won"
        print(f"battle {i + 1}/{args.games}: result={result} ({label})", flush=True)

    if args.games > 1:
        print(
            f"\nTally over {args.games} battles -> "
            f"player0: {tally[0]}, player1: {tally[1]}, draws: {tally[2]}"
        )
    print("OK: every battle reached a terminal result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
