#!/usr/bin/env python3
"""Re-instantiate a LIVE cg battle from a recorded trace.json game state.

Takes a trace produced by `record_battle.py --record-state` (each step carries a
full-state engine blob) and restores the battle at any recorded decision, then
either:

  --replay             re-feed the trace's own remaining actions and verify the
                       engine reproduces the recorded game exactly (obs + god
                       views + result). Only meaningful for traces recorded
                       with --seed: an unseeded trace ran on hardware RNG, so
                       its future can't be re-rolled — replay then reports
                       divergence rather than a defect.
  --p0 SPEC --p1 SPEC  play the position out with fresh agents (any
                       record_battle spec: random|rulebased|iono|<*.zip>|...),
                       one policy per engine seat. The decks in play are the
                       restored state's — agent specs only supply decisions.

Needs the locally rebuilt engine:
    bash utils/build_libcg.sh
    CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib" python utils/resume_battle.py \
        --trace t.json --step 30 --replay

    python utils/resume_battle.py --trace t.json --step 30 --p0 random --p1 iono
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import cg_state  # noqa: E402  (also puts CG_LIB_PATH on sys.path)
from cg.game import battle_finish, battle_select, visualize_data  # noqa: E402


def _strip(obs: dict) -> dict:
    """Recorded step.obs has search_begin_input dropped — compare like with like."""
    return {k: v for k, v in obs.items() if k != "search_begin_input"}


def _god_key(entry: dict) -> dict:
    """God-view snapshot minus the keys that legitimately differ on the first
    post-restore snapshot (the engine's log cursor had already advanced when the
    blob was saved — same normalization as utils/run_state_restore_test.py)."""
    return {k: v for k, v in entry.items() if k not in ("logs", "selected")}


def replay(game: dict, n: int) -> int:
    """Restore at step n, feed the trace's remaining actions, diff everything."""
    steps = game["steps"]
    header = game["header"]
    obs, start = cg_state.battle_start_from_state(steps[n]["state_blob"])
    if obs is None:
        raise SystemExit(f"restore failed: errorType={start.errorType} (wrong engine build?)")

    failures = []
    for k in range(n, len(steps)):
        obs = battle_select(steps[k]["action"])
        if k + 1 < len(steps) and _strip(obs) != steps[k + 1]["obs"]:
            failures.append(f"obs diverged at step {k + 1}")
            break

    god = json.loads(visualize_data())
    result = obs["current"]["result"]
    battle_finish()

    if not failures:
        # god[0] is the fresh post-restore snapshot (normalized); later ones and
        # the terminal snapshot must match the recording exactly.
        if _god_key(god[0]) != _god_key(steps[n]["god"]):
            failures.append(f"god view at restore step {n} differs")
        for i in range(1, len(god) - 1):
            if n + i < len(steps) and god[i] != steps[n + i]["god"]:
                failures.append(f"god view diverged at step {n + i}")
                break
        if god[-1] != game["final_god"]:
            failures.append("terminal god view differs")
        if result != header["result"]:
            failures.append(f"result {result} != recorded {header['result']}")

    played = len(steps) - n
    if failures:
        hint = "" if header.get("seed") else " (trace is unseeded: divergence is expected, not a defect)"
        print(f"REPLAY FAIL: restored at step {n}, {played} actions: " + "; ".join(failures) + hint)
        return 1
    print(f"REPLAY PASS: restored at step {n}, {played} recorded actions reproduced the game "
          f"exactly (result={result}, {len(god) - 1} board snapshots verified)")
    return 0


def play_on(game: dict, n: int, p0: str, p1: str, deck: str, greedy: bool) -> int:
    """Restore at step n and let fresh agents finish the game."""
    from record_battle import make_agent  # lazy: pulls torch/sb3
    from win_mode import classify_end

    agents = [make_agent(p0, deck, greedy), make_agent(p1, deck, greedy)]
    steps = game["steps"]
    obs, start = cg_state.battle_start_from_state(steps[n]["state_blob"])
    if obs is None:
        raise SystemExit(f"restore failed: errorType={start.errorType} (wrong engine build?)")

    moves = 0
    while obs["current"]["result"] < 0:
        obs = battle_select(agents[obs["current"]["yourIndex"]].fn(obs))
        moves += 1
    result = obs["current"]["result"]
    end = classify_end(obs, 0)  # seat-0 relative
    battle_finish()
    assert result in (0, 1, 2), f"invalid terminal result {result}"

    who = {0: f"seat 0 ({agents[0].name})", 1: f"seat 1 ({agents[1].name})", 2: "draw"}[result]
    print(f"PLAY-ON: restored at step {n} (of {len(steps)}), {moves} fresh moves -> "
          f"result={result} [{who}] end={end['mode']} turns={end['turns']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="trace.json from record_battle.py --record-state")
    ap.add_argument("--step", type=int, default=0, help="decision index to restore at (default 0)")
    ap.add_argument("--game", type=int, default=0, help="game index for a session trace (--games N)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", action="store_true",
                      help="verify the engine reproduces the recorded game from --step on")
    mode.add_argument("--p0", help="seat-0 agent spec to play the position out (with --p1)")
    ap.add_argument("--p1", help="seat-1 agent spec (agent mode)")
    ap.add_argument("--deck", default="lucario", help="deck name for any SB3 .zip agent spec")
    ap.add_argument("--greedy", action="store_true", help="SB3 agents pick argmax instead of sampling")
    args = ap.parse_args()

    if not cg_state.has_state_api():
        raise SystemExit(
            "the loaded libcg.so lacks the state API — build it with utils/build_libcg.sh "
            'and set CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib"'
        )

    with open(args.trace) as f:
        trace = json.load(f)
    game = trace["games"][args.game] if "games" in trace else trace
    steps = game["steps"]
    if not 0 <= args.step < len(steps):
        raise SystemExit(f"--step {args.step} out of range (trace has {len(steps)} decisions)")
    if "state_blob" not in steps[args.step]:
        raise SystemExit("no state_blob at that step — record the trace with --record-state")

    if args.replay:
        return replay(game, args.step)
    if not args.p1:
        raise SystemExit("agent mode needs both --p0 and --p1")
    return play_on(game, args.step, args.p0, args.p1, args.deck, args.greedy)


if __name__ == "__main__":
    raise SystemExit(main())
