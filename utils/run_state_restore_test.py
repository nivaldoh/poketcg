#!/usr/bin/env python3
"""Acceptance test for the engine state replicator (save/restore of live battles).

Records one deterministic battle (BattleStartSeeded via cg_state) saving a
full-state blob at every decision, then proves restore correctness in layers:

  (a) round-trip byte equality  — restore blob[N], re-save, compare bytes; and
      the restored fog obs (search_begin_input) must equal the recorded one.
  (b) god-view equality         — the restored battle's first visualize_data()
      snapshot matches the recorded one at step N, except the `logs` and
      `selected` keys: the engine advances its log cursor (nextLogStart) while
      producing the obs we save AFTER, so the first post-restore snapshot
      legitimately re-reports no logs / no previous selection. Everything else
      (`select`, `current`) must match exactly.
  (c) deterministic trajectory  — restore at mid-game, feed the recorded
      actions for the remaining steps: every obs (fog blob included), every
      later god view, and the terminal result must be identical.
  (d) smoke                     — restore at mid-game, play to completion with
      a random agent, expect a valid terminal result.

Needs the locally rebuilt engine:

    bash utils/build_libcg.sh
    CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib" python utils/run_state_restore_test.py
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import cg_state  # noqa: E402  (also puts CG_LIB_PATH on sys.path)
from cg.game import battle_finish, battle_select, visualize_data  # noqa: E402
from run_smoketest_battle import load_deck, random_agent  # noqa: E402


def record_seeded_game(deck: list[int], seed: int) -> dict:
    """Play one seeded battle to the end, saving a state blob at every step."""
    obs, start = cg_state.battle_start_seeded(deck, deck, seed)
    assert obs is not None, f"seeded battle_start failed: errorType={start.errorType}"

    blobs, obses, actions = [], [], []
    while obs["current"]["result"] < 0:
        blobs.append(cg_state.save_state())
        obses.append(obs)
        action = random_agent(obs)
        actions.append(action)
        obs = battle_select(action)
    final_blob = cg_state.save_state()
    god = json.loads(visualize_data())  # one snapshot per decision (+ terminal)
    result = obs["current"]["result"]
    battle_finish()

    assert len(god) == len(blobs) + 1, f"{len(god)} god views vs {len(blobs)} steps"
    return {
        "blobs": blobs, "obses": obses, "actions": actions,
        "final_blob": final_blob, "final_obs": obs, "god": god, "result": result,
    }


def normalized_god(entry: dict) -> dict:
    """A god-view snapshot minus the two keys that legitimately differ on the
    first post-restore snapshot (see module docstring)."""
    return {k: v for k, v in entry.items() if k not in ("logs", "selected")}


def check_roundtrip(rec: dict, n: int) -> None:
    """(a) restore at step n -> byte-identical re-save + identical fog obs."""
    obs, start = cg_state.battle_start_from_state(rec["blobs"][n])
    assert obs is not None, f"restore at step {n} failed: errorType={start.errorType}"
    blob2 = cg_state.save_state()
    fog_ok = obs["search_begin_input"] == rec["obses"][n]["search_begin_input"]
    battle_finish()
    assert blob2 == rec["blobs"][n], f"step {n}: round-trip blob differs"
    assert fog_ok, f"step {n}: restored search_begin_input differs"
    print(f"  step {n:3d}: blob round-trip byte-identical, fog obs identical")


def check_godview(rec: dict, n: int) -> None:
    """(b) restored first god view == recorded god[n] modulo logs/selected."""
    obs, _ = cg_state.battle_start_from_state(rec["blobs"][n])
    assert obs is not None
    god = json.loads(visualize_data())
    battle_finish()
    assert len(god) == 1, f"expected 1 fresh god view, got {len(god)}"
    want, got = normalized_god(rec["god"][n]), normalized_god(god[0])
    assert got == want, f"step {n}: god view differs beyond logs/selected"
    print(f"  step {n:3d}: god view identical (logs/selected excluded by design)")


def check_trajectory(rec: dict, n: int) -> None:
    """(c) restore at step n, replay recorded actions -> identical everything."""
    obs, _ = cg_state.battle_start_from_state(rec["blobs"][n])
    assert obs is not None
    steps = len(rec["actions"])
    for k in range(n, steps):
        obs = battle_select(rec["actions"][k])
        want = rec["obses"][k + 1] if k + 1 < steps else rec["final_obs"]
        assert obs == want, f"trajectory diverged at step {k + 1} (restored from {n})"
    god = json.loads(visualize_data())
    result = obs["current"]["result"]
    battle_finish()
    assert result == rec["result"], f"terminal result {result} != recorded {rec['result']}"
    # god[0] is the fresh post-restore snapshot (layer-b semantics); later
    # snapshots must match the recording exactly, logs included.
    for i in range(1, len(god)):
        assert god[i] == rec["god"][n + i], f"god view {i} differs after restore at {n}"
    print(
        f"  restore at step {n} + {steps - n} recorded actions -> identical obs/god "
        f"trajectory, same result ({result})"
    )


def check_smoke(rec: dict, n: int, agent_seed: int) -> None:
    """(d) restore at step n and let a random agent finish the game."""
    random.seed(agent_seed)
    obs, _ = cg_state.battle_start_from_state(rec["blobs"][n])
    assert obs is not None
    moves = 0
    while obs["current"]["result"] < 0:
        obs = battle_select(random_agent(obs))
        moves += 1
    result = obs["current"]["result"]
    battle_finish()
    assert result in (0, 1, 2), f"invalid terminal result {result}"
    print(f"  restore at step {n} + {moves} fresh random moves -> valid result ({result})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="engine + agent seed")
    parser.add_argument(
        "--deck", default=os.path.join(REPO_ROOT, "deck.csv"),
        help="60-card deck CSV used for both players",
    )
    args = parser.parse_args()

    if not cg_state.has_state_api():
        sys.exit(
            "the loaded libcg.so lacks the state API — build it with "
            'utils/build_libcg.sh and set CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib"'
        )

    deck = load_deck(args.deck)
    random.seed(args.seed)
    rec = record_seeded_game(deck, args.seed)
    steps = len(rec["blobs"])
    mid = steps // 2
    probe = sorted({0, mid, steps - 1})
    print(f"recorded seeded game: {steps} decisions, result={rec['result']}")

    print("(a) round-trip byte equality")
    for n in probe:
        check_roundtrip(rec, n)
    print("(a) terminal blob restore")
    obs, start = cg_state.battle_start_from_state(rec["final_blob"])
    assert obs is not None and obs["current"]["result"] == rec["result"]
    battle_finish()
    print(f"  terminal state restores with result={rec['result']}")

    print("(b) god-view equality at the restore step")
    for n in probe:
        check_godview(rec, n)

    print("(c) deterministic trajectory replay")
    check_trajectory(rec, mid)

    print("(d) restore-and-play-on smoke")
    check_smoke(rec, mid, agent_seed=args.seed + 1)

    print("OK: all state-restore layers passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
