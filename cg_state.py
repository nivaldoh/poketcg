"""Local-only save/restore bindings for the rebuilt cg engine (state replicator).

The stock Kaggle `libcg.so` only starts fresh battles. A locally rebuilt engine
(`bash utils/build_libcg.sh`, from the gitignored `ptcg_engine/` source) adds
three exports:

  * ``BattleStartSeeded(cards, seed)`` — deterministic battle (deviceRand off,
    fixed mt19937 seed), same return contract as ``BattleStart``.
  * ``SaveState(battle_ptr)`` — full god-view state blob (base64) of the live
    battle, including the RNG cursor. No fog-of-war erase.
  * ``BattleStartFromState(blob, len)`` — new live battle resumed from a blob.

Use them through this module, never through ``cg`` directly:

    CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib" python ...
    import cg_state
    obs, start = cg_state.battle_start_seeded(deck, deck, seed=7)
    blob = cg_state.save_state()
    ...
    obs, start = cg_state.battle_start_from_state(blob)

Everything degrades cleanly on the stock .so: ``has_state_api()`` is False and
the wrappers raise a pointed error instead of crashing. Blobs are raw memory
images — only readable by the exact engine build that wrote them (a magic word
rejects mismatches). LOCAL TOOL ONLY: none of this ships in a submission.

Self-check (needs the rebuilt lib): ``python cg_state.py``.
"""

import ctypes
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CG_LIB_PATH = os.environ.get("CG_LIB_PATH", os.path.join(_REPO_ROOT, "vendor", "cg-lib"))
if _CG_LIB_PATH not in sys.path:
    sys.path.insert(0, _CG_LIB_PATH)

from cg.game import _get_battle_data  # noqa: E402
from cg.sim import Battle, SerialData, StartData, lib  # noqa: E402

_configured = False


def has_state_api() -> bool:
    """True when the loaded libcg.so carries the local save/restore exports."""
    try:
        lib.SaveState, lib.BattleStartFromState, lib.BattleStartSeeded
        return True
    except AttributeError:
        return False


def _configure() -> None:
    """Set restype/argtypes once; raise a pointed error on the stock .so."""
    global _configured
    if _configured:
        return
    if not has_state_api():
        raise RuntimeError(
            f"the libcg.so loaded from {_CG_LIB_PATH!r} lacks the state API; "
            "build the local engine (bash utils/build_libcg.sh) and set "
            'CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib"'
        )
    lib.SaveState.restype = SerialData
    lib.SaveState.argtypes = [ctypes.c_void_p]
    lib.BattleStartFromState.restype = StartData
    lib.BattleStartFromState.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.BattleStartSeeded.restype = StartData
    lib.BattleStartSeeded.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
    _configured = True


def save_state() -> str:
    """Full-state blob (base64 str) of the live battle at its current decision."""
    _configure()
    sd = lib.SaveState(Battle.battle_ptr)
    if not sd.count:
        raise ValueError("SaveState failed (no live battle?)")
    # sd.json is NULL for SaveState; only data/count/selectPlayer are set.
    return ctypes.string_at(sd.data, sd.count).decode("ascii")


def battle_start_seeded(deck0: list[int], deck1: list[int], seed: int) -> tuple[dict | None, StartData]:
    """Deterministic `cg.game.battle_start`: fixed engine seed, no device RNG."""
    _configure()
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")
    cards = deck0 + deck1
    arg = (ctypes.c_int * len(cards))(*cards)
    start_data = lib.BattleStartSeeded(arg, ctypes.c_uint(seed))
    Battle.battle_ptr = start_data.battlePtr
    if not Battle.battle_ptr:
        return (None, start_data)
    return (_get_battle_data(), start_data)


def battle_start_from_state(blob: str) -> tuple[dict | None, StartData]:
    """Start a live battle from a `save_state` blob, at the saved decision point.

    The cg wrapper's battle is a process-global singleton: `battle_finish()`
    any live battle first. errorType 100 means the blob failed the build-magic
    check (wrong/rebuilt engine, or corrupt data).
    """
    _configure()
    raw = blob.encode("ascii")
    start_data = lib.BattleStartFromState(raw, len(raw))
    Battle.battle_ptr = start_data.battlePtr
    if not Battle.battle_ptr:
        return (None, start_data)
    return (_get_battle_data(), start_data)


def _self_check() -> int:
    """Seeded battle -> save -> restore -> byte-identical re-save."""
    from cg.game import battle_finish, battle_select

    sys.path.insert(0, os.path.join(_REPO_ROOT, "utils"))
    from run_smoketest_battle import load_deck, random_agent

    import random

    random.seed(0)
    deck = load_deck(os.path.join(_REPO_ROOT, "deck.csv"))

    obs, start = battle_start_seeded(deck, deck, seed=123)
    assert obs is not None, f"seeded start failed: {start.errorType}"
    for _ in range(3):
        obs = battle_select(random_agent(obs))
    blob = save_state()
    battle_finish()

    obs2, start2 = battle_start_from_state(blob)
    assert obs2 is not None, f"restore failed: errorType={start2.errorType}"
    assert obs2["search_begin_input"] == obs["search_begin_input"], "fog blob diverged"
    blob2 = save_state()
    battle_finish()
    assert blob2 == blob, (
        f"round-trip blob mismatch: {len(blob)} vs {len(blob2)} chars"
    )
    print(f"OK: save->restore->save round-trip byte-identical ({len(blob)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
