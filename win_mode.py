"""Terminal-observation win-mode classifier — the ONE shared "how did this game end" reader.

The cg engine reports WHY a game ended only in the battle log: the terminal observation's
``obs["logs"]`` contains an entry with ``type == LogType.RESULT`` (=23) carrying a ``reason``
code (grep ``reason (int)`` in vendor/cg-lib/cg/api.py):

    1 = prize-out  (a player took their last Prize card)
    2 = deck-out   (a player had to start their turn with 0 deck cards — that player LOSES)
    3 = bench-out  (a player had no Pokémon to promote to the Active Spot — that player LOSES)
    4 = card effect (a card's text ended the game; not inferable from counts)

``State.result`` (``obs["current"]["result"]``) only says WHO won (0/1 = player index,
2 = draw). Every driver loop in this repo holds the terminal obs (the loop condition is
``result < 0``); this module is where it finally gets read. Used by ``sb3_env.CGEnv.step``
(-> the terminal info dict) and ``eval._play_one`` (-> end-stats cells); ``summarize_ends``
is the ONE aggregation of those dicts.

Consistency channel: ``mode`` (log-derived) and ``mode_counts`` (inferred from terminal
prize/deck/in-play counts) are BOTH kept permanently. Terminal counts are read at the terminal
obs, not at a select — the CARD-select prize transients of work_log DISCOVERY 1 (2026-07-03)
are a per-decision phenomenon — but terminal-count trust is otherwise unmeasured, so
``mode == mode_counts`` agreement (where inferable) is the standing detector for either channel
drifting. Smoke recipe: grep ``endsmoke`` in work_log.

Seat conventions (all fields seat-relative to the ``seat`` argument):
    win        : +1 seat won, -1 seat lost, 0 draw
    prize_diff : opp_prizes_left - my_prizes_left, int in [-6, 6]; POSITIVE = seat is AHEAD on
                 prizes ((opp - own), so higher = better for seat).
                 A "prize-trailing win" is ``win == 1 and prize_diff < 0``.

Local-harness side only — never part of the Kaggle submission bundle (main.py + cg/ + deck.csv).
"""

import os
import sys

# Same idempotent local-SDK path insert as utils/run_smoketest_battle.py (this module lives at
# the repo root, so no dirname hop). Import cheaply: cg.api only (dataclass defs), no torch.
CG_LIB_PATH = os.environ.get(
    "CG_LIB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "cg-lib")
)
if CG_LIB_PATH not in sys.path:
    sys.path.insert(0, CG_LIB_PATH)

from cg.api import LogType  # noqa: E402

_RESULT_LOG_TYPE = int(LogType.RESULT)  # 23

# reason code -> mode name (see module docstring). Unknown/new codes -> "unknown".
MODE_NAMES = {1: "prize", 2: "deck", 3: "bench", 4: "effect"}


def _mode_from_counts(win, winner, loser):
    """Count-inferred end mode from the terminal per-player states (dicts).

    Fallback + standing consistency channel for the log-derived ``mode``. Reason 4 (card
    effect) is NOT count-inferable and comes back "unknown" — consistency checks must skip it.
    """
    if win == 0:
        return "draw"
    if len(winner["prize"]) == 0:
        return "prize"
    if loser["deckCount"] == 0:
        return "deck"
    if len(loser["active"]) + len(loser["bench"]) == 0:
        return "bench"
    return "unknown"


def classify_end(obs, seat):
    """Classify a FINISHED game from its terminal raw obs dict, relative to ``seat``.

    Args:
        obs: terminal observation dict (``obs["current"]["result"] >= 0``) as returned by the
             final ``battle_select`` — the driver holds it; the agent never sees it.
        seat: the player index of the side we account for (the learner/model seat).

    Returns a flat dict (see module docstring for conventions):
        win, mode, mode_counts, my_prizes_left, opp_prizes_left, my_deck, opp_deck,
        my_inplay, opp_inplay, prize_diff, turns
    """
    cur = obs["current"]
    result = cur["result"]
    win = 0 if result == 2 else (1 if result == seat else -1)

    me, opp = cur["players"][seat], cur["players"][1 - seat]
    if win == 0:
        winner, loser = me, opp  # unused by _mode_from_counts on draws
    else:
        winner, loser = (me, opp) if win == 1 else (opp, me)
    mode_counts = _mode_from_counts(win, winner, loser)

    reason = None
    for lg in obs["logs"]:
        if lg.get("type") == _RESULT_LOG_TYPE:
            reason = lg.get("reason")
            break
    if reason is not None:
        mode = MODE_NAMES.get(reason, "unknown")
    elif win == 0:
        mode = "draw"
    else:
        mode = mode_counts  # no RESULT log observed — count fallback keeps mode populated

    my_prizes_left, opp_prizes_left = len(me["prize"]), len(opp["prize"])
    return {
        "win": win,
        "mode": mode,
        "mode_counts": mode_counts,
        "my_prizes_left": my_prizes_left,
        "opp_prizes_left": opp_prizes_left,
        "my_deck": me["deckCount"],
        "opp_deck": opp["deckCount"],
        "my_inplay": len(me["active"]) + len(me["bench"]),
        "opp_inplay": len(opp["active"]) + len(opp["bench"]),
        "prize_diff": opp_prizes_left - my_prizes_left,
        "turns": cur["turn"],
    }


def summarize_ends(ends):
    """classify_end dicts -> the shared end-stats block (e.g. eval end-stats cells): win/loss mode
    histograms (zero counts omitted — read with .get), trailing_win_n (wins while BEHIND on prizes:
    prize_diff < 0), mean end prize_diff per class (None when the class is empty), and mean length
    over ALL games."""
    win_modes, loss_modes = {}, {}
    pd_w, pd_l = [], []
    trailing = 0
    for e in ends:
        if e["win"] == 1:
            win_modes[e["mode"]] = win_modes.get(e["mode"], 0) + 1
            pd_w.append(e["prize_diff"])
            trailing += e["prize_diff"] < 0
        elif e["win"] == -1:
            loss_modes[e["mode"]] = loss_modes.get(e["mode"], 0) + 1
            pd_l.append(e["prize_diff"])
    return {"win_modes": win_modes, "loss_modes": loss_modes, "trailing_win_n": trailing,
            "prize_diff_win_mean": sum(pd_w) / len(pd_w) if pd_w else None,
            "prize_diff_loss_mean": sum(pd_l) / len(pd_l) if pd_l else None,
            "turns_mean": sum(e["turns"] for e in ends) / len(ends) if ends else None}


def _selftest():
    """Constructed-dict cases: prize-out win, deck-out loss (prize-trailing pattern), draw."""

    def player(prizes, deck, active, bench):
        return {"prize": [None] * prizes, "deckCount": deck,
                "active": [object()] * active, "bench": [object()] * bench}

    def obs(result, p0, p1, logs, turn=30):
        return {"current": {"result": result, "turn": turn, "players": [p0, p1]}, "logs": logs}

    # 1) seat 0 wins by prize-out (took all 6; opponent has 4 left).
    e = classify_end(
        obs(0, player(0, 20, 1, 2), player(4, 25, 1, 1),
            [{"type": 5}, {"type": _RESULT_LOG_TYPE, "result": 0, "reason": 1}]),
        seat=0,
    )
    assert e["win"] == 1 and e["mode"] == "prize" and e["mode_counts"] == "prize", e
    assert e["prize_diff"] == 4 and e["my_prizes_left"] == 0 and e["turns"] == 30, e

    # 2) seat 0 LOSES by deck-out while AHEAD on prizes (the DISCOVERY-2 mirror image):
    #    seat 0 has 0 deck; prize_diff = opp_left - my_left = 5 - 2 = +3 (ahead, still lost).
    e = classify_end(
        obs(1, player(2, 0, 1, 3), player(5, 12, 1, 2),
            [{"type": _RESULT_LOG_TYPE, "result": 1, "reason": 2}]),
        seat=0,
    )
    assert e["win"] == -1 and e["mode"] == "deck" and e["mode_counts"] == "deck", e
    assert e["prize_diff"] == 3 and e["my_deck"] == 0 and e["opp_inplay"] == 3, e
    # Same terminal obs read from the OTHER seat: a prize-trailing deck-out WIN.
    e1 = classify_end(
        obs(1, player(2, 0, 1, 3), player(5, 12, 1, 2),
            [{"type": _RESULT_LOG_TYPE, "result": 1, "reason": 2}]),
        seat=1,
    )
    assert e1["win"] == 1 and e1["mode"] == "deck" and e1["prize_diff"] == -3, e1

    # 3) draw, no RESULT log at all -> mode falls back cleanly.
    e = classify_end(obs(2, player(3, 10, 1, 1), player(3, 11, 1, 1), []), seat=0)
    assert e["win"] == 0 and e["mode"] == "draw" and e["mode_counts"] == "draw", e

    # 4) reason 4 (card effect), counts non-terminal-looking -> mode "effect", counts "unknown".
    e = classify_end(
        obs(0, player(2, 9, 1, 1), player(3, 8, 1, 1),
            [{"type": _RESULT_LOG_TYPE, "result": 0, "reason": 4}]),
        seat=0,
    )
    assert e["mode"] == "effect" and e["mode_counts"] == "unknown", e

    # 5) summarize_ends: mixed set — prize win, prize-trailing deck win, deck loss, draw.
    s = summarize_ends([
        {"win": 1, "mode": "prize", "prize_diff": 4, "turns": 30},
        {"win": 1, "mode": "deck", "prize_diff": -3, "turns": 40},
        {"win": -1, "mode": "deck", "prize_diff": 3, "turns": 40},
        {"win": 0, "mode": "draw", "prize_diff": 0, "turns": 50},
    ])
    assert s["win_modes"] == {"prize": 1, "deck": 1} and s["loss_modes"] == {"deck": 1}, s
    assert s["trailing_win_n"] == 1 and s["prize_diff_win_mean"] == 0.5, s
    assert s["prize_diff_loss_mean"] == 3 and s["turns_mean"] == 40.0, s
    empty = summarize_ends([])
    assert empty["win_modes"] == {} and empty["turns_mean"] is None, empty

    print("win_mode selftest OK")


if __name__ == "__main__":
    _selftest()
