#!/usr/bin/env python3
"""Eval a train_sb3.py MaskablePPO checkpoint across the deck×opponent matrix — the SB3-stack
analog of eval_ppo.py (a PARALLEL path: eval_ppo hardcodes ppo_model.ActorCritic and its manifest
arch string, so it is not modified). Minimal first pass: matrix cells + Wilson CIs only (no
baseline anchor / manifest / json — deferred with the rest of telemetry parity).

The model plays EACH --decks deck against EACH rule-based opponent (its own deck), seat-swapped
via eval.play_match. The diagonal (same deck both seats) is the clean MIRROR skill read;
off-diagonal cells are deck-confounded (eval-discipline). Loads the SB3 zip (training-side —
importing SB3 here is fine, the submission boundary only constrains main.py).

Decision routing matches training exactly: non-policy nodes (and >N_MAX overflow nodes) are
auto_resolve'd; policy nodes featurize via sb3_env.obs_to_gym (the SAME code path CGEnv uses) and
go through MaskablePPO.predict with the action mask. `sampled` mode is the honest headline for a
stochastic policy; `greedy` = argmax (eval-discipline, same convention as eval_ppo).

    python eval_sb3.py --ckpt out/sb3ppo_sb3.zip --games 20            # smoke (below the n>=200 CI floor)
    python eval_sb3.py --ckpt out/sb3ppo_sb3.zip --games 200           # trustworthy CIs
"""

import argparse
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# sb3_env owns the CG_LIB_PATH import-order discipline (it imports tcg_model first); importing
# `eval` also inserts game_agents/rulebased/ on sys.path (idempotent).
from sb3_env import obs_to_gym, N_MAX                                  # noqa: E402
from tcg_model import is_policy_select, auto_resolve                   # noqa: E402
from utils.run_smoketest_battle import load_deck                       # noqa: E402
from cg.api import to_observation_class                                # noqa: E402
from eval import (Agent, play_match, wilson,                           # noqa: E402
                  make_rulebased, make_abomasnow, make_dragapult, make_iono)
from sb3_contrib import MaskablePPO                                    # noqa: E402

# Same registry shape as eval_ppo.OPPONENTS: keys match deck names so the matrix diagonal is a
# true mirror (eval.py names the Lucario factory make_rulebased).
OPPONENTS = {
    "lucario": make_rulebased,
    "abomasnow": make_abomasnow,
    "dragapult_ex": make_dragapult,
    "iono": make_iono,
}
DECK_NAMES = ["lucario", "abomasnow", "dragapult_ex", "iono"]


def sb3_agent(model, greedy=False):
    """Wrap a MaskablePPO as an agent_fn(obs_dict) -> list[int] with the training-time decision
    routing: auto_resolve non-policy/overflow nodes, masked predict on policy nodes."""
    def agent_fn(obs_dict):
        o = to_observation_class(obs_dict)
        sel = o.select
        if not is_policy_select(sel) or len(sel.option) > N_MAX:
            return auto_resolve(sel)
        gobs = obs_to_gym(o)
        action, _ = model.predict(gobs, action_masks=gobs["action_mask"].astype(bool),
                                  deterministic=greedy)
        return [int(action)]
    return agent_fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "out", "sb3ppo_sb3.zip"),
                    help="MaskablePPO zip saved by train_sb3.py")
    ap.add_argument("--decks", default=",".join(DECK_NAMES),
                    help="comma list of model-side decks (decks/<name>.csv)")
    ap.add_argument("--opponents", default=",".join(OPPONENTS),
                    help=f"comma list of {list(OPPONENTS)}")
    ap.add_argument("--mode", default="sampled", choices=["sampled", "greedy"],
                    help="sampled = the honest headline for a stochastic policy (eval-discipline)")
    ap.add_argument("--games", type=int, default=200,
                    help="seat-swapped games per cell; >=200 for a trustworthy CI (eval-discipline)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    deck_names = [s.strip() for s in args.decks.split(",") if s.strip()]
    opp_names = [s.strip() for s in args.opponents.split(",") if s.strip()]
    bad = [n for n in deck_names if n not in DECK_NAMES] + [n for n in opp_names if n not in OPPONENTS]
    if bad:
        raise SystemExit(f"unknown deck/opponent name(s): {bad}")

    model = MaskablePPO.load(args.ckpt, device="auto")
    model_decks = {n: load_deck(os.path.join(_REPO_ROOT, "decks", f"{n}.csv")) for n in deck_names}
    opponents = [(n, OPPONENTS[n]()) for n in opp_names]
    fn = sb3_agent(model, greedy=(args.mode == "greedy"))

    warn = "" if args.games >= 200 else "  (WARNING: <200 decisive is below the trustworthy CI floor)"
    print(f"SB3 deck×opponent matrix | ckpt={args.ckpt} | mode={args.mode} | "
          f"{args.games} games/cell{warn} | seed={args.seed}", flush=True)

    grid = {d: {} for d in deck_names}
    for dname in deck_names:
        me = Agent(f"sb3:{dname}:{args.mode}", fn, model_decks[dname])
        for oname, opp in opponents:
            t = time.time()
            random.seed(args.seed)   # comparable opponent RNG stream per cell (engine RNG stays uncontrolled)
            w, l, d = play_match(me, opp, args.games)
            p, lo, hi = wilson(w, w + l)
            grid[dname][oname] = (w, l, d, p, lo, hi)
            mirror = "*" if dname == oname else " "
            print(f"  {dname:12s} vs {oname:12s}{mirror}: {w}/{l}/{d}  "
                  f"win%={100 * p:5.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f})  "
                  f"[{time.time() - t:.0f}s]", flush=True)

    col_w = 16
    print(f"\n=== win% (95% CI, decisive; * = mirror diagonal) | {args.mode} ===", flush=True)
    print("model\\opp".ljust(14) + "".join(o.ljust(col_w) for o in opp_names), flush=True)
    for dname in deck_names:
        row = dname.ljust(14)
        for oname in opp_names:
            w, l, d, p, lo, hi = grid[dname][oname]
            mark = "*" if dname == oname else " "
            row += (f"{100 * p:4.1f}({100 * lo:2.0f}-{100 * hi:2.0f})" + mark).ljust(col_w)
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
