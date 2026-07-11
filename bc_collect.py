#!/usr/bin/env python3
"""bc_collect — teacher-vs-teacher games -> the behavior-cloning decision dataset.

The rule-based teachers beat the learner in all 16 eval cells, so they are free per-decision
supervision (docs/plans/plan.md §4.1). This script plays all 16 ordered archetype pairings (mirrors included),
seat-swapped per game, and records EVERY node that passes the learner's own routing
(`is_learnable_select` + the N_MAX-1 overflow cap — the exact predicate CGEnv/sb3_agent use), from
BOTH seats:

    {game_id, seat, deck, obs (trimmed raw dict), picks, min_count, max_count, outcome}

Design rules (each is load-bearing, see docs/plans/plan.md §4.1 + §7):
  - ONE teacher call per node — dragapult_ex accumulates obs.logs into module globals and lucario
    keeps sticky per-turn flags, so a second call on the same node corrupts them. The single call's
    return feeds BOTH the recorded sample and battle_select.
  - Fresh per-seat teacher module instances (importlib): in a mirror pairing one shared module
    would leak those globals ACROSS SEATS (a latent corruption eval.py's mirror cells live with;
    demonstrations shouldn't). Instances are reused across a pairing's games — the teachers reset
    their per-game state on their own turn-0 detection (the make_opponent contract).
  - Store the RAW trimmed obs, not featurized arrays — immune to featurization drift, keeps
    pick-order augmentation free. `_trim_obs` (credit_probe pattern) drops obs["logs"]: verified
    safe, no featurization path reads o.logs; the teacher's decision was already made live.
  - outcome = seat-relative {+1 win, -1 loss, 0 draw}, byte-identical to CGEnv._commit's G (the
    BC critic label IS PPO's gamma=1.0 return-to-go).
  - The cg battle is a process-global singleton: one live battle, battle_finish in finally.

Output: one pickle per pairing at {out_dir}/{a}_vs_{b}.pkl (a list of sample dicts) — sharded so
bc_train can stream. Every shard passes the replay self-test before it is written: the recorded
picks re-driven through the sub-step decomposition are legal under `_ms_mask` at every prefix,
STOP is legal exactly where the env would emit it, and (sampled) `obs_to_gym` featurizes each
prefix. This is the anti-drift guarantee that the offline data can replay 1:1 as env sub-steps.

`--teachers` restricts the pairing loop (e.g. `--teachers lucario` = the one mirror pairing —
the single-cell campaign). `--games-per-shard N` flushes {a}_vs_{b}_{NNN}.pkl chunks every N
games instead of one pairing-sized pickle: both this collector (raw dict rows in RAM) and
bc_train's per-shard unpickle scale with the largest shard, so large single-pairing collections
MUST chunk to keep RAM flat (7GB box). game_id stays globally increasing across chunks, so
bc_train's by-game holdout hash split is unaffected.

Smoke (2 games/pairing + FULL replay+featurize check + bytes/row):   python bc_collect.py --smoke
Real (default 125 games/pairing = 2000 games, ~75k decisions):        python bc_collect.py
"""

import argparse
import importlib.util
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# sb3_env owns the CG_LIB_PATH import-order discipline (it imports tcg_model first).
from sb3_env import is_learnable_select, _ms_mask, obs_to_gym, N_MAX, STOP_INDEX  # noqa: E402
from cg.api import to_observation_class                                # noqa: E402
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402

TEACHERS = ["lucario", "abomasnow", "dragapult_ex", "iono"]
_RB_DIR = os.path.join(_REPO_ROOT, "game_agents", "rulebased")


def fresh_teacher(name, tag):
    """A FRESH module instance of game_agents/rulebased/{name}.py (own globals — see docstring).
    `tag` uniquifies the module name so two instances of the same teacher never collide."""
    spec = importlib.util.spec_from_file_location(f"bc_teacher_{name}_{tag}",
                                                  os.path.join(_RB_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trim_obs(obs_dict):
    """Drop the bulky per-step fields the offline consumers never read (pattern lifted from
    credit_probe._trim_obs). logs is emptied, NOT omitted, so to_observation_class stays happy."""
    return {"current": obs_dict["current"], "select": obs_dict.get("select"), "logs": [],
            "search_begin_input": obs_dict.get("search_begin_input")}


def play_and_record(agents, decks, game_id, deck_names):
    """One battle; agents/decks/deck_names are seat-indexed [2]. Returns (result, rows) where rows
    are this game's recorded learnable decisions, outcome-labeled. eval._play_one pattern."""
    obs, start = battle_start(decks[0], decks[1])
    if start.errorPlayer >= 0:
        raise ValueError("deck rejected by engine")
    rows = []
    try:
        guard = 0
        while obs["current"]["result"] < 0:
            guard += 1
            if guard > 4000:
                raise RuntimeError("runaway game (>4000 nodes)")
            seat = obs["current"]["yourIndex"]
            sel = to_observation_class(obs).select
            picks = agents[seat](obs)          # THE one teacher call per node (stateful teachers)
            if is_learnable_select(sel) and len(sel.option) <= N_MAX - 1:
                rows.append({"game_id": game_id, "seat": seat, "deck": deck_names[seat],
                             "obs": _trim_obs(obs), "picks": list(picks),
                             "min_count": sel.minCount, "max_count": sel.maxCount})
            obs = battle_select(picks)
        result = obs["current"]["result"]
    finally:
        battle_finish()
    for r in rows:
        r["outcome"] = 0.0 if result == 2 else (1.0 if result == r["seat"] else -1.0)
    return result, rows


def replay_check(rows, featurize_every=1):
    """The B1 anti-drift self-test: every recorded sample must re-drive through the sub-step
    decomposition exactly as CGEnv/bc_train will run it. Every `featurize_every`-th row also runs
    obs_to_gym on each prefix (the expensive half). Returns #featurized rows."""
    feats = 0
    for k, r in enumerate(rows):
        o = to_observation_class(r["obs"])
        sel = o.select
        picks = r["picks"]
        assert len(set(picks)) == len(picks), f"duplicate picks recorded: {picks}"
        assert sel.minCount <= len(picks) <= sel.maxCount, \
            f"pick count {len(picks)} outside [{sel.minCount},{sel.maxCount}]"
        do_feat = (k % featurize_every == 0)
        picked = []
        for a in picks:
            mask = _ms_mask(sel, picked)
            assert mask[a], f"recorded pick {a} illegal under _ms_mask at prefix {picked}"
            if do_feat:
                obs_to_gym(o, picked)
            picked.append(a)
        if len(picked) < sel.maxCount:      # env would emit an explicit STOP here
            mask = _ms_mask(sel, picked)
            assert mask[STOP_INDEX], f"STOP illegal at prefix {picked} (mn={sel.minCount})"
            if do_feat:
                obs_to_gym(o, picked)
        if do_feat:
            feats += 1
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-per-pairing", type=int, default=125,
                    help="games per ordered pairing (16 pairings; default 125 -> 2000 games)")
    ap.add_argument("--teachers", default=",".join(TEACHERS),
                    help="comma list restricting the pairing loop (e.g. 'lucario' -> just the "
                         "lucario mirror)")
    ap.add_argument("--games-per-shard", type=int, default=0,
                    help="flush a {a}_vs_{b}_{NNN}.pkl chunk every N games (0 = one shard per "
                         "pairing); REQUIRED for large single-pairing collections — RAM at both "
                         "collect and bc_train ends scales with the largest shard")
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "out", "bc_data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="2 games/pairing, FULL replay+featurize self-test, bytes/row report")
    args = ap.parse_args()
    random.seed(args.seed)
    if args.smoke:
        args.games_per_pairing = 2
    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    unknown = [t for t in teachers if t not in TEACHERS]
    if unknown:
        raise SystemExit(f"unknown teachers {unknown} (one of {TEACHERS})")
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    total_rows = total_games = total_bytes = 0
    game_id = 0
    for a in teachers:
        for b in teachers:
            mods = {0: fresh_teacher(a, "s0"), 1: fresh_teacher(b, "s1")}
            per_shard = args.games_per_shard or args.games_per_pairing
            shard, wld, chunk, in_shard = [], [0, 0, 0], 0, 0
            for i in range(args.games_per_pairing):
                seat_a = i % 2                              # seat-swap (first-player symmetry)
                agents, decks, names = [None, None], [None, None], [None, None]
                agents[seat_a], decks[seat_a], names[seat_a] = mods[0].agent, mods[0].my_deck, a
                agents[1 - seat_a], decks[1 - seat_a], names[1 - seat_a] = \
                    mods[1].agent, mods[1].my_deck, b
                result, rows = play_and_record(agents, decks, game_id, names)
                wld[0 if result == seat_a else (2 if result == 2 else 1)] += 1
                shard.extend(rows)
                game_id += 1
                in_shard += 1
                if in_shard < per_shard and i < args.games_per_pairing - 1:
                    continue
                # gate the shard on the replay self-test BEFORE writing (featurize all in smoke,
                # every 25th row otherwise — the mask replay itself always covers every row)
                replay_check(shard, featurize_every=1 if args.smoke else 25)
                suffix = f"_{chunk:03d}" if args.games_per_shard else ""
                path = os.path.join(args.out_dir, f"{a}_vs_{b}{suffix}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
                nbytes = os.path.getsize(path)
                total_rows += len(shard)
                total_bytes += nbytes
                print(f"{a:>13} vs {b:<13} games {i + 1 - in_shard}-{i}  W/L/D(seat-a)="
                      f"{wld[0]}/{wld[1]}/{wld[2]} cum  rows={len(shard)}  "
                      f"{nbytes / max(1, len(shard)):.0f} B/row -> {path}", flush=True)
                shard, chunk, in_shard = [], chunk + 1, 0
            total_games += args.games_per_pairing

    dt = time.time() - t0
    print(f"OK: {total_games} games, {total_rows} decisions "
          f"({total_rows / max(1, total_games):.1f}/game), {total_bytes / 1e6:.1f} MB "
          f"({total_bytes / max(1, total_rows):.0f} B/row), {dt:.0f}s "
          f"({dt / max(1, total_games):.2f} s/game); replay self-test passed on every shard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
