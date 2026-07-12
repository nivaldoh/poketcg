#!/usr/bin/env python3
"""Record ONE cg battle to a step-by-step `trace.json` for the local battle visualizer.

Phase 1 (the headless backbone) of the visualizer plan: play one battle between two chosen agents
and write a JSON trace — one record per decision — that `visualizer.html` steps through. No UI here.

Each step record holds everything a human needs to stand in the model's shoes at that decision:
  * `obs`   — the fog-of-war observation the ACTING agent actually saw (search_begin_input dropped)
  * `god`   — the matching `visualize_data()` snapshot (a full-visibility board: BOTH players'
              hands/decks/prizes, with card names) — for a "reveal hidden info" toggle
  * `select`/`options` — the decision prompt + each legal option rendered human-readably
  * `action`— the option indices the agent chose (`chosen` flags them on the options)
  * `logs`  — the engine event feed since the previous decision (damage / KO / coin / draw), decoded
  * `model` — left null here; Phase 3 fills it with the SB3 policy's per-option probs + value

Reuses (nothing reimplemented): eval.py's `Agent` + `make_*` factories and its `_play_one` loop
SHAPE (copied, not edited, so eval/train routing stays untouched), `sb3_env.sb3_agent` for a `.zip`
model, `sb3_features._resolve_option` + `tcg_model.get_card`/`card_table` + `attack_table` for option
labels, `win_mode.classify_end` for the ending, and `cg.game.visualize_data` for the board.

    python record_battle.py --p0 random --p1 rulebased --out /tmp/t.json
    python record_battle.py --p0 out/sb3big_sb3.zip --p1 iono --deck lucario --out /tmp/m.json

Agent specs: random | rulebased(=lucario) | abomasnow | dragapult | iono | raging_bolt | real_raging_bolt | clefairy | a .pth (legacy MCTS,
plays sample_deck) | a .zip (SB3 MaskablePPO, plays --deck). The battle is one-at-a-time (the cg
engine is a process-global singleton) and, by default, unseeded (no CRN) — the written trace IS the
deterministic record you step through.

State replicator (needs the locally rebuilt engine — utils/build_libcg.sh + CG_LIB_PATH):
  * `--seed N`         — play deterministic engine-seeded battles (game i of a session uses N+i)
  * `--record-state`   — save a full-state engine blob per step (`step.state_blob`, plus
                         `final_state_blob`) so any recorded position can be re-instantiated as a
                         LIVE battle later via utils/resume_battle.py (replay or play-on). Opt-in:
                         blobs add ~15-60 KB/step.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Importing eval sets up the vendored cg path (via tcg_model) AND the game_agents/rulebased sys.path,
# so the cg.* and rule-based imports below resolve. No training/eval is triggered by import.
import torch as th  # noqa: E402  (already loaded via eval->tcg_model; used for the model trace)
from eval import (Agent, make_random, make_rulebased, make_abomasnow,  # noqa: E402
                  make_dragapult, make_iono, make_raging_bolt, make_real_raging_bolt,
                  make_lilies_clefairy,
                  make_model, git_commit, wilson)
from tcg_model import card_table, get_card, is_policy_select, auto_resolve  # noqa: E402
from sb3_features import (attack_table, _resolve_option, encode_state,  # noqa: E402
                          STATE_DIM, N_ENERGY_TYPES, N_SELECT_TYPES, BENCH_SLOTS)
from sb3_env import (is_learnable_select, obs_to_gym, _ms_mask,  # noqa: E402
                     N_MAX, STOP_INDEX)
from win_mode import classify_end, summarize_ends  # noqa: E402
from cg.api import (OptionType, SelectType, SelectContext, SpecialConditionType,  # noqa: E402
                    CardType, LogType, EnergyType, to_observation_class)
from cg.game import battle_start, battle_select, battle_finish, visualize_data  # noqa: E402
from utils.run_smoketest_battle import load_deck  # noqa: E402
import cg_state  # noqa: E402  (local save/restore bindings; inert on the stock .so)

# SelectContext value -> a plain-English prompt (mirrors the inline comments in cg/api.py's
# SelectContext; falls back to the prettified enum name for members added later in the competition).
PROMPTS = {
    0: "Main phase — choose an action", 1: "Choose your Active Pokémon (setup)",
    2: "Choose a Pokémon for your Bench (setup)", 3: "Choose a Pokémon to switch into the Active Spot",
    4: "Choose a Pokémon to put into your Active Spot", 5: "Choose a Pokémon to put on your Bench",
    6: "Choose a Pokémon to put into play", 7: "Choose a card to add to your hand",
    8: "Choose a card to discard", 9: "Choose a card to return to your deck",
    10: "Choose a card to return to the bottom of your deck", 11: "Choose a card to add to your prizes",
    12: "Choose a card to leave where it is", 13: "Choose a Pokémon to place damage counters on",
    14: "Choose a Pokémon to place damage counters on (as you like)",
    15: "Choose a Pokémon to deal damage to", 16: "Choose a Pokémon to remove damage counters from",
    17: "Choose a Pokémon to heal", 18: "Choose the Pokémon to evolve from",
    19: "Choose the Pokémon to evolve into", 20: "Choose a Pokémon to devolve",
    21: "Choose the Pokémon to attach the card to", 22: "Choose the card to attach to the Pokémon",
    23: "Choose the Pokémon to remove the card from", 24: "Choose a card to look at",
    25: "Choose a card to apply the effect to", 26: "Choose an Energy card to discard",
    27: "Choose a Pokémon Tool to discard", 28: "Choose an Energy card to replace",
    29: "Choose a card to discard", 30: "Choose the energy to discard",
    31: "Choose the energy to return to your hand", 32: "Choose the energy to return to the deck",
    33: "Choose the energy to switch", 34: "Choose the order of effect activation",
    35: "Choose the attack to use", 36: "Choose the attack to disable",
    37: "Choose the evolution source and target", 38: "Choose how many cards to draw",
    39: "Choose how many damage counters to place", 40: "Choose how many damage counters to remove",
    41: "Would you like to go first?", 42: "Would you like to redraw your cards? (mulligan)",
    43: "Would you like to activate the effect?", 44: "Would you like to select the first effect?",
    45: "Do you want to devolve it further?", 46: "Do you want to choose heads?",
    47: "Choose the special condition to affect", 48: "Choose the special condition to recover",
}


def _state_labels():
    """Per-dimension labels for `sb3_features.encode_state`'s state vector, in emit order, so the
    viewer can show each model-input number with its meaning. Generated from the SAME block widths
    the encoder uses (imported N_ENERGY_TYPES / N_SELECT_TYPES / BENCH_SLOTS), then asserted to equal
    STATE_DIM — if the featurization changes, this fails loudly instead of mislabeling silently.
    Each label is 'group: detail' (group ∈ {g, me, opp}) so the UI can fold it by side."""
    en = [EnergyType(i).name.title() for i in range(N_ENERGY_TYPES)]
    st = [SelectType(i).name for i in range(N_SELECT_TYPES)]

    def side(tag):
        L = [f"{tag}: prize/6", f"{tag}: deck/60", f"{tag}: hand/10", f"{tag}: discard/60",
             f"{tag}: bench/5", f"{tag}: inplay/6",
             f"{tag}: status poisoned", f"{tag}: status burned", f"{tag}: status asleep",
             f"{tag}: status paralyzed", f"{tag}: status confused",
             f"{tag}: active has", f"{tag}: active hp_frac", f"{tag}: active maxHp/400",
             f"{tag}: active n_energy/8"]
        L += [f"{tag}: active E[{e}]" for e in en]
        L += [f"{tag}: active n_tools/2", f"{tag}: active appearThisTurn", f"{tag}: active retreat/4"]
        L += [f"{tag}: active weak:{e}" for e in en]
        L += [f"{tag}: active resist:{e}" for e in en]
        L += [f"{tag}: active {f}" for f in ("basic", "stage1", "stage2", "ex", "megaEx")]
        for j in range(BENCH_SLOTS):
            L += [f"{tag}: bench{j} has", f"{tag}: bench{j} hp_frac", f"{tag}: bench{j} n_energy/8"]
            L += [f"{tag}: bench{j} E[{e}]" for e in en]
            L += [f"{tag}: bench{j} anyEx"]
        return L

    g = ["g: turn/20", "g: turnActionCount/10", "g: firstPlayer==me", "g: energyAttached",
         "g: supporterPlayed", "g: stadiumPlayed", "g: retreated", "g: stadium_present"]
    g += [f"g: selType:{s}" for s in st]
    g += ["g: remainEnergyCost/10", "g: remainDamageCounter/10"]
    labels = g + side("me") + side("opp")
    assert len(labels) == STATE_DIM, f"label count {len(labels)} != STATE_DIM {STATE_DIM}"
    return labels


STATE_LABELS = _state_labels()


def _enum_name(enum_cls, v):
    """Enum member name for `v`, tolerant of values added mid-competition (never raises)."""
    try:
        return enum_cls(v).name
    except (ValueError, KeyError):
        return f"UNKNOWN({v})"


def _card_name(cid, seen):
    """Human name for a card id (records the id in `seen` so the header card map covers it)."""
    if cid is None:
        return "?"
    seen.add(cid)
    cd = card_table.get(cid)
    return cd.name if cd else f"#{cid}"


def _label_option(o, opt, yi, seen):
    """Render one Option as a short human label, reusing sb3_features._resolve_option's card/attack
    resolution (the same mapping the model's featurizer uses). Falls back to the OptionType name."""
    try:
        t = OptionType(opt.type)
    except ValueError:
        return f"option (type {opt.type})"
    try:
        pid, _pp, tid, _tp, aid = _resolve_option(o, opt, yi)
    except Exception:
        pid = tid = aid = None
    cn = lambda cid: _card_name(cid, seen)
    if t == OptionType.ATTACK:
        a = attack_table.get(aid)
        return f"{a.name} (dmg {a.damage})" if a else f"Attack #{aid}"
    if t == OptionType.END:
        return "End turn"
    if t == OptionType.RETREAT:
        return f"Retreat {cn(pid)}" if pid is not None else "Retreat"
    if t == OptionType.PLAY:
        return f"Play {cn(pid)}"
    if t == OptionType.ATTACH:
        return f"Attach {cn(pid)} → {cn(tid)}"
    if t == OptionType.EVOLVE:
        return f"Evolve {cn(tid)} → {cn(pid)}"
    if t == OptionType.ABILITY:
        return f"Ability: {cn(pid)}"
    if t == OptionType.DISCARD:
        return f"Discard {cn(pid)}"
    if t == OptionType.SKILL:
        return f"Skill: {cn(opt.cardId)}"
    if t in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
        return cn(pid) if pid is not None else t.name.replace("_", " ").title()
    if t == OptionType.YES:
        return "Yes"
    if t == OptionType.NO:
        return "No"
    if t == OptionType.NUMBER:
        return f"Number: {opt.number}"
    if t == OptionType.SPECIAL_CONDITION:
        return f"Condition: {_enum_name(SpecialConditionType, opt.specialConditionType)}"
    return t.name.replace("_", " ").title()


def _option_cid(o, opt, yi):
    """The primary card id this option references (for coloring options in the viewer by their real
    card color), or None for options with no card subject (END/YES/NO/NUMBER/ATTACK). Reuses the same
    _resolve_option resolution as _label_option's label."""
    try:
        pid, _pp, _tid, _tp, _aid = _resolve_option(o, opt, yi)
    except Exception:
        return None
    return pid


def _decode_log(lg):
    """One raw log dict -> {kind, ...non-null fields}. `kind` names the LogType; the payload
    (playerIndex/cardId/value/head/reason/…) is passed through so the viewer can format it."""
    d = {"kind": _enum_name(LogType, lg.get("type"))}
    for k, v in lg.items():
        if k != "type" and v is not None:
            d[k] = v
    return d


def _build_record(obs, yi, actor, action, idx, seen, model_info, hero_seat):
    """One decision -> one trace record. `obs` is the acting seat's fog-of-war dict; `action` is the
    option indices chosen; `model_info` is the SB3 policy introspection (probs/value) or None;
    `hero_seat` is the seat the debugged agent occupies this game. `god` is attached later (from the
    single visualize_data() call)."""
    o = to_observation_class(obs)
    sel = o.select
    cur = obs["current"]
    chosen = set(action)
    rec = {
        "idx": idx,
        "turn": cur["turn"],
        "actor_seat": yi,
        "actor_name": actor.name,
        "hero_move": (yi == hero_seat),  # hero = the debugged agent (may sit at either seat, seat-swapped)
        "action": list(action),
        "logs": [_decode_log(lg) for lg in obs.get("logs", [])],
        "model": model_info,             # SB3 per-option probs + critic value (None for non-SB3 actors)
        # store the raw obs the agent saw, minus the opaque search-API blob
        "obs": {k: v for k, v in obs.items() if k != "search_begin_input"},
    }
    if sel is None:                      # deck-return node (not reached in the battle_start loop)
        rec["select"] = None
        rec["options"] = []
        return rec
    rec["select"] = {
        "context": int(sel.context),
        "context_name": _enum_name(SelectContext, sel.context),
        "prompt": PROMPTS.get(int(sel.context), _enum_name(SelectContext, sel.context).replace("_", " ").title()),
        "type": int(sel.type),
        "type_name": _enum_name(SelectType, sel.type),
        "minCount": sel.minCount,
        "maxCount": sel.maxCount,
        "n_options": len(sel.option),
        "policy": is_policy_select(sel),          # single most meaningful "real choice" flag
        "learnable": is_learnable_select(sel),    # the wider set the SB3 policy is allowed to decide
    }
    rec["options"] = [{"label": _label_option(o, opt, yi, seen), "chosen": k in chosen,
                       "cid": _option_cid(o, opt, yi)}   # cid → viewer colors the option by its card color
                      for k, opt in enumerate(sel.option)]
    if model_info is not None:
        # The exact state vector the model ingested (labels are the static header.state_labels).
        rec["features"] = {"state": [round(float(x), 4) for x in encode_state(o)[0].tolist()]}
    return rec


def _subtype(cd):
    """Compact stage/rarity label for the card hover, e.g. 'Stage 2 · ex' / 'Basic' / 'Mega ex' /
    'ACE SPEC'. Empty for a plain trainer/energy. (ex and megaEx are mutually exclusive in the engine
    — megaEx cards carry ex=False — so they're an either/or.)"""
    p = []
    if cd.basic:
        p.append("Basic")
    elif cd.stage1:
        p.append("Stage 1")
    elif cd.stage2:
        p.append("Stage 2")
    if cd.megaEx:
        p.append("Mega ex")
    elif cd.ex:
        p.append("ex")
    if cd.tera:
        p.append("Tera")
    if cd.aceSpec:
        p.append("ACE SPEC")
    return " · ".join(p)


def _attack_meta(aid):
    a = attack_table[aid]
    m = {"name": a.name, "damage": a.damage, "text": a.text}
    if a.energies:
        m["cost"] = [int(e) for e in a.energies]     # energy cost as EnergyType ints (pips in the UI)
    return m


def _card_meta(cid):
    """Static card data for the trace's card map — everything the visualizer's hover tooltip renders.
    Keys beyond name/hp/type/element/attacks are emitted only when meaningful (skills, subtype, and the
    Pokémon-only weakness/resistance/retreat/evolvesFrom) so the JSON stays lean and the viewer guards
    on presence. `skills` covers BOTH Pokémon abilities and Trainer (Item/Supporter/Stadium/Tool) rules."""
    cd = card_table.get(cid)
    if not cd:
        return {"name": f"#{cid}"}
    m = {
        "name": cd.name,
        "hp": cd.hp,
        "type": _enum_name(CardType, cd.cardType),
        "element": int(cd.energyType) if getattr(cd, "energyType", None) is not None else None,
        "attacks": [_attack_meta(aid) for aid in cd.attacks if aid in attack_table],
    }
    if cd.skills:
        m["skills"] = [{"name": sk.name, "text": sk.text} for sk in cd.skills]
    sub = _subtype(cd)
    if sub:
        m["subtype"] = sub
    if cd.cardType == CardType.POKEMON:
        if cd.weakness is not None:
            m["weakness"] = int(cd.weakness)
        if cd.resistance is not None:
            m["resistance"] = int(cd.resistance)
        m["retreat"] = cd.retreatCost
        if cd.evolvesFrom:
            m["evolvesFrom"] = cd.evolvesFrom
    return m


def record_match(a: Agent, b: Agent, seat_a: int = 0, seed: int = 0, record_state: bool = False) -> dict:
    """Play one battle (agent `a` at seat `seat_a`) and return the full trace dict. Mirrors
    eval._play_one's driver loop, capturing a record before each battle_select and one god-view
    snapshot array at the end (visualize_data() must be read BEFORE battle_finish frees the battle).
    `seed` != 0 plays an engine-deterministic battle; `record_state` attaches a full-state blob per
    step (both need the rebuilt engine, see module docstring)."""
    decks = [None, None]
    decks[seat_a] = a.deck
    decks[1 - seat_a] = b.deck
    if seed:
        obs, start = cg_state.battle_start_seeded(decks[0], decks[1], seed)
    else:
        obs, start = battle_start(decks[0], decks[1])
    if start.errorPlayer >= 0:
        raise ValueError(f"deck rejected: errorType={start.errorType} player={start.errorPlayer}")

    seen: set[int] = set(a.deck) | set(b.deck)
    steps = []
    while obs["current"]["result"] < 0:
        state_blob = cg_state.save_state() if record_state else None
        yi = obs["current"]["yourIndex"]
        actor = a if yi == seat_a else b
        action = actor.fn(obs)                       # run the agent on the FULL obs (search blob intact)
        holder = getattr(actor, "trace", None)       # traced SB3 agents stash their introspection here
        model_info = holder["last"] if holder else None
        steps.append(_build_record(obs, yi, actor, action, len(steps), seen, model_info, seat_a))
        if state_blob is not None:
            steps[-1]["state_blob"] = state_blob
        obs = battle_select(action)

    final_state_blob = cg_state.save_state() if record_state else None
    god = json.loads(visualize_data())               # one snapshot per step (+ terminal); read pre-finish
    result = obs["current"]["result"]
    end = classify_end(obs, seat_a)                  # hero(=agent a)-relative: how the game ended
    battle_finish()

    # Align god[i] (board when decision i was presented) with step i; god's last element is terminal.
    for i, st in enumerate(steps):
        st["god"] = god[i] if i < len(god) else None

    # Report seat-agnostically: p0 = the debugged agent `a` (hero, at seat `seat_a`), p1 = opponent `b`.
    result_str = {1: f"{a.name} (hero) wins", -1: f"{b.name} wins", 0: "draw"}[end["win"]]
    trace = {
        "header": {
            "p0": a.name, "p1": b.name, "hero_seat": seat_a,
            "result": result,                        # raw engine result (seat-based 0/1/2)
            "result_str": result_str,
            "win_mode": end,                         # hero-relative: how the game ended
            "seed": seed or None,                    # engine seed (None = stock nondeterministic run)
            "git_commit": git_commit(),
            "n_steps": len(steps),
            "cards": {str(cid): _card_meta(cid) for cid in sorted(seen)},
            "state_labels": STATE_LABELS,            # per-dim labels for step.features.state
        },
        "steps": steps,
        "final_god": god[-1] if god else None,
    }
    if final_state_blob is not None:
        trace["final_state_blob"] = final_state_blob
    return trace


def record_session(a: Agent, b: Agent, n: int, seed: int = 0, record_state: bool = False) -> dict:
    """Play `n` seat-swapped games (like eval.play_match: agent a at seat i%2) and bundle them with a
    running eval summary — win% + Wilson CI + win-mode histograms — all from the hero (agent a) POV.
    Reuses eval.wilson and win_mode.summarize_ends verbatim so the numbers match the eval harness.
    With `seed` != 0, game i is engine-seeded with seed+i."""
    games, ends = [], []
    w = l = d = 0
    for i in range(n):
        tr = record_match(a, b, seat_a=i % 2, seed=seed + i if seed else 0, record_state=record_state)
        end = tr["header"]["win_mode"]
        ends.append(end)
        w += end["win"] == 1
        l += end["win"] == -1
        d += end["win"] == 0
        games.append(tr)
        print(f"  game {i + 1}/{n}: W/L/D = {w}/{l}/{d}", end="\r", flush=True)
    print(" " * 50, end="\r")
    p, lo, hi = wilson(w, w + l)
    return {
        "session": {
            "p0": a.name, "p1": b.name, "n": n, "wins": w, "losses": l, "draws": d,
            "win_pct": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "decisive": w + l, "end_summary": summarize_ends(ends), "git_commit": git_commit(),
            "seed": seed or None,                    # game i used engine seed seed+i (None = unseeded)
        },
        "games": games,
    }


def traced_sb3_agent(model, greedy=False):
    """A read-only twin of sb3_env.sb3_agent: identical decision routing (so the played battle is
    unchanged), but on each learnable node it also captures the policy's per-option probabilities and
    the critic's value into `holder["last"]`. Returns (agent_fn, holder). The capture uses the SAME
    `gobs`/`mask` fed to model.predict, so it's exactly what the model saw. For a multi-select node it
    captures the FIRST sub-step's distribution (the model's opening preference over the shown options).

    win_prob = clamp((value+1)/2) is APPROXIMATE (flagged for review): the SB3 critic head is an
    unbounded Linear, so with terminal reward in {-1,+1} and gamma=1 the value only *approximates*
    E[return] in [-1,1] — read it as a rough win-probability, not a calibrated one."""
    holder = {"last": None}

    def agent_fn(obs_dict):
        o = to_observation_class(obs_dict)
        sel = o.select
        holder["last"] = None
        if not is_learnable_select(sel) or len(sel.option) > N_MAX - 1:
            return auto_resolve(sel)
        picked, first = [], True
        while True:
            gobs = obs_to_gym(o, picked)
            mask = _ms_mask(sel, picked)
            if first:                                # capture the fresh-node distribution + value
                with th.no_grad():
                    obs_t, _ = model.policy.obs_to_tensor(gobs)
                    probs = model.policy.get_distribution(obs_t, action_masks=mask).distribution.probs[0]
                    value = float(model.policy.predict_values(obs_t)[0, 0].item())
                probs = probs.cpu().numpy()
                n = len(sel.option)
                holder["last"] = {
                    "probs": [round(float(probs[i]), 4) for i in range(n)],
                    "p_commit": round(float(probs[STOP_INDEX]), 4),
                    "value": round(value, 4),
                    "win_prob": round(min(1.0, max(0.0, (value + 1) / 2)), 4),
                    "multi": sel.maxCount > 1 or sel.minCount == 0,   # probs are the first sub-step here
                }
                first = False
            action, _ = model.predict(gobs, action_masks=mask, deterministic=greedy)
            a = int(action)
            if a == STOP_INDEX:
                return picked
            picked.append(a)
            if len(picked) == sel.maxCount:
                return picked
    return agent_fn, holder


def make_agent(spec: str, deck_name: str, greedy: bool) -> Agent:
    """Resolve an agent spec (see module docstring) to an eval.Agent."""
    key = spec.lower()
    if key == "random":
        return make_random()
    if key in ("rulebased", "lucario"):
        return make_rulebased()
    if key == "abomasnow":
        return make_abomasnow()
    if key in ("dragapult", "dragapult_ex"):
        return make_dragapult()
    if key == "iono":
        return make_iono()
    if key in ("raging_bolt", "simple_raging_bolt_burn"):
        return make_raging_bolt()
    if key in ("real_raging_bolt", "real_raging_bolt_burn"):
        return make_real_raging_bolt()
    if key in ("clefairy", "lilies_clefairy", "lilies_clefairy_ex_mega_kangaskhan_ex"):
        return make_lilies_clefairy()
    if spec.endswith(".pth"):
        return make_model(spec)                      # legacy MyModel+MCTS; plays sample_deck
    if spec.endswith(".zip"):
        from sb3_contrib import MaskablePPO           # lazy: only the SB3 path needs it
        deck = load_deck(os.path.join(HERE, "decks", f"{deck_name}.csv"))
        model = MaskablePPO.load(spec, device="auto")
        fn, holder = traced_sb3_agent(model, greedy=greedy)
        ag = Agent(f"sb3:{os.path.basename(spec)}:{deck_name}", fn, deck)
        ag.trace = holder                            # record_match reads ag.trace["last"] per decision
        return ag
    raise SystemExit(f"unknown agent spec: {spec!r} "
                     "(random|rulebased|abomasnow|dragapult|iono|raging_bolt|real_raging_bolt|"
                     "clefairy|<*.pth>|<*.zip>)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default="random", help="agent for player 0 (the debugged 'hero')")
    ap.add_argument("--p1", default="rulebased", help="agent for player 1 (the opponent)")
    ap.add_argument("--deck", default="lucario",
                    help="deck name (decks/<name>.csv) for any SB3 .zip model agent")
    ap.add_argument("--greedy", action="store_true",
                    help="SB3 model picks argmax instead of sampling (default: sample)")
    ap.add_argument("--games", type=int, default=1,
                    help="record a SESSION of N seat-swapped games (running win%% + CI + win-modes) "
                         "instead of one battle; the viewer gets a game picker + session summary")
    ap.add_argument("--seed", type=int, default=0,
                    help="engine seed for deterministic battles (0 = stock nondeterministic; "
                         "game i of a session uses seed+i; needs the rebuilt engine)")
    ap.add_argument("--record-state", action="store_true",
                    help="attach a full-state engine blob to every step (resume/replay via "
                         "utils/resume_battle.py; needs the rebuilt engine)")
    ap.add_argument("--out", default="trace.json", help="output trace JSON path")
    args = ap.parse_args()

    if (args.seed or args.record_state) and not cg_state.has_state_api():
        raise SystemExit(
            "--seed/--record-state need the locally rebuilt engine: run "
            'utils/build_libcg.sh and set CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib"'
        )

    a = make_agent(args.p0, args.deck, args.greedy)
    b = make_agent(args.p1, args.deck, args.greedy)

    if args.games > 1:
        session = record_session(a, b, args.games, seed=args.seed, record_state=args.record_state)
        with open(args.out, "w") as f:
            json.dump(session, f)
        s = session["session"]
        assert len(session["games"]) == args.games and all(g["header"]["n_steps"] > 0 for g in session["games"])
        wm = s["end_summary"]["win_modes"]
        print(f"wrote {args.out}: SESSION {a.name} vs {b.name} | {args.games} games | "
              f"W/L/D={s['wins']}/{s['losses']}/{s['draws']} | win%={100 * s['win_pct']:.1f}% "
              f"(95% CI {100 * s['ci_lo']:.1f}-{100 * s['ci_hi']:.1f}, n={s['decisive']}) | win-modes {wm}")
        return 0

    trace = record_match(a, b, seed=args.seed, record_state=args.record_state)
    with open(args.out, "w") as f:
        json.dump(trace, f)

    # Smallest verification (README "verifiable results"): a real battle with a terminal result.
    h = trace["header"]
    assert h["n_steps"] > 0, "no decisions recorded"
    assert h["result"] in (0, 1, 2), f"bad result {h['result']}"
    assert h["win_mode"] and "mode" in h["win_mode"], "missing win_mode"
    print(f"wrote {args.out}: {h['n_steps']} steps | {h['result_str']} | "
          f"end={h['win_mode']['mode']} (hero win={h['win_mode']['win']}, "
          f"turns={h['win_mode']['turns']}) | {len(h['cards'])} cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
