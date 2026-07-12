# Rule-Based Agent for the Simplified Raging Bolt Burn deck
# Deck Concept
# Best fit: Turbo Raging Bolt ex / Teal Mask Ogerpon ex
#
# The closest Pokémon TCG equivalent to MTG Mono-Red Burn is not actually a Fire-type deck. It is
# Raging Bolt Ogerpon: it can attack on your first turn going second, it plays only Basic Pokémon
# (almost no setup), it converts Energy directly into damage, and its plan is simply to take a
# two-Prize knockout every turn.
#
# Bellowing Thunder requires one Lightning and one Fighting Energy. You may then discard any number
# of Basic Energy from any of your Pokémon, dealing 70 damage per Energy discarded:
#   2 -> 140, 3 -> 210, 4 -> 280, 5 -> 350, 6 -> 420
# This is the burn player's lethal calculation: discard exactly enough Energy for the knockout —
# never automatically discard everything.
#
# Pilot rules encoded below (from the deck guide):
#   1. Choose to go second (Supporter + attack on the first turn).
#   2. Build the same board almost every game: Active Raging Bolt; Bench 2x Ogerpon, backup Raging
#      Bolt, Latias, one utility Pokémon.
#   3. Ideal opening: bench Ogerpon, Teal Dance grass onto them, Crispin for Lightning+Fighting,
#      attach both to Raging Bolt, attack.
#   4. Utility Pokémon are mechanical: Meowth searches Supporters when benched; Kangaskhan draws in
#      the Active Spot and retreats free through Latias; Fezandipiti draws 3 after a knockout.
#   5. Reload after every attack: Teal Dance -> Crispin/draw -> recover/move Energy -> Boss a
#      valuable target -> Bellowing Thunder.
# The three rules that prevent most mistakes: attack whenever reasonably possible; discard only
# enough Energy for the knockout; prioritize two-Prize (and three-Prize Mega) targets.

import os
import sys
from collections import defaultdict
from math import ceil

# Local run: make the vendored `cg` simulator importable. Override with CG_LIB_PATH
# (the Kaggle BR kernel sets it). Repo root is two levels up from this file.
sys.path.insert(0, os.environ.get(
    "CG_LIB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "cg-lib")))

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                    Card, Pokemon, all_card_data, to_observation_class)

# Resolve this agent's deck: prefer the repo's committed list, then the
# Kaggle-packaged deck.csv (beside main.py) at submission time.
_here = os.path.dirname(os.path.abspath(__file__))
file_path = "deck.csv"
for _cand in (os.path.join(_here, "..", "..", "decks", "simple_raging_bolt_burn.csv"),
              os.path.join(_here, "decks", "simple_raging_bolt_burn.csv"), "deck.csv",
              "/kaggle_simulations/agent/deck.csv"):
    if os.path.exists(_cand):
        file_path = _cand
        break
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Fetch card metadata database and create an ID-to-Card lookup table
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
from cg.api import all_attack
attack_table = {a.attackId: a for a in all_attack()}

# Decklist
Raging_Bolt_ex = 63        # x4  71 Burst Roar {C}: discard hand, draw 6
                           #     72 Bellowing Thunder {L}{F}: 70 x each Basic Energy discarded
Teal_Mask_Ogerpon_ex = 96  # x4  Teal Dance: attach {G} from hand to self, draw 1
Fezandipiti_ex = 140       # x1  Flip the Script: draw 3 if KO'd last turn
Latias_ex = 184            # x2  Skyliner: your Basics have free retreat
Mega_Kangaskhan_ex = 756   # x2  Run Errand (Active only): draw 2
Meowth_ex = 1071           # x2  Last-Ditch Catch: on bench-play, search a Supporter
Prime_Catcher = 1088       # x1  ACE SPEC: gust + self switch
Night_Stretcher = 1097     # x2  Pokemon or basic energy from trash to hand
Glass_Trumpet = 1098       # x2  attach trash energy to up to 2 benched {C} Pokémon (needs Tera)
Energy_Switch = 1116       # x4  move a basic energy between your Pokémon
Energy_Retrieval = 1118    # x2  2 basic energy trash -> hand
Ultra_Ball = 1121          # x4  discard 2 -> search a Pokémon
Boss_Orders = 1182         # x2  gust
Crispin = 1198             # x4  search 2 basic energy of different types: 1 to hand, attach 1
Cyrano = 1205              # x2  search up to 3 Pokémon ex
Lillie_Determination = 1227  # x3  shuffle hand, draw 6 (8 at exactly 6 prizes)
Area_Zero_Underdepths = 1250  # x4 stadium: bench 8 while you have a Tera in play
Grass_Energy = 1           # x7
Lightning_Energy = 4       # x4
Fighting_Energy = 6        # x4

# Attack ids
ATK_BURST_ROAR = 71
ATK_BELLOWING = 72
ATK_LEAF_SHOWER = 120      # Ogerpon {G}{G}{G}: 30 + 30 per energy on both actives
ATK_CRUEL_ARROW = 183      # Fezandipiti {C}{C}{C}: 100 to any 1 opp Pokémon
ATK_RAPID_FIRE = 1092      # Kangaskhan {C}{C}{C}: 200 + 50 per heads
ATK_TUCK_TAIL = 1546       # Meowth: bounce self

CORE_TYPES = (EnergyType.LIGHTNING, EnergyType.FIGHTING)

# Per-turn state: this module is imported once and reused across many games (eval gauntlet).
pre_turn = -1
pending_attach_type = None  # energy type picked in a source select, consumed by the ATTACH_FROM target


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    score += pokemon.hp
    return score


def _my_pokemon(my_state) -> list[tuple[AreaType, int, Pokemon]]:
    out = []
    for i, p in enumerate(my_state.active):
        if p is not None:
            out.append((AreaType.ACTIVE, i, p))
    for i, p in enumerate(my_state.bench):
        if p is not None:
            out.append((AreaType.BENCH, i, p))
    return out


def _k_needed(defender: Pokemon) -> int:
    """Energies Bellowing Thunder must discard to KO the defender (70 per card, weakness x2)."""
    data = card_table[defender.id]
    per = 70
    hp = defender.hp
    if data.weakness == EnergyType.DRAGON:
        per = 140
    elif data.resistance == EnergyType.DRAGON:
        hp += 30
    return max(1, ceil(hp / per))


def _opp_can_ohko(op_state, our_hp: int) -> bool:
    """Whether the opponent's active has a payable attack whose BASE damage one-shots `our_hp`.
    Base damage only (variable attacks underestimate) — a deliberate, cheap proxy."""
    act = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    if act is None or act.id not in card_table:
        return False
    for aid in card_table[act.id].attacks:
        a = attack_table.get(aid)
        if a is not None and a.damage >= our_hp and len(act.energies) >= len(a.energies):
            return True
    return False


def _board_fuel(my_state) -> int:
    """Total Basic Energy cards attached to my Pokémon (all energy in this deck is basic)."""
    return sum(len(p.energyCards) for _, _, p in _my_pokemon(my_state))


def _active_bolt_ready(my_state) -> bool:
    """Active is a Raging Bolt with the {L}{F} attack cost attached."""
    act = my_state.active[0] if my_state.active else None
    if act is None or act.id != Raging_Bolt_ex:
        return False
    return (EnergyType.LIGHTNING in act.energies) and (EnergyType.FIGHTING in act.energies)


def _missing_core(my_state) -> tuple[Pokemon | None, list[EnergyType]]:
    """The priority Raging Bolt (active first, then most-fueled bench copy) and the core types
    ({L}/{F}) it still lacks. Returns (None, []) when no Bolt is in play or its core is complete."""
    bolts = [p for _, _, p in _my_pokemon(my_state) if p.id == Raging_Bolt_ex]
    if not bolts:
        return None, []
    act = my_state.active[0] if my_state.active else None
    if act is not None and act.id == Raging_Bolt_ex:
        target = act
    else:
        target = max(bolts, key=lambda p: sum(1 for t in CORE_TYPES if t in p.energies))
    missing = [t for t in CORE_TYPES if t not in target.energies]
    return target, missing


def _kill_options(my_state, op_state) -> dict:
    """What Bellowing Thunder could KO right now: the current defender and the best gustable
    bench target. Fuel is board-wide (the attack discards from any of my Pokémon). `ready`
    also covers an armed BENCH Bolt that can reach the active spot this turn (free/cheap
    retreat), so Boss -> retreat -> attack lines light up."""
    fuel = _board_fuel(my_state)
    ready = _active_bolt_ready(my_state)
    if not ready:
        act = my_state.active[0] if my_state.active else None
        latias = any(p is not None and p.id == Latias_ex
                     for p in list(my_state.active or []) + list(my_state.bench))
        bench_armed = any(p is not None and p.id == Raging_Bolt_ex
                          and all(t in p.energies for t in CORE_TYPES) for p in my_state.bench)
        if (bench_armed and act is not None and act.id != Raging_Bolt_ex
                and (latias or card_table[act.id].retreatCost <= 1)):
            ready = True
    my_need = len(my_state.prize)
    out = {"ready": ready, "fuel": fuel, "act_kill": False, "act_prizes": 0, "act_wins": False,
           "bench_i": -1, "bench_prizes": 0, "bench_wins": False}
    act = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    if act is not None and ready and _k_needed(act) <= fuel:
        out["act_kill"] = True
        out["act_prizes"] = prize_count(act)
        out["act_wins"] = out["act_prizes"] >= my_need
    best = -1
    for i, p in enumerate(op_state.bench):
        if p is None or not ready or _k_needed(p) > fuel:
            continue
        pr = prize_count(p)
        if pr > out["bench_prizes"] or (pr == out["bench_prizes"] and best >= 0
                                        and _k_needed(p) < _k_needed(op_state.bench[best])):
            out["bench_prizes"] = pr
            best = i
    out["bench_i"] = best
    out["bench_wins"] = out["bench_prizes"] >= my_need
    # A gust is worth it when it converts a KO the current defender doesn't offer, or a bigger one.
    out["gust_worth"] = best >= 0 and (out["bench_wins"] or not out["act_kill"]
                                       or out["bench_prizes"] > out["act_prizes"])
    return out


def _finish(scores: list[int], select, want: int | None = None) -> list[int]:
    """Sort options by score desc and return the top-k indices. Unlike the other rule-based agents
    (which always take maxCount), k is overridable: Bellowing Thunder's discard select has
    minCount=0 / maxCount=<all my energy>, where taking maxCount would torch the whole board."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    k = select.maxCount if want is None else want
    k = max(select.minCount, min(select.maxCount, k))
    return order[:k]


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive),
    with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    global pre_turn, pending_attach_type

    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Kaggle flow first call: return the deck and reset per-game state.
        pre_turn = -1
        pending_attach_type = None
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn != pre_turn:
        pre_turn = state.turn
        pending_attach_type = None

    eff = select.effect
    eff_id = eff.id if eff is not None else 0

    # ---- board census ----------------------------------------------------------------------
    mine = _my_pokemon(my_state)
    field_counts = defaultdict(int)
    for _, _, p in mine:
        field_counts[p.id] += 1
    hand_counts = defaultdict(int)
    for c in my_state.hand:
        hand_counts[c.id] += 1
    discard_counts = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    stadium_id = 0
    for c in state.stadium:
        stadium_id = c.id

    latias_in_play = field_counts[Latias_ex] > 0
    ogerpon_in_play = field_counts[Teal_Mask_Ogerpon_ex]
    ogerpon_target = 3 if (stadium_id == Area_Zero_Underdepths and my_state.benchMax >= 8) else 2
    trash_energy = discard_counts[Grass_Energy] + discard_counts[Lightning_Energy] + discard_counts[Fighting_Energy]
    deck_energy_left = 15 - trash_energy - sum(len(p.energyCards) for _, _, p in mine) \
        - hand_counts[Grass_Energy] - hand_counts[Lightning_Energy] - hand_counts[Fighting_Energy]
    supporter_in_hand = any(card_table[c.id].cardType == CardType.SUPPORTER for c in my_state.hand)
    total_in_play = len(mine)

    core_bolt, core_missing = _missing_core(my_state)
    kills = _kill_options(my_state, op_state)
    # Stall: a Bolt lacks its {L}{F} core and the hand has no way to finish it. The deck's own
    # answers are the reset buttons (Lillie's / Burst Roar) — loosen their gates in this state.
    missing_in_hand = any(hand_counts[e] > 0 for e in (Lightning_Energy, Fighting_Energy)
                          if card_table[e].energyType in core_missing)
    eswitch_fixable = (hand_counts[Energy_Switch] > 0 and core_bolt is not None
                       and any(p.serial != core_bolt.serial
                               and any(card_table[ec.id].energyType in core_missing
                                       for ec in p.energyCards if ec.id in card_table)
                               for _, _, p in mine))
    retrieval_fixable = (hand_counts[Energy_Retrieval] + hand_counts[Night_Stretcher] > 0
                         and any(discard_counts[e] > 0 for e in (Lightning_Energy, Fighting_Energy)
                                 if card_table[e].energyType in core_missing))
    stalled = (bool(core_missing) and not missing_in_hand and hand_counts[Crispin] == 0
               and not eswitch_fixable and not retrieval_fixable)

    # ---- per-hand-index usefulness (Ultra Ball cost, Lillie's / Burst Roar "dead hand" gate) --
    grass_keep = max(1, ogerpon_in_play)  # grass copies worth holding as Teal Dance fodder
    hand_usefulness = []
    seen = defaultdict(int)
    for c in my_state.hand:
        cid = c.id
        u = -10
        if cid == Raging_Bolt_ex:
            u = 200 if field_counts[cid] + seen[cid] < 2 else -30
        elif cid == Teal_Mask_Ogerpon_ex:
            u = 200 if field_counts[cid] + seen[cid] < ogerpon_target else -30
        elif cid == Latias_ex:
            u = 180 if field_counts[cid] + seen[cid] < 1 else -40
        elif cid == Meowth_ex:
            u = 120 if not supporter_in_hand and field_counts[cid] == 0 else -20
        elif cid == Fezandipiti_ex:
            u = 60 if seen[cid] == 0 and field_counts[cid] == 0 else -30
        elif cid == Mega_Kangaskhan_ex:
            u = -20
        elif cid in (Lightning_Energy, Fighting_Energy):
            u = 200 if core_missing else 40
        elif cid == Grass_Energy:
            u = 100 if seen[cid] < grass_keep else -20
        elif cid == Crispin:
            u = 150
        elif cid == Boss_Orders:
            u = 120
        elif cid == Prime_Catcher:
            u = 150
        elif cid == Cyrano:
            u = 80 if total_in_play < 4 else -10
        elif cid == Lillie_Determination:
            u = 60 if seen[cid] == 0 else -40
        elif cid == Ultra_Ball:
            u = 80
        elif cid == Energy_Switch:
            u = 60 if seen[cid] < 2 else -10
        elif cid == Energy_Retrieval:
            u = 50 if trash_energy >= 1 else -10
        elif cid == Night_Stretcher:
            u = 50
        elif cid == Glass_Trumpet:
            u = 30 if trash_energy >= 1 else -15
        elif cid == Area_Zero_Underdepths:
            u = 40 if stadium_id != Area_Zero_Underdepths and seen[cid] == 0 else -50
        hand_usefulness.append(u)
        seen[cid] += 1
    useful_hand = sum(1 for u in hand_usefulness if u > 0)

    # ==========================================================================================
    # Context-specific selects with their own semantics
    # ==========================================================================================

    if context == SelectContext.IS_FIRST:
        # Guide rule 1: choose to go SECOND (Supporter + attack on our first turn).
        scores = [1 if o.type == OptionType.NO else 0 for o in select.option]
        return _finish(scores, select)

    if context == SelectContext.DISCARD_ENERGY_CARD:
        # Bellowing Thunder's discard: 70 damage per card. Pick exactly enough for the KO,
        # spending renewable/bank fuel before the attacker's own {L}{F} core.
        defender = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
        k = _k_needed(defender) if defender is not None else 0
        infos = []
        for o in select.option:
            host = get_card(obs, o.area, o.index, my_index)
            eid = 0
            if host is not None and o.energyIndex is not None and o.energyIndex < len(host.energyCards):
                eid = host.energyCards[o.energyIndex].id
            infos.append((host, o.area, eid))
        # rank: 0 grass-on-Ogerpon (Teal Dance renews) < 1 bank (Kangaskhan/Meowth; Glass Trumpet
        # recycles) < 2 grass elsewhere < 3 L/F off-attacker < 4 surplus L/F on the attacker
        # < 9 the attacker's last {L}/{F} (breaks next turn's attack; spend only for the KO).
        # EXCEPT when the opponent's active can one-shot our attacker: everything attached to it
        # is trash next turn anyway, so spend the doomed attacker's energy FIRST and keep the
        # board's fuel.
        my_act = my_state.active[0] if my_state.active else None
        doomed = my_act is not None and _opp_can_ohko(op_state, my_act.hp)
        core_seen = {EnergyType.LIGHTNING: False, EnergyType.FIGHTING: False}
        ranks = []
        for host, area, eid in infos:
            hid = host.id if host is not None else 0
            etype = card_table[eid].energyType if eid in card_table else None
            on_attacker = hid == Raging_Bolt_ex and area == AreaType.ACTIVE
            if doomed and on_attacker:
                r = 0.1  # doomed fuel burns first
            elif hid == Teal_Mask_Ogerpon_ex and eid == Grass_Energy:
                r = 0.5
            elif hid in (Mega_Kangaskhan_ex, Meowth_ex):
                r = 1
            elif eid == Grass_Energy:
                r = 2
            elif not on_attacker:
                r = 3
            else:  # L/F on a surviving attacker: keep one of each if we can
                if etype in core_seen and not core_seen[etype]:
                    core_seen[etype] = True
                    r = 9
                else:
                    r = 4
            ranks.append(r)
        scores = [100000 - int(r * 1000) - i for i, r in enumerate(ranks)]
        if k and k <= len(infos):
            want = k
        else:
            # No KO available: chip only with free fuel — renewable Ogerpon grass plus anything
            # on a doomed attacker (it's lost either way). Never bleed the rest.
            want = min(sum(1 for r in ranks if r <= 0.5), k)
        return _finish(scores, select, want=want)

    if context in (SelectContext.SETUP_ACTIVE_POKEMON,):
        # The guide prefers a Kangaskhan start, but that assumes Latias shows up for the free
        # retreat; measured against random pilots a stuck 3-retreat active stalls the whole
        # attack plan. Start the attacker itself: it never needs to move.
        setup_active = {Raging_Bolt_ex: 100, Mega_Kangaskhan_ex: 80, Teal_Mask_Ogerpon_ex: 70,
                        Latias_ex: 60, Fezandipiti_ex: 40, Meowth_ex: 30}
        scores = []
        for o in select.option:
            c = get_card(obs, o.area, o.index, my_index)
            scores.append(setup_active.get(c.id, 0) if c is not None else 0)
        return _finish(scores, select)

    if context == SelectContext.SETUP_BENCH_POKEMON:
        # Bench the engine pieces; hold Meowth (Last-Ditch only triggers on an in-game bench play)
        # and Kangaskhan (3-prize liability). Engine loops this select one slot at a time.
        seen_bench = defaultdict(int)
        for _, _, p in mine:
            seen_bench[p.id] += 1
        scores = []
        for o in select.option:
            c = get_card(obs, o.area, o.index, my_index)
            s = -1
            if c is not None:
                cid = c.id
                if cid == Teal_Mask_Ogerpon_ex:
                    s = 100 if seen_bench[cid] < 1 else (95 if seen_bench[cid] < 2 else -1)
                elif cid == Raging_Bolt_ex:
                    s = 90 if seen_bench[cid] < 1 else (50 if seen_bench[cid] < 2 else -1)
                elif cid == Latias_ex:
                    s = 70 if seen_bench[cid] < 1 else -1
                elif cid == Fezandipiti_ex:
                    s = 10 if seen_bench[cid] < 1 else -1
                elif cid == Mega_Kangaskhan_ex:
                    s = 5 if seen_bench[cid] < 1 else -1  # 3-prize risk < bench-out insurance
            scores.append(s)
        return _finish(scores, select, want=sum(1 for s in scores if s > 0))

    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        scores = []
        for o in select.option:
            p = get_card(obs, o.area, o.index, o.playerIndex if o.playerIndex is not None else my_index)
            s = 0
            if p is None:
                scores.append(s)
                continue
            if o.playerIndex is not None and o.playerIndex != my_index:
                # Gust pick (Boss's Orders / Prime Catcher): drag up what we can KO, else the
                # weakest high-value piece.
                if kills["ready"] and _k_needed(p) <= kills["fuel"]:
                    s = 100000 + prize_count(p) * 1000
                    if prize_count(p) >= len(my_state.prize):
                        s += 50000
                else:
                    s = prize_count(p) * 1000 + 150 * len(p.energies) - p.hp
            else:
                # Promoting my own next active.
                cid = p.id
                if cid == Raging_Bolt_ex:
                    core_ok = all(t in p.energies for t in CORE_TYPES)
                    s = (1000 if core_ok else 600) + 10 * len(p.energies)
                elif cid == Teal_Mask_Ogerpon_ex:
                    s = 400 + 10 * len(p.energies)
                elif cid == Mega_Kangaskhan_ex:
                    s = 300 + (200 if latias_in_play else 0)
                    if len(op_state.prize) <= 3:
                        s -= 400  # a Kangaskhan KO would hand over the last 3 prizes
                elif cid == Latias_ex:
                    s = 200
                elif cid == Fezandipiti_ex:
                    s = 150
                elif cid == Meowth_ex:
                    s = 100
                if _opp_can_ohko(op_state, p.hp):
                    # The promotee likely dies on their next turn: weigh the prizes it feeds.
                    s -= 300 * prize_count(p)
            scores.append(s)
        return _finish(scores, select)

    if context == SelectContext.TO_HAND:
        scores = []
        for o in select.option:
            c = get_card(obs, o.area, o.index, my_index)
            s = 0
            if c is not None:
                s = _search_pick_score(c.id, eff_id, field_counts, hand_counts, core_missing,
                                       supporter_in_hand, ogerpon_in_play, ogerpon_target,
                                       kills, my_state, state)
                if card_table[c.id].cardType != CardType.BASIC_ENERGY:
                    s -= 100 * hand_counts[c.id]  # diversify Pokémon/trainer picks, not energy
            scores.append(s)
        return _finish(scores, select)

    if context == SelectContext.ATTACH_TO:
        # Picking WHICH energy card gets attached (Teal Dance hand grass, Crispin's deck attach,
        # Glass Trumpet's trash energy).
        scores = []
        for o in select.option:
            c = get_card(obs, o.area, o.index, my_index)
            s = 1
            if c is not None:
                et = card_table[c.id].energyType if c.id in card_table else None
                if et in core_missing:
                    s = 300 if et == EnergyType.FIGHTING else 290
                elif et in CORE_TYPES:
                    s = 150
                elif et == EnergyType.GRASS:
                    s = 100
            scores.append(s)
        pick = _finish(scores, select)
        if pick:
            c = get_card(obs, select.option[pick[0]].area, select.option[pick[0]].index, my_index)
            if c is not None and c.id in card_table:
                pending_attach_type = card_table[c.id].energyType
        return pick

    if context == SelectContext.SWITCH_ENERGY_CARD:
        # Energy Switch source: pull a core-type energy off any host EXCEPT the Bolt being
        # assembled (a bench Bolt's spare {F} moving to the active Bolt is a fine source).
        scores = []
        for o in select.option:
            host = get_card(obs, o.area, o.index, my_index)
            s = -1
            eid = 0
            if host is not None and o.energyIndex is not None and o.energyIndex < len(host.energyCards):
                eid = host.energyCards[o.energyIndex].id
            et = card_table[eid].energyType if eid in card_table else None
            not_target = host is not None and (core_bolt is None or host.serial != core_bolt.serial)
            if et in core_missing and not_target:
                s = 1000
            elif et in CORE_TYPES and not_target and host.id != Raging_Bolt_ex:
                s = 200
            elif et == EnergyType.GRASS and host is not None and host.id != Teal_Mask_Ogerpon_ex:
                s = 50
            scores.append(s)
        pick = _finish(scores, select)
        if pick:
            o = select.option[pick[0]]
            host = get_card(obs, o.area, o.index, my_index)
            if host is not None and o.energyIndex is not None and o.energyIndex < len(host.energyCards):
                eid = host.energyCards[o.energyIndex].id
                if eid in card_table:
                    pending_attach_type = card_table[eid].energyType
        return pick

    if context == SelectContext.ATTACH_FROM:
        # Picking WHICH of my Pokémon receives an energy (manual attach, Crispin's attach half,
        # Energy Switch destination, Glass Trumpet bench targets).
        et = pending_attach_type
        if eff_id in (Grass_Energy, Lightning_Energy, Fighting_Energy):
            et = card_table[eff_id].energyType  # manual attach: the effect IS the energy card
        # A stuck non-attacker active with an unpayable retreat cost blocks the whole plan when a
        # ready Bolt waits on the bench: funding its retreat beats any other attach target.
        act = my_state.active[0] if my_state.active else None
        ready_bench = any(p is not None and p.id == Raging_Bolt_ex
                          and all(t in p.energies for t in CORE_TYPES) for p in my_state.bench)
        fund_retreat = (act is not None and act.id != Raging_Bolt_ex and ready_bench
                        and not latias_in_play
                        and len(act.energies) < card_table[act.id].retreatCost)
        scores = []
        for o in select.option:
            p = get_card(obs, o.area, o.index, my_index)
            s = 0
            if p is not None:
                cid = p.id
                if fund_retreat and o.area == AreaType.ACTIVE and eff_id != Glass_Trumpet:
                    scores.append(1100)
                    continue
                if eff_id == Glass_Trumpet:
                    s = 100 if cid == Mega_Kangaskhan_ex else 90 if cid == Meowth_ex else 10
                elif et in CORE_TYPES:
                    if cid == Raging_Bolt_ex and et not in p.energies:
                        s = 1000 if o.area == AreaType.ACTIVE else 900
                    elif cid == Raging_Bolt_ex:
                        s = 500
                    elif cid in (Mega_Kangaskhan_ex, Meowth_ex):
                        s = 300
                    elif cid == Teal_Mask_Ogerpon_ex:
                        s = 250
                    else:
                        s = 100
                elif et == EnergyType.GRASS:
                    s = {Teal_Mask_Ogerpon_ex: 800, Mega_Kangaskhan_ex: 300, Meowth_ex: 280,
                         Raging_Bolt_ex: 250}.get(cid, 100)
                else:
                    # Unknown energy type: favor completing a Bolt, then the fuel bank.
                    if cid == Raging_Bolt_ex and any(t not in p.energies for t in CORE_TYPES):
                        s = 1000 if o.area == AreaType.ACTIVE else 900
                    else:
                        s = {Teal_Mask_Ogerpon_ex: 400, Mega_Kangaskhan_ex: 300,
                             Meowth_ex: 280}.get(cid, 100)
            scores.append(s)
        return _finish(scores, select)

    if context == SelectContext.DISCARD:
        own_hand = [o for o in select.option
                    if o.area == AreaType.HAND and (o.playerIndex is None or o.playerIndex == my_index)]
        if own_hand:
            # Ultra Ball's cost (and any other hand discard): pitch the least useful cards.
            scores = []
            for o in select.option:
                if o.area == AreaType.HAND and o.index < len(hand_usefulness):
                    scores.append(-hand_usefulness[o.index])
                else:
                    scores.append(0)
            return _finish(scores, select)
        # Bench shrink (Area Zero leaving): dump spent utility bodies, keep anything fueled.
        scores = []
        for o in select.option:
            p = get_card(obs, o.area, o.index, my_index)
            s = 0
            if p is not None and isinstance(p, Pokemon):
                base = {Meowth_ex: 1000, Fezandipiti_ex: 800, Teal_Mask_Ogerpon_ex: 300,
                        Raging_Bolt_ex: 200, Latias_ex: 150, Mega_Kangaskhan_ex: 100}.get(p.id, 0)
                s = base - 300 * len(p.energies)
                if field_counts[p.id] > 1:
                    s += 200
            scores.append(s)
        return _finish(scores, select)

    if context == SelectContext.DISCARD_ENERGY:
        # Retreat cost: pay with grass before the attack core.
        scores = []
        for o in select.option:
            et = None
            if o.energyIndex is not None:
                act = my_state.active[0] if my_state.active else None
                if act is not None and o.energyIndex < len(act.energies):
                    et = act.energies[o.energyIndex]
            scores.append(3 if et == EnergyType.GRASS else 1)
        return _finish(scores, select)

    # ==========================================================================================
    # MAIN (and any remaining context): score-and-sort over the generic option types
    # ==========================================================================================

    can_dance = any(o.type == OptionType.ABILITY
                    and (p := get_card(obs, o.area, o.index, my_index)) is not None
                    and p.id == Teal_Mask_Ogerpon_ex
                    for o in select.option)

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            score = _play_score(c, obs, state, my_state, op_state, field_counts, hand_counts,
                                discard_counts, stadium_id, core_missing, kills, useful_hand,
                                hand_usefulness, total_in_play, ogerpon_in_play, ogerpon_target,
                                trash_energy, deck_energy_left, can_dance, supporter_in_hand,
                                mine, stalled)
        elif o.type == OptionType.ATTACH:
            # Manual attach arrives as one option per (hand energy card x target Pokémon): the
            # energy sits at (area=HAND, index), the target at (inPlayArea, inPlayIndex). Score
            # the PAIR — attaching the wrong type to the right target wastes the turn's one attach.
            ecard = get_card(obs, o.area, o.index, my_index)
            target = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            et = card_table[ecard.id].energyType if (ecard is not None and ecard.id in card_table) else None
            score = 8000
            if target is not None:
                act = my_state.active[0] if my_state.active else None
                ready_bench = any(p is not None and p.id == Raging_Bolt_ex
                                  and all(t in p.energies for t in CORE_TYPES) for p in my_state.bench)
                fund_retreat = (act is not None and act.id != Raging_Bolt_ex and ready_bench
                                and not latias_in_play
                                and len(act.energies) < card_table[act.id].retreatCost)
                if fund_retreat and o.inPlayArea == AreaType.ACTIVE:
                    # Pay the stuck active's retreat so the ready Bolt can promote; spend grass.
                    score = 9610 if et == EnergyType.GRASS else 9600
                elif et in CORE_TYPES:
                    if target.id == Raging_Bolt_ex and et not in target.energies:
                        score = 9500 if o.inPlayArea == AreaType.ACTIVE else 9200
                    elif target.id == Raging_Bolt_ex:
                        score = 8150  # extra fuel rides on the attacker
                    elif target.id in (Mega_Kangaskhan_ex, Meowth_ex):
                        score = 8120
                    elif target.id == Teal_Mask_Ogerpon_ex:
                        score = 8110
                    else:
                        score = 8050
                elif et == EnergyType.GRASS:
                    if can_dance:
                        score = -1  # Teal Dance attaches it AND draws
                    elif ogerpon_in_play and hand_counts[Grass_Energy] <= 1:
                        score = -1  # hold the last grass for next turn's dance
                    else:
                        score = {Teal_Mask_Ogerpon_ex: 8060, Mega_Kangaskhan_ex: 8040,
                                 Meowth_ex: 8030}.get(target.id, 8000)
        elif o.type == OptionType.ABILITY:
            p = get_card(obs, o.area, o.index, my_index)
            score = -1
            if p is not None:
                if p.id == Fezandipiti_ex:
                    score = 58000 if my_state.deckCount > 3 else -1
                elif p.id == Mega_Kangaskhan_ex:
                    score = 57000 if my_state.deckCount > 2 else -1
                elif p.id == Teal_Mask_Ogerpon_ex:
                    score = 55000 if my_state.deckCount > 1 else -1
        elif o.type == OptionType.RETREAT:
            score = _retreat_score(my_state, hand_counts, latias_in_play)
        elif o.type == OptionType.ATTACK:
            score = _attack_score(o.attackId, my_state, op_state, kills, useful_hand,
                                  discard_counts, mine, stalled)
        elif o.type == OptionType.EVOLVE:
            score = 110000  # no evolutions in this deck; harmless default
        elif o.type == OptionType.END:
            score = 0
        scores.append(score)

    return _finish(scores, select)


def _search_pick_score(cid, eff_id, field_counts, hand_counts, core_missing, supporter_in_hand,
                       ogerpon_in_play, ogerpon_target, kills, my_state, state) -> int:
    """TO_HAND picks: deck/discard searches, branched on the activating card."""
    data = card_table.get(cid)
    et = data.energyType if data is not None else None
    if eff_id == Crispin:
        # Pick 1 of 2 (this one goes to HAND; the attach pick follows and must be a different
        # type). If two core types are missing, take one now and let the attach half place the
        # other. If at most one is missing, leave it for the attach half and BANK the other
        # scarce core type (only 4 {L} / 4 {F} in the deck; grass is plentiful).
        if len(core_missing) >= 2:
            if et == EnergyType.LIGHTNING:
                return 300
            if et == EnergyType.FIGHTING:
                return 290
            return 100
        if et in CORE_TYPES and et not in core_missing:
            return 300
        if et == EnergyType.GRASS:
            return 200 if hand_counts[Grass_Energy] < 2 else 120
        return 50
    if eff_id == Meowth_ex:
        # Last-Ditch Catch: fetch the Supporter this turn needs.
        if cid == Crispin and (core_missing or hand_counts[Lightning_Energy] + hand_counts[Fighting_Energy] < 2):
            return 300
        if cid == Boss_Orders and kills["gust_worth"]:
            return 280
        if cid == Lillie_Determination:
            return 260 if len(my_state.hand) <= 3 else 100
        if cid == Cyrano:
            return 240 if sum(field_counts.values()) < 4 else 90
        if cid == Crispin:
            return 230
        return 50
    if eff_id in (Energy_Retrieval, Night_Stretcher):
        if et in core_missing:
            return 300 if et == EnergyType.FIGHTING else 290
        if cid == Raging_Bolt_ex and field_counts[cid] <= 1:
            return 250
        if et == EnergyType.GRASS:
            return 200 if hand_counts[Grass_Energy] == 0 and ogerpon_in_play else 120
        if et in CORE_TYPES:
            return 150
        if cid == Teal_Mask_Ogerpon_ex and field_counts[cid] < ogerpon_target:
            return 140
        return 50
    # Ultra Ball / Cyrano / generic deck search: fill the target board.
    if cid == Raging_Bolt_ex:
        if field_counts[cid] == 0:
            return 320
        if field_counts[cid] + hand_counts[cid] < 2:
            return 280
        return 30
    if cid == Teal_Mask_Ogerpon_ex:
        return 300 if field_counts[cid] + hand_counts[cid] < ogerpon_target else 30
    if cid == Latias_ex:
        return 260 if field_counts[cid] + hand_counts[cid] < 1 else 20
    if cid == Meowth_ex:
        return 240 if not supporter_in_hand and field_counts[cid] == 0 and not state.supporterPlayed else 20
    if cid == Fezandipiti_ex:
        return 60 if field_counts[cid] + hand_counts[cid] < 1 else 10
    if cid == Mega_Kangaskhan_ex:
        return 50
    return 10


def _play_score(c, obs, state, my_state, op_state, field_counts, hand_counts, discard_counts,
                stadium_id, core_missing, kills, useful_hand, hand_usefulness, total_in_play,
                ogerpon_in_play, ogerpon_target, trash_energy, deck_energy_left, can_dance,
                supporter_in_hand, mine, stalled) -> int:
    """MAIN-phase PLAY option: the score band encodes within-turn ordering."""
    if c is None:
        return 0
    data = card_table[c.id]
    cid = c.id

    if data.cardType == CardType.POKEMON:
        # Bench-out safety valve: never be left with a lone active.
        if total_in_play <= 1:
            pref = {Teal_Mask_Ogerpon_ex: 0, Raging_Bolt_ex: 1, Latias_ex: 2, Fezandipiti_ex: 3,
                    Meowth_ex: 4, Mega_Kangaskhan_ex: 5}.get(cid, 6)
            return 200000 - pref
        if cid == Teal_Mask_Ogerpon_ex:
            return 100000 if field_counts[cid] < ogerpon_target else -1
        if cid == Raging_Bolt_ex:
            return 100000 if field_counts[cid] < 1 else (99500 if field_counts[cid] < 2 else -1)
        if cid == Latias_ex:
            return 100000 if field_counts[cid] < 1 else -1
        if cid == Meowth_ex:
            # Its whole point is the on-play Supporter search: bench it when we lack a supporter,
            # or lack CRISPIN specifically while the attack core is incomplete (Crispin is the
            # engine; a Boss in hand shouldn't block fetching it).
            need_crispin = bool(core_missing) and hand_counts[Crispin] == 0
            return 99000 if (field_counts[cid] == 0 and not state.supporterPlayed
                             and (not supporter_in_hand or need_crispin)) else -1
        if cid == Fezandipiti_ex:
            return 20000 if (state.turn >= 3 and field_counts[cid] == 0
                             and len(my_state.bench) + 2 <= my_state.benchMax) else -1
        if cid == Mega_Kangaskhan_ex:
            return -1  # 3-prize liability; it earns its spot only as the opening active
        return -1

    if data.cardType == CardType.BASIC_ENERGY:
        # Manual attach (the once-per-turn resource). Grass is Teal Dance fodder first.
        et = data.energyType
        if et in CORE_TYPES:
            for _, _, p in mine:
                if p.id == Raging_Bolt_ex and et not in p.energies:
                    return 9500
            return 8100
        if et == EnergyType.GRASS:
            if can_dance:
                return -1  # let Teal Dance attach it (same attach + a free card)
            if ogerpon_in_play and hand_counts[Grass_Energy] <= 1:
                return -1  # hold the last grass for next turn's dance
            return 8000
        return 8000

    if data.cardType == CardType.SUPPORTER:
        if cid == Boss_Orders:
            if kills["gust_worth"] and kills["bench_wins"]:
                return 130000  # gust for game
            # A gustable KO means the attack already works without Crispin's help this turn,
            # so converting it outranks every other supporter.
            return 27000 if kills["gust_worth"] else -1
        if cid == Crispin:
            if deck_energy_left <= 0:
                return -1
            if core_missing or hand_counts[Lightning_Energy] + hand_counts[Fighting_Energy] < 2:
                return 26000
            return 24000
        if cid == Cyrano:
            thin = (field_counts[Teal_Mask_Ogerpon_ex] + hand_counts[Teal_Mask_Ogerpon_ex] < 2
                    or field_counts[Raging_Bolt_ex] + hand_counts[Raging_Bolt_ex] < 2
                    or field_counts[Latias_ex] + hand_counts[Latias_ex] < 1)
            return 25000 if thin else -1
        if cid == Lillie_Determination:
            # A shuffle-draw-6 is the cheapest reset: it never discards, so on a stalled turn
            # (no way to finish the {L}{F} core) it outranks every other supporter.
            if stalled and not kills["act_kill"]:
                return 26500
            if useful_hand <= 3 and not kills["act_kill"]:
                return 23000 + (2000 if len(my_state.prize) == 6 else 0)
            return -1
        return -1

    if data.cardType == CardType.STADIUM:
        if cid == Area_Zero_Underdepths and stadium_id != Area_Zero_Underdepths and ogerpon_in_play:
            want_more = (hand_counts[Teal_Mask_Ogerpon_ex] > 0 or hand_counts[Raging_Bolt_ex] > 0
                         or hand_counts[Latias_ex] > 0)
            if len(my_state.bench) >= my_state.benchMax and want_more:
                return 110000  # unlock the bench before this turn's Pokémon plays
            if len(my_state.bench) >= 4 or stadium_id != 0:
                return 30000
        return -1

    # Items
    if cid == Ultra_Ball:
        expendable = sum(1 for u in hand_usefulness if u <= 0) - (1 if hand_usefulness else 0)
        wanted = (field_counts[Teal_Mask_Ogerpon_ex] + hand_counts[Teal_Mask_Ogerpon_ex] < ogerpon_target
                  or field_counts[Raging_Bolt_ex] + hand_counts[Raging_Bolt_ex] < 2
                  or field_counts[Latias_ex] + hand_counts[Latias_ex] < 1)
        return 45000 if expendable >= 2 and wanted else -1
    if cid == Energy_Switch:
        if core_missing:
            core_bolt, _ = _missing_core(my_state)
            for _, _, p in mine:
                if ((core_bolt is None or p.serial != core_bolt.serial)
                        and any(card_table[ec.id].energyType in core_missing
                                for ec in p.energyCards if ec.id in card_table)):
                    return 60000
        return -1
    if cid == Energy_Retrieval:
        missing_in_trash = any(discard_counts[e] > 0 for e in (Lightning_Energy, Fighting_Energy)
                               if card_table[e].energyType in core_missing)
        if missing_in_trash:
            return 62000  # the reload loop: trash -> hand -> manual attach -> attack
        if trash_energy >= 2:
            return 61000
        return 40000 if trash_energy == 1 else -1
    if cid == Night_Stretcher:
        needs = ((core_missing and any(discard_counts[e] > 0 for e in (Lightning_Energy, Fighting_Energy)))
                 or (discard_counts[Raging_Bolt_ex] > 0 and field_counts[Raging_Bolt_ex] <= 1)
                 or (discard_counts[Teal_Mask_Ogerpon_ex] > 0
                     and field_counts[Teal_Mask_Ogerpon_ex] < ogerpon_target))
        return 44000 if needs else -1
    if cid == Glass_Trumpet:
        bank = any(p.id in (Mega_Kangaskhan_ex, Meowth_ex) and a == AreaType.BENCH for a, i, p in mine)
        return 43000 if trash_energy >= 1 and bank else -1
    if cid == Prime_Catcher:
        if kills["gust_worth"] and kills["bench_wins"]:
            return 125000
        if kills["gust_worth"] and (state.supporterPlayed or hand_counts[Boss_Orders] == 0):
            return 21000
        return -1
    return -1


def _retreat_score(my_state, hand_counts, latias_in_play) -> int:
    """Retreat when the active isn't our attacker and a bench Bolt could take over. The engine
    only offers RETREAT when the cost is payable; the tiering keeps us from burning 2-3 attached
    energies unless the promoted Bolt is actually ready."""
    act = my_state.active[0] if my_state.active else None
    if act is None:
        return -1
    if act.id == Raging_Bolt_ex and all(t in act.energies for t in CORE_TYPES):
        return -1  # the attacker is already up
    bench_bolts = [p for p in my_state.bench if p is not None and p.id == Raging_Bolt_ex]
    if act.id == Raging_Bolt_ex:
        # Active is a Bolt still missing core: swapping for a sibling rarely helps.
        bench_bolts = [p for p in bench_bolts
                       if sum(1 for t in CORE_TYPES if t in p.energies)
                       > sum(1 for t in CORE_TYPES if t in act.energies)]
    if not bench_bolts:
        return -1
    ready = any(all(t in p.energies for t in CORE_TYPES) for p in bench_bolts)
    free = latias_in_play or card_table[act.id].retreatCost == 0
    cheap = card_table[act.id].retreatCost <= 1
    if ready:
        return 12000 if (free or cheap) else 11000  # even 2-3 energies: enabling the attacker pays
    # Partial Bolt: promote only when this turn can plausibly finish the core.
    fixable = (hand_counts[Lightning_Energy] + hand_counts[Fighting_Energy] > 0
               or hand_counts[Crispin] > 0 or hand_counts[Energy_Switch] > 0)
    if fixable:
        return 10500 if free else (10000 if cheap else -1)
    return -1


def _attack_score(attack_id, my_state, op_state, kills, useful_hand, discard_counts, mine,
                  stalled) -> int:
    """Attacks close the turn: lowest positive band. Bellowing lethal > utility attacks > chip."""
    defender = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    if attack_id == ATK_BELLOWING:
        if kills["act_kill"]:
            return 150000 if kills["act_wins"] else 3000
        # Chip with renewable grass only (the discard select enforces the count).
        chip = sum(len([ec for ec in p.energyCards if ec.id == Grass_Energy])
                   for _, _, p in mine if p.id == Teal_Mask_Ogerpon_ex)
        return 2000 if chip >= 1 else 50
    if attack_id == ATK_BURST_ROAR:
        # The hand reset of last resort: it torches the hand, so demand a dead one — but a
        # stalled hand that can't complete {L}{F} counts as dead enough.
        if my_state.deckCount > 6:
            if useful_hand <= 1:
                return 1500
            if stalled and useful_hand <= 5:
                return 1450
        return -1
    if attack_id == ATK_RAPID_FIRE:
        return 2400  # Kangaskhan is active with 3 energies: 200+ beats any chip
    if attack_id == ATK_LEAF_SHOWER:
        act = my_state.active[0] if my_state.active else None
        if act is not None and defender is not None:
            dmg = 30 + 30 * (len(act.energies) + len(defender.energies))
            if dmg >= defender.hp:
                return 2600
        return 700
    if attack_id == ATK_CRUEL_ARROW:
        targets = [p for p in ([defender] if defender else []) + list(op_state.bench)
                   if p is not None and p.hp <= 100]
        if targets:
            return 2300
        return 600
    if attack_id == ATK_TUCK_TAIL:
        return -1
    return 100


if __name__ == "__main__":
    # Smallest self-check: the module loads, the deck is intact, and the agent answers a synthetic
    # deck-return call. Full battles are exercised via utils/run_smoketest_battle.py and eval.py.
    assert len(my_deck) == 60, len(my_deck)
    assert my_deck.count(Raging_Bolt_ex) == 4 and my_deck.count(Teal_Mask_Ogerpon_ex) == 4
    assert agent({"select": None, "current": None, "logs": []}) == my_deck
    print(f"simple_raging_bolt_burn OK: deck of {len(my_deck)}, "
          f"{len(set(my_deck))} unique cards; agent() deck-return path works")
