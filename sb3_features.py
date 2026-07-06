#!/usr/bin/env python3
"""cg-battle featurization for the SB3 stack: `encode_state` / `encode_options` + their dims.

Moved verbatim (2026-07-06) out of the retired `legacy_ppo/ppo_model.py` — the SB3 env
(`sb3_env.obs_to_gym`) and the eval adapter are the only consumers now, so the featurization lives
here as an SB3-owned module rather than reaching back into the paused hand-rolled stack. The
functions are byte-identical to the originals (no logic change): existing `out/*.zip` checkpoints
were trained against these exact features and keep loading.

Both encoders take an `Observation` (from `to_observation_class`) framed from the ACTING seat and
return dense features PLUS integer card/attack id indices. The stock `MultiInputPolicy` has no
embedding tables, so `obs_to_gym` currently DISCARDS the id indices — they are preserved here for a
future embedding-capable policy (and so the move stays a pure move). See
`tcg_model.get_decoder_input` for the authoritative OptionType→field dispatch `encode_options`
mirrors.

Build + shape self-test:
    python sb3_features.py
"""

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: importing tcg_model runs the `CG_LIB_PATH` sys.path insert
# (tcg_model.py, the `_CG_LIB_PATH` block) that makes the vendored `cg` package importable —
# keep it above the `cg.api` import. Besides the select-contract helpers it provides the card
# DB + option-target resolver: `card_table[cardId] -> CardData` (static stats);
# `get_card(obs, area, index, playerIndex)` resolves an option's referenced card exactly as
# tcg_model.get_decoder_input does.
from tcg_model import (card_table, card_count, attack_count, get_card,  # noqa: E402
                       is_policy_select, auto_resolve)
from cg.api import (                                                    # noqa: E402
    OptionType, EnergyType, CardType, SelectType, SpecialConditionType,
    all_attack, to_observation_class,
)

# attackId -> Attack (damage, energies=cost). Built once at import (all_attack is cheap metadata).
attack_table = {a.attackId: a for a in all_attack()}


# ---------------------------------------------------------------------------------------------
# Enum cardinalities (discovered from the engine so the widths can't silently drift if the SDK
# adds a member). Card/attack vocab sizes come from tcg_model (max id + 1).
# ---------------------------------------------------------------------------------------------
N_ENERGY_TYPES = max(e.value for e in EnergyType) + 1        # 12
N_OPTION_TYPES = max(o.value for o in OptionType) + 1        # 17
N_CARD_TYPES = max(c.value for c in CardType) + 1            # 7
N_SELECT_TYPES = max(s.value for s in SelectType) + 1        # 11
N_SPECIAL_COND = max(s.value for s in SpecialConditionType) + 1  # 5

BENCH_SLOTS = 5             # engine benchMax is 5; we lay out 5 explicit (order-sensitive) slots
# Sentinel id per vocab (padding row): valid ids are [0, card_count-1] / [0, attack_count-1]; the
# sentinel is the extra index at card_count / attack_count, used for empty slots / unknown ids.
CARD_PAD = card_count
ATTACK_PAD = attack_count

# ---------------------------------------------------------------------------------------------
# Feature-block widths, summed into the two public dims. Defined as arithmetic over the constants
# above (NOT magic literals) so STATE_DIM / OPTION_DIM always equal what the encoders emit. The
# self-test (`python sb3_features.py`) asserts the emitted shapes match.
# ---------------------------------------------------------------------------------------------
# State ---------------------------------------------------------------------------------------
_GLOBAL_DIM = 8 + N_SELECT_TYPES + 2                          # turn/flags/stadium + selType + remain*
_ACTIVE_DIM = 4 + N_ENERGY_TYPES + 3 + N_ENERGY_TYPES + N_ENERGY_TYPES + 5  # 48
_BENCH_SLOT_DIM = 3 + N_ENERGY_TYPES + 1                      # 16
_SIDE_DIM = 6 + N_SPECIAL_COND + _ACTIVE_DIM + BENCH_SLOTS * _BENCH_SLOT_DIM
STATE_DIM = _GLOBAL_DIM + 2 * _SIDE_DIM                       # length of encode_state(o)[0]
N_STATE_SLOTS = 2 * (1 + BENCH_SLOTS)                         # card-id slots: 2 sides × (active+5 bench)

# Options -------------------------------------------------------------------------------------
_OPT_STATIC_DIM = N_CARD_TYPES + 5 + 1 + N_ENERGY_TYPES + 1   # primary card static block = 26
_OPT_PRIMARY_DYN_DIM = 3                                      # primary-if-Pokémon: hp_frac, n_energy, dmg_frac
_OPT_TARGET_DIM = 3                                           # ATTACH/EVOLVE in-play target: present, hp_frac, n_energy
_OPT_ATTACK_DIM = 1 + 1 + N_ENERGY_TYPES                      # damage, cost-len, cost histogram = 14
_OPT_MISC_DIM = 1 + 1 + N_SPECIAL_COND                        # number, energy-count, specialCond one-hot = 7
OPTION_DIM = (N_OPTION_TYPES + _OPT_STATIC_DIM + _OPT_PRIMARY_DYN_DIM
              + _OPT_TARGET_DIM + _OPT_ATTACK_DIM + _OPT_MISC_DIM)  # per-option dense width
N_OPT_CARD_SLOTS = 2                                          # [primary cardId, target cardId]
N_OPT_ATTACK_SLOTS = 1                                        # [attackId]


# ============================================================================================
# Featurization helpers (small, dense, None-safe). All normalize to ~[0,1].
# NOTE: this is STATIC hand-scaling (fixed divisors), NOT a running observation mean/std normalizer.
# The ICLR "37 details" flags running obs-normalization as "very helpful"; we rely on hand-picked
# scales instead — simpler, no running-stats state, but the divisors are only approximately right.
# ============================================================================================
def _onehot(idx, n):
    v = [0.0] * n
    if idx is not None and 0 <= idx < n:
        v[idx] = 1.0
    return v


def _is_pokemon(obj):
    """In-play Pokémon carry dynamic state (hp/energies); plain hand/deck/discard cards don't."""
    return obj is not None and hasattr(obj, "maxHp")


def _safe_card_id(cid):
    return cid if (cid is not None and 0 <= cid < card_count) else CARD_PAD


def _safe_attack_id(aid):
    return aid if (aid is not None and 0 <= aid < attack_count) else ATTACK_PAD


def _energy_hist(poke):
    """Count of attached effective energy per EnergyType (as the engine reports Pokemon.energies),
    scaled by /4 so a typical 1-4 energy Pokémon sits in ~[0,1]."""
    h = [0.0] * N_ENERGY_TYPES
    if poke is not None and getattr(poke, "energies", None):
        for e in poke.energies:
            ei = int(e)
            if 0 <= ei < N_ENERGY_TYPES:
                h[ei] += 0.25
    return h


def _card_static(card_id):
    """[_OPT_STATIC_DIM] static card attributes from card_table; all-zero if id unknown.

    ex / megaEx are emitted as SEPARATE flags. DOC-vs-ENGINE INCONSISTENCY (verify with
    all_card_data(), do not take on faith): the API doc ("docs/cabt engine/1. API module.md",
    CardData) says `ex` is "True if Pokémon ex (including Mega Evolution Pokémon ex)", but in the
    live card table `ex` and `megaEx` are MUTUALLY EXCLUSIVE — all 30 megaEx cards have ex=False and
    no card has both. Per the repo convention the compiled engine is ground truth over docs
    (CLAUDE.md "Conventions"; tcg-rules-critic skill), and the existing rule-based agents already
    treat the flags as disjoint (`3 if data.megaEx else 2 if data.ex else 1` — the KO prize
    liability — in game_agents/rulebased/lucario.py & dragapult_ex.py). We follow suit and read both.
    RISK: any downstream code that assumes `ex` ⊇ `megaEx` (e.g. "is it an ex? check `.ex`") will
    misclassify all Mega cards as non-ex — use `(ex or megaEx)` for "any ex Pokémon"."""
    cd = card_table.get(card_id) if card_id is not None else None
    ctype = _onehot(int(cd.cardType) if cd else None, N_CARD_TYPES)
    flags = [float(cd.basic), float(cd.stage1), float(cd.stage2), float(cd.ex), float(cd.megaEx)] if cd else [0.0] * 5
    hp = [(cd.hp / 400.0) if cd else 0.0]
    etype = _onehot(int(cd.energyType) if cd and cd.energyType is not None else None, N_ENERGY_TYPES)
    retreat = [(cd.retreatCost / 4.0) if cd else 0.0]
    return ctype + flags + hp + etype + retreat


def _active_block(poke):
    """[_ACTIVE_DIM] rich features for a side's Active Pokémon (None -> all zero). Weakness /
    Resistance (materially strategic: attack damage steps 3-4) included here as type one-hots."""
    has = poke is not None
    hp_frac = (poke.hp / poke.maxHp) if has and poke.maxHp else 0.0
    maxhp = (poke.maxHp / 400.0) if has else 0.0
    n_energy = (len(poke.energies) / 8.0) if has and poke.energies else 0.0
    hist = _energy_hist(poke)
    n_tools = (len(poke.tools) / 2.0) if has and getattr(poke, "tools", None) else 0.0
    appear = float(poke.appearThisTurn) if has else 0.0
    cd = card_table.get(poke.id) if has else None
    retreat = (cd.retreatCost / 4.0) if cd else 0.0
    weak = _onehot(int(cd.weakness) if cd and cd.weakness is not None else None, N_ENERGY_TYPES)
    resist = _onehot(int(cd.resistance) if cd and cd.resistance is not None else None, N_ENERGY_TYPES)
    flags = [float(cd.basic), float(cd.stage1), float(cd.stage2), float(cd.ex), float(cd.megaEx)] if cd else [0.0] * 5
    return [float(has), hp_frac, maxhp, n_energy] + hist + [n_tools, appear, retreat] + weak + resist + flags


def _bench_slot(poke):
    """[_BENCH_SLOT_DIM] compact per-bench-slot features (None -> all zero)."""
    has = poke is not None
    hp_frac = (poke.hp / poke.maxHp) if has and poke.maxHp else 0.0
    n_energy = (len(poke.energies) / 8.0) if has and poke.energies else 0.0
    hist = _energy_hist(poke)
    cd = card_table.get(poke.id) if has else None
    # (megaEx or ex) = "any ex" bench Pokémon = extra prize liability. ex/megaEx are disjoint in the
    # engine, so the union is the correct test — see _card_static for the doc-vs-engine note.
    ex = float(cd.megaEx or cd.ex) if cd else 0.0
    return [float(has), hp_frac, n_energy] + hist + [ex]


def _side_block(ps, ids_out):
    """[_SIDE_DIM] one player's board (counts, status, active, 5 bench slots). Appends this side's
    (1 active + 5 bench) card ids to `ids_out`. Opponent hidden zones use counts only."""
    occupied = len(ps.active) > 0
    active = ps.active[0] if occupied else None       # element may still be None (facedown at setup)
    counts = [
        len(ps.prize) / 6.0,
        ps.deckCount / 60.0,
        ps.handCount / 10.0,                          # handCount works for both seats (opp hand hidden)
        len(ps.discard) / 60.0,
        len(ps.bench) / 5.0,
        (len(ps.bench) + (1 if occupied else 0)) / 6.0,   # in-play count: the KO-out win-condition signal
    ]
    status = [float(ps.poisoned), float(ps.burned), float(ps.asleep), float(ps.paralyzed), float(ps.confused)]
    feats = counts + status + _active_block(active)
    ids_out.append(_safe_card_id(active.id if active is not None else None))
    for j in range(BENCH_SLOTS):
        p = ps.bench[j] if j < len(ps.bench) else None
        feats += _bench_slot(p)
        ids_out.append(_safe_card_id(p.id if p is not None else None))
    return feats


# ============================================================================================
# FEATURIZATION — encode_state / encode_options. Both take an `Observation` (from
# to_observation_class) framed from the ACTING seat (obs.current.yourIndex).
# ============================================================================================
def encode_state(o):
    """-> (dense: Tensor[STATE_DIM], card_ids: LongTensor[N_STATE_SLOTS]).

    Perspective-relative fixed-size board "dashboard" (me = players[yourIndex] laid out before the
    opponent) PLUS the id indices for the 12 in-play Pokémon slots (active + 5 bench, both sides),
    which an embedding policy can consume. The dense part feeds the value head AND is broadcast as
    shared context into every option's score. Prize/in-play/deck counts cover all 3 win conditions."""
    st = o.current
    sel = o.select
    yi = st.yourIndex
    me = st.players[yi]
    opp = st.players[1 - yi]

    g = [
        st.turn / 20.0,
        st.turnActionCount / 10.0,
        float(st.firstPlayer == yi),
        float(st.energyAttached),
        float(st.supporterPlayed),
        float(st.stadiumPlayed),
        float(st.retreated),
        float(len(st.stadium) > 0),
    ]
    g += _onehot(int(sel.type) if sel is not None else None, N_SELECT_TYPES)   # decision category
    g += [
        (sel.remainEnergyCost / 10.0) if sel is not None else 0.0,
        (sel.remainDamageCounter / 10.0) if sel is not None else 0.0,
    ]

    ids = []
    feats = g + _side_block(me, ids) + _side_block(opp, ids)
    return torch.tensor(feats, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)


def _resolve_option(o, opt, yi):
    """Map one Option to (primary_id, primary_poke, target_id, target_poke, attack_id), mirroring
    tcg_model.get_decoder_input's `match o.type` dispatch. `*_poke` is set only when the referenced
    thing is an in-play Pokémon (so its dynamic hp/energy can be featurized)."""
    t = OptionType(opt.type)
    me = o.current.players[yi]
    pid = ppoke = tid = tpoke = aid = None

    def _prim(obj):
        nonlocal pid, ppoke
        if obj is not None:
            pid = getattr(obj, "id", None)
            if _is_pokemon(obj):
                ppoke = obj

    if t == OptionType.PLAY:
        if me.hand and opt.index is not None and opt.index < len(me.hand):
            _prim(me.hand[opt.index])
    elif t in (OptionType.ATTACH, OptionType.EVOLVE):
        _prim(get_card(o, opt.area, opt.index, yi))
        tgt = get_card(o, opt.inPlayArea, opt.inPlayIndex, yi)
        if tgt is not None:
            tid = getattr(tgt, "id", None)
            if _is_pokemon(tgt):
                tpoke = tgt
    elif t in (OptionType.ABILITY, OptionType.DISCARD):
        _prim(get_card(o, opt.area, opt.index, yi))
    elif t == OptionType.RETREAT:
        _prim(me.active[0] if len(me.active) > 0 else None)
    elif t == OptionType.CARD:
        _prim(get_card(o, opt.area, opt.index, opt.playerIndex))
    elif t == OptionType.TOOL_CARD:
        c = get_card(o, opt.area, opt.index, opt.playerIndex)
        if c is not None and getattr(c, "tools", None) and opt.toolIndex is not None and opt.toolIndex < len(c.tools):
            _prim(c.tools[opt.toolIndex])
    elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
        c = get_card(o, opt.area, opt.index, opt.playerIndex)
        if c is not None and getattr(c, "energyCards", None) and opt.energyIndex is not None and opt.energyIndex < len(c.energyCards):
            _prim(c.energyCards[opt.energyIndex])
    elif t == OptionType.SKILL:
        pid = opt.cardId
    elif t == OptionType.ATTACK:
        aid = opt.attackId
    # NUMBER / YES / NO / END carry no card/attack reference — the type one-hot is their whole signal.
    return pid, ppoke, tid, tpoke, aid


def _option_row(pid, ppoke, tid, tpoke, aid, opt):
    """[OPTION_DIM] dense features for one option. Row order MUST match option index order (the
    sampled row i is returned verbatim as the cg selection [i]). The OptionType one-hot guarantees
    a nonzero bit so a zero-padded row (added when collated to N_MAX) is unambiguously distinguishable."""
    row = _onehot(int(opt.type), N_OPTION_TYPES)
    row += _card_static(pid)
    # primary dynamic (only if the primary is an in-play Pokémon)
    if ppoke is not None and ppoke.maxHp:
        row += [ppoke.hp / ppoke.maxHp, (len(ppoke.energies) / 8.0) if ppoke.energies else 0.0,
                (ppoke.maxHp - ppoke.hp) / ppoke.maxHp]
    else:
        row += [0.0, 0.0, 0.0]
    # ATTACH/EVOLVE in-play target
    if tpoke is not None and tpoke.maxHp:
        row += [1.0, tpoke.hp / tpoke.maxHp, (len(tpoke.energies) / 8.0) if tpoke.energies else 0.0]
    else:
        row += [0.0, 0.0, 0.0]
    # attack block (damage is THE signal for choosing among attacks)
    a = attack_table.get(aid) if aid is not None else None
    if a is not None:
        chist = [0.0] * N_ENERGY_TYPES
        for e in a.energies:
            ei = int(e)
            if 0 <= ei < N_ENERGY_TYPES:
                chist[ei] += 0.25
        row += [a.damage / 350.0, len(a.energies) / 5.0] + chist
    else:
        row += [0.0, 0.0] + [0.0] * N_ENERGY_TYPES
    # misc
    number = (opt.number / 10.0) if opt.number is not None else 0.0
    ecount = (opt.count / 4.0) if opt.count is not None else 0.0
    sc = _onehot(int(opt.specialConditionType) if opt.specialConditionType is not None else None, N_SPECIAL_COND)
    row += [number, ecount] + sc
    return row


def encode_options(o):
    """-> (dense: Tensor[n, OPTION_DIM], card_ids: LongTensor[n, 2], attack_ids: LongTensor[n, 1]),
    one row per legal option in `o.select.option` (n == len(option)), in index order. An embedding
    policy would embed the ids and score each row independently (no cross-option attention), so row
    i depends only on option i + the shared state.

    CAVEAT (marked): two options that resolve to the same card/attack with no distinguishing
    dynamic field get identical rows -> identical logits (the policy can't prefer one). Learned
    id-embeddings would mitigate the common case (different cards), but genuinely identical options
    remain indistinguishable — acceptable for v1 (they're usually equivalent)."""
    yi = o.current.yourIndex
    rows, cids, aids = [], [], []
    for opt in o.select.option:
        try:
            pid, ppoke, tid, tpoke, aid = _resolve_option(o, opt, yi)
            rows.append(_option_row(pid, ppoke, tid, tpoke, aid, opt))
        except (IndexError, AttributeError, TypeError):
            # A malformed/unexpected option can't crash a long training run — fall back to a
            # type-only row with padded ids. (Should not happen: engine surfaces only legal options.)
            pid = tid = aid = None
            rows.append(_option_row(None, None, None, None, None, opt))
        cids.append([_safe_card_id(pid), _safe_card_id(tid)])
        aids.append([_safe_attack_id(aid)])
    return (torch.tensor(rows, dtype=torch.float32),
            torch.tensor(cids, dtype=torch.long),
            torch.tensor(aids, dtype=torch.long))


if __name__ == "__main__":
    # Build + shape self-test: drive a sample_deck mirror with auto_resolve until a policy node and
    # assert the encoders emit exactly STATE_DIM / OPTION_DIM (+ the id-slot widths).
    import random
    from cg.game import battle_start, battle_select, battle_finish
    from tcg_model import sample_deck

    random.seed(0)
    checked = 0
    for _ in range(2):
        obs, start = battle_start(sample_deck, sample_deck)
        assert start.errorPlayer < 0, "deck rejected"
        while obs["current"]["result"] < 0:
            o = to_observation_class(obs)
            sel = o.select
            if is_policy_select(sel):
                n = len(sel.option)
                dense_s, state_ids = encode_state(o)
                dense_o, cids, aids = encode_options(o)
                assert dense_s.shape == (STATE_DIM,), (dense_s.shape, STATE_DIM)
                assert state_ids.shape == (N_STATE_SLOTS,) and state_ids.dtype == torch.long
                assert dense_o.shape == (n, OPTION_DIM), (dense_o.shape, OPTION_DIM)
                assert cids.shape == (n, N_OPT_CARD_SLOTS) and aids.shape == (n, N_OPT_ATTACK_SLOTS)
                assert cids.dtype == torch.long and aids.dtype == torch.long
                checked += 1
            obs = battle_select(auto_resolve(sel))
        battle_finish()
    print(f"OK: sb3_features shape self-test (STATE_DIM={STATE_DIM}, OPTION_DIM={OPTION_DIM}, "
          f"policy nodes checked={checked})")
