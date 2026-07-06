"""Reusable PPO implementation layer for the Pokémon TCG `cg` agent.

This is the *model/algorithm* half of the from-scratch PPO trainer — extracted from
`train_ppo.py` so the driver loop stays lean (mirrors how `tcg_model.py` holds the reusable
model layer for the other trainers). The training driver (battle collection, opponents, the
loop, argparse) lives in `train_ppo.py`, which imports `PPO`, `STATE_DIM`, `OPTION_DIM` from
here.

Design:
  * FRESH network from scratch — we do NOT reuse MyModel or its Transformer/sparse featurizers.
  * GAE(λ) advantages + TD(λ) value targets (value bootstrap between the learner's own decision
    steps; terminal-only ±1 reward). This replaced the original PPO-PyTorch-style Monte-Carlo
    returns after the measured broken-critic plateau (explained_var ~0.1 — docs/ppo/ppo_ceiling_report.md);
    γ=1.0 + λ=0.95 defaults follow the NashPG reference config for the same two-player zero-sum
    terminal-only reward shape (THEORY-REVIEW FLAG, see PPO.update).

FEATURIZATION (implemented; was the step-1 stub): `encode_state` / `encode_options` build a
RICH, hybrid representation — dense normalized scalars/one-hots PLUS integer card/attack ID
indices that `ActorCritic` embeds. This is deliberately richer than the original "flat scalar
dashboard" idea and, per an explicit design decision, DOES reintroduce a learned per-card /
per-attack vocabulary (which the very first sketch had ruled out) because the id-embeddings are
the single biggest richness lever. Typed features carry the deck-agnostic signal; the embeddings
specialize to the cards actually seen. See `tcg_model.get_decoder_input` for the authoritative
map of which `Option` fields each `OptionType` uses — `encode_options` mirrors that dispatch.

LOAD-BEARING invariant: the embeddings live INSIDE `ActorCritic` and are applied at act/evaluate
time from stored INDEX tensors. The encoders emit indices, NOT embeddings; the buffer stores the
indices; `PPO.update` re-embeds them so gradients reach the embedding tables. If we embedded in
the encoders and stored detached vectors, the tables would never train.

THE one structural departure from the PPO reference: cg's action space is NOT a fixed-width
discrete set. Each decision exposes a variable list `obs.select.option` whose length AND meaning
change every step, so we cannot use the reference's `Linear(H, action_dim) → Softmax`. Instead we
use a PER-OPTION SCORING head: featurize each legal option, score each to a scalar logit, then
take a (masked) softmax over exactly the options present this step.

v1 coverage limit: we only learn SINGLE-SELECT policy nodes (minCount==maxCount==1,
len(option)>1, not all-NUMBER — `is_policy_select`). Every other decision is auto-resolved
(`auto_resolve`) and emits no training record. Multi-select is a documented future extension.
"""

import math
import os
import sys

import torch
import torch.nn as nn
from torch.distributions import Categorical

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
HIDDEN = 256
# Learned id-embeddings. Sentinel row per table (padding_idx) = a zero vector w/ zero gradient,
# used for empty board slots and padded option rows. Valid ids are [0, card_count-1] /
# [0, attack_count-1]; the sentinel is the extra row at index card_count / attack_count.
D_CARD = 8
D_ATTACK = 8
CARD_PAD = card_count
ATTACK_PAD = attack_count

# ---------------------------------------------------------------------------------------------
# Feature-block widths, summed into the two public dims. Defined as arithmetic over the constants
# above (NOT magic literals) so STATE_DIM / OPTION_DIM always equal what the encoders emit. The
# self-test (`python ppo_model.py`) asserts the emitted shapes match.
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


def feature_groups():
    """Named groups over the 299 dense state dims -> dict[str, list[int]], built from the same
    width constants as STATE_DIM so it cannot drift from encode_state. Consumers: the feature-
    attribution probe (probe_credit.py --attrib) aggregates per-dim attributions into these
    groups. The groups exactly partition range(STATE_DIM) (self-test-asserted). Card-id
    embedding slots are NOT dense dims and are handled by the probe separately."""
    groups = {"turn": [0], "global_other": list(range(1, _GLOBAL_DIM))}
    for side, base in (("own", _GLOBAL_DIM), ("opp", _GLOBAL_DIM + _SIDE_DIM)):
        a = base + 6 + N_SPECIAL_COND                          # _active_block start
        b = a + _ACTIVE_DIM                                    # bench slots start
        bench_energy, bench_other = [], []
        for j in range(BENCH_SLOTS):
            s = b + j * _BENCH_SLOT_DIM                        # [has, hp_frac, n_energy, hist*12, ex]
            bench_energy += [s + 2] + list(range(s + 3, s + 3 + N_ENERGY_TYPES))
            bench_other += [s, s + 1, s + 3 + N_ENERGY_TYPES]
        groups.update({
            f"{side}_prizes": [base + 0],
            f"{side}_deck": [base + 1],
            f"{side}_hand": [base + 2],
            f"{side}_discard": [base + 3],
            f"{side}_board_counts": [base + 4, base + 5],
            f"{side}_status": list(range(base + 6, base + 6 + N_SPECIAL_COND)),
            # n_energy + energy hist of the active (the monotone energy-accumulation signal)
            f"{side}_active_energy": [a + 3] + list(range(a + 4, a + 4 + N_ENERGY_TYPES)),
            # has, hp_frac, maxhp + tools/appear/retreat + weak/resist one-hots + stage flags
            f"{side}_active_other": [a, a + 1, a + 2] + list(range(a + 4 + N_ENERGY_TYPES, a + _ACTIVE_DIM)),
            f"{side}_bench_energy": bench_energy,
            f"{side}_bench_other": bench_other,
        })
    return groups


def prize_potential(dense_state):
    """Phi(s) = (opp prizes remaining - own prizes remaining) / 6 in [-1, 1], read from the dense
    state (indices via the layout constants — the /6.0-normalized `len(ps.prize)` count leads each
    _side_block). Rises when WE take prizes (taking a prize removes a card from OUR OWN prize
    zone). Used as the potential for --shape prizediff (train_ppo.py) and logged as phi_seq by
    probe_credit.py; values are exact multiples of 1/6. Trust it only where
    prize_potential_valid() — the raw obs prize lists are select-type-dependent."""
    return float(dense_state[_GLOBAL_DIM + _SIDE_DIM] - dense_state[_GLOBAL_DIM])


def deck_potential(dense_state):
    """Phi_deck(s) = (own deckCount - opp deckCount) / 60 in [-1, 1], read from the dense state
    (the /60-normalized deckCount is slot base+1 of each _side_block). HIGHER = better for the
    acting seat: you LOSE by starting your turn with 0 deck cards, so own-minus-opp rising means
    the opponent hits deck-out first. The win-mode-aligned potential candidate for this policy's
    measured deck-out win mode (work_log 2026-07-03 DISCOVERY 2). NOT monotone within a game
    (Iono-style effects shuffle hands back into decks). Read it only where
    prize_potential_valid() — conservative reuse; whether deck counts share the CARD-select
    transients of the prize lists is unmeasured (see the deckCount transient mini-measurement in
    work_log)."""
    return float(dense_state[_GLOBAL_DIM + 1] - dense_state[_GLOBAL_DIM + _SIDE_DIM + 1])


def board_potential(dense_state):
    """Phi_board(s) = (own in-play count - opp in-play count) / 6 in [-1, 1] (the /6-normalized
    bench+active count is slot base+5 of each _side_block — the KO/bench-out win-condition
    signal). HIGHER = better for the acting seat: you LOSE with no Pokémon in play, so
    own-minus-opp rising means the opponent is closer to bench-out. Same validity caveat as
    deck_potential: gate reads on prize_potential_valid()."""
    return float(dense_state[_GLOBAL_DIM + 5] - dense_state[_GLOBAL_DIM + _SIDE_DIM + 5])


# selType one-hot offset inside the global block (derived, not a literal: the 8 pre-one-hot slots
# = _GLOBAL_DIM minus the one-hot and the 2 remain* slots).
_SELTYPE_OFF = _GLOBAL_DIM - N_SELECT_TYPES - 2


def prize_potential_valid(dense_state):
    """True when prize_potential is trustworthy at this decision. Measured 2026-07-03 (30 games,
    1104 MAIN readings, 0 violations): at MAIN selects both prize counts are monotone-decreasing
    from 6; at CARD selects your own prize zone can transiently read EMPTY during a prize pick
    (a spurious phi = +1 spike), and pre-deal setup selects read 0/0. So: MAIN select AND prizes
    dealt (not both-zero — a legitimate both-zero cannot occur mid-decision: taking the 6th
    prize ends the game immediately)."""
    return bool(dense_state[_SELTYPE_OFF + int(SelectType.MAIN)] == 1.0
                and (dense_state[_GLOBAL_DIM] > 0 or dense_state[_GLOBAL_DIM + _SIDE_DIM] > 0))


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
    which `ActorCritic` embeds. The dense part feeds the value head AND is broadcast as shared
    context into every option's score. Prize/in-play/deck counts cover all 3 win conditions."""
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
    a nonzero bit so a zero-padded row (added in _collate) is unambiguously distinguishable."""
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
    one row per legal option in `o.select.option` (n == len(option)), in index order. `ActorCritic`
    embeds the ids and scores each row independently (no cross-option attention), so row i depends
    only on option i + the shared state.

    CAVEAT (marked): two options that resolve to the same card/attack with no distinguishing
    dynamic field get identical rows -> identical logits (the policy can't prefer one). The learned
    id-embeddings mitigate the common case (different cards), but genuinely identical options remain
    indistinguishable — acceptable for v1 (they're usually equivalent)."""
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


# ============================================================================================
# NETWORK — fresh ActorCritic. Separate value + policy MLPs (no shared trunk); the policy is a
# per-option SCORER, not a fixed-width Softmax. Learned card/attack embeddings applied HERE (so
# gradients flow to them at update time) from the index tensors the encoders emit.
# ============================================================================================
class ActOutput:
    """What act() hands back to the collector (so the buffer can store the exact featurized inputs
    + the act-time logprob/value the PPO ratio must be computed against). Stores INDEX tensors, not
    embeddings — embedding happens in the net so update() can train the tables."""
    __slots__ = ("action", "logprob", "value", "dense_state", "state_ids",
                 "dense_opt", "opt_card_ids", "opt_attack_ids")

    def __init__(self, action, logprob, value, dense_state, state_ids,
                 dense_opt, opt_card_ids, opt_attack_ids):
        self.action = action                    # int, index into obs.select.option
        self.logprob = logprob                  # scalar tensor, log π(a|s) at act time (detached)
        self.value = value                      # scalar tensor, V(s) at act time (detached)
        self.dense_state = dense_state          # Tensor[STATE_DIM]        (detached)
        self.state_ids = state_ids              # LongTensor[N_STATE_SLOTS]
        self.dense_opt = dense_opt              # Tensor[n, OPTION_DIM]     (detached)
        self.opt_card_ids = opt_card_ids        # LongTensor[n, 2]
        self.opt_attack_ids = opt_attack_ids    # LongTensor[n, 1]


class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, option_dim=OPTION_DIM, hidden=HIDDEN,
                 d_card=D_CARD, d_attack=D_ATTACK):
        super().__init__()
        # Learned id-embeddings (padding_idx row = zero vector, zero gradient) for empty/pad slots.
        # LIMITATION — deck-specialization: an Embedding row gets gradient ONLY when its cardId/
        # attackId is actually looked up, so in v1's single-deck mirror only the ~11 distinct ids in
        # sample_deck (of ~1268) ever train; every other row stays at random init. The learned
        # card knowledge therefore MEMORIZES the training deck and does NOT transfer — an unseen
        # cardId at eval maps to an untrained (noise) row. Harmless in v1 (train == eval deck); it
        # bites only when generalizing to other decks/opponents. The typed features in _card_static/
        # _active_block are the deck-agnostic backbone that keeps the net functional on unseen cards;
        # the embeddings only add specialization on top. (Accepted tradeoff of the id-embedding fork.)
        self.card_emb = nn.Embedding(card_count + 1, d_card, padding_idx=CARD_PAD)
        self.attack_emb = nn.Embedding(attack_count + 1, d_attack, padding_idx=ATTACK_PAD)
        # COUPLING NOTE: card_emb is SHARED by both heads (value_head via _state_repr, policy_scorer
        # via _option_repr). The two MLPs are otherwise separate, but value-head gradient still flows
        # into the policy's card table — the value objective isn't perfectly quarantined from the
        # policy. Accepted for v1 (shared card knowledge is a feature); flagged so it's not a surprise.

        state_repr_dim = state_dim + N_STATE_SLOTS * d_card
        option_repr_dim = option_dim + N_OPT_CARD_SLOTS * d_card + N_OPT_ATTACK_SLOTS * d_attack

        # Value head: LINEAR output (no output squash). A final tanh was removed after the field
        # run measured it saturated under a ~90%-loss reward distribution: V pinned near −1,
        # holdout EV ≈ 0, and the tanh's (1−V²) gradient factor crushed exactly the rare-win
        # errors that carried the signal (docs/ppo/ppo_failure_investigation.md, verdict 1). The
        # terminal-only ±1 targets are bounded on their own, so an unbounded regressor is safe.
        self.value_head = nn.Sequential(
            nn.Linear(state_repr_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Policy scorer: scores ONE option from [state_repr ⊕ option_repr] → scalar logit.
        self.policy_scorer = nn.Sequential(
            nn.Linear(state_repr_dim + option_repr_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, *args, **kwargs):  # use act()/evaluate() directly, like the reference
        raise NotImplementedError("call act() / evaluate(), not forward()")

    def _state_repr(self, dense_state, state_ids):
        """dense_state[...,S], state_ids[...,N_STATE_SLOTS] -> [..., S + N_STATE_SLOTS*d_card]."""
        emb = self.card_emb(state_ids).flatten(start_dim=-2)
        return torch.cat([dense_state, emb], dim=-1)

    def _option_repr(self, dense_opt, opt_card_ids, opt_attack_ids):
        """dense_opt[...,O], ids -> [..., O + 2*d_card + 1*d_attack]."""
        ce = self.card_emb(opt_card_ids).flatten(start_dim=-2)
        ae = self.attack_emb(opt_attack_ids).flatten(start_dim=-2)
        return torch.cat([dense_opt, ce, ae], dim=-1)

    def _score(self, state_repr, option_repr):
        """[Srepr], [n, Orepr] → [n] option logits (batch=1, no padding)."""
        ctx = state_repr.unsqueeze(0).expand(option_repr.shape[0], -1)
        x = torch.cat([ctx, option_repr], dim=-1)
        return self.policy_scorer(x).squeeze(-1)

    def _value(self, state_repr):
        return self.value_head(state_repr).squeeze(-1)          # scalar or [B]

    def act(self, o) -> ActOutput:
        """Collection-time actor for ONE single-select node: featurize, embed, score, sample one
        option. Caller wraps in no_grad. All n options are valid here (no padding at batch=1)."""
        dev = self.card_emb.weight.device
        dense_state, state_ids = encode_state(o)
        dense_opt, opt_card_ids, opt_attack_ids = encode_options(o)
        dense_state, state_ids = dense_state.to(dev), state_ids.to(dev)
        dense_opt = dense_opt.to(dev)
        opt_card_ids, opt_attack_ids = opt_card_ids.to(dev), opt_attack_ids.to(dev)

        state_repr = self._state_repr(dense_state, state_ids)
        option_repr = self._option_repr(dense_opt, opt_card_ids, opt_attack_ids)
        logits = self._score(state_repr, option_repr)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return ActOutput(
            int(a.item()),
            dist.log_prob(a).detach(),
            self._value(state_repr).detach(),
            dense_state.detach(), state_ids,
            dense_opt.detach(), opt_card_ids, opt_attack_ids,
        )

    def act_index(self, o, greedy=False) -> int:
        """Eval-time action selection for ONE single-select node: return just the chosen option
        index (no buffer bookkeeping / logprob / value). Reuses the SAME featurize→score path as
        act(), so greedy=False is identical to act()'s sampling. greedy=True takes the argmax logit.
        Caller wraps in no_grad."""
        dev = self.card_emb.weight.device
        dense_state, state_ids = encode_state(o)
        dense_opt, opt_card_ids, opt_attack_ids = encode_options(o)
        state_repr = self._state_repr(dense_state.to(dev), state_ids.to(dev))
        option_repr = self._option_repr(dense_opt.to(dev), opt_card_ids.to(dev), opt_attack_ids.to(dev))
        logits = self._score(state_repr, option_repr)
        if greedy:
            return int(torch.argmax(logits).item())
        return int(Categorical(logits=logits).sample().item())

    def evaluate(self, dense_state_b, state_ids_b, dense_opt_b, opt_card_ids_b, opt_attack_ids_b,
                 mask, actions):
        """Update-time re-scoring of STORED actions under the CURRENT policy (K-epoch loop).
        Embeddings are applied here (from the stored indices) so their gradients flow.

          dense_state_b   : [B, STATE_DIM]
          state_ids_b     : [B, N_STATE_SLOTS]
          dense_opt_b     : [B, W, OPTION_DIM]   (zero-padded rows past each item's n)
          opt_card_ids_b  : [B, W, 2]            (sentinel-padded)
          opt_attack_ids_b: [B, W, 1]            (sentinel-padded)
          mask            : [B, W] bool          (True on real options)
          actions         : [B] long             (stored option index, always within the real region)
        Returns (logprobs[B], values[B], entropy[B])."""
        B, W, _ = dense_opt_b.shape
        state_repr = self._state_repr(dense_state_b, state_ids_b)          # [B, Srepr]
        option_repr = self._option_repr(dense_opt_b, opt_card_ids_b, opt_attack_ids_b)  # [B, W, Orepr]
        ctx = state_repr.unsqueeze(1).expand(-1, W, -1)                    # [B, W, Srepr]
        x = torch.cat([ctx, option_repr], dim=-1)
        logits = self.policy_scorer(x).squeeze(-1)                         # [B, W]
        # Finite large-negative (NOT -inf) so masked entropy stays NaN-safe (cf. NASHPG -1e9).
        logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), self._value(state_repr), dist.entropy()


# ============================================================================================
# ROLLOUT BUFFER — parallel lists like the reference, extended for the variable action space
# (per-step option feats/ids are ragged → padded at collate time) and the id indices.
# ============================================================================================
class RolloutBuffer:
    def __init__(self):
        self.dense_states = []    # list[Tensor[STATE_DIM]]
        self.state_ids = []       # list[LongTensor[N_STATE_SLOTS]]
        self.dense_opts = []      # list[Tensor[n_i, OPTION_DIM]]
        self.opt_card_ids = []    # list[LongTensor[n_i, 2]]
        self.opt_attack_ids = []  # list[LongTensor[n_i, 1]]
        self.actions = []         # list[int]
        self.logprobs = []        # list[float]  (old_logp, captured at act time)
        self.state_values = []    # list[float]  (V_old, captured at act time)
        self.rewards = []         # list[float]  (0 except a terminal G — written at game end)
        self.is_terminals = []    # list[bool]   (True only on a game's last learned decision)

    def add(self, dense_state, state_ids, dense_opt, opt_card_ids, opt_attack_ids, action, logp, value):
        self.dense_states.append(dense_state)
        self.state_ids.append(state_ids)
        self.dense_opts.append(dense_opt)
        self.opt_card_ids.append(opt_card_ids)
        self.opt_attack_ids.append(opt_attack_ids)
        self.actions.append(action)
        self.logprobs.append(logp)
        self.state_values.append(value)
        self.rewards.append(0.0)          # patched to G on the terminal step (collect_games)
        self.is_terminals.append(False)

    def __len__(self):
        return len(self.actions)

    def clear(self):
        for lst in (self.dense_states, self.state_ids, self.dense_opts, self.opt_card_ids,
                    self.opt_attack_ids, self.actions, self.logprobs, self.state_values,
                    self.rewards, self.is_terminals):
            del lst[:]


# ============================================================================================
# Within-game credit probe (work_log "2026-07-02 (later) — post-fix gate executed", Next item):
# does the critic discriminate STATES inside a game, or only (deck, opponent) identity?
# Module-level pure-Python helpers so probe_credit.py can reuse them on offline rollouts.
# ============================================================================================
def split_game_segments(state_values, is_terminals, rewards):
    """RolloutBuffer's flat per-decision lists → per-game segments [(v_seq, G), ...].

    A segment closes at each True terminal flag; its G is that step's patched terminal reward
    (collect_games writes rewards[-1] = G — the raw reward, so this holds at any γ). Defensive:
    a trailing unflagged remainder (never produced by collect_games; _selftest-style manual
    buffers could) is emitted as a final segment rather than dropped, with its own last reward
    as G (0.0 for an unfinished game)."""
    segments, seg = [], []
    for v, done, r in zip(state_values, is_terminals, rewards):
        seg.append(float(v))
        if done:
            segments.append((seg, float(r)))
            seg = []
    if seg:
        segments.append((seg, float(rewards[-1])))
    return segments


def signed_drift(v_seq, g):
    """One game's outcome-signed V drift: (mean V of 2nd half − mean V of 1st half) × G, balanced
    halves (odd length drops the middle decision); < 2 decisions → 0.0. Single definition shared
    by v_spread_stats and probe_credit.py (which needs per-game drifts for a standard error)."""
    t = len(v_seq)
    half = t // 2
    if half == 0:
        return 0.0
    return (sum(v_seq[t - half:]) / half - sum(v_seq[:half]) / half) * g


def v_spread_stats(segments):
    """Within-game credit probe over act-time V: does V vary inside games, and does it drift
    toward the eventual outcome? Tests the MATCHUP-ONLY-CRITIC hypothesis (work_log 2026-07-02
    CAVEAT: game-level R² of outcome on matchup identity ≥ every observed holdout EV, so
    ev_true_pre alone cannot rule out a critic that is a matchup lookup with game-constant V).

    Keys (all moments POPULATION/ddof=0 — deliberate: the ANOVA identity within+between = total
    is then exact and every n=1 case is 0.0 instead of NaN):
      v_within_std   mean over games of the per-game std of V (1-decision game → 0).
      v_between_std  std over games of the per-game mean V (single-game buffer → 0).
      v_within_frac  decision-weighted within-game share of V variance, SS_within/SS_total
                     (between share = 1 − within share). Guard: SS_total/N ≤ 1e-12 → 0.0
                     (no-contrast sentinel, mirrors the ev_true_pre var_g guard).
      v_drift_signed mean over games of (mean V of 2nd half − mean V of 1st half) × G, balanced
                     halves (odd length drops the middle decision); games with < 2 decisions and
                     draws (G=0) contribute exactly 0 (dilutes magnitude toward 0, never flips
                     sign; denominator = all recorded games, stable across updates).

    Interpretation: matchup-only critic signature = v_within_frac ≈ 0 AND v_drift_signed ≈ 0;
    a state-aware critic shows v_drift_signed > 0 (V drifts toward the realized outcome).
    v_drift_signed is the DISCRIMINATING readout — THEORY-REVIEW (verified algebra, flagged per
    convention): with game-constant realized targets (γ=1 ⇒ g_true ≡ G within a game), the
    residual decomposes as var(G − V) = between-game residual var + pooled within-game V var, so
    any within-game V spread mechanically LOWERS ev_true_pre — that metric structurally favors
    game-constant V and can never reward correct state-awareness. Corollary (THEORY-REVIEW,
    exact at γ=1): if V ≡ c within a game, GAE gives A_t = λ^(T−1−t)·(G − c) — same-sign,
    state-independent within the game, i.e. only λ-recency credit, no decision-quality credit."""
    if not segments:
        return {"v_within_std": 0.0, "v_between_std": 0.0,
                "v_within_frac": 0.0, "v_drift_signed": 0.0}
    means, within_stds, drifts, ss_within, n_total, v_sum = [], [], [], 0.0, 0, 0.0
    for v_seq, g in segments:
        t = len(v_seq)
        m = sum(v_seq) / t
        var = sum((x - m) ** 2 for x in v_seq) / t
        means.append(m)
        within_stds.append(var ** 0.5)
        ss_within += t * var
        n_total += t
        v_sum += sum(v_seq)
        drifts.append(signed_drift(v_seq, g))
    mu = v_sum / n_total
    ss_total = sum((x - mu) ** 2 for v_seq, _ in segments for x in v_seq)
    n_seg = len(segments)
    mean_of_means = sum(means) / n_seg
    return {
        "v_within_std": sum(within_stds) / n_seg,
        "v_between_std": (sum((m - mean_of_means) ** 2 for m in means) / n_seg) ** 0.5,
        # min(): SS_within <= SS_total holds exactly, but float summation order can push the
        # ratio to 1 + eps when SS_between ~ 0; consumers assume a share in [0, 1]
        "v_within_frac": min(1.0, ss_within / ss_total) if ss_total / n_total > 1e-12 else 0.0,
        "v_drift_signed": sum(drifts) / n_seg,
    }


# ============================================================================================
# PPO — mirrors the reference's PPO class (select_action / update / save / load).
# ============================================================================================
# Layer-1 telemetry contract (evaluation_harness_report.md §1a): the exact key set update()
# returns. Selftests assert against this tuple so the dict and its consumers can't drift apart.
# Aggregation semantics (since minibatching): all non-grad stats come from ONE full-batch
# no_grad pass under the POST-update θ (update() step 6); ratio_dev_e0 from a full-batch pass
# BEFORE any step; grad_norm_* are maxima over the epoch×minibatch training steps; ev_true_pre
# compares act-time (pre-update = holdout) values to TRUE outcomes — the honest critic read.
STATS_KEYS = (
    "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var",
    "ev_true_pre",
    "ratio_dev_e0", "ratio_max", "grad_norm_max", "grad_norm_policy_max", "grad_norm_value_max",
    "entropy_norm", "adv_mean_raw", "adv_std_raw",
    "return_mean", "return_std", "return_min", "return_max", "value_mean",
    # value-balance telemetry (--value-balance winloss; ≡ 1/1/0 under "none")
    "vw_win", "vw_loss", "vw_capped",
    # within-game credit probe (see v_spread_stats): matchup-only-critic diagnostic
    "v_within_std", "v_between_std", "v_within_frac", "v_drift_signed",
)


def _grad_norm(module):
    """Pre-clip L2 norm of the grads currently on `module`'s params (0.0 if backward hasn't run)."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm()) ** 2
    return total ** 0.5


class PPO:
    def __init__(self, state_dim, option_dim, lr, gamma, eps_clip, k_epochs,
                 vf_coef, ent_coef, device, gae_lambda=0.95, minibatches=4,
                 value_balance="none"):
        assert value_balance in ("none", "winloss"), value_balance
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.minibatches = minibatches
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.value_balance = value_balance
        self.policy = ActorCritic(state_dim, option_dim).to(device)
        # Single Adam/LR over ALL params (actor + critic + shared embeddings). The reference uses
        # separate lr_actor/lr_critic; folding them into one LR means the critic can't be tuned to a
        # different pace than the policy. Acceptable v1 simplification.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()   # rewards/is_terminals/state_values feed update()'s GAE

    def select_action(self, obs_dict):
        """Drive ONE decision. Returns (selection_list, recorded: bool). Single-select policy nodes
        are sampled on-policy and recorded; everything else is auto-resolved with NO record (so it's
        never trained — the v1 coverage limit)."""
        o = to_observation_class(obs_dict)
        sel = o.select
        if not is_policy_select(sel):
            return auto_resolve(sel), False
        with torch.no_grad():
            out = self.policy.act(o)
        self.buffer.add(out.dense_state, out.state_ids, out.dense_opt, out.opt_card_ids,
                        out.opt_attack_ids, out.action, float(out.logprob), float(out.value))
        return [out.action], True

    def _collate(self):
        """Buffer's ragged per-step option feats/ids → padded batch + validity mask.
        Returns (dense_state[B,S], state_ids[B,K], dense_opt[B,W,O], opt_card_ids[B,W,2],
                 opt_attack_ids[B,W,1], mask[B,W], actions[B], old_logp[B], old_v[B])."""
        dev = self.device
        dense_state = torch.stack(self.buffer.dense_states).to(dev)   # [B, STATE_DIM]
        state_ids = torch.stack(self.buffer.state_ids).to(dev)        # [B, N_STATE_SLOTS]
        n_list = [d.shape[0] for d in self.buffer.dense_opts]
        B, W = len(n_list), max(n_list)
        opt_b = dense_state.new_zeros(B, W, OPTION_DIM)
        # Pad id rows with the SENTINEL (not 0 — 0 is a real cardId); padding_idx zeroes them.
        cid_b = torch.full((B, W, N_OPT_CARD_SLOTS), CARD_PAD, dtype=torch.long, device=dev)
        aid_b = torch.full((B, W, N_OPT_ATTACK_SLOTS), ATTACK_PAD, dtype=torch.long, device=dev)
        mask = torch.zeros(B, W, dtype=torch.bool, device=dev)
        for i in range(B):
            n = n_list[i]
            opt_b[i, :n] = self.buffer.dense_opts[i].to(dev)
            cid_b[i, :n] = self.buffer.opt_card_ids[i].to(dev)
            aid_b[i, :n] = self.buffer.opt_attack_ids[i].to(dev)
            mask[i, :n] = True
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=dev)
        old_logp = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=dev)
        old_v = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=dev)
        return dense_state, state_ids, opt_b, cid_b, aid_b, mask, actions, old_logp, old_v

    def update(self):
        """One PPO update on the collected buffer: GAE(λ) advantages + TD(λ) value targets →
        clipped-surrogate policy loss + value MSE + entropy bonus, K epochs × M shuffled
        minibatches with per-minibatch advantage normalization. Clears the buffer and returns a
        diagnostics dict. Departures from PPO-PyTorch/PPO.py (which uses MC returns, full-batch
        epochs, whole-batch z-scored RETURNS) are deliberate: GAE + minibatching follow docs/ppo
        (ICLR "37 details" #5/#6/#7; "What Matters" §3.4/§3.5) and the NashPG reference config
        (γ=1.0, λ=0.95, 4 epochs × 4 minibatches — the same two-player zero-sum terminal-only
        reward shape); value targets stay raw / un-z-scored (see step 2) with a LINEAR value
        head (see ActorCritic.value_head). Value-loss clipping stays OMITTED on purpose — Engstrom et
        al. found no benefit and Andrychowicz et al. found it hurts.

        NO separate `policy_old` net (load-bearing): old_logp / V_old were captured at ACT time
        and stored in the buffer (select_action → act). `policy.evaluate` re-scores the STORED
        actions under the CURRENT policy, so the ratio π_θ/π_θ_old moves as θ updates over the K
        epochs. The reference instead freezes a `policy_old` snapshot to hold π_old fixed — but it
        only ever reads the buffered logprobs/values inside update(), never policy_old's params,
        so storing them at act time is functionally identical (and lighter). Recomputing old_logp
        here would force ratio≡1 every epoch and degenerate PPO.
        """
        if len(self.buffer) == 0:                       # driver always collects first; be safe anyway
            return {}

        # 1. GAE(λ) advantages + TD(λ) value targets (returns = gae + V), backward recursion,
        #    reset at each episode boundary. is_terminals marks a game's LAST recorded step and its
        #    reward already carries the full terminal G, so gae/next_v reset BEFORE that step's
        #    delta — nothing bootstraps past a game end and nothing leaks across games. Buffer
        #    adjacency == within-game adjacency of the learner's own policy-selects (only the PPO
        #    seat records; games are sequential), so V(s_{t+1}) is the learner's NEXT decision
        #    state — the opponent/auto-resolved steps in between are part of the environment
        #    transition (semi-MDP view). Single-step episodes degenerate to exactly the old MC
        #    target (gae = r − V, return = r), which the update-test relies on.
        advantages, returns = [], []
        gae = next_v = 0.0
        for r, done, v in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals),
                              reversed(self.buffer.state_values)):
            if done:
                gae = next_v = 0.0                      # terminal: no bootstrap past game end
            delta = r + self.gamma * next_v - v
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)                  # TD(λ) value target (the λ-return)
            next_v = v
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # 1b. TRUE Monte-Carlo return per step (exact — rewards are terminal-only), for the
        #    ev_true_pre holdout gate below: the act-time values in the buffer were predicted
        #    BEFORE this update trained on these games, so measuring them against the true
        #    outcomes is out-of-sample by construction. (explained_var is NOT: it is computed
        #    post-fit against targets containing the critic's own bootstrapped values — it read
        #    ~0.4 on the run whose holdout EV was ≈ 0, docs/ppo/ppo_failure_investigation.md.)
        g_true = []
        ret = 0.0
        for r, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if done:
                ret = 0.0
            ret = r + self.gamma * ret
            g_true.insert(0, ret)
        g_true = torch.tensor(g_true, dtype=torch.float32, device=self.device)
        # 1c. Win/loss-balanced per-sample VALUE-loss weights (--value-balance winloss; Stage
        #    B(a) of the critic-detrending plan). Measured motive: in ~90%-loss batches the MSE
        #    gradient is dominated by loss states and the fitted V hedges on time-proxy features
        #    (CLOCK attribution drift +0.130 vs lucario, work_log 2026-07-03 'later 2').
        #    Balancing win/loss sample mass makes V a BALANCED-PRIOR conditional estimator, NOT
        #    E[G|s] under the collection distribution [THEORY-REVIEW: baseline biases upward in
        #    loss-heavy cells; ev_true_pre is still measured against unweighted outcomes and may
        #    dip while win/loss discrimination improves — gate on the probe drift, not on the
        #    historical +0.16]. Class = sign(g_true) (terminal-only reward ⇒ sign(g_true) =
        #    sign(G); draws stay at 1). Cap before renorm: a 1-win batch would otherwise put
        #    ~B/2 weight on one sample. Renorm to mean 1 keeps the value-loss scale (vf_coef,
        #    grad telemetry) comparable. No wins or no losses in the batch → uniform (the
        #    starved-batch guard). Policy loss and the RAW targets are untouched.
        w_value = torch.ones_like(g_true)
        vw_win = vw_loss = 1.0
        vw_capped = 0.0
        if self.value_balance == "winloss":
            n_w = int((g_true > 0).sum())
            n_l = int((g_true < 0).sum())
            if n_w > 0 and n_l > 0:
                half = (n_w + n_l) / 2.0
                raw_w = torch.ones_like(g_true)
                raw_w[g_true > 0] = half / n_w
                raw_w[g_true < 0] = half / n_l
                vw_capped = float((raw_w > 5.0).float().mean())
                w_value = raw_w.clamp(max=5.0)
                w_value = w_value * (w_value.numel() / w_value.sum())
                vw_win = float(w_value[g_true > 0][0])
                vw_loss = float(w_value[g_true < 0][0])
        # 2. Keep RAW TD(λ) targets — deliberately NOT z-scored (the PPO-PyTorch reference
        #    z-scores its returns): with terminal-only ±1 reward the targets are already bounded
        #    and well-scaled, and z-scoring ERASES the win/loss signal in a skewed batch (a
        #    mostly-loss batch would recenter losses to ~0). Raw targets keep the outcome signal;
        #    V learns E[return|s]. (The old companion reason — keeping targets feasible for a
        #    tanh-bounded head — retired with the tanh output, which saturated; see
        #    ActorCritic.value_head.)

        # 3. Collate the ragged buffer once; full-batch PRE-normalization advantage stats (the
        #    training loop normalizes per-minibatch below). Bug signatures: adv_mean_raw
        #    persistently far from 0 = biased V; adv_std_raw ≈ return_std = V contributes nothing
        #    (the broken-critic symptom, seen from the advantage side).
        (dense_state, state_ids, opt_b, cid_b, aid_b,
         mask, actions, old_logp, old_v) = self._collate()
        adv_mean_raw = float(advantages.mean())
        adv_std_raw = float(advantages.std())
        # ev_true_pre: HOLDOUT explained variance of the act-time V vs the true outcome (step 1b)
        # — the standing critic gate (healthy ≫ 0). 0.0 = no-contrast batch (single outcome,
        # var ≈ 0, uninformative; the guard also absorbs the B==1 NaN of unbiased var()).
        var_g = float(g_true.var())
        ev_true_pre = (1.0 - float((g_true - old_v).var()) / var_g) if var_g > 1e-12 else 0.0
        # Within-game credit probe on the same act-time (holdout) values — computed here, before
        # buffer.clear(); the buffer's Python floats avoid a device round-trip. Definitions +
        # interpretation (matchup-only-critic signature): see v_spread_stats.
        vspread = v_spread_stats(split_game_segments(
            self.buffer.state_values, self.buffer.is_terminals, self.buffer.rewards))

        # 4. Epoch-0 full-batch ratio must be ≡1 BEFORE any optimizer step: old_logp was captured
        #    under this exact θ, so any deviation beyond fp noise means evaluate() failed to
        #    reconstruct act()'s distribution (collate/mask/featurization drift) — the ICLR
        #    "check ratio=1" rule. A dedicated no_grad pass now that training steps run per-
        #    minibatch (the first trained minibatch would only check a shuffled subset).
        with torch.no_grad():
            logp0 = self.policy.evaluate(dense_state, state_ids, opt_b, cid_b, aid_b,
                                         mask, actions)[0]
            ratio_dev_e0 = float((torch.exp(logp0 - old_logp) - 1.0).abs().max())

        # 5. K epochs × M shuffled minibatches of clipped-surrogate ascent over the same rollout
        #    (ICLR "37 details" #6; NashPG uses 4×4). Advantages are normalized PER-MINIBATCH
        #    (#7, matching baselines/NashPG) — scaling post-baseline advantages fixes the policy-
        #    gradient step size WITHOUT destroying the win/loss signal the baseline absorbs. The
        #    B//2 floor is load-bearing: std() of a single element is NaN (and torch.chunk can
        #    still emit a size-1 tail chunk, hence the numel guard too).
        B = len(self.buffer)
        n_mb = max(1, min(self.minibatches, B // 2))
        grad_norm_max = grad_norm_policy_max = grad_norm_value_max = 0.0
        for _ in range(self.k_epochs):
            for idx in torch.randperm(B, device=self.device).chunk(n_mb):
                mb_adv = advantages[idx]
                if mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                logp, V, entropy = self.policy.evaluate(dense_state[idx], state_ids[idx],
                                                        opt_b[idx], cid_b[idx], aid_b[idx],
                                                        mask[idx], actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # plain (optionally win/loss-balanced) MSE; no value-clipping (hurts).
                # w_value ≡ 1 under value_balance="none" — identical to the unweighted loss.
                value_loss = (w_value[idx] * (V - returns[idx]) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # Per-head PRE-clip grad norms, max over all epoch×minibatch steps (spikes are the
                # signal): the global norm can mask a value-head-dominated gradient — the value/
                # policy loss-scale-imbalance signature. Shared embeddings: total norm only.
                grad_norm_policy_max = max(grad_norm_policy_max,
                                           _grad_norm(self.policy.policy_scorer))
                grad_norm_value_max = max(grad_norm_value_max, _grad_norm(self.policy.value_head))
                # THEORY-REVIEW FLAG: the named reference (PPO-PyTorch/PPO.py) uses NO gradient
                # clip; we keep this one-line global-norm clip (ICLR "37 details" #11, norm 0.5)
                # as a standard stabilizer. Minor, deliberate divergence — see the plan/notes.
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                grad_norm_max = max(grad_norm_max, float(grad_norm))
                self.optimizer.step()

        # 6. Diagnostics: ONE full-batch no_grad pass under the POST-update θ (a per-minibatch
        #    read would describe a random 1/M of the data; the pre-minibatching code read the
        #    last epoch instead — comparable, slightly earlier θ). The diagnostic policy_loss/
        #    loss use FULL-batch-normalized advantages: training stepped on per-minibatch norms,
        #    so these are consistent update-over-update but NOT the exact quantities stepped on.
        with torch.no_grad():
            logp, V, entropy = self.policy.evaluate(dense_state, state_ids, opt_b, cid_b, aid_b,
                                                    mask, actions)
            ratio = torch.exp(logp - old_logp)
            adv_n = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            surr1 = ratio * adv_n
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_n
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (w_value * (V - returns) ** 2).mean()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
            # k3 KL estimator (Schulman): mean(ratio − 1 − log ratio) ≥ 0, so this is a
            # NaN-free, always-nonneg sanity signal.
            approx_kl = (ratio - 1.0 - (logp - old_logp)).mean()
            clip_frac = ((ratio - 1.0).abs() > self.eps_clip).float().mean()
            explained_var = 1.0 - (returns - V).var() / (returns.var() + 1e-8)
            stats = {
                "loss": float(loss),
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "entropy": float(entropy.mean()),
                "approx_kl": float(approx_kl),
                "clip_frac": float(clip_frac),
                "explained_var": float(explained_var),
                # THE critic health read (see step 1b/3): holdout EV of act-time V vs true
                # outcome. explained_var above stays for continuity but is in-sample/post-fit —
                # trust ev_true_pre when they disagree.
                "ev_true_pre": ev_true_pre,
                # Layer-1 additions (see STATS_KEYS / evaluation_harness_report.md §1a).
                # ratio_max: tail importance-weight violations hide behind a healthy mean
                # KL ("Truly PPO"). entropy_norm ∈ (0,1]: entropy / ln(n_options) — raw
                # entropy is uninterpretable under variable action counts; →0 collapse,
                # →1 flatten (is_policy_select guarantees n ≥ 2, so ln(n) > 0).
                "ratio_dev_e0": ratio_dev_e0,
                "ratio_max": float(ratio.max()),
                "grad_norm_max": grad_norm_max,
                "grad_norm_policy_max": grad_norm_policy_max,
                "grad_norm_value_max": grad_norm_value_max,
                "entropy_norm": float((entropy / mask.sum(dim=1).float().log()).mean()),
                "adv_mean_raw": adv_mean_raw,
                "adv_std_raw": adv_std_raw,
                # returns are TD(λ) targets. With the unbounded linear value head, V (and so a
                # λ-return) can drift somewhat outside [-1,1]; LARGE excursions still mean the
                # per-episode reset in the GAE recursion broke (bootstrap/reward leaking across
                # episode boundaries). value_mean: act-time V, catches sign bias / pinning
                # alongside ev_true_pre.
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "return_min": float(returns.min()),
                "return_max": float(returns.max()),
                "value_mean": float(old_v.mean()),
                # value-balance telemetry (step 1c): post-renorm class weights + fraction of
                # samples whose raw weight hit the 5.0 cap. All ≡ 1/1/0 under "none" and on
                # degenerate (single-outcome) batches — the uniform fallback signature.
                "vw_win": vw_win,
                "vw_loss": vw_loss,
                "vw_capped": vw_capped,
                # within-game credit probe (act-time V, step 3) — work_log 2026-07-02 (later):
                # the matchup-only-critic suspect behind the reward-rich plateau.
                **vspread,
            }

        # 5. Clear the now-off-policy buffer.
        self.buffer.clear()
        # 6. Diagnostics for the training-loop log line (train_ppo.train uses stats['loss']).
        return stats

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================================================
# SELF-TEST (roadmap step 1 verification): drive real random self-play battles and assert the
# featurization shapes / invariants, a net forward pass, and that the embeddings receive gradient.
#   TORCHDYNAMO_DISABLE=1 python ppo_model.py [--games N]
# ============================================================================================
def _selftest(games=2, seed=0):
    import random
    from cg.game import battle_start, battle_select, battle_finish
    from tcg_model import sample_deck

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    ppo = PPO(STATE_DIM, OPTION_DIM, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=1,
              vf_coef=0.5, ent_coef=0.01, device=device)
    deck = sample_deck
    checked = 0

    # feature_groups partitions the dense dims exactly (no gap, no overlap, no out-of-range)
    fg = feature_groups()
    all_idx = sorted(i for idxs in fg.values() for i in idxs)
    assert all_idx == list(range(STATE_DIM)), "feature_groups does not partition range(STATE_DIM)"
    # prize_potential reads the two /6-normalized prize counts: constructed state, own 2 prizes
    # remaining vs opp 5 -> Phi = (5 - 2) / 6
    ds = torch.zeros(STATE_DIM)
    ds[_GLOBAL_DIM] = 2 / 6.0
    ds[_GLOBAL_DIM + _SIDE_DIM] = 5 / 6.0
    assert abs(prize_potential(ds) - 3 / 6.0) < 1e-6, "prize_potential index/sign broken"  # float32 eps
    ds[_SELTYPE_OFF + int(SelectType.MAIN)] = 1.0
    assert prize_potential_valid(ds), "MAIN select with dealt prizes must be valid"
    ds[_SELTYPE_OFF + int(SelectType.MAIN)] = 0.0
    ds[_SELTYPE_OFF + int(SelectType.CARD)] = 1.0
    assert not prize_potential_valid(ds), "CARD select must be invalid (prize-pick artifact)"
    ds[_SELTYPE_OFF + int(SelectType.CARD)] = 0.0
    ds[_SELTYPE_OFF + int(SelectType.MAIN)] = 1.0
    ds[_GLOBAL_DIM] = ds[_GLOBAL_DIM + _SIDE_DIM] = 0.0
    assert not prize_potential_valid(ds), "pre-deal 0/0 must be invalid"
    # deck_potential/board_potential: index cross-check vs feature_groups + sign (higher = opp
    # closer to deck-out / bench-out). own deck 40/60 vs opp 20/60 -> +20/60; own 4 in play vs
    # opp 1 -> +3/6.
    assert [_GLOBAL_DIM + 1] == fg["own_deck"] and [_GLOBAL_DIM + _SIDE_DIM + 1] == fg["opp_deck"], \
        "deck_potential slots drifted from feature_groups"
    assert _GLOBAL_DIM + 5 in fg["own_board_counts"] and _GLOBAL_DIM + _SIDE_DIM + 5 in fg["opp_board_counts"], \
        "board_potential slots drifted from feature_groups"
    ds2 = torch.zeros(STATE_DIM)
    ds2[_GLOBAL_DIM + 1], ds2[_GLOBAL_DIM + _SIDE_DIM + 1] = 40 / 60.0, 20 / 60.0
    assert abs(deck_potential(ds2) - 20 / 60.0) < 1e-6, "deck_potential index/sign broken"
    ds2[_GLOBAL_DIM + 5], ds2[_GLOBAL_DIM + _SIDE_DIM + 5] = 4 / 6.0, 1 / 6.0
    assert abs(board_potential(ds2) - 3 / 6.0) < 1e-6, "board_potential index/sign broken"

    for _ in range(games):
        obs, start = battle_start(deck, deck)
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
                assert int(state_ids.max()) <= CARD_PAD and int(cids.max()) <= CARD_PAD
                assert int(aids.max()) <= ATTACK_PAD
                # exactly one OptionType one-hot bit per row (pad-distinguishability contract)
                assert torch.all(dense_o[:, :N_OPTION_TYPES].sum(dim=1) == 1.0), "type one-hot broken"
                checked += 1
            action, _ = ppo.select_action(obs)   # exercises policy.act() (net forward) for both seats
            obs = battle_select(action)
        battle_finish()

    # Gradient sanity: one evaluate()+backward must reach the card embedding table.
    assert len(ppo.buffer) > 0, "no policy nodes recorded"
    batch = ppo._collate()
    logp, V, ent = ppo.policy.evaluate(*batch[:6], batch[6])
    # act↔evaluate invariant (the no-policy_old ratio trick depends on it: ratio≡1 on epoch 0). The
    # logp captured at act() time (batch=1, unmasked, n options — stored in buffer.logprobs == batch[7])
    # must equal evaluate()'s batched + zero-padded + (-1e9)-masked re-score of the SAME stored action.
    # No optimizer.step ran during collection, so the net is unchanged and these must match to ~1e-6.
    assert torch.allclose(logp.detach(), batch[7], atol=1e-5), \
        "act()-time logp disagrees with evaluate()'s batched/masked re-score (ratio!=1 on epoch 0)"
    (logp.sum() + V.sum() + ent.sum()).backward()
    assert ppo.policy.card_emb.weight.grad is not None, "embeddings received no gradient"

    # Real-buffer update() smoke: exercise the full PPO update end-to-end on GENUINE cg
    # observations (real featurization, ragged options → the _collate pad/mask path, real
    # embeddings). The buffer already carries act-time old_logp/V (so the π_old invariant holds
    # and ratio≈1 falls out); one tail-terminal makes the whole buffer one pseudo-episode with a
    # non-degenerate GAE curve. Placed AFTER the grad check because update() clears the buffer.
    n_before = len(ppo.buffer)
    ppo.buffer.is_terminals[-1] = True
    ppo.buffer.rewards[-1] = 1.0
    stats = ppo.update()
    # Telemetry contract: the full Layer-1 key set, all finite, and the epoch-0 ratio invariant
    # holding THROUGH update() (the allclose check above runs pre-update; this one exercises the
    # real collate→evaluate path inside the K-epoch loop on genuine cg observations).
    assert set(stats) == set(STATS_KEYS), f"stats keys drifted from STATS_KEYS: {set(stats) ^ set(STATS_KEYS)}"
    assert all(math.isfinite(v) for v in stats.values()), f"non-finite stat: {stats}"
    assert stats["ratio_dev_e0"] < 1e-4, \
        f"epoch-0 ratio != 1 through update() (recompute drift): {stats['ratio_dev_e0']:.2e}"
    # V-spread probe invariants: the manual tail-terminal makes the whole buffer ONE segment →
    # between-game std is exactly 0 and the within share is a fraction by construction.
    assert stats["v_between_std"] == 0.0, \
        f"single-segment buffer must give zero between-game std: {stats['v_between_std']}"
    assert 0.0 <= stats["v_within_frac"] <= 1.0, f"v_within_frac not a share: {stats['v_within_frac']}"

    print(f"[selftest] OK  STATE_DIM={STATE_DIM} OPTION_DIM={OPTION_DIM} "
          f"N_STATE_SLOTS={N_STATE_SLOTS}  policy_nodes_checked={checked} buffer={n_before} "
          f"update_loss={stats['loss']:.4f}")


# ============================================================================================
# UPDATE LEARNING TEST (roadmap step 3 verification): a deterministic, battle-free unit test that
# runs TWO subsequent PPO updates on a tiny hand-built buffer with small hardcoded weights and
# asserts the network actually learns — the +advantage action's probability rises, the
# −advantage action's falls, and the value estimates move toward their targets.
#   TORCHDYNAMO_DISABLE=1 python ppo_model.py --update-test
# ============================================================================================
def _make_transitions():
    """Two single-step episodes: a GOOD state (took option 0, reward +1) and a DISTINCT bad state
    (took option 1, reward −1), each with two DISTINGUISHABLE options. Deterministic tensors of the
    exact shapes RolloutBuffer.add expects. Design rationale (all load-bearing):
      * single-step-per-episode ⇒ the GAE target == reward exactly (γ/λ-independent: the
        terminal reset zeroes the bootstrap, so gae = r − V and return = r) and exercises the
        per-episode reset on every step;
      * ±1 rewards give the advantage VARIANCE normalization needs (equal returns ⇒ ~0 advantages);
      * distinct states keep the value targets learnable (V(S_g)→+, V(S_b)→−);
      * the two option rows must differ (distinct dense bit + distinct card id) or identical logits
        would pin the softmax (the encode_options CAVEAT);
      * the +advantage action (opt0, dense bit 0) and the −advantage action (opt1, dense bit 1)
        are DIFFERENT option features on purpose: the per-option scorer is SHARED across options,
        so distinct features let the two signals REINFORCE (up-weight bit-0, down-weight bit-1)
        instead of fighting over one index — otherwise "generally down-weight that option" would
        satisfy the −advantage sample while suppressing the +advantage one."""
    def _one(fill, cid0, action, reward):
        dense_state = torch.full((STATE_DIM,), fill, dtype=torch.float32)
        state_ids = torch.full((N_STATE_SLOTS,), cid0, dtype=torch.long)
        dense_opt = torch.zeros(2, OPTION_DIM, dtype=torch.float32)
        dense_opt[0, 0] = 1.0                       # option 0: OptionType one-hot bit 0
        dense_opt[1, 1] = 1.0                       # option 1: OptionType one-hot bit 1 (distinct row)
        opt_card_ids = torch.tensor([[cid0, CARD_PAD], [cid0 + 1, CARD_PAD]], dtype=torch.long)
        opt_attack_ids = torch.full((2, N_OPT_ATTACK_SLOTS), ATTACK_PAD, dtype=torch.long)
        return dict(dense_state=dense_state, state_ids=state_ids, dense_opt=dense_opt,
                    opt_card_ids=opt_card_ids, opt_attack_ids=opt_attack_ids,
                    action=action, reward=reward)
    return [_one(0.1, 0, action=0, reward=1.0), _one(-0.1, 10, action=1, reward=-1.0)]


def _populate(buffer, transitions):
    """Add the hand-built transitions, then patch the terminal reward/flag by index — exactly how
    train_ppo.collect_games writes the terminal signal. logp/value are placeholders here; the test
    overwrites them from the current net before each update (the π_old invariant)."""
    for t in transitions:
        buffer.add(t["dense_state"], t["state_ids"], t["dense_opt"], t["opt_card_ids"],
                   t["opt_attack_ids"], t["action"], 0.0, 0.0)
    for i, t in enumerate(transitions):
        buffer.rewards[i] = t["reward"]
        buffer.is_terminals[i] = True


def _set_old_from_current(ppo):
    """Store π_old for every buffer entry = the CURRENT net's logp/V of the stored action (one
    no_grad evaluate over the collated buffer) — exactly what act() captures at collection time.
    This makes epoch-0 ratio == 1; without it the ratio uses a wrong baseline (the whole reason
    there is no separate policy_old net)."""
    with torch.no_grad():
        logp, V, _ = ppo.policy.evaluate(*ppo._collate()[:7])
    for i in range(len(ppo.buffer)):
        ppo.buffer.logprobs[i] = float(logp[i])
        ppo.buffer.state_values[i] = float(V[i])


def _measure(ppo, probe):
    """(good_logp, bad_logp, V(S_g), V(S_b)) under the current net. `probe` is a persistent
    _collate()[:7] snapshot (feats/ids/actions only), so it is unaffected by buffer clears/rewrites."""
    with torch.no_grad():
        logp, V, _ = ppo.policy.evaluate(*probe)
    return float(logp[0]), float(logp[1]), float(V[0]), float(V[1])


def _update_learns_test(seed=0):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    # lr=1e-3 (was 1e-2 under the old tanh-squashed value head): with the LINEAR head, 1e-2 on
    # this tiny 2-sample buffer overshoots V past the ±1 targets within one update (the shared
    # card embeddings, 12 identical id slots per state here, dominate state_repr), which flips
    # the advantage sign on the next update and yanks the policy back — a synthetic-regime
    # step-size artifact, not a learning failure (real batches are ~500 fresh samples).
    ppo = PPO(STATE_DIM, OPTION_DIM, lr=1e-3, gamma=0.99, eps_clip=0.2, k_epochs=4,
              vf_coef=0.5, ent_coef=0.01, device=device)
    # Small, deterministic, VARIED weight fill → logits start ≈0 (near-uniform policy) with plenty
    # of head-room, and reproducible across torch versions (nn.init defaults are not). A constant
    # fill would freeze the softmax at uniform (unlearnable); sin(idx) keeps per-weight variation.
    with torch.no_grad():
        for p in ppo.policy.parameters():
            flat = p.view(-1)
            flat.copy_(0.01 * torch.sin(torch.arange(flat.numel(), dtype=torch.float32)))
        # Re-zero the embedding padding rows the blanket fill overwrote (padding_idx must stay 0).
        ppo.policy.card_emb.weight[CARD_PAD].zero_()
        ppo.policy.attack_emb.weight[ATTACK_PAD].zero_()

    transitions = _make_transitions()
    _populate(ppo.buffer, transitions)
    probe = ppo._collate()[:7]                      # persistent; survives buffer.clear()
    g0, b0, vg0, vb0 = _measure(ppo, probe)

    # ---- UPDATE 1 ----
    _set_old_from_current(ppo)
    with torch.no_grad():                           # epoch-0 ratio == 1 (the no-policy_old invariant)
        logp_now = ppo.policy.evaluate(*probe)[0]
    assert torch.allclose(logp_now, torch.tensor(ppo.buffer.logprobs), atol=1e-6), \
        "epoch-0 ratio != 1: stored old_logp disagrees with the current net"
    stats1 = ppo.update()                           # clears the buffer
    g1, b1, vg1, vb1 = _measure(ppo, probe)

    # ---- UPDATE 2 ---- (re-populate the SAME transitions; re-derive π_old vs the UPDATED net)
    _populate(ppo.buffer, transitions)
    _set_old_from_current(ppo)
    stats2 = ppo.update()
    g2, b2, vg2, vb2 = _measure(ppo, probe)

    # Learning-curve table (P = exp(logp)); minimal analog of PPO-PyTorch's reward plots.
    def _p(lp):
        return math.exp(lp)
    print("[update-test] step   P(good)  P(bad)   V(S_g)   V(S_b)   loss")
    print(f"              init    {_p(g0):.3f}    {_p(b0):.3f}    {vg0:+.3f}   {vb0:+.3f}   -")
    print(f"              upd1    {_p(g1):.3f}    {_p(b1):.3f}    {vg1:+.3f}   {vb1:+.3f}   {stats1['loss']:.3f}")
    print(f"              upd2    {_p(g2):.3f}    {_p(b2):.3f}    {vg2:+.3f}   {vb2:+.3f}   {stats2['loss']:.3f}")

    # Assertions — direction only, with a small tie-guard (magnitude thresholds would be brittle).
    eps = 1e-4
    assert math.isfinite(stats1["loss"]) and math.isfinite(stats2["loss"]), "non-finite loss"
    assert stats1["approx_kl"] >= -1e-5 and stats2["approx_kl"] >= -1e-5, "approx_kl unexpectedly negative"
    # Layer-1 telemetry: full key set, finite; both episodes are single-step ±1 terminals, so the
    # GAE targets must be exactly ±1 (any excursion outside [-1,1] = broken episode-boundary reset).
    assert set(stats1) == set(STATS_KEYS), f"stats keys drifted: {set(stats1) ^ set(STATS_KEYS)}"
    assert all(math.isfinite(v) for v in stats1.values()), f"non-finite stat: {stats1}"
    assert -1.0 - 1e-6 <= stats1["return_min"] and stats1["return_max"] <= 1.0 + 1e-6, \
        f"returns escaped [-1,1]: min={stats1['return_min']} max={stats1['return_max']}"
    # V-spread probe invariants: two single-decision episodes → no within-game variation
    # anywhere; all three within-game stats must be EXACTLY 0 (ddof=0 conventions, no NaN).
    assert stats1["v_within_std"] == 0.0 and stats1["v_within_frac"] == 0.0 \
        and stats1["v_drift_signed"] == 0.0, \
        f"single-decision episodes must zero the within-game stats: {stats1}"
    assert g1 > g0 + eps and g2 > g1 + eps, f"good-action prob did not rise: {g0:.4f} {g1:.4f} {g2:.4f}"
    assert b1 < b0 - eps and b2 < b1 - eps, f"bad-action prob did not fall: {b0:.4f} {b1:.4f} {b2:.4f}"
    assert vg1 > vg0 and vg2 > vg1, f"V(S_g) did not rise toward +: {vg0:.4f} {vg1:.4f} {vg2:.4f}"
    assert vb1 < vb0 and vb2 < vb1, f"V(S_b) did not fall toward −: {vb0:.4f} {vb1:.4f} {vb2:.4f}"
    assert stats2["value_loss"] < stats1["value_loss"], "value loss did not decrease across updates"
    # vw_* telemetry contract under the default: all three are the uniform sentinel.
    assert stats1["vw_win"] == 1.0 and stats1["vw_loss"] == 1.0 and stats1["vw_capped"] == 0.0, \
        f"value_balance='none' must report uniform vw_*: {stats1}"

    # ---- UPDATE 3: --value-balance winloss on an UNBALANCED batch (1 win + 3 losses) ----
    # half=(1+3)/2=2 → raw weights win 2.0 / loss 2/3; no cap; Σw = n_w·half/n_w + n_l·half/n_l
    # = n_w+n_l = numel, so the mean-1 renorm is exactly a no-op (the no-cap/no-draw invariant).
    # Direction asserts still hold: balancing reweights the VALUE loss only.
    ppo.value_balance = "winloss"
    _populate(ppo.buffer, transitions + [transitions[1], transitions[1]])
    _set_old_from_current(ppo)
    stats3 = ppo.update()
    g3, b3, vg3, vb3 = _measure(ppo, probe)
    assert set(stats3) == set(STATS_KEYS), f"stats keys drifted: {set(stats3) ^ set(STATS_KEYS)}"
    assert all(math.isfinite(v) for v in stats3.values()), f"non-finite stat: {stats3}"
    assert abs(stats3["vw_win"] - 2.0) < 1e-6 and abs(stats3["vw_loss"] - 2.0 / 3.0) < 1e-6 \
        and stats3["vw_capped"] == 0.0, \
        f"winloss weights wrong on 1w/3l: {stats3['vw_win']} {stats3['vw_loss']} {stats3['vw_capped']}"
    assert g3 > g2 and b3 < b2, f"policy direction broke under value balancing: {g3:.4f} {b3:.4f}"
    assert vg3 > vg2 and vb3 < vb2, f"V direction broke under value balancing: {vg3:.4f} {vb3:.4f}"
    print(f"[update-test] winloss pass: vw_win={stats3['vw_win']:.3f} "
          f"vw_loss={stats3['vw_loss']:.3f} vw_capped={stats3['vw_capped']:.3f}")
    print("[update-test] OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ppo_model featurization + PPO update self-tests")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--update-test", action="store_true",
                    help="run the synthetic two-step PPO update learning test instead of the "
                         "battle-driven featurization self-test")
    args = ap.parse_args()
    if args.update_test:
        _update_learns_test(seed=args.seed)
    else:
        _selftest(games=args.games, seed=args.seed)
