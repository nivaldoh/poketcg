"""Reusable PPO implementation layer for the Pokémon TCG `cg` agent.

This is the *model/algorithm* half of the from-scratch PPO trainer — extracted from
`train_ppo.py` so the driver loop stays lean (mirrors how `tcg_model.py` holds the reusable
model layer for the other trainers). The training driver (battle collection, opponents, the
loop, argparse) lives in `train_ppo.py`, which imports `PPO`, `STATE_DIM`, `OPTION_DIM` from
here.

Design:
  * FRESH network from scratch — we do NOT reuse MyModel or its Transformer/sparse featurizers.
  * MONTE-CARLO discounted returns (advantage = return − V), exactly like PPO-PyTorch/PPO.py.
    GAE can slot in later behind the same `update()` interface.

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

# Import order is load-bearing: `eval_mymodel` imports `tcg_model` at its own top, which runs
# the `CG_LIB_PATH` sys.path insert (tcg_model.py, the `_CG_LIB_PATH` block) that makes the
# vendored `cg` package importable. Keep eval_mymodel above the `cg.api` / `tcg_model` imports.
from eval_mymodel import is_policy_select, auto_resolve                 # noqa: E402
# Reuse the card DB + option-target resolver from the shared model layer (already loaded above).
# `card_table[cardId] -> CardData` (static stats); `get_card(obs, area, index, playerIndex)`
# resolves an option's referenced card exactly as tcg_model.get_decoder_input does.
from tcg_model import card_table, card_count, attack_count, get_card    # noqa: E402
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

        # Value head: V(s) ∈ [-1, 1] via a final tanh — well-matched to the RAW MC returns
        # (γ^k·(±1) ∈ [-1,1]) that PPO.update targets. LOAD-BEARING coupling: this works ONLY because
        # update() keeps returns UN-normalized (see update() step 2). If you re-introduce return
        # z-scoring, targets exceed ±1 and this tanh saturates — drop the final tanh in that case.
        self.value_head = nn.Sequential(
            nn.Linear(state_repr_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Tanh(),
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
# PPO — mirrors the reference's PPO class (select_action / update / save / load).
# ============================================================================================
# Layer-1 telemetry contract (evaluation_harness_report.md §1a): the exact key set update()
# returns. Selftests assert against this tuple so the dict and its consumers can't drift apart.
STATS_KEYS = (
    "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var",
    "ratio_dev_e0", "ratio_max", "grad_norm_max", "grad_norm_policy_max", "grad_norm_value_max",
    "entropy_norm", "adv_mean_raw", "adv_std_raw",
    "return_mean", "return_std", "return_min", "return_max", "value_mean",
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
                 vf_coef, ent_coef, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.policy = ActorCritic(state_dim, option_dim).to(device)
        # Single Adam/LR over ALL params (actor + critic + shared embeddings). The reference uses
        # separate lr_actor/lr_critic; folding them into one LR means the critic can't be tuned to a
        # different pace than the policy. Acceptable v1 simplification.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()   # GAE could later slot in behind this + update()

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
        """One PPO update on the collected buffer: Monte-Carlo returns → clipped-surrogate policy
        loss + value MSE + entropy bonus, K epochs of full-batch gradient ascent. Clears the
        buffer and returns a diagnostics dict. Mirrors PPO-PyTorch/PPO.py (MC returns, clipped
        surrogate, advantage = returns − V_old) EXCEPT we normalize the ADVANTAGES, not the raw
        RETURNS the reference z-scores (deliberate, tied to our tanh value head — see step 2); GAE /
        minibatching / value-clipping are future drop-ins behind this same interface.

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

        # 1. Monte-Carlo discounted returns, backward, reset at each episode boundary. With our
        #    terminal-only ±1 reward, gamma<1 just discounts the outcome toward the later decisions.
        returns = []
        R = 0.0
        for r, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if done:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # 2. Keep RAW MC returns as the value target — deliberately NOT z-scored (the PPO-PyTorch
        #    reference z-scores returns, but it pairs that with an UNBOUNDED critic). Two reasons this
        #    matters HERE (independently reviewed — see the plan / docs/ppo_iclr_blog_post.md #7):
        #      (a) our value head ends in tanh (V ∈ [-1,1]); raw returns are γ^k·(±1) ∈ [-1,1] so they
        #          are tanh-FEASIBLE, whereas z-scored returns routinely exceed ±1 → tanh saturates
        #          (vanishing grad exactly where error is largest; explained_var pinned near 0).
        #      (b) with terminal-only ±1 reward, z-scoring the returns ERASES the win/loss signal: an
        #          all-wins batch re-centers to ~0 mean, leaving only the intra-game γ-discount order
        #          as gradient ("later-in-game = good" regardless of the actual result) — a CONFIDENTLY
        #          WRONG signal that worsens as win-rate climbs (batches get win-heavy). Raw returns
        #          keep the outcome signal; V learns E[return|s] to absorb it.

        # 3. Collate the ragged buffer once. Advantage = raw return − stored act-time V (the MC
        #    baseline); detached — a constant target w.r.t. the policy gradient.
        (dense_state, state_ids, opt_b, cid_b, aid_b,
         mask, actions, old_logp, old_v) = self._collate()
        advantages = (returns - old_v).detach()
        # PRE-normalization advantage stats (post-norm mean/std are 0/1 by construction under this
        # FULL-batch normalization — log them too if minibatching ever lands). Bug signatures:
        # adv_mean_raw persistently far from 0 = biased V; adv_std_raw ≈ return_std = V contributes
        # nothing (the broken-critic symptom, seen from the advantage side).
        adv_mean_raw = float(advantages.mean())
        adv_std_raw = float(advantages.std())
        # Normalize the ADVANTAGES (not the returns) — ICLR "37 details" #7. This is the coherent
        # place to normalize: advantages are post-baseline (≈zero-mean by construction), so scaling
        # them only fixes the policy-gradient step size WITHOUT destroying the win/loss information the
        # baseline exists to handle. +1e-8 guards a zero-variance batch (→ ~0 advantages, nothing to
        # learn). NOTE: a good baseline is load-bearing here — early on V≈0 so adv≈returns and this
        # degrades toward return-normalization; it becomes correct as the (now tanh-feasible) V learns.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. K epochs of full-batch clipped-surrogate ascent over the same rollout.
        stats = {}
        ratio_dev_e0 = float("nan")
        grad_norm_max = grad_norm_policy_max = grad_norm_value_max = 0.0
        for epoch in range(self.k_epochs):
            logp, V, entropy = self.policy.evaluate(dense_state, state_ids, opt_b, cid_b, aid_b,
                                                    mask, actions)
            ratio = torch.exp(logp - old_logp)
            if epoch == 0:
                # Epoch-0 ratio must be ≡1: old_logp was captured under this exact θ, so any
                # deviation beyond fp noise means evaluate() failed to reconstruct act()'s
                # distribution (collate/mask/featurization drift) — the ICLR "check ratio=1" rule.
                ratio_dev_e0 = float((ratio.detach() - 1.0).abs().max())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((V - returns) ** 2).mean()    # plain MSE; value-clipping omitted (optional)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # Per-head PRE-clip grad norms, max over the K epochs (spikes are the signal): the
            # global norm can mask a value-head-dominated gradient — the value/policy loss-scale-
            # imbalance signature. Shared embeddings are covered by the total norm only.
            grad_norm_policy_max = max(grad_norm_policy_max, _grad_norm(self.policy.policy_scorer))
            grad_norm_value_max = max(grad_norm_value_max, _grad_norm(self.policy.value_head))
            # THEORY-REVIEW FLAG: the named reference (PPO-PyTorch/PPO.py) uses NO gradient clip;
            # we keep this one-line global-norm clip (ICLR "37 details" #11, norm 0.5) as a
            # standard stabilizer. Minor, deliberate divergence — see the plan/notes.
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            grad_norm_max = max(grad_norm_max, float(grad_norm))
            self.optimizer.step()

            if epoch == self.k_epochs - 1:              # diagnostics from the last epoch (detached)
                with torch.no_grad():
                    # k3 KL estimator (Schulman): mean(ratio − 1 − log ratio) ≥ 0, so this is a
                    # NaN-free, always-nonneg sanity signal (≈0 on epoch 0 where ratio≡1).
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
                        # returns are γ^k·(±1) ∈ [-1,1] by construction — min/max outside that
                        # range means the per-episode reset in the MC recursion broke (reward
                        # leaking across episode boundaries). value_mean: act-time V, catches
                        # tanh-head saturation / sign bias alongside explained_var.
                        "return_mean": float(returns.mean()),
                        "return_std": float(returns.std()),
                        "return_min": float(returns.min()),
                        "return_max": float(returns.max()),
                        "value_mean": float(old_v.mean()),
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
    # and ratio≈1 falls out); one tail-terminal gives a non-degenerate MC return curve. Placed
    # AFTER the grad check because update() clears the buffer.
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
      * single-step-per-episode ⇒ MC return == reward (gamma-independent) and exercises the
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
    ppo = PPO(STATE_DIM, OPTION_DIM, lr=1e-2, gamma=0.99, eps_clip=0.2, k_epochs=4,
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
    # MC returns must be exactly ±1 (any excursion outside [-1,1] = broken episode-boundary reset).
    assert set(stats1) == set(STATS_KEYS), f"stats keys drifted: {set(stats1) ^ set(STATS_KEYS)}"
    assert all(math.isfinite(v) for v in stats1.values()), f"non-finite stat: {stats1}"
    assert -1.0 - 1e-6 <= stats1["return_min"] and stats1["return_max"] <= 1.0 + 1e-6, \
        f"returns escaped [-1,1]: min={stats1['return_min']} max={stats1['return_max']}"
    assert g1 > g0 + eps and g2 > g1 + eps, f"good-action prob did not rise: {g0:.4f} {g1:.4f} {g2:.4f}"
    assert b1 < b0 - eps and b2 < b1 - eps, f"bad-action prob did not fall: {b0:.4f} {b1:.4f} {b2:.4f}"
    assert vg1 > vg0 and vg2 > vg1, f"V(S_g) did not rise toward +: {vg0:.4f} {vg1:.4f} {vg2:.4f}"
    assert vb1 < vb0 and vb2 < vb1, f"V(S_b) did not fall toward −: {vb0:.4f} {vb1:.4f} {vb2:.4f}"
    assert stats2["value_loss"] < stats1["value_loss"], "value loss did not decrease across updates"
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
