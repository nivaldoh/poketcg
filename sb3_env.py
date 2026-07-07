#!/usr/bin/env python3
"""Gymnasium wrapper around the `cg` battle loop for reinforcement learning.

This is the SB3 experiment's env layer: the hand-rolled PPO collection loop inverted into
`reset()`/`step()` semantics. Between two of the learner's decisions the env services everything
else internally — opponent turns via the `make_opponent` provider (below), and the learner's own
non-learnable nodes via `auto_resolve`. `step()` surfaces LEARNABLE nodes (`is_learnable_select`):
single-select and the full multi-select range — optional-single [0..1] (including the single-option
decline), fixed-k (min==max), and true-variable (min<max, max>1). A multi-select node is decomposed
into a sequence of masked single-select sub-steps (a reserved STOP slot commits the picked set;
auto-commit at maxCount) — see `plan multiselect.md`.

Fixed shapes for SB3's rollout buffer: the variable-length option list is padded to `N_MAX` rows
(zero rows are unambiguous — every real row has a nonzero OptionType one-hot bit, see
sb3_features._option_row) with a validity mask. Masking rides TWO channels: the `action_mask` obs key
(available to the net as input) and the `action_masks()` method (the sb3-contrib MaskablePPO
protocol — what actually constrains sampling). Card/attack id embedding inputs are deferred (the
stock MultiInputPolicy has no embedding tables to feed).

The cg battle is a PROCESS-GLOBAL SINGLETON (cg/sim.py `Battle` class attrs): one live CGEnv per
process, no VecEnv n>1 — `battle_finish()` runs on terminal, on live-reset, on `close()`, and on
any engine exception, because a leaked live battle corrupts every later game in the process.

Reward: terminal-only G ∈ {+1 win, -1 loss, 0 draw}, seat-relative, undiscounted (pair with
gamma=1.0 — THEORY-REVIEW flag). Intermediate steps reward 0.

Smallest build check (imports + spaces resolve):
    python sb3_env.py
Full check: utils/run_smoketest_sb3_env.py
"""

import os
import random
import sys
import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: importing tcg_model runs the `CG_LIB_PATH` sys.path insert
# (tcg_model.py, the `_CG_LIB_PATH` block) that makes the vendored `cg` package importable. The
# `sb3_features`/`eval`/`cg` imports below depend on that having happened — keep this ordering.
from tcg_model import auto_resolve, random_agent                       # noqa: E402
from sb3_features import encode_state, encode_options, STATE_DIM, OPTION_DIM  # noqa: E402
from eval import make_rulebased, make_abomasnow, make_dragapult, make_iono  # noqa: E402  (opponent field)
from win_mode import classify_end                                      # noqa: E402  (terminal-obs telemetry)
from cg.api import to_observation_class, OptionType, SelectType        # noqa: E402
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402

# Padded option-list width. A 65-option MAIN policy node was OBSERVED in a 48-game rulebased
# round-robin (2026-07-06 review measurement), so 64 was too tight; MAIN scales with hand size ×
# actions, and non-policy nodes measured ≤20 (bounded by the 60-card pool). A policy node exceeding
# this is auto-resolved instead of surfaced (counted in `overflows` / warned once) — if that
# counter ever fires on rulebased play, measure and raise this again.
N_MAX = 128
STOP_INDEX = N_MAX - 1           # reserved top slot: the "commit what I've picked" action (multi-select)

# Extended obs widths for multi-select. The base encoders in sb3_features stay FROZEN (they see only
# the static Observation, not the running sub-step state), so the extra fields are appended HERE:
#   options gains 2 per-row bits  -> is_stop (row STOP_INDEX), already_picked (picked earlier this node)
#   state   gains 4 board scalars -> in_multi, num_picked, remaining_min, remaining_max
OPTION_DIM_LOCAL = OPTION_DIM + 2
STATE_DIM_LOCAL = STATE_DIM + 4
_MS_COUNT_NORM = 8.0             # keep the appended counts ~unit-scale (MultiInputPolicy isn't VecNormalize-wrapped)


def is_learnable_select(sel):
    """The nodes the policy is allowed to decide — a STRICT SUPERSET of is_policy_select. Learnable
    iff the node carries real discretion: not an all-NUMBER "how many", maxCount >= 1, and
    minCount < len (which rejects both "must pick all" and the no-choice forced single-option
    [1..1]). This admits the FULL multi-select range — single-select [1..1], optional-single [0..1]
    (including the single-option decline, len==1), fixed-k (min==max), and true-variable
    (min<max, max>1) — each decomposed into masked single-select sub-steps (see the module docstring
    / `plan multiselect.md`). Multi-pick SKILL nodes are excluded: their index ORDER is semantically
    meaningful (SKILL_ORDER, §3f), which the set-based sub-step loop doesn't model; single-pick SKILL
    is unaffected. is_policy_select (tcg_model) stays as-is for other callers."""
    if sel is None:
        return False
    n = len(sel.option)
    if n < 1:                                           # len==1 is allowed (the [0..1] decline); the
        return False                                    #   mn>=n guard below drops forced-single [1..1]
    if all(OptionType(o.type) == OptionType.NUMBER for o in sel.option):
        return False
    mn, mx = sel.minCount, sel.maxCount
    if mx < 1 or mn >= n:                               # no discretion: must-pick-all or forced [1..1]
        return False
    if SelectType(sel.type) == SelectType.SKILL and mx > 1:   # order-dependent; deferred to rules review
        return False
    return True


def _ms_mask(sel, picked):
    """bool[N_MAX] selectability at a (possibly mid-pick) node — the ONE mask shared by
    CGEnv.action_masks, obs_to_gym's action_mask key, and sb3_agent's eval loop, so all three agree.
    Real option i selectable <=> i not already picked and p < maxCount; STOP (slot STOP_INDEX)
    selectable <=> minCount <= p < maxCount. Provably non-empty whenever a node is presented (a node
    is only shown at p < maxCount <= len(option), so >=1 unpicked real option always remains)."""
    n = len(sel.option)
    p = len(picked)
    mask = np.zeros(N_MAX, dtype=bool)
    if p < sel.maxCount:
        mask[:n] = True
        for i in picked:
            mask[i] = False
        if p >= sel.minCount:
            mask[STOP_INDEX] = True
    return mask

# TODO: write comments with the shapes here so we can keep track of them more easily
def obs_to_gym(o, picked=()):
    """Observation dataclass at a learnable node -> the 3-key gym obs dict, rendered for the current
    multi-select sub-step (`picked` = indices chosen so far this node; () = fresh node / single-
    select). Shared by CGEnv and sb3_agent so train- and eval-time featurization can't drift. Base
    features come from the frozen sb3_features encoders; the STOP row + is_stop/already_picked bits
    and the 4 board-level count scalars are appended here (they depend on runtime sub-step state).
    Padding rows stay all-zero (distinguishable from real rows by the OptionType one-hot; the STOP
    row is distinguishable by its is_stop bit)."""
    sel = o.select
    dense, _state_ids = encode_state(o)
    opt_dense, _cids, _aids = encode_options(o)
    n = opt_dense.shape[0]
    assert 0 < n <= N_MAX - 1, f"obs_to_gym: {n} options exceeds N_MAX-1={N_MAX - 1} (STOP reserved)"
    picked = list(picked)
    p = len(picked)
    mn, mx = sel.minCount, sel.maxCount

    options = np.zeros((N_MAX, OPTION_DIM_LOCAL), dtype=np.float32)
    options[:n, :OPTION_DIM] = opt_dense.numpy()
    for i in picked:
        options[i, OPTION_DIM + 1] = 1.0                # already_picked
    options[STOP_INDEX, OPTION_DIM] = 1.0               # is_stop (static marker, present every node)

    if mn == 1 and mx == 1:
        extra = np.zeros(4, dtype=np.float32)           # single-select carries no multi context
    else:
        extra = np.array([1.0, p / _MS_COUNT_NORM,      # in_multi, num_picked, remaining_min/max
                          max(0, mn - p) / _MS_COUNT_NORM, (mx - p) / _MS_COUNT_NORM], dtype=np.float32)
    state = np.concatenate([dense.numpy().astype(np.float32), extra])

    return {"state": state,
            "options": options,
            "action_mask": _ms_mask(sel, picked).astype(np.int8)}


def sb3_agent(model, greedy=False):
    """Wrap a MaskablePPO (or anything with a masked `.predict`) as an agent_fn(obs_dict) ->
    list[int] with the training-time decision routing: auto_resolve non-learnable/overflow nodes,
    masked predict on learnable nodes. Multi-select nodes are resolved by running the sub-step loop
    INTERNALLY (no engine calls — the intermediate states are computed from `sel` + the growing
    `picked` set) and returning the full pick list in one shot, so the eval driver's single
    battle_select gets an engine-valid answer. Shared by eval_sb3 (model seat) and self-play
    (opponent seat); the loop mirrors CGEnv.step exactly (same obs_to_gym / _ms_mask / branch
    conditions) so train- and eval-time routing can't drift."""
    def agent_fn(obs_dict):
        o = to_observation_class(obs_dict)
        sel = o.select
        if not is_learnable_select(sel) or len(sel.option) > N_MAX - 1:
            return auto_resolve(sel)
        picked = []
        while True:
            gobs = obs_to_gym(o, picked)
            action, _ = model.predict(gobs, action_masks=_ms_mask(sel, picked),
                                      deterministic=greedy)
            a = int(action)
            if a == STOP_INDEX:
                return picked
            picked.append(a)
            if len(picked) == sel.maxCount:
                return picked
    return agent_fn


def make_opponent(kind):
    """Return a per-game provider next_opponent() -> (opp_fn, opp_deck_or_None, opp_name) for the
    non-learner seat; opp_deck None = mirror (opponent plays the learner's deck). `kind` may be a
    COMMA LIST (e.g. 'rulebased,random'): one entry is drawn uniformly per game, and repeating a
    kind weights it ('rulebased,rulebased,random' = 2:1) — the curriculum lever for reward
    starvation. An `@eps` suffix (e.g. 'rulebased@0.25') corrupts each opponent DECISION to a random
    legal action with prob eps (an on-distribution difficulty dial). Default: rulebased.

    Moved from the retired `train_ppo.make_opponent`; the live-policy `self`/`frozen` branches are
    dropped — the SB3 self-play path installs a policy opponent on `CGEnv` directly (plan Feature 1)."""
    kinds = [k.strip() for k in kind.split(",") if k.strip()]
    if not kinds:
        raise ValueError(f"empty opponent spec: {kind!r}")
    if len(kinds) > 1:
        providers = [make_opponent(k) for k in kinds]
        return lambda: random.choice(providers)()
    kind = kinds[0]
    if "@" in kind:
        base, _, eps_s = kind.partition("@")
        eps = float(eps_s)
        if not 0.0 < eps < 1.0:
            raise ValueError(f"eps must be in (0,1): {kind!r}")
        inner = make_opponent(base)

        def next_weakened():
            fn, deck, name = inner()

            def noisy_fn(obs):
                # ALWAYS run the base agent, then maybe discard its pick: the rule-based modules
                # keep module-level per-turn state (turn-0 reset / plan / can_attack) that must
                # see every decision to stay coherent.
                sel = fn(obs)
                return random_agent(obs) if random.random() < eps else sel
            return noisy_fn, deck, f"{name}@{eps:g}"
        return next_weakened
    if kind == "random":
        return lambda: (random_agent, None, "random")
    if kind == "rulebased":
        # THE FIELD (default): one of the 4 rule-based archetypes per game, each playing ITS OWN
        # deck (asymmetric battle_start). Reuses eval.py's factories; the modules reset their
        # per-game state on their own turn-0 detection, so reuse across games is safe.
        pool = [make_rulebased(), make_abomasnow(), make_dragapult(), make_iono()]

        def next_opponent():
            a = random.choice(pool)
            return a.fn, a.deck, a.name               # eval.Agent.name (make_rulebased = "rulebased")
        return next_opponent
    raise ValueError(f"unknown opponent kind: {kind!r} (self/frozen use CGEnv's policy-opponent path)")


class CGEnv(gym.Env):
    """One learner seat vs the opponent field, one battle per episode.

    `decks`: list of (name, 60-card list) pairs; the learner plays a random one per episode and
    the seat index alternates per episode (symmetry). `opponent_kind` is the `make_opponent` spec
    (comma mix, @eps suffix), OR the bare spec `self` for self-play — the learner's own live policy
    drives the other seat, installed after model creation via `set_policy_opponent`."""

    metadata = {"render_modes": []}

    def __init__(self, decks, opponent_kind="rulebased"):
        super().__init__()
        self._decks = list(decks)
        self._policy_opponent_fn = None         # installed post-model via set_policy_opponent (self-play)
        self._policy_opponent_spec = None
        if opponent_kind.strip() == "self":
            # Policy-backed opponent: the learner's own live policy drives the other seat. The fn is
            # injected AFTER model creation (set_policy_opponent) — the env is built before
            # MaskablePPO exists — so _next_opponent reads it lazily at reset() time.
            self._policy_opponent_spec = opponent_kind.strip()
            self._next_opponent = self._next_policy_opponent
        else:
            # self/frozen are only valid as the exact bare spec in v1 (make_opponent can't build
            # them; mixing into the rule-based field or an @eps suffix is a documented follow-on).
            for part in opponent_kind.split(","):
                base = part.strip().partition("@")[0]
                if base in ("self", "frozen"):
                    raise ValueError(f"opponent kind {base!r} is only supported as the exact bare "
                                     f"spec in v1 (no comma-mix, no @eps): got {opponent_kind!r}")
            self._next_opponent = make_opponent(opponent_kind)
        self._episode = 0
        self._live = False          # a cg battle is live (battle_finish still owed)
        self._raw = None            # latest raw obs dict (loop control reads the dict)
        self._cur = None            # Observation dataclass at the pending policy node
        self._seat = 0
        self._opp_fn = None
        self._opp_name = self._deck_name = None
        self._picked = []           # indices chosen so far in the current multi-select node
        self.overflows = 0          # learnable nodes auto-resolved because n_options > N_MAX-1
        # multi-select telemetry (mirror the overflows pattern; read by the smoketest)
        self.ms_optional = 0        # optional-single [0..1] nodes surfaced (any option count)
        self.ms_fixed = 0           # fixed-k (min==max>1) nodes surfaced
        self.ms_variable = 0        # true-variable (min<max, max>1) nodes surfaced
        self.ms_single_option = 0   # single-option [0..1] decline nodes surfaced (len==1; ms_optional subset)
        self.ms_empty_submit = 0    # battle_select([]) committed (STOP at minCount==0 — the decline)

        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM_LOCAL,), np.float32),
            "options": spaces.Box(-np.inf, np.inf, (N_MAX, OPTION_DIM_LOCAL), np.float32),
            "action_mask": spaces.MultiBinary(N_MAX),
        })
        self.action_space = spaces.Discrete(N_MAX)

    # -- self-play opponent ----------------------------------------------------------------
    def set_policy_opponent(self, agent_fn):
        """Install the policy that drives the opponent seat under `--opponent self`. Called once
        after MaskablePPO is built (the env is created before the model exists); MaskablePPO wraps
        this env by reference, so the installed fn reaches the same object model.learn() steps."""
        self._policy_opponent_fn = agent_fn

    def _next_policy_opponent(self):
        """Per-game provider for the policy-backed opponent: the injected policy fn drives the other
        seat, playing the learner's own deck (mirror, opp_deck=None). Read lazily so the fn can be
        installed after the env is built."""
        fn = self._policy_opponent_fn
        assert fn is not None, ("self-play opponent not installed: call "
                                "env.set_policy_opponent(sb3_agent(model)) after model creation, "
                                "before reset()")
        return fn, None, self._policy_opponent_spec

    # -- engine plumbing ------------------------------------------------------------------
    def _finish_if_live(self):
        if self._live:
            self._live = False
            self._cur = None        # no pending policy node may outlive its battle
            self._picked = []       # nor its per-node pick state
            battle_finish()

    def _start_battle(self):
        self._seat = self._episode % 2                      # alternate seats for symmetry
        self._episode += 1
        self._deck_name, learner_deck = random.choice(self._decks)
        self._opp_fn, opp_deck, self._opp_name = self._next_opponent()
        seat_decks = [None, None]
        seat_decks[self._seat] = learner_deck
        seat_decks[1 - self._seat] = opp_deck if opp_deck is not None else learner_deck
        obs, start = battle_start(seat_decks[0], seat_decks[1])
        if start.errorPlayer >= 0:
            raise ValueError("deck rejected by engine")
        self._live = True
        self._raw = obs

    def _advance_to_policy(self):
        """Service opponent decisions and the learner's non-learnable nodes until the learner's next
        learnable node (returns its Observation, with a fresh empty `picked`) or terminal (finishes
        the battle, returns None). Any engine exception finishes the battle before propagating
        (singleton discipline)."""
        obs = self._raw
        try:
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] != self._seat:
                    obs = battle_select(self._opp_fn(obs))
                    continue
                o = to_observation_class(obs)
                sel = o.select
                if is_learnable_select(sel):
                    if len(sel.option) <= N_MAX - 1:        # reserve slot STOP_INDEX
                        self._raw = obs
                        self._picked = []                   # fresh node
                        self._count_shape(sel)
                        return o
                    # TODO: this seems terrible. We need to start keeping track of how often this happens
                    # A learnable node too wide to surface — auto-resolved, out of learning.
                    self.overflows += 1
                    if self.overflows == 1:
                        warnings.warn(f"learnable node with {len(sel.option)} options > N_MAX-1="
                                      f"{N_MAX - 1} auto-resolved (counted in CGEnv.overflows)")
                obs = battle_select(auto_resolve(sel))
        except BaseException:
            self._finish_if_live()
            raise
        self._raw = obs
        self._finish_if_live()                              # terminal: pay the finish NOW
        return None

    def _count_shape(self, sel):
        """Tally a surfaced node's multi-select shape (telemetry mirrored to the smoketest)."""
        mn, mx = sel.minCount, sel.maxCount
        if len(sel.option) == 1:            # a surfaced len==1 node is necessarily the [0..1] decline
            self.ms_single_option += 1      #   (separate if: it's ALSO counted as optional below)
        if mx <= 1 and mn == 0:
            self.ms_optional += 1
        elif mn == mx and mx > 1:
            self.ms_fixed += 1
        elif mn < mx and mx > 1:
            self.ms_variable += 1
        # else single-select (mn == mx == 1): not a multi node, not counted

    # -- gymnasium API --------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)       # deck/opponent draws + auto_resolve paths use the global random stream
        self._finish_if_live()      # mid-episode reset (check_env does this; DummyVecEnv never)
        info = {}
        for _ in range(20):
            self._start_battle()
            # advance until it's our turn to act (or terminal)
            self._cur = self._advance_to_policy()
            if self._cur is not None:
                return obs_to_gym(self._cur, self._picked), info
            # Game ended before the learner ever saw a policy node (rare; auto-resolve carried the
            # whole game). Not a learnable episode — start another.
            info["skipped_games"] = info.get("skipped_games", 0) + 1
        raise RuntimeError("20 consecutive games without a learner policy node. Something is wrong")

    def step(self, action):
        assert self._cur is not None, "step() called without a pending policy node (reset first)"
        info = {}
        sel = self._cur.select
        a = int(action)
        mask = _ms_mask(sel, self._picked)
        if not (0 <= a < N_MAX and mask[a]):
            # Masked-out slot: only raw-space samplers hit this (env_checker); MaskablePPO never
            # does. Substitute a random LEGAL pick (drawn FROM the live mask, so it can't duplicate
            # an already-picked index -> no spurious IndexError) rather than crash a run.
            info["invalid_action"] = True
            a = int(random.choice(np.flatnonzero(mask)))
        if a == STOP_INDEX:
            return self._commit(info)                       # submit picked (== [] when minCount==0)
        self._picked.append(a)
        if len(self._picked) == sel.maxCount:
            return self._commit(info)                       # reached the ceiling -> auto-commit
        # Non-final pick: re-present the SAME node (picked option now masked out), reward 0.
        return obs_to_gym(self._cur, self._picked), 0.0, False, False, info

    # So this exists just because of multi-select? E.g. "You may draw up to 2 cards and discard 1"
    def _commit(self, info):
        """Submit the accumulated `picked` set to the engine and advance to the next learner node.
        Shared tail of the STOP and auto-commit branches of step()."""
        picked = list(self._picked)
        if not picked:
            self.ms_empty_submit += 1                       # STOP at minCount==0 — the decline path
        try:
            self._raw = battle_select(picked)
        except BaseException:
            self._finish_if_live()
            raise
        self._cur = self._advance_to_policy()               # resets self._picked when it surfaces a node
        if self._cur is None:                               # terminal (battle already finished)
            result = self._raw["current"]["result"]
            G = 0.0 if result == 2 else (1.0 if result == self._seat else -1.0)
            info.update(win_mode=classify_end(self._raw, self._seat),
                        opp_name=self._opp_name, deck_name=self._deck_name,
                        overflows=self.overflows)
            return self._terminal_obs(), G, True, False, info
        return obs_to_gym(self._cur, self._picked), 0.0, False, False, info

    def _terminal_obs(self):
        """Placeholder terminal observation (DummyVecEnv stores it as info['terminal_observation']
        and immediately auto-resets; nothing learns from it). Slot 0 stays valid in the mask so the
        dict is safe even if something ever softmaxes it."""
        obs = {"state": np.zeros(STATE_DIM_LOCAL, dtype=np.float32),
               "options": np.zeros((N_MAX, OPTION_DIM_LOCAL), dtype=np.float32),
               "action_mask": np.zeros(N_MAX, dtype=np.int8)}
        obs["action_mask"][0] = 1
        return obs

    def action_masks(self):
        """The sb3-contrib MaskablePPO channel: bool[N_MAX], True = selectable action (a real option
        or STOP). The SAME _ms_mask the obs's action_mask key carries, so the two channels never
        disagree. Falls back to a slot-0-only mask when no node is pending (terminal / pre-reset)."""
        if self._cur is None:
            mask = np.zeros(N_MAX, dtype=bool)
            mask[0] = True
            return mask
        return _ms_mask(self._cur.select, self._picked)

    def close(self):
        self._finish_if_live()


if __name__ == "__main__":
    # Smallest build check: spaces construct and one battle reaches a policy node and steps once.
    from utils.run_smoketest_battle import load_deck
    deck = load_deck(os.path.join(_REPO_ROOT, "decks", "lucario.csv"))
    env = CGEnv([("lucario", deck)])
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.observation_space.spaces), "obs keys != space keys"
    assert env.observation_space.contains(obs), "reset obs outside the declared space"
    obs2, r, term, trunc, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert r == 0.0 or term, "non-terminal step must pay 0"
    assert env.observation_space.contains(obs2), "post-step obs outside the declared space"
    env.close()
    print(f"OK: CGEnv reset+step (STATE_DIM_LOCAL={STATE_DIM_LOCAL}, "
          f"OPTION_DIM_LOCAL={OPTION_DIM_LOCAL}, N_MAX={N_MAX})")
