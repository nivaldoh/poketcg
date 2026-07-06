#!/usr/bin/env python3
"""Gymnasium wrapper around the `cg` battle loop for reinforcement learning.

This is the SB3 experiment's env layer: the hand-rolled PPO collection loop inverted into
`reset()`/`step()` semantics. Between two of the learner's decisions the env services everything
else internally — opponent turns via the `make_opponent` provider (below), and the learner's own
non-policy nodes via `auto_resolve` — so `step()` only ever surfaces SINGLE-SELECT policy nodes
(`is_policy_select`), the v1 coverage. Multi-select is a planned extension (see plan.md Feature 2).

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
from tcg_model import is_policy_select, auto_resolve, random_agent     # noqa: E402
from sb3_features import encode_state, encode_options, STATE_DIM, OPTION_DIM  # noqa: E402
from eval import make_rulebased, make_abomasnow, make_dragapult, make_iono  # noqa: E402  (opponent field)
from win_mode import classify_end                                      # noqa: E402  (terminal-obs telemetry)
from cg.api import to_observation_class                                # noqa: E402
from cg.game import battle_start, battle_select, battle_finish         # noqa: E402

# Padded option-list width. A 65-option MAIN policy node was OBSERVED in a 48-game rulebased
# round-robin (2026-07-06 review measurement), so 64 was too tight; MAIN scales with hand size ×
# actions, and non-policy nodes measured ≤20 (bounded by the 60-card pool). A policy node exceeding
# this is auto-resolved instead of surfaced (counted in `overflows` / warned once) — if that
# counter ever fires on rulebased play, measure and raise this again.
N_MAX = 128


def obs_to_gym(o):
    """Observation dataclass at a policy node (1 < n_options <= N_MAX) -> the 3-key gym obs dict.
    Shared by CGEnv and eval_sb3's agent adapter so train- and eval-time featurization can't
    drift. Padding rows are all-zero (distinguishable from real rows by the OptionType one-hot)."""
    dense, _state_ids = encode_state(o)
    opt_dense, _cids, _aids = encode_options(o)
    n = opt_dense.shape[0]
    assert 0 < n <= N_MAX, f"obs_to_gym called on a node with {n} options (N_MAX={N_MAX})"
    options = np.zeros((N_MAX, OPTION_DIM), dtype=np.float32)
    options[:n] = opt_dense.numpy()
    mask = np.zeros(N_MAX, dtype=np.int8)
    mask[:n] = 1
    return {"state": dense.numpy().astype(np.float32),
            "options": options,
            "action_mask": mask}


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
    """One learner seat vs the rule-based opponent field, one battle per episode.

    `decks`: list of (name, 60-card list) pairs; the learner plays a random one per episode and
    the seat index alternates per episode (symmetry). `opponent_kind` is the `make_opponent` spec
    (comma mix, @eps suffix); `self`/`frozen` need a live policy and are rejected here (deferred —
    plan.md Feature 1)."""

    metadata = {"render_modes": []}

    def __init__(self, decks, opponent_kind="rulebased"):
        super().__init__()
        for part in opponent_kind.split(","):
            base = part.strip().partition("@")[0]
            if base in ("self", "frozen"):
                raise ValueError(f"opponent kind {base!r} needs a live policy — unsupported in CGEnv (deferred)")
        self._decks = list(decks)
        self._next_opponent = make_opponent(opponent_kind)
        self._episode = 0
        self._live = False          # a cg battle is live (battle_finish still owed)
        self._raw = None            # latest raw obs dict (loop control reads the dict)
        self._cur = None            # Observation dataclass at the pending policy node
        self._seat = 0
        self._opp_fn = None
        self._opp_name = self._deck_name = None
        self.overflows = 0          # policy nodes auto-resolved because n_options > N_MAX

        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            "options": spaces.Box(-np.inf, np.inf, (N_MAX, OPTION_DIM), np.float32),
            "action_mask": spaces.MultiBinary(N_MAX),
        })
        self.action_space = spaces.Discrete(N_MAX)

    # -- engine plumbing ------------------------------------------------------------------
    def _finish_if_live(self):
        if self._live:
            self._live = False
            self._cur = None        # no pending policy node may outlive its battle
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
        """Service opponent decisions and the learner's non-policy nodes until the learner's next
        policy node (returns its Observation) or terminal (finishes the battle, returns None).
        Any engine exception finishes the battle before propagating (singleton discipline)."""
        obs = self._raw
        try:
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] != self._seat:
                    obs = battle_select(self._opp_fn(obs))
                    continue
                o = to_observation_class(obs)
                sel = o.select
                if is_policy_select(sel):
                    if len(sel.option) <= N_MAX:
                        self._raw = obs
                        return o
                    # A learnable node too wide to surface — auto-resolved, out of learning.
                    self.overflows += 1
                    if self.overflows == 1:
                        warnings.warn(f"policy node with {len(sel.option)} options > N_MAX={N_MAX} "
                                      f"auto-resolved (counted in CGEnv.overflows)")
                obs = battle_select(auto_resolve(sel))
        except BaseException:
            self._finish_if_live()
            raise
        self._raw = obs
        self._finish_if_live()                              # terminal: pay the finish NOW
        return None

    # -- gymnasium API --------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)       # deck/opponent draws + auto_resolve paths use the global random stream
        self._finish_if_live()      # mid-episode reset (check_env does this; DummyVecEnv never)
        info = {}
        for _ in range(20):
            self._start_battle()
            self._cur = self._advance_to_policy()
            if self._cur is not None:
                return obs_to_gym(self._cur), info
            # Game ended before the learner ever saw a policy node (rare; auto-resolve carried the
            # whole game). Not a learnable episode — start another.
            info["skipped_games"] = info.get("skipped_games", 0) + 1
        raise RuntimeError("20 consecutive games without a learner policy node. Something is wrong")

    def step(self, action):
        assert self._cur is not None, "step() called without a pending policy node (reset first)"
        info = {}
        a = int(action)
        n = len(self._cur.select.option)
        if not 0 <= a < n:
            # Masked-out (padding) slot: only raw-space samplers hit this (env_checker); MaskablePPO
            # never does. Substitute a random legal pick rather than crash a run.
            info["invalid_action"] = True
            a = random.randrange(n)
        try:
            self._raw = battle_select([a])
        except BaseException:
            self._finish_if_live()
            raise
        self._cur = self._advance_to_policy()
        if self._cur is None:                               # terminal (battle already finished)
            result = self._raw["current"]["result"]
            G = 0.0 if result == 2 else (1.0 if result == self._seat else -1.0)
            info.update(win_mode=classify_end(self._raw, self._seat),
                        opp_name=self._opp_name, deck_name=self._deck_name,
                        overflows=self.overflows)
            return self._terminal_obs(), G, True, False, info
        return obs_to_gym(self._cur), 0.0, False, False, info

    def _terminal_obs(self):
        """Placeholder terminal observation (DummyVecEnv stores it as info['terminal_observation']
        and immediately auto-resets; nothing learns from it). Slot 0 stays valid in the mask so the
        dict is safe even if something ever softmaxes it."""
        obs = {"state": np.zeros(STATE_DIM, dtype=np.float32),
               "options": np.zeros((N_MAX, OPTION_DIM), dtype=np.float32),
               "action_mask": np.zeros(N_MAX, dtype=np.int8)}
        obs["action_mask"][0] = 1
        return obs

    def action_masks(self):
        """The sb3-contrib MaskablePPO channel: bool[N_MAX], True = selectable option index."""
        mask = np.zeros(N_MAX, dtype=bool)
        n = len(self._cur.select.option) if self._cur is not None else 1
        mask[:n] = True
        return mask

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
    env.close()
    print(f"OK: CGEnv reset+step (STATE_DIM={STATE_DIM}, OPTION_DIM={OPTION_DIM}, N_MAX={N_MAX})")
