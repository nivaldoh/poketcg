"""Reusable PPO implementation layer for the Pokémon TCG `cg` agent.

This is the *model/algorithm* half of the from-scratch PPO trainer — extracted from
`train_ppo.py` so the driver loop stays lean (mirrors how `tcg_model.py` holds the reusable
model layer for the other trainers). The training driver (battle collection, opponents, the
loop, argparse) lives in `train_ppo.py`, which imports `PPO`, `STATE_DIM`, `OPTION_DIM` from
here.

This remains a deliberately staged, *reviewable skeleton*. The scaffolding and the net are
written out concretely; the two genuinely design-heavy pieces — state/option FEATURIZATION
(`encode_state` / `encode_options`) and the PPO UPDATE math (`PPO.update`) — are left as
documented stubs that raise `NotImplementedError`. We fill those in one at a time (see the
roadmap at the bottom of `train_ppo.py`). Nothing here trains yet.

Design (locked):
  * FRESH network from scratch — we do NOT reuse MyModel or its sparse featurizers.
  * MONTE-CARLO discounted returns (advantage = return − V), exactly like PPO-PyTorch/PPO.py.
    GAE can slot in later behind the same `update()` interface.

Reference: PPO-PyTorch/PPO.py (RolloutBuffer + ActorCritic + PPO in one file) and
docs/stackoverflow_ppo_explanation.txt (clipped objective + K epochs).

THE one structural departure from the reference: cg's action space is NOT a fixed-width
discrete set. Each decision exposes a variable list `obs.select.option` whose length AND
meaning change every step, so we cannot use the reference's `Linear(H, action_dim) → Softmax`.
Instead we use a PER-OPTION SCORING head: featurize each legal option, score each to a scalar
logit, then take a (masked) softmax over exactly the options present this step.

v1 coverage limit: we only learn SINGLE-SELECT policy nodes (minCount==maxCount==1,
len(option)>1, not all-NUMBER — `is_policy_select`). Every other decision is auto-resolved
(`auto_resolve`) and emits no training record. Multi-select is a documented future extension.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.distributions import Categorical

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: `eval_mymodel` imports `tcg_model` at its own top, which runs
# the `CG_LIB_PATH` sys.path insert (tcg_model.py, the `_CG_LIB_PATH` block) that makes the
# vendored `cg` package importable. Keep eval_mymodel above the `cg.api` import below.
from eval_mymodel import is_policy_select, auto_resolve                 # noqa: E402
from cg.api import to_observation_class                                 # noqa: E402


# ---------------------------------------------------------------------------------------------
# Feature dimensions. PLACEHOLDERS — these must equal the widths `encode_state` / `encode_options`
# actually emit once implemented. The net (ActorCritic.__init__) is built against them, so this
# is the one number to keep in sync when we write featurization in the next step.
# ---------------------------------------------------------------------------------------------
STATE_DIM = 64      # length of encode_state(o)
OPTION_DIM = 32     # per-option feature width from encode_options(o)
HIDDEN = 64         # reference uses 64-wide Tanh MLPs


# ============================================================================================
# FEATURIZATION — STUBS (fill in step 1). Both take an `Observation` (already converted via
# to_observation_class) framed from the ACTING seat's perspective (obs.current.yourIndex).
# ============================================================================================
def encode_state(o) -> torch.Tensor:
    """STUB → Tensor[STATE_DIM]. A fresh, hand-engineered, fixed-size state vector.

    Candidate features (all observable in `o.current` / `o.current.players[*]`, see cg.api
    PlayerState/State), framed from the acting seat (yourIndex) vs the opponent:
      * prize counts both seats (len(player.prize)); the prize DIFFERENTIAL is the win signal
      * active Pokémon HP / damage, and bench HP/damage (your side fully visible)
      * bench / hand / deck counts (deckCount, handCount); opponent hand is counts-only
      * active status conditions (poisoned/burned/asleep/paralyzed/confused)
      * turn number, and the per-turn flags (energyAttached, supporterPlayed, retreated, ...)
    Keep it SMALL and numeric for a minimal first net; normalize counts to ~[0,1].
    """
    raise NotImplementedError("encode_state: featurization is step 1 of the fill-in roadmap")


def encode_options(o) -> torch.Tensor:
    """STUB → Tensor[n, OPTION_DIM], one row per legal option in `o.select.option`.

    Per-option features from each cg.api `Option` (n == len(o.select.option)):
      * a one-hot of OptionType(option.type) (PLAY/ATTACH/EVOLVE/RETREAT/ATTACK/CARD/...)
      * a few scalars where present: area, index, count, cardId-bucketed, attackId, energyIndex
    The policy scores each row independently (no cross-option attention), so row i must depend
    only on option i (+ the shared state). Row order MUST match option index order — the sampled
    row index is returned directly as the cg selection `[i]`.
    """
    raise NotImplementedError("encode_options: featurization is step 1 of the fill-in roadmap")


# ============================================================================================
# NETWORK — fresh ActorCritic (concrete layers + act/evaluate; calls the featurization stubs).
# ============================================================================================
class ActOutput:
    """What act() hands back to the collector (so the buffer can store the exact featurized
    inputs + the act-time logprob/value the PPO ratio must be computed against)."""
    __slots__ = ("action", "logprob", "value", "state_feat", "option_feats")

    def __init__(self, action, logprob, value, state_feat, option_feats):
        self.action = action            # int, index into obs.select.option
        self.logprob = logprob          # scalar tensor, log π(a|s) at act time (detached)
        self.value = value              # scalar tensor, V(s) at act time (detached)
        self.state_feat = state_feat    # Tensor[STATE_DIM]  (detached, stored for update)
        self.option_feats = option_feats  # Tensor[n, OPTION_DIM] (detached, stored for update)


class ActorCritic(nn.Module):
    """Separate value + policy MLPs (no shared trunk), mirroring the reference — but the policy
    is a per-option SCORER instead of a fixed-width Softmax (see module docstring)."""

    def __init__(self, state_dim=STATE_DIM, option_dim=OPTION_DIM, hidden=HIDDEN):
        super().__init__()
        # Value head: V(s) ∈ [-1, 1] (terminal reward is ±1, so a tanh output is well-matched).
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )
        # Policy scorer: scores ONE option from [state ⊕ option] → scalar logit. Applied to every
        # legal option, then softmax over the present options.
        self.policy_scorer = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, *args, **kwargs):  # use act()/evaluate() directly, like the reference
        raise NotImplementedError("call act() / evaluate(), not forward()")

    def _score(self, state_feat, option_feats):
        """[STATE_DIM], [n, OPTION_DIM] → [n] option logits (batch=1, no padding)."""
        ctx = state_feat.unsqueeze(0).expand(option_feats.shape[0], -1)   # [n, STATE_DIM]
        x = torch.cat([ctx, option_feats], dim=-1)                        # [n, STATE_DIM+OPTION_DIM]
        return self.policy_scorer(x).squeeze(-1)                          # [n]

    def _value(self, state_feat):
        return self.value_head(state_feat).squeeze(-1)                    # scalar

    def act(self, o) -> ActOutput:
        """Collection-time actor for ONE single-select node: featurize, score, sample one option.
        Caller must wrap in no_grad. All n options are valid here (no padding at batch=1)."""
        state_feat = encode_state(o)            # [STATE_DIM]
        option_feats = encode_options(o)        # [n, OPTION_DIM]
        logits = self._score(state_feat, option_feats)   # [n]
        dist = Categorical(logits=logits)
        a = dist.sample()
        return ActOutput(
            int(a.item()),
            dist.log_prob(a).detach(),
            self._value(state_feat).detach(),
            state_feat.detach(),
            option_feats.detach(),
        )

    def evaluate(self, state_b, option_b, mask, actions):
        """Update-time re-scoring of STORED actions under the CURRENT policy (K-epoch loop).

        Ragged option counts are padded to a common width W with a validity `mask`:
          state_b  : [B, STATE_DIM]
          option_b : [B, W, OPTION_DIM]   (zero-padded rows past each item's n)
          mask     : [B, W] bool          (True on real options)
          actions  : [B] long             (stored option index, always within the real region)
        Returns (logprobs[B], values[B], entropy[B]).
        """
        B, W, _ = option_b.shape
        ctx = state_b.unsqueeze(1).expand(-1, W, -1)          # [B, W, STATE_DIM]
        x = torch.cat([ctx, option_b], dim=-1)                # [B, W, STATE_DIM+OPTION_DIM]
        logits = self.policy_scorer(x).squeeze(-1)            # [B, W]
        # Finite large-negative (NOT -inf) so masked entropy stays NaN-safe (cf. NASHPG -1e9).
        logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), self.value_head(state_b).squeeze(-1), dist.entropy()


# ============================================================================================
# ROLLOUT BUFFER — fresh; parallel lists like the reference, extended for the variable action
# space (per-step option_feats are ragged → padded at collate time).
# ============================================================================================
class RolloutBuffer:
    def __init__(self):
        self.states = []         # list[Tensor[STATE_DIM]]
        self.option_feats = []   # list[Tensor[n_i, OPTION_DIM]]
        self.actions = []        # list[int]
        self.logprobs = []       # list[float]  (old_logp, captured at act time)
        self.state_values = []   # list[float]  (V_old, captured at act time)
        self.rewards = []        # list[float]  (0 except a terminal G — written at game end)
        self.is_terminals = []   # list[bool]   (True only on a game's last learned decision)

    def add(self, state_feat, option_feats, action, logp, value):
        self.states.append(state_feat)
        self.option_feats.append(option_feats)
        self.actions.append(action)
        self.logprobs.append(logp)
        self.state_values.append(value)
        self.rewards.append(0.0)         # patched to G on the terminal step (collect_games)
        self.is_terminals.append(False)

    def __len__(self):
        return len(self.actions)

    def clear(self):
        for lst in (self.states, self.option_feats, self.actions, self.logprobs,
                    self.state_values, self.rewards, self.is_terminals):
            del lst[:]


# ============================================================================================
# PPO — fresh; mirrors the reference's PPO class (select_action / update / save / load).
# ============================================================================================
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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer() # TODO: is this where we could eventually add GAE as an alternative? Would that be an isolated change?

    def select_action(self, obs_dict):
        """Drive ONE decision. Returns (selection_list, recorded: bool). Single-select policy
        nodes are sampled on-policy and recorded; everything else is auto-resolved with NO
        record (so it's never trained — the v1 coverage limit)."""
        o = to_observation_class(obs_dict)
        sel = o.select
        # TODO: what is this?
        if not is_policy_select(sel):
            return auto_resolve(sel), False
        with torch.no_grad():
            out = self.policy.act(o)
        self.buffer.add(out.state_feat, out.option_feats, out.action,
                        float(out.logprob), float(out.value))
        return [out.action], True

    def _collate(self):
        """Buffer's ragged per-step option_feats → a padded batch + validity mask (concrete).
        Returns (states[B,S], option_b[B,W,O], mask[B,W], actions[B], old_logp[B], old_v[B])."""
        states = torch.stack(self.buffer.states).to(self.device)          # [B, STATE_DIM]
        n_list = [of.shape[0] for of in self.buffer.option_feats]
        B, W = len(n_list), max(n_list)
        option_b = states.new_zeros(B, W, OPTION_DIM)
        mask = torch.zeros(B, W, dtype=torch.bool, device=self.device)
        for i, of in enumerate(self.buffer.option_feats):
            n = of.shape[0]
            option_b[i, :n] = of.to(self.device)
            mask[i, :n] = True
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logp = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        old_v = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=self.device)
        return states, option_b, mask, actions, old_logp, old_v

    def update(self):
        """STUB → dict of diagnostics. The core PPO math — implemented in step 3 of the roadmap.

        Intended body (reference: PPO-PyTorch/PPO.py:200-251), all on the collated buffer:

          1. MONTE-CARLO discounted returns, computed backward, reset at episode boundaries:
                 R = 0
                 for (r, done) in reversed(zip(rewards, is_terminals)):
                     if done: R = 0
                     R = r + gamma * R
                     returns.insert(0, R)
             (With our terminal-only ±1 reward and gamma=1 this is just G per step; gamma<1
              discounts toward the decisions nearer the result.)
          2. returns = tensor(returns); normalize: (returns - mean) / (std + 1e-7).
          3. advantages = returns - V_old   (the stored act-time values; Monte-Carlo baseline).
          4. states, option_b, mask, actions, old_logp, old_v = self._collate().
          5. for _ in range(k_epochs):
                 logp, V, entropy = policy.evaluate(states, option_b, mask, actions)
                 ratio = exp(logp - old_logp)
                 surr1 = ratio * advantages
                 surr2 = clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
                 loss = -min(surr1, surr2).mean()
                        + vf_coef * MSE(V, returns)
                        - ent_coef * entropy.mean()
                 optimizer.zero_grad(); loss.backward()
                 clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
          6. self.buffer.clear()
          7. return diagnostics (loss parts, approx_kl, clip_frac, explained_var) for logging.

        NOTE vs reference: no separate `policy_old` net — we stored old_logp/V_old at act time
        (load-bearing: recomputing them here would make ratio≡1 and PPO degenerate). The
        reference's `policy_old` copy is an equivalent way to get the same baseline.
        """
        raise NotImplementedError("PPO.update: clipped-surrogate math is step 3 of the roadmap")

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
