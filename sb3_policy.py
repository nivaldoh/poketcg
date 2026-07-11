#!/usr/bin/env python3
"""TcgMaskablePolicy — the thin MaskableActorCriticPolicy wrapper around `tcg_net.TcgCore`.

Replaces the stock MultiInputPolicy (positional action head, id-blind — the measured ~13% plateau
architecture). The whole override surface is the four compute methods + `_build`; everything else
(predict/_predict/obs_to_tensor/set_training_mode/save/load plumbing) is INHERITED — verified
against sb3_contrib 2.9.0 source: MaskablePPO touches the policy ONLY through {ctor, forward,
evaluate_actions, predict_values, predict, obs_to_tensor, set_training_mode, optimizer,
parameters(), to()}, and the inherited `_predict` routes through our `get_distribution`, which is
what keeps `sb3_agent`, eval_sb3, record_battle and the self-play snapshot round-trip working
unchanged. We keep Discrete(N_MAX) + the STOCK MaskableCategoricalDistribution (masked logits
become -1e8, never -inf), so masking semantics are bit-identical to the stock policy.

Notes pinned by verification (do not "fix" these):
  - The base __init__ builds a parameter-free FlattenExtractor before `_build` — unused, harmless.
  - Our `_build` must NOT call super()._build(): the base version constructs
    mlp_extractor/action_net/value_net and ortho-inits them BY NAME; we build only `self.core`
    so the state_dict and Adam stay clean.
  - Int64 id obs keys reach `core` intact: SB3's float-casting preprocess_obs lives behind
    extract_features, which nothing here calls.
  - NEVER rename/move this file or tcg_net.py once checkpoints exist — MaskablePPO zips pickle
    this class by module path. ALL architecture params flow via policy_kwargs["core_kwargs"]
    (stored in the zip, restored on load); core_kwargs is ctor-REQUIRED, no defaults.

Self-test (contract shapes incl. B=1, masked probs ~0, save->load logit identity, predict path):
    python sb3_policy.py
"""

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # noqa: E402
from tcg_net import TcgCore                                                 # noqa: E402


class TcgMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy whose entire torso is the named submodule `self.core` (a bare
    TcgCore — the same module bc_train.py trains offline; `--bc-init` loads it via
    `policy.core.load_state_dict`, no prefix surgery)."""

    def __init__(self, observation_space, action_space, lr_schedule, core_kwargs=None, **kwargs):
        assert core_kwargs, ("core_kwargs is required — pass MaskablePPO(..., policy_kwargs="
                             "{'core_kwargs': {...}}); no module-level architecture defaults")
        self._core_kwargs = dict(core_kwargs)   # stash BEFORE super().__init__ (it calls _build last)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        # Full replacement of the base _build (no mlp_extractor/action_net/value_net; the
        # action_dist was already created by the base __init__). Optimizer construction is stock.
        self.core = TcgCore(**self._core_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        # Only used by bare-policy save/load (BaseModel.save); MaskablePPO zips rebuild from the
        # algorithm's stored policy_kwargs instead. Included for completeness.
        data = super()._get_constructor_parameters()
        data.update(core_kwargs=self._core_kwargs)
        return data

    # -- the one compute path -------------------------------------------------------------
    def _distribution(self, logits, action_masks):
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def forward(self, obs, deterministic=False, action_masks=None):
        logits, values = self.core(obs)                       # [B,N_MAX], [B,1]
        dist = self._distribution(logits, action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values, dist.log_prob(actions)        # [B], [B,1], [B]

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self.core(obs)
        dist = self._distribution(logits, action_masks)
        return values, dist.log_prob(actions), dist.entropy()  # [B,1], [B], [B]

    def get_distribution(self, obs, action_masks=None):
        logits, _ = self.core(obs)
        return self._distribution(logits, action_masks)

    def predict_values(self, obs):
        _, values = self.core(obs)
        return values


if __name__ == "__main__":
    # Contract self-test on a real CGEnv obs: shapes/dtypes incl. B=1, masked probs ~0, the
    # model.predict path (what sb3_agent uses), a gradient step through evaluate_actions, and
    # save -> MaskablePPO.load -> bit-identical logits (exercises the pickle-by-module-path route).
    import numpy as np
    from sb3_contrib import MaskablePPO
    from sb3_env import CGEnv, N_MAX, _ms_mask
    from utils.run_smoketest_battle import load_deck

    kw = dict(d_card=8, d_atk=4, h=32, d_ctx=64)              # small = fast; sizes are not the contract
    env = CGEnv([("lucario", load_deck(os.path.join(_REPO_ROOT, "decks", "lucario.csv")))])
    # device pinned to cpu: the save->load check below asserts BIT-identical logits, which only
    # holds when both sides run the same kernels (cuda vs cpu differ in low-order bits).
    model = MaskablePPO(TcgMaskablePolicy, env, policy_kwargs={"core_kwargs": kw},
                        n_steps=64, batch_size=32, seed=0, verbose=0, device="cpu")
    policy = model.policy
    gobs, _ = env.reset(seed=0)
    mask = env.action_masks()
    sel = env._cur.select
    assert (mask == _ms_mask(sel, [])).all()

    obs_t, _ = policy.obs_to_tensor(gobs)                     # B=1 path
    assert obs_t["opt_card_ids"].dtype == torch.int64, "int ids were cast — dtype path broken"
    with torch.no_grad():
        dist = policy.get_distribution(obs_t, action_masks=mask)
        probs = dist.distribution.probs                       # record_battle's exact read
        values = policy.predict_values(obs_t)
        actions, v2, logp = policy(obs_t, deterministic=True, action_masks=mask)
    assert probs.shape == (1, N_MAX) and values.shape == (1, 1)
    assert actions.shape == (1,) and v2.shape == (1, 1) and logp.shape == (1,)
    assert probs[0, ~mask].max() < 1e-6, "masked slots must have ~0 probability"
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert mask[int(actions[0])], "sampled action outside the mask"

    # evaluate_actions: shapes + gradients flow to the embedding tables through the SB3 wrapper
    values3, logp3, ent = policy.evaluate_actions(obs_t, actions, action_masks=mask[None])
    assert values3.shape == (1, 1) and logp3.shape == (1,) and ent.shape == (1,)
    (values3.sum() + logp3.sum() + ent.sum()).backward()
    assert policy.core.card_emb.weight.grad is not None
    policy.optimizer.zero_grad()

    # model.predict — the sb3_agent/eval path (numpy dict in, masked int out)
    act, _ = model.predict(gobs, action_masks=mask, deterministic=True)
    assert mask[int(act)]

    # save -> load -> bit-identical logits (pickle path; loads WITHOUT env like eval_sb3 does)
    zpath = os.path.join("/tmp", "tcg_policy_selftest.zip")
    model.save(zpath)
    m2 = MaskablePPO.load(zpath, device="cpu")
    with torch.no_grad():
        l1, v1 = policy.core(obs_t)
        obs_t2, _ = m2.policy.obs_to_tensor(gobs)
        l2, v2b = m2.policy.core(obs_t2)
    assert torch.equal(l1.cpu(), l2.cpu()) and torch.equal(v1.cpu(), v2b.cpu()), \
        "save->load logits/value not bit-identical"
    assert m2.policy._core_kwargs == kw
    env.close()
    os.remove(zpath)
    print(f"OK: TcgMaskablePolicy contract self-test (B=1 shapes, masked probs<1e-6, predict, "
          f"grad step, save->load bit-identical; device={model.device})")
