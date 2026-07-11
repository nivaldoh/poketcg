#!/usr/bin/env python3
"""TcgCore — the id-aware per-option-scorer network (plain torch, NO stable-baselines3 imports).

Plain torch on purpose: `bc_train.py` trains this exact module offline (no SB3 in the loop), and
`train_sb3.py --bc-init` then loads it into the SB3 policy as the named submodule
`TcgMaskablePolicy.core` — one weights file, no state_dict prefix surgery. It consumes the gym obs
dict `sb3_env.obs_to_gym` emits (float dashboard/menu + int64 card/attack id keys) and returns
`(logits[B, N_MAX], value[B, 1])`; MASKING IS THE CALLER'S JOB (the SB3 distribution or the BC
masked cross-entropy — pad/illegal rows get garbage logits here by design).

Shape story (embed every id -> describe every option -> summarize the board -> score each option
against the summary):
  - one shared `card_emb` row per card id (board slots, option primary/target, hand, discards) and
    one `attack_emb` row per attack id — a learned "cheat-sheet" per card beyond the ~26 dense
    stats. Tables are ZERO-INITIALIZED: only ~40-60 cards appear in the 4-deck world, so unseen
    rows contribute nothing (they read like PAD) instead of frozen random noise; `padding_idx`
    keeps the PAD rows exact zeros so masked-mean pooling is just sum/count.
  - `opt_latents[B,N,h]` = MLP over each option row's dense features ++ its id embeddings.
  - `ctx[B,d_ctx]` = MLP over: state[303] ++ the 12 board-slot embeddings CONCATENATED (fixed
    positional slots — pooling would erase WHOSE card is where; deliberate deviation from
    docs/plans/plan.md §3.1, flagged there) ++ pooled hand/my-discard/opp-discard embeddings ++ masked-mean of
    opt_latents over selectable rows (the menu) and over already-picked rows (options col
    OPTION_DIM+1) — the autoregressive multi-select conditioning.
  - `logits[:, i] = MLP_score(ctx ++ opt_latents[:, i])` shared across i (the per-option scorer:
    "what this card is" drives "how good this pick is", slot-position-free);
    `value = MLP_val(ctx)`, final layer plain linear (PPO value semantics; BC target in {-1,0,+1}).

Init: torch defaults, except the final score layer scaled x0.01 (near-uniform initial policy —
PPO stability) and the zero-init embedding tables.

ARCHITECTURE PARAMS (d_card/d_atk/h/d_ctx) are ctor-required — no defaults. They must flow through
`policy_kwargs["core_kwargs"]` (train_sb3) / the bc_core.pt checkpoint (bc_train), never a
module-level default someone can silently change under an existing checkpoint. NEVER rename/move
this file once checkpoints exist: SB3 zips pickle `sb3_policy.TcgMaskablePolicy` by module path and
that class imports this one.

Self-test (shapes, grads reach both tables, permutation + pad-leak probes, real-env batch):
    python tcg_net.py
"""

import os
import sys

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Vocab sizes + pad sentinels + obs widths come from the featurization ground truth (tcg_model /
# sb3_features / sb3_env), so this module can't drift from what obs_to_gym emits.
from tcg_model import card_count, attack_count                          # noqa: E402
from sb3_features import (CARD_PAD, ATTACK_PAD, N_STATE_SLOTS,          # noqa: E402
                          N_OPT_CARD_SLOTS, N_OPT_ATTACK_SLOTS)
from sb3_env import N_MAX, STATE_DIM_LOCAL, OPTION_DIM_LOCAL            # noqa: E402

_PICKED_COL = OPTION_DIM_LOCAL - 1        # options col 71 = already_picked (sb3_env.obs_to_gym)

# THE canonical architecture, shared by train_sb3 (via policy_kwargs, stored in every zip) and
# bc_train (stored in bc_core.pt). NOT a ctor default — TcgCore still requires explicit kwargs;
# every load boundary (--init-from, --bc-init, MaskablePPO.load) asserts stored == current, so
# changing this starts a new checkpoint lineage loudly, never silently.
CORE_KWARGS = dict(d_card=32, d_atk=16, h=128, d_ctx=256)


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


def _pool_ids(emb, ids, pad):
    """Masked-mean embedding of an id zone [B,K] -> [B,d]: PAD rows are exact zeros (padding_idx),
    so mean = sum / #non-pad (clamped; an empty zone pools to zeros)."""
    count = (ids != pad).sum(dim=1, keepdim=True).clamp(min=1)
    return emb(ids).sum(dim=1) / count


def _masked_mean(x, m):
    """[B,N,h] mean over rows where m[B,N]==1 (0 rows -> zeros). Excluded rows contribute exactly
    0 (0*x), so perturbing them cannot leak into the pool."""
    m = m.unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class TcgCore(nn.Module):
    """See module docstring. forward(obs: dict[str, Tensor]) -> (logits[B,N_MAX], value[B,1])."""

    def __init__(self, d_card, d_atk, h, d_ctx):
        super().__init__()
        self.core_kwargs = dict(d_card=d_card, d_atk=d_atk, h=h, d_ctx=d_ctx)
        self.card_emb = nn.Embedding(card_count + 1, d_card, padding_idx=CARD_PAD)
        self.attack_emb = nn.Embedding(attack_count + 1, d_atk, padding_idx=ATTACK_PAD)
        nn.init.zeros_(self.card_emb.weight)          # zero-init: unseen cards read like PAD (see docstring)
        nn.init.zeros_(self.attack_emb.weight)

        opt_in = OPTION_DIM_LOCAL + N_OPT_CARD_SLOTS * d_card + N_OPT_ATTACK_SLOTS * d_atk
        ctx_in = (STATE_DIM_LOCAL                     # the dashboard
                  + N_STATE_SLOTS * d_card            # 12 board-slot embeddings, concatenated
                  + 3 * d_card                        # pooled hand + my/opp discard embeddings
                  + 2 * h)                            # pooled menu + pooled already-picked latents
        self.mlp_opt = _mlp(opt_in, h, h)
        self.mlp_ctx = _mlp(ctx_in, d_ctx, d_ctx)
        self.mlp_score = _mlp(d_ctx + h, h, 1)
        self.mlp_val = _mlp(d_ctx, h, 1)
        with torch.no_grad():                         # near-uniform initial policy (PPO stability)
            self.mlp_score[-1].weight.mul_(0.01)
            self.mlp_score[-1].bias.zero_()

    def forward(self, obs):
        state = obs["state"].float()                          # [B,303]
        options = obs["options"].float()                      # [B,128,72]
        sel_mask = obs["action_mask"].float()                 # [B,128] (i8/bool -> float weights)
        cids = obs["opt_card_ids"].long()                     # [B,128,2]  (.long(): safety net —
        aids = obs["opt_attack_ids"].long()                   #  int64 end-to-end is verified)
        opt_x = torch.cat([options,
                           self.card_emb(cids).flatten(-2),
                           self.attack_emb(aids).flatten(-2)], dim=-1)
        lat = self.mlp_opt(opt_x)                             # [B,128,h] per-option latents

        picked_mask = options[..., _PICKED_COL]               # already_picked bit (col 71)
        ctx = self.mlp_ctx(torch.cat([
            state,
            self.card_emb(obs["state_card_ids"].long()).flatten(1),
            _pool_ids(self.card_emb, obs["hand_ids"].long(), CARD_PAD),
            _pool_ids(self.card_emb, obs["my_discard_ids"].long(), CARD_PAD),
            _pool_ids(self.card_emb, obs["opp_discard_ids"].long(), CARD_PAD),
            _masked_mean(lat, sel_mask),                      # what's on the menu
            _masked_mean(lat, picked_mask),                   # what I already picked (AR conditioning)
        ], dim=1))                                            # [B,d_ctx]

        ctx_rows = ctx.unsqueeze(1).expand(-1, lat.shape[1], -1)
        logits = self.mlp_score(torch.cat([ctx_rows, lat], dim=-1)).squeeze(-1)   # [B,128]
        return logits, self.mlp_val(ctx)                      # value [B,1]


if __name__ == "__main__":
    # Self-test: shapes/finiteness, grads reach BOTH tables, permutation + pad-leak probes on a
    # synthetic batch, then a real CGEnv obs through the exact np->torch collation the SB3/BC
    # consumers use.
    torch.manual_seed(0)
    kw = dict(d_card=32, d_atk=16, h=128, d_ctx=256)
    core = TcgCore(**kw)
    n_params = sum(p.numel() for p in core.parameters())

    def synth(B, n_real=6, n_picked=1, seed=1):
        """A structurally-valid random batch: rows [0,n_real) real (random dense + real ids), the
        rest PAD; first n_picked real rows flagged picked (masked out), STOP row selectable."""
        g = torch.Generator().manual_seed(seed)
        obs = {"state": torch.randn(B, STATE_DIM_LOCAL, generator=g),
               "options": torch.zeros(B, N_MAX, OPTION_DIM_LOCAL),
               "action_mask": torch.zeros(B, N_MAX, dtype=torch.int8),
               "state_card_ids": torch.randint(0, card_count, (B, N_STATE_SLOTS), generator=g),
               "opt_card_ids": torch.full((B, N_MAX, N_OPT_CARD_SLOTS), CARD_PAD),
               "opt_attack_ids": torch.full((B, N_MAX, N_OPT_ATTACK_SLOTS), ATTACK_PAD),
               "hand_ids": torch.randint(0, card_count, (B, 20), generator=g),
               "my_discard_ids": torch.full((B, 60), CARD_PAD),
               "opp_discard_ids": torch.full((B, 60), CARD_PAD)}
        obs["options"][:, :n_real] = torch.randn(B, n_real, OPTION_DIM_LOCAL, generator=g)
        obs["options"][:, :, _PICKED_COL] = 0.0
        obs["options"][:, :n_picked, _PICKED_COL] = 1.0
        obs["opt_card_ids"][:, :n_real] = torch.randint(0, card_count, (B, n_real, 2), generator=g)
        obs["opt_attack_ids"][:, :n_real] = torch.randint(0, attack_count, (B, n_real, 1), generator=g)
        obs["action_mask"][:, n_picked:n_real] = 1
        obs["action_mask"][:, N_MAX - 1] = 1                  # STOP selectable
        return obs

    # 1) shapes + finiteness + grads reach BOTH embedding tables (the direct regression test
    # against "ids discarded": if ids were dropped, these grads would be zero).
    obs = synth(B=4)
    logits, value = core(obs)
    assert logits.shape == (4, N_MAX) and value.shape == (4, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (logits.sum() + value.sum()).backward()
    assert core.card_emb.weight.grad.abs().sum() > 0, "no gradient reached card_emb"
    assert core.attack_emb.weight.grad.abs().sum() > 0, "no gradient reached attack_emb"
    core.zero_grad()

    # 2) permutation probe: permute option rows + their ids + mask TOGETHER -> logits permute
    # identically, value unchanged (catches silent id/row misalignment structurally). allclose,
    # not equal: pooled sums legitimately reorder floating-point addition.
    obs = synth(B=2, n_real=7, n_picked=2, seed=2)
    with torch.no_grad():
        base_logits, base_value = core(obs)
        perm = torch.randperm(N_MAX)
        pobs = dict(obs)
        for k in ("options", "opt_card_ids", "opt_attack_ids"):
            pobs[k] = obs[k][:, perm]
        pobs["action_mask"] = obs["action_mask"][:, perm]
        perm_logits, perm_value = core(pobs)
    assert torch.allclose(perm_logits, base_logits[:, perm], atol=1e-5), "permutation probe FAILED"
    assert torch.allclose(perm_value, base_value, atol=1e-5), "permutation probe FAILED (value)"

    # 3) pad-leak probe: perturb dense features AND ids of rows that are neither selectable nor
    # picked -> logits at selectable+picked rows AND value must be EXACTLY unchanged (excluded rows
    # enter pools as 0*x; value is the likeliest leak).
    with torch.no_grad():
        keep = (obs["action_mask"].bool() | (obs["options"][..., _PICKED_COL] > 0))   # [B,128]
        dirty = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
        junk = ~keep
        dirty["options"][junk.unsqueeze(-1).expand_as(dirty["options"])] = 7.7
        dirty["options"][..., _PICKED_COL][junk] = 0.0        # keep the picked flags themselves
        dirty["opt_card_ids"][junk] = 3
        dirty["opt_attack_ids"][junk] = 3
        d_logits, d_value = core(dirty)
    assert torch.equal(d_logits[keep], base_logits[keep]), "pad-leak probe FAILED (logits)"
    assert torch.equal(d_value, base_value), "pad-leak probe FAILED (value)"

    # 4) a REAL env obs through the exact np->torch collation consumers use (dtype plumbing).
    from utils.run_smoketest_battle import load_deck
    from sb3_env import CGEnv
    env = CGEnv([("lucario", load_deck(os.path.join(_REPO_ROOT, "decks", "lucario.csv")))])
    gobs, _ = env.reset(seed=0)
    env.close()
    batch = {k: torch.as_tensor(v)[None] for k, v in gobs.items()}
    with torch.no_grad():
        logits, value = core(batch)
    assert logits.shape == (1, N_MAX) and torch.isfinite(logits).all() and torch.isfinite(value).all()

    print(f"OK: TcgCore self-test ({n_params:,} params; grads reach both tables; permutation + "
          f"pad-leak probes pass; real-env obs forward OK)")
