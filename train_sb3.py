#!/usr/bin/env python3
"""Train MaskablePPO (sb3-contrib) on the cg field via sb3_env.CGEnv — the SB3 experiment driver.

SB3 owns the whole PPO algorithm (rollout buffer, GAE, clipped update, logging); this file is only
argparse + env construction + save. Policy = sb3_policy.TcgMaskablePolicy (tcg_net.TcgCore): the
id-embedding per-option scorer — each option row is scored by one shared MLP against the board
context, so "what this card is" (learned card/attack embeddings) drives "how good this pick is".
This REPLACED the stock MultiInputPolicy outright (docs/plans/plan.md: the positional head + id-blindness
were two of the three measured plateau causes; the stock policy can't honestly consume the int64
id obs keys — preprocess_obs would float-cast them). Architecture params live in CORE_KWARGS below
and flow through policy_kwargs into every zip (never module-level defaults in the net; changing
CORE_KWARGS orphans --init-from/--bc-init sources, which assert loudly).

Hyperparameters: gamma=1.0 (undiscounted terminal-only reward — THEORY-REVIEW flag), gae_lambda=0.95,
clip 0.2, 4 epochs, ent 0.01, vf 0.5, lr 3e-4. n_steps=512 with batch_size=128 keeps the 4-minibatch
shape; SB3 counts learner DECISIONS (~15-30/game), not games, so 512 steps ≈ 20-30 games per update.

Multi-select is live (all shapes): CGEnv decomposes each optional-single / fixed-k / true-variable /
single-option node into masked single-select sub-steps (a k-pick node = k learner decisions), so
n_steps now counts sub-steps and games-per-update drops slightly. gamma=1.0 is now LOAD-BEARING (not
merely a flag): under undiscounted terminal-only reward every sub-step of a node shares the node's
return-to-go — a gamma<1 would discount a k-pick node's reward unevenly across its sub-steps and
distort intra-node credit. CHECKPOINT BREAK (2026-07-09, the id-obs + custom-policy change): the
obs gained the int64 id keys and the policy state_dict is now core.* — every earlier out/*.zip
(multi-select-era included) is dead; retrain from scratch under a fresh --out-prefix. --init-from
rejects old zips loudly (obs-space assert + strict load_state_dict).

Outputs (out/, gitignored): {prefix}_sb3.zip (full SB3 save — resume/eval via MaskablePPO.load)
— training-side artifacts only, OUTSIDE the submission boundary (a deployable agent needs the
deferred SB3-free torch mirror; see the plan).

Smallest build check:      python train_sb3.py --help
Smallest training check:   python train_sb3.py --smoke   (~4 updates, <2 min)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# sb3_env owns the CG_LIB_PATH import-order discipline (it imports tcg_model first).
from sb3_env import CGEnv, N_MAX, sb3_agent                        # noqa: E402
from sb3_policy import TcgMaskablePolicy                           # noqa: E402
from tcg_net import CORE_KWARGS                                    # noqa: E402  (THE canonical arch)
from utils.run_smoketest_battle import load_deck                   # noqa: E402
from sb3_contrib import MaskablePPO                                # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402


def _refresh_selfplay_snapshot(model, env, path):
    """Freeze the current policy as the self-play opponent: save -> reload -> install. Loading from
    the zip guarantees the on-disk artifact IS the live opponent (and the roundtrip is the
    battle-tested SB3 serialization; ~1-2s, ~15MB)."""
    model.save(path)
    frozen = MaskablePPO.load(path, device=model.device)
    env.set_policy_opponent(sb3_agent(frozen, greedy=False))
    print(f"[selfplay-refresh] snapshot @{model.num_timesteps} -> {path}", flush=True)


class SelfPlayRefreshCallback(BaseCallback):
    """Re-freeze the self-play opponent every `freq` env steps. Takes effect from the NEXT episode —
    CGEnv._start_battle latches the opponent fn per game, so a game in flight is never switched."""

    def __init__(self, env, path, freq):
        super().__init__()
        self._env, self._path, self._freq = env, path, freq

    def _on_step(self):
        if self.n_calls % self._freq == 0:
            _refresh_selfplay_snapshot(self.model, self._env, self._path)
        return True


def build_env(args):
    deck_names = [s.strip() for s in args.decks.split(",") if s.strip()]
    decks = [(n, load_deck(os.path.join(_REPO_ROOT, "decks", f"{n}.csv"))) for n in deck_names]
    return CGEnv(decks, opponent_kind=args.opponent), deck_names


def train(args) -> int:
    if args.smoke:
        args.n_steps, args.batch_size, args.total_timesteps = 256, 64, 1024
    env, deck_names = build_env(args)
    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = os.path.join(args.out_dir, f"{args.out_prefix}_sb3.zip")

    model = MaskablePPO(
        TcgMaskablePolicy, env, policy_kwargs={"core_kwargs": CORE_KWARGS},
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=4, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
        ent_coef=args.ent_coef, vf_coef=0.5, target_kl=args.target_kl, seed=args.seed, verbose=1,
    )
    # BC warm-start: load the offline-trained bare TcgCore into the policy's named submodule
    # (this is WHY bc_train trains the bare core — no state_dict prefix surgery), then prove the
    # SB3 wrapper computes the identical function: one fixed obs through bc_train's collation and
    # through policy.obs_to_tensor must give bit-identical logits/value (the consumer-mismatch
    # tripwire; both paths hit the same weights on the same device, so any difference is an obs
    # plumbing bug). `--bc-init X --total-timesteps 0` saves the eval-able pure-BC zip.
    if args.bc_init:
        if args.init_from:
            raise SystemExit("--bc-init and --init-from are mutually exclusive")
        ckpt = torch.load(args.bc_init, map_location=model.device, weights_only=False)
        assert ckpt["core_kwargs"] == CORE_KWARGS, (
            f"bc core arch {ckpt['core_kwargs']} != CORE_KWARGS {CORE_KWARGS} — "
            f"retrain bc_core or align tcg_net.CORE_KWARGS")
        model.policy.core.load_state_dict(ckpt["state_dict"])
        gobs, _ = env.reset(seed=args.seed)
        with torch.no_grad():
            bc_batch = {k: torch.as_tensor(np.stack([v])).to(model.device) for k, v in gobs.items()}
            l_bc, v_bc = model.policy.core(bc_batch)
            obs_t, _ = model.policy.obs_to_tensor(gobs)
            l_sb3, v_sb3 = model.policy.core(obs_t)
        assert torch.equal(l_bc, l_sb3) and torch.equal(v_bc, v_sb3), \
            "BC->SB3 logit-identity FAILED: obs plumbing differs between bc_train and SB3"
        print(f"[bc-init] core <- {args.bc_init} (holdout metrics at save: "
              f"{ckpt.get('metrics')}) | logit-identity OK", flush=True)
    # Warm start: policy weights ONLY (fresh Adam, fresh lr schedule, num_timesteps=0). Deliberately
    # NOT MaskablePPO.load(): load() restores every zip hyperparam not overridden plus the old
    # optimizer state — a warm leg should be exactly a fresh run with warm initial weights.
    if args.init_from:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(args.init_from, device=model.device)
        assert data["observation_space"] == env.observation_space, (
            f"--init-from obs-space mismatch vs {args.init_from} (N_MAX / obs width changed since "
            f"it was saved — pre-multi-select zips don't load; see module docstring)")
        model.policy.load_state_dict(params["policy"])
        print(f"[init-from] policy weights <- {args.init_from}", flush=True)

    # Optional intermediate checkpoints for an offline learning curve (eval each with eval_sb3.py).
    # save_freq counts env steps == learner decisions here (one non-vec env); files land at
    # out/{prefix}_ckpt_{steps}_steps.zip. Training-side only — outside the submission boundary.
    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.out_dir,
                                            name_prefix=f"{args.out_prefix}_ckpt"))

    # Self-play: any spec containing "self" (bare or mixed, e.g. "rulebased,self") puts a policy on
    # the opponent seat. The env was built before the model existed, so inject the fn now —
    # MaskablePPO wraps the env by reference, so this reaches the same object model.learn() steps.
    # Default = LIVE mirror (model.predict reads current weights, win-rate ~50% by construction on
    # the self draws); --selfplay-refresh N freezes a snapshot instead, re-saved every N env steps.
    # The t=0 install must happen HERE: learn() resets the env (_setup_learn) BEFORE
    # on_training_start, and a mixed spec's first game may draw "self" at that reset.
    if env._policy_opponent_spec is not None:
        if args.selfplay_refresh > 0:
            snap = os.path.join(args.out_dir, f"{args.out_prefix}_selfplay_opp.zip")
            _refresh_selfplay_snapshot(model, env, snap)
            callbacks.append(SelfPlayRefreshCallback(env, snap, args.selfplay_refresh))
        else:
            env.set_policy_opponent(sb3_agent(model, greedy=False))
            print("[self-play] installed policy opponent (opponent seat = current learner policy)",
                  flush=True)
        assert env._policy_opponent_fn is not None
    elif args.selfplay_refresh > 0:
        raise SystemExit(f"--selfplay-refresh requires 'self' in --opponent (got {args.opponent!r})")

    print(f"MaskablePPO train | opponent={args.opponent} | decks={deck_names} | "
          f"n_steps={args.n_steps} batch={args.batch_size} | total_timesteps={args.total_timesteps} | "
          f"ckpt_freq={args.checkpoint_freq} | "
          f"N_MAX={N_MAX} | device={model.device} | -> {zip_path}", flush=True)

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)   # masking picked up from env.action_masks()
        model.save(zip_path)
    finally:
        env.close()     # SB3 rollouts cut mid-episode: pay the battle_finish (singleton discipline)

    assert os.path.exists(zip_path), f"save failed: {zip_path}"
    if env.overflows:
        print(f"[WARN] {env.overflows} policy nodes exceeded N_MAX={N_MAX} and were auto-resolved "
              f"— measure and raise N_MAX", flush=True)
    if env.hand_overflows:
        print(f"[WARN] {env.hand_overflows} surfaced nodes had a hand wider than H_MAX "
              f"(hand_ids truncated) — consider raising sb3_features.H_MAX", flush=True)
    print(f"OK: learn() completed ({args.total_timesteps} decisions, {time.time() - t0:.0f}s) "
          f"-> {zip_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=512,
                    help="learner decisions per rollout/update (~15-30 per game)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="minibatch size (512/128 = 4 minibatches)")
    ap.add_argument("--total-timesteps", type=int, default=51200,
                    help="total learner decisions (default = 100 updates of 512)")
    ap.add_argument("--checkpoint-freq", type=int, default=0,
                    help="save an intermediate zip every N learner decisions (0 = off); files land "
                         "at out/{prefix}_ckpt_{steps}_steps.zip for an offline learning curve")
    ap.add_argument("--opponent", default="rulebased",
                    help="opponent spec (sb3_env.make_opponent): comma mix over {rulebased,random,"
                         "self}, optional @eps suffix (e.g. 'rulebased,rulebased@0.25' or "
                         "'rulebased,rulebased,self' for 2:1 field:self-play); 'self' entries put "
                         "the learner's own policy on the opponent seat (live, or frozen via "
                         "--selfplay-refresh)")
    ap.add_argument("--selfplay-refresh", type=int, default=0,
                    help="freeze the 'self' opponent to a snapshot (out/{prefix}_selfplay_opp.zip) "
                         "re-saved every N env steps (0 = live mirror of current weights)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start policy weights (only) from a prior train_sb3 zip; hyperparams, "
                         "optimizer state and timestep counters stay fresh (chained campaign legs)")
    ap.add_argument("--bc-init", default=None,
                    help="warm-start the policy core from a bc_train checkpoint (out/bc_core.pt); "
                         "mutually exclusive with --init-from. With --total-timesteps 0 this saves "
                         "the eval-able pure-BC zip (the G1 gate + anti-forgetting anchor)")
    ap.add_argument("--target-kl", type=float, default=None,
                    help="MaskablePPO native early-stop per update epoch loop (fine-tune legs; "
                         "default off, SB3 default)")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="entropy bonus (default = the stock recipe). Set 0 for BC fine-tune "
                         "legs: the bonus actively pushes a sharp clone back toward uniform")
    ap.add_argument("--decks", default="lucario,abomasnow,dragapult_ex,iono",
                    help="comma list of deck names in decks/<name>.csv the learner cycles through")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds python/np/torch + agent-side draws; the cg engine RNG stays unseedable")
    ap.add_argument("--out-prefix", default="sb3ppo")
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "out"))
    ap.add_argument("--smoke", action="store_true",
                    help="micro run (n_steps=256, batch=64, total=1024) asserting the loop + save work")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
