#!/usr/bin/env python3
"""Train MaskablePPO (sb3-contrib) on the cg field via sb3_env.CGEnv — the SB3 experiment driver.

SB3 owns the whole PPO algorithm (rollout buffer, GAE, clipped update, logging); this file is only
argparse + env construction + save. Stock "MultiInputPolicy" — ACCEPTED TRADE-OFF (flagged for
theory review): its action head is a positional Linear(latent, N_MAX), so logit i means "option
slot i" whose meaning shifts every decision; per-option features are only visible through the
flattened obs. A custom maskable per-option scorer on the same env is the documented upgrade path
(plan.md "Deferred levers").

Hyperparameters: gamma=1.0 (undiscounted terminal-only reward — THEORY-REVIEW flag), gae_lambda=0.95,
clip 0.2, 4 epochs, ent 0.01, vf 0.5, lr 3e-4. n_steps=512 with batch_size=128 keeps the 4-minibatch
shape; SB3 counts learner DECISIONS (~15-30/game), not games, so 512 steps ≈ 20-30 games per update.

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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# sb3_env owns the CG_LIB_PATH import-order discipline (it imports tcg_model first).
from sb3_env import CGEnv, N_MAX                                   # noqa: E402
from utils.run_smoketest_battle import load_deck                   # noqa: E402
from sb3_contrib import MaskablePPO                                # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402


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
        "MultiInputPolicy", env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=4, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, seed=args.seed, verbose=1,
    )
    # Optional intermediate checkpoints for an offline learning curve (eval each with eval_sb3.py).
    # save_freq counts env steps == learner decisions here (one non-vec env); files land at
    # out/{prefix}_ckpt_{steps}_steps.zip. Training-side only — outside the submission boundary.
    callback = None
    if args.checkpoint_freq > 0:
        callback = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.out_dir,
                                      name_prefix=f"{args.out_prefix}_ckpt")

    print(f"MaskablePPO train | opponent={args.opponent} | decks={deck_names} | "
          f"n_steps={args.n_steps} batch={args.batch_size} | total_timesteps={args.total_timesteps} | "
          f"ckpt_freq={args.checkpoint_freq} | "
          f"N_MAX={N_MAX} | device={model.device} | -> {zip_path}", flush=True)

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)   # masking picked up from env.action_masks()
        model.save(zip_path)
    finally:
        env.close()     # SB3 rollouts cut mid-episode: pay the battle_finish (singleton discipline)

    assert os.path.exists(zip_path), f"save failed: {zip_path}"
    if env.overflows:
        print(f"[WARN] {env.overflows} policy nodes exceeded N_MAX={N_MAX} and were auto-resolved "
              f"— measure and raise N_MAX", flush=True)
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
                    help="opponent spec (sb3_env.make_opponent): comma mix over {rulebased,random}, "
                         "optional @eps suffix (e.g. 'rulebased,rulebased@0.25'); self/frozen "
                         "unsupported here (deferred — plan.md Feature 1)")
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
