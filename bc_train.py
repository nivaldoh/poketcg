#!/usr/bin/env python3
"""bc_train — behavior-clone the rule-based teachers into a bare `tcg_net.TcgCore` (pure torch,
no stable-baselines3 anywhere in the loop).

Consumes the `bc_collect.py` shards (raw trimmed obs + live teacher picks + seat-relative game
outcome) and trains policy + value jointly: masked cross-entropy on the teacher's next pick
(supervised per-decision signal — no credit-assignment problem) + lambda_v * MSE(value, outcome)
(the critic warm-start; the label IS PPO's gamma=1.0 return-to-go, and the credit probe showed a
supervised fit at ev 0.167 vs the RL critic's 0.089). `train_sb3.py --bc-init out/bc_core.pt`
loads the result into `TcgMaskablePolicy.core` for PPO fine-tuning.

THE ONE-OBS-BUILDER INVARIANT: featurization here is `sb3_env.obs_to_gym` + `_ms_mask`, imported —
never re-implemented — so BC features are bit-identical to what CGEnv/eval build (§4 of docs/plans/plan.md;
the classic silent offline-IL killer, closed by construction).

Sample expansion (mirrors the env's sub-step decomposition EXACTLY — never train a state the env
can't visit): teacher picks are a semantically-unordered set, shuffled fresh at every access
(order augmentation ≈ the set objective); each prefix emits (obs_to_gym(o, prefix), next_pick),
plus a STOP example iff len(picks) < maxCount (the env auto-commits at maxCount — no STOP there).
Masked slots get -1e8 in the CE, mirroring MaskableCategorical's HUGE_NEG.

Split is BY GAME (game_id hash): decisions within a game share one outcome label, so the effective
sample count is games, not decisions (the credit probe's lesson). Metrics: holdout top-1 masked
accuracy vs the random-legal floor (mean 1/|legal|), split single-select vs multi sub-steps, and
value ev_true (raw, on first-sub-step examples — the per-decision convention the 0.167 bar used).

RAM (7GB box): rows are kept as their pickle BYTES (~4KB each) and unpickled+featurized lazily per
batch; --frac-games subsamples games deterministically (uniform across pairing shards).

Smoke (tiny slice overfit + featurization throughput):  python bc_train.py --data <dir> --smoke
Real:  python bc_train.py --data out/bc_data --out out/bc_core.pt
"""

import argparse
import glob
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

from sb3_env import obs_to_gym, _ms_mask, STOP_INDEX               # noqa: E402  (ONE obs builder)
from tcg_net import TcgCore, CORE_KWARGS                           # noqa: E402
from cg.api import to_observation_class                            # noqa: E402

_OBS_KEYS = ("state", "options", "action_mask", "state_card_ids", "opt_card_ids",
             "opt_attack_ids", "hand_ids", "my_discard_ids", "opp_discard_ids")


def _hash01(gid, salt):
    """Deterministic uniform-ish [0,1) per game id (stable across runs; uniform across the
    per-pairing contiguous id blocks, unlike a plain prefix cut)."""
    return ((gid * 2654435761 + salt * 40503) % (2 ** 32)) / 2 ** 32


def load_row_bytes(data_dir, frac_games, holdout_frac, seed):
    """-> (train_rows, holdout_rows) as lists of pickled-row bytes (compact; unpickled lazily)."""
    train, hold = [], []
    shards = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    assert shards, f"no shards in {data_dir}"
    for path in shards:
        with open(path, "rb") as f:
            for row in pickle.load(f):
                gid = row["game_id"]
                if frac_games < 1.0 and _hash01(gid, seed + 2) >= frac_games:
                    continue
                blob = pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)
                (hold if _hash01(gid, seed + 1) < holdout_frac else train).append(blob)
    return train, hold


class BCDataset(Dataset):
    """Map-style over ROWS; __getitem__ expands one row into its sub-step example list (fresh
    pick-order shuffle every access = per-epoch order augmentation). The collate_fn flattens."""

    def __init__(self, row_blobs, augment):
        self.blobs = row_blobs
        self.augment = augment          # False for holdout: fixed order = stable metrics

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        row = pickle.loads(self.blobs[i])
        o = to_observation_class(row["obs"])
        sel = o.select
        picks = list(row["picks"])
        if self.augment and len(picks) > 1:
            random.shuffle(picks)       # teacher picks are a set; order is our augmentation axis
        is_multi = not (row["min_count"] == 1 and row["max_count"] == 1)
        out = []

        def emit(picked_prefix, target, first):
            gobs = obs_to_gym(o, picked_prefix)
            mask = gobs["action_mask"]
            assert mask[target], f"target {target} illegal (game {row['game_id']})"
            out.append((gobs, target, row["outcome"], is_multi, first, int(mask.sum())))

        for t in range(len(picks)):
            emit(picks[:t], picks[t], t == 0)
        if len(picks) < sel.maxCount:   # env would emit an explicit STOP (never at maxCount)
            emit(picks, STOP_INDEX, len(picks) == 0)
        return out


def collate(row_lists):
    ex = [e for lst in row_lists for e in lst]
    batch = {k: torch.as_tensor(np.stack([e[0][k] for e in ex])) for k in _OBS_KEYS}
    return (batch,
            torch.as_tensor([e[1] for e in ex], dtype=torch.long),
            torch.as_tensor([e[2] for e in ex], dtype=torch.float32),
            torch.as_tensor([e[3] for e in ex], dtype=torch.bool),
            torch.as_tensor([e[4] for e in ex], dtype=torch.bool),
            torch.as_tensor([e[5] for e in ex], dtype=torch.float32))


def _masked_logits(core, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    logits, value = core(batch)
    return logits.masked_fill(batch["action_mask"] == 0, -1e8), value.squeeze(-1)


@torch.no_grad()
def evaluate(core, loader, device):
    """Holdout metrics: masked top-1 accuracy (overall/single/multi), random-legal floor, and
    value ev_true on first-sub-step examples (per-decision, the credit-probe convention)."""
    core.eval()
    n = hit = n_s = hit_s = n_m = hit_m = 0
    floor_sum = 0.0
    y, pred = [], []
    for batch, tgt, outcome, is_multi, first, n_legal in loader:
        ml, val = _masked_logits(core, batch, device)
        top1 = ml.argmax(dim=1).cpu()
        ok = (top1 == tgt)
        n += len(tgt); hit += int(ok.sum())
        n_s += int((~is_multi).sum()); hit_s += int(ok[~is_multi].sum())
        n_m += int(is_multi.sum()); hit_m += int(ok[is_multi].sum())
        floor_sum += float((1.0 / n_legal).sum())
        y.append(outcome[first]); pred.append(val.cpu()[first])
    y = torch.cat(y).numpy(); pred = torch.cat(pred).numpy()
    ev = float(1.0 - np.var(y - pred) / max(np.var(y), 1e-9))
    core.train()
    return {"acc": hit / max(n, 1), "acc_single": hit_s / max(n_s, 1),
            "acc_multi": hit_m / max(n_m, 1), "floor": floor_sum / max(n, 1),
            "ev": ev, "n_examples": n, "n_single": n_s, "n_multi": n_m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(_REPO_ROOT, "out", "bc_data"))
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "out", "bc_core.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=256,
                    help="rows per batch (~1.2 sub-step examples each)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-v", type=float, default=0.5, help="value-MSE weight in the loss")
    ap.add_argument("--frac-games", type=float, default=1.0,
                    help="deterministic game subsample (learning-curve reads, RAM relief)")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader featurization workers. Default 0 on purpose: fork workers "
                         "gradually COPY the in-RAM blob store (python refcounts touch COW pages) "
                         "— on the 7GB box that is an OOM, not a speedup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny-slice overfit (loss plumbing) + featurization throughput report")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    if args.smoke:
        # small batches + hot lr: the point is "can this loss memorize a tiny slice", so give it
        # enough optimizer steps to actually do it (batch 256 would mean ~7 updates/epoch)
        args.epochs, args.workers, args.batch_rows, args.lr = 80, 0, 64, 1e-3
    train_blobs, hold_blobs = load_row_bytes(args.data, args.frac_games, args.holdout_frac,
                                             args.seed)
    if args.smoke:
        train_blobs = train_blobs[:1500]        # tiny fixed slice — overfit = loss-plumbing check
    assert train_blobs and hold_blobs, (len(train_blobs), len(hold_blobs))
    ram_mb = sum(map(len, train_blobs + hold_blobs)) / 1e6
    print(f"rows: train={len(train_blobs)} holdout={len(hold_blobs)} "
          f"(~{ram_mb:.0f} MB as blobs) | device={device} | load {time.time() - t0:.0f}s",
          flush=True)

    train_ds = BCDataset(train_blobs, augment=True)
    hold_ds = BCDataset(hold_blobs, augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_rows, shuffle=True,
                          num_workers=args.workers, collate_fn=collate)
    hold_ld = DataLoader(hold_ds, batch_size=args.batch_rows, shuffle=False,
                         num_workers=args.workers, collate_fn=collate)

    core = TcgCore(**CORE_KWARGS).to(device)
    opt = torch.optim.Adam(core.parameters(), lr=args.lr)
    best = {"acc": -1.0}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        seen = ce_sum = mse_sum = 0
        for batch, tgt, outcome, _m, _f, _n in train_ld:
            ml, val = _masked_logits(core, batch, device)
            ce = F.cross_entropy(ml, tgt.to(device))
            mse = F.mse_loss(val, outcome.to(device))
            opt.zero_grad()
            (ce + args.lambda_v * mse).backward()
            opt.step()
            seen += len(tgt)
            ce_sum += ce.detach().item() * len(tgt)
            mse_sum += mse.detach().item() * len(tgt)
        m = evaluate(core, hold_ld, device)
        keep = m["acc"] > best["acc"]
        if keep:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        print(f"epoch {epoch}/{args.epochs}: train ce={ce_sum / seen:.4f} mse={mse_sum / seen:.4f} "
              f"({seen} ex, {seen / (time.time() - te):.0f} ex/s) | holdout acc={m['acc']:.3f} "
              f"(single={m['acc_single']:.3f} multi={m['acc_multi']:.3f}, floor={m['floor']:.3f}) "
              f"ev={m['ev']:.3f}{'  *best*' if keep else ''}", flush=True)

    if args.smoke:
        # Loss-plumbing gate on SINGLE-SELECT train accuracy: single-select has a unique target,
        # so it must be memorizable (identical-option argmax ties cap it at ~99%, measured).
        # Multi sub-steps are NOT gated: the pick-order shuffle deliberately trains the SET, so
        # the recorded first-pick order is a coin-flip under top-1 — a metric artifact, not a bug.
        tm = evaluate(core, DataLoader(BCDataset(train_blobs, augment=False),
                                       batch_size=args.batch_rows, collate_fn=collate), device)
        assert tm["acc_single"] > 0.9, \
            f"smoke overfit failed: single-select train acc {tm['acc_single']:.3f} <= 0.9"
        print(f"OK: --smoke overfit single-select train acc={tm['acc_single']:.3f} (> 0.9; "
              f"overall={tm['acc']:.3f}, multi={tm['acc_multi']:.3f}); no checkpoint written")
        return 0

    torch.save({"state_dict": best_state, "core_kwargs": dict(CORE_KWARGS), "metrics": best,
                "train_rows": len(train_blobs), "holdout_rows": len(hold_blobs),
                "args": vars(args)}, args.out)
    print(f"OK: saved best-by-holdout-acc core -> {args.out} | acc={best['acc']:.3f} "
          f"(single={best['acc_single']:.3f} multi={best['acc_multi']:.3f}, "
          f"floor={best['floor']:.3f}) ev={best['ev']:.3f} vs supervised bar 0.167 | "
          f"{time.time() - t0:.0f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
