#!/usr/bin/env python3
"""Held-out move-agreement: does a checkpoint's greedy policy match the rule-based teacher?

The LOW-VARIANCE progression/regression signal. Outcome win-rate can't separate near-identical
checkpoints -- the Lucario mirror is ~50/50, so a few-point skill edge needs ~800-5000 games to
resolve. This instead scores each checkpoint over THOUSANDS of fixed teacher decision points, so its
Wilson CI shrinks far faster and can tell near-identical checkpoints apart. Caveat: it measures
imitation fidelity to the teacher (an imperfect, non-optimal reference) and treats a tiny deviation
like a blunder -- read it alongside outcome eval (eval.py) and, later, a calibrated value model
(Stage 2) / puzzle suite (Stage 3).

The cg engine RNG is unseedable, so a fresh teacher-vs-teacher set differs every run. We therefore
BUILD the decision set once and FREEZE it to disk; every checkpoint is then scored on the identical,
held-out positions (these games are never used in training).

    TORCHDYNAMO_DISABLE=1 python eval_agreement.py \\
        --ckpts out/anchor_model0.pth,out/mymodel_lucario_bc_ce.pth
"""

import argparse
import os
import pickle
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

import tcg_model  # noqa: E402  (sets up the vendored cg path; loads card metadata)
from tcg_model import MyModel, eval_nn  # noqa: E402
from train_mymodel_bc import record_teacher_games  # noqa: E402  (reuse the teacher decision capture)
from eval import wilson, sha256, deck_hash  # noqa: E402


def build_or_load_set(set_path: str, deck: list[int], games: int, rebuild: bool) -> list:
    """The frozen held-out decision set = the teacher's single-select choices over `games` mirror
    games. Built once (the engine RNG is unseedable, so it can't be regenerated identically) and
    reused so every checkpoint is scored on the SAME positions."""
    if os.path.exists(set_path) and not rebuild:
        with open(set_path, "rb") as f:
            blob = pickle.load(f)
        m = blob["meta"]
        print(f"Loaded frozen agreement set: {set_path} ({len(blob['samples'])} decisions, "
              f"built from {m['games']} games, deck {m['deck_hash']})", flush=True)
        return blob["samples"]
    print(f"Building frozen agreement set ({games} teacher games)...", flush=True)
    samples = record_teacher_games(deck, games)
    samples = [s for s in samples if 1.0 in s.policy]   # keep well-formed single-select decisions
    blob = {"meta": {"games": games, "deck_hash": deck_hash(deck), "n_decisions": len(samples)},
            "samples": samples}
    os.makedirs(os.path.dirname(set_path), exist_ok=True)
    with open(set_path, "wb") as f:
        pickle.dump(blob, f)
    print(f"Froze {len(samples)} decisions -> {set_path} (held out of training).", flush=True)
    return samples


def agreement(model: MyModel, samples: list, topk: int) -> tuple[int, int, int]:
    """Top-1 and top-k agreement of the model's greedy policy with the teacher's recorded choice.
    Each sample's sv_dec has one decoder bag per legal option, so the policy vector is per-option."""
    top1 = topk_hits = 0
    with torch.inference_mode():
        for s in samples:
            teacher_idx = s.policy.index(1.0)
            _, policy = eval_nn(s.sv_enc, s.sv_dec, model)
            n_opt = len(policy)
            order = sorted(range(n_opt), key=lambda i: policy[i], reverse=True)
            if order[0] == teacher_idx:
                top1 += 1
            if teacher_idx in order[:min(topk, n_opt)]:
                topk_hits += 1
    return top1, topk_hits, len(samples)


def random_baseline(samples: list, topk: int) -> tuple[float, float]:
    """Analytic expected agreement of a uniform-random picker over the same decisions (a floor)."""
    t1 = sum(1.0 / len(s.policy) for s in samples) / len(samples)
    tk = sum(min(topk, len(s.policy)) / len(s.policy) for s in samples) / len(samples)
    return t1, tk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", required=True,
                    help="comma-separated checkpoint paths to score on the frozen set")
    ap.add_argument("--set", default=os.path.join(_REPO_ROOT, "out", "agreement_set.pkl"),
                    help="frozen held-out decision set (built on first use, then reused)")
    ap.add_argument("--games", type=int, default=200,
                    help="teacher games to record when BUILDING the set (ignored once it exists)")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--deck", default=os.path.join(_REPO_ROOT, "deck.csv"))
    ap.add_argument("--rebuild", action="store_true", help="force-rebuild the frozen set")
    args = ap.parse_args()

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)   # reproducible forward pass
    except Exception:
        pass

    deck = [int(x) for x in open(args.deck).read().split()]
    samples = build_or_load_set(args.set, deck, args.games, args.rebuild)
    if not samples:
        print("ERROR: no decisions in the set.")
        return 1
    rb1, rbk = random_baseline(samples, args.topk)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 78}")
    print(f"Move-agreement vs teacher | {len(samples)} held-out decisions | top-1 & top-{args.topk}")
    print(f"set={args.set} (sha {sha256(args.set)[:12]}) | random baseline: "
          f"top-1={100 * rb1:.1f}% top-{args.topk}={100 * rbk:.1f}%")
    print('=' * 78)
    for path in [p.strip() for p in args.ckpts.split(",") if p.strip()]:
        if not os.path.exists(path):
            print(f"  {os.path.basename(path):30s} MISSING -> skipped")
            continue
        model = MyModel(128, 2, 256, 1, 1).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        t1, tk, n = agreement(model, samples, args.topk)
        p1, lo1, hi1 = wilson(t1, n)
        pk, lok, hik = wilson(tk, n)
        print(f"  {os.path.basename(path):30s} "
              f"top-1={100 * p1:5.1f}% (CI {100 * lo1:.1f}-{100 * hi1:.1f})  "
              f"top-{args.topk}={100 * pk:5.1f}% (CI {100 * lok:.1f}-{100 * hik:.1f})  "
              f"[sha {sha256(path)[:12]}]")
    print("\nHigher = more teacher-like. Compare overlapping Wilson CIs, not point estimates. This is")
    print("imitation fidelity, not absolute skill (the teacher is not optimal) — read with eval.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
