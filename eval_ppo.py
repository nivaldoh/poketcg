#!/usr/bin/env python3
"""Eval a trained PPO ActorCritic vs the random opponent on the sample_deck mirror (seat-swapped,
95% Wilson CIs). This is roadmap step 4 for train_ppo.py: did the trained policy's win-rate CLEAR
the UNTRAINED fresh-net baseline?

Why "beat the baseline" and not just ">50%": the fresh net is NOT ~50% vs random — it uses the real
`auto_resolve` heuristic on every non-policy-select node (only single-selects are ~uniform at init),
so it starts with an edge over uniform-random. SUCCESS is therefore the trained CI clearing the
baseline CI (and ideally the point estimate above ~50%), not merely beating 50%.

Reports BOTH selection modes: `sampled` (the honest headline for a stochastic policy, per the
eval-discipline skill — argmax misrepresents a diffuse policy) and `greedy` (argmax, shows the mode).

    TORCHDYNAMO_DISABLE=1 python eval_ppo.py --ckpt out/ppo_final.pth --baseline out/ppo_init.pth --games 200
"""

import argparse
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: tcg_model first runs the CG_LIB_PATH sys.path insert that makes the
# vendored `cg` package importable; the PPO stack below depends on that having happened.
from tcg_model import random_agent, sample_deck                       # noqa: E402
from ppo_model import ActorCritic, STATE_DIM, OPTION_DIM              # noqa: E402
from train_ppo import policy_agent   # shared wrapper: SAME decision routing train + eval  # noqa: E402
from eval import wilson                                              # noqa: E402
from eval_mymodel import play        # reuse the seat-swapped win/loss/draw loop  # noqa: E402


def load_net(path, device):
    net = ActorCritic(STATE_DIM, OPTION_DIM).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()                        # no dropout/BN in the net, so eval() is behavior-neutral (conventional)
    return net


def eval_net(label, net, deck, games, seed):
    """Play `games` seat-swapped vs random for BOTH sampled and greedy selection; print + return
    {mode: (p, lo, hi)} decisive-only Wilson CIs (draws excluded from the denominator, matching
    eval.py / eval_mymodel.py — NOT train_ppo's inline wins/total)."""
    out = {}
    for mode in ("sampled", "greedy"):
        t = time.time()
        me_fn = policy_agent(net, greedy=(mode == "greedy"))
        # NOTE: play() reseeds Python's `random` to `seed` each call, so the opponent's random stream
        # is identical across nets/modes (comparable). The cg engine's own RNG is unseedable (known
        # limitation), so games are not bit-identical — hence still eval at n>=200 for a real CI.
        w, l, d = play(me_fn, random_agent, deck, games, seed)
        p, lo, hi = wilson(w, w + l)
        print(f"  {label:8s} {mode:8s} vs random: {w}/{l}/{d}  win%={100 * p:5.1f}% "
              f"(95% CI {100 * lo:.1f}-{100 * hi:.1f})  [{time.time() - t:.0f}s]", flush=True)
        out[mode] = (p, lo, hi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "out", "ppo_final.pth"),
                    help="trained PPO checkpoint (ActorCritic state_dict from train_ppo.py)")
    ap.add_argument("--baseline", default=os.path.join(_REPO_ROOT, "out", "ppo_init.pth"),
                    help="untrained fresh-net checkpoint to anchor against (train_ppo saves {prefix}_init.pth)")
    ap.add_argument("--games", type=int, default=200,
                    help="seat-swapped games per (net,mode); >=200 for a trustworthy CI (eval-discipline)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deck = sample_deck                 # the training mirror deck (both seats)
    print(f"PPO eval | sample_deck mirror | {args.games} games/mode | seed={args.seed} | device={device}",
          flush=True)

    results = {}
    for tag, path in (("baseline", args.baseline), ("trained", args.ckpt)):
        if not os.path.exists(path):
            print(f"  {tag}: checkpoint not found at {path} — skipping", flush=True)
            continue
        results[tag] = eval_net(tag, load_net(path, device), deck, args.games, args.seed)

    # Verdict on the sampled headline: training helped iff the trained CI clears the baseline CI.
    if "trained" in results and "baseline" in results:
        tp, tlo, thi = results["trained"]["sampled"]
        bp, blo, bhi = results["baseline"]["sampled"]
        clears = tlo > bhi
        verdict = ("training CLEARS baseline (CIs disjoint)" if clears
                   else "CIs OVERLAP — trained not distinguishable from fresh net")
        above50 = " and lower CI > 50%" if tlo > 0.5 else ""
        print(f"[verdict] sampled: trained {100 * tp:.1f}% (CI {100 * tlo:.1f}-{100 * thi:.1f}) vs "
              f"baseline {100 * bp:.1f}% (CI {100 * blo:.1f}-{100 * bhi:.1f}) -> {verdict}{above50}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
