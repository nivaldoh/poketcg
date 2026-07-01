#!/usr/bin/env python3
"""Eval the BC-pretrained starter MyModel on the Lucario mirror (seat-swapped, 95% Wilson CIs).

THE crux: `mcts_agent(MyModel-BC)` vs the rule-based teacher — a real AlphaZero (BC prior + the
BC-trained VALUE HEAD + PUCT search, no rollouts). Win-rate targets vs the teacher:
  >=48%  -> strong (matches the best search-based agent we've measured)
  >50%   -> it BEATS the teacher
Also greedy MyModel-BC (no search) as a sanity baseline.

    TORCHDYNAMO_DISABLE=1 python eval_mymodel.py --ckpt out/mymodel_lucario_bc.pth --games 60 --search-count 10
"""

import argparse
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "game_agents", "rulebased"))

import tcg_model  # noqa: E402
from tcg_model import (  # noqa: E402
    MyModel, mcts_agent, random_agent, eval_nn, get_encoder_input, get_decoder_input,
)
from cg.api import OptionType, to_observation_class  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import lucario as rulebased_lucario  # noqa: E402
from eval import wilson  # noqa: E402


def is_policy_select(sel) -> bool:
    if sel.minCount != 1 or sel.maxCount != 1 or len(sel.option) <= 1:
        return False
    if all(OptionType(o.type) == OptionType.NUMBER for o in sel.option):
        return False
    return True


def auto_resolve(sel) -> list[int]:
    L = len(sel.option)
    if all(OptionType(o.type) == OptionType.NUMBER for o in sel.option):
        return [max(range(L), key=lambda i: sel.option[i].number)]
    if L == 1:
        return [0]
    if sel.minCount == 0:
        return list(range(min(sel.maxCount, L)))
    return list(range(sel.minCount))


def greedy_agent(obs_dict, deck, model):
    """MyModel policy argmax on single-select policy decisions; auto-resolve everything else."""
    o = to_observation_class(obs_dict)
    sel = o.select
    if not is_policy_select(sel):
        return auto_resolve(sel)
    n_opt = len(sel.option)
    _, policy = eval_nn(get_encoder_input(o, deck), get_decoder_input(o, [[i] for i in range(n_opt)]), model)
    return [int(max(range(n_opt), key=lambda i: policy[i]))]


# The codebase's fixed policy convention: logits ≡ decoder_out × 10 (MCTS uses exp(p·10)),
# so π = softmax(10·tanh(decoder_out)). SINGLE SOURCE OF TRUTH — train_nashpg imports this so
# its act-time / update-time / eval-time distributions stay identical (PPO ratio correctness).
LOGIT_SCALE = 10.0


def sampled_agent(obs_dict, deck, model):
    """No-search BARE-POLICY agent: SAMPLE (not argmax) from softmax(policy·LOGIT_SCALE) on
    single-select policy decisions; auto-resolve everything else. Use this (not greedy_agent)
    to evaluate a trained policy: it's a mixed/stochastic policy, and argmax misrepresents it."""
    o = to_observation_class(obs_dict)
    sel = o.select
    if not is_policy_select(sel):
        return auto_resolve(sel)
    n_opt = len(sel.option)
    _, policy = eval_nn(get_encoder_input(o, deck), get_decoder_input(o, [[i] for i in range(n_opt)]), model)
    probs = torch.softmax(torch.tensor(policy) * LOGIT_SCALE, dim=0)
    return [int(torch.multinomial(probs, 1).item())]


def play(me_fn, opp_fn, deck, n_games, seed):
    random.seed(seed)
    w = l = d = 0
    for i in range(n_games):
        seat = i % 2
        obs, start = battle_start(deck, deck)
        if start.errorPlayer >= 0:
            raise ValueError("deck rejected")
        while obs["current"]["result"] < 0:
            yi = obs["current"]["yourIndex"]
            obs = battle_select(me_fn(obs) if yi == seat else opp_fn(obs))
        battle_finish()
        r = obs["current"]["result"]
        d += (r == 2); w += (r == seat); l += (r == 1 - seat)
    return w, l, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "out", "mymodel_lucario_bc.pth"))
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--search-count", type=int, default=10)
    ap.add_argument("--value-mode", default="net", choices=["net", "rollout"],
                    help="MCTS leaf value: net (MyModel value head) or rollout (depth-capped playout)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--agents", default="mcts,greedy", help="comma list of: mcts,greedy")
    ap.add_argument("--opponents", default="random,teacher", help="comma list of: random,teacher")
    args = ap.parse_args()

    tcg_model.SEARCH_COUNT = args.search_count
    deck = [int(x) for x in open(os.path.join(_REPO_ROOT, "deck.csv")).read().split()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"MyModel-BC ({args.ckpt}) | Lucario mirror | {args.games} games | "
          f"search_count={args.search_count} value_mode={args.value_mode}", flush=True)

    me_fns = {
        "mcts": lambda obs: mcts_agent(obs, deck, model, value_mode=args.value_mode)[0],
        "greedy": lambda obs: greedy_agent(obs, deck, model),
    }
    opp_fns = {"random": random_agent, "teacher": rulebased_lucario.agent}

    with torch.inference_mode():
        for a in args.agents.split(","):
            for op in args.opponents.split(","):
                t = time.time()
                w, l, d = play(me_fns[a.strip()], opp_fns[op.strip()], deck, args.games, args.seed)
                p, lo, hi = wilson(w, w + l)
                print(f"  {a.strip():6s} vs {op.strip():8s}: {w}/{l}/{d}  win%={100 * p:5.1f}% "
                      f"(95% CI {100 * lo:.1f}-{100 * hi:.1f})  [{time.time() - t:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
