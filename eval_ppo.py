#!/usr/bin/env python3
"""Eval a trained PPO ActorCritic (playing its fixed sample_deck) against a field of opponents,
seat-swapped, with 95% Wilson CIs.

Opponents (via `--opponents`, default `all`):
  * random    — random-action agent on sample_deck: a clean MIRROR (same deck both seats) that
                isolates policy skill from deck strength. This is the roadmap step-4 FLOOR.
  * lucario / abomasnow / dragapult / iono — the four rule-based agents, each bringing its OWN
                archetype deck. These are genuine head-to-head but DECK-CONFOUNDED (win-rate mixes
                policy skill with the deck matchup), exactly like eval.py's gauntlet "FIELD".
                (sample_deck is an Abomasnow-ARCHETYPE deck sharing ~85% of cards (51/60) with
                decks/abomasnow.csv — NOT identical — so vs `abomasnow` is a near-mirror, not a mirror.)

For each opponent it evaluates BOTH the untrained fresh-net baseline (`--baseline`, saved by
train_ppo as {prefix}_init.pth) and the trained checkpoint (`--ckpt`), then prints a per-opponent
"did training help?" summary. Selection defaults to `sampled` (the honest headline for a stochastic
policy, per the eval-discipline skill — argmax misrepresents a diffuse policy); `--modes
sampled,greedy` also shows the argmax mode.

Why "beat the baseline" and not just ">50%": the fresh net is NOT ~50% vs random — it uses the real
`auto_resolve` heuristic on every non-policy-select node (only single-selects are ~uniform at init),
so it starts with an edge over uniform-random. SUCCESS vs the random floor is the trained CI clearing
the baseline CI, not merely beating 50%.

This reuses eval.py's two-deck machinery wholesale (`Agent`, `play_match`, `wilson`,
`elo_from_winrate`, and the `make_*` opponent factories) — the PPO net is wrapped as an
`eval.Agent` via train_ppo's `policy_agent`, which is deck-agnostic so it plays any opponent's deck
unmodified. CAVEAT: the PPO card-embeddings only trained on sample_deck's cards, so an opponent's
off-deck cards hit untrained (noise) embedding rows; the typed features still carry signal, the
embeddings just add none there (documented in ppo_model.py).

    TORCHDYNAMO_DISABLE=1 python eval_ppo.py --opponents all --games 200
    TORCHDYNAMO_DISABLE=1 python eval_ppo.py --opponents random --games 200   # just the floor
"""

import argparse
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: tcg_model first runs the CG_LIB_PATH sys.path insert that makes the
# vendored `cg` package importable; the PPO stack below depends on that having happened. Importing
# `eval` also re-runs that same setup plus the game_agents/rulebased sys.path insert (idempotent).
from tcg_model import sample_deck                                     # noqa: E402
from ppo_model import ActorCritic, STATE_DIM, OPTION_DIM             # noqa: E402
from train_ppo import policy_agent   # shared wrapper: SAME decision routing train + eval  # noqa: E402
from eval import (                                                    # noqa: E402
    Agent, play_match, wilson, elo_from_winrate,
    make_random, make_rulebased, make_abomasnow, make_dragapult, make_iono,
)


# Opponent registry: name -> factory returning a ready eval.Agent (policy fused with the deck it
# plays). Insertion order = display order (random floor first). NOTE: eval.py names the Lucario
# factory `make_rulebased` (its Agent.name is "rulebased"); we expose it here under "lucario".
OPPONENTS = {
    "random": make_random,        # mirror: plays sample_deck (clean floor; saturates high)
    "lucario": make_rulebased,    # deck-confounded: own Lucario deck
    "abomasnow": make_abomasnow,  # near-mirror: sample_deck shares ~85% (51/60) with decks/abomasnow.csv
    "dragapult": make_dragapult,  # deck-confounded: own Dragapult ex deck
    "iono": make_iono,            # deck-confounded: own Iono deck
}


def load_net(path, device):
    net = ActorCritic(STATE_DIM, OPTION_DIM).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()                        # no dropout/BN in the net, so eval() is behavior-neutral (conventional)
    return net


def resolve_opponents(spec):
    """'all' -> every registry key (field order); else a comma list validated against the registry."""
    if spec.strip() == "all":
        return list(OPPONENTS)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in OPPONENTS]
    if bad:
        raise SystemExit(f"unknown opponent(s) {bad}; choose from {list(OPPONENTS)} or 'all'")
    return names


def run_match(net, mode, opp, deck, games, seed):
    """Play the PPO net (as `deck`, i.e. sample_deck) vs one opponent Agent (its OWN deck) for
    `games` seat-swapped battles in one selection `mode`. Returns (w, l, d, p, lo, hi) — decisive-only
    Wilson CI (draws excluded from the denominator, matching eval.py / eval_mymodel.py)."""
    me = Agent(f"ppo:{mode}", policy_agent(net, greedy=(mode == "greedy")), deck)
    # play_match (unlike eval_mymodel.play) does NOT reseed; reseed here so the opponent's Python-RNG
    # stream is identical across nets/modes (comparable). The cg engine RNG stays unseedable, so games
    # are not bit-identical — hence still eval at n>=200 for a real CI.
    random.seed(seed)
    w, l, d = play_match(me, opp, games)
    p, lo, hi = wilson(w, w + l)
    return w, l, d, p, lo, hi


def _fmt_elo(p):
    e = elo_from_winrate(p)
    if e == float("inf"):
        return "+inf"
    if e == float("-inf"):
        return "-inf"
    return f"{e:+.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "out", "ppo_final.pth"),
                    help="trained PPO checkpoint (ActorCritic state_dict from train_ppo.py)")
    ap.add_argument("--baseline", default=os.path.join(_REPO_ROOT, "out", "ppo_init.pth"),
                    help="untrained fresh-net checkpoint to anchor against (train_ppo saves {prefix}_init.pth)")
    ap.add_argument("--opponents", default="all",
                    help="comma list of {random,lucario,abomasnow,dragapult,iono} or 'all' (default)")
    ap.add_argument("--modes", default="sampled",
                    help="comma list of {sampled,greedy}; sampled is the honest headline (default)")
    ap.add_argument("--games", type=int, default=200,
                    help="seat-swapped games per (net,opponent,mode); >=200 for a trustworthy CI "
                         "(eval-discipline). NOTE: total battles = nets x opponents x modes x games.")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deck = sample_deck                 # the PPO seat's fixed deck (mirror vs random; confounded vs rule-based)
    opp_names = resolve_opponents(args.opponents)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad_modes = [m for m in modes if m not in ("sampled", "greedy")]
    if bad_modes:
        raise SystemExit(f"unknown mode(s) {bad_modes}; choose from ['sampled', 'greedy']")

    # Build each opponent Agent ONCE and reuse across nets/games — exactly like eval.run_gauntlet
    # (the agents self-reset their per-game state via turn-detection; play_match already reuses them
    # this way). Each rule-based opponent brings its own deck; random plays sample_deck.
    opponents = [(name, OPPONENTS[name]()) for name in opp_names]

    print(f"PPO field eval | PPO plays sample_deck | {args.games} games/cell | modes={modes} | "
          f"opponents={opp_names} | seed={args.seed} | device={device}", flush=True)

    # results[tag][name][mode] = (w, l, d, p, lo, hi)
    results = {}
    for tag, path in (("baseline", args.baseline), ("trained", args.ckpt)):
        if not os.path.exists(path):
            print(f"  {tag}: checkpoint not found at {path} — skipping", flush=True)
            continue
        net = load_net(path, device)
        results[tag] = {}
        for name, opp in opponents:
            results[tag][name] = {}
            for mode in modes:
                t = time.time()
                w, l, d, p, lo, hi = run_match(net, mode, opp, deck, args.games, args.seed)
                results[tag][name][mode] = (w, l, d, p, lo, hi)
                print(f"  {tag:8s} {mode:8s} vs {name:10s}: {w}/{l}/{d}  win%={100 * p:5.1f}% "
                      f"(95% CI {100 * lo:.1f}-{100 * hi:.1f})  [{time.time() - t:.0f}s]", flush=True)

    # Per-opponent "did training help?" summary on the headline mode (sampled if present). Compares the
    # trained CI to the baseline CI per opponent — CIs disjoint = a real move, overlap = indistinguishable.
    if "baseline" in results and "trained" in results:
        head = "sampled" if "sampled" in modes else modes[0]
        print(f"\n[summary] did training help? ({head} mode, trained vs baseline per opponent)", flush=True)
        for name, _ in opponents:
            bp, blo, bhi = results["baseline"][name][head][3:6]
            tp, tlo, thi = results["trained"][name][head][3:6]
            if tlo > bhi:
                verdict = "training CLEARS baseline (CIs disjoint)"
            elif thi < blo:
                verdict = "REGRESSED vs baseline (CIs disjoint)"
            else:
                verdict = "CIs overlap — indistinguishable"
            kind = "mirror floor" if name == "random" else "deck-confounded"
            print(f"  vs {name:10s} [{kind:14s}]: baseline {100 * bp:5.1f}% (CI {100 * blo:.1f}-{100 * bhi:.1f}) "
                  f"-> trained {100 * tp:5.1f}% (CI {100 * tlo:.1f}-{100 * thi:.1f})  "
                  f"[trained Elo vs opp {_fmt_elo(tp)}]  -> {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
