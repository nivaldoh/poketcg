#!/usr/bin/env python3
"""Stage-1 evaluation harness: head-to-head win rates for the TCG agents.

This is our *primary* way to verify model performance, implementing the
"do-first" outcome layer from `eval_info.md`: play full battles between agents,
swap seats every game (to cancel first-player advantage), and report
win / loss / draw, a win rate with a **95% Wilson confidence interval**, and a
rough **Elo** over the pool.

Outcome metrics are noisy here — binary win/loss, hidden cards, coin flips — so
read the CIs and sample sizes, not just the point estimates. Process-level
metrics (Win-Probability-Added, puzzle suites) are a later stage and are
deliberately NOT implemented yet: they need a *calibrated* value model, which we
don't have (current checkpoints are only weak smoke-run models).

Usage:
    python eval.py                       # default sanity battery
    python eval.py --games 40            # more games per matchup (tighter CIs)
    python eval.py --checkpoint out/mixed_final.pth --baseline-checkpoint out/model0.pth
    python eval.py --search-count 8      # deeper MCTS for the model agents (slower)
"""

import argparse
import math
import os
import random

import torch

# Import the shared model/agent layer FIRST: it puts the vendored `cg` SDK on
# sys.path, so the `cg.game` import below resolves. (No training is triggered.)
import tcg_model
from tcg_model import MyModel, mcts_agent, random_agent, sample_deck
from cg.game import battle_start, battle_select, battle_finish
import rulebased_lucario

HERE = os.path.dirname(os.path.abspath(__file__))


class Agent:
    """A named policy paired with the 60-card deck it plays."""

    def __init__(self, name: str, fn, deck: list[int]):
        self.name = name          # display label, also the Elo key
        self.fn = fn              # obs_dict -> list[int] (option indices)
        self.deck = deck          # 60 card IDs


def make_random() -> Agent:
    # Plays sample_deck so a model-vs-random match is a clean *mirror* (same deck
    # both sides) that isolates agent skill from deck strength.
    return Agent("random", random_agent, sample_deck)


def make_rulebased() -> Agent:
    # The heuristics are hardcoded for the Lucario deck, so it must play that deck.
    return Agent("rulebased", rulebased_lucario.agent, rulebased_lucario.my_deck)


def make_model(path: str) -> Agent:
    """Load a checkpoint into an MCTS agent (plays sample_deck).

    MCTS depth is the module-global `tcg_model.SEARCH_COUNT`, set once in main().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    def fn(obs_dict):
        with torch.inference_mode():
            return mcts_agent(obs_dict, sample_deck, model)[0]  # drop training sample

    return Agent(f"model:{os.path.basename(path)}", fn, sample_deck)


def play_match(a: Agent, b: Agent, n_games: int) -> tuple[int, int, int]:
    """Play n_games between a and b, swapping seats each game.

    Returns (wins_a, losses_a, draws). Routing follows the canonical driver loop:
    `obs["current"]["yourIndex"]` says who acts; we call the seated agent.
    """
    wins = losses = draws = 0
    for i in range(n_games):
        seat_a = i % 2                       # alternate which player index a occupies
        decks = [None, None]
        decks[seat_a] = a.deck
        decks[1 - seat_a] = b.deck
        obs, start = battle_start(decks[0], decks[1])
        if start.errorPlayer >= 0:
            raise ValueError(
                f"deck rejected: errorType={start.errorType} player={start.errorPlayer}"
            )
        while obs["current"]["result"] < 0:
            actor = a if obs["current"]["yourIndex"] == seat_a else b
            obs = battle_select(actor.fn(obs))
        battle_finish()
        r = obs["current"]["result"]
        if r == 2:
            draws += 1
        elif r == seat_a:
            wins += 1
        else:
            losses += 1
        print(f"    game {i + 1}/{n_games}: W/L/D so far = {wins}/{losses}/{draws}",
              end="\r", flush=True)
    print(" " * 60, end="\r")  # clear progress line
    return wins, losses, draws


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """95% Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def fit_elo(names: list[str], wins: dict) -> dict:
    """Rough Bradley-Terry / Elo fit from pairwise decisive-win counts.

    `wins[(i, j)]` = times agent i beat agent j. Mean-anchored, with a light L2
    prior so perfect win/loss records (common vs random) don't run to +/-inf.
    Returns {name: elo}. This is a coarse summary, not a calibrated rating.
    """
    n = len(names)
    idx = {nm: k for k, nm in enumerate(names)}
    r = [0.0] * n
    for _ in range(3000):
        grad = [0.0] * n
        for (wi, wj), w in wins.items():
            i, j = idx[wi], idx[wj]
            nij = w + wins.get((wj, wi), 0)
            if nij == 0:
                continue
            pij = 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
            grad[i] += w - nij * pij
        for i in range(n):
            grad[i] -= 0.01 * r[i]          # L2 prior toward 0 (keeps it finite)
            r[i] += 0.05 * grad[i]
        m = sum(r) / n                       # anchor mean to 0
        for i in range(n):
            r[i] -= m
    scale = 400.0 / math.log(10.0)
    return {nm: r[idx[nm]] * scale for nm in names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20,
                    help="games per model-involving matchup (seat-swapped)")
    ap.add_argument("--baseline-games", type=int, default=200,
                    help="games for model-free matchups (cheap, so run many for tight CIs)")
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "out", "mixed_final.pth"),
                    help="trained model to evaluate")
    ap.add_argument("--baseline-checkpoint", default=os.path.join(HERE, "out", "model0.pth"),
                    help="earlier checkpoint, for a progress comparison (model vs baseline)")
    ap.add_argument("--checkpoints", default=None,
                    help="comma-separated checkpoints -> round-robin (random + each, all "
                         "playing sample_deck = clean mirror skill ranking). Overrides the battery.")
    ap.add_argument("--search-count", type=int, default=4,
                    help="MCTS rollouts per move for model agents (higher = stronger, slower)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tcg_model.SEARCH_COUNT = args.search_count

    # Round-robin mode (--checkpoints): pool = random + the listed checkpoints, all
    # playing sample_deck (pure mirror, NO deck confound) -> a clean skill ranking,
    # e.g. to see whether training improved along a checkpoint sequence. Otherwise
    # run the default sanity battery.
    rr_paths = [p.strip() for p in (args.checkpoints or "").split(",") if p.strip()]
    round_robin = bool(rr_paths)
    if round_robin:
        present = [p for p in rr_paths if os.path.exists(p)]
        for p in rr_paths:
            if p not in present:
                print(f"NOTE: checkpoint {p} missing -> skipped.")
        pool = [make_random()] + [make_model(p) for p in present]
        matchups = [(pool[i], pool[j], "round-robin (mirror, sample_deck)")
                    for i in range(len(pool)) for j in range(i + 1, len(pool))]
    else:
        # Build the agent pool. Model agents are skipped (with a note) if missing.
        rnd, rb = make_random(), make_rulebased()
        model = make_model(args.checkpoint) if os.path.exists(args.checkpoint) else None
        base = (make_model(args.baseline_checkpoint)
                if os.path.exists(args.baseline_checkpoint) else None)
        if model and base and os.path.basename(args.checkpoint) == os.path.basename(args.baseline_checkpoint):
            base = None  # same file -> skip the redundant self comparison

        # The sanity battery. Each tuple is (agent_a, agent_b, note).
        matchups = [
            (rnd, rnd, "harness self-check: expect ~50% (validates seat-swap)"),
            (rb, rnd, "validates agent ordering: expect rule-based clearly ahead"),
        ]
        if model:
            matchups.append((model, rnd, "mirror (sample_deck both sides): isolates skill"))
            matchups.append((model, rb, "progress signal; DECK-CONFOUNDED (sample vs Lucario)"))
            if base:
                matchups.append((model, base, "training progress (mirror): newer vs older ckpt"))

    print("=" * 78)
    print(f"Stage-1 eval | games/matchup={args.games} (seat-swapped) | "
          f"MCTS search-count={args.search_count} | seed={args.seed}")
    if not round_robin and not os.path.exists(args.checkpoint):
        print(f"NOTE: checkpoint {args.checkpoint} missing -> model matchups skipped.")
    print("=" * 78)

    elo_wins: dict = {}   # (winner_name, loser_name) -> count, for the Elo fit
    pool_names: list[str] = []

    def is_model(ag: Agent) -> bool:
        return ag.name.startswith("model:")

    for a, b, note in matchups:
        n = args.games if (is_model(a) or is_model(b)) else args.baseline_games
        label = f"{a.name} vs {b.name}"
        print(f"\n{label}")
        print(f"  ({note})")
        w, l, d = play_match(a, b, n)
        decisive = w + l
        p, lo, hi = wilson(w, decisive)
        print(f"  W/L/D = {w}/{l}/{d}   "
              f"win% = {100 * p:5.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f}, "
              f"n={decisive} decisive, {d} draws)")
        if a.name != b.name:  # accumulate for Elo (skip self-matches)
            elo_wins[(a.name, b.name)] = elo_wins.get((a.name, b.name), 0) + w
            elo_wins[(b.name, a.name)] = elo_wins.get((b.name, a.name), 0) + l
            for nm in (a.name, b.name):
                if nm not in pool_names:
                    pool_names.append(nm)

    if len(pool_names) >= 3:
        elo = fit_elo(pool_names, elo_wins)
        print("\n" + "-" * 78)
        confound = "clean mirror, sample_deck" if round_robin else "ROUGH, deck-confounded"
        print(f"Elo (Bradley-Terry, mean-anchored; {confound} — see caveats):")
        for nm, e in sorted(elo.items(), key=lambda kv: -kv[1]):
            print(f"  {nm:28s} {e:+7.0f}")

    print("\n" + "-" * 78)
    print("Caveats (see eval_info.md): win% excludes draws (counted separately).")
    print("  * vs rule-based mixes agent skill with the deck matchup (sample_deck vs")
    print("    Lucario deck); the mirror matches isolate skill better.")
    print("  * High outcome variance -> wide CIs at small n; raise --games for tighter.")
    print("  * Value-model / WPA process metrics are deferred (need a calibrated value")
    print("    model). Absolute numbers reflect weakly-trained checkpoints for now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
