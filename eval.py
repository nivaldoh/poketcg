#!/usr/bin/env python3
"""Stage-1 evaluation harness: head-to-head win rates for the TCG agents.

This is our *primary* way to verify model performance, implementing the
"do-first" outcome layer from `docs/eval_info.md`: play full battles between agents,
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
import hashlib
import json
import math
import os
import random
import subprocess
import sys

import torch

# Import the shared model/agent layer FIRST: it puts the vendored `cg` SDK on
# sys.path, so the `cg.game` import below resolves. (No training is triggered.)
import tcg_model
from tcg_model import MyModel, mcts_agent, random_agent, sample_deck
from cg.game import battle_start, battle_select, battle_finish
# The rule-based agents live in game_agents/rulebased/ as flat modules; add that dir to sys.path so
# they import by bare name here AND on Kaggle (push_sweep.py stages them flat in the BR dataset).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_agents", "rulebased"))
import lucario as rulebased_lucario
# Each deck is one clean module exposing `agent` + `my_deck`; they reset their per-game state on the
# first call (see each file's obs.select-None / turn-0 handling) so reuse across the gauntlet is clean.
import abomasnow as rb_abomasnow
import dragapult_ex as rb_dragapult
import iono as rb_iono

HERE = os.path.dirname(os.path.abspath(__file__))

# Trustworthy-n floor (decisive games) for model matchups. Empirical, from memory
# `eval-training-not-helping-yet` (n=60 once swung 49 vs the reliable 40); used only to
# warn. The real fix for "did it get better?" is --mode sprt / --mode gauntlet.
TRUST_N = 200


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


# The newer rule-based archetypes, each hardcoded to (and playing) its own deck.
def make_abomasnow() -> Agent:
    return Agent("abomasnow", rb_abomasnow.agent, rb_abomasnow.my_deck)


def make_dragapult() -> Agent:
    return Agent("dragapult_ex", rb_dragapult.agent, rb_dragapult.my_deck)


def make_iono() -> Agent:
    return Agent("iono", rb_iono.agent, rb_iono.my_deck)


# How checkpoints play, set once in main() (like tcg_model.SEARCH_COUNT). "sampled" = no-search
# bare policy (honest "did the POLICY improve?" read); "mcts" = the default search-on-top agent.
AGENT_KIND = "mcts"


def make_model(path: str) -> Agent:
    """Load a checkpoint into an Agent that plays sample_deck. Kind set by the module-global
    AGENT_KIND (main()): "mcts" (depth = tcg_model.SEARCH_COUNT) or "sampled" (no-search)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    if AGENT_KIND == "sampled":
        # RISK (load-bearing): import sampled_agent LAZILY here, NOT at module top. eval_mymodel
        # does `from eval import wilson` at its top, so a top-level `from eval_mymodel import
        # sampled_agent` would run before wilson is defined -> circular ImportError. A function-body
        # import runs only after eval is fully initialized.
        from eval_mymodel import sampled_agent

        def fn(obs_dict):
            with torch.inference_mode():
                return sampled_agent(obs_dict, sample_deck, model)
    else:
        def fn(obs_dict):
            with torch.inference_mode():
                return mcts_agent(obs_dict, sample_deck, model)[0]  # drop training sample

    # Keep the "model:" name prefix either way so battery Elo bookkeeping (is_model) still works.
    return Agent(f"model:{os.path.basename(path)}", fn, sample_deck)


def _play_one(a: Agent, b: Agent, seat_a: int) -> int:
    """Play one battle with agent `a` seated at player index `seat_a`. Returns the engine
    result (0/1 = that seat won, 2 = draw). Shared by play_match and play_match_sequential.

    Routing follows the canonical driver loop: `obs["current"]["yourIndex"]` says who acts.
    NOTE: the cg engine seeds its own RNG from OS entropy (no seed hook in the ABI), so every
    battle's shuffle/coin-flips are independent and uncontrollable — there is no common-random-
    numbers pairing to exploit (see EVAL.md "Reliable A/B comparison").
    """
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
    return obs["current"]["result"]


def play_match(a: Agent, b: Agent, n_games: int) -> tuple[int, int, int]:
    """Play n_games between a and b, swapping seats each game. Returns (wins_a, losses_a, draws)."""
    wins = losses = draws = 0
    for i in range(n_games):
        seat_a = i % 2                       # alternate which player index a occupies
        r = _play_one(a, b, seat_a)
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


# --- reproducibility + reliable-comparison helpers (see EVAL.md "Reliable A/B comparison") ---

def sha256(path: str) -> str:
    """Content hash of a checkpoint — its only stable identity (out/ is gitignored and filenames
    get reused, so the run manifest pins by hash, not by name)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deck_hash(deck: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, deck)).encode()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def elo_from_winrate(p: float) -> float:
    """Logistic Elo gap implied by a decisive win-rate p — the comparable scalar vs a frozen
    anchor. +/-inf at a clean sweep."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return 400.0 * math.log10(p / (1.0 - p))


def anchored_elo(names: list[str], wins: dict, anchor: str) -> dict:
    """Like fit_elo but pegs `anchor` to 0 every iteration instead of mean-anchoring the pool,
    so a checkpoint's rating is COMPARABLE ACROSS RUNS (mean-anchoring makes it depend on who else
    happened to be in the pool). Same light L2 prior keeps saturated records finite."""
    n = len(names)
    idx = {nm: k for k, nm in enumerate(names)}
    a = idx[anchor]
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
            grad[i] -= 0.01 * r[i]
            r[i] += 0.05 * grad[i]
        shift = r[a]                         # peg anchor to 0 -> comparable across runs
        for i in range(n):
            r[i] -= shift
    scale = 400.0 / math.log(10.0)
    return {nm: r[idx[nm]] * scale for nm in names}


def sprt(sum_x: float, sum_x2: float, n: int,
         elo0: float = 0.0, elo1: float = 10.0,
         alpha: float = 0.05, beta: float = 0.05) -> tuple[str, float]:
    """Generalized SPRT (normal approximation) on the per-game score x in {0, 0.5, 1} — the
    computer-chess "did this patch help?" test, drawing the FEWEST games for controlled
    (alpha, beta) error. H0: strength = elo0; H1: strength = elo1 (Elo of agent a over b).
    Returns (verdict, llr), verdict in {"h1","h0","continue"}.

    LLR = (mu1-mu0)/var * (Sx - n*(mu0+mu1)/2). `var` is the per-game score variance shrunk toward
    the max bounded-score variance (0.25, by Popoviciu) with a weak prior, so a short identical
    streak (var ~ 0) can't blow the normal-approx LLR up into a false early verdict; the shrinkage
    vanishes as n grows. Draws are handled natively via the 0.5 score. Bounds: log((1-beta)/alpha),
    log(beta/(1-alpha))."""
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if n < 2:
        return ("continue", 0.0)
    mu = sum_x / n
    var = sum_x2 / n - mu * mu                        # MLE variance of the per-game score
    var = max((n * var + 4.0 * 0.25) / (n + 4.0), 1e-6)   # weak shrink toward 0.25 (conservative)
    mu0 = 1.0 / (1.0 + 10 ** (-elo0 / 400.0))
    mu1 = 1.0 / (1.0 + 10 ** (-elo1 / 400.0))
    llr = (mu1 - mu0) / var * (sum_x - n * (mu0 + mu1) / 2.0)
    if llr >= upper:
        return ("h1", llr)
    if llr <= lower:
        return ("h0", llr)
    return ("continue", llr)


def play_match_sequential(a: Agent, b: Agent, elo0: float, elo1: float,
                          alpha: float, beta: float, max_games: int) -> tuple:
    """Seat-swapped games (via _play_one) streamed into the SPRT; stop at a decision bound or
    max_games. Returns (verdict, llr, wins_a, losses_a, draws, n_played)."""
    sum_x = sum_x2 = 0.0
    w = l = d = 0
    verdict, llr = "continue", 0.0
    n = 0
    for i in range(max_games):
        seat_a = i % 2
        r = _play_one(a, b, seat_a)
        if r == 2:
            x = 0.5; d += 1
        elif r == seat_a:
            x = 1.0; w += 1
        else:
            x = 0.0; l += 1
        sum_x += x; sum_x2 += x * x; n += 1
        verdict, llr = sprt(sum_x, sum_x2, n, elo0, elo1, alpha, beta)
        print(f"    game {n}: W/L/D={w}/{l}/{d}  LLR={llr:+.2f}  [{verdict}]",
              end="\r", flush=True)
        if verdict != "continue":
            break
    print(" " * 70, end="\r")
    return verdict, llr, w, l, d, n


def gauntlet(model: Agent, pool: list, n: int) -> list:
    """Play `model` vs each fixed opponent in `pool` (a list of (Agent, kind, note)); return one
    result row per opponent. `kind` in {"floor","skill","confound"}."""
    rows = []
    for opp, kind, note in pool:
        print(f"\n  vs {opp.name}  ({note})")
        w, l, d = play_match(model, opp, n)
        p, lo, hi = wilson(w, w + l)
        rows.append(dict(opp=opp.name, kind=kind, note=note, w=w, l=l, d=d,
                         p=p, lo=lo, hi=hi, decisive=w + l, elo=elo_from_winrate(p)))
    return rows


def run_manifest(mode: str, args, ckpt_paths: list, extra: dict = None) -> dict:
    """The comparability contract: two runs compare only if these fields match. The engine RNG is
    UNCONTROLLED (no seed hook), so reproducibility comes from pinned config + n, not seeds."""
    man = {
        "mode": mode,
        "search_count": args.search_count,
        "seed": args.seed,
        "engine_rng": "UNCONTROLLED (OS entropy; --seed affects agent-side sampling only)",
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "determinism": ("use_deterministic_algorithms(warn_only); "
                        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset')}"),
        "arch": "MyModel(128,2,256,1,1)",
        "git_commit": git_commit(),
        "checkpoints": [{"path": p, "sha256": sha256(p)}
                        for p in dict.fromkeys(ckpt_paths) if p and os.path.exists(p)],
        "decks": {"sample_deck": deck_hash(sample_deck),
                  "lucario": deck_hash(rulebased_lucario.my_deck),
                  "abomasnow": deck_hash(rb_abomasnow.my_deck),
                  "dragapult_ex": deck_hash(rb_dragapult.my_deck),
                  "iono": deck_hash(rb_iono.my_deck)},
    }
    if extra:
        man.update(extra)
    return man


def run_gauntlet(args) -> int:
    """Score ONE checkpoint against the FROZEN, hash-pinned gauntlet -> one comparable scalar."""
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: --checkpoint {args.checkpoint} missing.")
        return 1
    if not os.path.exists(args.anchor):
        print(f"ERROR: --anchor {args.anchor} missing. Create it once from the frozen-strongest "
              f"checkpoint, e.g.:  cp out/model0.pth {args.anchor}")
        return 1
    print(json.dumps(run_manifest("gauntlet", args, [args.checkpoint, args.anchor],
                                  extra={"gauntlet_games": args.gauntlet_games}), indent=2))
    n = args.gauntlet_games
    if n < TRUST_N:
        print(f"WARNING: --gauntlet-games {n} < trustworthy floor ~{TRUST_N} "
              f"(memory eval-training-not-helping-yet); CIs will be wide.")
    model = make_model(args.checkpoint)
    pool = [
        (make_random(), "floor", "mirror sample_deck — sanity FLOOR (saturates; everyone beats random)"),
        (make_model(args.anchor), "skill", "mirror sample_deck vs FROZEN anchor — the discriminating SKILL axis"),
        # Deck-confounded archetype field: how the checkpoint fares across a diverse metagame, not just
        # the Lucario teacher. Each plays its own deck, so these mix skill with the deck matchup.
        (make_rulebased(), "confound", "vs teacher (Lucario) — DECK-CONFOUNDED yardstick"),
        (make_abomasnow(), "confound", "vs Abomasnow ex — DECK-CONFOUNDED archetype"),
        (make_dragapult(), "confound", "vs Dragapult ex — DECK-CONFOUNDED archetype"),
        (make_iono(), "confound", "vs Iono — DECK-CONFOUNDED archetype"),
    ]
    print(f"\n{'=' * 78}\nGauntlet | {os.path.basename(args.checkpoint)} | {n} games/opponent "
          f"(seat-swapped) | search-count={args.search_count}\n{'=' * 78}")
    rows = gauntlet(model, pool, n)
    print("\n" + "-" * 78)
    for r in rows:
        print(f"  {r['kind']:8s} vs {r['opp']:26s} W/L/D={r['w']}/{r['l']}/{r['d']}  "
              f"win%={100 * r['p']:5.1f}% (95% CI {100 * r['lo']:.1f}-{100 * r['hi']:.1f}, "
              f"n={r['decisive']} dec, {r['d']} draws)")
    skill = next(r for r in rows if r["kind"] == "skill")
    floor = next(r for r in rows if r["kind"] == "floor")
    field = [r for r in rows if r["kind"] == "confound"]
    e = skill["elo"]
    e_str = "+inf" if e == float("inf") else ("-inf" if e == float("-inf") else f"{e:+.0f}")
    print("\n" + "-" * 78)
    print("HEADLINE (the comparable scalar — track this across checkpoints vs the SAME anchor):")
    print(f"  SKILL axis : win% vs frozen anchor = {100 * skill['p']:5.1f}%  "
          f"(95% CI {100 * skill['lo']:.1f}-{100 * skill['hi']:.1f})  => {e_str} Elo vs anchor")
    print(f"  FLOOR      : win% vs random        = {100 * floor['p']:5.1f}%  (expect high; saturated)")
    print("  FIELD      : win% vs each rule-based archetype (DECK-CONFOUNDED — mixes skill + matchup):")
    for r in field:
        print(f"               {r['opp']:14s} = {100 * r['p']:5.1f}%  "
              f"(95% CI {100 * r['lo']:.1f}-{100 * r['hi']:.1f}, n={r['decisive']} dec)")
    if field:
        print(f"               {'field mean':14s} = {100 * sum(r['p'] for r in field) / len(field):5.1f}%")
    print("\nProgression/regression = change in the SKILL-axis win% (or Elo) vs this same frozen")
    print("anchor across runs. For a controlled ship/no-ship verdict use --mode sprt. Near-identical")
    print("checkpoints won't separate here (mirror ~50/50) — use eval_agreement.py for those.")
    return 0


def run_sprt(args) -> int:
    """Sequential 'is A (--checkpoint) better than B (--baseline-checkpoint)?' at controlled error."""
    A, B = args.checkpoint, args.baseline_checkpoint
    if not (os.path.exists(A) and os.path.exists(B)):
        print("ERROR: --mode sprt needs both --checkpoint (A) and --baseline-checkpoint (B) to exist.")
        return 1
    print(json.dumps(run_manifest("sprt", args, [A, B], extra={
        "sprt": dict(elo0=args.sprt_elo0, elo1=args.sprt_elo1,
                     alpha=args.sprt_alpha, beta=args.sprt_beta, max_games=args.max_games)}), indent=2))
    a, b = make_model(A), make_model(B)
    print(f"\n{'=' * 78}")
    print(f"SPRT | H0: {a.name} <= {args.sprt_elo0:+.0f} Elo over {b.name}   vs   "
          f"H1: >= {args.sprt_elo1:+.0f} Elo")
    print(f"     | alpha={args.sprt_alpha} beta={args.sprt_beta} max-games={args.max_games} "
          f"search-count={args.search_count} (seat-swapped)\n{'=' * 78}")
    verdict, llr, w, l, d, n = play_match_sequential(
        a, b, args.sprt_elo0, args.sprt_elo1, args.sprt_alpha, args.sprt_beta, args.max_games)
    p, lo, hi = wilson(w, w + l)
    e = elo_from_winrate(p)
    e_str = "+inf" if e == float("inf") else ("-inf" if e == float("-inf") else f"{e:+.0f}")
    msg = {
        "h1": f"ACCEPT H1 — {a.name} is BETTER than {b.name} (ship).",
        "h0": f"ACCEPT H0 — no >= {args.sprt_elo1:+.0f} Elo gain for {a.name} over {b.name}.",
        "continue": f"INCONCLUSIVE at max-games={args.max_games} — need more games or a larger effect.",
    }[verdict]
    print("\n" + "-" * 78)
    print(f"  games={n}  W/L/D={w}/{l}/{d}  win%={100 * p:.1f}% "
          f"(95% CI {100 * lo:.1f}-{100 * hi:.1f})  implied {e_str} Elo  LLR={llr:+.2f}")
    print(f"  VERDICT: {msg}")
    return 0


def run_battery(args) -> int:
    """The default sanity battery / --checkpoints round-robin (outcome metrics + anchored Elo)."""
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
        ckpt_paths = present
    else:
        # Build the agent pool. Model agents are skipped (with a note) if missing.
        rnd, rb = make_random(), make_rulebased()
        model = make_model(args.checkpoint) if os.path.exists(args.checkpoint) else None
        base = (make_model(args.baseline_checkpoint)
                if os.path.exists(args.baseline_checkpoint) else None)
        if model and base and os.path.basename(args.checkpoint) == os.path.basename(args.baseline_checkpoint):
            base = None  # same file -> skip the redundant self comparison
        matchups = [
            (rnd, rnd, "harness self-check: expect ~50% (validates seat-swap)"),
            (rb, rnd, "validates agent ordering: expect rule-based clearly ahead"),
        ]
        if model:
            matchups.append((model, rnd, "mirror (sample_deck both sides): isolates skill"))
            matchups.append((model, rb, "progress signal; DECK-CONFOUNDED (sample vs Lucario)"))
            if base:
                matchups.append((model, base, "training progress (mirror): newer vs older ckpt"))
        ckpt_paths = [p for p in [args.checkpoint, args.baseline_checkpoint] if os.path.exists(p)]

    print(json.dumps(run_manifest("round-robin" if round_robin else "battery", args, ckpt_paths), indent=2))
    print("=" * 78)
    print(f"Stage-1 eval | games/matchup={args.games} (seat-swapped) | "
          f"MCTS search-count={args.search_count} | seed={args.seed}")
    if args.games < TRUST_N:
        print(f"WARNING: --games {args.games} < trustworthy floor ~{TRUST_N} for model matchups "
              f"(memory eval-training-not-helping-yet). For a real verdict use --mode sprt / "
              f"--mode gauntlet, or raise --games.")
    if not round_robin and not os.path.exists(args.checkpoint):
        print(f"NOTE: checkpoint {args.checkpoint} missing -> model matchups skipped.")
    print("=" * 78)

    elo_wins: dict = {}   # (winner_name, loser_name) -> count, for the Elo fit
    pool_names: list[str] = []

    def is_model(ag: Agent) -> bool:
        return ag.name.startswith("model:")

    for a, b, note in matchups:
        n = args.games if (is_model(a) or is_model(b)) else args.baseline_games
        print(f"\n{a.name} vs {b.name}")
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
        # Anchor to a stable present agent so the rating is comparable across runs: the rule-based
        # teacher in the battery, else the first non-random checkpoint (random saturates, bad anchor).
        anchor = ("rulebased" if "rulebased" in pool_names
                  else next((nm for nm in pool_names if nm != "random"), pool_names[0]))
        elo = anchored_elo(pool_names, elo_wins, anchor)
        print("\n" + "-" * 78)
        if round_robin:
            print(f"Anchored Elo (Bradley-Terry; clean mirror sample_deck; {anchor}=0):")
        else:
            print(f"Anchored Elo (Bradley-Terry; {anchor}=0) — ROUGH & DECK-CONFOUNDED, NOT a skill")
            print("  rating (mixes mirror + vs-Lucario axes; use --mode gauntlet for a clean axis):")
        for nm, e in sorted(elo.items(), key=lambda kv: -kv[1]):
            print(f"  {nm:28s} {e:+7.0f}")
        print("  NOTE: pairs vs 'random' saturate (~100-0); those ratings are prior-driven, not data.")

    print("\n" + "-" * 78)
    print("Caveats (see docs/eval_info.md / EVAL.md): win% excludes draws (counted separately).")
    print("  * vs rule-based mixes agent skill with the deck matchup (sample_deck vs Lucario); the")
    print("    mirror matches isolate skill better. For comparable cross-run numbers use --mode")
    print("    gauntlet (frozen anchor) or --mode sprt (controlled A/B verdict).")
    print("  * High outcome variance -> wide CIs at small n; the engine RNG is uncontrolled, so")
    print("    reliability comes from n, not seeds (see EVAL.md 'Reliable A/B comparison').")
    print("  * Value-model / WPA process metrics are deferred (need a calibrated value model).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["battery", "gauntlet", "sprt"], default="battery",
                    help="battery = sanity battery / --checkpoints round-robin (default); "
                         "gauntlet = score one ckpt vs the frozen anchor pool -> a comparable "
                         "scalar; sprt = sequential A/B verdict (--checkpoint vs --baseline-checkpoint)")
    ap.add_argument("--games", type=int, default=20,
                    help="games per model-involving matchup (seat-swapped)")
    ap.add_argument("--baseline-games", type=int, default=200,
                    help="games for model-free matchups (cheap, so run many for tight CIs)")
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "out", "mixed_final.pth"),
                    help="trained model to evaluate (agent A in sprt mode)")
    ap.add_argument("--baseline-checkpoint", default=os.path.join(HERE, "out", "model0.pth"),
                    help="earlier checkpoint for a progress comparison (agent B in sprt mode)")
    ap.add_argument("--checkpoints", default=None,
                    help="comma-separated checkpoints -> round-robin (random + each, all "
                         "playing sample_deck = clean mirror skill ranking). Overrides the battery.")
    ap.add_argument("--anchor", default=os.path.join(HERE, "out", "anchor_model0.pth"),
                    help="frozen, hash-pinned reference checkpoint for the gauntlet SKILL axis")
    ap.add_argument("--gauntlet-games", type=int, default=200,
                    help="games per opponent in --mode gauntlet (seat-swapped)")
    ap.add_argument("--sprt-elo0", type=float, default=0.0, help="SPRT H0 Elo (null: not better)")
    ap.add_argument("--sprt-elo1", type=float, default=10.0,
                    help="SPRT H1 Elo (the minimum edge worth shipping)")
    ap.add_argument("--sprt-alpha", type=float, default=0.05, help="SPRT type-I error")
    ap.add_argument("--sprt-beta", type=float, default=0.05, help="SPRT type-II error")
    ap.add_argument("--max-games", type=int, default=400, help="SPRT game cap (seat-swapped)")
    ap.add_argument("--search-count", type=int, default=4,
                    help="MCTS rollouts per move for model agents (higher = stronger, slower)")
    ap.add_argument("--agent-kind", choices=["mcts", "sampled"], default="mcts",
                    help="how checkpoints play: mcts (default) or sampled = no-search bare policy "
                         "(use for an honest 'did the POLICY improve?' read; --search-count then unused)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        # Reproducible forward pass given inputs (eval is inference-only). warn_only so an op
        # without a deterministic kernel degrades gracefully instead of crashing the eval.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    tcg_model.SEARCH_COUNT = args.search_count
    global AGENT_KIND
    AGENT_KIND = args.agent_kind   # read by make_model; mirrors the SEARCH_COUNT-set-in-main pattern

    if args.mode == "gauntlet":
        return run_gauntlet(args)
    if args.mode == "sprt":
        return run_sprt(args)
    return run_battery(args)


if __name__ == "__main__":
    raise SystemExit(main())
