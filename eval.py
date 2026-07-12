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
    python eval.py --mode field --field-games 200   # rule-based field: rich per-pairing telemetry
    python eval.py --mode field --smoke             # fast self-checking telemetry smoke
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter

import torch

# Import the shared model/agent layer FIRST: it puts the vendored `cg` SDK on
# sys.path, so the `cg.game` import below resolves. (No training is triggered.)
import tcg_model
from tcg_model import MyModel, mcts_agent, random_agent, sample_deck
from cg.game import battle_start, battle_select, battle_finish
from win_mode import classify_end, summarize_ends
# The rule-based agents live in game_agents/rulebased/ as flat modules; add that dir to sys.path so
# they import by bare name here AND on Kaggle (push_sweep.py stages them flat in its code dataset).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_agents", "rulebased"))
import lucario as rulebased_lucario
# Each deck is one clean module exposing `agent` + `my_deck`; they reset their per-game state on the
# first call (see each file's obs.select-None / turn-0 handling) so reuse across the gauntlet is clean.
import abomasnow as rb_abomasnow
import dragapult_ex as rb_dragapult
import iono as rb_iono
import simple_raging_bolt_burn as rb_raging_bolt
import real_raging_bolt_burn as rb_real_raging_bolt
import lilies_clefairy_ex_mega_kangaskhan_ex as rb_lilies_clefairy

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


def make_raging_bolt() -> Agent:
    return Agent("raging_bolt", rb_raging_bolt.agent, rb_raging_bolt.my_deck)


def make_real_raging_bolt() -> Agent:
    return Agent("real_raging_bolt", rb_real_raging_bolt.agent, rb_real_raging_bolt.my_deck)


def make_lilies_clefairy() -> Agent:
    return Agent("lilies_clefairy", rb_lilies_clefairy.agent, rb_lilies_clefairy.my_deck)


def make_model(path: str) -> Agent:
    """Load a checkpoint into an mcts_agent (depth = tcg_model.SEARCH_COUNT) playing sample_deck.
    The "model:" name prefix drives the battery Elo bookkeeping (is_model)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    def fn(obs_dict):
        with torch.inference_mode():
            return mcts_agent(obs_dict, sample_deck, model)[0]  # drop training sample

    return Agent(f"model:{os.path.basename(path)}", fn, sample_deck)


def _play_one(a: Agent, b: Agent, seat_a: int) -> tuple[int, dict]:
    """Play one battle with agent `a` seated at player index `seat_a`. Returns (result, end):
    the engine result (0/1 = that seat won, 2 = draw) and the win_mode.classify_end dict
    relative to seat_a (win mode + terminal margins — pure-additive telemetry read from the
    already-held terminal obs). Shared by play_match and play_match_sequential.

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
    return obs["current"]["result"], classify_end(obs, seat_a)


def play_match(a: Agent, b: Agent, n_games: int, ends_out: list | None = None) -> tuple[int, int, int]:
    """Play n_games between a and b, swapping seats each game. Returns (wins_a, losses_a, draws).
    `ends_out`: optional list that receives each game's classify_end dict (a-relative), in game
    order — opt-in win-mode telemetry; the return shape is unchanged for existing callers."""
    wins = losses = draws = 0
    for i in range(n_games):
        seat_a = i % 2                       # alternate which player index a occupies
        r, end = _play_one(a, b, seat_a)
        if ends_out is not None:
            ends_out.append(end)
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


# --- rich per-matchup telemetry (--mode field) --------------------------------------------------
# A PARALLEL collection path to play_match: same seat-swapped loop, but it also records illegal
# actions, first/second (true coin-flip), prizes, length, and action usage. play_match / _play_one
# stay byte-for-byte unchanged so the frozen callers (eval_sb3, record_battle, gauntlet, sprt) are
# untouched. It reuses win_mode.classify_end + summarize_ends and wilson() — no new metric math.

def _validate_selection(sel, sd):
    """Mirror the engine's OWN select validation so an illegal action can be COUNTED and
    CATEGORIZED, instead of collapsing into the bare IndexError battle_select raises (the engine
    exposes no per-move errorType — that pair is deck-legality-only on battle_start's StartData).
    Categories are exactly the engine's own codes: cg.game.Select's list[int] pre-check, then the
    search_step validation (4=count / 5=range / 6=duplicate). This is the COMPLETE illegal space
    because obs.select.option is pre-filtered to legal options, so any in-range, non-duplicate
    index of an allowed count is legal. `sd` is the raw obs["select"] dict; returns None if legal,
    else the category string."""
    if not (isinstance(sel, list) and all(isinstance(x, int) for x in sel)):
        return "type"                                   # cg.game.Select's list[int] pre-check
    if not (sd["minCount"] <= len(sel) <= sd["maxCount"]):
        return "count"                                  # search_step error 4
    n_opt = len(sd["option"])
    if any(not (0 <= x < n_opt) for x in sel):
        return "range"                                  # search_step error 5
    if len(set(sel)) != len(sel):
        return "duplicate"                              # search_step error 6
    return None


def _usage_label(o, opt, yi):
    """Compact, stable action label for the usage histogram. Reuses sb3_features._resolve_option
    (the positional Option->card dereference) + tcg_model.card_table / sb3_features.attack_table
    for runtime names (no hardcoded card list — all via all_card_data()). Form
    '{OptionType}:{card-or-attack}', or the bare OptionType for card-less options (END/YES/NO/
    NUMBER)."""
    from sb3_features import _resolve_option, attack_table   # lazy: only field mode needs it
    from cg.api import OptionType
    pid, _pp, tid, _tp, aid = _resolve_option(o, opt, yi)
    ot = OptionType(opt.type).name
    if aid is not None:
        atk = attack_table.get(aid)
        return f"{ot}:{getattr(atk, 'name', None) or aid}"
    if pid is not None:
        c = tcg_model.card_table.get(pid)
        return f"{ot}:{getattr(c, 'name', None) or pid}"
    return ot


def _play_field_one(a: Agent, b: Agent, seat_a: int, usage_a: Counter, usage_b: Counter) -> dict:
    """One telemetry battle (see play_field_pair). Mutates usage_a/usage_b (keyed by the agent that
    made each pick). Returns the rich a-relative per-game record. Unlike _play_one, an illegal
    action is pre-validated and RECORDED (not crashed on) — the game is then marked error and
    excluded from W/L/D, but its selects still count toward the illegal-action rate."""
    from cg.api import to_observation_class                  # lazy: only field mode needs it
    decks = [None, None]
    decks[seat_a] = a.deck
    decks[1 - seat_a] = b.deck
    obs, start = battle_start(decks[0], decks[1])
    if start.errorPlayer >= 0:
        raise ValueError(f"deck rejected: errorType={start.errorType} player={start.errorPlayer}")

    # Tallies keyed by SEAT index (0/1); mapped to a/b at the end. usage/agents route by seat too.
    selects = [0, 0]
    illegal = [0, 0]
    illegal_cat = [Counter(), Counter()]
    usage = {seat_a: usage_a, 1 - seat_a: usage_b}
    agents = {seat_a: a, 1 - seat_a: b}
    compute_s = 0.0
    error_seat = error_cat = None

    while obs["current"]["result"] < 0:
        yi = obs["current"]["yourIndex"]
        actor = agents[yi]
        t0 = time.perf_counter()
        chosen = actor.fn(obs)
        compute_s += time.perf_counter() - t0
        selects[yi] += 1
        cat = _validate_selection(chosen, obs["select"])
        if cat is not None:
            illegal[yi] += 1
            illegal_cat[yi][cat] += 1
            error_seat, error_cat = yi, cat
            break
        try:                                                # usage decode must never break a battle
            o = to_observation_class(obs)
            for k in chosen:
                usage[yi][_usage_label(o, o.select.option[k], yi)] += 1
        except Exception:
            usage[yi]["<decode_error>"] += 1
        t1 = time.perf_counter()
        try:
            obs = battle_select(chosen)
            compute_s += time.perf_counter() - t1
        except IndexError:                                  # backstop: validator passed, engine didn't
            compute_s += time.perf_counter() - t1
            illegal[yi] += 1
            illegal_cat[yi]["engine"] += 1
            error_seat, error_cat = yi, "engine"
            break
        except ValueError as e:
            if "battle_ptr" in str(e):
                raise                                       # engine corruption, not an agent action
            compute_s += time.perf_counter() - t1
            illegal[yi] += 1
            illegal_cat[yi]["engine"] += 1
            error_seat, error_cat = yi, "engine"
            break

    battle_finish()

    def _side(seat):
        return {"selects": selects[seat], "illegal": illegal[seat],
                "illegal_cat": dict(illegal_cat[seat])}

    if error_seat is not None:
        return {"error": True, "win": None, "seat_a": seat_a, "compute_s": compute_s,
                "error_seat": ("a" if error_seat == seat_a else "b"), "error_cat": error_cat,
                "a": _side(seat_a), "b": _side(1 - seat_a)}

    cur = obs["current"]
    fp = cur.get("firstPlayer", -1)
    rec = dict(classify_end(obs, seat_a))                   # win, mode, prize_diff, *_prizes_left, turns...
    rec.update(error=False, result=cur["result"], seat_a=seat_a,
               went_first=(None if fp not in (0, 1) else (fp == seat_a)),
               compute_s=compute_s, a=_side(seat_a), b=_side(1 - seat_a))
    return rec


def play_field_pair(a: Agent, b: Agent, n_games: int):
    """Telemetry twin of play_match: n seat-swapped games that ALSO collect a rich per-game record
    + per-agent action-usage Counters, WITHOUT touching play_match's frozen 3-tuple contract.
    Returns (wins_a, losses_a, draws, records, usage_a, usage_b). Records are a-relative."""
    wins = losses = draws = 0
    records = []
    usage_a, usage_b = Counter(), Counter()
    for i in range(n_games):
        rec = _play_field_one(a, b, i % 2, usage_a, usage_b)
        records.append(rec)
        if rec["error"]:
            pass                                             # excluded from W/L/D; counts in illegal-rate
        elif rec["win"] == 1:
            wins += 1
        elif rec["win"] == -1:
            losses += 1
        else:
            draws += 1
        print(f"    {a.name} vs {b.name} game {i + 1}/{n_games}: W/L/D={wins}/{losses}/{draws}",
              end="\r", flush=True)
    print(" " * 72, end="\r")
    return wins, losses, draws, records, usage_a, usage_b


def summarize_matchup(name_a, name_b, w, l, d, records, usage_a, usage_b, top_k=12) -> dict:
    """Roll one play_field_pair result into the per-pairing telemetry block. Reuses wilson() for
    EVERY rate and win_mode.summarize_ends for the terminal-obs end block (mode histograms /
    prize_diff means / turns_mean). Counts (not rates) only where a proportion has no denominator."""
    valid = [r for r in records if not r["error"]]
    dec = w + l
    p, lo, hi = wilson(w, dec)
    es = summarize_ends(valid)                              # valid recs are supersets of classify_end dicts

    def _split(want_first):
        ww = sum(1 for r in valid if r["win"] == 1 and r["went_first"] is want_first)
        ll = sum(1 for r in valid if r["win"] == -1 and r["went_first"] is want_first)
        return (ww, ll, *wilson(ww, ww + ll))

    def _illegal(side):
        sel = sum(r[side]["selects"] for r in records)
        ill = sum(r[side]["illegal"] for r in records)
        cats = Counter()
        for r in records:
            cats.update(r[side]["illegal_cat"])
        pr, plo, phi = wilson(ill, sel)
        return {"selects": sel, "illegal": ill, "rate": pr, "lo": plo, "hi": phi, "by_cat": dict(cats)}

    prizes = [6 - r["my_prizes_left"] for r in valid]       # a's prizes taken = 6 - a's prizes left
    pdiffs = [r["prize_diff"] for r in valid]
    return {
        "a": name_a, "b": name_b, "w": w, "l": l, "d": d, "decisive": dec,
        "win_p": p, "win_lo": lo, "win_hi": hi,
        "error_games": sum(1 for r in records if r["error"]),
        "first": _split(True), "second": _split(False), "end": es,
        "prizes_taken_mean": (sum(prizes) / len(prizes)) if prizes else None,
        "prize_diff_mean": (sum(pdiffs) / len(pdiffs)) if pdiffs else None,
        "compute_s_mean": (sum(r["compute_s"] for r in valid) / len(valid)) if valid else None,
        "illegal_a": _illegal("a"), "illegal_b": _illegal("b"),
        "usage_a": usage_a.most_common(top_k), "usage_b": usage_b.most_common(top_k),
    }


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
    """Rough Bradley-Terry / Elo fit from pairwise decisive-win counts (`wins[(i, j)]` = times
    agent i beat agent j), pegging `anchor` to 0 every iteration so a checkpoint's rating is
    COMPARABLE ACROSS RUNS (mean-anchoring would make it depend on who else happened to be in the
    pool). A light L2 prior keeps saturated records (common vs random) finite. Coarse summary,
    not a calibrated rating."""
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
        r, _ = _play_one(a, b, seat_a)   # end telemetry unused in the SPRT path
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
                  "iono": deck_hash(rb_iono.my_deck),
                  "raging_bolt": deck_hash(rb_raging_bolt.my_deck),
                  "real_raging_bolt": deck_hash(rb_real_raging_bolt.my_deck),
                  "lilies_clefairy": deck_hash(rb_lilies_clefairy.my_deck)},
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
    print("checkpoints won't separate here (mirror ~50/50) at any feasible n.")
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


# --- --mode field: rule-based round-robin telemetry report --------------------------------------

def _fmt_wil(p, lo, hi) -> str:
    return f"{100 * p:5.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f})"


def _fold_side(g, win, went_first, mode, prizes, pdiff, turns, compute):
    """Fold ONE agent's perspective of a single game into its pooled overall accumulator `g`."""
    if win == 1:
        g["w"] += 1; g["win_modes"][mode] += 1
        if went_first is True:
            g["first_w"] += 1
        elif went_first is False:
            g["second_w"] += 1
    elif win == -1:
        g["l"] += 1; g["loss_modes"][mode] += 1
        if went_first is True:
            g["first_l"] += 1
        elif went_first is False:
            g["second_l"] += 1
    else:
        g["d"] += 1
    g["prizes"].append(prizes); g["pdiff"].append(pdiff)
    g["turns"].append(turns); g["compute"].append(compute)


def _accumulate_overall(agg, name_a, name_b, records):
    """Fold a pairing's a-relative records into BOTH agents' pooled overall stats. b's view is the
    mirror: win=-win, prizes taken = 6-opp_prizes_left, went_first = not a.went_first, mode
    win<->loss swapped (handled by _fold_side via the flipped win sign)."""
    for r in records:
        agg[name_a]["selects"] += r["a"]["selects"]; agg[name_a]["illegal"] += r["a"]["illegal"]
        agg[name_a]["illegal_cat"].update(r["a"]["illegal_cat"])
        agg[name_b]["selects"] += r["b"]["selects"]; agg[name_b]["illegal"] += r["b"]["illegal"]
        agg[name_b]["illegal_cat"].update(r["b"]["illegal_cat"])
        if r["error"]:
            agg[name_a]["errors"] += 1; agg[name_b]["errors"] += 1
            continue
        _fold_side(agg[name_a], r["win"], r["went_first"], r["mode"],
                   6 - r["my_prizes_left"], r["prize_diff"], r["turns"], r["compute_s"])
        wf_b = None if r["went_first"] is None else (not r["went_first"])
        _fold_side(agg[name_b], -r["win"], wf_b, r["mode"],
                   6 - r["opp_prizes_left"], -r["prize_diff"], r["turns"], r["compute_s"])


def _new_agg():
    return {"w": 0, "l": 0, "d": 0, "errors": 0, "first_w": 0, "first_l": 0,
            "second_w": 0, "second_l": 0, "prizes": [], "pdiff": [], "turns": [], "compute": [],
            "selects": 0, "illegal": 0, "illegal_cat": Counter(),
            "win_modes": Counter(), "loss_modes": Counter()}


def _overall_summary(names, agg, usage_tot, top_k) -> dict:
    """JSON-friendly per-agent overall rollup (single source for both the print + the --json-out)."""
    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None
    out = {}
    for nm in names:
        g = agg[nm]
        dec = g["w"] + g["l"]
        out[nm] = {
            "w": g["w"], "l": g["l"], "d": g["d"], "errors": g["errors"], "decisive": dec,
            "win": wilson(g["w"], dec),
            "first": wilson(g["first_w"], g["first_w"] + g["first_l"]),
            "first_n": g["first_w"] + g["first_l"],
            "second": wilson(g["second_w"], g["second_w"] + g["second_l"]),
            "second_n": g["second_w"] + g["second_l"],
            "prizes_taken_mean": _mean(g["prizes"]), "prize_diff_mean": _mean(g["pdiff"]),
            "turns_mean": _mean(g["turns"]), "compute_s_mean": _mean(g["compute"]),
            "win_modes": dict(g["win_modes"]), "loss_modes": dict(g["loss_modes"]),
            "selects": g["selects"], "illegal": g["illegal"],
            "illegal_rate": wilson(g["illegal"], g["selects"]), "illegal_by_cat": dict(g["illegal_cat"]),
            "usage_top": usage_tot[nm].most_common(top_k), "distinct_actions": len(usage_tot[nm]),
        }
    return out


def _print_matchup(s):
    err = f", {s['error_games']} ERROR" if s["error_games"] else ""
    print(f"  W/L/D = {s['w']}/{s['l']}/{s['d']}   win% = {_fmt_wil(s['win_p'], s['win_lo'], s['win_hi'])}"
          f"  (n={s['decisive']} dec, {s['d']} draws{err})")
    fw, fl, fp, flo, fhi = s["first"]
    sw, sl, sp, slo, shi = s["second"]
    print(f"    going 1st: {_fmt_wil(fp, flo, fhi)} (n={fw + fl})    "
          f"going 2nd: {_fmt_wil(sp, slo, shi)} (n={sw + sl})")
    es = s["end"]
    print(f"    end modes  win={dict(es['win_modes'])}  loss={dict(es['loss_modes'])}  "
          f"trailing_wins={es['trailing_win_n']}")
    pt, pd, tm, cm = s["prizes_taken_mean"], s["prize_diff_mean"], es["turns_mean"], s["compute_s_mean"]
    bits = [f"prizes_taken(a)={pt:.2f}" if pt is not None else "prizes_taken(a)=n/a",
            f"prize_diff(a)={pd:+.2f}" if pd is not None else "prize_diff(a)=n/a",
            f"turns={tm:.1f}" if tm is not None else "turns=n/a",
            f"{cm * 1000:.0f}ms/game" if cm is not None else "compute=n/a"]
    print("    " + "   ".join(bits))
    for who, nm, il in (("a", s["a"], s["illegal_a"]), ("b", s["b"], s["illegal_b"])):
        cat = f" {il['by_cat']}" if il["by_cat"] else ""
        print(f"    illegal[{nm}] = {il['illegal']}/{il['selects']} selects "
              f"({_fmt_wil(il['rate'], il['lo'], il['hi'])}){cat}")
    for nm, use in ((s["a"], s["usage_a"]), (s["b"], s["usage_b"])):
        print(f"    top actions [{nm}]: " + (", ".join(f"{k}×{v}" for k, v in use) or "(none)"))


def _print_matrix(names, matrix):
    col_w = 16
    print(f"\n{'=' * 78}")
    print("WIN% MATRIX — row agent vs column opponent deck (decisive; '--' = self) — DECK-CONFOUNDED")
    print("agent\\opp".ljust(14) + "".join(o[:col_w - 1].ljust(col_w) for o in names))
    for row in names:
        line = row.ljust(14)
        for col in names:
            if row == col:
                line += "--".ljust(col_w)
            else:
                w, l, d, p, lo, hi = matrix[row][col]
                line += (f"{100 * p:4.1f}({100 * lo:2.0f}-{100 * hi:2.0f})").ljust(col_w)
        print(line)


def _print_overall(names, overall):
    print(f"\n{'=' * 78}")
    print("PER-AGENT OVERALL — pooled across its pairings (DECK-CONFOUNDED aggregate):")
    for nm in names:
        o = overall[nm]
        err = f", {o['errors']} err" if o["errors"] else ""
        print(f"\n  {nm}:  win% {_fmt_wil(*o['win'])}  (n={o['decisive']} dec, {o['d']} draws{err})")
        print(f"     1st {_fmt_wil(*o['first'])} (n={o['first_n']})    "
              f"2nd {_fmt_wil(*o['second'])} (n={o['second_n']})")
        pt = f"{o['prizes_taken_mean']:.2f}" if o["prizes_taken_mean"] is not None else "n/a"
        tm = f"{o['turns_mean']:.1f}" if o["turns_mean"] is not None else "n/a"
        print(f"     prizes_taken/game={pt}  turns={tm}  "
              f"win_modes={o['win_modes']}  loss_modes={o['loss_modes']}")
        cat = f" {o['illegal_by_cat']}" if o["illegal_by_cat"] else ""
        print(f"     illegal={o['illegal']}/{o['selects']} ({_fmt_wil(*o['illegal_rate'])}){cat}")
        top = ", ".join(f"{k}×{v}" for k, v in o["usage_top"]) or "(none)"
        print(f"     top actions ({o['distinct_actions']} distinct): {top}")


def _print_field_caveats():
    print(f"\n{'-' * 78}")
    print("Caveats:")
    print("  * DECK-CONFOUNDED: each agent hardcodes its own deck, so every cell mixes agent skill")
    print("    with the deck matchup — NOT a clean skill ranking (eval-discipline). No mirror here.")
    print("  * Illegal-action rate: battle_select exposes only a bare IndexError (no errorType — that")
    print("    is deck-legality-only, on battle_start). count/range/duplicate/type mirror the engine's")
    print("    OWN checks (cg.game list[int] pre-check + search_step codes 4/5/6); the option list is")
    print("    pre-filtered to legal options, so 0 illegal is expected — it doubles as a contract-honor")
    print("    guard for the four agents. An 'engine' category would mean the backstop fired.")
    print("  * 'going 1st/2nd' = the true coin-flip firstPlayer, not seat parity (which the harness")
    print("    only alternates to cancel player-index bias).")
    print("  * Wall-clock is agent-think + engine-step per game (usage-decode excluded); win% excludes draws.")


def _field_smoke_asserts(names, summaries, matrix, overall, usage_tot):
    assert summaries, "no pairings summarized"
    for key, s in summaries.items():
        for k in ("win_p", "first", "second", "end", "illegal_a", "illegal_b",
                  "prizes_taken_mean", "compute_s_mean", "usage_a"):
            assert k in s, f"missing {k} in summary {key}"
        assert 0.0 <= s["win_p"] <= 1.0, s
        for il in (s["illegal_a"], s["illegal_b"]):
            assert il["selects"] > 0 and 0.0 <= il["rate"] <= 1.0, il
        assert s["end"]["turns_mean"] is None or s["end"]["turns_mean"] > 0, s
        assert s["compute_s_mean"] is None or s["compute_s_mean"] > 0, s
    for nm in names:
        assert overall[nm]["selects"] > 0, f"{nm} made no selects"
        assert len(usage_tot[nm]) > 0, f"{nm} recorded no action usage"
    for r in names:                                          # matrix filled consistently from both sides
        for c in names:
            if r == c:
                continue
            assert matrix[r][c][0] == matrix[c][r][1] and matrix[r][c][1] == matrix[c][r][0], (r, c)


def _dump_field_json(path, args, names, summaries, matrix, overall):
    payload = {
        "manifest": run_manifest("field", args, [], extra={"field_games": args.field_games}),
        "agents": names,
        "pairings": [{**s, "usage_a": dict(s["usage_a"]), "usage_b": dict(s["usage_b"])}
                     for s in summaries.values()],
        "matrix": {r: {c: {"w": v[0], "l": v[1], "d": v[2], "p": v[3], "lo": v[4], "hi": v[5]}
                       for c, v in row.items()} for r, row in matrix.items()},
        "overall": overall,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def run_field(args) -> int:
    """Round-robin the 4 rule-based agents with rich per-matchup telemetry: win-by-opponent-deck,
    illegal-action rate, first/second split, end-condition/prize/length breakdown, action usage.
    A Layer-1 competence read of the field (docs/reports/evaluation_harness_report.md). Every
    pairing is DECK-CONFOUNDED — each agent hardcodes its own deck (eval-discipline)."""
    # Relabel the Lucario factory's "rulebased" -> "lucario" so the matrix reads cleanly (matches
    # eval_sb3's keying); the other three already name themselves after their deck.
    pool = [Agent("lucario", rulebased_lucario.agent, rulebased_lucario.my_deck),
            make_abomasnow(), make_dragapult(), make_iono(), make_raging_bolt(),
            make_real_raging_bolt(), make_lilies_clefairy()]
    names = [ag.name for ag in pool]
    n = 3 if args.smoke else args.field_games
    n_pairs = len(pool) * (len(pool) - 1) // 2

    print(json.dumps(run_manifest("field", args, [], extra={"field_games": n, "smoke": args.smoke}),
                     indent=2))
    if not args.smoke and n < TRUST_N:
        print(f"WARNING: --field-games {n} < trustworthy floor ~{TRUST_N} "
              f"(memory eval-training-not-helping-yet); CIs will be wide.")
    print("=" * 78)
    print(f"Rule-based FIELD telemetry | {len(pool)} agents, {n_pairs} pairings | "
          f"{n} games/pairing (seat-swapped){' | SMOKE' if args.smoke else ''}")
    print("DECK-CONFOUNDED: each agent plays its OWN hardcoded deck — every cell mixes skill + deck.")
    print("=" * 78)

    summaries = {}                                           # (i,j) -> summary dict
    usage_tot = {nm: Counter() for nm in names}
    matrix = {nm: {} for nm in names}                        # matrix[row][col] = (w,l,d,p,lo,hi), row-relative
    agg = {nm: _new_agg() for nm in names}

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            print(f"\n{a.name} vs {b.name}")
            w, l, d, records, ua, ub = play_field_pair(a, b, n)
            s = summarize_matchup(a.name, b.name, w, l, d, records, ua, ub, top_k=args.top_actions)
            summaries[(i, j)] = s
            usage_tot[a.name] += ua
            usage_tot[b.name] += ub
            _print_matchup(s)
            matrix[a.name][b.name] = (w, l, d, *wilson(w, w + l))
            matrix[b.name][a.name] = (l, w, d, *wilson(l, w + l))   # mirror cell, b-relative
            _accumulate_overall(agg, a.name, b.name, records)

    overall = _overall_summary(names, agg, usage_tot, args.top_actions)
    _print_matrix(names, matrix)
    _print_overall(names, overall)
    _print_field_caveats()

    if args.json_out:
        _dump_field_json(args.json_out, args, names, summaries, matrix, overall)
        print(f"\nwrote structured telemetry -> {args.json_out}")
    if args.smoke:
        _field_smoke_asserts(names, summaries, matrix, overall, usage_tot)
        print("\nfield telemetry smoke OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["battery", "gauntlet", "sprt", "field"], default="battery",
                    help="battery = sanity battery / --checkpoints round-robin (default); "
                         "gauntlet = score one ckpt vs the frozen anchor pool -> a comparable "
                         "scalar; sprt = sequential A/B verdict (--checkpoint vs --baseline-checkpoint); "
                         "field = rule-based round-robin with rich per-matchup telemetry (no ckpts)")
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
    ap.add_argument("--field-games", type=int, default=100,
                    help="games per pairing in --mode field (seat-swapped); >=%d for tight CIs" % TRUST_N)
    ap.add_argument("--top-actions", type=int, default=12,
                    help="how many top action/card usage entries to show per agent (--mode field)")
    ap.add_argument("--json-out", default=None,
                    help="--mode field: also dump the full structured telemetry to this JSON path")
    ap.add_argument("--smoke", action="store_true",
                    help="--mode field: fast self-checking run (few games/pairing + asserts)")
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

    if args.mode == "gauntlet":
        return run_gauntlet(args)
    if args.mode == "sprt":
        return run_sprt(args)
    if args.mode == "field":
        return run_field(args)
    return run_battery(args)


if __name__ == "__main__":
    raise SystemExit(main())
