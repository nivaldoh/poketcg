#!/usr/bin/env python3
"""Eval a train_ppo.py-trained PPO ActorCritic across a DECK×OPPONENT matrix: the model plays EACH
of the 4 game decks (lucario, abomasnow, dragapult_ex, iono) against EACH of the 4 rule-based
opponents (each on its OWN deck), seat-swapped, with 95% Wilson CIs. NOTE: train_ppo's default
opponent is now this same rule-based field, so the matrix is then an IN-DISTRIBUTION eval
(deliberate — beating this field is the project goal), not a generalization test.

The matrix DIAGONAL (model on deck-X vs the rule-based agent that also plays deck-X) is a clean
MIRROR — same deck both seats — so it isolates policy skill from deck strength. OFF-DIAGONAL cells
are deck-CONFOUNDED (win-rate mixes policy skill with the deck matchup), like eval.py's gauntlet FIELD.

For each cell it also (unless `--skip-baseline`) evaluates the untrained fresh-net baseline
(`--baseline`, saved by train_ppo as {prefix}_init.pth) so a per-cell "did training help?" verdict
compares the trained CI to the baseline CI (disjoint = a real move). The fresh net is NOT ~50%: it
uses the real `auto_resolve` heuristic on every non-policy node, so it starts with an edge — success
is the trained CI clearing the baseline CI. Selection defaults to `sampled` (the honest headline for a
stochastic policy, per the eval-discipline skill); `--modes sampled,greedy` also shows argmax.

This reuses eval.py's two-deck machinery wholesale (`Agent`, `play_match`, `wilson`,
`elo_from_winrate`, and the `make_*` opponent factories) — the PPO net is wrapped as an `eval.Agent`
via train_ppo's `policy_agent`, which is deck-agnostic, so the SAME trained net plays any deck we hand
its `Agent`. The 4 model decks ARE the 4 rule-based training decks, so every eval-time card was seen
in training (no untrained/noise embedding rows — cf. the embedding caveat in ppo_model.py).

Total battles = nets × decks × opponents × modes × games (a full 4×4 both-nets sweep at 200/cell is
6400 battles — use --skip-baseline / fewer --decks/--opponents / smaller --games to trim).

    TORCHDYNAMO_DISABLE=1 python eval_ppo.py --games 200                     # full 4x4 matrix, both nets
    TORCHDYNAMO_DISABLE=1 python eval_ppo.py --games 200 --skip-baseline     # trained only (half cost)
    TORCHDYNAMO_DISABLE=1 python eval_ppo.py --decks lucario --opponents lucario --games 200  # one cell
"""

import argparse
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: the FIRST import must run the CG_LIB_PATH sys.path insert that makes
# the vendored `cg` package importable before any bare `cg` import. utils.run_smoketest_battle
# (load_deck's module) does that setup itself at module top, as do the PPO imports below; importing
# `eval` re-runs it plus the game_agents/rulebased sys.path insert (idempotent).
from utils.run_smoketest_battle import load_deck                     # noqa: E402  (model-deck CSV loader; sets CG_LIB_PATH)
from ppo_model import ActorCritic, STATE_DIM, OPTION_DIM             # noqa: E402
from train_ppo import policy_agent   # shared wrapper: SAME decision routing train + eval  # noqa: E402
from eval import (                                                    # noqa: E402
    Agent, play_match, wilson, elo_from_winrate,
    make_random, make_rulebased, make_abomasnow, make_dragapult, make_iono,
)


# Opponent registry: name -> factory returning a ready eval.Agent (policy fused with the deck it
# plays). Keys MATCH the model-deck names (DECK_NAMES) so the matrix diagonal (same name on both
# axes) is a true mirror. NOTE: eval.py names the Lucario factory `make_rulebased` (its Agent.name is
# "rulebased"); we expose it here under "lucario".
OPPONENTS = {
    "lucario": make_rulebased,       # rule-based Lucario (own Lucario deck)
    "abomasnow": make_abomasnow,     # rule-based Abomasnow (own deck)
    "dragapult_ex": make_dragapult,  # rule-based Dragapult ex (own deck)
    "iono": make_iono,               # rule-based Iono (own deck)
    "random": make_random,           # optional floor: random agent on sample_deck (NOT a game deck)
}
# Model-side decks (decks/<name>.csv). These ARE the 4 rule-based decks, so the diagonal is a mirror
# and every eval card was seen in training.
DECK_NAMES = ["lucario", "abomasnow", "dragapult_ex", "iono"]


def load_net(path, device):
    net = ActorCritic(STATE_DIM, OPTION_DIM).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()                        # no dropout/BN in the net, so eval() is behavior-neutral (conventional)
    return net


def resolve_names(spec, valid, kind):
    """'all' -> every valid name (registry order); else a comma list validated against `valid`."""
    if spec.strip() == "all":
        return list(valid)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in valid]
    if bad:
        raise SystemExit(f"unknown {kind}(s) {bad}; choose from {list(valid)} or 'all'")
    return names


def run_match(net, mode, opp, deck_name, deck, games, seed):
    """Play the PPO net (as model deck `deck`) vs one opponent Agent (its OWN deck) for `games`
    seat-swapped battles in one selection `mode`. Returns (w, l, d, p, lo, hi) — decisive-only
    Wilson CI (draws excluded from the denominator, matching eval.py / eval_mymodel.py)."""
    me = Agent(f"ppo:{deck_name}:{mode}", policy_agent(net, greedy=(mode == "greedy")), deck)
    # play_match (unlike eval_mymodel.play) does NOT reseed; reseed here so the opponent's Python-RNG
    # stream is identical across nets/modes/cells (comparable). The cg engine RNG stays unseedable, so
    # games are not bit-identical — hence still eval at n>=200 for a real CI.
    random.seed(seed)
    w, l, d = play_match(me, opp, games)
    p, lo, hi = wilson(w, w + l)
    return w, l, d, p, lo, hi


def _fmt_cell(res):
    """One matrix cell -> 'win%(lo-hi)' from a (w,l,d,p,lo,hi) tuple ('-' if missing)."""
    if res is None:
        return "-"
    _, _, _, p, lo, hi = res
    return f"{100 * p:4.1f}({100 * lo:2.0f}-{100 * hi:2.0f})"


def print_matrix(grid, deck_names, opp_names, tag, mode):
    """Print a decks×opponents win% grid. grid[deck][opp] = (w,l,d,p,lo,hi). A trailing '*' marks the
    MIRROR diagonal (model deck == opponent deck) — the clean skill read; other cells are deck-confounded."""
    col_w = 16
    print(f"\n=== {tag} | {mode} | win% (95% CI, decisive; * = mirror diagonal) ===", flush=True)
    print("model\\opp".ljust(14) + "".join(o.ljust(col_w) for o in opp_names), flush=True)
    for d in deck_names:
        row = d.ljust(14)
        for o in opp_names:
            mark = "*" if d == o else " "
            row += (_fmt_cell(grid[d].get(o)) + mark).ljust(col_w)
        print(row, flush=True)


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
                    help="trained PPO checkpoint(s) (ActorCritic state_dict from train_ppo.py); "
                         "comma list evaluates each, tagged by file basename (e.g. a curve)")
    ap.add_argument("--baseline", default=os.path.join(_REPO_ROOT, "out", "ppo_init.pth"),
                    help="untrained fresh-net checkpoint to anchor against (train_ppo saves {prefix}_init.pth)")
    ap.add_argument("--decks", default="all",
                    help="model-side decks: comma list of {lucario,abomasnow,dragapult_ex,iono} or 'all'")
    ap.add_argument("--opponents", default="lucario,abomasnow,dragapult_ex,iono",
                    help="comma list of {lucario,abomasnow,dragapult_ex,iono,random} or 'all' (adds random floor)")
    ap.add_argument("--modes", default="sampled",
                    help="comma list of {sampled,greedy}; sampled is the honest headline (default)")
    ap.add_argument("--games", type=int, default=200,
                    help="seat-swapped games per cell; >=200 for a trustworthy CI (eval-discipline). "
                         "Total battles = nets x decks x opponents x modes x games.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="eval only the trained ckpt (skip the fresh-net baseline; halves total battles)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deck_names = resolve_names(args.decks, DECK_NAMES, "deck")
    opp_names = resolve_names(args.opponents, OPPONENTS, "opponent")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad_modes = [m for m in modes if m not in ("sampled", "greedy")]
    if bad_modes:
        raise SystemExit(f"unknown mode(s) {bad_modes}; choose from ['sampled', 'greedy']")

    # Load each model deck once (decks/<name>.csv). Build each opponent Agent ONCE and reuse across
    # nets/cells — exactly like eval.run_gauntlet (agents self-reset per-game via turn-detection;
    # play_match reuses them this way). Each rule-based opponent brings its own deck.
    model_decks = {n: load_deck(os.path.join(_REPO_ROOT, "decks", f"{n}.csv")) for n in deck_names}
    opponents = [(name, OPPONENTS[name]()) for name in opp_names]

    ckpt_paths = [s.strip() for s in args.ckpt.split(",") if s.strip()]
    # Single ckpt keeps the historical "trained" tag; a list is tagged per-file (checkpoint curves).
    ckpt_tags = (["trained"] if len(ckpt_paths) == 1 else
                 [os.path.splitext(os.path.basename(p))[0] for p in ckpt_paths])
    nets = ([] if args.skip_baseline else [("baseline", args.baseline)]) + list(zip(ckpt_tags, ckpt_paths))
    total = len(nets) * len(deck_names) * len(opponents) * len(modes) * args.games
    warn = "" if args.games >= 200 else "  (WARNING: <200 decisive is below the trustworthy CI floor)"
    print(f"PPO deck×opponent matrix | decks={deck_names} | opponents={opp_names} | modes={modes} | "
          f"{args.games} games/cell | ~{total} battles{warn} | seed={args.seed} | device={device}",
          flush=True)

    # results[tag][mode][deck][opp] = (w, l, d, p, lo, hi)
    results = {}
    for tag, path in nets:
        if not os.path.exists(path):
            print(f"  {tag}: checkpoint not found at {path} — skipping", flush=True)
            continue
        net = load_net(path, device)
        results[tag] = {m: {d: {} for d in deck_names} for m in modes}
        for dname in deck_names:
            for oname, opp in opponents:
                for mode in modes:
                    t = time.time()
                    cell = run_match(net, mode, opp, dname, model_decks[dname], args.games, args.seed)
                    results[tag][mode][dname][oname] = cell
                    w, l, d, p, lo, hi = cell
                    mirror = "*" if dname == oname else " "
                    print(f"  {tag:8s} {mode:8s} {dname:12s} vs {oname:12s}{mirror}: {w}/{l}/{d}  "
                          f"win%={100 * p:5.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f})  "
                          f"[{time.time() - t:.0f}s]", flush=True)

    # Per-(tag,mode) win% grids.
    for tag in results:
        for mode in modes:
            print_matrix(results[tag][mode], deck_names, opp_names, tag, mode)

    # "Did training help?" per cell on the headline mode — each ckpt's CI vs baseline CI (disjoint = real).
    if "baseline" in results:
        head = "sampled" if "sampled" in modes else modes[0]
        for tag in results:
            if tag == "baseline":
                continue
            print(f"\n[summary] did training help? ({head} mode, {tag} vs baseline per cell; "
                  f"* mirror = clean skill read)", flush=True)
            for dname in deck_names:
                for oname, _ in opponents:
                    b = results["baseline"][head][dname].get(oname)
                    tr = results[tag][head][dname].get(oname)
                    if b is None or tr is None:
                        continue
                    bp, blo, bhi = b[3:6]
                    tp, tlo, thi = tr[3:6]
                    if tlo > bhi:
                        verdict = "CLEARS baseline"
                    elif thi < blo:
                        verdict = "REGRESSED"
                    else:
                        verdict = "overlap (indistinguishable)"
                    kind = "mirror*" if dname == oname else "confounded"
                    print(f"  {dname:12s} vs {oname:12s} [{kind:10s}]: "
                          f"baseline {100 * bp:5.1f}% ({100 * blo:.0f}-{100 * bhi:.0f}) -> "
                          f"{tag} {100 * tp:5.1f}% ({100 * tlo:.0f}-{100 * thi:.0f}) "
                          f"[Elo {_fmt_elo(tp)}] -> {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
