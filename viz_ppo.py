#!/usr/bin/env python3
"""Visual reports for the PPO eval stack — reads the EXISTING structured outputs (no engine, no
torch, no `cg`: pure stdlib + numpy + matplotlib, so it runs on a fresh clone and stays outside
the Kaggle submission boundary) and writes PNG figures plus one self-contained HTML report
(figures base64-embedded next to a brief context caption each) under out/viz/<input-stem>/.

Dependency: `pip install matplotlib` (the repo's only viz dep; numpy ships with the env).

Three subcommands, one per data source:
  train  — a train_ppo.py per-update metrics jsonl ({out_dir}/{out_prefix}_metrics.jsonl).
           Telemetry curves per the Layer-1 health guide
           (.claude/skills/eval-discipline/ppo-health-reference.md — captions paraphrase it).
  probe  — a probe_credit.py --json dump (within-game credit: drift heatmaps, per-game V
           trajectories win-vs-loss, drift-vs-length, ANOVA shares).
  eval   — an eval_ppo.py results json (written by its --json arg): deck×opponent win% matrix
           heatmap, baseline-vs-trained CI forest plot, checkpoint curves for --ckpt lists.

Schema drift is expected and tolerated: every metric is read with .get, a figure whose keys are
absent is skipped with a visible "(not logged …)" note in the report, never a crash (e.g. the
four v_* credit keys exist only in metrics jsonls written after 2026-07-02; pre-decomposition
probe jsons lack drift_win/drift_loss/games).

The report is the chart view; the input file itself is the table view (EVAL.md: never a bare
win% — figures annotate n and 95% Wilson CIs throughout, draws excluded from eval denominators).

    python viz_ppo.py train --metrics out/ppo_epsmix_metrics.jsonl
    python viz_ppo.py probe --json out/probe_credit_epsmix_winloss.json
    python viz_ppo.py eval  --json out/eval_ppo_epsmix150_g200_s11.json
    # open from WSL2:  explorer.exe "$(wslpath -w out/viz/ppo_epsmix_metrics/index.html)"
"""

import argparse
import base64
import html
import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless (WSL2) — must precede pyplot import
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.patheffects as pe        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- palette (dataviz skill reference instance, light mode; validated 2026-07-03:
# categorical set + diverging poles pass all checks; aqua/yellow are sub-3:1 on the surface,
# so every use below carries direct value annotation or a legend — the required relief). ---
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"   # categorical slots 1-3, fixed order
RED = "#e34948"                                        # diverging warm pole (slot 6, not a series color here)
WIN, LOSS = BLUE, RED                                  # outcome encoding, consistent across all figures
GOOD, CRITICAL = "#0ca30c", "#d03b3b"                  # status — verdicts only, never a series
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, HAIRLINE = "#e1e0d9", "#c3c2b7"
NEUTRAL_MID = "#f0efec"
# Win-mode (classify_end "mode") categorical encoding, shared by all win-mode figures. WIN/LOSS
# stay the outcome channel; modes get the remaining categorical slots.
MODE_ORDER = ("prize", "deck", "bench", "effect", "unknown")
MODE_COLORS = {"prize": BLUE, "deck": YELLOW, "bench": AQUA, "effect": INK2, "unknown": MUTED}
DIVERGING = LinearSegmentedColormap.from_list(
    "repo_div", [LOSS, NEUTRAL_MID, WIN]).with_extremes(bad="#eeede9")  # blue=+ / red=− poles


def _ink_for(rgba):
    """Annotation color readable on a filled cell: white on dark fills, INK on light."""
    r, g, b = rgba[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45 else INK

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": HAIRLINE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13,
    "lines.linewidth": 2.0, "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
    "legend.frameon": False,
})


# ============================================================================================
# Small numeric helpers
# ============================================================================================
def wilson(wins, n, z=1.96):
    """95% Wilson interval, (p, lo, hi). Copied from eval.wilson (grep "def wilson" eval.py —
    that stays the source of truth); duplicated so this module never imports torch/cg."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _roll_sum(a, window):
    """Trailing-window sums per index."""
    c = np.concatenate([[0], np.cumsum(a)])
    return np.array([c[i + 1] - c[max(0, i + 1 - window)] for i in range(len(a))])


def rolling_wilson(win_counts, n_counts, window):
    """Pooled trailing-window win rate per update: (p, lo, hi) arrays. Pooling (not averaging
    per-update rates) keeps the Wilson n honest when games/update varies or an arm is thin."""
    w, n = _roll_sum(win_counts, window), _roll_sum(n_counts, window)
    p = np.full(len(n), np.nan)
    lo, hi = p.copy(), p.copy()
    for i in range(len(n)):
        if n[i] > 0:
            p[i], lo[i], hi[i] = wilson(int(w[i]), int(n[i]))
    return p, lo, hi


def _updates(rows):
    """The x-axis: each row's logged update index, 1-based row order when absent (schema drift)."""
    return np.array([r.get("update", i + 1) for i, r in enumerate(rows)])


def series(rows, key):
    """(updates, values) for one metrics key, rows lacking the key skipped (schema drift)."""
    pts = [(r.get("update", i + 1), r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
    if not pts:
        return np.array([]), np.array([])
    xs, ys = zip(*pts)
    return np.asarray(xs), np.asarray(ys)


def resample_unit(v_seq, k=21):
    """Linear-interp a per-decision V sequence onto normalized game time [0,1] (None if < 2)."""
    if len(v_seq) < 2:
        return None
    return np.interp(np.linspace(0, 1, k), np.linspace(0, 1, len(v_seq)), v_seq)


def parse_update_from_tag(tag):
    """Checkpoint tag -> training update: trailing digits (ppo_epsmix150 -> 150), *_init -> 0,
    else None (caller falls back to file order)."""
    if tag.endswith("init"):
        return 0
    m = re.search(r"(\d+)$", tag)
    return int(m.group(1)) if m else None


# ============================================================================================
# Report writer — one self-contained HTML file; PNGs embedded so it survives moves/sharing.
# ============================================================================================
_CSS = """
body { margin:0; background:%(page)s; color:%(ink)s;
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.page { max-width: 1080px; margin: 0 auto; padding: 28px 20px 60px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 6px; }
.sub { color:%(muted)s; font-size: 12.5px; margin: 0 0 18px; }
.banner { background:#fdf3e0; border:1px solid #eda10055; border-radius:8px;
          padding:10px 14px; font-size:13px; margin: 0 0 18px; }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin: 0 0 22px; }
.tile { background:%(surface)s; border:1px solid rgba(11,11,11,0.10); border-radius:10px;
        padding:10px 16px; min-width:130px; }
.tile .label { color:%(ink2)s; font-size:11.5px; margin-bottom:2px; }
.tile .value { font-size:19px; font-weight:600; }
section { background:%(surface)s; border:1px solid rgba(11,11,11,0.10); border-radius:10px;
          padding:16px 18px; margin: 0 0 18px; }
.cap { color:%(ink2)s; font-size:13px; line-height:1.45; margin: 0 0 10px; }
.note { color:%(muted)s; font-size:12.5px; font-style:italic; margin: 6px 0 0; }
img { max-width:100%%; height:auto; display:block; }
.footer { color:%(muted)s; font-size:11.5px; margin-top: 8px; }
""" % {"page": PAGE, "surface": SURFACE, "ink": INK, "ink2": INK2, "muted": MUTED}


def save_fig(fig, outdir, name):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _b64(png_path):
    with open(png_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def write_report(outdir, title, subtitle, tiles, sections, banner=None):
    """tiles = [(label, value, status|None)]; sections = [(title, png_path|None, caption,
    note|None)] — a section with no png still renders (title + caption + note), which is how a
    skipped figure stays VISIBLE in the report instead of silently absent."""
    esc = html.escape
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{esc(title)}</title><style>{_CSS}</style></head><body>",
             "<div class='page'>", f"<h1>{esc(title)}</h1>", f"<p class='sub'>{esc(subtitle)}</p>"]
    if banner:
        parts.append(f"<div class='banner'>&#9888; {esc(banner)}</div>")
    if tiles:
        parts.append("<div class='tiles'>")
        for label, value, status in tiles:
            color = {"good": GOOD, "bad": CRITICAL}.get(status, INK)
            mark = {"good": "&#10003; ", "bad": "&#10007; "}.get(status, "")
            parts.append(f"<div class='tile'><div class='label'>{esc(label)}</div>"
                         f"<div class='value' style='color:{color}'>{mark}{esc(str(value))}</div></div>")
        parts.append("</div>")
    for sec_title, png, caption, note in sections:
        parts.append(f"<section><h2>{esc(sec_title)}</h2><p class='cap'>{esc(caption)}</p>")
        if png:
            parts.append(f"<img src='{_b64(png)}' alt='{esc(sec_title)}'>")
        if note:
            parts.append(f"<p class='note'>{esc(note)}</p>")
        parts.append("</section>")
    parts.append(f"<p class='footer'>generated by viz_ppo.py — the input file is the table view "
                 f"(every plotted number is a JSON read away)</p></div></body></html>")
    path = os.path.join(outdir, "index.html")
    with open(path, "w") as f:
        f.write("".join(parts))
    return path


def _outdir(args, input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    d = args.out_dir or os.path.join(_REPO_ROOT, "out", "viz", stem)
    os.makedirs(d, exist_ok=True)
    return d


def _section_adder(outdir, sections, missing_note=None):
    """The one report-section builder (train/probe/eval): add(fig_id, title, fig, note=None)
    saves the figure (when there is one) and appends (title, png, caption, note); a None fig
    keeps the section with `missing_note` so the report says WHY a figure is absent."""
    def add(fig_id, title, fig, note=None):
        png = save_fig(fig, outdir, fig_id) if fig is not None else None
        sections.append((title, png, CAPTIONS[fig_id.split("__")[0]],
                         note if note is not None else (None if png else missing_note)))
    return add


def _n_figs(sections):
    return sum(1 for _, png, _, _ in sections if png)


def _finish(outdir, report_path, n_figs):
    print(f"[viz] {n_figs} figure(s) -> {outdir}")
    print(f"[viz] report: {report_path}")
    print(f'[viz] open:   explorer.exe "$(wslpath -w {report_path})"')


# ============================================================================================
# Captions — the single home for every figure's context text. Health reads paraphrase
# .claude/skills/eval-discipline/ppo-health-reference.md (per-metric guide + bug->signal map),
# the v_spread_stats docstring (ppo_model.py), probe_credit.py's [interpretation] block, and the
# eval_ppo.py module docstring. Trend language by design — imported absolute bands have misled
# here before; the only hard numbers are the repo's own invariants and measured observations.
# ============================================================================================
CAPTIONS = {
    "win_rate":
        "Per-update win fraction (dots) and a rolling win rate pooled over the trailing window "
        "(line, 95% Wilson band). The overall learning signal: healthy = a rising trend; flat at "
        "the fresh-net base rate is the plateau under investigation. Draws count as non-wins "
        "(measured ~0 in this env).",
    "matchup_opponent":
        "Rolling win rate split by opponent arm (from per_game). The reward-starvation read: an "
        "arm at ~0 wins across many updates contributes no learning signal at all (loss-only "
        "batches are ~pure noise). 'rulebased' is the Lucario-deck opponent; '@eps' arms are the "
        "per-decision-weakened curriculum versions that supply wins. Arms are thin (~2 "
        "games/arm/update) so bands are wide — trust trends, not wiggles.",
    "matchup_deck":
        "Rolling win rate split by the deck the learner played, pooled over opponents. A deck "
        "pinned at the floor while others move means the learner is not converting that deck's "
        "positions — cross-check against the eval matrix row for the same deck.",
    "entropy":
        "entropy_norm = entropy / ln(n_options), in (0,1] — the interpretable entropy across "
        "variable action counts. Toward 0 = policy collapse; sustained ~1.0 early = policy not "
        "learning; late RE-expansion with a flat win rate = the erosion signature (entropy bonus "
        "eating the learned policy) [measured-here].",
    "policy_step":
        "Policy-update trend signals — logged, not thresholded (the blog's kl<0.02 band was "
        "already falsified by a healthy run here). approx_kl (k3 estimator) exploding on the "
        "very FIRST update points at input scaling / weight init, not learning dynamics. "
        "clip_frac stuck at ~0 = policy barely moving (the broken run sat at ~0; the improved "
        "run 5-12%). ratio_max spikes were measured NOT to drive damage here — don't re-suspect "
        "without new evidence.",
    "critic":
        "THE critic gate is ev_true_pre: HOLDOUT explained variance of act-time (pre-update) V "
        "against the true outcome G — healthy = well above 0, sustained (exactly 0.0 is the "
        "no-contrast-batch sentinel, not a score). explained_var is IN-SAMPLE on the just-fitted "
        "batch: never accept it as critic health — it read 0.4-0.5 while holdout was -0.013 "
        "(2026-07-02 incident); divergence between the two curves is that memorization trap. "
        "value_mean pinned at a constant = saturated critic; value_loss is trend-only (same trap).",
    "grad_norms":
        "Pre-clip gradient-norm maxima per update, split by head. Per-head imbalance = "
        "loss-scale imbalance (value ~40x policy was observed alongside the broken critic); the "
        "global max just tracks the louder head.",
    "adv_return":
        "Advantage and value-target scale checks. adv_std_raw ~ 0 = a no-contrast batch (all "
        "outcomes identical — expected under reward starvation); wild scale drift = GAE/target "
        "bug. return_mean with its min-max envelope: TD(lambda) targets outside ~[-1,1] (dashed "
        "lines) mean a broken episode-boundary reset (GAE leaking across games) — an exact "
        "invariant, a violation is a bug, full stop.",
    "credit":
        "Within-game credit probe (v_spread_stats, logged since 2026-07-02). Matchup-only-critic "
        "signature = v_within_frac ~ 0 AND v_drift_signed ~ 0; a state-aware critic shows "
        "v_drift_signed > 0 (V drifts toward the realized outcome). Drift is the DISCRIMINATING "
        "readout: ev_true_pre structurally favors game-constant V and can never reward correct "
        "state-awareness (THEORY-REVIEW derivation in the ppo_model.py docstring).",
    "win_modes":
        "HOW games end (engine RESULT-log reason via win_mode.classify_end): prize = took all "
        "prizes; deck = opponent started a turn with 0 deck; bench = no Pokémon left in play; "
        "effect = a card text ended it. The DISCOVERY-2 standing readout (2026-07-03): this "
        "policy's wins vs the field were suspected to be largely deck/bench-out wins while "
        "TRAILING on prizes — the bottom panel (share of wins with prize_diff < 0 at the end) "
        "measures exactly that. Deck/bench dominance means the policy wins the attrition race, "
        "not the prize race — credit signals paying for prizes pay the wrong currency. Arms are "
        "thin per update; trust the rolling trend, not wiggles (telemetry, not an eval claim).",
    # --- probe (win-mode additions) ---
    "drift_by_mode":
        "Per-game raw V drift (2nd-half − 1st-half mean V) grouped by HOW the game ended, wins "
        "(blue) vs losses (red), per opponent column + pooled. The win-mode-conditioned credit "
        "read: if V rises mainly in deck/bench-mode games regardless of outcome, the 'clock' "
        "trend is attrition-race tracking; a critic with real win-mode-aligned credit should "
        "separate win from loss WITHIN each mode. n annotations are per class-mode bucket — "
        "buckets are small; ±SE bars, telemetry not eval.",
    "end_margin":
        "Terminal margins from classify_end: prize_diff at game end (positive = learner ahead; "
        "win mass at NEGATIVE prize_diff = prize-trailing wins, the deck-out-exploit anatomy — "
        "e.g. the observed learner win at 1-vs-5 prizes taken) and game length in engine turns "
        "(plies). Compare loss vs win length: deck-out wins need long games, so a length gap "
        "between win modes feeds the CLOCK V-trend interpretation.",
    # --- eval (win-mode additions) ---
    "end_modes":
        "Per-cell end-mode histograms from the eval matrix's parallel end_stats block: learner "
        "wins by mode (left, trailW = wins while behind on prizes) and losses by mode (right). "
        "Reads the flaw geography cell by cell: where wins are deck/bench-mode the policy is "
        "exploiting the field's attrition weakness, not out-playing it on prizes; where losses "
        "are all prize-mode the opponent simply races past. Mode shares inherit the cell's n — "
        "below n=200 treat as telemetry (eval-discipline).",
    # --- probe ---
    "drift_heatmap":
        "Half-split V drift per cell (learner deck x opponent); color = SIGN (blue positive, red "
        "negative), not good/bad. v_drift_signed = mean (2nd-half V - 1st-half V) x G: > 0 = "
        "state-aware credit. The win/loss decomposition disambiguates a negative cell: drift_win "
        "< 0 = V falls during wins (bug-like); drift_loss > 0 = V RISES DURING LOSSES — the "
        "critic tracking its own progress instead of win probability (the vs-lucario finding). "
        "The init checkpoint row is the noise floor; the color scale is shared across all panels "
        "so rows compare directly.",
    "v_traj":
        "Act-time V over normalized game time, one thin line per recorded game (wins blue, "
        "losses red), bold = class mean. Healthy credit = win curves rising toward +1 and loss "
        "curves falling toward -1 as the game resolves. Both classes drifting UP together is the "
        "outcome-independent progress trend that inflates the GAE baseline in winnable cells "
        "(work_log 2026-07-03). Games with <2 decisions are skipped.",
    "drift_vs_len":
        "Per-game raw drift (2nd-half mean V - 1st-half mean V, unsigned by outcome) vs game "
        "length, colored by outcome. The length-confound control: trained games run ~2x longer "
        "than init, so |corr(drift, len)| >~ 0.3 flags a length-driven drift artifact rather "
        "than genuine within-game credit. Annotated r = pooled corr_drift_len.",
    "within_frac":
        "Decision-weighted share of V variance that lives WITHIN games, per cell (ANOVA: within "
        "+ between = 1). ~0 = V is game-constant = a matchup lookup. Read together with drift: "
        "ev_true (legend, pooled) structurally favors game-constant V, so a low share plus zero "
        "drift is the matchup-only signature, while spread WITHOUT drift is noise, not credit.",
    # --- eval ---
    "matrix":
        "Deck x opponent win% (denominator = decisive games; 95% Wilson CI in parentheses; '~' "
        "prefix = CI includes 50%). The OUTLINED diagonal is a same-deck MIRROR — the clean "
        "skill read isolating policy from deck strength; off-diagonal cells are deck-confounded "
        "(win% mixes policy skill with the deck matchup). Blue = above 50%, red = below.",
    "verdicts":
        "Per-cell 'did training help?': untrained fresh-net baseline CI (gray) vs trained CI "
        "(blue). Disjoint CIs = a real move (clears / regressed); overlap = indistinguishable — "
        "never compare point estimates across overlapping CIs. The fresh net is NOT ~50%: it "
        "keeps the auto_resolve heuristic on non-policy nodes, so it starts with an edge. "
        "* marks mirror cells (the clean skill read).",
    "ckpt_curve":
        "Win% vs training update, one eval per checkpoint (95% Wilson bands). Per-opponent "
        "panels pool the 4 learner decks; the mirror panel pools only same-deck cells — the "
        "clean skill read. Gray band = the untrained baseline. Checkpoint-selection discipline: "
        "pick the eval-best checkpoint, never _final — curves rise then erode.",
}


# ============================================================================================
# train — figures over a train_ppo metrics jsonl
# ============================================================================================
def _arm_counts(rows, keyfn):
    """{key: (wins[U], n[U])} from per_game; keyfn(deck, opp) -> group key."""
    U = len(rows)
    arms = {}
    for i, r in enumerate(rows):
        for rec in r.get("per_game") or []:
            d, o, g = rec[0], rec[1], rec[2]
            k = keyfn(d, o)
            w, n = arms.setdefault(k, (np.zeros(U), np.zeros(U)))
            n[i] += 1
            if g > 0:
                w[i] += 1
    return arms


def fig_win_rate(rows, window):
    xs = _updates(rows)
    wins = np.array([r.get("wins", 0) for r in rows], dtype=float)
    ns = np.array([max(r.get("games", 0), 1) for r in rows], dtype=float)
    p, lo, hi = rolling_wilson(wins, ns, window)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.scatter(xs, wins / ns, s=9, color=HAIRLINE, zorder=2)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.13, linewidth=0)
    ax.plot(xs, p, color=BLUE, zorder=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("update")
    ax.set_ylabel("win rate")
    ax.set_title(f"win rate (rolling {window}-update pool)")
    return fig


def fig_matchup(rows, window, by):
    """Small multiples of rolling win rate: by='opponent' -> panel per opponent base name with
    one line per @eps arm; by='deck' -> panel per learner deck (single line)."""
    if by == "opponent":
        arms = _arm_counts(rows, lambda d, o: (o.split("@")[0], o.split("@")[1] if "@" in o else ""))
        panels = list(dict.fromkeys(k[0] for k in arms))                  # first-appearance order
        eps_levels = sorted({k[1] for k in arms}, key=lambda e: float(e or 0))
        eps_color = {}                                                    # entity = eps level, fixed slot order
        for e, c in zip(eps_levels, (BLUE, AQUA, YELLOW)):
            eps_color[e] = c
        dropped = eps_levels[3:]                                          # no 4th hue — folded out, noted
    else:
        arms = _arm_counts(rows, lambda d, o: (d, ""))
        panels = list(dict.fromkeys(k[0] for k in arms))
        eps_color, dropped = {"": BLUE}, []
    if not arms:
        return None, None
    xs = _updates(rows)
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 2.6 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, panels):
        for (base, eps), (w, n) in arms.items():
            if base != name or eps not in eps_color:
                continue
            p, lo, hi = rolling_wilson(w, n, window)
            ax.fill_between(xs, lo, hi, color=eps_color[eps], alpha=0.12, linewidth=0)
            ax.plot(xs, p, color=eps_color[eps])
        ax.set_ylim(0, 1)
        ax.set_title(name)
    for ax in axes[-1]:
        ax.set_xlabel("update")
    for ax in axes[:, 0]:
        ax.set_ylabel("win rate")
    if by == "opponent" and len(eps_color) > 1:
        handles = [plt.Line2D([], [], color=c, label=("full strength" if e == "" else f"@{e}"))
                   for e, c in eps_color.items()]
        fig.legend(handles=handles, loc="upper right", ncol=len(handles))
    fig.suptitle(f"win rate by {by} (rolling {window}-update pool)", y=1.02)
    fig.tight_layout()
    note = (f"eps arms beyond 3 hues folded out of the figure: {sorted(dropped)}"
            if dropped else None)
    return fig, note


def fig_entropy(rows):
    xs, ys = series(rows, "entropy_norm")
    if not len(xs):
        return None
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(xs, ys, color=BLUE)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("update")
    ax.set_ylabel("entropy_norm")
    ax.set_title("normalized policy entropy")
    return fig


def _stack_panels(rows, specs, title):
    """Stacked single-metric panels sharing x (never dual-axis). specs = [(key, log_y)]."""
    present = [(k, lg) for k, lg in specs if len(series(rows, k)[0])]
    if not present:
        return None
    fig, axes = plt.subplots(len(present), 1, figsize=(9, 2.0 * len(present)), sharex=True,
                             squeeze=False)
    for ax, (key, log_y) in zip(axes.flat, present):
        xs, ys = series(rows, key)
        ax.plot(xs, ys, color=BLUE)
        if log_y and np.min(ys) > 0:
            ax.set_yscale("log")
        ax.set_ylabel(key)
    axes.flat[-1].set_xlabel("update")
    axes.flat[0].set_title(title)
    fig.tight_layout()
    return fig


def fig_policy_step(rows):
    return _stack_panels(rows, [("approx_kl", False), ("clip_frac", False), ("ratio_max", True)],
                         "policy-update signals")


def fig_critic(rows):
    x1, ev_h = series(rows, "ev_true_pre")
    x2, ev_i = series(rows, "explained_var")
    if not len(x1) and not len(x2):
        return None
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    ax = axes[0]
    if len(x1):
        ax.plot(x1, ev_h, color=BLUE, label="ev_true_pre (holdout)")
    if len(x2):
        ax.plot(x2, ev_i, color=YELLOW, label="explained_var (in-sample)")
    ax.axhline(0, color=HAIRLINE, linewidth=1)
    ax.set_ylabel("explained var")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8)  # above the axis, clear of data
    for ax, key in ((axes[1], "value_mean"), (axes[2], "value_loss")):
        xs, ys = series(rows, key)
        if len(xs):
            ax.plot(xs, ys, color=BLUE)
            if key == "value_loss" and np.min(ys) > 0:
                ax.set_yscale("log")
        ax.set_ylabel(key)
    axes[2].set_xlabel("update")
    axes[0].set_title("critic health", loc="left")   # short + left so the above-axis legend fits
    fig.tight_layout()
    return fig


def fig_grad_norms(rows):
    keys = [("grad_norm_policy_max", "policy head", BLUE),
            ("grad_norm_value_max", "value head", AQUA),
            ("grad_norm_max", "global", YELLOW)]
    plotted = False
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ymin = np.inf
    for key, label, color in keys:
        xs, ys = series(rows, key)
        if len(xs):
            ax.plot(xs, ys, color=color, label=label)
            ymin = min(ymin, float(np.min(ys)))
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    if ymin > 0:
        ax.set_yscale("log")
    ax.set_xlabel("update")
    ax.set_ylabel("pre-clip grad-norm max")
    ax.set_title("gradient norms by head")
    ax.legend(ncol=3)
    return fig


def fig_adv_return(rows):
    x1, astd = series(rows, "adv_std_raw")
    x2, rmean = series(rows, "return_mean")
    if not len(x1) and not len(x2):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
    ax = axes[0]
    if len(x1):
        ax.plot(x1, astd, color=BLUE, label="adv_std_raw")
    xm, am = series(rows, "adv_mean_raw")
    if len(xm):
        ax.plot(xm, am, color=AQUA, label="adv_mean_raw")
    ax.axhline(0, color=HAIRLINE, linewidth=1)
    ax.set_ylabel("advantage (raw)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax = axes[1]
    if len(x2):
        ax.plot(x2, rmean, color=BLUE, label="return_mean")
        xlo, rlo = series(rows, "return_min")
        xhi, rhi = series(rows, "return_max")
        if len(xlo) == len(x2) and len(xhi) == len(x2):
            ax.fill_between(x2, rlo, rhi, color=BLUE, alpha=0.10, linewidth=0,
                            label="return min-max")
    for ref in (-1, 1):                       # dashed = the exact invariant bound, not a gridline
        ax.axhline(ref, color=MUTED, linewidth=1, linestyle="--")
    ax.set_ylabel("TD(lambda) targets")
    ax.set_xlabel("update")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8)  # above the axis, clear of the band
    axes[0].set_title("advantage contrast & value-target range")
    fig.tight_layout()
    return fig


def fig_credit(rows):
    x1, drift = series(rows, "v_drift_signed")
    x2, frac = series(rows, "v_within_frac")
    if not len(x1) and not len(x2):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
    if len(x1):
        axes[0].plot(x1, drift, color=BLUE)
    axes[0].axhline(0, color=HAIRLINE, linewidth=1)
    axes[0].set_ylabel("v_drift_signed")
    if len(x2):
        axes[1].plot(x2, frac, color=BLUE)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("v_within_frac")
    axes[1].set_xlabel("update")
    axes[0].set_title("within-game credit probe")
    fig.tight_layout()
    return fig


def _end_counts(rows):
    """Per-update win/loss counts by end mode + trailing-prize-win counts, from the per_game
    4th-element classify_end dicts. None when no row carries them (pre-2026-07-03 jsonl)."""
    U = len(rows)
    win_m = {m: np.zeros(U) for m in MODE_ORDER}
    loss_m = {m: np.zeros(U) for m in MODE_ORDER}
    wins, losses, trail = np.zeros(U), np.zeros(U), np.zeros(U)
    seen = False
    for i, r in enumerate(rows):
        for rec in r.get("per_game") or []:
            if len(rec) <= 3 or not isinstance(rec[3], dict):
                continue
            seen = True
            e = rec[3]
            m = e.get("mode", "unknown")
            m = m if m in MODE_ORDER else "unknown"
            if e.get("win") == 1:
                wins[i] += 1
                win_m[m][i] += 1
                trail[i] += e.get("prize_diff", 0) < 0
            elif e.get("win") == -1:
                losses[i] += 1
                loss_m[m][i] += 1
    return (win_m, loss_m, wins, losses, trail) if seen else None


def fig_win_modes(rows, window):
    """Rolling end-mode composition of wins and losses (stacked shares) + the rolling share of
    wins taken while BEHIND on prizes (prize_diff < 0), Wilson band over the rolling win pool."""
    counts = _end_counts(rows)
    if counts is None:
        return None
    win_m, loss_m, wins, losses, trail = counts
    xs = _updates(rows)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
    for ax, per_mode, tot, cls in ((axes[0], win_m, wins, "wins"),
                                   (axes[1], loss_m, losses, "losses")):
        tot_r = np.maximum(_roll_sum(tot, window), 1e-9)
        shares = {m: _roll_sum(per_mode[m], window) / tot_r for m in MODE_ORDER}
        present = [m for m in MODE_ORDER if shares[m].max() > 0]
        if present:
            ax.stackplot(xs, [shares[m] for m in present], labels=present,
                         colors=[MODE_COLORS[m] for m in present], alpha=0.85)
            ax.legend(loc="upper left", fontsize=8, ncols=len(present))
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"share of {cls}")
    axes[0].set_title(f"end-mode composition (rolling {window}-update pool)")
    p, lo, hi = rolling_wilson(trail, wins, window)
    axes[2].fill_between(xs, lo, hi, color=BLUE, alpha=0.13, linewidth=0)
    axes[2].plot(xs, p, color=BLUE)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("trailing-win share")
    axes[2].set_xlabel("update")
    fig.tight_layout()
    return fig


def cmd_train(args):
    rows = load_jsonl(args.metrics)
    if not rows:
        raise SystemExit(f"no jsonl rows in {args.metrics}")
    outdir = _outdir(args, args.metrics)
    window = max(1, min(args.window, len(rows)))

    wins = sum(r.get("wins", 0) for r in rows[-window:])
    games = sum(r.get("games", 0) for r in rows[-window:])
    p, lo, hi = wilson(wins, games)
    rd = [r["ratio_dev_e0"] for r in rows if r.get("ratio_dev_e0") is not None]
    tiles = [("updates", len(rows), None),
             ("total games", sum(r.get("games", 0) for r in rows), None),
             (f"win rate (last {window} updates)", f"{100 * p:.1f}% ({100 * lo:.0f}-{100 * hi:.0f})", None)]
    if rd:
        bad = max(rd) >= 1e-4
        tiles.append(("ratio_dev_e0 max (invariant < 1e-4)", f"{max(rd):.1e}",
                      "bad" if bad else "good"))

    sections = []
    add = _section_adder(outdir, sections, missing_note="(not logged in this run)")

    add("win_rate", "Win rate", fig_win_rate(rows, window))
    fig, note = fig_matchup(rows, window, "opponent")
    add("matchup_opponent", "Win rate by opponent arm", fig, note)
    fig, _ = fig_matchup(rows, window, "deck")
    add("matchup_deck", "Win rate by learner deck", fig)
    add("entropy", "Policy entropy", fig_entropy(rows))
    add("policy_step", "Policy-update signals", fig_policy_step(rows))
    add("critic", "Critic health", fig_critic(rows))
    add("grad_norms", "Gradient norms", fig_grad_norms(rows))
    add("adv_return", "Advantages & value targets", fig_adv_return(rows))
    add("credit", "Within-game credit (v_* keys)", fig_credit(rows))
    add("win_modes", "Win/loss end modes (per_game end telemetry)", fig_win_modes(rows, window))

    report = write_report(outdir, f"PPO training telemetry — {os.path.basename(args.metrics)}",
                          f"{len(rows)} updates | rolling window {window} | source: {args.metrics}",
                          tiles, sections)
    _finish(outdir, report, _n_figs(sections))
    return 0


# ============================================================================================
# probe — figures over a probe_credit.py --json dump
# ============================================================================================
def _ordered_cell_keys(data):
    """Union of all tags' cell keys, first-appearance order (dicts preserve the probe's
    canonical deck/opponent order — never collapse to a set, it scrambles the grid)."""
    return list(dict.fromkeys(k for t in data for k in data[t].get("cells", {})))


def _grid_axes(cells_keys):
    """First-appearance deck/opponent orders from 'deck|opp' cell keys."""
    decks = list(dict.fromkeys(k.split("|")[0] for k in cells_keys))
    opps = list(dict.fromkeys(k.split("|")[1] for k in cells_keys))
    return decks, opps


def _heat(ax, mat, decks, opps, vmax, se=None):
    """One deck x opponent heatmap on the shared symmetric diverging scale, every cell
    annotated (the relief for sub-3:1 fills; text color picked by fill luminance)."""
    ax.grid(False)
    ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap=DIVERGING, aspect="auto")
    ax.set_xticks(range(len(opps)), opps, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(decks)), decks, fontsize=8)
    for i in range(len(decks)):
        for j in range(len(opps)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            label = f"{v:+.3f}"
            if se is not None and not np.isnan(se[i, j]):
                label += f"\n±{se[i, j]:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    color=_ink_for(DIVERGING((v + vmax) / (2 * vmax))), fontsize=7.5)


def _cell_mat(cells, decks, opps, key):
    mat = np.full((len(decks), len(opps)), np.nan)
    for ck, cell in cells.items():
        d, o = ck.split("|")
        if d in decks and o in opps and cell.get(key) is not None:
            mat[decks.index(d), opps.index(o)] = cell[key]
    return mat


def fig_drift_heatmaps(data):
    tags = list(data)
    decks, opps = _grid_axes(_ordered_cell_keys(data))
    has_wl = any("drift_win" in c for t in tags for c in data[t].get("cells", {}).values())
    metrics = [("v_drift_signed", "v_drift_se")] + \
              ([("drift_win", "drift_win_se"), ("drift_loss", "drift_loss_se")] if has_wl else [])
    mats = {(t, m): _cell_mat(data[t]["cells"], decks, opps, m) for t in tags for m, _ in metrics}
    finite = [abs(v) for mat in mats.values() for v in mat.flat if not np.isnan(v)]
    if not finite:
        return None
    vmax = max(finite) or 1e-3
    fig, axes = plt.subplots(len(tags), len(metrics),
                             figsize=(3.6 * len(metrics) + 1, 0.62 * len(decks) * len(tags) + 1.6),
                             squeeze=False)
    for i, t in enumerate(tags):
        for j, (m, se_key) in enumerate(metrics):
            se = _cell_mat(data[t]["cells"], decks, opps, se_key)
            _heat(axes[i][j], mats[(t, m)], decks, opps, vmax, se=se)
            axes[i][j].set_title(f"{t}\n{m}" if j == 0 else m, fontsize=9)
    fig.suptitle("within-game V drift by cell (shared color scale)", y=1.0)
    fig.tight_layout()
    return fig


def fig_within_frac(data):
    tags = list(data)
    decks, opps = _grid_axes(_ordered_cell_keys(data))
    cell_keys = [f"{d}|{o}" for d in decks for o in opps
                 if any(f"{d}|{o}" in data[t].get("cells", {}) for t in tags)]
    if not cell_keys:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(cell_keys) + 1.4))
    slots = iter((BLUE, AQUA, YELLOW))                 # init = gray reference; slots go to trained tags
    colors = {t: (MUTED if t.endswith("init") else next(slots, MUTED)) for t in tags}
    for t in tags:
        vals = [data[t].get("cells", {}).get(k, {}).get("v_within_frac") for k in cell_keys]
        ys = np.arange(len(cell_keys))
        ev = data[t].get("pooled", {}).get("ev_true")
        label = f"{t} (pooled ev_true {ev:+.3f})" if ev is not None else t
        ax.scatter([v for v in vals if v is not None],
                   [y for y, v in zip(ys, vals) if v is not None],
                   color=colors[t], s=42, label=label, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)      # surface ring on overlapping dots
    ax.set_yticks(range(len(cell_keys)), cell_keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("v_within_frac (within-game share of V variance)")
    ax.set_title("within-game variance share by cell")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def _cell_games(data, tag, ck):
    return data[tag].get("cells", {}).get(ck, {}).get("games") or []


def fig_v_traj(data, opp, cells_filter):
    """Rows = decks (cells vs `opp`), cols = checkpoint tags; thin per-game V lines + class means."""
    tags = list(data)
    decks, _ = _grid_axes(_ordered_cell_keys(data))
    all_keys = set(_ordered_cell_keys(data))
    decks = [d for d in decks if f"{d}|{opp}" in all_keys
             and (not cells_filter or f"{d}|{opp}" in cells_filter)]
    if not decks:
        return None, 0
    fig, axes = plt.subplots(len(decks), len(tags), figsize=(4.2 * len(tags), 2.1 * len(decks)),
                             sharex=True, sharey=True, squeeze=False)
    n_skipped = 0
    K = 21
    for i, d in enumerate(decks):
        for j, t in enumerate(tags):
            ax = axes[i][j]
            groups = {1: [], -1: []}
            for g in _cell_games(data, t, f"{d}|{opp}"):
                curve = resample_unit(g.get("v_seq") or [], K)
                if curve is None:
                    n_skipped += 1
                    continue
                if g.get("G", 0) != 0:
                    groups[1 if g["G"] > 0 else -1].append(curve)
            for sign, color in ((1, WIN), (-1, LOSS)):
                for c in groups[sign]:
                    ax.plot(np.linspace(0, 1, K), c, color=color, alpha=0.10, linewidth=0.8)
            for sign, color in ((1, WIN), (-1, LOSS)):
                if groups[sign]:
                    ax.plot(np.linspace(0, 1, K), np.mean(groups[sign], axis=0), color=color,
                            linewidth=2.5,
                            path_effects=[pe.Stroke(linewidth=4, foreground=SURFACE), pe.Normal()])
            ax.axhline(0, color=HAIRLINE, linewidth=1)
            ax.set_title(f"{d}|{opp} — {t}  (W{len(groups[1])}/L{len(groups[-1])})", fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("game time (normalized)")
    for ax in axes[:, 0]:
        ax.set_ylabel("act-time V")
    handles = [plt.Line2D([], [], color=WIN, label="wins"),
               plt.Line2D([], [], color=LOSS, label="losses")]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(f"per-game V trajectories vs {opp}", y=1.01)
    fig.tight_layout()
    return fig, n_skipped


def fig_drift_vs_len(data):
    tags = list(data)
    if not any(_cell_games(data, t, ck) for t in tags for ck in data[t].get("cells", {})):
        return None
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 3.4), sharex=True,
                             sharey=True, squeeze=False)
    for ax, t in zip(axes.flat, tags):
        for sign, color, label in ((1, WIN, "win"), (-1, LOSS, "loss")):
            pts = [(g["n_decisions"], g["drift_raw"]) for ck in data[t].get("cells", {})
                   for g in _cell_games(data, t, ck) if sign * g.get("G", 0) > 0]
            xs, ys = zip(*pts) if pts else ([], [])
            ax.scatter(xs, ys, s=13, alpha=0.45, color=color, label=label, linewidths=0)
        ax.axhline(0, color=HAIRLINE, linewidth=1)
        corr = data[t].get("pooled", {}).get("corr_drift_len")
        ax.set_title(f"{t}" + (f"  (pooled r={corr:+.2f})" if corr is not None else ""), fontsize=9.5)
        ax.set_xlabel("decisions in game")
    axes.flat[0].set_ylabel("raw drift (2nd-half V − 1st-half V)")
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("drift vs game length", y=1.02)
    fig.tight_layout()
    return fig


def _gather_by_mode(data, tag, opp):
    """{(cls, mode): [drift_raw, ...]} over the tag's per-game records with an `end` dict;
    opp=None pools all cells, else only cells vs that opponent. Draws excluded."""
    acc = {}
    for ck in data[tag].get("cells", {}):
        if opp is not None and ck.split("|")[1] != opp:
            continue
        for g in _cell_games(data, tag, ck):
            e = g.get("end")
            if not e or g.get("G", 0) == 0:
                continue
            cls = "win" if g["G"] > 0 else "loss"
            m = e.get("mode", "unknown")
            acc.setdefault((cls, m if m in MODE_ORDER else "unknown"), []).append(g["drift_raw"])
    return acc


def fig_drift_by_mode(data):
    """Mean per-game raw drift ± SE grouped by end mode, wins vs losses; one panel per
    opponent column + pooled, rows = checkpoint tags."""
    tags = list(data)
    if not any(_gather_by_mode(data, t, None) for t in tags):
        return None
    opps = list(dict.fromkeys(k.split("|")[1] for k in _ordered_cell_keys(data)))
    cols = opps + ["pooled"]
    fig, axes = plt.subplots(len(tags), len(cols),
                             figsize=(3.3 * len(cols) + 0.6, 2.8 * len(tags)),
                             sharey=True, squeeze=False)
    for i, t in enumerate(tags):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            acc = _gather_by_mode(data, t, None if c == "pooled" else c)
            ticks = []
            for k, m in enumerate(MODE_ORDER):
                shown = False
                for cls, color, off in (("win", WIN, -0.16), ("loss", LOSS, +0.16)):
                    vals = acc.get((cls, m))
                    if not vals:
                        continue
                    shown = True
                    mean = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
                    ax.errorbar(k + off, mean, yerr=se, fmt="o", color=color,
                                markersize=5, capsize=3, linewidth=1.4)
                    ax.annotate(f"n{len(vals)}", (k + off, mean), textcoords="offset points",
                                xytext=(0, 7), fontsize=6.5, color=MUTED, ha="center")
                if shown:
                    ticks.append((k, m))
            ax.axhline(0, color=HAIRLINE, linewidth=1)
            ax.set_xticks([k for k, _ in ticks], [m for _, m in ticks], fontsize=8)
            ax.set_xlim(-0.6, len(MODE_ORDER) - 0.4)
            ax.set_title(f"{t} | {c}", fontsize=9)
        axes[i][0].set_ylabel("raw drift")
    fig.suptitle("per-game V drift by end mode (win=blue, loss=red)", y=1.0)
    fig.tight_layout()
    return fig


def fig_end_margin(data):
    """End prize_diff distribution and game length (turns), split win/loss, pooled over cells;
    rows = checkpoint tags."""
    tags = list(data)

    def gather(t):
        pd_w, pd_l, tn_w, tn_l = [], [], [], []
        for ck in data[t].get("cells", {}):
            for g in _cell_games(data, t, ck):
                e = g.get("end")
                if not e or g.get("G", 0) == 0:
                    continue
                (pd_w if g["G"] > 0 else pd_l).append(e.get("prize_diff", 0))
                (tn_w if g["G"] > 0 else tn_l).append(e.get("turns", 0))
        return pd_w, pd_l, tn_w, tn_l

    gathered = {t: gather(t) for t in tags}
    if not any(any(lst) for tup in gathered.values() for lst in tup):
        return None
    fig, axes = plt.subplots(len(tags), 2, figsize=(9.6, 2.9 * len(tags)), squeeze=False)
    for i, t in enumerate(tags):
        pd_w, pd_l, tn_w, tn_l = gathered[t]
        ax = axes[i][0]
        ax.hist([pd_l, pd_w], bins=np.arange(-6.5, 7.5, 1), color=[LOSS, WIN],
                label=["loss", "win"])
        ax.axvline(0, color=HAIRLINE, linewidth=1)
        ax.set_title(f"{t} | end prize_diff (+ = learner ahead)", fontsize=9)
        ax.set_ylabel("games")
        ax.legend(fontsize=8)
        ax = axes[i][1]
        tmax = max(tn_w + tn_l, default=0)
        ax.hist([tn_l, tn_w], bins=np.arange(0, tmax + 4, 2), color=[LOSS, WIN],
                label=["loss", "win"])
        ax.set_title(f"{t} | game length (turns = plies)", fontsize=9)
    axes[-1][0].set_xlabel("prize_diff at game end")
    axes[-1][1].set_xlabel("turns")
    fig.tight_layout()
    return fig


def cmd_probe(args):
    with open(args.json) as f:
        data = json.load(f)
    if not data:
        raise SystemExit(f"empty probe json {args.json}")
    outdir = _outdir(args, args.json)
    cells_filter = set(s.strip() for s in args.cells.split(",") if s.strip()) if args.cells else None
    tags = list(data)
    has_games = any(_cell_games(data, t, ck) for t in tags for ck in data[t].get("cells", {}))

    tiles = [("checkpoints", " / ".join(tags), None),
             ("games/cell", next((c["n_games"] for t in tags
                                  for c in data[t].get("cells", {}).values() if "n_games" in c), "?"), None)]
    for t in tags:
        pooled = data[t].get("pooled", {})
        d, se = pooled.get("v_drift_signed"), pooled.get("v_drift_se")
        if d is not None:
            tiles.append((f"{t} pooled v_drift_signed",
                          f"{d:+.4f}" + (f" ±{se:.4f}" if se is not None else ""), None))

    sections = []
    add = _section_adder(outdir, sections)

    add("drift_heatmap", "Signed V drift by cell", fig_drift_heatmaps(data))
    no_games_note = ("(per-game records absent from this json — rerun probe_credit.py --json "
                     "with the win/loss decomposition build to store v_seq)")
    if has_games:
        opps = list(dict.fromkeys(k.split("|")[1] for k in _ordered_cell_keys(data)))
        for opp in opps:
            fig, skipped = fig_v_traj(data, opp, cells_filter)
            if fig is not None:
                note = f"({skipped} games with <2 decisions skipped)" if skipped else None
                add(f"v_traj__{opp}", f"V trajectories vs {opp}", fig, note)
        add("drift_vs_len", "Drift vs game length", fig_drift_vs_len(data))
    else:
        sections.append(("V trajectories", None, CAPTIONS["v_traj"], no_games_note))
        sections.append(("Drift vs game length", None, CAPTIONS["drift_vs_len"], no_games_note))
    end_note = ("(no `end` win-mode records in this json — rerun probe_credit.py --json with "
                "the 2026-07-03 win-mode telemetry build)")
    has_end = any(g.get("end") for t in tags for ck in data[t].get("cells", {})
                  for g in _cell_games(data, t, ck))
    if has_end:
        add("drift_by_mode", "Drift by end mode", fig_drift_by_mode(data))
        add("end_margin", "End margins (prize_diff, turns)", fig_end_margin(data))
    else:
        sections.append(("Drift by end mode", None, CAPTIONS["drift_by_mode"], end_note))
        sections.append(("End margins (prize_diff, turns)", None, CAPTIONS["end_margin"], end_note))
    add("within_frac", "Within-game variance share", fig_within_frac(data))

    report = write_report(outdir, f"Within-game credit probe — {os.path.basename(args.json)}",
                          f"checkpoints: {', '.join(tags)} | source: {args.json}",
                          tiles, sections)
    _finish(outdir, report, _n_figs(sections))
    return 0


# ============================================================================================
# eval — figures over an eval_ppo.py --json results file
# ============================================================================================
def _cell(results, tag, mode, d, o):
    """[w,l,d,p,lo,hi] or None."""
    return results.get(tag, {}).get(mode, {}).get(d, {}).get(o)


def fig_matrix(results, tag, mode, decks, opps):
    mat = np.full((len(decks), len(opps)), np.nan)
    ann = {}
    for i, d in enumerate(decks):
        for j, o in enumerate(opps):
            c = _cell(results, tag, mode, d, o)
            if c is None:
                continue
            p, lo, hi = c[3:6]
            mat[i, j] = p
            straddle = "~" if lo <= 0.5 <= hi else ""
            ann[(i, j)] = f"{straddle}{100 * p:.0f}%\n({100 * lo:.0f}-{100 * hi:.0f})"
    if not ann:
        return None
    fig, ax = plt.subplots(figsize=(1.5 * len(opps) + 2, 0.9 * len(decks) + 1.6))
    ax.grid(False)
    ax.imshow(mat, vmin=0, vmax=1, cmap=DIVERGING, aspect="auto")     # midpoint = 50%
    ax.set_xticks(range(len(opps)), opps, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(decks)), decks, fontsize=9)
    ax.set_xlabel("opponent (its own deck)")
    ax.set_ylabel("model deck")
    for (i, j), label in ann.items():
        ax.text(j, i, label, ha="center", va="center",
                color=_ink_for(DIVERGING(mat[i, j])), fontsize=8.5)
    for i, d in enumerate(decks):                                     # outline the mirror diagonal
        for j, o in enumerate(opps):
            if d == o:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, linewidth=1.6))
    ax.set_title(f"{tag} | {mode}")
    return fig


def fig_verdicts(results, tag, mode, decks, opps):
    rows = []
    for d in decks:
        for o in opps:
            b, t = _cell(results, "baseline", mode, d, o), _cell(results, tag, mode, d, o)
            if b and t:
                rows.append((d, o, b, t))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.4))
    for y, (d, o, b, t) in enumerate(rows):
        bp, blo, bhi = b[3:6]
        tp, tlo, thi = t[3:6]
        ax.plot([blo, bhi], [y - 0.16] * 2, color=HAIRLINE, linewidth=2.4)
        ax.plot(bp, y - 0.16, "o", color=MUTED, markersize=5)
        ax.plot([tlo, thi], [y + 0.16] * 2, color=BLUE, linewidth=2.4)
        ax.plot(tp, y + 0.16, "o", color=BLUE, markersize=5)
        if tlo > bhi:
            verdict, color = "▲ clears", GOOD
        elif thi < blo:
            verdict, color = "▼ regressed", CRITICAL
        else:
            verdict, color = "· overlap", MUTED
        ax.text(1.02, y, verdict, color=color, fontsize=8.5, va="center",
                transform=ax.get_yaxis_transform())
    labels = [f"{d} vs {o}" + (" *" if d == o else "") for d, o, _, _ in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0.5, color=HAIRLINE, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("win rate (95% Wilson CI, decisive games)")
    handles = [plt.Line2D([], [], color=HAIRLINE, linewidth=2.4, label="baseline (untrained)"),
               plt.Line2D([], [], color=BLUE, linewidth=2.4, label=tag)]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title(f"did training help? — {tag} vs baseline | {mode}")
    return fig


def _pooled_curve(results, tags, mode, decks, cell_pred):
    """[(update_x, p, lo, hi)] per tag, pooling w/l over cells passing cell_pred(deck, opp)."""
    pts = []
    for x, tag in tags:
        w = l = 0
        for d in decks:
            for o in results.get(tag, {}).get(mode, {}).get(d, {}):
                if cell_pred(d, o):
                    c = _cell(results, tag, mode, d, o)
                    if c:
                        w, l = w + c[0], l + c[1]
        if w + l:
            pts.append((x, *wilson(w, w + l)))
    return pts


def fig_ckpt_curve(results, tags, mode, decks, opps):
    """tags = non-baseline (tag) list; x from trailing digits, else file order."""
    xs = [parse_update_from_tag(t) for t in tags]
    ordinal = any(x is None for x in xs)
    tagged = list(zip(range(len(tags)) if ordinal else xs, tags))
    panels = [(o, lambda d, oo, o=o: oo == o) for o in opps] + \
             [("mirror diagonal (pooled)", lambda d, oo: d == oo)]
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 2.6 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    drew = False
    for ax, (title, pred) in zip(axes.flat, panels):
        pts = _pooled_curve(results, tagged, mode, decks, pred)
        if pts:
            x, p, lo, hi = map(np.asarray, zip(*pts))
            order = np.argsort(x)
            ax.fill_between(x[order], lo[order], hi[order], color=BLUE, alpha=0.13, linewidth=0)
            ax.plot(x[order], p[order], color=BLUE, marker="o", markersize=4)
            drew = True
        base = _pooled_curve(results, [(0, "baseline")], mode, decks, pred)
        if base:
            _, bp, blo, bhi = base[0]
            ax.axhspan(blo, bhi, color=HAIRLINE, alpha=0.45, linewidth=0)
            ax.axhline(bp, color=MUTED, linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel("checkpoint (file order)" if ordinal else "training update")
    for ax in axes[:, 0]:
        ax.set_ylabel("win rate")
    fig.suptitle(f"checkpoint curve | {mode} (gray = untrained baseline)", y=1.01)
    fig.tight_layout()
    return fig if drew else None


def fig_end_modes(end_stats, tag, mode, decks, opps):
    """Per-cell horizontal stacked bars: learner wins by end mode (left) and losses by end mode
    (right); the win panel annotates the trailing-prize win count."""
    grid = end_stats.get(tag, {}).get(mode, {})
    cells = [(d, o) for d in decks for o in opps if grid.get(d, {}).get(o)]
    if not cells:
        return None
    labels = [f"{d}|{o}" + ("*" if d == o else "") for d, o in cells]
    ys = np.arange(len(cells))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 0.42 * len(cells) + 1.9), sharey=True)
    for ax, key, cls in ((axes[0], "win_modes", "wins"), (axes[1], "loss_modes", "losses")):
        left = np.zeros(len(cells))
        for m in MODE_ORDER:
            vals = np.array([grid[d][o].get(key, {}).get(m, 0) for d, o in cells], dtype=float)
            if vals.max() > 0:
                ax.barh(ys, vals, left=left, color=MODE_COLORS[m], label=m, height=0.62)
            left += vals
        ax.set_title(cls, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("games")
    for y, (d, o) in enumerate(cells):
        tw = grid[d][o].get("trailing_win_n", 0)
        total_w = sum(grid[d][o].get("win_modes", {}).values())
        if total_w:
            axes[0].text(total_w + 0.3, y, f"trailW {tw}", va="center", fontsize=7,
                         color=MUTED)
    axes[0].set_yticks(ys, labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"end modes by cell — {tag} | {mode} (* = mirror)", y=1.0)
    fig.tight_layout()
    return fig


def cmd_eval(args):
    with open(args.json) as f:
        data = json.load(f)
    results = data.get("results", {})
    if not results:
        raise SystemExit(f"no results in {args.json}")
    outdir = _outdir(args, args.json)
    decks, opps, modes = data.get("decks", []), data.get("opponents", []), data.get("modes", [])
    games = data.get("games", 0)
    head = "sampled" if "sampled" in modes else (modes[0] if modes else "sampled")
    tags = list(results)
    ckpt_tags = [t for t in tags if t != "baseline"]

    banner = (f"games/cell = {games} < 200 — below the trustworthy-CI floor (eval-discipline): "
              f"plumbing/smoke readout only, not an eval claim.") if games < 200 else None
    tiles = [("games/cell", games, "good" if games >= 200 else "bad"),
             ("seed", data.get("seed", "?"), None),
             ("modes", ", ".join(modes), None),
             ("checkpoints", " / ".join(tags), None)]

    sections = []
    add = _section_adder(outdir, sections)

    for tag in tags:
        for mode in modes:
            add(f"matrix__{tag}_{mode}", f"Win% matrix — {tag} | {mode}",
                fig_matrix(results, tag, mode, decks, opps))
    end_stats = data.get("end_stats", {})
    for tag in tags:
        for mode in modes:
            fig = fig_end_modes(end_stats, tag, mode, decks, opps)
            add(f"end_modes__{tag}_{mode}", f"End modes — {tag} | {mode}", fig,
                None if fig is not None
                else "(no end_stats block — pre-win-mode-telemetry eval json)")
    if "baseline" in results:
        for tag in ckpt_tags:
            add(f"verdicts__{tag}", f"Baseline vs {tag}",
                fig_verdicts(results, tag, head, decks, opps))
    if len(ckpt_tags) > 1:
        add("ckpt_curve", "Checkpoint curve",
            fig_ckpt_curve(results, ckpt_tags, head, decks, opps))

    report = write_report(outdir, f"PPO eval matrix — {os.path.basename(args.json)}",
                          f"{games} games/cell | seed {data.get('seed', '?')} | source: {args.json}",
                          tiles, sections, banner=banner)
    _finish(outdir, report, _n_figs(sections))
    return 0


# ============================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="figures for a train_ppo metrics jsonl")
    p.add_argument("--metrics", required=True, help="out/<prefix>_metrics.jsonl from train_ppo.py")
    p.add_argument("--window", type=int, default=25,
                   help="rolling pool width in updates (per-arm samples are thin: ~2 "
                        "games/arm/update in the 8-arm curriculum runs)")
    p.add_argument("--out-dir", default=None, help="default: out/viz/<input-stem>/")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("probe", help="figures for a probe_credit.py --json dump")
    p.add_argument("--json", required=True, help="out/probe_credit_*.json")
    p.add_argument("--cells", default=None,
                   help="comma list of 'deck|opp' cells for the trajectory figures (default all)")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("eval", help="figures for an eval_ppo.py --json results file")
    p.add_argument("--json", required=True, help="out/eval_*.json written by eval_ppo.py")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
