#!/usr/bin/env python3
r"""Within-game credit probe for EXISTING PPO checkpoints (the offline arm of the diagnostic
proposed in work_log "2026-07-02 (later) — post-fix gate executed": "log per-decision V spread
inside games" before trusting GAE for more than matchup baselining).

Question: is the critic a MATCHUP-ONLY predictor — V(s) ≈ constant within each game, varying only
with (deck, opponent) identity? The standing holdout gate (ev_true_pre) cannot answer this: the
measured game-level R² of outcome on matchup identity has been ≥ every observed holdout EV, and
against game-constant realized targets any within-game V spread mechanically LOWERS that EV
(THEORY-REVIEW — derivation in ppo_model.v_spread_stats). If V is game-constant, GAE at γ=1
degenerates to A_t = λ^(T−1−t)·(G − c): only λ-recency credit, no decision-quality credit.

Per checkpoint it plays `--games` games per (model-deck × opponent) cell through the TRAINING
decision path (PPO.select_action: sampling policy, records act-time V into the buffer — the
distribution the critic is trained on; eval_ppo's act_index path never touches the critic) and
reports per cell + pooled:
  * the 4 v_spread_stats keys (see ppo_model.v_spread_stats for definitions + interpretation),
    with a per-game standard error on v_drift_signed (std/√n over ppo_model.signed_drift);
  * ev_true — per-decision explained variance of act-time V vs the game outcome G;
  * pooled 3-level decision-weighted variance decomposition of V: between-cell /
    between-game-within-cell / within-game shares (shares sum to 1);
  * the matchup-identity SHORTCUT BOUND, two variants: game-level R² of G on cell identity (the
    work-log CAVEAT's bound, over ALL games) and a decision-weighted variant (THEORY-REVIEW:
    only the decision-weighted one is apples-to-apples with the per-decision ev_true);
  * ev_cellmean_v — ev_true of the predictor that replaces every decision's V with its cell's
    mean V: ev_true ≈ ev_cellmean_v ⇒ within-cell V variation adds nothing beyond matchup identity.

MATCHUP-ONLY SIGNATURE: between-cell share dominates, within-game share ≈ 0, v_drift_signed ≈ 0
(compare against the untrained {prefix}_init.pth as the noise floor), ev_true ≈ ev_cellmean_v.
A state-aware critic shows v_drift_signed > 0 (V drifts toward the realized outcome) — that is
the discriminating readout, NOT ev_true (see above).

WIN/LOSS-CONDITIONED DECOMPOSITION (added 2026-07-03, the "Next lever" of work_log "later 4"):
a cell's v_drift_signed mixes wins and losses (drift = ΔV×G), so a negative cell can mean
falling-V-in-wins (bug-like) or rising-V-in-losses (critic tracks own progress). Each cell (and
the pooled block) now also reports drift_win / drift_loss = mean UNSIGNED half-split ΔV over
recorded wins / losses (draws excluded) ± SE with counts n_win_rec/n_loss_rec, length stats
(len_mean/len_median, corr_drift_len as the game-length control), and per-game `games` records
{G, n_decisions, drift_raw, v_seq@4dp} (drift_raw from the unrounded sequence) so slope/quartile/
length re-analyses are offline JSON reads. Reconciliation identity, asserted per cell:
v_drift_signed·n_recorded = drift_win·n_win_rec − drift_loss·n_loss_rec.

WIN-MODE TELEMETRY (added 2026-07-03, the DISCOVERY-2 standing readout): every recorded game's
record also carries the three POTENTIAL_SEQS channels (phi_seq / phi_deck_seq / phi_board_seq —
sign conventions + the offline V_adj recipe at the POTENTIAL_SEQS definition) and an "end" dict
(win_mode.classify_end: engine RESULT-log win mode + terminal prize/deck/in-play margins,
seat-relative); each cell row and the pooled block gain win_modes/loss_modes histograms,
trailing_win_n (wins while behind on prizes), prize_diff_{win,loss}_mean, turns_mean over ALL
games. phi_seq keeps its exact pre-2026-07-03 semantics but is now emitted --attrib or not.

--attrib gxi|ig|both adds feature attribution on V over the same decisions (which encode_state
feature groups drive the within-game drift; ppo_model.feature_groups + own_ids/opp_ids): per-cell
and per-opponent-column pooled group drift stats (all/win/loss), win/loss binned trajectories,
SUPER_GROUPS rollups (CLOCK/SCORE/OWN_DEV/OPP_DEV/GLOBAL_OTHER), a survivorship calibration curve
(v_excess_rise: rise of meanV over the alive-set's meanG — miscalibration vs survivorship), and
per-game group_drift_<m> records. IG-Δ (baseline = the game's first decision) is the
pre-registered arbiter over grad×input on disagreement. 'none' keeps the JSON schema identical
to pre-attrib runs (modulo the always-on win-mode/potential-seq additions above).

Caveats: collect_games pins the learner to seat 0 at n_games=1 (`ppo_seat = g % 2`, g local per
call) whereas training alternates seats — likely expectation-neutral since the engine coin-flips
firstPlayer per game independently of seat, but flagged, not verified. Win rates here are small-n
telemetry, NOT eval claims — eval_ppo.py at n≥200 remains the eval (eval-discipline). The cg
engine RNG is unseedable, so --seed only fixes the Python/torch streams.

Reused symbols (grep recipes): PPO/STATE_DIM/OPTION_DIM/v_spread_stats/signed_drift from
ppo_model (`grep -n "^def v_spread_stats\|^def signed_drift\|^class PPO" ppo_model.py`);
collect_games from train_ppo (`grep -n "def collect_games" train_ppo.py`); OPPONENTS/DECK_NAMES/
resolve_names from eval_ppo (`grep -n "^OPPONENTS\|^DECK_NAMES\|def resolve_names" eval_ppo.py`);
summarize_ends/MODE_NAMES from win_mode; load_deck from utils.run_smoketest_battle.

    TORCHDYNAMO_DISABLE=1 python probe_credit.py --ckpt out/<prefix>_init.pth,out/<ckpt>.pth --games 50
    TORCHDYNAMO_DISABLE=1 python probe_credit.py --ckpt out/<ckpt>.pth --decks lucario --opponents random --games 4  # smoke
    TORCHDYNAMO_DISABLE=1 python probe_credit.py --ckpt out/<prefix>_init.pth,out/<ckpt>.pth \
        --decks all --opponents lucario,iono --games 100 --json out/probe_credit_<tag>_winloss.json  # w/l decomposition
    TORCHDYNAMO_DISABLE=1 python probe_credit.py --ckpt out/<prefix>_init.pth,out/<ckpt>.pth \
        --decks all --opponents lucario,iono --games 100 --attrib both \
        --json out/probe_credit_<tag>_attrib.json  # feature attribution: which groups drive the drift
"""

import argparse
import json
import os
import random
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# Import order is load-bearing: the FIRST import must run the CG_LIB_PATH sys.path insert that
# makes the vendored `cg` package importable (utils.run_smoketest_battle does it at module top;
# eval_ppo/train_ppo/ppo_model below re-run it idempotently).
from utils.run_smoketest_battle import load_deck                      # noqa: E402
from ppo_model import (PPO, STATE_DIM, OPTION_DIM, N_STATE_SLOTS, v_spread_stats,  # noqa: E402
                       signed_drift, feature_groups, prize_potential, prize_potential_valid,
                       deck_potential, board_potential)
from train_ppo import collect_games                                   # noqa: E402
from eval import Agent                                                 # noqa: E402
from eval_ppo import OPPONENTS, DECK_NAMES, resolve_names             # noqa: E402
from tcg_model import random_agent                                    # noqa: E402
from win_mode import summarize_ends, MODE_NAMES                       # noqa: E402


# ============================================================================================
# Pure stat helpers — population (ddof=0) moments throughout, matching v_spread_stats (exact
# variance decompositions, no n=1 NaN paths). Guards return None (= "no contrast", printed '-').
# ============================================================================================
def _pmean(xs):
    return sum(xs) / len(xs)


def _pvar(xs):
    m = _pmean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _ev(preds, targets):
    """Explained variance 1 − popvar(target − pred)/popvar(target) — same residual-about-its-mean
    convention as ppo_model's ev_true_pre. None when the targets have no contrast."""
    vt = _pvar(targets)
    if vt <= 1e-12:
        return None
    return 1.0 - _pvar([t - p for p, t in zip(preds, targets)]) / vt


def raw_drift(v_seq):
    """Unsigned half-split ΔV = 2nd-half mean V − 1st-half mean V (balanced halves, < 2 decisions
    → 0.0) = signed_drift(v_seq, 1.0) — ppo_model.signed_drift stays the single drift definition,
    so signed_drift(v, g) == raw_drift(v) * g exactly (including the short-game 0.0 path)."""
    return signed_drift(v_seq, 1.0)


def _pmedian(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _pcorr(xs, ys):
    """Pearson correlation (population moments); None when either variable has no contrast."""
    vx, vy = _pvar(xs), _pvar(ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    mx, my = _pmean(xs), _pmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (vx * vy) ** 0.5


def drift_conditional(segs, sign):
    """(mean, se, n) of raw_drift over recorded games whose G matches `sign` (+1 → wins, −1 →
    losses; draws G=0 excluded from both). Games with < 2 decisions stay in as exact-0
    contributions (v_spread_stats' dilution convention), which keeps the reconciliation identity
    EXACT: v_drift_signed·n_recorded = drift_win·n_win_rec − drift_loss·n_loss_rec (draws and
    short games contribute 0 to both sides; counts are RECORDED games, not the w/l tallies which
    include no-record games). No matching games → (None, None, 0)."""
    ds = [raw_drift(v_seq) for v_seq, g in segs if sign * g > 0]
    if not ds:
        return None, None, 0
    return _pmean(ds), (_pvar(ds) ** 0.5) / (len(ds) ** 0.5), len(ds)


def drift_len_corr(segs):
    """Pearson correlation of (raw_drift, n_decisions) over recorded games — the in-script
    control for the game-length confound on the half-split drift (trained games run ~2× longer
    than init; a |corr| ≳ 0.3 flags a length-driven drift). None when either variable has no
    contrast or n < 2. Deeper length reads (per-game OLS slope of V on decision index) are
    offline analysis over the stored per-game v_seq."""
    if len(segs) < 2:
        return None
    return _pcorr([raw_drift(v_seq) for v_seq, _ in segs],
                  [float(len(v_seq)) for v_seq, _ in segs])


def _flat_pred_target(cells, pred_of_cell):
    """Per-decision (pred, target=G) pairs over all recorded segments; the prediction is a
    per-cell constant from pred_of_cell(cell_key) (None → use the decision's own V)."""
    preds, targets = [], []
    for key, segs in cells.items():
        c = pred_of_cell(key) if pred_of_cell else None
        for v_seq, g in segs:
            for x in v_seq:
                preds.append(x if c is None else c)
                targets.append(g)
    return preds, targets


def cell_mean_v(segs):
    """Decision-weighted mean act-time V of one cell's recorded segments."""
    vs = [x for v_seq, _ in segs for x in v_seq]
    return _pmean(vs) if vs else None


def three_level_decomposition(cells):
    """Decision-weighted ANOVA of V over {cell_key: [(v_seq, G), ...]}: shares of total variance
    at between-cell / between-game-within-cell / within-game level (they sum to 1 exactly; the
    assert is the identity check). None when V has no contrast at all."""
    all_v = [x for segs in cells.values() for v_seq, _ in segs for x in v_seq]
    n = len(all_v)
    mu = _pmean(all_v)
    ss_total = sum((x - mu) ** 2 for x in all_v)
    if ss_total / n <= 1e-12:
        return None
    ss_cell = ss_game = ss_within = 0.0
    for segs in cells.values():
        cell_vs = [x for v_seq, _ in segs for x in v_seq]
        if not cell_vs:
            continue
        mu_c = _pmean(cell_vs)
        ss_cell += len(cell_vs) * (mu_c - mu) ** 2
        for v_seq, _ in segs:
            m = _pmean(v_seq)
            ss_game += len(v_seq) * (m - mu_c) ** 2
            ss_within += sum((x - m) ** 2 for x in v_seq)
    shares = {"between_cell": ss_cell / ss_total, "between_game": ss_game / ss_total,
              "within_game": ss_within / ss_total}
    assert abs(sum(shares.values()) - 1.0) < 1e-6, f"ANOVA identity broken: {shares}"
    return {"total_var": ss_total / n, **shares}


# ============================================================================================
# Feature attribution on V (--attrib) — which encode_state feature groups drive the within-game
# V movement. Two methods over the SAME collected decisions (no extra engine rollouts):
#   gxi  grad×input at each decision (fast; can misattribute via gradient drift on features that
#        are constant within the game).
#   ig   integrated gradients along the straight line from the game's FIRST decision input to
#        each decision's input ("IG-Δ"): attributes V(x_t) − V(x_0) with a completeness identity,
#        and constant-in-game features get exactly 0 — the right semantics for "what drives the
#        within-game CHANGE". Half-split drift is shift-invariant, so per-group IG drifts
#        decompose raw_drift(v_seq) up to the (asserted) completeness error.
# Pre-registered arbiter (plan Stage A): on gxi-vs-ig disagreement, IG wins.
# ============================================================================================
SUPER_GROUPS = {
    "CLOCK": ["turn", "own_deck", "opp_deck"],
    "SCORE": ["own_prizes", "opp_prizes"],
    "OWN_DEV": ["own_hand", "own_discard", "own_board_counts", "own_status", "own_active_energy",
                "own_active_other", "own_bench_energy", "own_bench_other", "own_ids"],
    "OPP_DEV": ["opp_hand", "opp_discard", "opp_board_counts", "opp_status", "opp_active_energy",
                "opp_active_other", "opp_bench_energy", "opp_bench_other", "opp_ids"],
    "GLOBAL_OTHER": ["global_other"],
}
N_TRAJ_BINS = 10


def _bin_means(cols, nbins=N_TRAJ_BINS):
    """cols: per-decision rows [n][G] -> per-group binned means [G][nbins] on normalized game
    time (None where a short game leaves a bin empty)."""
    n, n_groups = len(cols), len(cols[0])
    sums = [[0.0] * nbins for _ in range(n_groups)]
    cnts = [0] * nbins
    for t, row in enumerate(cols):
        b = min(nbins - 1, (nbins * t) // n)
        cnts[b] += 1
        for gi in range(n_groups):
            sums[gi][b] += row[gi]
    return [[(sums[gi][b] / cnts[b]) if cnts[b] else None for b in range(nbins)]
            for gi in range(n_groups)]


class Attributor:
    """Per-game feature attribution of the critic's V over the just-collected buffer states.
    Built per checkpoint (after ppo.load); `game()` consumes ppo.buffer.dense_states/state_ids/
    state_values BEFORE buffer.clear(). Aggregates the 299 dense dims into
    ppo_model.feature_groups() plus 'own_ids'/'opp_ids' (card-embedding slots 0-5 / 6-11 —
    encode_state appends the acting side's ids first)."""

    IG_STEPS = 32
    IG_STEPS_RETRY = 256      # re-run when the drift-level completeness gap exceeds IG_GAP_TOL
    IG_GAP_TOL = 0.02

    def __init__(self, policy, device, mode):
        assert mode in ("gxi", "ig", "both")
        self.policy, self.device = policy, device
        self.methods = ("gxi", "ig") if mode == "both" else (mode,)
        fg = feature_groups()
        self.names = list(fg) + ["own_ids", "opp_ids"]
        self.n_dense_groups = len(fg)
        gmap = torch.zeros(STATE_DIM, dtype=torch.long)
        for gi, nm in enumerate(fg):
            for i in fg[nm]:
                gmap[i] = gi
        self.gmap = gmap.to(device)

    def _to_groups(self, dense_attr, emb_attr):
        """[n,STATE_DIM] + [n,N_STATE_SLOTS*d_card] -> [n, n_groups_total]."""
        n = dense_attr.shape[0]
        out = torch.zeros(n, len(self.names), device=dense_attr.device)
        out[:, :self.n_dense_groups].index_add_(1, self.gmap, dense_attr)
        per_slot = emb_attr.view(n, N_STATE_SLOTS, -1).sum(-1)
        half = N_STATE_SLOTS // 2
        out[:, self.n_dense_groups] = per_slot[:, :half].sum(1)
        out[:, self.n_dense_groups + 1] = per_slot[:, half:].sum(1)
        return out

    def _ig(self, sr, steps):
        """IG-Δ on the concatenated value-head input, baseline = row 0 -> [n, D] attributions."""
        n, dim = sr.shape
        base, diff = sr[0:1], sr - sr[0:1]
        alphas = (torch.arange(steps, device=sr.device, dtype=sr.dtype) + 0.5) / steps  # midpoint
        pts = (base.unsqueeze(0) + alphas.view(-1, 1, 1) * diff.unsqueeze(0)) \
            .reshape(steps * n, dim).detach().requires_grad_(True)
        self.policy.value_head(pts).sum().backward()
        return diff * pts.grad.view(steps, n, dim).mean(0)

    def game(self, dense_states, state_ids, v_seq):
        """-> per-game record: group drifts + binned trajectories per method, fidelity gaps.
        Asserts the recompute invariant (attribution runs on EXACTLY the act-time V). The
        potential sequences (phi_seq etc.) are NOT built here — collect_cell emits them for
        every recorded game via valid_carried_seq, --attrib or not."""
        ds = torch.stack(dense_states).to(self.device)
        ids = torch.stack(state_ids).to(self.device)
        with torch.no_grad():
            emb0 = self.policy.card_emb(ids).flatten(start_dim=-2)
            v_repr = self.policy._value(self.policy._state_repr(ds, ids))
        gap = (v_repr - torch.tensor(v_seq, device=self.device)).abs().max().item()
        assert gap <= 1e-4, f"V recompute deviates from act-time V by {gap}"
        d_raw = raw_drift(v_seq)
        rec = {"gd": {}, "traj": {}, "fid_gap": {}, "ig_point_gap": None}

        def group_drifts(groups):
            cols = groups.detach().cpu().tolist()               # [n][n_groups]
            return cols, [raw_drift([row[gi] for row in cols]) for gi in range(len(self.names))]

        for m in self.methods:
            if m == "gxi":
                ds_g = ds.clone().requires_grad_(True)
                emb_g = emb0.clone().requires_grad_(True)
                v = self.policy._value(torch.cat([ds_g, emb_g], dim=-1))  # _state_repr cat order
                assert (v - v_repr).abs().max().item() <= 1e-5, "manual cat != _state_repr"
                v.sum().backward()
                cols, gd = group_drifts(self._to_groups(ds_g.grad * ds, emb_g.grad * emb0))
            else:
                sr = torch.cat([ds, emb0], dim=-1)
                for steps in (self.IG_STEPS, self.IG_STEPS_RETRY):
                    ig = self._ig(sr, steps)
                    cols, gd = group_drifts(self._to_groups(ig[:, :STATE_DIM], ig[:, STATE_DIM:]))
                    if abs(sum(gd) - d_raw) <= self.IG_GAP_TOL:
                        break
                # per-decision completeness of the final pass (telemetry, not gated)
                point_gap = (ig.sum(-1) - (v_repr - v_repr[0])).abs().max().item()
                rec["ig_point_gap"] = point_gap
            rec["gd"][m] = gd
            rec["traj"][m] = _bin_means(cols)
            rec["fid_gap"][m] = abs(sum(gd) - d_raw)
        if "ig" in rec["gd"]:
            assert rec["fid_gap"]["ig"] <= self.IG_GAP_TOL, \
                f"IG drift-completeness gap {rec['fid_gap']['ig']} > {self.IG_GAP_TOL} even at " \
                f"{self.IG_STEPS_RETRY} steps"
        return rec


def _attrib_cell_stats(attr_games, segs, names):
    """Per-cell aggregation of per-game attribution records (aligned with segs). Returns the
    JSON block: per-method fine-group drift stats (all/win/loss + SE), win/loss mean binned
    trajectories, super-group rollups, and fidelity stats."""
    d_raws = [raw_drift(v_seq) for v_seq, _ in segs]
    gs = [g for _, g in segs]
    out = {}
    for m in attr_games[0]["gd"]:
        gds = [a["gd"][m] for a in attr_games]

        def _stats(sel):
            rows = [gd for gd, g in zip(gds, gs) if sel(g)]
            if not rows:
                return None, None, 0
            n = len(rows)
            mean = [_pmean([r[gi] for r in rows]) for gi in range(len(names))]
            se = [(_pvar([r[gi] for r in rows]) ** 0.5) / (n ** 0.5) for gi in range(len(names))]
            return mean, se, n

        d_all, se_all, n_all = _stats(lambda g: True)
        d_win, se_win, n_win = _stats(lambda g: g > 0)
        d_loss, se_loss, n_loss = _stats(lambda g: g < 0)

        def _traj(sel):
            rows = [a["traj"][m] for a, g in zip(attr_games, gs) if sel(g)]
            if not rows:
                return None
            traj = []
            for gi in range(len(names)):
                bins = []
                for b in range(N_TRAJ_BINS):
                    vals = [r[gi][b] for r in rows if r[gi][b] is not None]
                    bins.append(_pmean(vals) if vals else None)
                traj.append(bins)
            return traj

        sup = {}
        for sname, members in SUPER_GROUPS.items():
            idx = [names.index(nm) for nm in members]
            per_game = [sum(gd[i] for i in idx) for gd in gds]
            for cls, sel in (("all", lambda g: True), ("win", lambda g: g > 0),
                             ("loss", lambda g: g < 0)):
                vals = [d for d, g in zip(per_game, gs) if sel(g)]
                sup.setdefault(sname, {})[f"drift_{cls}"] = _pmean(vals) if vals else None
                sup[sname][f"drift_{cls}_se"] = \
                    (_pvar(vals) ** 0.5) / (len(vals) ** 0.5) if vals else None

        sums = [sum(gd) for gd in gds]
        out[m] = {
            "drift_all": d_all, "drift_all_se": se_all, "n_all": n_all,
            "drift_win": d_win, "drift_win_se": se_win, "n_win": n_win,
            "drift_loss": d_loss, "drift_loss_se": se_loss, "n_loss": n_loss,
            "traj_win": _traj(lambda g: g > 0), "traj_loss": _traj(lambda g: g < 0),
            "super": sup,
            "fidelity_corr": _pcorr(sums, d_raws),
            "fidelity_gap_mean": _pmean([a["fid_gap"][m] for a in attr_games]),
        }
    ig_pts = [a["ig_point_gap"] for a in attr_games if a["ig_point_gap"] is not None]
    if ig_pts:
        out["ig"]["point_gap_mean"] = _pmean(ig_pts)
        out["ig"]["point_gap_max"] = max(ig_pts)
    return out


def calibration_curve(segs, min_alive=10):
    """Survivorship calibration (the A0 pre-read, now standing in the probe): at decision index
    t, mean V vs mean G over recorded games still alive (n_decisions > t), grid up to the cell's
    median length while >= min_alive games remain. v_excess_rise = half-split drift of
    (meanV - meanG_alive): ~0 = the within-game V rise is survivorship (alive games really are
    more winnable); >> 0 = outcome-independent miscalibration."""
    if not segs:
        return None
    t_med = int(_pmedian([len(v) for v, _ in segs]))
    grid, mv, mg, na = [], [], [], []
    for t in range(t_med + 1):
        alive = [(v, g) for v, g in segs if len(v) > t]
        if len(alive) < min_alive:
            break
        grid.append(t)
        mv.append(_pmean([v[t] for v, _ in alive]))
        mg.append(_pmean([g for _, g in alive]))
        na.append(len(alive))
    if len(grid) < 2:
        return None
    excess = [v - g for v, g in zip(mv, mg)]
    return {"grid_t": grid, "mean_v": [round(x, 4) for x in mv],
            "mean_g_alive": [round(x, 4) for x in mg], "n_alive": na,
            "v_rise": raw_drift(mv), "g_alive_rise": raw_drift(mg),
            "v_excess_rise": raw_drift(excess)}


# ============================================================================================
# Collection — one game per collect_games call so the (V-sequence, G) alignment is exact even
# for games that record no policy decision (records are unconditional, buffer entries are not),
# and the buffer (whose padded option tensors are the heavy part) never grows past one game.
# ============================================================================================

# The three potential channels emitted per recorded game (name -> potential fn). Sign
# conventions (all: HIGHER = better for the learner seat, exact multiples of 1/6 or 1/60):
#   phi_seq       (opp − own prizes)/6      — rises when WE take prizes (prize_potential)
#   phi_deck_seq  (own − opp deckCount)/60  — rises as the OPPONENT nears deck-out (deck_potential)
#   phi_board_seq (own − opp in-play)/6     — rises as the OPPONENT nears bench-out (board_potential)
# Offline V_adj recipe (the shape-gate pattern, no training run needed): V_adj_t =
# v_seq[t] + s*(phi_x_seq[t] − phi_x_seq[0]), then re-run the drift/gap reads on V_adj.
POTENTIAL_SEQS = {"phi_seq": prize_potential,
                  "phi_deck_seq": deck_potential,
                  "phi_board_seq": board_potential}


def valid_carried_seq(dense_states, potential_fn):
    """Potential per decision with --shape's validity semantics: readings taken only where
    prize_potential_valid (MAIN selects, prizes dealt) and carried forward across invalid
    decisions, backfilled from the first valid one (no valid reading in the game → all 0.0).
    So V_adj_t = v_t + s*(phi_t − phi_0) recovers outcome-prediction from a shaped-run critic
    offline. The MAIN gate is measured for the prize lists (work_log DISCOVERY 1); for
    deck/board counts it is a conservative reuse — non-MAIN transients there are unmeasured
    (see the deckCount transient mini-measurement in work_log)."""
    vals, last = [], None
    for x in dense_states:
        if prize_potential_valid(x):
            last = potential_fn(x)
        vals.append(last)
    first_valid = next((v for v in vals if v is not None), 0.0)
    return [first_valid if v is None else v for v in vals]


def collect_cell(ppo, opp_agent, deck_name, deck, games, attributor=None):
    """Returns (segments=[(v_seq, G) per recorded game], all_g=[G per game], n_norecord,
    attr_games, seg_extras, all_ends). attr_games is aligned 1:1 with segments (None when
    attributor is None); seg_extras (aligned with segments) carries the POTENTIAL_SEQS channels
    per recorded game; all_ends (aligned with all_g) carries win_mode.classify_end dicts for
    ALL games, recorded or not. The attribution consumes the buffer's dense_states/state_ids
    (index-aligned with state_values) BEFORE the clear — no extra engine rollouts."""
    def next_opp():
        return opp_agent.fn, opp_agent.deck, opp_agent.name
    segments, all_g, all_ends, n_norecord = [], [], [], 0
    seg_extras = []
    attr_games = [] if attributor is not None else None
    for _ in range(games):
        ppo.buffer.clear()
        rec = collect_games(ppo, next_opp, [(deck_name, deck)], 1)
        g = rec[0][2]
        all_g.append(g)
        all_ends.append(rec[0][3])
        v_seq = [float(v) for v in ppo.buffer.state_values]
        if v_seq:
            segments.append((v_seq, g))
            extras = {name: valid_carried_seq(ppo.buffer.dense_states, fn)
                      for name, fn in POTENTIAL_SEQS.items()}
            extras["end"] = rec[0][3]
            seg_extras.append(extras)
            if attributor is not None:
                attr_games.append(attributor.game(ppo.buffer.dense_states,
                                                  ppo.buffer.state_ids, v_seq))
        else:
            n_norecord += 1
    ppo.buffer.clear()
    return segments, all_g, n_norecord, attr_games, seg_extras, all_ends


def drift_with_se(segs):
    """(mean signed drift, std-error over games) — the per-game spread v_spread_stats averages away."""
    drifts = [signed_drift(v_seq, g) for v_seq, g in segs]
    if not drifts:
        return 0.0, 0.0
    return _pmean(drifts), (_pvar(drifts) ** 0.5) / (len(drifts) ** 0.5)


def _fmt_modes(modes):
    """'prize a / deck b / bench c / effect d' with zero counts skipped. (Mode shares at probe n
    are flaw-localization telemetry, not eval claims — quote Wilson CIs on any share cited.)"""
    parts = [f"{m} {modes[m]}" for m in (*MODE_NAMES.values(), "unknown")
             if modes.get(m)]
    return " / ".join(parts) if parts else "-"


def _fmt(x, w=6, p=3):
    return "-".rjust(w) if x is None else f"{x:+{w}.{p}f}"


def _fmt_cond(d, se, n):
    """One conditional-drift column: '+0.123±0.045(n40)' or '-' when the class is empty."""
    return "-".rjust(18) if d is None else f"{d:+.3f}±{se:.3f}(n{n:d})".rjust(18)


def probe_ckpt(tag, path, deck_names, opp_agents, model_decks, games, seed, device,
               attrib="none"):
    """Run the full cell grid for one checkpoint; returns the result dict (also printed)."""
    ppo = PPO(STATE_DIM, OPTION_DIM, lr=3e-4, gamma=1.0, eps_clip=0.2, k_epochs=1,
              vf_coef=0.5, ent_coef=0.0, device=device)   # hypers inert: update() is never called
    ppo.load(path)
    ppo.policy.eval()                                     # behavior-neutral (no dropout/BN); conventional
    attributor = Attributor(ppo.policy, device, attrib) if attrib != "none" else None

    cells = {}        # (deck, opp) -> [(v_seq, G), ...] recorded segments
    cell_rows = {}    # (deck, opp) -> printed/JSON per-cell stats
    cell_attr = {}    # (deck, opp) -> per-game attribution records (aligned with cells[...])
    all_games = {}    # (deck, opp) -> [G, ...] ALL games (incl. no-record) for the game-level bound
    cell_ends = {}    # (deck, opp) -> [classify_end dict, ...] ALL games (win-mode telemetry)
    print(f"\n=== {tag} ({os.path.basename(path)}) | {games} games/cell ===", flush=True)
    header = (f"{'deck':12s} {'opp':12s} {'games':>5s} {'rec':>4s} {'W/L/D':>8s} "
              f"{'v_win_std':>9s} {'v_btw_std':>9s} {'v_win_frac':>10s} "
              f"{'drift±SE':>15s} {'driftW±SE(n)':>18s} {'driftL±SE(n)':>18s} "
              f"{'len':>6s} {'ev_true':>8s}")
    print(header, flush=True)
    for dname in deck_names:
        for oname, opp in opp_agents:
            t0 = time.time()
            # Reseed per cell (eval_ppo.run_match pattern): identical Python/torch streams across
            # checkpoints per cell. Engine RNG stays unseedable — this is comparability, not CRN.
            random.seed(seed)
            torch.manual_seed(seed)
            segs, all_g, n_norec, attr_games, seg_extras, all_ends = collect_cell(
                ppo, opp, dname, model_decks[dname], games, attributor=attributor)
            cells[(dname, oname)] = segs
            cell_attr[(dname, oname)] = attr_games
            all_games[(dname, oname)] = all_g
            cell_ends[(dname, oname)] = all_ends
            spread = v_spread_stats(segs)
            drift, drift_se = drift_with_se(segs)
            preds, targets = _flat_pred_target({(dname, oname): segs}, None)
            ev_cell = _ev(preds, targets) if preds else None
            w = sum(1 for g in all_g if g > 0)
            l = sum(1 for g in all_g if g < 0)
            d = len(all_g) - w - l
            # Win/loss-conditioned decomposition of the signed drift (the 2026-07-02 "later 4"
            # ambiguity: negative cell drift = falling-V-in-wins vs rising-V-in-losses).
            dw, dw_se, n_w_rec = drift_conditional(segs, +1)
            dl, dl_se, n_l_rec = drift_conditional(segs, -1)
            lens = [len(v_seq) for v_seq, _ in segs]
            # Reconciliation identity (exact up to fp summation; see drift_conditional docstring).
            recon = (dw or 0.0) * n_w_rec - (dl or 0.0) * n_l_rec
            assert abs(spread["v_drift_signed"] * len(segs) - recon) < 1e-9, \
                f"win/loss drift does not reconcile with v_drift_signed: {spread['v_drift_signed']} {recon}"
            cell_rows[(dname, oname)] = {
                "n_games": len(all_g), "n_recorded": len(segs), "n_norecord": n_norec,
                "w": w, "l": l, "d": d, **spread,
                "v_drift_se": drift_se, "ev_true": ev_cell,
                "drift_win": dw, "drift_win_se": dw_se, "n_win_rec": n_w_rec,
                "drift_loss": dl, "drift_loss_se": dl_se, "n_loss_rec": n_l_rec,
                "n_decisions": sum(lens),
                "len_mean": _pmean(lens) if lens else None,
                "len_median": _pmedian(lens) if lens else None,
                "corr_drift_len": drift_len_corr(segs),
                **summarize_ends(all_ends),
                # Per-game records: v_seq rounded to 4 dp (size), drift_raw from the UNROUNDED
                # sequence (the number the stats above actually use) — future slope/quartile/
                # length re-analysis is an offline JSON read, no engine re-run. The three
                # POTENTIAL_SEQS channels + the classify_end dict ride along on every recorded
                # game (see the POTENTIAL_SEQS comment for the offline V_adj recipe).
                "games": [{"G": g, "n_decisions": len(v_seq), "drift_raw": raw_drift(v_seq),
                           "v_seq": [round(x, 4) for x in v_seq],
                           **{k: [round(x, 4) for x in ex[k]] for k in POTENTIAL_SEQS},
                           "end": ex["end"]}
                          for (v_seq, g), ex in zip(segs, seg_extras)],
            }
            if attributor is not None and segs:
                cell_rows[(dname, oname)]["attrib"] = {
                    **_attrib_cell_stats(attr_games, segs, attributor.names),
                    "calibration": calibration_curve(segs),
                }
                for grec, arec in zip(cell_rows[(dname, oname)]["games"], attr_games):
                    for m in arec["gd"]:
                        grec[f"group_drift_{m}"] = [round(x, 4) for x in arec["gd"][m]]
            print(f"{dname:12s} {oname:12s} {len(all_g):5d} {len(segs):4d} "
                  f"{f'{w}/{l}/{d}':>8s} {spread['v_within_std']:9.3f} "
                  f"{spread['v_between_std']:9.3f} {spread['v_within_frac']:10.3f} "
                  f"{_fmt(drift)} ±{drift_se:5.3f}  {_fmt_cond(dw, dw_se, n_w_rec)} "
                  f"{_fmt_cond(dl, dl_se, n_l_rec)} {_pmean(lens) if lens else 0:6.1f} "
                  f"{_fmt(ev_cell)}  [{time.time() - t0:.0f}s]",
                  flush=True)
            if attributor is not None and segs:
                a = cell_rows[(dname, oname)]["attrib"]
                cal = a["calibration"]
                for m in attributor.methods:
                    sup = a[m]["super"]
                    parts = "  ".join(
                        f"{s}={_fmt(sup[s]['drift_loss'], w=6)}" for s in SUPER_GROUPS)
                    fc = a[m]["fidelity_corr"]
                    print(f"{'':12s} attrib[{m}] loss-drift: {parts}  "
                          f"| fid r={_fmt(fc, w=5, p=2)} gap={a[m]['fidelity_gap_mean']:.3f}",
                          flush=True)
                if cal:
                    print(f"{'':12s} calibration: v_rise={cal['v_rise']:+.3f} "
                          f"g_alive_rise={cal['g_alive_rise']:+.3f} "
                          f"v_excess_rise={cal['v_excess_rise']:+.3f}", flush=True)

    # ---- pooled over all cells ----
    all_segs = [s for segs in cells.values() for s in segs]
    pooled = v_spread_stats(all_segs)
    pooled_drift, pooled_se = drift_with_se(all_segs)
    decomp = three_level_decomposition(cells)
    cell_means = {k: cell_mean_v(v) for k, v in cells.items()}
    preds, targets = _flat_pred_target(cells, None)
    ev_true = _ev(preds, targets) if preds else None
    preds_cm, targets_cm = _flat_pred_target(cells, lambda k: cell_means[k])
    ev_cellmean = _ev(preds_cm, targets_cm) if preds_cm else None
    # Matchup-identity shortcut bounds: R² of G on cell identity (predictor = the cell's mean G).
    # Game-level over ALL games (the work-log CAVEAT's bound) and decision-weighted (recorded
    # games, each G replicated per decision — apples-to-apples with per-decision ev_true).
    g_preds = [_pmean(all_games[k]) for k in all_games for _ in all_games[k]]
    g_targets = [g for k in all_games for g in all_games[k]]
    bound_game = _ev(g_preds, g_targets)
    cell_mean_g = {k: _pmean([g for _, g in segs]) if segs else None for k, segs in cells.items()}
    preds_b, targets_b = _flat_pred_target(cells, lambda k: cell_mean_g[k])
    bound_dec = _ev(preds_b, targets_b) if preds_b else None

    p_dw, p_dw_se, p_nw = drift_conditional(all_segs, +1)
    p_dl, p_dl_se, p_nl = drift_conditional(all_segs, -1)
    p_lens = [len(v) for v, _ in all_segs]

    n_dec = sum(len(v) for v, _ in all_segs)
    print(f"\n--- pooled | {tag} ---", flush=True)
    print(f"games={sum(len(g) for g in all_games.values())} recorded={len(all_segs)} "
          f"decisions={n_dec}", flush=True)
    if decomp:
        print(f"V variance decomposition (decision-weighted shares, sum=1): "
              f"between-cell {decomp['between_cell']:.3f} | "
              f"between-game(within-cell) {decomp['between_game']:.3f} | "
              f"within-game {decomp['within_game']:.3f}   (total var {decomp['total_var']:.4f})",
              flush=True)
    else:
        print("V variance decomposition: no contrast (V constant everywhere)", flush=True)
    print(f"v_within_std={pooled['v_within_std']:.4f}  v_between_std={pooled['v_between_std']:.4f}  "
          f"v_within_frac={pooled['v_within_frac']:.4f}  "
          f"v_drift_signed={pooled_drift:+.4f} ±{pooled_se:.4f}", flush=True)
    print(f"drift_win={_fmt_cond(p_dw, p_dw_se, p_nw).strip()}  "
          f"drift_loss={_fmt_cond(p_dl, p_dl_se, p_nl).strip()}  "
          f"len mean/median={(_pmean(p_lens) if p_lens else 0):.1f}/"
          f"{(_pmedian(p_lens) if p_lens else 0):.1f}  "
          f"corr(drift,len)={_fmt(drift_len_corr(all_segs))}", flush=True)
    print(f"ev_true(per-decision)={_fmt(ev_true)}   ev_cellmean_v={_fmt(ev_cellmean)}   "
          f"shortcut bound: game-level R2={_fmt(bound_game)}  decision-weighted R2={_fmt(bound_dec)}",
          flush=True)
    pooled_ends = summarize_ends([e for ends in cell_ends.values() for e in ends])
    n_w_all = sum(pooled_ends["win_modes"].values())
    print(f"win modes: {_fmt_modes(pooled_ends['win_modes'])} | "
          f"loss modes: {_fmt_modes(pooled_ends['loss_modes'])} | "
          f"trailing-prize wins {pooled_ends['trailing_win_n']}/{n_w_all}", flush=True)

    # Per-opponent-COLUMN pooled attribution (the A2 decision-rule granularity: rules read the
    # lucario column pooled over the 4 deck cells).
    columns = {}
    if attributor is not None:
        for oname in dict.fromkeys(o for _, o in cells):  # first-seen order (stable JSON keys)
            col_segs = [s for (d, o), segs in cells.items() if o == oname for s in segs]
            col_attr = [a for (d, o), ag in cell_attr.items() if o == oname and ag
                        for a in ag]
            if not col_attr:
                continue
            columns[oname] = {**_attrib_cell_stats(col_attr, col_segs, attributor.names),
                              "calibration": calibration_curve(col_segs)}
            print(f"\n--- column vs {oname} pooled | {tag} | attribution ---", flush=True)
            cal = columns[oname]["calibration"]
            if cal:
                print(f"calibration: v_rise={cal['v_rise']:+.3f} "
                      f"g_alive_rise={cal['g_alive_rise']:+.3f} "
                      f"v_excess_rise={cal['v_excess_rise']:+.3f}", flush=True)
            for m in attributor.methods:
                sup = columns[oname][m]["super"]
                for cls in ("win", "loss"):
                    parts = "  ".join(
                        f"{s}={_fmt(sup[s][f'drift_{cls}'], w=7, p=3)}"
                        f"±{(sup[s][f'drift_{cls}_se'] or 0):.3f}" for s in SUPER_GROUPS)
                    print(f"attrib[{m}] {cls}-drift: {parts}", flush=True)

    return {
        "ckpt": path,
        **({"group_names": attributor.names, "super_groups": SUPER_GROUPS,
            "columns": columns} if attributor is not None else {}),
        "cells": {f"{d}|{o}": row for (d, o), row in cell_rows.items()},
        "pooled": {**pooled, "v_drift_signed": pooled_drift, "v_drift_se": pooled_se,
                   "decomposition": decomp, "ev_true": ev_true, "ev_cellmean_v": ev_cellmean,
                   "bound_game_level": bound_game, "bound_decision_weighted": bound_dec,
                   "n_decisions": n_dec, "n_recorded_games": len(all_segs),
                   "drift_win": p_dw, "drift_win_se": p_dw_se, "n_win_rec": p_nw,
                   "drift_loss": p_dl, "drift_loss_se": p_dl_se, "n_loss_rec": p_nl,
                   "len_mean": _pmean(p_lens) if p_lens else None,
                   "len_median": _pmedian(p_lens) if p_lens else None,
                   "corr_drift_len": drift_len_corr(all_segs),
                   **pooled_ends},
    }


def make_probe_opponent(name):
    """OPPONENTS factory + optional '@eps' suffix (e.g. 'lucario@0.25'): each DECISION is
    replaced by a random legal action with prob eps — train_ppo make_opponent's noisy-fn
    pattern, so probe columns can match the training curriculum's weakened arms. The base agent
    ALWAYS runs first (the rule-based modules keep per-turn module state that must see every
    decision to stay coherent)."""
    base_name, _, eps_s = name.partition("@")
    if base_name not in OPPONENTS:
        raise SystemExit(f"unknown opponent {name!r}; choose from {list(OPPONENTS)} "
                         f"(optionally with @eps) or 'all'")
    base = OPPONENTS[base_name]()
    if not eps_s:
        return base
    eps = float(eps_s)
    if not 0.0 < eps < 1.0:
        raise SystemExit(f"eps must be in (0,1): {name!r}")

    def noisy_fn(obs):
        sel = base.fn(obs)
        return random_agent(obs) if random.random() < eps else sel
    return Agent(f"{base.name}@{eps:g}", noisy_fn, base.deck)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="PPO checkpoint(s) (ActorCritic state_dict from train_ppo.py); comma "
                         "list probes each, tagged by file basename. Include the run's "
                         "{prefix}_init.pth as the noise-floor control.")
    ap.add_argument("--decks", default="all",
                    help="model-side decks: comma list of {lucario,abomasnow,dragapult_ex,iono} or 'all'")
    ap.add_argument("--opponents", default="lucario,abomasnow,dragapult_ex,iono",
                    help="comma list of {lucario,abomasnow,dragapult_ex,iono,random} or 'all'")
    ap.add_argument("--games", type=int, default=50,
                    help="games per (deck x opponent) cell (V-spread telemetry, not a win-rate eval)")
    ap.add_argument("--attrib", default="none", choices=["none", "gxi", "ig", "both"],
                    help="feature attribution on V over the collected decisions (no extra "
                         "battles): grad×input (gxi), IG-Δ vs the game's first decision (ig), "
                         "or both (IG = the pre-registered arbiter on disagreement). 'none' "
                         "keeps the output schema identical to pre-attrib runs.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", default=None,
                    help="optional output path for the full result dict (e.g. out/probe_credit_<tag>.json)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deck_names = resolve_names(args.decks, DECK_NAMES, "deck")
    opp_names = list(OPPONENTS) if args.opponents.strip() == "all" else \
        [s.strip() for s in args.opponents.split(",") if s.strip()]
    model_decks = {n: load_deck(os.path.join(_REPO_ROOT, "decks", f"{n}.csv")) for n in deck_names}
    opp_agents = [(name, make_probe_opponent(name)) for name in opp_names]

    ckpt_paths = [s.strip() for s in args.ckpt.split(",") if s.strip()]
    tags = [os.path.splitext(os.path.basename(p))[0] for p in ckpt_paths]
    total = len(ckpt_paths) * len(deck_names) * len(opp_agents) * args.games
    print(f"within-game credit probe | ckpts={tags} | decks={deck_names} | opponents={opp_names} | "
          f"{args.games} games/cell | ~{total} battles | seed={args.seed} | device={device}", flush=True)

    results = {}
    for tag, path in zip(tags, ckpt_paths):
        if not os.path.exists(path):
            print(f"  {tag}: checkpoint not found at {path} — skipping", flush=True)
            continue
        results[tag] = probe_ckpt(tag, path, deck_names, opp_agents, model_decks,
                                  args.games, args.seed, device, attrib=args.attrib)

    print("\n[interpretation] matchup-only critic signature: between-cell share dominates, "
          "within-game share ≈ 0, v_drift_signed ≈ 0 (vs the init ckpt's noise floor), and "
          "ev_true ≈ ev_cellmean_v (≤ the shortcut bound). A state-aware critic shows "
          "v_drift_signed > 0. THEORY-REVIEW: ev_true structurally favors game-constant V "
          "(ppo_model.v_spread_stats docstring) — the drift is the discriminating readout. "
          "A NEGATIVE cell drift decomposes via driftW/driftL: driftW < 0 = V falls during wins "
          "(bug-like miscalibration); driftL > 0 = V rises during losses (critic tracks own "
          "progress, not win probability). "
          "Win rates here are small-n telemetry, NOT eval claims (use eval_ppo.py, n≥200).",
          flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[json] wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
