#!/usr/bin/env python3
"""Offline win-probability credit-signal gate (Phase 0 of the credit-signal plan).

NO TRAINING. This script answers ONE pre-registered question before we spend any compute on
reward shaping:

    Is there a per-decision signal, computable from the obs the agent already receives, that
    predicts the game's TRUE outcome (win/loss) substantially better than the naive prize-diff
    that already gate-FAILED on the legacy stack (~0.18 explained-variance)?

We build candidate win-probability signals Φ(state) -> [-1,+1] (≈ P(win) − P(loss)), score each by
explained-variance-of-the-true-outcome (`ev_true`) on a HELD-OUT set of games, and compare against
two reference baselines on the same holdout: raw prize-diff, and (optionally) the trained SB3
critic's own value head. A candidate must clear its PRE-REGISTERED bar (below) to justify a training
run. If none does, we've spent minutes, not GPU-days, and learned the obs snapshot is
win-uninformative (which would redirect us to IL / play-time search).

Candidates:
  * phi_sup : a supervised win-prob head fit offline on (encode_state -> outcome). The stronger,
              still-cheap generalization of "event-margin from obs['logs']" — encode_state already
              carries the prize/deck/bench/HP/energy counts, so a logistic/MLP fit on it IS the best
              cheap linear-ish read of the board. Doubles as a critic warm-start / PBRS potential.
  * phi_mc  : determinized MC-playout win-prob — reuse tcg_model.search_begin/search_step, played to
              TERMINAL (not the prize-diff leaf), averaged over K determinizations × M rollouts.
              ≈ true win-prob = the ideal potential. Expensive (subsampled).

Holdout = (state -> true seat-relative outcome) pairs from two sources, split BY GAME:
  * the 47 on-disk Kaggle replays (real top-leaderboard play), and
  * instrumented rule-based mirrors across the committed decks (competent, on-mirror states).

Stages (cache the dataset to disk between them so the expensive gate can iterate without regen):
    python credit_probe.py --stage gen  [--rb-games N]      # build + cache the labeled dataset
    python credit_probe.py --stage gate [--mc] [--critic CKPT.zip]   # fit signals + report the gate
    python credit_probe.py --smoke                          # tiny end-to-end sanity (few games)
"""

import argparse
import glob
import json
import math
import os
import pickle
import random
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

# tcg_model first: it puts the vendored `cg` on sys.path (see sb3_env's import-order note).
import tcg_model  # noqa: E402
from tcg_model import search_begin, search_step, search_end  # noqa: E402
from sb3_features import encode_state, STATE_DIM  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

# Rule-based agents import by bare name (same path trick as eval.py).
sys.path.insert(0, os.path.join(_REPO_ROOT, "game_agents", "rulebased"))

DATASET_PATH = os.path.join(_REPO_ROOT, "out", "credit_probe_dataset.pkl")
RESULTS_PATH = os.path.join(_REPO_ROOT, "out", "credit_probe_results.json")
REPLAY_GLOB = os.path.join(_REPO_ROOT, "out", "kaggle_replays", "episode-*-replay.json")

# ------------------------------------------------------------------------------------------------
# PRE-REGISTERED GATE (fixed BEFORE looking at any result — do not tune to the outcome).
# A candidate Φ PASSES iff, on the held-out games:
#   (a) its point ev_true >= EV_FLOOR                (worth the training cost at all), AND
#   (b) its 95% bootstrap-CI lower bound > the prize-diff baseline's ev_true UPPER bound
#       (i.e. disjoint-CI ABOVE the signal that already failed).
# Rationale: prize-diff scored ~0.18 and still gate-failed downstream; a signal worth wiring into
# training must be both absolutely useful and unambiguously better than that.
EV_FLOOR = 0.35
BOOTSTRAP_N = 2000
# ------------------------------------------------------------------------------------------------


def _seat_label(result, seat):
    """Seat-relative terminal outcome in {+1 win, -1 loss, 0 draw}."""
    return 0.0 if result == 2 else (1.0 if result == seat else -1.0)


def _prize_diff(obs_dict, seat):
    """(opp_prizes_left − my_prizes_left)/6, seat-relative. The naive baseline signal in [-1,1]."""
    p = obs_dict["current"]["players"]
    return (len(p[1 - seat]["prize"]) - len(p[seat]["prize"])) / 6.0


def _trim_obs(obs_dict):
    """Keep only what the signals need (current + select), drop the per-step `logs` history to keep
    the cached dataset small. search_begin (Φ_mc) and encode_state (Φ_sup) both read only
    current/select; the RL critic reads the same via obs_to_gym. `logs: []` is kept because
    to_observation_class requires the key (it filters unknown keys but errors on missing required
    ones). `search_begin_input` is LOAD-BEARING for Φ_mc: search_begin reads it to reconstruct the
    determinized search (api.py:546 — None -> "Not agent observation.")."""
    return {"current": obs_dict["current"], "select": obs_dict.get("select"), "logs": [],
            "search_begin_input": obs_dict.get("search_begin_input")}


# ================================================================================================
# STAGE gen — build the labeled (state -> outcome) dataset
# ================================================================================================
def _cap_game(rows, k):
    """Evenly subsample ONE game's decision rows to at most k (memory + de-correlation)."""
    if k <= 0 or len(rows) <= k:
        return rows
    idx = sorted(set(np.linspace(0, len(rows) - 1, k).round().astype(int).tolist()))
    return [rows[i] for i in idx]


def _replay_samples(max_per_game=40):
    """Parse the on-disk Kaggle replays into per-decision samples. Each seat's first action is its
    60-card deck (captured for Φ_mc determinization); each later ACTIVE record with a real `select`
    is a decision, labeled by the episode's terminal `rewards`. Capped per game INLINE so we never
    materialize the full un-subsampled set (memory)."""
    out = []
    files = sorted(glob.glob(REPLAY_GLOB))
    for fpath in files:
        try:
            d = json.load(open(fpath))
        except Exception as e:
            print(f"  [replay] skip {os.path.basename(fpath)}: {e}", flush=True)
            continue
        rewards = d.get("rewards")
        if not rewards or len(rewards) != 2 or rewards[0] is None:
            continue  # unfinished / errored episode
        result = 2 if rewards[0] == rewards[1] else (0 if rewards[0] > rewards[1] else 1)
        gid = f"replay:{d.get('info', {}).get('EpisodeId', os.path.basename(fpath))}"
        decks = {}  # seat -> 60-card deck (first action)
        rows = []
        for step in d["steps"]:
            for seat, rec in enumerate(step):
                act = rec.get("action")
                if seat not in decks and isinstance(act, list) and len(act) == 60:
                    decks[seat] = list(act)  # first 60-int action = the deck submission
                obs = rec.get("observation")
                if rec.get("status") != "ACTIVE" or not obs or obs.get("select") is None:
                    continue
                if isinstance(act, list) and len(act) == 60:
                    continue  # the deck-submission node itself is not a board decision
                rows.append(dict(game_id=gid, seat=seat, label=_seat_label(result, seat),
                                 obs=_trim_obs(obs), prize_diff=_prize_diff(obs, seat),
                                 deck=None, source="replay"))
        for r in rows:
            r["deck"] = decks.get(r["seat"])
        out.extend(_cap_game(rows, max_per_game))
        del d  # free the ~4MB replay JSON before the next file
    print(f"  [replay] {len(files)} files -> {len(out)} decision samples (<= {max_per_game}/game)",
          flush=True)
    return out


def _rulebased_samples(n_games, max_per_game=40):
    """Instrumented rule-based MIRRORS across the committed decks: snapshot every decision, label by
    the game's result. Mirrors isolate skill and give competent, on-distribution board states."""
    import lucario as rb_luc
    import abomasnow as rb_abo
    import dragapult_ex as rb_dra
    import iono as rb_ion
    decks = [("lucario", rb_luc.my_deck, rb_luc.agent),
             ("abomasnow", rb_abo.my_deck, rb_abo.agent),
             ("dragapult", rb_dra.my_deck, rb_dra.agent),
             ("iono", rb_ion.my_deck, rb_ion.agent)]
    out = []
    for gi in range(n_games):
        name, deck, agent = decks[gi % len(decks)]
        snaps = []
        obs, start = battle_start(deck, deck)
        if start.errorPlayer >= 0:
            battle_finish()
            raise ValueError(f"deck rejected: {name} errorType={start.errorType}")
        guard = 0
        try:
            while obs["current"]["result"] < 0:
                seat = obs["current"]["yourIndex"]
                snaps.append((seat, _trim_obs(obs), _prize_diff(obs, seat)))
                obs = battle_select(agent(obs))
                guard += 1
                if guard > 4000:
                    raise RuntimeError("runaway game")
            result = obs["current"]["result"]
        finally:
            battle_finish()
        gid = f"rb:{name}:{gi}"
        rows = [dict(game_id=gid, seat=seat, label=_seat_label(result, seat),
                     obs=tobs, prize_diff=pdiff, deck=deck, source=f"rb:{name}")
                for seat, tobs, pdiff in snaps]
        out.extend(_cap_game(rows, max_per_game))   # cap INLINE: never hold the full raw set
        print(f"    [rb] game {gi + 1}/{n_games} ({name}): result={result}, "
              f"{len(snaps)} decisions", end="\r", flush=True)
    print(" " * 70, end="\r")
    print(f"  [rb] {n_games} mirror games -> {len(out)} decision samples", flush=True)
    return out


def _subsample_per_game(samples, max_per_game, seed=0):
    """Cap decisions per game (evenly across the game's timeline). GAME COUNT drives generalization,
    not decisions-per-game; capping shrinks within-game correlation + dataset size without losing
    game coverage. Keeps a roughly uniform spread over each game's decision sequence."""
    if max_per_game <= 0:
        return samples
    by_game = {}
    for s in samples:
        by_game.setdefault(s["game_id"], []).append(s)
    out = []
    for g, ss in by_game.items():
        if len(ss) <= max_per_game:
            out.extend(ss)
        else:
            idx = np.linspace(0, len(ss) - 1, max_per_game).round().astype(int)
            out.extend(ss[i] for i in sorted(set(idx.tolist())))
    return out


def build_dataset(rb_games, max_per_game=40):
    print("[gen] building labeled (state -> outcome) dataset ...", flush=True)
    # Generators cap per game INLINE (memory: never materialize the full un-subsampled set).
    samples = _replay_samples(max_per_game) + _rulebased_samples(rb_games, max_per_game)
    print(f"[gen] {len(samples)} decisions over "
          f"{len(set(s['game_id'] for s in samples))} games", flush=True)
    # Precompute the Φ_sup feature vector once (encode_state is deterministic on the obs).
    print(f"[gen] featurizing {len(samples)} states ...", flush=True)
    for i, s in enumerate(samples):
        o = to_observation_class(s["obs"])
        s["feat"] = encode_state(o)[0].numpy().astype(np.float32)
        if (i + 1) % 5000 == 0:
            print(f"    [feat] {i + 1}/{len(samples)}", flush=True)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    # Atomic write: dump to a temp file then rename, so an interrupted/OOM'd write can never leave a
    # truncated pickle that later loads as EOFError (what bit us the first time).
    tmp = DATASET_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"samples": samples, "state_dim": STATE_DIM}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, DATASET_PATH)
    n_games = len(set(s["game_id"] for s in samples))
    print(f"[gen] wrote {len(samples)} samples over {n_games} games -> {DATASET_PATH}", flush=True)
    return samples


# ================================================================================================
# STAGE gate — fit the signals + score ev_true with game-level bootstrap CIs
# ================================================================================================
def _split_by_game(samples, holdout_frac=0.3, seed=0):
    """Deterministic train/holdout split BY GAME (never leak states from one game across the split)."""
    games = sorted(set(s["game_id"] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(games)
    n_hold = max(1, int(len(games) * holdout_frac))
    hold = set(games[:n_hold])
    tr = [s for s in samples if s["game_id"] not in hold]
    ho = [s for s in samples if s["game_id"] in hold]
    return tr, ho


def _affine_fit(x, y):
    """Least-squares a,b minimizing ||a*x+b − y||. Returns (a,b)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def _ev(y, pred):
    """Explained variance of the true outcome y by prediction pred: 1 − Var(y−pred)/Var(y)."""
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    vy = np.var(y)
    return float("nan") if vy == 0 else 1.0 - np.var(y - pred) / vy


def _fit_phi_sup(train, seed=0):
    """Win-prob head encode_state(299) -> outcome in [-1,1] (tanh, MSE). Returns predict(feats).

    CRITICAL: every decision in a game shares that game's single ±1 label, so the EFFECTIVE sample
    size is the number of GAMES, not decisions. A flexible net memorizes game-specific features and
    anti-generalizes. Defenses: (1) a small, strongly-L2-regularized MLP, (2) a by-GAME validation
    split with early stopping on held-out-game ev (never on decisions from a training game)."""
    import torch
    torch.manual_seed(seed)
    # by-game fit/val split
    games = sorted(set(s["game_id"] for s in train))
    rng = random.Random(seed)
    rng.shuffle(games)
    n_val = max(1, int(0.2 * len(games)))
    val_games = set(games[:n_val])
    fit = [s for s in train if s["game_id"] not in val_games]
    val = [s for s in train if s["game_id"] in val_games]

    Xf = torch.tensor(np.stack([s["feat"] for s in fit]))
    yf = torch.tensor(np.array([s["label"] for s in fit], dtype=np.float32)).unsqueeze(1)
    Xv = torch.tensor(np.stack([s["feat"] for s in val]))
    yv = np.array([s["label"] for s in val], dtype=np.float32)
    mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True) + 1e-6
    Xfn, Xvn = (Xf - mu) / sd, (Xv - mu) / sd

    net = torch.nn.Sequential(
        torch.nn.Linear(Xf.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 1), torch.nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
    lossf = torch.nn.MSELoss()

    best_ev, best_state, patience, wait = -1e9, None, 40, 0
    for epoch in range(1500):
        net.train(); opt.zero_grad()
        lossf(net(Xfn), yf).backward(); opt.step()
        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                vp = net(Xvn).squeeze(1).numpy()
            ev = _ev(yv, vp)
            if ev > best_ev + 1e-4:
                best_ev, best_state, wait = ev, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    print(f"    [phi_sup] fit {len(fit)} / val {len(val)} decisions "
          f"({len(games) - n_val}/{n_val} games); best val_ev={best_ev:+.3f}", flush=True)

    def predict(feats):
        with torch.no_grad():
            X = (torch.tensor(np.stack(feats)) - mu) / sd
            return net(X).squeeze(1).numpy()
    return predict, best_ev


def _bootstrap_ev_ci(holdout, pred_key, seed=0):
    """95% CI on ev_true via bootstrap resampling of HOLDOUT GAMES (respects within-game
    correlation). pred_key(s) -> signal value already on the affine-mapped/outcome scale."""
    by_game = {}
    for s in holdout:
        by_game.setdefault(s["game_id"], []).append(s)
    games = list(by_game.keys())
    rng = random.Random(seed)
    evs = []
    for _ in range(BOOTSTRAP_N):
        pick = [rng.choice(games) for _ in games]
        ys, ps = [], []
        for g in pick:
            for s in by_game[g]:
                ys.append(s["label"]); ps.append(pred_key(s))
        e = _ev(ys, ps)
        if not math.isnan(e):
            evs.append(e)
    if len(evs) < 20 or len(by_game) < 2:
        return float("nan"), float("nan")   # too few games for a meaningful CI
    evs.sort()
    lo = evs[int(0.025 * len(evs))]
    hi = evs[int(0.975 * len(evs))]
    return lo, hi


# ---- Φ_mc: determinized playout win-prob (offline; no live battle needed) ----
def _playout_winprob(obs_dict, deck, n_det, n_roll, max_depth=600):
    o = to_observation_class(obs_dict)
    yi = o.current.yourIndex
    st = o.current
    deck = list(deck) if deck else None
    if deck is None:
        return float("nan")  # no known deck (shouldn't happen for our sources)
    vals = []
    for _ in range(n_det):
        active = st.players[1 - yi].active
        try:
            ss = search_begin(
                o,
                your_deck=random.sample(deck, st.players[yi].deckCount),
                your_prize=random.sample(deck, len(st.players[yi].prize)),
                opponent_deck=random.sample(deck, st.players[1 - yi].deckCount),
                opponent_prize=[1] * len(st.players[1 - yi].prize),
                opponent_hand=random.sample(deck, st.players[1 - yi].handCount),
                opponent_active=[677] if len(active) > 0 and active[0] is None else [])
        except Exception:
            continue
        for _ in range(n_roll):
            s = ss
            hit = False
            for _step in range(max_depth):
                cur = s.observation.current
                if cur.result >= 0:
                    vals.append(_seat_label(cur.result, yi)); hit = True
                    break
                sel = s.observation.select
                s = search_step(s.searchId, random.sample(range(len(sel.option)), sel.maxCount))
            if not hit:
                vals.append(float("nan"))
    search_end()
    fin = [v for v in vals if not math.isnan(v)]
    return sum(fin) / len(fin) if fin else float("nan")


def _rl_critic_values(samples, ckpt):
    """Load a MaskablePPO checkpoint and return {id(s): critic_value} for samples at fresh nodes."""
    from sb3_contrib import MaskablePPO
    from sb3_env import obs_to_gym
    import torch
    model = MaskablePPO.load(ckpt, device="cpu")
    vals = {}
    ok = bad = 0
    for s in samples:
        try:
            o = to_observation_class(s["obs"])
            gobs = obs_to_gym(o)  # fresh node (picked=())
            t, _ = model.policy.obs_to_tensor(gobs)
            with torch.no_grad():
                v = float(model.policy.predict_values(t).squeeze().cpu().numpy())
            vals[id(s)] = v; ok += 1
        except Exception:
            vals[id(s)] = float("nan"); bad += 1
    print(f"  [critic] {os.path.basename(ckpt)}: valued {ok} states ({bad} skipped)", flush=True)
    return vals


def run_gate(use_mc, critic_ckpt, mc_det, mc_roll, mc_cap, seed):
    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)
    samples = data["samples"]
    train, holdout = _split_by_game(samples, seed=seed)
    print(f"[gate] {len(samples)} samples | train {len(train)} / holdout {len(holdout)} "
          f"(by game); pre-registered EV_FLOOR={EV_FLOOR}", flush=True)
    print(f"[gate] holdout label balance: "
          f"{sum(1 for s in holdout if s['label']>0)}W/"
          f"{sum(1 for s in holdout if s['label']<0)}L/"
          f"{sum(1 for s in holdout if s['label']==0)}D", flush=True)

    results = {}

    def register(name, signal_of, need_train_fit=True):
        """Fit an affine map signal->outcome on TRAIN, score ev_true + bootstrap CI on HOLDOUT."""
        if need_train_fit:
            xtr = [signal_of(s) for s in train]
            ytr = [s["label"] for s in train]
            m = [(x, y) for x, y in zip(xtr, ytr) if not (isinstance(x, float) and math.isnan(x))]
            a, b = _affine_fit([x for x, _ in m], [y for _, y in m])
        else:
            a, b = 1.0, 0.0
        pred = lambda s: a * signal_of(s) + b
        ho = [s for s in holdout if not (isinstance(signal_of(s), float) and math.isnan(signal_of(s)))]
        ev = _ev([s["label"] for s in ho], [pred(s) for s in ho])
        lo, hi = _bootstrap_ev_ci(ho, pred, seed=seed)
        results[name] = dict(ev=ev, ci_lo=lo, ci_hi=hi, n=len(ho), a=float(a), b=float(b))
        print(f"  {name:10s}  ev_true={ev:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={len(ho)}", flush=True)
        return results[name]

    # --- baseline: raw prize-diff (the ~0.18 that gate-failed) ---
    register("prizediff", lambda s: s["prize_diff"])

    # --- candidate: phi_sup (supervised head on encode_state) ---
    predict_sup, _sup_val_ev = _fit_phi_sup(train, seed=seed)
    sup_cache = {id(s): float(v) for s, v in zip(holdout + train,
                 predict_sup([s["feat"] for s in holdout + train]))}
    register("phi_sup", lambda s: sup_cache[id(s)], need_train_fit=True)

    # --- optional baseline: the trained SB3 critic's own value head ---
    if critic_ckpt:
        cvals = _rl_critic_values(holdout + train, critic_ckpt)
        register("rl_critic", lambda s: cvals[id(s)])

    # --- candidate: phi_mc (determinized playouts to terminal; subsampled) ---
    if use_mc:
        rng = random.Random(seed)
        def subsample(pool):
            pool = [s for s in pool if s.get("deck")]
            rng.shuffle(pool)
            return pool[:mc_cap]
        mc_tr, mc_ho = subsample(list(train)), subsample(list(holdout))
        print(f"  [mc] running playouts: {len(mc_tr)} train + {len(mc_ho)} holdout states "
              f"× {mc_det}det×{mc_roll}roll ...", flush=True)
        mc_cache = {}
        for i, s in enumerate(mc_tr + mc_ho):
            mc_cache[id(s)] = _playout_winprob(s["obs"], s["deck"], mc_det, mc_roll)
            if (i + 1) % 25 == 0:
                print(f"    [mc] {i + 1}/{len(mc_tr) + len(mc_ho)}", end="\r", flush=True)
        print(" " * 40, end="\r")
        # fit affine on mc_tr, eval on mc_ho (separate subsamples, no leakage)
        m = [(mc_cache[id(s)], s["label"]) for s in mc_tr
             if not math.isnan(mc_cache[id(s)])]
        a, b = _affine_fit([x for x, _ in m], [y for _, y in m])
        ho = [s for s in mc_ho if not math.isnan(mc_cache[id(s)])]
        pred = lambda s: a * mc_cache[id(s)] + b
        ev = _ev([s["label"] for s in ho], [pred(s) for s in ho])
        lo, hi = _bootstrap_ev_ci(ho, pred, seed=seed)
        results["phi_mc"] = dict(ev=ev, ci_lo=lo, ci_hi=hi, n=len(ho), a=float(a), b=float(b))
        print(f"  {'phi_mc':10s}  ev_true={ev:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={len(ho)}", flush=True)

    # --- pre-registered gate decision ---
    base_hi = results["prizediff"]["ci_hi"]
    print("\n[gate] PRE-REGISTERED decision (EV_FLOOR=%.2f, must beat prize-diff CI_hi=%.3f):"
          % (EV_FLOOR, base_hi), flush=True)
    verdict = {}
    for name in ("phi_sup", "phi_mc"):
        if name not in results:
            continue
        r = results[name]
        passed = bool((r["ev"] >= EV_FLOOR) and (r["ci_lo"] > base_hi))
        verdict[name] = passed
        print(f"  {name:10s}  ev={r['ev']:+.3f} CI_lo={r['ci_lo']:+.3f}  "
              f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    any_pass = any(verdict.values())
    print(f"\n[gate] OVERALL: {'A CANDIDATE PASSED -> proceed to Phase 1' if any_pass else 'NO CANDIDATE PASSED -> stop; obs snapshot is win-uninformative on this holdout'}",
          flush=True)

    out = dict(results=results, verdict=verdict, any_pass=any_pass,
               ev_floor=EV_FLOOR, n_samples=len(samples),
               n_train=len(train), n_holdout=len(holdout))
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gate] wrote {RESULTS_PATH}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gen", "gate"], help="gen: build dataset; gate: run the gate")
    ap.add_argument("--rb-games", type=int, default=400, help="rule-based mirror games to generate")
    ap.add_argument("--max-per-game", type=int, default=40, help="cap decisions kept per game")
    ap.add_argument("--mc", action="store_true", help="also run the (expensive) Φ_mc playout signal")
    ap.add_argument("--critic", default=None, help="MaskablePPO .zip to add its value head as a baseline")
    ap.add_argument("--mc-det", type=int, default=4, help="Φ_mc determinizations per state")
    ap.add_argument("--mc-roll", type=int, default=3, help="Φ_mc random rollouts per determinization")
    ap.add_argument("--mc-cap", type=int, default=250, help="Φ_mc states per split (subsample cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")
    args = ap.parse_args()

    if args.smoke:
        build_dataset(rb_games=6)
        run_gate(use_mc=True, critic_ckpt=None, mc_det=2, mc_roll=2, mc_cap=20, seed=0)
        return

    if args.stage == "gen":
        build_dataset(rb_games=args.rb_games, max_per_game=args.max_per_game)
    elif args.stage == "gate":
        if not os.path.exists(DATASET_PATH):
            sys.exit(f"no dataset at {DATASET_PATH}; run --stage gen first")
        run_gate(use_mc=args.mc, critic_ckpt=args.critic, mc_det=args.mc_det,
                 mc_roll=args.mc_roll, mc_cap=args.mc_cap, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
