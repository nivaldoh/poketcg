# PPO Self-Play: Ceiling & Why (2026-07-01)

Goal: push the from-scratch PPO agent (`ppo_model.py` + `train_ppo.py`) further from **self-play
cold-start**, **without changing the model/training code**, to find its performance ceiling and
explain it. All runs are HIDDEN=256, single-select PPO, MC returns, terminal-only ±1 reward; scaled
purely via existing CLI flags. Progress read from `eval_ppo.py` vs the rule-based field (mirror
diagonal = clean skill read), never from the self-play collect win-rate (~0.5 by construction).

## TL;DR

- **The ceiling is ~5–33% win-rate vs the rule-based field (deck-dependent; best clean mirror is
  lucario 32.5% at n=200), reached by ~200–400 updates.** Training 8× longer (800 updates) does not
  raise it, and the agent loses all 16 deck×opponent matchups — it never beats any rule-based agent.
- **It is NOT a coverage problem** (the net controls 80% of decisions), **NOT a capacity problem**
  (it learns to beat *random* 18%→76%), and **NOT a training-length problem** (plateaus early).
- **It IS a learning-signal problem**: (1) the **critic never fits** the ~50/50 self-play mirror
  (explained-variance `ev` ≈ 0.1 for all 800 updates → advantages dominated by noise), and (2)
  **self-play does not transfer** to countering the specific rule-based strategies (on dragapult/iono
  the net crushes random yet scores ~0–5% vs the heuristic). Hyperparameter sweeps (entropy, lr) and
  alternative opponents (random, frozen) do not break the ceiling.

## What was run (all local, CPU, cold-start, no code changes)

| Run | Config | Purpose |
|---|---|---|
| `ppo_sp_s0`, `ppo_sp_s1` | self-play, 4 decks, **800 updates**, games/upd 16, seeds 0/1 | headline ceiling curve |
| `ppo_sw_ent03`, `ppo_sw_lr1e4` | self-play 500u, `--ent-coef 0.03`, `--lr 1e-4` | can hyperparameters break it? |
| `ppo_opp_random`, `ppo_opp_frozen` | 500u, `--opponent random` / `frozen` | does a different training signal transfer? |
| anchor | existing `ppo_selfplay_final` (100u) | 4×4 matrix on disk |

Throughput: CPU ~3.3 s/update at games=16 (faster + lighter than CUDA for this tiny model — GPU
transfer latency dominates). Box is RAM-bound (~1.6 GB free, swap exhausts at 3+ procs) → ran ≤2
concurrent processes.

## Finding 1 — Coverage is NOT the bottleneck (refutes the leading prior hypothesis)

Decision-coverage instrumentation over 60 random self-play battles (9993 decision nodes):

- **LEARNED (single-select policy nodes the net controls): 80.3%** (89.4% of non-forced decisions).
- AUTO_RESOLVE (fixed rules): 19.7% — of which `forced-single` (only one option) 10.3%,
  `optional-multi->firstK` 6.0%, `mandatory-multi->firstK` 3.2%, `number->max` 0.3%.
- The net steers PLAY (36%), ATTACH (21%), CARD (20%), ABILITY (9.5%), EVOLVE (5.4%).

So the earlier "auto_resolve does the heavy lifting" story is wrong: the net already controls the
overwhelming majority of decisions. The ceiling is in *how it learns*, not *what it controls*.

## Finding 2 — The ceiling: plateaus at ~15–30% vs the field by ~300 updates (both seeds)

Mirror win% vs the rule-based agent on the same deck (`eval_ppo.py`, n=150/cell, sampled, seed 11):

Seed 1 (`out/ppo_curve_s1.log`):

| update | lucario | abomasnow | dragapult | iono |
|---|---|---|---|---|
| 0 (init) | 16.0 | 0.7 | 1.3 | 0.7 |
| 100 | 18.7 | 7.3 | 2.7 | 1.3 |
| 200 | 14.7 | 13.3 | 0.0 | 1.3 |
| 400 | 33.3 | 16.0 | 2.7 | 1.3 |
| 800 (final) | 31.3 | 16.7 | 5.3 | 2.7 |

Seed 0 (`out/ppo_curve_s0.log`): lucario 13→24, abomasnow 1→23, dragapult 1→3, iono 0→0.7 — same shape.

Definitive full 4×4 matrix on `ppo_sp_s1_final` (n=200, sampled, `out/ppo_matrix_sp_final.log`):
mirror diagonals **lucario 32.5%** (26–39), abomasnow 17.0% (12–23), dragapult 5.0% (3–9), iono
4.5% (2–8); **every one of the 16 deck×opponent cells is 2.0–32.5% — it loses all matchups.**

- Reproducible ceiling: lucario ~24–31%, abomasnow ~17–23%, **dragapult ~3–5% (floor), iono ~1–3% (floor)**.
- **Plateau is early**: abomasnow flat from update 200, lucario from ~400; the extra 400–800 updates
  add nothing vs the field. All mirrors < 50% → loses to every rule-based agent.

## Finding 3 — It learns vs random but NOT vs rule-based (self-play non-transfer)

Same checkpoints, vs the random floor (from the curve logs):

| deck | vs random 0→800 | vs rule-based (mirror) 0→800 |
|---|---|---|
| lucario | 19 → 76–79% | 13–16 → 24–31% |
| abomasnow | 45–54 → 79% | 1–3 → 17–23% |
| dragapult | 7–11 → 33–45% | 1 → 3–5% |
| iono | 13–22 → 48–73% | 0–1 → 1–3% |

The net acquires real generic competence (big vs-random gains on every deck) but **cannot counter the
specific strong strategies** the rule-based agents use — starkest on dragapult/iono, where it beats
random ~33–73% yet scores ~0–5% vs the heuristic on the *same* deck. Self-play converges to a
symmetric mutual equilibrium with no pressure to discover those counters.

## Finding 4 — The critic never fits the mirror (broken advantage signal)

Across all 800 headline updates (both seeds), `explained_var` (ev) stays ≈ **0.05–0.20** (occasionally
negative), i.e. the value head explains ~10–20% of return variance at best; `value_loss` ~0.35–0.6.
Entropy declines 1.65 → ~0.95 (not a hard collapse); `approx_kl` stays ~0.0005–0.0018 (stable, no
divergence). Root cause: with terminal-only ±1 reward on a ~50/50 mirror and coarse credit (G pinned
to the last policy-select), the value target of a mid-game state is close to a coin flip — hard to
predict from features — so advantages `= return − V` are dominated by noise, weakening the policy
gradient and capping improvement. This is a *structural* consequence of the reward/critic design,
not a tuning artifact (see Finding 5).

## Finding 5 — Hyperparameters don't break the ceiling

`ent-coef 0.03` and `lr 1e-4` (self-play 500u) leave the critic just as unfit: `ev` stays
{0.25,−0.03,0.09,0.03} and {0.16,0.09,0.19,0.20} respectively (vs headline ~0.1). Higher entropy
only keeps the policy less peaked (ent ~1.5 vs ~0.95); lower lr slows the entropy decline. **Final
mirror win% (n=150) stays in the same envelope** — ent0.03: lucario 20.7, abomasnow 13.3, dragapult
1.3, iono 0.0; lr1e4: lucario 20.0, abomasnow 16.7, dragapult 3.3, iono 0.7 — neither clears the
headline ceiling (both ≤ the best self-play seed).

## Finding 6 — No opponent regime transfers to beating rule-based

Training vs `random` (collect win-rate climbs 0.5→0.9+, learning to beat random) and vs a `frozen`
init-net do not produce a rule-based-beating policy. **Final mirror win% (n=150), same envelope** —
opponent=random: lucario 30.7, abomasnow 13.3, dragapult 2.0, iono 1.3; opponent=frozen: lucario
13.3, abomasnow 20.0, dragapult 2.0, iono 1.3. The non-transfer is regime-independent: no self /
random / frozen signal produces a policy that beats the rule-based field (dragapult/iono stay ~0–5%).

## Root cause (ranked)

1. **Self-play non-transfer** — the dominant cause on 2 of 4 decks; the equilibrium optimizes vs a
   copy of itself, not vs the rule-based field, so it never discovers the required counters.
2. **Broken critic under terminal-only ±1 reward + coarse credit** — `ev`≈0.1 throughout starves PPO
   of a clean advantage; the policy improves slowly and plateaus early.
3. Neither coverage (80% learned), net capacity (beats random), nor training length (plateaus by
   ~300 updates) is limiting; hyperparameters (entropy, lr) and opponent choice don't move it.

## Recommendations to raise the ceiling (documented, NOT implemented — out of scope)

- **Train against the rule-based field directly** (asymmetric decks) or add it to an opponent pool /
  league — the single biggest lever given Finding 3 (needs the deferred `rulebased` opponent + the
  asymmetric-deck driver; a small driver change).
- **Denser reward / better credit assignment**: per-step reward shaping (prize-diff, KO events) or
  GAE(λ) instead of terminal-only MC — attacks the `ev`≈0.1 critic (Finding 4). GAE was flagged as a
  drop-in behind `PPO.update`.
- **Opponent curriculum / lagged-snapshot league** rather than live-mirror or fixed-init self-play.
- Lesser levers: value-function warm start / separate critic lr; running observation normalization
  (currently hand-scaled); larger net (capacity isn't the current binding constraint, so low priority).

## Artifacts / reproduce

- Curves: `out/ppo_curve_s{0,1}.log`; anchor matrix: `out/ppo_matrix_selfplay100.log`,
  `docs/ppo/ppo_anchor_summary.txt`; training logs: `out/ppo_sp_s{0,1}.log`, `out/ppo_sw_*.log`,
  `out/ppo_opp_*.log`; sweep/opp finals: `out/ppo_finals_eval.log`, `out/ppo_frozen_final_eval.log`;
  final 4×4 matrix: `out/ppo_matrix_sp_final.log`.
- Throwaway diagnostics (scratchpad, read-only, no core edits): `coverage_probe.py`, `curve_eval.py`.
- Training: `CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 python train_ppo.py --opponent self
  --updates 800 --games-per-update 16 --save-every 50 --seed S --out-prefix ppo_sp_sS`.
