# SB3 MaskablePPO — large-run report

*2026-07-06. Covers: eval-harness verification, a ~3000-update training run, the deck×opponent
results, how battles actually end, and the stack's known limitations. All numbers are from the runs
described here (artifacts under `out/`, gitignored). Goal name in the request was `bs3_report.md`;
written as `sb3_report.md`.*

## TL;DR

- **The eval harness works.** `sb3_env.py`, `utils/run_smoketest_sb3_env.py`, `train_sb3.py --smoke`,
  and `eval_sb3.py` all pass end-to-end; the 4×4 matrix prints with ordered Wilson CIs and a marked
  mirror diagonal.
- **A ~3000-update run (1.536M learner decisions, 1.83 h, one CPU-bound process, cuda for the net)
  reaches ≈ 40% on its best mirror (Lucario) and loses all 16 deck×opponent cells.** Pooled mirror
  win-rate climbs 7.8% → 14.8% over the run, but ~⅔ of that gain lands in the first ~300 updates;
  the last ~2700 updates buy only ~+2.4 pp. This matches the documented legacy-PPO ceiling
  (~5–33%, `docs/ppo/ppo_ceiling_report.md`) and slightly exceeds it on Lucario.
- **Why it stalls is measured, not guessed:** the critic is severely unstable under the terminal-only
  ±1 reward (`explained_variance` ranged **−8.63 to +0.906** across updates), and the agent does not
  contest the prize race — **69% of its losses are prize-outs** (mean end prize deficit −4.26) while
  **only 12% of its wins are prize-race wins**; the rest are deck-out / bench-out attrition, half of
  them while *trailing* on prizes. This is the same behavioral signature the hand-rolled PPO showed.
- **Not deployable yet:** the checkpoint is an SB3 `.zip` (pickles gymnasium spaces); a Kaggle
  submission still needs the not-yet-built SB3-free torch inference mirror.

## What the SB3 stack is

- **`sb3_env.CGEnv`** — a gymnasium wrapper that inverts the `cg` battle loop into `reset()`/`step()`.
  Between two of the learner's decisions it internally services opponent turns (via
  `train_ppo.make_opponent`) and the learner's own non-policy nodes (`auto_resolve`), so `step()`
  only ever surfaces **single-select policy nodes**. The variable option list is padded to
  `N_MAX=128` rows with a validity mask exposed on two channels: the `action_mask` obs key and the
  `action_masks()` method (the sb3-contrib protocol that actually constrains sampling).
- **Driver:** `train_sb3.py` runs stock **`MaskablePPO` (sb3-contrib 2.9.0)** with the stock
  **`MultiInputPolicy`**. SB3 owns the whole algorithm (rollout buffer, GAE, clipped update, logging).
- **Reward:** terminal-only, seat-relative `G ∈ {+1 win, −1 loss, 0 draw}`, undiscounted (`gamma=1.0`).
  Intermediate steps pay 0.
- **Opponent:** the rule-based field on the learner's own deck. Training field == eval field here
  (a deliberate `train_ppo` convention), so the mirror diagonal is a clean skill read.

## Eval-harness verification

Run before the long run so a harness bug would surface cheaply. All four passed:

| Check | Command | Result |
|---|---|---|
| Build | `python sb3_env.py` | `OK: CGEnv reset+step (STATE_DIM=299, OPTION_DIM=70, N_MAX=128)` |
| Env contract | `python utils/run_smoketest_sb3_env.py --games 5` | `check_env: OK`; 5 masked episodes (reward ∈ {−1,0,1}, `win_mode` telemetry present); 0 overflows; no singleton leak |
| Train loop | `python train_sb3.py --smoke` | 4 updates / 1024 decisions / 6 s; zip saved |
| Matrix eval | `python eval_sb3.py --ckpt out/sb3ppo100_sb3.zip --games 20` | full 4×4 matrix; win% ∈ [0,100]; Wilson CIs ordered `lo ≤ p ≤ hi`; mirror diagonal marked |

`play_match` returns `(w, l, d)` from the model's perspective and `eval_sb3.py` computes
`wilson(w, w+l)` (decisive-games CI), seat-swapping every game — verified by reading `eval.py`.

## The training run

Command (only code change this session: `train_sb3.py` gained `--checkpoint-freq`, a stock
`CheckpointCallback`, for the offline curve):

```
python train_sb3.py --total-timesteps 1536000 --checkpoint-freq 153600 \
  --opponent rulebased --decks lucario,abomasnow,dragapult_ex,iono \
  --out-prefix sb3big --seed 0
```

| Item | Value |
|---|---|
| Algorithm | MaskablePPO / `MultiInputPolicy` |
| Learner decisions | 1,536,000 (= 3000 updates × `n_steps` 512) |
| Wall-clock | 6595 s (1.83 h), one process, net on `cuda` |
| Throughput | ≈ 233 decisions/s (≈ 10–14 games/s) |
| Hyperparameters | `batch_size` 128, `n_epochs` 4, `gamma` 1.0, `gae_lambda` 0.95, `clip_range` 0.2, `ent_coef` 0.01, `vf_coef` 0.5, `lr` 3e-4 |
| `ep_rew_mean` | −1.0 → ≈ −0.73 (bounces −0.62…−0.86, no trend after ~100 updates) ≈ 13% terminal win vs the mixed field |
| `explained_variance` | **−8.63 … +0.906** across updates — a severely unstable critic |
| `ep_len_mean` | ≈ 16–22 decisions/game |
| `N_MAX` overflows | **0** (no policy node exceeded 128 options in the whole run) |
| Checkpoints | 10 intermediate zips (`out/sb3big_ckpt_{steps}_steps.zip`) + `out/sb3big_sb3.zip` |

**Critic health caveat:** `explained_variance` here is SB3's *collection-time* metric (value fit on
the rollout the update was computed from), **not** a holdout gate. The hand-rolled stack's
`ev_true_pre` holdout-critic gate is not ported to SB3 yet, so read this as "the critic is visibly
unstable," not a calibrated out-of-sample number.

## Types of battles

- **Matchups:** 4 learner decks × 4 rule-based opponents = 16 cells, seat-swapped every game. The
  **mirror diagonal** (same deck both seats) isolates skill; **off-diagonal** cells mix skill with the
  deck matchup (deck-confound — read with care).
- **Decision coverage:** the learner only ever acts on **single-select policy nodes**. Non-policy
  nodes (and any node with > `N_MAX` options — none occurred) are `auto_resolve`d and are outside
  learning. `train_sb3.py`'s `--opponent` also accepts an `@eps` per-decision-corruption curriculum
  (unused here; this run trained on the clean field to match eval).
- **How games actually end** (final ckpt, sampled, 4 mirror cells pooled, 800 games,
  `win_mode.classify_end` = engine RESULT-log reason):

| | prize-out | bench-out | deck-out | total |
|---|---|---|---|---|
| **Wins** (123, ≈15%) | 15 (12%) | 39 (32%) | 69 (56%) | 123 |
| **Losses** (677) | 469 (69%) | 189 (28%) | 19 (3%) | 677 |

  Mean end prize-diff: **wins −0.50**, **losses −4.26** (positive = ahead). **61 of 123 wins (~50%)
  were prize-*trailing*** (won by deck/bench-out while behind on prizes). Mean game length ≈ 12 turns.
  Read plainly: **the agent cannot win the prize race** — it gets prized out in most losses by a wide
  margin, and the wins it does get come from opponent deck-out/bench-out attrition, often while
  behind. Per deck the winning mode differs: **Lucario wins by deck-out** (62/69 wins), **Abomasnow by
  bench-out** (23/28), and **Dragapult / Iono barely win** (16 / 10 of 200).

## Stats — deck×opponent win% (n=200/cell, seed 11, 95% Wilson CI; `*` = mirror)

**Sampled** (the honest headline for a stochastic policy):

```
model\opp     lucario         abomasnow       dragapult_ex    iono
lucario       40.0(33-47)*    15.0(11-21)     15.0(11-21)     18.0(13-24)
abomasnow     26.0(20-32)     20.5(15-27)*    19.0(14-25)      9.0( 6-14)
dragapult_ex  24.0(19-30)     13.0( 9-18)      7.5( 5-12)*     7.5( 5-12)
iono          15.5(11-21)     13.0( 9-18)      2.5( 1- 6)      5.5( 3-10)*
```

**Greedy** (argmax cross-check):

```
model\opp     lucario         abomasnow       dragapult_ex    iono
lucario       33.5(27-40)*    11.0( 7-16)      9.5( 6-14)     15.0(11-21)
abomasnow     24.0(19-30)     14.5(10-20)*    19.5(15-26)      5.5( 3-10)
dragapult_ex  20.0(15-26)     10.0( 7-15)      7.5( 5-12)*     9.0( 6-14)
iono          19.5(15-26)     12.0( 8-17)      3.0( 1- 6)      4.5( 2- 8)
```

- **Loses all 16 cells** (every CI excludes 50%). Best cell = **Lucario mirror ≈ 40%** (sampled).
- **Deck order** (mirror, sampled): Lucario 40.0 > Abomasnow 20.5 > Dragapult 7.5 ≈ Iono 5.5. The
  Iono/Dragapult mirrors are near-floor; the **Iono column stays dead** — the same pattern the legacy
  stack showed (Iono's deck recycles, so the deck-out weapon the agent leans on is structurally
  useless there).
- Sampled ≥ greedy on the Lucario mirror (40.0 vs 33.5); elsewhere they're within overlapping CIs.

**Learning curve** (pooled mirror-diagonal, sampled, 800 games/point):

| Checkpoint | updates | pooled mirror win% (CI) | Lucario | Abomasnow | Dragapult | Iono |
|---|---|---|---|---|---|---|
| `sb3ppo100` | 100 | 7.8 (6.1–9.8) | 22.5 | 6.0 | 1.5 | 1.0 |
| `sb3big` 153600 | 300 | 12.4 (10.3–14.8) | 28.5 | 16.0 | 2.5 | 2.5 |
| `sb3big` 460800 | 900 | 11.1 (9.1–13.5) | 30.5 | 11.0 | 2.5 | 0.5 |
| `sb3big` 768000 | 1500 | 12.5 (10.4–15.0) | 27.5 | 13.0 | 6.5 | 3.0 |
| `sb3big` 1075200 | 2100 | 14.6 (12.3–17.2) | 33.0 | 11.5 | 9.5 | 4.5 |
| `sb3big` 1536000 | 3000 | 14.8 (12.5–17.4) | 33.0 | 15.5 | 8.0 | 2.5 |

Most of the gain (7.8 → 12.4) lands in the first ~300 updates; the remaining ~2700 updates add only
~+2.4 pp — a near-plateau with a slow crawl, consistent with the "reached by ~200–400 updates, no
gain at 800" legacy prior. **Variance note:** this curve's final Lucario mirror reads 33% while the
headline matrix (independent n=200 sampled run) reads 40% — two honest samples of the same policy
bracket ≈ 33–40%, illustrating that n=200 gives only ≈ ±7 pp resolution at these rates (the README's
high-outcome-variance caveat).

## Known limitations

1. **Positional action head (accepted trade-off).** `MultiInputPolicy`'s head is
   `Linear(latent, N_MAX)`, so logit *i* means "option slot *i*" — nonstationary semantics, and
   per-option features are only visible through the flattened obs. *Observed:* it still learns and
   even edged past the legacy per-option scorer on Lucario, so in this run the handicap did **not**
   visibly cap performance below the hand-rolled stack — but it remains the obvious architectural
   upgrade (a custom per-option maskable policy on the same env).
2. **Terminal-only ±1 reward → unstable critic → no prize-race credit (the core limiter).**
   `explained_variance` swings −8.6…+0.9; with `gamma=1.0` every step's return is just the game
   outcome, so there is no per-decision, win-probability-aligned signal to teach the agent to contest
   prizes. The end-mode data is the symptom: 69% of losses are prize-outs at −4.26.
3. **Single-select coverage only.** Multi-select policy nodes are `auto_resolve`d (not learned).
   Overflow nodes > `N_MAX` are also auto-resolved (0 this run — `N_MAX=128` was sufficient).
4. **`N_MAX=128` is baked into the checkpoint's obs space** — a zip trained at one `N_MAX` will not
   load under code with another.
5. **No VecEnv / RAM-bound.** The `cg` battle is a process-global singleton, so there is one live env
   per process and no `SubprocVecEnv`; the box was swap-exhausted, so the run was a single serial
   process. Wall-clock scales linearly with updates; there is no local parallelism.
6. **Not deployable (submission blocker).** The `.zip` is an SB3 save that pickles gymnasium spaces;
   `MaskablePPO.load`/SB3 imports must never enter `main.py`. A Kaggle submission needs the still-
   missing **SB3-free torch inference mirror** of the policy.
7. **No telemetry parity.** The hand-rolled metrics jsonl and the `ev_true_pre` holdout critic gate
   are not ported; only SB3's collection-time logger is available (hook = an SB3 `BaseCallback`).
8. **`self`/`frozen` opponents unsupported** (`CGEnv` rejects them — they need a live policy).
9. **Eval variance / deck-confound.** High outcome variance (≈ ±7 pp at n=200); off-diagonal cells
   mix skill with the deck matchup.

## Comparison to the legacy hand-rolled PPO

| | legacy PPO ceiling | SB3 large run (3000u) |
|---|---|---|
| Best mirror | Lucario ~32.5% | Lucario **40%** sampled / 33.5% greedy |
| Cells won (of 16) | 0 | 0 |
| Plateau reached by | ~200–400 updates | ~300 updates (then slow crawl) |
| Critic | broken (`ev≈0.1`) | unstable (`ev` −8.6…+0.9) |
| Failure mode | doesn't contest prize race | **same** (69% losses prize-out) |

The SB3 stack reproduces the legacy result's shape and failure mode, and is competitive-to-slightly-
better on the best deck — i.e. the positional-head trade-off was not the binding constraint. The
binding constraint is the **terminal-only reward / prize-race blindness**, shared by both stacks.

## Recommended next steps

1. **Win-probability-aligned per-decision credit** (the open direction from the legacy diagnosis) —
   the most likely lever to break the ceiling, since the prize-race blindness is the measured cause.
   Note the legacy stack found a naïve `Φ_deck` potential shaping *worsened* the critic, so this needs
   a principled per-decision signal, not just any potential.
2. **Custom per-option maskable policy** on the same env — removes the positional-head handicap.
3. **Telemetry parity** via an SB3 `BaseCallback` (holdout `ev`, win-mode logging, metrics jsonl).
4. **SB3-free torch inference mirror** — prerequisite for any Kaggle submission.
5. **Kaggle wiring for `train_sb3.py`** (`SubprocVecEnv` + ship the SB3 deps) if larger runs are
   wanted — though this run shows more single-arm updates alone will not break the ceiling.

## Reproduce

```
# training (1.83 h)
python train_sb3.py --total-timesteps 1536000 --checkpoint-freq 153600 \
  --opponent rulebased --out-prefix sb3big --seed 0
# eval (matrix + curve + end-modes) — see out/sb3big_eval_*.txt, out/sb3big_curve.txt, out/sb3big_endmodes.txt
python eval_sb3.py --ckpt out/sb3big_sb3.zip --games 200 --mode sampled
python eval_sb3.py --ckpt out/sb3big_sb3.zip --games 200 --mode greedy
```
