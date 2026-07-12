# BC warm-start campaign report (2026-07-09)

Implementation + measurement write-up for [`plan.md`](../plans/plan.md) (id-aware per-option policy +
multi-select conditioning + imitation warm-start). All numbers are seat-swapped `eval_sb3.py`
sampled reads, n=200/cell, Wilson 95% CIs; "pooled mirror" = the 4 mirror (diagonal) cells pooled
(n=800). Historical baseline: the SB3 stock-policy stack plateaued at ~13% pooled mirror
(CI 11.1–15.8) by ~150k steps and lost all 16 cells (`sb3_selfplay_report.md`).

## Headline

| model | pooled mirror | vs plateau |
|---|---|---|
| old stack (stock MultiInputPolicy, RL-only) | ~13.0 (11.1–15.8) | — |
| **Phase M**: new net, PPO-only from scratch, 204.8k steps | 13.9 (11.6–16.4) | no change (overlap) |
| **pure BC** (`out/bcpure_sb3.zip`, gate G1) | **20.0 (17.4–22.9)** | **CI-separated above** |
| **BC + tuned PPO fine-tune** (`out/bcft3_ckpt_153600_steps.zip`, eval-best) | **24.0 (21.1–27.1)** | **CI-separated above** |

- **G1 PASSED**: the pure-BC clone alone beats the plateau, and every historically dead mirror is
  alive (abomasnow 30.0 vs 10.5, dragapult 7.0 vs 2.0, iono 14.5 vs 7.5).
- **Fine-tune improves on BC** (+4pp pooled; dragapult mirror 22.5 (17.3–28.8) vs BC's 7.0 (4–11),
  disjoint — the first CI-separated per-deck gain PPO has ever produced here) and holds it for
  307k steps with no forgetting. The confirmation matrix re-read matched the point pick (24.3 →
  24.0), no winner's curse.
- **The architecture alone is NOT the lever** (Phase M: 13.9%): capacity+ids without per-decision
  credit lands back on the plateau — direct confirmation of the standing credit diagnosis.
- Still loses all 16 cells vs the teachers (a perfect clone would read ~50% in mirrors; per-decision
  holdout accuracy is 0.796 and errors compound over ~70 recorded decisions/game). Iono mirror
  stays weakest (10–12%).

Eval-best matrix (`out/bcft3_best_matrix.log`, model rows × opponent cols):

```
model\opp     lucario         abomasnow       dragapult_ex    iono
lucario       36.5(30-43)*    26.5(21-33)     22.5(17-29)     22.5(17-29)
abomasnow     33.0(27-40)     31.5(25-38)*    28.5(23-35)     11.5( 8-17)
dragapult_ex  22.5(17-29)     19.5(15-26)     16.0(12-22)*    19.0(14-25)
iono          14.5(10-20)     22.5(17-29)      6.0( 3-10)     12.0( 8-17)*
```

## The fine-tune recipe that worked (and the one that didn't)

- **Leg 1 (FAILED — the plan's predicted failure mode):** stock recipe + BC init, lr 5e-5,
  target_kl 0.02. Pooled flat-to-below anchor; iono mirror decayed monotonically to 2.5%
  (CI-disjoint below anchor) — slow forgetting under tiny approx_kl, driven by the entropy bonus
  pushing a sharp clone toward uniform + a near-zero-ev critic at start.
- **Leg 2/3 (WORKED):** `--ent-coef 0 --lr 3e-5 --target-kl 0.01`. Climbs 17→24% over 153.6k
  steps, plateaus ~24% in the continuation leg (`--init-from` chain). ent_coef=0 is now a CLI flag
  with the rationale in its help string.

## Artifacts (out/, gitignored)

- `bc_data/` — 16 shards, 2000 games, 286,035 decisions (143/game), 1.19 GB; every shard passed
  the replay+featurize self-test at write time. `bc_collect.log`.
- `bc_core.pt` — BC core (holdout: acc 0.796 / single 0.803 / multi 0.765 vs floor 0.217;
  value ev −0.03, see open items). `bc_train.log`.
- `bcpure_sb3.zip` — the t=0 pure-BC policy (G1 anchor). `bcpure_eval.log`.
- `bcft1/2/3` checkpoint families + `bcft*_curve.log` + `bcft3_best_matrix.log`;
  **eval-best = `bcft3_ckpt_153600_steps.zip`**.
- `idnet_m_sb3.zip` + checkpoints + `idnet_m_curve.log` (Phase M).

## Open items / next levers (ranked by expected value)

1. **More BC data + epochs**: holdout accuracy was still climbing at epoch 4 on 2000 games; the
   learning-curve read (plan §4.1) says collect 4k games and re-train before anything fancier.
2. **Critic warm-start is unsolved**: value ev peaked 0.106 (epoch 2) then overfit to −0.03 by the
   best-by-acc epoch. PPO's own vf loss recovered ev to ~0.4–0.6 during fine-tune, so it wasn't
   blocking, but a value-only refit (early-stopped on ev) is a cheap candidate improvement.
3. **Iono mirror** (10–12% vs teacher 50%): the longest games / weakest credit; the natural probe
   is accuracy-by-deck in bc_train (deck field is already recorded per row).
4. BC-anchored KL regularization (plan's flagged v2) if longer fine-tune legs start drifting.
5. Deferred as before: SB3-free submission mirror, Kaggle-replay IL, deck conditioning.

## Verification trail

Phase 0: `python sb3_features.py` / `python sb3_env.py` / `utils/run_smoketest_sb3_env.py` (incl.
check_env on the id-extended space). Phase 1: `python tcg_net.py` (grads reach both embedding
tables; permutation + pad-leak probes). Phase 2: `python sb3_policy.py` (save→load bit-identical,
cpu-pinned), `train_sb3.py --smoke` (plain + self-play snapshot round-trip), `eval_sb3.py`,
`record_battle.py` (probs/value trace verified). B1: per-shard replay self-test (`_ms_mask`
legality at every prefix + STOP placement + featurizability). B2: `--smoke` overfit gate on
single-select train acc (0.971 vs measured ~99.4% identical-option tie ceiling). B3: BC→SB3
logit-identity assert inside `--bc-init`. `hand_overflows`/`overflows` = 0 across all runs
(H_MAX=20 never truncated).
