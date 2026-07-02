# PPO failure investigation — findings (2026-07-02)

Evidence-first diagnosis of `train_ppo.py` + `ppo_model.py` on the latest field run
(`out/ppo_field_metrics.jsonl`, 300 updates × 16 games vs the 4-archetype rule-based field;
eval anchor `out/ppo_field_eval.log`, n=200/cell). Pre-registered hypotheses and decision rules
were fixed before running probes (plan: PPO Failure Investigation). All probe scripts are
throwaway diagnostics (session scratchpad); no core files were modified. Eval discipline:
n ≥ 100 with 95% Wilson CIs; conclusions only from disjoint CIs.

## Verdict — two fundamental failures, one misleading telemetry signal

### 1. The value baseline is broken on holdout data (implementation-level failure)

**Measured** (critic probe: 800 fresh games vs the field with `ppo_field_final.pth`, same
sampled/auto_resolve routing as training; 22,889 policy decisions; γ=1 + terminal-only reward
means the true return at every decision IS the final G, so holdout EV is exact):

- Holdout explained variance vs true outcome: **−0.013** (i.e. no better than predicting the mean).
- Mean V = **−0.965** vs mean G = **−0.743**: the tanh value head sits pinned near −1.
  V|win ranges −0.925 (early game) to −0.812 (late); V|loss ≈ −0.98 at every phase.
- The critic *ranks* fine — AUC(V→win) = 0.76 in the first phase quartile, 0.85 in the last —
  but as a regressor it is a saturated near-constant.

**Arithmetic consequence** (with γ=1.0, λ=0.95 GAE): the terminal TD-delta on a **loss** is
−1 − (−0.98) ≈ **−0.02…−0.04**; on a **win** it is +1 − (−0.9) ≈ **+1.9**. Loss episodes
contribute ~nothing but critic-wiggle noise; win episodes carry ~all the gradient signal.
The algorithm degenerates to *win-only REINFORCE plus noise* — which mechanically produces
the other observed pathologies:

- **No wins ⇒ no learning at all** (see failure 2: the dragapult/iono floors).
- **Zero-contrast batches are pure noise steps**: 56/300 updates had 0 wins in 16 games yet
  carried 22.0% of the run's cumulative approx_kl (0.575/2.614), with per-minibatch advantage
  normalization amplifying critic residuals (adv_std_raw 0.166) to unit scale.
- Consistent with the raw-advantage telemetry: adv_mean_raw +0.026 on mixed batches vs −0.085
  on zero-win batches; per-batch value_mean swings −0.56…−0.99 (batch-composition-driven, not
  state-calibrated).

**The training telemetry hid this.** In-sample `explained_var` averaged 0.387 (0.576 on
contrastful batches) — but it is computed post-update on the batch the critic just fit for
4 epochs, against TD(λ) targets that themselves contain the critic's own bootstrapped values.
Holdout EV vs true outcomes is ≈ 0. The prior claim "critic FIXED (ev ~0.55)" is hereby
corrected: the critic fits batches, it does not evaluate states.

Plausible causes (flagged for the fix pass, NOT established by measurement): tanh value head
saturating under a ~90%-loss reward distribution (rare wins get vanishing tail gradients);
one shared Adam LR for actor+critic; 4 epochs × 4 minibatches on ~520-sample batches inviting
memorization. Reference material (docs/ppo: "What Matters" §3.3) ranks value normalization /
reward scaling among the strongest-effect choices.

### 2. Reward starvation vs the rule-based field (training-setup failure)

- **Dose-response** (existing n=200 matrix, pooled per opponent column, n=800): training gain
  is monotone in the baseline win rate — vs lucario 10.4→24.4% (+14.0), abomasnow 2.6→8.0%
  (+5.4), dragapult_ex 2.25→2.38% (+0.1), iono 1.4→0.6% (−0.8). Per-cell Spearman(baseline,
  delta) = +0.48 (16 cells; below the pre-registered 0.6, but the column ordering is 4/4).
- **Expected positive examples** over the whole 300-update run: ~27 wins vs dragapult, ~8–16
  vs iono (out of ~1200 games each) — ~0.1 wins/update. With failure 1, losses teach nothing,
  so those matchups receive essentially zero gradient signal.
- **Learning curves** (checkpoints init/50/…/300, n=200 riser cells / n=100 floor cells):
  learning happens exactly where wins exist and nowhere else. Lucario mirror 20.0→25.5→31.5→
  45.5(peak, ckpt150)→37.5→33.5→33.5; abomasnow mirror 1.0→19.5 by ckpt50. Floor cells never
  move at ANY checkpoint: iono vs dragapult_ex 0,0,0,0,0,1,1%; iono mirror ≤2% throughout.
- **C1 causal probe** (dragapult-only field via monkey-patched `make_opponent`, 150 updates,
  2× the entire field run's dragapult exposure): fresh net flat — 50-update blocks 2.9/4.8/3.6%
  (first vs last overlap). Resumed-from-`ppo_field_final` net crawls 2.0/3.8/5.2% (first vs last
  block 16/800 vs 42/800, disjoint CIs) — signal is not strictly unreachable for a pre-skilled
  net, but the bootstrap rate is glacial (~+3pp per 100 updates, from 2%).

### Symptom, not cause: mid-run peak then erosion (the "plateau" is really rise-then-decay)

- Lucario mirror ckpt150 vs ckpt300, pooled n=600/point across two seeds: **44.2% (40.3–48.2)
  → 34.8% (31.1–38.7)** — disjoint, z≈3.3.
- vs-random meter (random plays sample_deck; pooled 4 decks, n=400/ckpt): 25.5% (init) → 30.2%
  (ckpt100) → 30.0% → **21.5%** (ckpt300); ckpt100→300 disjoint (25.9–35.0 vs 17.7–25.8).
  Per-deck: lucario-deck 21→5% (disjoint decline), abomasnow-deck 85→55% (disjoint decline
  from ckpt100).
- Timing coincides with entropy re-expansion: entropy 50-update block means 1.22/0.88/1.03/
  0.98/1.07/1.20 — the policy flattens back out over the last 100 updates with zero win-rate
  gain, as expected when the advantage signal is mostly noise (failure 1) and the fixed
  ent_coef=0.01 bonus keeps pushing toward uniform.

## Hypotheses killed by the numbers

- **H3 destructive ratio-tail updates — killed.** Spikes exist (ratio_max 35.3 at upd 173 with
  kl 0.086; 21.7 at 147; no target-KL early stop) but do no measurable damage: all 8 spike
  pre/post 320-game windows within ±1.25σ except one at **+2.09σ (improvement)**;
  Pearson(kl_t, next-10-update win rate) = +0.018. (Missing target-kl stays on the hygiene
  list only.)
- **H2 zero-contrast noise updates as CAUSE — demoted.** C2 ablation (identical run but
  skipping every no-contrast batch; 73/300 updates skipped): lucario mirror 35.0% (28.7–41.8)
  vs original 33.5–36.5 — indistinguishable; iono mirror still 0.0%; vs-random pooled 22.0%
  vs original 21.5%. (Abomasnow mirror came out lower, 1.5% vs 14.5% — single-run
  strategy-discovery variance; either way, no rescue.) The noise-update mechanism is real but
  removing it changes nothing — it is downstream of failure 1.
- **H4a duplicate option rows — not the driver.** Fraction of policy decisions where the
  chosen action's encoding was identical to another option's (trained net vs field): lucario
  22.2%, abomasnow 36.4%, dragapult_ex 14.6%, iono 28.6%. Real ceiling component (and an upper
  bound — copies of the same card are harmless duplicates), but uncorrelated with the failure
  pattern: the dead decks have the LOWEST rates.
- **H4b "critic only detects decided games" — refuted in that form** (early-phase AUC 0.76,
  not 0.5); the actual critic failure is scale/calibration (see verdict 1).
- **H5 plateau — wrong description**: curves rise to ckpt50–150 then *erode* (disjoint CIs);
  they do not flatten.

## Cross-checks / audits

- Training draws = 0 across all 4800 games; eval draws = 0 in all 32 cells (0% cells are pure
  losses, not draw artifacts).
- `ppo_field300.pth` and `ppo_field_final.pth` are tensor-identical (md5 differs only via
  torch.save serialization).
- Curve-eval endpoints tie out with the original eval log (ckpt300 lucario mirror 33.5%
  (27.3–40.3) vs original 36.5% (30–43)); the engine RNG is unseedable, so identical-seed
  re-evals still fluctuate ~±2σ (init lucario mirror re-measured 20.0% vs original 12.5%) —
  all conclusions above rest on disjoint CIs or ≥3σ pooled effects.

## Recommendations (for the NEXT pass — no fixes applied in this one)

1. **Fix the value baseline first**; everything else is downstream. Candidates to A/B with the
   holdout-EV probe as the gate (target: holdout EV ≫ 0 on fresh games): linear (non-tanh)
   value head or value-target normalization, separate critic LR, fewer critic epochs.
   The probe script pattern (fresh-game EV vs true G) should become a standing eval, since
   in-sample `explained_var` is demonstrably misleading (0.5 in-sample vs −0.01 holdout).
2. **Fix the reward reachability**: mix beatable opponents into the field (curriculum), or
   the learner never sees a win vs dragapult/iono. C1 shows raw exposure alone does not fix
   this within realistic budgets.
3. Telemetry gaps to add for future runs: per-game opponent archetype + learner deck in the
   metrics jsonl (the flat pooled collect win-rate masked per-opponent trends this entire
   investigation had to reconstruct from checkpoints); holdout/pre-update EV; and treat
   entropy re-expansion with flat win-rate as an erosion alarm.
4. Checkpoint selection: prefer the eval-best checkpoint (here ckpt150, 44.2% lucario mirror
   at n=600) over `_final` (34.8%) — the last update is not the best policy.

## Probe inventory (reproduction)

| Probe | Script (session scratchpad) | Compute |
|---|---|---|
| A: artifact analysis | `analyze_field_run.py` | none |
| B1/B1b: checkpoint curves + vs-random | `run_b1_curves.sh` (logs `b1_*`, `b1b_*`) | ~10k games |
| B2: duplicate options | `dup_option_probe.py` | ~160 games |
| B3: holdout critic EV | `critic_probe.py` | 800 games |
| C1: dragapult-only arms | `train_single_opp.py` (out prefixes `ppo_dragonly_*`) | 4800 games |
| C2: zero-contrast skip | `train_skip_zero.py` (out prefix `ppo_skipzero`) | 4800 games + 900 eval |
| erosion recheck | `eval_ppo.py --ckpt out/ppo_field{150,300}.pth --decks lucario --opponents lucario --games 400 --seed 23` | 800 games |
