# PPO training-run health — distilled reference

Companion to `SKILL.md` → "PPO training-run health (Layer-1)". Distills the vendored theory
sources so evaluating a run doesn't require re-reading them, and records which claims are
settled, contested, or ours. Scope: the from-scratch PPO (`ppo_model.py` + `train_ppo.py`),
telemetry read from `{out_dir}/{out_prefix}_metrics.jsonl` (discover: `ls out/*_metrics.jsonl`).

## Attribution key — every theory claim below carries one

| Label | Source | Weight |
|---|---|---|
| `[measured-here]` | this repo's own experiments (`out/ppo_failure_investigation.md`, `work_log.txt`, memory) | strongest — overrides literature on this env |
| `[blog]` | `docs/ppo/ppo_iclr_blog_post.md` — Huang et al., "The 37 Implementation Details of PPO" (ICLR blog track) | Atari/MuJoCo-derived heuristics |
| `[paper]` | `docs/ppo/what_matters_in_on_policy_rl_paper.pdf` — Andrychowicz et al. 2020, "What Matters in On-Policy RL" | continuous-control MuJoCo only (5 envs) |
| `[engstrom-code]` | `reference_repos/implementation-matters/` — Engstrom et al. ICLR 2020 code release | verified in the vendored code; its README states **no** quantitative findings |
| `[paper-memory]` | recalled from a paper, NOT verifiable in the vendored sources | weakest — flag for independent review before acting on it |

**Standing caveats.**
1. `[paper]` studied continuous-control MuJoCo; its bands and rankings do not transfer 1:1 to a
   discrete, variable-action-set card game. Treat as priors, not rules.
2. `[paper]` is a final-performance study, not a telemetry guide — it contains no "explained
   variance" or "clip fraction" prescriptions anywhere (verified by full-text search, 2026-07-02).
3. Imported absolute bands have already misled once: the blog's "approx_kl generally stays below
   0.02" `[blog]` was rejected as a tripwire because a healthy legacy run here had kl=0.039 on
   update 1 `[measured-here]`. Trust trends and our own measured baselines over literature
   constants. Exact invariants (ratio ≡ 1 on epoch 0) are the exception — always enforceable.

## Per-metric guide

Authoritative key set: `grep -n "STATS_KEYS" ppo_model.py` — the self-tests assert exact-set
equality against it, so names can't silently drift; per-stat semantics live in comments where
each stat is computed (grep the stat name).

### Correctness invariants (exact — a violation is a bug, full stop)

| Metric | Healthy | Violation means | Source |
|---|---|---|---|
| `ratio_dev_e0` | ≈ fp noise (< 1e-4; WARN tripwire in `train_ppo.py`) | act/evaluate recompute drift — the rollout probability distribution was not reconstructed at update time. Canonical "check if ratio=1 on the first epoch" bug detector. | `[blog]` Recommendations + `[measured-here]` |
| `return_min` / `return_max` | inside ≈[−1, 1] for terminal-only ±1 rewards at γ=1 (small drift from unbounded linear-V bootstraps is normal) | large excursions = broken episode-boundary reset (GAE leaking across games) | `[measured-here]`; `--update-test` asserts targets ∈ [−1,1] on synthetic episodes |

### The critic (the part that has already failed us once)

| Metric | Reading | Source |
|---|---|---|
| `ev_true_pre` | **THE critic gate.** Holdout explained variance of act-time V vs the true final outcome G. Out-of-sample by construction (act-time values predate the update). Healthy = ≫ 0 sustained; 0.0 is the no-contrast-batch sentinel, not a score. | `[measured-here]` + `evaluation_harness_report.md` §1a; closest literature analogue is Engstrom's *held-out* value-loss logging (`log_value_losses` in `src/policy_gradients/logging.py`) `[engstrom-code]` |
| `explained_var` | IN-SAMPLE, post-update, on the just-fitted batch, against targets containing the critic's own bootstraps. **Never accept as evidence the critic works**: it read 0.4–0.5 while holdout EV was −0.013 (2026-07-02 incident). | `[measured-here]` |
| `value_mean` | pinned near a constant (e.g. ≈ the base rate, −0.965 in the incident) = saturated/collapsed critic even if rank-ordering (AUC) looks fine | `[measured-here]` |
| `value_loss` | trend signal only; low value_loss on the training batch is NOT critic health (same memorization trap as explained_var) | `[measured-here]`; curve-shape-vs-reference check `[blog]` |

Generalized lesson: **any metric computed on the just-fitted batch against self-referential
targets can be a memorization artifact — demand a holdout.** None of the three vendored sources
prescribe EV telemetry; the holdout gate is our own construction, so its semantics live only in
`ppo_model.py` comments (`grep -n "ev_true_pre" ppo_model.py`).

### Policy-update signals (trends, logged-not-thresholded)

| Metric | Reading | Source |
|---|---|---|
| `approx_kl` | ours is the k3 estimator `(ratio − 1) − log ratio` — unbiased, lower-variance; the blog's shipped code logs the noisier k1 `(−logratio)`, so don't compare raw numbers across codebases. KL exploding on the very FIRST update points at input scaling / weight init, not learning dynamics. No absolute band (see caveat 3). | `[blog]` Debug variables + scaling detail; estimator verified in `ppo_model.py` |
| `clip_frac` | ≈ 0 on epoch 0 by construction; stuck at ≈ 0 forever = policy barely moving (pre-fix broken run ~0; improved run 5–12%) | `[blog]` (defines it, no band) + `[measured-here]` |
| `ratio_max` | tail importance-weight indicator. Spikes were measured NOT to drive damage here (up to 35 with no pre/post-window effect, corr(kl, future win) ≈ 0) — don't re-suspect without new evidence. | `[measured-here]` |
| `entropy` / `entropy_norm` | entropy_norm = entropy/ln(n_options) ∈ (0,1]: →0 collapse, →1 flatten (≈1.0 sustained early = policy not learning). Late RE-expansion with flat win rate = the erosion signature (entropy bonus eating learned policy). | `[measured-here]` |
| `policy_loss` / `loss` | no absolute meaning; compare curve shape to a known-good run. Non-finite loss has a WARN tripwire. | `[blog]` Recommendations |

### Scale / balance signals

| Metric | Reading | Source |
|---|---|---|
| `grad_norm_policy_max` / `grad_norm_value_max` / `grad_norm_max` | pre-clip maxima over epoch×minibatch steps. Per-head imbalance = loss-scale imbalance / critic error dominating (value ≈ 40× policy in the first n=2 smoke after the fix pass — observational, read as consistent with the broken-critic symptoms). Persistent clip saturation (≫ 0.5) worth noting, but grad clipping itself is secondary. | `[measured-here]`; grad-clip "small boost, threshold barely matters" `[paper]` §3.3 |
| `adv_mean_raw` / `adv_std_raw` | pre-normalization advantage stats. `adv_std_raw` ≈ 0 = no-contrast batch (all outcomes identical — expect under reward starvation); wild scale drift = GAE/target bug. | `[measured-here]` |

### Driver-side jsonl fields

`update, games, wins, draws, losses, n_samples, games_per_s, per_game`. `per_game` =
`[[learner_deck, opponent, G], …]` — the per-matchup breakdown. Read it for **reward starvation**:
zero wins vs a specific opponent across many updates means that matchup contributes no learning
signal at all (loss-only batches are ~pure noise under a broken or weak baseline)
`[measured-here]` (verdict #2 of the 2026-07-02 investigation).

## Bug → signal map

| Implementation/conceptual bug | Surfaces as |
|---|---|
| rollout distribution not reconstructed at update (act/evaluate drift, stale state, LSTM-style state loss) | `ratio_dev_e0` > 1e-4 `[blog]` `[measured-here]` |
| input scaling / weight-init bug | approx_kl explosion on the very first update `[blog]` |
| episode-boundary / GAE reset bug (bootstrapping across games) | `return_min`/`return_max` excursions; `--update-test` target-range assert `[measured-here]` |
| advantage-normalization bug (wrong scope, 1-element minibatch NaN) | `adv_mean_raw`/`adv_std_raw` anomalies; the `n_mb = max(1, min(M, B//2))` floor exists because std of 1 element is NaN `[measured-here]` |
| critic saturation (bounded head + skewed outcomes) | `value_mean` pinned + `ev_true_pre` ≈ 0 while `explained_var` looks fine `[measured-here]` |
| critic memorization masquerading as fit | `explained_var` high, `ev_true_pre` ≈ 0 `[measured-here]` |
| loss-scale imbalance (critic dominating the shared trunk) | `grad_norm_value_max` ≫ `grad_norm_policy_max` `[measured-here]` |
| policy collapse / flatten / erosion | `entropy_norm` → 0 / → 1 / late re-expansion with flat win rate `[measured-here]` |
| reward starvation vs a too-strong opponent | `per_game`: ~0 wins for that opponent across updates; all-loss batches `[measured-here]` |
| any of the above after touching `update()` | run both self-tests first: `TORCHDYNAMO_DISABLE=1 python ppo_model.py` and `… --update-test` |

## "What Matters" recommendation boxes vs our implementation

Andrychowicz et al. `[paper]` — closely paraphrased, with where we deliberately diverge. MuJoCo
caveat applies throughout. Verify current defaults with `python train_ppo.py --help`, not this table.

| § (page) | Their recommendation | Us |
|---|---|---|
| 3.1 (p4) | Use the PPO loss; start clip threshold 0.25, try lower/higher | PPO clip, default 0.2 (`--eps-clip`) — aligned |
| 3.2 (p5–6) | Init last policy layer 100× smaller; tanh activations; **separate** wide value MLP | tanh MLP heads ✓; heads are separate but SHARE the state repr + card/attack embeddings (coupling noted in `ppo_model.py` comments); **no small last-layer init — PyTorch defaults** (open cross-check candidate; the paper calls policy init its most surprising high-impact finding) |
| 3.3 (p6) | Always normalize observations; check value-function normalization; grad clip secondary | features hand-normalized at encode time; value targets kept RAW — deliberate, z-scoring erases win/loss signal in skewed batches `[measured-here]`; grad clip 0.5 kept |
| 3.4 (p6) | GAE λ=0.9; neither Huber nor PPO-style value clipping | GAE λ=0.95 (theory-flagged); MSE ✓; value clipping omitted ✓ |
| 3.5 (p7) | Multiple epochs; shuffle transitions into minibatches; recompute advantages once per data pass | 4 epochs × 4 shuffled minibatches ✓; advantages computed ONCE pre-update (stale across epochs) — diverges from the paper but matches the canonical implementations `[blog]` |
| 3.6 (p7) | γ is one of the most important hypers; start 0.99; no special timeout handling at large step limits | **γ=1.0** — deliberate for terminal-only zero-sum reward (NashPG precedent), standing THEORY-REVIEW FLAG; no truncation handling (draws measured ≈ 0 in >10k games `[measured-here]`) |
| 3.7 (p8) | Adam, β1=0.9, lr 3e-4 safe default; LR decay secondary | Adam lr 3e-4 ✓; no annealing |
| 3.8 (p8, no formal box) | No regularizer (entropy or KL) helped significantly on their benchmark | ent_coef 0.01; late-run entropy re-expansion associated with erosion `[measured-here]` — a live risk, not settled either way here |

## Contested claims — never state these as settled

1. **Value-loss clipping.** OpenAI baselines uses it `[blog]`; Engstrom et al. report no evidence
   it helps `[paper-memory]` (their paper PDF here has no text layer; the repo README states no
   findings); Andrychowicz find it HURTS regardless of threshold `[paper]` §3.4. Ours: omitted.
2. **Entropy bonus.** Helps exploration on Atari (Mnih et al., via `[blog]`); no evidence it helps
   continuous control `[paper]` §3.8. Ours: 0.01, with the measured erosion caveat above.
3. **Value-function normalization.** Very strong but env-dependent — crucial on some MuJoCo envs,
   significantly HURTS Walker2d; the paper itself notes its conclusion differs from Engstrom's
   `[paper]` §3.3. Ours: raw targets, deliberate `[measured-here]`.
4. **Per-minibatch advantage normalization.** A fidelity detail `[blog]`; placement matches
   Engstrom (`adv_normalize` inside `surrogate_reward`, applied per-minibatch at loss time)
   `[engstrom-code]`; measured impact low `[paper]` §3.3. Ours: per-minibatch.
5. **Parallel-env count / batch scaling.** Paper: more envs can sharply hurt sample complexity
   `[paper]` §3.5; blog argues that's the wrong frame (wall-clock) `[blog]`. Moot here (serial cg),
   recorded for completeness.
6. **Timeout/truncation bootstrapping.** Theoretically one should bootstrap at truncation `[blog]`;
   empirically small effect when the step limit is large `[paper]` §3.6.
7. **KL estimator.** k1 is biased/noisy; k3 recommended `[blog]`. Ours is k3 — but any literature
   KL band was quoted for k1, one more reason not to import thresholds.

## Cross-check recipes (when touching `update()` or suspecting a subtle divergence)

Diff against `reference_repos/implementation-matters/src/policy_gradients/` by symbol, not line:
- Advantage-norm placement: `grep -n "adv_normalize" …/steps.py` (inside `surrogate_reward` = per-minibatch at loss time).
- GAE/returns construction: `grep -n "advantage_and_return" …/agent.py` (note: their `returns` are discounted raw rewards, independent of V — ours are TD(λ) `gae + v`; their value target is `old_vs + advantages` in `value_loss_gae`, the same GAE-implied family).
- Held-out trust-region + value diagnostics worth imitating: `grep -n "paper_constraints_logging\|log_value_losses" …/logging.py` — they log avg-KL / max-ratio / value loss on a FRESHLY COLLECTED held-out batch; `ev_true_pre` is our cheaper stand-in.
- Do NOT copy blindly: their trainer is single-actor continuous-control; their discrete `entropies` returns `Σ p·log p` WITHOUT the negating sign (sign quirk in `models.py`) `[engstrom-code]`.
- Their variance methodology: 40 identical trials per arm, no seed control, final score = mean reward of the last 5 iterations — variance is measured empirically, which is also our only option (cg RNG unseedable).

## Standing repo verdicts (pointers — numbers live in the reports, not here)

- **Critic gate PENDING** as of 2026-07-02: next field run must show `ev_true_pre` ≫ 0 sustained,
  then `eval_ppo.py` matrix vs the ckpt150 anchor. → memory `ppo-failure-investigation-verdict`,
  `out/ppo_failure_investigation.md`.
- **Checkpoint selection: eval-best, never `_final`** — curves rise then ERODE. → same report.
- **Don't re-add a bounding tanh to the value head** (saturated under skewed outcomes); don't
  re-introduce value-target z-scoring. → memory `ppo-from-scratch-skeleton`.
- **Verdict bar:** disjoint Wilson CIs or pooled ≥3σ only; same-seed re-evals swing ~±2σ (engine
  RNG unseedable). → memory `cg-engine-rng-unseedable-crn-infeasible`, `EVAL.md` §5.
- **Ruled out as causes (don't re-suspect without new evidence):** ratio-tail spikes, duplicate
  option rows as the driver, zero-contrast-batch skipping, draws. → `out/ppo_failure_investigation.md`.
