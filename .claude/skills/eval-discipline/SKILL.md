---
name: eval-discipline
description: >-
  Run the right eval harness and draw only the conclusions it supports. Use when comparing
  checkpoints or agents, claiming the model progressed or regressed, picking a gauntlet / SPRT
  signal, interpreting eval.py / eval_mymodel.py / eval_ppo.py
  results, judging a PPO training run's health from its metrics jsonl (the ev_true_pre holdout
  critic gate vs in-sample explained_var), or selecting a checkpoint. Enforces a trustworthy n,
  overlapping Wilson CIs over point estimates, mirror vs deck-confounded matchups, anchored (not
  per-run) Elo, a locked --search-count, holdout-over-in-sample metrics, and eval-best (not
  _final) checkpoint selection.
---

# eval-discipline

The judgment layer over the existing eval harnesses — no new script. Encode the *rules* here; look up
the *numbers* (trustworthy `n`, today's defaults, the best checkpoint) at use time, because they
drift.

## Pick the right harness (read defaults from `--help`, don't memorize them)
| Question | Harness | Matchup | Notes |
|---|---|---|---|
| Compare checkpoints/agents broadly, rough Elo | `eval.py` | `sample_deck` battery / round-robin | the primary Stage-1 signal |
| "Did this checkpoint progress/regress?" — one comparable scalar | `eval.py --mode gauntlet` | vs the FROZEN hash-pinned anchor (+ random, teacher) | track the SKILL axis (mirror vs anchor); vs-random saturates |
| "Ship this change?" — A-vs-B verdict | `eval.py --mode sprt` | `--checkpoint` vs `--baseline-checkpoint`, seat-swapped | sequential, controlled error, fewest games; "inconclusive" is valid |
| Is the BC `MyModel` any good on a clean skill test | `eval_mymodel.py` | Lucario **mirror** (same deck both sides) | needs `TORCHDYNAMO_DISABLE=1`; isolates skill |
| "Did PPO training beat its untrained init / the field?" | `eval_ppo.py` | deck×opponent matrix vs the rule-based field, seat-swapped | mirror diagonal (`*`) = the skill read; off-diagonal deck-confounded. IN-DISTRIBUTION by design (training field == eval field). `--skip-baseline` for cheap per-checkpoint learning curves. Sampled mode is the honest headline (greedy of an untrained net is degenerate). `TORCHDYNAMO_DISABLE=1` |
| "Is this PPO training run healthy?" / "did an `update()` change break something?" | `train_ppo.py` metrics jsonl + `ppo_model.py` self-tests | — (telemetry, no games) | read per **`ppo-health-reference.md`** (this skill dir); self-tests: `TORCHDYNAMO_DISABLE=1 python ppo_model.py` and `… --update-test` |

Read each harness's current flags/defaults at use time: `python eval.py --help` (incl. `--mode`),
`python eval_ppo.py --help`, `python train_ppo.py --help`, `TORCHDYNAMO_DISABLE=1 python
eval_mymodel.py --help`. They differ across scripts and will drift — don't hardcode them in this
skill.

## PPO training-run health (Layer-1) — reading order

Full per-metric semantics, bug→signal map, and theory attributions: **`ppo-health-reference.md`**
in this skill directory. The order matters:

1. **Correctness invariants first** (exact — a violation is a bug): `ratio_dev_e0` ≈ fp noise
   (epoch-0 ratio ≡ 1; the canonical distribution-not-reconstructed detector), the two
   `ppo_model.py` self-tests pass, `return_min`/`return_max` show no large excursions (broken
   episode-boundary reset).
2. **The critic gate:** `ev_true_pre` ≫ 0 sustained. NEVER accept in-sample `explained_var` as
   evidence the critic works — it read 0.4–0.5 while holdout EV was −0.013 (the 2026-07-02
   incident). General principle: *any metric computed on the just-fitted batch against
   self-referential targets can be a memorization artifact — demand a holdout.*
3. **Trend signals, logged-not-thresholded:** `entropy_norm` (→0 collapse / →1 flatten / late
   re-expansion with flat win rate = erosion), `approx_kl` + `clip_frac` + `ratio_max`, per-head
   grad-norm imbalance, `adv_*_raw`, `per_game` (reward starvation: ~0 wins vs an opponent = no
   signal from that matchup). No hard thresholds on purpose: literature bands come from
   Atari/MuJoCo and a healthy run here already violated the blog's kl<0.02 — trust trends and our
   measured baselines over imported constants.
4. **Checkpoint selection: eval-best, never `_final`.** Curves rise then ERODE; build the curve
   with `eval_ppo.py --skip-baseline` over saved checkpoints and pick by eval. Verdicts only from
   disjoint Wilson CIs or pooled ≥3σ — the engine RNG is unseedable and same-seed re-evals swing
   ~±2σ.

## Interpretation checklist
- **Use a trustworthy `n`.** Small-n win-rates swing the verdict. Fetch the current figure (and the
  swing that motivated it) from memory `eval-training-not-helping-yet`; re-verify rather than trusting
  a stale number.
- **Compare overlapping Wilson CIs, not point estimates.** Always report `n` + CI, never a bare win%.
- **Exclude draws** from win-rate.
- **Lock `--search-count` across the agents you compare** — different search budget = different agent
  strength, so a different `--search-count` is an unfair comparison.
- **Mirror vs deck-confounded.** Only a mirror (same deck both sides) isolates *skill*; the
  `sample_deck` vs Lucario matchup — and every off-diagonal `eval_ppo.py` cell — is deck-confounded.
- **In-distribution vs generalization.** The `eval_ppo.py` matrix equals the default training field
  on purpose (beating this field IS the goal) — never present it as a generalization result.
- **Holdout over in-sample.** Prefer metrics that are out-of-sample by construction (`ev_true_pre`,
  held-out move-agreement) over anything computed on the data just trained on.
- **Checkpoint timing:** `modelN` is saved *before* iteration N's training, not after. And pick
  checkpoints by eval, not recency — `_final` has been measurably worse than the mid-run peak.
- **Elo is meaningful only in a pure mirror** (round-robin, or the gauntlet's vs-anchor axis) and is
  distorted by non-transitive (rock-paper-scissors) archetype matchups — report the pool, not a bare Elo.
- **CRN is impossible** (the engine seeds its RNG from OS entropy, no ABI hook), so you can't cancel
  luck by re-seeding — reliability comes from `n` / SPRT, not paired seeds (`EVAL.md` §5).
- **Prefer the frozen-anchor scalar over a moving baseline.** Track win% / implied Elo vs the
  hash-pinned anchor (`--mode gauntlet`), not vs an ever-changing `--baseline-checkpoint`, so numbers
  compare across runs. Report **anchored** Elo only — the per-run mean-anchored Elo isn't comparable.
- **Use `--mode sprt` for ship/no-ship**, not a small fixed-`n` win%. Near-identical checkpoints won't
  separate on outcomes at any feasible `n` — don't claim a difference between them. (The retired
  `eval_agreement.py` move-agreement harness once covered this niche; git history has it.)
- **Value/WPA metrics need a calibrated value model first.** Validate the value model's *calibration*
  before trusting any value-based or win-probability-added metric (this gates the deferred Stage-2
  work — see `docs/eval_info.md`).
- **Theory-claims discipline.** When a conclusion leans on PPO theory, cite it with the attribution
  labels in `ppo-health-reference.md` (`[measured-here]` > vendored sources > `[paper-memory]`);
  present the contested items there AS contested; anything not verifiable in the vendored sources
  gets flagged for independent review (repo convention: theory claims need review).

## Don't re-run ruled-out experiments
Before concluding "training will fix it", recall the guardrails: untrained `model0` has been the
strongest MyModel checkpoint and training regressed it (memory `eval-training-not-helping-yet`);
self-play diverges for MyModel and non-transfers for PPO (memory `ppo-selfplay-multideck-eval`);
the value head is miscalibrated on the ~50/50 mirror; the AMAGO/Metamon port was abandoned (memory
`tcgmon-port-plan`). For PPO specifically, ratio spikes / duplicate options / zero-contrast batches
are measured non-causes (memory `ppo-failure-investigation-verdict`) — don't re-suspect them
without new evidence. MCTS, not training, has been providing MyModel's strength.

## Single source of truth + known duplicates
- Methodology canon: **`docs/eval_info.md`** (process vs outcome metrics; staged plan; calibration gate) —
  inherit its framing, don't restate it. Volatile specifics live in memory
  (`eval-training-not-helping-yet`, `eval-info-staged-methodology`) — look them up, don't copy here.
- Comparison method (frozen gauntlet / SPRT / move-agreement; why CRN is impossible): **`EVAL.md` §5**
  — point at it, don't restate.
- PPO telemetry design + the 3-layer roadmap: **`evaluation_harness_report.md`** (§1a is what
  `STATS_KEYS` implements); distilled reading guide: **`ppo-health-reference.md`** here.

## Drift-resilience
Point at `docs/eval_info.md`, `EVAL.md`, and memory; don't restate numbers. Read defaults from
`--help`. Reference checkpoints by discovery (`ls out/`), never a hardcoded `out/*.pth`; find the
telemetry with `ls out/*_metrics.jsonl`. The telemetry contract is `grep -n "STATS_KEYS"
ppo_model.py` (self-tests assert exact-set equality, so the key set can't silently drift) and each
stat's semantics live in comments beside its computation (e.g. `grep -n "ev_true_pre"
ppo_model.py`). Confirm a flag/file still exists before relying on it.

## Where this fits
This is the **interpret-the-numbers** step of the repo's verify-loop (`docs/skills_and_hooks.md` → "How
these compose"): author/debug an agent under `cg-agent-contract` → run the `utils/run_smoketest_battle.py` smoke
assertion → **read head-to-head numbers here with discipline** → package with `kaggle-submission` only
at the end. Use `tcg-rules-critic` wherever a step depends on the rules.
