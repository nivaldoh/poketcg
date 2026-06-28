---
name: eval-discipline
description: >-
  Run the right eval harness and draw only the conclusions it supports. Use when comparing
  checkpoints or agents, claiming the model progressed or regressed, picking a gauntlet / SPRT /
  move-agreement signal, or interpreting eval.py / eval_mymodel.py / eval_agreement.py results.
  Enforces a trustworthy n, overlapping Wilson CIs over point estimates, mirror vs deck-confounded
  matchups, anchored (not per-run) Elo, and a locked --search-count.
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
| Separate NEAR-IDENTICAL checkpoints | `eval_agreement.py` | held-out teacher decisions (no games) | low-variance; win-rate can't (mirror ~50/50). `TORCHDYNAMO_DISABLE=1` |
| Is the BC `MyModel` any good on a clean skill test | `eval_mymodel.py` | Lucario **mirror** (same deck both sides) | needs `TORCHDYNAMO_DISABLE=1`; isolates skill |

Read each harness's current flags/defaults at use time: `python eval.py --help` (incl. `--mode`),
`TORCHDYNAMO_DISABLE=1 python eval_agreement.py --help`, `TORCHDYNAMO_DISABLE=1 python eval_mymodel.py
--help`. They differ across scripts and will drift — don't hardcode them in this skill.

## Interpretation checklist
- **Use a trustworthy `n`.** Small-n win-rates swing the verdict. Fetch the current figure (and the
  swing that motivated it) from memory `eval-training-not-helping-yet`; re-verify rather than trusting
  a stale number.
- **Compare overlapping Wilson CIs, not point estimates.** Always report `n` + CI, never a bare win%.
- **Exclude draws** from win-rate.
- **Lock `--search-count` across the agents you compare** — different search budget = different agent
  strength, so a different `--search-count` is an unfair comparison.
- **Mirror vs deck-confounded.** Only a mirror (same deck both sides) isolates *skill*; the
  `sample_deck` vs Lucario matchup is deck-confounded — don't read it as skill.
- **Checkpoint timing:** `modelN` is saved *before* iteration N's training, not after.
- **Elo is meaningful only in a pure mirror** (round-robin, or the gauntlet's vs-anchor axis) and is
  distorted by non-transitive (rock-paper-scissors) archetype matchups — report the pool, not a bare Elo.
- **CRN is impossible** (the engine seeds its RNG from OS entropy, no ABI hook), so you can't cancel
  luck by re-seeding — reliability comes from `n` / SPRT, not paired seeds (`EVAL.md` §5).
- **Prefer the frozen-anchor scalar over a moving baseline.** Track win% / implied Elo vs the
  hash-pinned anchor (`--mode gauntlet`), not vs an ever-changing `--baseline-checkpoint`, so numbers
  compare across runs. Report **anchored** Elo only — the per-run mean-anchored Elo isn't comparable.
- **Use `--mode sprt` for ship/no-ship**, not a small fixed-`n` win%. Near-identical checkpoints won't
  separate on outcomes at any feasible `n` — use `eval_agreement.py` (held-out move-agreement) there.
- **Value/WPA metrics need a calibrated value model first.** Validate the value model's *calibration*
  before trusting any value-based or win-probability-added metric (this gates the deferred Stage-2
  work — see `docs/eval_info.md`).

## Don't re-run ruled-out experiments
Before concluding "training will fix it", recall the guardrails: untrained `model0` has been the
strongest checkpoint and training regressed it (memory `eval-training-not-helping-yet`); self-play
diverges and the value head is miscalibrated on the ~50/50 mirror; the AMAGO/Metamon port was
abandoned (memory `tcgmon-port-plan`). MCTS, not training, has been providing the strength.

## Single source of truth + known duplicates
- Methodology canon: **`docs/eval_info.md`** (process vs outcome metrics; staged plan; calibration gate) —
  inherit its framing, don't restate it. Volatile specifics live in memory
  (`eval-training-not-helping-yet`, `eval-info-staged-methodology`) — look them up, don't copy here.
- Comparison method (frozen gauntlet / SPRT / move-agreement; why CRN is impossible): **`EVAL.md` §5**
  — point at it, don't restate.

## Drift-resilience
Point at `docs/eval_info.md` and memory; don't restate numbers. Read defaults from `--help`. Reference
checkpoints by discovery (`ls out/`), never a hardcoded `out/*.pth`. Confirm a flag/file still exists
before relying on it.

## Where this fits
This is the **interpret-the-numbers** step of the repo's verify-loop (`docs/skills_and_hooks.md` → "How
these compose"): author/debug an agent under `cg-agent-contract` → run the `run_battle.py` smoke
assertion → **read head-to-head numbers here with discipline** → package with `kaggle-submission` only
at the end. Use `tcg-rules-critic` wherever a step depends on the rules.
