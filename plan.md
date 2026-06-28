# Plan — reaching a strong Pokémon TCG RL agent

## Context (why this plan exists)

We've been iterating on this repo for a while but optimizing the wrong thing. The bulk of recent effort went into **behavior cloning** the one rule-based teacher (`rulebased_lucario`). BC works mechanically — `train_mymodel_bc.py --policy-loss ce` reaches ~85% move-agreement and the `bc_ce` checkpoint is our best learned policy — but its **ceiling is a single weak heuristic agent**. Cloning a weak teacher can never produce a strong agent. Meanwhile the one thing that would produce a strong agent — **RL that actually improves the model** — currently *regresses* it: the untrained `model0` + MCTS is still our strongest checkpoint, and every self-play / BC-warm-start-then-self-play run gets *worse*.

This plan refocuses on the real bottleneck — **making RL improve the policy** — and sequences the work simplest-first, because (a) we don't yet have evidence that a reward-driven gradient can move this model at all, and (b) our local box is too weak for big runs, so each step must be a *tiny* local experiment that proves a mechanism before we spend Kaggle compute scaling it.

Intended outcome: a clear, ordered experimental ladder where each rung has a hypothesis, a tiny local experiment, a go/no-go metric from the eval harness we already trust, and the specific model/code change it implies.

## The core diagnosis (what is actually wrong with RL today)

Established by reading `starter_rl_kaggle.py` + `tcg_model.py` and the work log:

1. **No improving reward signal.** Self-play happens on a **Lucario mirror** (both sides the same deck). A mirror's game outcome is ≈ a coin flip, *uncorrelated with board state*, so there is nothing for a return-driven gradient to climb.
2. **A divergent value-bootstrap loop.** The value target blends game outcome with the MCTS root backup (`grep -n "label = (value + sample.value)" starter_rl_kaggle.py`, and `LAMBDA`), and that backup comes from the network's **miscalibrated** value head (`mcts_agent` defaults to `value_mode="net"`). Training the head off its own miscalibrated estimate + a coin-flip outcome makes it drift further each iteration.
3. **Strength comes from search, not the net.** PUCT + random-rollout leaf eval (`value_mode="rollout"`) is what makes `model0` strong; the learned value head actively *hurts* search on the mirror.

Net effect: training has nothing good to learn from and a feedback loop that makes things worse — exactly the "model0 is strongest, training regresses" result in memory.

## North star

An agent whose strength comes from a **learned** policy (improved by reward + search), able to **exceed the single weak teacher** and climb the Kaggle leaderboard — not merely inference-time MCTS bolted onto an untrained network. Strength that scales with compute, not strength that is purely search-time.

## Operating principles (decisions for this plan)

- **RL is the focus.** BC is an *optional tool only* — usable as an RL warm-start or value-pretrain *if a tiny experiment shows it helps*; keep `bc_ce` as a fixed eval opponent. Not a development thrust.
- **Simplest-first.** Escalate to a harder rung only after the simpler one passes its go/no-go. Never add search/value/league complexity to diagnose something a simpler setup answers.
- **Tiny local to prove a mechanism; Kaggle to measure small effects.** Local runs can only resolve *large* effects (see MDE below). Defer building Kaggle scale-up tooling until a tiny recipe *demonstrably* improves the model.
- **Change the model only when a limitation binds, measured.** Don't grow capacity or retarget the value head preemptively.
- **Eval discipline is non-negotiable** (the `eval-discipline` skill): one **locked `--search-count`** across every comparison, **n ≥ 200** decisive games for win-rate claims, promotion **gated by SPRT** (not point estimates), anchored Elo, and the **mirror-only skill axis** (avoid the `sample_deck`-vs-Lucario deck confound).

## Eval & tiny-experiment protocol (our measuring stick — set up once in R1, then freeze)

- **Primary metric:** win% vs the rule-based teacher on the **Lucario mirror** via `eval_mymodel.py` (both sides play `deck.csv`, so it's a clean skill axis, *not* deck-confounded). Bars from its docstring: **≥48% strong, >50% beats the teacher.** Run at **n ≥ 200**.
- **Low-variance corroborator:** movement on the frozen held-out decision set via `eval_agreement.py` (one forward pass per decision; CIs shrink fast — tells us "the policy moved" long before win-rate can).
- **Regression guard (cross-cutting):** `eval.py --mode gauntlet` vs the hash-pinned `out/anchor_model0.pth` (the `sample_deck` skill axis) — also tells us whether a Lucario-trained gain *generalizes* off its training deck.
- **Promotion gate:** a new checkpoint replaces the current best only when `eval.py --mode sprt --checkpoint NEW --baseline-checkpoint BEST` returns **H1**. Never promote on a raw win% point estimate.
- **MDE reality (why we scale on Kaggle):** detecting 55% needs ~780 decisive games, 60% ~190, 65% ~80 (`EVAL.md` §5). So **tiny local runs can only confirm large effects (≥60–65%)**; sub-5-point gains are a Kaggle-scale measurement. Use SPRT to stop early on big effects.
- **CRN is impossible** (engine RNG unseedable) — reliability comes from n + SPRT + the frozen anchor, never paired replicates.

> Discover defaults/paths at runtime (`--help`, the harness manifest), per the repo's drift-resilience contract. Don't hardcode `out/*.pth` names or flag defaults; verify a symbol/flag exists before relying on it.

## The ladder — major high-level steps

Each rung: **hypothesis → minimal experiment → go/no-go (which eval) → code/model change (real symbols).**

### R1 — Lock the protocol + re-confirm the baseline (no training)
- **Hypothesis:** the bar to beat is `model0` + MCTS(rollout); our metric must resolve effects at feasible n.
- **Experiment:** run `eval_mymodel.py` (greedy + mcts; vs random + teacher) and `eval.py --mode gauntlet` once on `model0` and `bc_ce`; record numbers + the comparability manifest; (re)build the frozen `agreement_set` for `eval_agreement.py`.
- **Go/no-go:** numbers reproduce prior figures within CI; harness self-checks pass (random-vs-random ≈ 50%); `anchor_model0` sha matches `EVAL.md`.
- **Change:** none. Pin the chosen `--search-count` in the run notes.

### R2 — Can a reward move the policy at all? *(the simplest RL — START HERE)*
- **Hypothesis:** plain **REINFORCE on the policy head**, vs a **fixed** opponent, moves the *raw greedy policy* in the winning direction — unlike the mirror's coin-flip outcome. (BC already proved the optimizer/featurization can move the policy *toward a label*; this proves a *reward* can.)
- **Experiment (tiny):** init from `model0`; substrate = Lucario `deck.csv`; opponent = **fixed `random_agent`** (max headroom, unconfounded). Act by **sampling** `softmax(policy/τ)` over single-select options (reuse `auto_resolve` / `is_policy_select` from `eval_mymodel.py` for non-policy nodes); **no `mcts_agent` in the loop.** Loss `= −Σ logπ(a)·(R − b̄) − β·H(π)` with `R` = final **prize-differential** (dense; same quantity `_rollout_value` uses), `b̄` = batch-mean baseline, entropy `β ≈ 0.01`. **Value/encoder head frozen** (`requires_grad_(False)`). ~200–500 games/iter, lr ~3e-4, 3–5 iters — fits the local box.
- **Go/no-go:** greedy win% of the trained **raw policy** vs fixed random (`eval_mymodel.py --agents greedy --opponents random`) rises CI-separated above the `model0` greedy baseline at n ≥ 80 (a ~50→≥65% move is detectable); corroborated by `eval_agreement.py` movement beyond CI. If it does **not** move, that itself diagnoses a coverage limit (see R5b).
- **Change:** focused rewrite of data-gen + loss in `starter_rl_kaggle.py` (sample-act instead of `mcts_agent`; single policy-gradient loss; drop the value loss). Reuse `get_encoder_input` / `get_decoder_input` / `eval_nn` / `LearnInput` from `tcg_model.py`. Note: `eval.py make_model` is **MCTS-only**, so for raw-policy A/B use `eval_mymodel.py` greedy win% + Wilson CI, or add a greedy/`--search-count 0` actor to `make_model`.

### R3 — Expert Iteration: search as the policy-improvement operator *(value dropped)*
- **Hypothesis:** the real gains come from **distilling the search-improved policy** — a feedforward policy can't represent the teacher's/search's lookahead (the Metamon/AMAGO ceiling). MCTS+rollout is a policy-improvement operator that does **not** need the broken value head.
- **Experiment:** ExIt loop vs a **fixed** opponent (start: teacher on `deck.csv`). Act with `mcts_agent(..., value_mode="rollout")`; **policy target = MCTS visit-count distribution, cross-entropy loss** (CE beat Huber decisively in BC); **no value loss.** Add **Dirichlet root noise + a visit temperature** so search actually explores — today `mcts_agent` is deterministic over a near-one-hot `softmax(policy*10)` prior, so visit-count distillation would be a no-op fixed point.
- **Go/no-go:** vs-teacher win% on `deck.csv` crosses the BC ceiling; promotion gated by `eval.py --mode sprt` vs current best; regression-guarded by the gauntlet vs `anchor_model0`.
- **Change in `tcg_model.py`:** pass `value_mode="rollout"` at the `mcts_agent` call sites; replace the advantage-style policy target (`grep -n "sample.policy\[i\]" tcg_model.py`) with normalized visit counts + CE; add root noise/temperature in `mcts_agent`; stop writing `sample.value` and drop the value branch.

### R4 — League play vs frozen snapshots *(kill the mirror divergence)*
- **Hypothesis:** simultaneous mirror self-play is the divergence source (non-stationary opponent + coin-flip reward). A **pool of frozen** past checkpoints keeps the opponent stationary while still self-improving toward super-teacher strength.
- **Experiment:** maintain a frozen pool `{model0, bc_ce, each SPRT-accepted RL ckpt}`; sample one opponent per game; train only the learner (R3 engine). Promote into the pool **only on SPRT-H1 vs current best** (fictitious self-play / AlphaStar-lite).
- **Go/no-go:** a monotone, SPRT-gated climb on vs-teacher win% **and** non-regression on the frozen gauntlet anchor (catches "beats the league but forgot how to beat model0").
- **Change in `starter_rl_kaggle.py`:** replace the `--rb-frac` simultaneous-mirror block with a frozen-opponent sampler; **delete the value-target blend** (`label = (value + sample.value)*0.5`) and the `LAMBDA` bootstrap.

### R5 — Fix model/representation limits as they bind
- **R5a — Determinization correctness (a real bug; pull earlier than the rest):** `mcts_agent`'s `search_begin` is hard-wired to Lucario — it injects a Lucario basic for the face-down active and samples the opponent deck from *your* deck (`grep -n "opponent_active=\|opponent_deck=" tcg_model.py`). Off the Lucario mirror this simulates impossible worlds and corrupts search **today**. Fix: parameterize the opponent model; sample **multiple** determinizations per move and average (mitigates PIMC strategy-fusion). Go/no-go: gauntlet skill axis vs anchor holds/improves on `sample_deck`.
- **R5b — Capacity / coverage (only when it binds):** the policy is trained on **single-select** nodes only, and the action enumerator builds **maxCount-sized combos only** — so "select up to N" / number-menu decisions are `auto_resolve`'d and the gradient never touches them. This likely caps how much outcome the policy can move (and is the prime suspect if R2 shows no movement). Fix: extend `get_decoder_input` + the enumerator to `minCount..maxCount` and train on those nodes **before** spending parameters on `d_model`/depth. Gate each change on SPRT.

### R6 — Scale on Kaggle *(three distinct items; build once a tiny recipe works)*
- **(a) Push *training* to Kaggle compute — does not exist; the real engineering item.** Package the repo + a train entrypoint as a Kaggle kernel (`kaggle kernels push` + `kernel-metadata.json`, GPU on); pull `out/*.pth` back; validate parity by running the *same tiny config* locally and on the kernel.
- **(b) Submit to validate the proxy — already exists** (`kaggle-submission` skill / `pack_submission.py`). Periodically submit and check that local vs-teacher / gauntlet gains track leaderboard movement.
- **(c) Replay loader — does not exist; highest-leverage data unlock.** Replays are already in SDK dict form; a small parser feeds `get_encoder_input` / `get_decoder_input` directly → diverse frozen opponents for R4, breaking single-teacher Goodhart. Parallel enabler, off the critical RL path.

## Pitfalls to avoid (and where the ladder handles each)

| Pitfall | Defense |
|---|---|
| Non-stationary mirror self-play (tail-chasing) | fixed opponent (R2–R3); frozen-snapshot league (R4); never simultaneous mirror |
| Uninformative reward (mirror ≈ coin flip) | fixed *asymmetric* opponent so outcome ∝ policy; delete the value blend (R4) |
| Sparse binary reward / weak credit assignment | prize-diff shaping (R2); MCTS-rollout does credit assignment in ExIt (R3) |
| Deadly triad off a miscalibrated critic | no learned critic early — freeze value head, MC baseline, rollout leaves (R2–R3) |
| Determinized-MCTS pathologies (strategy fusion, the Lucario hardcode) | correct + multi-sample determinization (R5a); never peek at hidden info |
| Policy/entropy collapse under a peaky `tanh×10` policy | entropy bonus + sampling temperature (R2); KL-to-init/BC trust region if needed |
| Eval self-deception (small-n swings, vs-random saturation, deck confound, search budget) | SPRT/gauntlet/agreement, n ≥ 200, mirror-only skill axis, locked `--search-count` (R1) |
| Goodhart on one teacher/deck | league diversity (R4) + replay opponents (R6c) + periodic Kaggle-proxy (R6b) |

## What we will NOT retry (do-not-retry facts)
- Simultaneous mirror self-play with the outcome+backup value blend (diverges — the thing this plan replaces).
- `value_mode="net"` MCTS on the mirror (miscalibrated; rollout is stronger).
- The Metamon/AMAGO/tcgmon port (matched the simple starter; no benefit).
- Treating BC as the primary path (ceiling = one weak teacher).
- Trusting vs-teacher win-rate at n ≈ 60, or vs-random as a skill signal for an MCTS agent (saturated).

## Critical files
- `starter_rl_kaggle.py` — R2 REINFORCE rewrite (sample-act, single PG loss, freeze value head); R4 frozen-opponent sampler; delete the value blend + `LAMBDA`.
- `tcg_model.py` — `mcts_agent` rollout/visit-count-CE target + root noise (R3); determinization fix (R5a); enumerator/decoder coverage (R5b).
- `eval_mymodel.py` — primary R2–R4 metric (greedy/MCTS vs random/teacher on `deck.csv`).
- `eval.py` — `--mode sprt` promotion gate + `--mode gauntlet` regression guard (consider a greedy actor in `make_model` for raw-policy A/B).
- `eval_agreement.py` — low-variance corroborator on the frozen set.
