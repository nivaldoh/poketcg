# NASHPG-Style Self-Play Agent for Pokémon TCG — Implementation Guidance

**Audience:** Claude Code (the implementing agent).
**Goal:** Build a small Transformer policy+value network for the Kaggle PTCG AI Battle Challenge, trained by **self-play with a NASHPG-style regularized PPO objective** (PPO + KL-toward-a-periodically-reset reference policy).

This document explains the *concepts and decisions*, not the code. It assumes you have the starter notebook (`MyModel`, the sparse encoders, `mcts_agent`, `random_agent`, `sample_deck`), the rulebook, and the NASHPG paper available. **Verify SDK interfaces against the actual `cg` library rather than assuming** — several details below are stated from the starter code and must be confirmed.

---

## 0. The single most important framing

The starter notebook is an **AlphaZero-style** system: MCTS generates improved policy targets, the net learns to imitate MCTS visit counts and predict the outcome, and hidden information is handled by *determinization* (sampling concrete opponent cards to roll the simulator forward).

**We are not doing that.** NASHPG is **model-free**: the network acts directly on its own observation, and learns from the returns of real self-play games. There is no search at action-selection time and no determinization needed to play.

So the plan is: **reuse the starter's architecture and feature encoders; replace its entire training method with NASHPG.** Keep the two paradigms cleanly separated. (MCTS can optionally come back later as a *test-time* search layer on top of a trained policy — out of scope for the first version.)

---

## 1. What NASHPG is, distilled

NASHPG (paper, Algorithm 4) is essentially **Independent PPO self-play plus one extra loss term**. The paper notes its JAX implementation differs from IPPO by ~20 lines. Do not implement the variational-inequality theory; you only need the practical recipe:

- Each player optimizes the usual expected return, **minus** `α · KL( π(·|o) ‖ ρ(·|o) )` averaged over visited information sets `o`, where `ρ` is a **frozen reference policy**.
- Structure is two nested loops:
  - **Outer loop (T iterations):** after the inner loop finishes, reset the reference `ρ ← current policy`.
  - **Inner loop (K updates):** standard regularized PPO updates against the *fixed* `ρ` from the start of this outer iteration.
- **Why it works (intuition):** plain self-play PPO can cycle and converge to an exploitable policy in imperfect-information games. The KL anchor stabilizes the inner optimization toward a *regularized* equilibrium; periodically resetting the anchor to the latest policy is what chases the *actual* (unexploitable) equilibrium. The paper proves this refinement scheme reduces distance to Nash in the idealized version (IMMD); NASHPG is the practical approximation.
- **Do NOT anneal `α` to zero over training.** The paper's Appendix C shows annealing is unstable under stochastic gradients. The multi-round *reference reset* is the mechanism, not a decaying coefficient. Keep `α` fixed.

`α`, `K`, and `T` are the only new knobs over PPO. The paper uses `α = 0.2` across all of its games and finds a broad U-shaped sensitivity (too small → noise dominates; too large → stable but slow). Start at `0.2`.

---

## 2. Reuse vs. drop (mapping onto the starter)

**Reuse as-is:**
- `MyModel` (encoder = value head, decoder = policy head) as the inference substrate.
- `get_encoder_input` and `get_decoder_input` — the sparse feature engineering is domain-specific and already correct. **Do not rebuild feature extraction.**
- `LearnInput` — use it to assemble batched sparse inputs for minibatches.
- Card/attack metadata loading (`all_card_data`, `all_attack`, `card_table`).
- `random_agent` — keep as an evaluation baseline.
- `sample_deck` — the deck to train/self-play with for the first version.

**Drop / ignore for training (these are MCTS-specific):**
- `mcts_agent`, `Node`, `Child`, `create_node`, `_rollout_value`, `SEARCH_COUNT`.
- The determinization sampling inside `mcts_agent` (sampling opponent deck/hand/prizes). **Real self-play does not need it** — the engine holds the true hidden state and each agent acts on its own observation.
- `search_begin` / `search_step` / `search_end` (these drive determinized planning).
- The MCTS-only output transforms: the `tanh` on the policy head and the `exp(policy[i] * 10.0)` prior sharpening. The visit-count-based policy targets and the rollout value mode.

---

## 3. Environment & data essentials

**Running a game:** Use the battle-play APIs (`battle_start`, `battle_select`, `battle_finish` from `cg.game`) to run real self-play games. **Inspect their exact signatures and the game-loop contract first** — the starter only shows them imported, not used end-to-end. Expect a loop of: get an observation for the player to move → that player picks an action → submit via `battle_select` → repeat until terminal.

**Observation structure** (from `to_observation_class(obs_dict)`):
- `obs.current` — game state: `players[...]` (each with `active`, `bench`, `hand`/`handCount`, `prize`, `discard`, `deckCount`, special-condition flags), `stadium`, `turn`, `firstPlayer`, `yourIndex`, and `result`.
- `obs.select` — the current decision: `option` (list of legal options), `maxCount` (how many to select), `context`, and area lists (`deck`, `looking`) used by some option types.

**Reading the outcome for reward** (grounded in the starter's `create_node` / `_rollout_value`): the game is terminal when `obs.current.result >= 0`. From the perspective of the player whose index is `idx`:
- `result == 2` → draw → reward `0`
- `result == idx` → win → reward `+1`
- otherwise → loss → reward `-1`

**Actions:** An action is a list of `maxCount` option indices chosen from `obs.select.option` (see `random_agent`, which does `random.sample(range(len(option)), maxCount)`). For most decision points `maxCount == 1`; the combinatorial case (e.g. "discard 2") is where action handling gets more involved (Section 4).

---

## 4. Model & action-space handling

**Actor-critic split** (already in `MyModel.forward`): the encoder produces `value = V(state)` independent of the candidate actions; the decoder cross-attends to the encoder output and produces **one logit per candidate action**, i.e. `π(action | state, candidates)`.

**Keep the value head's `tanh`** — rewards live in `[-1, 1]`, so a `tanh`-bounded value is appropriate. **Remove `tanh` from the policy path** and treat the decoder output as raw logits to be softmaxed. (The `tanh` + `×10` in the starter exists only to turn the head into MCTS priors.)

**Variable / combinatorial action space:** The policy is a categorical distribution over the *enumerated candidate actions* at each decision point. When `maxCount == 1` this is a plain categorical over legal options. For `maxCount > 1`, enumerate candidate index-lists (the starter caps this at 64 in `create_node`). To act: enumerate candidates → decoder logits → masked softmax → sample one candidate → submit its index-list. Store `(state features, candidate set, chosen index, log-prob)`.

Two consistency requirements that will silently corrupt PPO if violated:
- **The candidate enumeration must be identical at action-time and at update-time** (same order, same cap). Store the candidate set with the transition, or regenerate it deterministically.
- **Padding for batching:** `MyModel`'s decoder does `p.reshape(batch_size, -1, d_model)`, which assumes every batch item has the *same* number of candidate words. The encoder is fine (always 24 words/state); only the policy side has this issue. **Pad each candidate set to a fixed `K_max` (e.g. 64)** and apply a legality/padding mask in the softmax (set padded/illegal logits to `-inf`). This makes batching valid and the categorical well-defined.

**Network sharing:** Use a **single shared network** for both seats. The encoder already takes the acting player's deck (`get_encoder_input(obs, your_deck)`), so one network conditioned on `your_deck` can pilot either side. This is simplest and correct for a mirror match (same deck both sides), and naturally extends to differing decks later since the deck is an input feature. The frozen reference `ρ` is just a snapshot of this one network.

---

## 5. Training loop

Instantiate Algorithm 4 with PPO as the inner optimizer:

1. Init policy+value net `θ`; set reference `ρ ← θ`.
2. **Outer loop** `t = 0 … T-1`:
   - **Inner loop** `k = 0 … K-1`:
     - Collect self-play games with the current net acting for **both** seats.
     - Compute per-player advantages/returns (see below).
     - PPO update (clipped surrogate + value MSE + entropy bonus) **plus** the KL term `α · KL(π_θ ‖ ρ)` over visited info sets.
   - Reset `ρ ← θ`.

**Per-player credit assignment (the part most likely to break):** A game is an *interleaved* sequence of decisions by both players, but each player only sees its own information sets. **Do not** treat the interleaved sequence as one agent's MDP trajectory. Instead:
- Split the game into each player's own subsequence of decisions.
- From player A's view, A's opponent's moves are folded into the environment dynamics: the "next state" after A's decision is A's *next* decision.
- With `γ = 1` and reward only at termination, every decision in a won game gets return `+1` (lost game `-1`) from that player's perspective; player B's terminal reward is the negation of A's (zero-sum).
- Use GAE (`λ = 0.95`) with `V(state)` as the baseline to control the (high) variance from sparse, delayed rewards.
- With a shared net, concatenate both players' `(state, candidate set, action, advantage, return, reference logits)` samples into one PPO batch.

**KL term mechanics:** At each visited info set, you have the current policy's distribution over that decision's candidate set and the reference `ρ`'s distribution over the *same* candidates. `KL = Σ π log(π/ρ)`. Run the **frozen** `ρ` (no gradient) on the stored states to get its logits, compute the KL, and add `α · KL` to the loss. This is one extra forward pass through a frozen network per minibatch — cheap, and conceptually identical to a KL-to-anchor term in RLHF.

**Everything else is vanilla PPO.** The only deltas from Independent PPO are: (1) the added `α · KL` loss term, and (2) the periodic `ρ` reset every `K` inner updates.

Note on the value head: the starter avoids its value head in MCTS because mirror-BC "can't calibrate" it. That concern does not apply here — PPO trains the value head directly against returns, so it will calibrate over training. Use it as the critic; no rollout value needed.

---

## 6. Reward design

Start with the **sparse terminal reward**: `+1` win, `-1` loss, `0` draw (Section 3). This is unbiased, bounded, and matches the `tanh` value head. The paper uses pure outcome rewards throughout and relies on the value baseline + GAE to make it learnable.

Optional later lever if learning is too slow: a **prize-differential shaping** signal, e.g. `(opp_prize_remaining − your_prize_remaining) / 6` from the acting player's perspective (this is exactly the proxy the starter's rollout used). Keep it **off** for the first version — it biases toward prize-greedy play, which is not always optimal (board control sometimes matters more), and adds a tuning surface you don't need yet.

---

## 7. Test-time behavior

**Sample from the policy at test time; do not take argmax.** A deterministic policy is exploitable — the whole point of the equilibrium objective is a *mixed* strategy. Argmax throws that away.

For a Kaggle ladder there is a nuance: a Nash-style policy is *unexploitable* but not necessarily *maximally exploitative* of weak opponents, and the ladder rewards exploiting weak opponents. Default to sampling from the trained policy; you can experiment with a softmax temperature as a single knob (slightly sharper to exploit weak fields, softer to stay robust), but treat that as a late optimization, not a first-version concern.

---

## 8. Evaluation

Because of coin flips and card draws, **single games are very noisy** — every metric needs many games and a few seeds.

- **Primary — head-to-head win rate** against a fixed opponent set: `random_agent`, the MCTS starter agent, and frozen *past versions of self*. Play hundreds-to-thousands of games per matchup with **randomized seat assignment** (control for first-player advantage), and report mean ± confidence interval. The paper uses 1,024 games × 4 seeds. Win rate vs. the MCTS starter is your cheapest meaningful **progress signal** during development.
- **Secondary — approximate exploitability:** train a fresh PPO best-response against your *frozen* policy and measure how much it beats you (the paper's "Approx-BR"). This directly measures the thing NASHPG is supposed to deliver (low exploitability), but it's expensive — it trains a whole new agent — so run it periodically (e.g. at checkpoints), not every step.
- **Cheap sanity checks:** (a) value-head calibration — does predicted `V(state)` track the eventual outcome? (b) policy entropy — is it collapsing toward determinism (bad) or staying healthily stochastic?

Keep the first-version eval to the primary win-rate signal plus occasional exploitability checks. Resist building an elaborate intermediate-decision-quality benchmark suite before the pipeline learns at all.

---

## 9. Initial hyperparameters

From the paper's shared PPO config (Table 4), which is a sound starting point:
- Optimizer **AdamW**, learning rate **3e-4**
- PPO epochs **4**, minibatches **4**
- Discount **γ = 1** (games are finite/episodic)
- GAE **λ = 0.95**, clip **ε = 0.2**
- Entropy coefficient **0.1** (deliberately high — encourages the stochastic/mixed policy you want here)
- Max grad norm **0.5**

NASHPG-specific:
- KL coefficient **α = 0.2**
- Outer iterations **T** and inner updates **K**: the paper uses `K = 10,000`, `T = 25` (≈250k updates). **That assumes a JAX-vectorized environment.** For your first runs with the Python `cg` simulator, start much smaller (e.g. `K ≈ 500–2000`, `T ≈ 5–15`) just to validate the pipeline, then scale with your compute budget. `K`/`T` are pure budget knobs.

Network (start small, scale later):
- `d_model ≈ 128`, `num_heads = 4`, `2` encoder layers + `2` decoder layers. (The paper's most complex game, Hold'em, uses a 512-dim / 4-head Transformer; TCG is complex and long-horizon, so you can grow toward that, but begin small.)
- Candidate cap `K_max ≈ 64`.

**Measure the simulator's steps/sec before sizing anything else** — rollout throughput, not GPU, will set your effective scale.

---

## 10. Risks & pitfalls

- **Simulator throughput is the main bottleneck.** The `cg` SDK is a Python rules engine, not a JAX-vectorized environment like the paper's. Rollout collection will dominate wall-clock. Measure steps/sec early; parallelize *independent games across processes* if the SDK allows. The first goal is a working learning signal, not paper-scale update counts.
- **Per-player sign/perspective bugs.** Getting the value and reward perspective wrong for one seat is the classic self-play bug. The starter's value sign-flipping (`v = value if state.yourIndex == your_index else -value`) is a hint at how careful this is. Test it explicitly.
- **Action-enumeration inconsistency** between act-time and update-time → wrong log-prob ratios → silently broken PPO. Store/regenerate candidates deterministically.
- **Candidate padding/masking.** Forgetting the uniform-count requirement (Section 4) will either crash the reshape or batch garbage. Mask padded/illegal candidates to `-inf` in the softmax.
- **Leaving `tanh` on the policy head.** Use raw logits for the policy.
- **Sparse + long-horizon credit.** Long games with terminal-only reward stress the value baseline; if learning stalls, check value calibration before adding reward shaping.
- **Single-deck mirror self-play may not generalize.** The competition involves many decks and unknown opponents. Training only mirror self-play on `sample_deck` produces an agent good at *that* matchup. This is fine for getting the pipeline working; flag deck diversity (and proper opponent modeling) as a deliberate later step, not a silent assumption.
- **KL reset cadence.** If `K` is too small the anchor barely constrains anything; too large and it over-constrains and slows refinement. Tune around the paper's ratio once the basics work.
- **Do not reintroduce MCTS determinization into self-play.** It's unnecessary for model-free play and, configured wrong, actively incorrect.

---

## 11. Suggested first-milestone path

1. **Wire the game loop** with `random_agent` on both sides; confirm you can run full games and read outcomes via `result`. (Pure plumbing — no learning yet.)
2. **Plug the net in as actor-critic** and train with **PPO only (`α = 0`)** until it reliably beats `random_agent`. This validates rollout collection, per-player credit assignment, batching/masking, and log-prob bookkeeping — independently of the NASHPG-specific machinery.
3. **Add the KL-to-reference term + outer reset** (full NASHPG). Confirm stability (no divergence) and that it beats the MCTS starter and earlier checkpoints in head-to-head.
4. **Scale** net size / `K` / `T` as compute allows, and add periodic approximate-exploitability checks.

Throughout: keep `α = 0.2`, sample (don't argmax) at test time, and prefer the simplest correct version of each component over a clever one.