# NASHPG self-play for Pokémon TCG — implementation plan

*Refined, code-verified plan. Supersedes the earlier concept-only draft (an external instance that had the NashPG paper but not this codebase). Every load-bearing SDK/architecture claim below was checked against `tcg_model.py`, the vendored `cg` source, and the in-tree `train_reinforce.py` (whose header already documents the PPO-porting traps for exactly this transition). Now also cross-checked line-by-line against the **original NashPG implementation** in `nash_policy_gradient_public/` — see §2.5. Reference symbols by name + a grep recipe (no pinned line numbers); the `cg` engine and `all_card_data()` are ground truth.*

## 0. What this is

Train `MyModel` (the starter's Transformer: encoder→value, decoder→policy) by **model-free self-play PPO + a KL anchor to a periodically-reset frozen reference policy `ρ`** (NASHPG). No MCTS, no determinization in the training loop — the engine holds the true hidden state and each agent acts on its own observation.

**Two new knobs over plain PPO:** (1) add `α·KL(π_θ ‖ ρ)` to the loss over visited info sets; (2) every `K` inner updates, reset `ρ ← θ`. Keep `α=0.2` fixed (do **not** anneal — the reset is the mechanism). Intuition: plain self-play PG can cycle and converge to an exploitable policy; the KL anchor stabilizes each inner optimization toward a regularized equilibrium, and resetting the anchor chases the actual (unexploitable) one.

**Why now (lineage):** our own `reinforce-stage1-regresses` finding was that plain REINFORCE brackets between *flatten* (entropy too high) and *collapse* (gameable prize-diff) and "needs a **trust region** (KL-to-init/prev-iterate)." NASHPG's `α·KL(π‖ρ)` + reset *is* that trust region. So this supersedes the earlier ladder's plain-REINFORCE step and the divergent value-blend mirror self-play step.

**Locked v1 decisions:** **single-select policy nodes only** (multi-select & NUMBER decisions stay heuristic via `auto_resolve`); **from-scratch init** (fresh `MyModel`). **Goal of v1 = validate the learning mechanism locally on a tiny budget**, not win Kaggle (deferred).

## 1. Reuse vs. drop

**Reuse as-is** (from `tcg_model.py` unless noted): `MyModel`; `get_encoder_input(obs, your_deck)` / `get_decoder_input(obs, actions)` (sparse features — do **not** rebuild); `LearnInput` (sparse-batch assembler) + `SparseVector`; `eval_nn`; `random_agent`, `sample_deck`, card metadata. From `eval_mymodel.py`: `is_policy_select`, `auto_resolve`. From `eval.py`: `play_match`/`_play_one` (seat-swap), `wilson`, `sprt`, `Agent`. The driver loop from `cg/game.py` / `run_battle.py`. The **collection + masked-minibatch skeleton** from the in-tree `train_reinforce.py` (`act_sample`, `train_iter`) — it already does empty-bag padding + `masked_fill(..., -1e9)`, and its header enumerates the PPO-porting traps (R3/R4).

**Drop / ignore** (all MCTS-specific): `mcts_agent`, `Node`, `Child`, `create_node`, `_rollout_value`, `SEARCH_COUNT`, the determinization in `mcts_agent` (`search_begin/step/end`), the `exp(policy·10)` prior sharpening, and the visit-count policy targets.

## 2. Verified facts & corrections (don't trust the old draft blindly)

| Claim from the concept draft | Verdict | Reality / consequence |
|---|---|---|
| net = 128-dim / **4 heads / 2+2 layers** | ⚠️ wrong | everywhere uses `MyModel(128, 2, 256, 1, 1)` → **2 heads, 1+1 layers**. Keep for v1. |
| 24 encoder words/state | ✅ | `grep "^num_words_encoder" tcg_model.py`. |
| "classic value sign-flip bug" | ⚠️ misframed | `get_encoder_input` frames everything by `yourIndex` (`players[i ^ your_index]`, `players[your_index].hand`). So **V is acting-seat-relative → NO sign-flip needed** in per-player self-play. (The MCTS flip is a tree-sharing artifact.) Residual risk is only pairing each record with the right seat's terminal reward. |
| "remove tanh, use raw logits" | ⚠️ not free | the policy `tanh` is **inside `forward`** (`grep "p = torch.tanh" tcg_model.py`). **Don't touch the model** — reuse the codebase's existing convention **`logits ≡ decoder_out × 10`** (MCTS uses `exp(p·10)`; BC-CE uses `logits=(od·10)`), so `π = softmax(10·tanh(decoder_out))` over masked-legal candidates. Zero model change; tanh-bounded logits are a free mild trust region. Escape hatch (R7). |
| `exp(policy·10)` is in the policy head | ⚠️ location | it's in `create_node` (MCTS only), not `forward`; irrelevant once model-free. |
| keep value `tanh`; reward `<0`/`==idx`/`==2` | ✅ | `v = tanh(v.mean(0))` ∈ [-1,1] matches ±1 reward; `result` semantics confirmed in `docs/cg-sdk.md` + `game.py`. |
| candidate cap 64 | ✅ but MCTS-only | for single-select v1, candidates are just `[[0]…[n_opt-1]]`; the 64-combo enumeration is unused. |
| no RNG seed in the ABI | ✅ | CRN impossible → lean on n + Wilson CIs + SPRT. |

**Two facts the old draft omitted that shape the design:**
- **The battle API is a process-global singleton** — `battle_select(list[int])` takes no battle handle. ⇒ collection is **serial within a process**; parallelism is **multi-process only** (no threads). Note: we want to start simple, so we shouldn't worry about parallelization yet.
- **Model-free play sidesteps the determinization bug for free** — `mcts_agent` hardcodes opponent active `[677]` (Riolu) and mirror-determinizes with `your_deck`. NASHPG never calls `search_begin`, so this is moot.

## 2.5 Provenance check vs. the original NashPG code (`nash_policy_gradient_public/`)

The original (JAX/Flax/`distrax`) was read in full. **Our recipe is faithful**: NASHPG = PPO-clip + entropy + **forward-KL to a frozen "magnet" `ρ`, reset `ρ←θ` every outer iteration, α (`mag_coef`) = 0.2 fixed.** Verbatim (`grep -n "ppo_loss\|mag_loss" train/core/update_agent.py`):

```
ratio      = exp(log_prob - dataset.log_prob)
ppo_loss   = -mean( min(ratio*adv, clip(ratio, 1±ε)*adv) )       # PPO-clip, ε=0.2
entropy    = -mean( dists.entropy() )
mag_kl     = mean( KL(π_θ ‖ π_ρ) )      # FORWARD KL, closed-form over the full categorical
actor_loss = ppo_loss + ent_coef*entropy + mag_coef*mag_kl       # + a PPO-clipped critic loss
```
Reset (`grep -n "mag_agent" train/nash_pg.py`): `learner_state.mag_agent = nnx.clone(learner_state.agent)`, once per **outer** iteration. Config (`conf/algorithm/nash_pg.yaml`): `mag_coef 0.2`, `ent_coef 0.05`, `clip 0.2`, `num_inner_update 1000`, `num_outer_update 50`, grad-norm-clip 0.5, AdamW, `gamma 1.0`. **MMD vs NashPG, same repo:** MMD passes `mag_agent=None` → backward-KL to the *previous iterate* (`ratio·log_ratio−(ratio−1)`); **RNaD** folds `η·logπ` into the *reward* with V-trace+NERD. We deliberately target the NashPG (frozen-magnet, **loss**-KL) variant — keep it that way.

**Three divergences from this plan — all ⚑ flagged for independent theory review:**

| Knob | This plan (before) | Original (ground truth) | Decision |
|---|---|---|---|
| **Advantage** | MC λ=1; GAE deferred (R14) | **GAE λ=0.95 + learned critic** (reverse scan, per-player carry) | ⚑ **Promote GAE to first-class.** Variance is our R1 binding constraint and GAE is the original's main variance tool. MC λ=1 stays the *smoke* default; GAE λ=0.95 is the real-run path. |
| **Entropy coef** | 0.01 | **0.05** | ⚑ Sweep 0.01–0.05 (CLI flag). Our 0.01 came from *plain* REINFORCE (no anchor, gameable reward); 0.05 is *with* the KL anchor + GAE — a different regime. Watch entropy and `mag_kl`. |
| **Value loss** | plain MSE `(V−ret)²` | **PPO-clipped**: `0.5·max((v−vt)², (v_clip−vt)²)`, `v_clip = v_old + clip(v−v_old, ±ε)` | ⚑ Adopt the clipped form — guards the shared trunk (R7), standard, ~free. |

Matches we already had right: per-batch advantage normalization (R9), `gamma=1`, terminal ±1 reward (no shaping), `-1e9` masking (R6 — they use `distrax` masking; framework difference, behaviorally identical), `num_ppo_epoch=4` at *their* scale (we keep E=1–2 locally, R9).

## 3. The trainer — new file `train_nashpg.py`

Standalone local trainer importing `tcg_model`; never entangled with the Kaggle `main.py` boundary.

**Per-decision record** (extend `act_sample` — it currently drops value & log-prob), adopting the original's richer `Transition`/`Dataset` field set (`grep -n "class Transition\|class Dataset" train/mytypes.py`): `(sv_enc, sv_dec, n_opt, chosen_idx, old_logp, V_old, current_player, is_new_eps)`, plus `ret`, `adv`, `valid_mask` filled at game end. **Storing the realized `sv_dec` makes the act-time and update-time candidate sets byte-identical** (see R12). The `current_player`/`is_new_eps` fields are what make the **per-player GAE carry** clean (R14): the original's `calculate_one_env_gae` (`grep -n "calculate_one_env_gae" train/core/prepare_data.py`) keys each TD error off `next_value[current_player]`, i.e. it sequences each *seat's own* next decision with `V(terminal)=0` — copy that bookkeeping when GAE is enabled.

**Policy primitive:**
```
value, p = forward(sv_enc, sv_dec)      # p ∈ [-1,1]^K (post-tanh)
logits = p * 10.0;  logits[pad/illegal] = -1e9     # finite, NOT -inf (R6)
logp = log_softmax(logits); pi = softmax(logits)
```

**Collection** — one inner step = `G` self-play games; the *same* `θ` pilots both seats:
```
obs,_ = battle_start(sample_deck, sample_deck)
recs = [[], []]
while obs["current"]["result"] < 0:
    yi = obs["current"]["yourIndex"]; sel = to_observation_class(obs).select
    if is_policy_select(sel):
        idx ~ pi   # SAMPLE AT τ=1 (R2); store old_logp=logp[idx], V_old=value
        recs[yi].append(record); selected=[idx]
    else:
        selected = auto_resolve(sel)        # no record
    obs = battle_select(selected)
battle_finish(); r = obs["current"]["result"]
for seat in (0,1):
    G_seat = 0.0 if r==2 else (1.0 if r==seat else -1.0)   # already seat-relative
    for rec in recs[seat]: rec.ret = G_seat; rec.adv = G_seat - rec.V_old   # MC baseline (GAE λ=1)
buffer += recs[0] + recs[1]
```

**PPO update:**
```
for epoch in range(E):                       # E small for tiny buffers (R8)
  for mb in minibatches(buffer):             # pad each item to W candidates per-item (R5)
    value, p = θ.forward(...);  with no_grad: _, p_ref = ρ.forward(...)
    logp=log_softmax(p*10 masked); pi=softmax(...); logp_ref=log_softmax(p_ref*10 masked)
    ratio = exp(logp[chosen] - old_logp)
    L_clip = -min(ratio*adv, clamp(ratio,1±ε)*adv).mean()
    L_v    = ((value.squeeze(-1) - ret)**2).mean()           # v1 SMOKE: plain MSE
    # REAL RUNS — original's PPO-CLIPPED value loss (guards the shared trunk, R7):
    #   v_clip = V_old + clamp(value - V_old, -ε, +ε)
    #   L_v    = 0.5 * max((value-ret)**2, (v_clip-ret)**2).mean()
    H      = -(pi*logp).sum(-1).mean()
    KL     = (pi*(logp - logp_ref)).sum(-1).mean()
    loss = L_clip + c_v*L_v - c_H*H + α*KL
    loss.backward(); clip_grad_norm_(θ, 0.5); opt.step()
```

**Outer loop & reset:**
```
θ = MyModel(128,2,256,1,1); ρ = frozen_copy(θ); opt = AdamW(θ, 3e-4)   # ρ: eval(), no_grad, NOT in opt (R12)
for t in range(T):
    for k in range(K): collect(G); ppo_update()
    ρ.load_state_dict(θ.state_dict())          # reference reset
    torch.save(θ.state_dict(), out_path(t)); eval_checkpoint(t)
```

**CLI:** `--outer T --inner K --games G --epochs E --batch-size --lr --clip --alpha --ent-coef --vf-coef --eval-games --out-prefix`. **Smoke** (smallest verifiable signal): `--outer 2 --inner 2 --games 4` ⇒ finite loss every update + every game reaches `result>=0`.

## 4. Hyperparameters

AdamW **lr 3e-4**, **γ=1**, **clip 0.2**, **grad-clip 0.5**. **α (KL) = 0.2 fixed.** **Entropy coef — sweep 0.01–0.05** (⚑ the original uses 0.05; our earlier "0.01, not the paper's 0.1" reasoning came from *plain* REINFORCE with no KL anchor + a gameable reward — a different regime now that the anchor + GAE are in. Watch entropy and `mag_kl`; raise if it collapses, lower if it flattens). **Advantage: MC λ=1 for the v1 smoke default, GAE λ=0.95 as the first-class real-run path** (⚑ promoted off R14's deferred list — GAE is the original's primary variance reducer and variance is our R1 bottleneck; needs the per-player carry from §3). **Value loss: plain MSE for smoke, PPO-clipped for real runs** (§3). **Reward = terminal ±1/0 only** (prize-diff is Goodhart-gameable — confirmed in our run). Net `(128,2,256,1,1)`. **K/T/G/E are budget knobs — measure simulator games/sec FIRST**, then size them to minutes/inner-step locally; start tiny.

## 5. Risk register — what can go wrong

Severity: 🔴 likely to bite and/or silent · 🟡 real, detectable · 🟢 handled by design (stated so it stays handled).

### Conceptual (method may not learn even if coded perfectly)
- 🔴 **R1 — Mirror coin-flip variance is the binding constraint, and NASHPG does not fix it.** In a symmetric mirror the expected outcome is 50/50, so returns are ±1 with high variance; the value baseline only reduces that variance once `V` can predict mid-game advantage — which it learns *slowly from the same noisy ±1 returns* (chicken-and-egg, worsened by from-scratch init). The KL anchor cures *flatten/collapse* (the trust-region problem) but **not** the *credit-assignment/variance* problem the `reinforce-stage1-regresses` memo identified. **At locally-feasible `G`, the gradient SNR may be too low to demonstrate improvement at all** → an ambiguous "did it learn?" result. Mitigations: maximize `G` per update within the time budget; as a *debugging* signal, mix in some games vs the fixed rule-based teacher (asymmetric → clearer advantages, but off-Nash); accept that a clean demonstration may need Kaggle scale. Treat "loss moved" as necessary-not-sufficient; the real check is win-rate vs frozen past-self with CI separation.
- 🟡 **R2-conceptual — A single-deck-mirror Nash ≠ ladder strength.** The equilibrium of the `sample_deck` mirror is a specific mixed strategy; the competition is multi-deck vs unknown opponents, and a Nash policy is *unexploitable* but not *maximally exploitative* of weak opponents (which the ladder rewards). **v1 validates the mechanism, not Kaggle performance — don't over-claim.** Deck diversity / frozen-snapshot league is a deliberate later step.

### PPO-correctness (silently corrupts the objective — no crash, just wrong/zero learning)
- 🔴 **R3 — Collect on-policy at τ=1.** PPO's ratio assumes the behavior policy equals the policy being optimized. If you sample with a sharpening temperature (as the inherited REINFORCE `act_sample` did at τ=0.25/0.5) but optimize the untempered `π`, the importance ratio is wrong and updates are silently off-policy. **Sample directly from `π` (τ=1) during collection; reserve temperature for test-time only.**
- 🔴 **R4 — `old_logp` and `V_old` must be captured at collection time** from the same masked forward, under the exact distribution you sampled from. The inherited `act_sample` returns neither — extend it to emit both (the forward already computes `value` and `policy`). Recomputing `old_logp` later (e.g. from `θ` after an update, or from `ρ`) makes `ratio≈1` always → the clip never engages and PPO degenerates.
- 🟡 **R5 — Pad each batch item to `W` candidates immediately after its real candidates, per item.** `forward` does `p.reshape(batch, -1, d_model)`, which assumes a uniform candidate count. If you append all real words then all padding at the end, the reshape **scrambles candidates across items** — silent garbage. The existing trainers pad per-sample inside the loop; preserve that.
- 🟡 **R6 — Mask with `-1e9`, never `-inf`.** `0 · -inf = NaN` poisons the entropy and KL sums (a `0`-probability padded candidate times an `-inf` log term). `-1e9` is finite, so `exp(-1e9)=0` exactly and the product is `0`. This is exactly the NaN the in-tree REINFORCE code was fixed to avoid.

### Optimization stability
- 🔴 **R7 — Shared encoder trunk: the value loss can destabilize the policy.** The encoder is shared (value head reads `encoder_out`; the policy decoder cross-attends to the same `encoder_out`). Value MSE against ±1 targets produces large gradients into the encoder that the policy depends on. **Keep `c_v` modest (≈0.5), and monitor `L_v` vs `L_clip` magnitudes**; if the policy degrades while value falls, lower `c_v` or stop-gradient the encoder for the value head.
- 🟡 **R8 — `tanh` saturation on the policy head** (from the `×10` convention) → vanishing gradient for confident actions. Tolerable because we want a mixed/high-entropy policy and the KL anchor resists collapse. **Escape hatch:** add a `policy_logits` flag to `forward` that skips the final `tanh` (then `logits = decoder_out` directly; re-tune the `×10`).
- 🟡 **R9 — PPO epochs overfit tiny local buffers.** `epochs=4` assumes paper-scale rollouts; with tiny `G` the buffer is small, so 4 clipped epochs over-optimize it. Use `E=1–2` locally. Advantage normalization (per-minibatch mean/std) helps SNR but interacts with the self-play sign — apply carefully and watch entropy.

### Throughput / feasibility
- 🔴 **R10 — `T·K·G` games can be locally infeasible.** Python rules engine + process-global-singleton battle + serial collection. Even `T=5,K=200,G=32` ≈ 32k games — hours at ~1 game/s. **Measure games/sec before sizing anything**; parallelize only across processes if needed. The smoke test proves plumbing, not learning; a "see if it moves" run must be sized to the measured rate (and may still be too small to clear R1).

### Evaluation
- 🟡 **R11-eval — `eval.py`'s `make_model` wraps checkpoints in `mcts_agent` (search).** Evaluating that way measures *MCTS-on-top-of-policy*, confounding "did the bare policy improve?". **Add a no-search sampled policy agent** (template: `eval_mymodel.py`'s `greedy_agent`; sample, don't argmax) and evaluate with that. Use the **rule-based teacher** (live) + **frozen past-self** (progress) as signals — **not vs-random** (saturated ~83% even untrained). n≥200, Wilson CIs, SPRT-gated promotion, locked `--search-count`. Engine is unseeded → no exact repro; conclusions ride on CIs. For an *absolute* (not just relative-to-past-self) progress number, add the RL-best-response exploitability milestone (§6.4) — the only game-tree-free way to ask "how exploitable is this policy?".

### Handled by design (kept here so a future change doesn't break them)
- 🟢 **R12 — Candidate-set consistency is safe because decoder candidates are mutually independent.** `DecoderLayer` does cross-attention to the encoder only (`self.attention(x, encoder_out, encoder_out)`) with **no self-attention among candidate words** — so each candidate's logit depends only on itself + the encoder, padding can't corrupt real candidates, and the batch=1 collection forward equals the padded batched update forward. Storing `sv_dec` then guarantees byte-identical candidates. **Re-verify this independence if the decoder ever gains self-attention** (it would silently break PPO ratios).
- 🟢 **R13 — `ρ` is a genuinely frozen separate module:** built via `load_state_dict` into its own `MyModel`, `.eval()`, forwarded under `no_grad`, and **never added to the optimizer**. The optimizer is created once on `θ` and its Adam state persists across resets (only `ρ` resets). The first outer iteration anchors to the random init (high-entropy ≈ uniform — fine).
- 🟢 **R14 — MC advantage (λ=1) needs no per-player next-state bookkeeping** — every decision in a seat's subsequence gets the terminal `G`; this is the v1 *smoke* default. The now-first-class GAE λ=0.95 path (§4, §2.5) **does** require sequencing each seat's *own* next decision state with `V(terminal)=0` — the original's `calculate_one_env_gae` does exactly this via a per-`current_player` carry (§3). Copy that bookkeeping when enabling GAE; until then the MC path is correct and self-contained.

## 6. Milestones & verification

**Instrument every run with the original's PPO health diagnostics** (copy the metric set from `grep -n "MultiMetric" train/nash_pg.py`): log per update `ppo_loss`, `entropy`, `critic_loss`, `approx_kl`, `mag_kl`, `clip_frac`, `explained_var`. These turn "did it learn?" from a guess into signal — `clip_frac` says the clip is actually engaging (R3/R4), `approx_kl` flags too-large steps, `explained_var` says the critic is learning (central to R1), `mag_kl` says the anchor is doing something. A tiny JSON-lines logger suffices (their `BaseLogger`/`MultiLogger` is the shape to imitate); TensorBoard optional.

1. **Smoke** (plumbing): `python train_nashpg.py --outer 2 --inner 2 --games 4 --eval-games 20` → finite loss each update, every game reaches `result>=0`, the diagnostics above print, win-rates with Wilson CIs.
2. **PPO-only (α=0)**: a short run moves **win-rate-vs-teacher up with CI separation from frozen init** — explicitly *not* "beats random" (true at init). If R1 bites, this is where it shows (flat/noisy returns, `explained_var` stuck near 0).
3. **Full NASHPG (α=0.2)**: no divergence (entropy healthy, `mag_kl` nonzero-but-bounded, `explained_var` rising — predicted `V` tracks outcome), beats frozen past-self in the mirror.
4. **Approximate exploitability via RL best-response** (copied from `eval/compute_exploitability_rl.py`; the principled answer to R1 — needs **no game tree**, so it runs on the black-box `cg`): freeze a checkpoint as the fixed opponent, train a *fresh* policy as a best-response (BR) against it (PPO, single-seat — learn **only** from the BR's own decisions, `only_use_player0_experience` in the original), and its converged **average return is the exploitability estimate** (lower ⇒ closer to unexploitable). **We get this nearly free:** it's `train_nashpg.py` with `α=0`, `mag_coef=0`, opponent = a frozen checkpoint, learning only our seat — e.g.
   ```
   python train_nashpg.py --br-vs out/nashpg_t.pth --br-updates 500 --alpha 0   # BR return ⇒ exploitability
   ```
   ⚑ Caveat: a *weak/under-trained* BR **under-estimates** exploitability (false sense of safety) — report BR return **as a curve over BR-updates** and read it as a *lower bound*. (Exact OpenSpiel/EAS exploitability is **not** applicable — both need an enumerable game tree; `cg` has none.)

## 6.5 Reference repo (`nash_policy_gradient_public/`): copy vs. skip

**Verdict — do NOT port the code; reimplement from it as the spec.** It is JAX/Flax(`nnx`)/`optax`/`distrax` built around **vectorized batched envs** (`num_envs` stepped via `jax.vmap` + `nnx.scan`; collection is `nnx.scan(num_steps) → nnx.vmap(num_envs)` in `train/core/prepare_data.py`); we are PyTorch `MyModel` + the **serial `cg` process-global singleton**. Their throughput *design* (parallel envs) is exactly what we can't reuse, and JAX→torch + de-vectorizing the loop is more work than a clean reimplement. Two architecture facts seal it: (a) **their agents have a fixed action head** (`Linear(feat→NUM_ACTIONS)`, self-attention-only encoder) — our `MyModel` instead *cross-attends a variable candidate set per-candidate* (R12), so their agent/model code is non-reusable, which conversely **validates our single-select candidate-enumeration design** as the right adapter for a variable action space; (b) their `AutoResetWrapper` is `jax.lax.cond` — we use a plain Python `if done: battle_finish(); battle_start(...)` (our serial-collection stance, §2, is correct and unavoidable).

**Copy as spec / schema / concept:**

| Reference | What we take |
|---|---|
| `train/core/update_agent.py` loss | the exact PPO + entropy + forward-KL + clipped-critic algebra (reimplement in torch) |
| `train/core/prepare_data.py` GAE (`calculate_one_env_gae`) | the per-`current_player` reverse-scan carry = the R14 bookkeeping |
| `train/mytypes.py` `Transition`/`Dataset` | the field schema (§3 record upgrade) |
| `train/nash_pg.py` outer/inner + `mag_agent = nnx.clone(agent)` | the loop structure (1:1 with §3) |
| `eval/compute_exploitability_rl.py` | the RL-best-response exploitability milestone (§6.4) |
| `nnx.MultiMetric` diagnostics + JSON/TB loggers | the diagnostics to log + a small `BaseLogger`-shaped JSONL logger (§6 header) |
| `agents/mixture_agent.py` | concept for the **deferred** frozen-snapshot league (PSRO-style meta-distribution over snapshots) |

**Skip:** their envs, vectorized collection, `optax`/Orbax checkpointing, OpenSpiel/EAS exact-exploitability (need an enumerable game tree — impossible for `cg`), `auto_reset_wrapper`, and Hydra config groups (heavier than our argparse + the CLAUDE.md "minimal" rule — keep argparse for v1; note Hydra as a later option). Their `head2head.py` reports only mean±std — **our `eval.py` is already stronger** (seat-swapped `play_match` + Wilson CIs + SPRT); keep ours, only graft "frozen past-self as opponent" as a first-class progress signal (R11).

## 7. Deferred

Full action enumeration (multi-select/NUMBER coverage) · deck diversity / frozen-snapshot league (the `mixture_agent` concept, §6.5) · Kaggle push-training + replay loader · *routine/periodic* approx-exploitability at every checkpoint (the one-off BR milestone itself is now promoted to §6.4) · removing the policy `tanh`. (GAE λ<1 is **no longer deferred** — promoted to first-class in §4.)
