# NASHPG self-play for Pokémon TCG — implementation plan

*Refined, code-verified plan. Supersedes the earlier concept-only draft (an external instance that had the NashPG paper but not this codebase). Every load-bearing SDK/architecture claim below was checked against `tcg_model.py`, the vendored `cg` source, and the deleted prior art `git show 6da2bfc:train_reinforce.py`. Reference symbols by name + a grep recipe (no pinned line numbers); the `cg` engine and `all_card_data()` are ground truth.*

## 0. What this is

Train `MyModel` (the starter's Transformer: encoder→value, decoder→policy) by **model-free self-play PPO + a KL anchor to a periodically-reset frozen reference policy `ρ`** (NASHPG). No MCTS, no determinization in the training loop — the engine holds the true hidden state and each agent acts on its own observation.

**Two new knobs over plain PPO:** (1) add `α·KL(π_θ ‖ ρ)` to the loss over visited info sets; (2) every `K` inner updates, reset `ρ ← θ`. Keep `α=0.2` fixed (do **not** anneal — the reset is the mechanism). Intuition: plain self-play PG can cycle and converge to an exploitable policy; the KL anchor stabilizes each inner optimization toward a regularized equilibrium, and resetting the anchor chases the actual (unexploitable) one.

**Why now (lineage):** our own `reinforce-stage1-regresses` finding was that plain REINFORCE brackets between *flatten* (entropy too high) and *collapse* (gameable prize-diff) and "needs a **trust region** (KL-to-init/prev-iterate)." NASHPG's `α·KL(π‖ρ)` + reset *is* that trust region. So this supersedes the earlier ladder's plain-REINFORCE step and the divergent value-blend mirror self-play step.

**Locked v1 decisions:** **single-select policy nodes only** (multi-select & NUMBER decisions stay heuristic via `auto_resolve`); **from-scratch init** (fresh `MyModel`). **Goal of v1 = validate the learning mechanism locally on a tiny budget**, not win Kaggle (deferred).

## 1. Reuse vs. drop

**Reuse as-is** (from `tcg_model.py` unless noted): `MyModel`; `get_encoder_input(obs, your_deck)` / `get_decoder_input(obs, actions)` (sparse features — do **not** rebuild); `LearnInput` (sparse-batch assembler) + `SparseVector`; `eval_nn`; `random_agent`, `sample_deck`, card metadata. From `eval_mymodel.py`: `is_policy_select`, `auto_resolve`. From `eval.py`: `play_match`/`_play_one` (seat-swap), `wilson`, `sprt`, `Agent`. The driver loop from `cg/game.py` / `run_battle.py`. The **collection + masked-minibatch skeleton** from the deleted `train_reinforce.py` (`act_sample`, `train_iter`) — it already does empty-bag padding + `masked_fill(..., -1e9)`.

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
- **The battle API is a process-global singleton** — `battle_select(list[int])` takes no battle handle. ⇒ collection is **serial within a process**; parallelism is **multi-process only** (no threads).
- **Model-free play sidesteps the determinization bug for free** — `mcts_agent` hardcodes opponent active `[677]` (Riolu) and mirror-determinizes with `your_deck`. NASHPG never calls `search_begin`, so this is moot.

## 3. The trainer — new file `train_nashpg.py`

Standalone local trainer importing `tcg_model`; never entangled with the Kaggle `main.py` boundary.

**Per-decision record** (extend `act_sample` — it currently drops value & log-prob): `(sv_enc, sv_dec, n_opt, chosen_idx, old_logp, V_old)`, plus `ret`, `adv` filled at game end. **Storing the realized `sv_dec` makes the act-time and update-time candidate sets byte-identical** (see R11).

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
    L_v    = ((value.squeeze(-1) - ret)**2).mean()
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

AdamW **lr 3e-4**, **γ=1**, **clip 0.2**, **grad-clip 0.5**. **α (KL) = 0.2 fixed.** **Entropy coef ≈ 0.01** (NOT the paper's 0.1 — high entropy *flattened* our REINFORCE policy; the KL anchor already preserves stochasticity; raise only if it collapses). Advantage = MC (λ=1); GAE λ=0.95 only as an upgrade (R14). **Reward = terminal ±1/0 only** (prize-diff is Goodhart-gameable — confirmed in our run). Net `(128,2,256,1,1)`. **K/T/G/E are budget knobs — measure simulator games/sec FIRST**, then size them to minutes/inner-step locally; start tiny.

## 5. Risk register — what can go wrong

Severity: 🔴 likely to bite and/or silent · 🟡 real, detectable · 🟢 handled by design (stated so it stays handled).

### Conceptual (method may not learn even if coded perfectly)
- 🔴 **R1 — Mirror coin-flip variance is the binding constraint, and NASHPG does not fix it.** In a symmetric mirror the expected outcome is 50/50, so returns are ±1 with high variance; the value baseline only reduces that variance once `V` can predict mid-game advantage — which it learns *slowly from the same noisy ±1 returns* (chicken-and-egg, worsened by from-scratch init). The KL anchor cures *flatten/collapse* (the trust-region problem) but **not** the *credit-assignment/variance* problem the `reinforce-stage1-regresses` memo identified. **At locally-feasible `G`, the gradient SNR may be too low to demonstrate improvement at all** → an ambiguous "did it learn?" result. Mitigations: maximize `G` per update within the time budget; as a *debugging* signal, mix in some games vs the fixed rule-based teacher (asymmetric → clearer advantages, but off-Nash); accept that a clean demonstration may need Kaggle scale. Treat "loss moved" as necessary-not-sufficient; the real check is win-rate vs frozen past-self with CI separation.
- 🟡 **R2-conceptual — A single-deck-mirror Nash ≠ ladder strength.** The equilibrium of the `sample_deck` mirror is a specific mixed strategy; the competition is multi-deck vs unknown opponents, and a Nash policy is *unexploitable* but not *maximally exploitative* of weak opponents (which the ladder rewards). **v1 validates the mechanism, not Kaggle performance — don't over-claim.** Deck diversity / frozen-snapshot league is a deliberate later step.

### PPO-correctness (silently corrupts the objective — no crash, just wrong/zero learning)
- 🔴 **R3 — Collect on-policy at τ=1.** PPO's ratio assumes the behavior policy equals the policy being optimized. If you sample with a sharpening temperature (as the inherited REINFORCE `act_sample` did at τ=0.25/0.5) but optimize the untempered `π`, the importance ratio is wrong and updates are silently off-policy. **Sample directly from `π` (τ=1) during collection; reserve temperature for test-time only.**
- 🔴 **R4 — `old_logp` and `V_old` must be captured at collection time** from the same masked forward, under the exact distribution you sampled from. The inherited `act_sample` returns neither — extend it to emit both (the forward already computes `value` and `policy`). Recomputing `old_logp` later (e.g. from `θ` after an update, or from `ρ`) makes `ratio≈1` always → the clip never engages and PPO degenerates.
- 🟡 **R5 — Pad each batch item to `W` candidates immediately after its real candidates, per item.** `forward` does `p.reshape(batch, -1, d_model)`, which assumes a uniform candidate count. If you append all real words then all padding at the end, the reshape **scrambles candidates across items** — silent garbage. The existing trainers pad per-sample inside the loop; preserve that.
- 🟡 **R6 — Mask with `-1e9`, never `-inf`.** `0 · -inf = NaN` poisons the entropy and KL sums (a `0`-probability padded candidate times an `-inf` log term). `-1e9` is finite, so `exp(-1e9)=0` exactly and the product is `0`. This is exactly the NaN the deleted REINFORCE code was fixed to avoid.

### Optimization stability
- 🔴 **R7 — Shared encoder trunk: the value loss can destabilize the policy.** The encoder is shared (value head reads `encoder_out`; the policy decoder cross-attends to the same `encoder_out`). Value MSE against ±1 targets produces large gradients into the encoder that the policy depends on. **Keep `c_v` modest (≈0.5), and monitor `L_v` vs `L_clip` magnitudes**; if the policy degrades while value falls, lower `c_v` or stop-gradient the encoder for the value head.
- 🟡 **R8 — `tanh` saturation on the policy head** (from the `×10` convention) → vanishing gradient for confident actions. Tolerable because we want a mixed/high-entropy policy and the KL anchor resists collapse. **Escape hatch:** add a `policy_logits` flag to `forward` that skips the final `tanh` (then `logits = decoder_out` directly; re-tune the `×10`).
- 🟡 **R9 — PPO epochs overfit tiny local buffers.** `epochs=4` assumes paper-scale rollouts; with tiny `G` the buffer is small, so 4 clipped epochs over-optimize it. Use `E=1–2` locally. Advantage normalization (per-minibatch mean/std) helps SNR but interacts with the self-play sign — apply carefully and watch entropy.

### Throughput / feasibility
- 🔴 **R10 — `T·K·G` games can be locally infeasible.** Python rules engine + process-global-singleton battle + serial collection. Even `T=5,K=200,G=32` ≈ 32k games — hours at ~1 game/s. **Measure games/sec before sizing anything**; parallelize only across processes if needed. The smoke test proves plumbing, not learning; a "see if it moves" run must be sized to the measured rate (and may still be too small to clear R1).

### Evaluation
- 🟡 **R11-eval — `eval.py`'s `make_model` wraps checkpoints in `mcts_agent` (search).** Evaluating that way measures *MCTS-on-top-of-policy*, confounding "did the bare policy improve?". **Add a no-search sampled policy agent** (template: `eval_mymodel.py`'s `greedy_agent`; sample, don't argmax) and evaluate with that. Use the **rule-based teacher** (live) + **frozen past-self** (progress) as signals — **not vs-random** (saturated ~83% even untrained). n≥200, Wilson CIs, SPRT-gated promotion, locked `--search-count`. Engine is unseeded → no exact repro; conclusions ride on CIs.

### Handled by design (kept here so a future change doesn't break them)
- 🟢 **R12 — Candidate-set consistency is safe because decoder candidates are mutually independent.** `DecoderLayer` does cross-attention to the encoder only (`self.attention(x, encoder_out, encoder_out)`) with **no self-attention among candidate words** — so each candidate's logit depends only on itself + the encoder, padding can't corrupt real candidates, and the batch=1 collection forward equals the padded batched update forward. Storing `sv_dec` then guarantees byte-identical candidates. **Re-verify this independence if the decoder ever gains self-attention** (it would silently break PPO ratios).
- 🟢 **R13 — `ρ` is a genuinely frozen separate module:** built via `load_state_dict` into its own `MyModel`, `.eval()`, forwarded under `no_grad`, and **never added to the optimizer**. The optimizer is created once on `θ` and its Adam state persists across resets (only `ρ` resets). The first outer iteration anchors to the random init (high-entropy ≈ uniform — fine).
- 🟢 **R14 — MC advantage (λ=1) needs no per-player next-state bookkeeping** — every decision in a seat's subsequence gets the terminal `G`. Only a GAE λ<1 upgrade requires sequencing each seat's *own* next decision state with `V(terminal)=0`; do that bookkeeping then, not now.

## 6. Milestones & verification

1. **Smoke** (plumbing): `python train_nashpg.py --outer 2 --inner 2 --games 4 --eval-games 20` → finite loss each update, every game reaches `result>=0`, prints win-rates with Wilson CIs.
2. **PPO-only (α=0)**: a short run moves **win-rate-vs-teacher up with CI separation from frozen init** — explicitly *not* "beats random" (true at init). If R1 bites, this is where it shows (flat/noisy).
3. **Full NASHPG (α=0.2)**: no divergence (entropy healthy, `V` calibrates — check predicted `V` tracks outcome), beats frozen past-self in the mirror.
4. **Periodic (optional)**: approximate exploitability — train a fresh best-response vs the frozen policy; expensive, checkpoint cadence only.

## 7. Deferred

Full action enumeration (multi-select/NUMBER coverage) · deck diversity / frozen-snapshot league · GAE λ<1 · Kaggle push-training + replay loader · routine approx-exploitability · removing the policy `tanh`.
