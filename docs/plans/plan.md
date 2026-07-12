# Plan: an id-aware policy, real multi-select conditioning, and imitation warm-start

**Status: IMPLEMENTED 2026-07-09 — all phases (0/1/2/M/B1/B2/B3) landed and verified; results in `../reports/bc_warmstart_report.md` (headline: pure-BC 20.0% pooled mirror passes G1; BC+tuned fine-tune 24.0%, first lever to ever move the ~13% plateau).** Previously: approved 2026-07-09; refined 2026-07-09 (second review pass — simplifications, tightened phasing and verification). Supersedes the "Deferred levers" (§8) of [`plan-multiselect.md`](plan-multiselect.md), which stays as the reference for the multi-select machinery — this plan reuses that machinery unchanged.

Design facts below were verified against source **twice** (the second pass independently re-checked every load-bearing claim): `sb3_features.py`, `sb3_env.py`, `train_sb3.py`, `eval_sb3.py`, `tcg_model.py`, the installed `sb3_contrib` 2.9.0 (`common/maskable/policies.py`, `common/maskable/distributions.py`, `ppo_mask/ppo_mask.py`, `common/maskable/buffers.py`) and `stable_baselines3` (buffers, preprocessing, save/load, `obs_as_tensor`), `credit_probe.py`, `record_battle.py`, the four rule-based teachers, plus the findings in `latest_runs_findings.md` / `../reports/sb3_selfplay_report.md` / `../reports/credit_probe_report.md`.

## 1. Where we are and why these changes

Mental model of today's stack: at every decision the env hands the net a **dashboard** (`state[303]`: counts, HP bars, flags) and a **menu** (`options[128,72]`: one feature row per legal option, padded to 128 slots). The stock `MultiInputPolicy` flattens everything and a positional head emits one logit per menu slot.

Three measured problems (sources: `latest_runs_findings.md`, `../reports/sb3_selfplay_report.md`, `../reports/credit_probe_report.md`):

- **The net is card-blind.** A card is described only by ~26 hand-crafted stats (HP, type, stage…). Riolu and any other 70-HP basic look nearly identical; every ability/effect is invisible. The id indices that would fix this are already computed by `encode_state`/`encode_options` and then **thrown away** in `obs_to_gym` (`sb3_env.py`) because `MultiInputPolicy` can't embed integers. There are only 1268 card ids and 1557 attack ids in the whole competition (from the engine: `card_count`/`attack_count` in `tcg_model.py`) — small enough to give **every card its own learned embedding row**: a per-card "cheat-sheet" vector the net fills in during training with whatever matters about that card that the hand-crafted stats don't say. Tables cost ~(1269×32 + 1558×16) ≈ 65k params — negligible.
- **The positional head can't say "this option is Card X."** Logit *i* means "slot *i*", and slot *i* is a different card at every node. The fix (long flagged as **F6**) is a **per-option scorer**: one shared MLP scores each option row (its dense features + its card/attack embeddings) against the board context — the same score function applied to every row, so "what this card is" finally drives "how good this pick is".
- **The plateau is a credit problem, not just capacity.** PPO saturates by ~150k steps at ~13% pooled mirror and loses all 16 deck×opponent eval cells; the trained critic explains *less* outcome variance (ev 0.089) than a supervised fit on the same features (0.167). Scale, self-play, and ε-curriculum were all flat. So besides the architecture we add the classic fix: **clone the rule-based teachers first** (supervised, per-decision signal — no credit-assignment problem), warm-start the critic on real game outcomes, then let PPO fine-tune.

What we are deliberately NOT doing (and why) — see §8.

## 2. Change A — stop discarding ids; extend the observation

New obs keys in `obs_to_gym` (`sb3_env.py`), all integer Boxes (`dtype=np.int64`, finite bounds). The PAD sentinels **already exist** in `sb3_features.py`: `CARD_PAD = 1268` / `ATTACK_PAD = 1557` (the extra index past the last real id — id 0 is a real card, so 0 must never be used as filler). Only `H_MAX`/`D_MAX` and `encode_zone_ids` are new.

| key | shape | content |
|---|---|---|
| `state_card_ids` | [12] | active + 5 bench, both sides (already computed by `encode_state`) |
| `opt_card_ids` | [128,2] | per option: primary + target card (already computed by `encode_options`) |
| `opt_attack_ids` | [128,1] | per option: attack id (already computed) |
| `hand_ids` | [H_MAX=20] | NEW: my hand's card ids (`encode_zone_ids`, new in `sb3_features.py`) |
| `my_discard_ids` / `opp_discard_ids` | [60] each | NEW: both discard piles (public info; deck-size-bounded) |

Why the zone keys: today the net literally cannot see its own hand (only `handCount`) — unplayable-but-important cards (an evolution waiting for next turn) are invisible; discard piles matter for recycling decks (Iono). All three are plain reads off the `Observation` — verified: `PlayerState.hand` (own side) and `discard` (both sides) carry real cards. Hand overflow past `H_MAX` is truncated + counted in a telemetry counter (mirror of `CGEnv.overflows`). Raising `H_MAX` later is a *cheap* break — the pooled embedding means no weight shapes change — but it is still an obs-space break: SB3's `check_for_correct_spaces` rejects old zips.

Pad-row and STOP-row ids are `CARD_PAD`/`ATTACK_PAD` (never 0 — see §7 risk 1); id arrays are initialized to the PAD value before filling the first `n` rows. `observation_space`, `_terminal_obs`, and the module self-tests extend accordingly. **Checkpoint break** (the obs got wider): fresh `--out-prefix`; the `--init-from` obs-space assert (and SB3's own space check) rejects old zips loudly. No back-compat — project decision.

`sb3_features.py` gains `encode_zone_ids(o)` + `H_MAX`/`D_MAX` as a new sibling function; the frozen base encoders stay untouched.

## 3. Change B — the new network: `tcg_net.py` + `sb3_policy.py` (thin SB3 wrapper)

### 3.1 `TcgCore(nn.Module)` in new file `tcg_net.py` — plain torch

`TcgCore` is a plain torch module (imports torch only) for one practical reason: **BC (§4) trains this exact module directly, with no SB3 in the loop, and `--bc-init` then loads it into the SB3 policy as a named submodule** (`model.policy.core.load_state_dict(...)`) — one weights file, no state_dict prefix surgery. (Side effect, not a driver: a torch-only core also makes a future SB3-free submission mirror easy — deferred, see §8.)

Concrete picture, in order: **embed every id → describe every option → summarize the board → score each option against the summary.**

- **Embeddings**: `card_emb = nn.Embedding(card_count+1, d_card, padding_idx=CARD_PAD)` — ONE table shared by board slots, option primary/target, hand, discards (one cheat-sheet per card, learned from every context it appears in); `attack_emb` likewise. `padding_idx` makes PAD rows exact zeros, so masked-mean pooling is just `sum / count.clamp(min=1)`. **Both tables are zero-initialized**: only ~40–60 distinct cards appear in our 4-deck world, so most of the ~1268 rows never receive a gradient — zero-init means a never-seen card contributes *nothing beyond its dense features* (it reads like PAD in the embedding channel but stays distinguishable via its dense stats), instead of frozen random noise. Gradients flow into zero rows normally, so nothing is lost for trained cards.
- **Per-option latent**: `opt_latent_i = MLP_opt(concat(options[i, :72], card_emb(primary_i), card_emb(target_i), attack_emb(attack_i)))` → `[B,128,h]`. Applied to all 128 rows (the STOP row = its is_stop bit + PAD embeddings; pad rows produce garbage latents that the mask keeps out of every loss and pool).
- **Context vector** — the "board summary" every score sees, `ctx = MLP_ctx(concat(...))` of six named pieces:
  1. `state[303]` (the dashboard),
  2. pooled board-slot card embeddings (`state_card_ids`),
  3. pooled hand embedding (`hand_ids`),
  4. pooled my/opp discard embeddings,
  5. masked-mean of `opt_latents` over currently-selectable rows (what's on the menu),
  6. masked-mean of `opt_latents` over already-picked rows (col 71 flags them).
  Piece 6 is the **autoregressive multi-select conditioning**: mid-node, the net sees a summary of what it already picked — "I already took the Water energy, now pick accordingly."
- **Heads**: `logits[:, i] = MLP_score(concat(ctx, opt_latent_i))` shared across i → `[B,128]`; `value = MLP_val(ctx)` → `[B,1]`, final layer plain linear (no tanh — must match PPO value semantics and the BC regression target in {−1,0,+1}). Init: torch defaults, except final score layer scaled ×0.01 (keeps the initial policy near-uniform, which PPO stability relies on) and value gain 1.0.
- Sizes: start d_card=32, d_atk=16, h≈128, ctx≈256 — structural choices only, not sacred.

### 3.2 Multi-select: keep the sub-step machinery, upgrade the conditioning

Decision (confirmed with the user): **no separate AR model, no custom joint-action PPO loss.** The env already realizes π(set) = Π π(pickₜ | state, picked₍<ₜ₎) · π(commit) as ordinary MDP steps — that IS an autoregressive model; PPO is unbiased on it, and `gamma=1.0` keeps every sub-step of a node on the node's return (already load-bearing; see the `train_sb3.py` docstring). What was missing is that the old net couldn't *use* the conditioning; the picked-pool above + the existing 4 state scalars + per-row `already_picked` bit now close that gap. A literally separate model would train on only ~11% of decisions, need its own critic, and add a two-model drift surface.

Known residual limitations, accepted and recorded (THEORY-REVIEW flags, carried from `plan-multiselect.md`): order-redundancy (a k-set has up to k! sub-step paths — wasteful, not biased); STOP's logit rides entirely on the context vector (no content features of its own — the new pooled inputs are exactly what improves it); genuinely identical options remain indistinguishable; multi-pick SKILL nodes stay excluded (order-semantic).

### 3.3 `TcgMaskablePolicy(MaskableActorCriticPolicy)` in new file `sb3_policy.py`

Verified-against-source override surface (~120 lines). MaskablePPO's *entire* policy touch surface is {constructor, `forward`, `evaluate_actions`, `predict_values`, `predict`/`_predict`, `obs_to_tensor`, `set_training_mode`, `optimizer`, `parameters()`, `to()`} — nothing reaches `mlp_extractor`/`action_net`/`value_net`, and `use_sde` is forced False.

- `__init__`: stash `core_kwargs` BEFORE `super().__init__()` (the base calls `_build` at its end). `self.action_dist` is created by the base `__init__` *before* `_build`, so the custom `_build` inherits it — no need to recreate. (The base also builds a parameter-free `FlattenExtractor`; unused and harmless — don't be surprised it exists.)
- `_build`: replace entirely — construct `self.core = TcgCore(**core_kwargs)` + the optimizer exactly as stock: `self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)`. Do NOT build `mlp_extractor`/`action_net`/`value_net` (keeps the state_dict and Adam clean).
- `forward(obs, deterministic=False, action_masks=None)` / `evaluate_actions` / `get_distribution(obs, action_masks=None)` / `predict_values`: replace, all through one `_run_core(obs)` → `(logits[B,128], values[B,1])`, then the STOCK distribution path (`self.action_dist.proba_distribution(action_logits=...)` + `apply_masking`). Shape contract (verified): masks are reshaped to the logits' shape, so logits `[B,128]` is mandatory; `values` `[B,1]` (train() flattens, `[B]` would also work); `log_prob [B]`; `entropy [B]`. Masked logits become −1e8 (`HUGE_NEG`), not −inf.
- `_get_constructor_parameters`: include `core_kwargs`.
- Inherit `predict`/`_predict`/`obs_to_tensor`/`set_training_mode` untouched. Because we keep `Discrete(128)` + the stock `MaskableCategorical`, every existing consumer works unchanged: `sb3_agent` (train+eval routing), `eval_sb3.py` (plain `MaskablePPO.load`), `record_battle.py`'s `get_distribution(...).distribution.probs` + `predict_values`, and `train_sb3`'s self-play snapshot round-trip. `MaskablePPO.load` reconstructs the policy from the pickled class path + `policy_kwargs` — so `sb3_policy.py`/`tcg_net.py` must never be renamed once checkpoints exist, and ALL architecture params must flow through `policy_kwargs`, never module-level defaults.
- Int obs keys: verified all three tensor paths (rollout buffer allocates each key with the space's dtype; `obs_as_tensor` and `obs_to_tensor` preserve dtype). SB3's float-casting `preprocess_obs` is only reached via `extract_features`, which our overrides never call — int64 ids reach the core intact. `TcgCore` still casts ids with `.long()` internally as a safety net.

**Cut (was in the first draft): no `--policy {tcg,mlp}` flag.** The new policy replaces `MultiInputPolicy` outright and the stock path is deleted from `train_sb3.py`. Reason: the stock policy would ingest the new int64 id keys through `preprocess_obs`'s float cast — garbage features — so a "same-obs A/B baseline" is false advertising; keeping it honest would need obs-filtering machinery, i.e. real complexity for a number we already have. The architecture baseline is the *measured* ~13% plateau (Phase M). `train_sb3.py` threads `core_kwargs` into `policy_kwargs` (new plumbing — none exists today).

## 4. Change C — imitation warm-start (BC) + critic warm-start, then PPO fine-tune

Why: the rule-based agents beat the learner in all 16 cells, so they are free teachers; supervised per-decision labels sidestep the diagnosed terminal-reward credit problem for initial learning; and the credit probe already showed a supervised value fit (0.167) beats the RL critic (0.089) on identical features. Classic AlphaGo-style recipe: clone → fine-tune.

**Named invariant — ONE OBS BUILDER, four consumers.** Env training, eval (`sb3_agent`), `record_battle`, and BC all featurize through the same `obs_to_gym(o, picked)` + `_ms_mask(sel, picked)` (verified: free functions in `sb3_env.py` with no env/singleton state — importable offline). `bc_train` never re-implements featurization; if BC needs something new, it goes into `obs_to_gym`. This is the anti-drift guarantee that makes offline IL trustworthy (see §7 risk 3).

### 4.1 `bc_collect.py` (new) — teacher games → decision dataset

- Pattern lifted from `credit_probe.py::_rulebased_samples` (the instrumented game loop + `_trim_obs` + by-game labels).
- Teacher-vs-teacher games, both decks drawn from the 4 archetypes (16 pairings, BOTH seats produce samples). **Call each teacher exactly once per decision — load-bearing, not style**: dragapult accumulates `obs.logs` into module globals and lucario keeps sticky per-turn flags, so a second call on the same node corrupts them (verified). The one call feeds both the recorded sample and `battle_select`.
- Record at every node passing the SAME routing as training (`is_learnable_select`, ≤ N_MAX−1): `{game_id, seat, trimmed_obs, teacher_picks (raw set), minCount, maxCount}`, plus the seat-relative terminal outcome per game (the critic label; draw = 0, same convention as the env's terminal reward — the label IS PPO's gamma=1.0 return-to-go). Store **raw trimmed obs, not featurized arrays** (immune to featurization drift; pick-order augmentation stays free).
- Cost/scale (measured anchors: ~0.17 s/teacher-game; ~37 decisions/game; credit_probe pickle = 146 MB / 29k rows ≈ 5 KB/row): 16 pairings × 250 games ≈ 4k games ≈ ~12 min → ~150k decisions ≈ ~750 MB pickle, several GB RAM loaded. On this box **start small (~2k games)** and read the learning curve (holdout accuracy at 1k/2k/4k games) before collecting more; gzip or shard per pairing if size gets awkward; `--smoke` reports bytes/row.
- `--smoke` self-test: replay every recorded sample through the sub-step decomposition and assert `_ms_mask(sel, picked)[action] == 1` at each step, STOP emitted iff `len(picks) < maxCount`, and the reconstructed list is engine-shape-valid.

### 4.2 `bc_train.py` (new) — trains the bare `TcgCore`, pure torch

- Split train/holdout **by game_id** (decisions within a game share one outcome label — the effective sample count is games, not decisions; this is the credit probe's own lesson).
- Each sample expands on the fly into sub-step examples: **shuffle `teacher_picks` into a random order per epoch** (teachers return score-ordered lists that are semantically unordered sets — verified; the engine validates set-wise, and order-semantic SKILL nodes never reach this pipeline). Emit `(obs_to_gym(o, picked_prefix), next_pick)` per prefix, plus the STOP example only when `len(picks) < maxCount` (mirrors the env's auto-commit exactly — never train a state the env can't visit).
- Featurization cost: single-select samples (~89%) are order-invariant → featurize once and reuse; only multi-select prefixes need per-epoch regeneration. Measure in `--smoke` before building any caching.
- Loss: masked cross-entropy (masked-out slots set to −1e8, mirroring `MaskableCategorical`'s HUGE_NEG — never −inf) + `λ·MSE(value, outcome)` (the critic warm-start).
- Report on holdout: top-1 masked accuracy vs the random-legal floor (mean 1/|legal options|), accuracy split single-select vs multi sub-steps, and value explained-variance reusing `credit_probe`'s `ev_true` metric (the 0.167 supervised bar is the reference). Save `{state_dict, core_kwargs, metrics}` → `out/bc_core.pt`.
- `--smoke`: overfit a tiny slice to ~100% train accuracy (loss-plumbing check).
- Flagged v2 (not built): exact multi-label set objective (log of summed prob over all remaining teacher picks); the shuffle augmentation approximates it.

### 4.3 `train_sb3.py --bc-init out/bc_core.pt`

- After model creation: `model.policy.core.load_state_dict(...)` — loading into the **named submodule** sidesteps all state_dict prefix pain (this is why BC trains the bare `TcgCore`). Assert saved `core_kwargs` == current. Mutually exclusive with `--init-from`.
- `--bc-init X --total-timesteps 0` saves an eval-able zip of the pure BC policy (`learn(0)` no-op + `model.save`) — that zip is the headline BC evaluation, and the anchor against fine-tune forgetting.
- Fine-tune with reduced lr + MaskablePPO's native `target_kl` (early-stops each update's epoch loop), frequent `--checkpoint-freq`, eval-best selection (eval-discipline skill).

## 5. Phases, dependencies, verification (each phase lands only when its check passes)

Two independent tracks that meet at BC training. **B1 depends on nothing from phases 0–2** (it stores raw obs, so no later code change can invalidate the data) — build and run it whenever convenient, even first. Phase M is a background measurement that gates nothing.

| Phase | Work | Depends on | Est. | Verification |
|---|---|---|---|---|
| **0** | Obs extension: `encode_zone_ids` + `H_MAX`/`D_MAX` (`sb3_features.py`); 6 id keys + space + `_terminal_obs` + overflow telemetry (`sb3_env.py`) | — | ~half day | `python sb3_features.py` (extended: zone shapes/dtype/PAD asserts), `python sb3_env.py`, `utils/run_smoketest_sb3_env.py` |
| **1** | `tcg_net.py` (`TcgCore`) | 0 | ~half day | `python tcg_net.py`: shapes/finiteness; grads reach BOTH embedding tables (the direct regression test against "ids discarded"); **permutation probe** — permute option rows+ids+mask together, assert logits permute identically (catches silent id/row misalignment structurally); **pad-leak probe** — perturb rows that are neither selectable nor picked (dense features AND ids), assert real-row logits AND value unchanged (value is the likeliest pool leak) |
| **2** | `sb3_policy.py` + `train_sb3` integration (`policy_kwargs`; stock-policy path deleted) | 1 | ~half day | `python sb3_policy.py`: shape/dtype contract incl. B=1; masked slots prob≈0; save→`MaskablePPO.load`→bit-identical logits on a fixed obs (exercises the pickle path); single-obs `model.predict` (the `sb3_agent` path). Then `train_sb3.py --smoke` (+ once with `--opponent rulebased,self --selfplay-refresh 256` to cover the snapshot round-trip); smoke zip through `eval_sb3.py --games 20` and `record_battle.py` (probs+value trace renders) |
| **M** | Background measurement: ONE ~150–200k-step PPO-only run with the new net | 2 | ~20 min train + eval | `eval_sb3.py` 4×4, sampled, n=200/cell; pooled mirror vs the measured ~13% plateau with Wilson CIs (overlapping CIs ⇒ no claim). Measures the **arch+obs effect jointly vs a historical baseline** — a shakeout + directional read, not a clean ablation. Non-gating: runs while B1/B2 proceed |
| **B1** | `bc_collect.py` | — (independent) | ~half day + minutes of compute | `--smoke` replay-consistency self-test (§4.1); then a real collection (~2k games first; scale on the learning-curve read) |
| **B2** | `bc_train.py` | 0, 1, B1 | ~half day + measured train time | `--smoke` overfit test; holdout top-1 ≫ random-legal floor (split single vs multi sub-steps); value ev vs the 0.167 bar |
| **B3** | `--bc-init`; t=0 zip; PPO fine-tune legs | 2, B2 | runs ~20 min each + evals | **BC→SB3 logit-identity check**: one fixed obs through bc_train's forward and through `policy.core` inside the SB3 wrapper → bit-identical logits/value (~5 lines; catches any consumer mismatch in one assert). **Gate G1**: pure-BC zip 4×4 (n=200) pooled mirror CI-separated ABOVE the ~13% plateau (expectation, not promise: teacher-mirror is 50% by construction; a decent clone should land far above 13%). Then fine-tune legs vs the BC anchor — first sign of collapse below the anchor is itself the diagnostic |

Phases 0–2 are pure plumbing; B1 is parallel-friendly; B2–B3 are the payoff. Everything before B3 is cheap to redo — nothing bets a long run on unverified code.

## 6. What gets simpler / removed

- No back-compat: pre-existing `out/*.zip` are dead after Phase 0 (documented break, fresh `--out-prefix`).
- No `--policy` flag: the stock `MultiInputPolicy` path is deleted from `train_sb3.py` (it can't honestly consume the new obs anyway — §3.3).
- The SB3-submission concern is **out of scope** (moved to §8): no code, flags, or design constraints for it now. `tcg_net.py` being torch-only is motivated by BC, not by packaging.
- `plan-multiselect.md`'s "Deferred levers" (§8) is absorbed here; the multi-select machinery itself (`_ms_mask`, STOP, sub-step loop, `is_learnable_select`) is reused verbatim — zero changes to `CGEnv.step`/`sb3_agent` control flow.

## 7. Risks (ranked) and mitigations

1. **Silent id/row misalignment** (ids from a different pass than dense rows; pad ids left 0 = a real card). Mitigations: ids and dense rows come from the SINGLE existing `encode_options` call; id arrays initialized to `CARD_PAD`/`ATTACK_PAD` before filling `[:n]`; Phase-1 permutation + pad-leak probes; B1 mask-replay assert.
2. **Custom policy vs SB3 internals** (log_prob/entropy/value shapes, save/load, masking). Mitigations: we replace only the four verified methods and keep the stock distribution object; the full MaskablePPO→policy touch surface was enumerated from source (§3.3) and everything else is inherited; Phase-2 contract test incl. save→load logit-identity; the MaskablePPO algorithm itself is untouched.
3. **BC featurization drift** (offline training features silently diverge from what the env/eval build — the classic silent IL killer). Mitigated by construction: the ONE-OBS-BUILDER invariant (§4), the B1 `--smoke` replay assert, and the B3 BC→SB3 logit-identity check.
4. **BC ceiling / teacher mismatch**: teachers consult the full obs; if holdout accuracy is low, the obs may still lack decisive inputs — the accuracy-by-node-shape breakdown localizes it. BC also caps near teacher level; PPO fine-tune is the corrector. Teacher-vs-teacher states differ from learner states (standard IL distribution shift; same corrector).
5. **Fine-tune failure, both directions.** *Forgetting*: PPO's terminal-reward credit problem is NOT fixed by this plan — fine-tune may drift below the BC anchor. Mitigations: t=0 anchor eval, `target_kl`, low lr, checkpoint+eval-best; BC-anchored KL regularization is the flagged v2 if needed. *Too timid*: a BC-peaked (low-entropy) policy may barely move — tiny approx_kl, flat eval curve. That outcome is acceptable (≥ anchor) and diagnosable from approx_kl/clip_fraction; don't misread it as a bug.
6. **Module-path pinning**: the zip pickles `sb3_policy.TcgMaskablePolicy` by path (an SB3 save-format fact, independent of any packaging concern) — never rename/move `sb3_policy.py`/`tcg_net.py` after checkpoints exist; all arch params via `policy_kwargs` (module-level defaults are a time bomb). To be noted in both docstrings.
7. **Hand truncation** (`H_MAX=20`): telemetry counter; raise if it fires (cheap break: weights survive, obs-space doesn't — §2).
8. **Theory flags for independent review** (repo convention): sub-step factorization soundness under gamma=1 (carried F2/F4/F5 from `plan-multiselect.md`), shuffle-augmentation as a set-objective approximation, and the expectation that per-card embeddings are learnable from ~10³–10⁴ games (low-data regime; the BC holdout accuracy measures this directly).

## 8. Deferred (deliberately out of scope) — "Deferred levers"

- **SB3-free inference mirror / Kaggle submission packaging** — premature; revisit when a submission is actually near. `tcg_net.py`'s torch-only design makes it easy then.
- Separate multi-select AR model / custom joint-action PPO loss (decided against — see §3.2).
- Deck-identity conditioning (needs deck plumbing through `sb3_agent`'s closure; cheap later, skipped now to keep the diff lean — with hand/board ids visible, deck identity is largely inferable anyway).
- Kaggle-replay IL (only ~47 replays today; `fetch_kaggle_replays.py` accretes daily; `bc_collect`'s sample schema matches what a replay parser would emit — `credit_probe._replay_samples` is the template).
- Per-step reward shaping (repeatedly gate-failed), obs running-normalization, NUMBER-node learning, multi-pick SKILL nodes, parallel envs (engine-source rebuild), hyperparameter sweeps.

---

## Files

- **Modify**: `sb3_features.py` (add `encode_zone_ids`, `H_MAX`/`D_MAX`), `sb3_env.py` (`obs_to_gym` id keys, `observation_space`, `_terminal_obs`, telemetry counter), `train_sb3.py` (`policy_kwargs` plumbing, `--bc-init`, delete the stock-policy path).
- **New**: `tcg_net.py` (plain-torch `TcgCore`), `sb3_policy.py` (`TcgMaskablePolicy`), `bc_collect.py`, `bc_train.py`.
- **Untouched**: `CGEnv.step` control flow, `_ms_mask`/STOP machinery, `eval_sb3.py`, `eval.py`, `win_mode.py`, the rule-based agents, `record_battle.py` (verify only — it keeps working because Discrete(128) + the stock distribution are kept).
- **Reuse**: `credit_probe.py` collection/`_trim_obs`/ev-metric patterns; `make_opponent`/eval factories; `plan-multiselect.md` machinery.
