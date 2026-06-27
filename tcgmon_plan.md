# tcgmon — Porting Metamon to the Pokémon TCG simulator

## Context — why this report exists

`starter_rl_kaggle.py` + `tcg_model.py` give us a small Transformer (encoder=value, decoder=policy) trained by **MCTS self-play**. It has been trained for little time but it's unlikely to go very far. We should just use the starter RL code as a reference on how to work with the Pokémon TCG SDK. We're more interested in Metamon, which is a research codebase that gets a *model-free* sequence policy to strong human-level Pokémon **Showdown** play by a different recipe: a large **imitation-learning warm-up on human replays**, then **off-policy RL fine-tuning**, on top of the **AMAGO** long-sequence RL engine (timestep-encoder → trajectory-encoder → actor-critic). The goal is to reproduce that recipe for **Pokémon TCG** by writing a new domain adapter — **`tcgmon`** — that lives in *our* repo, uses `metamon/` only as a reference, reuses **AMAGO unchanged as a library dependency**, and feeds the **cg** simulator instead of poke-env/Showdown.

This document is the **plan/report only** — no implementation yet. It explains Metamon's structure and execution flow, maps every component to a `tcgmon` analog (keep / cut / rewrite), isolates potentially the hardest design decision (the action space), and attempts to ranks the risks.

> Scope note: I read `metamon/metamon/` core logic + AMAGO `amago/amago/` + our cg side (`tcg_model.py`, `docs/cg-sdk.md`, `vendor/cg-lib/cg/api.py`, `eval.py`). I did **not** read `metamon/server`, `metamon/tools`, `metamon/examples` (out of scope per request). Claims below cite `file:line`; a few are flagged **[verify]** where they should be confirmed empirically before building.

---

## ⚠ Verification pass (2026-06-26) — corrections that SUPERSEDE parts below

This report's `file:line` claims were independently verified (AMAGO, Metamon, cg/`tcg_model`)
and two design decisions were re-made with the user. Where this banner conflicts with Parts
A–I, **the banner wins**. Full record: the approved scrutiny doc at
`~/.claude/plans/sparkling-napping-quokka.md`.

**Decisions changed:**
- **Action space → POINTER HEAD FIRST** (reverses Part D / Part H.1's "LOCKED slot-`Discrete`,
  pointer = fallback"). Verified rationale: AMAGO's discrete heads have no cross-slot weight
  sharing, and Metamon's `Linear(emb→13)` only works because its slots have *fixed* semantics,
  which TCG canonical slots lack. The pointer **actor** (reuse `MyModel`'s decoder via
  `pass_obs_keys_to_actor`) is fork-free and carries Phases 1–2; the pointer **critic** is the
  only hard part and is deferred to Phase 3 (AMAGO passes `straight_from_obs` to the actor only;
  its critic is state-only — like `MyModel`'s own scalar value).

**Errors corrected:**
- **`deck.csv` was NOT the Lucario deck.** It was a Mega Abomasnow ex / Kyogre **Water** deck
  (721=Kyogre, 722=Snover, 723=Mega Abomasnow ex, energy 3=Water) ≈ `sample_deck`, sharing only
  **1 of 9** cards with `rulebased_lucario`'s hard-coded list — so the rule-based teacher could
  not pilot it. **FIXED:** `deck.csv` restored to the real Mega Lucario ex deck (the 60-card list
  in `rulebased_lucario.py:42-58`; **17 distinct** cards, not "~9"); verified
  rulebased-vs-random = **83% / 25-5** on the mirror (old `deck.csv` backed up as
  `deck_abomasnow.bak.csv`). All "single Lucario deck" references below are now true.
- **"Decoder embedding shrinks ~99%" is FALSE** — `decoder_size=73,847`/`encoder_size=22,000`
  are global constants; fixing the deck only leaves rows *untrained*. Reuse the EmbeddingBags
  as-is; drop the shrink claim.
- **Auto-resolve heuristic mis-cited** — `tcg_model.py:554` is `random_agent` (random pick), not
  a deterministic "pick-`maxCount`". Write a **deterministic** resolver; **gate multi-select
  deferral on a Phase-0 ablation** (run the teacher with auto-resolve; it must stay ≈ even vs the
  full teacher).
- **RL fine-tune is OFFLINE, not online** — Metamon's `il/exp_rl/binary_rl.gin` all use
  `online_coeff=0`, `start_collecting_at_epoch=inf`. Phase 3 = offline advantage-weighted BC over
  accumulated self-play (also moots the engine singleton).
- **Worked-trace names wrong** (Part I): `721`=Kyogre, `722`=Snover, `3`=Water (not Riolu/Lucario/
  Fighting). Indices/arithmetic are correct; only the names were guessed.

**Confirmed solid:** all AMAGO/Metamon/cg `file:line` claims, every Part-I constant & decoder
offset, enum sizes, `fake_filter` pure-BC, deps (torch 2.8 / gymnasium 0.29.1, both import), and
the engine singleton being a non-issue (search-in-live-battle already runs). **Verification
upgrades:** elevate top-1 action-agreement vs the teacher to a quantified Phase-2 gate; add the
Phase-0 deferral ablation.

---

## ⚠ Rulebook-consistency pass (2026-06-26) — corrections that SUPERSEDE parts below

This pass cross-checked `rulebook.md` (official rules, Sept 2025) + the verified cg contract
(`api.py` enums, `game.py` driver, `tcg_model.py` feature math, the restored Lucario `deck.csv`)
against this plan. The **architecture is unchanged**; these correct rule/contract-derived
assumptions, numbers, and scope. Where this conflicts with Parts A–I, **this banner wins.**
Full record: `~/.claude/plans/zazzy-popping-sunbeam.md`.

**Errors corrected (rule/contract-derived):**
- **R1 — Part I's worked trace is the WRONG DECK.** It is the legacy Mega Abomasnow ex / Kyogre
  replay (`episode-81475913`), not the committed **Lucario** deck, so its "Phase 0–2 acceptance
  numbers" are all wrong-deck. Corrected static IDs: **Riolu=677** (PLAY idx `1571+677=2248`, not
  2292=Kyogre 721); Fighting **energy card=6** (`2839+6=2845`, not 2842 — **`3` is WATER**;
  `EnergyType.WATER=3`/`FIGHTING=6`, api.py:25–37); attach **target 677/678** (`4784/4785`, not
  4829=Snover 722); encoder word17 card **677/678**, energy **6**. `END→1` is deck-independent.
  Decoder ATTACH/EVOLVE "S/T" are **card IDs**, not EnergyType enums (basic-energy card-id equals
  EnergyType by design, so 6 either way here). Regenerate the real Lucario trace in Phase 0.
- **R2 — Prizes-per-KO are 1/2/3, not 1.** Mega Lucario ex is `megaEx` ⇒ opponent takes **3**
  (rulebook Appendix 1 / glossary; `CardData.ex=2`/`megaEx=3`, api.py:476). The deferred shaped
  reward (Part E) must use prize-deltas **xor** a per-KO term, never both (they diverge up to 3×).
  Prize counts ARE in the obs (`len(ps.prize)`, tcg_model.py:200) but are **lumpy** (−3 on a Mega KO).
- **R3 — `result==2` is a real DRAW** (api.py:376; reason codes 1=prizes / 2=deck-out / 3=no-active /
  4=card-effect, api.py:319). Metamon's `BinaryReward` has no draw branch ⇒ map **draw → 0,
  done=True**. `[verify]` the engine actually emits 2 / how Sudden Death resolves.
- **R4 — "next select goes to the same player" (Part I) is FALSE.** KO-replacement
  (`SelectType.CARD`+`TO_ACTIVE`) is routed by `current.yourIndex` and **fires on the opponent's
  turn**. The recorder must attribute every select to `yourIndex` (as `eval.py:77`) — a turn is NOT
  a contiguous same-player run of selects (Checkup/KO selects interleave).
- **R5 — The pre-MAIN setup phase is policy-relevant and was omitted.** The engine surfaces
  `IS_FIRST=41` (go first/second — strategic), `SETUP_ACTIVE_POKEMON=1` (a single-select the policy
  learns when ≥2 basics), `SETUP_BENCH_POKEMON=2` (multi → auto-resolved). Phase-0 histograms and
  `N_max` must include these; "pure single-select over MAIN" understates scope.

**Notes / `[verify]` (Tier 3):**
- **R6** — auto-resolve via `random_agent` (`tcg_model.py:554`) randomizes genuine multi-choices;
  rulebook "up to X / any amount / you may" = `minCount==0` selects where a random/always-take
  resolution can destroy resources. Phase-0 ablation must cover `SETUP_BENCH` + multi-discard.
- **R7 `[verify]`** — engine treats Mega Lucario ex as **new-style, no turn-end on evolution**
  (Appendix 1, NOT Appendix 26's old turn-ending `-EX`); else the recorded select sequence truncates
  after a Mega evolve.
- **R8** — heavy coin-flip / hidden-info stochasticity ⇒ noisy terminal-only value target; budget
  wider eval CIs. **R9** — deck-out + 3-prize Mega KOs ⇒ short games; read Phase-0 selects-per-game
  with that in mind.

**Confirmed consistent:** deck legality (60 cards, ≤4 copies, basics present, 13 F energy);
Riolu→Mega Lucario ex direct evolution (no intermediate Lucario, Appendix 1); all three win
conditions present as engine reason codes; per-turn limits / evolution timing / special conditions /
Weakness-Resistance all engine-enforced via legal-option enumeration (validates "enumerate +
masking ON"); prize counts visible in the observation.

---

## First-cut scope — simplify & de-risk (2026-06-26 refinement)

The locked architecture below (Parts D, H) is **unchanged**. This section records a refinement pass that **shrinks the first cut** to the smallest end-to-end *learnable* system that yields a verifiable `eval.py` signal, deferring everything non-essential to later, independently-graded phases — per the README's "minimal, incremental, verifiable" ethos and the open question of whether the Metamon recipe pays off given starter RL is currently *regressing* ([[eval-training-not-helping-yet]]). These are **phasing choices, not architecture changes.**

**Confirmed with the user:** (1) **single deck = the Lucario `deck.csv`** for both self-play seats and all eval; (2) **multi-select deferred** — the first learnable policy is pure single-select.

**First-cut simplifications (detail folded into the Parts noted):**
1. **One fixed deck (Lucario), mirror self-play** — bounds the card/attack vocabulary to ~9 distinct cards (decoder embedding shrinks ~99%, `tcg_model.py:67`), removes deck-building, and makes *model-vs-rulebased a clean mirror* (kills the eval deck-confound). (Parts B/C/E/G)
2. **Defer multi-select + auto-resolve trivial selects in the env** — `maxCount>1`, must-take-all, and 1-legal-option selects are resolved by the existing pick-`maxCount` heuristic (`tcg_model.py:554`) in both data-gen *and* inference; the policy only ever sees single-select `Discrete(N_max)`. Removes the hardest new code (risk #4) and shrinks `N_max`/sequence length (risk #3). (Part D.4)
3. **Masking ON** (the `MetamonMaskedActor` default) — for TCG's mostly-illegal `Discrete(N_max)`, masking is both simpler (one fewer override) and safer than Metamon's no-mask default; the mask is free since we already enumerate legal options. (Part D.5)
4. **Binary win/loss reward only** (`BinaryReward`/`binary_rl.gin`, config-only) — defers shaped-reward design (risk #7), itself a suspected starter-RL regression cause. (Part E)
5. **Cheapest teacher first: distill `rulebased_lucario` before MCTS** — fastest verifiable "pipeline learns" milestone; the recorder is teacher-agnostic. (Parts A.4/G)
6. **Single-process, sequential, single frozen opponent** — sidesteps the engine-singleton risk (#5); no async/league. (Part E)

**Deferred out of the first cut** (each a later, eval-graded phase): multi-deck / opponent-deck generalization · learned multi-select (the locked decomposition) · shaped reward · human Kaggle-replay parser · async/parallel collection & co-evolving league · token-vocabulary `TransformerTurnEmbedding` encoder.

---

## Part A — How Metamon works (structure & execution flow)

### A.0 The three-layer stack (this is the whole mental model)

```
            POKÉMON SHOWDOWN                         POKÉMON TCG (our target)
        ┌────────────────────────┐              ┌────────────────────────────┐
LAYER 1 │ poke-env / Showdown    │   ──►  port  │ cg simulator (libcg.so)     │
(domain │ Battle objects, replays │              │ battle_start/select/finish, │
 engine)└────────────────────────┘              │ Kaggle episode replays      │
                  │                                         │
                  ▼                                         ▼
        ┌────────────────────────┐              ┌────────────────────────────┐
LAYER 2 │ metamon/metamon/  ◄─────┼── REWRITE ──►│ tcgmon/ (new, our repo)     │
(domain │ interface, tokenizer,  │  (this is     │ same structure, cg-native   │
 adapter│ il/model, data, env,   │   the work)   │ pieces                      │
 +glue) │ rl/metamon_to_amago    │              └────────────────────────────┘
        └────────────────────────┘                         │
                  │  (subclass + gin, no fork)              │ (identical pattern)
                  ▼                                         ▼
        ┌─────────────────────────────────────────────────────────────────────┐
LAYER 3 │ AMAGO (amago==3.4.0)  — REUSED UNCHANGED as a pip/library dependency  │
(engine)│ TstepEncoder → TrajEncoder → Actor-Critic, offline+online, data fmt   │
        └─────────────────────────────────────────────────────────────────────┘
```

The single most important architectural fact for "minimal changes": **Metamon does not modify AMAGO.** It only *subclasses* AMAGO classes and configures them through gin. All Metamon↔AMAGO glue is one file, `metamon/metamon/rl/metamon_to_amago.py` (1406 lines). `tcgmon` should mirror exactly this: AMAGO untouched, one `tcgmon_to_amago.py` glue file.

### A.1 AMAGO engine (Layer 3) — what we reuse verbatim

- **TstepEncoder** (`amago/amago/nets/tstep_encoders.py`): abstract base; you implement `inner_forward(obs: dict[str,Tensor], rl2s) -> (B,L,emb_dim)` and the `emb_dim` property. `obs` is a **dict of tensors** shaped `(B, L, *feat)`; `rl2s` is the previous action+reward vector `(B,L,1+action_dim)`. Ships `FFTstepEncoder` (MLP) and `CNNTstepEncoder`. **This is the only network we must write.**
- **TrajEncoder** (`amago/amago/nets/traj_encoders.py`): `TformerTrajEncoder` (causal decoder-only Transformer, fixed/learned/RoPE pos-emb, KV-cache for inference), plus `GRU`/`Mamba`/`FF`/`Identity`. **Reused as-is** (this is the "trajectory encoder" the user wants).
- **Agent / actor-critic** (`amago/amago/agent.py:489`): ties encoders → actor → critic ensemble. Off-policy actor-critic with **filtered behavior cloning**: actor loss = `online_coeff·(−Q(s,a∼π)) + offline_coeff·f(A(s,a))·log π(a|s)`; critic = clipped multi-critic TD with **multi-gamma** heads, PopArt, target nets. Key knobs: `offline_coeff`, `online_coeff`, `fbc_filter_func`, `gamma`, `reward_multiplier`. **Reused as-is.**
- **Action spaces AMAGO supports**: `Discrete(n)`, `MultiBinary(n)`, `Box`. Discrete policy = `Categorical` (`policy_dists.py:241`); discrete critic = a **Q-vector of size `n` from the state embedding alone** (`actor_critic.py:344` `NCritics`, discrete branch ~418/428) — DQN-style, **no per-action feature input**. ← pivotal for TCG (Part D).
- **Action masking**: AMAGO has **no built-in masking**, but exposes clean hooks:
  - `pass_obs_keys_to_actor` (`agent.py:626/661`) → those obs keys are forwarded as `straight_from_obs` into `actor.actor_network_forward(state, log_dict, straight_from_obs)` (`actor_critic.py:80/141`). Override that method to `masked_fill_(-inf)` illegal logits.
  - `edit_actor_mask` / `edit_critic_mask` (`agent.py:440/456`, applied at `:871-872`) default to identity; override to drop timesteps from the loss.
- **Env contract** (`amago/amago/envs/amago_env.py:30`): `AMAGOEnv` wraps a gymnasium env; obs forced to a **Dict** space; discrete actions one-hot'd as the "prev action" rl2 feature (`make_action_rep:90`). `SequenceWrapper:195` records trajectories to disk (`.traj`/`.npz`) and builds `rl2_space`.
- **Data** (`amago/amago/loading.py:25` `RLData`, `hindsight.py` `FrozenTraj`): a trajectory is obs-dict + `rews`/`dones`/`actions` (length T−1) + `time_idxs` (length T); `MAGIC_PAD_VAL=4.0` marks padding. Offline datasets subclass `RLDataset.sample_random_trajectory() -> RLData`. **Offline→online schedule** via `Experiment.start_learning_at_epoch` / `start_collecting_at_epoch`.
- **Experiment** (`amago/amago/experiment.py:43`): gin-configurable dataclass; `make_train_env`/`make_val_env` callables, `dataset`, encoder/traj/agent *types*, `max_seq_len`, etc. `.start()` then `.learn()`.

### A.2 Metamon domain layer (Layer 2) — the thing we rewrite

| File | Role | Key symbols |
|---|---|---|
| `interface.py` (60KB) | The domain abstraction. **Engine-agnostic** `UniversalState` (`:482`) built from poke-env `Battle`/replays; `ObservationSpace` (`state_to_obs → {"text": str, "numbers": float[48]}`, `DefaultObservationSpace:1010`); `TokenizedObservationSpace:1599` (text→`text_tokens` int ids); `ActionSpace` (`DefaultActionSpace = Discrete(13):815`, 4 moves+5 switches+4 tera, alphabetical order); `UniversalAction:648` (idx↔BattleOrder, `maybe_valid_actions:699` = legal set); `RewardFunction` (`DefaultShapedReward:870` = HP/status/KO deltas + ±100 win) | versioned, registry-based |
| `tokenizer/tokenizer.py` | `PokemonTokenizer:18`, word→int id; premade vocab JSONs (`:84`); ~3.5–4.5k tokens (species/moves/items/abilities + special tokens) | fixed token-list files |
| `il/model.py` | Shared NN building blocks: `TokenEmbedding:45`, `MultiModalEmbedding:67` (numbers→tokens), `TransformerTurnEmbedding:315` (token+numerical → per-turn embedding, `output_dim = scratch_tokens·d_model`), `GRUModel:465` (standalone IL). **These are shared by both IL and the RL TstepEncoder.** | the "tokenization" pipeline |
| `data/parsed_replay_dset.py` | `ParsedReplayDataset:532` over parsed human replays; `load_filename:466` → `(obs, action_infos{chosen,legal,missing}, rewards, dones)`. HF auto-download; 5.3M trajectories. | offline IL corpus |
| `env/wrappers.py` | `PokeEnvWrapper:170` (poke-env `OpenAIGymEnv`: `embed_battle`→obs, `action_to_move`, `calc_reward`, exposes `info["legal_actions"]`); `BattleAgainstBaseline`, `QueueOnLocalLadder`, `ChallengeByUsername` | **deeply poke-env-coupled → fully rewritten** |
| `rl/metamon_to_amago.py` | **The glue.** `MetamonTstepEncoder:566`, `MetamonDiscrete:337`, `MetamonMaskedActor:390`, `MetamonAMAGOWrapper:270`, `MetamonAMAGODataset:1259`, `MetamonAMAGOExperiment:1346`, `make_placeholder_env:132` | the file we clone for TCG |
| `rl/{train,finetune,pretrained}.py`, `rl/configs/{models,training,datasets}` | Entry points + gin presets (`il.gin`, `exp_rl.gin`, `binary_rl.gin`, `finetune.gin`; model sizes `small/medium/large_agent.gin`) | reused pattern, new presets |
| `config.py`, `__init__.py` | format registry; **hard pin `poke-env==0.8.3.3`** | drop poke-env pin |

### A.3 The glue, in detail (the template for `tcgmon_to_amago.py`)

- **TstepEncoder** (`MetamonTstepEncoder:566`): `inner_forward(obs, rl2s)` folds `rl2s` via `symlog`+linear into "extras", concatenates with `obs["numbers"]`, and runs `obs["text_tokens"]` + numbers through the **same `TransformerTurnEmbedding` used in IL** (`:618`). That shared module is the bridge that makes IL pre-training transfer into RL.
- **Masked actor** (`MetamonMaskedActor:390`): subclasses `Actor`, calls `super().actor_network_forward()` then `masked_fill_(-inf)` on `straight_from_obs["illegal_actions"]`, with a *no-legal-options* guard (`:439-448`). **Crucial nuance** (`:394-397`): the shipped models trained with `mask_illegal_actions=False` — they *don't* hard-mask; an illegal pick just triggers a random action in the env and "critic nets have no problem learning" to avoid it. So masking is optional.
- **Env wrapper** (`MetamonAMAGOWrapper:270`): subclasses `AMAGOEnv`; moves `info["legal_actions"]` into `obs["illegal_actions"]` (bool `[n]`) each reset/step.
- **Dataset adapter** (`MetamonAMAGODataset:1259`, `_process_data:1299`): replay trajectory → `RLData`; one-hots chosen actions; builds per-timestep `illegal_actions` `[T+1,n]`; **stashes `illegal_actions` and `missing_action_mask` inside the obs dict** ("a bit of a hack… let the network ignore it… mask the loss later", `:1324`).
- **Experiment** (`MetamonAMAGOExperiment:1346`): `init_model` monkey-patches `policy.edit_actor_mask/edit_critic_mask` to AND-in `~missing_action_mask` (`:1369-1393`); `train_step` injects the valid-state×valid-action mask into the FBC filter (`:1395-1406`).
- **Inference/loading** (`make_placeholder_env:132`, `make_placeholder_experiment:211`): a no-op env to instantiate the model and load a checkpoint for pure inference — **the template for our Kaggle `agent()`**.

### A.4 Execution flow (the recipe we want)

1. **Parse** human replays → on-disk parsed trajectories.
2. **IL warm-up**: `rl/train.py` with a model gin + **`il.gin`** = AMAGO Agent with high `offline_coeff` / `online_coeff≈0` ⇒ pure **filtered/behavior cloning** over the replay `RLDataset`, `start_collecting_at_epoch=∞` (offline only). Produces a checkpoint.
3. **RL fine-tune**: `rl/finetune.py` from that checkpoint with **`exp_rl.gin`** = exponential-advantage-filtered off-policy RL, now collecting online (self-play / vs baselines).
4. **Eval/inference**: load checkpoint into a placeholder experiment; run the policy (with TrajEncoder hidden state across the episode).

**Key accuracy point:** IL and RL are the **same AMAGO model**; "warm-up vs RL" is a **training-config (gin) difference**, *not* two architectures. The standalone `il/` GRU trainer exists but is secondary; the modern warm-up is the offline AMAGO objective.

---

## Part B — The TCG domain: what's genuinely different (the cg contract)

From `docs/cg-sdk.md` and `vendor/cg-lib/cg/api.py`:

- **Agent contract**: `agent(obs_dict) -> list[int]`. First call (`obs.select is None`) → return a **60-card deck**. Else return **option indices** into `obs.select.option`, length in `[select.minCount, select.maxCount]`, **no duplicates** (`docs/cg-sdk.md:21`).
- **Observation** (`api.py:438`): `Observation{select, logs, current, search_begin_input}`. `current` is full `State:366` (`turn`, `yourIndex`, `result`, two `PlayerState:350` with `active/bench/hand/discard/prize/deckCount/handCount` + status bools, `stadium`). Cards are `Card{id,serial,playerIndex}`; `Pokemon{id,hp,maxHp,energies,energyCards,tools,...}`. Static card metadata via `all_card_data()` (`tcg_model.py:54`).
- **Action = a `SelectData:398`**: `{type∈SelectType(11), context∈SelectContext(49), minCount, maxCount, option: list[Option(17 types)]}`. The option list is **variable-length and content-dependent** every decision (play/attach/evolve/ability/retreat/attack/yes-no/number/pick-card/…). **This is the crux.**
- **Reward / terminal**: `State.result` — `result==yourIndex` win, `==2` draw, `<0` ongoing (`docs/cg-sdk.md:48`, `eval.py:98`).
- **Driver**: `battle_start(deck0,deck1)` → loop `battle_select(idxs)` while `result<0` → `battle_finish()` (`vendor/cg-lib/cg/game.py:19/48/43`). Two agents routed by `current.yourIndex` (`eval.py:77`).
- **Hidden info / search**: determinizing `search_begin/step/end` API (`api.py:517`); `tcg_model.mcts_agent:460` fills opponent slots with placeholders.
- **Replays**: `data/replays/episode-81475913-replay.json` — Kaggle episode JSON whose **observation is already in cg-dict form** (per `work_log.txt`), but **no replay loader exists** → custom parser needed. We currently have **~1 sample game locally**. [I think there's probably very little replay data available from Kaggle at the moment, in which case we shouldn't spend time on this for now]
- **Engine is a global singleton** **[verify]**: `game.py` tracks one `Battle.battle_ptr` class-global, so one process holds one live battle at a time — matters for parallel collection (Part F) [we don't even have anything working so parallellizing things is not a priority].

How `tcg_model.py` already copes (reuse this know-how): `get_encoder_input:211` builds **24 sparse "words"** (per-player bench×8, active, player-state, hand, deck, stadium, turn) for an `EmbeddingBag`; `get_decoder_input:294` featurizes each **candidate action** by the *content* of its option(s); `MyModel:91` scores up to **64 enumerated candidates** (`create_node:425`) via a cross-attention **pointer decoder**. This is exactly the content-dependent-action machinery Metamon never needed.

---

## Part C — Component map: keep / cut / rewrite

| Metamon component | tcgmon analog | Disposition | Effort/Risk |
|---|---|---|---|
| AMAGO (Layer 3) | same | **Keep, unchanged dependency** | Low |
| `TformerTrajEncoder`, `Agent`, losses, gin | same | **Keep** | Low |
| `interface.UniversalState` (poke-env) | `tcgmon` state view over cg `Observation`/`State` | **Rewrite** (reuse `tcg_model` field extraction) | Med |
| `ObservationSpace` → `{text,numbers}` | `ObservationSpace` → `{card/enum tokens or sparse ids, numbers}` | **Rewrite** (reuse `get_encoder_input`) | Med |
| `ActionSpace = Discrete(13)` fixed-semantic | `Discrete(N_max)` **positional**, canonical option order | **Rewrite — the hard part** (Part D) | **High** |
| `RewardFunction` (HP/status/KO + win) | prize-deltas/damage/KO + win, over *select* steps | **Rewrite** (have all fields) | Low-Med |
| `PokemonTokenizer` + vocab JSONs | card-id/enum vocabulary from SDK metadata | **Rewrite** (or skip; use EmbeddingBag ids) | Low |
| `il/model.TransformerTurnEmbedding` | TCG turn/state embedding (token or sparse) | **Rewrite or adapt** | Med |
| `data/ParsedReplayDataset` + replay parser | Kaggle-episode-JSON → trajectory parser + dataset | **Rewrite** (no loader exists) | Med + **data-availability risk** |
| `env/PokeEnvWrapper` (poke-env OpenAIGymEnv) | cg gym env over `battle_start/select/finish` + self-play | **Rewrite from scratch** | Med-High |
| `rl/metamon_to_amago.py` glue | `tcgmon/rl/tcgmon_to_amago.py` (same 6 classes) | **Clone & adapt** | Med |
| `rl/{train,finetune,pretrained}.py`, gin configs | same names, TCG presets | **Clone & adapt** | Low-Med |
| `config.py`/poke-env pin, `server/`, `tools/` | — | **Cut** | — |
| Standalone `il/` GRU trainer | — | **Cut** (use AMAGO offline objective for warm-up) | — |
| (new, no Metamon analog) | **Multi-select decomposition** in env wrapper | **New** | **High** |
| (new) | **Kaggle `agent()` inference shim** with persistent hidden state + deck-on-first-call | **New** | Med |

---

## Part D — The central design decision: the action space

**Why it's hard.** Metamon's `Discrete(13)` has (near-)**fixed semantics** — slot 0 is "alphabetically-first move," and the observation lays out moves/switches in the *same* order, so a plain MLP head over the pooled state works. TCG has **no fixed action vocabulary**: option index *i* is "whatever the *i*-th legal option is *this* select," it's heterogeneous, variable in count, and can be **multi-select** (`minCount..maxCount`). AMAGO's discrete critic produces its Q-vector **from the state embedding only** — it never sees per-action features. So either (a) the per-option content must reach the heads *through the state embedding in a slot-aligned way*, or (b) we replace the heads with a pointer network.

**⚠ SUPERSEDED (see top banner) — the user chose the POINTER HEAD first; the slot-`Discrete` design below is no longer the primary path (it is now the fallback). The rest of this Part is kept for reference.**

**LOCKED — Metamon-faithful, minimal AMAGO risk: `Discrete(N_max)` over canonically-ordered, content-bearing option slots + masking + multi-select decomposition.**

1. **Fix `N_max`** = max single-select option count to support (pick from instrumentation, Part F; pad/mask the rest). One AMAGO action = **pick one option slot**.
2. **Canonical ordering**: deterministically sort `obs.select.option` (e.g. by `(type, area, index, cardId, attackId)`) so slot *i* is reproducible at train (replay) and inference time. This is the analog of Metamon's alphabetical move/switch order.
3. **Slot-aligned content features in the observation**: for each of the `N_max` slots, encode that option's *content* (its `OptionType` + referenced card/area/attack), reusing `tcg_model.get_decoder_input`'s per-option featurization (`tcg_model.py:294`). The TstepEncoder ingests these so the pooled state carries per-slot content → standard AMAGO actor/critic can produce per-slot logits/Q-values (exactly how Metamon's encoder feeds its 13 slots).
4. **Multi-select via env-side decomposition** (no Metamon analog): when `maxCount>1`, the wrapper presents the *same* select as **several AMAGO timesteps**, each picking one more option (mask already-chosen), with an explicit **STOP** slot enabled once `≥minCount`; the wrapper accumulates and issues a single `battle_select(full_list)`. Keeps the policy a clean single-select `Discrete`. (YES/NO and COUNT selects map to small fixed slot sets.) **First cut: DEFERRED via a precise predicate** — the policy makes a decision **iff `minCount==1 and maxCount==1 and len(option)>1`** ("pick exactly one of ≥2"); every other select (`maxCount>1` multi, `minCount==0` optional, or `len==1` forced) is auto-resolved by the existing pick-`maxCount` heuristic (`tcg_model.py:554`), *identically in data-gen and inference*, so the first learnable policy is pure single-select. **Caveat (from the worked trace, Part I):** auto-resolving `minCount==0` selects means the heuristic always *takes* the optional action (never declines) — legal but suboptimal; the principled fix is a synthetic **STOP/none** slot (the same machinery as this multi-select STOP), added in the same later phase. Graded by whether learning these selects beats the heuristic in `eval.py`.
5. **Masking — ON** (revised 2026-06-26; was "start with `mask_illegal_actions=False`"): keep `MetamonMaskedActor`'s **default `mask_illegal_actions=True`**. Metamon's no-mask choice fits its near-fixed-semantic `Discrete(13)` (mostly legal every turn); TCG's `Discrete(N_max)` is the opposite — **most slots are illegal on any given select**, so no-masking would have the early policy pick illegal slots most of the time → random env substitution → badly mis-attributed signal. Masking ON is both *simpler* (one fewer gin override) and *safer*, and the mask is **free** because we already enumerate the legal option set every decision to map canonical slots ↔ cg indices. Reuses the `illegal_actions`-in-obs + `edit_actor/critic_mask` machinery verbatim. Reversible via one flag if it fights training.

**Fallback only (tcg_model-faithful, higher AMAGO risk): pointer-head actor + critic** that score the up-to-64 enumerated candidates by content (port `MyModel`'s cross-attention decoder into a custom `Actor` *and* a custom `Critic`). Naturally handles content-dependence and multi-select-as-one-action, and reuses our proven head — **but** it requires a custom *critic* that takes per-candidate features (AMAGO's discrete critic doesn't), touching the most sensitive part of the loss (multi-gamma, PopArt, ensembles, FBC). Hold in reserve: switch to this only if the slot approach demonstrably won't learn (e.g. `N_max` must be huge, or slot-alignment fails to train).

Both approaches keep AMAGO unforked; they differ only in the actor/critic head and how multi-select is expressed.

---

## Part E — Change-list by aspect (data → obs → tokenize → encode → act → train → infer → reward → env)

- **Data** (self-play first): build a **self-play trajectory recorder** — record per-(single-)select `(obs_dict, chosen canonical-slot, legal_set, reward, done)` → `RLData` via a `RLDataset.sample_random_trajectory` that clones `MetamonAMAGODataset._process_data` (one-hot actions, `illegal_actions`/`missing_action_mask` stashed in obs). **First-cut teacher order: `rulebased_lucario` first** (fast, no search — cheapest path to a verifiable "pipeline learns" signal), **then `tcg_model.mcts_agent`** (stronger, slower — our strength source). The recorder is teacher-agnostic and is the trajectory→`RLData` core of the future Kaggle-replay parser — **deferred**, only the *source* (episode JSON vs our self-play) differs.
- **Input/observation**: define `tcgmon` `ObservationSpace.state_to_obs(cg Observation) -> {sparse/token state feats, numbers, candidate_slot_feats[N_max], illegal_actions[N_max], missing_action_mask}`. Reuse `get_encoder_input` for state; `get_decoder_input` (per single option) for slot feats. **Fixed-shape adaptation (from the worked trace, Part I — the main new encoder glue, not a verbatim reuse):** these builders return *variable-length* `SparseVector`s, but AMAGO obs must be fixed `(B,L,feat)` tensors — so `state_to_obs` pads to `enc_idx/enc_val (24,K_enc)` and `dec_idx/dec_val (N_max,K_dec)` (sizes from Phase 0; raise on overflow, never silently truncate).
- **Tokenization**: TCG "tokens" = card IDs + enum values (`SelectContext`, `OptionType`, `EnergyType`, `AreaType`). Either build a small vocab file (Metamon-style) or skip the tokenizer and use `EmbeddingBag` indices as `tcg_model` already does. Lean toward the latter for first cut.
- **Timestep encoder**: write `TcgmonTstepEncoder(amago…TstepEncoder)`; reuse `tcg_model`'s **feature builders + the two `EmbeddingBag`s** (NOT `MyModel`'s cross-attention pointer decoder — that's the Part-D *fallback*) to pool the padded obs into a `state_emb` (24 words) and a per-slot `slot_emb[N_max, d_slot]`, output as one `(B,L,emb_dim)` vector `≈ concat(state_emb, flatten(slot_emb))` so AMAGO's standard `Linear(emb_dim→N_max)` actor/critic heads are **slot-aligned**. Mirror `MetamonTstepEncoder` for the `rl2s`/`symlog` plumbing and `emb_dim`. *(Whether the heads actually learn the slot-block→logit mapping is risk #1 — the Phase-2 go/no-go.)*
- **Action/output**: `Discrete(N_max)` (single-select only — Part D.4) + `TcgmonMaskedActor` (clone `MetamonMaskedActor`, **masking ON** — Part D.5). Slot↔option de-canonicalization happens in the env/inference shim.
- **Training**: clone `train.py`/`finetune.py`; author `tcgmon` `il.gin` (offline FBC warm-up) and `exp_rl.gin` (RL); reward/obs/action selected by CLI like Metamon. Clone `TcgmonAMAGOExperiment` mask hooks.
- **Inference (Kaggle)**: `agent(obs_dict)` shim built on `make_placeholder_experiment`: on first call return deck (from `deck.csv`); else encode obs, run AMAGO `get_actions` **maintaining TrajEncoder hidden state across the whole game**, de-canonicalize the chosen slot(s) (driving the multi-select loop internally) → `list[int]`.
- **Reward** (first cut: **binary only**): use `BinaryReward` (+100/-100) via the existing `binary_rl.gin` — **config-only, zero code** — but **map `result==2` (draw) → 0 reward, `done=True`** (Metamon's BinaryReward has no draw branch; R3). Defers the shaped `RewardFunction` design (risk #7; over-shaping skews play, and a value/target sign-or-scaling error is a suspected starter-RL regression cause). Add shaping only if binary plateaus — and use prize-deltas (Δ of `len(your prize)`) **xor** a per-KO term, **never both**, since `megaEx`=3 prizes ≠ 1/KO (R2); plus damage and ±terminal from `State.result`.
- **Env**: `TcgmonEnv(gym.Env)` driving `battle_start/select/finish` **on the Lucario `deck.csv` (both seats)**, exposing dict obs + `info["legal_actions"]`, and **auto-resolving multi-select/forced selects internally** (Part D.4) so only single-select decisions surface. Wrap with `TcgmonAMAGOWrapper(AMAGOEnv)`. **First cut: single-process, sequential collection** (AMAGO sync, `parallel_actors=1`) — sidesteps the engine-singleton risk (#5); RL trains vs **one frozen opponent** (warm-up checkpoint or `rulebased_lucario`), not a co-evolving league.

---

## Part F — Risks & open questions (ranked)

1. **Action-space modeling (highest)** — content-dependent, variable, multi-select options vs AMAGO's fixed Discrete + state-only critic. Mitigation in Part D; fallback = pointer head. *De-risk first* with instrumentation + a tiny end-to-end smoke train.
2. **Warm-up teacher quality** (was "data availability") — *Resolved by decision 3:* we warm up by **distilling self-play (MCTS preferred, rule-based fallback)** instead of human replays, so there is no data blocker. The residual risk is *teacher strength*: MCTS is our strongest agent, so distillation should help — but if MCTS itself is weak on some decks, the warm-up inherits that ceiling. Human replays remain a later enhancement (parser deferred, recorder reused). Mitigation: prefer MCTS-sourced trajectories; measure the distilled policy vs the MCTS teacher in `eval.py`.
3. **Episode length / sequence budget** — a TCG game is **many selects per turn** (≫ Showdown's one decision/turn), so AMAGO sequences are long. Must instrument typical selects-per-game and set `max_seq_len`/context accordingly. **[verify]** *First-cut mitigation:* deferring multi-select + auto-resolving forced selects (Part D.4) removes those timesteps from the policy's sequence, shrinking it.
4. **Multi-select decomposition correctness** — new logic with no Metamon analog; bugs here corrupt both data parsing and live play. **Removed from the first cut** (Part D.4 — env auto-resolves multi-select with the existing heuristic); when re-introduced as its own phase it needs unit tests against the cg engine.
5. **cg engine global-singleton / process isolation [verify]** — one `Battle.battle_ptr` per process ⇒ parallel collection would need process isolation (AMAGO `env_mode="async"`); MCTS search shares the same global. **First-cut mitigation:** run single-process & sequential (Part E) so the singleton is a non-issue; verify `vendor/cg-lib/cg/game.py` reentrancy only when scaling collection later.
6. **Dropping MCTS** — *Decided:* the trained policy is **model-free** (no search at inference). MCTS does **not** disappear from the project — it becomes the **warm-up teacher / data generator** (decision 3). The hypothesis under test is that distilled-MCTS + RL + sequence memory matches or beats live MCTS while being far cheaper at inference. The Search API stays available for data-gen; a hybrid search-at-inference is explicitly out of scope for the first cut.
7. **Reward shaping mismatch** — most selects don't change prizes/HP ⇒ sparse signal; over-shaping skews play. **First cut: binary-only** (`binary_rl.gin`, Part E) sidesteps shaping bugs entirely; add a faithful prize/KO/win shaping only if binary plateaus.
8. **Canonical-order stability** — the slot ordering must be **identical** in the replay parser and at inference, or the policy is mis-aligned. Single source of truth for the comparator.
9. **Dep/runtime** — AMAGO 3.4.0 is vendored (matches metamon's pin) and wants `torch>=2.6` (we have 2.8) + `gymnasium<=0.29.1` **[verify our gymnasium]**. We **drop poke-env** entirely (good). `@torch.compile` on the encoder (`metamon_to_amago.py:610`) is optional — disable if it fights our setup.
10. **Inference statefulness** — the Kaggle `agent()` is called once per select with no explicit episode boundary; we must persist/reset the TrajEncoder hidden state per game (detect new game via `obs.select is None` / `turn`/state reset). Get this wrong → policy sees a corrupted history.
11. **Replay obs/action alignment [open question, parser-blocking]** — in `episode-81475913-replay.json`, naive same-step `(obs[t],action[t])` pairing **violates the `[minCount,maxCount]` contract at 2/12 steps**; both resolve if `action[t]` pairs with `obs[t-1]` (a Kaggle replay shift), contradicting the README's same-step assumption. **Does not block the first cut** (self-play controls alignment), but must be verified against the engine before the deferred replay parser, or BC targets are silently corrupted. Verify by replaying recorded actions through the engine and diffing state (Part I).
12. **`N_max` spans all single-select contexts** — `N_max` must cover the largest *policy-predicate* `len(option)` over **every** context (sub-selects like "pick 1 of N looking/deck cards" can exceed MAIN's). Phase 0 histograms `len(option)` split by the predicate (Part D.4), not just MAIN. Undersizing silently truncates legal options.

---

## Part G — Proposed phases (incremental, each independently verifiable)

- **Phase 0 — Recon (no model).** Instrument N **Lucario-mirror** games: histogram `len(option)` and `SelectType/Context` **split by single- vs multi-select**, selects-per-game, and the fraction of selects the env auto-resolves (Part D.4) — **and include the pre-MAIN setup selects** (`IS_FIRST`, `SETUP_ACTIVE_POKEMON`, `SETUP_BENCH_POKEMON`; `SETUP_ACTIVE`/`IS_FIRST` are policy single-selects, not just MAIN — R5). *Verify:* a short script prints the distributions; sets `N_max` (single-select only) and `max_seq_len`. (Singleton is moot under single-process collection.)
- **Phase 1 — Skeleton `tcgmon/` + AMAGO plumbing smoke.** `ObservationSpace`/`ActionSpace(Discrete(N_max))`/`BinaryReward` + `TcgmonEnv` (with auto-resolve) + `TcgmonAMAGOWrapper` + `TcgmonTstepEncoder`, **masking ON**. Random policy through AMAGO collection; one loss step. *Verify:* one Lucario game completes through AMAGO; an `RLData` round-trips; loss runs.
- **Phase 2 — Rule-based distillation warm-up (GO/NO-GO milestone).** Record `rulebased_lucario` self-play → `RLDataset`; train the `il.gin` offline filtered-BC objective. This single phase exercises the **entire pipeline** (obs → single-select action → encoder → data → AMAGO offline) end-to-end. *Verify:* the warm-started policy **beats `random_agent`** and **approaches the rule-based teacher** on the Lucario mirror (`eval.py`, seat-swapped, Wilson CI). *If it can't even imitate a hand-coded agent on one deck, stop and fix here — nothing downstream matters.*
- **Phase 2b — MCTS teacher.** Same recorder/objective, teacher = `tcg_model.mcts_agent`. *Verify:* distilled policy approaches its MCTS teacher in `eval.py`.
- **Phase 3 — RL fine-tune.** `binary_rl.gin` (online) from the warm-up checkpoint, **single frozen opponent**. *Verify:* win-rate vs rule-based improves over the warm-up checkpoint in `eval.py`.
- **Phase 4 — Kaggle inference shim.** `agent(obs_dict)` with persistent hidden state + the same auto-resolve loop (Part D.4) + deck-on-first-call from `deck.csv`. *Verify:* a `run_battle.py`-style loop completes a full Lucario game driven by the trained policy.

**Overall verification harness:** `eval.py` (head-to-head win rates + Wilson CIs + Elo) is the throughline — every phase is graded by it against `random_agent` and `rulebased_lucario`, now **clean mirrors** since everyone plays the Lucario deck.

---

## Part H — Decisions (settled) + one implementation default

**Settled with you (architecture — unchanged):**
1. **Action space** → `Discrete(N_max)` slots + masking + multi-select decomposition (pointer-head = fallback only).
2. **MCTS** → dropped from the policy; survives as warm-up teacher / data generator.
3. **Warm-up data** → self-play first (distillation); human-replay parser deferred.

**First-cut scope (2026-06-26 — phasing, not architecture; see "First-cut scope" up top):**
4. **Single deck = Lucario `deck.csv`** (both seats + all eval).
5. **Multi-select deferred** → env auto-resolves it (Part D.4); first policy is pure single-select.
6. **Masking ON** (Part D.5); **binary reward only** (Part E); **rule-based teacher first, then MCTS** (Part G); **single-process, single frozen opponent** (Part E).

**Implementation default (unchanged):**
7. **First-cut TstepEncoder** → **reuse `tcg_model`'s `EmbeddingBag` sparse encoder** (known-working on this exact obs, least new code) wrapped in AMAGO's `TstepEncoder` interface, rather than porting Metamon's token-vocabulary + `TransformerTurnEmbedding`. The `TstepEncoder` boundary makes a later swap local.

---

## Part I — Worked turn: concrete trace + verification checklist

*Derived from a real 12-decision turn in `data/replays/episode-81475913-replay.json` + constants read from `tcg_model.py`. These are the concrete acceptance criteria for Phases 0–2.*

> ⚠ **CORRECTED — see Rulebook-consistency banner (R1):** this trace is the legacy
> **Abomasnow/Kyogre** replay, NOT the committed Lucario deck — every concrete card ID/index
> below is wrong-deck and must not be used as Lucario acceptance numbers. Lucario values:
> Riolu **677** (PLAY `2248`), Fighting energy card **6** (`2845`), attach target **677/678**
> (`4784/4785`); `EnergyType.FIGHTING=6`, `WATER=3`. Regenerate on a real Lucario game in Phase 0.

**Exact constants:** `card_count=1268`, `attack_count=1557`, `num_words_encoder=24`, `encoder_size=22000`, `decoder_main_feature=8`, `decoder_attack_offset=14`, `decoder_card_offset=1571`, `decoder_size=73,847`. `N_max` TBD (Phase 0). `rl2s` dim `= 1+N_max`.

**Decoder feature indices** (`get_decoder_input`, `SparseVector.pos==0` throughout — confirmed no `add_pos`): `empty→0, END→1, YES→2, NO→3, SPECIAL_CONDITION s→4+s, NUMBER n→9+min(n,4), ATTACK aid→14+aid, PLAY C→1571+C, ATTACH(S,T)→{2839+S,4107+T}, EVOLVE→{5375+S,6643+T}, ABILITY C→7911+C, DISCARD C→9179+C, RETREAT A→10447+A, CARD/TOOL/ENERGY ctx→1571+(8+ctx)·1268+C`. (These are decoder-vocab indices, **distinct** from `OptionType` enum values.)

**A turn is a sequence of selects** (each MAIN option is a fully-specified action; `END` ends the turn, `RETREAT` doesn't; `battle_select` returns the next select routed by `current.yourIndex` (usually the same player mid-turn, but KO-replacement/Checkup selects go to the **other** player out-of-turn — R4); reward terminal-only). Under the Part-D.4 predicate a real **10-select turn → ~4 policy timesteps** (6 auto-resolved: optional draws, a 2-of-4 discard, forced 1-option evolves).

**One policy decision (MAIN, ctx=0), end-to-end:**
1. canonical-sort options → slots, keep `canon→raw` map; e.g. `slot0=PLAY Riolu(721)`, `slot1=ATTACH FightingEnergy(3)→Lucario(722)`, `slot2=END`, slots 3..N−1 PAD/illegal.
2. decoder bags: slot0 `idx[2292]`; slot1 `idx[2842,4829]`; slot2 `idx[1]` → `offset=[0,1,3]`.
3. encoder: 24 bags (e.g. word17 your-active = `[hp/400, card722, energy3]`).
4. pad → `obs{enc_idx/val (24,K_enc), dec_idx/val (N_max,K_dec), illegal_actions (N_max,)=[F,F,F,T…], missing_action_mask (1,)}`.
5. `rl2s (1,1,1+N_max) = [reward, onehot(prev_slot)]` (zeros at game start); `time_idxs` game-global.
6. TstepEncoder → `(1,1,emb_dim)` → TformerTrajEncoder(hidden carried) → actor `Linear→N_max` → `masked_fill_(illegal,-inf)` → sample slot.
7. de-canonicalize slot→raw → `battle_select([raw])`. Trajectory lengths: `obs/rl2s/time=T+1`, `actions/rews/dones=T`; `rews=0` until terminal ±1, `done` only at `result>=0`.

| Verification check | Method |
|---|---|
| canonical order deterministic | sort twice == ; round-trip slot→raw→re-canon→same slot over N selects |
| sparse parity | padded-obs EmbeddingBag pooling == native `SparseVector` pooling, on sampled States |
| no truncation | every option nnz ≤ `K_dec`, every word nnz ≤ `K_enc`; raise on overflow |
| `rl2s` correctness | `rl2s[t]==[rew[t-1],onehot(act[t-1])]`, `rl2s[0]==0`, dim `1+N_max` |
| masking / legality | sampled slot ∈ legal; **engine accepts every `battle_select`** (no ValueError) over N games |
| auto-resolve | resolution ∈ `[minCount,maxCount]`, no dupes; **same code path** in recorder & inference |
| trajectory shapes | `obs/rl2s/time=T+1`, `act/rew/done=T`; `time_idxs=0..T` monotonic across turns |
| recorder fidelity | replay recorded actions through engine → identical terminal result + state hashes (catches the off-by-one class, risk #11) |
| slot-alignment (risk #1) | top-1 option-match vs teacher on held-out states; then win-rate vs `random`/teacher (`eval.py`) |
| end-to-end (Phase 1) | one self-play Lucario game completes through AMAGO; an `RLData` round-trips; one loss step runs |
