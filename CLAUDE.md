# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project to train an AI agent for the **Pokémon TCG** Kaggle simulation competition (`pokemon-tcg-ai-battle`). Goal per `README.md`: get the Kaggle starter code running and training **locally** (Ubuntu WSL2) with verifiable results. Read `README.md` for goals/constraints and working style (minimal, incremental, verifiable changes; state assumptions);

## The `cg` simulator

The starters import a simulator SDK as `cg` — a **proprietary** compiled C++ engine (`libcg.so`, via `ctypes`) from the Kaggle competition. It is **not committed** (gitignored) but **vendored** at `vendor/cg-lib/cg/`, so code runs here. If it's missing (fresh clone), restore it from Kaggle — see `docs/cg-sdk.md`.

Environment: Python 3.13 (miniconda), `torch` 2.8 (CUDA available).

## Running locally

The SDK path is wired via `CG_LIB_PATH` (default `vendor/cg-lib`), replacing the Kaggle `glob('/kaggle/input/**/cg-lib')` injection.

No test framework yet. `vendor/`, and `out/` are gitignored.

## Modules

- `tcg_model.py` — the **reusable model/agent layer**, extracted from the trainer so tools can import it without launching training: `MyModel` (Transformer: encoder = value, decoder = policy), the sparse encoder/decoder feature builders, `mcts_agent`, `random_agent`, and `sample_deck`.
- `sb3_env.py` / `sb3_features.py` / `train_sb3.py` / `eval_sb3.py` — the **SB3 stack (active)**: `CGEnv` (gymnasium wrapper inverting the battle loop into `reset()`/`step()`; options padded to `N_MAX`, masking via `action_masks()`; one live env per process — the cg battle singleton) + a MaskablePPO driver running `TcgMaskablePolicy` (`sb3_policy.py`, wrapping `tcg_net.py`'s `TcgCore` — an id-embedding per-option scorer that replaced the stock `MultiInputPolicy`'s positional/id-blind head, the measured ~13% plateau architecture; **never rename/move `tcg_net.py` or `sb3_policy.py`** — MaskablePPO checkpoints pickle them by module path) + the matrix eval. `sb3_features.py` holds the cg-obs featurization (`encode_state`/`encode_options` → dense state/option tensors, moved out of the retired `legacy_ppo/` hand-rolled PPO stack — that stack's investigation findings live in `docs/ppo/`); the rule-based opponent field is `sb3_env.make_opponent`; reuses `eval.py` primitives. Deps (training-side only, OUTSIDE the submission boundary): `pip install -e reference_repos/stable-baselines3` + `pip install sb3-contrib` (both 2.9.0). Self-tests: `python sb3_features.py` / `python sb3_env.py` / `utils/run_smoketest_sb3_env.py` / `train_sb3.py --smoke`. Roadmap (id-aware policy + multi-select + BC warm-start): `docs/plans/plan.md` (status: implemented; `docs/plans/plan-multiselect.md` stays as the reference for the multi-select machinery).
- `bc_collect.py` / `bc_train.py` — the **imitation-learning warm-start pipeline**: `bc_collect.py` plays the rule-based teachers against each other and records every learnable decision to pickled shards; `bc_train.py` behavior-clones those shards into a bare `tcg_net.TcgCore` (policy + value), which `train_sb3.py --bc-init` then loads into `TcgMaskablePolicy.core` before PPO fine-tuning. Results: `docs/reports/bc_warmstart_report.md`.
- `win_mode.py` — the ONE shared terminal-obs win-mode classifier (`classify_end(obs, seat)`: engine RESULT-log reason 1=prize/2=deck/3=bench/4=effect + terminal prize/deck/in-play margins, seat-relative) + `summarize_ends` (the one aggregation of those dicts). Consumed by `sb3_env.CGEnv.step` (→ the terminal info dict) and `eval._play_one` (→ end-stats cells). Self-check: `python win_mode.py`.
- `eval.py` — (see README "Eval harness").
- `record_battle.py` + `visualizer.html` — the **local battle visualizer** (a human debug tool, NOT part of the submission). `record_battle.py` plays one battle — or a `--games N` seat-swapped **session** — between two agent specs (`random|rulebased|abomasnow|dragapult|iono|<*.pth>|<*.zip>`) and writes a per-decision `trace.json`: fog obs + full-visibility `visualize_data()` board + decoded `obs["logs"]` + human-readable options (reuses `sb3_features._resolve_option`) + `win_mode.classify_end`. For an SB3 `.zip` it also captures the policy's per-option probs + critic value (`traced_sb3_agent`, a read-only twin of `sb3_env.sb3_agent`) and the labeled 299-d state vector (`STATE_LABELS` mirrors `encode_state`). `visualizer.html` is self-contained/offline EXCEPT its one opt-in 🎮 Graphical replay button, which rebuilds the `visualize_data()` array from the trace (each entry stored verbatim in `step.god`/`final_god`) and POSTs it to the external Heroz renderer (`ptcgvis.heroz.jp`) in a new tab — the same path `kaggle_notebook_to_view_battles.py` uses. Otherwise fully local: open it, load a trace, step through the board + decision + probability bars + collapsible obs/feature/event panels; sessions add a win%+Wilson-CI bar (via `eval.wilson`/`win_mode.summarize_ends`) + game picker. Reuses eval/sb3/win_mode verbatim (doesn't edit them). Run: `python record_battle.py --p0 out/<ckpt>.zip --p1 iono --deck lucario --games 8 --out trace.json`. Plan: `~/.claude/plans/misty-coalescing-metcalfe.md`.
- `game_agents/rulebased/<deck>.py` — the heuristic agents, **one clean importable module per deck**: `lucario` (Mega Lucario ex), `abomasnow` (Mega Abomasnow ex), `dragapult_ex`, `iono` (Iono's Bellibolt/Voltorb), `simple_raging_bolt_burn` (Turbo Raging Bolt ex/Ogerpon; eval/record_battle spec `raging_bolt`; build+eval notes in `docs/reports/raging_bolt_agent_report.md`), `real_raging_bolt_burn` (post-rotation Raging Bolt ex with the Crispin/Glass Trumpet/Energy Switch engine; spec `real_raging_bolt`; notes in `docs/reports/real_raging_bolt_agent_report.md`), `lilies_clefairy_ex_mega_kangaskhan_ex` (NAIC-2026 Lillie's Clefairy ex/Mega Kangaskhan ex bench-scaling Rondo deck; spec `clefairy`/`lilies_clefairy`; the newest reference module — Telepath TO_BENCH handler, 5-type pair-scored attaches; notes in `docs/reports/lilies_clefairy_agent_report.md`). The reusable part of each is `agent()` and its scoring heuristics; each also exposes `my_deck` (its 60-card list). `eval.py` / `starter_rl_kaggle.py` add `game_agents/rulebased/` to `sys.path` and import them by bare name.

## The starter files

`starter_rl_kaggle.py` is a **Kaggle-notebook export, not a plain module** (it opens with `%%writefile main.py` and ends by building `submission.tar.gz`): RL + MCTS self-play training (Transformer: encoder = value, decoder = policy). Adapted to run locally (CLI flags, local `cg` path); the model/agent code now lives in `tcg_model.py`, which it imports. It is a **legacy** trainer kept as reference — active training is the SB3 stack (`train_sb3.py`).

Each rule-based agent (`game_agents/rulebased/<deck>.py`) loads its own 60-card deck from `decks/<name>.csv` (committed; the repo-root `decks/` is two levels up from the module), falling back to `deck.csv` then the Kaggle `/kaggle_simulations/agent/deck.csv` at submission time. Root `deck.csv` (gitignored, = the Lucario list, mirrors `decks/lucario.csv`) stays the learner-side default for the training/eval scripts.

Kaggle-specific glue to strip when porting `starter_rl_kaggle.py` (or a fresh Kaggle rule-based export) to a local module: the `glob('/kaggle/input/...')` path injection, `%%writefile`, and `tarfile` packaging. (The four rule-based agents have already been cleaned this way.)

## Architecture & API

The agent contract is one function: `agent(obs_dict) -> list[int]` returning **option indices** into `obs.select.option` (length in `[minCount, maxCount]`, no dupes); on the first call `obs.select is None` and it must return a 60-card deck instead. Battles run via `battle_start` → `battle_select` → `battle_finish` (see `utils/run_smoketest_battle.py`); MCTS uses a determinizing Search API to handle hidden opponent cards.

Full reference — observation/option/card model, driver loop, Search API & MCTS internals — is in **`docs/cg-sdk.md`** (ground truth from the vendored source).

## Conventions

- Be careful with assumptions and overconfident claims, especially on conceptual/theoretical aspects. You are an agent focused on coding. Theory claims need to be marked for independent review.
- New execution/training code should ship the smallest possible verification (e.g. a short battle asserting a `result`), per the README's "verifiable results" goal.
- Keep the Kaggle submission boundary clean: a deployable agent is ultimately `main.py` (`agent(obs_dict)`) + the `cg` package + `deck.csv`, packaged as `submission.tar.gz`. Don't entangle local training-harness code with it.
- Write against drift: in docs, skills, and memory, reference symbols by name + a grep recipe (never pinned line numbers) and discover paths/defaults at runtime (never hardcode `out/*.pth`, `deck.csv`, or flag defaults). Verify a file/symbol/flag still exists before recommending it; the `cg` engine and `all_card_data()` are ground truth over any doc where they conflict. (Fuller per-skill version: `docs/skills_and_hooks.md` → "Drift-resilience contract".)

## Environment
Our environment is Ubuntu WSL 2 through Windows 11. Kaggle CLI credentials have been set up at ~/.kaggle/access_token. The local env is very limited in terms of resources, so we won't be able to run large experiments locally. We should see if we're able to use the Kaggle CLI for this.