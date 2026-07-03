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
- `ppo_model.py` / `train_ppo.py` / `eval_ppo.py` — the **active training stack**: a clean-room PPO (fresh ActorCritic net, independent of `MyModel`), its trainer (default opponent = the rule-based field), and its deck×opponent matrix eval. Self-tests: `python ppo_model.py` / `--update-test`.
- `viz_ppo.py` — **visual reports over the PPO eval outputs** (offline; stdlib+numpy+matplotlib only, no torch/cg — outside the submission boundary; dep: `pip install matplotlib`). Subcommands `train` (a `*_metrics.jsonl`), `probe` (a `probe_credit_*.json`), `eval` (an `eval_ppo.py --json` results file); each writes PNGs + a self-contained captioned HTML report under `out/viz/<input-stem>/`.
- `win_mode.py` — the ONE shared terminal-obs win-mode classifier (`classify_end(obs, seat)`: engine RESULT-log reason 1=prize/2=deck/3=bench/4=effect + terminal prize/deck/in-play margins, seat-relative) + `summarize_ends` (the one aggregation of those dicts — eval_ppo `end_stats` cells, probe_credit rows). Consumed by `train_ppo.collect_games` (→ metrics `per_game`), `eval._play_one` (→ eval_ppo `end_stats`) and, via records, probe_credit per-game entries. Self-check: `python win_mode.py`.
- `eval.py` — (see README "Eval harness").
- `game_agents/rulebased/<deck>.py` — the heuristic agents, **one clean importable module per deck**: `lucario` (Mega Lucario ex), `abomasnow` (Mega Abomasnow ex), `dragapult_ex`, `iono` (Iono's Bellibolt/Voltorb). The reusable part of each is `agent()` and its scoring heuristics; each also exposes `my_deck` (its 60-card list). `eval.py` / `starter_rl_kaggle.py` add `game_agents/rulebased/` to `sys.path` and import them by bare name.

## The starter files

`starter_rl_kaggle.py` is a **Kaggle-notebook export, not a plain module** (it opens with `%%writefile main.py` and ends by building `submission.tar.gz`): RL + MCTS self-play training (Transformer: encoder = value, decoder = policy). Adapted to run locally (CLI flags, local `cg` path); the model/agent code now lives in `tcg_model.py`, which it imports. It is the **legacy** trainer — active training is `train_ppo.py`.

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