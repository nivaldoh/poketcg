# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project to train an AI agent for the **Pokémon TCG** Kaggle simulation competition (`pokemon-tcg-ai-battle`). Goal per `README.md`: get the Kaggle starter code running and training **locally** (Ubuntu WSL2) with verifiable results. Read `README.md` for goals/constraints and working style (minimal, incremental, verifiable changes; state assumptions; flag missing info — the rulebook and card CSVs/PDFs aren't provided yet); `ideas.md` for backlog.

## The `cg` simulator

The starters import a simulator SDK as `cg` — a **proprietary** compiled C++ engine (`libcg.so`, via `ctypes`) from the Kaggle competition. It is **not committed** (gitignored) but **vendored** at `vendor/cg-lib/cg/`, so code runs here. If it's missing (fresh clone), restore it from Kaggle — see `docs/cg-sdk.md`.

Environment: Python 3.13 (miniconda), `torch` 2.8 (CUDA available).

## Running locally

The SDK path is wired via `CG_LIB_PATH` (default `vendor/cg-lib`), replacing the Kaggle `glob('/kaggle/input/**/cg-lib')` injection.

```bash
# End-to-end battle smoke test (random vs random); asserts a terminal result.
python run_battle.py [--games N]

# RL+MCTS training. No args = original (heavy) Kaggle config; small numbers = fast
# smoke pass. Writes checkpoints to out/modelN.pth.
python starter_rl_kaggle.py --iterations 1 --eval-games 2 --selfplay-games 2 --search-count 2

# Stage-1 eval harness (our PRIMARY model-performance signal): head-to-head win
# rates with 95% Wilson CIs + a rough Elo. Default = a sanity battery of
# random / rule-based / checkpoint matchups (seat-swapped).
python eval.py --games 30 --search-count 6
```

No test framework yet; `run_battle.py` is the smallest verifiable signal the engine runs, and `eval.py` is the primary way to compare agents/checkpoints. `vendor/`, `deck.csv`, and `out/` are gitignored.

## Modules

- `tcg_model.py` — the **reusable model/agent layer**, extracted from the trainer so tools can import it without launching training: `MyModel` (Transformer: encoder = value, decoder = policy), the sparse encoder/decoder feature builders, `mcts_agent`, `random_agent`, and `sample_deck`. Importing it only loads card metadata (no side effects).
- `eval.py` — standalone **Stage-1 eval harness** built on `tcg_model` + `rulebased_lucario` (see README "Eval harness").

## The two starter files

Both are **Kaggle-notebook exports, not plain modules**:
- `starter_rl_kaggle.py` — RL + MCTS self-play training (Transformer: encoder = value, decoder = policy). Adapted to run locally (CLI flags, local `cg` path); the model/agent code now lives in `tcg_model.py`, which it imports.
- `starter_rulebased_lucario_kaggle.py` — heuristic agent for a Mega Lucario ex deck. **Still in Kaggle form, not yet ported**: starts with `%%writefile main.py`, ends by building `submission.tar.gz`; the reusable part is `agent()` and its scoring heuristics.

Kaggle-specific glue to strip when porting: the `glob('/kaggle/input/...')` path injection, `%%writefile`, the `/kaggle_simulations/agent/` deck-path fallback, and `tarfile` packaging.

## Architecture & API

The agent contract is one function: `agent(obs_dict) -> list[int]` returning **option indices** into `obs.select.option` (length in `[minCount, maxCount]`, no dupes); on the first call `obs.select is None` and it must return a 60-card deck instead. Battles run via `battle_start` → `battle_select` → `battle_finish` (see `run_battle.py`); MCTS uses a determinizing Search API to handle hidden opponent cards.

Full reference — observation/option/card model, driver loop, Search API & MCTS internals — is in **`docs/cg-sdk.md`** (ground truth from the vendored source).

## Conventions

- New execution/training code should ship the smallest possible verification (e.g. a short battle asserting a `result`), per the README's "verifiable results" goal.
- Keep the Kaggle submission boundary clean: a deployable agent is ultimately `main.py` (`agent(obs_dict)`) + the `cg` package + `deck.csv`, packaged as `submission.tar.gz`. Don't entangle local training-harness code with it.
