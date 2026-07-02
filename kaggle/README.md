# Kaggle remote-training path (PPO runs/sweeps)

Run `train_ppo.py` on Kaggle script-kernels (one kernel per seed) when a run is too heavy for the
local WSL2 box. This is **training infra only** — it never enters the `submission.tar.gz` boundary
(`main.py` + `cg/` + `deck.csv`).

## What runs where

- **cg engine:** comes from the **competition mount** (`competition_sources: ["pokemon-tcg-ai-battle"]`),
  not re-uploaded. `run_ppo.py` globs `/kaggle/input/**/cg/sim.py` (fallback `libcg.so`) and sets
  `CG_LIB_PATH` — defensive because the docs (`sample_submission/cg/`) and the starters (`cg-lib/cg`)
  disagree on the path. If the competition mount lacks the engine, upload `vendor/cg-lib/` as a
  **PRIVATE** dataset and add it to `dataset_sources`.
- **our code:** a PRIVATE Kaggle dataset `…/poketcg-ppo-code` containing the full import closure of
  `train_ppo.py` — note `eval.py` imports the rule-based agents, which at **import time**
  `open(decks/<name>.csv)` and call `all_card_data()`, so `decks/` ships too (flat; the kernel
  wrapper rebuilds `decks/` and `utils/`). **No checkpoints**: PPO trains from a fresh net
  (`train_ppo.py` has no `--load`).
- **outputs:** `ppo_<name>_s<seed>{_init,N,_final}.pth` ckpts (~1.5 MB each) + the per-update
  `*_metrics.jsonl` land in `/kaggle/working` and are pulled back locally.

## One-time + per-sweep flow (driver: `push_sweep.py`)

```bash
# 1) publish the code dataset (PRIVATE). Re-run `stage` (no --create) to push code updates.
python kaggle/push_sweep.py stage --create

# 2) render + push one CPU script-kernel per seed.  --dry-run to inspect first.
python kaggle/push_sweep.py sweep --name field --seeds 0 1 --dry-run
python kaggle/push_sweep.py sweep --name field --seeds 0 1

# 3) poll, then pull outputs once complete
python kaggle/push_sweep.py status --name field --seeds 0 1
python kaggle/push_sweep.py pull --name field --seeds 0 1

# 4) training-curve summary (collect win-rate + health diagnostics)
python kaggle/push_sweep.py report --name field --seeds 0 1
```

Pass extra `train_ppo.py` flags verbatim with `--extra` (must be the **last** flag):
`sweep --name mix --seeds 0 --extra --opponent rulebased,random --ent-coef 0.03`.
Smoke first: `sweep --name smoke --seeds 0 --updates 3 --games 2` is the smallest end-to-end check
(validates the cg glob, the flat-dataset `decks/`+`utils/` rebuild, and the import closure).

## Reading the result

`report` summarizes the training curve: collect win-rate vs the (fixed) opponent field is a genuine
progress signal, and `ev_true_pre` (holdout critic EV) is the health gate — trust it over in-sample
`explained_var` (eval-discipline skill → `ppo-health-reference.md`). The **verdict** is local:
`eval_ppo.py --games 200` on the pulled `_final` vs `_init` ckpts (per-cell Wilson-CI comparison).

## Notes & limits

- **CPU kernels** (`enable_gpu:"false"`): cg is CPU-bound (~4 Kaggle cores); the tiny net barely uses
  a GPU. Parallelism = **N independent kernels** (the cg battle is a process-global serial
  singleton — no intra-kernel parallelism).
- **Concurrency cap:** Kaggle runs at most **5 concurrent batch CPU sessions** — push in waves of ≤5.
- **Zombie kernels:** rejected/failed pushes can leave empty kernel records that poison the slug
  (`Notebook not found` on re-push). Use a **distinct `--name` per experiment**; the CLI has **no
  `kernels delete`** — clean zombies via the web UI.
- **Sessions cap ~9–12h; there is no resume.** `train_ppo.py` has no `--load` flag, so size
  `--updates` to fit one session (the 800×16 default is a few hours of games at cg's serial pace —
  check the jsonl's `games_per_s` from a smoke run before scaling up).
- `_build/` and `out_kaggle/` are gitignored (large + regenerable).
