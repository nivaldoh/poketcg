# Kaggle remote-training path (BR-exploitability sweep)

Run `train_nashpg.py` on Kaggle script-kernels for the §6.4 best-response (BR) exploitability sweep
(and, later, heavier NASHPG runs). This is **training infra only** — it never enters the
`submission.tar.gz` boundary (`main.py` + `cg/` + `deck.csv`).

## What runs where

- **cg engine:** comes from the **competition mount** (`competition_sources: ["pokemon-tcg-ai-battle"]`),
  not re-uploaded. `run_br.py` globs `/kaggle/input/**/cg/sim.py` (fallback `libcg.so`) and sets
  `CG_LIB_PATH` — defensive because the docs (`sample_submission/cg/`) and the starters (`cg-lib/cg`)
  disagree on the path. If the competition mount lacks the engine, upload `vendor/cg-lib/` as a
  **PRIVATE** dataset and add it to `dataset_sources`.
- **our code + probe ckpts:** a PRIVATE Kaggle dataset `…/poketcg-nashpg-code-ckpts` containing the
  full import closure of `train_nashpg.py` — note `eval.py` imports the rule-based starters, which at
  **import time** `open(decks/<name>.csv)` and call `all_card_data()`, so `decks/` ships too.
- **outputs:** ckpts + `*_metrics.jsonl` (per-update `br_return` curve) + `*_result.json` (the n≥200
  final BR-vs-target win% + Wilson CI) land in `/kaggle/working` and are pulled back locally.

## One-time + per-sweep flow (driver: `push_sweep.py`)

```bash
# 1) publish the code+ckpt dataset (PRIVATE). Re-run `stage` (no --create) to push new ckpts/code.
python kaggle/push_sweep.py stage --create

# 2) render + push one CPU script-kernel per (target, seed).  --dry-run to inspect first.
python kaggle/push_sweep.py sweep --dry-run
python kaggle/push_sweep.py sweep

# 3) poll, then pull outputs once complete
python kaggle/push_sweep.py status
python kaggle/push_sweep.py pull

# 4) aggregate: compares a0_final vs a02_final BR-win% via non-overlapping Wilson CIs
python kaggle/push_sweep.py report
```

Scope any step with `--targets <stems…> --seeds <ints…>` (defaults = the plan matrix:
`nashpg_a0_final nashpg_a02_final nashpg_a0_init nashpg_a02_init` × seeds `0 1 2`).
Smoke first: `sweep --targets nashpg_a0_init --seeds 0` is the smallest end-to-end check
(validates the cg glob, `--out-dir /kaggle/working`, and the dataset import closure).

## Reading the result

Higher BR-win% ⇒ the target is **more exploitable** (a weak BR under-estimates, so every number is a
**lower bound** — keep the `*_init` ceiling rows to prove the BR can exploit at all). **Headline:**
if `a02_final` BR-win% < `a0_final` BR-win% with **separated** Wilson CIs, the NASHPG KL anchor makes
the policy less exploitable (closer to Nash) — the rigorous method validation.

## Notes & limits

- **CPU kernels** (`enable_gpu:"false"`): cg is CPU-bound (~4 Kaggle cores); the tiny model barely uses
  the GPU. Reserve the ~30h/wk GPU quota for heavier NASHPG runs. Parallelism = **N independent kernels**
  (the cg battle is a process-global serial singleton — no intra-kernel parallelism).
- **Sessions cap ~9–12h.** For longer runs, pull the latest `*_br_*.pth`, `stage` it into the dataset,
  and re-push with `--load <ckpt>` added to `run_br.py`'s argv. (`--load` restores **weights only** —
  Adam state is not carried across sessions; a documented warm-restart.)
- `_build/` and `out_kaggle/` are gitignored (large + regenerable).
