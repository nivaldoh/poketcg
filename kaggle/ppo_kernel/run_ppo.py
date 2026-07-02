#!/usr/bin/env python3
"""Kaggle script-kernel entry for one train_ppo.py run (see kaggle/README.md).

`push_sweep.py` renders one copy of this per (name, seed) by overwriting the CONFIG block below,
then `kaggle kernels push`es it. The kernel runs `train_ppo.py` on Kaggle and writes checkpoints +
metrics to /kaggle/working (the downloadable kernel output).

Why a thin runpy wrapper instead of importing train_ppo: Kaggle kernels take no CLI args, so we
synthesize sys.argv and run the module as __main__. This keeps train_ppo.py itself Kaggle-agnostic
(the submission boundary stays clean -- this file is TRAINING infra, never part of submission.tar.gz)."""
import glob
import os
import shutil
import sys

# === CONFIG (push_sweep.py overwrites these five lines per kernel; keep them simple assignments) ===
RUN_NAME = "field"   # experiment name (also names the output prefix ppo_<name>_s<seed>)
SEED = 0
UPDATES = 800
GAMES = 16           # games per update
EXTRA_ARGS = []      # passthrough train_ppo.py flags, e.g. ["--ent-coef", "0.03"] (argparse last-wins)

# 1) Locate the cg engine in the attached competition/dataset and point CG_LIB_PATH at it BEFORE
#    importing anything that pulls in tcg_model. The competition layout is unverified (docs say
#    sample_submission/cg/, the starters glob cg-lib/cg), so glob defensively for the engine itself.
_hits = (glob.glob("/kaggle/input/**/cg/sim.py", recursive=True)
         or glob.glob("/kaggle/input/**/libcg.so", recursive=True))
assert _hits, "cg engine not found under /kaggle/input -- attach the competition or the cg fallback dataset"
_cg_dir = os.path.dirname(_hits[0])                    # .../cg
os.environ["CG_LIB_PATH"] = os.path.dirname(_cg_dir)   # parent of cg/  (what tcg_model expects)

# 2) Mirror our code dataset (read-only mount) into a WRITABLE dir, reconstructing the two
#    directories the flat dataset can't carry (Kaggle has no keep-as-folder dir mode):
#    decks/ -- the rule-based agents imported by eval.py `open(<module_dir>/decks/<name>.csv)` and
#    call all_card_data() AT IMPORT; utils/ -- train_ppo does a PACKAGE import
#    `from utils.run_smoketest_battle import load_deck` (namespace package, no __init__.py needed).
_code = glob.glob("/kaggle/input/**/train_ppo.py", recursive=True)
assert _code, "train_ppo.py not found -- attach the poketcg code dataset"
_src_dir = os.path.dirname(_code[0])
WORK = "/kaggle/working/_code"
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(os.path.join(WORK, "decks"))
os.makedirs(os.path.join(WORK, "utils"))
for _py in glob.glob(os.path.join(_src_dir, "*.py")):
    shutil.copy(_py, WORK)
shutil.move(os.path.join(WORK, "run_smoketest_battle.py"),
            os.path.join(WORK, "utils", "run_smoketest_battle.py"))
for _name in ("lucario", "abomasnow", "dragapult_ex", "iono"):
    _hit = glob.glob(f"/kaggle/input/**/{_name}.csv", recursive=True)
    if _hit:
        shutil.copy(_hit[0], os.path.join(WORK, "decks", f"{_name}.csv"))
sys.path.insert(0, WORK)

sys.argv = ["train_ppo.py",
            "--updates", str(UPDATES), "--games-per-update", str(GAMES),
            "--seed", str(SEED),
            "--out-dir", "/kaggle/working",            # code mirror is under here; outputs go here too
            "--out-prefix", f"ppo_{RUN_NAME}_s{SEED}",
            "--save-every", "50",                      # bound output size; EXTRA_ARGS can override
            ] + [str(a) for a in EXTRA_ARGS]
print("CG_LIB_PATH:", os.environ["CG_LIB_PATH"], flush=True)
print("kernel argv:", " ".join(sys.argv), flush=True)

import runpy
runpy.run_path(os.path.join(WORK, "train_ppo.py"), run_name="__main__")
