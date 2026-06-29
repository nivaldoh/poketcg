#!/usr/bin/env python3
"""Kaggle script-kernel entry for a BR-exploitability run (plan.md §6.4; see kaggle/README.md).

`push_sweep.py` renders one copy of this per (target, seed) by overwriting the CONFIG block below,
then `kaggle kernels push`es it. The kernel runs `train_nashpg.py --br-vs <target>` on Kaggle and
writes checkpoints + metrics to /kaggle/working (the downloadable kernel output).

Why a thin runpy wrapper instead of importing train_nashpg: Kaggle kernels take no CLI args, so we
synthesize sys.argv and run the module as __main__. This keeps train_nashpg.py itself Kaggle-agnostic
(the submission boundary stays clean -- this file is TRAINING infra, never part of submission.tar.gz)."""
import glob
import os
import shutil
import sys

# === CONFIG (push_sweep.py overwrites these four lines per kernel; keep them simple assignments) ===
BR_TARGET = "nashpg_a02_final"   # checkpoint STEM to best-respond to (globbed from the attached dataset)
SEED = 0
BR_UPDATES = 300
GAMES = 32

# 1) Locate the cg engine in the attached competition/dataset and point CG_LIB_PATH at it BEFORE
#    importing anything that pulls in tcg_model. The competition layout is unverified (docs say
#    sample_submission/cg/, the starters glob cg-lib/cg), so glob defensively for the engine itself.
_hits = (glob.glob("/kaggle/input/**/cg/sim.py", recursive=True)
         or glob.glob("/kaggle/input/**/libcg.so", recursive=True))
assert _hits, "cg engine not found under /kaggle/input -- attach the competition or the cg fallback dataset"
_cg_dir = os.path.dirname(_hits[0])                    # .../cg
os.environ["CG_LIB_PATH"] = os.path.dirname(_cg_dir)   # parent of cg/  (what tcg_model expects)

# 2) Mirror our code dataset (read-only mount) into a WRITABLE dir, reconstructing decks/. The
#    rule-based starters imported by eval.py do `open(<module_dir>/decks/<name>.csv)` and call
#    all_card_data() AT IMPORT, and the deck CSVs ship FLAT (Kaggle has no keep-as-folder dir mode).
_code = glob.glob("/kaggle/input/**/train_nashpg.py", recursive=True)
assert _code, "train_nashpg.py not found -- attach the poketcg code+ckpt dataset"
_src_dir = os.path.dirname(_code[0])
WORK = "/kaggle/working/_code"
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(os.path.join(WORK, "decks"))
for _py in glob.glob(os.path.join(_src_dir, "*.py")):
    shutil.copy(_py, WORK)
for _name in ("lucario", "abomasnow", "dragapult_ex", "iono"):
    _hit = glob.glob(f"/kaggle/input/**/{_name}.csv", recursive=True)
    if _hit:
        shutil.copy(_hit[0], os.path.join(WORK, "decks", f"{_name}.csv"))
sys.path.insert(0, WORK)

# 3) Resolve the frozen target checkpoint by stem.
_tgt = glob.glob(f"/kaggle/input/**/{BR_TARGET}.pth", recursive=True)
assert _tgt, f"target {BR_TARGET}.pth not found in attached datasets"

sys.argv = ["train_nashpg.py",
            "--br-vs", _tgt[0],
            "--out-dir", "/kaggle/working",            # code mirror is under here; outputs go here too
            "--games", str(GAMES), "--br-updates", str(BR_UPDATES),
            "--lam", "0.95", "--epochs", "4", "--clip-vloss",   # real BR config (reference recipe)
            "--final-eval-games", "200", "--seed", str(SEED),
            "--out-prefix", f"br_{BR_TARGET}_s{SEED}"]
print("CG_LIB_PATH:", os.environ["CG_LIB_PATH"], flush=True)
print("kernel argv:", " ".join(sys.argv), flush=True)

import runpy
runpy.run_path(os.path.join(WORK, "train_nashpg.py"), run_name="__main__")
