#!/usr/bin/env python3
"""Local driver for the Kaggle BR-exploitability sweep (plan.md §6.4; see kaggle/README.md).

Stages the code+ckpt dataset, renders one script-kernel per (target, seed), pushes them, and
polls/pulls/aggregates the results. This is TRAINING infra -- it never touches the submission
boundary (main.py + cg/ + deck.csv). Kaggle CLI auth is read from ~/.kaggle.

Usage:
  python kaggle/push_sweep.py stage --create     # first publish of the code+ckpt dataset
  python kaggle/push_sweep.py stage              # update the dataset (new ckpts / code)
  python kaggle/push_sweep.py sweep              # render+push one kernel per (target, seed)
  python kaggle/push_sweep.py sweep --dry-run    # render the build dirs but don't push
  python kaggle/push_sweep.py status             # poll every kernel
  python kaggle/push_sweep.py pull               # download outputs into kaggle/out_kaggle/
  python kaggle/push_sweep.py report             # aggregate *_result.json into a comparison table

Scope it with --targets / --seeds (defaults = the plan's matrix).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_BUILD = os.path.join(_HERE, "_build")
_OUT = os.path.join(_HERE, "out_kaggle")

DATASET_SLUG = "poketcg-nashpg-code-ckpts"

# The full import closure of train_nashpg.py (eval.py pulls in the rule-based agents, which open
# decks/<name>.csv and call all_card_data() AT IMPORT -- so decks/ MUST ship too). copy2 flattens
# each path to its basename, so the agents land flat in the dataset (run_br.py imports them by name).
CODE_FILES = ["train_nashpg.py", "tcg_model.py", "eval.py", "eval_mymodel.py",
              "game_agents/rulebased/lucario.py", "game_agents/rulebased/abomasnow.py",
              "game_agents/rulebased/dragapult_ex.py", "game_agents/rulebased/iono.py"]
# Probe checkpoints (the experiment matrix's targets, by stem -> out/<stem>.pth).
DEFAULT_TARGETS = ["nashpg_a0_final", "nashpg_a02_final", "nashpg_a0_init", "nashpg_a02_init"]
DEFAULT_SEEDS = [0, 1, 2]


def kuser():
    """Authenticated Kaggle username (for dataset/kernel ids). Drift-resilient: discovered, not hardcoded."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        sys.exit("could not determine Kaggle username (set KAGGLE_USERNAME or run `kaggle config view`)")
    return m.group(1)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def slug_for(target, seed):
    return f"poketcg-br-{target.replace('_', '-')}-s{seed}"


def stage(args):
    """Copy the import closure + decks/ + probe ckpts + dataset-metadata.json into _build/code_ckpt,
    then create (first time) or version (updates) the PRIVATE Kaggle dataset."""
    d = os.path.join(_BUILD, "code_ckpt")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in CODE_FILES:
        shutil.copy2(os.path.join(_REPO, f), d)
    # Deck CSVs ship FLAT (Kaggle datasets have no keep-as-folder dir mode); run_br.py rebuilds the
    # decks/ folder the rule-based starters expect. Flat == no subdirs == no --dir-mode needed.
    for csv in glob.glob(os.path.join(_REPO, "decks", "*.csv")):
        shutil.copy2(csv, d)
    for stem in args.targets:
        src = os.path.join(_REPO, "out", f"{stem}.pth")
        if not os.path.exists(src):
            sys.exit(f"missing checkpoint {src} -- can't stage it")
        shutil.copy2(src, d)
    shutil.copy2(os.path.join(_HERE, "code_ckpt", "dataset-metadata.json"),
                 os.path.join(d, "dataset-metadata.json"))
    print(f"staged {len(CODE_FILES)} modules + deck CSVs + {len(args.targets)} ckpts -> {d}")
    if args.create:
        # PRIVATE by default (no --public). The cg engine is NOT in this dataset (it comes from the
        # competition mount), so there's no proprietary-leak risk here -- only our code + our ckpts.
        run(["kaggle", "datasets", "create", "-p", d])
    else:
        run(["kaggle", "datasets", "version", "-p", d, "-m", args.message])


def _render_kernel(user, target, seed, dataset_id, br_updates, games):
    slug = slug_for(target, seed)
    kdir = os.path.join(_BUILD, "kernels", slug)
    os.makedirs(kdir, exist_ok=True)
    # render run_br.py: overwrite the four CONFIG constants, preserving inline comments
    src = open(os.path.join(_HERE, "br_kernel", "run_br.py")).read()
    consts = {"BR_TARGET": target, "SEED": seed, "BR_UPDATES": br_updates, "GAMES": games}
    for name, val in consts.items():
        v = f'"{val}"' if isinstance(val, str) else str(val)
        src = re.sub(rf"(?m)^{name} = [^\n#]*?(\s*#.*)?$",
                     lambda m, v=v: f"{name} = {v}" + (m.group(1) or ""), src, count=1)
    open(os.path.join(kdir, "run_br.py"), "w").write(src)
    # render kernel-metadata.json: id/title per kernel, dataset_sources -> our dataset
    meta = json.load(open(os.path.join(_HERE, "br_kernel", "kernel-metadata.json")))
    meta["id"] = f"{user}/{slug}"
    meta["title"] = slug
    meta["dataset_sources"] = [dataset_id]
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
    return slug, kdir


def sweep(args):
    user = kuser()
    dataset_id = f"{user}/{DATASET_SLUG}"
    for target in args.targets:
        for seed in args.seeds:
            slug, kdir = _render_kernel(user, target, seed, dataset_id, args.br_updates, args.games)
            print(f"rendered {slug} -> {kdir}  (br_updates={args.br_updates} games={args.games})")
            if not args.dry_run:
                run(["kaggle", "kernels", "push", "-p", kdir])
    if args.dry_run:
        print("\n--dry-run: inspect kaggle/_build/kernels/*/ then re-run without --dry-run to push.")


def status(args):
    user = kuser()
    for target in args.targets:
        for seed in args.seeds:
            run(["kaggle", "kernels", "status", f"{user}/{slug_for(target, seed)}"])


def pull(args):
    user = kuser()
    for target in args.targets:
        for seed in args.seeds:
            slug = slug_for(target, seed)
            run(["kaggle", "kernels", "output", f"{user}/{slug}",
                 "-p", os.path.join(_OUT, slug), "-o"])


def report(args):
    """Aggregate the *_result.json artifacts into an exploitability comparison (higher BR-win% =>
    more exploitable). Compare a0_final vs a02_final via NON-OVERLAPPING Wilson CIs."""
    rows = []
    for target in args.targets:
        for seed in args.seeds:
            slug = slug_for(target, seed)
            # run_br names it br_<target>_s<seed>_result.json
            cand = os.path.join(_OUT, slug, f"br_{target}_s{seed}_result.json")
            if os.path.exists(cand):
                rows.append(json.load(open(cand)))
    if not rows:
        sys.exit(f"no *_result.json found under {_OUT} -- run `pull` first")
    rows.sort(key=lambda r: (r["target"], r["seed"]))
    print(f"{'target':28s} {'seed':>4s} {'BR-win%':>8s} {'95% CI':>15s} {'expl_lb':>8s}")
    for r in rows:
        print(f"{r['target']:28s} {r['seed']:>4d} {r['br_win_pct']:>8.1f} "
              f"{r['ci_lo']:>6.1f}-{r['ci_hi']:<6.1f} {r['exploitability_lb']:>8.3f}")
    # per-target best (max over seeds = tightest lower bound on exploitability)
    best = {}
    for r in rows:
        if r["target"] not in best or r["br_win_pct"] > best[r["target"]]["br_win_pct"]:
            best[r["target"]] = r
    a0, a02 = best.get("nashpg_a0_final"), best.get("nashpg_a02_final")
    if a0 and a02:
        sep = a02["ci_hi"] < a0["ci_lo"] or a0["ci_hi"] < a02["ci_lo"]
        print(f"\nHEADLINE (max-over-seed lower bounds): a0_final BR-win%={a0['br_win_pct']:.1f} "
              f"[{a0['ci_lo']:.1f}-{a0['ci_hi']:.1f}] vs a02_final BR-win%={a02['br_win_pct']:.1f} "
              f"[{a02['ci_lo']:.1f}-{a02['ci_hi']:.1f}]")
        if sep and a02["br_win_pct"] < a0["br_win_pct"]:
            print("  => a02 (NASHPG) is LESS exploitable, CIs separated -> KL anchor validated.")
        elif sep:
            print("  => a0 (plain PPO) is less exploitable, CIs separated -> anchor did NOT help.")
        else:
            print("  => CIs OVERLAP -> inconclusive; need more BR budget / seeds / final-eval games.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage", "sweep", "status", "pull", "report"):
        p = sub.add_parser(name)
        p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
        p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
        if name == "stage":
            p.add_argument("--create", action="store_true", help="first publish (else version-update)")
            p.add_argument("--message", "-m", default="update code+ckpts")
        if name == "sweep":
            p.add_argument("--dry-run", action="store_true", help="render build dirs but don't push")
            p.add_argument("--br-updates", type=int, default=300, help="PPO updates per kernel (small for a smoke)")
            p.add_argument("--games", type=int, default=32, help="self-play games per update")
    args = ap.parse_args()
    {"stage": stage, "sweep": sweep, "status": status, "pull": pull, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
