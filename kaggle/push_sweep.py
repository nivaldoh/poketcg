#!/usr/bin/env python3
"""Local driver for Kaggle PPO training runs/sweeps (see kaggle/README.md).

Stages the train_ppo.py import closure as a PRIVATE Kaggle dataset, renders one script-kernel per
(name, seed), pushes them, and polls/pulls/summarizes the results. This is TRAINING infra -- it
never touches the submission boundary (main.py + cg/ + deck.csv). Kaggle CLI auth is read from
~/.kaggle.

Usage:
  python kaggle/push_sweep.py stage --create     # first publish of the code dataset
  python kaggle/push_sweep.py stage              # update the dataset (new code)
  python kaggle/push_sweep.py stage --dry-run    # stage locally, don't call the kaggle CLI
  python kaggle/push_sweep.py sweep --name field --seeds 0 1     # one kernel per seed
  python kaggle/push_sweep.py sweep --dry-run    # render the build dirs but don't push
  python kaggle/push_sweep.py status             # poll every kernel
  python kaggle/push_sweep.py pull               # download outputs into kaggle/out_kaggle/
  python kaggle/push_sweep.py report             # summarize the pulled *_metrics.jsonl

Scope with --name/--seeds; pass extra train_ppo.py flags verbatim via --extra (MUST be the last
flag, e.g. `sweep --name mix --seeds 0 --extra --opponent rulebased,random --ent-coef 0.03`).
The definitive strength read is LOCAL: pull, then run eval_ppo.py on the fetched ckpts (n=200,
per the eval-discipline skill) -- `report` is only the training-curve summary.
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

DATASET_SLUG = "poketcg-ppo-code"

# The full import closure of train_ppo.py (eval.py pulls in the rule-based agents, which open
# decks/<name>.csv and call all_card_data() AT IMPORT -- so decks/ MUST ship too). copy2 flattens
# each path to its basename; run_ppo.py rebuilds decks/ and utils/ beside the flat modules.
# eval_ppo.py is not in the closure but ships so a kernel/session can also run the matrix eval.
CODE_FILES = ["train_ppo.py", "ppo_model.py", "tcg_model.py", "eval.py", "win_mode.py",
              "eval_ppo.py", "utils/run_smoketest_battle.py",
              "game_agents/rulebased/lucario.py", "game_agents/rulebased/abomasnow.py",
              "game_agents/rulebased/dragapult_ex.py", "game_agents/rulebased/iono.py"]
DEFAULT_SEEDS = [0, 1]


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


def slug_for(name, seed):
    # Distinct slug per experiment: a failed push can leave a zombie kernel record that poisons
    # its slug, so never reuse one across experiments (see kaggle/README.md).
    return f"poketcg-ppo-{name.replace('_', '-')}-s{seed}"


def stage(args):
    """Copy the import closure + decks/ + dataset-metadata.json into _build/code, then create
    (first time) or version (updates) the PRIVATE Kaggle dataset."""
    d = os.path.join(_BUILD, "code")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in CODE_FILES:
        shutil.copy2(os.path.join(_REPO, f), d)
    # Deck CSVs ship FLAT (Kaggle datasets have no keep-as-folder dir mode); run_ppo.py rebuilds the
    # decks/ folder the rule-based agents expect. Flat == no subdirs == no --dir-mode needed.
    for csv in glob.glob(os.path.join(_REPO, "decks", "*.csv")):
        shutil.copy2(csv, d)
    shutil.copy2(os.path.join(_HERE, "code", "dataset-metadata.json"),
                 os.path.join(d, "dataset-metadata.json"))
    print(f"staged {len(CODE_FILES)} modules + deck CSVs -> {d}")
    if args.dry_run:
        print("--dry-run: inspect the staged dir, then re-run without --dry-run to publish.")
    elif args.create:
        # PRIVATE by default (no --public). The cg engine is NOT in this dataset (it comes from the
        # competition mount), so there's no proprietary-leak risk here -- only our code.
        run(["kaggle", "datasets", "create", "-p", d])
    else:
        run(["kaggle", "datasets", "version", "-p", d, "-m", args.message])


def _render_kernel(user, name, seed, dataset_id, updates, games, extra):
    slug = slug_for(name, seed)
    kdir = os.path.join(_BUILD, "kernels", slug)
    os.makedirs(kdir, exist_ok=True)
    # render run_ppo.py: overwrite the five CONFIG constants, preserving inline comments
    src = open(os.path.join(_HERE, "ppo_kernel", "run_ppo.py")).read()
    consts = {"RUN_NAME": name, "SEED": seed, "UPDATES": updates, "GAMES": games,
              "EXTRA_ARGS": list(extra)}
    for cname, val in consts.items():
        src = re.sub(rf"(?m)^{cname} = [^\n#]*?(\s*#.*)?$",
                     lambda m, v=repr(val): f"{cname} = {v}" + (m.group(1) or ""), src, count=1)
    open(os.path.join(kdir, "run_ppo.py"), "w").write(src)
    # render kernel-metadata.json: id/title per kernel, dataset_sources -> our dataset
    meta = json.load(open(os.path.join(_HERE, "ppo_kernel", "kernel-metadata.json")))
    meta["id"] = f"{user}/{slug}"
    meta["title"] = slug
    meta["dataset_sources"] = [dataset_id]
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
    return slug, kdir


def sweep(args):
    user = kuser()
    dataset_id = f"{user}/{DATASET_SLUG}"
    for seed in args.seeds:
        slug, kdir = _render_kernel(user, args.name, seed, dataset_id,
                                    args.updates, args.games, args.extra)
        print(f"rendered {slug} -> {kdir}  (updates={args.updates} games={args.games} "
              f"extra={args.extra})")
        if not args.dry_run:
            run(["kaggle", "kernels", "push", "-p", kdir])
    if args.dry_run:
        print("\n--dry-run: inspect kaggle/_build/kernels/*/ then re-run without --dry-run to push.")


def status(args):
    user = kuser()
    for seed in args.seeds:
        run(["kaggle", "kernels", "status", f"{user}/{slug_for(args.name, seed)}"])


def pull(args):
    user = kuser()
    for seed in args.seeds:
        slug = slug_for(args.name, seed)
        run(["kaggle", "kernels", "output", f"{user}/{slug}",
             "-p", os.path.join(_OUT, slug), "-o"])


def report(args):
    """Training-curve summary of the pulled metrics jsonls: last-K-update collect win-rate (pooled
    + per-opponent from the per_game records) and the final health diagnostics. Collect win-rate vs
    a fixed opponent is a genuine progress signal, but the health gate is ev_true_pre (holdout
    critic EV) and the VERDICT is local: eval_ppo.py on the pulled _final/_init ckpts, n=200."""
    K = args.last
    found = False
    for seed in args.seeds:
        slug = slug_for(args.name, seed)
        path = os.path.join(_OUT, slug, f"ppo_{args.name}_s{seed}_metrics.jsonl")
        if not os.path.exists(path):
            print(f"{slug}: no metrics jsonl at {path} -- run `pull` first?")
            continue
        found = True
        rows = [json.loads(line) for line in open(path) if line.strip()]
        tail = rows[-K:]
        games = [g for r in tail for g in r.get("per_game", [])]
        wins = sum(1 for _, _, G in games if G > 0)
        by_opp = {}
        for _, opp, G in games:
            w, n = by_opp.get(opp, (0, 0))
            by_opp[opp] = (w + (1 if G > 0 else 0), n + 1)
        last = rows[-1]
        print(f"\n{slug}: {len(rows)} updates completed")
        print(f"  collect win-rate, last {len(tail)} updates: "
              f"{wins}/{len(games)} = {wins / max(len(games), 1):.3f}")
        for opp, (w, n) in sorted(by_opp.items()):
            print(f"    vs {opp:14s} {w}/{n} = {w / n:.3f}")
        print(f"  last update diagnostics: ev_true_pre={last.get('ev_true_pre', float('nan')):+.3f} "
              f"approx_kl={last.get('approx_kl', float('nan')):.4f} "
              f"entropy={last.get('entropy', float('nan')):.3f}")
    if not found:
        sys.exit(f"no metrics found under {_OUT} -- run `pull` first")
    print("\nDefinitive strength read (local, per the eval-discipline skill):")
    print(f"  TORCHDYNAMO_DISABLE=1 python eval_ppo.py --games 200 \\")
    print(f"      --ckpt kaggle/out_kaggle/<slug>/ppo_{args.name}_s<seed>_final.pth \\")
    print(f"      --baseline kaggle/out_kaggle/<slug>/ppo_{args.name}_s<seed>_init.pth")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage", "sweep", "status", "pull", "report"):
        p = sub.add_parser(name)
        if name == "stage":
            p.add_argument("--create", action="store_true", help="first publish (else version-update)")
            p.add_argument("--message", "-m", default="update code")
            p.add_argument("--dry-run", action="store_true", help="stage locally, don't call the kaggle CLI")
        else:
            p.add_argument("--name", default="field", help="experiment name (slug + output prefix)")
            p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
        if name == "sweep":
            p.add_argument("--dry-run", action="store_true", help="render build dirs but don't push")
            p.add_argument("--updates", type=int, default=800, help="PPO updates per kernel (small for a smoke)")
            p.add_argument("--games", type=int, default=16, help="games per update")
            p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                           help="extra train_ppo.py flags, passed verbatim (must be the LAST flag)")
        if name == "report":
            p.add_argument("--last", type=int, default=50, help="updates in the win-rate window")
    args = ap.parse_args()
    {"stage": stage, "sweep": sweep, "status": status, "pull": pull, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
