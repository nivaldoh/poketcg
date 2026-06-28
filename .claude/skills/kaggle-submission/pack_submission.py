#!/usr/bin/env python3
"""Assemble + validate a Kaggle `submission.tar.gz` for the `pokemon-tcg-ai-battle` competition.

This script is the **executable single source of truth** for the submission layout: a valid
bundle is exactly three top-level entries — `main.py`, `cg/`, `deck.csv` — packed with no nested
prefix. The same fact is *prose-duplicated* in `README.md`, `CLAUDE.md`, and the Kaggle starter
`starter_rulebased_lucario_kaggle.py` (its `tarfile` block). Those are copies; reconcile them
against this script's assertions rather than re-deriving the layout by hand.

It does NOT submit. Submitting is a separate, explicit step (see the skill's SKILL.md).

Usage (run from the repo root):
    python .claude/skills/kaggle-submission/pack_submission.py --main path/to/main.py
    # validate a deck without building (no main.py needed):
    python .claude/skills/kaggle-submission/pack_submission.py --check-deck-only --deck deck.csv

Paths are discovered/defaulted at runtime, never hardcoded:
  --main   path to the agent entrypoint (main.py); required unless --check-deck-only
  --cg     the `cg` package dir to bundle   [default: ${CG_LIB_PATH:-<repo>/vendor/cg-lib}/cg]
  --deck   the 60-card deck list (CSV/one id per line)            [default: <repo>/deck.csv]
  --out    output tarball                                  [default: <repo>/submission.tar.gz]
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from collections import Counter


def _find_repo_root(start: str) -> str:
    """Walk up from `start` to the dir that holds `tcg_model.py` (the repo root)."""
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "tcg_model.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(start)  # fall back; let later checks report what's missing
        d = parent


REPO_ROOT = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CG_LIB = os.environ.get("CG_LIB_PATH", os.path.join(REPO_ROOT, "vendor", "cg-lib"))


def _read_deck_ids(deck_path: str) -> list[int]:
    """Read one card id per line, skipping blanks and any non-integer header lines."""
    ids: list[int] = []
    with open(deck_path) as f:
        for line in f:
            tok = line.strip().split(",")[0].strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                continue  # tolerate a header row
    return ids


def validate_deck(deck_path: str, cg_pkg_dir: str) -> list[str]:
    """Return a list of legality errors (empty == legal). Card facts come from the cg engine
    (`all_card_data()`), the ground truth — never guessed from ids. Rules enforced:
    60 cards / >=1 Basic Pokemon / <=1 ACE SPEC / <=4 copies of any non-(basic-energy) card."""
    errors: list[str] = []
    if not os.path.isfile(deck_path):
        return [f"deck not found: {deck_path}"]

    # Import the engine from the parent of the cg package dir (mirrors run_battle.py's wiring).
    cg_lib = os.path.dirname(os.path.abspath(cg_pkg_dir))
    if cg_lib not in sys.path:
        sys.path.insert(0, cg_lib)
    try:
        from cg.api import CardType, all_card_data
    except Exception as e:  # noqa: BLE001 - report cleanly instead of a traceback
        return [f"could not import the `cg` engine from {cg_lib!r} to check legality: {e}. "
                "Restore vendor/cg-lib/cg or set CG_LIB_PATH (see docs/cg-sdk.md)."]

    table = {c.cardId: c for c in all_card_data()}
    ids = _read_deck_ids(deck_path)

    if len(ids) != 60:
        errors.append(f"deck has {len(ids)} cards, must be exactly 60")

    unknown = sorted({i for i in ids if i not in table})
    if unknown:
        errors.append(f"unknown card id(s) not in all_card_data(): {unknown}")

    known = [i for i in ids if i in table]
    counts = Counter(known)
    for cid, n in sorted(counts.items()):
        if n > 4 and table[cid].cardType != CardType.BASIC_ENERGY:
            errors.append(f"card {cid} ({table[cid].name}) appears {n} times, max 4 "
                          "(only Basic Energy is exempt)")

    basics = [i for i in known
              if table[i].cardType == CardType.POKEMON and getattr(table[i], "basic", False)]
    if not basics:
        errors.append("deck has no Basic Pokemon; at least one is required")

    aces = sorted({i for i in known if getattr(table[i], "aceSpec", False)})
    if len(aces) > 1:
        errors.append(f"deck has {len(aces)} ACE SPEC cards {aces}, max 1")

    return errors


def build_tarball(main_path: str, cg_pkg_dir: str, deck_path: str, out_path: str) -> None:
    """Write `out_path` with exactly top-level main.py + cg/ + deck.csv, then assert the layout."""
    for label, p, want_dir in (("--main", main_path, False),
                               ("--cg", cg_pkg_dir, True),
                               ("--deck", deck_path, False)):
        if want_dir and not os.path.isdir(p):
            raise SystemExit(f"{label} directory not found: {p}")
        if not want_dir and not os.path.isfile(p):
            raise SystemExit(f"{label} file not found: {p}")

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(main_path, arcname="main.py")
        tar.add(cg_pkg_dir, arcname="cg")
        tar.add(deck_path, arcname="deck.csv")

    # Assert top-level layout: no absolute/parent paths, and all three entries present at root.
    with tarfile.open(out_path, "r:gz") as tar:
        names = tar.getnames()
    bad = [n for n in names if n.startswith("/") or n.startswith("..") or n == ""]
    if bad:
        raise SystemExit(f"submission has non-top-level paths: {bad}")
    roots = {n.split("/", 1)[0] for n in names}
    missing = {"main.py", "cg", "deck.csv"} - roots
    if missing:
        raise SystemExit(f"submission is missing top-level entries: {sorted(missing)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", help="path to the agent entrypoint main.py "
                                   "(required unless --check-deck-only)")
    ap.add_argument("--cg", default=os.path.join(DEFAULT_CG_LIB, "cg"),
                    help="the cg package directory to bundle")
    ap.add_argument("--deck", default=os.path.join(REPO_ROOT, "deck.csv"),
                    help="60-card deck list")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "submission.tar.gz"),
                    help="output tarball")
    ap.add_argument("--check-deck-only", action="store_true",
                    help="validate the deck and exit; do not require main.py or build")
    args = ap.parse_args()

    errors = validate_deck(args.deck, args.cg)
    if errors:
        print("DECK INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"deck OK: {args.deck} (60 cards, legal)")

    if args.check_deck_only:
        return 0

    if not args.main:
        ap.error("--main is required unless --check-deck-only is given")
    build_tarball(args.main, args.cg, args.deck, args.out)
    print(f"built {args.out} (top-level main.py + cg/ + deck.csv).")
    print("NOT submitted. To submit (only when explicitly asked), see the kaggle-submission skill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
