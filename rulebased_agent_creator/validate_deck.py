#!/usr/bin/env python3
"""Validate a decklist CSV against the competition's allowed card list.

Deck CSV format: a single column of card IDs, one row per copy, no header
(see rulebased_agent_creator/creating_rulebased_agents.md).
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD_DATA_CSV = REPO_ROOT / "data" / "EN_Card_Data.csv"
MAX_COPIES = 4
UNLIMITED_STAGE_TYPE = "Basic Energy"
EXPECTED_DECK_SIZE = 60


def load_card_data(path):
    cards = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cards[row["Card ID"].strip()] = {
                "name": row["Card Name"].strip(),
                "stage_or_type": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip(),
            }
    return cards


def load_deck(path):
    ids = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row:
                continue
            card_id = row[0].strip()
            if card_id:
                ids.append(card_id)
    return ids


def validate(deck_ids, card_data):
    counts = Counter(deck_ids)
    issues = []
    rows = []
    for card_id, count in sorted(counts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        card = card_data.get(card_id)
        if card is None:
            issues.append(f"Unknown card ID {card_id} (not in allowed card list)")
            rows.append((card_id, "<unknown>", count, "INVALID"))
            continue
        limit_ok = card["stage_or_type"] == UNLIMITED_STAGE_TYPE or count <= MAX_COPIES
        if not limit_ok:
            issues.append(
                f"Card {card_id} ({card['name']}) has {count} copies, exceeds max of {MAX_COPIES}"
            )
        rows.append((card_id, card["name"], count, "OK" if limit_ok else "TOO MANY COPIES"))

    if len(deck_ids) != EXPECTED_DECK_SIZE:
        issues.append(f"Deck has {len(deck_ids)} cards, expected {EXPECTED_DECK_SIZE}")

    return rows, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck_csv", help="Path to a decklist CSV (one card ID per row)")
    args = parser.parse_args()

    card_data = load_card_data(CARD_DATA_CSV)
    deck_ids = load_deck(args.deck_csv)
    rows, issues = validate(deck_ids, card_data)

    for card_id, name, count, status in rows:
        print(f"{card_id:>8}  x{count:<3} {name:<40} {status}")

    print()
    if issues:
        print(f"INVALID DECK ({len(issues)} issue(s)):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print(f"Deck is valid ({len(deck_ids)} cards, {len(rows)} unique).")
        sys.exit(0)


if __name__ == "__main__":
    main()
