---
name: cg-agent-contract
description: >-
  Write or modify a cg agent() without tripping the contract. Use when authoring or debugging an
  agent, a deck, or option-handling code: returning option indices (not card IDs), honoring
  minCount/maxCount, the first-call 60-card deck, opponent-zone visibility, and determinization
  assumptions. Forces the contract checklist and the run_battle.py smoke assertion.
---

# cg-agent-contract

Author/debug a cg `agent()` against the contract instead of from memory. The canonical prose is
`docs/cg-sdk.md` — this skill is the forcing trigger + the consolidated gotchas, not a copy of it.

## Read the contract first
`docs/cg-sdk.md` is short and navigable by heading: `grep -n '^#\{1,4\} ' docs/cg-sdk.md` →
"Agent contract", "Observation structure", "Options are typed", "Game driver loop",
"Search API / MCTS". Read the section you're touching; the engine under `vendor/cg-lib/cg/` is
ground truth where any prose disagrees.

## Contract checklist
- **Return option *indices*, never card IDs.** The agent returns a list of indices into
  `obs.select.option`. Resolve an option back to a card with the `get_card()` resolver in
  `tcg_model.py` (locate it: `grep -n 'def get_card' tcg_model.py`) — option index ≠ card id.
- **Honor `minCount`/`maxCount`** on `obs.select`: the returned list length must be in
  `[minCount, maxCount]`, with **no duplicate indices**.
- **First call returns a 60-card legal deck** (Kaggle flow only): when `obs.select is None`, return
  a deck, not option indices. Deck legality (60 / ≤4 per non-Basic-Energy / ≥1 Basic / ≤1 ACE SPEC)
  belongs to `tcg-rules-critic` + `docs/rulebook.md` and is asserted by `kaggle-submission`'s packer —
  don't re-derive it here.
- **Opponent zones are counts-only, not visible cards.** Don't read hidden opponent cards from the
  observation; treat them as counts.
- **Determinization is not free for asymmetric decks.** MCTS determinizes hidden cards (see the
  Search API section); a mirror (same deck both sides) is the clean case — don't assume the same
  assumptions hold when the two decks differ.

## Smoke test every change (the repo convention)
Ship the smallest "engine still runs" signal with any agent edit: the `play_one` driver loop and its
terminal assertion in `run_battle.py` (locate: `grep -n 'def play_one\|assert result' run_battle.py`,
which asserts `result in (0, 1, 2)`). Run `python run_battle.py [--games N]` after a change.

## Single source of truth + known duplicates
- Canonical contract prose: **`docs/cg-sdk.md`** (read from the vendored source; already referenced
  by `CLAUDE.md`). The deck-legality rule is paraphrased in `CLAUDE.md` and lives canonically in
  `docs/rulebook.md`/`tcg-rules-critic`. Point at those; don't add another copy here.

## Drift-resilience
Point at `docs/cg-sdk.md`, don't restate it. Reference symbols by name + a grep recipe (never line
numbers). Discover paths at runtime (`CG_LIB_PATH`). The compiled `cg` engine and `all_card_data()`
are ground truth over any doc; confirm a symbol/heading still exists before relying on it.

## Where this fits
This is the **author/debug** step of the repo's verify-loop (`docs/skills_and_hooks.md` → "How these
compose"): **write the agent here** → run the `run_battle.py` smoke assertion → interpret any
head-to-head numbers under `eval-discipline` → package with `kaggle-submission` only at the end.
Use `tcg-rules-critic` wherever a step depends on the rules. Heed the "what does NOT help" guardrails
(self-play diverges; the value head is miscalibrated on the mirror; the AMAGO/Metamon port was
abandoned — memory `tcgmon-port-plan`) before reaching for a ruled-out approach.
