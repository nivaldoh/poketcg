---
name: tcg-rules-critic
description: >-
  Check Pokémon TCG game-logic reasoning against the rulebook before trusting it.
  Use whenever correctness depends on the rules — deck legality, evolution chains,
  energy/retreat costs, prize/KO logic, status conditions, determinization
  assumptions, or when designing heuristics, features, or eval claims. Pulls only
  the relevant rulebook section into context instead of guessing at a rule.
---

# tcg-rules-critic

Verify a rules claim against the **actual rulebook** instead of guessing, then return a
short rules-consistency verdict. `docs/rulebook.md` (~1.6k lines) is too big to hold in
context, so navigate it on demand and read only the section that matters.

## Ground truth (read this first)
1. The compiled **`cg` engine is ground truth** wherever it conflicts with `docs/rulebook.md` —
   the engine is what the competition actually scores; the rulebook is the human reference.
   When they disagree, say so and trust the engine (e.g. the engine's `EnergyType` enum has
   no Fairy even though the rulebook lists one; the engine decides which options are legal).
2. **Card facts come from `all_card_data()`**, never guessed from a card ID. Resolve a
   name / type / HP / attack / flags with the SDK, e.g.
   `python -c "import tcg_model as t; c={x.cardId:x for x in t.all_card}; k=c[678]; print(k.name, k.cardType, k.hp, getattr(k,'megaEx',None))"`
   (this loads the vendored `cg`; see `CLAUDE.md` for the path wiring). Memory
   `poketcg-card-ids-and-decks` lists a few known IDs as a convenience, not as the source.

## Recipe
1. **Dump the map** — list every section, appendix, and glossary heading:
   `grep -n '^## \|^### ' docs/rulebook.md`
2. **Jump to the relevant part** — read the line range under the matching heading (e.g.
   "Parts of a Turn", "Deck Building", "Special Conditions", "Appendix 1: Mega Evolution
   Pokémon ex", or a Glossary entry). Read the section; don't infer from the title.
3. **Cross-check the claim** against what the section actually says.
4. **If the claim touches a specific card**, confirm its real data via `all_card_data()`.
5. **If the rulebook and the engine could differ** (an energy type, a first-turn restriction,
   a prize count, which options are legal), note the discrepancy and defer to the engine.

## Verdict format
Return **consistent / inconsistent / unsupported**, the rulebook heading(s) you read, and the
corrected statement if the claim was wrong. Cite the heading, not a line number — lines drift,
headings don't.

## Single source of truth + known duplicates
- Canonical rules: **`docs/rulebook.md`** (extracted by `extract_rulebook.py` from `rulebook.pdf`).
- Deck-legality is also paraphrased in `CLAUDE.md`; card IDs / decks also live in memory
  `poketcg-card-ids-and-decks`. Those are copies — reconcile them against `docs/rulebook.md` +
  `all_card_data()`; don't treat a copy as authoritative.

## Drift-resilience
Point at the rulebook, don't restate its prose here. Reference code symbols by name + a grep
recipe (never pinned line numbers). Discover paths and card facts at runtime. Confirm a heading
or symbol still exists before relying on it.

## Where this fits
This is the rules check in the repo's verify-loop (`docs/skills_and_hooks.md` → "How these compose"):
author/debug an agent under `cg-agent-contract` → run the `run_battle.py` smoke assertion → read
head-to-head numbers under `eval-discipline` → package with `kaggle-submission` →
**run this skill wherever a step depends on the rules**. All four skills now exist (see
`docs/skills_and_hooks.md` for the index).
