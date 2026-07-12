# Rule-based agent: `simple_raging_bolt_burn` — implementation & eval report

**Date:** 2026-07-11. **Deliverable:** `game_agents/rulebased/simple_raging_bolt_burn.py` (agent for
`decks/simple_raging_bolt_burn.csv`, the Turbo Raging Bolt ex / Teal Mask Ogerpon ex list), registered
in `eval.py` (`make_raging_bolt`, the `--mode field` pool, the manifest deck-hash dict — NOT the frozen
gauntlet pool) and `record_battle.py` (`raging_bolt` spec).

## What the agent does

Same skeleton as the other four deck agents (score-every-option, sort desc, take top `maxCount`), with
two structural deviations discovered to be necessary:

1. **Count-aware answers** (`_finish(scores, select, want)`): Bellowing Thunder's discard select
   (`SelectContext.DISCARD_ENERGY_CARD`, minCount=0, maxCount=all attached basic energy) is where the
   deck's lethal math lives — `want = ceil(defender_hp / 70)` exactly (weakness ×2), never the template's
   unconditional `[:maxCount]`, which would torch the whole board's energy every attack. Verified in
   traces: 86/86 discard selects chose exactly-lethal or capped-chip counts.
2. **Pair-scored manual attach**: the engine surfaces manual energy attach as `OptionType.ATTACH`
   options, one per (hand energy card × target Pokémon) — energy at `(area=HAND, index)`, target at
   `(inPlayArea, inPlayIndex)`. Scoring only the target (the iono/lucario idiom, safe for their
   mono-energy decks) silently attaches the wrong TYPE in a two-type-cost deck.

Deck-specific policy highlights: answers NO to `SelectContext.IS_FIRST` (goes second, per the guide);
turn-scoped kill plan (`_kill_options`: board-wide fuel vs `ceil(hp/70)` per target, gust targets
included, `ready` covers a promotable armed bench Bolt); doomed-attacker discard inversion (if the
opponent's active can OHKO our attacker, spend ITS energy for the KO and keep the board's); stall
detector (core incomplete + hand can't fix it) that unlocks Lillie's Determination / Burst Roar as
core-diggers; Boss's Orders outranks every supporter when a gustable KO exists; OHKO-aware promotion
(prize liability of the promotee); Meowth benched specifically to fetch Crispin.

## Results (seat-swapped; Wilson 95% CIs)

- **Floor:** vs `random` (sample_deck): 78% at first cut → **96%** (CI 87–99, n=50) after the tempo fixes.
- **Field** (`eval.py --mode field --field-games 200`, deck-confounded — each agent plays its own deck):

| raging_bolt vs | win% | CI |
|---|---|---|
| lucario | 14% | 9–19 |
| abomasnow | 30% | 24–36 |
| dragapult_ex | 26% | 21–33 |
| iono | 16% | 11–21 |

Field context (mean win% across the 5-agent round-robin): lucario 68, dragapult_ex 57, iono 54,
abomasnow 50, **raging_bolt 21**. For scale: iono only takes 24% off lucario — hard counters are normal
in this field.

Iteration trajectory (loss-mining via `record_battle.py` traces, fix, re-run): first attack
turn 9.6 → 4.2; attack cadence 26% → 58% of hero turns; field overall 17% → 21–23% (last two
iterations' CIs overlap on every pairing = stop criterion).

## Why it plateaus (deck-level, not pilot-level)

- Every Pokémon in the list is an ex/Mega: each of our KOs feeds 2–3 prizes; the field's decks all carry
  1-prize chaff. The opponents need 3 KOs; we usually need 4–6.
- The field's aces are 320–350 HP tanks (Mega Lucario 340, Mega Abomasnow 350, Dragapult ex 320) that
  OHKO back: killing one needs 5 discards while sustainable fuel income is ~2–3/turn (Teal Dance + one
  manual attach), and each of our deaths trashes an assembled {L}{F} core that takes 1–2 turns to rebuild
  (only 4 {L} / 4 {F} in the list).
- Remaining non-attacking turns in traces are dominated by the genuine once-per-turn-attach constraint
  (hand holds both core pieces), not decision errors.

Next levers if anyone returns to this: multi-turn fuel planning (pre-arm the backup Bolt before the
active dies — partially there), Fezandipiti/Cruel Arrow bench-snipe lines, and gust-farm sequencing of
1-prize bench chaff (Boss ×2 + Prime Catcher = only 3 gusts per game).

## Repro

```
python game_agents/rulebased/simple_raging_bolt_burn.py          # module self-check
python utils/run_smoketest_battle.py --deck decks/simple_raging_bolt_burn.csv --games 5
python record_battle.py --p0 raging_bolt --p1 random --games 50 --out out/rbb_vs_random.json
python eval.py --mode field --field-games 200 --json-out out/field_rbb_final.json
python eval.py --mode field --smoke                              # harness asserts, 5-agent pool
```
