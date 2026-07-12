# Lillie's Clefairy ex / Mega Kangaskhan ex rule-based agent — build + eval report

**Date:** 2026-07-11.
**Module:** `game_agents/rulebased/lilies_clefairy_ex_mega_kangaskhan_ex.py` (spec `clefairy` /
`lilies_clefairy` in `record_battle.py`; `make_lilies_clefairy` in `eval.py`, added to the
`--mode field` pool + manifest, NOT the frozen gauntlet).
**Deck:** `decks/lilies_clefairy_ex_mega_kangaskhan_ex.csv` — Kowalski's NAIC-2026-winning list
(4 Mega Kangaskhan ex 756 / 4 Meowth ex 1071 / 4 Lillie's Clefairy ex 272 / 3 Latias ex 184 /
2 Wellspring Ogerpon ex 108 / 2 Fezandipiti ex 140 / Moltres 791 / Chien-Pao 209 / Koraidon ex
979 / 4 Crispin / 3 Boss's / 2 Ciphermaniac / Cyrano / 4 Ultra Ball / 4 Dusk Ball / 3 Wondrous
Patch / Prime Catcher / 2 Lillie's Pearl / 4 Area Zero / 4 {P} 2 {W} 2 {F} 1 {R} 1 Telepath).

## Headline results (7-agent field, n=200/pairing seat-swapped, Wilson 95% CIs)

`python eval.py --mode field --field-games 200` (final run: `out/field_clefairy_v2.log/.json`):

| vs | win% (CI) |
|---|---|
| lucario | 45.0 (38–52) |
| abomasnow | 18.0 (13–24) |
| dragapult_ex | 53.5 (47–60) |
| iono | 22.0 (17–28) |
| raging_bolt | 85.0 (79–89) |
| real_raging_bolt | 82.5 (77–87) |
| **pooled** | **51.0 (48.2–53.8), n=1200** |

Field pooled ranking: lucario ~68, dragapult ~64, iono ~62, abomasnow ~58, **lilies_clefairy 51**,
raging_bolt ~26, real_raging_bolt ~21. The new agent lands solidly mid-field — roughly double the
pooled win rate of the two raging-bolt agents, ~9-18 points under the four ace decks (all
matchups deck-confounded per eval-discipline). Floor vs random: 90% (36/4, n=40). Illegal-action
rate 0/53k selects.

## Architecture

Self-contained flat module cloned from `real_raging_bolt_burn.py`: card/attack ID constants,
board census, SelectContext-specific handlers, then a generic MAIN score-and-sort whose score
bands encode within-turn ordering (abilities 50k+ > plays/items/supporters 15k-46k > attach ~8-9k
> retreat 9.8-12k > chip attacks < lethal 140k). Deviations from the reference: a `ctx` dict
replaces the reference's ~25-positional-arg helpers, and `_kill_options`/`_retreat_score` are
generalized to "best payable attack among active + free-retreat-reachable bench" (Skyliner makes
every Basic free) instead of a single-attacker check.

## Engine facts verified during the build (doc-vs-engine inconsistencies flagged in code)

- **Telepath Psychic Energy (19) provides {P} only**; its real effect is an on-attach rider —
  attached from hand to a {P} Pokémon it benches up to 2 Basic {P} Pokémon from the deck via a
  `SelectContext.TO_BENCH` select (new handler; no prior agent had one). The printed "provides
  every type" reading does not apply.
- **Koraidon ex (979) is Tera** — a third Area Zero enabler. Its `all_attack()` metadata is
  shifted: id 1408 IS Orichalcum Fang (50+, {F}{C}, engine auto-applies the +120 was-KO'd rider),
  id 1409 IS Impact Blow (200).
- Lillie's Pearl arrives as a MAIN `OptionType.ATTACH` pair like energy — branch on the card id
  BEFORE reading `energyType`.
- Dusk Ball resolves via `AreaType.LOOKING` + a TO_HAND select; Wondrous Patch via
  ATTACH_TO/ATTACH_FROM (targets restricted to benched {P} = Clefairy/Latias); Prime Catcher is
  two SWITCH selects; Area Zero's bench-shrink is a bench DISCARD select (keep ≥1 Tera or it
  cascades).
- The was-KO'd condition (Orichalcum/Fez) is not in obs — tracked by watching the opponent's
  prize count between our turns.

## Loss-mined fixes that mattered (trace → narrow rule, each re-verified)

1. **Bench starvation** (0-6 loss vs random, Rondo stuck at 20 for 22 turns): the generic
   "bench a body = +20 Rondo" band required Area Zero to be up, and an `opp_big_hitter` veto
   kept all 4 Kangaskhans in hand permanently vs abomasnow. Dropped the stadium requirement and
   the veto on the FIRST Kangaskhan. Largest single gain (abomasnow h2h 15→37% at n=20/30).
2. **Wrong-type energy routing**: non-{P} onto an energyless Clefairy scored above banking
   Kangaskhan's {C}{C}{C}; two {W} died on a Clefairy that never attacked. Now 140 vs 180.
3. **`fund_retreat` on a 3-retreat active**: a stuck Kangaskhan soaked the {P} a dry bench
   Clefairy needed, "funding" a retreat that needed 3 attaches. Now only fires when one attach
   completes the cost.
4. **Ultra Ball never firing**: the `expendable` count (u<=0) was too strict through a 7-turn
   brick; now counts u<=30 cards as pitchable.
5. **Sacrifice ordering**: a dry 2-prize Clefairy was promoted into OHKO decks before 1-prize
   bodies; dry-Clefairy promote base cut 500→220 so the −300/prize OHKO penalty sorts fodder
   by prizes fed.
6. **Eon Blade tank softener**: flat 200 into a 250+HP wall Rondo can't OHKO now outranks a
   thin-bench Rondo chip.
7. **Go first** (guide-vs-measurement inconsistency): the guide says go second for Crispin, but
   every strong local agent elects first and wins 10-30 points more on the play; our A/B trended
   positive (lucario 40→54 at n=40) and the full-field re-run went 49.5→51.0 pooled (CIs
   overlap — kept as neutral-to-positive). `CLEFAIRY_GO_FIRST=0` restores the guide's choice.

## Where the remaining losses live (assessed structural, not misplays)

Abomasnow (18%) and iono (22%) mirror the walls that capped the raging-bolt agents: 280-350HP
aces (Mega Abomasnow ex is 350, not ~300 — above Rondo's cap even at 8-bench) that Full Moon
Rondo (cap ~300, realistically 140-240) never OHKOs, trading 240-damage swings into our 190HP
2-prize attacker. Trace review of those losses shows locally-correct play (Boss correctly
withheld when the active was the same-prize kill, Patch/promote picks right); the deficit is
the prize-trade arithmetic. NOTE: the "hard for everyone" framing was wrong — lucario beats
abomasnow 53% and iono 77%, dragapult 54.5%/66% — which motivated the v3 investigation below.

## v3 investigation: are abomasnow/iono structural or fixable? (2026-07-11, follow-up)

Loss-mined both matchups (30-game trace sessions, then 100-150-game h2h iterations), tested
three engine-verified levers, kept what didn't regress. **Verdict: predominantly structural at
the deck level; the levers are real but their measured effect is within CI noise.**

What was found and shipped (all in the agent module, grep the symbols):

1. **`_est_attack_dmg` + real-damage `_opp_can_ohko`.** The old promote penalty read BASE
   attack damage, and both killers' real attacks are base 0-20 in the metadata (Hammer-lanche
   0 → real ~300; Voltaic Chain 20 → real 200-400; Riptide 0). Mining showed ~half the prizes
   fed in losses were promotes into an already-real-lethal active. The estimator parses the
   "damage for each" text (energy-attached / discard-pile / benched / in-play / coin-flip /
   hidden→x3 tiers). CAUTION (measured): keep it ACTIVE-only — a bench-and-Streamer-aware
   worst-case version penalized every promote equally and collapsed iono h2h 25%→12% (n=100);
   a dry 300HP Kangaskhan promoted into a not-yet-ramped board is a WALL that buys turns, not
   a misplay.
2. **`koraidon_lever`** (board-conditional, no per-opponent branch): a fighting-weak wall
   (printed HP >= 250 — Bellibolt ex 280, opposing Mega Kangaskhan 300) on their board flips
   {F}/Koraidon routing: energy-target 940, Crispin {F} fetch/attach bands, search picks above
   the 2nd Clefairy, bench-play band 21800. Iono's Bellibolt ex IS fighting-weak and Impact
   Blow (200x2=400) OHKOs it — but with 1 Koraidon + 2 {F} in 60 cards the lever arms rarely
   (benched 39% of games after the boosts, 2{F} almost never; Orichalcum's 1-{F} 340-after-a-KO
   line is the practical arm). Koraidon-benched games won 31% vs 22% without (n=150, noise-level).
3. **Bench-out win valuation**: KOing their active with an EMPTY bench ends the game — it's our
   top win mode vs abomasnow (bench-out was 17-24 of every ~25 wins) and was invisible to the
   lethal bands (`_attack_score`/`_kill_options` `opp_bench_empty`, and `gust_worth` no longer
   gusts away a game-winning active kill).
4. **Gust tiebreak**: equal-prize gust targets break toward the energized engine piece
   (`200*len(energies) + hp//10`) — a fueled Kyogre (the only Hammer-lanche recycler, alive at
   the end of 20/22 mined losses) over an empty Snover; mirrored in `_kill_options` so the
   worth-it decision and the target pick can't disagree.

Full-field v3 vs v2 (n=200/pairing, every clefairy pairing's Wilson CIs overlap — nothing
regressed, nothing provably gained): lucario 45.0→47.5, abomasnow 18.0→21.0, dragapult
53.5→54.0, iono 22.0→22.5, raging_bolt 85.0→81.0, real_raging_bolt 82.5→86.0, **pooled
51.0→52.0 (49.2-54.8)**. Pooled current-code targeted sessions: abomasnow 19.7% (n=350), iono
24.0% (n=350) — both overlap their anchors.

Why it stays structural with THIS 60: abomasnow's ace out-HPs Rondo's ceiling behind a
worst-cased −30 shield while Hammer-lanche (~E300+) OHKOs every body we own, and our deck
feeds 2-3 prizes per KO with only two 1-prize bodies; iono's wincon is a 70HP 1-prize attacker
with an uncapped attack plus recursion, so both the prize trade and the attrition race are
sold before play begins. The winning lines that remain (bench-out denial, Koraidon-on-Bellibolt)
are real but low-frequency. A deck-level change (2nd Koraidon / 3rd-4th {F}, or Metal tech for
Mega Abomasnow's weakness) is the untested lever left — out of scope while we play Kowalski's
list verbatim.

## Repro

```
python game_agents/rulebased/lilies_clefairy_ex_mega_kangaskhan_ex.py     # module self-check
python rulebased_agent_creator/validate_deck.py decks/lilies_clefairy_ex_mega_kangaskhan_ex.csv
python utils/run_smoketest_battle.py --deck decks/lilies_clefairy_ex_mega_kangaskhan_ex.csv --games 5
python record_battle.py --p0 clefairy --p1 random --games 40 --out out/floor.json
python eval.py --mode field --field-games 200 --json-out out/field.json
```
