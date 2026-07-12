# real_raging_bolt: the post-rotation Raging Bolt burn agent

**What**: `game_agents/rulebased/real_raging_bolt_burn.py` + `decks/real_raging_bolt_burn.csv` —
the "real" (JRobrueda post-rotation) Raging Bolt ex list with the Crispin / Glass Trumpet /
Energy Switch energy network, built per the strategy guide embedded as the comment block at the
top of the agent file. This is the complex sibling of `simple_raging_bolt_burn.py` (eval spec
`raging_bolt`; report: `docs/reports/raging_bolt_agent_report.md`).

**Specs**: eval spec `real_raging_bolt` (`eval.py make_real_raging_bolt`, in the `--mode field`
pool + manifest; `record_battle.py --p0 real_raging_bolt`). Deck: 60 cards, 26 unique
(`python rulebased_agent_creator/validate_deck.py decks/real_raging_bolt_burn.csv` passes;
2× Raging Bolt ex, 3× Teal Mask Ogerpon ex, 3× Mega Kangaskhan ex, 3× Meowth ex, 2× Latias ex,
1 each Wellspring Ogerpon / Fezandipiti / Lillie's Clefairy / Passimian / Chien-Pao; only
2 {L} + 2 {F} — core-energy scarcity is this list's defining constraint vs the simple deck's 4/4).

## Final results (2026-07-11, seat-swapped, Wilson 95% CIs)

Floor: **91.0% (83.8–95.2, n=100)** vs `random`
(`record_battle.py --p0 real_raging_bolt --p1 random --games 100`). Simple agent's floor was 96%.

Field (`python eval.py --mode field --field-games 200 --json-out out/field_real_rbb_final.json`,
6-agent pool, DECK-CONFOUNDED — mixes pilot skill with the deck matchup):

| real_raging_bolt vs | win% | CI | simple raging_bolt, same run | CI |
|---|---|---|---|---|
| lucario | 17.0% | 12–23 | 15.0% | 11–21 |
| abomasnow | 30.0% | 24–37 | 23.0% | 18–29 |
| dragapult_ex | 23.0% | 18–29 | 19.0% | 14–25 |
| iono | 16.0% | 12–22 | 16.5% | 12–22 |
| each other (head-to-head) | **46.0%** | 39–53 | 54.0% | 47–61 |

Pooled overall (all 5 pairings): **real 26.4% (23.8–29.2)** vs **simple 25.5% (22.9–28.3)**,
n=1000 decisive each. Illegal actions: 0/54,463.

**Verdict**: the real agent lands at parity-or-slightly-ahead of the simple agent — ahead on
every non-mirror point estimate except iono, behind on the head-to-head point estimate — but
**every pairing's CIs overlap**, so the honest read is a tie. Note the simple agent's own
numbers moved between runs (its original report: lucario 14 (9–19), abomasnow 30 (24–36),
dragapult 26 (21–33), iono 16 (11–21)); cross-run point-estimate deltas of ±5 are noise at
n=200. The 4-opponent mean is 21.5% for real vs 18.4% for simple in this run (old report: 21%).

## What the agent does beyond the simple template

Ported the `simple_raging_bolt_burn.py` architecture (per-`SelectContext` handlers,
`_finish(want=…)` count-clamped returns, pair-scored manual attach, `pending_attach_type`
handshake, `_kill_options` turn plan) and added, grounded in the engine source
(`ptcg_engine/ptcgProgram 22/CardImpl.h`):

- **`ACTIVATE` handler** (new): Chien-Pao's Snow Sink says YES only to an opponent's stadium —
  the generic YES fallthrough would trash our own Area Zero; Torrential Pump's shuffle-3 rider
  fires only when it converts a bench KO.
- **`DAMAGE` handler** (new): Cruel Arrow (100 anywhere) / Pump bench snipe (120) take the
  most-prizes KO, else soften the biggest piece toward a cheaper Bellowing (70-multiple math).
- **`TO_DECK`** (Ciphermaniac stack-2) + **`TO_DECK_ENERGY`/`TO_HAND_ENERGY`** handlers.
- **Per-type energy census** (`deck_energy_by_id`) and a **`core_bricked` detector** ({L}/{F}
  gone from every reachable zone → pivot: pile any energy on Kangaskhan, Rapid-Fire {C}{C}{C} 200
  becomes the win condition; retreat ping-pong toward an unarmable Bolt disabled).
- **Fairy Zone damage model**: live weakness is not in the obs — `_weak_mult`/`_attack_damage`
  rewrite opponent Dragons' weakness to {P} when Clefairy is in play (Full Moon Rondo/Eon Blade
  doubling), and `_k_needed` knows Fairy Zone can only *remove* a Dragon weakness for Bellowing.
- **`_damage_shield`**: data-driven "takes N less damage" scan of the defender's payable attacks
  (Mega Abomasnow's Frost Barrier) padded into every KO calculation.
- **Draw-engine management**: this list has NO hand-reset supporter — Run Errand IS the draw.
  Cyrano fetches Mega Kangaskhan (confirmed offered despite `ex=False, megaEx=True`), the bench
  gate weighs its 3-prize liability against dead-board risk, and free (Skyliner) retreats rotate
  it through the Active Spot with a deck-count guard.
- **Matchup gates**: `opp_big_hitter` (any ≥150-base attack on their board, e.g. Phantom Dive)
  blocks benching/promoting Kangaskhan where it's a free 3-prize 2HKO; Clefairy/Chien-Pao/
  Passimian bench only when their plan is live (Dragons in play / opponent stadium / late-game
  1-prize walls).

## Iteration log (loss-mining vs `record_battle.py` traces, n=100 sessions)

| iteration | change (from mined evidence) | key metric |
|---|---|---|
| 0 | initial port | 90% vs random but **first attack turn ~6.5**, turn-20 Bellowings in losses |
| 1 | pre-Bolt core banking (`core_missing` fallback when no Bolt in play) + Bolt-finder priority (Cyrano 26.5k / Ultra Ball 46k / Meowth→Cyrano when Bolt absent) | first attack 6.5→5.4 |
| 2 | draw-engine access (Cyrano fetches Kangaskhan; bench gate; Run-Errand rotation; armed-secondary promotion aligned with the retreat trigger); brick pivot + Burst Roar deck guard | mirror vs simple 39%→46%; **dragapult regressed 27→16** (benched Kangaskhan = Phantom Dive food) |
| 3 | `opp_big_hitter` gate on Kangaskhan (bench + search picks) and `_damage_shield` Frost Barrier pad | dragapult 16→20, abomasnow 22→27, mirror 47% |
| final | Kangaskhan promotion penalty vs big hitters | field table above |

Stop criterion reached: across the last two iterations every pairing's CIs overlap and the
trace review shows no recurring decision-error class — remaining losses are the deck-structural
ones the simple agent's report already diagnosed (every KO we give up feeds 2–3 prizes; the
field's 320–350hp aces need k=5 discards vs ~2–3/turn sustainable fuel income; attack cadence
plateaus ~0.5 of hero turns), now sharpened by this list's 2{L}/2{F} (prized core = brick, ~2%
of games vs random end 0-attack).

## What did NOT pay (tried/considered and bounded)

- Ciphermaniac is gated to stalled-with-controlled-draw and almost never fires — without a
  same-turn Run Errand/Teal Dance the next shuffle eats the stack; Crispin dominates otherwise.
- Wellspring Pump / Passimian / Clefairy lines assemble rarely (1 {W}, 2 {P}); they're kept as
  cheap opportunistic branches (the `DAMAGE`/`ACTIVATE` handlers), not the plan.
- 0-discard Bellowings still occur when the board is fuel-dry — harmless (an attack-shaped END)
  but a visible symptom of the income ceiling, not a scoring bug.

## Repro

```
python game_agents/rulebased/real_raging_bolt_burn.py                      # self-check
python rulebased_agent_creator/validate_deck.py decks/real_raging_bolt_burn.csv
python utils/run_smoketest_battle.py --deck decks/real_raging_bolt_burn.csv --games 5
python record_battle.py --p0 real_raging_bolt --p1 random --games 100 --out out/floor.json
python eval.py --mode field --field-games 200 --json-out out/field.json
```
