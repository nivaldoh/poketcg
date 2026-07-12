# The heuristic agent factory (design doc)

Status: design, revised 2026-07-11. Nothing here is implemented yet except where a repo file is
named explicitly. Companion docs: `creating_rulebased_agents.md` (the manual authoring guide this
factory automates), `docs/cg-sdk.md` (agent contract ground truth), the per-agent build reports in
`docs/reports/` (`raging_bolt_agent_report.md`, `real_raging_bolt_agent_report.md`,
`lilies_clefairy_agent_report.md` — the manual version of the loop we are automating).

We do **not** need an evolutionary league, PSRO, Nash mixtures, exploiters, or sophisticated
population management yet. We need a **deck-specific heuristic-agent generator with strong
mechanical verification**.

The goal is not:

> Discover strategically optimal agents.

It is:

> Produce many agents that reliably execute each deck's basic game plan, make mostly sensible
> legal decisions, and generate trajectories that are better than random self-play — i.e. more
> teachers for the BC warm-start pipeline (`bc_collect.py` → `bc_train.py` → `train_sb3.py
> --bc-init`).

That is much more achievable. We already built seven such agents by hand
(`game_agents/rulebased/`); the factory's job is to industrialize that process, not invent a new
kind of agent. We need to make "the process of creating a rule-based agent" easier and better.

## Assumptions

* Each generated agent plays one fixed 60-card deck, stored as `decks/<name>.csv` (one card ID per
  line, copies repeated; validate with `rulebased_agent_creator/validate_deck.py`).
* The engine exposes legal options and the player-visible state through the cg `obs` dict; the
  agent contract is exactly `agent(obs_dict) -> list[int]` of option indices (details in
  "The contract", below).
* We can construct and replay specific game states **locally** via the modified engine
  (`cg_state.py`: `battle_start_seeded`, `save_state`, `battle_start_from_state`; recorded with
  `record_battle.py --seed --record-state`, resumed with `utils/resume_battle.py`). The stock
  Kaggle `libcg.so` lacks this API — scenario tooling is local-only by construction.
* The downstream model improves later through RL or search; the heuristic agents only need to be a
  useful initialization. Teachers are **local-only** artifacts. They are not Kaggle submissions,
  so they may import a shared framework module. (If a teacher ever needs to be submitted, a
  bundling step that inlines the framework into a single `main.py` is a separate, later task —
  see "Repo layout".)
* We care more about broad deck coverage than squeezing the last few percentage points from each
  heuristic. Two hand-built data points say the last percentage points are often not reachable
  anyway: for abomasnow/iono and for clefairy, iterated fixes landed inside CI noise and the
  residual deficit was judged **deck-structural**, not pilot error (see
  `docs/reports/lilies_clefairy_agent_report.md`). Acceptance criteria below are designed around
  that fact.

Under those assumptions, the factory has five components.

---

# 0. The contract and the engine-quirks ledger (prerequisite knowledge)

Everything the factory generates must satisfy the cg contract (full reference: `docs/cg-sdk.md`;
checklist: the `cg-agent-contract` skill):

* `agent(obs_dict) -> list[int]` returns **option indices** into `obs.select.option` — never card
  IDs — with no duplicates and length in `[select.minCount, select.maxCount]`.
* On the very first call `obs.select is None` and the agent must return its 60-card deck
  (`my_deck`). This call is also where per-game module state must be reset (agents are module
  singletons reused across games by eval/bc_collect; a leaked flag from game N corrupts game N+1).
* Many selects are **variable-count**: taking `maxCount` is often wrong (e.g. Bellowing Thunder's
  `DISCARD_ENERGY_CARD` offers all your energy; the right answer is "exactly enough for the KO").
  The established pattern is the burn-family `_finish(scores, select, want)` clamp — sort by
  score, take `want` clamped to `[minCount, maxCount]` (grep `_finish` in
  `game_agents/rulebased/simple_raging_bolt_burn.py`).

The factory must also bake in the engine quirks we have already paid to discover. These are
verified behaviors, each handled somewhere in the existing agents (grep the named symbol). This
ledger is a **living framework artifact**: the shared framework should encode each item once, and
the repair loop's checklist should reference it.

| Quirk | Where handled today |
|---|---|
| Manual ATTACH options are (hand-energy × target) **pairs** — score the pair, not the card | every agent's `OptionType.ATTACH` branch; pair-scorers `attach_score` (dragapult_ex), `_energy_target_score` (lilies) |
| Two-step energy selects: ATTACH_TO / SWITCH_ENERGY_CARD pick the energy, the following ATTACH_FROM picks the recipient — needs cross-select state | `pending_attach_type` module global, burn family |
| `select.effect` / `select.contextCard` identify the card causing the select — branch search/discard scoring on it | `_search_pick_score` (burn family), `attach_score` (dragapult_ex) |
| `IS_FIRST` is a choosable select; going **first** measurably beats the "go second" guide advice | IS_FIRST blocks in the burn family; finding in `docs/reports/lilies_clefairy_agent_report.md` |
| `ACTIVATE` (stadium/ability yes-no) will happily trash your **own** stadium — answer NO unless the standing stadium is the opponent's | ACTIVATE handler in `real_raging_bolt_burn.py` / lilies |
| Effect snipes arrive as `SelectContext.DAMAGE` (and spread counters as DAMAGE_COUNTER / DAMAGE_COUNTER_ANY) | burn-family DAMAGE block; dragapult's counter placement |
| Some `all_attack()` rows are mislabeled (Koraidon ids 1408/1409 swapped) — the engine's behavior, not the row text, is ground truth | `ATK_*` constants + comment in lilies |
| Variable-damage attacks read as base 0 in metadata — threat estimates must scale from effect text or accept undercount | `_est_attack_dmg` (lilies), `_opp_can_ohko` |
| "Takes N less damage" shields are not observable — assume active when payable, pad KO math | `_damage_shield` (burn family) |
| No "was KO'd last turn" obs flag — infer from prize-count delta or MOVE_CARD logs | `was_koed`/`opp_prize_prev` (lilies), `pre_ko` (dragapult_ex) |
| Weakness rewrites (Fairy Zone) have no obs field — recompute in code | `_weak_mult` (lilies) |
| Opponent-imposed "choose what you lose" selects need **inverted** scoring | Crispin second-pick / DISCARD handlers (`hand_score(..., True)`, `-hand_usefulness`) |

When engine behavior contradicts a doc or card text (as with Koraidon's attack rows or Telepath's
printed wording), treat it as a documented inconsistency: trust the engine, leave a code comment,
and add the case to this ledger.

---

# 1. A standard deck-strategy specification

We should not have to infer everything directly from raw card text each time, so we will represent every deck using a structured strategy file, `strategy.yaml`, keyed by **card ID** (names
are ambiguous and reprints exist; ids are the engine's ground truth — never guess a name from an
id or vice versa, resolve through the card table):

```yaml
deck: simple_raging_bolt_burn        # basename of decks/<name>.csv
deck_display_name: Raging Bolt ex / Teal Mask Ogerpon ex

primary_attackers: [{id: <ID>, name: Raging Bolt ex}]
support_attackers: [{id: <ID>, name: Teal Mask Ogerpon ex}]
# <ID> = placeholders; the generator fills real ids from the deck CSV + card table.
# Never hand-type ids from memory — id→name guesses have burned us before.

setup_goals:
  turn_1:
    - establish 2+ Ogerpon
    - attach energy
    - find draw support
  turn_2:
    - prepare Raging Bolt
    - reach required damage threshold

resource_roles:
  - id: <ID>           # Earthen Vessel
    purpose: find energy
    preserve_if: [hand already lacks discardable cards]
  - id: <ID>           # Energy Retrieval
    purpose: recover energy for damage scaling

strategic_rules:
  - avoid benching unnecessary two-prize Pokémon
  - preserve enough energy recovery for the late game
  - prefer knockouts that improve the prize race

failure_conditions:
  - no attacker prepared by turn 2
  - discarding all copies of a critical recovery card

quirk_flags:            # which ledger entries this deck exercises
  - two_step_energy_selects
  - variable_damage_attack
```

This is the intermediate representation between the deck list, the card definitions, strategy
knowledge (human or generated), and the executable policy. The generator produces it first from
four inputs — `decks/<name>.csv`, piloting/strategy guide for the deck, `data/EN_Card_Data.csv` (human-readable effect text), and
`cg.all_card_data()` (machine ground truth for types/stages/ex flags; **the CSV describes, the
engine decides — on conflict, the engine wins**) — then the spec is reviewed/tested, and only then
is code generated from it.

The spec itself gets a mechanical lint before any code generation:

* every card in the deck CSV appears in exactly one role (attacker / setup / draw / search /
  energy accel / switching / recovery / disruption / energy);
* every id referenced in the spec is actually in the deck;
* the deck passes `validate_deck.py`;
* every `quirk_flags` entry names a ledger item.

---

# 2. A reusable heuristic policy framework

Most deck agents should share the same engine; only deck-specific scoring and priorities change.

We do not need to design this framework from scratch — it should be **extracted from the existing
agents**, which already converged on a shape. Two generations exist in `game_agents/rulebased/`:

* the classic family (`lucario`, `abomasnow`, `iono`, `dragapult_ex`): one monolithic option loop
  branching on `OptionType`;
* the burn family (`simple_raging_bolt_burn`, `real_raging_bolt_burn`,
  `lilies_clefairy_ex_mega_kangaskhan_ex`): dispatch on `SelectContext` first (IS_FIRST, ACTIVATE,
  DAMAGE, TO_HAND, ATTACH_FROM, …), each handler building a `scores` vector and returning through
  `_finish`, with a generic MAIN loop as the fallthrough.

**The framework adopts the burn-family shape** (context-primary dispatch + `_finish` clamping) —
it is the most evolved, and it is the only shape that handles variable-count selects and
minCount correctly (the classic family always takes `maxCount`; only dragapult drops
negative-scored options).

What gets factored into the shared framework (all of this is currently copy-pasted, in several
cases byte-identical, across the seven modules — grep any of these names across
`game_agents/rulebased/` to see the duplication):

* Contract plumbing: the `to_observation_class` prologue, the `obs.select is None → my_deck`
  first-call branch **including per-game state reset**, deck-CSV loading (repo `decks/` →
  co-located → `deck.csv` → Kaggle path fallback chain), `all_card_data()` /
  card-table / attack-table setup.
* Option resolution: `get_card(obs, area, index, player_index)` — verbatim ×7 today, and
  re-implemented an eighth time as `sb3_features._resolve_option`.
* Board census: `field_counts` / `hand_counts` / `discard_counts` / `attached_counts` /
  `deck_energy_by_id` construction.
* Generic evaluation: `prize_count`, `pokemon_score`, `_my_pokemon`, `_attack_damage` /
  `_best_payable_dmg` / `_can_pay` / `_k_needed`, `_weak_mult`, `_damage_shield`,
  `_opp_can_ohko` / `_opp_big_hitter`, `_kill_options` (the KO planner).
* Return path: `_finish(scores, select, want)` — **and** the framework's `_finish` must emit the
  full per-option score vector to the caller/logger, not just the chosen indices (see
  "Data-generation considerations": every agent already computes this vector and throws it away).
* Cross-select state (the `pending_attach_type` mechanism) managed by the framework, not by
  per-deck globals.
* Deterministic tie-breaking (score ties broken by option index, as `_finish` already does),
  logging, and explanations.

What each deck supplies — the **deck policy plugin**:

* the named card-ID / attack-ID constants (from the strategy spec, not hand-typed);
* per-context score functions, following the signatures that already exist in the burn family:
  `_play_score`, `_search_pick_score`, `_energy_target_score`, `_attack_score`, `_retreat_score`,
  `hand_usefulness`, setup-active / setup-bench scoring;
* deck-specific tables (`DECK_ENERGY_TOTALS`, `ONE_PRIZE_IDS`, the attack-id → damage-formula
  switch);
* optionally a multi-turn planner (lucario's `AttackPlan`, dragapult's `plan_a`/`plan_b`) — most
  decks won't need one, and the factory should not generate one by default;
* a `parameters.json` of named numeric weights the score functions read (this is what variants
  perturb — see "Multiple variants per deck").

The LLM fills in the plugin; it does not rebuild the architecture. The seven existing agents are
the reference corpus — and the first validation of the framework is a **regression port**: re-express
one classic agent (lucario) and one burn agent (lilies) as plugins and show behavioral equivalence
on a fixed seeded scenario set (same chosen indices on every decision) before generating any new
deck. If exact equivalence is impractical for the classic agent (it changes the return-path
semantics), equivalence on the burn agent plus no-regression field numbers for the port is the
fallback gate.

Two constraints the framework must preserve:

* **Importability contract**: `eval.py`, `record_battle.py`, and `bc_collect.py` import each agent
  as a bare module from `game_agents/rulebased/` exposing module-level `agent()` and `my_deck`.
  Generated decks must keep that surface (a thin per-deck module that instantiates the framework
  with its plugin), or all three harnesses grow a registry — keeping the surface is cheaper.
* **Fresh-instance semantics**: `bc_collect.fresh_teacher` loads a second copy of a module so two
  seats don't share globals. Framework state must live where that trick still works (module-level
  is fine; process-level singletons are not).

---

# 3. Scenario-based verification

For this use case, scenario tests matter more than competitive win rate: we need to know whether
the agent understands the core loop of the deck, and full-game win rate can't tell us that
(deck-structural plateaus dominate the signal — see Assumptions).

## The mechanism (this exists today)

The state replicator makes scenarios concrete:

1. **Capture**: `record_battle.py --seed N --record-state` plays a deterministic game and attaches
   a full-state blob to every decision (`step.state_blob` in the trace).
2. **Replay a decision**: `utils/resume_battle.py --trace T --step K` restores the battle at
   decision K (`battle_start_from_state`), then either re-feeds the recorded actions (`--replay`)
   or plays the position out with fresh agents (`--p0/--p1`).
3. **Determinism**: `battle_start_seeded(deck0, deck1, seed)` gives reproducible games, so a
   scenario is stable across runs.

One hard constraint shapes the scenario format: **state blobs are raw memory images readable only
by the exact engine build that wrote them** (build-magic check; `errorType 100` on mismatch).
Therefore a scenario's canonical form is not a blob. It is:

```
scenario:
  id: rb_t1_nest_ball
  decks: [simple_raging_bolt_burn, lucario]     # both seats
  seed: 12345
  reach: [[0], [2], [0,3], ...]   # the action prefix (per-decision index lists) driving
                                  # the game from battle_start_seeded to the decision under test
  asserts: ...
```

with the blob as a regenerable **cache** (re-derived by replaying `reach` after any engine
rebuild). `utils/run_state_restore_test.py` already demonstrates exactly this
record-seeded-game / restore / replay-suffix pattern. Two more operational constraints: the cg
battle is a process-global singleton (one scenario at a time per process; parallelize with
processes), and none of this runs on the stock Kaggle engine (fine — teachers are local-only).

Scenarios come from two sources:

* **Mined**: the loss-mining workflow the manual reports document — run a 30-game seeded trace
  session, find the decision where the game was lost, freeze it as a scenario. The repair loop
  (component 4) automates this: every diagnosed failure becomes a permanent regression scenario.
* **Curated/generated**: the generator proposes scenarios from the strategy spec's `setup_goals`
  and `failure_conditions` (it can construct positions by scripting the `reach` prefix, since with
  a fixed seed both hands are known after one recording pass).

## What to test

Per deck, cover at least:

**Opening setup** — which Pokémon to bench / make active, what to search first, where the first
energy goes, the IS_FIRST answer.

**Midgame** — which attacker to prepare; pursue a knockout vs continue setup; what to preserve;
when to switch a damaged Pokémon.

**Endgame** — take an immediate winning line when one exists; pick the KO target with the right
prize count; don't self-deck; spend vs conserve.

**Resource traps** — don't discard the last copy of a necessary card; don't fill the bench with
junk; don't attach energy to a Pokémon that can never attack; don't waste a once-per-turn ability;
don't search for an unusable card.

**Quirk coverage** — for every `quirk_flags` entry in the strategy spec, at least one scenario
exercising that ledger item (e.g. a two-step energy select, an opponent-stadium ACTIVATE).

## Assertion API

A scenario rarely has one exact correct move. Assertions operate on what the framework already
exposes — the chosen index list, the per-option score vector, and the resolved cards
(via `get_card` / the option labels `sb3_features._resolve_option` produces):

```python
# all evaluated at the restored decision, or over a short play_on rollout
assert set(chosen) <= acceptable_index_sets          # membership, not equality
assert resolved(chosen[0]).id in {OGERPON_ID, ...}   # card-level, robust to option reordering
assert not any(resolved(i).id in CRITICAL_LAST_COPY for i in chosen)
assert scores[best_ko_option] > scores[pass_option]  # relative, not absolute
assert rollout(n_decisions=6).benched_count(OGERPON_ID) >= 2   # play_on-based setup check
```

Prefer card-ID-level assertions over option-index-level ones: option order is an engine detail,
card identity is the intent.

---

# 4. The generate–test–repair loop

The harness is small. Concretely, per deck:

```text
1. INPUTS      decks/<name>.csv, piloting/strategy guide, data/EN_Card_Data.csv, cg.all_card_data(),
               the framework package, the quirks ledger, the reference plugins.
2. SPEC        Generate strategy.yaml → run the spec lint → (for important decks) human review.
3. PLUGIN      Generate/modify the deck plugin module.
4. STATIC      Import check; contract lint (every handler returns through _finish; no bare
               maxCount takes; no module state outside the framework's reset hook).
5. SMOKE       utils/run_smoketest_battle.py --deck decks/<name>.csv --games 5   (deck legality +
               engine-completes smoke; it drives RANDOM agents, so it tests the deck, not the
               agent). The agent-contract smoke is step 7 — the battle loop raises on any illegal
               return, so a completed seeded batch IS the contract check.
6. SCENARIOS   Run the deck's scenario suite; each failure produces a structured report.
7. GAMES       A small seeded batch (~30) vs 2–3 field opponents via record_battle sessions;
               telemetry: completion, illegal-action count, win_mode.summarize_ends loss modes,
               key-mechanism usage (did it ever use its main attack/ability? — read from trace
               logs), floor game vs random.
8. REPORT      Summarize failures (format below).
9. REPAIR      The coding agent edits ONLY the plugin + parameters.json (not the framework, not
               the scenarios) given the report. Framework bugs discovered here are routed to a
               human/separate task — otherwise the repair loop will "fix" shared code per-deck.
10. REPEAT     until acceptance (component 5) or until two consecutive iterations change nothing
               outside CI noise (the manual stop criterion: successive iterations' Wilson CIs
               overlap on every pairing → stop, the residual is deck-structural).
```

**Who the coding agent is**: a headless Claude Code invocation (`claude -p` / Agent SDK) with a
fixed prompt template stored in `rulebased_agent_creator/prompts/`, whose editable surface is
restricted to the deck's plugin files. The generator scripts (steps 2–3) are the same mechanism
with different templates. This is a decision we can revisit, but the doc should not leave "the
coding agent" abstract.

The failure report is concrete and machine-assembled from the scenario runner + traces:

```text
Scenario: rb_t1_nest_ball (turn-1 setup with Nest Ball available)
Assertion failed: resolved(chosen[0]).id in {OGERPON_ID}
Chosen:    [4]  → "Nest Ball → Raging Bolt ex"
Acceptable:[2]  → "Nest Ball → Teal Mask Ogerpon ex"
Scores:    idx2=+340  idx4=+410  idx0=-50 ...     ← the miswiring is visible in the scores
Spec rule violated: setup_goals.turn_1 "establish 2+ Ogerpon"
Trace: out/factory/rb/iter3/trace_g12.json step 7   (resume: utils/resume_battle.py --trace ... --step 7)
```

That is much more useful than `Agent win rate: 41%`. Complete-game evaluation stays as a sanity
layer — does it finish games, execute its setup, use its mechanisms, beat random — not as the
repair signal. Thirty seeded games per iteration, not thousands.

---

# 5. Acceptance criteria designed for data generation

Acceptance reflects whether the trajectories are useful for imitation learning, not tournament
strength. Tiers, each with a measurable gate on existing harnesses:

### Tier 0: mechanically valid
* `utils/run_smoketest_battle.py --deck decks/<name>.csv --games 5` passes (deck legal, engine
  completes).
* Zero illegal returns and zero crashes across the seeded game batch driving the actual agent;
  games complete.
* (Deck-outs are **not** in this tier's gate — decking out is a legitimate loss mode, tracked via
  `win_mode.classify_end` `mode == "deck"`. What we gate on is *pathological* deck-out: losing to
  deck-out from a position the telemetry says was winning, or draw-looping.)

### Tier 1: deck-aware
* Core-loop scenarios (opening + "uses its primary attacker/ability") pass.
* Key-mechanism usage ≥ threshold across the game batch (from trace logs).
* Floor check: wins vs `random` at roughly the level the hand-built agents established (~90–96%),
  measured with `eval.wilson` at a trustworthy n (`eval.py` warns below `TRUST_N`).

### Tier 2: competent heuristic
* Resource-trap and endgame scenarios pass; quirk-coverage scenarios pass.
* Field eval vs the existing rule-based pool (`eval.py --mode field`) completes with no
  catastrophic loss-mode signature (e.g. >X% of losses by bench-out from self-inflicted board
  wipes). **No absolute pooled-win% gate**: the field is deck-confounded, and hand-built agents
  span ~21%–52% pooled purely on deck structure. Judge per-pairing with overlapping-CI discipline
  (the `eval-discipline` skill), relative to iteration-over-iteration progress, not to a fixed bar.

### Tier 3: strong heuristic
* Matchup-sensitive and long-horizon decisions. Explicitly optional — not required for the
  warm-start dataset.

Tier 1–2 agents are already valuable teachers. The tier reached, the scenario pass rate, the
loss-mode distribution, and the field numbers all go into the exported agent's metadata.

---

# What should be automated

Per deck, the factory automates:

1. Parse the deck list (`decks/<name>.csv`) and validate it.
2. Retrieve structured card effects (CSV text + `all_card_data()` flags).
3. Classify card roles (attacker / setup / draw / search / energy accel / switching / recovery /
   disruption / energy).
4. Generate the strategy spec + run the spec lint.
5. Generate the deck plugin (score functions + ID tables + parameters.json).
6. Generate the scenario suite (spec-derived + trivially, the quirk-coverage set).
7. Run the loop: smoke → scenarios → seeded games → failure report → repair.
8. Export the accepted agent + metadata, and run the **registration checklist** — this is real,
   enumerable work the original draft omitted; today each new agent touches:
   * `eval.py`: a `make_<deck>` factory + the field pool + the run-manifest deck-hash dict;
   * `record_battle.make_agent`: the spec string;
   * `bc_collect.TEACHERS` (or the `--teachers` universe);
   * `CLAUDE.md`'s module list (docs).
   The factory should either patch these mechanically or (better, one-time refactor) replace the
   hand-maintained lists with discovery over `game_agents/rulebased/` + per-agent metadata, so
   registration is automatic.

Human-in-the-loop points: reviewing `strategy.yaml` for important decks, and triaging anything the
repair loop wants to change outside the plugin (framework bugs, ledger additions, scenario
disputes).

---

# Data-generation considerations

The agents will be flawed; the trajectories are still useful if we record enough.

`bc_collect.py` already records, per learnable decision:
`{game_id, seat, deck, obs (trimmed), picks, min_count, max_count, outcome}` — with every shard
gated by a `replay_check` (picks re-verified legal against the re-driven mask). The factory
extends this record; it does not reinvent it:

```text
+ agent_id / variant_id / agent_version    (which teacher, which parameters.json)
+ scores                                   (per-option heuristic score vector)
+ select context + turn number             (cheap, useful for slicing)
+ confidence label                          (derived — see below)
```

The `scores` field is nearly free: **every existing agent already computes a full index-aligned
per-option score vector and discards it** — the framework's `_finish` just has to hand it to the
recorder. Instead of learning only "the teacher picked A", the model can learn a ranking or soft
target:

```text
A: 8.5   B: 7.9   C: 2.1   D: -10.0
```

Two honest caveats, flagged for independent review before we build training changes on them
(theory, not verified in this repo):

* **Scores are not calibrated and not comparable across contexts or decks.** The heuristics use
  band constants (±100000 forcing bands, small tie-break offsets), so a raw softmax over scores is
  meaningless; any soft-target scheme needs per-context normalization (rank-based, or per-select
  temperature) — or should be dropped in favor of using scores only for the confidence label.
* **`bc_train.py` today is hard one-hot** (sequential multi-select decomposition + STOP examples +
  scalar outcome for the value head). Soft targets and per-example weights are a `bc_train`
  change, to be gated the same way as any training change (holdout acc + downstream eval), not
  assumed to help.

## Filter the data by confidence

Label decisions:

* **High confidence**: forced/only-legal moves, immediate-win lines, large score gaps.
* **Medium**: clear heuristic preference.
* **Low**: near-tied scores (tie-break arbitrary), contexts the plugin doesn't handle (generic
  fallthrough), states flagged unfamiliar.

A weight like `w = f(score gap between best and second-best)` captures "the heuristic believes
this is clear" — subject to the normalization caveat above (use rank gap or a per-context scale,
not raw score units). Initial use: train more heavily on high-confidence examples, and keep the
label in the shard either way so the experiment is possible without recollecting.

---

# Use multiple variants per deck

Create several variants of the same deck agent — aggressive, setup-focused,
resource-conservative, simple-greedy — not for a strategic league, but to reduce imitation
collapse onto one narrow behavioral style.

Variants share the plugin code and differ only in `parameters.json` (feature weights, thresholds,
risk tolerance, bench-development aggressiveness). They are produced by parameter sampling, not by
separate LLM generation.

One check the hand-built history demands: **verify variants actually behave differently.** Several
hand-tuned "levers" shipped inside CI noise with no measurable behavioral change. The cheap gate:
run the variant pair over the fixed seeded scenario/game set and measure the action-disagreement
rate; a "variant" that disagrees with the base on <~5% of decisions is not a variant, it's noise —
resample. (Threshold to be tuned; the point is the gate exists.)

Each variant is a distinct `variant_id` in the trajectory data.

---

# What we do not need yet

Omitted at this stage: Nash computation, PSRO, MAP-Elites, evolutionary populations, Elo-promotion
systems, best-response exploiters, full round-robin tournaments, general any-deck agents,
automatic deck construction, and end-to-end LLM-written agents with unrestricted code access.
These may matter later; they are unnecessary for a decent behavior-cloning bootstrap dataset.

---

# The main risk

Not that the agents are weak — we expect that. The risk is that they are **systematically wrong in
the same way**, so the learned policy inherits brittle habits: always spending search cards
immediately, overvaluing short-term damage, never contesting the prize race, never preserving
late-game resources, collapsing outside states the heuristics visit. (This is not hypothetical:
the PPO line's standing diagnosis is exactly a prize-race blind spot, and the one 82.6% BC+PPO
result won by perfecting a deck-out exploit — teacher biases propagate.)

Mitigations, in order of leverage:

* Multiple policy variants per deck (above), with the disagreement gate.
* Store per-option scores + confidence so training can down-weight arbitrary choices.
* Include losing trajectories and recovery states; don't filter to wins.
* Seeded randomized starts (vary `--seed` and opponents) so coverage isn't one canonical line.
* Treat imitation as initialization: self-play / RL / DAgger-style relabeling comes after, and the
  state replicator makes DAgger-style "restore a learner state, ask the teacher" mechanically
  possible later.

Distribution shift remains: once the learned policy deviates from the teacher it visits states
absent from teacher data. That is a normal BC limitation, addressed downstream, not here.

---

# Repo layout

Extend the existing structure instead of inventing a parallel `deck_factory/` tree — the
importability contract (eval/record_battle/bc_collect import bare modules from
`game_agents/rulebased/`) and the committed `decks/*.csv` format already fix half the layout:

```text
game_agents/rulebased/
    common/                      # NEW: the shared framework (extracted, not written fresh)
        runtime.py               #   contract wrapper, dispatch scaffold, _finish, reset hook
        resolve.py               #   get_card / option resolution / board census
        scoring.py               #   prize_count, pokemon_score, damage/threat/KO helpers
        quirks.md                #   the living engine-quirks ledger
    <deck>.py                    # per-deck module (thin: plugin + framework), same import surface
    <deck>_params/               # parameters.json variants (exact shape TBD at implementation)

rulebased_agent_creator/
    heuristic_agent_generator.md # this doc
    creating_rulebased_agents.md # the manual guide (stays: it's the generator's source material)
    validate_deck.py             # exists
    prompts/                     # generation + repair prompt templates
    specs/<deck>/strategy.yaml
    scenarios/<deck>/*.yaml      # canonical (decks, seed, reach, asserts); blobs are cache only
    runner/                      # spec lint, scenario runner, failure reporter, loop driver

decks/<name>.csv                 # unchanged, committed
out/factory/<deck>/              # iteration artifacts: traces, blobs, reports (gitignored)
```

Deck lists stay CSV (not the originally-sketched `decklist.json`) — the loaders, the Kaggle
fallback chain, and `validate_deck.py` all already speak CSV.

---

# Open questions (decide at implementation time, not silently)

1. **Framework extraction order**: refactor all seven existing agents onto `common/`, or freeze
   them as-is and only new decks use it? (Freezing avoids regression risk on measured teachers;
   refactoring gives the regression-port validation for free on two of them. Current lean:
   port lucario + lilies as the validation pair, leave the other five frozen.)
2. **Registry vs hand-edited lists** for eval/record_battle/bc_collect registration (see
   "What should be automated" §8).
3. **Scenario `reach` authoring**: mined-only at first, or invest in a position-scripting helper
   early? (Lean: mined-only for v1; scripting when a deck's core loop can't be reached by mining.)
4. **Soft targets / confidence weighting in `bc_train`**: record the data now, gate the training
   change behind its own experiment (this doc only commits to recording).
5. **How many decks** the first factory run targets, and from what source (NAIC-style meta lists
   vs. what the engine's ~1268-card pool supports) — determines how much spec-review time to
   budget.

---

The central design principle:

> **Generate small deck-specific policy plugins inside a strong shared framework extracted from
> the seven hand-built agents, and improve them through scenario failures replayed on the state
> replicator — not through open-ended tournament evolution.**

That is appropriately scoped for a factory of flawed-but-useful heuristic teachers, and far more
likely to produce many usable deck agents within days than a general autoresearch system aimed at
discovering strong play.
