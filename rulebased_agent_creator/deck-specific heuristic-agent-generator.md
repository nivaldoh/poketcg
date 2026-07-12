We **not** need an evolutionary league, PSRO, Nash mixtures, exploiters, or sophisticated population management yet. We need a **deck-specific heuristic-agent generator with strong mechanical verification**.

The goal is not:

> Discover strategically optimal agents.

It is:

> Produce many agents that reliably execute each deck’s basic game plan, make mostly sensible legal decisions, and generate trajectories that are better than random self-play.

That is much more achievable.

## Assumptions

* Each generated agent plays one fixed deck list.
* The simulator exposes legal actions and the current player-visible state.
* We can replay or construct at least some specific game states through the modified version of the PTCG engine source code available locally.
* The downstream model will later improve through RL or stronger search, so the heuristic agents only need to provide a useful initialization.
* We care more about broad deck coverage than squeezing the last few percentage points from each heuristic.

Under those assumptions, we will build the rule-based agent factory around five components.

---

# 1. A standard deck-strategy specification

Do not ask the coding agent to infer everything directly from raw card text each time.

Represent every deck using a structured strategy file:

```yaml
deck_name: Raging Bolt ex / Teal Mask Ogerpon ex

primary_attackers:
  - Raging Bolt ex

support_attackers:
  - Teal Mask Ogerpon ex

setup_goals:
  turn_1:
    - establish 2 or more Ogerpon
    - attach energy
    - find draw support
  turn_2:
    - prepare Raging Bolt
    - reach required damage threshold

resource_roles:
  Earthen Vessel:
    purpose: find energy
    preserve_if:
      - hand already lacks discardable cards
  Energy Retrieval:
    purpose: recover energy for damage scaling

strategic_rules:
  - avoid benching unnecessary two-prize Pokémon
  - preserve enough energy recovery for the late game
  - prefer knockouts that improve the prize race

failure_conditions:
  - no attacker prepared by turn 2
  - discarding all copies of a critical recovery card
```

This becomes the intermediate representation between:

* Deck list.
* Card definitions.
* Human or generated strategy knowledge.
* Executable policy.

The factory can generate this specification first, review or test it, and then generate code from it.

This will be much more reliable than asking an LLM to repeatedly write an entire agent from scratch.

---

# 2. A reusable heuristic policy framework

Most deck agents should share the same engine.

Only deck-specific scoring rules and priorities should change.

For example:

```python
class DeckPolicy:
    def choose_action(self, state, legal_actions):
        phase = self.identify_phase(state)

        winning_actions = self.find_immediate_wins(state, legal_actions)
        if winning_actions:
            return self.rank(winning_actions, state)[0]

        safe_actions = self.remove_obviously_bad_actions(
            state,
            legal_actions,
        )

        scored_actions = [
            (action, self.score_action(state, action, phase))
            for action in safe_actions
        ]

        return max(scored_actions, key=lambda item: item[1])[0]
```

The shared framework should provide:

* Legal-action handling.
* Turn-phase detection.
* Immediate-win detection.
* Attack-damage calculation.
* Prize-trade estimation.
* Bench-space management.
* Resource counting.
* Search-card resolution.
* Retreat and switching logic.
* Generic setup heuristics.
* Deterministic tie-breaking.
* Logging and explanations.

Then each deck supplies functions like:

```python
score_bench_candidate(...)
score_energy_attachment(...)
score_search_target(...)
score_attacker(...)
score_supporter(...)
score_attack(...)
```

This is important because otherwise the factory will create dozens of agents with duplicated, inconsistent, and buggy mechanics.

The LLM should mostly fill in a **deck policy plugin**, not rebuild the complete agent architecture.

We can use the existing abomasnow, dragapult_ex, iono and lucario rule-based agents as initial references.

---

# 3. Scenario-based verification

For this use case, scenario tests matter more than competitive win rate.

We need to know whether the agent understands the core loop of the deck.

Each deck should have generated or manually curated scenarios such as:

### Opening setup

* Which Pokémon should be benched?
* Which basic Pokémon should be active?
* What should be searched from the deck?
* Where should the first energy be attached?

### Midgame

* Which attacker should be prepared?
* Should the agent pursue a knockout or continue setting up?
* Which resources should be preserved?
* Which damaged Pokémon should be switched?

### Endgame

* Is there an immediate winning line?
* Which target produces the correct prize count?
* Should the agent avoid decking itself out?
* Should it conserve or spend its remaining resources?

### Resource traps

* Do not discard the final copy of a necessary card.
* Do not fill the bench with irrelevant Pokémon.
* Do not attach energy to a Pokémon that cannot become useful.
* Do not use a once-per-turn ability on the wrong target.
* Do not search for a card that cannot be used.

A scenario does not always need one exact correct move. You can verify broader conditions:

```python
assert chosen_action in acceptable_actions
assert not destroys_critical_resource(chosen_action)
assert improves_setup_score(next_state)
```

This is especially useful because Pokémon positions often have several reasonable actions.

---

# 4. A simple generate–test–repair loop

The autoresearch harness can be quite small:

```text
1. Provide the deck list, card definitions (available in data/EN_Card_Data.csv), policy framework,
   and strategy specification.

2. Generate or modify the deck-specific policy module.

3. Run static and mechanical checks.

4. Create and run the deck’s scenario suite.

5. Run a small number of complete games.

6. Summarize failures.

7. Ask the coding agent to repair the policy.

8. Repeat until it reaches an acceptance threshold.
```

The failure report should be concrete:

```text
Scenario: Turn 1 setup with Nest Ball available

Expected behavior:
- bench Teal Mask Ogerpon ex
- preserve active Raging Bolt ex

Observed behavior:
- searched for a second Raging Bolt ex
- bench became too attacker-heavy

Relevant state:
...

Chosen action:
...

Available actions:
...
```

This is much more useful than:

```text
Agent win rate: 41%
```

Complete-game evaluation is still useful, but mainly as a sanity check:

* Does the agent complete games?
* Does it execute its intended setup consistently?
* Does it frequently deadlock?
* Does it use major deck mechanisms?
* Is it clearly better than random or a generic baseline?

You do not need thousands of games per candidate.

---

# 5. Acceptance criteria designed for data generation

Your acceptance test should reflect whether the trajectories are useful for imitation learning.

A deck agent might be acceptable if it achieves something like:

* Virtually zero illegal actions.
* Virtually zero crashes and timeouts/deckouts.
* High success on core deck scenarios.
* Consistent execution of its setup plan.
* Clearly better results than a random legal agent.
* No frequent catastrophic resource mistakes.
* Reasonable action diversity where several plays are valid.

We can define quality tiers.

### Tier 0: mechanically valid

* Completes games.
* Always selects legal actions.

### Tier 1: deck-aware

* Uses the deck’s central Pokémon, abilities, and attacks.
* Performs its basic setup.

### Tier 2: competent heuristic

* Manages resources reasonably.
* Identifies straightforward knockouts.
* Handles common disruption and recovery situations.

### Tier 3: strong heuristic

* Makes matchup-sensitive and long-horizon decisions.

For your warm-start dataset, Tier 1 and Tier 2 agents may already be valuable. You do not necessarily need Tier 3 for every deck.

---

# What should be automated

The factory can automate the following for each deck:

1. Parse the deck list.
2. Retrieve structured card effects.
3. Classify card roles:

   * attacker;
   * setup Pokémon;
   * draw;
   * search;
   * energy acceleration;
   * switching;
   * recovery;
   * disruption.
4. Generate a first-pass strategy specification.
5. Generate deck-specific scoring functions.
6. Generate basic scenario tests.
7. Run games and identify repeated failures.
8. Repair the policy.
9. Export the accepted agent and its metadata.

I would still manually inspect the generated strategy specification for important decks. A bad high-level strategy will otherwise produce code that is internally consistent but fundamentally misguided.

---

# Data-generation considerations

The agents will be flawed, but the trajectories can still be useful if you record enough information.

At every decision, store:

```text
deck ID
agent version
player-visible observation
legal action set
chosen action
optional heuristic scores for all actions
game phase
eventual result
turn number
complete trajectory ID
```

The heuristic score for each legal action can be especially useful. Instead of learning only:

> The teacher selected action A.

The model can learn a ranking or soft target:

```text
A: 8.5
B: 7.9
C: 2.1
D: -10.0
```

Those scores will not be perfectly calibrated, but they contain more information than a one-hot action.

## Filter the data by confidence

Not every heuristic choice should be treated as equally authoritative.

You can label decisions as:

* **High confidence:** immediate win, required legal sequence, obvious setup action.
* **Medium confidence:** strong heuristic preference.
* **Low confidence:** close scores, arbitrary tie-breaking, unfamiliar state.

For imitation learning, you could initially train more heavily on high-confidence examples.

For example:

[
w_t = f(\text{score gap between best and second-best action})
]

A large score gap suggests the heuristic believes the action is clear. A small gap suggests uncertainty.

This helps prevent the model from overfitting to arbitrary choices made by a crude teacher.

---

# Use multiple variants per deck

Create several variants of the same deck agent:

* Aggressive.
* Setup-focused.
* Resource-conservative.
* Simple greedy.
* Matchup-specific, where applicable.

Not because you need a strategic league, but because it reduces imitation collapse onto one narrow behavioral style.

These variants can share 95% of their code and differ primarily in:

* Feature weights.
* Thresholds.
* Risk tolerance.
* Preferred prize mapping.
* Bench-development aggressiveness.

Numeric variants can be produced cheaply through parameter sampling or an optimizer rather than separate LLM-generated codebases.

---

# What we do not need yet

For this stage, we should omit:

* Nash equilibrium computation.
* PSRO.
* MAP-Elites.
* Large evolutionary populations.
* Elo-based promotion systems.
* Best-response exploiters.
* Full round-robin tournaments.
* General agents that can play arbitrary decks.
* Automatic deck construction.
* End-to-end LLM-written agents with unrestricted code access.

Those may become relevant later for the overarching codebase, but they are unnecessary here for producing a decent behavior-cloning bootstrap dataset.

---

# The main risk

The main risk is not that the agents are weak. We already expect that.

The risk is that they are **systematically wrong in the same way**, causing the learned policy to inherit brittle habits.

Examples:

* Always spending search cards immediately.
* Overvaluing short-term damage.
* Failing to plan prize trades.
* Never preserving resources for the late game.
* Behaving badly outside states commonly visited by the heuristic.

Mitigations:

* Use multiple policy variants.
* Store confidence and action scores.
* Include losing trajectories and recovery states.
* Generate randomized but plausible starting situations.
* Later fine-tune through self-play or RL.
* Treat imitation as initialization, not as the final objective.

There will still be distribution shift: once the learned policy deviates from the teacher, it will visit states absent from the teacher data. That is a normal limitation of behavioral cloning. Self-play, online RL, DAgger-like relabeling, or search-based improvement can address it later.

## A practical minimal architecture

```text
deck_factory/
    common_policy/
        state_features.py
        action_filters.py
        generic_scoring.py
        policy_runtime.py

    decks/
        raging_bolt/
            decklist.json
            strategy.yaml
            policy.py
            parameters.json
            scenarios/
        charizard/
            ...

    evaluator/
        scenario_runner.py
        match_runner.py
        invariant_checks.py
        failure_report.py

    generator/
        strategy_generator.py
        policy_generator.py
        repair_agent.py

    trajectories/
        writer.py
        schemas.py
```

The central design principle is:

> **Generate small deck-specific policy modules inside a strong shared framework, and improve them primarily through scenario failures rather than open-ended tournament evolution.**

That is appropriately scoped for building a factory of flawed but useful heuristic teachers. It is also much more likely to produce many usable deck agents within days than a general autoresearch system aimed at discovering strategically strong play.
