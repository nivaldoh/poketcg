# The `cg` simulator SDK

Note: this file was created before the rest of the engine documentation was copied to this repo, so it's probably redundant and needs to be reviewed.

Reference for the proprietary Pokémon TCG simulator (`cg`). The API below is **ground truth** read from the vendored source at `vendor/cg-lib/cg/` (`api.py`, `game.py`, `sim.py`) — not inferred. SDK docs upstream: https://matsuoinstitute.github.io/cabt/.

## Obtaining / restoring the engine

`cg` is a compiled C++ engine (`libcg.so` on Linux, `cg.dll` on Windows), loaded via `ctypes` in `sim.py`, which also calls `GameInitialize()` at import time. It is gitignored (proprietary, binary) and must be present at `vendor/cg-lib/cg/` for anything to run. To restore on a fresh clone (needs a Kaggle account/token in `~/.kaggle/`):

```bash
pip install kaggle
SLUG=pokemon-tcg-ai-battle   # Simulation track; ships the engine under sample_submission/cg/
for f in __init__.py api.py game.py sim.py utils.py libcg.so; do
  kaggle competitions download -c $SLUG -f sample_submission/cg/$f -p vendor/cg-lib/cg
done
kaggle competitions download -c $SLUG -f sample_submission/deck.csv -p .
```

The companion competition `pokemon-tcg-ai-battle-challenge-strategy` (Strategy track) ships only card data (CSVs/PDFs), not the engine.

## Agent contract

**Entry point:** `agent(obs_dict: dict) -> list[int]`. Returns a list of **option indices** into `obs.select.option`, with length in `[select.minCount, select.maxCount]` and no duplicates. Convert the raw dict with `to_observation_class(obs_dict)`. On the very first call `obs.select is None` — the agent must instead return the **60-card deck** as a list of card IDs (TCG deck rules: ≤4 of any non-basic-energy card, ≥1 Basic Pokémon, ≤1 Ace Spec). Note: in the `battle_start`-driven local loop, decks are passed directly, so `obs.select is None` only occurs in the Kaggle submission flow.

## Observation structure

- `obs.current` — game state: `players[2]`, `yourIndex`, `firstPlayer`, `turn`, `stadium`, `looking`, and `result` (`< 0` = ongoing; `0`/`1` = that player won; `2` = draw). Per-turn flags like `energyAttached`, `supporterPlayed`.
- `obs.current.players[i]` — `active[]`, `bench[]`, `hand`/`handCount`, `discard`, `prize`, `deckCount`, and status conditions (`poisoned`, `burned`, `asleep`, `paralyzed`, `confused`). Your opponent's hidden zones are counts only.
- `obs.select` — the current decision: `context` (a `SelectContext`, e.g. `MAIN`, `SWITCH`, `TO_ACTIVE`, `TO_HAND`, `ATTACH_FROM`, `SETUP_ACTIVE_POKEMON`), the list of legal `option`s, and `minCount`/`maxCount`.

## Options are typed

Each `option` has a `type` (`OptionType`: `PLAY`, `ATTACH`, `EVOLVE`, `ABILITY`, `RETREAT`, `ATTACK`, `DISCARD`, `CARD`, `ENERGY`, `YES`/`NO`, `NUMBER`, `END`, ...) plus type-specific fields (`area`, `index`, `inPlayArea`, `inPlayIndex`, `attackId`, `number`, `playerIndex`, ...). The agent maps each option to a card/effect and scores it. Both starters share a near-identical `get_card(obs, area, index, player_index)` helper that resolves an option's `(area, index)` to the actual `Card`/`Pokemon` via an `AreaType` match — the canonical pattern for interpreting options; reuse it.

**Card metadata:** `all_card_data()` returns all cards; build `card_table = {c.cardId: c}` for lookups. Flags used by heuristics: `cardType`, `stage1`/`stage2`, `ex`/`megaEx`, `weakness`/`resistance` (an `EnergyType`), `name`.

## Game driver loop

From `cg.game` (see `utils/run_smoketest_battle.py` and `starter_rl_kaggle.py`):

```python
obs, start_data = battle_start(deck0, deck1)   # both decks are 60 IDs; errorPlayer >= 0 (or obs is None) => deck rejected
while obs["current"]["result"] < 0:            # loop on the raw dict
    selected = agent_fn(obs)                   # -> list of option indices
    obs = battle_select(selected)              # raises ValueError/IndexError on invalid selection
battle_finish()                                # takes no args; then read obs["current"]["result"]
```

`result == your_index` is a win, `== 2` a draw, else a loss. This loop is the natural place to plug in win/loss measurement — win rate is the trivially-checkable signal the README asks for; intermediate metrics are an open problem to design later.

## Search API / MCTS

`search_begin(obs, your_deck=, your_prize=, opponent_deck=, opponent_hand=, opponent_active=, ...)` → `SearchState`; then `search_step(searchId, select)` advances a hypothetical, `search_end()` releases. Because opponent cards are hidden, the starter **determinizes** them: it samples your own deck/prize from known IDs and fills opponent slots with placeholder card IDs (e.g. Snorlax `1072`, basic energy `1`). The Search API also lets a rule-based agent ask "what would this attack actually do?" without committing.

In the MCTS code (`mcts_agent`, `Node`/`Child`, `create_node`), the model produces a scalar **value** (encoder, tanh) and a per-candidate-action **policy** (decoder). Action candidates are enumerated as index-combinations up to 64; selection uses a PUCT-style score. Self-play emits `LearnSample`s whose value/policy labels are filled by backprop after the game (TD-style with `LAMBDA = 0.9`).
