# How our model works (architecture + execution flow)

This document explains, in concrete detail, how **our current Pokémon-TCG agent** works:
what the model is, how it turns a battle observation into a decision, how it was trained,
and exactly what happens during one in-game turn. It is written to be read top-to-bottom
by someone who has *not* memorised the codebase.

**Ground truth & drift note.** The compiled `cg` engine and `all_card_data()` are the
source of truth; this doc points at symbols (find them with the `grep` recipes shown) rather
than pinning line numbers, and discovers paths/defaults/card-names at runtime. Card facts here
were resolved with `all_card_data()`, never guessed from IDs. For the SDK contract (observation
schema, Search API) the canonical reference is [`docs/cg-sdk.md`](docs/cg-sdk.md); for the game
rules it is [`docs/rulebook.md`](docs/rulebook.md) (with the engine winning on any conflict — see
[§11](#11-rules-consistency-notes)). The numbers in the worked example come from one real,
captured battle (the engine reshuffles every run, so the specific cards will differ if you
re-run, but the mechanics do not).

---

## 1. TL;DR — what "the model" is

Our model is **`MyModel`** in `tcg_model.py` (`grep -n 'class MyModel' tcg_model.py`): a small
Transformer with two heads.

| Head | Built from | Produces |
| --- | --- | --- |
| **Encoder = value** | a sparse board encoding → `TransformerEncoder` | one scalar `value ∈ [−1, 1]` ("am I winning?") |
| **Decoder = policy** | a sparse per-action encoding, cross-attending to the encoder | one scalar score `∈ [−1, 1]` *per legal action* ("how good is this option?") |

It was produced by **behaviour cloning (BC)**: we recorded the rule-based teacher
`rulebased_lucario` playing the Mega Lucario ex deck against itself, then trained `MyModel` to
imitate it (`train_mymodel_bc.py --policy-loss ce` → `out/mymodel_lucario_bc_ce.pth`, the
checkpoint these docs describe — confirm it exists with `ls out/`).

At play time the model is used in one of two ways:
- **Greedy** (`eval_mymodel.py:greedy_agent`): one forward pass, pick the highest-scoring option.
- **MCTS** (`tcg_model.py:mcts_agent`): use the policy as a search **prior** and the value (or a
  random rollout) as the **leaf evaluation**, run PUCT search over a *determinised* copy of the
  game, and play the most-visited action. This is an AlphaZero-style agent.

The rest of this doc unpacks each piece.

---

## 2. The agent contract and the battle loop (top-level execution flow)

### 2.1 The contract
Every agent — random, rule-based, or ours — is a single function:

```python
def agent(obs_dict: dict) -> list[int]
```

- **First call only (Kaggle submission flow):** `obs["select"] is None`, and the agent must
  return a **60-card deck** as a list of card IDs. In our *local* driver this never happens —
  decks are passed straight to `battle_start`, so `obs.select` is always populated (see the
  note in `run_battle.py:random_agent`).
- **Every other call:** return a list of **option indices** into `obs.select.option`. The list
  length must satisfy `minCount ≤ len ≤ maxCount`, with **no duplicate indices**. Option
  indices are *positions in the option list*, **not** card IDs.

### 2.2 The driver loop
A battle is three SDK calls (`from cg.game import battle_start, battle_select, battle_finish`).
The canonical loop is `run_battle.py:play_one` (`grep -n 'def play_one' run_battle.py`):

```python
obs, start_data = battle_start(deck0, deck1)        # decks: 60 card IDs each
if start_data.errorPlayer >= 0 or obs is None:
    raise ValueError("deck rejected by engine")     # e.g. illegal deck
while obs["current"]["result"] < 0:                 # < 0 means "game ongoing"
    obs = battle_select(agent(obs))                 # apply the agent's choice, advance
battle_finish()                                     # release engine memory
result = obs["current"]["result"]                   # 0 = player 0 won, 1 = player 1 won, 2 = draw
```

So the game is just a sequence of **decision points**: the engine hands you an observation
describing the current state and the legal options, you return which option(s) to take, and it
steps forward to the next decision (which may belong to either player — check
`obs["current"]["yourIndex"]`). When `result ≥ 0` the game is over. The engine also records
*why* it ended (`api.py` `LogType.RESULT.reason`): `1` took all Prize cards, `2` started a turn
unable to draw (deck-out), `3` no Pokémon left in the Active Spot, `4` a card effect — which
lines up with the rulebook's three win conditions ([§11](#11-rules-consistency-notes)).

A single decision is **not** a whole turn — one turn is many `battle_select` calls (bench a
Pokémon, attach an Energy, play a Trainer, …, then "END turn"). The worked example in
[§9](#9-worked-example--one-real-turn-with-several-actions) is *one* such decision.

---

## 3. The observation model (the model's raw input)

`obs = to_observation_class(obs_dict)` deserialises the dict into dataclasses defined in
`vendor/cg-lib/cg/api.py` (the ground-truth schema; `grep -n '^class \|^@dataclass' vendor/cg-lib/cg/api.py`).
The pieces the model actually reads:

- **`Observation`**: `.select` (the current decision), `.current` (the game `State`), `.logs`
  (events since the last decision).
- **`State`** (`obs.current`): `turn`, `yourIndex` (0/1 — whose decision this is),
  `firstPlayer`, the per-turn flags `energyAttached` / `supporterPlayed` / `stadiumPlayed` /
  `retreated`, `result`, `stadium` (0–1 cards), and `players` (exactly 2 `PlayerState`s).
- **`PlayerState`**: `active` (0–1 `Pokemon`), `bench` (list of `Pokemon`), `deckCount`,
  `discard`, `prize`, `handCount`, `hand`, and the five status booleans `poisoned` / `burned`
  / `asleep` / `paralyzed` / `confused`.
- **`Pokemon`**: `id`, `hp`/`maxHp`, `energies` (list of `EnergyType`), `energyCards`, `tools`,
  `preEvolution`.
- **`SelectData`** (`obs.select`): `type` (`SelectType`), `context` (`SelectContext`),
  `minCount`, `maxCount`, and `option` — the list of legal `Option`s.
- **`Option`**: a tagged union keyed by `type` (`OptionType`). The fields that matter depend on
  the type, e.g. `PLAY` uses `index` (into hand); `ATTACH` uses `(area, index)` for the card and
  `(inPlayArea, inPlayIndex)` for the target Pokémon; `ATTACK` uses `attackId`; `END` has none.

**Visible vs hidden.** You see your own hand, Active, Bench, and (face-up) discard; you see only
**counts** of the opponent's hand and deck (`handCount`, `deckCount`), and face-down prizes are
`None`. Hidden information is why MCTS has to *determinise* (see [§8](#8-inference-path-b--mcts)).

The enums are small and worth knowing (verbatim from `api.py`): `OptionType` = `NUMBER=0, YES=1,
NO=2, CARD=3, TOOL_CARD=4, ENERGY_CARD=5, ENERGY=6, PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10,
DISCARD=11, RETREAT=12, ATTACK=13, END=14, SKILL=15, SPECIAL_CONDITION=16`; `SelectType` ranges
`MAIN=0 … SPECIAL_CONDITION=10`; `SelectContext` ranges `MAIN=0 … RECOVER_SPECIAL_CONDITION=48`.

---

## 4. Feature encoding (observation → sparse tensors)

`MyModel` does not read the dataclasses directly. Two builders turn them into **sparse
bag-of-features** tensors, fed to `torch.nn.EmbeddingBag(mode="sum")` layers (each "word" is a
bag whose feature embeddings are summed). Helpers: `add_card`, `add_cards`, `add_pokemon`,
`add_player` build a `SparseVector` of `(index, value)` pairs grouped into words by `word_start()`.

### 4.1 The encoder input — exactly 24 "words"
`get_encoder_input(obs, your_deck)` (`grep -n 'def get_encoder_input' tcg_model.py`) emits
`num_words_encoder = 24` words, always ordered **you first** (it indexes players with
`i ^ your_index`, so the encoding is perspective-invariant):

| Words | Content | Per-word features |
| --- | --- | --- |
| 0–7 | your Bench (8 slots) | empty-flag, `hp/400`, card identity, tools, energies (weight 0.5) |
| 8–15 | opponent Bench (8 slots) | same |
| 16 | your Active | same |
| 17 | opponent Active | same |
| 18 | your player state | `deckCount/60`, `discard/60`, `handCount/8`, `bench/5`, prize-count one-hot, 5 status flags, discard contents (0.25) |
| 19 | opponent player state | same |
| 20 | your hand | bag of your hand cards (weight 0.25) |
| 21 | your deck | bag of your full deck list (weight 0.25) |
| 22 | stadium | stadium card(s) |
| 23 | meta | constant bias, `turn/10`, `firstPlayer == you` |

The `EmbeddingBag` vocabulary `encoder_size = 22000` is a fixed cap set above the highest
feature index the 24 words actually use (≈21,600 in this build, since the indices accumulate
across words); the card-identity slots — `card_count = 1268` cards each — make up most of that
space. A neat detail: the 8 bench slots
of a player share the same feature sub-space (the builder resets `sv.pos` between slots), so the
model treats the bench as an **order-invariant set**, not 8 positional slots.

### 4.2 The decoder input — one bag per candidate action
`get_decoder_input(obs, actions)` emits one word per candidate action. Vocabulary layout
(`decoder_size = 73847` for this build): indices `0–13` are option-type tokens (`END=1`,
`YES=2`, `NO=3`, special-condition `4–8`, number `9–13`); `14 … 14+attack_count` are attack IDs;
and everything from `decoder_card_offset = 14 + attack_count = 1571` onward is
**`card_id` × context block** (a card embedded differently depending on *how* it appears in the
option — played, attach source, attach target, evolve, etc.). See the concrete bags in
[§9.3](#93-the-exact-features).

---

## 5. The model architecture

`MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1)`
— this is exactly how every loader instantiates it (`grep -n 'MyModel(' *.py`). `forward` takes
the encoder/decoder sparse tensors and returns `(value, policy)`:

```
ENCODER (value head)
  encoder_bag(index,value,offset)          -> [batch, 24, 128]   (sum-pool each word)
  TransformerEncoder (1 layer, 2 heads)    -> [24, batch, 128]   (self-attention across the 24 words)
  encoder_fc -> tanh(mean over 24 words)   -> value  [batch, 1]   in [-1, 1]

DECODER (policy head)
  decoder_bag(index,value,offset)          -> [batch, A, 128]    (A = #candidate actions)
  for layer in decoder:                                          (cross-attention: actions attend
      x = layer(x, encoder_out)                                   to the 24 encoded board words)
  decoder_fc -> tanh                       -> policy [batch, A]   in [-1, 1]
```

So **board words talk to each other** (encoder self-attention) to judge the position, and each
**candidate action attends to the board** (decoder cross-attention) to be scored in context.
Two things to internalise:

1. Both outputs are **`tanh`-bounded scalars in [−1, 1]**, *not* probabilities. The value is a
   win-expectation; each policy score is a per-option preference.
2. When MCTS needs a probability **prior**, it sharpens the policy: `prior ∝ exp(score × 10)`,
   normalised over options (`create_node` in `tcg_model.py`). The `×10` makes the prior nearly
   one-hot on the model's favourite option (see the numbers in [§9](#9-worked-example--one-real-turn-with-several-actions)).
   The BC trainer's cross-entropy loss uses the same `×10`, so training and search agree.

At inference the policy length is just the number of candidate actions; the "64" you'll see in
the trainer is only the fixed padding width used to batch variable-sized option sets.

---

## 6. How it was trained (behaviour cloning)

`train_mymodel_bc.py` (`grep -n 'def record_teacher_games\|def train' train_mymodel_bc.py`):

1. **Collect** (`record_teacher_games`): drive **teacher-vs-teacher** Lucario-mirror games
   (`battle_start(deck, deck)` with `rulebased_lucario.agent` on both seats). For each decision
   that is a genuine single-select policy choice — `is_policy_select`: `minCount == maxCount == 1`,
   `> 1` option, and not an all-`NUMBER` menu — record a training sample:
   - **encoder input** = `get_encoder_input` of that board;
   - **decoder input** = one bag per legal option;
   - **policy target** = teacher's chosen option → `+1`, every other option → `−1`;
   - **value target** = the eventual **game outcome from that seat** (`+1` win / `−1` loss /
     `0` draw), filled in once the game finishes.
2. **Train** (`train`): `AdamW(lr=3e-4)`, defaults `--games 250 --epochs 4 --batch-size 128`.
   The value head uses `HuberLoss`. The policy head uses one of:
   - `--policy-loss huber` (default): regress each option toward its ±1 target;
   - `--policy-loss ce` (**our checkpoint**): cross-entropy over the legal options toward the
     teacher's choice, on logits `score × 10` — sharper imitation that matches the MCTS prior.

The output path is whatever `--out` says (ours: `out/mymodel_lucario_bc_ce.pth`); read current
defaults with `python train_mymodel_bc.py --help` rather than trusting this paragraph.

> **Important consequence — read [§10](#10-risks--limitations).** The policy head only ever sees
> *single-select* decisions, and the value head's target is the outcome of a **mirror** game
> (teacher vs identical teacher ≈ a coin flip). Both facts shape — and limit — what the trained
> model can do.

---

## 7. Inference path A — greedy (no search)

`eval_mymodel.py:greedy_agent(obs_dict, deck, model)`:

1. If the decision is **not** a single-select policy choice (`is_policy_select` is false), don't
   ask the model — `auto_resolve` it with a fixed heuristic (all-`NUMBER` → pick the max; a lone
   option → take it; `minCount == 0` → take `maxCount`; else take the first `minCount`).
2. Otherwise run **one** `eval_nn` forward pass with one bag per option and return
   `argmax_i policy[i]`.

This is the cheapest way to play and a clean sanity baseline, but the model only governs
non-trivial single-select decisions; everything else is heuristic.

---

## 8. Inference path B — MCTS

`tcg_model.py:mcts_agent(obs_dict, your_deck, model, value_mode="net")` is an AlphaZero-style
search. Conceptually: repeatedly walk from the root down to a leaf by a selection rule, expand
the leaf with the network, and back up the result; then play the action visited most often.

**Step 1 — determinise the hidden state** (`search_begin`). MCTS needs a *concrete* world to
simulate, but the opponent's hand/deck/prizes are hidden, so it fills them in:
- your deck & prizes: random samples of *your* known deck list;
- **opponent deck & hand: sampled from your own deck** — a **mirror assumption** (the opponent
  pilots the same deck);
- opponent prizes: an energy-card placeholder;
- opponent's face-down Active: card `677` = **Riolu** (the Basic in the Lucario line).

**Step 2 — expand a node** (`create_node`). For a non-terminal state it enumerates up to 64
candidate **actions** — combinations of exactly `maxCount` option indices — featurises them with
`get_decoder_input`, and calls the network once to get `value` and per-action `policy`. Priors
are `softmax(policy × 10)`. The leaf value is either the network's value head (`value_mode="net"`)
or a depth-capped random **rollout** that scores the leaf by prize-difference
(`_rollout_value`, `value_mode="rollout"`). Terminal nodes get `±1`/`0` directly.

**Step 3 — selection (PUCT)**. From a node, pick the child maximising

```
UCB(child) = Q(child) + c · prior(child) / (1 + visits(child)),   c = 0.4 · sqrt(visits(node))
```

where `Q` is the child's mean backed-up value, negated at the opponent's nodes (negamax) so the
search always maximises from *our* perspective. The chosen leaf is expanded, its value is
back-propagated up the path (`Node.backprop`), and the loop repeats `SEARCH_COUNT` times
(`tcg_model.SEARCH_COUNT`, default **10**; eval scripts override it via `--search-count`).

**Step 4 — act.** Return the root child with the **most visits** (deterministic; no temperature
or root noise).

`value_mode` matters: see [§10](#10-risks--limitations) for *why* `rollout` is often the better
choice here (`eval_mymodel.py --value-mode {net,rollout}`).

---

## 9. Worked example — one real turn with several actions

Captured from a real Lucario-mirror battle with our checkpoint
(`out/mymodel_lucario_bc_ce.pth`). It is **turn 1**, and we are the **first player** (`yourIndex
= 0`, `firstPlayer = 0`).

### 9.1 The state the model sees

```
turn=1  yourIndex=0  firstPlayer=0  energyAttached=False  supporterPlayed=False  retreated=False
YOU  Active = Solrock (id=676, 110/110 HP, no Energy)      Bench = []   deck=46  hand=7  prizes=6
OPP  Active = Lunatone (id=675, 110/110 HP)                Bench = []   deck=47  hand=6  prizes=6
YOUR HAND:
  [0] Lillie's Determination  (SUPPORTER)
  [1] Dusk Ball               (ITEM)
  [2] Basic {F} Energy        (BASIC_ENERGY)
  [3] Boss's Orders           (SUPPORTER)
  [4] Premium Power Pro       (ITEM)
  [5] Premium Power Pro       (ITEM)
  [6] Mega Lucario ex         (POKEMON, Stage 1, evolves from Riolu)
SELECT: type=MAIN  context=MAIN  minCount=1  maxCount=1   (choose exactly one action)
```

### 9.2 The legal options (and why these and only these)

```
[0] PLAY hand[1] Dusk Ball                       (ITEM)
[1] ATTACH Basic {F} Energy -> ACTIVE[0] Solrock
[2] PLAY hand[4] Premium Power Pro               (ITEM)
[3] PLAY hand[5] Premium Power Pro               (ITEM)   <- the duplicate copy
[4] END turn
```

Every absence is **rule-driven** (verified in [§11](#11-rules-consistency-notes)):
- **No bench play** — there is no *Basic* Pokémon in hand; Mega Lucario ex (hand[6]) is a Stage 1,
  so it can't be benched.
- **Mega Lucario ex can't be played** — it evolves from Riolu, but we have no Riolu in play, and
  you can't evolve on a Pokémon's first turn anyway.
- **Both Supporters are withheld** (Lillie's Determination, Boss's Orders) — *the player going
  first may not play a Supporter on their first turn* (rulebook line 518; the engine enforces it).
- **The Energy is an `ATTACH`, not a `PLAY`** — option [1]; you may attach once per turn.
- **No `ATTACK` option** — the first player skips the attack step on turn 1.

### 9.3 The exact features

The encoder turns this board into **24 words / 96 non-zero features**. A few decoded so the
mapping is unambiguous (`(index, value)` pairs; `card_count = 1268`):

```
word16 active_you : (7613, 0.275)  = Solrock HP 110/400
                    (8290, 1.0)    = Solrock identity  (= 7612 + 2 + card_id 676)
word18 player_you : (15224, 0.767) = deckCount 46/60
                    (15226, 0.875) = handCount 7/8
                    (15234, 1.0)   = "6 prizes remaining" one-hot
word23 meta       : (21596, 1.0)   = constant bias
                    (21597, 0.1)   = turn 1/10
                    (21598, 1.0)   = firstPlayer == you
words 0-15 (bench) : one "empty slot" flag each;  word22 (stadium): none
```

The decoder makes one bag per option (`decoder_card_offset = 1571`):

```
opt0 PLAY Dusk Ball            : (2673, 1.0)              = 1571 + 0*1268 + card 1102
opt1 ATTACH Energy -> Solrock  : (2845, 1.0)(4783, 1.0)   = energy source (1571+1268+6) + target Solrock (1571+2*1268+676)
opt2 / opt3 PLAY Power Pro     : (2712, 1.0)              = 1571 + card 1141  (identical -> identical bag)
opt4 END turn                  : (1, 1.0)                 = the END token
```

### 9.4 What the model outputs

One forward pass over this state and its 5 option-bags:

| Option | policy score | MCTS prior `∝ e^(10·score)` |
| --- | --- | --- |
| **[0] PLAY Dusk Ball** | **+0.498** | **0.9997** |
| [1] ATTACH Energy → Solrock | −0.315 | 0.0003 |
| [2] PLAY Premium Power Pro | −0.830 | ~0 |
| [3] PLAY Premium Power Pro | −0.830 | ~0 |
| [4] END turn | −0.889 | ~0 |

- **Value head:** `value = −0.479`. (Flagged — this should be roughly even on turn 1 of a mirror;
  see [§10](#10-risks--limitations).)
- **Greedy choice:** option `[0]` → **PLAY Dusk Ball** (a Poké-Ball Item that digs for Pokémon —
  a sensible "develop the board" play, and what the teacher tends to do).
- **MCTS choice** (`SEARCH_COUNT = 20`): option `[0]` with `value_mode="net"` **and** with
  `value_mode="rollout"` — both agree with greedy here.

Note how the `×10` sharpening turns a modest score gap (`+0.498` vs `−0.315`) into a near-one-hot
prior (`0.9997`), so the search spends almost all its budget confirming Dusk Ball. In closer
positions the prior is flatter and the search genuinely reshuffles the ranking; here it didn't
need to.

---

## 10. Risks & limitations

1. **The value head is miscalibrated.** Its training target is the outcome of a **mirror** game
   (teacher vs identical teacher ≈ 50/50), i.e. almost a coin flip uncorrelated with the board,
   so it learns little. The worked example shows it: `value = −0.479` on an even turn-1 position
   that should read ≈ 0. **Mitigation:** prefer `value_mode="rollout"` in MCTS (a random playout
   scored by prize difference is a more honest leaf estimate). Memory also records that *further
   training tends to regress* strength versus the untrained init — MCTS, not the value head, is
   where the strength comes from.
2. **Determinisation is hard-wired to the Lucario mirror.** `mcts_agent` samples the opponent's
   deck/hand from *your* deck and guesses a face-down Active as `677` (Riolu). That is correct
   for the mirror it's evaluated on, but **wrong against any other deck**, and it is coupled to
   `deck.csv` being the Lucario deck (the unrelated `tcg_model.sample_deck` is a *different*
   deck used by `eval.py`). Against a non-mirror opponent the search reasons about the wrong cards.
3. **MCTS always selects exactly `maxCount` options.** `create_node` enumerates `maxCount`-sized
   combinations only, so for a "select *up to* N" decision (`minCount < maxCount`) it can't
   propose taking fewer than `maxCount`.
4. **The policy head was trained on single-select decisions only.** Greedy `auto_resolve`s every
   other decision type heuristically (not with the model); MCTS *will* query the net on
   multi-select nodes, but those priors are effectively untrained (only the value back-ups refine
   them).
5. **It's a tiny model with shallow search.** `d_model=128`, 1 encoder + 1 decoder layer, 2 heads,
   `SEARCH_COUNT` default 10. Strength depends heavily on `--search-count` — so comparisons must
   hold it fixed across agents.
6. **Outputs are bounded scalars, not calibrated probabilities.** Read `value`/`policy` as
   preferences; the only place a distribution appears is the `softmax(policy × 10)` MCTS prior.
7. **Engine vs rulebook gaps exist — trust the engine.** E.g. `docs/rulebook.md` lists a *Fairy* energy
   type, but the engine's `EnergyType` enum has none; the engine also reports a 4th game-end reason
   ("card effect") beyond the rulebook's three. Where they differ, the `cg` engine decides.

---

## 11. Rules-consistency notes

Checked against `docs/rulebook.md` with the `tcg-rules-critic` recipe
(`grep -n '^## \|^### ' docs/rulebook.md` → read the section). Card facts via `all_card_data()`. The
engine is authoritative on conflict.

| Claim used above | Rulebook section | Verdict |
| --- | --- | --- |
| 3 ways to win: all Prizes / KO all opponent Pokémon / opponent can't draw | "How to Win the Game" | consistent (engine adds reason `4` = card effect) |
| Turn = draw → (bench / evolve / attach 1 Energy / Trainers / retreat / Abilities, any order) → attack → checkup | "Parts of a Turn" | consistent |
| One Energy attach, one Supporter, one Stadium, one retreat **per turn** | "Turn Actions" (C, D, E) | consistent (mirrors `State.energyAttached/supporterPlayed/stadiumPlayed/retreated`) |
| First player may **not** play a Supporter on turn 1; and skips the attack step on turn 1 | "Play Trainer cards" (line 518); "Attack and End Your Turn" (line 607) | consistent — explains the withheld Supporters & missing ATTACK in [§9](#9-worked-example--one-real-turn-with-several-actions) |
| You can't evolve a Pokémon on its first turn / on your first turn | "Evolve Pokémon" (lines 501–503) | consistent — explains why Mega Lucario ex is unplayable in [§9](#9-worked-example--one-real-turn-with-several-actions) |
| KO when total damage ≥ HP; KO grants the attacker **1 Prize** (Active only for Weakness/Resistance) | "Attack and End Your Turn" | consistent |
| **Mega Lucario ex gives up 3 Prize cards** when KO'd; it's a Stage 1 that evolves **from Riolu** | "Appendix 1: Mega Evolution Pokémon ex" (lines 951, 957) | consistent with `all_card_data()` (id 678: `megaEx`, `stage1`, `evolvesFrom=Riolu`, 340 HP) |
| Deck legality: 60 cards, ≤4 of a non-energy name, ≥1 Basic, ≤1 ACE SPEC | "Deck Building"; "Appendix 3: ACE SPEC Cards" | consistent — our deck's lone ACE SPEC is Hero's Cape ×1 |
| Special Conditions (Asleep/Burned/Poisoned/Paralyzed/Confused) affect only the Active and clear on bench/evolve | "Special Conditions" | consistent (mirrors the `PlayerState` status booleans) |

---

## Reproduce / explore

```bash
# Greedy & MCTS vs the teacher on the Lucario mirror (our checkpoint):
TORCHDYNAMO_DISABLE=1 python eval_mymodel.py \
    --ckpt out/mymodel_lucario_bc_ce.pth --games 60 --search-count 20 --value-mode rollout

# Smallest "engine runs" smoke test (random vs random):
python run_battle.py --games 5
```

To capture a decision like [§9](#9-worked-example--one-real-turn-with-several-actions): drive a
mirror battle with `rulebased_lucario.agent`, stop at the first decision where
`is_policy_select(obs.select)` holds and `len(obs.select.option) > 1`, then print the board,
`get_encoder_input`/`get_decoder_input`, and `eval_nn`. The engine reshuffles every run, so your
cards will differ — but the mechanics are identical.

**See also:** [`docs/cg-sdk.md`](docs/cg-sdk.md) (SDK contract — observation/option/card model,
Search API), [`docs/rulebook.md`](docs/rulebook.md) (game rules), `tcg_model.py` (the model/agent layer),
`train_mymodel_bc.py` (training), `eval_mymodel.py` (evaluation).
