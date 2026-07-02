# Execution flow — a layered mental model

A map of how this project runs, built so you can **hold it two steps at a time**, not all at once.

**How to read the step blocks:** each step is a small box —
**IN** (what arrives) · **DO** (what happens) · **OUT** (what leaves).
The **OUT of one step is the IN of the next**, so you only ever track two boxes.

**How this page is organized:** the part you see now (the map) is the whole project at the
highest level. Everything deeper lives in **click-to-expand boxes** below — they start
**collapsed**, so by default you only see the map. Open **one box at a time**, top-down, when
you want to zoom in. Boxes are self-contained, so some inputs are repeated across them on purpose.

---

## Flow 0: The big picture (the project lifecycle)

One idea runs through everything. A game state comes in, gets turned into numbers, a small
neural net (`MyModel`) scores the legal choices, and an agent returns which choice(s) to make:

```
obs (game state, a dict)  ->  features  ->  MyModel  ->  option indices (list of ints)
```

Everything talks to the game through the same **3 engine calls** (the `cg` simulator):

```
battle_start(deck0, deck1)  ->  battle_select(indices)  [many times]  ->  battle_finish()
```

And an **agent is just one function**: `agent(obs) -> list[int]`. It returns *option indices* —
positions in the list of legal choices — **never card IDs**.

Zoomed all the way out, the project is a loop of **make data → train → score**:

```
decks + card data        TRAIN (self-play battles)        EVAL (head-to-head)
  deck.csv / decks/*.csv     a trainer plays games           eval.py pits two
  sample_deck (in code)  ->  with MyModel, then       ->     agents, swaps seats,  ->  win% + CI
  card info from the cg      updates the net & saves         counts wins              + Elo
  SDK at runtime             out/<name>.pth                                           (the verdict)
```

*(Heavy training/eval runs are offloaded to Kaggle — see Flow 6.)*

Files:

| File | Role | In one sentence |
|---|---|---|
| `tcg_model.py` | the **brain** | turns an obs into features, runs `MyModel`, and wraps it as `mcts_agent` / `random_agent`; also holds `sample_deck`. |
| `utils/run_smoketest_battle.py` | the **driver** | plays one battle start→finish with `random_agent`; the smallest "engine works" test. |
| `train_ppo.py` (+ `ppo_model.py`, `eval_ppo.py`) | **training** (active) | clean-room PPO vs the rule-based field; independent of `MyModel` (see the file docstrings). |
| `train_nashpg.py` | training (retired) | self-play NashPG; superseded by `train_ppo.py` (removed — git history). |
| `starter_rl_kaggle.py` | training (legacy) | RL + MCTS self-play; the original trainer. |
| `train_reinforce.py` | training (retired) | plain REINFORCE; tried, regressed (removed — git history). |
| `eval.py` | **scoring** | pits two agents head-to-head and reports win rates + CI + Elo. |
| `eval_mymodel.py` | scoring (extra) | Lucario-mirror win-rate signal for the BC `MyModel`. |
| `kaggle/push_sweep.py` | **offload** | ships the `train_ppo.py` closure to Kaggle script-kernels for heavy runs. |

Where each loads its deck: `utils/run_smoketest_battle.py` reads a CSV (`--deck`, default `deck.csv`);
`eval.py` model-agents use `sample_deck` (a list baked into `tcg_model.py`); `train_ppo.py` /
`eval_ppo.py` load `decks/<name>.csv` per `--decks`;
`eval_mymodel.py` reads `deck.csv` (the Lucario deck).

---

## Execution flows

| Flow | Zoom level | What it shows |
|---|---|---|
| **1** | spine | the battle loop — start → decide/apply (repeat) → finish |
| **2** | concrete | one real game (seed 0), with actual values |
| **3** | the brain | obs → choice; nests **3a** features · **3b** model · **3c** MCTS |
| **4** | training | the 3 trainers; nests **4a** NashPG loop · **4b** one update · **4c** the others |
| **5** | eval | one match; nests **5a** play_match · **5b** the modes & extra tools |
| **6** | offload | running heavy jobs on Kaggle |

---

<flow1>
Flow 1: battle loop

`utils/run_smoketest_battle.py` is the simplest place to see it (`grep -n 'def play_one' utils/run_smoketest_battle.py`).

```
STEP 0 · Pick a deck
  IN   deck.csv          (60 lines, one card ID per line)
  DO   load_deck("deck.csv")
  OUT  deck = [673, 673, 674, ...]   (a list of 60 ints)
        |
        v   both players use this deck
```
```
STEP 1 · Start the battle
  IN   deck  (used twice — same deck both sides = a "mirror" match)
  DO   battle_start(deck, deck)
  OUT  obs   (the first game state, a dict)   + some StartData (errors, if any)
        |
        v   obs feeds the agent
```
```
STEP 2 · Agent decides
  IN   obs
  DO   look at obs["select"]["option"]  (the legal choices) and pick some
  OUT  indices   e.g. [4]               (which options to take)
        |
        v   indices feed the engine
```
```
STEP 3 · Engine applies the choice
  IN   indices   e.g. [4]
  DO   battle_select([4])
  OUT  next obs  (a new game state)
        |
        v   back up to Step 2  ...while obs["current"]["result"] < 0
```

**Steps 2 and 3 repeat** — decide, apply, decide, apply — until the game ends.

```
STEP 4 · Finish
  IN   the last obs   (its result is no longer -1)
  DO   battle_finish()        (frees memory; returns nothing)
  OUT  result = obs["current"]["result"]
        -1 = still going  ·  0 = player 0 won  ·  1 = player 1 won  ·  2 = draw
```

Small print (skip on first read):
- The agent returns **option indices, not card IDs**.
- It must return between `minCount` and `maxCount` indices, with **no repeats**.
- There's a special first-ever call where an agent returns a 60-card *deck* instead — but that
  never happens in `utils/run_smoketest_battle.py`, because both decks are handed over up front in Step 1.

</flow1>

<details>
<summary><b>▸ Flow 2 · A real game (seed 0)</b> — the same loop, with actual values</summary>

Captured from one real battle (both sides playing randomly). The full obs dict is large, so only
the key fields are shown. *The field values are exact, copied from the run; the plain-English
reads are interpretation, not checked against the rulebook.*

**Decision #1 — the opening.** The very first thing the engine asks:

```
current:  turn=0   yourIndex=0   result=-1
select :  type=YES_NO   context=IS_FIRST   minCount=1   maxCount=1
options:  [0] YES
          [1] NO        <-- agent chose this
returned indices: [1]
```
Read: a yes/no question (context `IS_FIRST` — reads as "take the first turn?"). Two options; the
random agent returned `[1]`, i.e. the option at index 1 = "NO".

**A mid-game decision.** A normal turn (`type=MAIN`):

```
current:  turn=1   yourIndex=1   result=-1
select :  type=MAIN   context=MAIN   minCount=1   maxCount=1
options:  [0] ATTACH  area=HAND index=3  -> onto ACTIVE(0)
          [1] PLAY    index=4
          [2] PLAY    index=5
          [3] ATTACH  area=HAND index=6  -> onto ACTIVE(0)
          [4] END                          <-- agent chose this
returned indices: [4]
```
Read: it's player 1's turn (`yourIndex=1`). Five legal options — attach a hand card onto the
active Pokémon, play a hand card, or just end the turn. The random agent returned `[4]` = END.

**The end.** This game ran **149 decisions** total, then:
```
final result = 0   (player 0 won)
```

So the loop in Flow 1 ran ~149 times (Step 2 ↔ Step 3), one decision each pass, until `result`
flipped from `-1` to `0`.

</details>

<details>
<summary><b>▸ Flow 3 · The brain: obs → choice</b> — how an agent actually decides</summary>

`random_agent` ignores everything and picks at random. The real agent, `mcts_agent`, runs the
data spine from Flow 0:

```
STEP A · obs -> features
  IN   obs
  DO   get_encoder_input(obs, deck)  +  get_decoder_input(obs, actions)
  OUT  two "SparseVectors"  (the board, and the candidate actions, as number-bags)
        |
        v
```
```
STEP B · features -> scores
  IN   the two SparseVectors
  DO   MyModel(...)   (called via eval_nn)
  OUT  (value, policy)
         value  = one number ~ -1..+1, "who's winning?"   (encoder head)
         policy = one score per candidate action           (decoder head)
        |
        v
```
```
STEP C · scores -> choice
  IN   value + policy scores
  DO   mcts_agent runs a small search (and "determinizes" the opponent's hidden cards)
  OUT  the option indices to play
```

`mcts_agent` slots straight into **Step 2 of Flow 1** — same IN (obs), same OUT (indices). It
just thinks harder than `random_agent`. The three nested boxes below zoom into A, B, and C.

<details>
<summary>▸ 3a · Features — obs → SparseVectors (Step A)</summary>

A **SparseVector** is not a dense array — it's a *bag of features* fed to a `torch.nn.EmbeddingBag`
(`grep -n "class SparseVector" tcg_model.py`). It carries three parallel lists:
`index[]` (which feature slots are "on"), `value[]` (the weight of each), and `offset[]` (which
splits the indices into **"words"**). The model sums the embedding rows in each word.
*Quirk: if an index repeats inside a word, the weights add up.*

```
STEP · the board -> encoder input
  IN   obs (full game state)  +  your_deck (60 ids)
  DO   get_encoder_input(obs, your_deck)   -- walk every zone from YOUR view (i ^ your_index)
  OUT  a SparseVector with EXACTLY 24 words:
         16 bench  +  2 active  +  2 player  +  1 hand  +  1 deck  +  1 stadium  +  1 global
```
```
STEP · the choices -> decoder input
  IN   obs  +  actions = [[0],[1],...,[n-1]]   (one entry per legal option)
  DO   get_decoder_input(obs, actions)   -- encode "what kind of action, on which card"
  OUT  a SparseVector with ONE word per candidate action (up to 64)
```

Concrete values:
- **Global word**, for `turn=3` and you're *not* the first player → `value = [1.0, 0.3]` (a bias
  `1`, then `turn/10`; the "am I first?" feature is 0 so it drops out).
- **Deck word**: one `(card_id, 0.25)` entry per deck card. Because duplicates sum, a card that
  appears 33× lands at weight ≈ `33 × 0.25 = 8.25` — so one 128-dim vector encodes the whole
  deck list.
- **Decoder, a MAIN decision with 12 options** → 12 words; e.g. an `END` option's word is the
  single index `1`, an `ATTACK` with attackId 57 is index `14+57 = 71`.

Constants (`grep -n "num_words_encoder\|encoder_size" tcg_model.py`): `num_words_encoder = 24`,
`encoder_size = 22000` (the flat feature vocabulary).

</details>

<details>
<summary>▸ 3b · MyModel forward — SparseVectors → (value, policy) (Step B)</summary>

```
STEP · run the net
  IN   the two SparseVectors (board + candidate actions)
  DO   MyModel.forward(...)   -- encoder reads the 24 board words; decoder reads the action
       words and cross-attends to the board
  OUT  (value, policy)
         value  : shape (batch, 1)        -- tanh, in [-1, 1]   "who's winning?"
         policy : shape (batch, n_actions)-- tanh, one score per action, each in [-1, 1]
```

The model is built the same everywhere: `MyModel(128, 2, 256, 1, 1)` =
`d_model=128, heads=2, feed-forward=256, encoder-layers=1, decoder-layers=1`
(`grep -n "class MyModel" tcg_model.py`).

`eval_nn(sv_enc, sv_dec, model)` is the thin batch-of-1 wrapper used at decision time
(`grep -n "def eval_nn" tcg_model.py`): it packs the two SparseVectors into tensors, calls the
model, and returns `(value: float, policy: list[float])`.

*Note: the policy scores aren't probabilities yet. Everywhere in the codebase the convention is
`logits = policy × 10` (`LOGIT_SCALE = 10.0`), so a probability is `softmax(policy × 10)`.*

</details>

<details>
<summary>▸ 3c · mcts_agent internals — search + determinization (Step C)</summary>

`mcts_agent(obs, your_deck, model, value_mode="net")` (`grep -n "def mcts_agent" tcg_model.py`)
returns `(select, sample)` — `select` is the chosen list of option indices (the move), `sample`
is a training record.

```
STEP · guess the hidden cards (determinize)
  IN   obs  +  your_deck
  DO   the opponent's hidden cards are unknown, so fill them in by guessing:
         - opponent deck/hand  <- random sample from YOUR deck (assume a mirror match)
         - opponent hidden prizes  <- energy placeholder, card id 1
         - opponent face-down active  <- Riolu, card id 677
  OUT  a fully-known hypothetical state  ->  search_begin(...)
```
```
STEP · search a little
  IN   the hypothetical state
  DO   create_node (enumerate up to 64 action combos, call the model once for value+policy,
       set child priors = softmax(policy*10)); then SEARCH_COUNT = 10 PUCT iterations, each
       stepping the hypothetical with search_step(...) and backing up the leaf value
  OUT  visit counts on each root child
```
```
STEP · pick + finish
  IN   the searched tree
  DO   choose the MOST-VISITED root child; search_end() releases the search
  OUT  select = that child's option indices   (+ a LearnSample for training)
```

*Accuracy note:* the determinization above is the **live code** (mirror sampling + Riolu `677` +
energy `1`), and `docs/cg-sdk.md` describes the same scheme.
(`grep -n "search_begin\|677\|random.sample" tcg_model.py`.)

</details>

</details>

<details>
<summary><b>▸ Flow 4 · Training (legacy MyModel trainers)</b> — self-play games → a better MyModel</summary>

**This whole flow is historical**: all three MyModel trainers below are retired or legacy; the
active trainer is `train_ppo.py` (a separate, fresh-net PPO stack — see its docstring and
`ppo_model.py`). Kept because the shared spine and the act-function evolution still explain the
codebase's story.

There are three trainers. They share **one spine**: play battles with the current model, then use
who-won to nudge the model, then save a checkpoint.

```
THE SHARED SPINE
  battle_start(deck, deck)
  while result < 0:
      decide a move (act) ; battle_select(...)     <- record what happened
  battle_finish()
  -> use the result to update MyModel -> save out/<name>.pth
```

The "decide a move" step is what evolved across trainers (this *is* the project's story):

```
mcts_agent      ->   act_sample        ->   act_collect
(search; older)      (sample a move;        (sample a move at full
                      train_reinforce)       randomness AND store the
                                             info PPO needs: old_logp + V_old)
```

`train_nashpg.py` was the last MyModel trainer (retired; removed — git history); boxes 4a/4b zoom
into it. 4c covers the other two.

<details>
<summary>▸ 4a · NashPG — the outer / inner / reset loop (train_nashpg.py)</summary>

Setup: build a fresh `model` (θ); make a **frozen copy** `ref` (ρ); snapshot a frozen `init` for
later comparison. Then two nested loops:

```
STEP a · Collect (inner step)
  IN   model, deck
  DO   play `--games` self-play battles; collect_game(...) records every decision
  OUT  buffer = list[Record]   (each Record: the features, the chosen option, old_logp, V_old,
                                and after the game: ret + adv)
        |
        v
```
```
STEP b · Update
  IN   buffer, model (theta), ref (rho)
  DO   ppo_update(...)   -- one PPO step (see box 4b)
  OUT  a slightly-changed model + a metrics row -> out/<prefix>_metrics.jsonl
        |
        v   repeat a->b for `--inner` steps, then:
```
```
STEP c · Reset the anchor (once per OUTER iter)
  IN   model (theta)
  DO   ref.load_state_dict(model.state_dict())     -- rho <- theta
  OUT  a refreshed frozen anchor
        |
        v
```
```
STEP d · Save + eval
  IN   model
  DO   torch.save(...) ; eval_checkpoint(...)  vs frozen init + the rule-based teacher
  OUT  out/<prefix>{t}.pth  + eval lines      (repeat the whole thing for `--outer` iters)
```

Produces: `out/<prefix>_init.pth`, per-outer `out/<prefix>{t}.pth`, `out/<prefix>_final.pth`, and
`out/<prefix>_metrics.jsonl`. Defaults: `--outer 2 --inner 2 --games 4 --epochs 1
--batch-size 64`, lr `3e-4`, KL coeff α `0.2` (fixed), clip `0.2`, value-coeff `0.5`,
entropy `0.01`, lam `1.0`. Reward = terminal `+1 / -1 / 0` only.

**Real numbers** (from `work_log.txt`, no run needed):
- Smoke `--outer 2 --inner 2 --games 4`: finite loss each update, `clip_frac` ~0.1–0.3,
  `explained_var` 0.06→0.52, entropy ~0.5, ~1 game/s.
- A real run (`6/4/128`, epochs 2, lam 0.95, entropy 0.05): ~1.7 games/s, ~50 min.
- SPRT check, `a0_final` vs its own `a0_init`: ACCEPT H1, `33/2 = 94.3%` over 35 games → it learns.

</details>

<details>
<summary>▸ 4b · NashPG — what one update does (collect → returns → PPO+KL)</summary>

```
STEP · collect one game
  IN   model, deck
  DO   collect_game(...): run the battle; at each of YOUR decisions, act_collect(...) samples a
       move from softmax(policy*10) and stores a Record with old_logp + V_old
  OUT  per-seat lists of Records
        |
        v
```
```
STEP · turn results into targets
  IN   one seat's Records + that seat's terminal G (+1 win / -1 loss / 0 draw)
  DO   compute_gae(recs, G, lam)
         lam = 1.0 (smoke): adv = G - V_old,  ret = G       (simple Monte-Carlo)
         lam = 0.95 (real): the GAE smoothing of the same idea
  OUT  every Record now has .adv and .ret
        |
        v
```
```
STEP · the PPO + KL update
  IN   the buffer of Records, model (theta), ref (rho)
  DO   ppo_update(...): for each minibatch, run theta AND frozen rho; build masked logits
       (mask = -1e9); the loss is
            L = L_clip  +  vf * value_loss  -  ent * entropy  +  alpha * KL(theta || rho)
       (PPO-clipped policy term, a value term, an entropy bonus, and the KL pull toward the
        frozen anchor); clip grads to 0.5; opt.step()
  OUT  an updated model + metrics (ppo_loss, entropy, approx_kl, clip_frac, mag_kl, explained_var)
```

The KL-to-frozen-ρ term (and the periodic ρ←θ reset in 4a) is the one thing NashPG adds over
plain PPO — it keeps self-play from cycling into an exploitable policy.

</details>

<details>
<summary>▸ 4c · The other two trainers (older / experiment)</summary>

```
starter_rl_kaggle.py  (the original; uses MCTS)
  IN   model
  DO   play self-play games with mcts_agent; label each decision by who won, blended backward
       (LAMBDA=0.9, value targets ±1); train MyModel in batches of 128
  OUT  out/<prefix>{N}.pth (per iter) + out/<prefix>_final.pth
```
```
train_reinforce.py  (plain REINFORCE; a tried experiment that regressed)
  IN   model
  DO   play games with act_sample; reward = prize-difference / 6  (a "gameable" proxy);
       update toward higher-reward moves
  OUT  out/<prefix>{it}.pth
  WHY IT MATTERS  it stores NEITHER old_logp NOR V_old — the exact gap that act_collect (4b)
                  fixes. The prize-diff reward let the policy "cheat" the metric -> it regressed.
```

Defaults: `starter_rl_kaggle.py` → `--iterations 5 --selfplay-games 100 --batch-size 128`;
`train_reinforce.py` → `--games 200 --iterations 4 --tau 0.5 --beta 0.01`.

</details>

</details>

<details>
<summary><b>▸ Flow 5 · Eval</b> — is one agent better than another?</summary>

`eval.py` answers one question: *did the agent get better?* It runs the Flow-1 battle loop between
**two** agents, swaps seats so first-player luck cancels, and turns the wins into a win% with a
confidence interval and a rough Elo.

```
STEP · set up two agents
  IN   two checkpoints / baselines  (e.g. a model + "random", or two checkpoints)
  DO   make_model(path) / make_random() / make_rulebased()  -> each is {name, fn, deck}
  OUT  agent A, agent B
        |
        v
```
```
STEP · play N seat-swapped battles
  IN   A, B, n
  DO   play_match(A, B, n)  -- run the battle loop, alternating who is seat 0 each game
  OUT  (wins_A, losses_A, draws)
        |
        v
```
```
STEP · report
  IN   the W/L/D counts
  DO   wilson(...) for a 95% CI ; anchored Elo over the pool
  OUT  a printed line, e.g.
       W/L/D = 31/27/2   win% = 53.4%  (95% CI 40.8-65.6, n=58 decisive, 2 draws)
```

Every run also prints a JSON **manifest** (checkpoint hashes, search-count, decks, git commit) so
two runs are only compared when their settings match.

<details>
<summary>▸ 5a · One match — play_match & the two-agent driver</summary>

```
STEP · one battle between two agents
  IN   A, B, seat_a (0 or 1)
  DO   _play_one(A, B, seat_a): same battle_start -> while result<0: battle_select -> battle_finish
       loop as utils/run_smoketest_battle.py, BUT each decision is routed to the right agent by
       obs["current"]["yourIndex"]  (== seat_a ? A : B)
  OUT  result  (0/1 = that seat won, 2 = draw)
        |
        v
```
```
STEP · many battles, seat-swapped
  IN   A, B, n
  DO   play_match: for i in range(n): seat_a = i % 2 ; _play_one(...) ; tally A-relative W/L/D
  OUT  (wins_A, losses_A, draws)
```

Grep: `grep -n "def _play_one\|def play_match" eval.py`. The same shape (route by `yourIndex`,
seat-swap by `i % 2`) appears in `eval_mymodel.py::play` and, with a single agent, in
`utils/run_smoketest_battle.py::play_one`.

</details>

<details>
<summary>▸ 5b · The modes & the other eval tools</summary>

`eval.py --mode` (default `battery`; `--games 20`, `--baseline-games 200`, `--search-count 4`,
`--agent-kind {mcts,sampled}`):

```
battery  (default)  a sanity battery of matchups (random-vs-random ~50% validates the harness;
                    rule-based > random validates ordering) + an Elo table; or, with
                    --checkpoints, a clean mirror round-robin.
gauntlet            one --checkpoint vs a FROZEN anchor + the rule-based field -> one comparable
                    headline scalar (win% vs anchor => Elo).
sprt                a sequential ship/no-ship A/B test of two checkpoints — stops as soon as the
                    evidence is decisive. e.g. "VERDICT: ACCEPT H1 — model:A is BETTER (ship)."
```

The other eval tool:
```
eval_mymodel.py    Lucario MIRROR (deck.csv both sides, search-count 10); "is this checkpoint any
                   good on a clean skill test?"  -> win% + CI per agent×opponent cell.
```

Watch the **deck axis**: `eval.py` model-agents play `sample_deck`; `eval_mymodel.py` plays
`deck.csv` (Lucario). Win rates across different decks (or different `--search-count`) aren't
comparable — lock both when comparing.

</details>

</details>

<details>
<summary><b>▸ Flow 6 · Heavy runs on Kaggle</b> — offloading big jobs</summary>

Local hardware is small (WSL2), so heavy training/eval is pushed to Kaggle. Driver:
`kaggle/push_sweep.py` (subcommands `stage / sweep / status / pull / report`).

```
STEP · stage
  IN   the eval/train code + decks (no checkpoints -- PPO trains from a fresh net)
  DO   bundle them into a PRIVATE Kaggle dataset (outside the submission boundary)
  OUT  a staged dataset
        |
        v
```
```
STEP · sweep
  IN   the staged dataset
  DO   push script-kernels (run_ppo.py -> runs train_ppo.py, --out-dir /kaggle/working)
  OUT  kernels running on Kaggle CPUs
        |
        v
```
```
STEP · status / pull / report
  IN   the running kernels
  DO   poll, then download results
  OUT  results in kaggle/out_kaggle/  + a training-curve summary (verdict = local eval_ppo.py)
```

Real lessons (`work_log.txt` / README "Running heavy jobs on Kaggle"):
- Kaggle caps **5 concurrent** CPU sessions — push in waves of ≤5.
- Failed pushes leave **zombie kernel records**; reusing that slug then fails — use fresh slugs.
- Each CPU kernel runs ~1–1.5 h; the CLI has **no kernel-delete** (remove zombies via the web UI).

</details>
