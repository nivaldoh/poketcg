# Plan: Multi-select for the SB3 stack

## Reference — Observation, SelectData, and key terms

Read this once; later sections point back here instead of re-explaining.

**The observation an agent sees.** `agent(obs_dict)` receives an `Observation`
(`vendor/cg-lib/cg/api.py`) with four top-level fields:

- `select` — the current decision, a `SelectData` (see below), or `None` on the very
  first call (deck selection).
- `current` — the board `State`: `turn`, `result` (winner index, `-1` while live),
  `yourIndex`, and `players[2]`. Each `PlayerState` has `active`, `bench`,
  `hand`/`handCount` (opponent's is a count only), `prize`, `discard`, `deckCount`,
  and the five status booleans (`poisoned`/`burned`/`asleep`/`paralyzed`/`confused`).
  Opponent hidden zones surface as counts / facedown `None`s.
- `logs` — events since the last decision, including the RESULT log (reason
  `1`=prize, `2`=deck-out, `3`=no-bench, `4`=effect).
- `search_begin_input` — the Search-API determinization token (MCTS only; not used
  by the SB3 stack).

**`SelectData`** (`obs.select`) has ten fields; the ones this design uses:

- `minCount` / `maxCount` — how many options we **must** / **may** choose. Both are
  always-present `int`s (no default, never `None`) whenever `select` is not `None`.
  `minCount` can be `0`; `maxCount` never exceeds `len(option)`. They are absent
  *only* at the first deck-selection call, where `select is None`.
- `option` — the list of legal `Option`s. Each `Option` has a `type` (one of 17
  `OptionType`s — see the category table in §3c) plus fields locating the card /
  attack it refers to.
- (`type`, `context`, `remainDamageCounter`, `remainEnergyCost`, `deck`,
  `contextCard`, `effect` round out the ten; not central here.)

We answer by returning a **list of option indices** into `option`. The engine
validates that list as a **set**: length in `[minCount, maxCount]`, every index in
range, and **no duplicates** (`battle_select([1,1])` is rejected — the Search API
labels this "Duplicate select elements"). It applies **no ordering rule** — `[a,b]`
and `[b,a]` are the same answer — with one semantic exception noted in §3f.

**Glossary.**

- **node** — one decision the engine hands us: one `obs.select` on our seat's turn.
- **slot** — a fixed position `0..N_MAX-1` in the action space / option array.
- **row** — one **feature vector** in the `(N_MAX, OPTION_DIM)` options array
  describing the option at that slot (see §3a). Not an integer — a vector of floats.
- **STOP** — the reserved commit action at slot `N_MAX-1` (§3a).
- **picked** — the indices chosen so far *within the current node* (§3b).
- **auto-commit** — automatically submitting `picked` once it hits `maxCount` (§3b).
- **learnable node** — a node the policy is allowed to decide (§3c); everything else
  is handled by the `auto_resolve` heuristic.

---

## 1. The problem

**Where we are today.** The learner (MaskablePPO) chooses actions from an
`action_space = Discrete(N_MAX)` (`N_MAX = 128`). This is a **fixed-width positional
head**, not a truncation of what we can pick: SB3's rollout buffer needs a fixed
observation/action shape every step, so the variable-length option list is *padded* to
128 slots. Each step the policy emits **one integer** — an index into the current
option list. A node with more than 128 options (the widest ever observed was 65) is
handed to the `auto_resolve` heuristic instead; the alternative to a fixed positional
head is a variable-length / custom per-option head, which is the planned upgrade (§8,
flag **F6**).

The consequence: the policy can only ever answer **"pick exactly one of these."**
Anything that asks for *zero, two, or a range* of picks is handed to `auto_resolve`
and the policy **never sees it**.

**What a decision looks like.** See the reference box above for the full obs and
`SelectData`. The parts that matter here are `minCount`/`maxCount` (how many to pick)
and `option` (the choices), and the fact that we answer with a **set of positional
indices** — order does not matter (§3f).

**Because `minCount`/`maxCount` are always present, we always know a node's shape from
the `SelectData` alone.** The four shapes below are a **coarse partition of the
`(minCount, maxCount)` space** — convenient labels, not an engine-enforced set. The
only hard rules the engine imposes are `0 ≤ minCount`, `maxCount ≤ len(option)`, and
the set-validation above. So an "optional-multi" node (`0..k`, `k>1`) is perfectly
legal; it just falls in the "true variable" bucket.

| Shape | `min..max` | Example | Valid answers | What the policy does today |
|---|---|---|---|---|
| **Single-select** | `1..1` | "Which action this turn?" options `[attack, attach energy, play Riolu, …, END]` | `[k]` | **Learns it** (`is_policy_select`) |
| **Fixed-k** | `2..2` | "Discard 2 cards" from `[A,B,C,D]` | any 2 distinct indices, e.g. `[0,3]` | `auto_resolve` → first 2: `[0,1]` |
| **Optional-single** | `0..1` | "You may attach 1 Energy from discard" from `[E1,E2]` | `[]`, `[0]`, or `[1]` | `auto_resolve` → **`[0]`** (never declines, never picks E2) |
| **True variable** | `min<max, max>1` | "Discard 1–3 cards" / "attach 0–2" | any distinct set with size in `[min,max]` | `auto_resolve` → first `min` (or first `max` if `min==0`) |

So every multi-card decision is invisible to the learner.

**Measured scale (one 120-game rulebased sample — approximate, and see the caveat).**
Multi-select is ~10.8% of non-trivial decisions and 100% auto-resolved today.
Breakdown: optional-single `[0..1]` ≈ **940** (the big one — for `[0..1]`,
`auto_resolve` does "option 0"), fixed `k=2` ≈ **418**, true variable
(`min<max, max>1`) ≈ **490**, all-NUMBER multi = **0**. So `[0..1]` + fixed-k ≈ **73%**
of multi nodes.

> **Caveat — these numbers are not reproducible from committed code.** No shape-census
> script exists in the repo; the 940/418/490 counts live only in this doc. (`N_MAX`'s
> own comment cites a *separate* 48-game round-robin, where the 65-option node was
> seen.) §5 plans a small committed census script to make them reproducible and to
> settle the §3c open question.

**Goal:** let the policy learn multi-card decisions.

---

## 2. Why not just use a multi-select action space?

`MultiDiscrete` / `MultiBinary` are natively supported by MaskablePPO, but they are
**independent factored heads**: each sub-choice is sampled independently and its
mask is fixed *before* sampling. That means they **cannot** express the two
constraints every multi-select needs:

- **distinctness** — a factored `MultiDiscrete` could easily emit `[1,1]`, which the
  engine rejects.
- **a min..max cardinality window** — "pick between 1 and 3" couples the choices;
  independent per-component masks can't enforce a total count.

**Phase 1 keeps the stock policy** (stock `MultiInputPolicy`, no custom network) for
simplicity; the custom per-option head is a **planned** later step, not a mere
maybe (§8, **F6**). Given the stock policy, the faithful approach is **sequential
decomposition**: turn one multi-select node into a short *sequence* of masked
single-select sub-steps, each an ordinary `Discrete(N_MAX)` decision realized as a
real env transition. Because each sub-step is just a normal single-select step, SB3's
existing log-prob / advantage / masking machinery applies unchanged — we reuse the
single-select path, not invent a new head. (The custom autoregressive head inside the
network is the real alternative; it means writing a custom policy, so it's deferred —
§8, **F6**.)

Worked example — **"discard 2 of `[A,B,C,D]`"** becomes:

```
sub-step 1: options [A,B,C,D] + STOP. STOP is masked off (must reach min=2 first).
            policy picks B (index 1).  picked = [1].  reward 0, present next obs.
sub-step 2: options [A,B,C,D] + STOP, but B (index 1) is now masked off (already
            picked). policy picks D (index 3).  picked = [1,3].
            len(picked) == maxCount == 2  ->  AUTO-COMMIT: battle_select([1,3]).
```

The machinery guarantees a valid answer (2 distinct indices) by construction:
distinctness comes from masking already-picked options; cardinality comes from the
min/max mask logic plus a STOP action (§3a–b).

---

## 3. Design

### 3a. STOP and the positional head — a fixed reserved top slot

**How the positional head reads options.** The `options` observation is a 2-D array of
shape `(N_MAX, OPTION_DIM) = (128, 70)`. **Row `i` is a 70-float feature vector**
describing the option at slot `i` (its `OptionType` one-hot, the card's static stats,
an attack block, etc. — built by `_option_row` in `sb3_features.py`) — *not* a single
integer. The stock policy's action head is a `Linear(latent, N_MAX)`: it emits **one
scalar logit per slot**, and slot `i`'s logit means "how good is picking the option in
row `i`?"

Concretely, at a MAIN node with options `[attack, attach energy, play Riolu, END]`
(`n=4`):

```
slot:   0        1              2            3     4..126   127
option: attack   attach energy  play Riolu   END   (none)   STOP
logit:  z0       z1             z2           z3    z4..z126 z127
mask:   T        T              T            T     F        F (single-select: STOP off)
```

The net scores `z0..z3`; `action_masks()` unmasks only the real slots (and STOP when
applicable), so sampling is confined to legal options. At a *different* node — say a
DISCARD node with options `[cardA, cardB, cardC]` — slot 0 now means "cardA." The
slot's meaning shifts node-to-node; that pre-existing weakness is flag **F6**,
unchanged by this work.

**Where STOP goes.** Multi-select needs one extra action: **"I'm done, commit what
I've picked."** We permanently reserve the **top slot**, `STOP_INDEX = N_MAX-1 = 127`,
for it. This is *one* reserved slot — not "indices beyond the options," and not per-pick
slots. Real options keep indices `0..n-1`, so `battle_select([i])` needs **no index
remapping**. STOP is *always* slot 127, so the net can learn one stable "commit"
meaning. Real option counts top out around 65 (observed), so reserving slot 127 costs
nothing. The *number* of picks is not encoded by extra slots — it is tracked by the
sub-step loop's `picked` list and surfaced through the `num_picked` / `remaining_*`
features (§3d).

**Marking STOP's row.** Row 127 carries an **`is_stop = 1`** feature bit. This is a
**static marker, not a toggle**: `is_stop` is `1` on row 127 and `0` on every real
option row, in *every* node — single-select and multi-select alike. It never changes
mid-node. It just lets the net tell the STOP row apart from a real option and from an
empty padding row. What *does* change per sub-step is whether STOP is **unmasked**
(selectable) — that's an `action_masks()` decision (§3b), independent of the marker.

Note this *doesn't compound* the positional-head weakness: while we resolve one
multi-select node, the option list is fixed across sub-steps (same 4 cards to discard);
we only mask out the ones already picked. So "slot `i`" means the same option across
all sub-steps of that node.

### 3b. One unified sub-step engine (single-select is the degenerate case)

We route **every** learnable decision through the same sub-step loop. A "node" is
**one `obs.select` on our seat's turn** (see glossary). While resolving a node we
track:

```
sel     = the SelectData
n       = len(sel.option)
mn, mx  = sel.minCount, sel.maxCount
picked  = indices chosen so far in THIS node   (starts empty)
p       = len(picked)
```

**Mask, recomputed each sub-step:**

- A real option `i` is selectable  ⇔  `i not in picked`  and  `p < mx`.
- STOP (slot 127) is selectable  ⇔  `p >= mn`  and  `p < mx`.

**Transitions:**

- **Pick a real option `i`:** append `i` to `picked`.
  - If now `p == mx` → **auto-commit**: `battle_select(picked)`, advance the engine.
  - Else → emit the next sub-observation (**reward 0**), with `i` newly masked out.
- **Pick STOP:** `battle_select(picked)` (this is `[]` when `mn == 0` and nothing was
  picked — a *decline*), advance the engine.

**Do we need both the `maxCount` ceiling *and* a STOP token?** Yes — they cover
different cases and neither replaces the other:

- **`maxCount` (auto-commit)** handles *"you have to stop now."* Once `p == mx` there
  is no legal move left (you can't pick more without exceeding `maxCount`), so we
  submit automatically rather than forcing a pointless extra STOP step.
- **STOP** handles *"you may stop early, by choice."* It is only offered while
  `mn ≤ p < mx` — i.e. only when the count is genuinely up to the learner (optional or
  true-variable nodes). For fixed-k (`mn == mx`) STOP is **never** offered: you pick
  `k` times and it auto-commits.

**Why emit a next sub-obs after a non-final pick?** The node isn't resolved yet — we
still owe more picks — so we present the *same* node again as a fresh observation
(picked option masked out, picked-count features updated) and let the policy make the
next pick. Reward is 0 because reward is paid only at the end of the battle
(terminal-only; unchanged).

**End-to-end trace with real values.** Two nodes through the one loop (mask shown over
the relevant slots; slots `4..126` are always padding-masked and omitted):

```
Fixed-k  "discard 2 of [A,B,C,D]"   n=4, mn=2, mx=2
  sub-step 1  picked=[]     state: in_multi=1 num_picked=0 remaining_min=2 remaining_max=2
              mask  slots 0 1 2 3 =T T T T | STOP(127)=F   (p=0 < mn=2 -> STOP off)
              rows 0..3 already_picked=0 ; row127 is_stop=1
              policy picks slot 1 (B) -> picked=[1], p=1 != mx -> next sub-obs, reward 0
  sub-step 2  picked=[1]    state: num_picked=1 remaining_min=1 remaining_max=1
              mask  slots 0 1 2 3 =T F T T | STOP=F        (slot 1 already picked)
              row 1 already_picked=1
              policy picks slot 3 (D) -> picked=[1,3], p=2 == mx -> AUTO-COMMIT
              battle_select([1,3])

Optional "you may attach 1 energy of [E1,E2]"   n=2, mn=0, mx=1
  sub-step 1  picked=[]     state: in_multi=1 num_picked=0 remaining_min=0 remaining_max=1
              mask  slots 0 1 =T T | STOP=T                (p=0 >= mn=0 and p < mx -> STOP on)
              -> if policy picks STOP  : battle_select([])   (decline)
              -> if policy picks slot 0: picked=[0], p=1 == mx -> AUTO-COMMIT battle_select([0])
```

How the four shapes flow through this one loop:

```
single-select [1..1], n=3:  p=0 pick -> p=1==mx -> auto-commit [a].   STOP never offered.
optional      [0..1], n=2:  p=0: may pick OR STOP(decline). pick -> p=1==mx -> auto-commit.
                            STOP at p=0 submits [].
fixed-k       [2..2], n=4:  p=0 pick; p=1 pick; p=2==mx -> auto-commit.  STOP never offered.
variable      [1..3], n=5:  p=0 pick; p>=1 -> STOP or pick; ... p=3==mx -> auto-commit.
              (Phase B)     policy chooses to stop at 1, 2, or 3.
```

Single-select is provably the degenerate `mn==mx==1` case: one pick → auto-commit
`[a]`, STOP never offered. Its **engine interaction and action semantics are identical
to today**; the only difference is the observation now carries a few extra fields that
are all zero for single-select. Folding it into the unified loop removes the separate
single-select code path, so train and eval can't drift on two different code paths.

### 3c. Which nodes to surface — `is_learnable_select`

New predicate (in `sb3_env.py`) that decides what the policy sees, replacing the
`is_policy_select` gate the env and `sb3_agent` use today. A node is learnable iff
**all** of:

- `len(option) > 1` — a real choice (more than one thing to pick from),
- **not** all-NUMBER — leave "how many" nodes on `auto_resolve`,
- `maxCount >= 1`,
- `minCount < len(option)` — reject "you must pick all of them," which has no
  discretion (e.g. `min==max==3` with exactly 3 options → the only legal answer is
  `[0,1,2]`).

**A "not learnable" node is exactly one with no real discretion** — `≤1` option, "must
pick all," or all-NUMBER "how many." `auto_resolve` handles those deterministically and
the policy never sees them; there is nothing worth learning there.

This is a **strict superset** of `is_policy_select`: every single-select policy node
(`min==max==1`, `len>1`, not all-NUMBER) still qualifies, so single-select behavior
is unchanged; multi-select nodes now *additionally* qualify. `is_policy_select` in
`tcg_model.py` stays as-is for any other caller.

**Category enumeration — have we exhausted the possibilities?** Two orthogonal axes
define a node: the `SelectType` (what kind of selection) and the `(minCount, maxCount)`
window (its cardinality). Both are enumerable, so exhaustiveness is checkable.

*The 17 `OptionType`s and where they land:*

| Handling | OptionTypes |
|---|---|
| **Learnable** (surfaced to the policy when the shape allows) | PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, CARD, TOOL_CARD, ENERGY_CARD, ENERGY, YES, NO, SKILL, SPECIAL_CONDITION |
| **Always auto-resolved** | NUMBER ("how many" — its own `SelectType.COUNT` node; `auto_resolve` takes the largest `.number`) |

*The 11 `SelectType` node kinds:* MAIN, CARD, ATTACHED_CARD, CARD_OR_ATTACHED_CARD,
ENERGY, SKILL, ATTACK, EVOLVE, COUNT, YES_NO, SPECIAL_CONDITION. `COUNT` is the
all-NUMBER kind (auto-resolved); `SKILL` includes the order-dependent case (§3f). The
rest carry the learnable OptionTypes above.

*The `(minCount, maxCount)` shape partition* is the four rows of §1's table, with
"true variable" absorbing every `min<max` case including `0..k`. Combined with the
predicate, that is the complete map from node → learnable vs auto-resolved.

> **Open question (flag):** `len > 1` **excludes** a `[0..1]` node with a *single*
> option ("attach this energy — or not"), which is still a real do-or-decline choice
> and part of the EV story for the 940 `[0..1]` nodes. If a meaningful share of those
> 940 are single-option, relax the `[0..1]` branch to `len >= 1` — the STOP/decline
> machinery already handles `len==1` fine. **This is unresolved because the measurement
> that would settle it is not reproducible** (§1 caveat). Resolution: the §5 census
> script will tally single-option `[0..1]` nodes; make the `len>1` vs `len>=1` call
> from that data before locking the predicate.

### 3d. Extra features (append-only, fixed shape)

The net needs to *see* the multi-select context to choose well. We add a few small
features, **always present** (SB3's rollout buffer needs one fixed observation shape
every step), zeroed for single-select and at reset. They ride the two data-carrying
keys of the gym obs `Dict` — `state` and `options` (the third key, `action_mask`, is
the network's mask input and is unchanged in width):

- **2 per-option bits** (each `options` row gains): `is_stop` (1 on the STOP row) and
  `already_picked` (1 if this option was chosen earlier in this same node).
- **4 board-level numbers** (appended to `state`): `in_multi` (are we mid-multi-select?),
  `num_picked`, `remaining_min`, `remaining_max` (how many more we must / may pick).

So `OPTION_DIM` grows by 2 (70 → 72) and `STATE_DIM` by 4 (299 → 303).

**The encoding is not skipped — it is extended.** `obs_to_gym` still calls the base
encoders `encode_state(o)` / `encode_options(o)` (in `sb3_features.py`) exactly as
today, then **appends** the new fields. Those base encoders see only the static
`Observation`; they do **not** know `picked` or the running counts — that state lives
in the `CGEnv` sub-step loop. So the natural home for the new features is where the
state exists: the extended `obs_to_gym` and `CGEnv`, both in `sb3_env.py`. We leave
`encode_state` / `encode_options` / `STATE_DIM` / `OPTION_DIM` in `sb3_features.py`
untouched (they stay the documented, self-tested "base" featurization) and define the
extended widths in `sb3_env.py`:

```
OPTION_DIM_LOCAL = OPTION_DIM + 2      # + is_stop, already_picked
STATE_DIM_LOCAL  = STATE_DIM  + 4      # + in_multi, num_picked, remaining_min, remaining_max
```

`obs_to_gym` fills rows `0..n-1` with the base option features plus the 2 new bits,
and fills row 127 (STOP) with all-zero base features plus `is_stop=1`.

**Checkpoint break:** the obs gets wider, so existing `out/*.zip` checkpoints can no
longer load — retrain from scratch (fine — we don't rely on the current ones) and bump
`--out-prefix` so you don't overwrite them.

### 3e. Eval must mirror the sub-step loop internally

At eval time the driver (`eval._play_one`) calls the agent **once per node** and
expects the **full answer list** back, *without the agent touching the engine* — the
driver does the single `battle_select`. But training resolves a multi-select as several
engine sub-steps. So `sb3_agent` (already in `sb3_env.py`, shared with eval) must run
the sub-step loop **internally**: repeatedly build the extended obs from the same
`SelectData` + growing `picked` set (no engine calls — the intermediate states are
computed locally), call `model.predict`, accumulate picks until STOP / auto-commit,
then return the full `picked` list in one shot.

Train and eval share the same helpers — `is_learnable_select`, the mask function
`_ms_mask(sel, picked)`, the extended `obs_to_gym`, `STOP_INDEX` — all in `sb3_env.py`,
so the two paths can't drift. Single-select collapses to one `predict` → `[a]`,
identical to today.

### 3f. The discard-then-draw ordering question

*"If a decision means discard one card and draw another, doesn't the index order decide
which pick is discarded vs. drawn? Does this design handle that automatically?"*

Yes, automatically — because that is **not one node**. In this engine, **one `select`
node is one homogeneous choice**: its `SelectData.context` says what is being chosen,
and every option in the node serves that same purpose. "Discard one, then draw one" is
surfaced as **two separate select nodes** (first the discard choice, then the draw
choice), each routed independently through this same machinery. Within a single node,
all picks have the same role, so *which* card is discarded vs. drawn is decided by
**which node** you're answering — never by the order of indices inside one `picked`
list.

And the engine validates a `battle_select` list only on **count, index range, and
no-duplicates** (§Reference) — **no order constraint** — so pick order within one node
doesn't change the outcome. This is exactly theory flag **F3**: the net may waste a
little capacity because `{a,b}` is reachable as `[a,b]` or `[b,a]`, but it is *not* a
correctness problem.

> **One confirmed order-dependent exception.** `SelectType.SKILL` /
> `SelectContext.SKILL_ORDER` ("select the order of effect activation") is a node whose
> *whole purpose* is choosing an order — there the returned index order **is**
> semantically meaningful, even though the generic validator enforces nothing about it.
> "One select node = one homogeneous, order-independent set-pick" holds for every other
> node type. Flag for rules review before relying on order-independence for any
> SKILL-ordering node.

---

## 4. Phasing

- **Phase A — `[0..1]` optional-single + fixed-k** (`min==max`). ~73% of multi nodes,
  highest EV (the 940 `[0..1]` "option 0" losses). This builds **all** the machinery:
  STOP slot, extra features, sub-step loop, shared eval loop. Gate
  `is_learnable_select` to `maxCount <= 1 or minCount == maxCount`; `auto_resolve` the
  rest. Phase A already exercises **both** the decline path (`[0..1]`, `mn=0` → STOP
  submits `[]`) and auto-commit (fixed-k reaches max via picks).
- **Phase B — true variable** (`min < max, max > 1`, e.g. "discard 1–3"). **Zero new
  machinery** — just widen the gate. The only new behavior is *intermediate* STOP (stop
  after picking some-but-not-all), which the §3b mask rules already implement. Do this
  once A is proven; it's where the autoregressive chains get longest (highest theory
  variance). **Also revisit the all-NUMBER exclusion here** — the 120-game sample saw
  zero multi-select NUMBER nodes, but that measurement is approximate (§1 caveat); the
  §5 census can confirm whether any exist before Phase B locks the gate.

---

## 5. Files to change

- **`sb3_env.py`** (most of the work): new consts `STOP_INDEX`, `STATE_DIM_LOCAL`,
  `OPTION_DIM_LOCAL`; `is_learnable_select`; shared `_ms_mask(sel, picked)`; extended
  `obs_to_gym` (append the new fields; **keep the name** — eval imports it); widen the
  `observation_space` dims; `_advance_to_policy` (switch the gate to
  `is_learnable_select`, overflow cap → `len(option) <= N_MAX-1`); `step` (the sub-step
  transitions + auto-commit + STOP → `battle_select`); `action_masks` (delegate to
  `_ms_mask`); `_terminal_obs` (new dims); `_finish_if_live` (clear the per-node
  `picked` state); `sb3_agent` (add the internal multi-pick loop for eval);
  `__main__` shape asserts. **`sb3_agent` already lives here** and is shared with eval —
  no relocation needed.
- **`eval_sb3.py`** — nothing structural: it already imports `sb3_agent` from
  `sb3_env`. Confirm it still routes correctly.
- **`train_sb3.py`** — no structural change; document the checkpoint break, bump
  `--out-prefix`.
- **`utils/run_smoketest_sb3_env.py`** — add the multi-node assertions in §6.
- **A small shape-census script (new, follow-up)** — reuse the `sb3_features.py`
  self-test scaffold (it already drives `battle_start` + `auto_resolve` /
  `is_policy_select` to policy nodes) and add a tally of
  `(minCount, maxCount, len(option), all-NUMBER)` per select node over N games. This
  makes the §1 numbers reproducible and settles the §3c open question. It is a separate,
  optional task — not required for the Phase A code.
- **Do not touch** the base encoders in `sb3_features.py`, nor `tcg_model.py` /
  `eval.py` — the new features depend on sub-step state those don't have, and freezing
  them keeps the base-featurization self-test valid and the diff localized.

---

## 6. Verification

- **One-time empty-submit check (do this FIRST — it's the riskiest assumption):** drive
  to a `minCount == 0` node and confirm `battle_select([])` is accepted (no exception).
  The engine's own rule says `len == 0` is legal iff `minCount == 0`, but this is **not
  verified anywhere in the repo today** — confirm it live before relying on the decline
  path.
- **Extend `utils/run_smoketest_sb3_env.py`** (`--games 10`, play uniformly at random
  over the live mask): assert both dominant multi classes fired (optional + fixed
  counters `> 0`); assert **`battle_select` never raises** on the generated lists (the
  engine is the oracle for distinctness + cardinality + empty-submit); assert
  `check_env` passes with the new shapes and the battle singleton isn't leaked.
  - Note: an invalid `battle_select` raises **`IndexError`** (bad count / out-of-range /
    duplicates); only a broken battle pointer raises `ValueError`. The richly-labeled
    cardinality/range/duplicate `ValueError`s live on the Search API's `search_step`,
    not `battle_select`. So assert **"no exception"**, and if you want the labeled
    reason for a failure, cross-check with `search_step`.
- **Two scripted micro-drives:** "always STOP when legal" (exercises the `[]`
  empty-submit + early stop) and "always pick max" (exercises auto-commit).
- **`eval_sb3.py --games 20`** on a fresh smoke checkpoint completes. The driver raises
  on any invalid selection, so completing proves `sb3_agent` emits engine-valid lists
  for multi nodes via the shared code. Parity comes from **shared code**, not a matched
  trajectory (the cg RNG is unseedable — no common-random-numbers).

---

## 7. Theory-review flags

Per repo convention, these are theory claims flagged for independent review. I reviewed
the *mechanics* (control flow, engine contract), not the RL theory.

- **F2 — Off-MDP intermediate states.** Partial-pick states are fictitious (the engine
  never observes them); the critic must value states outside the true game MDP. Bounded
  by F5.
- **F3 — Order on an unordered set.** `{a,b}` reachable as `[a,b]` or `[b,a]`; the net
  must learn order-invariance or waste a little capacity. (This is the discard/draw
  answer — a capacity cost, not a correctness risk. The lone exception is SKILL-ordering
  nodes, §3f.)
- **F4 — Sample-weight distortion.** A k-pick node emits `k` gradient samples vs. 1 for
  single-select, so multi nodes are up-weighted; `n_steps` now counts sub-steps, so
  games-per-update drops (~11% more nodes).
- **F5 (benign) — `gamma=1` return invariance.** Under terminal-only undiscounted
  reward, all sub-steps of a node share the same return-to-go, so the *reward* credit is
  identical to a hypothetical joint decision; the only risk is the critic/GAE treatment
  of the inserted states (F2–F4), which bounds it.
- **F6 (carried) — Positional action head.** Pre-existing; mitigated *within* a node
  (stable slots), unchanged across decisions. The custom per-option head (§8) is the
  planned fix.
- **F1** (live-policy re-entrancy during self-play rollout) belongs to the **self-play
  plan**, not here — cross-reference only.

---

## 8. Deferred levers

Out of scope for Phase A/B, captured so they aren't lost:

- **Custom per-option-head policy (planned next step, not just optional).** Replace the
  stock `MultiInputPolicy` (positional head, flag F6) with a maskable policy that scores
  each *option row* independently (the design the retired hand-rolled stack used;
  `encode_options` already preserves the card/attack id indices it would need). Directly
  addresses F6 and is the in-network alternative to sequential decomposition. Compare
  head-to-head.
- **Automated hyperparameter search** — sweep `lr` / `n_steps` / `batch_size` /
  `total_timesteps` / `ent_coef` rather than hand-picking; pairs with a future Kaggle
  run harness (the retired `kaggle/push_sweep.py` pattern, recoverable from git,
  re-targeted at `train_sb3.py`).

---

## Correctness review notes (this revision)

Facts below were verified against source (`tcg_model.py`, `sb3_env.py`,
`sb3_features.py`, and `vendor/cg-lib/cg/api.py` / `game.py`).

1. **The gym obs `Dict` has THREE keys**, not two: `state (299,)`, `options (128,70)`,
   and `action_mask (128,)`. Masking rides two channels — the `action_mask` obs key (a
   network input) and the separate `action_masks()` method (what actually constrains
   MaskablePPO sampling). The new features go in `state`/`options`; STOP selectability is
   an `action_masks()` concern (§3a, §3d).
2. **`battle_select` raises `IndexError`, not `ValueError`**, on invalid picks (bad
   count / range / duplicates); only a broken battle pointer is `ValueError`. The
   labeled validity errors are on `search_step` (Search API). §6 asserts the right thing.
3. **`auto_resolve`'s optional (`minCount==0`) branch returns the first `maxCount`
   options** (`list(range(min(maxCount, L)))`), not "always option 0." For the specific
   `[0..1]` shape it *does* reduce to `[0]` (maxCount=1), so the "940 → option 0" figure
   stands; the NUMBER branch maximizes the option's **`.number`** field.
4. **Empty-submit `battle_select([])` for `minCount==0` is NOT verified** anywhere in
   the repo. It's *plausible* per the engine's own `min <= len <= max` rule, but §6 adds
   a one-time live check as the first verification step.
5. **The 120-game shape distribution is not reproducible from committed code** — the
   counts live only in this doc, and `N_MAX`'s comment cites a separate 48-game
   measurement. §1 labels them approximate; §5 plans a census script to fix this.
6. **`sb3_features.py` is frozen for a real reason:** the new features depend on runtime
   sub-step state (`picked`, counts) that the pure encoders don't have, so they must
   live in `obs_to_gym` / `CGEnv`. (Checkpoints break regardless of where the dims are
   added — that is inherent to widening the obs, not a reason to touch the encoders.)
7. **Order is set-based with one exception:** validation enforces count/range/no-dup
   only (no ordering), but `SKILL_ORDER` nodes are semantically order-dependent (§3f).
8. **`sb3_agent` already lives in `sb3_env.py`** (shared with eval), so the eval-mirror
   home already exists — no relocation step.
