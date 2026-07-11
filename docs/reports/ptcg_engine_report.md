# Can we use `ptcg_engine`? — a plain-language assessment

[we want to avoid using any "easy" information from the engine that a player wouldn't have direct access to. We want to stay with model-free RL, instead of having a model of the environment readily available for model-based RL. We should do another pass on this plan based on that]
[for external things like creating a visualizer that would exclusively help me (a human) see what's going on during a match, generating better data, etc, it would be okay to display information that a player wouldn't necessarily have.]
[we need to update our docs with references to the engine.]


## Bottom line

- **What it is:** the C++ source of *our own* competition engine (the "cabt" engine that
  becomes `libcg.so`). Not a new simulator — the guts of the one we already run.
- **Can we use it?** **Yes, for building and testing our entry locally** — which is
  exactly training and evaluation. That's the license's stated purpose and what the
  README calls "suitable for local debugging and reinforcement learning." The limits are
  about *sharing*, not *using*: don't republish it, don't ship it inside the Kaggle
  submission, delete it when the competition ends. (Details in "Rules" below.)
- **Does it help?** **Yes — meaningfully.** It removes guesswork we've been living with,
  and it puts three things on the table that the sealed binary flatly denied us:
  a denser training reward, reproducible (seeded) evaluation, and faster parallel
  training. None of these are magic bullets, but all three attack real, documented
  limiters in our current stack.

## What it actually is

Our whole stack treats the game engine as a black box: we hand `libcg.so` some decks and
actions through `cg`, and it hands back observations and a win/loss. We reverse-engineered
what the fields mean by poking at it, and `docs/cg-sdk.md` even flags itself as
"probably redundant and needs to be reviewed."

`ptcg_engine` is the source behind that box. Concretely it contains:
- the full public API (`Export.cpp`) — the same functions our wrapper already calls;
- the complete game rules as executable code (turn flow, prizes, KOs, status, deck-out);
- **~1,268 fully-implemented cards** with real attack logic *and* English rules text
  (`CardImpl.h`, ~878 KB) — effectively the rulebook we never had (README line 107:
  "I haven't provided you an actual rule book yet");
- the search/determinization model used for MCTS (`Search.h`) — a clean, cloneable
  "what-if" copy of the game state.

## The rules question (can we use it, and how)

I read the license and README directly. In plain terms:

**Allowed (and intended):**
- Use it *only* to **build and test your competition entries**, *only while the
  competition is running*. Local RL training and local evaluation are squarely "building
  and testing your entry." The README literally markets the simulator as being for local
  RL.

**Not allowed / be careful:**
- **Don't redistribute or publish it.** ✅ Already handled — `ptcg_engine/` is in
  `.gitignore`, so it won't be committed or pushed.
- **Don't put this source (or a private recompiled build of it) inside the Kaggle
  submission.** The submission must keep using the stock competition `cg` package. Our
  submission boundary (`main.py` + official `cg` + `deck.csv`) already does this — so a
  local seedable build is a *training/eval tool only*, never shipped.
- **Delete your copy when the competition ends.**
- Anything "derived from" the source is deemed "Pokémon Elements" (belongs to Pokémon).

**One caveat to verify yourself, not on my word:** I'm a coding agent, not a lawyer. The
*binding* terms are the Kaggle Competition Rules you accepted, not this summary. The one
nuance worth a 2-minute check on the rules page: whether a *model trained against* the
engine counts as a restricted "derivative." The competition's entire premise is that you
train an agent and submit it, so this is almost certainly fine — but confirm it's not
carved out. Everything else above is low-risk and already the intended use.

## What it unlocks (ranked by value ÷ effort)

### 1. Ground truth — kills a pile of guesswork (cheap, do first)
Right now several things in our code are educated guesses that we can now *verify*:
- **Hand-identified card / attack IDs and damage numbers** in the rule-based agents
  (e.g. `game_agents/rulebased/lucario.py` hardcodes `Riolu=677`, `Mega Brave attackId=983`,
  damage constants like 130/210/270, and penalized IDs `144/322/323/337`) — every label
  was inferred from play and can now be checked against the card table in `CreateCard.h`.
- **The `ex` vs `megaEx` inconsistency** already flagged in `sb3_features._card_static`
  ("DOC-vs-ENGINE INCONSISTENCY… do not take on faith") — `Card.h` settles it.
- **Magic normalizers** in `sb3_features.py` (HP ÷ 400, damage ÷ 350, prize ÷ 6, …, which
  the file itself admits are "only approximately right") — the source has the true maxima,
  so these become exact instead of approximate.
- **`docs/cg-sdk.md`'s hedged claims** (the determinization placeholders, the option-field
  mapping, the first-call deck convention) can be audited against `AddOption.h` /
  `SelectProc.h` / `Api.h` instead of trusting reverse-engineered notes.

*Why it matters:* these are silent correctness risks in features and heuristics. Fixing
them is low-effort and de-risks everything built on top.

### 2. A denser training reward — and the best version needs *no* rebuild (high value)
Our single biggest documented limiter (memory + `sb3_report.md`): the reward is
**terminal-only ±1**, so the agent gets no per-decision credit and never contests the
prize/KO race (69% of losses are prize-outs).

The source reveals something we didn't realize: the engine already tracks **dense
per-step information** — prize counts, deck sizes, damage dealt, KOs, prizes taken this
turn, energy attached, HP, turn number — and a typed **event log** of all of it. Crucially,
**this event log is already returned in `obs["logs"]` on *every* `battle_select`**, not
just at the end. We currently only read it at the terminal node (`win_mode.classify_end`)
and throw the rest away.

*Why it matters:* the raw material for a prize-race / win-probability-aligned per-decision
reward is **already flowing to us** — no engine change needed, just stop discarding it. The
source's job here is to tell us *exactly* what each log event and field means so we build
the signal correctly.
*Honest caveat (flagged for independent review):* past reward-shaping attempts
gate-FAILED, and our standing read is "the problem is reward *structure*, not
implementation." So this is not a guaranteed fix — but the source removes the "we're not
sure what these fields mean" confound that may have undermined earlier attempts, and lets
us build a *verified-correct* signal. Treat as a promising, not proven, direction.

### 3. Reproducible evaluation — CRN becomes possible in a local build (high value, medium effort)
Our eval can't do "common random numbers" (replay the *exact same luck* for both agents so
an A/B comparison is fair) — `EVAL.md` §5 and `eval.py` document that the shipped `libcg.so`
seeds itself from OS entropy and exposes no seed. The source confirms this at the source
level *and* shows precisely why: `BattleStart` re-seeds from hardware entropy and sets
`deviceRand=true`, which makes shuffles/coin-flips bypass the seeded RNG entirely.

But the engine was *designed* to be seedable — `GameConfig` has a `seed` field and a
`deviceRand` flag; seed `0` means "randomize," any nonzero seed is deterministic. So a
**local build with a fixed seed + `deviceRand=false`** (three small edits) would give us
deterministic, replayable battles — turning our "CRN is impossible, period" constraint into
"impossible with the stock binary, but doable with our own eval build."
*Caveat:* `std::shuffle`'s exact output can differ across compilers/standard libraries, so
determinism holds within one build, not necessarily identically to Kaggle's binary. That's
fine for *our* eval reproducibility (the goal), just don't assume byte-identical replays vs
Kaggle.

### 4. Faster local training — break the one-env-per-process wall (high value, medium effort)
Our training runs **one live battle per process** (`sb3_env.py`: "PROCESS-GLOBAL SINGLETON…
no VecEnv n>1"), which throttles throughput on our small WSL2 box. I confirmed this is a
**wrapper limitation, not an engine one**: `vendor/cg-lib/cg/sim.py` stores the battle
handle as a *class attribute* (`class Battle: battle_ptr = None`) — one global handle for
the whole process — so two battles would clobber each other. But the C++ engine gives every
battle its own heap `ApiData*` and has **no** hidden process-global battle state (audited:
the only globals are the read-only card tables).

*Why it matters:* we can run many battles in one process (a real vectorized env) by keeping
N distinct handles instead of one class-global — either via a thin custom wrapper or a
per-instance build. On limited local hardware, this is the most direct throughput win, and
it's the difference between "training is painfully slow" and "we can actually iterate."

### 5. The rulebook, on tap (ongoing utility)
The `tcg-rules-critic` skill and `ideas.md` both want a reliable source of game rules.
`CardImpl.h` + `EffectInstant.h`/`GameProc.h` *are* the authoritative rules — deck legality
(`DECK_SAME_CARD_MAX`, ≤1 ACE SPEC, ≥1 Basic), evolution chains, prize/KO/deck-out logic,
status effects — all as executable ground truth. Anytime a heuristic, feature, or eval claim
depends on "how does the game actually work here," this answers it definitively.

## What it does *not* give us

- It does **not** by itself solve the prize-race reward problem (see #2 caveat).
- It does **not** change the Kaggle submission — that still uses the stock `cg`.
- A local recompiled build is a **tool for us**, never something we ship.
- It's not a shortcut to a stronger model; it's better *instrumentation and ground truth*
  around the model we're already trying to train.

## Recommended next steps (cheap → expensive; each independently useful)

1. **(Cheap, do first)** Dump the real card table via the engine's `AllCard()`/`AllAttack()`
   and reconcile our hardcoded IDs, damage constants, magic normalizers, and the
   `ex`/`megaEx` flag against it. Fix `sb3_features.py` normalizers + audit
   `game_agents/rulebased/*`. Also refresh/annotate `docs/cg-sdk.md` from source.
2. **(Cheap–medium)** Start reading `obs["logs"]` on every step (not just terminal) and log
   prize/KO/damage deltas per decision — first as *telemetry only*, to see the prize-race
   dynamics we're currently blind to. This is a prerequisite to any per-decision reward and
   costs nothing in correctness risk.
3. **(Medium)** Compile a local `libcg.so` from the source
   (`g++ -std=c++20 -O2 -fPIC -shared -DNDEBUG Export.cpp -o libcg.so`) and confirm it loads
   in our `cg` wrapper and reproduces a battle. This is the gate for #4 and #5 below.
4. **(Medium)** From that build, make a **seedable eval variant** (fixed seed +
   `deviceRand=false`) so eval gets reproducible/CRN comparisons.
5. **(Medium)** Prototype a **parallel/vectorized env** by holding N battle handles per
   process (custom thin wrapper), removing the singleton throughput cap.

## How to verify the load-bearing claims (per our "verifiable results" rule)

- **"It's our engine":** `grep -o 'GAME_API [A-Za-z_]*' "ptcg_engine/ptcgProgram 22/Export.cpp"`
  and compare the symbol names to what `vendor/cg-lib/cg/` imports from `libcg.so`.
- **"It builds on Linux":** run the `g++ … -o libcg.so` command above; expect a `.so` with no
  Windows deps (audited: no `windows.h`, the one `__declspec` has a gcc fallback).
- **"Same card data":** load our new `.so`, call `AllCard()`, diff the JSON against the
  vendored engine's `all_card_data()` — expect identical.
- **"Seeding makes it deterministic":** in the local build, fix the seed + set
  `deviceRand=false`, run the same battle twice, assert identical logs.

---

*Confidence: the "it's our engine," rules, seeding, telemetry-already-in-obs, singleton
cause, and buildability claims are verified directly from the engine source
(`ptcg_engine/ptcgProgram 22/`) and our repo. The reward-shaping payoff (#2) is
promising-but-unproven — flagged for independent review per our convention on theory
claims. All source references are by symbol/file (grep-reproducible), not pinned lines.*
