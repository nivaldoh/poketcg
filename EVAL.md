# EVAL.md — our evaluation strategy

The single entry point for *how we decide whether an agent got better*. Every change to the model
or its search should be justifiable against something in this doc. It is the **strategy / playbook**
layer; it deliberately does **not** restate the methodology, the interpretation rules, or the
volatile numbers — those live elsewhere and are linked below.

### Where this fits (read this map first)

| Layer | Lives in | Role |
|---|---|---|
| **Strategy / entry point** | **this file** | what we measure, what's built, what to build next, how to run it |
| Methodology canon + references | `docs/eval_info.md` | the staged theory; chess/poker/MTG/Pokémon precedent; the credit-assignment problem |
| Enforced interpretation rules | `eval-discipline` skill (`.claude/skills/eval-discipline/`) | trustworthy `n`, CIs over point estimates, mirror vs deck-confound, locked search budget |
| Volatile numbers / what failed | memory (`eval-training-not-helping-yet`, `eval-info-staged-methodology`, `tcgmon-port-plan`) + `work_log.txt` | best checkpoint, trustworthy `n`, ruled-out experiments — they age fast |
| Model limits that gate eval | `evaluation_harness_report.md` | why the value head is miscalibrated (gates Stage 2) |

**Discipline:** inherit `docs/eval_info.md`'s framing, don't repeat it; encode *rules* in the skill and
*numbers* in memory, not here. Reference every symbol by name + a grep recipe (never a line number),
and read flag defaults from `--help` at use time (never hardcode them).

---

## 1. The atomic unit, and the one question

Every metric is an aggregation over a single bit: who won one battle. A battle ends when
`obs["current"]["result"] >= 0`, with `0` = seat 0 won, `1` = seat 1 won, `2` = draw — equivalently,
`result == your seat index` is a win (`grep -n "def play_one" utils/run_smoketest_battle.py`; fuller contract in
`docs/cg-sdk.md`). Win rate, Elo, and puzzle scores are all just ways of averaging this bit while
fighting the noise around it.

The one question this whole stack exists to answer: **did the agent get better?** Because the signal
is a noisy single bit, answering it credibly is mostly about *statistics and confounds*, not about
collecting more numbers.

## 2. The target vs. the proxy (and the gap between them)

- **Real target — the Kaggle ladder.** Submissions play head-to-head episodes (~168 per submission)
  against other teams; the leaderboard ranks teams; top-rated episodes are exported daily for
  BC/RL/IL. The pull chain (`kaggle competitions leaderboard / team-submissions / episodes / replay`)
  is in `README.md` → *Training §2*; a sample replay and the card metadata are already local
  (`data/replays/episode-81475913-replay.json`, `data/EN_Card_Data.csv`).
- **What we can't see.** The **exact leaderboard scoring formula** (Elo? raw win rate? points?), the
  **opponent/deck distribution** we'll face, and the **rulebook proper** (`README.md` notes it isn't
  provided — be careful with rule assumptions; use the `tcg-rules-critic` skill).
- **The proxy.** Local head-to-head win rate vs. fixed baselines stands in for ladder placement. The
  gap is real: different opponents, different decks, an unknown formula. **Rule of thumb:** treat
  every local number as a *proxy* until a real submission validates it (see §6 → Kaggle-proxy).

## 3. Principles (the best-practice spine)

Headlines only — the enforced, current versions live in the `eval-discipline` skill (rules) and
memory (numbers).

- **Separate process from outcome.** Outcome metrics (win rate, Elo) are the ultimate truth but
  noisy — need large `n`. Process metrics (move-quality vs. a reference, agreement, WPA) discriminate
  on far fewer games. (`docs/eval_info.md` §3; the poker "resulting" fallacy.)
- **Never a bare win%.** Always report `n` + a **95% Wilson CI** (`grep -n "def wilson" eval.py`),
  and compare **overlapping CIs**, not point estimates. Exclude draws from win%.
- **Cancel first-player advantage.** Seat-swap every game (`grep -n "def play_match" eval.py`).
  *Common-random-numbers pairing is NOT possible here* — the `cg` engine seeds its own RNG from OS
  entropy with no seed in the ABI, so A and B can't replay the same luck (see §5). Reliability comes
  from large-effect axes + SPRT + process metrics instead.
- **Mirror vs. deck-confounded.** Only a mirror (same deck both sides) isolates *skill*. The
  `sample_deck`-vs-Lucario matchup mixes skill with the deck matchup — don't read it as skill.
- **Lock the search budget.** Different `--search-count` = different agent strength; comparing across
  budgets is unfair. Hold it fixed across the agents you compare.
- **Use a trustworthy `n`.** Small-`n` win rates swing the verdict; look up the current figure in
  memory `eval-training-not-helping-yet` (empirically ~200 vs. the teacher) rather than trusting a
  stale number.
- **The harness tests itself.** `eval.py`'s battery includes sanity checks (random-vs-random ≈ 50%
  validates seat-swap; rule-based clearly > random validates ordering). A test you don't trust isn't
  a test.
- **Hold test positions out of training; calibrate before trusting value metrics.** Puzzle suites
  must be held out (no leakage). Any value-based metric (WPA, regret) inherits the value model's
  calibration — validate that predicted-70% spots win ~70% *first* (`docs/eval_info.md` §2A, Caveats).
- **Don't re-run ruled-out experiments** (see §8).

## 4. Current state — what's built and what it tells us

**Built: Stage 1 (outcome metrics).**

| Harness | Question it answers | Matchup | Key caveat |
|---|---|---|---|
| `eval.py` | broad checkpoint/agent comparison + rough Elo | `sample_deck` battery / `--checkpoints` round-robin | Elo only meaningful in a pure mirror; vs-rulebased is deck-confounded |
| `eval_sb3.py` | how does an SB3 checkpoint fare, per cell? | 4 decks × 4 rule-based opponents, seat-swapped; diagonal = clean mirror | mirror diagonal (`*`) = the skill read; off-diagonal cells deck-confounded; sampled mode is the honest headline |

Both report W/L/D, win% over decisive games with a 95% Wilson CI, and seat-swap. `eval.py` adds an
anchored Bradley-Terry Elo over the pool (a stable agent pegged to 0). (The BC-`MyModel` mirror
harness `eval_mymodel.py` was removed 2026-07-03 with the rest of the retired BC line.)

**What we currently believe** (hedged — verify at n≥200, don't trust stale figures):
- The **untrained `model0` has been the strongest checkpoint**; further training tends to *regress*
  strength (memory `eval-training-not-helping-yet`).
- **MCTS, not the network, provides the strength.** The value head is miscalibrated on the ~50/50
  mirror, so prefer `value_mode="rollout"` for MCTS leaf eval (`evaluation_harness_report.md`;
  `grep -n "def mcts_agent" tcg_model.py`).

**Honest limits of today's signal:** deck confound in the vs-rulebased matchup; wide CIs at small
`n` (raise `--games`); a single deck pool (no opponent/archetype diversity); Elo distorted by
non-transitive ("rock-paper-scissors") matchups; and the whole thing is **unvalidated against the
Kaggle ladder**.

## 5. Reliable A/B comparison (the approach)

Answering "did it get better?" credibly is mostly experimental design. The obvious variance-killer is
unavailable, so reliability rests on two feasible levers + a manifest.

**Common random numbers are impossible here.** Replaying the same shuffle/coin-flips for A and B (so
luck cancels in the difference) would be the biggest variance reducer — but the `cg` engine seeds its
own RNG from OS entropy and exposes **no seed** in the ABI (`battle_start` takes only the decks; the
`libcg.so` symbol table has no `Seed`/`SRand` — `grep -n "BattleStart" vendor/cg-lib/cg/sim.py`).
`--seed` controls only *agent-side* sampling (random moves, MCTS determinization), never the deal. So
paired / McNemar designs are out. *Self-checkable:* two battles with identical decks **and** identical
`random.seed` still diverge.

**Match the tool to the effect size** (the core idea):

| To detect… | Use | Why it's the reliable choice |
|---|---|---|
| a **large** change (help / regress) | gauntlet win-rates + SPRT | big effects resolve at feasible `n` |
| comparability **over time** | frozen gauntlet → one scalar + anchored Elo + manifest | numbers stop moving with the baseline |

**Minimum detectable effect** (decisive games to see a win-rate vs 50%, α=.05 / 80% power): 55%→~780,
60%→~190, 65%→~80, 70%→~40. So `--games 20` only sees ≥~25-pt gaps, and two ~50/50 mirror checkpoints
are effectively unresolvable on win-rate — detect progress on the **large-effect axes + process
metrics**, never on near-mirror duels.

**Frozen gauntlet + the comparable scalar** — `python eval.py --mode gauntlet --checkpoint X`. A
fixed, **hash-pinned** opponent set: `random` (floor; saturates), the frozen anchor (the discriminating
**SKILL** axis, a clean mirror), and the teacher (large-effect **yardstick**, deck-confounded). The
headline scalar = win% (and implied Elo) vs the frozen anchor — track it across checkpoints. All model
agents play `sample_deck` (via `make_model`), so this is the *sample_deck* skill axis — a Lucario-deck
BC checkpoint doesn't belong here (the BC line and its `eval_mymodel.py` harness are retired). Pin
the anchor by **content hash**, not filename (`out/` is gitignored, names get reused). Anchor = the
frozen-strongest `model0`, `out/anchor_model0.pth`, sha256
`5bf4b8ae0cc977dfb98f3a915d994990979d2db6e4ad4a61e5dd99396f8c4122`. Re-running anchor-vs-random over
time is a harness-drift detector.

**SPRT — the ship/no-ship verdict** — `python eval.py --mode sprt --checkpoint A --baseline-checkpoint
B`. The computer-chess sequential test: stream seat-swapped games, stop the moment the log-likelihood
ratio crosses a bound (accept "better" / "not better") or hit `--max-games`. Controlled `(α, β)` error
at the **fewest games** — terminates fast on a large effect, returns "inconclusive" only on a genuine
coin-flip. H0/H1 are Elo gaps (`--sprt-elo0` / `--sprt-elo1`, default `0` vs `+10`).

*(Retired: a held-out move-agreement harness, `eval_agreement.py`, once filled the
near-identical-checkpoint niche — one forward pass per frozen teacher decision, so its CI shrank far
faster than win-rate. It depended on the deleted BC-teacher tooling and was removed; git history has
it. Today near-identical checkpoints are simply unresolvable — judge on the large-effect axes.)*

**The comparability contract.** Every `eval.py` run prints a JSON **manifest**; two runs compare only
if it matches — checkpoint sha256s, `--search-count`, decks, `n`, draw rate, torch version / device /
determinism, arch, git commit. Because the engine RNG is uncontrolled, reproducibility comes from
**pinned config + adequate `n`, not seeds**; `torch.use_deterministic_algorithms(True, warn_only=True)`
only fixes the forward pass (on CUDA also set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for bit-exactness, or
run eval on CPU — the manifest records which).

**Stats fixes folded in** (`grep -n "def anchored_elo\|def sprt\|TRUST_N" eval.py`): anchored Elo
(pegged to a fixed anchor) replaces the per-run mean-anchored Elo, so ratings are comparable across
runs; deck-confounded vs-teacher results are kept **off** the skill axis; saturated pairs (vs random)
are flagged as prior-driven, not data; and a sub-`TRUST_N` (~200) game count prints a warning.

## 6. Roadmap (stages inherit `docs/eval_info.md`; status + acceptance gate added)

- **Stage 1 — outcome metrics — ✅ BUILT** (`eval.py`, `eval_sb3.py`). *Maintenance:* keep
  `--search-count` locked, raise `--games` for tighter CIs, consider adding baselines / a second
  opponent deck for archetype diversity.

- **Stage 3 — puzzle / known-answer suite — 🎯 RECOMMENDED NEXT.** It *leapfrogs* Stage 2: immune to
  outcome luck, needs no calibrated value model, sidesteps every known blocker, and directly tests
  "did the model learn each deck's key plays" (the `ideas.md` ask). Precedent: PokéChamp's 1,000
  curated scenarios, chess STS, Hearthstone "find lethal" (`docs/eval_info.md` §2D, Recommendations
  Stage 3).
  - *Sketch:* curate positions by category — find lethal / correct removal target / sequencing /
    keep-or-mulligan. Source "correct" answers from deep MCTS search + a rules consult
    (`tcg-rules-critic`). **Hold every position out of training.** Score by top-1 / top-k agreement,
    reported **per category** with CIs (a skill profile, not one number).
  - *Acceptance gate:* a runnable `eval_puzzles.py` that loads a held-out position set and prints
    per-category accuracy + Wilson CIs. **Specified here, not built in this change.**

- **Stage 2 — calibrated value model → Win-Probability-Added — ⛔ DEFERRED (blocked).** WPA / blunder
  rate need a value model whose probabilities are *calibrated*. Ours isn't — its mirror training
  target is ≈ a coin flip (`evaluation_harness_report.md`). **Gate:** validate calibration (predicted-70% spots win
  ~70%) before trusting any value-based move score. Borrow MTG-Causal-RL's six factors as a starting
  feature set (`docs/eval_info.md` §6).

- **Stage 4 — per-factor credit / off-policy eval — optional, later** (`docs/eval_info.md` §4, §6).

- **Kaggle-proxy validation — cross-cutting anchor.** The only thing that closes the §2 gap: submit,
  pull episode replays, and check whether local win-rate/Elo gains actually track leaderboard
  movement. Gated behind an explicit submission (`kaggle-submission` skill) — high-stakes, do only
  when asked.

## 7. How to run (quick reference — read defaults from `--help`)

```bash
python eval.py                                   # default sanity battery
python eval.py --mode gauntlet --checkpoint out/X.pth        # X vs the frozen anchor -> comparable scalar
python eval.py --mode sprt --checkpoint out/A.pth --baseline-checkpoint out/B.pth  # is A > B? (controlled)
python eval.py --checkpoints out/a.pth,out/b.pth # clean mirror round-robin (skill ranking)
python utils/run_smoketest_battle.py                             # smallest signal that the engine runs at all
```

Run `python eval.py --help` / `python eval_sb3.py --help` for the current flags and defaults
(don't hardcode them). For *which* harness to use and *what* you may conclude, follow the
`eval-discipline` skill.

## 8. Pitfalls & ruled-out paths (don't redo these)

Verified dead ends — see `work_log.txt` and memory before re-attempting:
- **Self-play training diverges** (cold *and* warm-started from BC) vs. the teacher.
- **Value-head MCTS hurts** on the ~50/50 mirror (the head is miscalibrated) — use rollout leaves.
- **Further training regresses** strength relative to the untrained init (memory
  `eval-training-not-helping-yet`).
- **The AMAGO / Metamon port was abandoned** — the simple starter matched it (memory
  `tcgmon-port-plan`).

## 9. Reading order

New here? Read this file → `docs/eval_info.md` (the why) → the `eval-discipline` skill (the rules) →
memory for today's numbers. When this doc and the live engine / `--help` disagree, **the engine
wins** and this doc is stale — fix it.
