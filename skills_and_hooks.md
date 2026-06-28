# Skill ideas (backlog)

Candidate Claude Code Skills to build *later* — not yet created. Sibling to `ideas.md`.

**The bar a skill must clear here.** This repo already carries strong always-on context (`CLAUDE.md`, `README.md`, `docs/cg-sdk.md`, `eval_info.md`, auto-memory). A skill earns its place only by adding value *beyond* that: (1) **forcing** an error-prone domain procedure to be followed at the right moment (sharp trigger), (2) **progressive disclosure** — pulling a large doc in only when relevant, or (3) **hiding a fiddly, infrequent procedure** behind a script + checklist. Skills must *point to* existing docs, not copy them. One job each. Don't force them.

---

## Drift-resilience contract (rules every skill here must follow)

These skills will sit dormant for weeks and must still be correct after the repo moves under them. The whole point of writing them well *now* is that we won't notice when they rot. Before building any skill below, make it obey all six:

1. **Point, don't copy.** Reference the canonical doc; never restate its prose. A skill that copies a fact becomes a second source that silently disagrees with the first.
2. **Reference symbols by name + a discovery recipe, never pinned line numbers.** Write "the `get_card()` resolver in `tcg_model.py` (locate it: `grep -n 'def get_card' tcg_model.py`)", not `tcg_model.py:258`. Line numbers are wrong after the next edit; symbol names + a grep survive refactors.
3. **Discover paths and defaults at runtime; don't hardcode them.** Find checkpoints with `ls out/`, read CLI defaults from the script's `--help`, get decks from their source module. A hardcoded `out/mixed_final.pth` / `deck.csv` / `sample_deck` goes stale without warning.
4. **Separate stable judgment from volatile specifics.** Encode the *rule* (e.g. "compare overlapping CIs, not point estimates"; "lock `--search-count` across the agents you compare") in the skill; fetch the *current numbers* (the trustworthy `n`, today's defaults, which checkpoint is best) from memory / `--help` at use time. Rules age slowly; numbers age fast.
5. **Verify before you recommend.** Confirm a file/symbol/flag still exists before citing it (the harness already asks this of recalled memory). Where they conflict, the compiled `cg` engine and `all_card_data()` are ground truth over any doc or memory.
6. **Name the single source of truth + its known duplicates.** Each skill states where the canonical fact lives and lists the other copies that need reconciling, so the skill never becomes copy N+1. (Today's worst offenders: the submission layout, the `--batch-size` gotcha, and the card-ID/deck facts each live in 3–4 places — see each skill.)

## How these compose with the workflow loop

The skills reinforce the repo's existing verify-loop; they don't replace it. The intended chain when changing agent code: author/debug under **`cg-agent-contract`** → run the `run_battle.py`-style smoke assertion (the smallest "engine still runs" signal) → interpret any head-to-head numbers under **`eval-discipline`** at a trustworthy `n`. `tcg-rules-critic` slots in wherever a step depends on game rules; `kaggle-submission` only at the very end. Each skill should close by pointing back at that loop rather than inventing a parallel one — and should cross-link the hard-won "what does NOT help" guardrails (self-play diverges, the value head is miscalibrated on the mirror, the AMAGO/Metamon port was abandoned — memory `tcgmon-port-plan`) so a future session doesn't burn a day re-running a ruled-out experiment.

---

## 1. `eval-discipline` — Tier 1

Run the right eval harness and draw only the conclusions it supports.

- **Triggers when:** comparing checkpoints/agents, claiming "the model improved/regressed", or interpreting `eval.py` / `eval_mymodel.py` win-rates.
- **Prevents:**
  - Trusting small-n results. **Use a trustworthy `n`** — the size at which game-to-game variance stops swinging the verdict (memory `eval-training-not-helping-yet` records the current figure and the swing that motivated it; re-verify rather than hardcoding a number here).
  - Comparing point estimates instead of overlapping Wilson CIs.
  - Confusing the **deck-confounded** matchup (`sample_deck` vs Lucario) with a clean **mirror** (same deck both sides) — only the mirror isolates skill.
  - Grabbing the wrong harness: `eval.py` (sample_deck; round-robin/battery) vs `eval_mymodel.py` (Lucario mirror; BC `MyModel`; needs `TORCHDYNAMO_DISABLE=1`). Read each one's defaults from `python eval.py --help` / `python eval_mymodel.py --help` at use time — don't memorize them here (they differ between the two scripts and will drift).
  - Comparing runs at different `--search-count` (unfair — different agent strength); wrong checkpoint timing (`modelN` is saved *before* iter N's training); reading Elo as meaningful outside a pure mirror round-robin.
- **Best-practice framing (inherit from `eval_info.md`, don't restate it):** process metrics vs outcome metrics (process metrics discriminate skill on far fewer games; win-rate/Elo need large `n` to average out luck); **validate a value model's *calibration* before trusting any value/WPA metric** — this is the gate on the deferred Stage-2 work; always report `n` + CI, never a bare win%; Elo is distorted by non-transitive (rock-paper-scissors) archetype matchups, so report the pool.
- **SKILL.md (lean):** a decision table — *question I'm asking → harness, matchup, trustworthy `n`, locked `--search-count`* — plus an interpretation checklist (CIs not points; draws excluded; fetch the current trustworthy `n` from memory). Volatile specifics (defaults, best checkpoint) are looked up, not written in.
- **Reuses:** `eval_info.md` (staged methodology, Stages 1–4); memory `eval-training-not-helping-yet`, `eval-info-staged-methodology`. No new script — the judgment layer over existing harnesses.

## 2. `tcg-rules-critic` — Tier 1 (on-demand Skill, not a subagent)

Check game-logic reasoning against the actual rulebook.

- **Triggers when:** reasoning about deck legality, evolution chains, energy/retreat costs, prize/KO logic, status conditions, determinization assumptions, or designing heuristics/features — i.e. whenever correctness depends on TCG rules.
- **Prevents:** guessing at rules. `rulebook.md` (~1650 lines) is too big to hold in context; the skill pulls in the relevant section only when rules matter.
- **SKILL.md (lean):** a navigation recipe rather than a copied map — `grep -n '^## \|^### ' rulebook.md` to dump the section / appendix / glossary headings, then read the matching line range; return a rules-consistency verdict. **Caveat to encode:** the compiled `cg` engine is ground truth where it conflicts with the rulebook; card facts come from `all_card_data()`, never guessed from IDs.
- **Reuses:** `rulebook.md`; memory `poketcg-card-ids-and-decks`. (Origin: the `/goal` critic idea in `ideas.md`; settled as an inline Skill rather than a subagent.)

## 3. `cg-agent-contract` — Tier 2

Write/modify a cg `agent()` without tripping the contract.

- **Triggers when:** authoring or debugging an agent, deck, or option-handling code.
- **Prevents:** option indices ≠ card IDs (resolve via the canonical `get_card()` resolver — find it with `grep -n 'def get_card' tcg_model.py`); violating `minCount`/`maxCount`; duplicate indices; forgetting the 60-card legal deck on the first call (`obs.select is None`, Kaggle flow only); deck-legality rules (60 cards / ≤4 per non-energy / ≥1 Basic / ≤1 ACE SPEC); treating opponent zones as visible (they're counts only); assuming mirror-determinization holds for asymmetric decks.
- **SKILL.md (lean):** the contract checklist + canonical snippets (the `run_battle.py` driver loop and its `assert result in (0,1,2)` smoke pattern — locate with `grep -n 'def play_one\|assert result' run_battle.py`; the `get_card()` resolution pattern) + the convention "ship a `run_battle.py`-style smoke assertion."
- **Reuses:** `docs/cg-sdk.md` (the canonical contract prose — ~54 lines, navigable by `##`/`###` headers, already referenced by `CLAUDE.md`), so the value is the forcing trigger + consolidated gotchas, not the prose.

## 4. `kaggle-submission` — Tier 2

Assemble & submit a valid `submission.tar.gz`.

- **Triggers when:** explicitly asked to package/submit a deployable agent. (Rare + high-stakes → skill memory beats human memory.)
- **Prevents:** nested paths (silently break the submission — must be top-level `main.py` + `cg/` + `deck.csv`); wrong/missing files; forgetting the "submit **only when explicitly asked**" guardrail; mis-typing the `tar -czvf` + `kaggle competitions submit` invocation.
- **SKILL.md + bundled script:** an assembler+validator that stages `main.py`/`cg/`/`deck.csv`, asserts top-level layout & deck legality, then builds the tarball; plus the submit command and the explicit gate. **Single-source-of-truth note:** the layout fact is currently duplicated across `README.md`, `CLAUDE.md`, and the kaggle starter — make this script's assertions the *executable* source of truth and list those copies in the skill so they get reconciled, not re-copied.
- **Adjacent (lighter, same boundary), fold in as reference if useful:**
  - *Notebook → local module port:* strip `%%writefile`, `glob('/kaggle/input/...')`, the `/kaggle_simulations/agent/` fallback, and `tarfile` (done twice for `rulebased_lucario.py` — mostly a checklist).
  - *Fetch ladder replays:* the `leaderboard → team-submissions → episodes → replay` CLI chain (a bundled script; relevant once the BC-from-replays task in `ideas.md` starts).

---

## Hook

- **Work-log discipline** (append a lean entry after substantive changes): an always-on habit, not a triggered task — **implemented as a Stop hook** in `.claude/settings.local.json` (nudges only when tracked source files changed but `work_log.txt` didn't), not as a skill.
- **Training-runner wrappers** (`rl-train` / `bc-pretrain`): commands + the batch-size gotcha already live in `README.md`/`CLAUDE.md`, and "what's already ruled out" lives in memory (self-play diverges; value-head miscalibrated; AMAGO abandoned; CE-BC is the keeper). A skill would just duplicate passive context.
