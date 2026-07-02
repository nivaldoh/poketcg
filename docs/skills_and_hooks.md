# Skills & Hooks (quick reference)

The Claude Code **skills** and **hooks** for this repo. These are **implemented** (this file was a
backlog; it's now the index). Skills live in `.claude/skills/<name>/SKILL.md`; the hook is wired in
`.claude/settings.local.json`. Each skill sharpens the repo's verify-loop and obeys the
**Drift-resilience contract** below.

## Skills

| Skill | Use when | Reuses / source of truth | Path |
|---|---|---|---|
| **`tcg-rules-critic`** | correctness depends on TCG rules — deck legality, evolution, energy/retreat, prize/KO, status, determinization, heuristic/feature/eval design | `docs/rulebook.md`; `all_card_data()`; memory `poketcg-card-ids-and-decks` | `.claude/skills/tcg-rules-critic/` |
| **`eval-discipline`** | comparing checkpoints/agents, claiming the model improved/regressed, interpreting `eval.py` / `eval_mymodel.py` / `eval_ppo.py` win-rates, judging a PPO run's health from its metrics jsonl / picking a checkpoint | `docs/eval_info.md`; `evaluation_harness_report.md`; bundled `ppo-health-reference.md`; memory `eval-training-not-helping-yet`, `ppo-failure-investigation-verdict` | `.claude/skills/eval-discipline/` |
| **`cg-agent-contract`** | authoring/debugging an `agent()`, a deck, or option-handling code | `docs/cg-sdk.md`; `utils/run_smoketest_battle.py`; `get_card()` in `tcg_model.py` | `.claude/skills/cg-agent-contract/` |
| **`kaggle-submission`** | explicitly asked to package/submit a deployable agent (rare, high-stakes) | bundled `pack_submission.py` (executable layout SoT) | `.claude/skills/kaggle-submission/` |

## Hook

- **Work-log nudge** — a **Stop hook** (`.claude/settings.local.json` → `.claude/work-log-nudge.sh`).
  Advisory, loop-safe, read-only: if tracked files changed this turn but `work_log.txt` didn't, it
  nudges once to append a lean dated entry. It's an always-on habit, not a triggered task — hence a
  hook, not a skill.

---

## The bar a skill must clear

This repo already carries strong always-on context (`CLAUDE.md`, `README.md`, `docs/cg-sdk.md`,
`docs/eval_info.md`, auto-memory). A skill earns its place only by adding value *beyond* that:
(1) **forcing** an error-prone domain procedure at the right moment (sharp trigger),
(2) **progressive disclosure** — pulling a large doc in only when relevant, or
(3) **hiding a fiddly, infrequent procedure** behind a script + checklist. Skills *point to* existing
docs, never copy them. One job each. Don't force them.

## Drift-resilience contract (rules every skill must follow)

These skills sit dormant for weeks and must still be correct after the repo moves under them — we
won't notice when they rot. Every skill here obeys all six; hold new ones to the same bar:

1. **Point, don't copy.** Reference the canonical doc; never restate its prose. A copied fact becomes
   a second source that silently disagrees with the first.
2. **Reference symbols by name + a discovery recipe, never pinned line numbers.** Write "the
   `get_card()` resolver in `tcg_model.py` (`grep -n 'def get_card' tcg_model.py`)", not a line
   number. Names + a grep survive refactors; line numbers don't.
3. **Discover paths and defaults at runtime.** Find checkpoints with `ls out/`, read CLI defaults from
   `--help`, get decks from their source. A hardcoded `out/*.pth` / `deck.csv` goes stale silently.
4. **Separate stable judgment from volatile specifics.** Encode the *rule* in the skill (e.g. "compare
   overlapping CIs"; "lock `--search-count`"); fetch the *numbers* (trustworthy `n`, today's defaults,
   best checkpoint) from memory / `--help` at use time. Rules age slowly; numbers age fast.
5. **Verify before you recommend.** Confirm a file/symbol/flag still exists before citing it. Where
   they conflict, the compiled `cg` engine and `all_card_data()` are ground truth over any doc/memory.
6. **Name the single source of truth + its known duplicates.** Each skill states where the canonical
   fact lives and lists the copies to reconcile, so it never becomes copy N+1. (The submission layout
   is now *executable* in `kaggle-submission`'s `pack_submission.py`; the card-ID/deck facts live in
   memory `poketcg-card-ids-and-decks` + `all_card_data()`.)

## How these compose with the verify-loop

The skills reinforce the repo's verify-loop; they don't replace it. The chain when changing agent
code: author/debug under **`cg-agent-contract`** → run the `utils/run_smoketest_battle.py`-style smoke assertion (the
smallest "engine still runs" signal) → interpret head-to-head numbers under **`eval-discipline`** at a
trustworthy `n` → package with **`kaggle-submission`** only at the very end (submit only when
explicitly asked). **`tcg-rules-critic`** slots in wherever a step depends on game rules. Each skill
closes by pointing back at this loop and cross-links the hard-won "what does NOT help" guardrails
(self-play diverges; the value head is miscalibrated on the mirror; the AMAGO/Metamon port was
abandoned — memory `tcgmon-port-plan`) so a future session doesn't re-run a ruled-out experiment.

## Deliberately *not* a skill

- **Training-runner wrappers** (`rl-train` / `bc-pretrain`): the commands + the `--batch-size` gotcha
  already live in `README.md`/`CLAUDE.md`, and "what's already ruled out" lives in memory. A skill
  would just duplicate passive context.

## Adding a skill

Create `.claude/skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`) and a lean body
that obeys the contract above; bundle any script in the same directory (see `kaggle-submission`).
Copy the structure of an existing `SKILL.md` (e.g. `tcg-rules-critic`).
