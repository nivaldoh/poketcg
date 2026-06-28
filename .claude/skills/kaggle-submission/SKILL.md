---
name: kaggle-submission
description: >-
  Assemble and submit a valid Kaggle submission.tar.gz for the pokemon-tcg-ai-battle
  competition. Use only when explicitly asked to package or submit a deployable agent —
  rare and high-stakes. Builds the top-level main.py + cg/ + deck.csv bundle via a bundled
  validator script and gates the actual submit behind explicit user request.
---

# kaggle-submission

Package a deployable agent into a valid `submission.tar.gz` and (only on explicit request) submit
it. This is rare and high-stakes, so lean on the bundled script's assertions instead of human
memory of the layout.

## Gate (read first)
- **Build freely; submit only when explicitly asked.** The bundled script *packages and validates*
  but never submits. Run the `kaggle competitions submit` step only when the user has clearly asked
  to submit — never as a side effect of packaging.

## The bundled script is the source of truth
`pack_submission.py` (in this skill's directory) is the **executable single source of truth** for
the submission layout: exactly three top-level entries — `main.py` + `cg/` + `deck.csv` — with no
nested prefix (a nested path silently breaks the submission). It validates deck legality against the
**`cg` engine** (`all_card_data()`), not against guesses, then builds the tarball and asserts the
layout. Don't hand-roll a `tar` invocation; run the script.

## Recipe
1. **Package + validate** (run from the repo root; paths are discovered/defaulted at runtime):
   ```bash
   python .claude/skills/kaggle-submission/pack_submission.py --main path/to/main.py
   ```
   - Defaults: `--cg ${CG_LIB_PATH:-vendor/cg-lib}/cg`, `--deck deck.csv`, `--out submission.tar.gz`.
   - Validate a deck alone (no `main.py` needed): add `--check-deck-only`.
   - It enforces 60 cards / ≥1 Basic Pokémon / ≤1 ACE SPEC / ≤4 copies of any non-(Basic-Energy)
     card, and that the tarball has only top-level paths. A non-zero exit means **do not submit**.
2. **Confirm the layout** if you want a second look: `tar -tzf submission.tar.gz` → only top-level
   `main.py`, `cg/…`, `deck.csv`.
3. **Submit — only when explicitly asked:**
   ```bash
   kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m "<message>"
   ```
   Verify the exact flags first with `kaggle competitions submit --help` (the CLI form can drift;
   the competition slug `pokemon-tcg-ai-battle` is confirmed in `docs/cg-sdk.md` / `README.md`).

## Single source of truth + known duplicates
- Canonical layout: **`pack_submission.py`'s assertions**.
- The same layout is *prose-duplicated* in `README.md`, `CLAUDE.md`, and the Kaggle starter
  `starter_rulebased_lucario_kaggle.py` (its `tarfile` block — `grep -n 'tarfile\|arcname'
  starter_rulebased_lucario_kaggle.py`). Reconcile those against the script; don't add copy N+1.

## Adjacent: notebook → local module port (brief checklist)
When porting a Kaggle starter to a local module (the reusable `agent()`), strip the Kaggle glue:
`%%writefile main.py`, `glob('/kaggle/input/...')` path injection, the `/kaggle_simulations/agent/`
deck-path fallback, and the `tarfile` packaging. (Already done for `rulebased_lucario.py`.)

## Drift-resilience
Point at the script, don't restate the layout here. Reference symbols by name + a grep recipe (never
line numbers). Discover paths/defaults at runtime (`--help`, `CG_LIB_PATH`). Confirm the slug and CLI
flags still exist before submitting; the `cg` engine and `all_card_data()` are ground truth.

## Where this fits
This is the **final** step of the repo's verify-loop (`docs/skills_and_hooks.md` → "How these compose"):
author/debug an agent under `cg-agent-contract` → run the `run_battle.py` smoke assertion → read
head-to-head numbers under `eval-discipline` → **package here, and submit only when explicitly
asked**. Check game-rule assumptions with `tcg-rules-critic` if deck legality is ever in doubt.
