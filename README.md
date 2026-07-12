# General instructions
- Start with minimal code that can be verified, always aim for simplicity, avoid verbosity, avoid boilerplate and things like backwards-compatibility. We want to keep our code clean and readable. Code/repo changes should ideally be minimal, incremental/decremental and verifiable.
- Be careful with assumptions and overconfidence.
- We should aim to structure the repo such that it makes it easier for you (Claude Code) to work with it.
- Don't get overly attached to existing code. Refactor and rewrite freely if you see good opportunities to simplify things.
- We should keep the work log lean and up-to-date so that we can properly keep track of what has been tried/done so far.

# Roadmap
- The roadmap is in [`plan.md`](docs/plans/plan.md) (marked implemented as of 2026-07-09 — see its header for open items). For evaluation strategy see [`EVAL.md`](EVAL.md); for the underlying methodology see [`docs/eval_info.md`](docs/eval_info.md).

# Context

## Overview and description
Overview
This project aims to enhance the performance of an AI Training Agent with the Pokémon Trading Card Game (TCG). The research focuses on training AI Training Agents for competitive play in a system where probability, unknown elements, and strategic planning are key determinants of success.

As a strategic game of its own, Pokémon TCG players must make gameplay decisions while being mindful of their opponent's own strategies, decks and hands. When formulating their approach to the game, players need to take into consideration various Pokémon types and thousands of different card combinations to consider various gameplay possibilities. In addition, other factors such as card draws and coin tosses introduce additional gameplay variables. The AI Training Agent’s training will be conducted within this competitive simulation framework.

Not knowing what cards an opponent holds presents a core challenge for an AI Training Agent. Participants are encouraged to explore novel methodologies for strategy learning and decision making in this dynamic environment.

Participants will be provided with a simulator (SDK) for training and testing. This toolkit uses the same logic as the Kaggle competition environment, making it suitable for local debugging and reinforcement learning.

Using rule-based programming alone may not ensure a high ranking. Winning a Pokémon TCG game requires forward thinking, real-time adaptation, and optimal decision-making. With many different deck variations available and evolving strategies, no two games are alike. The AI Training Agent must demonstrate high analytical capacity, adaptability, and be ready for the unexpected.

## Kaggle Dataset Description (The 4 files mentioned here haven't been provided yet)
Episode Replay
You can access episode replays for your Submissions from the Submissions tab, or you can download them via the CLI (or MCP) read more here: https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md

You can download replay files from other teams from the Leaderboard, and we will enable a daily episode export of the top rated episodes (to help BC/RL/IL). This will be posted in the competition forums.

Dataset Description
This competition provides card metadata and reference materials for the Pokémon Trading Card Game environment used in the simulator.

The dataset contains card identifiers, card names, expansion information, and reference images that correspond to the cards available in the competition environment. These files are provided to help participants understand the card pool and map card IDs used in the simulator to their corresponding card details.

Two versions of the card data are provided: an English version and a Japanese version. The content is identical except for the language of the card names and descriptions.

Files
The dataset includes the following files:

Card\_ID\_List\_EN.pdf - A reference document listing all cards available in the competition environment. Each entry includes the card ID, card name, expansion set, collection number, and an image of the card.

Card\_ID\_List\_JP.pdf - The Japanese version of the card reference document. The structure is identical to the English version, but card names and descriptions are written in Japanese.

EN Card Data.csv - A structured dataset containing metadata for each card in English.

JP Card Data.csv - The Japanese version of the card metadata dataset.

CSV File Structure
Both CSV files share the same schema and contain structured metadata for all cards used in the competition environment.

Card ID
Unique identifier for each card used by the simulator.

Card Name
The name of the card.

Expansion
The expansion set the card belongs to.

Collection No.
The card's collection number within the expansion.

Stage (Pokémon) / Type (Energy and Trainer)
Indicates the Pokémon evolution stage (Basic, Stage 1, Stage 2) or the card type for Energy and Trainer cards.

Rule
Special rule text associated with the card, if applicable.

Category
The category of the card (e.g., Pokémon, Trainer, Energy).

Previous stage
The previous evolution stage required for this Pokémon card.

HP
Hit Points of the Pokémon.

Type
Pokémon type (e.g., Grass, Fire, Water).

Weakness
Type weakness of the Pokémon.

Resistance (Type)
Type resistance of the Pokémon.

Retreat
Retreat cost required to switch the Pokémon.

Move Name
Name of the attack or move.

Cost
Energy cost required to use the move.

Damage
Damage dealt by the move.

Effect Explanation
Description of the move effect or additional rule text.

## API documentation (to interact with the game simulator)
- https://matsuoinstitute.github.io/cabt/ (TODO: add tooling for this in the future?)

## Pokémon TCG rules and specifics
- These will probably be important for some things. I haven't provided you an actual rule book yet, so for now just be careful with assumptions about the game.

## Rule-based agent creation (`rulebased_agent_creator/`)
- `rulebased_agent_creator/creating_rulebased_agents.md` — the workflow for building a new rule-based agent for a deck (see also `game_agents/rulebased/<deck>.py` for existing examples).
- `rulebased_agent_creator/validate_deck.py` — validates a decklist CSV (one card ID per row) against `data/EN_Card_Data.csv`: flags unknown card IDs and copy-count violations (max 4 per card, unlimited for Basic Energy). Run: `python rulebased_agent_creator/validate_deck.py decks/<deck>.csv`.

## Modified local engine — state replicator (`ptcg_engine/`)
The competition engine's C++ source (`ptcg_engine/ptcgProgram 22/`, the source of the vendored `libcg.so`) is committed here **with local modifications** that the stock engine lacks: battles can be seeded (deterministic) and saved/re-instantiated from an arbitrary mid-game state. Three added C-ABI exports — `BattleStartSeeded`, `SaveState`, `BattleStartFromState` (`Export.cpp`/`Api.h`; canonical serialization in `State.h`/`FixedList.h`/`GameFunction.h`) — power:
- `record_battle.py --seed N --record-state` — deterministic traces carrying a full-state engine blob per step;
- `utils/resume_battle.py` — restore any recorded position as a *live* battle, then verify byte-exact replay (`--replay`) or play it out with fresh agents (`--p0/--p1`);
- `cg_state.py` — the Python bindings (degrade cleanly on the stock `.so`; `has_state_api()`).

Build with `bash utils/build_libcg.sh`, then swap in via `CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib"`. Acceptance test: `utils/run_state_restore_test.py` (round-trip byte equality, god-view equality, deterministic replay, play-on smoke).

**License note:** the engine is competition-use-only (`ptcg_engine/ptcgProgram 22/LICENSES`) — this repo must stay **private**, the source must not be redistributed, and neither the source nor a rebuilt `.so` ever goes into a Kaggle submission bundle (submissions use the stock vendored `cg/`).

# Skills
- Described in docs/skills_and_hooks.md.

# Benchmark, verifiable results, evals, tests
- We need to make our work have verifiable results as much as possible, to ensure we're actually making progress as we iterate. Some things will be hard to verify, so we'll have to start small and gradually think more about this as we go. For instance, checking whether a battle resulted in a win or loss should be trivial, but benchmarking intermediate factors to see what led to that result will be more challenging.

## Eval harness (`eval.py`) — our primary model-performance signal

**Strategy & roadmap: see [`EVAL.md`](EVAL.md)** — the single entry point (what we measure, what's built, what to build next). This section is the quickstart.

`docs/eval_info.md` lays out a staged methodology for benchmarking decision quality under sparse, binary (win/loss) rewards. We implement its **Stage 1** ("cheap, do-first"): head-to-head **win rate vs fixed baselines, with confidence intervals and seat-swapping, plus a rough Elo over the pool**. This is the directly-checkable signal for "did the model get better?".

```bash
python eval.py                       # default sanity battery
python eval.py --games 50            # more games per model matchup (tighter CIs)
python eval.py --search-count 8      # deeper MCTS for the model (stronger, slower)
python eval.py --mode gauntlet --checkpoint out/model0.pth   # one MyModel-arch ckpt vs frozen anchor + the 4 rule-based decks
```

What it does (`eval.py`, built on `tcg_model.py` + `game_agents/rulebased/lucario.py`):
- **Agents:** `random`, `rulebased` (Mega-Lucario, plays `deck.csv`), and any checkpoint (`MyModel` + `mcts_agent`, plays `sample_deck`). Defaults compare `out/mixed_final.pth` vs `out/model0.pth`.
- **`play_match`** alternates seats every game (cancels first-player advantage) and routes each decision by `obs.current.yourIndex`.
- **Reports** W/L/D, win% over *decisive* games with a **95% Wilson CI**, and an anchored Bradley-Terry **Elo** (a stable pool agent pegged to 0). Cheap model-free matchups run with `--baseline-games` (default 200) for tight CIs; model matchups use `--games`.

Key caveats (printed by the tool; from `docs/eval_info.md`):
- **Deck confound:** win rate vs rule-based mixes agent skill with the deck matchup (`sample_deck` vs Lucario). The *mirror* matches (vs random, ckpt-vs-ckpt — same deck both sides) isolate skill better.
- **High outcome variance** → wide CIs at small n; raise `--games`. Win% excludes draws (reported separately).

**Current standing:** volatile — read it from the tools, not this file. Run `python eval.py --mode gauntlet --checkpoint out/<ckpt>.pth` (one checkpoint vs the frozen anchor + the rule-based field) or `python eval_sb3.py --ckpt out/<ckpt>.zip` (SB3 deck×opponent matrix); dated verdicts live in `work_log.txt` and the memory notes. Standing headline so far: the hand-written rule-based agents remain ahead of every trained checkpoint.

**Deferred (Stage 2+, not yet built):** the value-model / Win-Probability-Added process metrics and puzzle suites from `docs/eval_info.md`. WPA needs a *calibrated* value model; ours (the encoder value head) is only weakly trained, and `docs/eval_info.md` warns every value-metric inherits the model's calibration — so this waits until we have a properly trained checkpoint and can validate calibration.

# Training

Two trainer generations plus an unwired data source. The **active** stack is the SB3 MaskablePPO
experiment (`train_sb3.py`); the original MCTS self-play starter is kept as legacy reference. (A
hand-rolled from-scratch PPO stack preceded SB3 and was retired 2026-07-06 — its featurization
lives on in `sb3_features.py` and its findings in `docs/ppo/`.)

## 1. SB3 MaskablePPO (active): `train_sb3.py`

Built on stable-baselines3 (sb3-contrib MaskablePPO): `sb3_env.py` (`CGEnv`, a gymnasium wrapper
inverting the cg battle loop into `reset()`/`step()`, with the rule-based opponent field via
`make_opponent`), `sb3_features.py` (the cg-obs featurization — dense state/option tensors),
`train_sb3.py` (argparse + env construction + SB3's PPO), and `eval_sb3.py` (deck × opponent matrix
with Wilson CIs). Read the file docstrings and `--help` for the contract; the upcoming self-play +
multi-select roadmap is `docs/plans/plan.md` (implemented); run history and the PPO-ceiling findings live in `work_log.txt`,
`docs/reports/sb3_report.md`, and `docs/ppo/` (`ppo_ceiling_report.md`, `ppo_failure_investigation.md`,
`ppo_saturation_report.md`). SB3 deps (training-side only): `pip install -e
reference_repos/stable-baselines3` + `pip install sb3-contrib`.

Fast local smoke (~4 tiny updates, writes a checkpoint zip under `out/`):
```bash
python train_sb3.py --smoke
```

## 2. Legacy self-play starter (`starter_rl_kaggle.py`)

`starter_rl_kaggle.py` is a working RL+MCTS self-play trainer. Per iteration it:
- runs the `battle_start → battle_select → battle_finish` loop;
- drives MCTS via the Search API (`search_begin/search_step/search_end`), *determinizing* the opponent's hidden cards (mirror determinization: opponent deck/hand sampled from *your own* deck list, basic-energy `1` placeholders for hidden prizes, Riolu `677` for a face-down active — `grep -n "search_begin" tcg_model.py`);
- emits `LearnSample`s (a scalar **value** + per-action **policy** + sparse encoder/decoder vectors); targets come from MCTS visit stats and a TD(λ=0.9) backup of the game result;
- trains the Transformer and checkpoints to `out/modelN.pth` (saved at the *start* of iteration N, so `modelN` is the model *before* iteration N's updates).

Card metadata is read at runtime from the compiled SDK (`all_card_data()`, `all_attack()`), not from any file.

Fast local smoke pass (guaranteed ≥1 gradient step):
```bash
python starter_rl_kaggle.py --iterations 2 --eval-games 2 \
    --selfplay-games 4 --search-count 2 --batch-size 8
```
Prints per-iteration `samples=… batches=… mean_loss=…` and eval win-rate vs **both** a random opponent and the rule-based agent. Gotcha: the default `--batch-size` is 128, so a handful of self-play games produces no full batch (`batch_count = samples // batch_size == 0`) and thus *no* gradient step — hence the small `--batch-size` for smoke runs. No args = heavy Kaggle defaults (5 iters, 50 eval / 100 self-play games, search 10, batch 128).

**Opponents:** self-play also faces the rule-based Mega-Lucario agent (importable port: `game_agents/rulebased/lucario.py`) for a fraction of games set by `--rb-frac` (default 0.5; the learner plays `sample_deck`, the opponent plays the Lucario `deck.csv`). Continue an existing checkpoint with `--load out/<ckpt>.pth`, name checkpoints with `--out-prefix`, and the fully-trained model is saved to `out/<prefix>_final.pth` (the per-iteration `out/<prefix>N.pth` are saved *before* each iteration's training). Example "train further" run:
```bash
python starter_rl_kaggle.py --load out/model1.pth --out-prefix mixed \
    --iterations 3 --eval-games 6 --selfplay-games 20 --search-count 8 --batch-size 32
```

## 3. Episode replays / ladder data (available, not yet wired up)

Top-ladder games can be pulled from Kaggle for behavior cloning / imitation learning. Auth is already configured (the `kaggle` CLI v2.2.2 reads `~/.kaggle/access_token`). The chain (competition `pokemon-tcg-ai-battle`):
```bash
kaggle competitions leaderboard pokemon-tcg-ai-battle --show   # → teamId
kaggle competitions team-submissions <team_id>                 # → submission id + score
kaggle competitions episodes <submission_id>                   # → episode ids (~168/sub)
kaggle competitions replay <episode_id> -p data/replays        # → episode-<id>-replay.json
```
(Kaggle simulation-competitions guide: https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md)

A replay is a Kaggle episode JSON: `rewards` (final result, e.g. `[1,-1]`) + `steps`, a list of per-step `[agent0, agent1]` entries (`action`, `observation`, `reward`, `status`). The **acting** agent has `status="ACTIVE"` with a populated `observation.select`; its `action` is the option indices it chose (the first real action is the 60-card deck). Crucially, `observation` is already in the **same dict format the cg SDK uses** (`current`/`select`/…), so BC/IL can reuse `get_encoder_input`/`get_decoder_input` to turn each `(observation, action)` into a supervised policy target — no env reverse-engineering. What's missing is a small parser/training script: **the `cg` SDK has no replay loader.**

Already fetched locally (all under gitignored `data/`): a sample replay `data/replays/episode-81475913-replay.json` (top team, 1 game) and the card-metadata `data/EN_Card_Data.csv` / `data/JP_Card_Data.csv` (the matching ID-list PDFs are ~320 MB and were skipped; both CSVs and PDFs are attached to `pokemon-tcg-ai-battle` and the strategy track).

**Open decision:** build the replay → BC/IL parser next, or push further on self-play first.

# Running heavy jobs on Kaggle (operational lessons)

Local hardware is small (WSL2), so heavy training/eval can be offloaded to Kaggle script-kernels. (The PPO-specific sweep driver `kaggle/push_sweep.py` was retired with the hand-rolled PPO stack on 2026-07-06 — recoverable from git at commit `4ee6421` as a template for a future SB3 Kaggle run.) The operational lessons below are general and still apply:

- **Concurrency cap:** Kaggle runs at most **5 concurrent batch CPU sessions**. We pushed 12 kernels at once; the extra 7 were rejected with a session-cap error. **Push in waves of ≤5** and plan the matrix around that cap rather than firing everything at once.
- **Zombie kernels:** rejected/failed pushes can leave **empty "zombie" kernel records**; re-pushing to those same slugs then fails with a misleading `Notebook not found`. **Use a distinct slug per job** so a failed push doesn't poison the slug.
- **No CLI delete:** the Kaggle CLI has **no `kernels delete`** — zombie/partial kernels must be removed via the **web UI**.
- **Throughput:** each CPU kernel ran **~1–1.5 h**; results pull back via the CLI. Net effect of the above: only **6 of 12** kernels completed that sweep.

# How to Submit to this Competition (only do this when explicitly asked)
See kaggle-submission skill.