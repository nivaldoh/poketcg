# SB3 MaskablePPO ceiling campaign — self-play mixtures, ε-curriculum, saturation (2026-07-08)

**Question.** After low-hanging fruit and a reasonable amount of training, how far does the current
architecture get — SB3 MaskablePPO, stock `MultiInputPolicy` positional head, terminal-only ±1
reward (`gamma=1.0`), multi-select sub-step decomposition (obs 303/72) — before a significantly
different approach is warranted?

**Answer (headline).** The architecture saturates by **~150k decisions** at **~30% Lucario-mirror /
~13% pooled-mirror** (sampled, n=200/cell), and **none of the three cheap levers moves it**: more
scale (819k), frozen self-play grounded 2:1 in the rule-based field, or an ε-weakened curriculum.
The campaign-best checkpoint still **loses all 16 deck×opponent cells** (every CI excludes 50%).
This is consistent with, and further confirms, the standing prize-race diagnosis: the binding
constraint is the credit signal (terminal-only reward), not capacity, steps, or opponent mix.
A significantly different approach is warranted (see Recommendation).

## What was run

New training-side machinery (uncommitted; ~50 lines, all smoke-gated, regression smoketest green):
`self` now composes into opponent mixtures (`--opponent rulebased,rulebased,self`, repetition =
weighting; `self@eps` works), `--selfplay-refresh N` freezes the self opponent to a snapshot zip
re-saved every N steps (`SelfPlayRefreshCallback`; 0 = live mirror), and `--init-from <zip>`
warm-starts a leg from a prior checkpoint (policy weights only — fresh Adam/lr/timesteps).

Three legs (cuda, `n_steps=512, batch=128`, defaults, seed 0; eval: sampled, seed 11, 200 g/cell,
Wilson CIs; checkpoints every 51200):

| Leg | Run | Opponent | Steps | Wall | Result (Lucario-mirror curve / pooled mirror) |
|---|---|---|---|---|---|
| 1 | `rb800k` (fresh) | `rulebased` | 819,200 | 51 min | flat 24.0–32.5% across all 16 ckpts; pooled 11.25–13.25% |
| 2 | `mix500k` (init-from rb800k@153k) | `rulebased,rulebased,self` + refresh 51200 | 512,000 | 40 min | flat 25.0–35.0%; pooled 11.4–12.4% |
| 3 | `eps500k` (init-from rb800k@153k) | `rulebased,rulebased@0.25` | 512,000 | 34 min | flat 25.5–36.0%; pooled 13.15–13.6% |

Every curve point's CI overlaps every other point's, within and across legs: there is no knee to
find — the plateau is reached by the first or third checkpoint and never left. Legs 2 and 3 are
parallel branches off the same Leg-1 eval-best base (`rb800k_ckpt_153600`), so their reads are
directly comparable.

## Campaign-best matrix (eps500k_ckpt_460800 — a noise-level pick among statistical equals)

```
model\opp     lucario         abomasnow       dragapult_ex    iono
lucario       30.5(25-37)*    10.0( 7-15)      4.0( 2- 8)      8.5( 5-13)
abomasnow     16.0(12-22)     10.5( 7-16)*    11.0( 7-16)      6.0( 3-10)
dragapult_ex  20.0(15-26)      5.0( 3- 9)      2.0( 1- 5)*     2.5( 1- 6)
iono          12.0( 8-17)     12.5( 9-18)      3.5( 2- 7)      7.5( 5-12)*
```
Loses all 16 cells. Mirror pattern unchanged from every prior era: Lucario learnable (~30%),
Abomasnow weak (~10%), Dragapult/Iono dead (2–8%) — the ε-curriculum did not revive the dead
columns (its target mechanism, reward de-starvation, produced no measurable effect).

## Specific findings

1. **Self-play non-transfer now extends to grounded mixtures.** The legacy finding (pure self-play
   crushes random, ~0–5% vs the field) is now confirmed under SB3 + multi-select for the stronger
   design: frozen snapshots (51.2k refresh), majority rule-based grounding, warm-started from a
   field-trained base. The mixture neither helped nor hurt any field read. Training-time reward
   matched the mixture arithmetic exactly (ep_rew_mean ≈ −0.45 vs −0.8 pure-field), i.e. the
   machinery works; the transfer just isn't there.
2. **Multi-select-era stack reads slightly below the pre-multi-select reference.** `sb3big`
   (299/70 obs, 1.5M steps) pooled ≈ 18.4% vs this campaign's ~13% — pooled CIs barely separate
   (15.7–21.1 vs 11.1–15.8). Confounded (different obs width, node distribution, and step count;
   the flat curves argue steps are not the difference) — **flagged for theory review, not claimed
   as a regression**. Plausible mechanism: multi-select roughly doubles decisions/game
   (ep_len ~37 vs ~16–22), diluting per-decision credit under terminal-only reward.
3. **Saturation is extremely early** (~150k decisions ≈ 15 min of training on this box). Anything
   beyond ~200k steps on this recipe is wasted compute.
4. Throughput for future planning: ~267 fps pure-field, ~216 fps with self-play draws (opponent
   inference cost ~20%). Snapshot refresh cost negligible (~1–2s per 51.2k steps).

## Caveats (eval discipline)

n=200/cell ⇒ ±7pp per cell, pooled 800 ⇒ ±3.5pp; point-best checkpoint picks are winner's-curse
prone (the "13.6%" pooled pick re-read 12.6% in the full matrix — as expected). cg RNG is
unseedable (no CRN; seat-swapped, fixed agent-side seed). The eval field equals the training field
(no holdout opponent) — standing blind spot; generalization is untested. All theory-flavored
claims above (credit-signal binding, multi-select credit dilution) are marked for independent
review per repo convention.

## Recommendation — what "significantly different" should mean

In priority order (existing roadmap items, now with the cheap alternatives exhausted):
1. **Win-probability-aligned per-decision credit** — the named core lever; the engine's per-step
   event log (`obs["logs"]`, see `ptcg_engine_report.md`) already carries damage/KO/prize events.
   Not naive Φ_deck/PBRS (gate-FAILED in legacy); needs its own pre-registered gate.
2. **IL warm-start from real matches** — 47 top-leaderboard Kaggle replays already on disk
   (`out/kaggle_replays/`, `fetch_kaggle_replays.py` accretes more daily), matching the ideas.md
   data-flywheel thesis; then RL fine-tune.
3. **Per-option scoring head** (custom maskable policy) — removes the positional-head flag; on its
   own it was not the binding constraint (sb3_report), so pair it with 1 or 2.

## Artifacts (all under gitignored `out/`)

- **Checkpoints (303/72-obs MaskablePPO zips, ~15MB each):** `rb800k_ckpt_{51200..819200}_steps.zip`
  + `rb800k_sb3.zip`; `mix500k_ckpt_*` + `mix500k_sb3.zip` + `mix500k_selfplay_opp.zip`;
  `eps500k_ckpt_*` + `eps500k_sb3.zip`. Recommended reuse anchor: `rb800k_ckpt_153600_steps.zip`
  (Leg-1 eval-best) or `eps500k_ckpt_460800_steps.zip` (campaign-best).
- **Eval logs:** `{rb800k,mix500k,eps500k}_curve.log`, `{rb800k,mix500k,eps500k}_pooled_mirror.log`,
  `campaign_best_matrix.log`; training logs `{rb800k,mix500k,eps500k}_train.log`.
- **Trajectory data:** `out/traces/campaignbest_vs_{rulebased,abomasnow,dragapult,iono}.json` +
  `campaignbest_selfmirror.json` (8-game seat-swapped sessions; per-decision fog obs, god-view
  board, option probs, critic values — visualizer.html-compatible); `out/kaggle_replays/`
  (47 episodes, ~200MB).
