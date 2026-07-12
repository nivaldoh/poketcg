# Lucario-mirror specialization campaign (2026-07-10)

Goal (user directive): shrink the problem to ONE deck (lucario) vs its rule-based mirror teacher;
get above 50% in that single cell, or find the saturation point and pinpoint the cause.
Plan: `~/.claude/plans/ticklish-frolicking-knuth.md`. All reads = `eval_sb3.py --decks lucario
--opponents lucario`, sampled unless noted, seat-swapped, Wilson 95% CIs. Teacher-vs-teacher
lucario mirror is balanced (bc_collect 4k games: seat-a 1999/1999/2) → 50% = teacher parity.

## Headline

**82.6% (CI 79.8–85.1), 661/139/0, n=800 sampled** — `out/lucb2_ckpt_204800_steps.zip`.
Greedy read: 85.0% (79.4–89.3), n=200. Zero draws in every eval of the campaign.

The >50% target is exceeded with the Wilson lower bound at ~80%. **BUT (flagged, pre-registered
signature F): it is a degenerate exploit of the fixed teacher.** Win-mode audit
(`record_battle.py`, 100 games, `out/lucb2_audit_trace.json`): **all 82 model wins end by
DECK-OUT**, vs the teacher-mirror norm of prize wins ({prize: 19, deck: 3} over 50 teacher-mirror
games). The model learned to outlast the scripted teacher until its deck ran empty — a legal,
draw-free, reproducible win strategy against THIS opponent, not general strength. Expect it not to
transfer off-cell (untested, out of scope per directive).

| stage | ckpt | read |
|---|---|---|
| historical best (4-deck bcft3) | `bcft3_ckpt_153600` | **37.2 (32.7–42.1)** n=400 (A_ft) |
| 4-deck pure BC clone | `bcpure_sb3` | 22.8 (18.9–27.1) n=400 (A_bc) |
| + pin opponent, more quiet PPO (Ph 2) | lucpin1 best | 33.5 (29.1–38.3) n=400 — FLAT |
| lucario-only BC clone, 4k games (Ph 3) | `bclucpure_sb3` | **38.8 (34.1–43.6)** n=400 |
| + quiet PPO ×2 legs (Ph 4 Arm A) | lucft1/2 best | **45.9–46.8** n=400 — saturated |
| + bold PPO leg 1 (Ph 4 Arm B) | lucb1_307200 | 84.0 (78.3–88.4) n=200, still rising |
| + bold PPO leg 2 (ceiling) | **lucb2_204800** | **82.6 (79.8–85.1) n=800** — saturated ~83–85 |

## What moved the needle (in causal order)

1. **Single-pairing BC data at scale** (4000 mirror games / 537,922 decisions vs 125 games
   before): holdout acc 0.796 → **0.934** (single 0.925 / multi 0.964; floor 0.203), still
   climbing at epoch 8 — and the pure clone alone matched the old all-deck+PPO best
   (38.8 vs 37.2). Win rate tracked clone accuracy (0.796→22.8%, 0.934→38.8%).
2. **PPO fine-tune ON the clone** (quiet recipe `--ent-coef 0 --lr 3e-5 --target-kl 0.01`):
   +8pp to ~46–47%, then saturated across two 307k legs.
3. **The optimizer budget, not the model, was the binding constraint** (pre-registered signature
   D: approx_kl 0.0025–0.0067 ≪ target 0.01, clip_fraction ~0.02). Arm B (`--lr 1e-4
   --target-kl 0.02`, everything else identical) took the SAME policy from 46.8 → 84.0 in one
   307k leg (~26 min). A second bold leg saturated at ~83–85% (4 checkpoints, no CI-clearing
   gain) = the setup's ceiling.

## Ceiling diagnosis (the pre-registered tree)

- **D (too timid) — CONFIRMED and was THE blocker at ~46%.** Fixed by Arm B; worth 37pp.
- **F (degenerate teacher exploit) — CONFIRMED at the top.** 100% deck-out wins (audit above);
  ep_len_mean fell ~65 → ~39 learner decisions/game in the exploit regime. The ~83–85% ceiling is
  the exploit's reliability limit vs teacher+engine stochasticity, not a capacity limit.
- **A (clone accuracy limit) — NOT the ceiling**: acc still climbing at epoch 8; and PPO blew past
  the clone level anyway.
- **C (forgetting) — did not occur** under either recipe here (no decay below any leg's t=0).
- **E (credit-limited) — did NOT bind in this regime**: with BC-initialized policy+critic
  (lucario-only value ev 0.170 at epoch 1 ≈ the 0.167 supervised bar; 0.092 at best-by-acc epoch 8)
  and enough update budget, terminal-only reward was sufficient for a 37pp climb. Historical
  context preserved: on the 4-deck task with stock nets, every credit lever failed — this campaign
  shows the plateau there was at least partly optimizer-budget, and PPO answered the credit
  question here by ROUTING AROUND the prize race (deck-out), not solving it.
- **B (distribution shift) — moot** above the clone; the clone's own 38.8-vs-parity gap is
  consistent with compounding per-decision error (0.925 single-acc over ~70 decisions).

## Phase log (details)

- **Phase 0** anchors n=400: A_ft 37.2 (149/251/0), A_bc 22.8 (91/309/0). Gate vs report read: OK.
- **Phase 1** plumbing (~25 lines): `sb3_env.make_opponent` gained the `rulebased:<deck>` pinned
  spec (composes with `@eps`/comma-mix; self-test asserts added, green); `bc_collect.py` gained
  `--teachers` + `--games-per-shard` chunked writes (replay self-test per chunk; RAM protection —
  a 4k-game pairing would otherwise be a ~2.3GB single pickle at both collect and train ends).
  All smokes green (`sb3_env.py`, `run_smoketest_sb3_env.py`, `train_sb3 --smoke` with pinned
  opponent, `bc_collect --smoke` chunked + default).
- **Phase 2** (pin opponent, fine-tune old best, 307k): curve flat 27–40.5 (n=200); point-best
  40.5 failed n=400 confirm (33.5). Verdict: the old clone's specialization headroom was spent.
- **Phase 3** (lucario BC at scale): collect 407s/4000 games; bc_train 8 epochs 48 min
  (`out/bc_core_lucario.pt`); value ev peaked 0.172 @ epoch 2 → 0.092 @ epoch 8 (known
  critic-overfit pattern, did not block). Pure-clone gate: 38.8 (34.1–43.6) — G-acc + G-clone
  passed.
- **Phase 4** Arm A legs 1–2 (quiet): 45.9 (41.0–50.8) then 46.8 (41.9–51.6) — saturated.
  Arm B legs 1–2 (bold): 84.0 → ceiling ~83–85; eval-best lucb2_204800 confirmed 82.6 at n=800.
  (n=200 curve points of 50.0/51.0 in the quiet legs repeatedly failed n=400 confirms —
  winner's-curse discipline earned its keep.)
- **Phase 5** (self-play mix): never triggered — its condition (saturation <50%) did not occur.
- Wall-clock: ~5 bold/quiet PPO legs ≈ 26 min each + ~13 min curve each; BC 48 min; collect 7 min.

## Artifacts (out/, gitignored)

- `bc_data_lucario/` (16 chunk shards, 2.3GB), `bc_collect_lucario.log`
- `bc_core_lucario.pt`, `bc_train_lucario.log`, `bclucpure_sb3.zip`, `bclucpure_eval.log`
- `lucpin1_*`, `lucft1_*`, `lucft2_*`, `lucb1_*`, `lucb2_*` (ckpt families + `_train.log` +
  `_curve.log`), `luc_anchor_ft.log`, `luc_anchor_bc.log`, `lucb2_headline.log`
- **Eval-best: `out/lucb2_ckpt_204800_steps.zip`**; audit trace `out/lucb2_audit_trace.json`
  (open in `visualizer.html` to step through the deck-out games)

## Next levers (ranked)

1. **Make the win honest**: mix opponents to break the single-teacher exploit —
   `rulebased:lucario,self` (frozen snapshots) or `rulebased:lucario@0.1` — and watch whether
   prize-mode wins appear while staying >50%. The `rulebased:<deck>` spec makes per-cell
   curricula cheap now.
2. **Field transfer read** (1 command): `eval_sb3.py --ckpt out/lucb2_ckpt_204800_steps.zip`
   full 4×4 — quantifies the specialist's off-cell regression before any scale-up decision.
3. **Recipe upgrade for the 4-deck task**: the bold budget (lr 1e-4, target-kl 0.02) was never
   tried on the 4-deck BC warm-start (bcft3 used the quiet recipe) — the 24% pooled plateau may
   be partly the same artifact. Cheap to test with the existing `bc_core.pt`.
4. **More BC data/epochs** still un-plateaued (acc 0.934 rising) if a stronger clone is wanted.
5. Deferred as before: SB3-free submission mirror, value-only critic refit (ev overfit persists
   but did not block), Kaggle-replay IL.
