# Have we saturated PPO? — a plain-language verdict

*2026-07-06. This answers one question: has our PPO agent stopped improving, and are there
worthwhile improvements left? It's written to be readable without the code open. The detailed
numbers behind it live in `sb3_report.md`, `docs/ppo/ppo_ceiling_report.md`, and the training/eval
outputs under `out/` (gitignored). Where a claim is a measurement I say "measured"; where it's my
best explanation I say "our read (worth a second opinion)", per the repo convention of flagging
theory for independent review.*

---

## Short answer

**Yes — for the way we currently train it, PPO has saturated.** The agent has stopped getting
better and still loses to every opponent we test it against. Two separate versions of the trainer
hit the *same* wall, and training much longer barely helps.

**There are a few things left to try, but none is a sure win.** One is genuinely new (and the most
promising); the rest are either long shots or a change of direction. My honest recommendation is at
the bottom.

---

## What "saturated" means here

We measure skill on a grid of **16 matchups**: our 4 decks × the 4 rule-based opponent decks. The
fairest test is the **"mirror"** matchups — the same deck on both sides — because there the only
difference is skill, not the deck.

- A win rate of **50%** would mean "as good as the opponent."
- Our best matchup is about **40%** (the Lucario mirror). Every one of the 16 matchups is a loss.

"Saturated" means the agent's win rate has flattened out well below that 50% mark and extra training
no longer moves it.

---

## The evidence (measured)

- **Two independent trainers hit the same ceiling.** The older hand-built PPO and the newer one
  (built on the standard Stable-Baselines3 library) both top out at roughly **5–40%** across the
  grid, both lose all 16 matchups, and both peak on the Lucario mirror (~33–40%).
- **Training longer barely helps.** In a large run (~3,000 training rounds, ~1.8 hours), most of the
  improvement arrived in the first **~300 rounds**. The remaining **~2,700 rounds added only about
  +2 percentage points** — a flat line with a slow crawl.
- **The standard tuning knobs are already set correctly** (the value estimator uses a plain output,
  raw targets, normalized updates, etc.). They weren't the thing holding it back.

---

## Why it's stuck (the core problem)

**Measured — the agent doesn't fight for prizes.** Taking all your prize cards is the main way to
win this game. Our agent is bad at it:

- **69% of its losses** are "prize-outs" — the opponent takes all their prizes first — and it's
  usually **about 4 of 6 prizes behind** when it loses.
- Only **~12% of its wins** come from actually winning the prize race. It mostly wins when the
  *opponent* runs out of cards or Pokémon — and often while itself behind on prizes.

**Measured — its internal "who's winning?" estimate is unreliable.** PPO trains a helper called the
*value function* that's supposed to guess how likely the agent is to win from the current board. A
standard health check on it swung wildly (from very wrong to okay across the run), and a stricter,
fairer version of that check sits near **0.1** — barely better than guessing.

**Our read (worth a second opinion) — why those two things happen.** The agent only ever gets a
reward at the *very end* of a game: **+1 for a win, −1 for a loss, and nothing in between.** With no
signal during the game, the trainer has no way to tell which individual *moves* made winning more or
less likely — so it can't learn to play for prizes move-by-move. (The deeper version of this
argument — roughly, that every move in a lost game ends up blamed about equally, good moves and bad
moves alike — is already flagged as a theory claim needing review in our own code comments; treat it
as a strong hypothesis, not proven fact.)

---

## Why the obvious fixes won't work (already tried)

- **"Reward getting ahead on prizes" (reward shaping).** We built this the correct way — it can't be
  gamed, the totals still add up to just win/loss. It **didn't help**. *Our read:* the agent actually
  survives by out-lasting the opponent (attrition) *while behind* on prizes, so nudging it toward the
  prize race may fight the only strategy that's working for it. (This "fights its own strategy" part
  is interpretation, not a direct measurement.)
- **"Reward pushing the opponent toward running out of cards."** Tested without a full training run;
  it made the "who's winning?" estimate *worse*, so it was rejected.
- **Bigger network, more training, self-play, tougher opponents.** All tried; all plateau. Self-play
  in particular learned to beat *itself* but that skill didn't carry over to the real opponents.
- **A fancier move-picking design.** The newer trainer used a *simpler, arguably handicapped*
  move-picker and still matched the older one — so the move-picker was **not** the bottleneck.

---

## What's left to try (honest, and ranked)

1. **Cheap checks first (near-zero cost).** Two quick experiments before spending any real compute:
   a small settings comparison that's been noted but never run, and an **offline test** of a smarter
   "who's winning?" signal. The offline test reuses tooling we already have and needs **no training
   run at all** — it just tells us whether a better signal *would* help before we pay to train with
   it.

2. **The one genuinely new idea (most promising, but not a sure thing).** Give the agent a real
   *win-probability* signal by having the game's own simulator quickly play out the rest of the game
   a few times from the current board and see how often it wins. This is the only signal we have that
   truly reflects "am I winning?" rather than a hand-picked stand-in — and the code to compute it
   **already exists** (it's used elsewhere, just never fed into training). *Caveats:* it's slow on
   our limited machine, and a quick rough playout can be noisy, so it may need care to be reliable.

3. **Change direction: add planning at play-time.** Instead of relying only on the agent's trained
   reflexes, let it *think ahead* a few moves using search when it actually plays. Our own past
   results say this is where real strength has come from on this game ("the search provides the
   strength; training the reflexes alone tends to regress"). This is a bigger change with a higher
   ceiling, and it side-steps the reward problem above entirely.

**Not worth more effort** (the data already argues against these): a fancier move-picker, more raw
training time, self-play, or harder opponents.

---

## Bottom line

We've hit the ceiling of this PPO setup, and we understand *why*: the agent only learns from the
final win/loss, so it never learns to fight for prizes during the game. The remaining ideas are real
but not guaranteed. If we want the best shot at a real jump, the honest move is probably to **add
search at play-time** rather than keep tuning the reward — with the cheap offline check in step 1 as
a low-risk first thing to run.
