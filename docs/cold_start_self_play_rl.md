# Cold-Start Self-Play RL for the Pokémon TCG and Structurally Similar Games — A Reference Skill

- expalin that we use MaskedPPO
- explain that likely causes of plateau are: correctness bug (action masking / reward sign / value target), a credit-assignment failure, or a representation defect


- **The three highest-leverage design decisions for a PTCG-like game are [big claim]: (1) permutation-invariant set encoders (DeepSets/Set Transformer) for hand/bench/deck [do we need to train an entire transformer for this?] (2) Monte-Carlo-style return targets (γ near 1, high GAE-λ) because reward is terminal and episodes are finite [this is another claim we'd have to check] and (3) a critic that may see privileged/perfect information — PerfectDou's "perfect information distillation," a "perfect-training-imperfect-execution framework" (Guan Yang et al., NeurIPS 2022) — while the actor stays imperfect-information.** These are validated *in this game family* [more claims].
- **Fix evaluation before touching the algorithm.** Win-rate-against-self is nearly uninformative in a zero-sum game (it trends to 50%). Build fixed reference opponents, a relative-population Elo / round-robin payoff matrix. Use telemetry (explained variance, entropy, approx-KL, advantage stats) to run the differential diagnosis in Section 7.

---

## Section 1 — Prerequisites and Assumptions

**What must be true before cold-start self-play can climb at all:**

- **Representation must be Markov-sufficient for the acting player's information set and permutation-correct.** If the observation is not a sufficient statistic of the information set, or if it imposes spurious order on unordered zones, the value function cannot fit and PPO stalls. (Do this: set encoders; see Section 2.)
- **The critic must be learnable.** With terminal-only ±1 reward and long horizons, a bootstrapping critic has almost nothing to latch onto early. Monte-Carlo return regression is more robust here [verify claim] (Section 3).
- **Legal-action masking must be correct end-to-end** (sampling, log-prob, entropy, value) or gradients silently corrupt (Section 6).
- **The opponent distribution must be non-stationary but not chaotic.** Pure latest-snapshot self-play can cycle; you need either historical snapshots or regularization to a reference policy (Section 4).

**Standing theoretical assumptions and their practical consequences:**

- **Solution concept.** A two-player zero-sum imperfect-information game has a value, and its equilibria are (mixed) Nash equilibria; the maximin/minimax strategy is unexploitable. This is the target our training implicitly chases. Crucially, in imperfect-information games the equilibrium is generally **mixed** (stochastic), so a policy that collapses to determinism is provably suboptimal — this is *opposite* to the perfect-information intuition where deterministic optimal policies exist. **Consequence:** keep entropy alive; do not anneal it to zero.
- **No convergence guarantee for plain policy-gradient self-play.** Naive self-play policy gradient / gradient ascent has **no general convergence guarantee to Nash and can cycle indefinitely** (Poincaré recurrence around the equilibrium). This is well-established theory (Mertikopoulos et al.; Perolat et al. 2021). Two mitigations have provable guarantees: **(a) average-iterate convergence** — the *time-average* of iterates converges (fictitious play, CFR, plain mirror descent), which requires storing/averaging historical policies; **(b) last-iterate convergence** — the *current* policy converges, achievable by adding regularization toward a reference policy (magnetic mirror descent / MMD, Sokota et al., ICLR 2023; R-NaD, Perolat et al., Science 2022).
- **Practical consequence for you:** vanilla MaskablePPO in latest-vs-latest self-play is in the "no guarantee, may cycle" regime. The cheapest fixes that align you with theory are (i) high entropy regularization (approximates a regularized / quantal-response equilibrium), (ii) a KL-to-anchor term (approximates MMD), and (iii) an opponent pool of past snapshots (approximates fictitious play / average-iterate). All three are in-scope for a small team.

**Transfer note:** The Nash/last-iterate theory is proven for two-player zero-sum; PTCG singles fits this cleanly. If your PTCG variant has deck-construction asymmetry or any non-zero-sum element, the guarantees weaken.

---

## Section 2 — Representation / Features / Observation Design

**Do:**
- **Encode unordered zones (hand, bench, deck knowledge, discard) with permutation-invariant set encoders** — DeepSets (Zaheer et al. 2017: `ρ(Σ φ(x))`) or Set Transformer (Lee et al., ICML 2019, attention-based, models inter-card interactions). Validated in this family: VGC-Bench (Pokémon VGC, 2506.10326) and the Grigsby et al. Pokémon transformer (2504.04395) embed each Pokémon/card and aggregate with a Transformer encoder; MAGENTA uses permutation-invariant entity transformers for multi-game RL.
- **Represent each card by a learned identity embedding** plus typed feature slots (HP, energy attached, damage counters, status, attack costs), then pool. Zone/count features (prize counts remaining for both players, hand size, deck size), turn/phase indicator, whose-turn, and public opponent info (active Pokémon, bench, discard, known abilities) go in the global vector.
- **Give the opponent's public state explicitly** — active/bench/discard/prizes-taken — since that is the observable part of the information set.

**Avoid:**
- **Order-sensitive encodings of sets** (a flat concatenation of "hand slot 1..7" fed to an MLP, or an RNN over an unordered hand). Mechanism: the network wastes capacity and samples learning that all `k!` orderings are equivalent, badly hurting sample efficiency; RNNs are order-sensitive by construction (per Set Transformer and MAGENTA motivation). This is a common, silent efficiency killer.

**Imperfect information / belief:**
- **Two validated options.** (a) **Feed only observable info to both actor and critic** (simplest, correct). (b) **Asymmetric / "privileged critic":** let the *value network* see hidden state (opponent hand, deck order) while the *policy* sees only the information set — this is PerfectDou's "perfect information distillation," described as a "perfect-training-imperfect-execution framework that allows the agents to utilize the global information to guide the training of the policies" (Guan Yang et al., NeurIPS 2022), trained "with Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) by self-play." DeepNash uses the same idea (privileged value function). It reduces value-estimation variance dramatically and is one of the strongest known tricks for sparse-reward card games. **Do this if your simulator can expose ground truth at train time** (yours can — it's a Kaggle sim). It costs nothing at inference because the actor never uses privileged info.

**Determinization as data augmentation — and its pitfalls:**
- Determinization = sample a concrete world consistent with the info set and treat it as perfect-information. Useful for *search*, but it has two **well-documented failure modes** (Cowling, Powley, Whitehouse 2012; Frank & Basin): **strategy fusion** — the agent implicitly assumes it can make different decisions in states it cannot actually distinguish, so it "fuses" strategies it could never execute; and **non-locality** — the value of hidden information depends on the opponent's historical choices (e.g., "opponent wouldn't have led that card if they held X"), which perfect-information subgame decomposition ignores. **Consequence for you:** do NOT train the *policy* on determinized perfect-information states as if they were the true observation — you will teach it to rely on information it won't have, capping deployed performance. Determinization is acceptable only inside a belief-aware method (Information-Set MCTS) or as privileged *critic* input, never as actor input.

**Transfer note:** Set encoders, privileged critic, and the determinization pitfalls are all validated in card/imperfect-info games specifically — high confidence.

---

## Section 3 — Credit Assignment Under Sparse Terminal Reward

**The Deep Monte Carlo lesson (well-established, this family):** DouZero (Zha et al., ICML 2021) showed that for a large-action imperfect-information card game with episodic terminal reward, **Monte-Carlo return regression beats TD-bootstrapping**. The paper notes that "DQN and A3C can not make satisfactory progress in DouDizhu... shown to have less than 20% winning percentage against random agents that sample legal moves uniformly." Mechanism: (1) the game is episodic and finite, so MC has no incomplete-episode problem; (2) bootstrapping injects bias and instability when the value function is poorly fit early (exactly the cold-start regime); (3) MC's high variance is tamed by generating many parallel episodes. **This is the single most important credit-assignment insight for your setting.**

**Concrete implications for GAE-λ and γ in PPO:**
- **Set γ very close to 1 (γ = 0.997–1.0).** Since reward is terminal and episodes are bounded (a PTCG game ends), discounting mainly adds bias that under-credits early, pivotal decisions (setup turns). PerfectDou used **γ = 1.0** exactly (Table 7); gym-locm used γ = 0.99 "to slightly encourage shorter episodes yet still reward all choices in a battle" (Vieira et al., SBGames 2022). **Default: start at γ = 0.999**; use 1.0 only if episodes are strictly bounded (turn cap / deck-out) and you handle the value target carefully.
- **Set GAE-λ high (0.95–1.0).** High λ pushes GAE toward the Monte-Carlo return (less bootstrap bias, more variance) — the DouZero-consistent choice. PerfectDou used **λ = 0.95** (Table 7). **Default: λ = 0.95, and try λ = 1.0** (pure MC advantage) as an ablation given the sparse-terminal structure.
- Diagnostic: if value **explained variance stays near 0 or negative** at 150k steps, your critic is not learning the return — raise λ toward 1, consider the privileged critic, and verify reward bookkeeping (Section 9).

**Potential-based reward shaping (PBRS) — what it provably can and cannot do:**
- **Theorem (Ng, Harada & Russell 1999):** shaping of the form `F(s,s') = γΦ(s') − Φ(s)` leaves the set of optimal policies **invariant** — it cannot change *what* is optimal, only potentially *how fast* you learn. Wiewiora (2003): PBRS is equivalent to Q-value initialization. Devlin & Kudenko (2012): the potential may be time-varying (dynamic PBRS) and invariance still holds.
- **Consequence for your prize-differential shaping:** if you implemented it as a proper potential difference (`γΦ(s') − Φ(s)`, Φ = f(prize differential)), it is *guaranteed not to change the optimum* — so it CANNOT be the cause of a wrong policy, but it ALSO explains why it "did not fix" the plateau: if the bottleneck is representation, masking, or self-play dynamics, PBRS provably cannot help. If you added a **raw prize-differential bonus that is NOT a potential difference** (e.g., +c each time you take a prize), you have **changed the optimal policy** — this can induce reward-hacking (e.g., trading inefficiently to grab prizes) and must be removed (the classic "bug" Ng et al. warn about).
- **In a two-player zero-sum game, per-agent PBRS must preserve the zero-sum structure and must not introduce/remove Nash equilibria** (general-sum extensions). Keep shaping symmetric and potential-based, or drop it.
- **Verdict:** correctly-formed PBRS is low-risk but low-reward here; it will not break a representation/credit/self-play plateau. PerfectDou's "oracle reward" (distance-to-win via a dynamic-programming estimate of minimum steps to empty the hand, scaled and then removed in fine-tuning) worked because it is a *dense, informative* potential — a much stronger signal than prize count, and it required a domain-specific solver.

**Learned credit assignment (RUDDER etc.):**
- RUDDER (Arjona-Medina et al. 2018) redistributes delayed reward via an LSTM return-decomposition to make expected future reward ≈ 0, and is "return-equivalent" (preserves optimal policies). It is powerful for delayed reward in *single-agent* MDPs but adds substantial complexity (monotonic-LSTM, contribution analysis, auxiliary losses) and is **not validated in two-player self-play card games**. **Verdict: defer.** For your setting, the privileged critic + MC returns gets most of the credit-assignment benefit at a fraction of the risk.

**Transfer note:** DMC/MC-return superiority and PBRS theory are well-established and family-validated. RUDDER transfer to self-play card games is speculative.

---

## Section 4 — Self-Play Dynamics and Opponent Management

**Why naive latest-snapshot self-play fails (well-established):**
- **Cycling / non-transitivity:** in games with rock-paper-scissors structure, "improvement" against the current opponent can be undone by the next; the policy chases its tail (Balduzzi et al. 2019; Vinyals et al. 2019 note self-play "is more forgetful").
- **Overfitting to the current self:** the policy specializes to beat one opponent and forgets general skill.
- **This is last-iterate vs average-iterate again (Section 1):** naive self-play tracks the (non-convergent) last iterate.

**Fixes, ordered by cost:**

1. **Snapshot opponent pool (fictitious self-play, Heinrich et al. 2015 / Heinrich & Silver 2016).** Freeze copies of the policy periodically; sample opponents from the pool (uniform = FSP). Cheap, robust, directly attacks forgetting/cycling. **Do this first.** gym-locm (Vieira et al.) refreshed the frozen opponent "every 10, 100, or 1,000 episodes" as a tuned hyperparameter — snapshot cadence matters.
2. **KL-to-anchor regularization (approximates MMD / R-NaD; last-iterate convergence).** Add a penalty pulling the policy toward a fixed reference policy that is periodically updated to the current policy. MMD (Sokota et al. 2023) and R-NaD/DeepNash (Perolat et al. 2022) both use this to converge to (regularized) Nash instead of cycling. R-NaD's mechanism: transform per-turn reward by `−η·log(π/π_reg)` for the acting player and `+η·log(π/π_reg)` for the opponent (keeping it zero-sum), run the dynamics to a fixed point, then set `π_reg ← fixed point` and repeat (best regularization strength η ≈ 0.05 in Perolat et al. 2021, annealed). Low code cost in PPO — add a backward-KL term to the loss (Rudolph et al. 2025's "PPO+KL"). **Do this second if snapshots alone still cycle.**
3. **Prioritized Fictitious Self-Play (PFSP, Vinyals et al. 2019 / AlphaStar).** Sample opponents proportional to a function of their win-rate against you — `f(x)=(1−x)^p` to focus on hard opponents, or `f(x)=x(1−x)` to match your own level. Buys faster improvement and a less-exploitable result than uniform FSP. **Overkill for the first milestone; adopt once basics work.**
4. **Full league training (main/main-exploiter/league-exploiter, AlphaStar).** Powerful but heavy — multiple concurrent agents, large compute. **Overkill for a solo/small team on one GPU. Defer/skip.**

**Practical defaults (transfer: medium — from AlphaStar/gym-locm, not PTCG specifically):**
- Snapshot cadence: every ~5–20 policy updates (or a few hundred to ~1000 episodes). Too frequent = pool ≈ latest (cycling returns); too rare = stale opponents.
- Pool size: keep ~5–20 diverse snapshots; sample a rule-based heuristic-bot fraction (~10–30%) to anchor absolute skill and prevent joint drift into degenerate mutual strategies.
- Keep a fixed reference-opponent set OUTSIDE the training pool for evaluation only (Section 7).
- Rule-based bots as a minority opponent are a legitimate curriculum floor — but do NOT put their trajectories into the PPO buffer as if on-policy (Section 8); on-policy PPO must learn from its own action distribution.

---

## Section 5 — Exploration in Large Masked Discrete Action Spaces

**Do:**
- **Use a substantially larger entropy coefficient than single-agent defaults.** Rudolph et al. (2025) found "the entropy coefficients with the best average performance are between 0.05 and 0.2, larger than any of the default entropy coefficients for PPO in Stable Baselines, CleanRL, RLlib, OpenSpiel... which range between 0 and 0.01." **This is possibly the highest-leverage single knob for your plateau** — if you are running SB3's default `ent_coef=0.0`, that alone can cause premature determinism and a scale-invariant plateau. (PerfectDou used an entropy weight of 0.1, consistent with this range.) **Default: sweep ent_coef ∈ {0.01, 0.03, 0.05, 0.1, 0.2}.**
- Because equilibria are mixed (Section 1), sustained stochasticity is *correct*, not just exploratory.

**Detecting exploration collapse (diagnostic decision rule):**
- Track **policy entropy over the masked distribution** each update. **If entropy drops sharply early and flatlines near a low value while win-rate-vs-reference stops improving → exploration collapse.** Suspect too-low `ent_coef`, too-high learning rate (aggressive updates collapse entropy — well-documented), or a masking bug shrinking the effective action set.
- **Normalize for the mask:** entropy in states with 3 legal actions is naturally lower than in states with 40. Track entropy *relative to log(#legal actions)* (normalized entropy in [0,1]) or you will misread healthy low-entropy endgame states as collapse.

**Curriculum:**
- The rule-based-bot minority (Section 4) is your cheapest curriculum. Explicit exploration curricula (progressively harder opponents via PFSP `f(x)=x(1−x)`) help but are secondary.

**Keeping exploration alive without breaking on-policy stability:**
- Prefer the entropy bonus + KL-to-anchor over ad-hoc action noise. Watch **approx-KL per update**: if it spikes (> ~0.02–0.05) you are taking destabilizing steps that also crush entropy; lower LR or clip range. (Transfer: PPO tuning heuristics, medium confidence for card games.)

---

## Section 6 — Action-Masking Correctness (Implementation-Level)

**The theory (Huang & Ontañón 2020/2022, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"):** masking replaces invalid-action logits with −∞ before softmax, renormalizing over valid actions. This is a **state-dependent differentiable function**; the resulting update is a **valid policy gradient**, and gradients to masked logits are **zero** (correct — they should not be pushed). Masking is empirically critical and its importance grows as the invalid fraction grows (exactly your case: large action space, few legal moves per state).

**How MaskablePPO (sb3-contrib) handles it correctly:** it applies the mask consistently in the `MaskableCategorical` distribution used for **action sampling, log-probability, and entropy**, stores masks in the rollout buffer, and re-applies them during the update so the ratio and entropy are computed over the masked distribution. You MUST use `MaskableEvalCallback` / the maskable `evaluate_policy`, not the base SB3 ones (documented requirement). (gym-locm's agents are built exactly this way: PPO with invalid action masking via sb3-contrib, "before the softmax activation, all logits that refer to invalid actions are set to −∞.")

**Common bugs (each silently caps performance, no error thrown):**
- **Masking only at sampling time, not in the loss.** If log-prob/entropy are computed over the *full* distribution while sampling used the masked one, the importance ratio is inconsistent and gradients are biased. (MaskablePPO avoids this — but if you customized the policy, re-check.)
- **Entropy over the full vs masked distribution.** Full-distribution entropy rewards spreading probability onto *invalid* actions — meaningless and destabilizing. Must be masked-distribution entropy.
- **Gradients leaking through masked logits.** Using a large finite negative (e.g., −1e9) instead of true −∞ masking, or masking after the softmax, can leak tiny gradients and cause NaNs or slow drift. Use the library's masked distribution.
- **Mask/observation desync.** The mask passed to the update must correspond to the *exact* state whose observation is stored. Off-by-one (mask from `s_{t+1}` applied to `s_t`) is a classic silent bug — verify by asserting the sampled action is always legal under the stored mask.
- **Value function conditioning:** the critic does not need the mask (it estimates state value, not action choice), but it MUST see enough of the state to be Markov. Do not accidentally feed the mask as the only signal about legality-relevant state.

**Composite / factored actions (highly relevant to PTCG):** a PTCG turn is a *sequence* of sub-decisions (play energy, use ability, retreat, attack, choose targets). Two encodings:
- **Flat enumeration of full turn-actions** — mask enumerates only legal complete actions. Simple, but the action set can explode combinatorially.
- **Sequential/factored decomposition with a STOP/PASS token** — decide sub-actions autoregressively, each step masked, until STOP. This is how LOCM/µRTS-style agents (Huang & Ontañón) and AlphaStar handle combinatorial actions. **Correctness subtlety:** each sub-action's mask depends on sub-actions already chosen this turn; the log-prob of the composite action is the **sum of masked sub-action log-probs**, and entropy must likewise be summed over the factored distribution. Getting this partially right (e.g., masking only the first factor) is a common silent cap. sb3-contrib's MaskablePPO supports `MultiDiscrete` masks; verify the mask tensor shape matches the factorization.

**Diagnostic:** log the mean number of legal actions and assert 100% of sampled actions are legal. If your agent ever "wants" illegal actions (pre-mask logits concentrated on masked entries) that's fine; if *sampled* actions are ever illegal, masking is broken.

**Transfer note:** action-masking correctness is domain-general and directly validated in strategy/card games (LOCM used MaskablePPO). High confidence.

---

## Section 7 — Evaluation and Diagnostics (HIGHEST PRIORITY)

**Why win-rate-against-self is misleading:** in a symmetric zero-sum game, the current policy vs a copy of itself trends to ~50% *by construction*, regardless of skill. It measures nothing about absolute strength and hides cycling entirely (A beats B beats C beats A can all show 50% in self-play). **Never use self-play win rate as your primary metric.**

**Build trustworthy evaluation (in priority order):**
1. **Fixed reference opponents held out of training:** rule-based heuristic bots of varying quality + a few frozen checkpoints you never train against. Report win-rate vs each. This gives an *absolute, comparable* signal across runs.
2. **Round-robin payoff matrix + relative-population Elo (RP-Elo) / Nash averaging.** Because Elo is unreliable in non-transitive games (Balduzzi et al., "Re-evaluating Evaluation," 2018; "On the Limitations of Elo: Real-World Games are Transitive, not Additive," 2022), compute the full opponent-vs-opponent win matrix and use **Relative Population Performance** (Balduzzi et al. 2019: `v(A,B)=pᵀM q` at the Nash of the payoff matrix) or Nash averaging. RP-Elo (as in JiangJun, 2308.04719) tracks whether a *population* is genuinely improving vs merely cycling.
3. **Exploitability / best-response estimation.** The gold standard for imperfect-info games: how much can a best-responder beat your policy? If your PTCG state/action space is small enough, compute exact exploitability (cf. the exp-a-spiel library, Rudolph et al. 2025). Otherwise, **train a "best-response" agent from scratch against your frozen policy** (approximate exploitability) — if it reaches a high win-rate, you are far from equilibrium. Cautionary example: ByteRL, the LOCM/Hearthstone SOTA, was shown highly exploitable despite beating a top-tier human — Haluska & Kowalski (2024) report "Although ByteRL beat a top-10 Hearthstone player from China, we show that its play in Legends of Code and Magic is highly exploitable." Head-to-head strength ≠ unexploitability.
4. **Transitivity measurement:** count rock-paper-scissor cycles in the payoff matrix; a high cycle count means your snapshot ladder is non-transitive and single-number metrics will mislead.

**Training telemetry — what to watch and how to read it:**

| Channel | Healthy | Stuck / pathological |
|---|---|---|
| **Value explained variance** | rising toward 0.5–0.9 | stuck ≤0 → critic can't predict return |
| **Policy entropy (mask-normalized)** | high early, slow decline, plateaus above floor | crashes to ~0 fast → exploration collapse |
| **Approx-KL per update** | small, stable (~0.005–0.02) | spikes → destabilizing steps |
| **Advantage mean/std** | mean ≈ 0, std stable | std → 0 → no signal; huge → value mis-scaled |
| **Gradient norm** | stable | exploding/vanishing |

**Differential diagnosis of the plateau (the crux — these look identical on a raw win-rate curve):**

- **Credit-assignment-limited:** **explained variance stays near 0/negative** and advantage std is tiny while entropy is still healthy. The critic never learns the terminal return → advantages are noise → policy can't improve. **Check/fix:** raise GAE-λ→1, γ→1, add privileged critic, verify reward reaches the buffer with correct sign (Section 9). *Top suspect for a terminal-reward game plateauing at 150k.*
- **Self-play-dynamics-limited:** telemetry looks *healthy* (good explained variance, healthy entropy, stable KL) but **win-rate vs fixed references oscillates or is flat while self-play win-rate sits at 50%**, and the round-robin matrix shows cycles. **Check/fix:** add snapshot pool / KL-to-anchor / PFSP (Section 4).
- **Representation-limited:** explained variance plateaus at a mediocre level and *no* hyperparameter change (LR, λ, entropy) moves it; the network cannot fit the value even in-sample. **Check/fix:** set encoders, card embeddings, add missing observation features (Section 2). Test: can a supervised probe predict game outcome from your observation encoding? If not, representation is the cap.
- **Exploration-limited:** **entropy collapses early**; increasing `ent_coef` measurably raises the plateau. **Check/fix:** Section 5.

**Decision rule summary:** *Look at explained variance and entropy first.* Low EV → credit/representation. Collapsed entropy → exploration. Both healthy but flat vs references → self-play dynamics. This ordering resolves the "scale-invariant plateau" that raw win-rate cannot.

**Transfer note:** RP-Elo, RPP, exploitability, and Elo's limitations are peer-reviewed and family-validated. The telemetry thresholds are general PPO heuristics (medium confidence for card games).

---

## Section 8 — What NOT to Do (Anti-Patterns)

- **Treating raw sample throughput as the lever when the plateau is scale-invariant.** Your own observation (invariant to scale) is evidence the bottleneck is *bias/correctness/dynamics*, not variance. Mechanism: more samples reduce gradient variance but cannot fix a biased value target, a masking bug, or a cycling opponent. **Diagnose (Section 7) before scaling.**
- **Expecting offline RL (CQL/IQL) to help with no dataset.** CQL/IQL exploit a *fixed dataset* with pessimism; with essentially no data they have nothing to constrain to. The Pokémon offline-RL success (Grigsby et al., 2504.04395) used "a dataset of real human battles spanning more than a decade that grows larger every day" — you explicitly lack this. **Skip.**
- **Expecting learned-model rollouts (MBPO/Dreamer-style) to help when you already have a perfect simulator.** Model-based RL's payoff is sample efficiency when the environment is expensive/unknown. You have a cheap exact simulator; a learned model would only add approximation error. **Skip.**
- **Dumping scripted-bot trajectories into the PPO buffer as an "expert prior."** PPO is on-policy; off-distribution data makes the importance ratio invalid and biases the gradient. Scripted bots are useful as *opponents* (Section 4), not as *behavior data* for the learner. (Using them as data is imitation/behavior-cloning pretraining — a different pipeline, and you said no pretraining.)
- **Reward shaping that changes the optimum** (non-potential prize bonuses). Provably alters the optimal policy (Ng et al. 1999) → reward hacking. **Use only potential-based shaping or none** (Section 3).
- **Evaluating only against self.** Hides cycling and measures nothing absolute (Section 7).
- **Over-investing in CFR/PSRO/NFSP machinery.** Rudolph et al. (2025), across five large imperfect-info games and 7,000+ runs, found FP/DO/CFR-based methods (NFSP, PSRO, ESCHER, R-NaD) **fail to outperform generic policy gradient (PPO/MMD/PPG)** on exact exploitability. Mechanism: these methods add heavy machinery (regret tables, best-response oracles, reservoir buffers) whose benefit is dominated, in practice, by simply using PPO with the *right* entropy coefficient and a stabilizer. **For a small team, a well-tuned MaskablePPO + snapshots + KL-anchor is the rational frontier; deep-CFR is not.**

---

## Section 9 — Correctness / Reproducibility Gotchas Specific to Self-Play

- **Reward sign / zero-sum bookkeeping.** The most common silent capper: from the learner's perspective the terminal reward must be +1 for *its* win and −1 for *its* loss, correctly attributed to the acting player. In self-play both seats generate data; if you log both seats' transitions into one buffer, you MUST flip the reward (and any value target) for the opponent seat, or you train the policy to *lose*. **Assert:** average episodic reward over a batch of self-play games ≈ 0 (zero-sum). Systematically non-zero → sign/side bookkeeping is broken.
- **First-player advantage / side balancing.** PTCG has a going-first advantage. In evaluation, **play every matchup an equal number of times on each side** (swap who goes first) or you will measure side bias, not skill. gym-locm trained *separate* first/second-player networks and swapped sides every episode — at minimum, balance sides in eval.
- **Discount in sparse episodic settings.** γ<1 with terminal-only reward systematically *under-credits* early setup moves; prefer γ≈0.999–1.0 (Section 3). If you use γ=1 you must ensure episodes always terminate (deck-out / turn cap) or returns diverge.
- **Value normalization.** With ±1 terminal reward the value target is already in [−1,1]; heavy reward/return normalization (VecNormalize-style) can distort this. Per-minibatch advantage normalization is fine and standard; observation normalization for set inputs must be per-feature, not across the pooled set. **Don't double-normalize** an already-bounded reward.
- **Snapshot management / train-eval leakage.** Opponents you *train against* must be disjoint from opponents you *evaluate against*, or you overfit to your eval and report inflated skill. Freeze eval references at project start and never train on them.
- **Deterministic, fair evaluation.** Fix and log seeds; freeze opponent weights during eval (no learning); run enough games for tight confidence intervals (win-rate CIs are wide — hundreds of games per matchup).
- **Stochastic-policy evaluation.** Because the equilibrium is mixed, evaluate the *stochastic* policy (sample actions), not a greedy argmax — greedy evaluation of a mixed-optimal policy can be much weaker and is a common misleading step. (DeepNash applies a small post-processing threshold that zeroes out very-low-probability actions rather than going fully greedy.)
- **Mask/observation desync and off-by-one on episode boundaries** (Section 6) — assert legality every step.

**Transfer note:** all of these are correctness invariants, domain-general, and directly applicable.

---

## Section 10 — Prioritized Default Recipe

**Phase 0 — Correctness gate (do first; usable immediately, no infra):**
1. Verify action masking end-to-end (assert sampled actions always legal; entropy & log-prob over masked dist; use MaskablePPO's maskable eval/callback). *Silent cap risk: high.*
2. Verify zero-sum reward bookkeeping (batch mean episodic reward ≈ 0; correct sign per seat). *Silent cap risk: high.*
3. Balance sides in evaluation; freeze held-out reference opponents. *Enables trustworthy signal.*

**Phase 1 — Make the critic learn (attack the most likely plateau cause):**
4. Set **γ = 0.999 (or 1.0 with guaranteed termination), GAE-λ = 0.95 (try 1.0)**.
5. Adopt the **privileged/perfect-information critic** (actor sees info set, critic sees hidden state) — PerfectDou/DeepNash trick; your simulator supports it. *Highest-leverage credit-assignment fix.*
6. Watch **explained variance**; it must climb. If not, revisit representation.

**Phase 2 — Representation (if EV or plateau persists):**
7. **Set encoders (DeepSets → Set Transformer) for hand/bench/deck; learned card embeddings; explicit prize/turn/public-opponent features.** *Highest-leverage sample-efficiency fix.*

**Phase 3 — Exploration & self-play stability:**
8. **Raise entropy coefficient into 0.05–0.2** (sweep). *Possibly the fastest single win given a scale-invariant plateau on default ent_coef.*
9. **Add a snapshot opponent pool (FSP)** with ~10–30% rule-based bots; cadence every few hundred episodes.
10. If cycling persists, **add KL-to-anchor regularization (PPO+KL ≈ MMD)** with a periodically-updated reference policy.

**Phase 4 — Only after basics validate (defer):**
11. PFSP opponent prioritization; approximate best-response/exploitability evaluation as a standing metric.
12. **Defer/skip:** full league training, offline RL, learned world-models, RUDDER, heavy CFR machinery, and test-time MCTS (ruled out by your 10-min no-GPU inference constraint — a cheap feed-forward masked policy is exactly what you want to ship, and is what DeepNash and DouZero deploy: no search at inference).

**Suggested MaskablePPO starting config (transfer caveats in brackets):**
- `gamma = 0.999` [family-validated: PerfectDou 1.0, LOCM 0.99]
- `gae_lambda = 0.95` [family-validated: PerfectDou 0.95]
- `ent_coef = 0.05` then sweep to 0.2 [family-validated: Rudolph et al. 2025, "between 0.05 and 0.2"; PerfectDou used 0.1]
- `clip_range = 0.2`, `learning_rate = 3e-4`, `max_grad_norm = 0.5` [general PPO; PerfectDou LR = 3e-4]
- `n_epochs = 3–5`, per-minibatch advantage normalization on [general PPO]
- `normalize_advantage = True`; do NOT re-normalize the ±1 reward [correctness]
- Many parallel envs help variance but are **infra-dependent**; the plateau is scale-invariant, so prioritize Phases 0–2 before scaling envs.

**What depends on infrastructure vs usable now:** Phases 0–2 and the entropy sweep are usable on one GPU immediately. Large parallel-env throughput, PFSP matchmaking bookkeeping, and exact-exploitability solvers benefit from more infra but are not prerequisites for breaking the current plateau.

---

### Consolidated epistemic ledger
**Well-established AND validated in card/board/imperfect-info games:** DMC/MC-return superiority for sparse-reward card games (DouZero, ICML 2021); privileged critic / perfect-information distillation (PerfectDou, NeurIPS 2022; DeepNash, Science 2022); set encoders for card observations (VGC-Bench; Grigsby et al. 2025; Set Transformer, ICML 2019); action-masking validity (Huang & Ontañón; used in LOCM MaskablePPO); PBRS policy invariance (Ng, Harada & Russell 1999; Wiewiora 2003; Devlin & Kudenko 2012); no-Nash-guarantee/cycling of naive self-play and last-iterate fixes (Perolat et al. 2021/2022; Sokota et al. 2023 MMD); PPO competitive with CFR/PSRO/NFSP on exploitability with high entropy (Rudolph et al. 2025, five games / 7,000+ runs); FSP/PFSP/league (Heinrich & Silver 2016; Vinyals et al. 2019); Elo limitations / RPP / exploitability evaluation (Balduzzi et al. 2018/2019; JiangJun 2023; Haluska & Kowalski 2024); determinization strategy-fusion/non-locality (Cowling, Powley & Whitehouse 2012).
**Speculative / extrapolated / thin transfer:** RUDDER in self-play card games; exact telemetry thresholds for card games; specific snapshot cadence / pool-size numbers (from AlphaStar/LOCM, not PTCG); MMD regularization-coefficient defaults (per-game tuning; α ≈ 0.05 in reproductions/benchmarks). Where a specific number was unavailable — e.g., PerfectDou's PPO clip range, and DeepNash-Stratego's exact regularization strength and anchor-update interval (in the Science supplement) — this is flagged rather than guessed.