# The problem
- Stochastic two-player zero-sum imperfect information game
- Huge action space
- Little to no training data available for warm-start imitation learning
- Multiple actions allower per turn
- Sparse win/loss binary reward
- 10-minute match clock?
- cabt engine for the Pokémon TCG simulator (not parallellizable unless we do something)

# High-level Plan
- Claim: Build a PPO self-play agent from scratch (or copy from the cloned PPO-PyTorch repo?) that scores legal state–action pairs (DouZero-style action-as-feature encoding) and decomposes each multi-action turn into a sequence of single-action timesteps. Two potential templates for this recipe are the Big 2 self-play paper (arXiv 2605.28863, 2026) and "Reevaluating Policy Gradient Methods for Imperfect-Information Games" (Rudolph et al., 2025/ICLR 2026), with "Superhuman AI for Generals.io" (Straka et al., 2026) as a demonstration that pure self-play policy gradient works under sparse win/loss reward.
- Claim: Multiple actions per turn is best handled by the environment, not the algorithm: expose each intra-turn sub-decision as its own MDP timestep with its own legal-action mask. The Kaggle "cabt Engine" already supposedly presents only legal moves each turn and expects an action index back, which fits this decomposition exactly. The ygo-agent project (unpublished) supposedly handles Yu-Gi-Oh's long chains this way via a "multi-select state machine."
- Claim: Sparse binary reward + self-play from scratch is supposedly feasible without reward shaping, as supposedly shown by the Generals.io agent (2026): "Trained for four days on 4x NVIDIA H200 GPUs, our agent reaches #1 on the public 1v1 leaderboard of over 5,000 human players... and beats the two top-ranked humans head-to-head with a combined 199–70 record across 269 ladder matches" — trained purely with sparse win/loss reward and policy gradient, no behavior cloning, no league, plus two simple tricks (an exponential moving average of policy weights, and top-advantage sample filtering).
- I think there was also a paper with RL for Hearthstone
- Refer to PPO implementation cloned from github in the PPO-PyTorch folder. Last updated 3 years ago.
- Deckbuilding and metagame need to be handled separately
- Clone paper repos as extra references

## Stage 1 — Build the simplest viable baseline (start here).
- PPO: policy gradient with clipping (or alternatively, KL divergence). Two networks, one for the policy and another for the state-action values. Has an entropy bonus to encourage exploration.
- Claim: Implement PPO self-play from scratch with: an encoder that embeds board/hand/global state (refer to game rulebook) plus each legal action as a feature vector; an actor that scores legal (state, action) pairs with masking; a value head; entropy regularization. Model each intra-turn sub-decision as its own timestep. Use gamma≈1.0 and GAE. This is supposedly the Big 2 (2026) recipe scaled up, supposedly justified by Rudolph et al. (2025).
- Benchmark to clear: beat a random-legal-move agent and rule-based agents heuristic decisively (the Big 2 and ygo-agent projects both use random/greedy/heuristic opponents as the first ladder).
- Claim: The reference's discrete actor ends in nn.Linear(64, action_dim) → Softmax — a fixed
 action space. cg's action space is not fixed: each state exposes a variable list
 obs.select.option whose length and meaning change every decision. So instead of a fixed-width
 Softmax we use a per-option scoring head: featurize each legal option, score each one to a
 scalar logit, then take a masked softmax over exactly the options present this step. This is
 the correct minimal design for cg and is the single place our PPO departs structurally from the
 reference. v1 scope = single-select policy nodes only (minCount == maxCount == 1,
 len(option) > 1, not all-NUMBER); every other node is auto-resolved and never trained
 (consistent with the established single-select v1).

## Stage 2 — Add the cheap, high-leverage tricks.
- Claim: Adopt current-policy self-play (Big 2 finding) and add a parameter EMA policy for deployment plus top-advantage sample filtering (Generals.io). Add a history encoder (LSTM or small transformer) for opponent/hidden-state inference.
- Benchmark to clear: monotonically rising win rate vs. frozen past checkpoints; stable (non-collapsing) entropy.

## Stage 3 — Scale representation to the full card pool.
- Claim: If generalization across the ~1,300-card pool (confirm competition card pool) is weak, add LLM-generated card-text embeddings (ygo-agent) or learned action embeddings (Axie Infinity). Optionally borrow Metamon's transformer-over-trajectory design for stronger opponent adaptation.

## Stage 4 — Only if you measure high exploitability on the ladder.
- Claim: Add KL-to-uniform regularization first (a one-line magnet term). If still exploitable, consider R-NaD via OpenSpiel (Solly/DeepNash) — but budget materially more engineering and tuning time.

# Ideas and code to take from Paper A, as well as main differences (verify)
A. "Self-Play Reinforcement Learning under Imperfect Information in Big 2"
Authors/venue/year: arXiv:2605.28863 (2026).
What it solves: Self-play RL for Big 2, a card game with hidden information, a large combinatorial action space, delayed/sparse terminal rewards, and action sets that "change sharply between turns." arXivarXiv
Method: A self-play framework that compares PPO against Monte-Carlo Q approximation, SARSA, and Q-learning under a common budget. Architecture separately encodes the information state and the legal candidate actions, then scores each state–action pair (shared card-embedding table over card IDs). Findings directly useful to you: (1) PPO outperforms the value-based methods; (2) "moderate entropy regularization improves PPO by preventing the policy from becoming overly deterministic" (without it, PPO "often sampled its top action with 90+% probability"); (3) "current-policy self-play provides a stronger finite-budget curriculum than checkpoint self-play or fixed-opponent training."
Match to constraints: Very high. It is a card game, imperfect-information, combinatorial actions, sparse terminal reward, self-play from scratch, no dataset. Main caveat: it is 4-player (general-sum) rather than 2-player zero-sum, and Big 2's action space is far smaller than the Pokémon TCG's.
Simplicity: High on both axes. PPO + masking + a state/action encoder is standard and well-documented.
Reusable directly: the state–action scoring architecture, the entropy-regularization finding, and the current-policy self-play recommendation. Needs adaptation: scaling the action encoder to a ~2,000-card pool; handling the 2-player zero-sum structure.

# Ideas and code to take from Paper B, as well as main differences (verify)
B. "Reevaluating Policy Gradient Methods for Imperfect-Information Games"
Authors/venue/year: Max Rudolph, Nathan Lichtlé, Sobhan Mohammadpour, Alexandre Bayen, J. Zico Kolter, Amy Zhang, Gabriele Farina, Eugene Vinitsky, Samuel Sokota; arXiv:2502.08938 (2025), published at ICLR 2026.
What it solves: Tests "The Policy Gradient Hypothesis" — that appropriately tuned generic PG methods (sharing an ethos with magnetic mirror descent: maximize value, regularize toward a magnet/uniform policy, control update size) are competitive with or superior to FP-/DO-/CFR-based DRL in two-player zero-sum imperfect-information games.
Method: Releases the first broadly accessible exact exploitability computations for five large games and conducts the largest-ever exploitability comparison (over 7,000 training runs); concludes the specialized methods "fail to outperform generic policy gradient methods." OpenReview
Match to constraints: This is your justification document. It tells you it is sound to skip CFR/NFSP/PSRO and use PPO with entropy/KL regularization in an imperfect-information self-play setting.
Simplicity: This is precisely a paper about choosing the simpler method. Note: its exploitability results are on benchmark games smaller than the Pokémon TCG; the extrapolation to a game this large is an inference, not a proven result.
Reusable: the regularization recipe (entropy bonus / KL-to-uniform as a "magnet") and the empirical permission to keep the method simple.

# Ideas and code to take from Paper C, as well as main differences (verify)
C. "Superhuman AI for Generals.io Using Self-Play Reinforcement Learning"
Authors/venue/year: Matej Straka, Viliam Lisý, Martin Schmid; arXiv:2606.23348 (2026).
What it solves: A real-time strategy game with strong imperfect information (fog of war), long-horizon planning, and a sparse win/loss reward — trained from scratch by self-play.
Method: A single vision-transformer policy trained end-to-end with a policy-gradient loop and sparse win/loss reward. Key ingredients isolated by ablation: (1) a parameter exponential moving average (EMA) is a consistently stronger deployment policy than the last iterate; (2) top-advantage sample filtering (training on only the highest-advantage fraction of samples) improves both wall-clock and sample efficiency; (3) a spawn-distance curriculum. Crucially, it deliberately drops the "outer loop" (league/PSRO/NFSP) and shows a well-tuned inner loop suffices. arxiv
Match to constraints: High on the hardest parts — sparse binary reward and self-play from scratch with no dataset. Main caveat: "A key enabler is a JAX-native simulator that reaches tens of millions of frames per second on a single GPU, roughly a 10,000x speedup over the prior simulator." Your Kaggle "cabt Engine" simulator will be far slower, so your sample budget will be much smaller.
Simplicity: The recipe is intentionally minimal; EMA and top-advantage filtering are each a few lines.
Reusable: the from-scratch sparse-reward self-play recipe, EMA of policy weights, and top-advantage filtering. Needs adaptation: the throughput assumptions.

# Extra sources of inspiration (verify)
- D. Metamon — "Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers" (the anchor)
Authors/venue/year: Jake Grigsby, Yuqi Xie, Justin Sasek, Steven Zheng, Yuke Zhu; arXiv:2504.04395; RLC 2025. Code/data at metamon.tech.
What it solves: Competitive Pokémon Singles on Pokémon Showdown via offline RL on "5M+ trajectories 'reconstructed' from real human battles and 20M+ generated by self-play between agents," using large sequence (transformer) models that adapt to the opponent from the input trajectory with no search. Reported outcome: "our best agents climb to rankings inside the top 10% of active players," and Metamon "served as both a starter kit and winning baseline for the NeurIPS 2025 PokéAgent Challenge."
Match to constraints: This is your domain anchor and its observation/representation engineering and opponent-adaptation framing transfer well. But the three differences you flagged are real and decisive: (i) its strong results lean on a large dataset for the imitation-learning/offline-RL warmup you do not have; (ii) it has a small fixed action set (moves + switches), whereas you have an enormous combinatorial space; (iii) it acts once per turn, whereas you act multiple times per turn.
Reusable: transformer-over-trajectory for implicit opponent modeling under imperfect information; the self-play fine-tuning stage (their progression ends in offline fine-tuning on self-play data); their observation encoding philosophy. Not reusable: the offline/IL-dependent pipeline as a whole.

- E. DouZero family — action-as-feature encoding for enormous action spaces
Anchor paper: DouZero (Zha et al., ICML 2021) — pre-2023, so cited only as the source of the technique, not as a compliant candidate. Post-2023 continuations: DouZero+ (opponent modeling) and AlphaDou (arXiv:2407.10279, 2024, adds bidding and win-rate-aware advantage).
Technique to borrow: encode each card combination/action as a feature matrix and learn a Q-network that scores (state, legal-action) pairs, so the network generalizes across a huge, turn-varying action set instead of using a fixed softmax head. DouZero used Deep Monte-Carlo (averaged returns) which is also a natural fit for sparse terminal win/loss reward and avoids Q-value overestimation in large action spaces.
Simplicity: Conceptually simple; the DMC variant is extremely simple to implement and parallelize, though sample-hungry.

- F. ygo-agent (Yu-Gi-Oh!) — engineering reference, not a paper
Status: Open-source project by GitHub user sbl1996 (2024). There is no associated arXiv/conference paper — treat it as an engineering reference, not a citable post-2023 paper. (An unrelated arXiv paper, "Deciding winning strategies in Yu-Gi-Oh! TCG is hard," is a computability result and irrelevant here.) GitHub
Why it is the closest structural analog to the Pokémon TCG: it handles (i) a variable per-step action set (up to ~24 legal candidate actions, each a 12-dim feature vector) by scoring legal state–action pairs, with action attributes embedded and cross-attention to the referenced card; (ii) multiple sub-decisions per turn via an environment-level "multi-select state machine" that emits each choice as its own timestep; (iii) a huge card variety via LLM-generated card-text embeddings; (iv) self-play from scratch with a PPO/V-trace actor-critic, undiscounted (gamma=1.0), sparse win/loss reward, with optional Random Network Distillation for exploration and an asymmetric critic that can see both players' states during training. A strong checkpoint was trained "for over 100M games (equivalent to training on 32 4090 GPUs for 5 days)." deepwikiGitHub
Caveats: trained on a limited deck set, not the full card pool; no measured human-vs-AI results published; MCTS/league/Nash items are roadmap, not implemented. (Aggregator sites describing it as "DQN/PyTorch" are inaccurate; the codebase is JAX/Flax with PPO/V-trace.)

- G. DeepNash / R-NaD (Perolat et al., 2022, Stratego, arXiv:2206.15378) and Solly / Liar's Poker (Dewey et al., arXiv:2511.03724, 2025).
What they offer: model-free self-play to an approximate Nash equilibrium in two-player zero-sum imperfect-information games, with low exploitability — exactly the unexploitability property you might eventually want in a competitive ladder. Liar's Poker (Solly) is a post-2023 demonstration that R-NaD (via OpenSpiel) trains a strong agent from scratch with no dataset and reaches elite human level. arXiv
Why not primary: R-NaD (reward transformation + NeuRD update + v-trace adaptation) is materially more complex to implement and tune than PPO, and DeepNash itself notes model-free self-play is "empirically challenging to stabilize when scaling up to complex games." Given your explicit simplicity priority and the Rudolph et al. (2025) evidence, this is a second-line option.

## Thresholds that change the plan:
- Claim: If the simulator is too slow to generate the hundreds of millions of frames pure self-play wants, prioritize Deep Monte-Carlo (DouZero) for sample-efficient terminal-reward learning and consider light reward shaping. Or try to enable parallellization in the simulator somehow?
- Claim: If you later obtain any sizable corpus of game logs, revisit Metamon's offline-RL + transformer pipeline as a warmup — that single change would make Metamon a far closer fit.

## Techniques to Borrow, by Problem (verify)
- Enormous discrete action space:
Primary (simplest): DouZero-style action-as-feature encoding — represent each legal action as a feature vector and score (state, action) pairs, with action masking over the legal set each step (as in Big 2, ygo-agent, and the Jaipur self-play work, which used action masks for a 25,469-action space). This is simpler than learning a separate action-embedding model.
Secondary: learned action embeddings (e.g., the Axie Infinity "Towards Modern Card Games with Large-Scale Action Spaces" approach) if generalization to unseen actions becomes important.

- Multiple actions per turn:
Primary (simplest): decompose each turn into a sequence of single-action MDP timesteps, each with its own legal-action mask — exactly how the Kaggle cabt Engine presents the game and how ygo-agent's multi-select state machine works. No algorithm change needed; PPO/GAE handles the longer effective horizon.
Secondary: autoregressive / factored action heads (AlphaStar-style; Charlesworth's multi_action_head_PPO; the security-game PPO-AR work that factors π(a,n|o)=π(a|n,o)·π(n|o)) only if you must emit a structured composite action atomically. arxiv

- Sparse binary (win/loss) reward in self-play:
Undiscounted or near-undiscounted return (gamma≈1.0) with GAE; the Generals.io tricks (parameter EMA, top-advantage filtering); entropy regularization (Big 2 finding). Deep Monte-Carlo (DouZero) is a viable alternative that targets terminal returns directly. Consider light potential-based shaping (e.g., prize cards taken) only if from-scratch exploration stalls — but the Generals.io result suggests pure sparse reward can work.

- Imperfect information:
A recurrent or transformer encoder over the action/observation history to implicitly infer hidden state (Metamon's trajectory transformer; ygo-agent's history-action transformer + LSTM). A critic that can see privileged/both-player information during training while the actor stays partial-observation (ygo-agent's oppo_info critic; the asymmetric actor-critic pattern). If exploitability becomes a measured problem, add KL-to-uniform regularization (cheap) before reaching for full R-NaD.