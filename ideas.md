Here are just some personal notes I'm scribbling. Agents should ignore this file. Do NOT modify this.

- write replay parser to add off-policy daily data for training
- come up with proper eval, benchmark
- improve the existing mcts model to see if it can serve as a teacher for IL? It seems to diverge. It's probably bad
- submit model officially to see how it does

- create some kind of long term memory for opponent play styles
- build the verif, benchmarking stack first
- libero,etc as a reference. also ENPIRE, karpathy autoresearch for the repo
- procedural gen tasks, agents, decks
- find a way to come up with a training set containing key plays (like puzzles), or at least have some way to ensure that the model has seen the main strategy of all major metagame decks
- experiment with data generation
- take metagame report, ensure it matches available cards, find deck lists and make sure the model knows how to play at least one of these decks against all of the others
- use kaggle replays for IL
- look for similar problems including in different domains to gain inspiration for different approaches. I think this is a two-player zero-sum imperfect-information game
- use an llm to build a curriculum for rl? might need to build a trajectory parser for this, then we get a verifiable thing to fine-tune an llm with. Would this be an RL env? the LLM gets an easier job since it's not meant to win directly. It's can build easy replays to help warm up the RL
- what if I tried to train the model with a backwards curriculum, like, take full trajectories, remove the last step from them, use them to teach the model how to win. Then once it learns that, remove the second-last step.
- turn it into a multi-agent problem
- use language model policy to play
- create visualization to see inputs (obs, etc) and metrics while showing what's going on in a match
- try to have a model that is able to play both TCG and Showdown (more adaptable meta-RL?) in a less data-hungry way than Metamon?       
    - wouldn't need a faster env (like TCG Jax) for this since we want the model to learn with LESS data, NOT more.
    - What is it the makes Metamon/AMAGO data-hungry anyway? Is this the right question to ask? Maybe it's a decent starting point. 
    - Maybe a better question is: how does Metamon/AMAGO learn anyway?
    - Should I focus on using LESS data, or on enabling the model to scale better + generate MORE and BETTER data relatively to Metamon?
        - The kaggle competition has a limit for number of MB uploaded in submissions, so the model can't be too large for that
        - If I were to use Metamon, I'd need AMAGO, which in turn needs lots of off-policy data. Maybe I just need to generate that? Metamon used competitive data from high-skill players though, so how would I get that? This game might also be even more data-intensive? (this I still need to check and compare)
        - Would it be enough to adapt Metamon to the TCG env and simply throw it enough training data? Or does Metamon make some kind of assumption that wouldn't work here?
    - [TODO: try to take ideas from metamon codebase + paper. need to keep in mind the differences between Showdown and TCG. Can look into that after this?]
    - to adapt metamon I'd probably have to start adapting AMAGO. So I should start by reading its paper and checking if we break any assumptions there? Other than requiring lots of data for off-policy training
    - i should read the paper from amy zhang and eugene vinitsky that revisits PPO for imperfect-information games, see if that applies here and if so, take ideas and code from there. Can eventually try to reach out to one of the authors to ask questions.
- should probably start training by overfitting on a single deck against a single other deck and opponent, and then slowly expand from there checking what limitations appear. I have the custom visualizer and it will probably be useful here
    - should I try to start with cold-start self-play and then benchmark it as I go and solve saturation points as they appear?
- i should at some point count the total amount of variables involved in this game and compare it to showdown
- need to improve the repo for agents to work with. better skills, organize files, facilitate subagent exploration, etc
    - create skills like how to do self-play properly, etc?
    - need to create hook to help claude code ramp up
        - context: tcg kaggle competition
        - plan: self-play cold start
        - repo structure: ...
        - ...
- so there are two options to start with (?): 1.cold-start self-play RL, 2.having lots of data for IL warm-start
    - being able to do similarly to metamon would be great, but then I'd need lots of competent gameplay data (including expert-level) for the warm-start IL, otherwise it probably wouldn't work well?
        - if I were to gather data, what options do I have?
        - set claude code on loop to auto-generate code for more rulebased agents? it probably needs to keep verifying correctness turn by turn by interacting directly with the sim? Need to give it one deck at a time. Verify that they generate valid agents that would serve as a decent baseline
        - should I gather data from web (e.g. competitive TCG video subtitles) to feed to a coding agent to have it generate rule-based agents that would serve as a decent source of data for IL warm start?
            - so first I'd have to gather as much info about competitive TCG discussions as possible?
            - then feed it to agent
            - have an LLM pick an easy deck for a rule-based agent to pilot?
            - I can start by seeing how a coding agent does with limited info + starter template decks as examples. Might need to give it simulator API docs. Write Skills for it
            - Should I train an LLM with RL using the simulator as a verifier in order to have it generate good rule-based agent code?
            - Then I can see how it performs
- should I try to start by benchmarking a lot of things that are game-related without being tied to a specific algorithm, like how often the agent wins by deck-out, so that I can directly measure and address specific flaws as they come up?

- is synthetic data bad for self-play? could i generate a good curriculum with it?


what is my focus here? What is the question to ask? Am I testing RL? Am I trying to win? Am I creating a benchmark? Should I try to generate good data to give the model a curriculum?


________

What do we need to win? An agent that combines leaderboard performance + Hackathon evaluation (innovation).

Major areas:
- Data
- Agent
- Evals

Agent:
- Doesn't have to be necessarily trained
    - Rule-based / heuristic
    - Pre-trained LLMs, Pokéchamp-style
- Trained agents:
    - Full-supervised
    - IL warm-start + RL (Metamon)
    - RL with cold start?
    - Fine-tuned LLM
    - etc etc

Trained agents will need data.
- Sources:
    - Kaggle replays (likely low-quantity and low quality?)
    - Rule-based / heuristic bots
    - Self-play
    - Web (extreme option. Probably we won't have time for this)
    - Manual... (at least to have a few very high-quality examples?)
    - Artificially generated (e.g. can we get an LLM to generate trajectories?)

Weaker agents should be useful to generate data for stronger agents. Generating data (high-quality curriculum?) is probably our priority. We need to collect increasingly higher-quality checkpoints and trajectories.

Our long-term goal is to have an ideal agent that:
- Knows how to play a minimal amount of decks that it can rotate to do well in the competition's ladder (there's probably no single deck that will clear the field)
- Knows how to generalize/adapt as well as possible against any opponent style and deck
- Doesn't get exploited
- Doesn't leave us bankrupt
- Runs within the time limit