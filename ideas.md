Here are just some personal notes I'm scribbling. Agents should ignore this file. Do NOT modify this.

- finished trivial training with rule-based deck + simple RL model

- write replay parser to add off-policy daily data for training
- come up with proper eval, benchmark
- improve the existing mcts model to see if it can serve as a teacher for IL
- submit model officially to see how it does

- create some kind of long term memory for opponent play styles
- build the verif, benchmarking stack first
- libero,etc as a reference. also ENPIRE, karpathy autoresearch for the repo
- procedural gen tasks, agents, decks
- find a way to come up with a training set containing key plays (like puzzles), or at least have some way to ensure that the model has seen the main strategy of all major metagame decks
- experiment with data generation
- take metagame report, ensure it matches available cards, find deck lists and make sure the model knows how to play at least one of these decks against all of the others
- download rule book and ensure the code is aligned with it
- use kaggle replays
- look for similar problems including in different domains to gain inspiration for different approaches. I think this is a two-player zero-sum imperfect-information game

probably do next: have a /goal for a critic that has access to TCG rules and checks if we're making any conceptual mistakes