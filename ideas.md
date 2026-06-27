Here are just some personal notes I'm scribbling. For now you can ignore this.

- finished trivial training with rule-based deck + simple RL model

- write replay parser to add off-policy daily data for training
- come up with proper eval, benchmark
- improve the existing mcts model to see if it can serve as a teacher for IL
- submit model officially to see how it does
- port metamon to tcg

- create some kind of long term memory for opponent play styles
- build the verif, benchmarking stack first
- libero,etc as a reference. also ENPIRE, karpathy autoresearch for the repo
- procedural gen tasks, agents, decks
- find a way to come up with a training set containing key plays (like puzzles), or at least have some way to ensure that the model has seen the main strategy of all major metagame decks
- experiment with data generation
- take metagame report, ensure it matches available cards, find deck lists and make sure the model knows how to play at least one of these decks against all of the others
- download rule book and ensure the code is aligned with it
- use kaggle replays

what I ran as /goal:
Write a detailed report explaining Metamon's high-level structure and execution flow, and how we might adapt it for our Pokémon TCG problem with minimal changes. Check the metamon repo at the root of our repo, and then its metamon folder, containing its core logic. You don't need to check other directories in metamon, so ignore server, tools, etc. You will also need to check the AMAGO repo that Metamon uses. My goal is to adapt Metamon to receive inputs from the Pokémon TCG simulator instead of its Pokémon Showdown server, and then train a model that outputs an action to play Pokémon TCG through the simulator provided, instead of outputting an action for Pokémon Showdown. I want to avoid overcomplicating things so I'm thinking that our adaptation (let's call it tcgmon) should have its code as part of our root repo, so we'd use the metamon repo as a reference but would write our adaptation outside of that repo, cutting anything that's not relevant to our problem and changing things as needed. Ideally I figure we want to make minimal changes in order to minimize the risk of having a broken model. Metamon is already quite complex as it is, so we don't want to go around making arbitrary changes to it. I want to use the same architecture (timestep encoder -> trajectory encoder -> actor critic) and overall approach (tokenization, imitation learning to "warm-up" the model before starting with RL, etc). For now we just need to plan our approach, so don't worry about implementing anything yet. Be very careful with assumptions: Metamon is a research project, I haven't given you access to its paper and many of its components are highly sensitive to change. Make sure to highlight what exactly needs to change (and why), covering all relevant aspects of Metamon (data, input processing, training, inference, output, ...) and raising any potential risks. We can partially treat this as a code migration problem, where we're essentially porting Metamon to work on a different domain. Make sure the report is exhaustive and accurate.

probably do next: have a /goal for a critic that has access to TCG rules and checks if we're making any conceptual mistakes