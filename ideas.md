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
- use kaggle replays
- look for similar problems including in different domains to gain inspiration for different approaches. I think this is a two-player zero-sum imperfect-information game

from time to time have a /goal for a critic that has access to TCG rules and checks if we're making any conceptual mistakes


I need to externalize the structure and execution flow of the code to understand what's going on so that I don't have to keep track of everything in my head. I'm trying to create an initial mental model. I'd like you to help me write this overview in execution_flow.md. First of all I need to understand things at a high-level. What are the main ideas happening, what are the files being used, what is the execution flow of the data. We need to keep the language simple to make it easier for me to absorb. Probably it will be ideal to always show what happens in the execution flow from one step to the next. I mean, if there's too much info for me to hold in my head all at once I would struggle, but maybe if I only have to hold info in these tiny sequential two-by-two pieces it will be easier? Keep it short and focus on the high-level ideas. We can start expanding it once I got those down. I need to see the execution in a concrete manner to actually understand it. I know that logs tend to help me with this as they show what is happening and in what order, but I'm trying to avoid polluting the code, at least for now. However, seeing a concrete example of inputs and outputs, with their actual values, would certainly help. We have something already written in execution_flow.md, but it doesn't cover the entire codebase. We need to refine it to become more comprehensive in that sense, but without forcing me to overextend my memory. Also, probably having a single execution flow is not ideal. The code has multiple execution flows at different levels, so it's probably best if we write them separately. We should have a main execution flow from the perspective of the main high-level ideas (dumb example: "preprocessing -> training -> eval" would be one level of execution flow, then "inputs -> battle -> outputs" would be another, then another one could "zoom in" in the NashPG algorithm, and so on). We can think of these different execution flows as collapsible boxes that we can show or hide as needed, making it easier to understand things from top-down. It's totally fine to have some repeated info accross these execution flows (e.g. showing the inputs in multiple places). Don't worry about explaining the theory. You also don't need to worry about explaining things like how the rule-based agents work (if needed I could just request a separate "execution flow" for that). Help me write this so I can finally start grasping the code!