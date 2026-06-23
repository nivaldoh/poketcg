# General instructions
- Start with minimal code that can be verified, always aim for simplicity, avoid verbosity, avoid boilerplate and things like backwards-compatibility. We want to keep our code clean and readable. Code/repo changes should ideally be minimal, incremental/decremental and verifiable.
- Be careful with assumptions, and always explicitly state any that you think you need to make. Point out any information that you think you might be missing, and I will help you find it and make key decisions.
- We should aim to structure the repo such that it makes it easier for you (Claude Code) to work with it.

# First goal
- Set up the repo to work locally (the provided starter code was meant to run in a Kaggle Notebook. Our environment is Ubuntu WSL 2 through Windows 11). Run an initial test with the provided starter code to ensure we have a functional starting point before we start making modifications. We can take the starter RL+MCTS code and get it to run and do some training.

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
- https://matsuoinstitute.github.io/cabt/

## Pokémon TCG rules and specifics
- These will probably be important for some things. I haven't provided you an actual rule book yet, so for now just be careful with assumptions about the game.

# Skills
- We need to think about what repeatable processes exist and what domain-specific things could use specific Skills. We don't need to create them yet, but we think about Skills that might be useful to create later on.

# Benchmark, verifiable results, evals, tests
- We need to make our work have verifiable results as much as possible, to ensure we're actually making progress as we iterate. Some things will be hard to verify, so we'll have to start small and gradually think more about this as we go. For instance, checking whether a battle resulted in a win or loss should be trivial, but benchmarking intermediate factors to see what led to that result will be more challenging.

# Training
- There seems to be two ways to gather training data: self-play and retrieving ladder data from time to time. I don't know more details about how either of these would work, so we'll have to gather more info as we go.