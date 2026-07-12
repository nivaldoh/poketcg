# Creating rule-based agents

Take CSV decklist and check its validity with the competition's allowed card list in En_Card_Data.csv, in the repo's root data folder. The CSV has format Quantity,Card,Category. To create a deck, write a CSV file with a single column, containing only the card IDs that are used in the deck. If a card has multiple copies in the deck, it should appear in multiple subsequent rows repeatedly.

1. Understand the deck

We prioritize strong decks with:

a relatively linear game plan,
few complicated card interactions,
consistent setup,
manageable branching.

A slightly weaker human deck may be better if it is much easier to encode reliably. Make the most out of the decklist provided.

2. Consider the available references

simulator API documentation,
game rules and relevant card rulings,
existing rule-based agents,
the exact deck list,
a short deck strategy guide

3. Use simulator API

Make the most of existing helpers and create new ones as needed. Examples:

play_card(...)
attach_energy(...)
retreat(...)
use_ability(...)
attack(...)

This keeps simulator details separate from strategic logic and makes the code easier for us to modify safely.

4. Implement a straightforward priority-based policy

Use a clear hierarchy:

Take a guaranteed win.
Prevent an immediate loss.
Take a valuable knockout.
Prepare the next attacker.
Improve the board and hand.
Disrupt the opponent.
End the turn safely.

Add deck-specific rules inside this structure. Avoid complex planning or search initially.

5. Check the evaluation harness

Run many games against:

available agents,
older versions of your own agent

Track the available metrics

6. Improve the agent from concrete losses

For important losses:

inspect the replay,
find the first major mistake,
add a test reproducing that situation,
implement a narrow rule to fix it,
rerun the evaluation suite.

7. Consider advanced options as a last resort

belief-state modeling,
tree search or beam search,
automated heuristic optimization,
etc