game module
This module provides high-level functions to control a battle session. It wraps native C++ backend functions such as battle initialization, selection, and visualization. battle_start() returns both the first observation and the low-level StartData result.

Functions

Functions
game.battle_start(deck0, deck1)
Start a new battle with two decks.

Parameters
:
deck0 (list[int]) – The first player’s 60-card deck as a list of card IDs.

deck1 (list[int]) – The second player’s 60-card deck as a list of card IDs.

Returns
:
A tuple containing the initial observation and the low-level start result. The observation is None when battle initialization fails and StartData contains the error details.

Return type
:
tuple[dict | None, StartData]

Raises
:
ValueError – If either deck does not contain exactly 60 cards.

game.battle_finish()
Finish and clean up the current battle.

This frees native resources associated with the current battle instance.

game.battle_select(select_list)
Apply a selection from the player and retrieve the updated game state.

Parameters
:
select_list (list[int]) – List of selected option indices.

Returns
:
The updated game state after applying the selection.

Return type
:
dict

Raises
:
ValueError – If input is not a list of integers or internal state is broken.

IndexError – If the selection index is invalid.

game.visualize_data()
Retrieve the data to be used by the visualizer.

Returns
:
The data to be used by the visualizer.

Return type
:
str