# This is sample code for Reinforcement Learning and MCTS using Transformer like models.
# It also serves as an example for the Search API and the Battle Start/Finish APIs. Pokémon TCG cards have a wide variety of effects. It can be difficult even to determine things such as how much damage an attack would deal to the opponent's Pokémon right now, or whether an Item card drawn from the deck would be playable.
# By using the Search API, you can find out, for example, what the actual outcome would be if an attack were used. This should be useful for rule-based agents as well.


import os
import random
import sys

import torch
import torch.nn
import torch.optim

# Local run: vendored `cg` simulator under vendor/cg-lib/ (see README/CLAUDE.md);
# override with CG_LIB_PATH. Self-contained, mirroring run_battle.py.
_CG_LIB_PATH = os.environ.get(
    "CG_LIB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "cg-lib"),
)
sys.path.insert(0, _CG_LIB_PATH)
from cg.game import battle_start, battle_finish, battle_select

# Reusable model/agent layer, extracted so eval.py and other tools can import the
# model and agents without triggering this trainer's module-level training run.
import tcg_model
from tcg_model import (
    MyModel,
    mcts_agent,
    random_agent,
    sample_deck,
    LearnSample,
    LearnInput,
    progress,
)

# The rule-based Mega-Lucario agent: a second training/eval opponent.
from rulebased_lucario import agent as rulebased_agent, my_deck as lucario_deck

# Loop sizes are CLI-tunable so the training loop can run a tiny local smoke
# pass (e.g. --iterations 1 --eval-games 2 --selfplay-games 2 --search-count 2)
# while keeping the Kaggle defaults when run with no arguments.
import argparse
_p = argparse.ArgumentParser(description="RL+MCTS self-play training")
_p.add_argument("--iterations", type=int, default=5)
_p.add_argument("--eval-games", type=int, default=50)
_p.add_argument("--selfplay-games", type=int, default=100)
_p.add_argument("--search-count", type=int, default=tcg_model.SEARCH_COUNT)
_p.add_argument("--batch-size", type=int, default=128)
_p.add_argument("--load", type=str, default=None, help="checkpoint to continue training from")
_p.add_argument("--out-prefix", type=str, default="model", help="checkpoint filename prefix under out/")
_p.add_argument("--rb-frac", type=float, default=0.5, help="fraction of self-play games played vs the rule-based agent")
_args = _p.parse_args()
tcg_model.SEARCH_COUNT = _args.search_count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(128, 2, 256, 1, 1)
model = model.to(device)
if _args.load:
    model.load_state_dict(torch.load(_args.load, map_location=device))
    print(f"Loaded initial weights from {_args.load}", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
os.makedirs("out", exist_ok=True)

def eval_winrate(n_games: int, opponent: str) -> tuple[int, list[int]]:
    """Win rate (%) of the MCTS agent (sample_deck) vs an opponent over n_games.

    opponent: "random" (mirror sample_deck) or "rulebased" (Lucario deck).
    Returns (win_rate, [wins, losses, draws]).
    """
    results = [0, 0, 0]
    for i in progress(n_games, f"Evaluating vs {opponent}... "):
        your_index = i % 2
        if opponent == "rulebased":
            decks = [sample_deck, sample_deck]
            decks[1 - your_index] = lucario_deck  # opponent plays Lucario
            obs, start_data = battle_start(decks[0], decks[1])
        else:
            obs, start_data = battle_start(sample_deck, sample_deck)
        if start_data.errorPlayer >= 0:
            raise ValueError(f"Deck rejected: errorType={start_data.errorType}")
        while obs["current"]["result"] < 0:
            if obs["current"]["yourIndex"] == your_index:
                selected, _ = mcts_agent(obs, sample_deck, model)
            elif opponent == "rulebased":
                selected = rulebased_agent(obs)
            else:
                selected = random_agent(obs)
            obs = battle_select(selected)
        battle_finish()
        r = obs["current"]["result"]
        if r == 2:
            results[2] += 1
        elif r == your_index:
            results[0] += 1
        else:
            results[1] += 1
    total = results[0] + results[1]
    return (100 * results[0] // total if total else 0, results)

# The main training loop.
for counter in range(_args.iterations):
    torch.save(model.state_dict(), f"out/{_args.out_prefix}{counter}.pth")  # Save the current model.
    sample_list:list[LearnSample] = []  # List of training data samples.
    
    model.eval()
    with torch.inference_mode():
        # Evaluation against a random opponent (baseline) and the rule-based agent
        # (the meaningful progress signal).
        rate_rng, _ = eval_winrate(_args.eval_games, "random")
        print(f"Evaluation win rate vs random: {rate_rng}%", flush=True)
        rate_rb, res_rb = eval_winrate(_args.eval_games, "rulebased")
        print(f"Evaluation win rate vs rule-based: {rate_rb}% (W/L/D={res_rb})", flush=True)

        # Self Play — alternate the opponent between the RL agent itself
        # (self-play; learn from both sides) and the rule-based Lucario agent
        # (learn from the RL learner's side only). --rb-frac controls the mix.
        for g in progress(_args.selfplay_games, "Training Data Collecting... "):
            vs_rb = int((g + 1) * _args.rb_frac) > int(g * _args.rb_frac)
            learner = g % 2  # alternate which player index the RL learner is
            if vs_rb:
                decks = [sample_deck, sample_deck]
                decks[1 - learner] = lucario_deck  # opponent plays Lucario
                obs, _ = battle_start(decks[0], decks[1])
            else:
                obs, _ = battle_start(sample_deck, sample_deck)
            samples:list[list[LearnSample]] = [[], []]  # [Player0 samples, Player1 samples]
            while True:
                if obs["current"]["result"] >= 0:
                    break

                yi = obs["current"]["yourIndex"]
                if vs_rb and yi != learner:
                    selected = rulebased_agent(obs)  # opponent move; no sample collected
                else:
                    # The MCTS agent generates an action and a training sample.
                    selected, sample = mcts_agent(obs, sample_deck, model)
                    samples[yi].append(sample)
                obs = battle_select(selected)

            battle_finish()  # Finalize the game.

            # Calculate the training labels and add them to the training data list.
            # In rule-based games only the learner side has samples; the other
            # side's empty list makes the loop below a no-op for that player.
            for i in range(2):
                LAMBDA = 0.9
                # The final value is 1.0 for a win and -1.0 for a loss.
                value = 1.0 if i == obs["current"]["result"] else -1.0

                # Iterate backwards from the end of the game to calculate values.
                for sample in reversed(samples[i]):
                    label = (value + sample.value) * 0.5
                    value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                    sample.value = label
                    sample_list.append(sample)

    # Train on the training data collected through self-play.
    print("Training Start.")
    model.train()
    random.shuffle(sample_list)
    BATCH_SIZE = _args.batch_size
    batch_count = len(sample_list) // BATCH_SIZE
    loss_sum = 0.0
    for i in range(batch_count):
        # Prepare a batch of data.
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask = []
        label_enc = []
        label_dec = []
        start = BATCH_SIZE * i
        for j in range(start, start + BATCH_SIZE):
            sample = sample_list[j]
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            for _ in range(len(sample.policy)):
                mask.append(1.0)
            for _ in range(64 - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        # Convert data to PyTorch tensors.
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        mask_tensor = mask_tensor.view(BATCH_SIZE, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device)
        label_tensor_enc = label_tensor_enc.view(BATCH_SIZE, -1)
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device)
        label_tensor_dec = label_tensor_dec.view(BATCH_SIZE, -1)

        optimizer.zero_grad()

        # Get model predictions for the batch.
        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device))
        
        # Calculate loss.
        loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
        loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
        loss_dec = loss_dec * mask_tensor
        loss_dec = loss_dec.sum() / float(BATCH_SIZE)
        loss = loss_enc + loss_dec

        # Backpropagate the loss and update model parameters.
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    mean_loss = loss_sum / batch_count if batch_count else float("nan")
    print(
        f"Training Finish. samples={len(sample_list)} batches={batch_count} "
        f"mean_loss={mean_loss:.4f}",
        flush=True,
    )

# Save the fully-trained model. The per-iteration "out/<prefix>N.pth" checkpoints
# above are written *before* each iteration's training, so this final save captures
# the last iteration's updates (which would otherwise be lost).
final_path = f"out/{_args.out_prefix}_final.pth"
torch.save(model.state_dict(), final_path)
print(f"Saved final trained model to {final_path}", flush=True)