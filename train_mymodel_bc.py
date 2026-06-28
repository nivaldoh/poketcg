#!/usr/bin/env python3
"""BC-pretrain the starter MyModel on the rule-based teacher (Lucario mirror).

Distills `rulebased_lucario` into MyModel's **policy AND value heads** — the value head is the payoff:
`mcts_agent` can use it as the leaf eval. Same MyModel + same HuberLoss recipe as `starter_rl_kaggle.py`;
only the data source changes (teacher imitation instead of self-play). Pure starter stack — no external
RL library.

    python train_mymodel_bc.py [--games 250] [--epochs 4] [--batch-size 128] [--deck deck.csv]
    -> out/mymodel_lucario_bc.pth  (then: TORCHDYNAMO_DISABLE=1 python eval_mymodel.py)
"""

import argparse
import os
import random
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)

import tcg_model  # noqa: E402  (sets up the vendored cg path; loads card metadata)
from tcg_model import (  # noqa: E402
    MyModel, get_encoder_input, get_decoder_input, LearnSample, LearnInput, progress,
)
from cg.api import OptionType, to_observation_class  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import rulebased_lucario  # noqa: E402


def is_policy_select(sel) -> bool:
    """A single-select policy decision (canonical predicate; excludes all-NUMBER selects)."""
    if sel.minCount != 1 or sel.maxCount != 1 or len(sel.option) <= 1:
        return False
    if all(OptionType(o.type) == OptionType.NUMBER for o in sel.option):
        return False
    return True


def record_teacher_games(deck: list[int], n_games: int) -> list[LearnSample]:
    """Drive teacher-vs-teacher Lucario-mirror games; record one sample per single-select policy
    decision (per-option decoder bags == how mcts_agent featurizes single-select nodes). Value target
    = the game's Monte-Carlo outcome from the acting seat (+1/-1/0); policy target = teacher's choice
    -> +1, others -> -1."""
    teacher = rulebased_lucario.agent
    out: list[LearnSample] = []
    for _ in progress(n_games, "Recording teacher games... "):
        obs, start = battle_start(deck, deck)
        if start.errorPlayer >= 0:
            raise ValueError(f"deck rejected: errorType={start.errorType}")
        per_seat: list[list[LearnSample]] = [[], []]
        while obs["current"]["result"] < 0:
            o = to_observation_class(obs)
            sel = o.select
            choice = teacher(obs)
            if is_policy_select(sel):
                yi = o.current.yourIndex
                n_opt = len(sel.option)
                policy = [-1.0] * n_opt
                for c in choice:
                    if 0 <= c < n_opt:
                        policy[c] = 1.0
                per_seat[yi].append(LearnSample(
                    0.0, policy, get_encoder_input(o, deck),
                    get_decoder_input(o, [[i] for i in range(n_opt)])))
            obs = battle_select(choice)
        battle_finish()
        result = obs["current"]["result"]
        for seat in range(2):
            outcome = 0.0 if result == 2 else (1.0 if seat == result else -1.0)
            for s in per_seat[seat]:
                s.value = outcome
                out.append(s)
    return out


def train(model: MyModel, samples: list[LearnSample], epochs: int, batch_size: int, device,
          policy_loss: str = "huber", value_weight: float = 1.0) -> None:
    """The starter's batching recipe (starter_rl_kaggle.py:168-216): each sample padded to 64
    candidate-bags. `policy_loss`: 'huber' (starter-native, per-option +-1 regression) or 'ce'
    (cross-entropy over the legal options vs the teacher's choice — sharper imitation, matches
    mcts_agent's softmax(*10) read). `value_weight` scales the HuberLoss value head (down-weight to
    cut interference with the policy)."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)
    model.train()
    for ep in range(epochs):
        random.shuffle(samples)
        n_batches = len(samples) // batch_size
        loss_sum = 0.0
        for i in range(n_batches):
            input_enc, input_dec = LearnInput(), LearnInput()
            mask, label_enc, label_dec, chosen = [], [], [], []
            for s in samples[i * batch_size:(i + 1) * batch_size]:
                input_enc.add(s.sv_enc)
                input_dec.add(s.sv_dec)
                label_enc.append(s.value)
                label_dec.extend(s.policy)
                chosen.append(s.policy.index(1.0) if 1.0 in s.policy else 0)   # teacher's option idx
                for _ in range(len(s.policy)):
                    mask.append(1.0)
                for _ in range(64 - len(s.policy)):           # pad to 64 candidate-bags per sample
                    mask.append(0.0)
                    label_dec.append(0.0)
                    input_dec.offset.append(len(input_dec.index))
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
            le = torch.tensor(label_enc, dtype=torch.float32, device=device).view(batch_size, -1)
            ld = torch.tensor(label_dec, dtype=torch.float32, device=device).view(batch_size, -1)
            opt.zero_grad()
            oe, od = model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device))
            loss_enc = loss_fn_enc(oe, le)
            if policy_loss == "ce":
                logits = (od * 10.0).masked_fill(mask_t == 0, -1e9)   # *10 matches mcts_agent's read
                logp = torch.log_softmax(logits, dim=1)
                ci = torch.tensor(chosen, dtype=torch.long, device=device)
                loss_dec = -logp[torch.arange(batch_size, device=device), ci].mean()
            else:
                loss_dec = (loss_fn_dec(od, ld) * mask_t).sum() / float(batch_size)
            loss = value_weight * loss_enc + loss_dec
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"  epoch {ep}: batches={n_batches} mean_loss={loss_sum / max(n_batches, 1):.4f}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--policy-loss", default="huber", choices=["huber", "ce"],
                    help="huber = starter-native per-option regression; ce = cross-entropy (sharper imitation)")
    ap.add_argument("--value-weight", type=float, default=1.0, help="scale on the value HuberLoss")
    ap.add_argument("--deck", default=os.path.join(_REPO_ROOT, "deck.csv"))
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "out", "mymodel_lucario_bc.pth"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    deck = [int(x) for x in open(args.deck).read().split()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BC-pretrain MyModel on rulebased_lucario | {args.games} games | device={device}", flush=True)

    samples = record_teacher_games(deck, args.games)
    print(f"Recorded {len(samples)} single-select policy samples.", flush=True)

    model = MyModel(128, 2, 256, 1, 1).to(device)
    print(f"  policy_loss={args.policy_loss} value_weight={args.value_weight}", flush=True)
    train(model, samples, args.epochs, args.batch_size, device,
          policy_loss=args.policy_loss, value_weight=args.value_weight)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved BC-pretrained MyModel to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
