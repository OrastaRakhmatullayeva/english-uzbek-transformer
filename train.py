
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import TransformerNMT


SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "english_uzbek_100.csv"

OUTPUT_DIR = ROOT / "checkpoints"

SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>"
]


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    return SPECIAL_TOKENS + chars


def encode(text, stoi):
    return torch.tensor(
        [stoi["<BOS>"]]
        + [stoi.get(char, stoi["<UNK>"]) for char in text]
        + [stoi["<EOS>"]],
        dtype=torch.long
    )


def collate_fn(batch):
    src, tgt = zip(*batch)

    src = pad_sequence(
        src,
        batch_first=True,
        padding_value=0
    )

    tgt = pad_sequence(
        tgt,
        batch_first=True,
        padding_value=0
    )

    return src, tgt


def main(epochs):
    random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(DATA_PATH)

    if df.shape[1] != 2:
        raise ValueError(
            "Expected two columns: English and Uzbek."
        )

    df.columns = ["english", "uzbek"]

    df = df.dropna().drop_duplicates()

    for column in ["english", "uzbek"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    df = df[
        df["english"].str.len().between(1, 98)
        & df["uzbek"].str.len().between(1, 98)
    ].drop_duplicates()

    en_vocab = build_vocab(df["english"])
    uz_vocab = build_vocab(df["uzbek"])

    en_stoi = {
        token: index
        for index, token in enumerate(en_vocab)
    }

    uz_stoi = {
        token: index
        for index, token in enumerate(uz_vocab)
    }

    dataset = [
        (
            encode(en, en_stoi),
            encode(uz, uz_stoi)
        )
        for en, uz in zip(
            df["english"],
            df["uzbek"]
        )
    ]

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = TransformerNMT(
        len(en_vocab),
        len(uz_vocab)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_tokens = 0

        for src, tgt in loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            decoder_input = tgt[:, :-1]
            expected_output = tgt[:, 1:]

            optimizer.zero_grad()

            logits = model(src, decoder_input)

            loss = criterion(
                logits.reshape(-1, len(uz_vocab)),
                expected_output.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            tokens = (expected_output != 0).sum().item()

            total_loss += loss.item() * tokens
            total_tokens += tokens

        epoch_loss = total_loss / total_tokens

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss
        })

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {epoch_loss:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "english_uzbek_transformer_100.pth"
    )

    with open(
        OUTPUT_DIR / "vocabularies.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "en_vocab": en_vocab,
                "uz_vocab": uz_vocab
            },
            file,
            ensure_ascii=False,
            indent=2
        )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print("Training completed.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=200
    )

    args = parser.parse_args()
    main(args.epochs)
