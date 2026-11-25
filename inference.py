
import json
from pathlib import Path

import torch

from model import TransformerNMT


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"

DEVICE = torch.device("cpu")

SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}


def load_model():
    """Load trained weights and character vocabularies."""

    vocab_path = CHECKPOINT_DIR / "vocabularies.json"
    model_path = CHECKPOINT_DIR / "english_uzbek_transformer_100.pth"

    with vocab_path.open(encoding="utf-8") as file:
        vocabs = json.load(file)

    en_vocab = vocabs["en_vocab"]
    uz_vocab = vocabs["uz_vocab"]

    model = TransformerNMT(
        len(en_vocab),
        len(uz_vocab)
    ).to(DEVICE)

    weights = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    return model, en_vocab, uz_vocab


model, en_vocab, uz_vocab = load_model()

en_stoi = {
    token: index
    for index, token in enumerate(en_vocab)
}

uz_stoi = {
    token: index
    for index, token in enumerate(uz_vocab)
}


def validate_input(word):
    """Validate English word input."""

    word = word.strip().lower()

    if not word:
        raise ValueError("Please enter an English word.")

    if len(word) > 98:
        raise ValueError("Input is too long.")

    if any(char not in en_stoi for char in word):
        raise ValueError(
            "Input contains unsupported characters."
        )

    if any(char.isspace() for char in word):
        raise ValueError(
            "This model supports individual words only."
        )

    return word


@torch.inference_mode()
def predict(word, max_len=25):
    """Generate an Uzbek translation."""

    word = validate_input(word)

    if not 1 <= max_len <= 99:
        raise ValueError("Invalid decoding length.")

    src_ids = (
        [en_stoi["<BOS>"]]
        + [en_stoi[char] for char in word]
        + [en_stoi["<EOS>"]]
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    memory, src_padding_mask = model.encode(src)

    generated = [uz_stoi["<BOS>"]]

    for _ in range(max_len):
        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model.decode(
            tgt,
            memory,
            src_padding_mask
        )

        token_id = logits[0, -1].argmax().item()

        if token_id == uz_stoi["<EOS>"]:
            break

        generated.append(token_id)

    result = "".join(
        uz_vocab[index]
        for index in generated[1:]
        if uz_vocab[index] not in SPECIAL_TOKENS
    )

    return result


def translate(word, max_len=25):
    """User-facing translation function."""

    try:
        return predict(word, max_len) or "(No translation generated)"
    except ValueError as error:
        return str(error)
