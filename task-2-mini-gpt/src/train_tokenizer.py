"""One-time training pipeline for the byte-level BPE tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tokenizer import BPETokenizer


@dataclass(frozen=True)
class TokenizerTrainConfig:
    """Inputs and output location for one tokenizer training run."""

    corpus_path: Path = Path("data/train.txt")
    output_path: Path = Path("ckpt/tokenizer.json")
    vocab_size: int = 400


def load_training_text(path: str | Path) -> str:
    """Read and validate the UTF-8 corpus used to learn BPE merge rules."""

    text = Path(path).read_text(encoding="utf-8")

    if not text:
        raise ValueError("Training corpus is empty")

    return text


def train_tokenizer(
    text: str,
    vocab_size: int,
) -> BPETokenizer:
    """Construct a fresh tokenizer and learn merge rules from training text."""
    bpe = BPETokenizer(vocab_size)

    bpe.train(text)

    return bpe


def validate_tokenizer(
    tokenizer: BPETokenizer,
    samples: list[str],
) -> None:
    """Raise an error when encode/decode cannot reproduce any sample exactly."""
    for sample in samples:
        ids = tokenizer.encode(sample)
        temp_sample = tokenizer.decode(ids)

        if temp_sample != sample:
            raise ValueError("Wrong sample")


def save_tokenizer(
    tokenizer: BPETokenizer,
    path: str | Path,
) -> None:
    """Create the artifact directory and persist the trained tokenizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)



def train_and_save_tokenizer(
    config: TokenizerTrainConfig,
) -> BPETokenizer:
    """Coordinate corpus loading, BPE training, validation, and persistence."""


    all_text = load_training_text(config.corpus_path)

    all_text_split = all_text.split("\n")

    train_text = all_text_split[len(all_text_split)//10:]
    valid_text = all_text_split[:len(all_text_split) // 10]

    bpe = train_tokenizer(all_text, config.vocab_size)
    validate_tokenizer(bpe, valid_text)

    save_tokenizer(bpe, config.output_path)

    return bpe


def main() -> None:
    """Run one tokenizer training job with the configured corpus and vocabulary."""
    bpe = train_and_save_tokenizer(TokenizerTrainConfig())


if __name__ == "__main__":
    main()
