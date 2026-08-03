"""Byte-level BPE tokenizer interface for Task 2."""

# %%
from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# %%


class BPETokenizer:
    """A trainable byte-level BPE tokenizer.

    The implementation will keep a byte vocabulary and an ordered list of
    merge rules.  Text encoding and decoding must be lossless for UTF-8 text.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        """Initialize an untrained tokenizer with a target vocabulary size."""
        self.vocab_id_to_token = {i: bytes([i]) for i in range(256)}
        self.vocab_token_to_id = {bytes([i]): i for i in range(256)}

        self.now_vocab_size = 256

        self.target_vocab_size = vocab_size

        self.merge_rule = []

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens currently present in the vocabulary."""
        return self.now_vocab_size

    @classmethod
    def int_list_to_bytes_list(cls, lst):
        lst = [bytes([x]) for x in lst]
        return lst

    @classmethod
    def str_to_bytes(cls, text):
        return bytes.fromhex(text)

    @classmethod
    def bytes_to_str(cls, obj):
        return obj.hex()

    def train(self, text: str) -> None:
        """Learn BPE merge rules from training text."""

        code_list = text.encode("utf-8")
        code_list = self.int_list_to_bytes_list(code_list)

        if self.target_vocab_size <= 256:
            raise OverflowError("训练个锤子")
        for i in range(self.target_vocab_size - 256):
            pair_count = Counter(itertools.pairwise(code_list))
            if not pair_count:
                break
            pair, _ = pair_count.most_common(1)[0]
            new_token = pair[0] + pair[1]
            self.vocab_token_to_id[new_token] = self.now_vocab_size
            self.vocab_id_to_token[self.now_vocab_size] = new_token
            self.now_vocab_size += 1
            self.merge_rule.append(pair)

            merged_list = []

            index = 0
            while index < len(code_list):
                if index + 1 >= len(code_list):
                    if index < len(code_list):
                        merged_list.append(code_list[index])
                    break
                pair_token = code_list[index] + code_list[index + 1]
                if pair == (code_list[index], code_list[index + 1]):
                    merged_list.append(pair_token)
                    index += 2
                else:
                    merged_list.append(code_list[index])
                    index += 1

            code_list = merged_list

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 text into BPE token IDs."""

        code_list = text.encode("utf-8")
        code_list = self.int_list_to_bytes_list(code_list)

        for rule in self.merge_rule:
            merged_list = []

            index = 0
            while index < len(code_list):
                if index + 1 >= len(code_list):
                    if index < len(code_list):
                        merged_list.append(code_list[index])
                    break
                pair_token = code_list[index] + code_list[index + 1]
                if rule == (code_list[index], code_list[index + 1]):
                    merged_list.append(pair_token)
                    index += 2
                else:
                    merged_list.append(code_list[index])
                    index += 1
            code_list = merged_list

        return [self.vocab_token_to_id[x] for x in code_list]

    def decode(self, ids: Iterable[int]) -> str:
        """Reconstruct UTF-8 text from BPE token IDs."""
        tokens = [self.vocab_id_to_token[x] for x in ids]
        return b"".join(tokens).decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Save the vocabulary and ordered merge rules as JSON."""

        properties = {}

        properties["merge_rule"] = [
            (pair[0].hex(), pair[1].hex())
            for pair in self.merge_rule
        ]

        properties["now_vocab_size"] = self.now_vocab_size
        properties["target_vocab_size"] = self.target_vocab_size

        properties["vocab_token_to_id"] = {
            BPETokenizer.bytes_to_str(token): v for token, v in self.vocab_token_to_id.items()
        }

        properties["vocab_id_to_token"] = {
            token: BPETokenizer.bytes_to_str(v) for token, v in self.vocab_id_to_token.items()
        }

        with open(path, encoding="utf-8", mode="w+") as file:
            json.dump(properties, file)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> BPETokenizer:
        """Load a previously saved tokenizer without retraining it."""

        with open(path, encoding="utf-8") as file:
            data = json.load(file)

        bpe = BPETokenizer()
        bpe.merge_rule = [
            (
                BPETokenizer.str_to_bytes(pair[0]),
                BPETokenizer.str_to_bytes(pair[1]),
            )
            for pair in data["merge_rule"]
        ]
        bpe.now_vocab_size = data["now_vocab_size"]
        bpe.target_vocab_size = data["target_vocab_size"]
        bpe.vocab_token_to_id = {
            BPETokenizer.str_to_bytes(k): int(v) for k, v in data["vocab_token_to_id"].items()
        }
        bpe.vocab_id_to_token = {
            int(k): BPETokenizer.str_to_bytes(v) for k, v in data["vocab_id_to_token"].items()
        }

        return bpe


# %%

# %%
