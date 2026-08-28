"""Dataset and collation interfaces for Task 3."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .chat import ChatMessage, ConversationEncoding, encode_conversation

SFT_SYSTEM_PROMPT = "You are a helpful, honest and harmless assistant."


@dataclass(frozen=True)
class PreferenceRecord:
    """One DPO prompt with a preferred and rejected response."""

    prompt: list[ChatMessage]
    chosen: str
    rejected: str


@dataclass(frozen=True)
class PreferenceEncoding:
    """Tokenized chosen/rejected sequences and response-only labels."""

    chosen_input_ids: Tensor
    chosen_attention_mask: Tensor
    chosen_labels: Tensor
    rejected_input_ids: Tensor
    rejected_attention_mask: Tensor
    rejected_labels: Tensor


@dataclass(frozen=True)
class ToolTurn:
    """One MOSS plugin turn with its complete tool-use trajectory."""

    human: str
    inner_thoughts: str
    commands: str
    tool_responses: str
    assistant: str


@dataclass(frozen=True)
class PluginRecord:
    """One MOSS with-tools conversation used by bonus goal S5."""

    conversation_id: str
    meta_instruction: str
    turns: tuple[ToolTurn, ...]


def load_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load at most ``max_samples`` JSON objects from a JSONL file."""

    result = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.isspace():
                continue
            if max_samples is not None and len(result) >= max_samples:
                return result
            result.append(json.loads(line))

    return result


def parse_moss_messages(record: Mapping[str, Any]) -> list[ChatMessage]:
    """Normalize one MOSS SFT record into role/content messages."""

    ordered_record = record["chat"]

    result = []

    result.append(ChatMessage(role="system", content=SFT_SYSTEM_PROMPT))

    for chat in ordered_record:
        user_content: str = (
            ordered_record[chat]["Human"]
            .removeprefix("<|Human|>:")
            .removesuffix("<eoh>\n")
            .strip()
        )
        moss_content: str = (
            ordered_record[chat]["MOSS"]
            .removeprefix("<|MOSS|>:")
            .removesuffix("<eom>\n")
            .strip()
        )

        result.append(ChatMessage(role="user", content=user_content))
        result.append(ChatMessage(role="assistant", content=moss_content))

    return result


def parse_preference_record(record: Mapping[str, Any]) -> PreferenceRecord:
    """Normalize one chosen/rejected dataset record."""

    role_dict = {"human": "user", "system": "system", "gpt": "assistant"}

    conversations = record["conversations"]

    prompts = []

    for chat in conversations:
        if chat["from"] not in role_dict:
            raise ValueError("no such role")

        prompts.append(ChatMessage(role=role_dict[chat["from"]], content=chat["value"]))

    if not prompts:
        raise ValueError("prompts cant be empty")
    if record["rejected"]["from"] != "gpt" or record["chosen"]["from"] != "gpt":
        raise ValueError("answers not from gpt")
    if len(prompts) > 0 and prompts[-1]["role"] != "user":
        raise ValueError("last talk must be user")
    if len(prompts) > 0 and prompts[0]["role"] != "system":
        prompts = [ChatMessage(role="system", content=SFT_SYSTEM_PROMPT)] + prompts

    return PreferenceRecord(
        prompt=prompts,
        chosen=record["chosen"]["value"].strip(),
        rejected=record["rejected"]["value"].strip(),
    )


def parse_moss_plugin_record(record: Mapping[str, Any]) -> PluginRecord:
    """Normalize one MOSS with-tools record without dropping tool traces."""

    build_turns = [{key.strip(): value} for key, value in record["chat"].items()]

    if len(build_turns) != record["num_turns"]:
        raise ValueError("wrong num turns")

    build_turns.sort(key=lambda x: int(next(iter(x.keys())).split("_")[-1]))

    build_turns = [next(iter(turn.items()))[1] for turn in build_turns]

    result = []

    for turn in build_turns:
        roles = ["Human", "Inner Thoughts", "Commands", "Tool Responses", "MOSS"]
        for role in roles:
            if role not in turn:
                raise ValueError("Missing Roles")
        result.append(
            ToolTurn(
                turn["Human"].strip(),
                turn["Inner Thoughts"].strip(),
                turn["Commands"].strip(),
                turn["Tool Responses"].strip(),
                turn["MOSS"].strip(),
            )
        )

    return PluginRecord(
        str(record["conversation_id"]),
        record["meta_instruction"].strip(),
        tuple(result),
    )


def plugin_record_to_messages(record: PluginRecord) -> list[ChatMessage]:
    """Convert a plugin trajectory into messages suitable for SFT masking."""

    result = []

    result.append(ChatMessage(role="system", content=record.meta_instruction))

    for turn in record.turns:
        result.append(ChatMessage(role="user", content=turn.human))
        result.append(
            ChatMessage(
                role="assistant", content=turn.inner_thoughts + "\n" + turn.commands
            )
        )
        # result.append(ChatMessage(role="user", content=turn.commands))
        result.append(ChatMessage(role="user", content=turn.tool_responses))
        result.append(ChatMessage(role="assistant", content=turn.assistant))

    return result


class SFTDataset(Dataset[ConversationEncoding]):
    """Lazily encode MOSS conversations for assistant-only SFT."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        """Return the number of SFT conversations."""
        return len(self.records)

    def __getitem__(self, index: int) -> ConversationEncoding:
        """Return one tokenized and masked conversation."""
        if index < 0 or index >= len(self.records):
            raise RuntimeError("out of bound")
        return encode_conversation(
            self.tokenizer,
            parse_moss_messages(self.records[index]),
            self.max_length,
            self.ignore_index,
        )


class PluginSFTDataset(Dataset[ConversationEncoding]):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
        ignore_index: int = -100,
    ) -> None:
        self.records = records

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ConversationEncoding:
        if index < 0 or index >= len(self.records):
            raise IndexError("dataset index out of range")

        record = self.records[index]

        pluginrecord:PluginRecord = parse_moss_plugin_record(record)

        msg = plugin_record_to_messages(pluginrecord)



        return encode_conversation(
            self.tokenizer,
            msg,
            self.max_length,
            self.ignore_index
        )




class PreferenceDataset(Dataset[PreferenceEncoding]):
    """Lazily encode chosen/rejected pairs for DPO."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.records = records

    def __len__(self) -> int:
        """Return the number of preference pairs."""
        return len(self.records)

    def __getitem__(self, index: int) -> PreferenceEncoding:
        """Return one tokenized chosen/rejected pair."""
        record = self.records[index]

        record: PreferenceRecord = parse_preference_record(record)

        chose_prompts = record.prompt + [
            ChatMessage(role="assistant", content=record.chosen)
        ]
        rejected_prompts = record.prompt + [
            ChatMessage(role="assistant", content=record.rejected)
        ]

        from .chat import build_labels, format_messages

        formated_base = format_messages(record.prompt)
        formated_chosen = format_messages(chose_prompts)
        formated_rejected = format_messages(rejected_prompts)

        tokenized_base = self.tokenizer(formated_base, return_tensors="pt")
        tokenized_chosen = self.tokenizer(formated_chosen, return_tensors="pt")
        tokenized_rejected = self.tokenizer(formated_rejected, return_tensors="pt")

        chosen_label_input = tokenized_chosen.input_ids[0].long().clone()
        chosen_label_input[: tokenized_base.input_ids.shape[-1]] = self.ignore_index
        rejected_label_input = tokenized_rejected.input_ids[0].long().clone()
        rejected_label_input[: tokenized_base.input_ids.shape[-1]] = self.ignore_index

        chosen_label = build_labels(
            chosen_label_input, None, self.tokenizer, self.ignore_index
        )
        rejected_label = build_labels(
            rejected_label_input, None, self.tokenizer, self.ignore_index
        )

        # tokenized_message = tokenizer(formated_message, return_tensors="pt")
        # input_ids = tokenized_message.input_ids.long()

        base_length = tokenized_base.input_ids.shape[-1]

        prompt_keep = min(base_length, self.max_length // 2)

        return PreferenceEncoding(
            tokenized_chosen.input_ids[0].long()[
                base_length - prompt_keep : base_length + self.max_length - prompt_keep
            ],
            tokenized_chosen.attention_mask[0][
                base_length - prompt_keep : base_length + self.max_length - prompt_keep
            ],
            chosen_label[
                base_length - prompt_keep : base_length + self.max_length - prompt_keep
            ],
            tokenized_rejected.input_ids[0].long()[
                base_length - prompt_keep : base_length + self.max_length - prompt_keep
            ],
            tokenized_rejected.attention_mask[0][
                base_length - prompt_keep : base_length + self.max_length - prompt_keep
            ],
            rejected_label[
                base_length - prompt_keep : base_length + self.max_length - prompt_keep
            ],
        )


def collate_sft_batch(
    examples: Sequence[ConversationEncoding],
    pad_token_id: int,
    ignore_index: int = -100,
) -> dict[str, Tensor]:
    """Pad SFT examples into ``input_ids``, ``attention_mask``, and labels."""
    max_len = 0
    result = {}
    for example in examples:
        max_len = max(max_len, example.input_ids.numel())

    for example in examples:
        input_ids = torch.nn.functional.pad(
            example.input_ids,
            (0, max_len - example.input_ids.numel()),
            value=pad_token_id,
        ).unsqueeze(0)
        attention_mask = torch.nn.functional.pad(
            example.attention_mask,
            (0, max_len - example.input_ids.numel()),
            value=False,
        ).unsqueeze(0)
        labels = torch.nn.functional.pad(
            example.labels, (0, max_len - example.input_ids.numel()), value=ignore_index
        ).unsqueeze(0)

        result["input_ids"] = (
            input_ids
            if "input_ids" not in result
            else torch.cat([result["input_ids"], input_ids], dim=0)
        )
        result["attention_mask"] = (
            attention_mask
            if "attention_mask" not in result
            else torch.cat([result["attention_mask"], attention_mask], dim=0)
        )
        result["labels"] = (
            labels
            if "labels" not in result
            else torch.cat([result["labels"], labels], dim=0)
        )
    return result


def collate_preference_batch(
    examples: Sequence[PreferenceEncoding],
    pad_token_id: int,
    ignore_index: int = -100,
) -> dict[str, Tensor]:
    """Pad chosen and rejected encodings into one DPO batch."""
    max_len = 0
    result = {}

    for example in examples:
        max_len = max(max_len, example.chosen_labels.shape[-1])
        max_len = max(max_len, example.rejected_labels.shape[-1])

    for example in examples:
        chosen_labels = torch.nn.functional.pad(
            example.chosen_labels,
            (0, max_len - example.chosen_labels.numel()),
            value=ignore_index,
        ).unsqueeze(0)
        chosen_input_ids = torch.nn.functional.pad(
            example.chosen_input_ids,
            (0, max_len - example.chosen_input_ids.numel()),
            value=pad_token_id,
        ).unsqueeze(0)
        chosen_attention_mask = torch.nn.functional.pad(
            example.chosen_attention_mask,
            (0, max_len - example.chosen_attention_mask.numel()),
            value=False,
        ).unsqueeze(0)
        rejected_attention_mask = torch.nn.functional.pad(
            example.rejected_attention_mask,
            (0, max_len - example.rejected_attention_mask.numel()),
            value=False,
        ).unsqueeze(0)
        rejected_input_ids = torch.nn.functional.pad(
            example.rejected_input_ids,
            (0, max_len - example.rejected_input_ids.numel()),
            value=pad_token_id,
        ).unsqueeze(0)
        rejected_labels = torch.nn.functional.pad(
            example.rejected_labels,
            (0, max_len - example.rejected_labels.numel()),
            value=ignore_index,
        ).unsqueeze(0)
        result["chosen_input_ids"] = (
            chosen_input_ids
            if "chosen_input_ids" not in result
            else torch.cat([result["chosen_input_ids"], chosen_input_ids], dim=0)
        )
        result["chosen_attention_mask"] = (
            chosen_attention_mask
            if "chosen_attention_mask" not in result
            else torch.cat(
                [result["chosen_attention_mask"], chosen_attention_mask], dim=0
            )
        )
        result["chosen_labels"] = (
            chosen_labels
            if "chosen_labels" not in result
            else torch.cat([result["chosen_labels"], chosen_labels], dim=0)
        )
        result["rejected_input_ids"] = (
            rejected_input_ids
            if "rejected_input_ids" not in result
            else torch.cat([result["rejected_input_ids"], rejected_input_ids], dim=0)
        )
        result["rejected_attention_mask"] = (
            rejected_attention_mask
            if "rejected_attention_mask" not in result
            else torch.cat(
                [result["rejected_attention_mask"], rejected_attention_mask], dim=0
            )
        )
        result["rejected_labels"] = (
            rejected_labels
            if "rejected_labels" not in result
            else torch.cat([result["rejected_labels"], rejected_labels], dim=0)
        )

    return result
