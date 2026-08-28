"""Qwen chat formatting and assistant-only label construction interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict,get_args

from torch import Tensor


Role = Literal["system", "user", "assistant"]
# %%
import torch
# a = torch.Tensor([1,2,3,4])
# aa = torch.Tensor([[1,2], [2,3], [3, 3 ]])
# aa.nonzero()
# (a.unfold(0,2,1) == aa).all(-1).nonzero(as_tuple=True)
# %%

class ChatMessage(TypedDict):
    """One role/content item in a chat conversation."""

    role: Role
    content: str


@dataclass(frozen=True)
class ConversationEncoding:
    """One tokenized SFT sample before batch padding."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


def validate_messages(messages: Sequence[Mapping[str, str]]) -> None:
    """Validate roles, content, and conversation ordering."""
    loop_order = 2
    for (index, message) in enumerate(messages):
        if message["role"] not in get_args(Role):
            raise TypeError("Invalid role")
        message = ChatMessage(**message)
        if message["role"] == "system":
            if index != 0:
                raise RuntimeError("Wrong Order")

        elif message["role"] == "user":
            if (loop_order == 1):
                raise RuntimeError("Wrong Order")
            else:
                loop_order = 1
        else:
            if (loop_order == 2 or loop_order == 0):
                raise RuntimeError("Wrong Order")
            else:
                loop_order = 2


def format_messages(
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Render messages with Qwen ``im_start``/``im_end`` markers.

    The default one-argument form is required by ``eval/run.py``.
    """
    validate_messages(messages)
    result = ""
    start_tag = r"<|im_start|>"
    end_tag = r"<|im_end|>"
    for message in messages:
        message = ChatMessage(**message)
        result += start_tag + message["role"] + "\n"
        result += message["content"] + end_tag + "\n"
    if add_generation_prompt:
        result += start_tag + "assistant" + "\n"
    return result




def build_labels(
    input_ids: Tensor,
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Mask all non-assistant positions in an existing token sequence.

    The returned tensor must have the same shape as ``input_ids``. The
    optional tokenizer allows the implementation to recover exact token spans;
    the two-argument form remains compatible with the provided evaluator.
    """
    if tokenizer is None:
        raise RuntimeError("no tokenizer")
    start_tag = "<|im_start|>assistant\n"
    end_tag = "<|im_end|>\n"

    start_tag_token = torch.tensor(tokenizer(start_tag).input_ids, device=input_ids.device, dtype=torch.long)
    end_tag_token = torch.tensor(tokenizer(end_tag).input_ids, device=input_ids.device, dtype = torch.long)

    start_pos = (input_ids.unfold(0, start_tag_token.numel(), 1) == start_tag_token).all(-1).nonzero(as_tuple=True)[0]
    end_pos = (input_ids.unfold(0, end_tag_token.numel(), 1) == end_tag_token).all(-1).nonzero(as_tuple=True)[0]

    result = torch.full_like(input_ids, ignore_index)

    for i in range(start_pos.numel()):
        result[start_pos[i] + start_tag_token.numel() : end_pos[end_pos>=start_pos[i]].min() + end_tag_token.numel()] = input_ids[start_pos[i] + start_tag_token.numel() : end_pos[end_pos>=start_pos[i]].min() + end_tag_token.numel()]

    return result


def encode_conversation(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    max_length: int,
    ignore_index: int = -100,
) -> ConversationEncoding:
    """Format, tokenize, truncate, and label one SFT conversation."""
    formated_message = format_messages(messages, False)
    tokenized_message = tokenizer(formated_message, return_tensors="pt")
    input_ids = tokenized_message.input_ids.long()
    labels = build_labels(input_ids[0], messages, tokenizer, ignore_index)

    return ConversationEncoding(input_ids[0,:max_length], tokenized_message.attention_mask[0, :max_length], labels[:max_length])
