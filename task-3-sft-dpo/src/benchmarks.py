"""C-Eval interfaces for measuring catastrophic forgetting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from .data import PluginRecord


@dataclass(frozen=True)
class CEvalExample:
    """One four-choice C-Eval question."""

    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str


@dataclass(frozen=True)
class CEvalMetrics:
    """Accuracy summary for one model on a fixed C-Eval subset."""

    correct: int
    total: int
    accuracy: float
    accuracy_by_subject: Mapping[str, float]


@dataclass(frozen=True)
class ForgettingReport:
    """Base/SFT C-Eval scores and their accuracy delta."""

    base: CEvalMetrics
    sft: CEvalMetrics
    accuracy_delta: float


@dataclass(frozen=True)
class ToolCallExampleResult:
    """Generated and expected command details for one plugin turn."""

    conversation_id: str
    turn_index: int
    generated_text: str
    generated_commands: str | None
    expected_commands: str
    format_valid: bool
    exact_match: bool


@dataclass(frozen=True)
class ToolCallMetrics:
    """Aggregate command-format and exact-match metrics for plugin SFT."""

    turn_count: int
    format_valid_count: int
    format_valid_rate: float
    exact_match_count: int
    exact_match_rate: float
    examples: tuple[ToolCallExampleResult, ...]


def extract_tool_commands(text: str) -> str | None:
    """Extract the first non-empty command block terminated by ``<eoc>``."""
    import re

    match = re.search(
        r"<\|Commands\|>\s*:\s*(.*?)\s*<eoc>",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return None

    commands = match.group(1).strip()
    return commands or None


def evaluate_plugin_tool_calls(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[PluginRecord],
    max_new_tokens: int = 128,
    device: str | None = None,
) -> ToolCallMetrics:
    """Generate one command per turn and summarize tool-call quality."""
    import re

    import torch

    from .chat import format_messages

    target_device = torch.device(device or "cpu")
    model = model.eval().to(target_device)
    results = []

    for record in records:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": record.meta_instruction}
        ]

        for turn_index, turn in enumerate(record.turns):
            messages.append({"role": "user", "content": turn.human})
            prompt = format_messages(messages, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_strings=["<eoc>"],
                    tokenizer=tokenizer,
                )

            generated_text = tokenizer.decode(
                output_ids[0, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            generated_commands = extract_tool_commands(generated_text)
            expected_commands = extract_tool_commands(turn.commands)
            if expected_commands is None:
                raise ValueError("Expected plugin command is malformed")

            def normalize(value: str) -> str:
                return re.sub(r"\s+", " ", value).strip()

            exact_match = (
                generated_commands is not None
                and normalize(generated_commands) == normalize(expected_commands)
            )
            results.append(
                ToolCallExampleResult(
                    conversation_id=record.conversation_id,
                    turn_index=turn_index,
                    generated_text=generated_text,
                    generated_commands=generated_commands,
                    expected_commands=expected_commands,
                    format_valid=generated_commands is not None,
                    exact_match=exact_match,
                )
            )

            # Teacher-force the gold trajectory before evaluating the next turn.
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": turn.inner_thoughts + "\n" + turn.commands,
                    },
                    {"role": "user", "content": turn.tool_responses},
                    {"role": "assistant", "content": turn.assistant},
                ]
            )

    turn_count = len(results)
    format_valid_count = sum(result.format_valid for result in results)
    exact_match_count = sum(result.exact_match for result in results)
    return ToolCallMetrics(
        turn_count=turn_count,
        format_valid_count=format_valid_count,
        format_valid_rate=format_valid_count / turn_count if turn_count else 0.0,
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_count / turn_count if turn_count else 0.0,
        examples=tuple(results),
    )


def load_ceval_examples(
    path: str | Path,
    max_samples: int | None = None,
) -> list[CEvalExample]:
    """Load a deterministic C-Eval JSONL subset."""
    import json

    samples = []
    with open(path) as f:
        for line in f:
            if not line.isspace():
                if max_samples is not None and len(samples) >= max_samples:
                    break
                record = json.loads(line.strip())
                samples.append(parse_ceval_example(record))
    return samples


def parse_ceval_example(record: Mapping[str, Any]) -> CEvalExample:
    """Normalize one prepared C-Eval record."""

    return CEvalExample(
        record["subject"],
        record["question"],
        (record["A"], record["B"], record["C"], record["D"]),
        record["answer"],
    )


def format_ceval_prompt(example: CEvalExample) -> str:
    """Render one C-Eval question as a four-choice model prompt."""

    s = "请回答下列选择题，答案只能从A B C D中选择\n"

    s += "问题：" + example.question + "\n"
    s += "A. " + example.choices[0] + "\n"
    s += "B. " + example.choices[1] + "\n"
    s += "C. " + example.choices[2] + "\n"
    s += "D. " + example.choices[3] + "\n"
    s += "答案："

    return s


def extract_choice(response: str) -> str | None:
    """Extract an independent A, B, C, or D choice from a model response."""
    import re

    text = response.strip()
    if not text:
        return None

    patterns = [
        # 中文：答案是 B、我选择 C、正确选项为 D
        (
            r"(?:正确答案|答案|正确选项|选项|我选择|选择|我选|我认为|选)"
            r"\s*(?:是|为|：|:)?\s*(?:选项)?"
            r"\s*[（(\[]?\s*([A-D])"
        ),
        # 英文：answer is B、option C、I choose D
        (
            r"\b(?:ANSWER|OPTION|CHOICE|I\s+(?:CHOOSE|PICK))"
            r"\s*(?:IS|=|:)?\s*[(\[]?\s*([A-D])\b"
        ),
        # 回复以独立字母开头：A、B.、(C)、D：解释
        (
            r"^\s*[（(\[]?\s*([A-D])\s*[）)\]]?"
            r"(?=\s|[.。,:：、]|$)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if match:
            return match.group(1).upper()

    return None


def evaluate_ceval(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[CEvalExample],
    device: str | None = None,
) -> CEvalMetrics:
    """Evaluate one model on an already fixed C-Eval subset."""

    import dataclasses

    import torch

    model = model.eval().to(device) if model is not None else model

    mapp = {}
    correct = 0
    total = 0

    for example in examples:
        tokenized_example = tokenizer(format_ceval_prompt(example), return_tensors="pt")

        for key in tokenized_example:
            tokenized_example[key] = (
                tokenized_example[key].to(device)
                if device is not None
                else tokenized_example[key]
            )

        example = dataclasses.asdict(example)

        if example["subject"] not in mapp:
            mapp[example["subject"]] = {"total": 0, "correct": 0}

        with torch.inference_mode():
            output_ids = model.generate(**tokenized_example, max_new_tokens=8)
            # print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
            answer = tokenizer.decode(
                output_ids[0, tokenized_example.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
        answer = extract_choice(answer)

        if answer == example["answer"]:
            mapp[example["subject"]]["correct"] += 1
        mapp[example["subject"]]["total"] += 1

    accuracy_mapp = {}
    for key in mapp:
        total += mapp[key]["total"]
        correct += mapp[key]["correct"]
        accuracy_mapp[key] = (
            mapp[key]["correct"] / mapp[key]["total"] if mapp[key]["total"] != 0 else 0
        )

    return CEvalMetrics(
        correct, total, correct / total if total != 0 else 0, accuracy_mapp
    )


def compare_catastrophic_forgetting(
    base_model: nn.Module,
    sft_model: nn.Module,
    tokenizer: Any,
    examples: Sequence[CEvalExample],
    device: str | None = None,
) -> ForgettingReport:
    """Compare base and SFT knowledge retention on identical examples."""

    base_report = evaluate_ceval(base_model, tokenizer, examples, device)
    sft_report = evaluate_ceval(sft_model, tokenizer, examples, device)

    return ForgettingReport(base_report, sft_report, sft_report.accuracy - base_report.accuracy)
