"""Tool-calling SFT pipeline for bonus goal S5."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    PluginSFTDataset,
    collate_sft_batch,
    load_jsonl,
    parse_moss_plugin_record,
)
from train_sft import SFTTrainState


@dataclass(frozen=True)
class PluginSFTConfig:
    """Paths and hyperparameters for MOSS plugin supervised fine-tuning."""

    model_path: Path = Path("models/Qwen2.5-0.5B")
    train_path: Path = Path("data/moss-plugin/train.jsonl")
    eval_path: Path = Path("data/moss-plugin/eval.jsonl")
    output_dir: Path = Path("ckpt/plugin-sft")
    max_length: int = 768
    max_samples: int | None = None
    eval_max_samples: int | None = 10
    eval_max_new_tokens: int = 128
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    seed: int = 42
    device: str | None = None


def load_plugin_model_and_tokenizer(
    config: PluginSFTConfig,
) -> tuple[nn.Module, Any]:
    """Load the base policy and tokenizer for tool-calling SFT."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config.model_path

    # The local CPU is dramatically faster with float32 than emulated bfloat16.
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def build_plugin_dataloader(
    tokenizer: Any,
    config: PluginSFTConfig,
    shuffle: bool = True,
) -> DataLoader[Any]:
    """Build a DataLoader from prepared MOSS with-tools conversations."""

    jsonls = load_jsonl(config.train_path, config.max_samples)

    for record in jsonls:
        validate_tool_call_sample(record)

    dataset = PluginSFTDataset(jsonls, tokenizer, config.max_length)

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_sft_batch, pad_token_id=tokenizer.pad_token_id),
    )


def validate_tool_call_sample(record: dict[str, Any]) -> None:
    """Validate that one sample contains a tool request and tool result."""

    plugin_record = parse_moss_plugin_record(record)

    if len(plugin_record.turns) == 0:
        raise ValueError("Illegal Turn num")

    for turn in plugin_record.turns:
        if not turn.commands.strip() or not turn.tool_responses.strip():
            raise ValueError("Illegal Tool Call Sample")


def train_plugin_sft(config: PluginSFTConfig) -> SFTTrainState:
    """Run S5 LoRA SFT and save a tool-calling adapter."""
    import torch

    from src.lora import save_lora_adapter
    from train_sft import (
        build_lr_scheduler,
        build_optimizer,
        compute_sft_loss,
        prepare_model_for_sft,
        set_seed,
    )

    state = SFTTrainState()

    set_seed(config.seed)

    model, tokenizer = load_plugin_model_and_tokenizer(config)

    device = torch.device(config.device or "cpu")

    model = model.to(device)
    model.train()

    model = prepare_model_for_sft(model, config)

    optim = build_optimizer(model, config)
    dataloader = build_plugin_dataloader(tokenizer, config)
    scheduler = build_lr_scheduler(optim, config, dataloader)

    optim.zero_grad()
    for i in range(config.epochs):
        times = 0
        optim.zero_grad()

        limit_rount = len(dataloader) // config.gradient_accumulation_steps

        for index, train_data in enumerate(dataloader):
            train_data = {
                name: tensor.to(device) for name, tensor in train_data.items()
            }
            loss = compute_sft_loss(model, train_data)
            state.global_step += 1
            state.losses.append(loss.item())

            if index >= limit_rount * config.gradient_accumulation_steps:
                loss = loss / (len(dataloader) % config.gradient_accumulation_steps)

            else:
                loss = loss / config.gradient_accumulation_steps
            loss.backward()

            times += 1
            times %= config.gradient_accumulation_steps
            if times == 0 or index == len(dataloader) - 1:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config.max_grad_norm,
                )
                optim.step()
                scheduler.step()
                optim.zero_grad()
                state.optimizer_step += 1
    save_lora_adapter(
        model,
        config.output_dir,
        {
            "base_model": str(config.model_path),
            "target_modules": list(config.target_modules),
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
        },
    )
    return state


def main() -> None:
    """Run bonus goal S5 with the default configuration."""
    import time

    # ... 你的代码 ...

    start = time.perf_counter()  # 高精度单调时钟

    state = train_plugin_sft(
        PluginSFTConfig(
            max_samples=200,
            max_length=368,
            batch_size=1,
            gradient_accumulation_steps=2,
            epochs=1,

        )
    )
    print(state)
    end = time.perf_counter()
    print(f"耗时: {end - start:.4f} 秒")


if __name__ == "__main__":
    main()
