"""Supervised fine-tuning pipeline for Task 3."""
# %%
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lora import inject_lora

if TYPE_CHECKING:
    from train_plugin_sft import PluginSFTConfig


@dataclass(frozen=True)
class SFTConfig:
    """Paths and hyperparameters for LoRA-based supervised fine-tuning."""

    model_path: Path = Path("models/Qwen2.5-0.5B")
    train_path: Path = Path("data/moss-sft/train.jsonl")
    eval_path: Path = Path("data/moss-sft/eval.jsonl")
    output_dir: Path = Path("ckpt/sft")
    max_length: int = 128
    max_samples: int | None = 100
    eval_max_samples: int | None = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
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
    device: str = "cpu"


@dataclass
class SFTTrainState:
    """Mutable counters and metrics collected during SFT."""

    global_step: int = 0
    optimizer_step: int = 0
    losses: list[float] = field(default_factory=list)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch randomness used by the training run."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_tokenizer(config: SFTConfig) -> tuple[nn.Module, Any]:
    """Load the local Qwen base model and its matching tokenizer."""
    model_path = config.model_path

    # The local CPU is dramatically faster with float32 than emulated bfloat16.
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

# model, tokenizer = load_model_and_tokenizer(SFTConfig())
# text = tokenizer.apply_chat_template(
#     "123",
#     tokenize=False,
#     add_generation_prompt=True,  # 添加助手回复的起始标记
# )
# inputs = tokenizer(text, return_tensors="pt").to(model.device)

# # 6. 生成文本
# import torch

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#     )

# inputs.input_ids, outputs[0]
# %%

def prepare_model_for_sft(model: nn.Module, config: SFTConfig | PluginSFTConfig) -> nn.Module:
    """Freeze the base model and inject trainable LoRA modules."""

    for p in model.parameters():
        p.requires_grad = False

    model = inject_lora(
        model,
        config.target_modules,
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
    )

    return model


def build_sft_dataloader(tokenizer: Any, config: SFTConfig, shuffle=True) -> DataLoader[Any]:
    """Load MOSS records and build a padded SFT DataLoader."""
    from functools import partial

    from src.data import SFTDataset, collate_sft_batch, load_jsonl

    moss_jsons = load_jsonl(config.train_path, config.max_samples)

    dataset = SFTDataset(moss_jsons, tokenizer, config.max_length)

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_sft_batch, pad_token_id=tokenizer.pad_token_id),
    )


def build_optimizer(model: nn.Module, config: SFTConfig | PluginSFTConfig) -> Optimizer:
    """Build an optimizer over trainable LoRA parameters only."""

    from torch.optim import AdamW

    return AdamW(
        params=[p for p in model.parameters() if p.requires_grad == True],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def compute_sft_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    return outputs.loss


def build_lr_scheduler(optimizer: Optimizer, config: SFTConfig | PluginSFTConfig, dataloader: DataLoader):
    import math

    from transformers import get_cosine_schedule_with_warmup

    total_steps = config.epochs * math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )

    return get_cosine_schedule_with_warmup(
        optimizer, math.floor(total_steps * config.warmup_ratio), total_steps
    )


def train_sft(config: SFTConfig) -> SFTTrainState:
    """Run LoRA SFT and save the best or final adapter artifact."""

    from src.lora import save_lora_adapter
    state = SFTTrainState()

    set_seed(config.seed)

    model, tokenizer = load_model_and_tokenizer(config)

    device = torch.device(config.device)

    model = model.to(device)
    model.train()

    model = prepare_model_for_sft(model, config)

    optim = build_optimizer(model, config)
    dataloader = build_sft_dataloader(tokenizer, config)
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
    """Run Task 3 SFT with the default configuration."""

    state = train_sft(SFTConfig())

    print(state)


if __name__ == "__main__":
    main()

# %%
