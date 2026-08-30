"""Direct preference optimization pipeline for Task 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.losses import DPOLossOutput


@dataclass(frozen=True)
class DPOConfig:
    """Paths and hyperparameters for preference optimization."""

    model_path: Path = Path("models/Qwen2.5-0.5B")
    sft_adapter_path: Path = Path("ckpt/sft")
    train_path: Path = Path("data/dpo/train.jsonl")
    output_dir: Path = Path("ckpt/dpo")
    max_length: int = 512
    max_samples: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.1
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    seed: int = 42
    device: str | None = None


@dataclass
class DPOTrainState:
    """Mutable counters and diagnostics collected during DPO."""

    global_step: int = 0
    optimizer_step: int = 0
    losses: list[float] = field(default_factory=list)
    reward_margins: list[float] = field(default_factory=list)


def load_policy_reference_and_tokenizer(
    config: DPOConfig,
) -> tuple[nn.Module, nn.Module, Any]:
    """Load SFT-initialized policy and a frozen reference model."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.lora import inject_lora, load_lora_adapter

    model = AutoModelForCausalLM.from_pretrained(config.model_path, dtype=torch.float32)
    model = inject_lora(
        model,
        config.target_modules,
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
    )
    model = load_lora_adapter(model, config.sft_adapter_path)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    model2 = AutoModelForCausalLM.from_pretrained(
        config.model_path, dtype=torch.float32
    )
    model2 = inject_lora(
        model2,
        config.target_modules,
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
    )
    model2 = load_lora_adapter(model2, config.sft_adapter_path)

    return model, model2, tokenizer


def freeze_reference_model(model2: nn.Module) -> nn.Module:
    """Disable gradients and training-only behavior for the reference model."""

    model2.eval()

    for p in model2.parameters():
        p.requires_grad = False
    return model2


def build_dpo_dataloader(tokenizer: Any, config: DPOConfig, shuffle=True) -> DataLoader[Any]:
    """Load preference pairs and build a padded DPO DataLoader."""
    from src.data import PreferenceDataset, collate_preference_batch, load_jsonl

    jsons = load_jsonl(config.train_path, config.max_samples)

    dataset = PreferenceDataset(jsons, tokenizer, config.max_length)

    from functools import partial

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_preference_batch,
            pad_token_id=tokenizer.pad_token_id,
            ignore_index=-100,
        ),
    )


def build_optimizer(policy_model: nn.Module, config: DPOConfig) -> Optimizer:
    """Build an optimizer over policy LoRA parameters only."""

    from torch.optim import AdamW

    return AdamW(
        params=[p for p in policy_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def compute_dpo_batch_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    batch: dict[str, Tensor],
    beta: float,
) -> DPOLossOutput:
    """Run policy/ref on chosen/rejected inputs and compute DPO loss."""

    from src.losses import dpo_loss, sequence_log_probs

    policy_chosen_result = policy_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        use_cache=False,
    )
    policy_rejected_result = policy_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        use_cache=False,
    )
    reference_chosen_result = reference_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        use_cache=False,
    )
    reference_rejected_result = reference_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        use_cache=False,
    )

    return dpo_loss(
        policy_chosen_logps=sequence_log_probs(
            policy_chosen_result.logits, batch["chosen_labels"]
        ),
        policy_rejected_logps=sequence_log_probs(
            policy_rejected_result.logits, batch["rejected_labels"]
        ),
        reference_chosen_logps=sequence_log_probs(
            reference_chosen_result.logits, batch["chosen_labels"]
        ),
        reference_rejected_logps=sequence_log_probs(
            reference_rejected_result.logits, batch["rejected_labels"]
        ),
        beta=beta,
    )


def train_dpo(config: DPOConfig) -> DPOTrainState:
    """Run DPO from the SFT adapter and save the resulting adapter."""

    import random

    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    policy_model, reference_model, tokenizer = load_policy_reference_and_tokenizer(
        config
    )

    reference_model = freeze_reference_model(reference_model)

    dataloader = build_dpo_dataloader(tokenizer, config)

    optimizer = build_optimizer(policy_model, config)

    state_global_step = 0
    optimizer_step = 0
    losses = []
    reward_margins = []

    optimizer.zero_grad()
    for i in range(config.epochs):
        for index, batch in enumerate(dataloader):
            state_global_step += 1

            loss: DPOLossOutput = compute_dpo_batch_loss(
                policy_model, reference_model, batch, config.beta
            )

            import torch

            rounds = len(dataloader) // config.gradient_accumulation_steps
            last_index = (
                len(dataloader) % config.gradient_accumulation_steps
                if len(dataloader) % config.gradient_accumulation_steps != 0
                else config.gradient_accumulation_steps
            )

            if index < rounds * config.gradient_accumulation_steps:
                (loss.loss / config.gradient_accumulation_steps).backward()
            else:
                (loss.loss / last_index).backward()

            losses.append(loss.loss.item())
            reward_margins.append(loss.reward_margin.mean().item())

            if (index + 1) % config.gradient_accumulation_steps == 0 or index == len(
                dataloader
            ) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_model.parameters() if p.requires_grad],
                    max_norm=config.max_grad_norm,
                )
                optimizer_step += 1
                optimizer.step()
                optimizer.zero_grad()
    from src.lora import save_lora_adapter

    save_lora_adapter(
        policy_model,
        config.output_dir,
        {
            "base_model": str(config.model_path),
            "target_modules": list(config.target_modules),
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
        },
    )

    state = DPOTrainState(state_global_step, optimizer_step, losses, reward_margins)

    from src.experiments import plot_reward_margin_curve

    plot_reward_margin_curve(
        steps=range(1, state.global_step + 1),
        reward_margins=state.reward_margins,
        output_path=config.output_dir / "reward_margin_curve.png",
    )

    return state


def main() -> None:
    """Run Task 3 DPO with the default configuration."""
    import time

# ... 你的代码 ...

    start = time.perf_counter()  # 高精度单调时钟
    state = train_dpo(
        DPOConfig(
            max_samples=192,
            max_length=64,
            batch_size=2,
            gradient_accumulation_steps=4,
            epochs=1,
            learning_rate=2e-5
        )
    )
    print(state)
    end = time.perf_counter()
    print(f"耗时: {end - start:.4f} 秒")


if __name__ == "__main__":
    main()
