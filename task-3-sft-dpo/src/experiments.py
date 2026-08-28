"""Interfaces for all Task 3 bonus experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torch import nn

from train_sft import SFTConfig

FineTuneMode = Literal["full", "lora"]
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compare import CompareConfig, load_variant
from train_sft import (
    build_lr_scheduler,
    build_optimizer,
    build_sft_dataloader,
    compute_sft_loss,
)


@dataclass(frozen=True)
class TrainingRunMetrics:
    """Comparable resource and quality measurements from one SFT run."""

    name: str
    mode: FineTuneMode
    trainable_parameters: int
    total_parameters: int
    peak_memory_bytes: int
    wall_time_seconds: float
    final_train_loss: float
    eval_loss: float
    artifact_dir: Path


@dataclass(frozen=True)
class FullVsLoRAReport:
    """S1 result comparing full fine-tuning with LoRA."""

    full: TrainingRunMetrics
    lora: TrainingRunMetrics


@dataclass(frozen=True)
class RankAblationReport:
    """S2 measurements indexed by LoRA rank."""

    runs: dict[int, TrainingRunMetrics]


@dataclass(frozen=True)
class PreferenceMetrics:
    """Preference accuracy and reward-margin summary for one model."""

    pair_count: int
    chosen_win_rate: float
    mean_reward_margin: float
    reward_margins: tuple[float, ...]


@dataclass(frozen=True)
class PreferenceComparisonReport:
    """S4 comparison between the SFT-only and SFT+DPO policies."""

    sft: PreferenceMetrics
    dpo: PreferenceMetrics


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    trainable = 0
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        total += p.numel()

    return trainable, total


def run_sft_variant(
    config: SFTConfig,
    mode: FineTuneMode,
    experiment_name: str,
) -> TrainingRunMetrics:
    """Train and measure one full-fine-tune or LoRA SFT variant."""

    # training loop
    from train_sft import load_model_and_tokenizer, prepare_model_for_sft, set_seed

    set_seed(config.seed)

    model, tokenizer = load_model_and_tokenizer(config)

    if mode == "lora":
        model = prepare_model_for_sft(model, config)
    elif mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    device = torch.device(config.device or "cpu")
    model = model.to(device)
    model.train()

    dataloader = build_sft_dataloader(tokenizer, config)

    optim = build_optimizer(model, config)

    lr_scheduler = build_lr_scheduler(optim, config, dataloader)

    start = time.perf_counter()
    optim.zero_grad()
    losses = []
    for i in range(config.epochs):
        times = 0
        optim.zero_grad()

        limit_rount = len(dataloader) // config.gradient_accumulation_steps

        for index, train_data in enumerate(dataloader):
            train_data = {
                name: tensor.to(device) for name, tensor in train_data.items()
            }
            loss = compute_sft_loss(model, train_data)

            losses.append(loss.item())

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
                lr_scheduler.step()
                optim.zero_grad()
    wall_time_seconds = time.perf_counter() - start

    with torch.inference_mode():
        model.eval()
        from dataclasses import replace

        eval_config = replace(
            config, train_path=config.eval_path, max_samples=config.eval_max_samples
        )
        eval_dl = build_sft_dataloader(tokenizer, eval_config, False)
        eval_loss = []

        eval_loss_sum = 0
        eval_token_count = 0

        for batch in eval_dl:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            loss = compute_sft_loss(model, batch)
            valid_tokens = (batch["labels"][:, 1:] != -100).sum().item()

            eval_loss_sum += loss.item() * valid_tokens
            eval_token_count += valid_tokens

        eval_loss = eval_loss_sum / eval_token_count

    trainable_param_count, all_param_count = count_parameters(model)
    import resource
    import sys

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    peak_memory_bytes = (
        peak_rss
        if sys.platform == "darwin"  # macOS 返回 bytes
        else peak_rss * 1024  # Linux 返回 KiB
    )

    from src.lora import save_lora_adapter

    artifact_dir = config.output_dir / experiment_name

    if mode == "lora":
        save_lora_adapter(
            model,
            artifact_dir,
            {
                "base_model": str(config.model_path),
                "target_modules": list(config.target_modules),
                "r": config.lora_r,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
            },
        )
    else:
        model.save_pretrained(artifact_dir)
        tokenizer.save_pretrained(artifact_dir)

    return TrainingRunMetrics(
        experiment_name,
        mode,
        trainable_param_count,
        all_param_count,
        peak_memory_bytes,
        wall_time_seconds,
        sum(losses[-10:]) / len(losses[-10:]),
        eval_loss,
        config.output_dir / experiment_name,
    )


def _run_sft_variant_worker(
    config: SFTConfig,
    mode: FineTuneMode,
    experiment_name: str,
    result_queue: Any,
) -> None:
    """Run one variant in a child process and return its result or traceback."""
    import traceback

    try:
        result_queue.put(("ok", run_sft_variant(config, mode, experiment_name)))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))


def run_sft_variant_isolated(
    config: SFTConfig,
    mode: FineTuneMode,
    experiment_name: str,
) -> TrainingRunMetrics:
    """Measure one variant in a fresh spawned process."""
    import multiprocessing
    import queue

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_sft_variant_worker,
        args=(config, mode, experiment_name, result_queue),
    )
    process.start()
    process.join()

    try:
        if process.exitcode != 0:
            raise RuntimeError(
                f"{mode} child process exited with code {process.exitcode}"
            )

        try:
            status, payload = result_queue.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError(
                f"{mode} child process returned no metrics"
            ) from error

        if status == "error":
            raise RuntimeError(f"{mode} child process failed:\n{payload}")
        if not isinstance(payload, TrainingRunMetrics):
            raise TypeError(f"Unexpected {mode} worker result: {type(payload)!r}")
        return payload
    finally:
        result_queue.close()
        result_queue.join_thread()


def compare_full_finetune_vs_lora(config: SFTConfig) -> FullVsLoRAReport:
    """Run S1 with identical data, seed, and step budget."""
    lora_metrics = run_sft_variant_isolated(config, "lora", "s1-lora")
    full_metrics = run_sft_variant_isolated(config, "full", "s1-full")

    return FullVsLoRAReport(
        full=full_metrics,
        lora=lora_metrics,
    )


def run_lora_rank_ablation(
    config: SFTConfig,
    ranks: Sequence[int] = (4, 8, 16, 32),
) -> RankAblationReport:
    """Run S2 while changing only LoRA rank and matched output paths."""
    raise NotImplementedError("TODO: run the LoRA rank ablation")


import torch


@torch.no_grad()
def evaluate_preference_model(
    policy_model: nn.Module,
    reference_model: nn.Module,
    dataloader: Any,
    beta: float,
) -> PreferenceMetrics:
    """Measure chosen win rate and reward margins on fixed DPO pairs."""

    from src.losses import DPOLossOutput, dpo_loss, sequence_log_probs
    from train_dpo import compute_dpo_batch_loss

    pair_count = 0
    margins = []
    chosen_win_count = 0

    policy_model.eval()
    reference_model.eval()

    for batch in dataloader:
        loss: DPOLossOutput = compute_dpo_batch_loss(
            policy_model, reference_model, batch, beta
        )

        chosen_win_count += (loss.reward_margin > 1e-8).sum().item()
        chosen_win_count += (loss.reward_margin.abs() < 1e-8).sum().item() * 0.5

        pair_count += loss.reward_margin.numel()
        margins.extend(loss.reward_margin.detach().tolist())

    return PreferenceMetrics(
        pair_count,
        chosen_win_count / pair_count,
        sum(margins) / pair_count,
        tuple(margins),
    )


def compare_sft_vs_dpo_preference(
    sft_model: nn.Module,
    dpo_model: nn.Module,
    reference_model: nn.Module,
    dataloader: Any,
    beta: float,
) -> PreferenceComparisonReport:
    """Run S4 on the same held-out preference pairs."""

    sft_vs_ref = evaluate_preference_model(sft_model, reference_model, dataloader, beta)
    dpo_vs_ref = evaluate_preference_model(dpo_model, reference_model, dataloader, beta)

    return PreferenceComparisonReport(sft_vs_ref, dpo_vs_ref)


def plot_reward_margin_curve(
    steps: Sequence[int],
    reward_margins: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot and save the S4 reward-margin training curve."""

    import math

    import matplotlib.pyplot as plt

    step_values = list(steps)
    margin_values = [float(margin) for margin in reward_margins]

    if not step_values:
        raise ValueError("steps and reward_margins must not be empty")
    if len(step_values) != len(margin_values):
        raise ValueError("steps and reward_margins must have the same length")
    if not all(math.isfinite(margin) for margin in margin_values):
        raise ValueError("reward_margins must contain only finite values")

    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    # A trailing average makes the trend readable without hiding noisy batch values.
    window_size = min(
        20,
        len(margin_values),
        max(2, round(math.sqrt(len(margin_values)))),
    )
    smoothed_margins = []
    running_sum = 0.0
    for index, margin in enumerate(margin_values):
        running_sum += margin
        if index >= window_size:
            running_sum -= margin_values[index - window_size]
        count = min(index + 1, window_size)
        smoothed_margins.append(running_sum / count)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        step_values,
        margin_values,
        color="#8BB8A8",
        linewidth=1.2,
        alpha=0.55,
        label="Batch reward margin",
    )
    ax.plot(
        step_values,
        smoothed_margins,
        color="#D65A31",
        linewidth=2.2,
        label=f"Moving average (window={window_size})",
    )
    ax.axhline(0.0, color="#343A40", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("DPO Reward Margin During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward margin")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_experiment_report(report: Any, output_path: str | Path) -> Path:
    """Persist a dataclass experiment report as JSON."""
    import json
    from dataclasses import asdict, is_dataclass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(asdict(report) if is_dataclass(report) else report, default=str)
    )

    return Path(output_path)
