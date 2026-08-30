"""Unified entry point for Task 3 bonus goals S1-S5."""

# %%
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.compare import CompareConfig, load_variant
from src.experiments import compare_sft_vs_dpo_preference, save_experiment_report
from train_dpo import DPOConfig, build_dpo_dataloader
from train_plugin_sft import PluginSFTConfig
from train_sft import SFTConfig


@dataclass(frozen=True)
class BonusSuiteConfig:
    """Shared paths and experiment settings for all bonus goals."""

    sft: SFTConfig = SFTConfig()
    dpo: DPOConfig = DPOConfig()
    plugin_sft: PluginSFTConfig = PluginSFTConfig()
    ceval_path: Path = Path("data/ceval/validation.jsonl")
    dpo_eval_path: Path = Path("data/dpo/eval.jsonl")
    report_dir: Path = Path("reports/bonus")
    lora_ranks: tuple[int, ...] = (4, 8, 16, 32)
    s4_max_samples: int = 64


def run_s1_full_vs_lora(config: BonusSuiteConfig) -> Any:
    """Run the matched full-fine-tuning versus LoRA comparison."""
    from src.experiments import compare_full_finetune_vs_lora

    report = compare_full_finetune_vs_lora(config.sft)

    save_experiment_report(report, config.report_dir / "s1_bonus.json")
    return report


def run_s2_rank_ablation(config: BonusSuiteConfig) -> Any:
    """Run LoRA ranks 4, 8, 16, and 32 with fixed controls."""
    from src.experiments import run_lora_rank_ablation

    report = run_lora_rank_ablation(config.sft, config.lora_ranks)
    save_experiment_report(report, config.report_dir / "s2_bonus.json")
    return report


def run_s3_catastrophic_forgetting(config: BonusSuiteConfig) -> Any:
    """Compare base and SFT accuracy on the fixed C-Eval subset."""

    from src.benchmarks import (
        compare_catastrophic_forgetting,
        load_ceval_examples,
    )

    device = config.sft.device or "cpu"
    base_model, base_tok = load_variant(
        "base",
        CompareConfig(model_path=config.sft.model_path, device=device),
    )

    sft_model, _ = load_variant(
        "sft",
        CompareConfig(
            model_path=config.sft.model_path,
            sft_adapter_path=config.sft.output_dir,
            dpo_adapter_path=config.dpo.output_dir,
            device=device,
        ),
    )

    examples = load_ceval_examples(config.ceval_path, 80)

    report = compare_catastrophic_forgetting(
        base_model,
        sft_model,
        base_tok,
        examples,
        device,
    )

    save_experiment_report(report, config.report_dir / "s3_bonus.json")
    return report


@torch.no_grad()
def run_s4_preference_comparison(config: BonusSuiteConfig) -> Any:
    """Compare SFT and DPO preference metrics and save the margin curve."""

    device = config.dpo.device or "cpu"
    sft_model, _ = load_variant(
        "sft",
        CompareConfig(
            model_path=config.sft.model_path,
            sft_adapter_path=config.sft.output_dir,
            dpo_adapter_path=config.dpo.output_dir,
            device=device,
        ),
    )
    dpo_model, dpo_tok = load_variant(
        "dpo",
        CompareConfig(
            model_path=config.dpo.model_path,
            sft_adapter_path=config.sft.output_dir,
            dpo_adapter_path=config.dpo.output_dir,
            device=device,
        ),
    )

    from dataclasses import replace

    eval_config = replace(
        config.dpo,
        train_path=config.dpo_eval_path,
        max_samples=config.s4_max_samples,
    )

    dataloader = build_dpo_dataloader(dpo_tok, eval_config, False)

    report = compare_sft_vs_dpo_preference(
        sft_model, dpo_model, sft_model, dataloader, config.dpo.beta
    )
    save_experiment_report(report, config.report_dir / "s4_bonus.json")
    return report


@torch.no_grad()
def run_s5_plugin_sft(config: BonusSuiteConfig) -> Any:
    """Evaluate the trained MOSS tool-calling SFT adapter."""

    from src.benchmarks import evaluate_plugin_tool_calls
    from src.data import load_jsonl, parse_moss_plugin_record

    model, tok = load_variant(
        "sft",
        CompareConfig(
            model_path=config.plugin_sft.model_path,
            sft_adapter_path=config.plugin_sft.output_dir,
            device=config.plugin_sft.device or "cpu",
        ),
    )

    raw_records = load_jsonl(
        config.plugin_sft.eval_path,
        config.plugin_sft.eval_max_samples,
    )
    records = [parse_moss_plugin_record(record) for record in raw_records]
    report = evaluate_plugin_tool_calls(
        model,
        tok,
        records,
        max_new_tokens=config.plugin_sft.eval_max_new_tokens,
        device=config.plugin_sft.device or "cpu",
    )
    save_experiment_report(report, config.report_dir / "s5_bonus.json")
    return report


# %%


def run_bonus_suite(config: BonusSuiteConfig) -> dict[str, Any]:
    """Run S1-S5 and collect their reports under one directory."""
    return {
        "s1": run_s1_full_vs_lora(config),
        "s2": run_s2_rank_ablation(config),
        "s3": run_s3_catastrophic_forgetting(config),
        "s4": run_s4_preference_comparison(config),
        "s5": run_s5_plugin_sft(config),
    }


def parse_args() -> argparse.Namespace:
    """Select one bonus experiment while preserving spawn-safe script execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal",
        choices=("s1", "s2", "s3", "s4", "s5", "all"),
        default="s1",
        help="Bonus experiment to run (default: s1)",
    )
    return parser.parse_args()


def main() -> None:
    """Run one selected bonus experiment with default settings."""
    args = parse_args()
    runners = {
        "s1": run_s1_full_vs_lora,
        "s2": run_s2_rank_ablation,
        "s3": run_s3_catastrophic_forgetting,
        "s4": run_s4_preference_comparison,
        "s5": run_s5_plugin_sft,
        "all": run_bonus_suite,
    }
    runners[args.goal](BonusSuiteConfig())


if __name__ == "__main__":
    main()
