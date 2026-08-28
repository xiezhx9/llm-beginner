"""Base/SFT/DPO generation comparison interfaces."""

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torch import nn

ModelVariant = Literal["base", "sft", "dpo"]


@dataclass(frozen=True)
class CompareConfig:
    """Artifact paths and generation settings for model comparison."""

    model_path: Path = Path("models/Qwen2.5-0.5B")
    sft_adapter_path: Path = Path("ckpt/sft")
    dpo_adapter_path: Path = Path("ckpt/dpo")
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 0.9
    device: str | None = "cpu"


def load_variant(
    variant: ModelVariant,
    config: CompareConfig,
) -> tuple[nn.Module, Any]:
    """Load one base, SFT, or DPO model/tokenizer pair."""
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.lora import inject_lora, load_lora_adapter

    model = AutoModelForCausalLM.from_pretrained(config.model_path, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    if variant == "sft":
        metadata = json.loads((config.sft_adapter_path / "metadata.json").read_text())

        inject_lora(
            model,
            target_modules=metadata["target_modules"],
            r=metadata["r"],
            alpha=metadata["alpha"],
            dropout=metadata["dropout"],
        )

        lora_model = load_lora_adapter(model, config.sft_adapter_path, True)
        return lora_model.eval(), tokenizer
    if variant == "dpo":
        metadata = json.loads((config.dpo_adapter_path / "metadata.json").read_text())

        inject_lora(
            model,
            target_modules=metadata["target_modules"],
            r=metadata["r"],
            alpha=metadata["alpha"],
            dropout=metadata["dropout"],
        )

        lora_model = load_lora_adapter(model, config.dpo_adapter_path, True)
        return lora_model.eval(), tokenizer

    if variant == "base":
        return model.eval(), tokenizer


def generate_response(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    config: CompareConfig,
) -> str:
    """Format one user prompt and decode only the generated response."""

    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from src.chat import format_messages

    # 确保 pad_token 存在（Qwen2 通常使用 eos_token 作为 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, honest and harmless assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    # formated_messages = format_messages(messages, add_generation_prompt=True)

    # 5. 应用聊天模板并编码
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 添加助手回复的起始标记
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 6. 生成文本
    import torch

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        output_ids = outputs[0][inputs.input_ids.shape[1] :]

        return tokenizer.decode(output_ids, skip_special_tokens=True)


def compare_variants(
    prompts: list[str],
    config: CompareConfig,
) -> dict[str, dict[ModelVariant, str]]:
    """Run identical prompts through base, SFT, and DPO variants."""

    base_model, base_tokenizer = load_variant("base", config)

    sft_model, sft_tokenizer = load_variant("sft", config)

    dpo_model, dpo_tokenizer = load_variant("dpo", config)

    result = {}

    for prompt in prompts:
        base_response = generate_response(base_model, base_tokenizer, prompt, config)
        sft_response = generate_response(sft_model, sft_tokenizer, prompt, config)
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt, config)
        result[prompt] = {
            ("base"): base_response,
            ("sft"): sft_response,
            ("dpo"): dpo_response,
        }
    return result


def main() -> None:
    """Print a small base/SFT/DPO comparison report."""
    test_prompts = [
        "什么是机器学习？",
        "将以下句子翻译成英文：今天天气真好",
        "写一个Python函数计算斐波那契数列",
        "总结这段话的核心观点：人工智能正在深刻改变医疗、教育和金融等行业，但同时也带来了数据隐私、算法偏见和就业结构变化等挑战。专家呼吁建立完善的监管框架以确保技术向善。",
        "用户：你好！\n助手：",
        "列出三个减少碳排放的方法，用bullet point格式",
        "解释量子纠缠，假设听众是小学生",
        "以下说法对吗？地球是平的。请给出理由。",
        '把下面的JSON格式化并修复错误：{"name": "test", age: 25,}',
        "续写故事：深夜，他打开门发现…",
    ]
    result = compare_variants(test_prompts, CompareConfig(temperature=0.0))

    import json

    Path("eval/compare_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


# %%

if __name__ == "__main__":
    main()
