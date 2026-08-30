# S1 全量微调与 LoRA 对照实验报告

## 1. 实验目的

在相同 Qwen2.5-0.5B 基座、MOSS SFT 数据、随机种子和训练步数下，对比 LoRA
与全量微调的可训练参数量、峰值内存、训练耗时和验证损失，观察参数高效微调在
学习规模实验中的资源收益与质量差异。

本报告的原始数据见 [`reports/bonus/s1_bonus.json`](reports/bonus/s1_bonus.json)，
环境准备与跨机器复现细节见 [`S1_CODEX_HANDOFF.md`](S1_CODEX_HANDOFF.md)。

## 2. 固定配置

```text
Base model: Qwen2.5-0.5B
Data: MOSS-003-sft no-tools 固定子集
Device / dtype: Intel Core i5-13600KF CPU / FP32
System memory: 32 GB
GPU: RTX 4060 8 GB（本实验未使用）
Train / eval samples: 100 / 50
Max length: 128
Batch size: 1
Gradient accumulation: 4
Epochs: 1
Optimizer steps: 25
Learning rate: 2e-4
Seed: 42
LoRA rank / alpha: 8 / 16
LoRA targets: q_proj, v_proj
```

两个方案均在独立的 `spawn` 子进程中从同一基础模型开始，先执行 LoRA、再执行
全量微调，避免模型状态和峰值内存统计相互污染。由于实验使用 CPU，报告中的
`peak_memory_bytes` 是进程峰值常驻内存（RAM），不是 GPU 显存。

## 3. 实机结果

| 指标 | LoRA SFT | 全量 SFT |
|---|---:|---:|
| 可训练参数 | 540,672 | 494,032,768 |
| 总参数 | 494,573,440 | 494,032,768 |
| 可训练参数占比 | 0.109% | 100% |
| 峰值 RAM | 3.31 GB | 10.78 GB |
| 训练耗时 | 68.95 s | 130.20 s |
| 最后 10 batch 平均训练 loss | 1.1569 | 2.1785 |
| 验证 loss | **1.1015** | 2.3385 |
| 本地产物大小 | 约 2.2 MB | 约 1.98 GB |

由原始指标计算得到：

- LoRA 减少约 **99.891%** 的可训练参数；
- LoRA 降低约 **69.3%** 的峰值 RAM；
- LoRA 的训练速度约为全量微调的 **1.89 倍**；
- LoRA adapter 约为全量 checkpoint 大小的千分之一。

## 4. 结论

1. 在本学习规模实验中，LoRA 只训练 54 万个参数就完成了有效适配，显著降低了
   优化器状态、梯度和模型产物的资源成本。
2. LoRA 的主要模型权重被冻结，但反向传播仍需穿过基础模型，因此耗时不会按
   可训练参数比例缩小；本机实际加速约 1.89 倍，而不是数百倍。
3. 本次 LoRA 的验证 loss 明显低于全量微调，说明固定配置下 LoRA 更稳定。但两组
   为保证控制变量共用 `learning_rate=2e-4`，该学习率适合 LoRA、对全量微调偏大，
   因而不能将结果推广为“LoRA 在任何配置下都优于全量微调”。
4. 对 0.5B 模型做全量 FP32 微调时，优化器状态与梯度让峰值 RAM 达到 10.78 GB；
   LoRA 仅为 3.31 GB，更适合消费级设备和快速实验。
5. 当前“下游质量”使用同一验证集上的 assistant-only token loss 作为代理指标，
   尚未覆盖人工回答质量、指令遵循率或独立任务准确率。

## 5. 局限与后续实验

- 数据量仅 100 条训练记录、50 条验证记录，结论主要用于验证工程链路和资源趋势。
- 仅训练 1 个 epoch，最后 10 batch 的训练 loss 容易受 batch 内容影响，应优先比较
  同一验证集上的 loss。
- 更公平的最佳性能对比应保留本固定控制实验，并额外为全量微调搜索更低学习率，
  例如 `1e-5 / 2e-5 / 5e-5`。
- 本实验没有使用 RTX 4060。若追加 GPU 实验，应记录 CUDA PyTorch 版本、精度、
  梯度检查点和优化器类型，并将峰值显存与本报告的 CPU 峰值 RAM 分开解释。
- 若要评价真实下游能力，应增加固定提示集的人工或模型评分，而不只比较 loss。

## 6. 复现与验证

```bash
uv sync
uv run python data/download.py --s1-only
uv run python eval/run.py
uv run python run_bonus.py --goal s1
```

执行完成后检查：

```text
ckpt/sft/s1-lora/
ckpt/sft/s1-full/
reports/bonus/s1_bonus.json
```

本次自检结果：`lora_param_count`、`loss_masking`、`sft_vs_base` 三项全部通过。
