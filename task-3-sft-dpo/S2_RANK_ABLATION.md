# S2 LoRA Rank 消融实验

## 1. 实验目的

比较 LoRA rank `4 / 8 / 16 / 32` 对参数量、adapter 大小、训练耗时、峰值内存和
验证损失的影响。除了 `lora_r` 之外，数据、随机种子、训练步数和其他 LoRA
配置均保持一致。

LoRA 更新为：

\[
\Delta W = \frac{\alpha}{r}BA,
\qquad
A\in\mathbb{R}^{r\times d_{in}},\quad
B\in\mathbb{R}^{d_{out}\times r}.
\]

因此 LoRA 参数量为：

\[
N_{LoRA}=r(d_{in}+d_{out}),
\]

rank 翻倍时，可训练参数量和 adapter 大小应近似翻倍。

## 2. 固定配置

```text
Base model: Qwen2.5-0.5B
Device / dtype: Apple M1 CPU / FP32
Train / eval samples: 100 / 50
Max length: 128
Batch size: 1
Gradient accumulation: 4
Epochs: 1
Optimizer steps: 25
Learning rate: 2e-4
LoRA alpha: 16
LoRA dropout: 0
Target modules: q_proj, v_proj
Seed: 42
```

每个 rank 都在独立的 `spawn` 子进程中从同一个基础模型重新开始，四组实验串行
执行，避免模型权重和峰值内存统计互相污染。

## 3. 实机结果

| Rank | 可训练参数 | Adapter 大小 | 峰值 RAM | 训练耗时 | 最后 10 batch 训练 loss | 验证 loss |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 270,336 | 1.1 MB | 2.93 GB | 54.40 s | 1.0573 | **1.1012** |
| 8 | 540,672 | 2.1 MB | 3.21 GB | 65.16 s | 1.1569 | 1.1014 |
| 16 | 1,081,344 | 4.2 MB | 3.03 GB | 56.35 s | 1.1792 | 1.1039 |
| 32 | 2,162,688 | 8.3 MB | 2.72 GB | 62.14 s | 1.1071 | 1.1016 |

四轮训练循环合计约 238 秒。报告保存在本机
`reports/bonus/s2_bonus.json`，adapter 位于 `ckpt/sft/s2-rank-*/`，这些生成物
不提交 Git。

## 4. 结论

1. 可训练参数量和 adapter 大小随 rank 近似线性增长，符合公式预期。
2. 四组验证 loss 的最大差值不足 `0.003`，当前小数据、短训练设置下没有观察到
   增大 rank 带来的稳定质量收益。
3. rank 4 用约一半于 rank 8 的参数取得了本次最低验证 loss，因此在本实验预算下
   性价比最高；这不代表 rank 4 对更复杂任务始终最佳。
4. 峰值 RAM 和耗时没有随 rank 单调增加，因为 0.5B 基础模型占据主要成本，LoRA
   参数增量较小，同时操作系统缓存、内存分配和进程调度会产生测量噪声。
5. 训练 loss 只是最后 10 个 batch 的平均值，容易受 batch 内容影响。rank 消融的
   质量判断应优先使用同一验证集上的 loss。

## 5. Alpha 控制变量说明

本实验遵循“只修改 rank”的配置消融，固定 `alpha=16`。由于实际缩放系数是
`alpha / rank`，四组缩放分别为 `4 / 2 / 1 / 0.5`，所以结果同时包含了 rank 容量
和更新缩放变化的影响。

若要更纯粹地研究 rank 容量，可以追加第二组实验，令 `alpha = 2 * rank`，从而让
`alpha / rank = 2` 保持不变。该实验应作为新增对照，不能覆盖本次固定 alpha 的
结果。

## 6. 复现命令

先按照 `S1_CODEX_HANDOFF.md` 准备 uv 环境、基础模型和 MOSS SFT 数据，然后运行：

```bash
uv run python run_bonus.py --goal s2
```

`run_bonus.py` 会按 `4 -> 8 -> 16 -> 32` 串行运行，并将汇总结果写入
`reports/bonus/s2_bonus.json`。
