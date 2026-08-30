# Task 3 SFT、DPO 与扩展实验总结

## 完成范围

Task 3 已完成手写 LoRA、ChatML 格式化、assistant-only loss masking、MOSS SFT、
DPO、base/SFT/DPO 生成对比，以及 S1-S5 全部扩展实验。正式自检结果为：

```text
lora_param_count: PASS，540,672 / 494,573,440 = 0.109%
loss_masking: PASS，ignore ratio = 46.7%
sft_vs_base: PASS
base / SFT / DPO comparison: PASS，10 个相同提示
```

## 扩展实验结论

| 实验 | 结论 |
|---|---|
| [S1 全量 vs LoRA](S1_FULL_VS_LORA.md) | LoRA 使用 3.31 GB 峰值 RAM、训练 68.95 秒；全量微调使用 10.78 GB、训练 130.20 秒。固定学习率下 LoRA 验证 loss 更低。 |
| [S2 Rank 消融](S2_RANK_ABLATION.md) | rank 4/8/16/32 的验证 loss 差异不足 0.003；rank 4 参数最少且本次 loss 最低。 |
| [S3 灾难性遗忘](S3_CATASTROPHIC_FORGETTING.md) | C-Eval 从 30/80 提升到 34/80；当前小样本上没有观察到遗忘证据。 |
| [S4 偏好比较](S4_PREFERENCE_COMPARISON.md) | DPO chosen win rate 为 54.69%，mean margin 为 0.000536；方向正确但信号较弱。 |
| [S5 工具调用](S5_PLUGIN_SFT.md) | 20/20 格式有效，严格命令匹配 1/20；协议掌握稳定，参数精度不足。 |

## 实验观察

本任务最重要的认识是，后训练并不是“loss 下降就算成功”。LoRA 显著减少了可训练
参数和内存，但 rank 增大没有自动带来质量提升；DPO 的 reward margin 必须结合
win rate 和样本分布解释；工具调用也必须区分“格式正确”和“参数正确”。同时，
小规模实验的波动很大，S3 多答对 4 题、S4 多赢 6 对都不足以支持普遍性结论。
因此我们坚持固定数据与 seed、隔离进程测内存、在 held-out 数据上评估，并把结论
限制在当前实验预算内。这比单纯追求一个更低的训练 loss 更接近真实模型研发流程。

## 复现入口

```bash
uv sync
uv run python data/download.py
uv run python eval/run.py
uv run python run_bonus.py --goal s1
uv run python run_bonus.py --goal s2
uv run python run_bonus.py --goal s3
uv run python run_bonus.py --goal s4
uv run python run_bonus.py --goal s5
```

模型、数据、checkpoint、曲线和一般生成报告只保存在本地；S1 的固定实机指标
`reports/bonus/s1_bonus.json` 随代码提交，作为资源对照报告的原始数据。
