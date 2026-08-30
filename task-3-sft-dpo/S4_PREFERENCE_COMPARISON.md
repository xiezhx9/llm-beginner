# S4 SFT 与 DPO 偏好对比

## 1. 实验目的

DPO 希望策略模型相对于冻结的 SFT reference，更偏好 `chosen` 回答而不是
`rejected` 回答。每个样本的隐式 reward margin 为：

\[
m=\beta\left[
(\log\pi(y_c|x)-\log\pi_{ref}(y_c|x))
-(\log\pi(y_r|x)-\log\pi_{ref}(y_r|x))
\right].
\]

- `m > 0`：策略相对 reference 更偏向 chosen；
- `m < 0`：策略相对 reference 更偏向 rejected；
- `m = 0`：偏好强度没有变化。

SFT baseline 与它自身作为 reference 比较，因此 margin 必然为 0，win rate 按平局
计为 50%。真正需要观察的是 DPO policy 相对同一个 SFT reference 的变化。

## 2. 固定设置

```text
Policy: ckpt/dpo
Reference: ckpt/sft
Held-out pairs: 64
Max length: 512
Beta: 0.1
Evaluation: deterministic, no gradients
```

## 3. 验收结果

| 指标 | SFT baseline | DPO policy |
|---|---:|---:|
| 偏好对数量 | 64 | 64 |
| Chosen win rate | 50.00% | 54.69% |
| Chosen 胜 / 负 / 平 | 0 / 0 / 64 | 35 / 29 / 0 |
| Mean reward margin | 0 | 0.000536 |
| Median reward margin | 0 | 0.002895 |
| Margin 范围 | 0 | -0.1254 到 0.1485 |

训练阶段的 reward margin 曲线共有 100 个 step。batch margin 波动明显，移动平均
在零点附近震荡，但训练后半段和终点略偏正。

## 4. 结论

DPO 模型在 held-out 偏好对上的 chosen win rate 比 50% 基线高约 4.69 个百分点，
说明训练产生了方向正确但较弱的偏好变化。平均 margin 非常接近 0，而且仍有 29
对偏向 rejected，因此不能表述为 DPO 已稳定学会全部偏好。

更可靠的改进方向包括扩大训练与评估数据、使用多个 seed、检查偏好对质量、调节
`beta` 与学习率，并报告置信区间。当前实验的准确结论是：**DPO 带来了轻微的
chosen 偏好提升，但信号较弱。**

## 5. 复现

```bash
uv run python run_bonus.py --goal s4
```

本地结果写入 `reports/bonus/s4_bonus.json`，训练曲线位于
`ckpt/dpo/reward_margin_curve.png`；两者均不提交 Git。
