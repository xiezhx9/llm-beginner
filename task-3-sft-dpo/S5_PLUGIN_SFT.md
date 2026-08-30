# S5 工具调用 SFT 验收

## 1. 实验目的

S5 使用 MOSS with-tools 对话训练 LoRA adapter，让模型学习包含内部思考、工具命令
和工具返回值的轨迹。当前验收重点不是工具真的被执行，而是模型能否生成约定的
命令协议：

```text
<|Commands|>: ToolName(arguments)<eoc>
```

## 2. Adapter 与评估集

```text
Base model: Qwen2.5-0.5B
Adapter: ckpt/plugin-sft
LoRA rank / alpha: 8 / 16
Target modules: q_proj, v_proj
Trainable parameters: 540,672（96 个 LoRA 张量）
Held-out records: 10
Tool-call turns: 20
Max new tokens: 128
Generation: greedy
```

多轮记录采用 teacher forcing：评估当前工具调用后，将数据集中的标准工具轨迹加入
历史，再继续评估下一回合，避免第一回合生成错误污染后续回合。

## 3. 验收结果

| 指标 | 结果 |
|---|---:|
| 格式有效 | 20 / 20 |
| Format valid rate | **100%** |
| 命令精确匹配 | 1 / 20 |
| Exact match rate | **5%** |

20 个回合全部生成了可解析的 `Search(...)<eoc>`，说明模型已经学会工具调用格式。
精确匹配率低，主要因为该指标要求命令字符串几乎完全一致。例如模型生成：

```text
Search("人工智能 信息安全")
```

而标签是：

```text
Search("人工智能 在 信息安全 领域 应用"), Search("人工智能 信息安全 技术 算法")
```

两者语义相关，但搜索词数量和文字不同，因此被判为不匹配。

## 4. 结论

当前 plugin SFT 已可靠学会命令边界、工具名称和结束标记，格式能力验收通过；但对
标签参数的精确复现能力较弱。5% exact match 不能简单解释为 95% 的调用都无效，
也不能忽略它反映出的参数控制不足。

后续应增加结构化指标，例如工具名准确率、参数 JSON 可解析率、必需参数召回率，
并在真实或 mock 工具执行器中验证命令是否可执行。当前准确结论是：**模型会按
协议调用工具，但参数语义和精确度仍需提升。**

## 5. 复现

```bash
uv run python run_bonus.py --goal s5
```

本地结果写入 `reports/bonus/s5_bonus.json`，adapter 与报告均不提交 Git。
