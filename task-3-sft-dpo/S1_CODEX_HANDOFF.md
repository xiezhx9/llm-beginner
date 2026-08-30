# S1 从零运行交接纪要

本文用于指导另一台机器上的 Codex，从空目录开始恢复 Task 3 环境并正常执行
S1（全量微调与 LoRA 的对照实验）。

## 1. 实验范围

S1 使用同一个 Qwen2.5-0.5B 基座、同一批 MOSS SFT 数据、同一随机种子和同一
训练步数，依次执行：

1. LoRA SFT，产物写入 `ckpt/sft/s1-lora/`；
2. 全量 SFT，产物写入 `ckpt/sft/s1-full/`；
3. 汇总训练参数量、总参数量、峰值内存、训练耗时、训练损失和验证损失；
4. 报告写入 `reports/bonus/s1_bonus.json`。

两个实验由独立的 `spawn` 子进程顺序执行，避免前一个实验的内存峰值污染后一个
实验。Windows 下必须从脚本入口运行，不要直接在没有 `if __name__ == "__main__"`
保护的 Notebook 单元中启动多进程。

## 2. 获取代码

```bash
git clone https://github.com/xiezhx9/llm-beginner.git
cd llm-beginner/task-3-sft-dpo
```

MOSS 数据不会上传到 GitHub。后面的下载脚本会从公开数据源生成下列固定子集：

```text
data/moss-sft/train.jsonl  2000 条
data/moss-sft/eval.jsonl    200 条
```

默认训练配置只取其中前 100 条训练记录和前 50 条验证记录。

## 3. 安装 uv 和依赖

macOS/Linux：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```

项目锁定 Python 3.11，`uv sync` 会创建 `task-3-sft-dpo/.venv`。后续命令都从
`task-3-sft-dpo` 目录执行，并统一使用 `uv run`，不要混用系统 Python。

## 4. 下载基础模型和 S1 数据

模型和数据文件都不会提交到 GitHub。使用项目脚本只准备 S1 所需资源：

```bash
uv run python data/download.py --s1-only
```

如果 Hugging Face 提示未认证，可以先设置 `HF_TOKEN`，但公开模型通常不要求登录。
脚本会从 `Qwen/Qwen2.5-0.5B` 下载约 1GB 的基座，并流式读取
`OpenMOSS-Team/moss-003-sft-data` 的
`moss-003-sft-no-tools.jsonl.zip`；读取到 2200 条记录后立即停止，不下载完整压缩包。
下载完成后，`models/Qwen2.5-0.5B/config.json`、两份 JSONL 和
`data/manifest.json` 应当存在。

## 5. 运行前自检

```bash
uv run python -c "import torch; from pathlib import Path; from train_sft import SFTConfig; c=SFTConfig(); assert Path(c.model_path, 'config.json').is_file(); assert sum(1 for _ in Path(c.train_path).open()) == 2000; assert sum(1 for _ in Path(c.eval_path).open()) == 200; print(torch.__version__, c)"
```

默认关键配置位于 `train_sft.py`：

```text
max_length=128
max_samples=100
eval_max_samples=50
batch_size=1
gradient_accumulation_steps=4
epochs=1
device="cpu"
```

因此每种方案执行 100 次前向/反向传播、25 次优化器更新和 50 次验证前向传播。

## 6. 执行 S1

```bash
uv run python run_bonus.py
```

`run_bonus.py` 的 `--goal` 默认值是 `s1`。执行结束后检查：

```bash
ls ckpt/sft/s1-lora
ls ckpt/sft/s1-full
cat reports/bonus/s1_bonus.json
```

如果实验中途失败，先删除对应的不完整产物目录，再重新执行；不要把
`models/`、`ckpt/` 或 `reports/` 提交回 Git。

## 7. 4060 与内存注意事项

当前已同步版本为了兼容原 Mac，模型以 `float32` 加载且默认 `device="cpu"`。
因此在 RTX 4060 机器上原样执行仍然使用 CPU。32GB 系统内存足够完成默认 S1，
但耗时取决于该机器的 CPU。

不要只把 `device` 改成 `"cuda"`：4060 通常只有 8GB 显存，而 0.5B 模型的 FP32
参数、梯度和 AdamW 状态已经接近或超过 8GB，全量微调还需要激活值和 CUDA
运行空间，通常会 OOM。LoRA 可以直接放入 8GB 显存；若要求全量实验也使用 GPU，
应先由 Codex 实现并验证 BF16、梯度检查点和低显存优化器，或者使用 CPU offload，
同时保证 LoRA 与 full 的数据、seed、步数仍然一致，并在报告中记录不同精度设置。

## 8. 关键文件

- `train_sft.py`：SFT 配置、数据加载、损失、优化器和 scheduler。
- `src/lora.py`：LoRA 注入及 adapter 保存。
- `src/experiments.py`：单个变体训练、进程隔离和 S1 对照编排。
- `run_bonus.py`：S1 入口及 JSON 报告保存。
- `data/moss-sft/`：随仓库固定的 S1 数据。
- `data/manifest.json`：数据来源、规模和随机种子记录。

排查问题时，先运行第 5 节的自检，再检查模型目录、当前工作目录和 `uv run`
是否正确；不要先修改训练逻辑。

## 9. Windows 实机执行结果（2026-08-28）

S1 已在以下机器上完整执行并通过自检：

```text
CPU: Intel Core i5-13600KF
内存: 32 GB
GPU: NVIDIA GeForce RTX 4060 8 GB（本次未使用）
PyTorch: 2.13.0+cpu
训练设备与精度: CPU / FP32
```

实验严格沿用第 5 节默认配置。LoRA 与全量微调使用同一个基座、相同的 100 条
训练样本、50 条验证样本、`seed=42`、`max_length=128`、1 个 epoch 和 25 次
优化器更新。峰值内存是隔离子进程的峰值常驻内存（RAM），不是显存。

| 指标 | LoRA SFT | 全量 SFT |
|---|---:|---:|
| 可训练参数 | 540,672 | 494,032,768 |
| 总参数 | 494,573,440 | 494,032,768 |
| 峰值 RAM | 3,310,899,200 bytes（约 3.31 GB） | 10,777,518,080 bytes（约 10.78 GB） |
| 训练耗时 | 68.95 秒 | 130.20 秒 |
| 最后 10 batch 平均训练 loss | 1.1569 | 2.1785 |
| 验证 loss | 1.1015 | 2.3385 |

本机结果显示：LoRA 减少约 99.891% 的可训练参数、降低约 69.3% 的峰值 RAM，
训练速度约为全量微调的 1.89 倍。在这组固定超参数下 LoRA 的验证 loss 更低；
这不能解释为 LoRA 普遍优于全量微调，因为公平对照刻意让两组共用
`learning_rate=2e-4`，该学习率更适合 LoRA，对全量微调明显偏大。若继续研究质量，
应新增一组“各自调优学习率”的实验，但不要替换本 S1 的固定控制实验。

产物位于本机但按约定不提交 Git：

```text
ckpt/sft/s1-lora/lora_dict.pt
ckpt/sft/s1-lora/metadata.json
ckpt/sft/s1-full/model.safetensors
reports/bonus/s1_bonus.json
eval/result.json
```

最终自检结果为三项全部通过：

```text
lora_param_count: pass，trainable ratio = 0.00109
loss_masking: pass，mask ratio = 0.467
sft_vs_base: pass
```

## 10. 本次为跨机器复现补充的修复

1. `src/experiments.py` 的峰值内存统计已改为跨平台实现：Windows 使用
   `GetProcessMemoryInfo/PeakWorkingSetSize`，Linux/macOS 保留 `resource`，CUDA
   运行则使用 `torch.cuda.max_memory_allocated`。
2. `data/download.py` 的 MOSS 流式下载现在遵循 `HF_ENDPOINT`，国内机器可以让
   模型和数据统一走镜像。
3. `.gitignore` 已排除 `data/hf-home/`、S1 checkpoint 和实验报告，避免误上传
   大模型权重或本地缓存。

在另一台国内网络机器上若 Xet/CAS 返回 401，可这样准备资源：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
uv run python data/download.py --s1-only
```

另一台机器的 Codex 接手后应先按第 5 节自检，再运行 `uv run python run_bonus.py`。
不要直接将 `device` 改为 `cuda`：当前锁文件安装的是 CPU 版 PyTorch，而且 RTX
4060 8 GB 无法安全容纳 FP32 全量微调。若要追加 GPU 对照，应单独实现 CUDA 版
依赖、混合精度、梯度检查点和低显存优化器，并把精度与内存口径写入新报告。
