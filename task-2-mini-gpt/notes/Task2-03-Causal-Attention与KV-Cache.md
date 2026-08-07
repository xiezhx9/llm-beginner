---
tags:
  - LLM
  - Transformer
  - Attention
  - Causal-Mask
  - KV-Cache
aliases:
  - Task2 Attention
  - Causal Attention与KV Cache
---

# Task 2.3：Causal Attention 与 KV Cache

> [!summary]
> Causal self-attention 让每个 token 只能聚合自己和历史 token；多头机制让模型在多个特征子空间并行建模关系；KV cache 则在逐 token 生成时复用历史 K/V，避免重复计算。

返回总览：[[Task2-miniGPT学习索引]]；位置部分见 [[Task2-02-RoPE旋转位置编码]]。

## 1. 从输入推导 Q/K/V

输入隐藏状态：

$$
X\in\mathbb R^{B\times T\times D}
$$

线性投影：

$$
Q=XW_Q,qquad K=XW_K,qquad V=XW_V
$$

当前实现令 $W_Q,W_K,W_V\in\mathbb R^{D\times D}$，所以：

$$
Q,K,V\in\mathbb R^{B\times T\times D}
$$

Q/K/V 保持与 $X$ 相同形状是模型设计选择，不是注意力理论的硬性规定。这样方便拆成 $H$ 个 head，并在拼接后恢复 $D$ 维参与残差连接。

## 2. 多头拆分

令：

$$
d_h=\frac{D}{H}
$$

形状变化：

$$
[B,T,D]
\xrightarrow{reshape}
[B,T,H,d_h]
\xrightarrow{transpose}
[B,H,T,d_h]
$$

代码：

```python
x = x.reshape(B, T, H, head_dim).transpose(1, 2)
```

把 head 维放到前面后，PyTorch 会把 `[B,H]` 当作批量维，对每个样本的每个 head 并行执行最后两维矩阵乘法。

如果坚持使用 `[B,T,H,d_h]`，也不是数学上不能算，但 $T$ 和 $H$ 夹在矩阵维之间，不能直接写常见的 `Q @ K.transpose(-1,-2)`；通常需要 `einsum` 或更复杂的维度指定。

## 3. Scaled Dot-Product Attention

每个 head：

$$
S=\frac{QK^T}{\sqrt{d_h}}
$$

形状：

$$
[B,H,T_q,d_h]
@
[B,H,d_h,T_k]
\rightarrow
[B,H,T_q,T_k]
$$

矩阵元素：

$$
S_{b,h,i,j}
$$

表示第 $b$ 个样本、第 $h$ 个 head 中，第 $i$ 个 Query token 对第 $j$ 个 Key token 的匹配分数。

经过 mask 和 softmax：

$$
A=\operatorname{softmax}(\operatorname{Mask}(S),\text{dim}=-1)
$$

再聚合 Value：

$$
O=AV
$$

$$
[B,H,T_q,T_k]@[B,H,T_k,d_h]
\rightarrow[B,H,T_q,d_h]
$$

## 4. 为什么除以 $\sqrt{d_h}$

若 Q/K 各维近似独立、均值 0、方差 1，则：

$$
\operatorname{Var}(q\cdot k)\approx d_h
$$

缩放后：

$$
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{d_h}}\right)\approx1
$$

这不是要求训练中 Q/K 永远严格服从标准正态，而是控制 score 随维度增长的典型尺度，避免 softmax 过早饱和。

## 5. Causal mask 的语义

自回归模型预测位置 $i+1$ 时，只允许利用位置不大于 $i$ 的信息。无 cache 时，有效条件：

$$
j\le i
$$

长度 4 的有效矩阵：

```text
       Key 0  1  2  3
Query 0     T  F  F  F
Query 1     T  T  F  F
Query 2     T  T  T  F
Query 3     T  T  T  T
```

这里下三角 `True` 表示允许关注。代码用 `~mask` 选出无效位置并填充：

```python
scores = scores.masked_fill(~causal_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

因为：

$$
e^{-\infty}=0
$$

被屏蔽位置在 softmax 后权重为 0。Mask 并不是另一个概率输入，它是在 softmax 前修改 scores。

## 6. 为什么 mask 每个 token 占一行

Attention score 的行对应 Query，列对应 Key。每个 Query 所处的位置不同，允许查看的历史范围也不同，所以每个 token 必须有自己的一行。

Batch 和 head 不是写在这两个矩阵轴里，而是前导维：

$$
[B,H,T_q,T_k]
$$

基础 causal mask 只需 `[1,1,T_q,T_k]`，再广播到所有 batch 和 head。

## 7. 广播的含义

Mask：

$$
[1,1,T_q,T_k]
$$

Scores：

$$
[B,H,T_q,T_k]
$$

两个大小为 1 的维度分别重复 $B$ 次和 $H$ 次；不是把矩阵按某一列“拉长”，而是每个样本、每个 head 都使用同一张 causal 规则。

与 padding mask 的区别：

| Mask | 典型形状 | 解决的问题 |
|---|---|---|
| Causal mask | `[1,1,Tq,Tk]` | 屏蔽未来位置 |
| Padding mask | `[B,1,1,Tk]` | 屏蔽各样本中的 PAD Key |

两者可先广播，再做逻辑与。

## 8. 多头合并与 $W_O$

每个 head 输出 `[B,H,T,d_h]`，合并：

$$
[B,H,T,d_h]
\rightarrow[B,T,H,d_h]
\rightarrow[B,T,D]
$$

```python
x = x.transpose(1, 2).reshape(B, T, D)
```

`transpose` 后 Tensor 往往不连续；`reshape` 会在必要时复制。若改用 `view`，需先 `.contiguous()`。

最终投影：

$$
\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\ldots,head_H)W_O
$$

$W_O$ 重新混合各 head 信息，并保持输出为 $D$ 维以便与残差相加。

## 9. Attention Dropout

当前在 softmax 权重后应用 Dropout：

```python
weights = softmax(masked_scores)
weights = dropout(weights)
output = weights @ V
```

它在训练时随机移除一部分“Query → Key”连接，并按保留率缩放，其作用类似对注意力路径做正则化。`model.eval()` 时自动关闭。

这不是 LayerNorm。LayerNorm 稳定激活尺度；Dropout 引入随机扰动以降低共适应。

## 10. KV Cache 缓存什么

每一层缓存已经应用 RoPE 的历史 Key 和原始 Value：

$$
K_{past},V_{past}\in\mathbb R^{B\times H\times T_{past}\times d_h}
$$

新 token 到来时，只计算新 Q/K/V，再沿序列维拼接：

$$
K_{all}=\operatorname{cat}(K_{past},K_{new},\text{dim}=-2)
$$

$$
V_{all}=\operatorname{cat}(V_{past},V_{new},\text{dim}=-2)
$$

为什么是 `dim=-2`：最后一维是 `head_dim`，倒数第二维才是 token 序列长度。

## 11. Cache 模式下的形状

第一次处理 prompt：

$$
T_q=T_k=T_{prompt}
$$

后续每次只输入一个新 token：

$$
T_q=1,qquad T_k=T_{past}+1
$$

所以即使是 self-attention，cache 模式下 Query 长度和 Key 长度也可以不同。

此时 causal 有效条件写成绝对位置：

$$
j\le i+T_{past}
$$

代码：

```python
query_pos = torch.arange(query_len)[:, None] + past_len
key_pos = torch.arange(key_len)[None, :]
mask = query_pos >= key_pos
```

它同时兼容普通训练和增量推理。

## 12. KV Cache 省了什么

没有 cache，每生成一个 token 都重新投影全部历史 token 的 K/V。使用 cache 后，历史 K/V 只算一次，后续只计算新增 token。

KV cache：

- 主要用于自回归推理；
- 不改变模型参数和预测定义；
- 用显存换取计算速度；
- 不会让模型更准确；
- cache 会随生成长度线性增长。

训练通常一次输入完整序列，不需要 KV cache，因为所有 token 可以并行计算。

## 13. RoPE offset 与 Cache

历史 cache 已有 $T_{past}$ 个 token 时，新 Q/K 必须从位置 $T_{past}$ 开始旋转：

```python
Q_new, K_new = rope(Q_new, K_new, position_offset=T_past)
```

若每次都从位置 0 开始，新 token 的位置被重复使用，全量前向与 cache 前向不会一致。

## 14. 等价性自检

测试比较：

1. 一次输入完整序列得到 `logits_full`。
2. 每次输入一个 token 并传递 cache，拼出 `logits_inc`。
3. 比较最大绝对误差。

$$
\max|Z_{full}-Z_{cache}|<10^{-4}
$$

当前结果：

$$
2.86\times10^{-6}
$$

微小差异来自浮点矩阵运算顺序，并非逻辑错误。

## 15. 常见错误

| 错误 | 表现 |
|---|---|
| mask 真假语义写反 | 只能看未来或所有权重异常 |
| softmax 后才 masked_fill | 被屏蔽 token 已参与归一化 |
| cache 沿 head_dim 拼接 | shape 或语义错误 |
| 新 token 忘记 RoPE offset | cache 等价性失败 |
| 生成时仍传完整历史并追加 cache | 历史重复，长度快速膨胀 |
| `train()` 状态做等价测试 | Dropout 导致两次结果随机不同 |
| 超过 cache/位置上限 | 内存增长或 RoPE 越界 |

## 16. 自测

1. `scores[b,h,3,0]` 表示什么？
2. 为什么 causal mask 是 token×token，而不是 batch×token？
3. cache 已有 20 个 token 时，新一步的 Q/K/V 形状分别是什么？
4. 为什么 KV cache 缓存 K/V 而通常不缓存 Q？
5. 为什么等价测试必须 `model.eval()`？

> [!answer]- 答案
> 1. 第 b 个样本、第 h 个 head 中，第 4 个 Query 对第 1 个 Key 的分数。
> 2. mask 描述每个 Query 能看哪些 Key；batch/head 只是广播维。
> 3. Q/K/V 新值均为 `[B,H,1,dh]`，拼接后 K/V 为 `[B,H,21,dh]`。
> 4. 每一步只有新 Query 会被使用；历史 Query 不会再次参与新 token 的输出，而历史 K/V 会。
> 5. 关闭 Attention Dropout，确保差异只来自计算路径而非随机采样。
