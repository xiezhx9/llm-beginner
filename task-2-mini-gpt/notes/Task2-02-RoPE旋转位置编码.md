---
tags:
  - LLM
  - Transformer
  - RoPE
  - 位置编码
aliases:
  - Task2 RoPE
  - 旋转位置编码
---

# Task 2.2：RoPE 旋转位置编码

> [!summary]
> RoPE 不把位置向量加到 token embedding 上，而是按位置旋转每个注意力头的 Q/K。旋转后的点积自然依赖相对距离，因此非常适合 decoder-only 自注意力和 KV cache 增量生成。

返回总览：[[Task2-miniGPT学习索引]]

## 1. 为什么 Attention 需要位置编码

如果不加入任何位置信息，自注意力只根据 token 表示计算相似性。对输入 token 做同样的排列，输出也会跟着排列，模型无法仅凭 Attention 区分“谁在前、谁在后”。

正余弦绝对位置编码采用：

$$
X_p'=X_p+PE_p
$$

RoPE 采用：

$$
Q_p'=R_pQ_p,\qquad K_p'=R_pK_p
$$

它直接改变 Attention score 的位置关系，而不修改 V。

## 2. 旋转发生在哪些维度

Q/K 的形状：

$$
[B,H,T,d_h]
$$

在每个 batch、每个 head、每个 token 内，把相邻特征两两分组：

```text
(维度0, 维度1)
(维度2, 维度3)
...
```

因此 `head_dim` 必须是偶数。

> [!important] 不同 head 不是共享数据
> 每个 head 持有从 $D$ 维拆出的不同特征。各 head 可以使用同一套位置角度表，但被旋转的 Q/K 数值不同，所以旋转结果当然不同。

## 3. 频率与角度公式

第 $i$ 个二维组的频率：

$$
\omega_i=\text{base}^{-2i/d_h},
\qquad i=0,1,\ldots,\frac{d_h}{2}-1
$$

位置 $p$ 的角度：

$$
\theta_{p,i}=p\omega_i
$$

低维组变化快，高维组变化慢，从而同时表达短距离和长距离位置模式。

当前代码等价于：

```python
position = torch.arange(max_seq_len).unsqueeze(1)       # [L, 1]
inv_freq = base ** (
    -torch.arange(0, head_dim, 2).unsqueeze(0) / head_dim
)                                                     # [1, dh/2]
angles = position @ inv_freq                           # [L, dh/2]
```

## 4. 二维旋转公式

对位置 $p$、第 $i$ 组特征 $(x_{2i},x_{2i+1})$：

$$
\begin{bmatrix}
x'_{2i}\\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{p,i} & -\sin\theta_{p,i}\\
\sin\theta_{p,i} & \cos\theta_{p,i}
\end{bmatrix}
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}
$$

展开：

$$
x'_{2i}=x_{2i}\cos\theta-x_{2i+1}\sin\theta
$$

$$
x'_{2i+1}=x_{2i}\sin\theta+x_{2i+1}\cos\theta
$$

代码中的偶数/奇数切片正好实现这两个式子：

```python
result[..., ::2] = even * cos - odd * sin
result[..., 1::2] = even * sin + odd * cos
```

## 5. 同一位置的角度是否相同

需要同时看“位置”和“二维维度组”：

$$
\theta_{p,i}=p\omega_i
$$

- 相同 $p$、相同 $i$：角度相同。
- 相同 $p$、不同 $i$：频率不同，角度不同。
- 不同 $p$、相同 $i$：位置不同，角度不同。
- 不同 head、相同 $p/i$：使用相同角度表，但旋转不同的 Q/K 数据。

## 6. 为什么能表达相对位置

位置 $p$ 的 Query 与位置 $q$ 的 Key：

$$
(R_pQ)^T(R_qK)
=Q^TR_p^TR_qK
=Q^TR_{q-p}K
$$

最终点积依赖 $q-p$，即 token 间相对距离。这是 RoPE 相比“直接加绝对位置向量”的核心优势。

## 7. 为什么只作用于 Q/K

Attention：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

Q/K 决定“关注谁”，所以位置关系需要影响 Q/K 的匹配分数。V 表示被取出的内容，不负责匹配，无需旋转。

## 8. `cos/sin` 返回什么

当前 `_get_cos_sin` 返回两个 Tensor：

$$
\cos\Theta,\sin\Theta
\in\mathbb R^{1\times1\times T\times(d_h/2)}
$$

它们不是“一个返回偶数位置、一个返回奇数位置”。两个 Tensor 都覆盖全部 token 位置和全部二维维度组：一个保存 cosine，一个保存 sine。随后分别广播到 `[B,H,T,d_h/2]`。

## 9. `position_offset` 为什么存在

无 cache 时输入完整序列，位置从 0 开始：

```text
0, 1, 2, ..., T-1
```

有 cache 时，历史已经包含 $T_{past}$ 个 token，新 token 必须从该位置继续：

```text
T_past, T_past+1, ...
```

所以：

```python
cos, sin = self._get_cos_sin(seq_len, position_offset=T_past, dtype=q.dtype)
```

如果每轮都从位置 0 开始，全量前向和 cache 前向会使用不同角度，`kv_cache_equivalence` 必然失败。

## 10. 为什么预计算并注册 buffer

`cos/sin` 只由位置、维度和 base 决定，可以预计算避免每次 forward 重复做三角函数。

它们不是训练参数，却需要随模型移动设备：

```python
self.register_buffer("cos_arg_tensor", angles.cos(), persistent=False)
self.register_buffer("sin_arg_tensor", angles.sin(), persistent=False)
```

| 属性 | 结果 |
|---|---|
| 不出现在 `model.parameters()` | 优化器不会更新 |
| 随 `model.to(device)` 移动 | 不会出现 CPU/MPS/CUDA mismatch |
| `persistent=False` | 不写入 checkpoint，可重新构造 |

## 11. `pow` 与 `exp(log())`

数学上：

$$
b^x=\exp(x\log b)
$$

因此下面两种写法等价：

```python
base ** exponent
torch.exp(exponent * math.log(base))
```

当前 `base=10000`、指数范围约在 `[-1,0]`，结果在 $(10^{-4},1]$ 附近，远未达到溢出风险，直接 `pow` 清晰且足够稳定。

只有以下情况才更值得关注两种实现的数值路径：

- base 或指数极端大；
- 使用 float16 等低精度；
- 结果接近 underflow/overflow；
- 需要与某个既有实现逐位对齐。

## 12. `max_seq_len` 与模型 `block_size`

当前：

| 参数 | 数值 | 语义 |
|---|---:|---|
| RoPE `max_seq_len` | `2048` | cos/sin 表能索引的最大位置 |
| MiniGPT `block_size` | `64` | 训练和生成约定的上下文长度 |

RoPE 表更长没有问题，但不代表模型已学会 2048 长度。大幅超出训练长度属于位置外推，质量没有保证；当前 `generate` 也限制总 token 数不超过 `block_size`。

## 13. RoPE 是否必须结合 KV cache

不必须。

- 训练全序列时，RoPE 正常提供位置信息，不需要 cache。
- 增量推理时，RoPE 与 cache 经常一起使用，此时必须正确设置 offset。

RoPE 解决“位置如何进入注意力”，KV cache 解决“历史 K/V 如何复用”，二者职责不同。

## 14. 常见错误

| 错误 | 后果 |
|---|---|
| `head_dim` 为奇数 | 最后一维无法组成旋转对 |
| Q/K 使用不同位置角度 | 点积位置关系被破坏 |
| 给 V 也做 RoPE | 改变内容通道，偏离标准定义 |
| cache 模式忘记 offset | 全量与增量 logits 不一致 |
| cos/sin 在 CPU，Q/K 在 MPS | device mismatch |
| 超过 `max_seq_len` | 位置切片越界或长度不匹配 |

## 15. 自测

1. 位置 $p=3$ 时，是否所有特征维都使用同一个角度？
2. 不同 head 为什么可以共享同一张 RoPE 角度表？
3. RoPE 的相对位置信息如何出现在 $QK^T$ 中？
4. cache 已有 12 个 token 时，新 token 的 offset 应是多少？
5. 为什么 RoPE 最大长度 2048 不等于模型可靠上下文就是 2048？

> [!answer]- 答案
> 1. 不是；不同二维维度组有不同频率，角度也不同。
> 2. 位置频率规则相同，但每个 head 的 Q/K 内容不同，旋转结果不同。
> 3. $R_p^TR_q=R_{q-p}$，旋转后点积依赖相对距离 $q-p$。
> 4. `12`，新 token 使用第 12 个位置。
> 5. 前者只是频率表容量，后者还受训练长度、数据和模型约定限制。
