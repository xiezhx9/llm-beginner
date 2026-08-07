---
tags:
  - LLM
  - GPT
  - Decoder
  - PyTorch
aliases:
  - Task2 miniGPT模型
  - Decoder-only模型构建
---

# Task 2.4：mini-GPT 模型构建与生成

> [!summary]
> MiniGPT 把 token ID 变成 Embedding，依次经过多个 Pre-LN Decoder Block，再用 Final LayerNorm 和 LM head 输出每个位置对整个词表的 logits。生成时反复采样下一个 token，并通过 KV cache 只计算新增位置。

返回总览：[[Task2-miniGPT学习索引]]；核心注意力见 [[Task2-03-Causal-Attention与KV-Cache]]。

## 1. 当前配置

| 参数 | 数值 | 含义 |
|---|---:|---|
| `vocab_size` | `400` | token 类别数 |
| `block_size` | `64` | 训练/生成上下文长度 |
| `num_layers` | `2` | Decoder Block 数量 |
| `num_heads` | `4` | 每层 attention heads |
| `embed_dim` | `128` | 隐藏维度 $D$ |
| `head_dim` | `32` | $D/H$ |
| `ffn_multiplier` | `2` | FFN 隐藏维 `256` |
| `dropout` | `0.2` | 正则化比例 |

## 2. 为什么 `forward` 输入 IDs

Tokenizer 属于模型外的数据预处理：

$$
\text{text}\xrightarrow{tokenizer}\text{ids}
$$

神经网络只接收 Tensor：

$$
\text{ids}\xrightarrow{MiniGPT}\text{logits}
$$

这样训练、评估和部署都可复用 `forward(ids)`，模型不依赖字符串处理库。Tokenizer 并没有被忽略，而是在调用模型之前使用。

## 3. Embedding

```python
self.embedding = nn.Embedding(vocab_size, embed_dim)
```

Embedding 是 $V\times D$ 的可训练矩阵：

$$
H^{(0)}=E[\text{ids}]
$$

形状：

$$
[B,T]\rightarrow[B,T,D]
$$

初始向量是随机的，随后通过 next-token loss 与整个模型一起更新。Tokenizer 词表规定每个 ID 代表哪个 byte/subword；Embedding 学习该 ID 的语义表示。

## 4. FeedForward

每个 token 独立通过同一组前馈层：

$$
\operatorname{FFN}(x)=W_2\operatorname{GELU}(W_1x+b_1)+b_2
$$

形状：

$$
[B,T,D]\rightarrow[B,T,mD]\rightarrow[B,T,D]
$$

为什么叫 FeedForward：没有循环状态，也不在 token 之间通信，数据只从第一层流向第二层。

为什么先扩维：更宽的中间空间允许组合更多非线性特征，再压回 $D$ 维参与残差。

为什么 GELU：

$$
\operatorname{GELU}(x)=x\Phi(x)
$$

它平滑调节输入保留程度，不像 ReLU 在 0 处硬截断，是 Transformer 常用激活函数。

## 5. Pre-LN Decoder Block

数学公式：

$$
U=X+\operatorname{CausalMHA}(\operatorname{LN}_1(X))
$$

$$
Y=U+\operatorname{FFN}(\operatorname{LN}_2(U))
$$

代码骨架：

```python
attn = self.attn(self.ln1(x), kv_cache, return_cache)
x = x + attn
x = x + self.ffn(self.ln2(x))
```

Pre-LN 让 Attention/FFN 接收尺度稳定的输入，残差主干保持直接信息和梯度通道。残差要求子层输出仍为 `[B,T,D]`。

## 6. Encoder Block 与 Decoder Block

两者并不是完全相反的结构，核心骨架高度相似：Attention + FFN + Norm + Residual。

| 结构 | Attention 可见范围 | 常见用途 |
|---|---|---|
| Encoder Block | 通常双向看到完整有效输入 | 理解、分类、编码 |
| Decoder Block | causal，只看当前与历史 | 自回归生成 |
| Encoder-Decoder 的 Decoder | causal self-attention + cross-attention | 翻译、条件生成 |

GPT 类模型只堆 decoder block 很常见，因为 next-token prediction 与 causal mask 天然匹配。

## 7. N 个 Block 与 Final LayerNorm

```python
for block in self.blocks:
    x = block(x)
x = self.ln(x)
```

Pre-LN Block 的残差输出不会在 Block 尾部统一归一化，所以所有 Block 之后通常补一次 Final LN，稳定送入 LM head 的尺度。

## 8. LM Head 为什么输出 $V$ 维

每个位置要预测下一个 token，而可选类别正是词表中的 $V$ 个 token：

$$
Z=HW_{vocab}+b
$$

$$
[B,T,D]\rightarrow[B,T,V]
$$

`logits[b,t,v]` 表示第 $b$ 条序列在位置 $t$ 对 token $v$ 的未归一化分数。训练交给 CrossEntropy；生成时再转成概率并采样。

## 9. 为什么随机模型能学会关系

初始 Embedding、QKV 和 LM head 都随机，初始输出自然接近乱猜。预测错误后，梯度路径为：

```text
loss -> logits -> LM head -> Blocks
     -> attention weights -> Q/K/V -> Embedding
```

监督信号只规定正确 next token，并不直接标注应该关注谁。模型通过大量样本寻找能降低平均 loss 的内部表示，注意力关系因此逐渐形成。

## 10. `block_size` 为什么存成属性

外部代码需要知道模型的上下文约束：

- Dataset 按它切训练窗口；
- `generate` 检查 prompt + 新 token 是否越界；
- PPL 评测按它切 dev 文本；
- cache 需要控制最大长度。

RoPE 表支持更大位置，不代表模型训练上下文自动变大。修改 `block_size` 后通常应重新训练。

## 11. `forward` 与 KV cache 接口

```python
forward(ids, kv_cache=None, return_cache=False)
```

普通训练：

```python
logits = model(ids)
```

增量推理：

```python
logits, new_cache = model(ids, old_cache, return_cache=True)
```

模型有多个 Block，因此 cache 是“每层一个 `(K,V)`”的列表。传入 cache 数量必须与 Block 数一致。

## 12. Generate 逻辑

```text
prompt IDs
-> 第一次完整前向并建立 cache
-> 取最后位置 logits
-> 选择 next ID
-> 后续只输入这个新 ID
-> cache 延长
-> 重复
-> tokenizer.decode
```

第一次 `loop_ids=prompt_ids`，之后 `loop_ids=token_to_return`。如果后续仍把完整历史传入，同时又传 cache，历史 token 会被重复追加。

## 13. 四种采样控制

### Greedy

$$
y=\arg\max_i z_i
$$

确定、保守，但可能重复。当前 `temperature=0` 进入 greedy 分支，避免除零。

### Temperature

$$
p_i=\operatorname{softmax}(z_i/\tau)
$$

低温更确定，高温更多样。

### Top-k

只保留概率最大的 $k$ 个 token，再采样。候选数量固定。

### Top-p

按概率降序，保留累计概率达到 $p$ 的最小候选集合。候选数量随分布变化。

Top-k 与 top-p 不互斥。当前实现先 top-k，再在剩余候选中 top-p。

> [!note] Top-p 后的归一化
> `torch.multinomial` 接受未归一化的非负权重，所以屏蔽后不显式归一化仍能采样；生产代码最好重新归一化并检查总和大于 0，边界更清晰。

## 14. `load_for_eval` 为什么只要一个路径

从 `ckpt/best.pt` 的父目录推导：

```text
ckpt/best.pt
ckpt/model_config.json
ckpt/tokenizer.json
```

加载流程：

1. 读取 tokenizer。
2. 读取 config 并重建模型结构。
3. 加载 `state_dict`。
4. 调用 `model.eval()`。
5. 返回 `(model, tokenizer)`。

三份 artifact 必须来自同一实验。仅判断 `best.pt` 存在还不够，稳健实现也应检查 config/tokenizer 并给出明确错误。

## 15. Weight tying 是什么

调参阶段讨论过让输入 Embedding 与 LM head 共享权重：

$$
W_{head}=E^T
$$

它能减少约 $VD$ 个参数，并让输入与输出 token 空间直接关联。它是常见技巧，但当前回滚后的正式源码没有启用。

## 16. 自测

1. 为什么 FFN 不负责 token 之间通信？
2. 为什么 Decoder Block 的输出必须仍为 $D$ 维？
3. 为什么最终每个 token 要输出 $V$ 个 logits？
4. 生成第二步为什么只输入刚生成的一个 ID？
5. Top-k 与 top-p 是否互斥？

> [!answer]- 答案
> 1. FFN 独立作用于每个 `[D]` 向量，token 轴上没有矩阵混合。
> 2. 需要与残差输入相加，并保持 Block 可重复堆叠。
> 3. next-token prediction 是 $V$ 分类问题。
> 4. 历史 K/V 已在 cache 中，再输入完整历史会重复计算和追加。
> 5. 不互斥，可以组合为更严格的候选集合。
