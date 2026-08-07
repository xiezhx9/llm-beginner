---
tags:
  - LLM
  - Tokenizer
  - BPE
  - NLP
aliases:
  - Task2 BPE
  - Byte-level BPE
---

# Task 2.1：Byte-level BPE 分词器

> [!summary]
> BPE 是独立于 Transformer 的预处理模型。它先从训练语料中学习一套有顺序的相邻 token 合并规则，再固定下来负责 `text ↔ token IDs`；Embedding 才负责把 ID 变成可训练向量。

返回总览：[[Task2-miniGPT学习索引]]

## 1. 为什么不只按字或按词分

| 方案 | 优点 | 问题 |
|---|---|---|
| 按 byte | 覆盖任何 UTF-8 文本 | 序列很长，单个 token 语义弱 |
| 按字符 | 中文直观 | 未登录字符和多语言覆盖麻烦 |
| 按完整词 | token 语义较完整 | 词表巨大，低频词和新词难处理 |
| BPE 子词 | 覆盖能力与序列长度折中 | 需要提前训练 merge rules |

我们的实现从 `256` 个 byte 开始，因此任何 UTF-8 文本都可以退化为 byte 序列，不存在传统意义上的未登录字符。

```python
self.vocab_id_to_token = {i: bytes([i]) for i in range(256)}
self.vocab_token_to_id = {bytes([i]): i for i in range(256)}
```

## 2. 训练输入与输出

输入：训练语料字符串。

输出：

- `vocab_id_to_token`；
- `vocab_token_to_id`；
- 有顺序的 `merge_rule`；
- 可持久化的 `tokenizer.json`。

当前目标词表为 `400`，基础 byte 词表为 `256`，所以最多学习：

$$
400-256=144
$$

条规则。当前 artifact 的确包含 144 条 merge rule。

## 3. 一轮 BPE 如何训练

当前 token 序列记为：

$$
S=(s_0,s_1,\ldots,s_{n-1})
$$

统计所有相邻有序 pair：

$$
C(a,b)=\sum_{i=0}^{n-2}\mathbf 1[(s_i,s_{i+1})=(a,b)]
$$

选择频率最高的 pair：

$$
(a^*,b^*)=\arg\max_{(a,b)}C(a,b)
$$

创建新 token：

$$
z=a^*\Vert b^*
$$

然后从左到右，把所有互不重叠的 $(a^*,b^*)$ 替换为 $z$。

例如：

```text
原序列: a a a a
规则:   (a, a) -> aa
结果:   aa aa
```

索引匹配后前进 2，未匹配时前进 1，因此不会让同一个 token 同时参加两次合并。

## 4. 循环次数是什么意思

每一轮只新增一个词表 token。目标词表为 $V$ 时，理论循环次数为：

$$
V-256
$$

若序列已经没有相邻 pair，则提前停止。循环次数不是滑动窗口长度，而是“允许学习多少条合并规则”。

词表越大：

- 常见片段可能被合成更长 token；
- 平均 token 序列可能变短；
- Embedding 和 LM head 参数量增大；
- 低频 token 获得的训练样本可能更少。

## 5. Encode 为什么不是简单最长匹配

编码先把文本转成 UTF-8 bytes，再严格按训练顺序重放规则：

```python
for rule in self.merge_rule:
    # 合并当前序列中所有匹配 rule 的相邻 token
```

早期规则建立基础子词，后期规则依赖这些基础子词。若改成任意最长匹配，可能得到训练阶段从未产生过的边界组合。

> [!important] BPE 的模型参数就是 merge rule 的顺序
> 同样一组规则如果执行顺序不同，最终 tokenization 也可能不同。因此持久化时不能只存最终 vocab，还必须保存 merge 顺序。

## 6. 为什么 `(a, ab)` 和 `(aa, b)` 是不同 pair

虽然：

$$
a\Vert ab=aa\Vert b=aab
$$

但是：

$$
(a,ab)\ne(aa,b)
$$

BPE 统计的是当前符号序列中的两个相邻 token 及其边界。选择 `(a, ab)` 只能替换边界恰好为 `(a, ab)` 的位置，不能顺便替换 `(aa, b)`。

若只比较拼接后的 bytes，会把两个不同上下文边界混为同一规则，导致：

- pair 频率统计不再反映真实 token 序列；
- 替换位置与选中的 pair 不一致；
- encode 无法可靠重放训练过程。

> [!warning] 当前简化实现的潜在改进
> 内部词表以拼接后的 bytes 作为 token 身份。若两个不同 pair 产生相同 bytes，`vocab_token_to_id` 可能被新 ID 覆盖。更稳健的实现可以让序列始终保存 token ID，用 `(left_id, right_id)` 表示 merge，并显式避免重复 token 内容。

## 7. Decode 为什么能恢复中文

UTF-8 汉字通常由多个 byte 组成，单个 byte token 不一定能独立解码。正确做法是先拼接全部 bytes，再统一 decode：

```python
tokens = [self.vocab_id_to_token[token_id] for token_id in ids]
text = b"".join(tokens).decode("utf-8")
```

因此 roundtrip 应满足：

$$
\operatorname{decode}(\operatorname{encode}(x))=x
$$

## 8. 空格算不算字符

算。文本先执行 UTF-8 编码，普通英文空格会变成 byte `32`，与字母和标点一样进入 pair 统计。

成熟 tokenizer 常给“带前导空格的词”特殊处理，因为空格能表示英文词边界。我们的简化实现没有预分词器，空格只是普通 byte，但 BPE 仍可能学出包含空格的高频 token。

## 9. Tokenizer 与 Embedding

Tokenizer：

$$
\text{text}\leftrightarrow\text{integer IDs}
$$

Embedding：

$$
i\mapsto E_i\in\mathbb R^D
$$

Tokenizer 不返回语义向量。`nn.Embedding(V,D)` 本质是一张 $V\times D$ 的可训练查找表；输入 ID $i$ 就取第 $i$ 行。

训练顺序：

1. 用训练语料单独训练 BPE。
2. 保存并冻结 `tokenizer.json`。
3. 用 `tokenizer.vocab_size` 创建 Embedding 和 LM head。
4. 通过 next-token loss 训练 Embedding 与 Transformer。

BPE 与 Transformer 不共享计算图。训练语言模型时，token ID 是离散整数，梯度不会穿过 `encode` 回到 merge rules。

## 10. 与 BERT tokenizer 的区别

“BPE”是算法家族；`transformers` 中的 tokenizer 是包含算法、词表、规范化、预分词、特殊 token 和序列打包规则的完整产品。

例如 BERT 常使用 WordPiece，而 GPT-2 使用 byte-level BPE。成熟 tokenizer 通常额外处理：

- 大小写与 Unicode 规范化；
- 空格和单词边界；
- BOS/EOS/PAD/UNK；
- truncation、padding 和 batch 输出；
- 高性能 Rust 实现。

我们的实现重点是理解核心 merge 过程，而不是替代生产级 tokenizer。

## 11. 性能优化方向

当前训练每一轮都重新：

1. 扫描整个序列统计 pair；
2. 扫描整个序列执行 merge。

朴素复杂度大致为：

$$
O((V-256)N)
$$

可优化方向：

| 方法 | 思路 |
|---|---|
| 增量 pair 计数 | merge 后只更新受影响邻域，而非全量 Counter |
| 链表/邻接索引 | 快速删除和连接 token 节点 |
| 最大堆 | 更快获取最高频 pair，但要处理过期计数 |
| 按文档/词统计 | 相同片段用频次压缩，避免重复扫描 |
| token ID 内部表示 | 减少 bytes 拼接、哈希和重复对象 |
| 编码 trie | 当算法约定允许时加速候选匹配 |

## 12. 持久化与一致性

`tokenizer.json` 至少保存：

- ordered merge rules；
- token ↔ ID 映射；
- 当前和目标 vocab size。

模型、tokenizer 和 config 必须成套。换 tokenizer 后，即使词表大小相同，同一个 ID 的含义也可能完全不同，旧模型权重不能直接复用。

## 13. 自测

1. 为什么 byte-level BPE 没有传统 OOV？
2. 为什么 merge rule 的执行顺序必须保存？
3. 为什么 pair 相等要比较 tuple，而不能只比较拼接结果？
4. 为什么 decode 要先拼 bytes，再做 UTF-8 解码？
5. 缩小词表会同时带来哪些好处和代价？

> [!answer]- 答案
> 1. 256 个基础 token 覆盖所有 byte，任意 UTF-8 文本都可退化表示。
> 2. 后期 merge 依赖前期形成的符号，顺序变化会改变分词结果。
> 3. pair 表示当前 token 边界，不同边界是不同规则。
> 4. 一个 Unicode 字符可能跨多个 byte token，单独解码会失败。
> 5. 小词表减少输入/输出层参数并提高 token 频次，但会让序列更长、Attention 更贵。
