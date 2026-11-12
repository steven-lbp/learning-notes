# Transformer

## Scaled dot-product attention

每个 Q 与 K 计算匹配分数，再用得到的 Attention 权重对 V 加权求和。MHA 在各个 head 中分别完成这个计算，最后拼接输出。

Each query is scored against the keys, and the resulting attention weights are used to combine the values. Multi-head attention performs this computation independently in each head and concatenates the outputs.

$$
Attention(Q, K, V) = softmax\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

```python
# query: [batch, query_len, heads, head_dim]
# key, value: [batch, key_len, heads, head_dim]
energy = torch.einsum("nqhd,nkhd->nhqk", query, key)
# energy: [batch, heads, query_len, key_len]

attention = torch.softmax(energy / (query.shape[-1] ** 0.5), dim=-1)

out = torch.einsum("nhqk,nkhd->nqhd", attention, value)
# out: [batch, query_len, heads, head_dim]
out = out.flatten(start_dim=-2)
# out: [batch, query_len, heads * head_dim]
```

## Attention tensor shapes

下面跟踪一个 8-head MHA 的前向过程：QKV Projection、分头和 RoPE。输入转置后，后续张量统一使用 sequence-first 布局。

The following example traces an 8-head MHA forward pass through QKV projection, head reshaping, and RoPE. All tensors after the input transpose use a sequence-first layout.

### 1. Input and layout

三个样本分别有 5、3、4 个 Token，padding 后统一为 5 个位置。作为 Key 的 padding 位置通过 Attention Mask 屏蔽。交换前两个维度，把输入从 `[batch, seq, hidden]` 转为 `[seq, batch, hidden]`。

The three samples contain 5, 3, and 4 tokens. Padding gives a sequence length of 5; padded key positions are excluded by the Attention Mask. Transposing the first two dimensions changes the input from `[batch, seq, hidden]` to `[seq, batch, hidden]`.

```python
hidden_size = 512
num_heads = 8
head_dim = 64       # 512 / 8
batch_size = 3
seq_len = 5

# batch_input: [batch, seq, hidden] = [3, 5, 512]
hidden_states = batch_input.transpose(0, 1).contiguous()
# hidden_states: [seq, batch, hidden] = [5, 3, 512]
```

### 2. QKV Projection

Q、K、V 各需要 512 维，合并投影把每个 Token 从 512 维映射到 1536 维。这个例子按每个 head 的 Q、K、V 顺序存放投影结果。

Q, K, and V each need 512 dimensions. A fused projection maps each token from 512 to 1536 dimensions. In this example, the projected values are packed as Q, K, and V within each head.

```python
mixed_x_layer = self.query_key_value(hidden_states)
# [5, 3, 512] → [5, 3, 1536]
```

### 3. Reshape

展开最后一维，得到 24 组、每组 64 维：Q、K、V 各占 8 组。实际的 Attention head 数仍然是 8。

Reshaping the last dimension gives 24 groups of 64 values: 8 groups each for Q, K, and V. The number of Attention heads is still 8.

```python
new_tensor_shape = mixed_x_layer.size()[:-1] + (-1, head_dim)
mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
# [5, 3, 1536] → [5, 3, 24, 64]
```

### 4. Split Q, K, and V

把每个 head 的 Q、K、V 合到一起，形成 192 维，再沿最后一维拆成三个 64 维张量。拆开后的 shape 都是 `[seq, batch, heads, head_dim]`。

Group each head's Q, K, and V into 192 values, then split the last dimension into three chunks of 64. Each resulting tensor has the shape `[seq, batch, heads, head_dim]`.

```python
qkv = mixed_x_layer.view(seq_len, batch_size, num_heads, 3 * head_dim)
# [5, 3, 24, 64] → [5, 3, 8, 192]

query_layer, key_layer, value_layer = torch.split(qkv, head_dim, dim=-1)
# 192 = 64 (Q) + 64 (K) + 64 (V)
# query_layer: [5, 3, 8, 64]
# key_layer:   [5, 3, 8, 64]
# value_layer: [5, 3, 8, 64]
```

[Megatron-LM: Attention implementation](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py)

### 5. RoPE

RoPE 对 Q、K 做与位置相关的旋转，V 保持原样。每个 head 的 64 维组成 32 对分量，每对都同时使用 sin、cos。每个位置有 32 个旋转角，旋转前后的 shape 保持一致。

RoPE applies position-dependent rotations to Q and K, leaving V unchanged. The 64 dimensions in each head form 32 pairs. Each pair uses both sin and cos, giving 32 rotation angles per position while preserving the original tensor shape.

```python
query_layer = rotary_embedding(query_layer)
key_layer = rotary_embedding(key_layer)
# Q, K: [5, 3, 8, 64]
```

[RoPE paper](https://arxiv.org/html/2104.09864v5#S3.SS4.SSS2)

## KV Cache, Prefix Caching, and document memory

三者都复用已经算好的 K/V，区别在于复用范围，以及是否压缩文档信息。

| 机制 | 缓存内容 | 复用范围 | 节省的计算 | 文档压缩 |
| --- | --- | --- | --- | --- |
| KV Cache | 历史 Token 的逐层 K/V | 同一次生成的不同 Step | 重复计算历史 Token | 无 |
| Prefix Caching | 相同前缀的逐层 K/V | 共享前缀的不同请求 | 前缀的重复 Prefill | 无 |
| 文档压缩缓存 | 128 个 Memory Token 的逐层 K/V | 同一 Doc 的不同 Query | 重复编码 Doc，以及读取完整 Doc 序列 | 通过训练把信息压到 Memory |

**Prefix Caching 把 KV 复用扩展到不同请求；文档压缩缓存结合了学习式压缩与 KV 复用。**

All three mechanisms reuse computed K/V. They differ in the scope of reuse and whether they compress document information.

| Mechanism | Cached data | Reuse scope | Work saved | Document compression |
| --- | --- | --- | --- | --- |
| KV Cache | Per-layer K/V of past tokens | Steps within one generation | Recomputing past tokens | No |
| Prefix Caching | Per-layer K/V of an identical prefix | Requests sharing that prefix | Repeated prefix Prefill | No |
| Document memory | Per-layer K/V of 128 Memory Tokens | Queries for the same Doc | Re-encoding the Doc and reading its full token sequence | Learned compression into Memory |

**Prefix Caching extends KV reuse across requests. Document memory combines learned compression with KV reuse.**

### 1. KV Cache: reuse history during generation

标准 causal Transformer 中，历史位置看不到后面新增的 Token，因此历史 K/V 可以保持不变。

**缓存按层保存。** 各层的输入 Hidden States 和投影权重不同，需要各自对应的 K/V；最后一层的缓存无法代替其他层。

下面的例子中，输入 `A B C D` 后，Prefill 就能预测 `E`。接着输入 `E`，经过全部层，预测 `F`；再输入 `F`，经过全部层，预测 `G`。每一步都读取并更新各层自己的 KV。

**一个 Decode Step 是新增 Token 经过整个模型的一次前向传播。** Layer 是网络深度，Step 是生成时间步。

每层只计算新增 Token 的 Q/K/V，用新的 Q 读取历史 KV 和当前 Token 的 KV。历史 Q 不再使用，因此通常只缓存 K/V。

缓存避免了重新运行历史 Token。Full Attention 中，新的 Q 仍需读取所有可见 KV，因此上下文变长后，Attention 计算和内存访问开销仍会增加。

In a standard causal Transformer, past positions cannot attend to newly appended tokens, so their K/V remain unchanged.

**The cache is stored per layer.** Each layer has different input Hidden States and projection weights, so it needs its own K/V. The final layer's cache cannot replace the others.

For a prompt `A B C D` followed by generated tokens `E F G`:

| Phase | Current input | Network execution | Cache update | Prediction |
| --- | --- | --- | --- | --- |
| Prefill | `A B C D` | Process the prompt in parallel through all layers | Store each layer's KV for `A B C D` | `E` |
| Decode Step 1 | `E` | Pass the new token through all layers | Read each layer's past KV and append its KV for `E` | `F` |
| Decode Step 2 | `F` | Pass the new token through all layers | Read each layer's past KV and append its KV for `F` | `G` |

**One Decode Step is one forward pass of the new token through the entire model.** Layer refers to network depth; Step refers to generation time.

Each layer computes Q/K/V only for the new token. Its Q attends to the cached K/V and the current token's K/V. Past queries are no longer needed, so the cache usually stores only K/V.

Caching avoids rerunning past tokens. With full Attention, however, the new Q still reads all visible KV, so longer contexts increase the Attention compute and memory traffic.

[Hugging Face: Caching](https://huggingface.co/docs/transformers/v5.14.0/cache_explanation)

### 2. Prefix Caching: reuse a prefix across requests

下面几个请求的公共指令和 Doc 相同，后面的 Query 各不相同。

第一次请求计算并保存公共前缀的逐层 KV。后续请求命中时，直接加载这些 KV，只处理新的 Query；需要生成回答时，再接普通的 Decode。

能复用的原因是**前缀不依赖后面的 Query**。命中需要前缀 Token 和计算条件一致，包括模型权重、位置编码和 Attention Mask。前面的上下文也参与匹配，语义相近或只有中间一段文字相同，都不足以直接复用。

vLLM 按块管理前缀 KV，块的标识结合本块 Token、前面各块的信息及相关配置。

**Prefix Caching 主要省 Prefill。** 4096 Token 的 Doc 仍占据 4096 个 KV 位置，新的 Query 和生成 Token 仍需对这些位置做 Attention。

Suppose several requests start with the same instructions and document:

```text
Request 1: [Shared instructions + Doc] + Query A
Request 2: [Shared instructions + Doc] + Query B
Request 3: [Shared instructions + Doc] + Query C
```

The first request computes and stores the prefix's per-layer KV. On a cache hit, later requests load those KV and process only the new Query. Answer generation then continues with ordinary Decode.

Reuse works because **the prefix does not depend on the following Query**. A hit requires matching prefix tokens and computation settings, including model weights, position encoding, and Attention Mask. The preceding context is part of the match; similar meaning or an identical middle passage is insufficient.

vLLM manages prefix KV in blocks. A block's identifier incorporates its tokens, the preceding blocks, and relevant configuration.

**Prefix Caching mainly saves Prefill work.** A 4096-token Doc still occupies 4096 KV positions, and new Query or generated tokens still attend to them.

[vLLM: Prefix Caching implementation](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)

### 3. Document memory: Doc → 128 Memory Tokens → Query scoring

业务场景的输入是 `Doc + Cache（128 Token）+ Query`。这里将这 128 个 Cache Token 称为 **Memory Token**，用于承载压缩后的文档信息。

Attention Mask 的行表示发起 Attention 的位置，列表示可读取的位置：

| 读取方 ↓ / 被读取方 → | Doc | Memory | Query |
| --- | --- | --- | --- |
| Doc | 可见 | 可见 | 不可见 |
| Memory | 可见 | 可见 | 不可见 |
| Query | 不可见 | 可见 | 原设计的 Query Mask |

Doc 和 Memory 双向交互，Query 通过 Memory 读取文档信息。**Doc / Memory 在所有层都不依赖 Query**，因此它们的计算可以离线完成。

下面的流程分为两部分：离线对同一 Doc 计算一次，保存 Memory 在每层的 K/V；在线只计算 Query 的表征，逐层读取对应的 Memory KV。Query 内部的 Attention 保持原 Mask；打分头若还读取离线侧的其他表征，也将它们一并缓存。Query 完成一次前向后，就可以交给相关性打分头。

Doc 为 4096 Token、Memory 为 128 Token 时，文档侧缓存从 **4096 个位置变为 128 个位置**，KV 数量为原来的 **1/32**。Query 每层通过这 128 个 Memory 位置读取文档信息。

The business scenario uses the input `Doc + Cache (128 tokens) + Query`. We call those 128 Cache Tokens **Memory Tokens** because they carry compressed document information.

Rows in the Attention Mask represent the attending positions; columns represent the positions they can read:

| Attending positions ↓ / Attended positions → | Doc | Memory | Query |
| --- | --- | --- | --- |
| Doc | Visible | Visible | Blocked |
| Memory | Visible | Visible | Blocked |
| Query | Blocked | Visible | Original Query Mask |

Doc and Memory interact bidirectionally, while Query reads document information through Memory. **Doc and Memory never depend on Query at any layer**, so their computation can run offline.

```text
Offline, once per Doc:
Doc + 128 Memory Tokens
    → All Transformer layers
    → Save Memory K/V at every layer

Online, for each Query:
Query → Layer 1, read Memory KV[1]
      → Layer 2, read Memory KV[2]
      → ... → Layer L, read Memory KV[L]
      → Relevance scoring heads
```

Online computation updates only the Query representations and reads the corresponding Memory KV at each layer. Query-to-Query Attention keeps its original Mask. Any additional offline representations used by the scoring heads are cached as well. One Query forward pass produces the representations for relevance scoring.

For a 4096-token Doc and 128 Memory Tokens, the document-side cache shrinks from **4096 to 128 positions**, or **1/32 of the KV entries**. Each Query layer reads document information through those 128 Memory positions.

#### Why saving every layer matters

第一层中，Memory 读取 Doc 后，输出 Hidden States 才带上文档信息；第二层的 Memory K/V 由这个输出产生。更深的各层也有各自深度的表示。

**在线加载的是离线算好的每层 KV。** 如果去掉 Doc，再把原始 Memory Token 从头运行，第一层就读不到 Doc，第二层的 KV 随之改变。只保存最后一层 Hidden States，也无法代替各层对应的 KV。

复用时保持模型版本、Attention Mask 和位置编码方案一致。使用 RoPE 时，去掉 Doc 后仍需保留缓存 Memory 与 Query 的位置关系，或采用训练时约定的位置方案。

将 Doc 压到 128 个 Memory 位置，是模型通过任务训练学习的压缩；复用这些位置的逐层 KV，是对已训练模型的计算复用。

In Layer 1, Memory attends to Doc and produces Hidden States containing document information. Layer 2 derives its Memory K/V from those states. Deeper layers likewise have representations at their own depth.

**Online inference loads the per-layer KV computed offline.** If Doc is removed and the original Memory Tokens are rerun from scratch, Layer 1 can no longer read Doc, changing Layer 2's KV. Final-layer Hidden States alone cannot replace each layer's KV.

Reuse keeps the model version, Attention Mask, and position-encoding scheme consistent. With RoPE, removing Doc must preserve the positional relationship between cached Memory and Query, or follow the position scheme used during training.

Compressing Doc into 128 Memory positions is learned through task training. Reusing their per-layer KV saves computation within that trained architecture.

### 4. Training and inference

普通训练通常一次处理完整序列并反向传播，K/V 是中间激活；推理缓存用于跨生成步骤或跨请求复用。

文档压缩缓存方案训练时，让 Doc、Memory 和 Query 按指定 Mask 完成前向与反向传播，学习文档压缩和相关性打分。模型权重固定后，再离线生成可跨 Query 复用的 Memory KV；权重更新后，对应缓存需要重建。

Standard training usually processes a full sequence and backpropagates through it. K/V are intermediate activations; inference caches instead reuse K/V across generation steps or requests.

The document-memory model trains Doc, Memory, and Query together under the specified Mask to learn compression and relevance scoring. Once model weights are fixed, per-layer Memory KV can be computed offline and reused across queries. Updating the weights requires rebuilding the corresponding caches.

## Attention design choices

### Why use multiple heads?

多个 head 让 Attention 在不同表示子空间中捕捉不同关系。Head 数量需要通过实验选择，增加 head 数量不一定能提升效果。

Multiple heads let Attention capture different relationships in separate representation subspaces. The number of heads is an empirical design choice; increasing it does not necessarily improve performance.

### Why scale by the square root of the head dimension?

假设 Q、K 的各分量独立、均值为 0、方差为 1，它们的点积方差会随 head dimension 增大。除以 head dimension 的平方根，可将方差缩放回约 1，减少 softmax 输出过于极端的情况，保持有效的梯度。

Assuming independent, zero-mean Q and K components with unit variance, their dot product has variance proportional to the head dimension. Scaling by its square root brings the variance back to roughly 1, reducing extreme softmax outputs and keeping gradients useful.
