# Transformer

Attention 让每个 Token 根据相关性读取其他可见位置的信息。理解它的计算过程后，就能顺着理解多头、KV Cache，以及 GQA、MLA 等结构如何减少缓存。

Attention lets each token read information from visible positions according to relevance. Its computation provides the foundation for understanding multiple heads, KV caching, and architectures such as GQA and MLA.

- [1. 输入与 Hidden States](#1-输入与-hidden-states)
- [2. 单头 Attention：从 QKV 到输出](#2-单头-attention从-qkv-到输出)
- [3. 多头 Attention：分头、位置与维度](#3-多头-attention分头位置与维度)
- [4. 推理：Prefill、Decode 与 KV Cache](#4-推理prefilldecode-与-kv-cache)
- [5. 真实模型：MHA、GQA 与 MLA](#5-真实模型mhagqa-与-mla)
- [6. 深入 MLA：低秩投影与压缩缓存计算](#6-深入-mla低秩投影与压缩缓存计算)
  - [6.1 为什么需要压缩 KV Cache](#61-为什么需要压缩-kv-cache)
  - [6.2 一个 Token 如何生成 Q 和 KV](#62-一个-token-如何生成-q-和-kv)
  - [6.3 展开后如何计算 Attention](#63-展开后如何计算-attention)
  - [6.4 为什么先降维再升维](#64-为什么先降维再升维)
  - [6.5 如何直接使用压缩缓存](#65-如何直接使用压缩缓存)
  - [6.6 与 Qwen GQA 的异同](#66-与-qwen-gqa-的异同)
- [7. 应用：文档 Memory Cache](#7-应用文档-memory-cache)
- [附录：张量布局与实现](#附录张量布局与实现)

## 1. 输入与 Hidden States

### 1.1 文本如何变成向量？

以“小王去银行取钱”为例，假设 Tokenizer 将它切成 `小王 / 去 / 银行 / 取 / 钱` 这 5 个 Token。实际切分由模型的 Tokenizer 决定。每个 Token 先变成一个整数 ID，再用这个 ID 查 Embedding 表，得到一个 `hidden_size` 维的向量。

若 `hidden_size=512`，这句话的表示就是 `[5, 512]`：5 个位置，每个位置有 512 个数。**序列长度、词表大小、Embedding 维度是三个不同的量。** 向量中的分量由训练学习，不是预先指定的“地点”“动作”等字段。

For illustration, suppose the tokenizer splits the sentence into five tokens: `小王 / 去 / 银行 / 取 / 钱`. Actual tokenization depends on the model. Each token becomes an integer ID, which indexes an Embedding table to retrieve a `hidden_size`-dimensional vector.

With `hidden_size=512`, the sentence has shape `[5, 512]`: five positions with 512 values each. **Sequence length, vocabulary size, and Embedding dimension are distinct quantities.** Components are learned features rather than predefined fields such as location or action.

```text
Text → Token IDs [5] → Embedding lookup → Hidden States [5, 512]
```

### 1.2 Attention 在模型的什么位置？

在常见的 Decoder Transformer 中，Embedding 后是多层 Transformer Block。每层的 Attention 在 Token 之间读取信息，FFN / MoE 对各位置的特征继续变换，并配合归一化和残差连接。最后，LM Head 将模型表示映射为词表上的预测分数。

第一层继承输入 Embedding；后续层接收上一层输出的 Hidden States。因此，**每层都从自己的输入计算自己的 Q、K、V**。下面用 X 表示当前 Attention 的输入；具体实现中，X 可能是经过归一化的层输入。

In a typical Decoder Transformer, embeddings pass through multiple Transformer blocks. Attention exchanges information between token positions, while the FFN or MoE transforms features at each position. Normalization and residual connections accompany these operations. The LM Head maps the final representations to vocabulary prediction scores.

The first layer receives the input embeddings, and later layers receive the preceding layer's Hidden States. **Each layer computes its own Q, K, and V from its own input.** Below, X denotes the current Attention input, which may already be normalized.

## 2. 单头 Attention：从 QKV 到输出

先固定一个样本、一个 head，理解一次完整计算：**生成 QKV → 点积打分 → 缩放与 Mask → softmax → 加权 V**。

First fix one sample and one head: **project QKV → compute dot products → scale and mask → apply softmax → combine V**.

### 2.1 Q、K、V 从哪里来，各自做什么？

同一个输入 X 分别经过三个可学习的线性投影，得到 Q、K、V。这里采用行向量约定，省略 bias。

The same input X passes through three learned linear projections to produce Q, K, and V. The equations use row vectors and omit biases.

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

| 张量 / Tensor | 作用 / Role |
| --- | --- |
| Q — Query | 当前 Token 发起匹配的向量 / What the current token queries with |
| K — Key | 各个可见位置参与匹配的向量 / What each visible position is matched against |
| V — Value | 各个位置提供、被加权组合的内容 / The content each position contributes |

例如，“钱”这个位置的 Q 可以与前面“小王”“银行”“取”等位置的 K 匹配，再按权重组合它们的 V。实际权重由模型学习。**Q、K 决定各位置的权重，V 提供被加权的内容；同一位置的 K 与 V 对应同一个 Token。**

For example, the query at `钱` can match keys at preceding positions such as `小王`, `银行`, and `取`, then combine their values. The actual weights are learned. **Q and K determine weights over positions; V supplies the content. Each key is paired with the value at the same token position.**

### 2.2 点积如何得到匹配分数？

用一个二维例子手算：一个 Query 读取三个可见 Token。点积将对应分量相乘后相加，因此两个向量得到一个标量分数。

For a small numerical example, let one two-dimensional query read three visible tokens. A dot product multiplies corresponding components and sums them, producing one scalar score from two vectors.

```text
q  = [1, 0]
K₁ = [1, 0]     q · K₁ = 1×1 + 0×0 = 1
K₂ = [0, 1]     q · K₂ = 1×0 + 0×1 = 0
K₃ = [1, 0]     q · K₃ = 1×1 + 0×0 = 1
```

### 2.3 为什么缩放，Mask 和 softmax 又做什么？

**缩放：** 每头 Q/K 的维度记作 `d_k`，点积除以 `√d_k`。若各分量独立、均值为 0、方差为 1，点积的方差会随 `d_k` 增大；这样缩放能使方差保持在约 1，减少 softmax 过于极端。本例 `d_k=2`，缩放分数约为 `[0.707, 0, 0.707]`。

**Mask：** 先限制哪些位置可读，再分配权重。Causal Mask 屏蔽未来 Token，Padding Mask 屏蔽补齐的 Key 位置。实现中可将不可见位置的分数设为负无穷，使其 softmax 权重为 0。本例三个位置都可见。

**Softmax：** 对同一个 Query 的所有可见 Key 一起归一化，得到权重，权重之和为 1。本例约为 `[0.401, 0.198, 0.401]`。softmax 沿 Key 位置计算，不是对每个分数分别计算。

**Scaling:** Divide dot products by `√d_k`, where `d_k` is the Q/K dimension per head. With independent, zero-mean, unit-variance components, dot-product variance grows with `d_k`; scaling brings it back to roughly 1 and reduces extreme softmax outputs. Here, `d_k=2` gives scaled scores of approximately `[0.707, 0, 0.707]`.

**Masking:** Determine visibility before assigning weights. Causal masks block future tokens, and padding masks block padded key positions. Setting a blocked score to negative infinity gives it zero softmax weight. All three positions are visible in this example.

**Softmax:** Normalize across visible keys for the same query, producing weights that sum to 1. Here they are approximately `[0.401, 0.198, 0.401]`. Softmax operates over key positions, not independently on each scalar score.

```math
a_j=\frac{\exp(s_j)}{\sum_{\ell}\exp(s_\ell)},\qquad
s_j=\frac{q\cdot k_j}{\sqrt{d_k}}+m_j
```

### 2.4 权重如何变成输出？

取三个位置的 V 如下。每个权重乘以同一位置的 V，再将对应分量相加，得到当前 Query 的输出。

Using the following values, multiply each weight by the value at the same position, then sum corresponding components to obtain the current query's output.

```text
V₁ = [1, 2]     V₂ = [0, 1]     V₃ = [1, 0]

o ≈ 0.401 × [1, 2] + 0.198 × [0, 1] + 0.401 × [1, 0]
  ≈ [0.802, 1.000]
```

两个分量分别是 `0.401×1 + 0.198×0 + 0.401×1` 和 `0.401×2 + 0.198×1 + 0.401×0`。输出有 2 维，因为本例的 V 有 2 维。一般情况下，**Q 与 K 的每头维度必须匹配；V 可以有不同维度，输出维度由 V 决定。**

The two output components are `0.401×1 + 0.198×0 + 0.401×1` and `0.401×2 + 0.198×1 + 0.401×0`. The output has two components because V does. In general, **Q and K must have matching per-head dimensions; V may have a different dimension, which determines the output dimension.**

### 2.5 把所有 Query 一起计算

把各个 Query 按行放在 Q 中，就能一次计算它们与所有 Key 的关系。匹配矩阵和权重矩阵的**行表示 Query 位置，列表示 Key 位置**；softmax 沿每一行计算。M 在可见位置为 0，不可见位置为负无穷。

Stack queries as rows of Q to compute all query-key relationships together. **Rows index query positions, and columns index key positions** in both the score and weight matrices. Softmax operates along each row. M is zero for visible positions and negative infinity for blocked positions.

```math
A=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right),\qquad
O=AV
```

[Attention Is All You Need, §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)

## 3. 多头 Attention：分头、位置与维度

### 3.1 为什么需要多个 head？

不同 head 使用不同的投影参数，可以在不同表示子空间中学习不同的匹配和读取方式。它们的结果拼接后，再经输出投影 W_O 混合。head 的分工由训练形成，不需要人为指定“语法 head”“语义 head”；更多 head 也不保证效果更好。

本例选择 `hidden_size=512`、`num_heads=8`、`head_dim=64`。每个 Q/K/V 投影先得到 512 维，再拆成 `8 × 64`。**投影会计算新数值，reshape/split 只重新组织已有数值。** 每个 head 的投影可以使用完整的 512 维输入。

Different heads use different projection parameters to learn matching and reading patterns in different representation subspaces. Their outputs are concatenated and mixed by W_O. Training determines their roles; heads do not need predefined grammatical or semantic assignments, and adding more heads does not guarantee improvement.

This example chooses `hidden_size=512`, `num_heads=8`, and `head_dim=64`. Each Q/K/V projection produces 512 values, then reshapes them into `8 × 64`. **Projection computes new values; reshape and split reorganize existing ones.** Each head's projection can use the full 512-dimensional input.

### 3.2 用同一组数字走完前向过程

继续使用 5 个 Token，先省略 batch。每个 head 独立执行第 2 节的 Attention，最后合并 head 轴与特征轴。

Continue with five tokens and omit the batch dimension for now. Each head independently performs the Attention computation from Section 2. Concatenation then merges the head and feature axes.

| 阶段 / Stage | Shape |
| --- | --- |
| 输入 X / Input X | `[5, 512]` |
| 分别投影 Q、K、V / Project Q, K, V | 各为 / Each `[5, 512]` |
| 分头 / Split into heads | 各为 / Each `[5, 8, 64]` |
| 对 Q、K 应用 RoPE / Apply RoPE to Q and K | `[5, 8, 64]`，shape 不变 / Unchanged |
| 各 head 的分数与权重 / Scores and weights per head | `[8, 5, 5]` |
| 各 head 对 V 加权后的输出 / Weighted outputs per head | `[5, 8, 64]` |
| 拼接 8 个 head / Concatenate 8 heads | `[5, 512]` |
| 输出投影 W_O / Output projection W_O | `[5, 512]` |

固定一个 head 后，Q、K、V 都是 `[5, 64]`。QKᵀ 沿 64 维特征求和，得到每个 Query 对每个 Key 的 `5 × 5` 分数；A 乘 V 则沿 5 个 Key 位置求和，得到每个 Query 的 64 维输出。

For one head, Q, K, and V each have shape `[5, 64]`. QKᵀ sums over the 64 features to produce `5 × 5` position scores. Multiplying A by V sums over the five key positions to produce a 64-dimensional output for each query.

```math
\underbrace{Q}_{5\times64}\underbrace{K^T}_{64\times5}
=\underbrace{S}_{5\times5},\qquad
\underbrace{A}_{5\times5}\underbrace{V}_{5\times64}
=\underbrace{O}_{5\times64}
```

拼接不会把不同 Token 混在一起：每个 Token 收集自己的 8 个 head 输出，得到 `8 × 64 = 512` 维。W_O 在这些特征之间做线性变换，并输出与残差分支匹配的宽度。

Concatenation keeps token positions separate: each token collects its own eight head outputs into 512 dimensions. W_O linearly mixes those features and returns the width required by the residual branch.

### 3.3 位置信息如何进入计算：RoPE

只比较内容向量还不足以表达顺序。RoPE 在 Q、K 做点积之前，对它们施加与位置相关的旋转，使匹配分数能够反映相对位置；V 不参与这个旋转。

本例每头全部 64 维参与 RoPE，组成 32 对分量。每对都同时使用 sin、cos，因此每个位置有 32 个旋转角。旋转改变数值，保留 shape；具体模型也可以只对部分分量使用 RoPE。

Content vectors alone do not encode order. Before QK dot products, RoPE applies position-dependent rotations to Q and K so that matching can reflect relative position. V does not undergo this rotation.

In this example, all 64 components per head form 32 pairs. Each pair uses both sin and cos, giving 32 rotation angles per position. RoPE changes values while preserving shape; particular models may apply it to only part of each head.

[RoPE paper, §3.4](https://arxiv.org/html/2104.09864v5#S3.SS4)

### 3.4 加上 batch 和 Padding

若三个样本分别有 5、3、4 个 Token，Padding 后统一为 5 个位置。batch 轴表示不同样本，head 轴表示不同投影子空间，两者各自独立计算。Padding 的 Key 位置由 Mask 屏蔽；Mask 改变可见性，保留张量的 shape。

主线统一使用 batch-first 布局。sequence-first 以及合并 QKV 的打包方式放在附录中。

For samples containing 5, 3, and 4 tokens, padding gives five positions per sample. Batch indexes independent samples, while head indexes independently projected subspaces. Padding masks block padded key positions without changing tensor shapes.

The main example uses batch-first layout throughout. Sequence-first layout and fused QKV packing are covered in the appendix.

| 张量 / Tensor | 完整 shape / Full shape |
| --- | --- |
| 输入 X / Input X | `[3, 5, 512]` |
| Q、K、V / Q, K, V | 各为 / Each `[3, 5, 8, 64]` |
| Attention 分数、权重 / Scores and weights | `[3, 8, 5, 5]` |
| Attention 输出 / Per-head outputs | `[3, 5, 8, 64]` |
| 拼接与 W_O 后的输出 / Output after concatenation and W_O | `[3, 5, 512]` |

![Transformer QKV：从输入到注意力输出](images/qkv-attention-overview.jpg)

## 4. 推理：Prefill、Decode 与 KV Cache

### 4.1 模型如何逐个生成 Token？

输入 `A B C D` 后，模型先对整个 Prompt 做一次 Prefill，最后一个位置的输出用于预测 `E`。随后把 `E` 作为新输入，通过全部层，预测 `F`；再输入 `F`，预测 `G`。

**一个 Decode Step 是新 Token 经过整个模型的一次前向传播。** Layer 表示网络深度，Step 表示生成时间步。

Given prompt `A B C D`, Prefill processes the prompt, and its final position predicts `E`. Feeding `E` through all layers predicts `F`; feeding `F` predicts `G`.

**One Decode Step is a forward pass of the new token through the entire model.** Layer refers to network depth; Step refers to generation time.

| 阶段 / Phase | 当前输入 / Input | 每层缓存更新 / Per-layer cache update | 预测 / Prediction |
| --- | --- | --- | --- |
| Prefill | `A B C D` | 保存 / Store KV for `A B C D` | `E` |
| Decode 1 | `E` | 读取历史 KV，追加 E / Read history and append E | `F` |
| Decode 2 | `F` | 读取历史 KV，追加 F / Read history and append F | `G` |

### 4.2 为什么能缓存 KV，为什么通常不缓存 Q？

标准 causal Transformer 中，历史位置不能读取后面新增的 Token。在模型权重和计算条件一致时，历史位置的表示及其 K/V 保持不变，因此可以复用。

每层只为新 Token 计算 Q/K/V：新 Q 读取该层的历史 K/V 和当前 Token 的 K/V，新的 K/V 追加进缓存。历史 Q 已完成对应位置的输出计算，后续生成不再需要它。**缓存按层保存**，因为各层的输入与投影参数不同。

In a standard causal Transformer, past positions cannot read newly appended tokens. With unchanged model weights and computation settings, their representations and K/V remain valid for reuse.

Each layer computes Q/K/V only for the new token. Its new Q reads that layer's past and current K/V, and its new K/V are appended to the cache. Past queries have already produced their position outputs and are not needed for later generation. **Caches are stored per layer**, since layers have different inputs and projection parameters.

若当前共有 t 个可见位置，单 head 的一次 Decode 读取就是：

With t visible positions, one head's Decode attention has these shapes:

```math
[1,d_k]\times[d_k,t]\rightarrow[1,t],\qquad
[1,t]\times[t,d_v]\rightarrow[1,d_v]
```

KV Cache 避免重新运行历史 Token；Full Attention 中，新 Q 仍然需要读取所有可见 KV，因此上下文越长，读取缓存的开销仍然越大。

KV caching avoids rerunning past tokens. With full Attention, each new query still reads all visible KV, so cache-reading cost continues to grow with context length. [Hugging Face: How caching works](https://huggingface.co/docs/transformers/main/cache_explanation)

### 4.3 Prefix Caching：跨请求复用相同前缀

若多个请求的公共指令和 Doc 完全相同，第一次请求可以保存前缀的逐层 KV。后续命中时，加载这些 KV，只处理新 Query；回答生成继续使用普通 Decode。

命中的基础是前缀不依赖后面的 Query。复用需要前缀 Token、前序上下文、模型权重、位置编码和 Mask 等计算条件一致；语义相近或仅中间一段文字相同，都不足以直接复用。vLLM 按块管理前缀 KV，块的标识包含前序块信息、本块 Token 和相关配置。

If requests have an identical instruction-and-document prefix, the first request can save its per-layer KV. Later cache hits load those KV and process only the new query, followed by ordinary Decode when generating an answer.

Reuse works because the prefix does not depend on the following query. A hit requires matching prefix tokens, preceding context, model weights, positions, masks, and other computation settings. Semantic similarity or an identical middle passage is insufficient. vLLM identifies prefix-cache blocks using preceding block information, current tokens, and relevant configuration.

```text
Request 1: [Shared instructions + Doc] + Query A
Request 2: [Shared instructions + Doc] + Query B
Request 3: [Shared instructions + Doc] + Query C
```

**Prefix Caching 主要节省 Prefill。** 4096 Token 的 Doc 仍占 4096 个 KV 位置，后续 Query 和生成 Token 仍要对它们做 Attention。

**Prefix Caching mainly saves Prefill.** A 4096-token document still occupies 4096 KV positions, which later query and generated tokens continue to attend to. [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) · [Prefix Caching implementation](https://docs.vllm.ai/en/stable/design/prefix_caching/)

## 5. 真实模型：MHA、GQA 与 MLA

### 5.1 hidden_size 不一定等于 heads × head_dim

`hidden_size` 是层间 Hidden States 的宽度；head 数和每头维度描述 Attention 内部的表示。本例选择 `512 = 8 × 64`，但投影矩阵的输出维度可以不同于输入维度。

Q 的总维度是 `Q head 数 × d_k`。每个 Q head 得到一个 `d_v` 维输出，因此拼接维度是 `Q head 数 × d_v`，随后 W_O 将它映射回 `hidden_size`。MHA 中 Q/K/V head 数相同；GQA 中多个 Q heads 共享一个 KV head。

`hidden_size` is the width of Hidden States passed between layers. Head counts and dimensions describe representations inside Attention. This example chooses `512 = 8 × 64`, but learned projections may have different input and output widths.

The total query width is `query head count × d_k`. Each query head produces a `d_v`-dimensional output, so concatenation has width `query head count × d_v`; W_O maps it back to `hidden_size`. MHA uses equal Q/K/V head counts, while GQA shares a KV head across multiple query heads.

### 5.2 Qwen3-30B-A3B：共享 KV heads 的 GQA

Qwen3-30B-A3B 的 `hidden_size=2048`，Q 有 32 个 head，K/V 各有 4 个 head，每头为 128 维。因此一个 Token 投影后，Q 是 4096 维，K、V 各是 512 维。每 8 个 Q heads 共享一组 K/V，共 4 组；不同 Q heads 仍分别计算各自的权重。

Qwen3-30B-A3B has `hidden_size=2048`, 32 query heads, 4 key/value heads, and 128 dimensions per head. Each token projects to a 4096-dimensional Q and two 512-dimensional K/V vectors. Eight query heads share each KV head, forming four groups, while retaining separate query-dependent attention weights. [Official configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)

| 阶段 / Stage，省略 batch / Batch omitted | Shape |
| --- | --- |
| 5 个 Token 的输入 X / Input X | `[5, 2048]` |
| Q 投影并分头 / Project and reshape Q | `[5, 4096] → [5, 32, 128]` |
| K、V 投影并分头 / Project and reshape K and V | 各为 / Each `[5, 512] → [5, 4, 128]` |
| 各 Q head 的权重 / Weights per query head | `[32, 5, 5]` |
| 拼接各 head 输出 / Concatenate head outputs | `[5, 4096]` |
| W_O 输出 / W_O output | `[5, 2048]` |

这里 `2048/32=64`，实际每头却是 128 维，因此不能仅凭 hidden size 和 head 数推断每头维度。Qwen3 还包含 Q/K 归一化、RoPE 和 Mask 等操作。名字中的 A3B 指约 3B 激活参数，与 QKV 维度是不同概念。

Although `2048/32=64`, each head actually has 128 dimensions. Hidden size and head count alone therefore do not determine head dimension. Qwen3 also uses Q/K normalization, RoPE, and masking. A3B refers to roughly 3B active parameters, a separate concept from QKV dimensions.

### 5.3 DeepSeek-V3：压缩 KV 表示的 MLA

DeepSeek-V3 的 `hidden_size=7168`，有 128 个 Q head。每个 Q/K head 包含 128 维内容分量和 64 维 RoPE 分量，共 192 维；V 每头为 128 维。

Q 路径先经过 1536 维低秩中间表示，再生成各个 Q head。KV 路径把内容压到一个共享的 512 维潜在向量，并另外产生一个共享的 64 维位置 Key。**紧凑缓存只需保存这 512 + 64 = 576 个数**，不必缓存所有 head 展开后的完整 K/V。

DeepSeek-V3 uses `hidden_size=7168` and 128 query heads. Each Q/K head combines 128 content dimensions with 64 RoPE dimensions, totaling 192; V has 128 dimensions per head.

The query path uses a 1536-dimensional low-rank intermediate. The KV path produces a shared 512-dimensional content latent and a shared 64-dimensional positional key. **Compact caching stores these 512 + 64 = 576 values**, rather than all expanded K/V heads. [Official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) · [Technical report, §2.1.1](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)

```text
One token: hidden [7168]
  Q path:  [7168] → latent [1536] → Q [128, 192]
  KV path: [7168] → KV latent [512] + RoPE key [64]
                     ↓ conceptual expansion
                   K content [128, 128], V [128, 128]
                   K head = content [128] + shared RoPE key [64]

Per-query-head output: [128]
Concat: [128 × 128] = [16384] → W_O → hidden [7168]
```

这里 Q/K 的 192 维负责匹配，V 的 128 维决定每个 head 的输出宽度；因此拼接结果为 16384 维。优化实现可以吸收部分投影，避免显式展开全部 K/V。

Here, the 192-dimensional Q/K heads perform matching, while the 128-dimensional V heads determine output width. Concatenation therefore produces 16384 dimensions. Optimized implementations can absorb projections and avoid explicitly expanding all K/V.

低秩投影、参数量、Decoupled RoPE 和矩阵吸收将在[第 6 节](#6-深入-mla低秩投影与压缩缓存计算)继续展开。

[Section 6](#6-深入-mla低秩投影与压缩缓存计算) develops the low-rank projections, parameter counts, Decoupled RoPE, and weight absorption.

### 5.4 每层、每个 Token 缓存多少数？

缓存量由实际保存的表示决定。下面计算逻辑元素数，不含量化、Padding、分布式复制和缓存管理等开销；实际字节数还取决于数据类型。

Cache size depends on the representation actually stored. The following counts logical elements, excluding quantization, padding, distributed replication, and management overhead. Actual bytes also depend on data type.

| 结构 / Architecture | 缓存表示 / Cached representation | 元素数 / Elements |
| --- | --- | --- |
| 本文 MHA / This note's MHA | K、V 各 8 个 head，每头 64 维 / 8 K and V heads of 64 dimensions | `2 × 8 × 64 = 1024` |
| Qwen3-30B-A3B GQA | K、V 各 4 个 head，每头 128 维 / 4 K and V heads of 128 dimensions | `2 × 4 × 128 = 1024` |
| DeepSeek-V3 紧凑 MLA / Compact MLA | KV latent + 位置 Key / KV latent and positional key | `512 + 64 = 576` |

这些模型的架构规模不同。若固定 32 个 Q head、每头 128 维，标准 MHA 需要缓存 `2 × 32 × 128 = 8192` 个数；改为 4 个 KV head 的 GQA 后为 1024 个数，即 1/8。

These models have different architecture sizes. Holding 32 query heads and 128 dimensions per head fixed, standard MHA would cache 8192 values; GQA with four KV heads stores 1024, or one eighth as many.

## 6. 深入 MLA：低秩投影与压缩缓存计算

MLA（Multi-head Latent Attention）为每个 Token 保存一份共享的 KV 潜在表示，各个 head 通过自己的投影形成不同的 K/V。DeepSeek-V3 再将位置 Key 单独保存，使 Attention 能直接读取紧凑缓存。

MLA stores a shared KV latent for each token, from which head-specific projections produce distinct K/V representations. DeepSeek-V3 stores the positional key separately, allowing Attention to operate directly on the compact cache.

### 6.1 为什么需要压缩 KV Cache

#### 6.1.1 三种内存分别保存什么

生成新 Token 时，当前 Q 读取历史 K/V；当前 Q 完成这一层的计算后，通常不写入历史缓存。因此，Q 展开到很高维，与每个历史 Token 需要保存多少缓存，是两个不同的问题。

During generation, the current Q reads past K/V. Once it has completed the layer's computation, it is normally not retained in the history cache. The width of the current query and the storage required for each past token describe different resources.

| 资源 / Resource | 内容 / Contents | 与生成长度的关系 / Relation to generated length |
| --- | --- | --- |
| 模型权重 / Model weights | 每层的投影矩阵等参数 / Per-layer projection parameters | 固定模型的参数不随 Token 逐份复制 / Parameters are reused across tokens |
| 推理临时激活 / Temporary inference activations | 当前 Q、分数及计算工作区 / Current Q, scores, and workspace | 取决于 batch、序列和计算实现 / Depends on batch, sequence, and implementation |
| KV Cache | 各层历史位置的可复用表示 / Reusable per-layer history | 随保留的历史 Token 数累积 / Grows with retained token history |

#### 6.1.2 MLA 压缩的是每个历史位置的存储

若保留 T 个位置，每个位置每层需要 n 个数，batch 为 B、层数为 L，每个数占 s 字节，则逻辑缓存大小为 `B × L × T × n × s`。MLA 主要降低 n；不同层仍需要各自的缓存。

以下围绕 DeepSeek-V3 的单层、单个样本展开，采用行向量约定：`输出 = 输入 @ 权重`。t 表示当前 Token，j 表示可读取的 Key 位置，i 表示 head。

For T retained positions, n cached values per position per layer, batch B, L layers, and s bytes per value, logical cache size is `B × L × T × n × s`. MLA primarily reduces n; layers still require separate caches.

The following uses one DeepSeek-V3 layer and one sample, with row vectors: `output = input @ weight`. The indices t, j, and i denote the current token, a visible key position, and a head.

### 6.2 一个 Token 如何生成 Q 和 KV

#### 6.2.1 先固定各个维度的含义

输入 Hidden State 的宽度、低秩中间维度和每头维度分别描述不同对象。DeepSeek-V3 使用下面的配置；其中 1536 和 512 是架构选定的中间维度，不能只靠 hidden size 或 head 数唯一推导出来。

Hidden-state width, latent dimensions, and per-head dimensions describe different objects. DeepSeek-V3 uses the following configuration. Its latent widths of 1536 and 512 are architecture choices, not quantities uniquely determined by hidden size or head count. [Official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)

| 符号 / Symbol | 数值 / Value | 含义 / Meaning |
| --- | --- | --- |
| d_model | 7168 | 当前层输入宽度 / Layer input width |
| H | 128 | Q head 数 / Query head count |
| r_Q | 1536 | Q 的低秩中间维度 / Query latent width |
| r_KV | 512 | 共享 KV 潜在表示的宽度 / Shared KV latent width |
| d_C | 128 | 每头 Q/K 的内容维度 / Content dimensions per Q/K head |
| d_R | 64 | RoPE 分量的维度 / RoPE dimensions |
| d_V | 128 | 每头 V 及输出的维度 / Value and output dimensions per head |

#### 6.2.2 内容分支生成共享 KV Latent

当前 Token 的输入 `x_t: [7168]` 先经过降维投影和 RMSNorm，得到 `c_t^KV: [512]`。这是每个 Token 自己的潜在表示，之后所有 head 的内容 K 和 V 都由它生成。

The current token's input `x_t: [7168]` passes through a down-projection and RMSNorm to produce `c_t^KV: [512]`. Each token has its own latent, shared across all heads when producing content keys and values.

```math
c_t^{KV}=\mathrm{RMSNorm}(x_tW_{DKV}),\qquad
W_{DKV}\in\mathbb{R}^{7168\times512}
```

两个不同的升维投影分别生成内容 K 和 V。完整结果各有 `128 × 128 = 16384` 个数，可以拆成 128 个 head，每头 128 维。

Two different up-projections generate content K and V. Each complete result contains `128 × 128 = 16384` values, split into 128 heads of 128 dimensions.

```math
K_t^C=c_t^{KV}W_{UK},\qquad V_t=c_t^{KV}W_{UV},\qquad
W_{UK},W_{UV}\in\mathbb{R}^{512\times16384}
```

```text
x_t [7168] → down-projection + RMSNorm → c_t^KV [512]
                                          ├→ W_UK → K content [128, 128]
                                          └→ W_UV → V         [128, 128]
```

这里的“升维”是生成各 head 所需的特征，不是把原始 7168 维输入无损还原。模型在训练时就使用这套低秩结构，学习怎样把有用信息放进共享表示。

Up-projection generates the features needed by the heads; it does not losslessly reconstruct the original 7168-dimensional input. Training learns to place useful information into this shared latent within the low-rank architecture.

#### 6.2.3 共享 Latent 如何产生不同的 head

把 W_UK 按输出列拆开，第 i 个 head 使用自己的子矩阵 `W_UK^(i): [512, 128]`，V 同理。各个 head 的参数不同，所以共享输入可以产生不同结果。

例如，把维度缩小为 2：同一个 `c = [2, 3]`，经过两组不同参数，可以得到 `[2, 3]` 和 `[5, -1]`。

Split W_UK into output-column blocks. Head i uses its own `W_UK^(i): [512, 128]`, and V works the same way. Different head parameters produce different results from the shared input.

For a two-dimensional example, the same `c = [2, 3]` can produce `[2, 3]` and `[5, -1]` through different projections.

```math
[2,3]
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
=[2,3],\qquad
[2,3]
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
=[5,-1]
```

**MLA 共享每个 Token 的低维表示，各个 head 保留不同的投影和 Attention 权重。** 投影矩阵是该层的模型参数，所有 Token 共用；缓存只需保存各 Token 的潜在表示，不需要为每个 Token 再存一份矩阵。

**MLA shares each token's latent while retaining head-specific projections and attention weights.** Projection matrices are layer parameters reused across tokens. The cache stores token-specific latents, without duplicating those matrices per token.

#### 6.2.4 位置分支单独生成 64 维 Key

另一个分支从 x_t 生成 64 维向量，再按位置 t 应用 RoPE，得到 `k_t^R: [64]`。这个位置 Key 在全部 128 个 head 之间共享；各 head 的完整 Key 由自己的 128 维内容部分与这个 64 维位置部分拼接。

A separate branch projects x_t to 64 dimensions and applies RoPE at position t, producing `k_t^R: [64]`. This positional key is shared across all 128 heads. Each complete key concatenates its head-specific 128-dimensional content with this shared 64-dimensional position component.

```math
k_t^R=\mathrm{RoPE}_t(x_tW_{KR}),\qquad
W_{KR}\in\mathbb{R}^{7168\times64},\qquad
k_{t,i}=[k_{t,i}^C;k_t^R]\in\mathbb{R}^{192}
```

因此，一个 Token 写入紧凑缓存的内容是 **`c_t^KV [512] + k_t^R [64]`**。64 维位置 Key 只保存一份，内容 K/V 通过共享 latent 表示。

One token therefore contributes **`c_t^KV [512] + k_t^R [64]`** to the compact cache. The positional key is stored once, and the shared latent represents content K/V. [DeepSeek-V3 technical report, §2.1.1](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)

#### 6.2.5 Q 分支生成当前 Query

Q 使用另一组投影：`7168 → 1536 → 24576`，然后拆成 128 个 192 维的 head。每头再拆成 128 维内容 Query 和 64 维位置 Query，仅对位置部分应用 RoPE。Q 的位置分量按 head 生成；K 的位置分量则在 head 之间共享。

Q 的 1536 维 latent 和最终 24576 维 Query 都用于当前计算，通常不写入历史 KV Cache。

The query path uses `7168 → 1536 → 24576`, then splits into 128 heads of 192 dimensions. Each head has 128 content dimensions and 64 positional dimensions; only the latter receive RoPE. Query position components are head-specific, while the key position component is shared across heads.

Both the 1536-dimensional query latent and the final 24576-dimensional query serve the current computation and are normally not retained in the history KV cache.

```math
c_t^Q=\mathrm{RMSNorm}(x_tW_{DQ}),\qquad
Q_t=\mathrm{reshape}(c_t^QW_{UQ},[128,192])
```

```math
W_{DQ}\in\mathbb{R}^{7168\times1536},\qquad
W_{UQ}\in\mathbb{R}^{1536\times24576},\qquad
q_{t,i}=[q_{t,i}^C;q_{t,i}^R]
```

### 6.3 展开后如何计算 Attention

#### 6.3.1 分数由内容匹配与位置匹配组成

先按概念展开 K/V。对于当前 Token t 的第 i 个 head，分别计算内容分数和位置分数，相加后缩放，再对可见位置 j 做 softmax。当前 Token 自己的 K/V 也包含在可读取位置中。

Consider the conceptually expanded K/V first. For head i of the current token t, add content and positional scores, apply scaling, and normalize over visible key positions j. Visible positions include the current token's own K/V.

```math
s_{t,j,i}=\alpha\left[q_{t,i}^C(k_{j,i}^C)^T+q_{t,i}^R(k_j^R)^T\right]+M_{t,j},\qquad
a_{t,j,i}=\mathrm{softmax}_{j}(s_{t,j,i})
```

基础缩放为 `α = 1/√192`，对应每头 Q/K 的 `128 + 64` 维；具体实现可按长上下文 RoPE 方案调整该系数。Mask 在可见位置为 0，不可见位置为负无穷。

The base scale is `α = 1/√192`, matching the `128 + 64` dimensions per Q/K head. Implementations may adjust it for long-context RoPE scaling. The mask is zero at visible positions and negative infinity at blocked positions.

#### 6.3.2 输出宽度由 V 决定

每个 head 用自己的权重组合 128 维的 V，因此输出也是 128 维。128 个 head 拼接成 16384 维，再通过 W_O 映射回 7168 维，供残差连接和后续层使用。

Each head combines 128-dimensional values using its own weights, producing a 128-dimensional output. Concatenating 128 heads yields 16384 dimensions; W_O maps that back to 7168 for the residual connection and subsequent layers.

```math
o_{t,i}=\sum_{j\leq t}a_{t,j,i}v_{j,i}\in\mathbb{R}^{128},\qquad
y_t=[o_{t,1};\ldots;o_{t,128}]W_O,\qquad
W_O\in\mathbb{R}^{16384\times7168}
```

`24576 = 128 × 192` 是 Q/K 的展开宽度，`16384 = 128 × 128` 是各 head 输出的拼接宽度，`576 = 512 + 64` 是每个历史位置的紧凑缓存宽度。

The widths refer to distinct objects: 24576 for expanded Q/K, 16384 for concatenated head outputs, and 576 for each cached history position.

#### 6.3.3 缓存量随上下文如何增长

对照一种朴素实现：完整保存每个 head 的 192 维 K 和 128 维 V。这种展开缓存每个位置为 `128 × (192 + 128) = 40960` 个数，也重复保存了各 head 中共享的位置 Key。MLA 紧凑缓存每个位置为 576 个数，约为该基线的 1/71.1。

下表均为**单层、单样本、BF16 每个数 2 字节**的逻辑缓存，不含模型权重和计算工作区。

Use a naive baseline that caches each head's full 192-dimensional K and 128-dimensional V. It stores `128 × (192 + 128) = 40960` values per position, including repeated copies of the shared positional key. MLA stores 576, about one seventy-first as many.

The table gives logical cache sizes for **one layer, one sample, and BF16 at two bytes per value**, excluding model weights and workspace.

| 表示 / Representation | 每 Token / Per token | 1000 Token | 4096 Token |
| --- | --- | --- | --- |
| 完整展开 K/V / Fully expanded K/V | 80 KiB | 78.125 MiB | 320 MiB |
| MLA 紧凑缓存 / Compact MLA | 1.125 KiB | 约 / About 1.099 MiB | 4.5 MiB |

这个比例描述相同 head 尺寸下两种缓存表示的存储量；实际推理速度还取决于计算路径、访存、batch 和硬件。

This ratio compares storage representations at the same head dimensions. Inference speed also depends on computation strategy, memory access, batch size, and hardware.

### 6.4 为什么先降维再升维

#### 6.4.1 低秩瓶颈减少线性投影成本

直接从 m 维投影到 n 维需要 `m × n` 个线性权重。若先经过 r 维中间表示，两次投影共需 `m × r + r × n` 个权重；r 足够小时，参数量和这两层线性投影的乘加次数都更少。

输入、输出宽度决定能否对接下一步计算；中间宽度 r 则决定共享表示的容量。最终输出比输入宽，并不要求先做一次同样巨大的直接投影。

A direct projection from m to n dimensions has `m × n` linear weights. Factoring it through r dimensions uses `m × r + r × n`. With sufficiently small r, both weight count and the linear projections' multiply-accumulate count decrease.

Input and output widths determine compatibility with subsequent operations; r controls the capacity of the shared intermediate. A wide output does not require a single equally large direct projection.

```math
P_{direct}=mn,\qquad P_{latent}=r(m+n),\qquad
P_{latent}\lt P_{direct}\ \text{when}\ r\lt \frac{mn}{m+n}
```

#### 6.4.2 Q 与 KV 内容投影的参数量

Q 的直接投影宽度为 24576，低秩中间维度为 1536。内容 K 和 V 的输出宽度各为 16384，共享 512 维中间表示。下面只计线性权重；KV 两个方案都不计独立的位置 Key 分支，归一化参数和 bias 也不计入。

The direct query output width is 24576, with latent width 1536 in the factorized path. Content K and V each have output width 16384 and share a 512-dimensional latent. The comparison counts linear weights only; both KV alternatives exclude the positional-key branch, and normalization parameters and biases are omitted.

| 路径 / Path | 直接投影 / Direct projection | 低秩投影 / Low-rank projection | 减少 / Reduction |
| --- | --- | --- | --- |
| Q | `7168 × 24576 = 176,160,768` | `7168 × 1536 + 1536 × 24576 = 48,758,784` | 约 / About 72.3% |
| K 内容与 V / Content K and V | `7168 × (16384 + 16384) = 234,881,024` | `7168 × 512 + 512 × (16384 + 16384) = 20,447,232` | 约 / About 91.3% |

Q 路径的线性投影乘加次数同样约降到直接方案的 1/3.61。这个比较只覆盖投影运算；Attention 的整体成本还包含读取历史、匹配和加权求和。

The query projections likewise use about 27.7% of the direct path's multiply-accumulate operations. This concerns projection work; Attention also includes reading history, matching, and weighted aggregation.

#### 6.4.3 升维不会自动增加独立信息

例如 `[a, b]` 可以映射成 `[a, b, a+b, a-b, 2a, 3b]`。输出有 6 个数，但仍来自两个输入变量。MLA 的升维也在用共享特征构造各 head 的表示。

忽略中间的 RMSNorm 时，两次线性投影可写成一个有效矩阵，它的秩不超过中间维度 r。较小的 r 会约束表达能力，因此低秩结构不是任意直接投影的无损等价替换。实际 Q/KV latent 后有 RMSNorm，完整路径不能合并成一个固定线性矩阵，但仍经过低维瓶颈。

For example, `[a, b]` can map to `[a, b, a+b, a-b, 2a, 3b]`. Six outputs still derive from two input variables. MLA similarly constructs head representations from shared features.

Ignoring intermediate RMSNorm, the two linear projections form an effective matrix whose rank is at most r. A smaller r constrains expressiveness, so low-rank projection is not a lossless equivalent of every direct projection. Actual Q/KV paths include RMSNorm after the latent; the full path is not one fixed linear matrix, although it still has a low-dimensional bottleneck.

```math
W_{effective}=W_DW_U,\qquad \mathrm{rank}(W_{effective})\leq r
```

#### 6.4.4 Q 与 KV 低秩设计的收益不同

两条路径都能减少上述投影参数与乘加次数。Q 的低秩表示还用于减少训练激活内存；KV 的中间表示则需要随历史位置保存，其宽度直接影响推理缓存预算。1536 和 512 是表达能力与资源开销之间的架构取舍。

Both paths reduce the projection parameters and multiply-accumulate counts above. Query compression also reduces training activation memory. The KV latent is retained for historical positions, so its width directly controls inference-cache storage. Choosing 1536 and 512 balances representation capacity and resource costs. [DeepSeek-V3, query compression and model dimensions](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)

| 收益 / Benefit | Q 低秩 / Query latent | KV 低秩 / KV latent |
| --- | --- | --- |
| 上述线性投影的参数和运算 / Projection weights and operations above | 减少 / Reduced | 减少 / Reduced |
| 历史 KV Cache / Historical KV cache | 不存 Q / Q is not cached | 保存压缩表示 / Store compressed representation |
| 中间宽度 / Latent width | 1536 | 512 |

### 6.5 如何直接使用压缩缓存

#### 6.5.1 将 K 的投影转到当前 Q 上

若每一步都把全部历史 latent 展开成 K/V，虽然持久缓存较小，仍会产生临时张量和重复投影。内容 K 的生成是线性的，可以利用矩阵乘法结合律，把历史 K 的展开转移到当前 Query 上。

Re-expanding every historical latent at every step would preserve compact storage but introduce temporary tensors and repeated projections. Content-key generation is linear, so associativity can move its projection from historical keys to the current query.

```math
k_{j,i}^C=c_j^{KV}W_{UK}^{(i)},\qquad W_{UK}^{(i)}\in\mathbb{R}^{512\times128}
```

```math
q_{t,i}^C(k_{j,i}^C)^T
=q_{t,i}^C(W_{UK}^{(i)})^T(c_j^{KV})^T
=\widetilde q_{t,i}(c_j^{KV})^T
```

```math
\widetilde q_{t,i}=q_{t,i}^C(W_{UK}^{(i)})^T\in\mathbb{R}^{512}
```

对每个 head，当前内容 Query 从 128 维变成 512 维，随后直接与缓存的 `[T, 512]` latent 匹配。历史内容 K 无需先展开为 `[T, 128]`。每个 head 的变换和匹配权重仍各不相同。

For each head, the current content query transforms from 128 to 512 dimensions and matches the cached `[T, 512]` latents directly. Historical content keys need not first expand to `[T, 128]`. Heads retain distinct query transformations and attention weights.

#### 6.5.2 先聚合 Latent 再做 V 投影

V 也由 latent 线性生成，因此可先对 latent 加权求和，得到当前 head 的 512 维聚合表示，再投影成 128 维输出。这样就无需为所有历史位置显式生成完整 V。

V is also a linear projection of the latent. First aggregate latents into a 512-dimensional representation for the current head, then project it into the 128-dimensional output. This avoids explicitly producing V for every history position.

```math
\begin{aligned}
o_{t,i}
&=\sum_j a_{t,j,i}\left(c_j^{KV}W_{UV}^{(i)}\right)\\
&=\left(\sum_j a_{t,j,i}c_j^{KV}\right)W_{UV}^{(i)}
=z_{t,i}W_{UV}^{(i)},\qquad z_{t,i}\in\mathbb{R}^{512}
\end{aligned}
```

V 投影还可以与 W_O 的对应分块组合，形成输出侧的权重吸收。实际实现也可以先得到各 head 的 128 维输出，再调用 W_O；两者的数学重排不要求采用同一种 kernel。

The value projection can also combine with its corresponding W_O block for output-side weight absorption. Implementations may instead produce each 128-dimensional head output before applying W_O. The algebra permits different kernel strategies.

#### 6.5.3 为什么需要 Decoupled RoPE

上述重排依赖 W_UK 是位置无关的固定矩阵。如果把各历史位置的 RoPE 旋转混进内容 K 路径，旋转会随 j 改变，无法统一吸收到只由当前 Query 决定的一次变换中。

MLA 因此将内容与位置分开：内容分数直接读取 512 维 latent；位置分数由当前各 head 的 64 维 RoPE Query 与缓存的共享 RoPE Key 单独计算。两部分相加后，沿用原来的分数缩放和 softmax。

The rearrangement relies on a fixed, position-independent W_UK. If historical RoPE rotations were inserted into the content-key path, those rotations would vary with j and prevent a single transformation determined only by the current query.

Decoupled RoPE separates the two paths: content scores read 512-dimensional latents directly, while positional scores use each current head's 64-dimensional RoPE query and the shared cached positional key. The two scores are added before the original scaling and softmax. [DeepSeek-V2, Decoupled RoPE](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS3)

```math
s_{t,j,i}=\alpha\left[\widetilde q_{t,i}(c_j^{KV})^T+q_{t,i}^R(k_j^R)^T\right]+M_{t,j}
```

变换后内容 Query 是 512 维，位置分量是 64 维，但这只是等价重排；缩放系数仍由原始 Q/K 的 192 维计算和模型配置决定，不能改用 `1/√576`。

The transformed query uses 512 content and 64 positional dimensions, but this is an equivalent rearrangement. Scaling remains tied to the original 192-dimensional Q/K computation and model configuration; it must not become `1/√576`.

#### 6.5.4 数学等价与实现成本

展开计算与上述重排在精确算术下等价，浮点运算顺序可能造成小的数值差异。重排减少完整历史 K/V 的生成和存储需求，但变换后的内容匹配维度也从 128 变成 512；Prefill、Decode、batch 大小和硬件会影响实际选择。

DeepSeek 的参考实现分别提供 naive 与 absorb 路径：前者缓存展开 K/V，后者缓存归一化后的 KV latent 和位置 Key，并直接完成内容匹配和 latent 聚合。

The expanded and rearranged computations are equivalent in exact arithmetic; floating-point order may introduce small differences. Rearrangement avoids generating and storing full historical K/V, while the transformed content matching dimension increases from 128 to 512. Prefill, Decode, batch size, and hardware influence implementation choices.

DeepSeek's reference implementation provides naive and absorb paths. The former caches expanded K/V; the latter caches normalized KV latents and positional keys and computes scores and latent aggregation directly. [Official inference implementation, MLA](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)

单步 Decode 的完整运算顺序见[附录 A.3](#a3-mla-单步-decode-伪代码)。

See [Appendix A.3](#a3-mla-单步-decode-伪代码) for the complete single-step Decode computation.

### 6.6 与 Qwen GQA 的异同

GQA 通过减少 KV head 数量来共享 K/V；同组 Q heads 的 Query 不同，Attention 权重和输出仍可不同。Qwen 的 `512 = 4 × 128` 是分组后的 K 或 V，直接用于 Attention，不经过 MLA 式的 latent 升维。MLA 共享的是产生内容 K/V 的 latent，各个 head 使用不同的投影；DeepSeek 的 512 维 latent 同时生成内容 K 和 V，另加一份 64 维位置 Key。

GQA reduces the number of KV heads, with queries, weights, and outputs remaining distinct within each group. Qwen's `512 = 4 × 128` is the grouped K or V used directly by Attention, without MLA-style latent up-projection. MLA shares the latent that produces content K/V while retaining head-specific projections. DeepSeek's 512-dimensional latent generates both content K and V, accompanied by one 64-dimensional positional key.

| 项目 / Property | Qwen3-30B-A3B GQA | DeepSeek-V3 MLA |
| --- | --- | --- |
| Hidden size | 2048 | 7168 |
| Q 路径 / Query path | `2048 → 4096 = 32 × 128` | `7168 → 1536 → 24576 = 128 × 192` |
| 512 的含义 / Meaning of 512 | 4 个 K heads 或 4 个 V heads / 4 K or V heads | 共享 KV latent / Shared KV latent |
| 各 head 的内容 K/V / Content K/V across heads | 同组 Q heads 使用相同 KV / Query groups share KV | 各 head 使用不同投影 / Head-specific projections |
| 每层每 Token 紧凑缓存 / Compact cache per layer per token | `512 + 512 = 1024` 个数 / Values | `512 + 64 = 576` 个数 / Values |
| Q 是否先低秩压缩 / Low-rank query path | 否，直接投影 / No, direct projection | 是，中间 1536 维 / Yes, via 1536 dimensions |

1024 与 576 分别是这两个模型每层、每 Token 的缓存元素量。模型规模和计算路径不同，这两个数本身不能用来比较整体推理速度。

The values 1024 and 576 describe cached elements per token per layer in these two models. Their model scales and computation paths differ, so these counts alone do not compare overall inference speed.

## 7. 应用：文档 Memory Cache

### 7.1 为什么还需要压缩文档？

Prefix Caching 可以复用文档的 Prefill，但在线 Query 仍需读取整篇文档的 KV。若同一 Doc 被多个 Query 反复使用，可以训练一组 Memory Token 承载压缩后的文档信息，再缓存这些位置的逐层 KV。

这里考虑 `Doc + 128 Memory Tokens + Query` 的相关性打分场景。Doc 为 4096 Token 时，在线读取的文档侧位置从 4096 缩减到 128，即原来的 1/32。这是位置数和 KV 元素量的缩减，不代表整个在线任务必然加速 32 倍。

Prefix Caching reuses document Prefill, but online queries still read the document's full KV. For repeated queries against the same document, a model can learn to compress document information into Memory Tokens and cache their per-layer KV.

Consider relevance scoring with `Doc + 128 Memory Tokens + Query`. For a 4096-token document, the document-side positions read online shrink to 128, or one thirty-second as many. This reduces positions and KV elements, rather than guaranteeing a 32-fold end-to-end speedup.

### 7.2 哪些位置可以相互读取？

Mask 的行表示读取方，列表示被读取方。Doc 与 Memory 双向交互；Query 通过 Memory 读取文档信息，并保留原有的 Query 内部 Mask。**Doc / Memory 在所有层都不依赖 Query**，这是离线复用的前提。

Rows of the mask index attending positions; columns index positions being read. Doc and Memory interact bidirectionally. Query accesses document information through Memory and retains its original internal mask. **Doc and Memory do not depend on Query at any layer**, enabling offline reuse.

| 读取方 ↓ / 被读取方 → | Doc | Memory | Query |
| --- | --- | --- | --- |
| Doc | 可见 / Visible | 可见 / Visible | 不可见 / Blocked |
| Memory | 可见 / Visible | 可见 / Visible | 不可见 / Blocked |
| Query | 不可见 / Blocked | 可见 / Visible | 原有 Query Mask / Original mask |

### 7.3 离线保存什么，在线计算什么？

离线为同一 Doc 运行一次 Doc + Memory，保存 Memory 在每层的 K/V。在线只计算 Query 的表征，在每一层读取对应层的 Memory KV；Query 完成前向后，交给相关性打分头。

如果打分头还读取离线侧的其他表征，也需要一并保存。Query 自身的 Attention 按原 Mask 计算。

Offline, process Doc and Memory once per document and save Memory K/V at every layer. Online, compute Query representations while reading the corresponding layer's Memory KV, then pass the result to the relevance scoring head.

Cache any additional offline representations required by that head as well. Query's own attention follows its original mask.

```text
Offline, once per Doc:
Doc + 128 Memory Tokens → All layers → Memory KV[1], ..., Memory KV[L]

Online, for each Query:
Query → Layer 1, read Memory KV[1]
      → Layer 2, read Memory KV[2]
      → ... → Layer L, read Memory KV[L]
      → Relevance scoring head
```

### 7.4 为什么必须按层缓存？

第一层 Memory 读取 Doc 后，输出 Hidden States 才包含文档信息；第二层 Memory 的 K/V 由这一输出产生。更深的层也有各自深度的表示，因此最后一层的 Hidden States 无法代替所有层的 KV。

在线加载的是离线算好的每层 KV。如果删除 Doc 后，把原始 Memory Token 从头重新运行，第一层就读不到 Doc，后续层的 KV 也随之改变。复用还需要保持模型、Mask 和位置编码条件一致；使用 RoPE 时，移除 Doc 后仍需保留 Memory 与 Query 的位置关系，或遵守训练时约定的位置方案。

Memory's first-layer output becomes document-conditioned after attending to Doc. Layer 2 derives its K/V from that output, and deeper layers have their own representations. Final-layer Hidden States cannot replace all layers' KV.

Online inference loads the per-layer KV computed offline. Rerunning the original Memory Tokens without Doc would change their first-layer outputs and subsequent KV. Reuse also requires consistent model, mask, and position settings. With RoPE, removing Doc must preserve the Memory–Query position relationship or follow the position scheme used during training.

### 7.5 压缩如何学到，缓存何时生成？

训练时，Doc、Memory、Query 按上述 Mask 一起完成前向和反向传播，学习文档压缩与相关性打分。模型权重固定后，再离线生成可跨 Query 复用的 Memory KV；模型权重更新后，对应缓存需要重建。

普通训练中的 K/V 是中间激活；推理缓存则用于跨 Step 或跨请求复用。**压缩来自训练，缓存复用来自计算依赖关系。**

During training, Doc, Memory, and Query use the specified mask for forward and backward passes, learning compression and relevance scoring. Once weights are fixed, Memory KV can be generated offline and reused across queries. Updating weights requires rebuilding the corresponding caches.

K/V in ordinary training are intermediate activations; inference caches support reuse across steps or requests. **Training learns the compression; computational dependencies make reuse possible.**

| 机制 / Mechanism | 复用范围 / Reuse scope | 文档位置是否压缩 / Document compression |
| --- | --- | --- |
| KV Cache | 同一次生成的不同 Step / Steps within one generation | 否 / No |
| Prefix Caching | 共享相同前缀的请求 / Requests sharing an identical prefix | 否 / No |
| 文档 Memory Cache / Document memory cache | 同一 Doc 的不同 Query / Queries against the same document | 是，通过训练 / Yes, learned during training |

## 附录：张量布局与实现

### A.1 Batch-first MHA 的核心计算

下面对应第 3 节的 MHA。输入已经分头，并在需要时对 Q、K 应用了 RoPE。`allowed` 是布尔 Mask，True 表示可见，可以广播到 `[B, H, Tq, Tk]`。第 2 节的点积、缩放、Mask、softmax 和加权求和都在同一段代码中完成。

This implements the MHA from Section 3. Inputs are already split into heads, with RoPE applied to Q and K when required. `allowed` is a boolean visibility mask broadcastable to `[B, H, Tq, Tk]`, where True means visible. The function performs dot products, scaling, masking, softmax, and weighted aggregation.

```python
import math
import torch


def attention(query, key, value, allowed=None):
    # query: [B, Tq, H, dk]
    # key:   [B, Tk, H, dk]
    # value: [B, Tk, H, dv]
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
    scores = scores / math.sqrt(query.shape[-1])
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # [B, H, Tq, Tk]
    out = torch.einsum("bhqk,bkhd->bqhd", weights, value)
    return out.flatten(start_dim=-2)        # [B, Tq, H * dv]

# W_O is applied after concatenation.
```

### A.2 sequence-first 与合并 QKV 投影

有些实现先将 `[batch, seq, hidden]` 转成 `[seq, batch, hidden]`。这只交换轴顺序，不改变每个 Token 的特征维度。合并 QKV 投影则一次生成三组向量，减少分别调用投影的操作。

下面选择每个 head 内依次存放 Q/K/V：`[Q₁, K₁, V₁, Q₂, K₂, V₂, ...]`。这是打包约定，拆分必须与其对应。

Some implementations transpose `[batch, seq, hidden]` into `[seq, batch, hidden]`. This changes axis order without changing token feature width. Fused QKV projection produces all three sets of vectors in one operation.

The following packing order stores Q/K/V within each head: `[Q₁, K₁, V₁, Q₂, K₂, V₂, ...]`. Splitting must match that convention.

```python
# batch_input: [3, 5, 512]
hidden_states = batch_input.transpose(0, 1).contiguous()
# [5, 3, 512]

qkv_projection = torch.nn.Linear(512, 3 * 512)
mixed = qkv_projection(hidden_states)
# [5, 3, 1536]

groups = mixed.view(5, 3, 24, 64)
packed = groups.view(5, 3, 8, 192)
query, key, value = packed.split(64, dim=-1)
# Each: [5, 3, 8, 64] = [seq, batch, heads, head_dim]

# Convert each tensor to batch-first before calling attention() above:
query = query.transpose(0, 1)  # [3, 5, 8, 64]
key = key.transpose(0, 1)
value = value.transpose(0, 1)
```

| 数字 / Number | 含义 / Meaning |
| --- | --- |
| `1536 = 3 × 8 × 64` | Q/K/V 的总投影宽度 / Total fused QKV width |
| `24 = 3 × 8` | 每组 64 维的打包分组数，实际仍是 8 个 head / Packed groups of 64 values; still 8 heads |
| `192 = 3 × 64` | 一个 head 内 Q/K/V 的合并宽度 / Packed Q/K/V width within one head |
| `64 = 512/8` | 本例每头维度 / Per-head dimension in this example |
| `32 = 64/2` | 本例每头完整 RoPE 的分量对数 / Component pairs for full-head RoPE |

[Megatron-LM: Attention implementation](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py)

### A.3 MLA 单步 Decode 伪代码

下列伪代码按一个样本、一个新增 Token 展示 absorb 路径。`UK[i]`、`UV[i]` 为各 head 的投影子矩阵，shape 都是 `[512, 128]`。`q_norm` 与 `kv_norm` 是参数独立的 RMSNorm。cache 按层独立维护；写入当前位置后，T 包含历史位置与当前位置。RoPE 使用真实位置，scale 来自模型配置。

This pseudocode illustrates the absorb path for one sample and one new token. `UK[i]` and `UV[i]` are head-specific `[512, 128]` projection blocks. The RMSNorms `q_norm` and `kv_norm` have separate parameters. Each layer has its own cache. After appending the current position, T includes both past and current positions. RoPE uses actual positions, and scale comes from model configuration.

```python
# x_t: [7168]; learned matrices use row-vector convention.
q_latent = q_norm(x_t @ W_DQ)                   # [1536]
q = (q_latent @ W_UQ).reshape(128, 192)
q_content = q[:, :128]                         # [128, 128]
q_rope = rope(q[:, 128:], position)            # [128, 64]

c = kv_norm(x_t @ W_DKV)                       # [512]
k_rope = rope(x_t @ W_KR, position)            # [64]
cache.append(c, k_rope)
C = cache.content                             # [T, 512]
KR = cache.position_keys                      # [T, 64]

head_outputs = []
for i in range(128):
    q_effective = q_content[i] @ UK[i].T        # [512]
    scores = scale * (C @ q_effective + KR @ q_rope[i])
    weights = softmax(scores)                 # [T], all cached positions visible
    z = weights @ C                           # [512]
    head_outputs.append(z @ UV[i])            # [128]

y_t = concatenate(head_outputs) @ W_O         # [7168]
```
