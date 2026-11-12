# QLoRA

QLoRA 以 4-bit 保存冻结的主干权重，通过 LoRA 训练少量 Adapter 参数，降低大模型微调的显存开销。三个关键组件是 NF4、Double Quantization 和 Paged Optimizers。

QLoRA stores frozen backbone weights in 4-bit form and trains a small set of LoRA Adapter parameters, reducing fine-tuning memory requirements. Its key components are NF4, Double Quantization, and Paged Optimizers.

## 4-bit quantization

### NF4

把大权重矩阵分成多个 Block，每个 Block 使用自己的缩放因子，再映射到 NF4 的非均匀量化值。NF4 的取值适合近似正态分布的权重。

每个 Block 可用最大绝对值作为缩放因子。相较于统一的线性量化间隔，NF4 在更常见的数值区域保留更合适的分辨率，减小量化误差。

Split a large weight matrix into blocks, each with its own scale, and map values to NF4's nonuniform quantization levels. NF4 is designed for approximately normally distributed weights.

A block's maximum absolute value can provide its scale. Compared with uniform linear quantization, NF4 allocates resolution to more likely values to reduce quantization error.

$$
W_{i} \approx S_{i} \cdot Q_{W}
$$

### Double Quantization

缩放因子本身也占空间。Double Quantization 对这些量化常数再量化一次，进一步降低平均存储开销。原 QLoRA 方案将第一层的 FP32 量化常数进一步量化为 8-bit，并使用第二层缩放信息恢复。

Quantization scales also consume memory. Double Quantization quantizes those constants again to lower average storage overhead. The original QLoRA approach quantizes the first-level FP32 constants into 8-bit values with second-level scaling information.

## LoRA

冻结原始权重 W，只训练低秩增量。原始矩阵若为 10000 × 10000，可用 10000 × 6 的 A 和 6 × 10000 的 B 表示低秩更新，大幅减少可训练参数。

Freeze the original weights W and train a low-rank update. For a 10000 × 10000 weight matrix, a 10000 × 6 matrix A and a 6 × 10000 matrix B represent the update with far fewer trainable parameters.

$$
W' = W + \Delta W
$$

$$
\Delta W = AB
$$

## Paged Optimizers

Paged Optimizers 管理训练中的显存峰值，减少瞬时内存需求造成的 OOM。

Paged Optimizers manage memory spikes during training, reducing out-of-memory failures caused by transient demand.

[QLoRA paper](https://arxiv.org/abs/2305.14314)
