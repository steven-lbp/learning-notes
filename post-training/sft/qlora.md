# QLoRA
QLoRA (Quantized Low-Rank Adapter for Efficient Finetuning) is an advanced fine-tuning technique that enables efficient low-cost adaptation of large language models (LLMs) on consumer GPUs while maintaining high performance.
- 4-bit quantization (NF4 Quantization + Double Quantization)
- LoRA (Low-Rank Adapters) for efficient parameter fine-tuning
- Paged Optimizers to minimize memory footprint

### 4-bit quantization
#### NF4 Quantization(Non-Uniform)
1. The large weight matrix is divided into multiple smaller groups (blocks)
2. Each group has its own scaling factor $S_{i}$, ensuring
$$
W_{i} \approx S_{i} \cdot Q_{W}
$$
3. This allows each group to adapt to NF4’s predefined quantization table, preventing severe precision loss.

Each group computes its own max value $S_{i}$, which serves as the scaling factor.

Why:

If we apply traditional linear quantization:

Small values lose precision (e.g., 0.02 and 0.05 might be mapped to the same value).
Large values get over-compressed (e.g., 3.3 might be mapped to 2.0, leading to computational errors).

#### Double Quantization
Scaling factor ```S1, S2 and S3``` are 16 bit (FP16) values themselves, which still occupy a large amount of storage.

Therefore, they are  quantized into 8-bit representation and use a global FP16 scaling factor $S_{g}$ to ensure precision.

### LoRA
$$
W' = W + \Delta W
$$
$$
\Delta W = AB
$$
eg. W: 10000 \* 10000 A: 10000 \* 6 B: 6 * 10000

### Paged Optimizers
