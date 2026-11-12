# Megatron-LM

## Introduction

Megatron-LM 是 NVIDIA 开发的大模型训练框架，通过组合不同并行方式，在多 GPU、多机器上训练大型 Transformer。

Megatron-LM is NVIDIA's framework for training large Transformer models across multiple GPUs and machines by combining parallelism strategies.

| 方式 | 划分对象 |
| --- | --- |
| Tensor Parallelism（TP） | 单层中的矩阵计算 |
| Pipeline Parallelism（PP） | 不同层组成的 Stage |
| Sequence Parallelism（SP） | 部分激活的 sequence 维度，例如 LayerNorm、Dropout 的输入 |
| Data Parallelism（DP） | 不同 Batch 数据，各副本协同更新同一组参数 |

| Strategy | What is partitioned |
| --- | --- |
| Tensor Parallelism (TP) | Matrix computations within a layer |
| Pipeline Parallelism (PP) | Stages containing different layers |
| Sequence Parallelism (SP) | The sequence dimension of selected activations, such as LayerNorm and Dropout inputs |
| Data Parallelism (DP) | Batch data across model replicas that coordinate parameter updates |

[Megatron Core parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

## Megatron-LM and DeepSpeed

Megatron-LM 的 TP / PP 将模型计算分布到多个设备；DeepSpeed 的 ZeRO 将 Optimizer States、Gradients、Parameters 按阶段分片，降低模型状态的冗余存储。

DeepSpeed-Megatron 结合这两类能力：用 TP / PP 分配模型计算，用 ZeRO 和 Offload 降低显存占用。

Megatron-LM distributes model computation with TP and PP. DeepSpeed's ZeRO progressively shards Optimizer States, Gradients, and Parameters to reduce redundant model-state storage.

DeepSpeed-Megatron combines these capabilities: TP / PP distribute computation, while ZeRO and Offload reduce device memory requirements.
