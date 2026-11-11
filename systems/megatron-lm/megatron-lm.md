# Megatron-LM
## Introduction
A library developed by NVIDIA specifically for training **Super** Large Language Models (LLMs).
- ``Tensor Parallel, TP``: Splitting the Matrix of Transformer Layers
- ``Pipeline Parallel, PP``: Different Transformer layers
- ``Sequence Parallel, SP``: Splitting QKV
- ``Data Parallel, DP``: Traditional data parallelism

## Megatron-LM and Deepspeed
Megatron LM provides model parallelism (MP) for **splitting Transformer layers**, allowing super large models to be trained on multiple GPUs/nodes.

DeepSpeed provides ZeRO optimization (Zero Redundancy Optimizer) for splitting optimizer states, **reducing memory usage**, and improving computational efficiency.

Microsoft and NVIDIA have jointly developed DeepSpeed-Megatron, combining the advantages of both: 

- Megatron LM is responsible for tensor parallel (TP)+pipeline parallel (PP) → improving training efficiency

- DeepSpeed is responsible for ZeRO-3+Offload → reducing video memory usage