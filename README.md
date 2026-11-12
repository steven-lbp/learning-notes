# Learning Notes

个人学习笔记与代码练习，按基础原理、后训练、应用方法、训练工程、模型案例和算法练习分类。

Personal learning notes and coding exercises, organized by foundations, post-training, applications, training systems, model studies, and algorithm practice.

## 学习路线

- **模型原理与后训练：** 梯度与反向传播 → Transformer → QLoRA → 策略梯度与 PPO / GRPO → OPD → DeepSeek-R1。
- **应用方法：** LightRAG → Generative Agents → Deep Search。
- **训练工程：** 流水线并行 → DeepSpeed → Megatron-LM。

- **Model foundations and post-training:** Gradients and backpropagation → Transformer → QLoRA → Policy gradients and PPO / GRPO → OPD → DeepSeek-R1.
- **Applications:** LightRAG → Generative Agents → Deep Search.
- **Training systems:** Pipeline Parallelism → DeepSpeed → Megatron-LM.

## 内容索引

| 方向 | 笔记与练习 |
| --- | --- |
| 基础原理 `foundations/` | [梯度下降与反向传播](foundations/backpropagation.md) · [Transformer 与 KV Cache](foundations/transformer.md) |
| 后训练 `post-training/` | [QLoRA](post-training/sft/qlora.md) · [策略梯度、PPO 与 GRPO](post-training/rl/ppo-grpo.md) · [OPD](post-training/opd/opd.md) |
| 应用方法 `applications/` | [LightRAG](applications/rag/light-rag.md) · [Generative Agents](applications/agents/generative-agents.md) · [Deep Search](applications/deep-search/deep-search.md) |
| 训练工程 `systems/` | [流水线并行](systems/pipeline-parallelism/pipeline-parallelism.md) · [DeepSpeed](systems/deepspeed/deepspeed.md) · [Megatron-LM](systems/megatron-lm/megatron-lm.md) |
| 模型案例 `model-studies/` | [DeepSeek-R1](model-studies/deepseek-r1/deepseek-r1.md) · [DeepSeek V4.1 Flash](model-studies/deepseek-v4.1-flash.md) |
| 算法练习 `practice/` | [LeetCode](practice/leetcode/README.md) |

| Area | Notes and exercises |
| --- | --- |
| Foundations `foundations/` | [Gradients and backpropagation](foundations/backpropagation.md) · [Transformer and KV Cache](foundations/transformer.md) |
| Post-training `post-training/` | [QLoRA](post-training/sft/qlora.md) · [Policy gradients, PPO, and GRPO](post-training/rl/ppo-grpo.md) · [OPD](post-training/opd/opd.md) |
| Applications `applications/` | [LightRAG](applications/rag/light-rag.md) · [Generative Agents](applications/agents/generative-agents.md) · [Deep Search](applications/deep-search/deep-search.md) |
| Training systems `systems/` | [Pipeline Parallelism](systems/pipeline-parallelism/pipeline-parallelism.md) · [DeepSpeed](systems/deepspeed/deepspeed.md) · [Megatron-LM](systems/megatron-lm/megatron-lm.md) |
| Model studies `model-studies/` | [DeepSeek-R1](model-studies/deepseek-r1/deepseek-r1.md) · [DeepSeek V4.1 Flash](model-studies/deepseek-v4.1-flash.md) |
| Algorithm practice `practice/` | [LeetCode](practice/leetcode/README.md) |

## 存放约定

- 每个小节先中文、后英文，不加语言标记；常用术语保留英文，代码、公式和图片共用一份。
- 单篇笔记直接放在所属分类目录，多篇笔记或代码组成独立主题时再建子目录；目录和笔记文件名使用小写及连字符。
- 同一主题的 Markdown 与 `images/` 放在一起，图片通过相对路径引用。
- 文件名写清实际主题，例如 `qlora.md`、`light-rag.md`、`generative-agents.md`。
- 通用方法集中讲原理，模型案例讲具体应用。

- Put Chinese first and English second within each section, without language labels. Keep common technical terms in English and share code, formulas, and images.
- Place a single note directly in its category. Create a subdirectory when multiple notes or code files form a distinct topic. Use lowercase names with hyphens.
- Keep Markdown beside its `images/` directory and use relative image paths.
- Name files after their actual topics, such as `qlora.md`, `light-rag.md`, and `generative-agents.md`.
- Explain general principles in method notes and specific applications in model studies.
