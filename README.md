# Learning Notes

个人学习笔记与代码练习，按基础原理、后训练、应用方法、训练工程、模型案例和算法练习分类。

## 学习路线

- **模型原理与后训练：** Transformer → QLoRA → PPO / GRPO → OPD → DeepSeek-R1。
- **应用方法：** LightRAG → Generative Agents → Deep Search。
- **训练工程：** 流水线并行 → DeepSpeed → Megatron-LM。

## 内容索引

| 方向 | 笔记与练习 |
| --- | --- |
| 基础原理 `foundations/` | [Transformer](foundations/transformer/transformer.md) |
| 后训练 `post-training/` | [QLoRA](post-training/sft/qlora.md) · [PPO 与 GRPO](post-training/rl/ppo-grpo.md) · [OPD](post-training/opd/opd.md) |
| 应用方法 `applications/` | [LightRAG](applications/rag/light-rag.md) · [Generative Agents](applications/agents/generative-agents.md) · [Deep Search](applications/deep-search/deep-search.md) |
| 训练工程 `systems/` | [流水线并行](systems/pipeline-parallelism/pipeline-parallelism.md) · [DeepSpeed](systems/deepspeed/deepspeed.md) · [Megatron-LM](systems/megatron-lm/megatron-lm.md) |
| 模型案例 `model-studies/` | [DeepSeek-R1](model-studies/deepseek-r1/deepseek-r1.md) |
| 算法练习 `practice/` | [LeetCode](practice/leetcode/README.md) |

## 存放约定

- 按“学习方向 → 具体主题”存放，目录和笔记文件名使用小写及连字符，例如 `post-training/opd/opd.md`。
- 同一主题的 Markdown 与 `images/` 放在一起，图片通过相对路径引用。
- 文件名写清实际主题，例如 `qlora.md`、`light-rag.md`、`generative-agents.md`。
- 通用方法集中讲原理，模型案例讲具体应用，通过相对链接关联。
