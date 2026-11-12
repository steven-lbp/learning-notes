# DeepSearch

## Search-o1

Search-o1 在推理过程中协调何时搜索、怎样整合搜索结果，通过结构化的搜索增强生成提升回答，无需更新模型参数。

Search-O1 introduces a structured search-enhanced generation process that coordinates when to search and how to incorporate search results — without changing the model parameters.
<!-- 待补原图：images/1.png -->

## Search-R1

### Intro

Search-R1 将 RL 扩展到检索增强推理：模型在逐步推理中自主生成一个或多个搜索 Query，并利用实时结果继续推理。DeepSeek-R1 展示了 RL 对自检与推理能力的帮助，Search-R1 进一步把外部检索纳入训练过程。

Search-R1 is an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval.

In 2025, DeepSeek-R1 demonstrated that pure **reinforcement learning** can cultivate the self-verification ability of LLMs, but without integrating external retrieval. This paper proposes extending the reinforcement learning framework to a retrieval-augmented scenario.

### Key Concept

1. **搜索增强推理：** 模型动态决定是否需要补充信息，再发起检索。
2. **应用方向：** 适合科研、知识密集型问答、编程和数学推理等需要外部知识的任务。

The two key ideas are:

1. **Enhanced Reasoning with Search:** the model dynamically decides whether it needs additional information and retrieves it.
2. **Application Areas:** scientific research, knowledge-intensive applications, programming, and mathematical reasoning can benefit from knowledge that evolves or exceeds what model parameters retain.

### RL

<!-- 待补原图：images/2.png -->

### Training template

推理、搜索、读取知识、回答构成循环；检索结果可以触发下一轮推理与搜索。

Reasoning, search, retrieved knowledge, and answering form a loop. Retrieved results can trigger another round of reasoning and search.

```
Think ---> Search ---> Knowledge ---> Answer
 ^                          |
 |                          |
 |__________________________|
```
<!-- 待补原图：images/3.png -->

## MaskSearch

<!-- 待补原图：images/4.png -->

### 1. RAMP Task: Retrieval-Augmented Masked Prediction

RAMP 要求模型主动搜索信息，补全句子中被 Mask 的内容，将检索过程融入 Masked Prediction。

**Core idea**: Instead of simple masked language modeling, RAMP requires the model to actively search for the information needed to fill masked parts in sentences.
```
Andrew Barto received his [mask] with distinction in [mask] from the University of Michigan in 1970.
```

### 2. Multi-Agent Collaboration for SFT

MaskSearch 用多个 Agent 生成高质量 CoT 数据供 SFT 使用：

1. **Planner：** 决定要搜索的 Mask 项。
2. **Rewriter：** 改写搜索 Query。
3. **Observer：** 阅读结果并提取事实。
4. **Teacher：** 验证正确性。

To generate high-quality chain-of-thought (CoT) data for supervised fine-tuning (SFT), MaskSearch builds a multi-agent system:

1. Planner: Determines masked items to search.

2. Rewriter: Crafts effective search queries.

3. Observer: Reads and extracts facts.

4. Teacher: Validates correctness.

### 3. Reinforcement Learning with DAPO

SFT 后使用 DAPO 继续训练，主要结合两类奖励：

- **Format reward：** 检查输出结构。
- **Answer reward：** 判断答案与 Ground Truth 是否一致，可借助 Qwen2.5-72B-Instruct 等模型。

After SFT, MaskSearch uses reinforcement learning (RL) to further refine the model using DAPO, which optimizes two reward types:

**Format reward**: Ensures the output structure matches expected formats.

**Answer reward**: Evaluates correctness vs. the ground truth (using large-instruction models like Qwen2.5-72B-Instruct)

### 4. Curriculum Learning

训练数据设置不同数量的 Mask。模型先处理较少 Mask 的简单任务，再逐步学习多个 Mask 的复杂任务，提高训练稳定性和推理能力。

MaskSearch introduces difficulty-based curriculum learning:

- Prompts contain varying numbers of masked tokens.

- The model starts with simpler tasks (fewer masks) and gradually tackles more complex ones (multiple masks), improving stability and reasoning capability.

## Baichuan-Research

### structured generation with control tokens

通过 `<search>` 等控制 Token 识别工具调用边界，收集 Query，执行搜索，再把 `<result>` 包裹的结果加入上下文，继续生成，直到答案结束。

```
algorithm ReSearchInference:
    input: trained policy_model πθ, new question q
    prompt = create_prompt(q)
    output_seq = []
    current_text = ""
    done = false
    while not done:
        token = policy_model.generate_next_token(current_text)
        output_seq.append(token)
        current_text += token
        if token == "<search>":
            # gather query until '</search>'
            search_query = generate_until("</search>")
            result_text = SEARCH(search_query)
            output_seq.append("<result>")
            output_seq.append(result_text)
            output_seq.append("</result>")
            current_text += "<result>" + result_text + "</result>"
        if token == "</answer>" or end_of_text(token):
            done = true

    return output_seq  # The final answer is within <answer> ... </answer>
```

### Benchmarks

- **EM：** Exact Match，检查答案是否精确匹配。
- **LJ：** LLM-as-a-Judge，由 LLM 判断预测答案与参考答案是否语义一致。

| 数据集 | 特点 |
| --- | --- |
| HotpotQA | 跨多个 Wikipedia 段落的 Multi-hop QA，带支持事实 |
| 2Wiki | 从实体对构建的关系推理问答 |
| MuSiQue | 组合多个单跳问题，评估组合推理 |
| Bamboogle | 面向真实信息需求、需要外部搜索的问答 |

ReSearch 仅在 MuSiQue 上训练，并评估对其他三个数据集的泛化。

- `EM:` Exact Match

- `LJ:` LLM-as-a-Judge

LJ uses an LLM to judge whether the predicted answer matches the reference answer semantically.

| Dataset       | Description                                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **HotpotQA**  | A widely-used multi-hop QA dataset requiring reasoning across 2+ Wikipedia paragraphs. Includes supporting facts for supervision.               |
| **2Wiki**     | Entity-centric multi-hop QA built from Wikipedia entity pairs. Focuses on factual relational reasoning.                                         |
| **MuSiQue**   | Constructed by composing multiple atomic single-hop questions. Tests fine-grained semantic compositional reasoning.                             |
| **Bamboogle** | A realistic search-based QA dataset that requires triggering external retrieval to answer. Designed to simulate real-world information-seeking. |

>  **ReSearch is trained on only one dataset** (MuSiQue) but achieves **strong generalization** across all four.

### Baseline Methods Compared in ReSearch

| 方法 | 检索 | 输出结构 | 推理方式 | RL 训练 |
| --- | --- | --- | --- | --- |
| Naive Gen | 无 | 无 | 直接生成 | 无 |
| Naive RAG | 有 | 无 | 检索后生成 | 无 |
| Iter-RetGen | 有 | 部分循环结构 | 迭代检索与生成 | 无 |
| IRCoT | 有 | 交错组织 | 交替推理与检索 | 无 |
| ReSearch | 有 | `<think>`、`<search>` 等 | 学习多步推理与搜索 | GRPO |

| Baseline     | Retrieval? | Structured Output?               | Reasoning Depth               | RL-Trained?  |
| ------------ | ---------- | -------------------------------- | ----------------------------- | ------------ |
| Naive Gen    | No         |  No                              |  None                         |  No          |
| Naive RAG    | Yes        |  No                              |  Shallow                      |  No          |
| Iter-RetGen  | Yes        |  Partial (loop)                  |  Deeper                       |  No          |
| IRCoT        | Yes        |  Yes (interleaved)               |  Deep + precise               |  No          |
| **ReSearch** | Yes        |  Yes (`<think>`, `<search>`...)  |  Learned multi-step reasoning   |  Yes (GRPO)  |

[ReSearch paper](https://arxiv.org/html/2503.19470v1)
