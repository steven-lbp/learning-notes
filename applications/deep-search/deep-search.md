# DeepSearch
## Search-o1
Search-O1 introduces a structured search-enhanced generation process that coordinates when to search and how to incorporate search results — without changing the model parameters.
<!-- 待补原图：images/1.png -->

## Search-R1
### Intro
Search-R1 is an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval.

In 2025, DeepSeek-R1 demonstrated that pure **reinforcement learning** can cultivate the self-verification ability of LLMs, but without integrating external retrieval. This paper proposes extending the reinforcement learning framework to a retrieval-augmented scenario.

### Key Concept
1. Enhanced Reasoning with Search
- This search is **dynamic**, meaning the model can decide in real-time whether it needs additional data, and then retrieve it.
2. Application Areas
- Search-R1 is particularly useful for tasks in **scientific research, knowledge-intensive applications, programming, and mathematical reasoning**, where knowledge is constantly evolving or too large to be entirely stored in a model's parameters.

### RL
<!-- 待补原图：images/2.png -->
### Training template
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
**Core idea**: Instead of simple masked language modeling, RAMP requires the model to actively search for the information needed to fill masked parts in sentences.
```
Andrew Barto received his [mask] with distinction in [mask] from the University of Michigan in 1970.
```

### 2. Multi-Agent Collaboration for SFT
To generate high-quality chain-of-thought (CoT) data for supervised fine-tuning (SFT), MaskSearch builds a multi-agent system:

1. Planner: Determines masked items to search.

2. Rewriter: Crafts effective search queries.

3. Observer: Reads and extracts facts.

4. Teacher: Validates correctness.

### 3. Reinforcement Learning with DAPO
After SFT, MaskSearch uses reinforcement learning (RL) to further refine the model using Dynamic-Aware Policy Optimization (DAPO), which optimizes two reward types:

**Format reward**: Ensures the output structure matches expected formats.

**Answer reward**: Evaluates correctness vs. the ground truth (using large-instruction models like Qwen2.5-72B-Instruct)

### 4. Curriculum Learning
MaskSearch introduces difficulty-based curriculum learning:

- Prompts contain varying numbers of masked tokens.

- The model starts with simpler tasks (fewer masks) and gradually tackles more complex ones (multiple masks), improving stability and reasoning capability.

## Baichuan-Research
### structured generation with control tokens

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
- `EM:` Exact Match

- `LJ:` Longest Jump

LJ measures whether the model took the correct long path from the problem to the correct answer (rather than guessing).

| Dataset       | Description                                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **HotpotQA**  | A widely-used multi-hop QA dataset requiring reasoning across 2+ Wikipedia paragraphs. Includes supporting facts for supervision.               |
| **2Wiki**     | Entity-centric multi-hop QA built from Wikipedia entity pairs. Focuses on factual relational reasoning.                                         |
| **MuSiQue**   | Constructed by composing multiple atomic single-hop questions. Tests fine-grained semantic compositional reasoning.                             |
| **Bamboogle** | A realistic search-based QA dataset that requires triggering external retrieval to answer. Designed to simulate real-world information-seeking. |

>  **ReSearch is trained on only one dataset** (MuSiQue) but achieves **strong generalization** across all four.

### Baseline Methods Compared in ReSearch
| Baseline     | Retrieval? | Structured Output?               | Reasoning Depth               | RL-Trained?  |
| ------------ | ---------- | -------------------------------- | ----------------------------- | ------------ |
| Naive Gen    | No         |  No                              |  None                         |  No          |
| Naive RAG    | Yes        |  No                              |  Shallow                      |  No          |
| Iter-RetGen  | Yes        |  Partial (loop)                  |  Deeper                       |  No          |
| IRCoT        | Yes        |  Yes (interleaved)               |  Deep + precise               |  No          |
| **ReSearch** | Yes        |  Yes (`<think>`, `<search>`...)  |  Deepest (trained planning)   |  Yes (GRPO)  |

