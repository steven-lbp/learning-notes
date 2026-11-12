# DeepSpeed

## Why DeepSpeed

DeepSpeed 通过 ZeRO、Mixed Precision、Offload 等机制降低大模型训练的显存需求，并支持更大规模的分布式训练。

DeepSpeed reduces large-model training memory requirements through ZeRO, Mixed Precision, and Offload, supporting larger distributed training workloads.

## Communication backends

- **MPI：** 常见于跨节点分布式计算和 CPU 集群。
- **Gloo：** 常用于 CPU Tensor 的分布式通信。
- **NCCL：** NVIDIA 面向 GPU 的高性能集合通信库。

- **MPI:** commonly used for distributed computation across nodes and CPU clusters.
- **Gloo:** commonly used for distributed communication involving CPU tensors.
- **NCCL:** NVIDIA's high-performance collective communication library for GPUs.

## ZeRO

普通 DP 会在各副本中重复存储模型状态。ZeRO 按阶段将这些状态分片：

| 状态 | 含义 |
| --- | --- |
| Optimizer States | Optimizer 更新所需的动量、二阶统计等信息 |
| Gradients | 反向传播得到的参数梯度 |
| Parameters | 模型学习到的权重 |

| 阶段 | 分片内容 |
| --- | --- |
| ZeRO-0 | 不分片，使用普通 DP |
| ZeRO-1 | Optimizer States |
| ZeRO-2 | Optimizer States + Gradients |
| ZeRO-3 | Optimizer States + Gradients + Parameters |

显存节省随 DP 规模、状态精度和分片阶段变化。Offload 还可以将部分状态移到 CPU 或 NVMe。

Ordinary DP replicates model states across workers. ZeRO progressively partitions them:

| State | Meaning |
| --- | --- |
| Optimizer States | Momentum, second moments, and other state required for updates |
| Gradients | Parameter gradients produced by backpropagation |
| Parameters | The model's learned weights |

| Stage | Sharded states |
| --- | --- |
| ZeRO-0 | None; ordinary DP |
| ZeRO-1 | Optimizer States |
| ZeRO-2 | Optimizer States + Gradients |
| ZeRO-3 | Optimizer States + Gradients + Parameters |

Memory savings depend on the DP degree, state precision, and sharding stage. Offload can additionally move states to CPU or NVMe.

[DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)

## Mixed Precision

FP16 可以减少显存与计算开销，但其数值范围较小，训练中常配合 Loss Scaling，减少梯度下溢。配置 `fp16.enabled` 可以启用 FP16 训练。

BF16 的数值范围更大，通常不需要 Loss Scaling；配置 `bf16.enabled` 启用。不同训练模式中，梯度累积和 Optimizer 状态还可以使用更高精度。

FP16 reduces memory and compute costs, but its limited numerical range often requires Loss Scaling to reduce gradient underflow. Enable it through `fp16.enabled`.

BF16 has a wider numerical range and usually does not require Loss Scaling; enable it through `bf16.enabled`. Gradient accumulation and optimizer states can use higher precision depending on the training setup.

[DeepSpeed precision configuration](https://www.deepspeed.ai/docs/config-json/)

## Example code

### ds_zero2_config.json

下面的配置使用 ZeRO-2，将 Optimizer States Offload 到 CPU，并启用 FP16 与 Activation Checkpointing。

The configuration below uses ZeRO-2, offloads Optimizer States to CPU, and enables FP16 and Activation Checkpointing.

```
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "steps_per_print": 50,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    # Optimizer state moved to CPU
    "offload_optimizer": {
            "device": "cpu"
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  },
  "wall_clock_breakdown": false
}
```

### train.py

示例依次完成数据 Tokenization、模型加载、LoRA 配置和 Trainer 训练。`train_on_inputs=False` 时，输入 Prompt 对应的 Label 设为 −100，只监督 Response 部分。

The example tokenizes data, loads the model, configures LoRA, and trains with Trainer. When `train_on_inputs=False`, prompt labels are set to −100 so only the response contributes to the supervised objective.

```
import os
import torch
import random
import datasets
import numpy as np
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)

#--------------------- 1. Set the argument ---------------------#
# LoRA argument
LORA_R = 8  # rank, higher rank, higher information, higher cache consumption
LORA_ALPHA = 32  # effective_scaling(lr) = LORA_ALPHA / LORA_RANK
LORA_DROPOUT = 0.1  # Prevent overfitting

# training argument
EPOCHS = 3
LEARNING_RATE = 5e-5
OUTPUT_DIR = "./checkpoints"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS=3  # effective_batch_size = BATCH_SIZE * GRAD_ACCUMULATION

# other argument
MODEL_PATH = "model/llama-3.1-7B"
DATA_PATH = "./data/data4llama.json"
MAX_LENGTH = 512
PATTERN = "{}\n{}"
DeepSpeed_CONFIG = "ds_zero2_config.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
dataset = datasets.load_dataset("json", data_files=DATA_PATH)


#--------------------- 2. tokenize ---------------------#
def tokenize(text: str, add_eos_token=True):
    result = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None)
    # 判断是否要添加eos_token
    if (result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < MAX_LENGTH
        and add_eos_token):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess(example: Dict, train_on_inputs: bool = False):
    prompt = example["input"]
    response = example["target"]
    text = PATTERN.format(prompt, response)
    tokenized_inp = tokenize(text)
    # 若train_on_inputs为False，则将label中与input相关的token替换为-100
    if not train_on_inputs:
        tokenized_prompt = tokenize(prompt,add_eos_token=False)
        prompt_tokens_len = len(tokenized_prompt["input_ids"])
        tokenized_inp["labels"] = [-100]*prompt_tokens_len + tokenized_inp["labels"][prompt_tokens_len:]
    return tokenized_inp


train_data = dataset["train"].shuffle().map(preprocess, remove_columns=["id", "input", "target"])
print(train_data[0])

# pad_to_multiple_of=8表示padding的长度是8的倍数
collate_fn = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

#--------------------- 3. load the model ---------------------#
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# device_map指定模型加载的GPU;troch_dtype=torch.float16表示半精度加载模型
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map=device_map)


#--------------------- 4. LoRA ---------------------#
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R, # LoRA中低秩近似的秩
    lora_alpha=LORA_ALPHA, # 见上文中的低秩矩阵缩放超参数
    lora_dropout=LORA_DROPOUT, # LoRA层的dropout
)
# 转换模型
model = get_peft_model(model, lora_config)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))
# 打印模型中的可训练参数
model.print_trainable_parameters()


#--------------------- 5. training arg ---------------------#
args = TrainingArguments(
    output_dir=OUTPUT_DIR, # checkpoint的存储目录
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True, # 使用混合精度训练
    logging_steps=50,
    evaluation_strategy="no", # 不进行评估
    save_strategy="steps",
    save_steps=2000, # 保存checkpoint的step数
    save_total_limit=5, # 最多保存5个checkpoint
    deepspeed=DS_CONFIG
)


#--------------------- 6. training the model ---------------------#
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=None,
    args=args,
    data_collator=collate_fn
)
trainer.train()
model.save_pretrained("best_model")
```
