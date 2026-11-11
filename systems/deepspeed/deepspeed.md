# Deepspeed
## Why do we need deepspeed
- ZeRO: a kind of data Parallel

ZeRO can reduce memory usage, optimize large model training, and divide model parameters into three parts: Optimizer States, Gradient, and Model Parameters

- Mixed precision training
- Deepspeed supports larger scale model training

## Communication strategy
``MPI`` is a cross node communication library commonly used for distributed training on CPU clusters;

``Gloo`` is a high-performance distributed training framework that supports distributed training on both CPU and GPU;

``NCCL`` is a GPU specific communication library provided by NVIDIA, widely used for distributed training on GPUs.
## Zero Redundancy Optimizer
ZeRO divides model parameters into three parts: Optimizer States, Gradient, and Model Parameters
- ``Optimizer States`` are the data that Optimizer needs to use for gradient updates, such as Momentum in SGD.
- ``Gradient`` is the gradient information generated after backpropagation, which determines the direction of parameter updates.
- ``Model parameters`` refer to the information we 'learn' from data throughout the entire process.

``ZeRO-0``: Disable all types of shards and only use DeepSpeed as DDP (Distributed Data Parallel)

``ZeRO-1``: Splitting ``Optimizer States`` reduces memory by 4 times, with the same communication capacity and data parallelism

``ZeRO-2``: Split ``Optimizer States`` and ``Gradients``, reduce memory by 8x, with the same communication capacity and data parallelism

``ZeRO-3``: Splitting ``Optimizer States``, ``Gradients``, and ``Parameters``, memory reduction is linearly related to data parallelism and complexity

## Mixed precision training
Due to the low accuracy of FP16, there may be issues with gradient vanishing and model instability during the training process.

``"fp16.enabled": true``: During the training process, DeepSpeed will automatically convert some operations to FP16 format and dynamically adjust the accuracy scaling factor as needed to ensure the stability and accuracy of the training.

``"bf16.enabled": true``: BF16 can serve as a more accurate alternative for some critical computational operations, such as gradient accumulation and weight updates
## Example code
- ds_zero2_config.json
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
- train.py
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