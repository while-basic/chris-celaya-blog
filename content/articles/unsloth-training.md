---
cover: /minecraft.webp
author:
  name: Christopher B. Celaya
  avatarUrl: /generative-ai/storm.webp
  link: https://chriscelaya.com
date: 2025-01-22T00:00:00.000Z
description: Learn how to efficiently fine-tune large language models using the Unsloth framework, demonstrated through a practical Minecraft AI assistant implementation.
title: Efficient Fine-Tuning of Large Language Models - A Minecraft AI Assistant Tutorial
layout: article
tags: ['AI', 'Machine Learning', 'LLM', 'Minecraft', 'Tutorial', 'Python', 'Unsloth']
toc: true
---

# Efficient Fine-Tuning of Large Language Models

A Case Study on Minecraft Using the Unsloth Framework

::alert{type="info"}
🚀 **Source Code**: Find the complete implementation and run using Mindcraft [GitHub repository](https://github.com/kolbytn/mindcraft).

::

## Introduction

This tutorial demonstrates how to fine-tune the Qwen 7B model to create "Andy," a Minecraft AI assistant, using the Unsloth framework for efficient training. You'll learn how to leverage cutting-edge techniques like 4-bit quantization and LoRA to achieve scalable fine-tuning without requiring extensive computational resources.

### Key Steps Overview:

1. **Setup**: Install Unsloth and dependencies for memory-efficient training
2. **Model Initialization**: Load Qwen 7B with 4-bit quantization
3. **LoRA Adapters**: Add Low-Rank Adaptation for efficient fine-tuning
4. **Dataset Preparation**: Format the Minecraft-specific dataset
5. **Training**: Configure and run the fine-tuning process
6. **Evaluation**: Test the assistant's performance
7. **Model Saving**: Save and share your trained model

### Key Steps:

1. **Setup**: Install Unsloth and dependencies for memory-efficient training.
2. **Model Initialization**: Load Qwen 7B with 4-bit quantization for reduced resource usage.
3. **LoRA Adapters**: Add Low-Rank Adaptation (LoRA) to fine-tune select model layers efficiently.
4. **Dataset Preparation**: Format the Minecraft-specific Andy-3.5 dataset using ChatML templates.
5. **Training**: Fine-tune the model using lightweight hyperparameter configurations suitable for Google Colab.
6. **Evaluation**: Test Andy's performance on Minecraft-related queries, ensuring task-specific accuracy.
7. **Model Saving**: Save the fine-tuned model locally or share it via the Hugging Face Hub.

### Optimization Tips:

- Expand the dataset for broader Minecraft knowledge.
- Extend training steps and fine-tune hyperparameters for higher-quality outputs.
- Adjust inference parameters for more natural or diverse responses.

By leveraging cutting-edge techniques like 4-bit quantization and LoRA, this workflow achieves scalable fine-tuning without requiring extensive computational resources.

## Prerequisites

Before proceeding, ensure the following requirements are met:

::alert{type="warning"}
⚡ **GPU Requirements**: While a T4 GPU (available in Google Colab free tier) is sufficient, using an A100 or V100 will significantly speed up training.
::

- Access to **Google Colab** (free tier with T4 GPU is sufficient, but higher-tier GPUs are recommended for faster training).
- Familiarity with **Python** programming and basic concepts of deep learning.
- An optional **Hugging Face account** for model hosting and sharing.

## Environment Setup

To begin, install the required packages. The `unsloth` framework and its dependencies will facilitate model fine-tuning:

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

## Base Model Initialization

For this tutorial, we will utilize the **Qwen 7B** model, which has demonstrated strong performance in gaming-related tasks. The configuration ensures memory-efficient loading using 4-bit quantization:

```python
from unsloth import FastLanguageModel
import torch

# Model parameters
max_seq_length = 2048 # Max sequence length can be increased, but instability occurs around 8192, with 16,000 recommended.

dtype = None  # Automatically detects the optimal precision (e.g., bf16 or fp16)
load_in_4bit = True  # Enables memory-efficient quantization

# Load the pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

### Explanation:

The `FastLanguageModel.from_pretrained` method initializes a compressed, memory-optimized model, allowing for effective fine-tuning on hardware with limited resources.

## Incorporating LoRA Adapters

To further enhance efficiency, we utilize **Low-Rank Adaptation (LoRA)**. This approach enables us to fine-tune specific layers of the model, significantly reducing computational overhead:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank of the low-rank matrices
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Disables dropout for more deterministic training
    bias="none",
    use_gradient_checkpointing="unsloth",  # Reduces memory usage during backpropagation
    random_state=3407,  # Ensures reproducibility
)
```

### Explanation:

LoRA allows fine-tuning a small subset of parameters while keeping the majority of the model frozen. This is computationally efficient and ideal for domain-specific adaptation, such as Minecraft-related tasks. 

A full model tune can be done with LoRA, but it is better suited for small adjustments, as demonstrated in the Minecraft implementation.

## Defining a Chat Template

To enable structured interactions, we define a chat template using the **ChatML** format. This ensures that the assistant processes conversations correctly:

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    map_eos_token=True,
)
```

## Dataset Preparation

We utilize the **Andy-3.5** dataset, a curated collection of Minecraft-specific dialogues. The dataset is preprocessed to align with the ChatML format:

```python
from datasets import load_dataset

# Define a formatting function to process conversation data
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

# Load and preprocess the dataset
dataset = load_dataset("Sweaterdog/Andy-3.5", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Explanation:

The formatting step ensures that the dataset aligns with the model's expected input format, enhancing training consistency.

## Training Configuration

::alert{type="success"}
💡 **Pro Tip**: Monitor the training progress using Weights & Biases for better visualization and experiment tracking.
::

For a full run, set max_steps to 0. Include num_train_epochs = 1 to specify a single epoch.

We now configure and initiate the fine-tuning process. The parameters are optimized for efficiency on limited hardware:

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Define the training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,  # use the one supported by your model
    dataset_num_proc=4,  # Leverage multiprocessing for faster data prep
    packing=True,  # Enable packing to increase throughput for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=16,  # A100 can handle larger batch sizes
        gradient_accumulation_steps=1,  # Reduce as larger batches are feasible
        warmup_steps=500,  # Longer warmup for stability with large models
        max_steps=0,  # Adjust for the size of your dataset and desired epochs
        learning_rate=2e-5,  # Smaller for fine-tuning; scale up for larger models
        fp16=True,  # Use mixed precision for speed and memory savings
        bf16=False,  # Use BF16 if supported (A100 supports both; choose one)
        logging_steps=50,  # Reduce logging frequency for longer runs
        optim="adamw_torch",  # Switch to more efficient optimizer
        weight_decay=0.01,  # Regularization for better generalization
        lr_scheduler_type="cosine",  # Cosine decay works well for transformers
        seed=42,  # Reproducibility
        output_dir="outputs",
        gradient_checkpointing=True,  # Useful for extremely large models
    ),
)

# Start training
trainer_stats = trainer.train()
```

## Evaluation

After training, the model is ready for evaluation. We test it with a Minecraft-related query:

```python
FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Can you fetch me some wood to start us off?"},
]

# Format the input for inference
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Generate a response
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
)
```

## Saving the Model

To preserve the trained model, we save it locally or push it to the Hugging Face Hub for broader accessibility:

```python
# Save locally
model.save_pretrained("andy_minecraft_bot")
tokenizer.save_pretrained("andy_minecraft_bot")

# Save to Hugging Face (optional)
# model.push_to_hub("your-username/andy-minecraft-bot", token="your-token")
# tokenizer.push_to_hub("your-username/andy-minecraft-bot", token="your-token")
```

## Results and Recommendations

::div{class="flex justify-center my-8"}
  ![Training Results](/articles/training-results.webp)
  *Figure 3: Training Loss and Evaluation Metrics*
::

After implementing the training pipeline, here are key recommendations for optimal results:

### Dataset Enhancement
- Expand with advanced crafting recipes
- Add Redstone mechanism tutorials
- Include combat strategies
- Incorporate survival tips and tricks

### Training Optimization
- Start with 500-1000 training steps
- Experiment with learning rates (reduce as steps increase)
- Test different batch sizes for your hardware
- Try various scheduler types

### Inference Tuning
- Adjust temperature for creativity vs. consistency
- Fine-tune repetition penalty for natural responses
- Balance response length with max_new_tokens

## Conclusion

This tutorial demonstrated how to efficiently fine-tune a large language model for a specific domain using the Unsloth framework. The Minecraft AI assistant example shows how to:

1. Optimize memory usage with 4-bit quantization
2. Reduce computational overhead using LoRA
3. Prepare and process domain-specific training data
4. Configure training parameters for efficient learning
5. Save and deploy the fine-tuned model

The techniques covered here can be applied to various domain-specific applications beyond gaming, making it a valuable reference for anyone looking to create specialized AI assistants with limited computational resources.

---

::div{class="flex flex-col gap-4 mt-8"}
  ## Additional Resources
  - [Unsloth Documentation](https://github.com/unslothai/unsloth)
  - [LoRA Paper](https://arxiv.org/abs/2106.09685)
  - [Quantization Guide](https://huggingface.co/docs/transformers/quantization)
  - [Join our Discord Community](https://discord.gg/UwrCbURA)
::

::div{class="flex justify-between items-center mt-8 p-4 bg-gray-100 rounded-lg"}
  ### Share this tutorial
  - [Twitter](https://twitter.com/share?url=https://chriscelaya.com/articles/unsloth-training)
  - [LinkedIn](https://www.linkedin.com/shareArticle?url=https://chriscelaya.com/articles/unsloth-training)
  - [GitHub](https://github.com/kolbytn/mindcraft)
  - [Dataset](https://huggingface.co/Sweaterdog)
::

Written by:
- Christopher Celaya
- [chriscelaya.com](http://chriscelaya.com)
- [chris@chriscelaya.com](mailto:chris@chriscelaya.com)
- 

