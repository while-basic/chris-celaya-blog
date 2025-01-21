---
cover: /articles/medchat.webp
author:
  name: Christopher Celaya
  avatarUrl: /generative-ai/storm.webp
  link: https://twitter.com/Im_Mr_Chris
date: 2023-08-01T00:00:00.000Z
description: .

layout: article
---

# Efficient Fine-Tuning of Large Language Models for Domain-Specific Applications: 

A Case Study on Minecraft Using the Unsloth Framework

Written by:

- Christopher B. Celaya

- [chriscelaya.com](http://chriscelaya.com)

- [chris@chriscelaya.com](mailto:chris@chriscelaya.com)*

Here's a concise summary that highlights the core elements of what we will be working on:

---

## **Efficient Fine-Tuning with Unsloth**

This tutorial demonstrates how to fine-tune the Qwen 7B model for a Minecraft AI assistant, "Andy," using the Unsloth framework for efficient training.

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

---

Here's a full summary that highlights the code.

# **Efficient Fine-Tuning Using Unsloth: A Case Study on Training a Minecraft AI Assistant**

This tutorial demonstrates the process of fine-tuning a high-performance language model to create an AI assistant tailored for Minecraft. 

By leveraging the Unsloth framework, we will optimize training efficiency while achieving robust task performance. 

The resulting assistant, "Andy," will provide players with guidance on crafting, game strategies, and mechanics.

---

## **Prerequisites**

Before proceeding, ensure the following requirements are met:

- Access to **Google Colab** (free tier with T4 GPU is sufficient, but higher-tier GPUs are recommended for faster training).
- Familiarity with **Python** programming and basic concepts of deep learning.
- An optional **Hugging Face account** for model hosting and sharing.

---

## **Environment Setup**

To begin, install the required packages. The `unsloth` framework and its dependencies will facilitate model fine-tuning:

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

---

## **Base Model Initialization**

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

---

## **Incorporating LoRA Adapters**

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

---

## **Defining a Chat Template**

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

---

## **Dataset Preparation**

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

---

## **Training Configuration**

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

---

## **Evaluation**

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

---

## **Saving the Model**

To preserve the trained model, we save it locally or push it to the Hugging Face Hub for broader accessibility:

```python
# Save locally
model.save_pretrained("andy_minecraft_bot")
tokenizer.save_pretrained("andy_minecraft_bot")

# Save to Hugging Face (optional)
# model.push_to_hub("your-username/andy-minecraft-bot", token="your-token")
# tokenizer.push_to_hub("your-username/andy-minecraft-bot", token="your-token")
```

---

## **Optimization Recommendations**

1. **Dataset Augmentation**: Expand the dataset to include:
- Advanced crafting recipes
- Redstone mechanisms
- Combat strategies
- Survival tips
1. **Extended Training**: Increase max_steps (e.g., 500–1000) for improved performance.
2. **Hyperparameter Tuning**: Experiment with:

•	Learning rates (reduce as the number of steps increases for better stability)

•	Batch sizes

•	Scheduler types

4.	**Inference Tweaks**: Adjust temperature or repetition penalty during inference for tailored responses.

---

### **Recap**

In this tutorial, we fine-tuned the Qwen 7B model to create a Minecraft-specific AI assistant, "Andy," using the Unsloth framework. Key steps included setting up the environment, initializing the model with 4-bit quantization for efficient memory usage, adding LoRA adapters to reduce computational overhead, preparing the Minecraft-focused dataset, and configuring training parameters to optimize performance on Google Colab. After training, we tested Andy's responses to Minecraft queries, saving the model for future use or sharing.

### **Outro**

Congratulations on completing this tutorial! You've successfully fine-tuned an advanced language model for a specialized task with minimal computational resources. By leveraging tools like Unsloth, LoRA, and 4-bit quantization, you've unlocked the ability to efficiently train sophisticated AI models without the need for expensive hardware. We encourage you to further experiment with different datasets, training durations, and inference configurations to refine your model's performance. Happy training, and we look forward to seeing what you build next!

Written by:

*Christopher B. Celaya
[chriscelaya.com](http://chriscelaya.com)
[chris@chriscelaya.com](mailto:chris@chriscelaya.com)*

::hero
---
image: /generative-ai/yoda.webp
---

Written by:

*Christopher B. Celaya
[chriscelaya.com](http://chriscelaya.com)
[chris@chriscelaya.com](mailto:chris@chriscelaya.com)*

Here's a concise summary that highlights the core elements of what we will be working on:

# **Summary: Efficient Fine-Tuning with Unsloth**

This tutorial demonstrates how to fine-tune the Qwen 7B model for a Minecraft AI assistant, "Andy," using the Unsloth framework for efficient training.

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

---

Here's a full summary that highlights the code.

# **Efficient Fine-Tuning Using Unsloth: A Case Study on Training a Minecraft AI Assistant**

This tutorial demonstrates the process of fine-tuning a high-performance language model to create an AI assistant tailored for Minecraft. By leveraging the Unsloth framework, we will optimize training efficiency while achieving robust task performance. The resulting assistant, "Andy," will provide players with guidance on crafting, game strategies, and mechanics.

---

## **Prerequisites**

Before proceeding, ensure the following requirements are met:

- Access to **Google Colab** (free tier with T4 GPU is sufficient, but higher-tier GPUs are recommended for faster training).
- Familiarity with **Python** programming and basic concepts of deep learning.
- An optional **Hugging Face account** for model hosting and sharing.

---

## **Environment Setup**

To begin, install the required packages. The `unsloth` framework and its dependencies will facilitate model fine-tuning:

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

---

## **Base Model Initialization**

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

---

## **Incorporating LoRA Adapters**

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

---

## **Defining a Chat Template**

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

---

## **Dataset Preparation**

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

---

## **Training Configuration**

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

---

## **Evaluation**

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

---

## **Saving the Model**

To preserve the trained model, we save it locally or push it to the Hugging Face Hub for broader accessibility:

```python
# Save locally
model.save_pretrained("andy_minecraft_bot")
tokenizer.save_pretrained("andy_minecraft_bot")

# Save to Hugging Face (optional)
# model.push_to_hub("your-username/andy-minecraft-bot", token="your-token")
# tokenizer.push_to_hub("your-username/andy-minecraft-bot", token="your-token")
```

---

## **Optimization Recommendations**

1. **Dataset Augmentation**: Expand the dataset to include:
- Advanced crafting recipes
- Redstone mechanisms
- Combat strategies
- Survival tips
1. **Extended Training**: Increase max_steps (e.g., 500–1000) for improved performance.
2. **Hyperparameter Tuning**: Experiment with:

•	Learning rates (reduce as the number of steps increases for better stability)

•	Batch sizes

•	Scheduler types

4.	**Inference Tweaks**: Adjust temperature or repetition penalty during inference for tailored responses.

---

### **Recap**

In this tutorial, we fine-tuned the Qwen 7B model to create a Minecraft-specific AI assistant, "Andy," using the Unsloth framework. Key steps included setting up the environment, initializing the model with 4-bit quantization for efficient memory usage, adding LoRA adapters to reduce computational overhead, preparing the Minecraft-focused dataset, and configuring training parameters to optimize performance on Google Colab. After training, we tested Andy's responses to Minecraft queries, saving the model for future use or sharing.

### **Outro**

Congratulations on completing this tutorial! You've successfully fine-tuned an advanced language model for a specialized task with minimal computational resources. By leveraging tools like Unsloth, LoRA, and 4-bit quantization, you've unlocked the ability to efficiently train sophisticated AI models without the need for expensive hardware. 

We encourage you to further experiment with different datasets, training durations, and inference configurations to refine your model's performance. Happy training, and we look forward to seeing what you build next!

Written by:

- Christopher B. Celaya



