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

## Efficient Fine-Tuning of Large Language Models

A Case Study on Minecraft Using the Unsloth Framework

::alert{type="info"}
🚀 **Mindcraft Source Code**: Necessary for gameplay interaction
[GitHub repository](https://github.com/kolbytn/mindcraft).

::

::alert{type="info"}
🚀 **Colab Notebook**: Follow along with the [Colab Notebook](https://github.com/kolbytn/mindcraft).

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
%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

## Base Model Initialization

For this tutorial, we will utilize the **Qwen 7B** model, which has demonstrated strong performance in gaming-related tasks. The configuration ensures memory-efficient loading using 4-bit quantization:

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048   # Choose any! RoPE Scaling internally is supported!
dtype = None            # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True     # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models supported for 4x faster downloading + no OOMs
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/Qwen2.5-7B-bnb-4bit",
]

# Specify the desired data type (bfloat16 or float16)
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-bnb-4bit", # Eg. teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "your-token", # Use if using gated models like meta-llama/Llama-2-7b-hf
    trust_remote_code = True, # Add this line to trust remote code
)
```

## Incorporating LoRA Adapters

To further enhance efficiency, we utilize **Low-Rank Adaptation (LoRA)**. This approach enables us to fine-tune specific layers of the model, significantly reducing computational overhead:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 | Suggested 8, 16, 32, 64, 128
    # Include 'embed_tokens' and 'lm_head' in target_modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,     # Rank stabilized LoRA supported
    loftq_config = None,    # And LoftQ
)
```

### Explanation:

The `FastLanguageModel.from_pretrained` method initializes a compressed, memory-optimized model, allowing for effective fine-tuning on hardware with limited resources.

LoRA allows fine-tuning a small subset of parameters while keeping the majority of the model frozen. This is computationally efficient and ideal for domain-specific adaptation, such as Minecraft-related tasks. 

A full model tune can be done with LoRA, but it is better suited for small adjustments.

## Dataset Preparation

We utilize the Andy-3.5 dataset, curated and fine-tuned by Sweaterdog, specifically for Minecraft tasks. This dataset is designed to provide the AI with contextual knowledge of Minecraft gameplay, including crafting, exploration, and survival mechanics. The dataset is preprocessed to align with the ChatML format:

```python
# Loading the ChatML template
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

# Loading the dataset
from datasets import load_dataset
dataset = load_dataset("Sweaterdog/Andy-3.5", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

Using our `get_chat_template` function, we get the correct chat template.

Normally one has to train `<|im_start|>` and `<|im_end|>`.

Instead, map `<|im_end|>` is to be the EOS token, and `<|im_start|>` stays as is and requires no extra training for additional tokens.

```python
# ShareGPT style
{"from": "human", "value" : "Hi"}
``` 

## Initial ChatML Template and Dataset

Here is how the format works by printing the fifth element.

```python
# Print the fifth element
dataset[5]["conversations"]
```

```python
# Print the fifth element
print(dataset[5]["text"])
``` 

## Unsloth Template

This template defines how the conversation between a user and an assistant should be structured in a text format.

```python
# Define a template for structuring chat interactions between user and assistant
unsloth_template = \
    "{{ bos_token }}" \  # Placeholder for the Beginning of Sequence (BOS) token, start of a sequence
    "{{ 'You are a helpful assistant to the user\n' }}" \  # A static string indicating the assistant's role
    "{% endif %}" \  # Used for closing a conditional block in the template
    "{% for message in messages %}" \  # Start a loop over a list of messages to process each one
        "{% if message['role'] == 'user' %}" \  # Check if the message is from the user
            "{{ '>>> User: ' + message['content'] + '\n' }}" \  # Format the user's message with a prefix ">>> User:"
        "{% elif message['role'] == 'assistant' %}" \  # Check if the message is from the assistant
            "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}" \  # Format the assistant's message with a prefix ">>> Assistant:" and append the EOS token
        "{% endif %}" \  # Close the conditional block for the "if-elif"
    "{% endfor %}" \  # Close the loop that iterates through messages
    "{% if add_generation_prompt %}" \  # Check if we need to add a generation prompt
        "{{ '>>> Assistant: ' }}" \  # If the condition is true, add a prompt indicating the assistant should respond
    "{% endif %}" # Close the "if" block for the generation prompt condition

# Define the EOS (End of Sequence) token used to mark the end of the assistant's response.
unsloth_eos_token = "eos_token"

# The following block is a configuration that won't be executed because the condition is 'False'
if False:  # This condition prevents the code inside the block from executing
    tokenizer = get_chat_template(
        tokenizer,  # The tokenizer object to be updated
        chat_template = (unsloth_template, unsloth_eos_token,),  # Provide the template (unsloth_template) and EOS token.
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},  # Define a mapping to convert between different naming conventions (e.g., 'role' -> 'from', 'content' -> 'value')
        map_eos_token = True,  # A flag that, if set to True, will map <|im_end|> to </s> (used as an EOS token)
    )
```

For more information on chat templates, view [Unsloth Templates](https://github.com/unslothai/unsloth/wiki#chat-templates).

## Training Configuration

::alert{type="success"}
💡 **Pro Tip**: Monitor the training progress using Weights & Biases for better visualization and experiment tracking.
::

We now configure and initiate the fine-tuning process. The parameters are optimized for efficiency on limited hardware.

For a full run, it is recommended to set `max_steps` to `0` and a new parameter which is `num_train_epochs` to `1`.

Include `num_train_epochs = 1` to specify a single epoch.

```python
# Import necessary modules for training
from trl import SFTTrainer  # SFTTrainer is a specialized trainer for supervised fine-tuning (SFT).
from transformers import TrainingArguments  # TrainingArguments is used to configure training parameters.
from unsloth import is_bfloat16_supported  # Utility to check if bfloat16 (Brain Floating Point 16) is supported by the hardware.

# Initialize the SFTTrainer with the necessary parameters
trainer = SFTTrainer(
    model = model,      # The model that will be fine-tuned.
    tokenizer = tokenizer,  # The tokenizer to preprocess the input text data.
    train_dataset = dataset,  # The dataset used for training.
    dataset_text_field = "text",  # The name of the field in the dataset containing the text data.
    max_seq_length = max_seq_length,  # Maximum sequence length to truncate/pad the input sequences to.
    
    dataset_num_proc = 2,  # Number of processes to use for dataset preprocessing. Set to 1 to disable multiprocessing for debugging or resource constraints.
    
    packing = False,  # Enables packing of sequences into batches. Can speed up training for short sequences but might not work well for long sequences.
    
    # TrainingArguments define the hyperparameters for the training process.
    args = TrainingArguments(
        per_device_train_batch_size = 16,  # Batch size per device during training. You may need to reduce to 1 if running out of memory.
        gradient_accumulation_steps = 1,  # Number of steps to accumulate gradients before updating model weights. Set to 1 to update after every batch.
        warmup_steps = 500,  # Number of steps to perform learning rate warmup.
        max_steps = 1000,  # Total number of training steps.
        learning_rate = 2e-5,  # Default is typically higher, but this has been reduced to avoid overshooting.
        
        fp16 = not is_bfloat16_supported(),  # Enable mixed precision training using FP16 if bfloat16 is not supported by the hardware.
        bf16 = is_bfloat16_supported(),  # Enable bfloat16 training if supported by the hardware (this is common for TPU and some GPUs).
        
        logging_steps = 1,  # How often to log training information (in terms of steps). Here, it's set to log after every step.
        
        optim = "adamw_8bit",  # Use the AdamW optimizer with 8-bit precision to save memory and improve performance.
        weight_decay = 0.01,  # Apply weight decay regularization to avoid overfitting. Typically used with AdamW optimizers.
        
        lr_scheduler_type = "linear",  # Linear learning rate scheduler, decaying from the initial learning rate to 0.
        seed = 3407,  # Random seed for reproducibility of results.
        output_dir = "outputs",  # Directory where the training output (model checkpoints, logs, etc.) will be saved.
        
        gradient_checkpointing = True,  # Enable gradient checkpointing to reduce memory usage during training. Useful for large models.
        num_train_epochs = 1,
    ),
)
```

## Free Unused Memory

We explicitly release unused memory using `torch.cuda.empty_cache()`.

Call this function before starting the training loop.

```python
import torch
torch.cuda.empty_cache()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

## Train the Model

Here we invoke the actual training process.

This can take minutes to hours depending on your configuration.

::alert{type="warning"}
🚀 **Wandb.ai**: Enter your key to begin tracking the training process when prompted. [API Key](https://wandb.ai/authorize).

::

```python
trainer_stats = trainer.train()
```

## Final Memory and Time Stats

Review and collect the final memory and time statistics.

```python
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
```

## Inference

Now it is time to run the model.

Since we're using ChatML, use `apply_chat_template` with `add_generation_prompt` set to `True`.

```python
# Inference (single message)
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Build a large house with windows, doors, rooms, and beds. Place torches and other decorative materials around the house."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
```

## Streaming

You can also use a `TextStreamer` for continuous inference which allows you to see the generation token by token.

```python
# Streaming (single message)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Complete the achievments in order?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128, use_cache = True)
```

## Saving and Loading the Model

To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**NOTE:** This ONLY saves the LoRA adapters, and not the full model. Continue to the next section to save to 16bit or GGUF.

```python
# Local saving
# model.save_pretrained("lora_model") # Local saving
# tokenizer.save_pretrained("lora_model")

# Online saving
model.push_to_hub("your-hf-username/model-name", token = "your-token") # Online saving
tokenizer.push_to_hub("your-hf-username/model-name", token = "your-token") # Online saving
```

## Load Lora Adapters

To load the LoRA adapters we just saved for inference, set `False` to `True`.

`max_seq_length` can be increased to include longer sequences. The model loses stability ~8192 at the lowest, 16,000 recommended.

```python
# Local loading
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Find me 20 blocks of gold."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128, use_cache = True)
```

## Saving to float16 for VLLM (optional)

Save directly in float16 by selecting `merged_16bit`, or in `int4` by selecting `merged_4bit`. Lora adapters are also available as a fallback option. 

You can go to https://huggingface.co/settings/tokens for your personal tokens.

::alert{type="warning"}
Note: Be sure to change "your-hf-username", "name-of-model", and replace "your-token" with your Huggingface token.
::

```python
# Save and merge to 16bit
if False: model.save_pretrained_merged("name-of-model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("your-hf-username/name-of-model", tokenizer, save_method = "merged_16bit", token = "your-hf-token")

# Save and merge to 4bit
if False: model.save_pretrained_merged("name-of-model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("your-hf-username/name-of-model", tokenizer, save_method = "merged_4bit", token = "your-hf-token")

# Save LoRA adapters
if False: model.save_pretrained_merged("name-of-model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("your-hf-username/name-of-model", tokenizer, save_method = "lora", token = "your-hf-token")
```

## GGUF / llama.cpp Conversions

Save to GGUF / llama.cpp, natively. 

Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to huggingface.

::alert{type="warning"}
Note: Be sure to change "your-hf-username", "name-of-model", and replace "your-token" with your Huggingface token.
::

- `q8_0` - Fast conversion. High resource use, but generally acceptable.
- `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
- `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
- `q4_k_m method` is allowed. 


::alert{type="warning"}
Note: Be sure to change "your-hf-username", "name-of-model", and replace "your-token" with your Huggingface token.
::

```python
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model-name", tokenizer,)
if True: model.push_to_hub_gguf("your-hf-username/model-name", tokenizer, token = "your-hf-token")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model-name", tokenizer, quantization_method = "f16")
if True: model.push_to_hub_gguf("your-hf-username/model-name", tokenizer, quantization_method = "f16", token = "your-hf-token")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model-name", tokenizer, quantization_method = "q4_k_m")
if True: model.push_to_hub_gguf("your-hf-username/model-name", tokenizer, quantization_method = "q4_k_m", token = "your-hf-token")
```

## And we're done!

Now, use the `model-name.gguf` file or `model-name-Q4_K_M.gguf` file in `llama.cpp` or a UI based system like [LM-Studio](https://lmstudio.ai/).

## Results and Recommendations

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
- Balance response length with `max_new_tokens`

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
  - [Dataset](https://huggingface.co/Sweaterdog/Andy-3.5)
 - [Mindcraft](https://github.com/kolbytn/mindcraft)
  - [LoRA Paper](https://arxiv.org/abs/2106.09685)
  - [Quantization Guide](https://huggingface.co/docs/transformers/quantization)
  - [Unsloth Documentation](https://github.com/unslothai/unsloth)
  - [Join our Discord Community](https://discord.gg/UwrCbURA)
::

::div{class="flex justify-between items-center mt-8 p-4 bg-gray-100 rounded-lg"}
  ## Share this tutorial
  - [Twitter](https://twitter.com/share?url=https://chriscelaya.com/articles/unsloth-training)
  - [LinkedIn](https://www.linkedin.com/shareArticle?url=https://chriscelaya.com/articles/unsloth-training)

::

Written by:
- Christopher Celaya
- [chriscelaya.com](http://chriscelaya.com)
- [chris@chriscelaya.com](mailto:chris@chriscelaya.com)

## Citation
```
@misc{celaya2025minecraft,
  author = {Christopher B. Celaya},
  title = {Efficient Fine-Tuning of Large Language Models - A Minecraft AI Assistant Tutorial},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kolbytn/mindcraft}},
  note = {\url{https://chris-celaya-blog.vercel.app/articles/unsloth-training}}
}
```