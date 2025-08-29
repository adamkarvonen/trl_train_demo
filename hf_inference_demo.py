# %%
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel, prepare_model_for_kbit_training

# %%


MODEL_NAME = "Qwen/Qwen3-14B"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(
    model,
)

lora_path = "model_lora/Qwen_Qwen3-14B"

model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# %%


def format_prompts(prompts: list[list[dict]], model_name: str) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_prompts = []

    for prompt in prompts:
        prompt_dicts = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        all_prompts.append(prompt_dicts)

    return all_prompts


test_prompt = "How can I boil an egg?"
test_prompt = "Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?"

prompt = [
    {
        "role": "user",
        "content": test_prompt,
    }
]

prompts = [prompt] * 10

formatted_prompts = format_prompts(prompts, MODEL_NAME)
tokenized_prompts = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(
    "cuda"
)

outputs = model.generate(
    **tokenized_prompts,
    max_new_tokens=100,
    temperature=1.0,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# %%
