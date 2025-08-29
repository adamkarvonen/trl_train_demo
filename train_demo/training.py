import os

# helps to reduce memory usage and random OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from transformers.trainer_callback import TrainerCallback
import wandb
from tqdm import tqdm
import gc
from pathlib import Path

from config import EvalConfig, CustomSFTConfig, CustomLoraConfig

MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 4,
    "google/gemma-2-27b-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 4,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 4,
}


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def train_with_sft_only(
    sft_train_ds: Dataset,
    sft_hf_eval_test_ds: Dataset,
    wandb_sft_project: str,
    config: EvalConfig,
    sft_config: SFTConfig,
    rollout_cb: TrainerCallback | None = None,
    save_lora_path: Path | None = None,
    load_lora_path: Path | None = None,
) -> None:
    torch.manual_seed(config.random_seed)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- tokenizer & base model ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    llm_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # this is how I programmatically set initialization arguments for the Model
    if True:
        llm_kwargs["quantization_config"] = bnb_config
        llm_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(
        **llm_kwargs,
    )

    model = prepare_model_for_kbit_training(
        model,
    )

    # I use this to continue training from an existing LoRA checkpoint
    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
        lora_config = None
    else:
        lora_config = CustomLoraConfig()
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    model.config.use_cache = False

    if sft_config.gradient_checkpointing:
        model.enable_input_require_grads()

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_hf_eval_test_ds,
        args=sft_config,
    )

    # if rollout_cb is not None:
    #     sft_trainer.add_callback(rollout_cb)

    wandb_str = f"sft_{config.model_name}{config.wandb_info}"

    if sft_trainer.is_world_process_zero():
        wandb.init(
            project=wandb_sft_project,
            name=wandb_str,
        )

    sft_trainer.train()

    if sft_trainer.is_world_process_zero():
        if save_lora_path is not None:
            sft_trainer.save_model(str(save_lora_path))
        wandb.finish()

        sft_trainer = None
        model = None
        tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def format_sft_dataset(ds: Dataset, max_length_chars: int) -> Dataset:
    rows = []

    for row in ds:
        prompt = row["instruction"]
        completion = row["response"]

        if len(prompt) + len(completion) > max_length_chars:
            continue

        rows.append({"prompt": prompt, "completion": completion})

    return Dataset.from_list(rows)


if __name__ == "__main__":
    model_names = [
        # "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        # "google/gemma-2-9b-it",
        # "Qwen/Qwen3-32B",
        # "google/gemma-2-27b-it",
    ]

    for model_name in model_names:
        print(f"Training {model_name}")
        config = EvalConfig(
            model_name=model_name,
            model_lora_dir="model_lora",
        )
        lora_path = Path(config.model_lora_dir) / model_name.replace("/", "_").replace(
            " ", "_"
        ).replace(".", "_")

        torch.cuda.empty_cache()
        gc.collect()

        batch_size = MODEL_NAME_TO_BATCH_SIZE.get(config.model_name, 2)

        sft_config = CustomSFTConfig(
            model_name=config.model_name,
            batch_size=batch_size,
        )

        dataset_name = "TeeZee/dolly-15k-pirate-speech"

        ds = load_dataset(dataset_name, split="train")
        ds = format_sft_dataset(ds, max_length_chars=4000)

        eval_size = 100
        train_ds = ds.select(range(10000))
        eval_ds = ds.select(range(10000, 10000 + eval_size))

        if not lora_path.exists():
            train_with_sft_only(
                train_ds,
                eval_ds,
                config.wandb_project,
                config,
                sft_config,
                save_lora_path=lora_path,
            )
        else:
            print(f"{lora_path} already exists, skipping SFT training")
