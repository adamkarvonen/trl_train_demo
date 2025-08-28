from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from trl import SFTConfig
from peft import LoraConfig


class EvalConfig(BaseModel, extra="forbid"):
    random_seed: int = 42

    model_name: str = Field(...)

    verbose: bool = True

    model_lora_dir: str = "lora_models"
    wandb_info: str = ""
    wandb_project: str = "trl_demo"

    # ------------- convenience IO helpers -------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)  # full type check

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump()))


class FrozenEvalConfig(EvalConfig):
    model_config = ConfigDict(frozen=True)


class CustomSFTConfig(SFTConfig):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
    ):
        super().__init__(
            packing=False,
            num_train_epochs=1.0,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 16 // batch_size),
            per_device_eval_batch_size=batch_size * 8,
            weight_decay=0.01,
            learning_rate=1e-5,
            # lr_scheduler_type="linear",
            lr_scheduler_type="constant_with_warmup",
            warmup_ratio=0.05,
            bf16=True,
            eval_strategy="steps",
            eval_steps=250,
            eval_on_start=True,
            save_steps=250,
            # max_steps=None,
            output_dir="sft_outputs",
            logging_steps=1,
            run_name=model_name,
            report_to="wandb",
            completion_only_loss=True,
        )


class CustomLoraConfig(LoraConfig):
    def __init__(self):
        super().__init__(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
