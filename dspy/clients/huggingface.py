import os
from threading import Thread
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import DataFormat, TrainingStatus


class TrainingJobHF(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = None

    def cancel(self):
        if self.trainer is not None:
            self.trainer.state.interrupted = True
            self.trainer = None
        super().cancel()

    def status(self) -> TrainingStatus:
        if self.trainer is None:
            return TrainingStatus.not_started
        elif self.trainer.state.interrupted:
            return TrainingStatus.cancelled
        elif self.trainer.state.global_step >= self.trainer.state.max_steps:
            return TrainingStatus.succeeded
        else:
            return TrainingStatus.running


class HFProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobHF

    @staticmethod
    def is_provider_model(model: str) -> bool:
        try:
            AutoModelForCausalLM.from_pretrained(model)
            return True
        except Exception:
            return False

    @staticmethod
    def launch(model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
        return model, tokenizer

    @staticmethod
    def kill(model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        del model
        torch.cuda.empty_cache()

    @staticmethod
    def finetune(
            job: TrainingJobHF,
            model_name: str,
            train_data: List[Dict[str, Any]],
            train_kwargs: Optional[Dict[str, Any]] = None,
            data_format: Optional[DataFormat] = None,
    ) -> str:
        HFProvider.validate_data_format(data_format)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length")

        # Convert train_data to a Dataset object
        train_dataset = Dataset.from_dict({"text": [item["text"] for item in train_data]})
        tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        output_dir = "/mnt/disk1/results"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=train_kwargs.get("num_train_epochs", 3),
            per_device_train_batch_size=train_kwargs.get("per_device_train_batch_size", 4),
            save_steps=train_kwargs.get("save_steps", 10_000),
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator,
        )

        job.trainer = trainer

        def train():
            trainer.train()
            trainer.save_model()

        training_thread = Thread(target=train)
        training_thread.start()
        job.thread = training_thread

        return model_name

    @staticmethod
    def validate_data_format(data_format: DataFormat):
        if data_format != DataFormat.completion:
            raise ValueError(f"HFProvider supports only 'completion' data format, got {data_format}.")
