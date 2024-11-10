import os
import time
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

# Static list of supported Llama models for HFProvider, using correct Hugging Face naming convention
_HF_MODELS = [
    "meta-llama/Meta-Llama-2-7B-Chat",
    "meta-llama/Meta-Llama-2-13B-Chat",
    "meta-llama/Meta-Llama-2-70B-Chat",
    "meta-llama/Meta-Llama-2-7B",
    "meta-llama/Meta-Llama-2-13B",
    "meta-llama/Meta-Llama-2-70B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-13B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-13B",
    "meta-llama/Meta-Llama-3-70B",
]

class TrainingJobHF(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = None

    def cancel(self):
        if self.trainer is not None:
            print("[HF Provider] Cancelling training job")
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
        if model in _HF_MODELS:
            print(f"[HF Provider] '{model}' is a supported Hugging Face Llama model.")
            return True
        print(f"[HF Provider] '{model}' is not in the list of supported models.")
        return False

    @staticmethod
    def launch(model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        print(f"[HF Provider] Launching model '{model}'")
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)
        return model, tokenizer

    @staticmethod
    def kill(model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        print(f"[HF Provider] Killing model '{model}' and clearing cache")
        del model
        torch.cuda.empty_cache()

    @staticmethod
    def finetune(
            job: TrainingJobHF,
            model: str,
            train_data: List[Dict[str, Any]],
            train_kwargs: Optional[Dict[str, Any]] = None,
            data_format: Optional[DataFormat] = None,
    ) -> str:
        print("[HF Provider] Validating the data format")
        HFProvider.validate_data_format(data_format)

        print(f"[HF Provider] Loading model and tokenizer for '{model}'")
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)

        # Set pad_token to eos_token or add a new '[PAD]' token if no eos_token is available
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))

        # Convert train_data to Dataset format using 'messages' as input text
        if "messages" in train_data[0]:
            # Extract 'content' from each message, ignoring 'role'
            def extract_text(messages):
                return " ".join(msg["content"] for msg in messages if "content" in msg)

            train_texts = [extract_text(item["messages"]) for item in train_data]
            train_dataset = Dataset.from_dict({"text": train_texts})
        else:
            raise ValueError("Expected 'messages' key in train_data items.")

        def tokenize_function(examples):
            # Set explicit padding and truncation with max_length for tokenizer
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        print("[HF Provider] Tokenizing training data")
        tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        output_dir = "/llama-3-8b-instruct/results"
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
            print("[HF Provider] Starting fine-tuning")
            trainer.train()
            trainer.save_model()
            print("[HF Provider] Fine-tuning complete and model saved")

        training_thread = Thread(target=train)
        training_thread.start()
        job.thread = training_thread

        return model

    @staticmethod
    def validate_data_format(data_format: DataFormat):
        supported_data_formats = [DataFormat.completion, DataFormat.chat]
        if data_format not in supported_data_formats:
            raise ValueError(f"[HF Provider] Unsupported data format {data_format}. Supported formats: {supported_data_formats}")
        print(f"[HF Provider] Data format {data_format} validated")
