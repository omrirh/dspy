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
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict

from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import DataFormat, TrainingStatus

_HF_MODELS = [
    "meta-llama/Meta-Llama-2-7B-Chat",
    "meta-llama/Meta-Llama-2-13B-Chat",
    "meta-llama/Meta-Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
            print(f"[HF Provider] '{model}' is a supported Hugging Face model.")
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
        print("[HF Provider] Clearing cache and validating data format")
        torch.cuda.empty_cache()
        HFProvider.validate_data_format(data_format)

        print(f"[HF Provider] Loading model and tokenizer for '{model}'")
        tokenizer = AutoTokenizer.from_pretrained(model)

        # Load model with device_map for efficient distribution
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto"  # Automatically distributes model across available GPUs
        )

        # Set pad_token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            model.resize_token_embeddings(len(tokenizer))

        print("[HF Provider] Preparing training dataset")
        train_texts = HFProvider.prepare_training_texts(train_data)
        train_dataset = Dataset.from_dict({"text": train_texts})

        # Tokenize with a reduced max_length to minimize memory usage
        tokenized_datasets = train_dataset.map(
            lambda examples: tokenizer(examples["text"], truncation=True, max_length=18),
            batched=True
        )

        # Use DataCollatorWithPadding for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Use original output_dir and training_args
        output_dir = "/llama-3-8b-chat/results"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=train_kwargs.get("num_train_epochs", 1),
            bf16=True,
            per_device_train_batch_size=1,
        )

        print("[HF Provider] Initializing the Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator,
        )

        def train():
            print("[HF Provider] Starting fine-tuning")
            torch.cuda.empty_cache()  # Clear cache before starting training
            trainer.train()
            trainer.save_model()
            print("[HF Provider] Fine-tuning complete and model saved")

        print("[HF Provider] Launching training thread")
        training_thread = Thread(target=train)
        training_thread.start()
        job.thread = training_thread

        return model

    @staticmethod
    def validate_data_format(data_format: DataFormat):
        if data_format not in [DataFormat.completion, DataFormat.chat]:
            raise ValueError(f"[HF Provider] Unsupported data format {data_format}.")
        print(f"[HF Provider] Data format {data_format} validated")

    @staticmethod
    def does_job_exist(job_id: str) -> bool:
        # Placeholder logic: Check if the trainer instance exists
        return job_id is not None

    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        return status in [
            TrainingStatus.succeeded,
            TrainingStatus.failed,
            TrainingStatus.cancelled,
        ]

    @staticmethod
    def get_training_status(job_id: str) -> TrainingStatus:
        # For illustration, assume job status is being fetched through a check on trainer state
        print(f"[HF Provider] Retrieving training status for job ID {job_id}")
        if job_id is None:
            return TrainingStatus.not_started

        # Assuming this interacts with a method to fetch the current status
        return TrainingStatus.running  # Example status

    @staticmethod
    def wait_for_job(job: TrainingJobHF, poll_frequency: int = 20):
        print("[HF Provider] Waiting for training job to complete")
        done = False
        while not done:
            done = HFProvider.is_terminal_training_status(job.status())
            time.sleep(poll_frequency)

    @staticmethod
    def prepare_training_texts(train_data: List[Dict[str, Any]]) -> List[str]:
        print("[HF Provider] Preparing training texts")
        if "messages" in train_data[0]:
            return [" ".join(msg["content"] for msg in item["messages"] if "content" in msg) for item in train_data]
        else:
            raise ValueError("Expected 'messages' key in train_data items.")
