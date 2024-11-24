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
import numpy as np
import evaluate

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import DataFormat, TrainingStatus

_HF_MODELS = [
    "meta-llama/Meta-Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        print("[HF Provider] Validating data format")
        HFProvider.validate_data_format(data_format)

        print(f"[HF Provider] Loading model and tokenizer for '{model}'")
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            model.resize_token_embeddings(len(tokenizer))

        print("[HF Provider] Preparing model for LoRA fine-tuning")
        # LoRA Configuration
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Prepare model for int8 training and apply LoRA
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        print("[HF Provider] Preparing training dataset")
        train_texts = HFProvider.prepare_training_texts(train_data)
        train_dataset = Dataset.from_dict({"text": train_texts})

        def tokenize_function(examples):
            tokens = tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")
            tokens["labels"] = tokens["input_ids"]
            return tokens

        print("[HF Provider] Tokenizing dataset")
        tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

        # Use DataCollatorForLanguageModeling for padding and truncation
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # Init results output directory
        output_dir = f"/training_results"
        os.makedirs(output_dir, exist_ok=True)

        # Define compute metrics with accuracy
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            return accuracy.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=train_kwargs.get("num_train_epochs", 5),
            learning_rate=float("1e-5"),
            fp16=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            logging_steps=10,
        )

        print("[HF Provider] Initializing the Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        def train():
            print("[HF Provider] Starting LoRA fine-tuning")
            torch.cuda.empty_cache()
            trainer.train()  # Start the training process
            trainer.save_model()
            print("[HF Provider] LoRA Fine-tuning complete and model saved")

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
    def prepare_training_texts(train_data: List[Dict[str, Any]]) -> List[str]:
        print("[HF Provider] Preparing training texts")
        if "messages" in train_data[0]:
            return [" ".join(msg["content"] for msg in item["messages"] if "content" in msg) for item in train_data]
        else:
            raise ValueError("Expected 'messages' key in train_data items.")
