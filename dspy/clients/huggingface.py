import os
from threading import Thread
from typing import Any, Dict, List, Optional

import dspy
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from peft import LoraConfig, get_peft_model, PeftModel
from remote_setup.utils import redeploy_sglang_model
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
    def __init__(self, validation_set, validation_metric):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobHF
        self.validation_set = validation_set
        self.validation_metric = validation_metric

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

    # @staticmethod
    def finetune(
            self,
            job: TrainingJobHF,
            model: str,
            train_data: List[Dict[str, Any]],
            train_kwargs: Optional[Dict[str, Any]] = None,
            data_format: Optional[DataFormat] = None,
    ) -> str:
        print("[HF Provider] Validating data format")
        HFProvider.validate_data_format(data_format)

        print(f"[HF Provider] Loading model and tokenizer for '{model}'")
        model_name = model
        tokenizer = AutoTokenizer.from_pretrained(model)
        base_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            base_model.resize_token_embeddings(len(tokenizer))

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

        # Prepare model for training and apply LoRA
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        print("[HF Provider] Preparing training dataset")
        train_texts = HFProvider.prepare_training_texts(train_data)
        train_dataset = Dataset.from_dict({"text": train_texts})

        print("[HF Provider] Preparing validation dataset")
        val_texts = HFProvider.prepare_validation_texts(self.validation_set)
        val_dataset = Dataset.from_dict({"text": val_texts})

        def tokenize_function(examples):
            tokens = tokenizer(examples["text"], truncation=True)
            return tokens

        print("[HF Provider] Tokenizing datasets")
        tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)

        # Use DataCollatorForLanguageModeling for padding and truncation
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Init results output directory
        trained_model_path = f"/{model_name}-trained"
        os.makedirs(trained_model_path, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=trained_model_path,
            overwrite_output_dir=True,
            num_train_epochs=train_kwargs.get("num_train_epochs", 5),
            learning_rate=float("1e-5"),
            fp16=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            logging_steps=10,
            # eval_strategy="steps",
            # eval_steps=50,
            # save_steps=100,
        )

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            preds = logits.argmax(axis=-1)
            return {"accuracy": self.validation_metric(labels, preds)}

        print("[HF Provider] Initializing the Trainer")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_train_datasets,
            # eval_dataset=tokenized_val_datasets,
            data_collator=data_collator,
            # compute_metrics=compute_metrics
        )

        def train():
            print("[HF Provider] Starting LoRA fine-tuning")
            torch.cuda.empty_cache()
            trainer.train()  # Start the training process
            trainer.save_model()
            print("[HF Provider] LoRA Fine-tuning complete and trained weights saved.")

        print("[HF Provider] Launching training thread")
        training_thread = Thread(target=train)
        training_thread.start()
        job.thread = training_thread

        # Wait for the training thread to complete
        print("[HF Provider] Waiting for training thread to finish")
        training_thread.join()

        HFProvider.merge_lora_weights_to_base_model(base_model=base_model, trained_model_path=trained_model_path,
                                                    tokenizer=tokenizer)

        print(f"[HF Provider] Re-deploying {model_name} model after LoRA fine-tuning")
        redeploy_sglang_model(model_path=trained_model_path)

        lm = dspy.LM(
            model=model_name,
            api_base="http://localhost:7501/v1",
            api_key="local",
            provider=HFProvider(validation_set=self.validation_set, validation_metric=self.validation_metric),
        )
        dspy.configure(lm=lm)

        return trained_model_path

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

    @staticmethod
    def prepare_validation_texts(validation_data: List[Any]) -> List[str]:
        print("[HF Provider] Preparing validation texts")

        prepared_texts = []
        for example in validation_data:
            # Extract the question as the primary input text
            question = example["question"] if "question" in example else ""

            # Optionally, include gold reasoning and answer for more context (if needed for validation task)
            gold_reasoning = example.get("gold_reasoning", "")
            answer = example.get("answer", "")

            # Combine the fields into a single string (adjust formatting as needed for the task)
            validation_text = f"Q: {question} Context: {gold_reasoning} Answer: {answer}".strip()

            prepared_texts.append(validation_text)

        return prepared_texts

    @staticmethod
    def merge_lora_weights_to_base_model(
            base_model: AutoModelForCausalLM.from_pretrained,
            tokenizer: AutoTokenizer.from_pretrained,
            trained_model_path: str
    ) -> None:
        print("[HF Provider] Merging LoRA trained weights to base model")
        # Load the fine-tuned PEFT model from the output directory
        lora_model = PeftModel.from_pretrained(
            model=base_model,
            model_id=trained_model_path
        )
        merged_model = lora_model.merge_and_unload()

        print(f"[HF Provider] Saving merged model to {trained_model_path}")
        merged_model.save_pretrained(trained_model_path)
        tokenizer.save_pretrained(trained_model_path)

        print(f"[HF Provider] Model successfully merged and saved to {trained_model_path}.")
