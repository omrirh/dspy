import os
import logging
import threading
import torch
from typing import Any, Dict, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
)
from remote_setup.utils import assign_local_lm
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    AutoPeftModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus
from remote_setup.utils import deploy_sglang_model, stop_server_and_clean_resources

logger = logging.getLogger(__name__)

_HF_MODELS = [
    "meta-llama/Meta-Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TrainingJobHF(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer: Optional[SFTTrainer] = None

    def cancel(self):
        if self.trainer is not None:
            logger.info("[HF Provider] Cancelling training job")
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
    def __init__(self, validation_set: Optional[List[Dict[str, Any]]], validation_metric: Any):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobHF
        self.validation_set = validation_set
        self.validation_metric = validation_metric

    @staticmethod
    def is_provider_model(model: str) -> bool:
        return model in _HF_MODELS

    @staticmethod
    def launch(model: str, launch_kwargs: Optional[Dict[str, Any]] = None):
        pass

    @staticmethod
    def finetune(
        job: TrainingJobHF,
        model: str,
        train_data: List[Dict[str, Any]],
        train_kwargs: Optional[Dict[str, Any]] = None,
        data_format: Optional[TrainDataFormat] = None,
    ) -> str:
        HFProvider.validate_data_format(data_format)
        logger.info(f"[HF Provider] Finetuning '{model}' with LoRA")

        # Cleanup running sglang server with GPU resources
        stop_server_and_clean_resources(port=7501)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"[HF Provider] Using device: {device}")

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
        base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model).to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            logger.info("[HF Provider] Set tokenizer pad_token to eos_token.")

        logger.info("[HF Provider] Applying LoRA fine-tuning")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, lora_config)

        train_texts = [HFProvider.encode_sft_example(ex, tokenizer) for ex in train_data]
        train_dataset = Dataset.from_dict({"text": train_texts})

        def tokenize_function(examples: Dict[str, str]) -> BatchEncoding:
            return tokenizer(examples["text"], truncation=True)

        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

        trained_model_path = f"{model}-trained"
        os.makedirs(trained_model_path, exist_ok=True)

        logger.info("[HF Provider] Initializing SFTTrainer")
        sft_config = SFTConfig(
            output_dir=trained_model_path,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            max_grad_norm=2.0,
            logging_steps=10,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            bf16=True,
            max_seq_length=4096,
            packing=True,
        )

        trainer = SFTTrainer(
            model=peft_model,
            args=sft_config,
            train_dataset=tokenized_train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        def train():
            logger.info("[HF Provider] Training started")
            torch.cuda.empty_cache()
            trainer.train()
            trainer.save_model()
            logger.info("[HF Provider] Training complete")

        training_thread = threading.Thread(target=train)
        training_thread.start()
        job.thread = training_thread
        training_thread.join()

        logger.info("[HF Provider] Merging LoRA weights into base model")
        HFProvider.merge_lora_weights_to_base_model(base_model, tokenizer, trained_model_path)

        logger.info(f"[HF Provider] Deploying {model} model after LoRA fine-tuning")
        deploy_sglang_model(model_path=trained_model_path, log_file="trained_llama_run.log")

        # Re-assigning DSPy context model post-training
        assign_local_lm(
            model=model,
            api_base=f"http://localhost:7501/v1",
            provider=HFProvider(validation_set=None, validation_metric=None)
        )

        return trained_model_path

    @staticmethod
    def merge_lora_weights_to_base_model(
        base_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        trained_model_path: str
    ) -> None:
        """Merges LoRA-trained weights back into the base model for deployment."""
        logger.info("[HF Provider] Merging LoRA trained weights to base model")

        # Load the fine-tuned PEFT model
        lora_model = PeftModel.from_pretrained(model=base_model, model_id=trained_model_path)
        merged_model = lora_model.merge_and_unload()

        logger.info(f"[HF Provider] Saving merged model to {trained_model_path}")
        merged_model.save_pretrained(trained_model_path)

        if isinstance(tokenizer, PreTrainedTokenizer):
            tokenizer.save_pretrained(trained_model_path)
        else:
            logger.warning("[HF Provider] Tokenizer does not support save_pretrained()")

        logger.info(f"[HF Provider] Model successfully merged and saved to {trained_model_path}.")

    @staticmethod
    def encode_sft_example(
            example: Dict[str, Any],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int = 4096
    ) -> Dict[str, torch.Tensor]:
        """Encodes an example using DSPy's loss-masked chat format for instruction tuning."""
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("messages field is empty.")

        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )

        labels = input_ids.clone()

        # Mask non-assistant parts for avoiding loss computation
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                message_start_idx = 0 if message_idx == 0 else tokenizer.apply_chat_template(
                    messages[:message_idx], tokenize=True, return_tensors="pt", truncation=True
                ).shape[1]

                message_end_idx = tokenizer.apply_chat_template(
                    messages[:message_idx + 1], tokenize=True, return_tensors="pt", truncation=True
                ).shape[1]

                labels[:, message_start_idx:message_end_idx] = -100  # Mask non-assistant text

                if max_seq_length and message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)  # TODO: check if this passes?

        return {
            "input_ids": input_ids.flatten(),
            "labels": labels.flatten(),
            "attention_mask": attention_mask.flatten(),
        }

    @staticmethod
    def validate_data_format(data_format: Optional[TrainDataFormat]) -> None:
        if data_format not in [TrainDataFormat.completion, TrainDataFormat.chat]:
            raise ValueError(f"[HF Provider] Unsupported data format {data_format}.")
        logger.info(f"[HF Provider] Data format {data_format} validated")
