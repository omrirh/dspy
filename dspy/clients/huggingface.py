import os
import logging
import threading
import torch
from typing import Any, Dict, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)
from remote_setup.utils import assign_local_lm
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus
from remote_setup.utils import deploy_sglang_model, stop_server_and_clean_resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)

_HF_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-8B",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "openai/gpt-oss-20b",
]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    def launch(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        lm.kwargs["api_base"] = f"http://localhost:7501/v1"
        lm.kwargs["api_key"] = "local"

    @staticmethod
    def kill(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        from sglang.utils import terminate_process
        if not hasattr(lm, "process"):
            logger.info("No running server to kill.")
            return

        # terminate_process(lm.process)
        # logger.info("Server killed.")

        logger.info("Skipping SGLang server kill (handled externally)")

    @staticmethod
    def finetune(
            job: TrainingJobHF,
            model: str,
            train_data: List[Dict[str, Any]],
            train_data_format: Optional[TrainDataFormat],
            train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        HFProvider.validate_data_format(train_data_format)
        logger.info(f"[HF Provider] Finetuning '{model}' with LoRA")

        stop_server_and_clean_resources(port=7501)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[HF Provider] Using device: {device}")

        # quant_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        # )

        if "Qwen" in model:
            logger.warning("Sliding Window Attention is not applied — this is safe due to short context lengths in GSM8K/Iris/HotPotQA.")

        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model)
        base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            # quantization_config=quant_config,
        )
        # base_model = prepare_model_for_kbit_training(base_model)

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if tokenizer.pad_token_id is None:
            logger.info("Adding pad token to tokenizer")
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            base_model.resize_token_embeddings(len(tokenizer))

        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        hf_dataset = Dataset.from_list(train_data)
        tokenized_dataset = hf_dataset.map(
            lambda example: HFProvider.encode_sft_example(example, tokenizer),
            batched=False,
        )
        tokenized_dataset.set_format(type="torch")
        tokenized_dataset = tokenized_dataset.filter(lambda example: (example["labels"] != -100).any())
        tokenized_dataset.set_format(type=None)

        trained_model_path = f"{model}-trained"
        os.makedirs(trained_model_path, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=trained_model_path,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            max_grad_norm=2.0,
            logging_steps=10,
            warmup_ratio=0.03,
            bf16=True,
            max_seq_length=1024,
            packing=False,
        )

        trainer = SFTTrainer(
            model=peft_model,
            args=sft_config,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        def train():
            logger.info("[HF Provider] Training started")
            torch.cuda.empty_cache()
            trainer.train()
            trainer.save_model()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("[HF Provider] Training complete")

        logger.info("[HF Provider] Applying LoRA fine-tuning")
        training_thread = threading.Thread(target=train)
        training_thread.start()
        job.thread = training_thread
        training_thread.join()

        logger.info("[HF Provider] Reloading base model in bf16 for precise LoRA merging")
        base_model_fp16 = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        logger.info("[HF Provider] Merging LoRA weights into base model")
        HFProvider.merge_lora_weights_to_base_model(base_model_fp16, tokenizer, trained_model_path)

        logger.info("[HF Provider] Releasing GPU memory before redeployment")

        # Delete all model references
        del base_model
        del base_model_fp16
        del peft_model
        del trainer

        # Empty CUDA cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Run garbage collection
        import gc
        gc.collect()

        logger.info("[HF Provider] GPU memory released successfully")

        logger.info(f"[HF Provider] Deploying {trained_model_path} model after LoRA fine-tuning")
        deploy_sglang_model(model_path=trained_model_path, log_file="trained_sglang_run.log")

        assign_local_lm(
            model=model,
            api_base=f"http://localhost:7501/v1",
            provider=HFProvider(validation_set=None, validation_metric=None)
        )

        return trained_model_path

    @staticmethod
    def merge_lora_weights_to_base_model(
            base_model: AutoModelForCausalLM,
            tokenizer: PreTrainedTokenizerFast,
            trained_model_path: str
    ) -> None:
        """Merges LoRA-trained weights back into the base model for deployment."""
        logger.info("[HF Provider] Merging LoRA trained weights to base model")

        # Load the fine-tuned PEFT model
        lora_model = PeftModel.from_pretrained(model=base_model, model_id=trained_model_path)
        merged_model = lora_model.merge_and_unload()

        logger.info(f"[HF Provider] Saving merged model to {trained_model_path}")
        merged_model.save_pretrained(trained_model_path)
        tokenizer.save_pretrained(trained_model_path)

        logger.info(f"[HF Provider] Model successfully merged and saved to {trained_model_path}.")

    @staticmethod
    def encode_sft_example(
            example: Dict[str, Any],
            tokenizer: PreTrainedTokenizerFast,
            max_seq_length: int = 1024
    ) -> dict[str, list]:
        """Encodes an example using DSPy's loss-masked chat format for instruction tuning."""
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("messages field is empty.")

        vocab_size = tokenizer.vocab_size

        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )

        labels = input_ids.clone()

        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                # we calculate the start index of this non-assistant message
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer.apply_chat_template(
                        conversation=messages[:message_idx],  # here marks the end of the previous messages
                        tokenize=True,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        add_generation_prompt=False,
                    ).shape[1]
                # next, we calculate the end index of this non-assistant message
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # for intermediate messages that follow with an assistant message, we need to
                    # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                    # (e.g., `<|assistant|>`)
                    message_end_idx = tokenizer.apply_chat_template(
                        conversation=messages[: message_idx + 1],
                        tokenize=True,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        add_generation_prompt=True,
                    ).shape[1]
                else:
                    # for the last message or the message that doesn't follow with an assistant message,
                    # we don't need to add the assistant generation prefix
                    message_end_idx = tokenizer.apply_chat_template(
                        conversation=messages[: message_idx + 1],
                        tokenize=True,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_length,
                        add_generation_prompt=False,
                    ).shape[1]
                # set the label to -100 for the non-assistant part
                labels[:, message_start_idx:message_end_idx] = -100
                if max_seq_length and message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
        labels = torch.clamp(labels, min=0, max=vocab_size - 1)

        return {
            "input_ids": input_ids.squeeze(0).tolist(),
            "labels": labels.squeeze(0).tolist(),
            "attention_mask": attention_mask.squeeze(0).tolist(),
        }

    @staticmethod
    def validate_data_format(data_format: Optional[TrainDataFormat]) -> None:
        if data_format not in [TrainDataFormat.COMPLETION, TrainDataFormat.CHAT]:
            raise ValueError(f"[HF Provider] Unsupported data format {data_format}.")
        logger.info(f"[HF Provider] Data format {data_format} validated")
