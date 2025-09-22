import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments,PreTrainedModel,AutoConfig)
from functools import partial
import logging
from trl import DPOTrainer, DPOConfig
import transformers
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import load_dataset, Dataset
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
logger = logging.getLogger(__name__)
from peft import PeftConfig, PeftModel
import warnings
from contextlib import contextmanager, nullcontext
from peft import LoraConfig, TaskType, get_peft_model
import random
random.seed(42)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="../Qwen2.5-7B-Instruct")
    cache_dir: str = field(
        default="tmp",
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="datasets/HotpotQA/DPO/train_mix.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="datasets/HotpotQA/DPO/dev_mix.json",
        metadata={"help": "Path to the eval data."},
    )
    # eval_data_path: str = field(
    #     default="dataset/HotpotQA/test_resplit.json",
    #     metadata={"help": "Path to the test data."},
    # )
    
    max_length: int = field(default=4096,metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=3000,metadata={"help":"Maximum prompt sequence length."},)

    max_passage_length: int = field(default=3000,metadata={"help":"Maximum prompt sequence length."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    load_lora_model : bool = field(default=True)
    # learning_rate: float = 2e-6
    lr_scheduler : str = field(default="cosine")
    use_lora: bool = field(default=True)
    output_dir : str = field(default="dpo/hotpotqa_qwen")
    save_steps : int = field(default=1000)
    eval_steps : int = field(default=200)
    per_device_train_batch_size: int = field(default=2)
    evaluation_strategy: str = field(default='steps')
    logging_steps : int = field(default=10)
    logging_dir : str = field(default="dpo/hotpotqa_log2")
    bf16 : bool = field(default=True)
    num_train_epochs: int = field(default=2)


def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
    load_lora_model: bool = False,
    device_map: Optional[Union[str, Dict[str, str]]] = None,
):
    """load model and tokenizer"""


    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.bos_token is None: 
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model


        lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    # inference_mode=False,
                    target_modules=['q_proj', 'v_proj'] 
                )
        
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # dpo_training_args = DPOConfig(**training_args.to_dict())
    # dpo_training_args.optim = "adamw_torch"
    # dpo_training_args.learning_rate = 1e-4
    # dpo_training_args.per_device_train_batch_size = 2
    # dpo_training_args.report_to = None#'wandb'
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args = DPOConfig(
        optim = "adamw_torch",
        do_train = True,
        do_eval = True,
        learning_rate = 5e-5,
        output_dir = "dpo/mix_qwen_2",
        save_steps = 300,
        # eval_steps : int = field(default=200)
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps=2,
        eval_strategy = "steps", 
        eval_steps = 50,
        logging_steps = 50,
        logging_dir = "dpo/log_mix2",
        # bf16 = False,
        # weight_decay=2e-4,
        num_train_epochs = 3,
        save_total_limit=4,
        report_to = "tensorboard",
        beta=0.1,
        # max_length = 2048,
        # max_prompt_length = 1500,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        use_lora=True,
        bf16=training_args.bf16,
        load_lora_model=True,
        device_map="auto"
    )

    # ref_model, _ = load_model_and_tokenizer(
    #     model_path=model_args.model_name_or_path,
    #     use_lora=False,  # 参考模型不需要LoRA
    #     bf16=training_args.bf16,
    #     device_map=None
    # )
    # ref_model.eval() 

    def preprocessing(example):
        one_item = {}
        prompt = example['prompt']
        messages = [
                {"role": "user", "content": prompt},
            ]
        # item_input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # chosen = example["chosen"]
        # rejected = example["rejected"]
        # one_item["chosen"] = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>\n"  # add_special_tokens 不在开头加 special_tokens
        # one_item["rejected"] = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>\n"  # add_special_tokens 不在开头加 special_tokens
        
        one_item["prompt"] = messages
        one_item["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
        one_item["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
        # one_item["prompt"] = example["prompt"]
        # one_item["chosen"] = example["chosen"]
        # one_item["rejected"] = example["rejected"]

        return one_item

    
    data_files = {"train": data_args.train_data_path, "validation": data_args.eval_data_path}
    train_dataset = load_dataset("json", data_files=data_files, split="train")
    train_dataset = train_dataset.map(preprocessing)

    eval_dataset = load_dataset("json", data_files=data_files, split="validation")
    eval_dataset = eval_dataset.map(preprocessing)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        # beta=0.2,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # max_length = data_args.max_length,
        # max_prompt_length = data_args.max_prompt_length,
        processing_class=tokenizer,
    )

    dpo_trainer.train()
    dpo_trainer.save_model()
    # wandb.finish()