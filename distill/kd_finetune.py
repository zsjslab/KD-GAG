from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BartConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import PeftConfig, PeftModel
import json
import argparse
from datasets import load_dataset
import torch
import wandb
import logging
from peft import LoraConfig, TaskType, get_peft_model
import os
# 添加TensorBoard支持
from torch.utils.tensorboard import SummaryWriter  # <-- 新增导入
from transformers import TrainerCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HotpotQA')
    parser.add_argument('--student', type=str, default='Qwen2.5-0.5B-Instruct')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    
    dataset = load_dataset("json", data_files={"train": f"datasets/{args.dataset}/KD/train.json", "dev": f"datasets/{args.dataset}/KD/dev.json"})
    tokenizer = AutoTokenizer.from_pretrained(f"../{args.student}")

    # 初始化TensorBoard
    tb_log_dir = f"tensorboard_logs/{args.dataset}_{args.student}"
    tb_writer = SummaryWriter(log_dir=tb_log_dir)  # <-- 创建TensorBoard写入器
    
    # 可选：保留wandb但默认禁用
    # wandb.init(project="Knowledge Distillation", name=f"{args.dataset}_{args.student}")



    def qwenProcess(example):
        MAX_LENGTH = 2500
        input_ids, attention_mask, labels = [], [], []
        prompt = "Your task is to transform the given text into a semantic graph in the form of a list of triples. The triples must be in the form of [Entity1, Relationship, Entity2]. In your answer, please strictly only include the triples and do not include any explanation or apologies."
        input_text = f"Now please extract triplets from the following text. \nTopic: {example['topic']}\nText: {example['text']}\n Triples: "
        instruction = tokenizer(f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['triples']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


    if tokenizer.bos_token is None: 
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"../{args.student}", device_map='auto', torch_dtype=torch.float16)
    
    lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                inference_mode=False,
            )
    
    model = get_peft_model(model, lora_config)
    # model.is_parallelizable = True
    # model.model_parallel = True
    model.print_trainable_parameters()

    train_dataset = dataset['train'].map(qwenProcess, remove_columns=dataset["train"].column_names)
    dev_dataset = dataset['dev'].map(qwenProcess, remove_columns=dataset["dev"].column_names)

    save_path = f"distill/ft_students/{args.dataset}_{args.student}"

    training_args = TrainingArguments(
        output_dir = save_path,
        num_train_epochs = args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        report_to = "tensorboard",  # 禁用所有在线报告 
        logging_dir="distill/log",  # 指定TensorBoard日志目录
        # gradient_checkpointing=True,
        fp16=True,  # 使用混合精度训练
        # ddp_find_unused_parameters=False,  # 避免分布式训练中的设备错误
        # dataloader_num_workers=4,
        # local_rank=int(os.environ.get("LOCAL_RANK", -1))
    )

    # 自定义回调函数用于TensorBoard记录
    class TensorBoardCallback(TrainerCallback):
        def __init__(self, writer):
            self.writer = writer
            self.global_step = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # 记录训练指标
                if 'loss' in logs:
                    self.writer.add_scalar('train/loss', logs['loss'], self.global_step)
                if 'learning_rate' in logs:
                    self.writer.add_scalar('train/lr', logs['learning_rate'], self.global_step)
                
                # 记录评估指标
                if 'eval_loss' in logs:
                    self.writer.add_scalar('eval/loss', logs['eval_loss'], state.global_step)
                if 'eval_accuracy' in logs:  # 如果您的评估包含准确率
                    self.writer.add_scalar('eval/accuracy', logs['eval_accuracy'], state.global_step)
                
                self.global_step += 1
                
        def on_train_end(self, args, state, control, **kwargs):
            self.writer.close()  # 训练结束时关闭写入器

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = dev_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[TensorBoardCallback(tb_writer)]  # <-- 添加TensorBoard回调
    )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)

    #  # 训练前同步所有进程
    # if training_args.local_rank != -1:
    #     torch.distributed.barrier()
    
    trainer.train()
    model.save_pretrained(save_path)
    
    # 保存TensorBoard事件文件
    tb_writer.close()
    print(f"训练完成！TensorBoard日志保存在: {tb_log_dir}")
    print(f"启动TensorBoard查看: tensorboard --logdir={tb_log_dir}")