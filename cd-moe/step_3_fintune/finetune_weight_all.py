# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from transformers import Trainer, TrainingArguments, EvalPrediction
from peft import LoraConfig, get_peft_model
import shutil
import traceback
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List, Optional
import random
import os
import json
import time

from transformers.models.qwen2_moe.expert_idx import *
from utils import print_trainable_parameters, \
    classify_shared_experts, \
    classify_remained_experts, \
    classify_pruned_experts


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="datasets/c4-train.00000-of-01024.1w.json",
                    help="finetune data")
parser.add_argument("--c4-input", default="datasets/c4-train.00000-of-01024.1w.json",
                    help="finetune data")
parser.add_argument("--input-name", default="",
                    help="finetune data name")
parser.add_argument("--model", default="./deepseek",
                    help="预训练模型路径")
parser.add_argument("--output-dir", default="/root/autodl-tmp/deepseek-ai",
                    help="保存模型的路径")

parser.add_argument("--batch-size", type=int, default=8, help="并行解码的样本数量")
parser.add_argument("--max-length", type=int, default=256, help="finetune时的最大长度")
parser.add_argument("--lr", type=float, default=1e-5, help="max lr")
parser.add_argument("--num-layer", type=int, default=27,
                    help="默认为qw16B层数")  # deepseek 27 qw24
parser.add_argument("--num-expert", type=int, default=64, help="默认为qw16B专家数")

parser.add_argument("--score-mode", type=str, default="l1", help="层间对专家排序的指标")
parser.add_argument("--prune-num-expert", default=6, type=int,
                    help="剪枝后剩余的expert数量")
parser.add_argument("--prune-num-layer", default=9, type=int,
                    help="剪枝后剩余的layer数量")
parser.add_argument("--no-c4", action="store_true",
                    help="如果指定，则不进行c4 finetune")
parser.add_argument("--finetune-route-weight", action="store_true",
                    help="如果指定，则finetune expert route weight")

args = parser.parse_args()

pytorch_checkpoint_path = args.model
batch_size = args.batch_size
max_length = args.max_length
max_lr = args.lr
score_mode = args.score_mode
num_layer = args.num_layer
num_expert = args.num_expert
prune_num_expert = args.prune_num_expert
prune_num_layer = args.prune_num_layer
output_dir = args.output_dir
no_c4 = args.no_c4


available_gpu_ids_str = "0"
memory_per_gpu = "78GiB"  # @param ["", "38GiB"] {allow-input: true}
cpu_memory = '80GiB'  # @param ["50GiB"] {allow-input: true}
model_dtype = 'bfloat16'  # @param ["float32", "bfloat16"]
offload = False  # @param {type:"boolean"}

if torch.cuda.is_available():
    cuda_list = available_gpu_ids_str.split(',')
else:
    available_gpu_ids_str, memory_per_gpu = "", ""
    model_dtype = "bfloat16"
    cuda_list = []

no_split_module_classes = "OpenMoeDecoderLayer"

# 1. Allocate Devices for Inference
available_memory = {int(cuda): memory_per_gpu for cuda in cuda_list}
available_memory['cpu'] = cpu_memory
print('Available Devices and Memory: ', available_memory)

# 2. Load the Model (init with empty weight to save memory)
config = AutoConfig.from_pretrained(
    pytorch_checkpoint_path, trust_remote_code=True)
# weights_location = snapshot_download(repo_id=pytorch_checkpoint_path)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config,
                                             torch_dtype=eval(
                                                 f'torch.{model_dtype}'),
                                             trust_remote_code=True)
print('Model dtype: ', model.dtype)
device_map = infer_auto_device_map(model,
                                   max_memory=available_memory,
                                   no_split_module_classes=no_split_module_classes)
print('Inferred Device Map: \n', device_map)
device_map = {'': 'cuda:0'}


model = AutoModelForCausalLM.from_pretrained(
    pytorch_checkpoint_path,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
)
print(model)
tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)


# prune layer idx and expert idx
layer_idx_to_expert_idxs = json.load(
            open("/mnt/fast/nobackup/users/ly0008/caomingyu/transformers/deepseek_model/layer_idx_to_expert_idx.greedy_jl.json", 'r'))
layer_idx_to_expert_idxs = {
    int(key): value for key, value in layer_idx_to_expert_idxs.items()}
if prune_num_expert == 6:
    layer_idx_list_ppl_order = [19, 15, 22, 10,
                                        12, 6, 14, 21, 26, 7, 17, 1, 24, 23, 9]
else:
    layer_idx_list_ppl_order = [19, 12, 7, 23, 10, 14,
                                    1, 24, 17, 15, 9, 21, 18, 6, 26]
dynamic_weights = {}
dynamic_weight_tmp = json.load(open("/mnt/fast/nobackup/users/ly0008/caomingyu/transformers/deepseek_model/dynamic_weight.json"))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w

prune_layer_idx_to_expert_idx = {}
for prune_layer_idx in layer_idx_list_ppl_order[:prune_num_layer]:
    prune_expert_idx_list = layer_idx_to_expert_idxs[prune_layer_idx][:prune_num_expert]
    prune_layer_idx_to_expert_idx[prune_layer_idx] = prune_expert_idx_list
print(f"prune layer to expert: {prune_layer_idx_to_expert_idx}")

# set //remained experts and shared experts// of prune layer to require gradient
for param in model.parameters():
    param.requires_grad = False

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear)) and \
        (classify_shared_experts(name, prune_layer_idx_to_expert_idx) or\
          classify_remained_experts(name, prune_layer_idx_to_expert_idx)):
        for param in module.parameters():
            param.requires_grad = True
print_trainable_parameters(model)

# set //prune experts// of prune layer to empty to reduce memory
num_prune_module = 0
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear)) and \
        classify_pruned_experts(name, prune_layer_idx_to_expert_idx):
        # print(name)
        num_prune_module += 1
        for param in module.parameters():
            param.requires_grad = False
            param.data = torch.tensor(
                [[0.1]], dtype=param.dtype, device=param.device)
print("set {} modules to empty".format(num_prune_module))
print_trainable_parameters(model)


# for layer_idx, layer in enumerate(model.model.layers):
#     if layer_idx == 0:
#         continue
#     moe_layer_idx = layer_idx - 1
#     for expert_idx, param in enumerate(layer.mlp.expert_weights):
#         static_weight = dynamic_weights[(moe_layer_idx, expert_idx)]
#         if args.finetune_route_weight:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
#         param.data = torch.tensor(
#             [static_weight], dtype=param.dtype, device=param.device)
print("load static expert weight")
print_trainable_parameters(model)

# finetune
# 加载数据集
try:
    dataset = load_dataset('json', data_files=[
                        args.input], field='instances')
except:
    dataset = load_dataset('json', data_files=[
                       args.input])
c4_dataset = load_dataset('json', data_files=[
                       args.c4_input])
# eval_dataset = load_dataset(
#     'json', data_files=["datasets/sample_questions_from_6_dataset.json"])

tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)


def tokenize_function_c4(examples):
    x = tokenizer(examples['text'], padding="max_length",
                  truncation=True, max_length=256, return_tensors="pt")
    # Ensure labels are the same as input_ids and convert to FP32
    x["labels"] = x["input_ids"].clone()
    return x

def tokenize_function(examples):
    x = tokenizer(examples['text'], padding="max_length",
                  truncation=True, max_length=max_length, return_tensors="pt")
    # Ensure labels are the same as input_ids and convert to FP32
    x["labels"] = x["input_ids"].clone()
    return x


# 应用tokenize函数
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
c4_tokenized_datasets = c4_dataset.map(
    tokenize_function_c4, batched=True, remove_columns=["text"])
# eval_tokenized_datasets = eval_dataset.map(
#     tokenize_function, batched=True, remove_columns=["text"])

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    # Calculate loss
    loss = torch.nn.CrossEntropyLoss()(
        torch.tensor(logits), torch.tensor(labels)).item()
    return {"eval_loss": loss}


def get_custom_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr: float):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


######C4######
if not no_c4:
    training_args = TrainingArguments(
        output_dir=output_dir,          # 输出文件夹（注意：尽管设置了output_dir，但模型不会被保存）
        overwrite_output_dir=True,               # 覆盖输出文件夹
        num_train_epochs=1,                      # 训练轮数
        per_device_train_batch_size=8,           # 每个设备的batch大小
        save_steps=1000000000,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
        save_strategy="steps",
        save_total_limit=0,                      # 不保存任何检查点（虽然设置为0在某些情况下可能不是必需的，但这里为了明确性）
        logging_steps=5,                        # 日志记录的步数
        learning_rate=max_lr,
        # lr_scheduler_type="cosine",
        warmup_ratio=0.2,
        # eval_steps=100,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
        # eval_strategy="steps",
        logging_dir=os.path.join(output_dir, "run_log")
        # 注意：其他参数可以根据需要进行调整
    )
    # Calculate total training steps
    num_training_steps = len(
        c4_tokenized_datasets['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    # Define minimum learning rate
    min_lr = 5e-6

    # Create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_args.learning_rate)

    # Create the custom scheduler
    scheduler = get_custom_schedule_with_warmup(
        optimizer, training_args.warmup_steps, num_training_steps, min_lr)
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=c4_tokenized_datasets['train'],
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )
    trainer.train()

    # 删除c4 ft model
    rm_ft_model_cmd = "rm -r {}/checkpoint*".format(output_dir)
    os.system(rm_ft_model_cmd)


#######SFT#######
training_args = TrainingArguments(
    output_dir=output_dir,          # 输出文件夹（注意：尽管设置了output_dir，但模型不会被保存）
    overwrite_output_dir=True,               # 覆盖输出文件夹
    num_train_epochs=1,                      # 训练轮数
    per_device_train_batch_size=args.batch_size,           # 每个设备的batch大小
    save_steps=1000000000,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
    save_strategy="steps",
    save_total_limit=0,                      # 不保存任何检查点（虽然设置为0在某些情况下可能不是必需的，但这里为了明确性）
    logging_steps=50,                        # 日志记录的步数
    learning_rate=max_lr,
    # lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # eval_steps=100,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
    # eval_strategy="steps",
    logging_dir=os.path.join(output_dir, "run_log")
    # 注意：其他参数可以根据需要进行调整
)
# Calculate total training steps
num_training_steps = len(
    tokenized_datasets['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs

# Define minimum learning rate
min_lr = 5e-6

# Create the optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=training_args.learning_rate)

# Create the custom scheduler
scheduler = get_custom_schedule_with_warmup(
    optimizer, training_args.warmup_steps, num_training_steps, min_lr)
# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    # eval_dataset=eval_tokenized_datasets["train"],
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)
trainer.train()
tokenizer.save_pretrained(output_dir)  