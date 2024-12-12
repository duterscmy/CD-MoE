# -*- coding: utf-8 -*-
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
# import shortuuid
import time

from transformers.variables import *

import torch
import torch.nn.functional as F


def calculate_kl_divergence(probs_p, probs_q):
    """
    计算两个分布之间的KL散度
    :param probs_p: 真实分布的概率分布 (形状：[N, D])
    :param probs_q: 近似分布的概率分布 (形状：[N, D])
    :return: KL散度
    """
    epsilon = 1e-10
    probs_p = probs_p + epsilon
    probs_q = probs_q + epsilon
    # print("probs p {}, probs q {}".format(probs_p, probs_q))
    kl_div = F.kl_div(probs_q.log(), probs_p, reduction='batchmean')  # 计算KL散度
    return kl_div


def calculate_js_divergence(logits_p, logits_q):
    """
    计算两个分布之间的Jensen-Shannon散度
    :param logits_p: 真实分布的logits (形状：[N, D])
    :param logits_q: 近似分布的logits (形状：[N, D])
    :return: JS散度
    """
    p = F.softmax(logits_p, dim=-1)  # 将logits转化为概率分布
    q = F.softmax(logits_q, dim=-1)  # 将logits转化为概率分布
    # print("p {}, q {}".format(p, q))
    m = 0.5 * (p + q)
    kl_pm = calculate_kl_divergence(p, m)
    kl_qm = calculate_kl_divergence(q, m)
    js_div = 0.5 * (kl_pm + kl_qm)
    # print("kl per sample {} {} {}".format(kl_pm, kl_qm, js_div))
    return js_div


def get_layer_output(model, moe_layer_idx, tokenizer, input_strs, batch_size=1, add_special_tokens=True):
    model = model.eval()
    layer_idx = moe_layer_idx + 2  # add embedding layer and ffn layer

    def encode_text_batch(input_strs):
        inputs = tokenizer.batch_encode_plus(
            input_strs,
            padding='longest',
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            max_length=256,
            truncation=True
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        # print("input_ids {}".format(input_ids))
        # print("attention mask {}".format(attention_mask))
        return input_ids, attention_mask

    num_texts = len(input_strs)
    layer_outputs = []

    for i in range(0, num_texts, batch_size):
        text_list_batch = input_strs[i:i + batch_size]
        input_ids, attention_mask = encode_text_batch(text_list_batch)
        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_output = hidden_states[layer_idx]
            layer_output = layer_output.to(torch.float32)
            # print("layer output {}".format(layer_output))
            # print("layer output size {}".format(layer_output.size()))
            # Remove padding based on attention mask
            for j in range(len(text_list_batch)):
                # print(layer_output[j].size())
                # the valid length of the input
                length = attention_mask[j].sum().item()
                trimmed_output = layer_output[j, -length:, :]  # 左侧padding
                # print("trimmed output {}".format(trimmed_output))
                # print("trimeed output dtype {}".format(trimmed_output.dtype))
                layer_outputs.append(trimmed_output)
    return layer_outputs


def get_total_js_divergence(origin_layer_outputs, prune_layer_outputs):
    js_div_sum = 0.0
    for o, p in zip(origin_layer_outputs, prune_layer_outputs):
        js_div = calculate_js_divergence(o, p)
        js_div_sum += js_div.item()
    mean_js_div = js_div_sum / len(origin_layer_outputs)
    print("sum div {}, length dataset {}, mean div {}".format(
        js_div_sum, len(origin_layer_outputs), mean_js_div), flush=True)
    return mean_js_div


global_start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./data/calibration_data.json",
                    help="eval数据集路径")
parser.add_argument("--model", default="./deepseek",
                    help="模型路径")
parser.add_argument("--dynamic-weight-file", default="./data/dynamic_weight.json",
                    help="动态路由系数")
parser.add_argument("--output", default="./data/layer_idx_order.json",
                    help="输出结果的文件名")
parser.add_argument("--greedy-expert-file",  default="./data/layer_idx_to_expert_idx.greedy_jl.json",
                    help="逐层贪心搜索的专家")
parser.add_argument("--batch-size", type=int, default=8, help="并行解码的样本数量")
parser.add_argument("--num-layer", type=int, default=27,
                    help="默认为deepseek16B层数")
parser.add_argument("--num-expert", type=int,
                    default=64, help="默认为deepseek16B专家数")

parser.add_argument("--prune-layer", default=15, type=int,
                    help="剪枝层的数量")
parser.add_argument("--prune-expert", default=6, type=int,
                    help="剪枝专家的数量")
parser.add_argument("--prune-expert-strategy", default="greedy_jl", type=str,
                    help="剪枝专家的策略")
parser.add_argument("--reverse-experts", action="store_true",
                    help="如果指定，则剪枝时倒转expert顺序")

args = parser.parse_args()
output = args.output
batch_size = args.batch_size
num_layer = args.num_layer
num_expert = args.num_expert
num_prune_expert = args.prune_expert
num_prune_layer = args.prune_layer
prune_expert_strategy = args.prune_expert_strategy
pytorch_checkpoint_path = args.model
dynamic_weight_file = args.dynamic_weight_file
greedy_expert_file = args.greedy_expert_file

# @param ["", "0", "0,1", "0,1,2"] {allow-input: true}
available_gpu_ids_str = "0"
memory_per_gpu = "38GiB"  # @param ["", "38GiB"] {allow-input: true}
cpu_memory = '50GiB'  # @param ["50GiB"] {allow-input: true}
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


model = AutoModelForCausalLM.from_pretrained(
    pytorch_checkpoint_path,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)

# read benchmark
with open(args.input, 'r') as fp:
    questions = []
    for line in fp:
        line = line.strip()
        if line:
            question = json.loads(line)
            questions.append(question)
raw_questions = list(map(lambda x: x["text"], questions))

# load greedy expert result
if num_prune_expert == 0:
    layer_idx_to_expert_idxs = {idx: [] for idx in range(27)}
else:
    layer_idx_to_expert_idxs = json.load(
        open(args.greedy_expert_file, 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value[:num_prune_expert] for key, value in layer_idx_to_expert_idxs.items()}

# load pre-computed expert weights
dynamic_weight_file = args.dynamic_weight_file
dynamic_weight_tmp = json.load(open(dynamic_weight_file, 'r'))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w
# print(dynamic_weights)


# origin output (no prune)
prune_layer_idx = int(args.prune_layer)  # 每次只剪枝一层，逐层看效果
prune_layer_list.append({})
layer_num_list.append(num_layer)
s = time.time()
origin_get_layer_output = get_layer_output(
    model, num_layer-1, tokenizer, raw_questions, batch_size=batch_size)
e = time.time()
print("compute origin layer output cost {}".format(e-s))

# prune
beam_size = 1
max_greedy_layer_num = num_prune_layer
beam_prune_layer_idx_list = [[]]

while (len(beam_prune_layer_idx_list[0]) < max_greedy_layer_num):
    print("the {}th iteration".format(
        len(beam_prune_layer_idx_list[0])), flush=True)
    new_prune_layer_idx_list_with_jl = []

    for prune_layer_idx_list in beam_prune_layer_idx_list:
        candidate_layer_idx_list = [layer for layer in range(num_layer)
                                    if layer not in prune_layer_idx_list]
        # candidate_layer_idx_list = candidate_layer_idx_list[:beam_size]
        print("exist prune layers {}; candidate prune layers {}".format(
            prune_layer_idx_list, candidate_layer_idx_list), flush=True)

        for candidate_idx in candidate_layer_idx_list:  # greedy search expert
            start_time = time.time()
            tmp_layer_list = prune_layer_idx_list + [candidate_idx]
            print("try to eval layer idx list {}".format(
                tmp_layer_list), flush=True)

            prune_layer_idx_to_expert_idxs = {}
            for layer_idx in tmp_layer_list:
                prune_layer_idx_to_expert_idxs[layer_idx] = layer_idx_to_expert_idxs[layer_idx]
            print("exp prune layer idx to expert idxs {}".format(
                prune_layer_idx_to_expert_idxs), flush=True)
            # update prune variables
            prune_layer_list.append(prune_layer_idx_to_expert_idxs)
            layer_num_list.append(num_layer)

            # eval ppl on benchmark
            prune_get_layer_output = get_layer_output(
                model, num_layer-1, tokenizer, raw_questions, batch_size=batch_size)
            mean_jl = get_total_js_divergence(
                origin_get_layer_output, prune_get_layer_output)

            new_prune_layer_idx_list_with_jl.append(
                (tuple(tmp_layer_list), mean_jl))

    new_prune_layer_idx_list_with_jl = sorted(
        new_prune_layer_idx_list_with_jl, key=lambda x: x[1])
    new_prune_layer_idx_list_with_jl = new_prune_layer_idx_list_with_jl[:beam_size]

    beam_prune_layer_idx_list = [
        list(t) for t, j in new_prune_layer_idx_list_with_jl]


best_prune_layer_idx_list = beam_prune_layer_idx_list[0]
json.dump(best_prune_layer_idx_list, open(output, 'w'))

end_time = time.time()
print("greedy search layer batchsize {} for one layer cost: {} seconds".format(
    batch_size, end_time-global_start_time))
