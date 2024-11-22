## CONDENSE, DON’T JUST PRUNE: ENHANCING EFFICIENCY AND PERFORMANCE IN MOE LAYER PRUNING
Official PyTorch implementation of CD-MOE, as presented in our paper:

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

## Setup

Follow the setup of [Transformers](https://github.com/huggingface/transformers)

```bash
git clone https://github.com/duterscmy/CD-MoE.git
cd CD-MoE
pip install -e .
```

## Usage

The process mainly consists of three steps: (1) obtaining the average weights of the experts, (2) selecting experts and layers through greedy search, and (3) fine-tuning the experts. The first two steps are mandatory, while the last step is optional. First, you need to download the official deepseek16B-MOE model to the local directory `$model_path`.

### 1. Expert Weight

```bash
cp cd-moe/get_weight/modeling_deepseek.py $model_path
python cd-moe/get_weight/get_weight.py \
    --input $calibration_data_file \
    --output $expert_weight_file \
    --model $model_path
```

### 2. Greedy Search

The greedy search expert must be done before the greedy search layer.

#### Greedy Search Expert

```bash
cp cd-moe/greedy_search/modeling_deepseek.py $model_path
python cd-moe/greedy_search/greedy_search_expert.py \
    --input $calibration_data_file \
    --model $model_path \
    --dynamic-weight-file $expert_weight_file \
    --output $greedy_search_expert_result_file
```

#### Greedy Search Layer

```bash
python cd-moe/greedy_search/greedy_search_layer.py \
    --input $calibration_data_file \
    --model $model_path \
    --dynamic-weight-file $expert_weight_file \
    --greedy-expert-file $greedy_search_expert_result_file \
    --output $greedy_search_layer_result_file
```

### 3. Fine-tune (Optional)

```bash
cp cd-moe/modeling_deepseek.py cd-moe/exp_hyper.py $model_path
python cd-moe/finetune/finetune.py \
    --input $sft_data \
    --c4-input $lm_data \
    --model $model_path \
    --output-dir $sft_model_path
```

You can use the `--no-c4` option to skip lm fine-tuning and directly fine-tune for downstream tasks.

For some intermediate variables, we provide some already generated results. The open-source model and C4 training data need to be downloaded locally:
- calibration_data_file: `cd-moe/data/calibration_data.json`
- expert_weight_file: `cd-moe/data/dynamic_weight.json`
- greedy_search_expert_result_file: `cd-moe/data/layer_idx_to_expert_idx.greedy_jl.json`

## Evaluation

Install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)  
Evaluate the pruned model:
```bash
lm_eval --model hf \
    --model_args $modelpath \
    --tasks arc-challenge,boolq,piqa,rte,obqa,winogrande,mmlu,hellaswag \
    --device cuda:0 \
    --batch_size 8
```
Evaluate the fine-tuned model:
```bash
lm_eval --model hf \
    --model_args $modelpath \
    --tasks arc-challenge,boolq,piqa,rte,obqa,winogrande,mmlu,hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --ignore_mismatched_sizes
```
