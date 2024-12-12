def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def classify_remained_experts(module_name, prune_layer_idx_to_expert_idx):
    '''classify the remained experts in prune layers'''
    try:
        layer_id = int(module_name.split(".")[2])
        expert_id = int(module_name.split(".")[5])
    except:
        return False
    moe_layer_id = layer_id - 1
    # print(moe_layer_id, expert_id)
    if moe_layer_id in prune_layer_idx_to_expert_idx and expert_id in prune_layer_idx_to_expert_idx[moe_layer_id]:
        return True
    return False


def classify_shared_experts(module_name, prune_layer_idx_to_expert_idx):
    '''classify the shared experts in prune layers'''
    try:
        layer_id = int(module_name.split(".")[2])
    except:
        return False
    moe_layer_id = layer_id - 1
    # print(moe_layer_id, expert_id)
    if moe_layer_id in prune_layer_idx_to_expert_idx and "shared" in module_name:
        return True
    return False


def classify_pruned_experts(module_name, prune_layer_idx_to_expert_idx):
    '''classify the pruned experts in prune layers'''
    try:
        layer_id = int(module_name.split(".")[2])
        expert_id = int(module_name.split(".")[5])
    except:
        return False
    moe_layer_id = layer_id - 1
    # print(moe_layer_id, expert_id)
    if moe_layer_id in prune_layer_idx_to_expert_idx and expert_id not in prune_layer_idx_to_expert_idx[moe_layer_id]:
        return True
    return False