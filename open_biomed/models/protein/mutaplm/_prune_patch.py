import torch, torch.nn as nn
from typing import List, Optional
def find_pruneable_heads_and_indices(heads: List[int], n_heads: int, head_size: int, already_pruned_heads: set) -> tuple:
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for h in heads: mask[h] = 0
    mask = mask.view(-1).contiguous().eq(1)
    return heads, torch.arange(len(mask))[mask].long()
def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    assert dim in (0, 1)
    W = layer.weight.index_select(dim, index)
    b = None if layer.bias is None else (layer.bias if dim == 1 else layer.bias.index_select(0, index))
    new_size = list(layer.weight.size()); new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=b is not None)
    new_layer.weight.data.copy_(W)
    if b is not None: new_layer.bias.data.copy_(b)
    return new_layer
