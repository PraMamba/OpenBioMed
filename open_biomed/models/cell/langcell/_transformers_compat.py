"""补回 4.31+ 被移除的 transformers 内部工具函数"""
import torch
import torch.nn as nn
from typing import Optional, Callable, List

def apply_chunking_to_forward(
    chunk_size: int, chunk_dim: int, forward_fn: Callable, *input_tensors
):
    """
    原 transformers 4.30 实现：把输入沿 chunk_dim 分块推理，再拼接结果。
    """
    assert chunk_dim >= 0, "chunk_dim must be non-negative"
    assert chunk_size > 0,  "chunk_size must be positive"

    # 计算分块数
    tensor_shape = input_tensors[0].shape[chunk_dim]
    if tensor_shape <= chunk_size:
        return forward_fn(*input_tensors)

    num_chunks = (tensor_shape + chunk_size - 1) // chunk_size
    # 按 chunk_dim 切片
    def slice_chunk(t, i):
        return torch.cat(
            [
                t.narrow(chunk_dim, i * chunk_size, chunk_size),
            ],
            dim=chunk_dim,
        )

    outputs = []
    for i in range(num_chunks):
        chunk_inputs = [slice_chunk(t, i) for t in input_tensors]
        out = forward_fn(*chunk_inputs)
        outputs.append(out)

    # 拼接输出
    if isinstance(outputs[0], tuple):
        return tuple(torch.cat(o, dim=chunk_dim) for o in zip(*outputs))
    else:
        return torch.cat(outputs, dim=chunk_dim)

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: set
) -> tuple:
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for h in heads: mask[h] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    assert dim in (0, 1)
    W = layer.weight.index_select(dim, index)
    b = None if layer.bias is None else (
        layer.bias if dim == 1 else layer.bias.index_select(0, index)
    )
    new_size = list(layer.weight.size()); new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=b is not None)
    new_layer.weight.data.copy_(W)
    if b is not None: new_layer.bias.data.copy_(b)
    return new_layer
