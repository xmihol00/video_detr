from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import Tensor


def computeTopKAccuracy(logits: Tensor, targets: Tensor, topK: Iterable[int] = (1, 5)) -> List[Tensor]:
    maxK = max(topK)
    _, predictions = logits.topk(maxK, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    results: List[Tensor] = []
    batchSize = targets.size(0)
    for currentK in topK:
        correctK = correct[:currentK].reshape(-1).float().sum(0)
        results.append(correctK * (100.0 / batchSize))
    return results


def reduceTensorAverage(value: Tensor) -> Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value

    reduced = value.clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    reduced = reduced / torch.distributed.get_world_size()
    return reduced
