from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def isDistributedAvailableAndInitialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def getWorldSize() -> int:
    if not isDistributedAvailableAndInitialized():
        return 1
    return dist.get_world_size()


def getRank() -> int:
    if not isDistributedAvailableAndInitialized():
        return 0
    return dist.get_rank()


def isMainProcess() -> bool:
    return getRank() == 0


def setupDistributed() -> Tuple[bool, torch.device, int]:
    worldSize = int(os.environ.get("WORLD_SIZE", "1"))
    isDistributed = worldSize > 1

    if isDistributed:
        if torch.cuda.is_available():
            localRank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(localRank)
            dist.init_process_group(backend="nccl", init_method="env://")
            device = torch.device("cuda", localRank)
        else:
            dist.init_process_group(backend="gloo", init_method="env://")
            device = torch.device("cpu")
            localRank = 0
    else:
        localRank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")

    return isDistributed, device, localRank


def cleanupDistributed() -> None:
    if isDistributedAvailableAndInitialized():
        dist.barrier()
        dist.destroy_process_group()
