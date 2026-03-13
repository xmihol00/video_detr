from __future__ import annotations

import time
from typing import Dict, Tuple

import torch
from torch import nn


def estimateModelParameterMemoryMb(model: nn.Module) -> float:
    parameterBytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return parameterBytes / (1024.0 * 1024.0)


def measureLatencyMs(
    model: nn.Module,
    inputShape: Tuple[int, int, int, int],
    device: torch.device,
    warmupSteps: int = 10,
    measureSteps: int = 30,
) -> float:
    model.eval()
    dummyInput = torch.randn(*inputShape, device=device)

    with torch.no_grad():
        for _ in range(warmupSteps):
            _ = model(dummyInput)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    startTime = time.perf_counter()
    with torch.no_grad():
        for _ in range(measureSteps):
            _ = model(dummyInput)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - startTime
    return (elapsed / measureSteps) * 1000.0


def collectModelResourceMetrics(
    model: nn.Module,
    inputResolution: int,
    device: torch.device,
) -> Dict[str, float]:
    latencyMs = measureLatencyMs(
        model=model,
        inputShape=(1, 3, inputResolution, inputResolution),
        device=device,
    )
    memoryMb = estimateModelParameterMemoryMb(model)

    return {
        "latencyMs": latencyMs,
        "parameterMemoryMb": memoryMb,
    }
