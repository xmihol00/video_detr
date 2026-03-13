from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import random

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from cnnSearch.metrics import computeTopKAccuracy, reduceTensorAverage
from cnnSearch.search_space import ArchitectureConfig, DEFAULT_SEARCH_SPACE, sampleRandomArchitecture


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learningRate: float
    weightDecay: float
    ampEnabled: bool
    gradientClipNorm: float
    saveDir: str
    evalEveryEpochs: int


@dataclass(frozen=True)
class EvalResult:
    top1: float
    top5: float
    loss: float


def _sampleBatchArchitecture(referenceResolution: int) -> ArchitectureConfig:
    architecture = sampleRandomArchitecture(DEFAULT_SEARCH_SPACE)
    return ArchitectureConfig(
        inputResolution=referenceResolution,
        outputStride=architecture.outputStride,
        stageDepths=architecture.stageDepths,
        stageWidthMultipliers=architecture.stageWidthMultipliers,
        stemChannels=architecture.stemChannels,
    )


def trainOneEpoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainLoader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: GradScaler,
    epochIndex: int,
    ampEnabled: bool,
    gradientClipNorm: float,
    referenceResolution: int,
) -> Dict[str, float]:
    model.train()

    runningLoss = 0.0
    runningTop1 = 0.0
    runningTop5 = 0.0
    numSteps = 0

    for images, targets in trainLoader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        sampledArchitecture = _sampleBatchArchitecture(referenceResolution=referenceResolution)
        if images.shape[-1] != sampledArchitecture.inputResolution:
            images = F.interpolate(
                images,
                size=(sampledArchitecture.inputResolution, sampledArchitecture.inputResolution),
                mode="bilinear",
                align_corners=False,
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=ampEnabled and device.type == "cuda"):
            logits, _ = model(images, sampledArchitecture)
            loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        if gradientClipNorm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradientClipNorm)

        scaler.step(optimizer)
        scaler.update()

        top1Tensor, top5Tensor = computeTopKAccuracy(logits.detach(), targets, topK=(1, 5))

        runningLoss += loss.item()
        runningTop1 += float(top1Tensor.item())
        runningTop5 += float(top5Tensor.item())
        numSteps += 1

    metricTensor = torch.tensor(
        [runningLoss / max(1, numSteps), runningTop1 / max(1, numSteps), runningTop5 / max(1, numSteps)],
        device=device,
    )
    metricTensor = reduceTensorAverage(metricTensor)

    return {
        "loss": float(metricTensor[0].item()),
        "top1": float(metricTensor[1].item()),
        "top5": float(metricTensor[2].item()),
    }


def evaluate(
    model: nn.Module,
    valLoader: torch.utils.data.DataLoader,
    device: torch.device,
    architectureConfig: ArchitectureConfig,
    ampEnabled: bool,
) -> EvalResult:
    model.eval()

    runningLoss = 0.0
    runningTop1 = 0.0
    runningTop5 = 0.0
    numSteps = 0

    with torch.no_grad():
        for images, targets in valLoader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if images.shape[-1] != architectureConfig.inputResolution:
                images = F.interpolate(
                    images,
                    size=(architectureConfig.inputResolution, architectureConfig.inputResolution),
                    mode="bilinear",
                    align_corners=False,
                )

            with autocast(enabled=ampEnabled and device.type == "cuda"):
                logits, _ = model(images, architectureConfig)
                loss = F.cross_entropy(logits, targets)

            top1Tensor, top5Tensor = computeTopKAccuracy(logits, targets, topK=(1, 5))
            runningLoss += loss.item()
            runningTop1 += float(top1Tensor.item())
            runningTop5 += float(top5Tensor.item())
            numSteps += 1

    metricTensor = torch.tensor(
        [runningLoss / max(1, numSteps), runningTop1 / max(1, numSteps), runningTop5 / max(1, numSteps)],
        device=device,
    )
    metricTensor = reduceTensorAverage(metricTensor)

    return EvalResult(
        loss=float(metricTensor[0].item()),
        top1=float(metricTensor[1].item()),
        top5=float(metricTensor[2].item()),
    )


def saveCheckpoint(
    saveDir: str,
    epochIndex: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    bestMetric: float,
    extraState: Optional[Dict[str, object]] = None,
) -> str:
    Path(saveDir).mkdir(parents=True, exist_ok=True)
    checkpointPath = Path(saveDir) / f"checkpoint_epoch_{epochIndex:04d}.pth"

    state = {
        "epoch": epochIndex,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "bestMetric": bestMetric,
        "rngStateTorch": torch.get_rng_state(),
        "rngStatePython": random.getstate(),
    }
    if extraState is not None:
        state["extraState"] = extraState

    torch.save(state, checkpointPath)
    return str(checkpointPath)


def appendJsonLog(saveDir: str, metrics: Dict[str, object]) -> None:
    Path(saveDir).mkdir(parents=True, exist_ok=True)
    logPath = Path(saveDir) / "training_log.jsonl"
    with logPath.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")
