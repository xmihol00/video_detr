from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, cast
import json
import random

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from cnnSearch.logging_utils import EventLogger, getEventLogger
from cnnSearch.metrics import computeTopKAccuracy, reduceTensorAverage
from cnnSearch.search_space import ArchitectureConfig, DEFAULT_SEARCH_SPACE, sampleRandomArchitecture

LOGGER = getEventLogger(__name__)

@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learningRate: float
    weightDecay: float
    ampEnabled: bool
    gradientClipNorm: float
    auxiliaryLossWeight: float
    saveDir: str
    evalEveryEpochs: int


@dataclass(frozen=True)
class EvalResult:
    top1: float
    top5: float
    loss: float


def _extractLogitsAndAuxiliary(modelOutputs: object) -> Tuple[Tensor, List[Tensor]]:
    if isinstance(modelOutputs, tuple):
        if len(modelOutputs) == 3:
            logits = cast(Tensor, modelOutputs[0])
            auxiliaryLogits = cast(List[Tensor], modelOutputs[2])
            return logits, auxiliaryLogits
        if len(modelOutputs) >= 1:
            logits = cast(Tensor, modelOutputs[0])
            return logits, []

    logits = cast(Tensor, modelOutputs)
    return logits, []


def _sampleBatchArchitecture(referenceResolution: int) -> ArchitectureConfig:
    architecture = sampleRandomArchitecture(DEFAULT_SEARCH_SPACE)
    return ArchitectureConfig(
        inputResolution=referenceResolution,
        outputStride=architecture.outputStride,
        stageDepths=architecture.stageDepths,
        stageWidthMultipliers=architecture.stageWidthMultipliers,
        stemChannels=architecture.stemChannels,
        stemPathIndex=architecture.stemPathIndex,
        stagePathIndices=architecture.stagePathIndices,
        stageKernelSizes=architecture.stageKernelSizes,
        stageExtraStrides=architecture.stageExtraStrides,
        enableAuxiliaryHeads=True,
    )


def _getGpuMemoryStatsMb(device: torch.device) -> Optional[Dict[str, float]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    allocatedMb = float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0))
    reservedMb = float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0))
    peakAllocatedMb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return {
        "allocatedMb": allocatedMb,
        "reservedMb": reservedMb,
        "peakAllocatedMb": peakAllocatedMb,
    }


def trainOneEpoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainLoader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: GradScaler,
    epochIndex: int,
    ampEnabled: bool,
    gradientClipNorm: float,
    auxiliaryLossWeight: float,
    referenceResolution: int,
    eventLogger: Optional[EventLogger] = None,
    logIntervalSteps: int = 5,
) -> Dict[str, float]:
    model.train()
    activeLogger = eventLogger if eventLogger is not None else LOGGER

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    activeLogger.info(
        "Starting train epoch",
        epoch=epochIndex,
        ampEnabled=ampEnabled,
        gradientClipNorm=gradientClipNorm,
        auxiliaryLossWeight=auxiliaryLossWeight,
        referenceResolution=referenceResolution,
    )

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
            modelOutputs = model(images, sampledArchitecture)
            logits, auxiliaryLogits = _extractLogitsAndAuxiliary(modelOutputs)

            mainLoss = F.cross_entropy(logits, targets)
            if auxiliaryLogits:
                auxLoss = torch.stack([F.cross_entropy(auxLogits, targets) for auxLogits in auxiliaryLogits]).mean()
                loss = mainLoss + auxiliaryLossWeight * auxLoss
            else:
                loss = mainLoss

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

        gpuMemoryStats = _getGpuMemoryStatsMb(device)
        if gpuMemoryStats is None:
            memoryMessage = "gpuMem=cpu"
            gpuAllocatedMb = 0.0
            gpuReservedMb = 0.0
            gpuPeakAllocatedMb = 0.0
        else:
            gpuAllocatedMb = gpuMemoryStats["allocatedMb"]
            gpuReservedMb = gpuMemoryStats["reservedMb"]
            gpuPeakAllocatedMb = gpuMemoryStats["peakAllocatedMb"]
            memoryMessage = (
                f"gpuMemAlloc={gpuAllocatedMb:.1f}MB "
                f"gpuMemRes={gpuReservedMb:.1f}MB "
                f"gpuMemPeak={gpuPeakAllocatedMb:.1f}MB"
            )

        activeLogger.logEveryN(
            key=f"train.epoch.{epochIndex}.metrics",
            everyN=max(1, logIntervalSteps),
            message=(
                f"Training epoch {epochIndex} | "
                f"loss={float(loss.item()):.4f} "
                f"top1={float(top1Tensor.item()):.2f} "
                f"top5={float(top5Tensor.item()):.2f} "
                f"{memoryMessage}"
            ),
            epoch=epochIndex,
            step=numSteps,
            batchLoss=float(loss.item()),
            batchTop1=float(top1Tensor.item()),
            batchTop5=float(top5Tensor.item()),
            gpuMemAllocMb=gpuAllocatedMb,
            gpuMemReservedMb=gpuReservedMb,
            gpuMemPeakMb=gpuPeakAllocatedMb,
        )

    metricTensor = torch.tensor(
        [runningLoss / max(1, numSteps), runningTop1 / max(1, numSteps), runningTop5 / max(1, numSteps)],
        device=device,
    )
    metricTensor = reduceTensorAverage(metricTensor)

    epochMetrics = {
        "loss": float(metricTensor[0].item()),
        "top1": float(metricTensor[1].item()),
        "top5": float(metricTensor[2].item()),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        epochPeakGpuMb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        epochMemoryMessage = f"gpuMemPeak={epochPeakGpuMb:.1f}MB"
    else:
        epochPeakGpuMb = 0.0
        epochMemoryMessage = "gpuMem=cpu"

    activeLogger.info(
        (
            f"Completed train epoch | "
            f"loss={epochMetrics['loss']:.4f} "
            f"top1={epochMetrics['top1']:.2f} "
            f"top5={epochMetrics['top5']:.2f} "
            f"{epochMemoryMessage}"
        ),
        epoch=epochIndex,
        epochLoss=epochMetrics["loss"],
        epochTop1=epochMetrics["top1"],
        epochTop5=epochMetrics["top5"],
        epochGpuPeakMb=epochPeakGpuMb,
        steps=numSteps,
    )
    return epochMetrics


def evaluate(
    model: nn.Module,
    valLoader: torch.utils.data.DataLoader,
    device: torch.device,
    architectureConfig: ArchitectureConfig,
    ampEnabled: bool,
    eventLogger: Optional[EventLogger] = None,
    logIntervalSteps: int = 100,
    epochIndex: Optional[int] = None,
) -> EvalResult:
    model.eval()
    activeLogger = eventLogger if eventLogger is not None else LOGGER

    activeLogger.info(
        "Starting evaluation",
        outputStride=architectureConfig.outputStride,
        inputResolution=architectureConfig.inputResolution,
        ampEnabled=ampEnabled,
    )

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
                modelOutputs = model(images, architectureConfig)
                logits, _ = _extractLogitsAndAuxiliary(modelOutputs)
                loss = F.cross_entropy(logits, targets)

            top1Tensor, top5Tensor = computeTopKAccuracy(logits, targets, topK=(1, 5))
            runningLoss += loss.item()
            runningTop1 += float(top1Tensor.item())
            runningTop5 += float(top5Tensor.item())
            numSteps += 1

            activeLogger.logEveryN(
                key="evaluate.batch.progress",
                everyN=max(1, logIntervalSteps),
                message=f"Evaluation epoch {epochIndex} | loss={float(loss.item()):.4f} top1={float(top1Tensor.item()):.2f} top5={float(top5Tensor.item()):.2f}",
            )

    metricTensor = torch.tensor(
        [runningLoss / max(1, numSteps), runningTop1 / max(1, numSteps), runningTop5 / max(1, numSteps)],
        device=device,
    )
    metricTensor = reduceTensorAverage(metricTensor)

    result = EvalResult(
        loss=float(metricTensor[0].item()),
        top1=float(metricTensor[1].item()),
        top5=float(metricTensor[2].item()),
    )
    activeLogger.info(
        "Completed evaluation",
        loss=result.loss,
        top1=result.top1,
        top5=result.top5,
        steps=numSteps,
    )
    return result


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

    torch.save(state, checkpointPath, _use_new_zipfile_serialization=False)
    LOGGER.info(
        "Saved checkpoint",
        checkpointPath=str(checkpointPath),
        epoch=epochIndex,
        bestMetric=bestMetric,
    )
    return str(checkpointPath)


def appendJsonLog(saveDir: str, metrics: Dict[str, object]) -> None:
    Path(saveDir).mkdir(parents=True, exist_ok=True)
    logPath = Path(saveDir) / "training_log.jsonl"
    with logPath.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")
    LOGGER.logEveryN(
        key="trainer.jsonl.append",
        everyN=10,
        message="Appended training metrics to JSONL",
        logPath=str(logPath),
    )
