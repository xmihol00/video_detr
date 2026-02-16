# Copyright (c) 2026. All Rights Reserved.
"""
Training and evaluation functions for VideoDETR.

This module provides:
- trainOneEpoch: Single epoch training loop
- evaluate: Evaluation on validation set

The functions are adapted from DETR's engine.py to handle video sequences.
Logging is done via Python's standard ``logging`` module and per-batch /
per-epoch metrics are tracked in CSV files through ``MetricTracker``.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import logging
import math
import os
import time
import copy
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

import util.misc as utils
from vidDetr.logging_utils import MetricTracker

logger = logging.getLogger("vidDetr")


def _formatNonAuxLosses(lossDictScaled: Dict[str, Any]) -> str:
    """
    Build a compact string of **non-auxiliary** loss components.

    Auxiliary losses have keys like ``loss_ce_0``, ``loss_bbox_3``, etc.
    (base name + ``_<decoder_layer_index>``).  We keep only the base
    losses whose keys do NOT end with ``_<digit>``.
    """
    import re
    _auxSuffix = re.compile(r"_\d+$")
    parts: list = []
    for k, v in lossDictScaled.items():
        if _auxSuffix.search(k):
            continue  # skip auxiliary decoder-layer losses
        val = v.item() if hasattr(v, "item") else float(v)
        # Shorten key for readability: "loss_ce" -> "ce", "loss_giou" -> "giou"
        shortKey = k.replace("loss_", "")
        parts.append(f"{shortKey}: {val:.4f}")
    return "  ".join(parts)


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    
    Maintains a shadow copy of the model whose parameters are an exponential
    moving average of the training model.  The EMA model typically generalises
    better than the raw training model and is the standard trick used in
    DINO, DN-DETR, RT-DETR and most SOTA detection transformers.
    
    Args:
        model: The model whose parameters will be averaged.
        decay: EMA decay rate (default: 0.9997).  Higher = slower update.
        device: Device for the shadow model (None = same as model).
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9997, device=None):
        # Create a deep copy of the model parameters
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with the current model parameters."""
        # Adaptive decay: use a smaller decay at the beginning (warmup)
        for emaParam, modelParam in zip(
            self.module.state_dict().values(),
            model.state_dict().values()
        ):
            if self.device is not None:
                modelParam = modelParam.to(device=self.device)
            # EMA: ema_p = decay * ema_p + (1 - decay) * model_p
            if emaParam.dtype.is_floating_point:
                emaParam.mul_(self.decay).add_(modelParam, alpha=1.0 - self.decay)


def trainOneEpoch(
    model: nn.Module,
    criterion: nn.Module,
    dataLoader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    maxNorm: float = 0,
    accumSteps: int = 1,
    tracker: Optional[MetricTracker] = None,
    emaModel: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: VideoDETR model
        criterion: VideoCriterion loss module
        dataLoader: Training dataloader
        optimizer: Optimizer
        device: Device for computation
        epoch: Current epoch number
        maxNorm: Maximum gradient norm for clipping (0 to disable)
        accumSteps: Number of gradient accumulation steps
        tracker: Optional MetricTracker for CSV logging
        emaModel: Optional EMA model to update after each optimizer step

    Returns:
        Dict of mean metrics for the epoch
    """
    model.train()
    criterion.train()

    if tracker is not None:
        tracker.epochStart(epoch)

    totalBatches = len(dataLoader)
    logLines = 10 if totalBatches <= 100 else 20 if totalBatches <= 1000 else 40 if totalBatches <= 3500 else 80 
    logInterval = max(totalBatches // logLines, 1)  # ~20 log lines per epoch

    optimizer.zero_grad()
    batchIdx = -1  # in case dataLoader is empty

    epochStart = time.time()
    batchEnd = time.time()

    for batchIdx, (samples, targets) in enumerate(dataLoader):
        dataTime = time.time() - batchEnd

        # Move samples to device
        # samples is a list of NestedTensors, one per frame
        samples = [sample.to(device) for sample in samples]

        # Move targets to device
        # targets is a list of lists of dicts
        for batchTargets in targets:
            for frameTarget in batchTargets:
                for k, v in frameTarget.items():
                    if isinstance(v, torch.Tensor):
                        frameTarget[k] = v.to(device)

        # Forward pass (pass targets for label denoising during training)
        outputs = model(samples, targets=targets)

        # Compute losses
        lossDict = criterion(outputs, targets)
        weightDict = criterion.weightDict

        # Compute weighted sum of losses
        losses = sum(
            lossDict[k] * weightDict[k]
            for k in lossDict.keys()
            if k in weightDict
        )

        # Scale loss by accumulation steps
        losses = losses / accumSteps

        # Reduce losses for logging (use unscaled values for metrics)
        lossDictReduced = utils.reduce_dict(lossDict)
        lossDictReducedScaled = {
            k: v * weightDict[k]
            for k, v in lossDictReduced.items()
            if k in weightDict
        }
        batchLoss = sum(lossDictReducedScaled.values()).item()

        if not math.isfinite(batchLoss):
            logger.error("Loss is %s, stopping training", batchLoss)
            logger.error("Loss dict: %s", lossDictReduced)
            sys.exit(1)

        # Backward pass (accumulate gradients)
        losses.backward()

        # Step optimizer every accumSteps iterations (or at the end of epoch)
        if (batchIdx + 1) % accumSteps == 0:
            if maxNorm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), maxNorm)

            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA model after each optimizer step
            if emaModel is not None:
                emaModel.update(
                    model.module if hasattr(model, 'module') else model
                )

        iterTime = time.time() - batchEnd
        batchEnd = time.time()

        # ---- Collect per-batch metrics ----
        currentLr = optimizer.param_groups[0]["lr"]
        batchMetrics: Dict[str, float] = {
            "loss": batchLoss,
            "lr": currentLr,
            "iter_time": round(iterTime, 4),
            "data_time": round(dataTime, 4),
        }

        # Individual loss components (scaled)
        for k, v in lossDictReducedScaled.items():
            batchMetrics[k] = v.item()

        # Class error
        if "class_error" in lossDictReduced:
            batchMetrics["class_error"] = lossDictReduced["class_error"].item()

        # Tracking loss (unscaled for easy inspection)
        if "loss_tracking" in lossDictReduced:
            batchMetrics["loss_tracking_unscaled"] = lossDictReduced["loss_tracking"].item()

        if tracker is not None:
            tracker.update(step=batchIdx, metrics=batchMetrics)

        # ---- Periodic console / log-file output ----
        if batchIdx % logInterval == 0 or batchIdx == totalBatches - 1:
            memStr = ""
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                memStr = f"  mem: {mem:.0f}MB"

            # Build string of individual (non-aux) loss components
            lossDetails = _formatNonAuxLosses(lossDictReducedScaled)

            logger.info(
                "Epoch [%d] [%d/%d]  loss: %.4f  lr: %.6f  "
                "class_err: %.2f  %s%s",
                epoch,
                batchIdx,
                totalBatches,
                batchLoss,
                currentLr,
                batchMetrics.get("class_error", 0.0),
                lossDetails,
                memStr,
            )

    # Flush any remaining accumulated gradients
    if batchIdx >= 0 and (batchIdx + 1) % accumSteps != 0:
        if maxNorm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), maxNorm)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update EMA for the final partial accumulation step
        if emaModel is not None:
            emaModel.update(
                model.module if hasattr(model, 'module') else model
            )

    # ---- Epoch summary ----
    epochTime = time.time() - epochStart
    epochStats: Dict[str, float] = {}
    if tracker is not None:
        epochStats = tracker.writeEpochSummary()

    logger.info(
        "Epoch [%d] completed in %.1fs — mean loss: %.4f  lr: %.6f",
        epoch,
        epochTime,
        epochStats.get("loss_mean", 0.0),
        epochStats.get("lr_mean", epochStats.get("lr_last", 0.0)),
    )

    return epochStats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    postprocessors: Dict[str, nn.Module],
    dataLoader: Iterable,
    device: torch.device,
    outputDir: str,
    tracker: Optional[MetricTracker] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Evaluate on validation set.

    Args:
        model: VideoDETR model
        criterion: VideoCriterion loss module
        postprocessors: Dict of postprocessor modules
        dataLoader: Validation dataloader
        device: Device for computation
        outputDir: Directory for saving outputs
        tracker: Optional MetricTracker for CSV logging
        epoch: Current epoch (used for tracker epoch tagging)

    Returns:
        Dict of evaluation metrics (mean values across all batches)
    """
    model.eval()
    criterion.eval()

    if tracker is not None:
        tracker.epochStart(epoch)

    totalBatches = len(dataLoader)
    logInterval = max(totalBatches // 5, 1)
    totalSamples = 0

    evalStart = time.time()

    for batchIdx, (samples, targets) in enumerate(dataLoader):
        # Move samples to device
        samples = [sample.to(device) for sample in samples]

        # Move targets to device
        for batchTargets in targets:
            for frameTarget in batchTargets:
                for k, v in frameTarget.items():
                    if isinstance(v, torch.Tensor):
                        frameTarget[k] = v.to(device)

        # Forward pass (no targets needed — denoising is training only)
        outputs = model(samples)

        # Compute losses
        lossDict = criterion(outputs, targets)
        weightDict = criterion.weightDict

        # Reduce losses for logging
        lossDictReduced = utils.reduce_dict(lossDict)
        lossDictReducedScaled = {
            k: v * weightDict[k]
            for k, v in lossDictReduced.items()
            if k in weightDict
        }
        batchLoss = sum(lossDictReducedScaled.values()).item()

        # ---- Collect per-batch metrics ----
        batchMetrics: Dict[str, float] = {"loss": batchLoss}

        for k, v in lossDictReducedScaled.items():
            batchMetrics[k] = v.item()

        if "class_error" in lossDictReduced:
            batchMetrics["class_error"] = lossDictReduced["class_error"].item()

        if "loss_tracking" in lossDictReduced:
            batchMetrics["loss_tracking_unscaled"] = lossDictReduced["loss_tracking"].item()

        if tracker is not None:
            tracker.update(step=batchIdx, metrics=batchMetrics)

        if batchIdx % logInterval == 0 or batchIdx == totalBatches - 1:
            lossDetails = _formatNonAuxLosses(lossDictReducedScaled)

            logger.info(
                "Val [%d/%d]  loss: %.4f  class_err: %.2f  %s",
                batchIdx,
                totalBatches,
                batchLoss,
                batchMetrics.get("class_error", 0.0),
                lossDetails,
            )

        totalSamples += len(targets)

    # ---- Epoch summary ----
    evalTime = time.time() - evalStart
    epochStats: Dict[str, float] = {}
    if tracker is not None:
        epochStats = tracker.writeEpochSummary(
            extraMetrics={"num_samples": totalSamples}
        )

    logger.info(
        "Validation completed in %.1fs — mean loss: %.4f  samples: %d",
        evalTime,
        epochStats.get("loss_mean", 0.0),
        totalSamples,
    )

    return epochStats


def computeTrackingMetrics(
    predictions: List[Dict],
    targets: List[List[Dict]],
    iouThreshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute tracking-specific metrics.

    This function computes metrics for evaluating the quality of
    object tracking across frames.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts with tracking annotations
        iouThreshold: IoU threshold for matching

    Returns:
        Dict with tracking metrics:
        - MOTA: Multi-Object Tracking Accuracy
        - IDF1: ID F1 Score
        - ID_switches: Number of ID switches
        - fragmentations: Number of track fragmentations

    Note: This is a simplified implementation. For full MOT metrics,
    consider using the motmetrics library.
    """
    metrics = {
        "mota": 0.0,
        "idf1": 0.0,
        "id_switches": 0,
        "fragmentations": 0,
    }

    # TODO: Implement full MOT metrics computation
    # This requires:
    # 1. Running inference on full video sequences
    # 2. Associating detections across frames using tracking embeddings
    # 3. Comparing with ground truth track assignments
    # 4. Computing CLEAR MOT metrics

    return metrics


def associateDetectionsAcrossFrames(
    trackingEmbeddings: torch.Tensor,
    detectionScores: torch.Tensor,
    numFrames: int,
    queriesPerFrame: int,
    similarityThreshold: float = 0.5,
) -> List[List[int]]:
    """
    Associate detections across frames using tracking embeddings.

    This function implements a simple greedy association algorithm
    based on embedding similarity.

    Args:
        trackingEmbeddings: [B, numQueries, embedDim] tracking embeddings
        detectionScores: [B, numQueries] detection confidence scores
        numFrames: Number of frames
        queriesPerFrame: Number of queries per frame
        similarityThreshold: Minimum similarity for association

    Returns:
        List of tracks, where each track is a list of query indices
        across frames (or -1 if no detection in that frame)
    """
    import torch.nn.functional as F

    batchSize = trackingEmbeddings.shape[0]
    embedDim = trackingEmbeddings.shape[-1]

    allTracks = []

    for b in range(batchSize):
        # Get embeddings and scores for this sample
        embeddings = trackingEmbeddings[b]  # [numQueries, embedDim]
        scores = detectionScores[b]  # [numQueries]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Split by frame
        frameEmbeddings = []
        frameScores = []
        for f in range(numFrames):
            start = f * queriesPerFrame
            end = start + queriesPerFrame
            frameEmbeddings.append(embeddings[start:end])
            frameScores.append(scores[start:end])

        # Initialize tracks from first frame
        # Only consider high-confidence detections
        scoreThreshold = 0.5
        tracks = []

        for qIdx in range(queriesPerFrame):
            if frameScores[0][qIdx] > scoreThreshold:
                # Start new track
                track = [-1] * numFrames
                track[0] = qIdx
                tracks.append(
                    {"indices": track, "embedding": frameEmbeddings[0][qIdx]}
                )

        # Associate detections in subsequent frames
        for f in range(1, numFrames):
            # Compute similarity between track embeddings and frame detections
            if not tracks:
                # No tracks yet, start new ones
                for qIdx in range(queriesPerFrame):
                    if frameScores[f][qIdx] > scoreThreshold:
                        track = [-1] * numFrames
                        track[f] = qIdx
                        tracks.append(
                            {
                                "indices": track,
                                "embedding": frameEmbeddings[f][qIdx],
                            }
                        )
                continue

            # Stack track embeddings
            trackEmbs = torch.stack([t["embedding"] for t in tracks])

            # Compute similarity matrix
            similarity = torch.matmul(trackEmbs, frameEmbeddings[f].T)

            # Greedy matching
            usedDetections = set()
            usedTracks = set()

            # Sort by similarity (descending)
            flatSim = similarity.flatten()
            sortedIndices = flatSim.argsort(descending=True)

            for idx in sortedIndices:
                trackIdx = idx // queriesPerFrame
                detIdx = idx % queriesPerFrame

                if (
                    trackIdx.item() in usedTracks
                    or detIdx.item() in usedDetections
                ):
                    continue

                sim = similarity[trackIdx, detIdx]

                if (
                    sim > similarityThreshold
                    and frameScores[f][detIdx] > scoreThreshold
                ):
                    # Assign detection to track
                    tracks[trackIdx]["indices"][f] = detIdx.item()
                    # Update track embedding (exponential moving average)
                    alpha = 0.5
                    tracks[trackIdx]["embedding"] = (
                        alpha * tracks[trackIdx]["embedding"]
                        + (1 - alpha) * frameEmbeddings[f][detIdx]
                    )
                    tracks[trackIdx]["embedding"] = F.normalize(
                        tracks[trackIdx]["embedding"], p=2, dim=-1
                    )

                    usedTracks.add(trackIdx.item())
                    usedDetections.add(detIdx.item())

            # Start new tracks for unmatched high-confidence detections
            for qIdx in range(queriesPerFrame):
                if (
                    qIdx not in usedDetections
                    and frameScores[f][qIdx] > scoreThreshold
                ):
                    track = [-1] * numFrames
                    track[f] = qIdx
                    tracks.append(
                        {
                            "indices": track,
                            "embedding": frameEmbeddings[f][qIdx],
                        }
                    )

        # Extract track indices
        sampleTracks = [t["indices"] for t in tracks]
        allTracks.append(sampleTracks)

    return allTracks
