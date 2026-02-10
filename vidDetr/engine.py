# Copyright (c) 2026. All Rights Reserved.
"""
Training and evaluation functions for VideoDETR.

This module provides:
- trainOneEpoch: Single epoch training loop
- evaluate: Evaluation on validation set

The functions are adapted from DETR's engine.py to handle video sequences.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import math
import os
from typing import Iterable, Dict, Any, List

import torch
import torch.nn as nn

import util.misc as utils


def trainOneEpoch(
    model: nn.Module,
    criterion: nn.Module,
    dataLoader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    maxNorm: float = 0,
    accumSteps: int = 1
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
    
    Returns:
        Dict of average metrics for the epoch
    """
    model.train()
    criterion.train()
    
    metricLogger = utils.MetricLogger(delimiter="  ")
    metricLogger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metricLogger.add_meter('class_error', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metricLogger.add_meter('loss_tracking', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}]'
    printFreq = 10
    
    optimizer.zero_grad()
    
    for batchIdx, (samples, targets) in enumerate(metricLogger.log_every(dataLoader, printFreq, header)):
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
        lossDictReducedUnscaled = {
            f'{k}_unscaled': v
            for k, v in lossDictReduced.items()
        }
        lossDictReducedScaled = {
            k: v * weightDict[k]
            for k, v in lossDictReduced.items()
            if k in weightDict
        }
        lossesReducedScaled = sum(lossDictReducedScaled.values())
        
        lossValue = lossesReducedScaled.item()
        
        if not math.isfinite(lossValue):
            print(f"Loss is {lossValue}, stopping training")
            print(lossDictReduced)
            sys.exit(1)
        
        # Backward pass (accumulate gradients)
        losses.backward()
        
        # Step optimizer every accumSteps iterations (or at the end of epoch)
        if (batchIdx + 1) % accumSteps == 0:
            if maxNorm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), maxNorm)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics
        metricLogger.update(
            loss=lossValue,
            **lossDictReducedScaled,
            **lossDictReducedUnscaled
        )
        metricLogger.update(class_error=lossDictReduced['class_error'])
        metricLogger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Log tracking loss specifically
        if 'loss_tracking' in lossDictReduced:
            metricLogger.update(loss_tracking=lossDictReduced['loss_tracking'].item())
    
    # Flush any remaining accumulated gradients
    if (batchIdx + 1) % accumSteps != 0:
        if maxNorm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), maxNorm)
        optimizer.step()
        optimizer.zero_grad()
    
    # Gather stats from all processes
    metricLogger.synchronize_between_processes()
    print("Averaged stats:", metricLogger)
    
    return {k: meter.global_avg for k, meter in metricLogger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    postprocessors: Dict[str, nn.Module],
    dataLoader: Iterable,
    device: torch.device,
    outputDir: str
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
    
    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    criterion.eval()
    
    metricLogger = utils.MetricLogger(delimiter="  ")
    metricLogger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metricLogger.add_meter('loss_tracking', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Test:'
    
    # Tracking metrics
    totalSamples = 0
    
    for samples, targets in metricLogger.log_every(dataLoader, 10, header):
        # Move samples to device
        samples = [sample.to(device) for sample in samples]
        
        # Move targets to device
        for batchTargets in targets:
            for frameTarget in batchTargets:
                for k, v in frameTarget.items():
                    if isinstance(v, torch.Tensor):
                        frameTarget[k] = v.to(device)
        
        # Forward pass (no targets needed - denoising is training only)
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
        lossDictReducedUnscaled = {
            f'{k}_unscaled': v
            for k, v in lossDictReduced.items()
        }
        
        metricLogger.update(
            loss=sum(lossDictReducedScaled.values()),
            **lossDictReducedScaled,
            **lossDictReducedUnscaled
        )
        metricLogger.update(class_error=lossDictReduced['class_error'])
        
        if 'loss_tracking' in lossDictReduced:
            metricLogger.update(loss_tracking=lossDictReduced['loss_tracking'].item())
        
        totalSamples += len(targets)
    
    # Gather stats from all processes
    metricLogger.synchronize_between_processes()
    print("Averaged stats:", metricLogger)
    
    stats = {k: meter.global_avg for k, meter in metricLogger.meters.items()}
    stats['num_samples'] = totalSamples
    
    return stats


def computeTrackingMetrics(
    predictions: List[Dict],
    targets: List[List[Dict]],
    iouThreshold: float = 0.5
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
    # Placeholder for tracking metrics
    # Full implementation would require frame-by-frame association
    # and comparison with ground truth track IDs
    
    metrics = {
        'mota': 0.0,
        'idf1': 0.0,
        'id_switches': 0,
        'fragmentations': 0
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
    similarityThreshold: float = 0.5
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
                tracks.append({
                    'indices': track,
                    'embedding': frameEmbeddings[0][qIdx]
                })
        
        # Associate detections in subsequent frames
        for f in range(1, numFrames):
            # Compute similarity between track embeddings and frame detections
            if not tracks:
                # No tracks yet, start new ones
                for qIdx in range(queriesPerFrame):
                    if frameScores[f][qIdx] > scoreThreshold:
                        track = [-1] * numFrames
                        track[f] = qIdx
                        tracks.append({
                            'indices': track,
                            'embedding': frameEmbeddings[f][qIdx]
                        })
                continue
            
            # Stack track embeddings
            trackEmbeddings = torch.stack([t['embedding'] for t in tracks])
            
            # Compute similarity matrix
            similarity = torch.matmul(trackEmbeddings, frameEmbeddings[f].T)
            
            # Greedy matching
            usedDetections = set()
            usedTracks = set()
            
            # Sort by similarity (descending)
            flatSim = similarity.flatten()
            sortedIndices = flatSim.argsort(descending=True)
            
            for idx in sortedIndices:
                trackIdx = idx // queriesPerFrame
                detIdx = idx % queriesPerFrame
                
                if trackIdx.item() in usedTracks or detIdx.item() in usedDetections:
                    continue
                
                sim = similarity[trackIdx, detIdx]
                
                if sim > similarityThreshold and frameScores[f][detIdx] > scoreThreshold:
                    # Assign detection to track
                    tracks[trackIdx]['indices'][f] = detIdx.item()
                    # Update track embedding (exponential moving average)
                    alpha = 0.5
                    tracks[trackIdx]['embedding'] = (
                        alpha * tracks[trackIdx]['embedding'] +
                        (1 - alpha) * frameEmbeddings[f][detIdx]
                    )
                    tracks[trackIdx]['embedding'] = F.normalize(
                        tracks[trackIdx]['embedding'], p=2, dim=-1
                    )
                    
                    usedTracks.add(trackIdx.item())
                    usedDetections.add(detIdx.item())
            
            # Start new tracks for unmatched high-confidence detections
            for qIdx in range(queriesPerFrame):
                if qIdx not in usedDetections and frameScores[f][qIdx] > scoreThreshold:
                    track = [-1] * numFrames
                    track[f] = qIdx
                    tracks.append({
                        'indices': track,
                        'embedding': frameEmbeddings[f][qIdx]
                    })
        
        # Extract track indices
        sampleTracks = [t['indices'] for t in tracks]
        allTracks.append(sampleTracks)
    
    return allTracks
