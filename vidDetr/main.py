# Copyright (c) 2026. All Rights Reserved.
"""
Main training script for VideoDETR.

This script handles:
- Argument parsing with all hyperparameters
- Model, criterion, and optimizer setup
- Dataset and dataloader creation
- Training loop with evaluation
- Checkpointing and logging

Usage:
    python main.py --dataConfig data.yaml --numFrames 5 --epochs 100

For distributed training:
    torchrun --nproc_per_node=2 main.py --dataConfig data.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import argparse
import datetime
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, DistributedSampler

# Use file_system sharing strategy to avoid running out of file descriptors.
# The default 'file_descriptor' strategy uses one FD per shared tensor, which
# quickly exhausts the 1024 FD limit on shared machines with video batches.
torch.multiprocessing.set_sharing_strategy('file_system')

import util.misc as utils
from vidDetr.models import buildVideoDETR
from vidDetr.datasets import VideoSequenceDataset, buildVideoDataset, videoCollateFn
from vidDetr.datasets import TaoDataset, buildTaoDataset, taoCollateFn
from vidDetr.engine import trainOneEpoch, evaluate, ModelEMA
from vidDetr.logging_utils import setupLogging, MetricTracker

NUM_GPUS = 1

import safe_gpu
while True:
    try:
        safe_gpu.claim_gpus(NUM_GPUS)
        break
    except:
        print("Waiting for free GPU")
        time.sleep(5)
        pass

# ---------------------------------------------------------------------------
# Per-epoch hyperparameter scheduling
# ---------------------------------------------------------------------------

# Names of all arguments that support per-epoch scheduling (list of floats).
# Each is stored as a list; at epoch *e* the value used is
#     schedule[min(e, len(schedule) - 1)]
SCHEDULED_PARAM_NAMES: List[str] = [
    'focalAlpha', 'focalGamma',
    'setCostClass', 'setCostBbox', 'setCostGiou',
    'bboxLossCoef', 'giouLossCoef',
    'eosCoef',
    'trackingLossCoef', 'contrastiveTemp',
    'labelNoiseRatio', 'boxNoiseScale',
    'dnLossCoef', 'dupLossCoef',
    'dropout', 'dropPathRate',
    'clipMaxNorm',
]


def getScheduledValue(schedule: List[float], epoch: int) -> float:
    """
    Return the value from *schedule* for the given *epoch*.

    If the list is shorter than ``epoch + 1``, the last value is reused.
    """
    return schedule[min(epoch, len(schedule) - 1)]


def resolveEpochParams(args, epoch: int) -> Dict[str, float]:
    """
    Resolve all scheduled parameters for the given *epoch* and return
    a dict ``{paramName: value}``.
    """
    resolved: Dict[str, float] = {}
    for name in SCHEDULED_PARAM_NAMES:
        # Special case: dropout may have been converted to scalar by
        # convertArgsForTransformer; the original schedule is stored
        # in ``_dropoutSchedule``.
        if name == 'dropout':
            schedule = getattr(args, '_dropoutSchedule', None)
            if schedule is None:
                schedule = getattr(args, 'dropout', None)
        else:
            schedule = getattr(args, name, None)

        if schedule is not None and isinstance(schedule, list):
            resolved[name] = getScheduledValue(schedule, epoch)
        else:
            # Fallback: the attribute is already a scalar (shouldn't happen
            # after parser changes, but be safe).
            resolved[name] = float(schedule) if schedule is not None else 0.0
    return resolved


def logEpochScheduledParams(
    logger, epoch: int, params: Dict[str, float]
) -> None:
    """
    Log the resolved per-epoch hyperparameters in a readable table.
    """
    lines = [f"Epoch [{epoch}] scheduled hyperparameters:"]
    for name, value in params.items():
        lines.append(f"  {name:>20s} = {value}")
    logger.info("\n".join(lines))


def getArgsParser():
    """
    Create argument parser with all configuration options.
    
    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        'VideoDETR training and evaluation script',
        add_help=False
    )
    
    # Learning rate and optimization
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Base learning rate')
    parser.add_argument('--lrBackbone', default=1e-5, type=float,
                        help='Learning rate for backbone')
    parser.add_argument('--batchSize', default=48, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--weightDecay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--lrDrop', default=80, type=int,
                        help='Epoch to drop learning rate')
    parser.add_argument('--clipMaxNorm', default=[0.1], nargs='+', type=float,
                        help='Gradient clipping max norm; per-epoch schedule')
    
    # Warmup parameters
    parser.add_argument('--warmupEpochs', default=2, type=int,
                        help='Number of warmup epochs with linearly increasing LR')
    parser.add_argument('--warmupStartLr', default=1e-6, type=float,
                        help='Starting learning rate for warmup')
    
    # Gradient accumulation
    parser.add_argument('--accumSteps', default=1, type=int,
                        help='Gradient accumulation steps (effective batch = batchSize * accumSteps)')
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='CNN backbone architecture')
    parser.add_argument('--freezeBackbone', action='store_true', default=False,
                        help='Freeze backbone parameters during training')
    parser.add_argument('--dilation', action='store_true', default=False,
                        help='Use dilation in last backbone block')
    parser.add_argument('--positionEmbedding', default='sine', type=str,
                        choices=['sine', 'learned'],
                        help='Type of spatial positional embedding')
    parser.add_argument('--temporalEncoding', default='sine', type=str,
                        choices=['sine', 'learned'],
                        help='Type of temporal positional embedding')
    
    # Transformer parameters
    parser.add_argument('--encLayers', default=6, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--decLayers', default=6, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dimFeedforward', default=2048, type=int,
                        help='Feedforward dimension in transformer')
    parser.add_argument('--hiddenDim', default=256, type=int,
                        help='Transformer hidden dimension')
    parser.add_argument('--dropout', default=[0.0, 0.1, 0.1, 0.2], nargs='+', type=float,
                        help='Dropout rate in transformer and heads; per-epoch schedule')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--preNorm', action='store_true',
                        help='Use pre-normalization in transformer')
    
    # VideoDETR specific parameters
    parser.add_argument('--numFrames', default=4, type=int,
                        help='Number of frames per video clip')
    parser.add_argument('--queriesPerFrame', default=30, type=int,
                        help='Number of detection queries per frame')
    parser.add_argument('--trackingEmbedDim', default=128, type=int,
                        help='Dimension of tracking embeddings')
    parser.add_argument('--maxFrames', default=50, type=int,
                        help='Maximum frames for temporal encoding')
    
    # Loss parameters
    parser.add_argument('--auxLoss', action='store_true', default=True,
                        help='Use auxiliary losses at each decoder layer')
    parser.add_argument('--noAuxLoss', dest='auxLoss', action='store_false',
                        help='Disable auxiliary losses')
    parser.add_argument('--useFocalLoss', action='store_true', default=False,
                        help='Use focal loss instead of cross-entropy for classification')
    parser.add_argument('--noFocalLoss', dest='useFocalLoss', action='store_false',
                        help='Disable focal loss, use cross-entropy instead')
    parser.add_argument('--focalAlpha', default=[0.25], nargs='+', type=float,
                        help='Focal loss alpha (balancing factor); per-epoch schedule')
    parser.add_argument('--focalGamma', default=[2.0], nargs='+', type=float,
                        help='Focal loss gamma (focusing parameter); per-epoch schedule')
    parser.add_argument('--setCostClass', default=[4.0, 3.75, 3.5, 3.0], nargs='+', type=float,
                        help='Classification cost in matching; per-epoch schedule')
    parser.add_argument('--setCostBbox', default=[5.0], nargs='+', type=float,
                        help='L1 box cost in matching; per-epoch schedule')
    parser.add_argument('--setCostGiou', default=[2.0], nargs='+', type=float,
                        help='GIoU cost in matching; per-epoch schedule')
    parser.add_argument('--bboxLossCoef', default=[5.0], nargs='+', type=float,
                        help='L1 box loss coefficient; per-epoch schedule')
    parser.add_argument('--giouLossCoef', default=[2.0], nargs='+', type=float,
                        help='GIoU loss coefficient; per-epoch schedule')
    parser.add_argument('--eosCoef', default=[0.035, 0.05, 0.1, 0.15, 0.2], nargs='+', type=float,
                        help='No-object class weight (higher = fewer false positives); per-epoch schedule')
    parser.add_argument('--trackingLossCoef', default=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75], nargs='+', type=float,
                        help='Tracking contrastive loss coefficient; per-epoch schedule')
    parser.add_argument('--contrastiveTemp', default=[0.07], nargs='+', type=float,
                        help='Temperature for contrastive loss; per-epoch schedule')
    
    # Label denoising (DN-DETR / DINO) parameters
    parser.add_argument('--useDnDenoising', action='store_true', default=False,
                        help='Use label denoising for training stabilization')
    parser.add_argument('--noDnDenoising', dest='useDnDenoising', action='store_false',
                        help='Disable label denoising')
    parser.add_argument('--numDnGroups', default=5, type=int,
                        help='Number of denoising groups (each group = noised copy of all GTs)')
    parser.add_argument('--labelNoiseRatio', default=[0.5], nargs='+', type=float,
                        help='Probability of flipping a label to random class in DN queries; per-epoch schedule')
    parser.add_argument('--boxNoiseScale', default=[0.4], nargs='+', type=float,
                        help='Scale of box coordinate noise (fraction of box size); per-epoch schedule')
    parser.add_argument('--dnLossCoef', default=[1.0], nargs='+', type=float,
                        help='Coefficient for denoising losses (multiplied with base loss coefs); per-epoch schedule')
    
    # Duplicate suppression loss
    parser.add_argument('--dupLossCoef', default=[0.0, 0.1, 0.25, 0.5, 1.0, 1.5], nargs='+', type=float,
                        help='Weight for IoU-based duplicate suppression loss; per-epoch schedule')
    
    # EMA (Exponential Moving Average)
    parser.add_argument('--useEma', action='store_true', default=False,
                        help='Use EMA (exponential moving average) of model weights')
    parser.add_argument('--noEma', dest='useEma', action='store_false',
                        help='Disable EMA')
    parser.add_argument('--emaDecay', default=0.9997, type=float,
                        help='EMA decay rate')
    
    # Drop path (stochastic depth)
    parser.add_argument('--dropPathRate', default=[0.0, 0.025, 0.05, 0.075, 0.1], nargs='+', type=float,
                        help='Drop path rate for stochastic depth regularization; per-epoch schedule')
    
    # Dataset parameters
    parser.add_argument('--dataConfig', default='vidDetr/data.yaml', type=str,
                        help='Path to dataset configuration YAML')
    parser.add_argument('--numClasses', default=80, type=int,
                        help='Number of object classes')
    parser.add_argument('--framesPerSequence', default=50, type=int,
                        help='Total frames in each video sequence')
    parser.add_argument('--minFrameGap', default=1, type=int,
                        help='Minimum gap between sampled frames')
    parser.add_argument('--maxFrameGap', default=10, type=int,
                        help='Maximum gap between sampled frames')
    parser.add_argument('--maxSize', default=384, type=int,
                        help='Maximum image size after transforms')
    parser.add_argument('--minBoxSize', default=0.2, type=float,
                        help='Minimum GT box size as fraction of image width or height; '
                             'boxes whose normalised w AND h are both below this threshold are dropped')
    
    # TAO dataset parameters
    parser.add_argument('--taoDataRoot', default=None, #'/mnt/matylda5/xmihol00/tao/dataset/', 
                        type=str,
                        help='Root directory of TAO dataset (overrides --dataConfig)')
    parser.add_argument('--taoMaxCategories', default=None, type=int,
                        help='Keep only top-N most frequent TAO categories (None=all)')
    parser.add_argument('--taoWindowOverlap', default=0.5, type=float,
                        help='Overlap fraction between TAO video windows (0-1)')
    
    # Merge train + val
    parser.add_argument('--mergeTrainVal', action='store_true', default=False,
                        help='Merge training and validation sets into one training set '
                             '(no validation loop; a checkpoint is saved every epoch)')
    
    # Training parameters
    parser.add_argument('--outputDir', default='vidDetr_weights/',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='Device for training')
    parser.add_argument('--seed', default=random.randint(0, 2**32-1), type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--startEpoch', default=0, type=int,
                        help='Starting epoch number')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--numWorkers', default=4, type=int,
                        help='Number of dataloader workers')
    
    # Debug visualisation
    parser.add_argument('--debugFrames', action='store_true', default=True,
                        help='Save one debug frame per batch to debug_frames/ '
                             '(GT green dashed, predictions red solid)')
    
    # Distributed training
    parser.add_argument('--worldSize', default=NUM_GPUS, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--distUrl', default='env://',
                        help='URL for distributed training setup')
    
    # Pretrained weights
    parser.add_argument('--pretrainedDetr', default='/homes/eva/xm/xmihol00/video_detr/weights_2026-02-19/checkpoint_latest.pth', type=str,
                        help='Path to pretrained DETR weights')
    #parser.add_argument('--pretrainedDetr', default='/mnt/matylda5/xmihol00/video_detr/detr-r50-e632da11.pth', type=str,
    #                help='Path to pretrained DETR weights')
    
    return parser


def convertArgsForBackbone(args):
    """
    Convert VideoDETR args to format expected by backbone builder.
    
    The original DETR backbone builder uses snake_case args, so we need
    to provide compatibility.
    """
    # Create a compatible args object
    args.lr_backbone = args.lrBackbone
    args.position_embedding = args.positionEmbedding
    args.hidden_dim = args.hiddenDim
    args.masks = False  # No segmentation for now
    
    return args


def convertArgsForTransformer(args):
    """
    Convert VideoDETR args to format expected by transformer builder.
    
    Note: ``args.dropout`` may be a list (per-epoch schedule).  The DETR
    transformer builder expects a scalar, so we store the original list
    in ``args._dropoutSchedule`` and set ``args.dropout`` to the first
    value for the builder.  ``resolveEpochParams`` reads from the
    schedule list.
    """
    args.enc_layers = args.encLayers
    args.dec_layers = args.decLayers
    args.dim_feedforward = args.dimFeedforward
    args.hidden_dim = args.hiddenDim
    args.pre_norm = args.preNorm
    
    # Transformer builder expects a scalar dropout — preserve the
    # original schedule in a private attribute.
    if isinstance(args.dropout, list):
        args._dropoutSchedule = args.dropout
        args.dropout = args.dropout[0]
    else:
        args._dropoutSchedule = [args.dropout]
    
    return args


def main(args):
    """
    Main training function.
    
    Args:
        args: Parsed command-line arguments
    """
    # Initialize distributed mode
    utils.init_distributed_mode(args)

    # Set up structured logging (file + console)
    rank = utils.get_rank()
    distributed = getattr(args, 'distributed', False)
    vidDetrLogger = setupLogging(
        outputDir=args.outputDir,
        logFilename="training.log",
        distributed=distributed,
        rank=rank,
    )

    vidDetrLogger.info("git: %s", utils.get_sha())
    vidDetrLogger.info("args: %s", args)
    
    device = torch.device(args.device)
    
    # Set random seeds for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Convert args for compatibility
    args = convertArgsForBackbone(args)
    args = convertArgsForTransformer(args)
    
    # Resolve epoch-0 scheduled hyperparameters so that builders
    # (which expect scalar values) receive the correct initial values.
    epoch0Params = resolveEpochParams(args, epoch=args.startEpoch)
    for name, value in epoch0Params.items():
        setattr(args, f'_current_{name}', value)
    logEpochScheduledParams(vidDetrLogger, args.startEpoch, epoch0Params)
    
    # Build datasets FIRST so that dataset-driven numClasses is set
    # before the model and criterion are constructed.
    mergeTrainVal = getattr(args, 'mergeTrainVal', False)
    vidDetrLogger.info("Building datasets...")
    if args.taoDataRoot:
        datasetTrain, datasetVal = buildTaoDataset(args)
        collateFn = taoCollateFn
    else:
        datasetTrain, datasetVal = buildVideoDataset(args)
        collateFn = videoCollateFn
    vidDetrLogger.info("Train dataset: %d sequences", len(datasetTrain))
    if datasetVal is not None:
        vidDetrLogger.info("Val dataset: %d sequences", len(datasetVal))
    else:
        vidDetrLogger.info("Val dataset: None (merged into training set)")
    
    # Build model, criterion, and postprocessors
    model, criterion, postprocessors = buildVideoDETR(args)
    model.to(device)
    
    # Setup for distributed training
    modelWithoutDdp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        modelWithoutDdp = model.module
    
    # Count parameters
    nParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vidDetrLogger.info('Number of trainable parameters: %s', f'{nParameters:,}')
    
    # Setup EMA (Exponential Moving Average)
    emaModel = None
    if getattr(args, 'useEma', True):
        emaModel = ModelEMA(modelWithoutDdp, decay=getattr(args, 'emaDecay', 0.9997))
        vidDetrLogger.info(
            "EMA enabled with decay=%.5f", getattr(args, 'emaDecay', 0.9997)
        )
    
    # Setup optimizer with different learning rates
    paramDicts = [
        {
            "params": [
                p for n, p in modelWithoutDdp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in modelWithoutDdp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lrBackbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        paramDicts,
        lr=args.lr,
        weight_decay=args.weightDecay
    )
    
    # Learning rate scheduler: warmup + step decay
    # StepLR handles the main schedule (drop at lrDrop)
    stepScheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lrDrop)
    
    # Warmup scheduler (linear warmup over warmupEpochs)
    if args.warmupEpochs > 0:
        # Compute warmup factor: start from warmupStartLr / lr and linearly go to 1.0
        warmupStartFactor = args.warmupStartLr / args.lr
        warmupScheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmupStartFactor,
            end_factor=1.0,
            total_iters=args.warmupEpochs
        )
        lrScheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmupScheduler, stepScheduler],
            milestones=[args.warmupEpochs]
        )
    else:
        lrScheduler = stepScheduler
    
    vidDetrLogger.info("Optimizer: AdamW, lr=%s, lr_backbone=%s", args.lr, args.lrBackbone)
    vidDetrLogger.info(
        "LR schedule: warmup %d epochs (%s -> %s), step drop at epoch %d",
        args.warmupEpochs, args.warmupStartLr, args.lr, args.lrDrop,
    )
    vidDetrLogger.info(
        "Gradient accumulation: %d steps (effective batch size = %d)",
        args.accumSteps, args.batchSize * args.accumSteps,
    )
    
    # Setup samplers
    if args.distributed:
        samplerTrain = DistributedSampler(datasetTrain)
    else:
        samplerTrain = torch.utils.data.RandomSampler(datasetTrain)
    
    # Create dataloaders
    batchSamplerTrain = torch.utils.data.BatchSampler(
        samplerTrain, 
        args.batchSize, 
        drop_last=True
    )
    
    dataLoaderTrain = DataLoader(
        datasetTrain,
        batch_sampler=batchSamplerTrain,
        collate_fn=collateFn,
        num_workers=args.numWorkers,
        prefetch_factor=2 if args.numWorkers > 0 else None,
        persistent_workers=args.numWorkers > 0,
        pin_memory=True
    )
    
    dataLoaderVal = None
    if datasetVal is not None:
        if args.distributed:
            samplerVal = DistributedSampler(datasetVal, shuffle=False)
        else:
            samplerVal = torch.utils.data.SequentialSampler(datasetVal)
        dataLoaderVal = DataLoader(
            datasetVal,
            args.batchSize,
            sampler=samplerVal,
            drop_last=False,
            collate_fn=collateFn,
            num_workers=args.numWorkers,
            prefetch_factor=2 if args.numWorkers > 0 else None,
            persistent_workers=args.numWorkers > 0,
            pin_memory=True
        )
    
    # Setup output directory
    outputDir = Path(args.outputDir)
    
    # Load pretrained weights if specified
    if args.pretrainedDetr:
        vidDetrLogger.info("Loading pretrained DETR weights from %s", args.pretrainedDetr)
        checkpoint = torch.load(args.pretrainedDetr, map_location='cpu')
        
        # Load compatible weights (skip mismatched layers)
        modelDict = modelWithoutDdp.state_dict()
        pretrainedDict = {
            k: v for k, v in checkpoint['model'].items()
            if k in modelDict and v.shape == modelDict[k].shape
        }
        notLoaded = set(modelDict.keys()) - set(pretrainedDict.keys())
        if notLoaded:
            vidDetrLogger.warning("%d parameters not loaded from pretrained weights:", len(notLoaded))
            for k in list(notLoaded):
                vidDetrLogger.warning("  %s", k)
        modelDict.update(pretrainedDict)
        modelWithoutDdp.load_state_dict(modelDict, strict=False)
        vidDetrLogger.info("Loaded %d/%d pretrained weights", len(pretrainedDict), len(checkpoint['model']))
    
    # Resume from checkpoint
    bestMetric = float('inf')  # Initialize best metric tracker
    if args.resume:
        vidDetrLogger.info("Resuming from checkpoint: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        modelWithoutDdp.load_state_dict(checkpoint['model'])
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lrScheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.startEpoch = checkpoint['epoch'] + 1
            # Restore best metric if available
            if 'best_metric' in checkpoint:
                bestMetric = checkpoint['best_metric']
                vidDetrLogger.info("Restored best metric: %.4f", bestMetric)
        
        # Restore EMA state if available
        if emaModel is not None and 'ema_state_dict' in checkpoint:
            emaModel.module.load_state_dict(checkpoint['ema_state_dict'])
            vidDetrLogger.info("Restored EMA state")
    
    # Evaluation only
    if args.eval:
        if dataLoaderVal is None:
            vidDetrLogger.error("Cannot evaluate: no validation set (--mergeTrainVal is active)")
            return
        valTracker = MetricTracker(outputDir=args.outputDir, phase="val")
        testStats = evaluate(
            model, criterion, postprocessors,
            dataLoaderVal, device, args.outputDir,
            tracker=valTracker, epoch=0,
        )
        vidDetrLogger.info("Evaluation results: %s", testStats)
        return

    if args.freezeBackbone:
        model.freezeBackbone(freeze=False)
    
    # Create metric trackers for CSV logging
    trainTracker = MetricTracker(outputDir=args.outputDir, phase="train")
    valTracker = MetricTracker(outputDir=args.outputDir, phase="val") if dataLoaderVal is not None else None

    # Debug frames directory
    debugFramesDir = None
    if getattr(args, 'debugFrames', False):
        debugFramesDir = str(outputDir / 'debug_frames')
        os.makedirs(debugFramesDir, exist_ok=True)
        vidDetrLogger.info("Debug frames enabled → %s", debugFramesDir)

    # Training loop
    vidDetrLogger.info("Starting training...")
    if mergeTrainVal:
        vidDetrLogger.info("*** mergeTrainVal is ON — no validation will be performed ***")
    startTime = time.time()
    # bestMetric already initialized above (either from checkpoint or float('inf'))
    
    for epoch in range(args.startEpoch, args.epochs):
        if args.distributed:
            samplerTrain.set_epoch(epoch)
        
        # ---- Resolve per-epoch scheduled hyperparameters ----
        epochParams = resolveEpochParams(args, epoch)
        logEpochScheduledParams(vidDetrLogger, epoch, epochParams)
        
        # Update criterion (loss weights, focal params, contrastive temp, etc.)
        criterion.updateEpochParams(epochParams)
        
        # Update model (denoising noise ratios, dropout, etc.)
        rawModel = model.module if hasattr(model, 'module') else model
        rawModel.updateEpochParams(epochParams)
        
        # Resolve clipMaxNorm for this epoch (engine needs it as scalar)
        currentClipMaxNorm = epochParams['clipMaxNorm']
        
        # Train one epoch
        trainStats = trainOneEpoch(
            model, criterion, dataLoaderTrain,
            optimizer, device, epoch, currentClipMaxNorm,
            accumSteps=args.accumSteps,
            tracker=trainTracker,
            emaModel=emaModel,
            debugFramesDir=debugFramesDir,
        )
        
        lrScheduler.step()
        
        # Evaluate (skip when train+val are merged)
        testStats: Dict[str, float] = {}
        if dataLoaderVal is not None:
            evalModel = emaModel.module if emaModel is not None else model
            testStats = evaluate(
                evalModel, criterion, postprocessors,
                dataLoaderVal, device, args.outputDir,
                tracker=valTracker, epoch=epoch,
            )
        
        # Save checkpoint after every epoch
        if args.outputDir:
            checkpointPaths = []
            
            # 1. Save every epoch with unique name (video_detr_XXX.pth format)
            epochCheckpoint = outputDir / f'video_detr_{epoch+1:03d}.pth'
            #checkpointPaths.append(epochCheckpoint)
            
            # 2. Always update the latest checkpoint
            latestCheckpoint = outputDir / 'checkpoint_latest.pth'
            checkpointPaths.append(latestCheckpoint)
            
            # 3. Check if this is the best model so far (lowest validation loss)
            #    Only meaningful when a validation set exists.
            if testStats:
                currentMetric = testStats.get('loss_mean', float('inf'))
                if currentMetric < bestMetric:
                    bestMetric = currentMetric
                    bestCheckpoint = outputDir / 'video_detr_best.pth'
                    checkpointPaths.append(bestCheckpoint)
                    vidDetrLogger.info("*** New best model at epoch %d with loss=%.4f ***", epoch + 1, currentMetric)
            
            # Save all checkpoint paths
            checkpoint_data = {
                'model': modelWithoutDdp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lrScheduler.state_dict(),
                'epoch': epoch,
                'best_metric': bestMetric,
                'args': args,
            }
            
            # Include EMA state in checkpoint
            if emaModel is not None:
                checkpoint_data['ema_state_dict'] = emaModel.module.state_dict()
            
            for checkpointPath in checkpointPaths:
                utils.save_on_master(checkpoint_data, checkpointPath)
            
            vidDetrLogger.info("Checkpoints saved: %s", [p.name for p in checkpointPaths])
        
        # Log epoch summary to the structured log
        if testStats:
            vidDetrLogger.info(
                "Epoch %d summary — train_loss: %.4f  val_loss: %.4f  best: %.4f",
                epoch,
                trainStats.get('loss_mean', 0.0),
                testStats.get('loss_mean', 0.0),
                bestMetric,
            )
        else:
            vidDetrLogger.info(
                "Epoch %d summary — train_loss: %.4f",
                epoch,
                trainStats.get('loss_mean', 0.0),
            )
    
    totalTime = time.time() - startTime
    totalTimeStr = str(datetime.timedelta(seconds=int(totalTime)))
    vidDetrLogger.info('Training completed in %s', totalTimeStr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'VideoDETR training and evaluation',
        parents=[getArgsParser()]
    )
    args = parser.parse_args()
    
    # Create output directory
    if args.outputDir:
        outputDir = Path(args.outputDir)
        
        # Handle existing directory with versioning
        if outputDir.exists():
            suffix = outputDir.name.split('_')[-1]
            if suffix.isdigit():
                newSuffix = int(suffix) + 1
                args.outputDir = '_'.join(outputDir.name.split('_')[:-1]) + f'_{newSuffix}'
            else:
                args.outputDir = str(outputDir) + '_1'
        
        Path(args.outputDir).mkdir(parents=True, exist_ok=True)
    
    print(f"Using output directory: {args.outputDir}")  # before logger is set up
    main(args)
