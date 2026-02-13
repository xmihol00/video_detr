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
import json
import random
import time
from typing import Optional

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
from vidDetr.engine import trainOneEpoch, evaluate

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
    parser.add_argument('--batchSize', default=26, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--weightDecay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--lrDrop', default=80, type=int,
                        help='Epoch to drop learning rate')
    parser.add_argument('--clipMaxNorm', default=0.1, type=float,
                        help='Gradient clipping max norm')
    
    # Warmup parameters
    parser.add_argument('--warmupEpochs', default=5, type=int,
                        help='Number of warmup epochs with linearly increasing LR')
    parser.add_argument('--warmupStartLr', default=1e-6, type=float,
                        help='Starting learning rate for warmup')
    
    # Gradient accumulation
    parser.add_argument('--accumSteps', default=4, type=int,
                        help='Gradient accumulation steps (effective batch = batchSize * accumSteps)')
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='CNN backbone architecture')
    parser.add_argument('--freezeBackbone', action='store_true', default=True,
                        help='Freeze backbone parameters during training')
    parser.add_argument('--dilation', action='store_true',
                        help='Use dilation in last backbone block')
    parser.add_argument('--positionEmbedding', default='sine', type=str,
                        choices=['sine', 'learned'],
                        help='Type of spatial positional embedding')
    parser.add_argument('--temporalEncoding', default='learned', type=str,
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
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--preNorm', action='store_true',
                        help='Use pre-normalization in transformer')
    
    # VideoDETR specific parameters
    parser.add_argument('--numFrames', default=4, type=int,
                        help='Number of frames per video clip')
    parser.add_argument('--queriesPerFrame', default=50, type=int,
                        help='Number of detection queries per frame')
    parser.add_argument('--trackingEmbedDim', default=128, type=int,
                        help='Dimension of tracking embeddings')
    parser.add_argument('--maxFrames', default=100, type=int,
                        help='Maximum frames for temporal encoding')
    
    # Loss parameters
    parser.add_argument('--auxLoss', action='store_true', default=True,
                        help='Use auxiliary losses at each decoder layer')
    parser.add_argument('--noAuxLoss', dest='auxLoss', action='store_false',
                        help='Disable auxiliary losses')
    parser.add_argument('--useFocalLoss', action='store_true', default=False,
                        help='Use focal loss instead of cross-entropy for classification')
    parser.add_argument('--focalAlpha', default=0.4, type=float,
                        help='Focal loss alpha (balancing factor)')
    parser.add_argument('--focalGamma', default=1.4, type=float,
                        help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--setCostClass', default=3.0, type=float,
                        help='Classification cost in matching')
    parser.add_argument('--setCostBbox', default=4.0, type=float,
                        help='L1 box cost in matching')
    parser.add_argument('--setCostGiou', default=2.0, type=float,
                        help='GIoU cost in matching')
    parser.add_argument('--bboxLossCoef', default=4.0, type=float,
                        help='L1 box loss coefficient')
    parser.add_argument('--giouLossCoef', default=2.0, type=float,
                        help='GIoU loss coefficient')
    parser.add_argument('--eosCoef', default=0.01, type=float,
                        help='No-object class weight')
    parser.add_argument('--trackingLossCoef', default=2.0, type=float,
                        help='Tracking contrastive loss coefficient')
    parser.add_argument('--contrastiveTemp', default=0.07, type=float,
                        help='Temperature for contrastive loss')
    
    # Label denoising (DN-DETR / DINO) parameters
    parser.add_argument('--useDnDenoising', action='store_true', default=False,
                        help='Use label denoising for training stabilization')
    parser.add_argument('--noDnDenoising', dest='useDnDenoising', action='store_false',
                        help='Disable label denoising')
    parser.add_argument('--numDnGroups', default=5, type=int,
                        help='Number of denoising groups (each group = noised copy of all GTs)')
    parser.add_argument('--labelNoiseRatio', default=0.5, type=float,
                        help='Probability of flipping a label to random class in DN queries')
    parser.add_argument('--boxNoiseScale', default=0.4, type=float,
                        help='Scale of box coordinate noise (fraction of box size)')
    parser.add_argument('--dnLossCoef', default=1.0, type=float,
                        help='Coefficient for denoising losses (multiplied with base loss coefs)')
    
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
    
    # TAO dataset parameters
    parser.add_argument('--taoDataRoot', default='/mnt/matylda5/xmihol00/tao/dataset/', type=str,
                        help='Root directory of TAO dataset (overrides --dataConfig)')
    parser.add_argument('--taoMaxCategories', default=None, type=int,
                        help='Keep only top-N most frequent TAO categories (None=all)')
    parser.add_argument('--taoWindowOverlap', default=0.5, type=float,
                        help='Overlap fraction between TAO video windows (0-1)')
    
    # Training parameters
    parser.add_argument('--outputDir', default='vidDetr_weights/',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='Device for training')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--startEpoch', default=0, type=int,
                        help='Starting epoch number')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--numWorkers', default=4, type=int,
                        help='Number of dataloader workers')
    
    # Distributed training
    parser.add_argument('--worldSize', default=NUM_GPUS, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--distUrl', default='env://',
                        help='URL for distributed training setup')
    
    # Pretrained weights
    parser.add_argument('--pretrainedDetr', default='/mnt/matylda5/xmihol00/video_detr/vidDetr_weights_1/video_detr_best.pth', type=str,
                        help='Path to pretrained DETR weights')
    
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
    """
    args.enc_layers = args.encLayers
    args.dec_layers = args.decLayers
    args.dim_feedforward = args.dimFeedforward
    args.hidden_dim = args.hiddenDim
    args.pre_norm = args.preNorm
    
    return args


def main(args):
    """
    Main training function.
    
    Args:
        args: Parsed command-line arguments
    """
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    print(f"git:\n  {utils.get_sha()}\n")
    print(args)
    
    device = torch.device(args.device)
    
    # Set random seeds for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Convert args for compatibility
    args = convertArgsForBackbone(args)
    args = convertArgsForTransformer(args)
    
    # Build datasets FIRST so that dataset-driven numClasses is set
    # before the model and criterion are constructed.
    print("Building datasets...")
    if args.taoDataRoot:
        datasetTrain, datasetVal = buildTaoDataset(args)
        collateFn = taoCollateFn
    else:
        datasetTrain, datasetVal = buildVideoDataset(args)
        collateFn = videoCollateFn
    print(f"Train dataset: {len(datasetTrain)} sequences")
    print(f"Val dataset: {len(datasetVal)} sequences")
    
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
    print(f'Number of trainable parameters: {nParameters:,}')
    
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
    
    print(f"Optimizer: AdamW, lr={args.lr}, lr_backbone={args.lrBackbone}")
    print(f"LR schedule: warmup {args.warmupEpochs} epochs ({args.warmupStartLr} -> {args.lr}), "
          f"step drop at epoch {args.lrDrop}")
    print(f"Gradient accumulation: {args.accumSteps} steps "
          f"(effective batch size = {args.batchSize * args.accumSteps})")
    
    # Setup samplers
    if args.distributed:
        samplerTrain = DistributedSampler(datasetTrain)
        samplerVal = DistributedSampler(datasetVal, shuffle=False)
    else:
        samplerTrain = torch.utils.data.RandomSampler(datasetTrain)
        samplerVal = torch.utils.data.SequentialSampler(datasetVal)
    
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
        print(f"Loading pretrained DETR weights from {args.pretrainedDetr}")
        checkpoint = torch.load(args.pretrainedDetr, map_location='cpu')
        
        # Load compatible weights (skip mismatched layers)
        modelDict = modelWithoutDdp.state_dict()
        pretrainedDict = {
            k: v for k, v in checkpoint['model'].items()
            if k in modelDict and v.shape == modelDict[k].shape
        }
        notLoaded = set(modelDict.keys()) - set(pretrainedDict.keys())
        if notLoaded:
            print(f"Warning: {len(notLoaded)} parameters not loaded from pretrained weights:")
            for k in list(notLoaded):  # Print up to 10 missing keys
                print(f"  {k}")
        modelDict.update(pretrainedDict)
        modelWithoutDdp.load_state_dict(modelDict, strict=False)
        print(f"Loaded {len(pretrainedDict)}/{len(checkpoint['model'])} pretrained weights")
    
    # Resume from checkpoint
    bestMetric = float('inf')  # Initialize best metric tracker
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        modelWithoutDdp.load_state_dict(checkpoint['model'])
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lrScheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.startEpoch = checkpoint['epoch'] + 1
            # Restore best metric if available
            if 'best_metric' in checkpoint:
                bestMetric = checkpoint['best_metric']
                print(f"Restored best metric: {bestMetric:.4f}")
    
    # Evaluation only
    if args.eval:
        testStats = evaluate(
            model, criterion, postprocessors,
            dataLoaderVal, device, args.outputDir
        )
        print(f"Evaluation results: {testStats}")
        return

    if args.freezeBackbone:
        model.freezeBackbone(freeze=False)
    
    # Training loop
    print("Starting training...")
    startTime = time.time()
    # bestMetric already initialized above (either from checkpoint or float('inf'))
    
    for epoch in range(args.startEpoch, args.epochs):
        if args.distributed:
            samplerTrain.set_epoch(epoch)
        
        # Train one epoch
        trainStats = trainOneEpoch(
            model, criterion, dataLoaderTrain,
            optimizer, device, epoch, args.clipMaxNorm,
            accumSteps=args.accumSteps
        )
        
        lrScheduler.step()
        
        # Evaluate
        testStats = evaluate(
            model, criterion, postprocessors,
            dataLoaderVal, device, args.outputDir
        )
        
        # Save checkpoint after every epoch
        if args.outputDir:
            checkpointPaths = []
            
            # 1. Save every epoch with unique name (video_detr_XXX.pth format)
            epochCheckpoint = outputDir / f'video_detr_{epoch+1:03d}.pth'
            checkpointPaths.append(epochCheckpoint)
            
            # 2. Always update the latest checkpoint
            latestCheckpoint = outputDir / 'checkpoint_latest.pth'
            checkpointPaths.append(latestCheckpoint)
            
            # 3. Check if this is the best model so far (lowest validation loss)
            currentMetric = testStats.get('loss', float('inf'))
            if currentMetric < bestMetric:
                bestMetric = currentMetric
                bestCheckpoint = outputDir / 'video_detr_best.pth'
                checkpointPaths.append(bestCheckpoint)
                print(f"*** New best model at epoch {epoch+1} with loss={currentMetric:.4f} ***")
            
            # Save all checkpoint paths
            checkpoint_data = {
                'model': modelWithoutDdp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lrScheduler.state_dict(),
                'epoch': epoch,
                'best_metric': bestMetric,
                'args': args,
            }
            
            for checkpointPath in checkpointPaths:
                utils.save_on_master(checkpoint_data, checkpointPath)
            
            print(f"Checkpoints saved: {[p.name for p in checkpointPaths]}")
        
        # Log stats
        logStats = {
            **{f'train_{k}': v for k, v in trainStats.items()},
            **{f'test_{k}': v for k, v in testStats.items()},
            'epoch': epoch,
            'n_parameters': nParameters,
            'best_metric': bestMetric
        }
        
        if args.outputDir and utils.is_main_process():
            with (outputDir / "log.txt").open("a") as f:
                f.write(json.dumps(logStats) + "\n")
    
    totalTime = time.time() - startTime
    totalTimeStr = str(datetime.timedelta(seconds=int(totalTime)))
    print(f'Training completed in {totalTimeStr}')


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
    
    print(f"Using output directory: {args.outputDir}")
    main(args)
