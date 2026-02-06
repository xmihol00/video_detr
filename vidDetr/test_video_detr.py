#!/usr/bin/env python
# Copyright (c) 2026. All Rights Reserved.
"""
Test script for VideoDETR components.

This script tests that all components build correctly and can perform
a forward pass with synthetic data on CPU.

Usage:
    python test_video_detr.py
"""

import sys
from pathlib import Path
import argparse
import os

# Disable triton to avoid environment issues
os.environ['TRITON_DISABLE'] = '1'

# Add parent directory for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import torch
import torch.nn as nn


def testTemporalEncoding():
    """Test temporal position encoding module."""
    print("\n" + "=" * 60)
    print("Testing Temporal Position Encoding")
    print("=" * 60)
    
    from vidDetr.models.temporal_encoding import (
        TemporalPositionEncoding,
        TemporalPositionEncodingSine,
        TemporalPositionEncodingLearned
    )
    
    # Test sinusoidal encoding
    print("\n1. Testing sinusoidal temporal encoding...")
    sineEnc = TemporalPositionEncodingSine(numPosFeats=128, maxFrames=100)
    frameIndices = torch.arange(5)
    output = sineEnc(frameIndices, numFrames=5)
    print(f"   Input shape: {frameIndices.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (5, 256), f"Expected (5, 256), got {output.shape}"
    print("   ✓ Sinusoidal encoding OK")
    
    # Test learned encoding
    print("\n2. Testing learned temporal encoding...")
    learnedEnc = TemporalPositionEncodingLearned(numPosFeats=256, maxFrames=100)
    output = learnedEnc(frameIndices, numFrames=5)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (5, 256), f"Expected (5, 256), got {output.shape}"
    print("   ✓ Learned encoding OK")
    
    # Test combined encoding
    print("\n3. Testing combined spatial-temporal encoding...")
    combinedEnc = TemporalPositionEncoding(
        hiddenDim=256,
        temporalType='learned',
        maxFrames=100
    )
    spatialPos = torch.randn(2, 256, 25, 25)  # [B, C, H, W]
    output = combinedEnc(spatialPos, numFrames=5)
    print(f"   Spatial pos shape: {spatialPos.shape}")
    print(f"   Output shape: {output.shape}")
    print("   ✓ Combined encoding OK")
    
    print("\n✓ All temporal encoding tests passed!")


def testTrackingHead():
    """Test tracking head module."""
    print("\n" + "=" * 60)
    print("Testing Tracking Head")
    print("=" * 60)
    
    from vidDetr.models.tracking_head import TrackingHead
    
    # Test basic tracking head
    print("\n1. Testing tracking head forward pass...")
    trackingHead = TrackingHead(
        inputDim=256,
        hiddenDim=256,
        outputDim=128,
        numLayers=3,
        normalize=True
    )
    
    # Single batch input
    x = torch.randn(2, 375, 256)  # [B, numQueries, hiddenDim]
    output = trackingHead(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (2, 375, 128), f"Expected (2, 375, 128), got {output.shape}"
    
    # Check normalization
    norms = output.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings not normalized"
    print("   ✓ Output is L2-normalized")
    
    # Test with decoder layer dimension
    print("\n2. Testing with decoder layer dimension...")
    x = torch.randn(6, 2, 375, 256)  # [numLayers, B, numQueries, hiddenDim]
    output = trackingHead(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (6, 2, 375, 128)
    print("   ✓ Multi-layer input OK")
    
    print("\n✓ All tracking head tests passed!")


def testContrastiveLoss():
    """Test supervised contrastive loss."""
    print("\n" + "=" * 60)
    print("Testing Supervised Contrastive Loss")
    print("=" * 60)
    
    from vidDetr.losses.contrastive_loss import SupervisedContrastiveLoss
    
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    
    # Test case 1: Perfect embeddings (same track = same embedding)
    print("\n1. Testing with identical positive pairs...")
    embeddings = torch.randn(10, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    
    # 5 objects, each appearing in 2 frames
    trackIds = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    frameIndices = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    loss = loss_fn(embeddings, trackIds, frameIndices)
    print(f"   Loss value: {loss.item():.4f}")
    print("   ✓ Loss computed successfully")
    
    # Test case 2: Empty input
    print("\n2. Testing with empty input...")
    emptyEmbeddings = torch.randn(0, 128)
    emptyTrackIds = torch.tensor([])
    emptyFrameIndices = torch.tensor([])
    
    loss = loss_fn(emptyEmbeddings, emptyTrackIds, emptyFrameIndices)
    print(f"   Loss value: {loss.item():.4f}")
    assert loss.item() == 0.0, "Empty input should give 0 loss"
    print("   ✓ Empty input handled correctly")
    
    # Test case 3: No positive pairs
    print("\n3. Testing with no positive pairs...")
    embeddings = torch.randn(5, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    trackIds = torch.tensor([0, 1, 2, 3, 4])  # All different tracks
    frameIndices = torch.tensor([0, 0, 0, 0, 0])  # Same frame
    
    loss = loss_fn(embeddings, trackIds, frameIndices)
    print(f"   Loss value: {loss.item():.4f}")
    print("   ✓ No positive pairs handled correctly")
    
    print("\n✓ All contrastive loss tests passed!")


def testVideoMatcher():
    """Test video Hungarian matcher."""
    print("\n" + "=" * 60)
    print("Testing Video Hungarian Matcher")
    print("=" * 60)
    
    from vidDetr.losses.video_criterion import VideoHungarianMatcher
    
    matcher = VideoHungarianMatcher(
        costClass=1.0,
        costBbox=5.0,
        costGiou=2.0,
        numFrames=3,
        queriesPerFrame=10
    )
    
    # Create synthetic outputs and targets
    batchSize = 2
    numQueries = 30  # 3 frames * 10 queries
    numClasses = 10
    
    outputs = {
        'pred_logits': torch.randn(batchSize, numQueries, numClasses + 1),
        'pred_boxes': torch.rand(batchSize, numQueries, 4).sigmoid()
    }
    
    # Create targets: 2 objects per frame
    targets = []
    for b in range(batchSize):
        batchTargets = []
        for f in range(3):
            frameTarget = {
                'labels': torch.randint(0, numClasses, (2,)),
                'boxes': torch.rand(2, 4),
                'trackIds': torch.tensor([0, 1])
            }
            batchTargets.append(frameTarget)
        targets.append(batchTargets)
    
    print("\n1. Testing matching computation...")
    indices = matcher(outputs, targets)
    
    print(f"   Batch size: {batchSize}")
    print(f"   Num frames: 3")
    print(f"   Indices structure: {len(indices)} batches x {len(indices[0])} frames")
    
    # Verify structure
    assert len(indices) == batchSize
    for batchIndices in indices:
        assert len(batchIndices) == 3  # 3 frames
        for predIdx, tgtIdx in batchIndices:
            assert len(predIdx) == len(tgtIdx)
            assert len(predIdx) <= 10  # Max queries per frame
    
    print("   ✓ Matching structure OK")
    
    print("\n✓ All matcher tests passed!")


def testVideoCriterion():
    """Test video criterion (full loss computation)."""
    print("\n" + "=" * 60)
    print("Testing Video Criterion")
    print("=" * 60)
    
    from vidDetr.losses.video_criterion import VideoCriterion, VideoHungarianMatcher
    
    # Setup
    numFrames = 3
    queriesPerFrame = 10
    numClasses = 10
    batchSize = 2
    
    matcher = VideoHungarianMatcher(
        costClass=1.0,
        costBbox=5.0,
        costGiou=2.0,
        numFrames=numFrames,
        queriesPerFrame=queriesPerFrame
    )
    
    weightDict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
        'loss_tracking': 1.0
    }
    
    criterion = VideoCriterion(
        numClasses=numClasses,
        matcher=matcher,
        weightDict=weightDict,
        eosCoef=0.1,
        losses=['labels', 'boxes', 'cardinality', 'tracking'],
        numFrames=numFrames,
        queriesPerFrame=queriesPerFrame
    )
    
    # Create synthetic data
    numQueries = numFrames * queriesPerFrame
    
    outputs = {
        'pred_logits': torch.randn(batchSize, numQueries, numClasses + 1, requires_grad=True),
        'pred_boxes': torch.rand(batchSize, numQueries, 4, requires_grad=True).sigmoid(),
        'pred_tracking': torch.randn(batchSize, numQueries, 128, requires_grad=True)
    }
    outputs['pred_tracking'] = torch.nn.functional.normalize(
        outputs['pred_tracking'], p=2, dim=-1
    )
    
    targets = []
    for b in range(batchSize):
        batchTargets = []
        for f in range(numFrames):
            frameTarget = {
                'labels': torch.randint(0, numClasses, (3,)),
                'boxes': torch.rand(3, 4),
                'trackIds': torch.tensor([0, 1, 2])
            }
            batchTargets.append(frameTarget)
        targets.append(batchTargets)
    
    print("\n1. Testing full criterion forward pass...")
    losses = criterion(outputs, targets)
    
    print("   Computed losses:")
    for k, v in losses.items():
        print(f"     {k}: {v.item():.4f}")
    
    # Verify all expected losses are computed
    assert 'loss_ce' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    assert 'loss_tracking' in losses
    assert 'cardinality_error' in losses
    
    print("   ✓ All losses computed")
    
    # Test backward pass
    print("\n2. Testing backward pass...")
    totalLoss = sum(
        losses[k] * weightDict.get(k, 1.0)
        for k in losses.keys()
        if k in weightDict and losses[k].requires_grad
    )
    totalLoss.backward()
    print(f"   Total loss: {totalLoss.item():.4f}")
    print("   ✓ Backward pass OK")
    
    print("\n✓ All criterion tests passed!")


def testVideoDETRModel():
    """Test the full VideoDETR model."""
    print("\n" + "=" * 60)
    print("Testing VideoDETR Model (CPU)")
    print("=" * 60)
    
    # Create a minimal args namespace
    class Args:
        # Backbone
        backbone = 'resnet50'
        dilation = False
        position_embedding = 'sine'
        lr_backbone = 1e-5
        masks = False
        hidden_dim = 256
        
        # Transformer
        enc_layers = 2  # Reduced for faster testing
        dec_layers = 2
        dim_feedforward = 512  # Reduced for memory
        nheads = 8
        dropout = 0.1
        pre_norm = False
        
        # VideoDETR specific
        numFrames = 2  # Reduced for testing
        queriesPerFrame = 10  # Reduced for testing
        numClasses = 10
        auxLoss = True
        trackingEmbedDim = 64
        temporalEncoding = 'learned'
        
        # Loss
        setCostClass = 1.0
        setCostBbox = 5.0
        setCostGiou = 2.0
        bboxLossCoef = 5.0
        giouLossCoef = 2.0
        eosCoef = 0.1
        trackingLossCoef = 1.0
        contrastiveTemp = 0.07
        decLayers = 2
        
        device = 'cpu'
    
    args = Args()
    
    print("\n1. Building model components...")
    
    from models.backbone import build_backbone
    from models.transformer import build_transformer
    from vidDetr.models.video_detr import VideoDETR
    from vidDetr.losses.video_criterion import buildVideoCriterion
    
    print("   Building backbone...")
    backbone = build_backbone(args)
    print(f"   Backbone channels: {backbone.num_channels}")
    
    print("   Building transformer...")
    transformer = build_transformer(args)
    print(f"   Transformer d_model: {transformer.d_model}")
    
    print("   Building VideoDETR...")
    model = VideoDETR(
        backbone=backbone,
        transformer=transformer,
        numClasses=args.numClasses,
        numFrames=args.numFrames,
        queriesPerFrame=args.queriesPerFrame,
        auxLoss=args.auxLoss,
        trackingEmbedDim=args.trackingEmbedDim,
        temporalType=args.temporalEncoding
    )
    
    # Count parameters
    nParams = sum(p.numel() for p in model.parameters())
    nTrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {nParams:,}")
    print(f"   Trainable parameters: {nTrainable:,}")
    
    # Create synthetic input
    print("\n2. Testing forward pass...")
    from util.misc import NestedTensor
    
    batchSize = 1
    imgSize = 224  # Smaller for CPU testing
    
    samples = []
    for f in range(args.numFrames):
        tensors = torch.randn(batchSize, 3, imgSize, imgSize)
        mask = torch.zeros(batchSize, imgSize, imgSize, dtype=torch.bool)
        samples.append(NestedTensor(tensors, mask))
    
    print(f"   Input: {args.numFrames} frames of size {imgSize}x{imgSize}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(samples)
    
    print("   Output shapes:")
    print(f"     pred_logits: {outputs['pred_logits'].shape}")
    print(f"     pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"     pred_tracking: {outputs['pred_tracking'].shape}")
    
    expectedQueries = args.numFrames * args.queriesPerFrame
    assert outputs['pred_logits'].shape == (batchSize, expectedQueries, args.numClasses + 1)
    assert outputs['pred_boxes'].shape == (batchSize, expectedQueries, 4)
    assert outputs['pred_tracking'].shape == (batchSize, expectedQueries, args.trackingEmbedDim)
    
    if args.auxLoss:
        print(f"     aux_outputs: {len(outputs['aux_outputs'])} layers")
    
    print("   ✓ Forward pass OK")
    
    # Test with criterion
    print("\n3. Testing with criterion...")
    criterion = buildVideoCriterion(args)
    
    # Create targets
    targets = []
    for b in range(batchSize):
        batchTargets = []
        for f in range(args.numFrames):
            frameTarget = {
                'labels': torch.randint(0, args.numClasses, (2,)),
                'boxes': torch.rand(2, 4),
                'trackIds': torch.tensor([0, 1])
            }
            batchTargets.append(frameTarget)
        targets.append(batchTargets)
    
    model.train()
    outputs = model(samples)
    losses = criterion(outputs, targets)
    
    print("   Losses:")
    for k, v in sorted(losses.items()):
        if not k.endswith('_unscaled'):
            print(f"     {k}: {v.item():.4f}")
    
    # Test backward
    totalLoss = sum(
        losses[k] * criterion.weightDict.get(k, 1.0)
        for k in losses.keys()
        if k in criterion.weightDict
    )
    totalLoss.backward()
    print(f"   Total loss: {totalLoss.item():.4f}")
    print("   ✓ Backward pass OK")
    
    print("\n✓ All VideoDETR model tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VideoDETR Component Tests")
    print("=" * 60)
    
    try:
        testTemporalEncoding()
        testTrackingHead()
        testContrastiveLoss()
        testVideoMatcher()
        testVideoCriterion()
        testVideoDETRModel()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
