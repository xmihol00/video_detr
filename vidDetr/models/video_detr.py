# Copyright (c) 2026. All Rights Reserved.
"""
VideoDETR Model for Video Object Detection and Tracking.

This module extends the original DETR architecture to handle multiple video
frames simultaneously, with temporal positional encoding and a tracking head
for object association across frames.

Key modifications from DETR:
1. Process N frames through a shared backbone
2. Add temporal positional encoding to distinguish frames
3. Increase queries to N * queries_per_frame
4. Add tracking embedding head for cross-frame association
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple, Any

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from models.backbone import build_backbone
from models.transformer import build_transformer

from .temporal_encoding import TemporalPositionEncoding, buildTemporalEncoding
from .tracking_head import TrackingHead
from .denoising import DenoisingGenerator


class MLP(nn.Module):
    """
    Simple multi-layer perceptron (also called FFN).
    
    Used for bounding box regression head.
    """
    
    def __init__(self, inputDim: int, hiddenDim: int, outputDim: int, numLayers: int):
        super().__init__()
        self.numLayers = numLayers
        h = [hiddenDim] * (numLayers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([inputDim] + h, h + [outputDim])
        )
    
    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.numLayers - 1 else layer(x)
        return x


class VideoDETR(nn.Module):
    """
    VideoDETR: Video Object Detection and Tracking with Transformers.
    
    This model extends DETR to process multiple video frames simultaneously,
    enabling end-to-end detection and tracking.
    
    Architecture:
    1. Shared CNN backbone extracts features from each frame
    2. Temporal positional encoding is added to frame features
    3. All frame features are concatenated and processed by the transformer
    4. Detection heads predict class and box for each query
    5. Tracking head predicts embeddings for cross-frame association
    
    Args:
        backbone: CNN backbone module (e.g., ResNet50)
        transformer: Transformer encoder-decoder module
        numClasses: Number of object classes (excluding no-object)
        numFrames: Number of frames to process per clip
        queriesPerFrame: Number of detection queries per frame (default: 75)
        auxLoss: Whether to use auxiliary losses at each decoder layer
        hiddenDim: Transformer hidden dimension (inferred from transformer)
        trackingEmbedDim: Dimension of tracking embeddings (default: 128)
        temporalType: Type of temporal encoding ('sine' or 'learned')
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        numClasses: int,
        numFrames: int = 5,
        queriesPerFrame: int = 75,
        auxLoss: bool = True,
        trackingEmbedDim: int = 128,
        temporalType: str = 'learned',
        # Label denoising params
        useDnDenoising: bool = True,
        numDnGroups: int = 5,
        labelNoiseRatio: float = 0.5,
        boxNoiseScale: float = 0.4
    ):
        super().__init__()
        
        self.numFrames = numFrames
        self.queriesPerFrame = queriesPerFrame
        self.numQueries = numFrames * queriesPerFrame
        self.auxLoss = auxLoss
        self.numClasses = numClasses
        self.useDnDenoising = useDnDenoising
        
        # Core components
        self.backbone = backbone
        self.transformer = transformer
        
        # Get hidden dimension from transformer
        hiddenDim = transformer.d_model
        self.hiddenDim = hiddenDim
        
        # Input projection from backbone channels to transformer hidden dim
        self.inputProj = nn.Conv2d(backbone.num_channels, hiddenDim, kernel_size=1)
        
        # Temporal positional encoding
        self.temporalEncoding = TemporalPositionEncoding(
            hiddenDim=hiddenDim,
            temporalType=temporalType,
            maxFrames=100
        )
        
        # Query embeddings
        # We organize queries by frame: queries 0 to Q-1 for frame 0, Q to 2Q-1 for frame 1, etc.
        self.queryEmbed = nn.Embedding(self.numQueries, hiddenDim)
        
        # Frame-specific query embeddings (added to base query embeddings)
        self.frameQueryEmbed = nn.Embedding(numFrames, hiddenDim)
        
        # Detection heads (same as DETR)
        self.classEmbed = nn.Linear(hiddenDim, numClasses + 1)  # +1 for no-object
        self.bboxEmbed = MLP(hiddenDim, hiddenDim, 4, 3)
        
        # Tracking head for cross-frame association
        self.trackingHead = TrackingHead(
            inputDim=hiddenDim,
            hiddenDim=hiddenDim,
            outputDim=trackingEmbedDim,
            numLayers=3,
            normalize=True
        )
        
        # Label denoising generator (DN-DETR / DINO style)
        if useDnDenoising:
            self.denoisingGenerator = DenoisingGenerator(
                hiddenDim=hiddenDim,
                numClasses=numClasses,
                numDnGroups=numDnGroups,
                labelNoiseRatio=labelNoiseRatio,
                boxNoiseScale=boxNoiseScale,
                numFrames=numFrames,
                queriesPerFrame=queriesPerFrame
            )
            print(f"[VideoDETR] Label denoising enabled: "
                  f"{numDnGroups} groups, label_noise={labelNoiseRatio}, "
                  f"box_noise={boxNoiseScale}")
        else:
            self.denoisingGenerator = None
        
        self._initializeQueryEmbeddings()
    
    def _initializeQueryEmbeddings(self):
        """Initialize query embeddings with proper scheme."""
        nn.init.xavier_uniform_(self.queryEmbed.weight)
        nn.init.xavier_uniform_(self.frameQueryEmbed.weight)
    
    def _getFrameQueries(self) -> Tensor:
        """
        Get query embeddings with frame-specific components.
        
        Returns:
            Query embeddings of shape [numQueries, hiddenDim]
        """
        device = self.queryEmbed.weight.device
        
        # Base query embeddings [numQueries, hiddenDim]
        baseQueries = self.queryEmbed.weight
        
        # Add frame-specific embeddings
        # Create frame indices for each query
        frameIndices = torch.arange(self.numFrames, device=device)
        frameIndices = frameIndices.repeat_interleave(self.queriesPerFrame)
        
        # Get frame embeddings [numQueries, hiddenDim]
        frameEmbed = self.frameQueryEmbed(frameIndices)
        
        # Combine
        queries = baseQueries + frameEmbed
        
        return queries
    
    def freezeBackbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone parameters."""
        print(f"{'Freezing' if freeze else 'Unfreezing'} backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def forward(
        self,
        samples: List[NestedTensor],
        targets: Optional[List[List[Dict[str, Tensor]]]] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass for VideoDETR.
        
        Args:
            samples: List of N NestedTensors, one per frame. Each NestedTensor
                    contains batched images [B, 3, H, W] and masks [B, H, W]
            targets: Optional frame-first targets for denoising training.
                    targets[frameIdx][batchIdx] = dict with 'labels', 'boxes', etc.
                    Only needed during training when useDnDenoising=True.
        
        Returns:
            Dict with:
            - 'pred_logits': Classification logits [B, numQueries, numClasses+1]
            - 'pred_boxes': Box predictions [B, numQueries, 4] in cxcywh format
            - 'pred_tracking': Tracking embeddings [B, numQueries, trackingEmbedDim]
            - 'aux_outputs': Optional list of intermediate predictions
            - 'dn_meta': Optional denoising metadata (training only)
            - 'dn_pred_logits': Optional DN classification logits (training only)
            - 'dn_pred_boxes': Optional DN box predictions (training only)
            - 'dn_aux_outputs': Optional DN auxiliary outputs (training only)
        """
        # Validate input
        assert len(samples) == self.numFrames, \
            f"Expected {self.numFrames} frames, got {len(samples)}"
        
        # Process each frame through backbone
        allFeatures = []
        allPositions = []
        allMasks = []
        
        for frameIdx, frameSample in enumerate(samples):
            # Ensure NestedTensor format
            if isinstance(frameSample, (list, torch.Tensor)):
                frameSample = nested_tensor_from_tensor_list(frameSample)
            
            # Extract features and spatial positions from backbone
            features, pos = self.backbone(frameSample)
            
            # Get the last feature level (typically from layer4)
            src, mask = features[-1].decompose()
            spatialPos = pos[-1]
            
            # Project features to hidden dimension
            src = self.inputProj(src)
            
            # Add temporal position encoding
            temporalEnc = self.temporalEncoding.getTemporalEncoding(
                self.numFrames, src.device
            )
            # temporalEnc is [numFrames, hiddenDim]
            # Expand to [B, hiddenDim, H, W] for this frame
            batchSize, _, h, w = src.shape
            temporalEncFrame = temporalEnc[frameIdx].view(1, -1, 1, 1)
            temporalEncFrame = temporalEncFrame.expand(batchSize, -1, h, w)
            
            # Combine spatial and temporal position encodings
            combinedPos = spatialPos + temporalEncFrame
            
            allFeatures.append(src)
            allPositions.append(combinedPos)
            allMasks.append(mask)
        
        batchSize = allFeatures[0].shape[0]
        
        # Flatten each frame: [B, C, H, W] -> [H*W, B, C]
        flatFeatures = []
        flatPositions = []
        flatMasks = []
        
        for src, pos, mask in zip(allFeatures, allPositions, allMasks):
            srcFlat = src.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
            posFlat = pos.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
            maskFlat = mask.flatten(1)  # [B, H*W]
            
            flatFeatures.append(srcFlat)
            flatPositions.append(posFlat)
            flatMasks.append(maskFlat)
        
        # Concatenate along sequence dimension: [H*W*numFrames, B, C]
        srcConcat = torch.cat(flatFeatures, dim=0)
        posConcat = torch.cat(flatPositions, dim=0)
        maskConcat = torch.cat(flatMasks, dim=1)  # [B, H*W*numFrames]
        
        # Get matching query embeddings with frame-specific components
        matchQueryEmbed = self._getFrameQueries()  # [numQueries, hiddenDim]
        matchQueryEmbed = matchQueryEmbed.unsqueeze(1).repeat(1, batchSize, 1)  # [numQueries, B, hiddenDim]
        matchTgt = torch.zeros_like(matchQueryEmbed)
        
        # ---- Label Denoising (training only) ----
        dnInfo = None
        if self.training and self.useDnDenoising and self.denoisingGenerator is not None and targets is not None:
            dnInfo = self.denoisingGenerator(targets, srcConcat.device)
        
        if dnInfo is not None:
            # Prepend denoising queries: [dnQueries + matchQueries, B, hiddenDim]
            tgt = torch.cat([dnInfo['dn_query_content'], matchTgt], dim=0)
            queryEmbed = torch.cat([dnInfo['dn_query_pos'], matchQueryEmbed], dim=0)
            attnMask = dnInfo['dn_attn_mask']  # [totalQueries, totalQueries]
        else:
            tgt = matchTgt
            queryEmbed = matchQueryEmbed
            attnMask = None
        
        # Run transformer encoder
        memory = self.transformer.encoder(
            srcConcat,
            src_key_padding_mask=maskConcat,
            pos=posConcat
        )
        
        # Run transformer decoder with attention mask
        # hs shape: [numDecoderLayers, totalQueries, B, hiddenDim]
        hs = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=attnMask,
            memory_key_padding_mask=maskConcat,
            pos=posConcat,
            query_pos=queryEmbed
        )
        
        # Transpose to [numDecoderLayers, B, totalQueries, hiddenDim]
        hs = hs.transpose(1, 2)
        
        # ---- Split DN and matching outputs ----
        if dnInfo is not None:
            dnMeta = dnInfo['dn_meta']
            dnHs, matchHs = self.denoisingGenerator.splitDnOutputs(hs, dnMeta)
        else:
            dnMeta = None
            matchHs = hs
        
        # Compute matching outputs (same as before)
        outputsClass = self.classEmbed(matchHs)  # [numLayers, B, numQueries, numClasses+1]
        outputsCoord = self.bboxEmbed(matchHs).sigmoid()  # [numLayers, B, numQueries, 4]
        outputsTracking = self.trackingHead(matchHs)  # [numLayers, B, numQueries, trackingDim]
        
        # Build output dict (using last decoder layer)
        out = {
            'pred_logits': outputsClass[-1],
            'pred_boxes': outputsCoord[-1],
            'pred_tracking': outputsTracking[-1]
        }
        
        # Add auxiliary outputs if enabled
        if self.auxLoss:
            out['aux_outputs'] = self._setAuxLoss(
                outputsClass, outputsCoord, outputsTracking
            )
        
        # ---- Add denoising outputs (training only) ----
        if dnMeta is not None:
            # Apply same detection heads to DN outputs
            dnOutputsClass = self.classEmbed(dnHs)  # [numLayers, B, dnQueries, numClasses+1]
            dnOutputsCoord = self.bboxEmbed(dnHs).sigmoid()  # [numLayers, B, dnQueries, 4]
            
            out['dn_pred_logits'] = dnOutputsClass[-1]
            out['dn_pred_boxes'] = dnOutputsCoord[-1]
            out['dn_meta'] = dnMeta
            
            # DN auxiliary outputs
            if self.auxLoss:
                out['dn_aux_outputs'] = [
                    {
                        'dn_pred_logits': dnOutputsClass[i],
                        'dn_pred_boxes': dnOutputsCoord[i],
                    }
                    for i in range(dnOutputsClass.shape[0] - 1)
                ]
        
        return out
    
    @torch.jit.unused
    def _setAuxLoss(
        self,
        outputsClass: Tensor,
        outputsCoord: Tensor,
        outputsTracking: Tensor
    ) -> List[Dict[str, Tensor]]:
        """
        Create auxiliary output dicts for intermediate decoder layers.
        
        This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values.
        """
        return [
            {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_tracking': c
            }
            for a, b, c in zip(
                outputsClass[:-1],
                outputsCoord[:-1],
                outputsTracking[:-1]
            )
        ]
    
    def getFrameOutputs(
        self,
        outputs: Dict[str, Tensor],
        frameIdx: int
    ) -> Dict[str, Tensor]:
        """
        Extract outputs for a specific frame.
        
        Args:
            outputs: Full model outputs
            frameIdx: Index of the frame (0 to numFrames-1)
        
        Returns:
            Dict with outputs for the specified frame only
        """
        startIdx = frameIdx * self.queriesPerFrame
        endIdx = startIdx + self.queriesPerFrame
        
        return {
            'pred_logits': outputs['pred_logits'][:, startIdx:endIdx],
            'pred_boxes': outputs['pred_boxes'][:, startIdx:endIdx],
            'pred_tracking': outputs['pred_tracking'][:, startIdx:endIdx]
        }


def buildVideoDETR(args) -> Tuple[VideoDETR, nn.Module, Dict[str, nn.Module]]:
    """
    Build VideoDETR model, criterion, and postprocessors from args.
    
    Args:
        args: Argument namespace with model configuration
        
    Returns:
        Tuple of (model, criterion, postprocessors)
    """
    # Import criterion builder
    from vidDetr.losses import buildVideoCriterion
    from vidDetr.losses.video_criterion import PostProcess
    
    device = torch.device(args.device)
    
    # Build backbone
    backbone = build_backbone(args)
    
    # Build transformer
    transformer = build_transformer(args)
    
    # Build model
    model = VideoDETR(
        backbone=backbone,
        transformer=transformer,
        numClasses=args.numClasses,
        numFrames=args.numFrames,
        queriesPerFrame=args.queriesPerFrame,
        auxLoss=args.auxLoss,
        trackingEmbedDim=getattr(args, 'trackingEmbedDim', 128),
        temporalType=getattr(args, 'temporalEncoding', 'learned'),
        useDnDenoising=getattr(args, 'useDnDenoising', True),
        numDnGroups=getattr(args, 'numDnGroups', 5),
        labelNoiseRatio=getattr(args, 'labelNoiseRatio', 0.5),
        boxNoiseScale=getattr(args, 'boxNoiseScale', 0.4)
    )
    
    # Build criterion
    criterion = buildVideoCriterion(args)
    criterion.to(device)
    
    # Build postprocessors
    postprocessors = {'bbox': PostProcess()}
    
    return model, criterion, postprocessors
