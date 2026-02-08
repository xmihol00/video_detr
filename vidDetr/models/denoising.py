# Copyright (c) 2026. All Rights Reserved.
"""
Label Denoising for VideoDETR (DN-DETR / DINO style).

This module implements the denoising training technique from:
- DN-DETR: "DN-DETR: Accelerate DETR Training by Introducing Query Denoising"
- DINO: "DINO: DETR with Improved DeNoising Anchor Training"

The key idea:
- During training, we add extra "denoising queries" to the decoder alongside
  the regular matching queries.
- These denoising queries are constructed from ground-truth labels and boxes
  with added noise (label flips + box coordinate noise).
- Since we know which GT each denoising query came from, we compute their loss
  DIRECTLY without Hungarian matching — providing a clean gradient signal.
- An attention mask prevents information leakage between denoising and matching
  queries, and between different denoising groups.

For video: denoising queries are created per-frame, respecting the frame-based
query organization of VideoDETR.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from util import box_ops


class DenoisingGenerator(nn.Module):
    """
    Generates denoising queries from ground-truth annotations.
    
    For each frame in the video clip, this module:
    1. Takes GT labels and boxes
    2. Creates multiple denoising groups (each with a noised copy of all GTs)
    3. Applies label noise (random class flipping) and box noise (coordinate perturbation)
    4. Generates an attention mask to prevent information leakage
    
    The denoising queries are prepended to the matching queries before being
    fed to the transformer decoder.
    
    Args:
        hiddenDim: Transformer hidden dimension (d_model)
        numClasses: Number of object classes (excluding no-object)
        numDnGroups: Number of denoising groups (each group = one noised copy of all GTs)
        labelNoiseRatio: Probability of flipping a label to a random class
        boxNoiseScale: Scale of box coordinate noise (fraction of box size)
        numFrames: Number of video frames
        queriesPerFrame: Number of matching queries per frame
    """
    
    def __init__(
        self,
        hiddenDim: int = 256,
        numClasses: int = 80,
        numDnGroups: int = 5,
        labelNoiseRatio: float = 0.5,
        boxNoiseScale: float = 0.4,
        numFrames: int = 5,
        queriesPerFrame: int = 75
    ):
        super().__init__()
        
        self.hiddenDim = hiddenDim
        self.numClasses = numClasses
        self.numDnGroups = numDnGroups
        self.labelNoiseRatio = labelNoiseRatio
        self.boxNoiseScale = boxNoiseScale
        self.numFrames = numFrames
        self.queriesPerFrame = queriesPerFrame
        
        # Learnable label embedding: maps class index to query content
        # +1 for no-object class (used as indicator embedding)
        self.labelEmbed = nn.Embedding(numClasses + 1, hiddenDim)
        
        # Box-to-query MLP: maps noised box (4D) to positional embedding  
        self.boxEmbed = nn.Sequential(
            nn.Linear(4, hiddenDim),
            nn.ReLU(inplace=True),
            nn.Linear(hiddenDim, hiddenDim),
        )
        
        self._initWeights()
    
    def _initWeights(self):
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.labelEmbed.weight)
        for layer in self.boxEmbed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _getNumGtPerFrame(
        self,
        targets: List[List[Dict[str, Tensor]]]
    ) -> List[List[int]]:
        """
        Count GT objects per frame per batch.
        
        Args:
            targets: Frame-first targets: targets[frameIdx][batchIdx]
            
        Returns:
            numGt[frameIdx][batchIdx] = number of GT objects
        """
        numFrames = len(targets)
        batchSize = len(targets[0])
        
        numGt = []
        for f in range(numFrames):
            frameGt = []
            for b in range(batchSize):
                frameGt.append(len(targets[f][b]['labels']))
            numGt.append(frameGt)
        
        return numGt
    
    def forward(
        self,
        targets: List[List[Dict[str, Tensor]]],
        device: torch.device
    ) -> Optional[Dict[str, Tensor]]:
        """
        Generate denoising queries and attention mask.
        
        The denoising queries are organized as:
        [group0_frame0_gts, group0_frame1_gts, ..., group1_frame0_gts, ...]
        
        Each group contains noised copies of ALL GT objects across ALL frames,
        maintaining the per-frame structure.
        
        Args:
            targets: Frame-first targets: targets[frameIdx][batchIdx]
            device: Device for tensors
            
        Returns:
            None if no GT objects, otherwise dict with:
            - 'dn_query_content': [totalDnQueries, B, hiddenDim] content embeddings
            - 'dn_query_pos': [totalDnQueries, B, hiddenDim] positional embeddings
            - 'dn_attn_mask': [totalDnQueries + matchQueries, totalDnQueries + matchQueries]
              attention mask for the decoder self-attention
            - 'dn_meta': dict with metadata for loss computation:
                - 'dn_num_groups': number of DN groups
                - 'dn_num_queries': total number of DN queries per batch element
                - 'dn_pad_size': padded size per group (max_gt * numFrames per group)
                - 'gt_counts': numGt[frameIdx][batchIdx]
                - 'max_gt_per_frame': max GT across all frames and batch elements
                - 'known_labels': [B, totalDnQueries] GT labels for each DN query
                - 'known_boxes': [B, totalDnQueries, 4] GT boxes for each DN query
                - 'dn_valid_mask': [B, totalDnQueries] mask of valid (non-padded) queries
        """
        numFrames = len(targets)
        batchSize = len(targets[0])
        
        # Count GT per frame per batch
        numGt = self._getNumGtPerFrame(targets)
        
        # Find max GT per frame across all batches and frames
        maxGt = max(
            count
            for frameCounts in numGt
            for count in frameCounts
        )
        
        if maxGt == 0:
            # No GT objects in entire batch - skip denoising
            return None
        
        # Total DN queries per group = maxGt * numFrames (padded to same size)
        dnQueriesPerGroup = maxGt * numFrames
        totalDnQueries = dnQueriesPerGroup * self.numDnGroups
        
        # Gather all GT labels and boxes, padded to maxGt per frame
        # Shape: [B, numFrames, maxGt]
        allLabels = torch.full(
            (batchSize, numFrames, maxGt),
            self.numClasses,  # pad with no-object class
            dtype=torch.long,
            device=device
        )
        allBoxes = torch.zeros(
            (batchSize, numFrames, maxGt, 4),
            dtype=torch.float,
            device=device
        )
        validMask2d = torch.zeros(
            (batchSize, numFrames, maxGt),
            dtype=torch.bool,
            device=device
        )
        
        for f in range(numFrames):
            for b in range(batchSize):
                ngt = numGt[f][b]
                if ngt > 0:
                    allLabels[b, f, :ngt] = targets[f][b]['labels'][:ngt].to(device)
                    allBoxes[b, f, :ngt] = targets[f][b]['boxes'][:ngt].to(device)
                    validMask2d[b, f, :ngt] = True
        
        # Repeat for each DN group: [B, numDnGroups, numFrames, maxGt]
        labels = allLabels.unsqueeze(1).repeat(1, self.numDnGroups, 1, 1)
        boxes = allBoxes.unsqueeze(1).repeat(1, self.numDnGroups, 1, 1, 1)
        validMask = validMask2d.unsqueeze(1).repeat(1, self.numDnGroups, 1, 1)
        
        # ---- Apply label noise ----
        # Randomly flip labels to a different class
        if self.labelNoiseRatio > 0 and self.training:
            labelNoiseMask = (
                torch.rand_like(labels.float()) < self.labelNoiseRatio
            ) & validMask
            
            # Random class indices for flipped labels
            randomLabels = torch.randint_like(labels, 0, self.numClasses)
            labels = torch.where(labelNoiseMask, randomLabels, labels)
        
        # ---- Apply box noise ----
        # Add noise proportional to box size
        if self.boxNoiseScale > 0 and self.training:
            # boxes are in cxcywh format
            # Noise scale relative to box dimensions
            boxWh = boxes[..., 2:].clamp(min=1e-4)  # [B, G, F, maxGt, 2]
            
            # Random noise in [-1, 1] * scale * box_size
            noise = (torch.rand_like(boxes) * 2 - 1) * self.boxNoiseScale
            
            # Scale xy noise by wh, scale wh noise by wh
            noisedBoxes = boxes.clone()
            noisedBoxes[..., :2] = boxes[..., :2] + noise[..., :2] * boxWh
            noisedBoxes[..., 2:] = boxes[..., 2:] * (1 + noise[..., 2:])
            
            # Clamp to valid range [0, 1]
            noisedBoxes = noisedBoxes.clamp(0, 1)
            
            # Only apply noise to valid entries
            boxes = torch.where(
                validMask.unsqueeze(-1).expand_as(boxes),
                noisedBoxes,
                boxes
            )
        
        # ---- Store known (clean) GT for loss computation ----
        # These are the ORIGINAL labels and boxes before noise
        knownLabels = allLabels.unsqueeze(1).repeat(
            1, self.numDnGroups, 1, 1
        )  # [B, G, F, maxGt]
        knownBoxes = allBoxes.unsqueeze(1).repeat(
            1, self.numDnGroups, 1, 1, 1
        )  # [B, G, F, maxGt, 4]
        
        # ---- Generate content and positional embeddings ----
        # Reshape for embedding lookup: [B, G*F*maxGt]
        flatLabels = labels.reshape(batchSize, -1)  # [B, totalDnQueries]
        flatBoxes = boxes.reshape(batchSize, -1, 4)  # [B, totalDnQueries, 4]
        flatValidMask = validMask.reshape(batchSize, -1)  # [B, totalDnQueries]
        flatKnownLabels = knownLabels.reshape(batchSize, -1)
        flatKnownBoxes = knownBoxes.reshape(batchSize, -1, 4)
        
        # Content embedding from noised labels: [B, totalDnQueries, hiddenDim]
        contentEmbed = self.labelEmbed(flatLabels)
        
        # Positional embedding from noised boxes: [B, totalDnQueries, hiddenDim]
        posEmbed = self.boxEmbed(flatBoxes)
        
        # Transpose to [totalDnQueries, B, hiddenDim] for decoder
        contentEmbed = contentEmbed.transpose(0, 1)
        posEmbed = posEmbed.transpose(0, 1)
        
        # ---- Build attention mask ----
        totalQueries = totalDnQueries + self.numFrames * self.queriesPerFrame
        attnMask = torch.zeros(
            (totalQueries, totalQueries),
            dtype=torch.bool,
            device=device
        )
        
        # 1. DN queries cannot attend to matching queries
        attnMask[:totalDnQueries, totalDnQueries:] = True
        
        # 2. Matching queries cannot attend to DN queries
        attnMask[totalDnQueries:, :totalDnQueries] = True
        
        # 3. DN queries from different groups cannot attend to each other
        for i in range(self.numDnGroups):
            for j in range(self.numDnGroups):
                if i != j:
                    startI = i * dnQueriesPerGroup
                    endI = (i + 1) * dnQueriesPerGroup
                    startJ = j * dnQueriesPerGroup
                    endJ = (j + 1) * dnQueriesPerGroup
                    attnMask[startI:endI, startJ:endJ] = True
        
        # Build metadata for loss computation
        dnMeta = {
            'dn_num_groups': self.numDnGroups,
            'dn_num_queries': totalDnQueries,
            'dn_queries_per_group': dnQueriesPerGroup,
            'max_gt_per_frame': maxGt,
            'gt_counts': numGt,  # [numFrames][batchSize]
            'known_labels': flatKnownLabels,  # [B, totalDnQueries]
            'known_boxes': flatKnownBoxes,  # [B, totalDnQueries, 4]
            'dn_valid_mask': flatValidMask,  # [B, totalDnQueries]
        }
        
        return {
            'dn_query_content': contentEmbed,  # [totalDnQueries, B, hiddenDim]
            'dn_query_pos': posEmbed,  # [totalDnQueries, B, hiddenDim]
            'dn_attn_mask': attnMask,  # [totalQueries, totalQueries]
            'dn_meta': dnMeta,
        }
    
    def splitDnOutputs(
        self,
        hs: Tensor,
        dnMeta: Dict
    ) -> Tuple[Tensor, Tensor]:
        """
        Split decoder outputs into denoising and matching parts.
        
        Args:
            hs: Decoder output [numLayers, B, totalQueries, hiddenDim]
            dnMeta: Metadata from forward()
            
        Returns:
            (dnHs, matchHs):
            - dnHs: [numLayers, B, totalDnQueries, hiddenDim]
            - matchHs: [numLayers, B, numMatchQueries, hiddenDim]
        """
        totalDnQueries = dnMeta['dn_num_queries']
        
        dnHs = hs[:, :, :totalDnQueries, :]
        matchHs = hs[:, :, totalDnQueries:, :]
        
        return dnHs, matchHs
