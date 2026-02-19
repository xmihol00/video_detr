# Copyright (c) 2026. All Rights Reserved.
"""
Supervised Contrastive Loss for Object Tracking.

This module implements supervised contrastive learning loss that encourages
embeddings of the same object across different frames to be similar while
pushing apart embeddings of different objects.

Reference:
    "Supervised Contrastive Learning" - Khosla et al., NeurIPS 2020
    https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for cross-frame object association.
    
    This loss pulls together embeddings of the same object across frames
    (positive pairs) while pushing apart embeddings of different objects
    (negative pairs).
    
    The loss for a positive pair (i, j) where i and j are the same object
    in different frames is:
    
        L_ij = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
    
    where:
        - z_i, z_j are normalized embeddings
        - sim(a, b) = a · b (dot product for normalized vectors = cosine similarity)
        - τ is the temperature parameter
        - The sum is over all embeddings k ≠ i
    
    Args:
        temperature: Temperature parameter for scaling (default: 0.07)
        baseLossWeight: Weight for base contrastive term (default: 1.0)
        contrastAllFrames: If True, contrast across all frames; else only adjacent
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        baseLossWeight: float = 1.0,
        contrastAllFrames: bool = True
    ):
        super().__init__()
        
        self.temperature = temperature
        self.baseLossWeight = baseLossWeight
        self.contrastAllFrames = contrastAllFrames
    
    def updateTemperature(self, temperature: float) -> None:
        """Update the temperature parameter (for per-epoch scheduling)."""
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: Tensor,
        trackIds: Tensor,
        frameIndices: Tensor,
        validMask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: Tracking embeddings [N, D] where N is total number of
                       matched predictions across all frames in the batch
            trackIds: Track IDs [N] indicating which object each embedding belongs to
            frameIndices: Frame indices [N] indicating which frame each embedding is from
            validMask: Optional boolean mask [N] indicating valid embeddings
        
        Returns:
            Scalar loss value
        """
        device = embeddings.device
        
        # Handle empty input
        if embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Apply valid mask if provided
        if validMask is not None:
            embeddings = embeddings[validMask]
            trackIds = trackIds[validMask]
            frameIndices = frameIndices[validMask]
        
        numSamples = embeddings.shape[0]
        
        if numSamples < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure embeddings are normalized
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute all pairwise similarities
        # [N, N]
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create masks for positive and negative pairs
        # Positive: same track ID, different frame
        trackIdMatrix = trackIds.unsqueeze(0) == trackIds.unsqueeze(1)  # [N, N]
        frameIdMatrix = frameIndices.unsqueeze(0) == frameIndices.unsqueeze(1)  # [N, N]
        
        # Positive pairs: same track, different frame
        positiveMask = trackIdMatrix & ~frameIdMatrix
        
        # Negative pairs: different track (any frame)
        negativeMask = ~trackIdMatrix
        
        # Self-mask (exclude self-similarity)
        selfMask = torch.eye(numSamples, device=device, dtype=torch.bool)
        
        # Combine: we want to contrast against all samples except self
        # but only use positives for the numerator
        
        # For numerical stability, subtract max from similarities
        simMax, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - simMax.detach()
        
        # Compute exp(similarity)
        expSim = torch.exp(similarity)
        
        # Mask out self
        expSim = expSim * (~selfMask).float()
        
        # Denominator: sum over all samples except self
        denominator = expSim.sum(dim=1, keepdim=True)
        
        # Numerator: exp(similarity) for positive pairs
        positiveExpSim = expSim * positiveMask.float()
        
        # Count positive pairs per sample
        numPositives = positiveMask.sum(dim=1).float()
        
        # Avoid division by zero
        hasPositives = numPositives > 0
        
        if not hasPositives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log probability for positive pairs
        # -log(exp_pos / denominator) = -log(exp_pos) + log(denominator)
        # = -(similarity_pos - simMax) + log(denominator)
        
        # For each positive pair, compute the loss
        logProb = similarity - torch.log(denominator + 1e-8)
        
        # Average over positive pairs for each sample
        positiveLogProb = (logProb * positiveMask.float()).sum(dim=1)
        positiveLogProb = positiveLogProb / (numPositives + 1e-8)
        
        # Only include samples that have at least one positive pair
        loss = -positiveLogProb[hasPositives].mean()
        
        return loss * self.baseLossWeight


class HardNegativeContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss with hard negative mining.
    
    This variant focuses on hard negatives (similar embeddings from different
    objects) which are more informative for learning discriminative features.
    
    Args:
        temperature: Temperature parameter (default: 0.07)
        numHardNegatives: Number of hard negatives to use (default: 10)
        hardNegativeWeight: Weight for hard negative term (default: 1.0)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        numHardNegatives: int = 10,
        hardNegativeWeight: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        self.numHardNegatives = numHardNegatives
        self.hardNegativeWeight = hardNegativeWeight
        
        self.baseLoss = SupervisedContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        embeddings: Tensor,
        trackIds: Tensor,
        frameIndices: Tensor,
        validMask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute contrastive loss with hard negative mining.
        
        Args:
            embeddings: Tracking embeddings [N, D]
            trackIds: Track IDs [N]
            frameIndices: Frame indices [N]
            validMask: Optional boolean mask [N]
        
        Returns:
            Tuple of (total_loss, base_loss) for logging
        """
        # Base contrastive loss
        baseLoss = self.baseLoss(embeddings, trackIds, frameIndices, validMask)
        
        # Hard negative mining
        device = embeddings.device
        
        if validMask is not None:
            embeddings = embeddings[validMask]
            trackIds = trackIds[validMask]
        
        numSamples = embeddings.shape[0]
        
        if numSamples < 2:
            return baseLoss, baseLoss
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarities
        similarity = torch.matmul(embeddings, embeddings.T)
        
        # Identify negative pairs (different track IDs)
        negativeMask = trackIds.unsqueeze(0) != trackIds.unsqueeze(1)
        
        # Get hard negatives: highest similarity among negatives
        negSim = similarity.clone()
        negSim[~negativeMask] = -float('inf')
        
        # For each sample, get top-k hard negatives
        k = min(self.numHardNegatives, negativeMask.sum(dim=1).min().item())
        
        if k < 1:
            return baseLoss, baseLoss
        
        hardNegSim, _ = negSim.topk(k, dim=1)
        
        # Hard negative loss: push apart hard negatives
        # We want similarity to be low (negative after scaling)
        hardNegLoss = torch.relu(hardNegSim / self.temperature + 1.0).mean()
        
        totalLoss = baseLoss + self.hardNegativeWeight * hardNegLoss
        
        return totalLoss, baseLoss


def buildContrastiveLoss(args) -> SupervisedContrastiveLoss:
    """
    Build contrastive loss from args.
    
    Args:
        args: Argument namespace with:
            - contrastiveTemp: Temperature parameter
            - contrastiveWeight: Loss weight
            
    Returns:
        SupervisedContrastiveLoss module
    """
    return SupervisedContrastiveLoss(
        temperature=getattr(args, 'contrastiveTemp', 0.07),
        baseLossWeight=getattr(args, 'contrastiveWeight', 1.0),
        contrastAllFrames=getattr(args, 'contrastAllFrames', True)
    )
