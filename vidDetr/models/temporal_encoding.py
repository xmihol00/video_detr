# Copyright (c) 2026. All Rights Reserved.
"""
Temporal Position Encoding for VideoDETR.

This module implements temporal positional encodings that are added to the
spatial positional encodings from the original DETR. This allows the model
to distinguish between frames in a video sequence.

Two types of temporal encoding are supported:
1. Sinusoidal (sine): Fixed encoding based on frame index
2. Learned: Learnable embedding for each frame position
"""

import math
import torch
from torch import nn, Tensor
from typing import Optional


class TemporalPositionEncodingSine(nn.Module):
    """
    Sinusoidal temporal position encoding.
    
    Generates fixed sinusoidal embeddings based on frame index, similar to
    the positional encoding in the original Transformer paper.
    
    Args:
        numPosFeats: Number of features in the encoding (hidden_dim // 2)
        temperature: Temperature for the sinusoidal encoding (default: 10000)
        maxFrames: Maximum number of frames supported (default: 100)
        normalize: Whether to normalize frame indices (default: True)
        scale: Scale factor for normalized indices (default: 2*pi)
    """
    
    def __init__(
        self,
        numPosFeats: int = 128,
        temperature: float = 10000.0,
        maxFrames: int = 100,
        normalize: bool = True,
        scale: Optional[float] = None
    ):
        super().__init__()
        
        self.numPosFeats = numPosFeats
        self.temperature = temperature
        self.maxFrames = maxFrames
        self.normalize = normalize
        
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        # Precompute dimension factors
        self.register_buffer(
            'dimT',
            self.temperature ** (2 * (torch.arange(numPosFeats) // 2) / numPosFeats)
        )
    
    def forward(
        self, 
        frameIndices: Tensor,
        numFrames: int
    ) -> Tensor:
        """
        Generate temporal positional encoding.
        
        Args:
            frameIndices: Tensor of frame indices [batchSize] or [batchSize, numFrames]
            numFrames: Total number of frames in the sequence
            
        Returns:
            Temporal encoding of shape [numFrames, numPosFeats * 2] or 
            [batchSize, numFrames, numPosFeats * 2]
        """
        device = frameIndices.device if isinstance(frameIndices, Tensor) else 'cpu'
        
        # Create frame index tensor if not provided
        if frameIndices.dim() == 1:
            # frameIndices is [numFrames]
            t = frameIndices.float()
        else:
            # frameIndices is [batchSize, numFrames]
            t = frameIndices.float()
        
        # Normalize
        if self.normalize:
            t = t / (numFrames - 1 + 1e-6) * self.scale
        
        # Expand for broadcasting
        if t.dim() == 1:
            # [numFrames] -> [numFrames, 1]
            t = t.unsqueeze(-1)
        else:
            # [batchSize, numFrames] -> [batchSize, numFrames, 1]
            t = t.unsqueeze(-1)
        
        # Compute sinusoidal encoding
        dimT = self.dimT.to(device)
        posT = t / dimT
        
        # Stack sin and cos to get [*, numPosFeats * 2]
        posEnc = torch.cat([posT.sin(), posT.cos()], dim=-1)
        
        return posEnc


class TemporalPositionEncodingLearned(nn.Module):
    """
    Learned temporal position encoding.
    
    Uses learnable embeddings for each frame position, allowing the model
    to learn task-specific temporal representations.
    
    Args:
        numPosFeats: Number of features in the encoding (hidden_dim)
        maxFrames: Maximum number of frames supported (default: 100)
    """
    
    def __init__(
        self,
        numPosFeats: int = 256,
        maxFrames: int = 100
    ):
        super().__init__()
        
        self.numPosFeats = numPosFeats
        self.maxFrames = maxFrames
        
        # Learnable embedding for each frame position
        self.frameEmbed = nn.Embedding(maxFrames, numPosFeats)
        
        self._resetParameters()
    
    def _resetParameters(self):
        """Initialize parameters with uniform distribution."""
        nn.init.uniform_(self.frameEmbed.weight)
    
    def forward(
        self,
        frameIndices: Tensor,
        numFrames: int
    ) -> Tensor:
        """
        Generate temporal positional encoding.
        
        Args:
            frameIndices: Tensor of frame indices [numFrames] or [batchSize, numFrames]
            numFrames: Total number of frames (unused, for API consistency)
            
        Returns:
            Temporal encoding of shape [numFrames, numPosFeats] or
            [batchSize, numFrames, numPosFeats]
        """
        return self.frameEmbed(frameIndices.long())


class TemporalPositionEncoding(nn.Module):
    """
    Combined temporal and spatial position encoding for VideoDETR.
    
    This module wraps the temporal encoding and provides utilities for
    combining it with spatial position encoding from the backbone.
    
    Args:
        hiddenDim: Hidden dimension of the transformer (default: 256)
        temporalType: Type of temporal encoding ('sine' or 'learned')
        maxFrames: Maximum number of frames (default: 100)
        temperature: Temperature for sinusoidal encoding (default: 10000)
    """
    
    def __init__(
        self,
        hiddenDim: int = 256,
        temporalType: str = 'learned',
        maxFrames: int = 100,
        temperature: float = 10000.0
    ):
        super().__init__()
        
        self.hiddenDim = hiddenDim
        self.temporalType = temporalType
        self.maxFrames = maxFrames
        
        if temporalType == 'sine':
            # For sine, we use half the features
            self.temporalEnc = TemporalPositionEncodingSine(
                numPosFeats=hiddenDim // 2,
                temperature=temperature,
                maxFrames=maxFrames,
                normalize=True
            )
        elif temporalType == 'learned':
            self.temporalEnc = TemporalPositionEncodingLearned(
                numPosFeats=hiddenDim,
                maxFrames=maxFrames
            )
        else:
            raise ValueError(f"Unknown temporal encoding type: {temporalType}")
        
        # Optional projection layer to match dimensions if needed
        if temporalType == 'sine':
            # Sine encoding produces hiddenDim features, so no projection needed
            self.projection = None
        else:
            self.projection = None
    
    def forward(
        self,
        spatialPos: Tensor,
        numFrames: int,
        frameIndices: Optional[Tensor] = None
    ) -> Tensor:
        """
        Combine spatial and temporal position encoding.
        
        Args:
            spatialPos: Spatial position encoding from backbone [B, C, H, W]
            numFrames: Number of frames in the sequence
            frameIndices: Optional frame indices [numFrames] (default: 0, 1, ..., N-1)
            
        Returns:
            Combined position encoding [B*numFrames, C, H, W] where temporal
            encoding is added to each spatial position
        """
        device = spatialPos.device
        batchSize, channels, height, width = spatialPos.shape
        
        # Generate frame indices if not provided
        if frameIndices is None:
            frameIndices = torch.arange(numFrames, device=device)
        
        # Get temporal encoding [numFrames, hiddenDim]
        temporalEnc = self.temporalEnc(frameIndices, numFrames)
        
        # Project if needed
        if self.projection is not None:
            temporalEnc = self.projection(temporalEnc)
        
        # Ensure temporal encoding has shape [numFrames, hiddenDim]
        if temporalEnc.dim() == 3:
            # [batchSize, numFrames, hiddenDim] -> [numFrames, hiddenDim]
            temporalEnc = temporalEnc[0]
        
        # Expand temporal encoding to match spatial dimensions
        # [numFrames, hiddenDim] -> [numFrames, hiddenDim, 1, 1]
        temporalEnc = temporalEnc.view(numFrames, -1, 1, 1)
        
        # Expand to [numFrames, hiddenDim, H, W]
        temporalEnc = temporalEnc.expand(-1, -1, height, width)
        
        # The spatial pos is [B, C, H, W] but we need [B*numFrames, C, H, W]
        # Assuming spatialPos is already repeated for all frames or will be
        # combined by the caller, we return temporal encoding for combination
        
        return temporalEnc
    
    def getTemporalEncoding(
        self,
        numFrames: int,
        device: torch.device
    ) -> Tensor:
        """
        Get pure temporal encoding without spatial combination.
        
        Args:
            numFrames: Number of frames
            device: Device for the tensor
            
        Returns:
            Temporal encoding [numFrames, hiddenDim]
        """
        frameIndices = torch.arange(numFrames, device=device)
        return self.temporalEnc(frameIndices, numFrames)


def buildTemporalEncoding(args) -> TemporalPositionEncoding:
    """
    Build temporal position encoding from args.
    
    Args:
        args: Argument namespace with:
            - hiddenDim: Hidden dimension
            - temporalEncoding: Type ('sine' or 'learned')
            - maxFrames: Maximum number of frames
            
    Returns:
        TemporalPositionEncoding module
    """
    return TemporalPositionEncoding(
        hiddenDim=args.hiddenDim,
        temporalType=getattr(args, 'temporalEncoding', 'learned'),
        maxFrames=getattr(args, 'maxFrames', 100),
        temperature=getattr(args, 'temporalTemperature', 10000.0)
    )
