# Copyright (c) 2026. All Rights Reserved.
"""
Tracking Head for VideoDETR.

This module implements the tracking embedding head that produces embeddings
used for associating detections across frames via contrastive learning.
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional


class TrackingHead(nn.Module):
    """
    Tracking embedding head for object association across frames.
    
    This MLP produces embeddings that are trained with supervised contrastive
    loss to group the same object across different frames while separating
    different objects.
    
    Args:
        inputDim: Input dimension (transformer hidden_dim, default: 256)
        hiddenDim: Hidden dimension in the MLP (default: 256)
        outputDim: Output embedding dimension (default: 128)
        numLayers: Number of layers in the MLP (default: 3)
        dropout: Dropout probability (default: 0.0)
        normalize: Whether to L2-normalize output embeddings (default: True)
    """
    
    def __init__(
        self,
        inputDim: int = 256,
        hiddenDim: int = 256,
        outputDim: int = 128,
        numLayers: int = 3,
        dropout: float = 0.0,
        normalize: bool = True
    ):
        super().__init__()
        
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.numLayers = numLayers
        self.normalize = normalize
        
        # Build MLP layers
        layers = []
        dims = [inputDim] + [hiddenDim] * (numLayers - 1) + [outputDim]
        
        for i in range(numLayers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add ReLU and dropout for all but the last layer
            if i < numLayers - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        self._resetParameters()
    
    def _resetParameters(self):
        """Initialize parameters with Xavier initialization."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute tracking embeddings.
        
        Args:
            x: Input tensor from transformer decoder
               Shape: [batchSize, numQueries, inputDim] or
                      [numDecoderLayers, batchSize, numQueries, inputDim]
        
        Returns:
            Tracking embeddings of shape [batchSize, numQueries, outputDim] or
            [numDecoderLayers, batchSize, numQueries, outputDim]
        """
        # Apply MLP
        embeddings = self.mlp(x)
        
        # L2 normalize embeddings if specified
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class TrackingHeadWithMemory(nn.Module):
    """
    Enhanced tracking head with memory mechanism for temporal consistency.
    
    This version includes a memory bank that stores embeddings from previous
    frames, enabling more robust tracking by considering historical information.
    
    Note: This is an advanced variant for future experimentation.
    
    Args:
        inputDim: Input dimension (default: 256)
        hiddenDim: Hidden dimension (default: 256)
        outputDim: Output embedding dimension (default: 128)
        numLayers: Number of MLP layers (default: 3)
        memorySize: Size of memory bank per track (default: 10)
        useAttention: Whether to use attention over memory (default: True)
    """
    
    def __init__(
        self,
        inputDim: int = 256,
        hiddenDim: int = 256,
        outputDim: int = 128,
        numLayers: int = 3,
        memorySize: int = 10,
        useAttention: bool = True
    ):
        super().__init__()
        
        self.baseHead = TrackingHead(
            inputDim=inputDim,
            hiddenDim=hiddenDim,
            outputDim=outputDim,
            numLayers=numLayers,
            normalize=True
        )
        
        self.memorySize = memorySize
        self.useAttention = useAttention
        
        if useAttention:
            # Attention mechanism for aggregating memory
            self.memoryAttention = nn.MultiheadAttention(
                embed_dim=outputDim,
                num_heads=4,
                dropout=0.1
            )
        
        # Memory bank (to be managed externally during inference)
        self.register_buffer('memoryBank', None)
    
    def forward(
        self, 
        x: Tensor,
        memory: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute tracking embeddings with optional memory enhancement.
        
        Args:
            x: Input tensor [batchSize, numQueries, inputDim]
            memory: Optional memory tensor [batchSize, numQueries, memorySize, outputDim]
        
        Returns:
            Tracking embeddings [batchSize, numQueries, outputDim]
        """
        # Get base embeddings
        embeddings = self.baseHead(x)
        
        # If memory is provided and attention is enabled, enhance embeddings
        if memory is not None and self.useAttention:
            batchSize, numQueries, outputDim = embeddings.shape
            
            # Reshape for attention
            # Query: current embeddings [numQueries, batchSize, outputDim]
            q = embeddings.permute(1, 0, 2)
            
            # Key/Value: memory [memorySize, batchSize*numQueries, outputDim]
            memFlat = memory.view(batchSize * numQueries, self.memorySize, -1)
            kv = memFlat.permute(1, 0, 2)
            
            # Apply attention
            enhanced, _ = self.memoryAttention(q, kv, kv)
            
            # Reshape back
            enhanced = enhanced.permute(1, 0, 2)
            
            # Combine with original embeddings
            embeddings = F.normalize(embeddings + enhanced, p=2, dim=-1)
        
        return embeddings


def buildTrackingHead(args) -> TrackingHead:
    """
    Build tracking head from args.
    
    Args:
        args: Argument namespace with:
            - hiddenDim: Transformer hidden dimension
            - trackingEmbedDim: Output embedding dimension
            - trackingNumLayers: Number of MLP layers
            - trackingDropout: Dropout probability
            
    Returns:
        TrackingHead module
    """
    return TrackingHead(
        inputDim=args.hiddenDim,
        hiddenDim=args.hiddenDim,
        outputDim=getattr(args, 'trackingEmbedDim', 128),
        numLayers=getattr(args, 'trackingNumLayers', 3),
        dropout=getattr(args, 'trackingDropout', 0.0),
        normalize=True
    )
