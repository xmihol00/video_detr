# Copyright (c) 2026. All Rights Reserved.
"""
Video Criterion for VideoDETR.

This module extends the SetCriterion from DETR to handle video sequences
with tracking supervision. It computes:
1. Per-frame detection losses (classification, box regression, GIoU)
2. Tracking loss using supervised contrastive learning

The Hungarian matching is performed per-frame, and tracking correspondences
are established using the trackIds from the dataset.
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

from util import box_ops
from util.misc import (
    accuracy, 
    get_world_size, 
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list
)

from .contrastive_loss import SupervisedContrastiveLoss


def sigmoidFocalLoss(
    inputs: Tensor, 
    targets: Tensor, 
    numBoxes: float,
    alpha: float = 0.25, 
    gamma: float = 2.0
) -> Tensor:
    """
    Sigmoid focal loss for classification.
    
    Focal loss down-weights well-classified examples and focuses on hard ones.
    This is crucial for training with extreme class imbalance (many no-object queries).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        inputs: [B, numQueries, numClasses] raw logits (no sigmoid applied)
        targets: [B, numQueries, numClasses] one-hot targets
        numBoxes: normalization factor
        alpha: balancing factor (default: 0.25)
        gamma: focusing parameter (default: 2.0)
    
    Returns:
        Scalar focal loss
    """
    prob = inputs.sigmoid()
    ceTarget = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pT = prob * targets + (1 - prob) * (1 - targets)
    focalWeight = (1 - pT) ** gamma
    
    # Alpha weighting: alpha for positive, (1-alpha) for negative
    alphaT = alpha * targets + (1 - alpha) * (1 - targets)
    
    loss = alphaT * focalWeight * ceTarget
    return loss.mean(1).sum() / numBoxes


class VideoHungarianMatcher(nn.Module):
    """
    Hungarian matcher for video sequences.
    
    Performs per-frame Hungarian matching between predictions and targets,
    organizing queries by frame.
    
    Args:
        costClass: Weight for classification cost
        costBbox: Weight for L1 box cost
        costGiou: Weight for GIoU cost
        numFrames: Number of frames per clip
        queriesPerFrame: Number of queries per frame
    """
    
    def __init__(
        self,
        costClass: float = 1.0,
        costBbox: float = 5.0,
        costGiou: float = 2.0,
        numFrames: int = 5,
        queriesPerFrame: int = 75
    ):
        super().__init__()
        
        self.costClass = costClass
        self.costBbox = costBbox
        self.costGiou = costGiou
        self.numFrames = numFrames
        self.queriesPerFrame = queriesPerFrame
        
        assert costClass != 0 or costBbox != 0 or costGiou != 0, \
            "All costs cannot be 0"
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict[str, Tensor]]]
    ) -> List[List[Tuple[Tensor, Tensor]]]:
        """
        Perform Hungarian matching for each frame in the batch.
        
        Args:
            outputs: Model outputs with:
                - pred_logits: [B, numQueries, numClasses+1]
                - pred_boxes: [B, numQueries, 4]
            targets: List of B lists of numFrames target dicts
        
        Returns:
            List of B lists of numFrames tuples (predIndices, targetIndices)
        """
        from scipy.optimize import linear_sum_assignment
        
        batchSize = outputs['pred_logits'].shape[0]
        allIndices = []
        
        for batchIdx in range(batchSize):
            frameIndices = []
            
            for frameIdx in range(self.numFrames):
                # Get predictions for this frame
                startQuery = frameIdx * self.queriesPerFrame
                endQuery = startQuery + self.queriesPerFrame
                
                predLogits = outputs['pred_logits'][batchIdx, startQuery:endQuery]
                predBoxes = outputs['pred_boxes'][batchIdx, startQuery:endQuery]
                
                # Get targets for this frame
                target = targets[frameIdx][batchIdx]
                
                # Handle empty targets
                if len(target['labels']) == 0:
                    frameIndices.append((
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64)
                    ))
                    continue
                
                tgtLabels = target['labels']
                tgtBoxes = target['boxes']
                
                # Compute classification cost
                outProb = predLogits.softmax(-1)
                costClassMat = -outProb[:, tgtLabels]
                
                # Compute L1 box cost
                costBboxMat = torch.cdist(predBoxes, tgtBoxes, p=1)
                
                # Compute GIoU cost
                costGiouMat = -box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(predBoxes),
                    box_ops.box_cxcywh_to_xyxy(tgtBoxes)
                )
                
                # Combine costs
                costMatrix = (
                    self.costClass * costClassMat +
                    self.costBbox * costBboxMat +
                    self.costGiou * costGiouMat
                )
                
                # Run Hungarian algorithm
                costMatrix = costMatrix.cpu().numpy()
                predIdx, tgtIdx = linear_sum_assignment(costMatrix)
                
                frameIndices.append((
                    torch.as_tensor(predIdx, dtype=torch.int64),
                    torch.as_tensor(tgtIdx, dtype=torch.int64)
                ))
            
            allIndices.append(frameIndices)
        
        return allIndices


class VideoCriterion(nn.Module):
    """
    Video criterion for VideoDETR training.
    
    Computes losses for:
    1. Classification (cross-entropy or focal loss)
    2. Bounding box regression (L1 + GIoU)
    3. Object tracking (supervised contrastive)
    
    Args:
        numClasses: Number of object classes
        matcher: Hungarian matcher module
        weightDict: Dict of loss weights
        eosCoef: Weight for no-object class (used with cross-entropy)
        losses: List of loss names to compute
        numFrames: Number of frames per clip
        queriesPerFrame: Number of queries per frame
        contrastiveTemp: Temperature for contrastive loss
        useFocalLoss: Use sigmoid focal loss instead of cross-entropy (default: True)
        focalAlpha: Focal loss alpha parameter (default: 0.25)
        focalGamma: Focal loss gamma parameter (default: 2.0)
    """
    
    def __init__(
        self,
        numClasses: int,
        matcher: VideoHungarianMatcher,
        weightDict: Dict[str, float],
        eosCoef: float = 0.1,
        losses: List[str] = None,
        numFrames: int = 5,
        queriesPerFrame: int = 75,
        contrastiveTemp: float = 0.07,
        useFocalLoss: bool = True,
        focalAlpha: float = 0.25,
        focalGamma: float = 2.0
    ):
        super().__init__()
        
        self.numClasses = numClasses
        self.matcher = matcher
        self.weightDict = weightDict
        self.eosCoef = eosCoef
        self.losses = losses if losses is not None else ['labels', 'boxes', 'cardinality', 'tracking']
        self.numFrames = numFrames
        self.queriesPerFrame = queriesPerFrame
        self.useFocalLoss = useFocalLoss
        self.focalAlpha = focalAlpha
        self.focalGamma = focalGamma
        
        # Duplicate suppression: penalize pairs of same-frame predictions
        # that have high IoU with each other (both confidently predicting an object)
        self.dupIouThreshold = 0.7  # IoU above which pairs are penalised
        
        # Class weights for cross-entropy (downweight no-object) - used when not using focal loss
        emptyWeight = torch.ones(numClasses + 1)
        emptyWeight[-1] = eosCoef
        self.register_buffer('emptyWeight', emptyWeight)
        
        # Contrastive loss for tracking
        self.contrastiveLoss = SupervisedContrastiveLoss(
            temperature=contrastiveTemp
        )
        
        if useFocalLoss:
            print(f"[VideoCriterion] Using focal loss (alpha={focalAlpha}, gamma={focalGamma})")
        else:
            print(f"[VideoCriterion] Using cross-entropy (eos_coef={eosCoef})")
    
    def lossLabels(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]],
        indices: List[List[Tuple]],
        numBoxes: int,
        log: bool = True
    ) -> Dict[str, Tensor]:
        """
        Classification loss (focal loss or cross-entropy).
        
        Computed per-frame and averaged.
        """
        assert 'pred_logits' in outputs
        srcLogits = outputs['pred_logits']
        
        device = srcLogits.device
        batchSize = srcLogits.shape[0]
        
        # Build target classes tensor [B, numQueries]
        targetClasses = torch.full(
            srcLogits.shape[:2], 
            self.numClasses,  # no-object class
            dtype=torch.int64, 
            device=device
        )
        # Fill in matched targets
        # Note: targets are organized as [frameIdx][batchIdx] (frame-first)
        for batchIdx in range(batchSize):
            for frameIdx in range(self.numFrames):
                srcIdx, tgtIdx = indices[batchIdx][frameIdx]
                
                if len(srcIdx) == 0:
                    continue
                
                # Convert frame-local indices to global query indices
                globalSrcIdx = srcIdx + frameIdx * self.queriesPerFrame
                
                # Get target labels (targets are frame-first: targets[frameIdx][batchIdx])
                tgtLabels = targets[frameIdx][batchIdx]['labels'][tgtIdx]
                targetClasses[batchIdx, globalSrcIdx] = tgtLabels.to(device)
        
        if self.useFocalLoss:
            # Focal loss: operates on one-hot targets with sigmoid
            # srcLogits shape: [B, numQueries, numClasses+1]
            # Build one-hot targets of same shape
            targetClassesOnehot = torch.zeros_like(srcLogits)
            targetClassesOnehot.scatter_(2, targetClasses.unsqueeze(-1), 1)
            
            lossCe = sigmoidFocalLoss(
                srcLogits, targetClassesOnehot, numBoxes,
                alpha=self.focalAlpha, gamma=self.focalGamma
            )
        else:
            # Standard cross-entropy with class weighting
            lossCe = F.cross_entropy(
                srcLogits.transpose(1, 2),
                targetClasses,
                self.emptyWeight.to(device)
            )
        
        losses = {'loss_ce': lossCe}
        
        if log:
            # Compute classification error for logging
            allSrcIdx = []
            allTgtLabels = []
            
            for batchIdx in range(batchSize):
                for frameIdx in range(self.numFrames):
                    srcIdx, tgtIdx = indices[batchIdx][frameIdx]
                    if len(srcIdx) > 0:
                        globalSrcIdx = srcIdx + frameIdx * self.queriesPerFrame
                        allSrcIdx.append((
                            torch.full_like(globalSrcIdx, batchIdx),
                            globalSrcIdx
                        ))
                        allTgtLabels.append(
                            targets[frameIdx][batchIdx]['labels'][tgtIdx]
                        )
            
            if allSrcIdx:
                batchIdxCat = torch.cat([x[0] for x in allSrcIdx])
                srcIdxCat = torch.cat([x[1] for x in allSrcIdx])
                tgtLabelsCat = torch.cat(allTgtLabels).to(device)
                
                # For focal loss, class_error is computed on matched predictions only
                if self.useFocalLoss:
                    # Get the logits of matched queries and compute accuracy
                    predLogitsMatched = srcLogits[batchIdxCat, srcIdxCat]
                    # With focal loss, predict = argmax of sigmoid
                    predClasses = predLogitsMatched.sigmoid().argmax(-1)
                    correctPreds = (predClasses == tgtLabelsCat).float().mean() * 100
                    losses['class_error'] = 100 - correctPreds
                else:
                    predLogitsMatched = srcLogits[batchIdxCat, srcIdxCat]
                    losses['class_error'] = 100 - accuracy(predLogitsMatched, tgtLabelsCat)[0]
            else:
                losses['class_error'] = torch.tensor(0.0, device=device)
        
        return losses
    
    def lossBoxes(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]],
        indices: List[List[Tuple]],
        numBoxes: int
    ) -> Dict[str, Tensor]:
        """
        Bounding box losses (L1 + GIoU).
        """
        assert 'pred_boxes' in outputs
        device = outputs['pred_boxes'].device
        
        # Collect all matched boxes
        srcBoxes = []
        tgtBoxes = []
        
        batchSize = outputs['pred_boxes'].shape[0]
        
        # Note: targets are organized as [frameIdx][batchIdx] (frame-first)
        for batchIdx in range(batchSize):
            for frameIdx in range(self.numFrames):
                srcIdx, tgtIdx = indices[batchIdx][frameIdx]
                
                if len(srcIdx) == 0:
                    continue
                
                # Convert to global query indices
                globalSrcIdx = srcIdx + frameIdx * self.queriesPerFrame
                
                srcBoxes.append(outputs['pred_boxes'][batchIdx, globalSrcIdx])
                tgtBoxes.append(targets[frameIdx][batchIdx]['boxes'][tgtIdx].to(device))
        
        if not srcBoxes:
            return {
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True)
            }
        
        srcBoxes = torch.cat(srcBoxes, dim=0)
        tgtBoxes = torch.cat(tgtBoxes, dim=0)
        
        # L1 loss
        lossBbox = F.l1_loss(srcBoxes, tgtBoxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = lossBbox.sum() / numBoxes
        
        # GIoU loss
        lossGiou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(srcBoxes),
            box_ops.box_cxcywh_to_xyxy(tgtBoxes)
        ))
        losses['loss_giou'] = lossGiou.sum() / numBoxes
        
        return losses
    
    def lossCardinality(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]],
        indices: List[List[Tuple]],
        numBoxes: int
    ) -> Dict[str, Tensor]:
        """
        Cardinality error (for logging, not trained).
        """
        predLogits = outputs['pred_logits']
        device = predLogits.device
        batchSize = predLogits.shape[0]
        numFrames = len(targets)  # targets is [numFrames][batchSize]
        
        # Count target boxes per batch
        # Note: targets are organized as [frameIdx][batchIdx] (frame-first)
        tgtLengths = []
        for batchIdx in range(batchSize):
            totalBoxes = sum(len(targets[frameIdx][batchIdx]['labels']) for frameIdx in range(numFrames))
            tgtLengths.append(totalBoxes)
        tgtLengths = torch.as_tensor(tgtLengths, device=device)
        
        # Count predicted non-empty boxes
        cardPred = (predLogits.argmax(-1) != predLogits.shape[-1] - 1).sum(1)
        cardErr = F.l1_loss(cardPred.float(), tgtLengths.float())
        
        return {'cardinality_error': cardErr}
    
    def lossTracking(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]],
        indices: List[List[Tuple]],
        numBoxes: int
    ) -> Dict[str, Tensor]:
        """
        Tracking loss using supervised contrastive learning.
        
        Groups embeddings by track ID and computes contrastive loss to
        pull together embeddings of the same object across frames.
        """
        assert 'pred_tracking' in outputs
        device = outputs['pred_tracking'].device
        
        # Collect matched embeddings with their track IDs and frame indices
        allEmbeddings = []
        allTrackIds = []
        allFrameIndices = []
        
        batchSize = outputs['pred_tracking'].shape[0]
        
        # We need unique track IDs across batches
        trackIdOffset = 0
        
        # Note: targets are organized as [frameIdx][batchIdx] (frame-first)
        for batchIdx in range(batchSize):
            maxTrackId = 0
            
            for frameIdx in range(self.numFrames):
                srcIdx, tgtIdx = indices[batchIdx][frameIdx]
                
                if len(srcIdx) == 0:
                    continue
                
                # Get embeddings
                globalSrcIdx = srcIdx + frameIdx * self.queriesPerFrame
                embeddings = outputs['pred_tracking'][batchIdx, globalSrcIdx]
                
                # Get track IDs (from target annotations, frame-first indexing)
                trackIds = targets[frameIdx][batchIdx]['trackIds'][tgtIdx]
                trackIds = trackIds.to(device) + trackIdOffset
                
                # Frame indices
                frameIndices = torch.full_like(trackIds, frameIdx)
                
                allEmbeddings.append(embeddings)
                allTrackIds.append(trackIds)
                allFrameIndices.append(frameIndices)
                
                maxTrackId = max(maxTrackId, trackIds.max().item() + 1 if len(trackIds) > 0 else 0)
            
            # Update offset for next batch
            trackIdOffset = maxTrackId + 1000  # Large gap between batches
        
        if not allEmbeddings:
            return {'loss_tracking': torch.tensor(0.0, device=device, requires_grad=True)}
        
        # Concatenate all
        embeddings = torch.cat(allEmbeddings, dim=0)
        trackIds = torch.cat(allTrackIds, dim=0)
        frameIndices = torch.cat(allFrameIndices, dim=0)
        
        # Compute contrastive loss
        lossTracking = self.contrastiveLoss(embeddings, trackIds, frameIndices)
        
        return {'loss_tracking': lossTracking}
    
    def _getSrcPermutationIdx(
        self,
        indices: List[List[Tuple]]
    ) -> Tuple[Tensor, Tensor]:
        """Get batch and query indices for all matched predictions."""
        batchIdx = []
        srcIdx = []
        
        for b, frameIndices in enumerate(indices):
            for f, (src, _) in enumerate(frameIndices):
                if len(src) > 0:
                    batchIdx.append(torch.full_like(src, b))
                    srcIdx.append(src + f * self.queriesPerFrame)
        
        if batchIdx:
            return torch.cat(batchIdx), torch.cat(srcIdx)
        return torch.tensor([]), torch.tensor([])
    
    def getLoss(
        self,
        loss: str,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]],
        indices: List[List[Tuple]],
        numBoxes: int,
        **kwargs
    ) -> Dict[str, Tensor]:
        """Get a specific loss by name."""
        lossMap = {
            'labels': self.lossLabels,
            'cardinality': self.lossCardinality,
            'boxes': self.lossBoxes,
            'tracking': self.lossTracking,
            'duplicates': self.lossDuplicates
        }
        
        assert loss in lossMap, f"Unknown loss: {loss}"
        return lossMap[loss](outputs, targets, indices, numBoxes, **kwargs)
    
    def lossDuplicates(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]],
        indices: List[List[Tuple]],
        numBoxes: int,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Duplicate suppression loss.
        
        Penalises pairs of predictions within the same frame that both have
        high object confidence AND high IoU with each other.  This directly
        discourages the model from placing multiple boxes on a single object.
        
        The loss is:  mean( IoU(b_i, b_j) * min(conf_i, conf_j) )
        for all same-frame pairs where IoU > threshold.
        """
        assert 'pred_boxes' in outputs and 'pred_logits' in outputs
        device = outputs['pred_boxes'].device
        batchSize = outputs['pred_boxes'].shape[0]
        
        # Get objectness scores (probability of NOT being no-object)
        if self.useFocalLoss:
            # sigmoid-based: max prob across real classes
            objScores = outputs['pred_logits'].sigmoid().max(dim=-1).values
        else:
            # softmax-based: 1 - prob(no-object)
            probs = outputs['pred_logits'].softmax(dim=-1)
            objScores = 1.0 - probs[..., -1]  # [B, numQueries]
        
        totalDupLoss = torch.tensor(0.0, device=device, requires_grad=True)
        numPairs = 0
        
        for batchIdx in range(batchSize):
            for frameIdx in range(self.numFrames):
                startQ = frameIdx * self.queriesPerFrame
                endQ = startQ + self.queriesPerFrame
                
                frameBoxes = outputs['pred_boxes'][batchIdx, startQ:endQ]  # [Q, 4]
                frameScores = objScores[batchIdx, startQ:endQ]  # [Q]
                
                # Only consider predictions with moderate confidence
                confMask = frameScores > 0.3
                if confMask.sum() < 2:
                    continue
                
                confBoxes = frameBoxes[confMask]  # [M, 4]
                confScores = frameScores[confMask]  # [M]
                
                # Compute pairwise IoU
                ious = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(confBoxes),
                    box_ops.box_cxcywh_to_xyxy(confBoxes)
                )[0]  # [M, M]
                
                # Mask out diagonal and lower triangle
                M = confBoxes.shape[0]
                upperMask = torch.triu(torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1)
                highIouMask = (ious > self.dupIouThreshold) & upperMask
                
                if not highIouMask.any():
                    continue
                
                # For each high-IoU pair, penalty = IoU * min(score_i, score_j)
                pairIous = ious[highIouMask]
                rows, cols = torch.where(highIouMask)
                pairMinScores = torch.min(confScores[rows], confScores[cols])
                
                pairLoss = (pairIous * pairMinScores).sum()
                totalDupLoss = totalDupLoss + pairLoss
                numPairs += highIouMask.sum().item()
        
        if numPairs > 0:
            totalDupLoss = totalDupLoss / max(numPairs, 1)
        
        return {'loss_duplicates': totalDupLoss}
    
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[List[Dict]]
    ) -> Dict[str, Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: List of numFrames lists of B target dicts (frame-first: targets[frameIdx][batchIdx])
        
        Returns:
            Dict of all losses
        """
        # Filter out auxiliary outputs and denoising outputs for matching
        outputsWithoutAux = {
            k: v for k, v in outputs.items() 
            if k not in ('aux_outputs', 'dn_pred_logits', 'dn_pred_boxes', 
                         'dn_meta', 'dn_aux_outputs')
        }
        
        # Perform Hungarian matching
        indices = self.matcher(outputsWithoutAux, targets)
        
        # Count total target boxes for normalization
        # Note: targets are organized as [frameIdx][batchIdx] (frame-first)
        numBoxes = sum(
            len(t['labels'])
            for frameTargets in targets
            for t in frameTargets
        )
        numBoxes = torch.as_tensor(
            [numBoxes], 
            dtype=torch.float, 
            device=next(iter(outputs.values())).device
        )
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(numBoxes)
        numBoxes = torch.clamp(numBoxes / get_world_size(), min=1).item()
        
        # Compute all matching losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels':
                kwargs['log'] = True
            losses.update(self.getLoss(loss, outputs, targets, indices, numBoxes, **kwargs))
        
        # Compute auxiliary losses
        if 'aux_outputs' in outputs:
            for i, auxOutputs in enumerate(outputs['aux_outputs']):
                auxIndices = self.matcher(auxOutputs, targets)
                
                for loss in self.losses:
                    if loss == 'tracking':
                        # Skip tracking loss for intermediate layers (optional)
                        continue
                    if loss == 'duplicates':
                        # Skip duplicate loss for intermediate layers
                        continue
                    
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    
                    lDict = self.getLoss(loss, auxOutputs, targets, auxIndices, numBoxes, **kwargs)
                    lDict = {k + f'_{i}': v for k, v in lDict.items()}
                    losses.update(lDict)
        
        # ---- Label Denoising Losses (training only) ----
        if 'dn_meta' in outputs and outputs['dn_meta'] is not None:
            dnLosses = self._computeDenoisingLoss(outputs, numBoxes)
            losses.update(dnLosses)
            
            # Auxiliary DN losses
            if 'dn_aux_outputs' in outputs:
                for i, dnAuxOutputs in enumerate(outputs['dn_aux_outputs']):
                    auxDnLosses = self._computeDenoisingLoss(
                        {
                            'dn_pred_logits': dnAuxOutputs['dn_pred_logits'],
                            'dn_pred_boxes': dnAuxOutputs['dn_pred_boxes'],
                            'dn_meta': outputs['dn_meta'],
                        },
                        numBoxes
                    )
                    auxDnLosses = {k + f'_{i}': v for k, v in auxDnLosses.items()}
                    losses.update(auxDnLosses)
        
        return losses
    
    def _computeDenoisingLoss(
        self,
        outputs: Dict[str, Tensor],
        numBoxes: float
    ) -> Dict[str, Tensor]:
        """
        Compute denoising losses directly (no Hungarian matching needed).
        
        The DN queries have a known correspondence to GT objects, so we compute
        classification and box regression losses directly.
        
        Args:
            outputs: Must contain 'dn_pred_logits', 'dn_pred_boxes', 'dn_meta'
            numBoxes: Normalization factor
            
        Returns:
            Dict with 'loss_dn_ce' and 'loss_dn_bbox', 'loss_dn_giou'
        """
        dnPredLogits = outputs['dn_pred_logits']  # [B, totalDnQueries, numClasses+1]
        dnPredBoxes = outputs['dn_pred_boxes']  # [B, totalDnQueries, 4]
        dnMeta = outputs['dn_meta']
        
        device = dnPredLogits.device
        batchSize = dnPredLogits.shape[0]
        
        # Get known GT labels and boxes
        knownLabels = dnMeta['known_labels'].to(device)  # [B, totalDnQueries]
        knownBoxes = dnMeta['known_boxes'].to(device)  # [B, totalDnQueries, 4]
        validMask = dnMeta['dn_valid_mask'].to(device)  # [B, totalDnQueries]
        
        # ---- Classification loss ----
        if self.useFocalLoss:
            # Build one-hot targets
            targetClassesOnehot = torch.zeros_like(dnPredLogits)
            targetClassesOnehot.scatter_(2, knownLabels.unsqueeze(-1), 1)
            
            # Mask out invalid (padded) queries: set them to no-object
            # For invalid queries, we zero-out the one-hot (all zeros = no-object for sigmoid)
            targetClassesOnehot = targetClassesOnehot * validMask.unsqueeze(-1).float()
            
            lossDnCe = sigmoidFocalLoss(
                dnPredLogits, targetClassesOnehot, numBoxes,
                alpha=self.focalAlpha, gamma=self.focalGamma
            )
        else:
            # Cross-entropy: set invalid to no-object class
            targetLabels = knownLabels.clone()
            targetLabels[~validMask] = self.numClasses
            
            lossDnCe = F.cross_entropy(
                dnPredLogits.transpose(1, 2),
                targetLabels,
                self.emptyWeight.to(device)
            )
        
        # ---- Box losses (only for valid queries) ----
        if validMask.any():
            # Get valid predictions and targets
            validPredBoxes = dnPredBoxes[validMask]  # [numValid, 4]
            validTgtBoxes = knownBoxes[validMask]  # [numValid, 4]
            
            # L1 loss
            lossDnBbox = F.l1_loss(validPredBoxes, validTgtBoxes, reduction='none')
            lossDnBbox = lossDnBbox.sum() / numBoxes
            
            # GIoU loss
            lossDnGiou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(validPredBoxes),
                box_ops.box_cxcywh_to_xyxy(validTgtBoxes)
            ))
            lossDnGiou = lossDnGiou.sum() / numBoxes
        else:
            lossDnBbox = torch.tensor(0.0, device=device, requires_grad=True)
            lossDnGiou = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {
            'loss_dn_ce': lossDnCe,
            'loss_dn_bbox': lossDnBbox,
            'loss_dn_giou': lossDnGiou,
        }


class PostProcess(nn.Module):
    """
    Post-processor for VideoDETR outputs.
    
    Converts model outputs to the format expected by evaluation metrics.
    """
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targetSizes: Tensor,
        frameIdx: Optional[int] = None
    ) -> List[Dict[str, Tensor]]:
        """
        Convert outputs to evaluation format.
        
        Args:
            outputs: Model outputs
            targetSizes: [B, 2] tensor of (height, width) for each image
            frameIdx: Optional frame index to extract specific frame outputs
        
        Returns:
            List of result dicts with 'scores', 'labels', 'boxes'
        """
        outLogits = outputs['pred_logits']
        outBbox = outputs['pred_boxes']
        
        assert len(outLogits) == len(targetSizes)
        assert targetSizes.shape[1] == 2
        
        # Softmax for probabilities
        prob = F.softmax(outLogits, -1)
        scores, labels = prob[..., :-1].max(-1)
        
        # Convert boxes from cxcywh to xyxy
        boxes = box_ops.box_cxcywh_to_xyxy(outBbox)
        
        # Scale to image size
        imgH, imgW = targetSizes.unbind(1)
        scaleFct = torch.stack([imgW, imgH, imgW, imgH], dim=1)
        boxes = boxes * scaleFct[:, None, :]
        
        # Build results
        results = []
        for s, l, b in zip(scores, labels, boxes):
            result = {'scores': s, 'labels': l, 'boxes': b}
            
            # Add tracking embeddings if available
            if 'pred_tracking' in outputs:
                result['tracking'] = outputs['pred_tracking']
            
            results.append(result)
        
        return results


def buildVideoCriterion(args) -> VideoCriterion:
    """
    Build VideoCriterion from args.
    
    Args:
        args: Argument namespace with configuration
        
    Returns:
        VideoCriterion module
    """
    # Build matcher
    matcher = VideoHungarianMatcher(
        costClass=args.setCostClass,
        costBbox=args.setCostBbox,
        costGiou=args.setCostGiou,
        numFrames=args.numFrames,
        queriesPerFrame=args.queriesPerFrame
    )
    
    useFocalLoss = getattr(args, 'useFocalLoss', True)
    useDnDenoising = getattr(args, 'useDnDenoising', True)
    
    # Build weight dict
    # With focal loss, the CE weight needs to be higher since focal loss 
    # produces smaller values due to the (1-pt)^gamma factor
    ceLossCoef = 2.0 if useFocalLoss else 1.0
    
    weightDict = {
        'loss_ce': ceLossCoef,
        'loss_bbox': args.bboxLossCoef,
        'loss_giou': args.giouLossCoef,
        'loss_tracking': getattr(args, 'trackingLossCoef', 1.0),
        'loss_duplicates': getattr(args, 'dupLossCoef', 0.5)
    }
    
    # Add denoising loss weights (same coefficients as matching losses)
    if useDnDenoising:
        dnLossCoef = getattr(args, 'dnLossCoef', 1.0)
        weightDict['loss_dn_ce'] = ceLossCoef * dnLossCoef
        weightDict['loss_dn_bbox'] = args.bboxLossCoef * dnLossCoef
        weightDict['loss_dn_giou'] = args.giouLossCoef * dnLossCoef
    
    # Add auxiliary loss weights
    if args.auxLoss:
        auxWeightDict = {}
        for i in range(args.decLayers - 1):
            auxWeightDict.update({k + f'_{i}': v for k, v in weightDict.items()})
        # Remove tracking and duplicates from aux (computed only at the last layer)
        auxWeightDict = {k: v for k, v in auxWeightDict.items() if 'tracking' not in k and 'duplicates' not in k}
        weightDict.update(auxWeightDict)
        print(f"[buildVideoCriterion] Auxiliary losses enabled for {args.decLayers - 1} decoder layers")
    
    losses = ['labels', 'boxes', 'cardinality', 'tracking', 'duplicates']
    
    criterion = VideoCriterion(
        numClasses=args.numClasses,
        matcher=matcher,
        weightDict=weightDict,
        eosCoef=args.eosCoef,
        losses=losses,
        numFrames=args.numFrames,
        queriesPerFrame=args.queriesPerFrame,
        contrastiveTemp=getattr(args, 'contrastiveTemp', 0.07),
        useFocalLoss=useFocalLoss,
        focalAlpha=getattr(args, 'focalAlpha', 0.25),
        focalGamma=getattr(args, 'focalGamma', 2.0)
    )
    
    print(f"[buildVideoCriterion] Weight dict keys: {list(weightDict.keys())}")
    
    return criterion
