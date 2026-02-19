# Copyright (c) 2026. All Rights Reserved.
"""
Video Sequence Dataset for VideoDETR.

This module implements a dataset loader for video sequences in YOLO format with
tracking annotations. The key feature is that objects on the same line number
across label files in a sequence represent the same object (tracking correspondence).

Dataset structure:
    train/
    ├── images/
    │   ├── seq_000001_frame_0000.jpg
    │   ├── seq_000001_frame_0001.jpg
    │   └── ...
    └── labels/
        ├── seq_000001_frame_0000.txt
        ├── seq_000001_frame_0001.txt
        └── ...

Label format (YOLO style):
    class_id center_x center_y width height
    (normalized coordinates in [0, 1])
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import hashlib
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml

from datasets import transforms as T

# Cache version - increment when cache format changes
CACHE_VERSION = "1.0"


class VideoSequenceDataset(Dataset):
    """
    Dataset for loading video sequences with tracking annotations.
    
    Each sample consists of N frames from a single video sequence, along with
    bounding box annotations and tracking IDs (derived from line numbers in
    label files).
    
    Args:
        dataRoot: Root directory containing 'images' and 'labels' subdirs
        numFrames: Number of frames to sample per sequence (default: 5)
        transforms: Optional transforms to apply to each frame
        imageSet: 'train' or 'val' (affects augmentation behavior)
        framesPerSequence: Total frames in each sequence (default: 50)
        minFrameGap: Minimum gap between sampled frames (default: 1)
        maxFrameGap: Maximum gap between sampled frames (default: 10)
        classNames: Optional list of class names
        useCache: Whether to use cached sequence information (default: True)
        minBoxSize: Minimum GT box size as a fraction of image width or
                    height.  Boxes whose normalised width **and** height
                    are both smaller than this value are dropped.
                    Set to 0.0 to keep all boxes.  Default: 0.0.
    """
    
    # Regex pattern for parsing filenames: seq_XXXXXX_frame_XXXX
    FILENAME_PATTERN = re.compile(r'seq_(\d{6})_frame_(\d{4})')
    
    def __init__(
        self,
        dataRoot: str,
        numFrames: int = 5,
        transforms: Optional[Any] = None,
        imageSet: str = 'train',
        framesPerSequence: int = 50,
        minFrameGap: int = 1,
        maxFrameGap: int = 10,
        classNames: Optional[List[str]] = None,
        useCache: bool = True,
        minBoxSize: float = 0.0,
    ):
        super().__init__()
        
        self.dataRoot = Path(dataRoot)
        self.numFrames = numFrames
        self.transforms = transforms
        self.imageSet = imageSet
        self.framesPerSequence = framesPerSequence
        self.minFrameGap = minFrameGap
        self.maxFrameGap = maxFrameGap
        self.classNames = classNames
        self.useCache = useCache
        self.minBoxSize = minBoxSize
        
        # Paths to images and labels directories
        self.imagesDir = self.dataRoot / 'images'
        self.labelsDir = self.dataRoot / 'labels'
        
        # Validate directories exist
        assert self.imagesDir.exists(), f"Images directory not found: {self.imagesDir}"
        assert self.labelsDir.exists(), f"Labels directory not found: {self.labelsDir}"
        
        # Discover all sequences and their frames (with caching)
        self.sequences = self._loadOrDiscoverSequences()
        
        # Create list of sequence IDs for indexing
        self.sequenceIds = list(self.sequences.keys())
        
        print(f"[VideoSequenceDataset] Found {len(self.sequences)} sequences "
              f"with {numFrames} frames per sample")
    
    def _getCachePath(self) -> Path:
        """Get the path to the cache file for this dataset."""
        # Create a hash of the data root to make cache file unique
        rootHash = hashlib.md5(str(self.dataRoot.resolve()).encode()).hexdigest()[:8]
        return self.dataRoot / f".viddetr_cache_{rootHash}.json"
    
    def _isCacheValid(self, cachePath: Path) -> bool:
        """Check if the cache file is valid and up-to-date."""
        if not cachePath.exists():
            return False
        
        try:
            with open(cachePath, 'r') as f:
                cache = json.load(f)
            
            # Check cache version
            if cache.get('version') != CACHE_VERSION:
                print(f"[VideoSequenceDataset] Cache version mismatch, rebuilding...")
                return False
            
            # Check if directories have been modified
            cacheTime = cache.get('timestamp', 0)
            imgsDirMtime = os.path.getmtime(self.imagesDir)
            labelsDirMtime = os.path.getmtime(self.labelsDir)
            
            if imgsDirMtime > cacheTime or labelsDirMtime > cacheTime:
                print(f"[VideoSequenceDataset] Dataset modified since cache, rebuilding...")
                return False
            
            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[VideoSequenceDataset] Cache read error: {e}, rebuilding...")
            return False
    
    def _loadOrDiscoverSequences(self) -> Dict[str, List[int]]:
        """Load sequences from cache or discover them from disk."""
        cachePath = self._getCachePath()
        
        if self.useCache and self._isCacheValid(cachePath):
            print(f"[VideoSequenceDataset] Loading from cache: {cachePath}")
            with open(cachePath, 'r') as f:
                cache = json.load(f)
            sequences = {k: v for k, v in cache['sequences'].items()}
            # Filter sequences with enough frames
            sequences = {
                seqId: frames 
                for seqId, frames in sequences.items() 
                if len(frames) >= self.numFrames
            }
            return sequences
        
        # Discover sequences from disk
        print(f"[VideoSequenceDataset] Scanning dataset directory...")
        sequences = self._discoverSequences()
        
        # Save to cache
        if self.useCache:
            self._saveCache(cachePath, sequences)
        
        return sequences
    
    def _saveCache(self, cachePath: Path, sequences: Dict[str, List[int]]) -> None:
        """Save discovered sequences to cache file."""
        try:
            # Include all sequences in cache (before filtering by numFrames)
            # so cache can be reused with different numFrames settings
            allSequences = self._discoverSequencesUnfiltered()
            
            cache = {
                'version': CACHE_VERSION,
                'timestamp': max(
                    os.path.getmtime(self.imagesDir),
                    os.path.getmtime(self.labelsDir)
                ),
                'dataRoot': str(self.dataRoot.resolve()),
                'numSequences': len(allSequences),
                'sequences': allSequences
            }
            
            with open(cachePath, 'w') as f:
                json.dump(cache, f)
            
            print(f"[VideoSequenceDataset] Cache saved: {cachePath}")
        except OSError as e:
            print(f"[VideoSequenceDataset] Failed to save cache: {e}")
    
    def _discoverSequencesUnfiltered(self) -> Dict[str, List[int]]:
        """Discover all sequences without filtering by numFrames."""
        sequences = {}
        imageExtensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for imgFile in self.imagesDir.iterdir():
            if imgFile.suffix.lower() not in imageExtensions:
                continue
            
            match = self.FILENAME_PATTERN.match(imgFile.stem)
            if not match:
                continue
            
            seqId = match.group(1)
            frameIdx = int(match.group(2))
            
            labelFile = self.labelsDir / f"seq_{seqId}_frame_{frameIdx:04d}.txt"
            if not labelFile.exists():
                continue
            
            if seqId not in sequences:
                sequences[seqId] = []
            sequences[seqId].append(frameIdx)
        
        for seqId in sequences:
            sequences[seqId] = sorted(sequences[seqId])
        
        return sequences
    
    def _discoverSequences(self) -> Dict[str, List[int]]:
        """
        Discover all sequences and their available frame indices.
        
        Returns:
            Dictionary mapping sequence ID to list of frame indices
        """
        sequences = self._discoverSequencesUnfiltered()
        
        # Filter sequences with enough frames
        sequences = {
            seqId: frames 
            for seqId, frames in sequences.items() 
            if len(frames) >= self.numFrames
        }
        
        return sequences
    
    def _sampleFrameIndices(self, availableFrames: List[int]) -> List[int]:
        """
        Sample frame indices from available frames with controlled spread.
        
        This ensures temporal diversity while maintaining reasonable motion
        between frames. Uses stratified sampling for better coverage.
        
        Args:
            availableFrames: List of available frame indices in the sequence
            
        Returns:
            List of sampled frame indices (sorted)
        """
        numAvailable = len(availableFrames)
        
        if self.imageSet == 'train':
            # Training: Random sampling with controlled gaps
            # Strategy: Divide available frames into numFrames segments,
            # then randomly sample one frame from each segment
            
            segmentSize = numAvailable // self.numFrames
            sampledFrames = []
            
            for i in range(self.numFrames):
                segmentStart = i * segmentSize
                segmentEnd = (i + 1) * segmentSize if i < self.numFrames - 1 else numAvailable
                
                # Random frame from this segment
                frameIdx = random.randint(segmentStart, segmentEnd - 1)
                sampledFrames.append(availableFrames[frameIdx])
            
            # Additional randomization: sometimes shuffle the order
            # (disabled for now to maintain temporal order)
            
        else:
            # Validation: Uniform sampling for reproducibility
            step = (numAvailable - 1) / (self.numFrames - 1) if self.numFrames > 1 else 0
            sampledFrames = [
                availableFrames[min(int(i * step), numAvailable - 1)]
                for i in range(self.numFrames)
            ]
        
        return sorted(sampledFrames)
    
    def _loadImage(self, seqId: str, frameIdx: int) -> Image.Image:
        """Load a single image from the dataset.
        
        Eagerly loads pixel data and closes the file handle to avoid
        leaking file descriptors in multiprocessing workers.
        """
        imgPath = self.imagesDir / f"seq_{seqId}_frame_{frameIdx:04d}.jpg"
        
        # Try different extensions if .jpg doesn't exist
        if not imgPath.exists():
            for ext in ['.png', '.jpeg', '.bmp']:
                altPath = imgPath.with_suffix(ext)
                if altPath.exists():
                    imgPath = altPath
                    break
        
        # Open, force pixel decode, then close the file handle.
        # PIL uses lazy loading by default — the underlying file stays open
        # until .load() is called. In DataLoader workers this leaks FDs.
        with Image.open(imgPath) as img:
            img.load()  # force full decode into memory
            return img.convert('RGB')
    
    def _loadLabels(
        self, 
        seqId: str, 
        frameIdx: int,
        imgWidth: int,
        imgHeight: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load labels for a single frame.
        
        Args:
            seqId: Sequence identifier
            frameIdx: Frame index
            imgWidth: Image width for denormalization
            imgHeight: Image height for denormalization
            
        Returns:
            Tuple of (boxes, labels, trackIds) where:
            - boxes: [N, 4] tensor in cxcywh format (normalized)
            - labels: [N] tensor of class indices
            - trackIds: [N] tensor of tracking IDs (line numbers)
        """
        labelPath = self.labelsDir / f"seq_{seqId}_frame_{frameIdx:04d}.txt"
        
        boxes = []
        labels = []
        trackIds = []
        
        if labelPath.exists():
            with open(labelPath, 'r') as f:
                for lineIdx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    # YOLO format: class_id cx cy w h (normalized)
                    classId = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Validate coordinates
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        continue
                    
                    # Skip boxes that are too small (both w and h below threshold)
                    if self.minBoxSize > 0 and w < self.minBoxSize and h < self.minBoxSize:
                        continue
                    
                    boxes.append([cx, cy, w, h])
                    labels.append(classId)
                    trackIds.append(lineIdx)  # Line number = track ID
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            trackIds = torch.tensor(trackIds, dtype=torch.int64)
        else:
            # Empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            trackIds = torch.zeros((0,), dtype=torch.int64)
        
        return boxes, labels, trackIds
    
    def __len__(self) -> int:
        """Return number of sequences in the dataset."""
        return len(self.sequenceIds)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Get a sample consisting of N frames and their annotations.
        
        Args:
            idx: Index of the sequence to load
            
        Returns:
            Tuple of (images, targets) where:
            - images: List of N image tensors [3, H, W]
            - targets: List of N target dicts, each containing:
                - boxes: [M, 4] bounding boxes in cxcywh format
                - labels: [M] class labels
                - trackIds: [M] tracking IDs for cross-frame association
                - frameIdx: Frame index within the clip (0 to N-1)
                - origSize: Original image size [H, W]
                - size: Current image size [H, W]
                - imageId: Unique identifier for this frame
                - seqId: Sequence identifier
        """
        seqId = self.sequenceIds[idx]
        availableFrames = self.sequences[seqId]
        
        # Sample frame indices
        sampledFrames = self._sampleFrameIndices(availableFrames)
        
        images = []
        targets = []
        
        for clipFrameIdx, frameIdx in enumerate(sampledFrames):
            # Load image
            img = self._loadImage(seqId, frameIdx)
            imgWidth, imgHeight = img.size
            
            # Load labels
            boxes, labels, trackIds = self._loadLabels(
                seqId, frameIdx, imgWidth, imgHeight
            )
            
            # Prepare target dict
            # Note: We need 'iscrowd' and 'area' fields for DETR transforms compatibility
            numBoxes = len(boxes)
            
            # Compute area from normalized cxcywh boxes (w * h * imgW * imgH)
            if numBoxes > 0:
                # boxes are in cxcywh normalized format
                area = boxes[:, 2] * boxes[:, 3] * imgWidth * imgHeight
            else:
                area = torch.zeros((0,), dtype=torch.float32)
            
            target = {
                'boxes': boxes,  # cxcywh normalized
                'labels': labels,
                'trackIds': trackIds,
                'iscrowd': torch.zeros((numBoxes,), dtype=torch.int64),  # No crowd annotations
                'area': area,
                'frameIdx': torch.tensor([clipFrameIdx]),
                'origSize': torch.tensor([imgHeight, imgWidth]),
                'size': torch.tensor([imgHeight, imgWidth]),
                'imageId': torch.tensor([int(seqId) * 10000 + frameIdx]),
                'seqId': seqId,
                'seqFrameIdx': frameIdx,
            }
            
            # Convert boxes from cxcywh to xyxy for transforms (then back)
            if numBoxes > 0:
                # Convert normalized cxcywh to absolute xyxy for transforms
                boxesXyxy = self._cxcywhToXyxy(boxes, imgWidth, imgHeight)
                target['boxes'] = boxesXyxy
            
            # Apply transforms
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            
            images.append(img)
            targets.append(target)
        
        return images, targets
    
    def _cxcywhToXyxy(
        self, 
        boxes: torch.Tensor, 
        imgWidth: int, 
        imgHeight: int
    ) -> torch.Tensor:
        """
        Convert boxes from normalized cxcywh to absolute xyxy format.
        
        Args:
            boxes: [N, 4] tensor in normalized cxcywh format
            imgWidth: Image width
            imgHeight: Image height
            
        Returns:
            [N, 4] tensor in absolute xyxy format
        """
        cx, cy, w, h = boxes.unbind(-1)
        
        # Denormalize
        cx = cx * imgWidth
        cy = cy * imgHeight
        w = w * imgWidth
        h = h * imgHeight
        
        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)


def makeVideoTransforms(imageSet: str, maxSize: int = 800):
    """
    Create transforms for video dataset.
    
    Note: We use simpler transforms than DETR because we need consistency
    across frames in a clip. Heavy augmentation is applied at the clip level.
    
    Args:
        imageSet: 'train' or 'val'
        maxSize: Maximum image size
        
    Returns:
        Composed transforms
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    if imageSet == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.RandomSelect(
                T.RandomResize(scales, max_size=maxSize),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=maxSize),
                ])
            ),
            normalize,
            T.RandomErasing(p=0.1),
        ])
    
    if imageSet == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=maxSize),
            normalize,
        ])
    
    raise ValueError(f"Unknown image set: {imageSet}")


def videoCollateFn(batch: List[Tuple]) -> Tuple[List[Any], List[List[Dict]]]:
    """
    Collate function for video batches.
    
    Unlike standard DETR collate, we handle sequences of frames.
    
    Args:
        batch: List of (images, targets) tuples where images is a list of N tensors
        
    Returns:
        Tuple of (nestedTensorList, targetsList) where:
        - nestedTensorList: List of N NestedTensors, each of shape [B, 3, H, W]
        - targetsList: List of N lists of target dicts
    """
    # Import NestedTensor utilities
    from util.misc import nested_tensor_from_tensor_list
    
    # Unzip batch
    imagesBatch = [item[0] for item in batch]  # List of B lists of N tensors
    targetsBatch = [item[1] for item in batch]  # List of B lists of N dicts
    
    batchSize = len(imagesBatch)
    numFrames = len(imagesBatch[0])
    
    # Reorganize: from [B, N] to [N, B]
    framesPerTimestep = []
    targetsPerTimestep = []
    
    for frameIdx in range(numFrames):
        # Collect all images for this frame across batch
        frameImages = [imagesBatch[b][frameIdx] for b in range(batchSize)]
        frameTargets = [targetsBatch[b][frameIdx] for b in range(batchSize)]
        
        # Create NestedTensor for this frame
        nestedTensor = nested_tensor_from_tensor_list(frameImages)
        
        framesPerTimestep.append(nestedTensor)
        targetsPerTimestep.append(frameTargets)
    
    return framesPerTimestep, targetsPerTimestep


def buildVideoDataset(args) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Build train and validation datasets from args.
    
    Args:
        args: Argument namespace with dataset configuration.
              If ``args.mergeTrainVal`` is True, the validation split is
              loaded with training augmentations, concatenated with the
              training split, and returned as ``(mergedDataset, None)``.
        
    Returns:
        Tuple of (trainDataset, valDataset).  ``valDataset`` is ``None``
        when ``mergeTrainVal`` is enabled.
    """
    # Load data config from yaml
    dataConfigPath = Path(args.dataConfig)
    with open(dataConfigPath, 'r') as f:
        dataConfig = yaml.safe_load(f)
    
    trainRoot = dataConfig.get('train', '')
    valRoot = dataConfig.get('val', '')
    classNames = list(dataConfig.get('names', {}).values())
    
    # Replace 'images' suffix to get root directory
    if trainRoot.endswith('/images'):
        trainRoot = trainRoot[:-7]
    if valRoot.endswith('/images'):
        valRoot = valRoot[:-7]
    
    mergeTrainVal = getattr(args, 'mergeTrainVal', False)
    
    # Build datasets
    trainTransforms = makeVideoTransforms('train', maxSize=args.maxSize)
    
    minBoxSize = getattr(args, 'minBoxSize', 0.0)
    
    trainDataset = VideoSequenceDataset(
        dataRoot=trainRoot,
        numFrames=args.numFrames,
        transforms=trainTransforms,
        imageSet='train',
        framesPerSequence=args.framesPerSequence,
        minFrameGap=args.minFrameGap,
        maxFrameGap=args.maxFrameGap,
        classNames=classNames,
        minBoxSize=minBoxSize,
    )
    
    if mergeTrainVal:
        # Load validation split with *train* transforms so it acts as
        # additional training data.
        valAsTrainDataset = VideoSequenceDataset(
            dataRoot=valRoot,
            numFrames=args.numFrames,
            transforms=trainTransforms,
            imageSet='train',   # use train augmentations & sampling
            framesPerSequence=args.framesPerSequence,
            minFrameGap=args.minFrameGap,
            maxFrameGap=args.maxFrameGap,
            classNames=classNames,
            minBoxSize=minBoxSize,
        )
        merged = torch.utils.data.ConcatDataset([trainDataset, valAsTrainDataset])
        print(
            f"[buildVideoDataset] mergeTrainVal: {len(trainDataset)} + "
            f"{len(valAsTrainDataset)} = {len(merged)} sequences"
        )
        return merged, None
    
    valTransforms = makeVideoTransforms('val', maxSize=args.maxSize)
    
    valDataset = VideoSequenceDataset(
        dataRoot=valRoot,
        numFrames=args.numFrames,
        transforms=valTransforms,
        imageSet='val',
        framesPerSequence=args.framesPerSequence,
        minFrameGap=args.minFrameGap,
        maxFrameGap=args.maxFrameGap,
        classNames=classNames,
        minBoxSize=minBoxSize,
    )
    
    return trainDataset, valDataset
