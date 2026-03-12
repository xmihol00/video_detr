# Copyright (c) 2026. All Rights Reserved.
"""
Video Dataset for VideoDETR — single long-sequence format.

This module implements a dataset loader for a video dataset where **all
frames belong to one long video sequence** (per split).  Images are sorted
alphabetically to establish temporal order, and labels carry explicit
track IDs.

Dataset structure:
    <dataRoot>/
    ├── train/
    │   ├── images/
    │   │   ├── frame_000000.jpg
    │   │   ├── frame_000001.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── frame_000000.txt
    │       ├── frame_000001.txt
    │       └── ...
    └── val/
        ├── images/   (may be empty initially)
        └── labels/   (may be empty initially)

Label format (one object per line):
    track_id  class_id  cx  cy  w  h
    (cx, cy, w, h are normalised coordinates in [0, 1])

Because the dataset is a single sequence, the natural ``__len__`` equals
the number of valid clips that can be formed from the frames.  However,
since the sequence can be extremely long, the user can cap the number of
batches seen per epoch via ``batchesPerEpoch`` (default: 1000 for train,
50 for val).  The dataset's ``__len__`` is set to
``batchesPerEpoch * batchSize`` so that the dataloader exhausts it in
exactly the desired number of batches.

Frame sampling strategies (configurable via ``--videoSamplingStrategy``):

1. **random_walk** (default) — pick a random starting frame, then add a
   random offset in [0, ``maxFrameOffset``] for each subsequent frame.
   This is the simplest approach and gives good variety.

2. **exponential_stride** — pick a random base stride from a geometric
   distribution (small strides are more likely) and sample frames evenly
   at that stride.  Inspired by TAO / BURST temporal sampling.

3. **mixed** — SOTA-style strategy that blends short-range fine-grained
   clips and long-range coarse clips.  With probability 0.5 it samples a
   tight clip (offset 0-5), and with 0.5 it samples a wide clip (offset
   5-``maxFrameOffset``).  This is similar to the multi-scale temporal
   jittering used in TubeDETR / TrackFormer / MOTR.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from vidDetr.datasets import transforms as T

# Cache version — bump when on-disk format changes.
CACHE_VERSION = "1.0"


# ======================================================================
#  Sampling strategies
# ======================================================================

def _sampleRandomWalk(
    numFrames: int,
    totalFrames: int,
    maxOffset: int,
) -> List[int]:
    """
    Random-walk sampling: pick a random start, then step forward by a
    random offset in [0, maxOffset] for each subsequent frame.

    Guarantees:
    - All indices are in [0, totalFrames).
    - Consecutive frames are at most ``maxOffset`` frames apart.
    - The sequence is strictly non-decreasing.
    """
    # The maximum span of the clip is (numFrames - 1) * maxOffset.
    maxSpan = (numFrames - 1) * maxOffset
    # The latest possible start so that we can still fit the clip even
    # if all offsets are at maximum.
    latestStart = max(0, totalFrames - 1 - maxSpan)
    start = random.randint(0, max(0, totalFrames - 1)) if latestStart == 0 else random.randint(0, latestStart)

    indices = [start]
    for _ in range(numFrames - 1):
        offset = random.randint(0, maxOffset)
        nextIdx = min(indices[-1] + offset, totalFrames - 1)
        indices.append(nextIdx)
    return indices


def _sampleExponentialStride(
    numFrames: int,
    totalFrames: int,
    minGap: int,
    maxGap: int,
) -> List[int]:
    """
    Exponential-stride sampling: pick a stride with geometrically
    decreasing probability (smaller strides more likely).  Then
    sample at that uniform stride.
    """
    # Choose stride with P(stride=s) ∝ 0.5^(s - minGap)
    maxPossibleGap = min(maxGap, (totalFrames - 1) // max(numFrames - 1, 1))
    maxPossibleGap = max(maxPossibleGap, minGap)

    weights = [0.5 ** (g - minGap) for g in range(minGap, maxPossibleGap + 1)]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    stride = minGap
    for g, w in zip(range(minGap, maxPossibleGap + 1), weights):
        cumulative += w
        if r <= cumulative:
            stride = g
            break

    span = (numFrames - 1) * stride
    maxStart = max(0, totalFrames - 1 - span)
    start = random.randint(0, maxStart)
    return [start + i * stride for i in range(numFrames)]


def _sampleMixed(
    numFrames: int,
    totalFrames: int,
    maxOffset: int,
) -> List[int]:
    """
    SOTA mixed sampling: with p=0.5 use a tight clip (offset 0-5) and
    with p=0.5 use a wide clip (offset 5-maxOffset).  This gives the
    model exposure to both fine-grained and coarse temporal dynamics,
    similar to multi-scale temporal jittering in MOTR / TrackFormer.
    """
    tightMax = min(5, maxOffset)
    wideMin = min(5, maxOffset)

    if random.random() < 0.5:
        # Tight clip
        return _sampleRandomWalk(numFrames, totalFrames, tightMax)
    else:
        # Wide clip — use exponential stride for the wide part
        return _sampleExponentialStride(numFrames, totalFrames, wideMin, maxOffset)


SAMPLING_STRATEGIES = {
    "random_walk": _sampleRandomWalk,
    "exponential_stride": _sampleExponentialStride,
    "mixed": _sampleMixed,
}


# ======================================================================
#  Dataset
# ======================================================================

class VideoDataset(Dataset):
    """
    Dataset for a single long video sequence per split.

    The dataset scans all images under ``dataRoot/images/`` and sorts them
    alphabetically to establish temporal order.  Each ``__getitem__`` call
    samples ``numFrames`` frames from this sequence using the configured
    sampling strategy.

    Because the dataset is a single sequence, the ``__len__`` is set to
    ``epochLength`` so that one epoch corresponds to a fixed number of
    samples (= ``batchesPerEpoch * batchSize``).

    Args:
        dataRoot:           Path containing ``images/`` and ``labels/`` dirs.
        numFrames:          Frames per clip.
        transforms:         DETR-style per-frame transforms.
        imageSet:           ``'train'`` or ``'val'``.
        batchesPerEpoch:    Number of batches that constitute one epoch.
        batchSize:          Batch size (needed to compute epoch length).
        maxFrameOffset:     Maximum gap between consecutive sampled frames.
        samplingStrategy:   One of ``'random_walk'``, ``'exponential_stride'``,
                            or ``'mixed'``.
        classNames:         Optional list of class names.
        numClasses:         Number of object classes.
        useCache:           Cache the discovered frame list.
        minBoxSize:         Minimum normalised box size (boxes with both
                            w and h below this are dropped).
    """

    def __init__(
        self,
        dataRoot: str,
        numFrames: int = 5,
        transforms: Optional[Any] = None,
        imageSet: str = "train",
        batchesPerEpoch: int = 1000,
        batchSize: int = 32,
        maxFrameOffset: int = 30,
        samplingStrategy: str = "mixed",
        classNames: Optional[List[str]] = None,
        numClasses: int = 80,
        useCache: bool = True,
        minBoxSize: float = 0.0,
    ):
        super().__init__()

        self.dataRoot = Path(dataRoot)
        self.numFrames = numFrames
        self.transforms = transforms
        self.imageSet = imageSet
        self.batchesPerEpoch = batchesPerEpoch
        self.batchSize = batchSize
        self.maxFrameOffset = maxFrameOffset
        self.samplingStrategy = samplingStrategy
        self.classNames = classNames
        self.numClasses = numClasses
        self.useCache = useCache
        self.minBoxSize = minBoxSize

        self.imagesDir = self.dataRoot / "images"
        self.labelsDir = self.dataRoot / "labels"

        # The directories must exist, but may be empty (especially val).
        assert self.imagesDir.exists(), f"Images directory not found: {self.imagesDir}"
        assert self.labelsDir.exists(), f"Labels directory not found: {self.labelsDir}"

        # Discover frames (sorted alphabetically = temporal order).
        self.framePaths: List[Path] = self._loadOrDiscoverFrames()
        self.totalFrames = len(self.framePaths)

        # Epoch length = batchesPerEpoch * batchSize.  If fewer frames are
        # available (or zero), we clamp appropriately.
        if self.totalFrames == 0:
            self._epochLength = 0
        else:
            self._epochLength = batchesPerEpoch * batchSize

        print(
            f"[VideoDataset] {imageSet}: {self.totalFrames} frames, "
            f"epochLength={self._epochLength}, "
            f"strategy={samplingStrategy}, "
            f"maxOffset={maxFrameOffset}"
        )

    # ------------------------------------------------------------------
    #  Frame discovery (with caching)
    # ------------------------------------------------------------------
    def _getCachePath(self) -> Path:
        rootHash = hashlib.md5(str(self.dataRoot.resolve()).encode()).hexdigest()[:8]
        return self.dataRoot / f".video_dataset_cache_{rootHash}.json"

    def _isCacheValid(self, cachePath: Path) -> bool:
        if not cachePath.exists():
            return False
        try:
            with open(cachePath, "r") as f:
                cache = json.load(f)
            if cache.get("version") != CACHE_VERSION:
                return False
            cacheTime = cache.get("timestamp", 0)
            if os.path.getmtime(self.imagesDir) > cacheTime:
                return False
            return True
        except (json.JSONDecodeError, KeyError, OSError):
            return False

    def _loadOrDiscoverFrames(self) -> List[Path]:
        cachePath = self._getCachePath()
        if self.useCache and self._isCacheValid(cachePath):
            print(f"[VideoDataset] Loading frame list from cache: {cachePath}")
            with open(cachePath, "r") as f:
                cache = json.load(f)
            return [Path(p) for p in cache["frames"]]

        print(f"[VideoDataset] Scanning {self.imagesDir} ...")
        frames = self._discoverFrames()

        if self.useCache and frames:
            self._saveCache(cachePath, frames)
        return frames

    def _discoverFrames(self) -> List[Path]:
        """Return image paths sorted alphabetically (= temporal order)."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        frames = sorted(
            p for p in self.imagesDir.iterdir()
            if p.suffix.lower() in extensions
        )
        return frames

    def _saveCache(self, cachePath: Path, frames: List[Path]) -> None:
        try:
            cache = {
                "version": CACHE_VERSION,
                "timestamp": os.path.getmtime(self.imagesDir),
                "numFrames": len(frames),
                "frames": [str(p) for p in frames],
            }
            with open(cachePath, "w") as f:
                json.dump(cache, f)
            print(f"[VideoDataset] Cache saved: {cachePath}")
        except OSError as e:
            print(f"[VideoDataset] Failed to save cache: {e}")

    # ------------------------------------------------------------------
    #  Label loading
    # ------------------------------------------------------------------
    def _getLabelPath(self, imagePath: Path) -> Path:
        """Derive label file path from image path (same stem, .txt)."""
        return self.labelsDir / (imagePath.stem + ".txt")

    def _loadLabels(
        self, labelPath: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse a label file.

        Each line: ``track_id  class_id  cx  cy  w  h`` (normalised).

        Returns:
            boxes    : [N, 4]  cxcywh normalised
            labels   : [N]     class indices
            trackIds : [N]     track identity
        """
        boxes, labels, trackIds = [], [], []

        if labelPath.exists():
            with open(labelPath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue

                    tid = int(parts[0])
                    cid = int(parts[1])
                    cx = float(parts[2])
                    cy = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])

                    # Basic sanity
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        continue
                    if self.minBoxSize > 0 and w < self.minBoxSize and h < self.minBoxSize:
                        continue

                    boxes.append([cx, cy, w, h])
                    labels.append(cid)
                    trackIds.append(tid)

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(trackIds, dtype=torch.int64),
            )
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0,), dtype=torch.int64),
        )

    # ------------------------------------------------------------------
    #  Image loading
    # ------------------------------------------------------------------
    def _loadImage(self, imgPath: Path) -> Image.Image:
        with Image.open(imgPath) as img:
            img.load()
            return img.convert("RGB")

    # ------------------------------------------------------------------
    #  Coordinate conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _cxcywhToXyxy(
        boxes: torch.Tensor, imgW: int, imgH: int
    ) -> torch.Tensor:
        """Normalised cxcywh → absolute xyxy."""
        cx, cy, w, h = boxes.unbind(-1)
        cx, w = cx * imgW, w * imgW
        cy, h = cy * imgH, h * imgH
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------
    def _sampleIndices(self) -> List[int]:
        """Sample frame indices according to the configured strategy."""
        if self.samplingStrategy in ("random_walk", "mixed"):
            fn = SAMPLING_STRATEGIES[self.samplingStrategy]
            return fn(self.numFrames, self.totalFrames, self.maxFrameOffset)
        elif self.samplingStrategy == "exponential_stride":
            return _sampleExponentialStride(
                self.numFrames, self.totalFrames, minGap=0, maxGap=self.maxFrameOffset
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.samplingStrategy}")

    def _sampleIndicesVal(self) -> List[int]:
        """Deterministic uniform sampling for validation."""
        if self.totalFrames <= self.numFrames:
            indices = list(range(self.totalFrames))
            while len(indices) < self.numFrames:
                indices.append(indices[-1])
            return indices
        step = (self.totalFrames - 1) / (self.numFrames - 1)
        return [min(int(i * step), self.totalFrames - 1) for i in range(self.numFrames)]

    # ------------------------------------------------------------------
    #  Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._epochLength

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Return a clip of ``numFrames`` frames with annotations.

        The ``idx`` is **ignored** for training (we always do random
        sampling from the full sequence).  For validation, we use
        deterministic sampling seeded by ``idx`` to ensure reproducibility.
        """
        if self.totalFrames == 0:
            raise RuntimeError(
                f"[VideoDataset] No frames found in {self.imagesDir}. "
                "Cannot produce samples from an empty dataset."
            )

        if self.imageSet == "train":
            frameIndices = self._sampleIndices()
        else:
            # Deterministic per-sample: seed from idx so each val sample
            # covers a different part of the video.
            rng = random.Random(idx)
            start = rng.randint(0, max(0, self.totalFrames - 1))
            frameIndices = []
            cur = start
            for _ in range(self.numFrames):
                frameIndices.append(min(cur, self.totalFrames - 1))
                cur += rng.randint(0, self.maxFrameOffset)

        images = []
        targets = []

        for clipIdx, fIdx in enumerate(frameIndices):
            imgPath = self.framePaths[fIdx]
            img = self._loadImage(imgPath)
            imgW, imgH = img.size

            labelPath = self._getLabelPath(imgPath)
            boxes, labels, trackIds = self._loadLabels(labelPath)
            numBoxes = len(boxes)

            # Area (in absolute pixels) from normalised cxcywh
            if numBoxes > 0:
                area = boxes[:, 2] * boxes[:, 3] * imgW * imgH
            else:
                area = torch.zeros((0,), dtype=torch.float32)

            target: Dict[str, Any] = {
                "boxes": boxes,        # cxcywh normalised — converted below
                "labels": labels,
                "trackIds": trackIds,
                "iscrowd": torch.zeros((numBoxes,), dtype=torch.int64),
                "area": area,
                "frameIdx": torch.tensor([clipIdx]),
                "origSize": torch.as_tensor([imgH, imgW]),
                "size": torch.as_tensor([imgH, imgW]),
                "imageId": torch.tensor([fIdx]),
                "seqId": "video",
                "seqFrameIdx": fIdx,
            }

            # Convert normalised cxcywh → absolute xyxy for DETR transforms
            if numBoxes > 0:
                target["boxes"] = self._cxcywhToXyxy(boxes, imgW, imgH)

            # Apply transforms (resize, flip, normalise, …)
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            images.append(img)
            targets.append(target)

        return images, targets


# ======================================================================
#  Transforms  (reuse the same recipe as other datasets)
# ======================================================================

def makeVideoDatasetTransforms(imageSet: str, maxSize: int = 800):
    """
    Create DETR-compatible transforms for the VideoDataset.

    A ``SmartSquareCrop`` is applied first to crop wide (e.g. 16:9) images
    toward a 1:1 aspect ratio without losing any annotated objects.  The
    remaining pipeline is identical to ``makeVideoTransforms`` /
    ``makeTaoTransforms`` so that datasets are interchangeable.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if imageSet == "train":
        return T.Compose([
            #T.SmartSquareCrop(margin=0.15, randomise_pos=False),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.RandomResize(scales, max_size=maxSize),
            #T.RandomSelect(
            #    T.Compose([
            #        T.RandomResize([400, 500, 600]),
            #        T.RandomSizeCrop(384, 600),
            #        T.RandomResize(scales, max_size=maxSize),
            #    ]),
            #),
            normalize,
            #T.RandomErasing(p=0.1),
        ])

    if imageSet == "val":
        return T.Compose([
            #T.SmartSquareCrop(margin=0.15, randomise_pos=False),
            T.RandomResize([800], max_size=maxSize),
            normalize,
        ])

    raise ValueError(f"Unknown image set: {imageSet}")


# ======================================================================
#  Collate  — reuse videoCollateFn (identical output format)
# ======================================================================

def videoDatasetCollateFn(
    batch: List[Tuple],
) -> Tuple[List[Any], List[List[Dict]]]:
    """
    Collate function for ``VideoDataset`` batches.

    Output format is identical to ``videoCollateFn`` /
    ``taoCollateFn``, so we simply delegate.
    """
    from vidDetr.datasets.simulated_video_dataset import videoCollateFn
    return videoCollateFn(batch)


# ======================================================================
#  Builder
# ======================================================================

def buildVideoDatasetFromArgs(
    args,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Build train and (optionally) validation ``VideoDataset`` instances.

    Expected ``args`` attributes:
        videoDataRoot       : str  — root containing ``train/`` and ``val/``
        numFrames           : int
        maxSize             : int
        numClasses          : int
        batchSize           : int
        videoTrainBatches   : int  — batches per training epoch  (default 1000)
        videoValBatches     : int  — batches per validation epoch (default 50)
        videoMaxFrameOffset : int  — max gap between consecutive frames (default 30)
        videoSamplingStrategy : str — sampling strategy name
        minBoxSize          : float
        mergeTrainVal       : bool

    Returns:
        (trainDataset, valDataset).  ``valDataset`` is ``None`` when the
        val split is empty or ``mergeTrainVal`` is ``True``.
    """
    root = Path(args.videoDataRoot)
    trainRoot = root / "train"
    valRoot = root / "val"

    maxSize = getattr(args, "maxSize", 800)
    minBoxSize = getattr(args, "minBoxSize", 0.0)
    mergeTrainVal = getattr(args, "mergeTrainVal", False)

    trainBatches = getattr(args, "videoTrainBatches", 1000)
    valBatches = getattr(args, "videoValBatches", 50)
    maxOffset = getattr(args, "videoMaxFrameOffset", 30)
    strategy = getattr(args, "videoSamplingStrategy", "mixed")
    batchSize = getattr(args, "batchSize", 32)

    # Load class names from dataConfig if available.
    classNames = None
    dataConfigPath = getattr(args, "dataConfig", None)
    if dataConfigPath and Path(dataConfigPath).exists():
        with open(dataConfigPath, "r") as f:
            dataConfig = yaml.safe_load(f)
        classNames = list(dataConfig.get("names", {}).values())

    trainTransforms = makeVideoDatasetTransforms("train", maxSize=maxSize)
    trainDataset = VideoDataset(
        dataRoot=str(trainRoot),
        numFrames=args.numFrames,
        transforms=trainTransforms,
        imageSet="train",
        batchesPerEpoch=trainBatches,
        batchSize=batchSize,
        maxFrameOffset=maxOffset,
        samplingStrategy=strategy,
        classNames=classNames,
        numClasses=args.numClasses,
        minBoxSize=minBoxSize,
    )

    if mergeTrainVal:
        # Check if val has any frames before trying to merge.
        valImagesDir = valRoot / "images"
        if valImagesDir.exists() and any(valImagesDir.iterdir()):
            valAsTrainDataset = VideoDataset(
                dataRoot=str(valRoot),
                numFrames=args.numFrames,
                transforms=trainTransforms,
                imageSet="train",
                batchesPerEpoch=valBatches,
                batchSize=batchSize,
                maxFrameOffset=maxOffset,
                samplingStrategy=strategy,
                classNames=classNames,
                numClasses=args.numClasses,
                minBoxSize=minBoxSize,
            )
            merged = torch.utils.data.ConcatDataset([trainDataset, valAsTrainDataset])
            print(
                f"[buildVideoDataset] mergeTrainVal: {len(trainDataset)} + "
                f"{len(valAsTrainDataset)} = {len(merged)} samples"
            )
            return merged, None
        else:
            print("[buildVideoDataset] mergeTrainVal: val is empty, using train only")
            return trainDataset, None

    # Build val dataset — gracefully handle empty val dir.
    valImagesDir = valRoot / "images"
    if not valImagesDir.exists():
        print("[buildVideoDataset] val/images/ not found — skipping validation")
        return trainDataset, None

    valHasFrames = any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in valImagesDir.iterdir()
    ) if valImagesDir.exists() else False

    if not valHasFrames:
        print("[buildVideoDataset] val/images/ is empty — skipping validation")
        return trainDataset, None

    valTransforms = makeVideoDatasetTransforms("val", maxSize=maxSize)
    valDataset = VideoDataset(
        dataRoot=str(valRoot),
        numFrames=args.numFrames,
        transforms=valTransforms,
        imageSet="val",
        batchesPerEpoch=valBatches,
        batchSize=batchSize,
        maxFrameOffset=maxOffset,
        samplingStrategy=strategy,  # val uses deterministic sampling anyway
        classNames=classNames,
        numClasses=args.numClasses,
        minBoxSize=minBoxSize,
    )

    return trainDataset, valDataset
