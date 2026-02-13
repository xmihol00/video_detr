# Copyright (c) 2026. All Rights Reserved.
"""
TAO Dataset for VideoDETR.

This module implements a dataset loader for the TAO (Tracking Any Object)
benchmark. TAO provides large-scale video annotations with:
- COCO-style bounding box annotations (x, y, w, h in pixels)
- Track IDs linking detections across frames
- 1230 object categories from LVIS taxonomy
- Sparse annotations (not every frame is annotated)

The dataset implements a clever multi-window sampling strategy:
each long video is split into multiple overlapping windows so that
all parts of the video are covered during a single epoch. Within each
window, frames are sampled with variable temporal stride (larger strides
are chosen with decreasing probability) to expose the model to diverse
motion patterns.

Dataset structure expected:
    <dataRoot>/
    ├── annotations/
    │   ├── train.json
    │   └── validation.json
    └── frames/
        └── train/  (or val/)
            └── <dataset>/
                └── <video_name>/
                    ├── frame0000.jpg
                    ├── frame0001.jpg
                    └── ...
"""

import sys
from pathlib import Path

# Add parent directory to path for imports - must be before other imports
_parentDir = Path(__file__).resolve().parent.parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets import transforms as T


# ---------------------------------------------------------------------------
# Cache version — bump when the on-disk format of the index cache changes.
# ---------------------------------------------------------------------------
CACHE_VERSION = "1.1"


class TaoDataset(Dataset):
    """
    Dataset for loading TAO video sequences with tracking annotations.

    Each sample consists of ``numFrames`` frames from a single video, together
    with bounding-box annotations and track IDs.  Long videos are broken into
    overlapping *windows* so that the full dataset is covered every epoch.

    Args:
        dataRoot:           Root directory of the TAO dataset (contains
                            ``annotations/`` and ``frames/`` subdirs).
        annotationFile:     Path to the JSON annotation file (e.g.
                            ``train.json``).
        numFrames:          Number of frames to sample per clip.
        transforms:         Optional DETR-style transforms applied per-frame.
        imageSet:           ``'train'`` or ``'val'``.
        minFrameGap:        Minimum stride between sampled frames.
        maxFrameGap:        Maximum stride between sampled frames.
        windowOverlap:      Fraction of overlap between consecutive windows
                            (0.0 – 1.0).  Larger values ⇒ more windows per
                            video ⇒ more samples per epoch.
        useCache:           Cache the parsed annotation index to disk.
        maxCategoriesUsed:  If set, keep only the N most frequent categories
                            (by annotation count).  ``None`` keeps all.
    """

    def __init__(
        self,
        dataRoot: str,
        annotationFile: str,
        numFrames: int = 5,
        transforms: Optional[Any] = None,
        imageSet: str = "train",
        minFrameGap: int = 1,
        maxFrameGap: int = 10,
        windowOverlap: float = 0.5,
        useCache: bool = True,
        maxCategoriesUsed: Optional[int] = None,
    ):
        super().__init__()

        self.dataRoot = Path(dataRoot)
        self.annotationFile = Path(annotationFile)
        self.numFrames = numFrames
        self.transforms = transforms
        self.imageSet = imageSet
        self.minFrameGap = max(1, minFrameGap)
        self.maxFrameGap = max(self.minFrameGap, maxFrameGap)
        self.windowOverlap = windowOverlap
        self.useCache = useCache
        self.maxCategoriesUsed = maxCategoriesUsed

        # ── Parse annotations ─────────────────────────────────────────
        annData = self._loadAnnotations()

        # Category handling: remap sparse TAO category IDs → contiguous
        self.categories: Dict[int, dict] = {
            c["id"]: c for c in annData["categories"]
        }  # original id → category info

        # Build contiguous category mapping
        self._buildCategoryMapping(annData)

        # ── Build per-video index ──────────────────────────────────────
        # videoId → list of annotated image records (sorted by frame_index)
        self.videoImages: Dict[int, List[dict]] = defaultdict(list)
        imageById: Dict[int, dict] = {}
        for img in annData["images"]:
            imageById[img["id"]] = img
            self.videoImages[img["video_id"]].append(img)

        # Sort each video's images by frame_index
        for vid in self.videoImages:
            self.videoImages[vid].sort(key=lambda x: x["frame_index"])

        # ── Per-image annotations (grouped) ────────────────────────────
        # imageId → list of annotation dicts
        self.imageAnnotations: Dict[int, List[dict]] = defaultdict(list)
        for ann in annData["annotations"]:
            # Skip annotations whose category was pruned
            if ann["category_id"] not in self.catIdToContiguous:
                continue
            self.imageAnnotations[ann["image_id"]].append(ann)

        # Video metadata
        self.videoInfo: Dict[int, dict] = {v["id"]: v for v in annData["videos"]}

        # ── Build windows ──────────────────────────────────────────────
        self.windows = self._buildWindows()

        print(
            f"[TaoDataset] {imageSet}: {len(self.videoImages)} videos, "
            f"{len(self.windows)} windows, "
            f"{self.numClasses} classes, "
            f"{numFrames} frames/clip, "
            f"gap {self.minFrameGap}–{self.maxFrameGap}"
        )

    # ------------------------------------------------------------------
    #  Category mapping
    # ------------------------------------------------------------------
    def _buildCategoryMapping(self, annData: dict) -> None:
        """Build a mapping from original TAO category IDs to contiguous IDs."""
        if self.maxCategoriesUsed is not None:
            # Count annotations per category
            catCounts: Dict[int, int] = defaultdict(int)
            for ann in annData["annotations"]:
                catCounts[ann["category_id"]] += 1
            # Keep top-N
            topCats = sorted(catCounts, key=lambda c: catCounts[c], reverse=True)[
                : self.maxCategoriesUsed
            ]
            topCats.sort()  # deterministic ordering
        else:
            topCats = sorted(self.categories.keys())

        self.catIdToContiguous: Dict[int, int] = {
            origId: idx for idx, origId in enumerate(topCats)
        }
        self.contiguousToCatId: Dict[int, int] = {
            v: k for k, v in self.catIdToContiguous.items()
        }
        self.numClasses: int = len(self.catIdToContiguous)

    # ------------------------------------------------------------------
    #  Annotation loading (with optional caching)
    # ------------------------------------------------------------------
    def _getCachePath(self) -> Path:
        h = hashlib.md5(str(self.annotationFile.resolve()).encode()).hexdigest()[:8]
        return self.annotationFile.parent / f".tao_cache_{h}.json"

    def _loadAnnotations(self) -> dict:
        """Load the raw TAO JSON. No caching here — it's fast enough via json."""
        assert self.annotationFile.exists(), (
            f"Annotation file not found: {self.annotationFile}"
        )
        print(f"[TaoDataset] Loading annotations from {self.annotationFile} ...")
        with open(self.annotationFile, "r") as f:
            data = json.load(f)
        print(
            f"[TaoDataset]   {len(data['videos'])} videos, "
            f"{len(data['images'])} images, "
            f"{len(data['annotations'])} annotations, "
            f"{len(data['categories'])} categories"
        )
        return data

    # ------------------------------------------------------------------
    #  Window construction
    # ------------------------------------------------------------------
    def _buildWindows(self) -> List[Tuple[int, List[dict]]]:
        """
        Build a list of (videoId, imageSubset) windows.

        Each video is sliced into overlapping windows whose length (in
        annotated frames) is ``numFrames * maxFrameGap``.  Within each
        window the actual frames will be sampled at runtime with a random
        stride.  This guarantees that every annotated frame participates
        in at least one window, and long videos contribute proportionally
        more samples per epoch.
        """
        windows: List[Tuple[int, List[dict]]] = []

        for videoId, images in self.videoImages.items():
            # Only keep images that have at least one (mapped) annotation
            annotatedImages = [
                img for img in images if len(self.imageAnnotations[img["id"]]) > 0
            ]
            if len(annotatedImages) < self.numFrames:
                # Not enough annotated frames for a single clip
                # Still include if we have *any* annotated frames — pad later
                if len(annotatedImages) > 0:
                    windows.append((videoId, annotatedImages))
                continue

            # Window length in number of annotated images
            windowLen = max(
                self.numFrames,
                self.numFrames * 2,  # allow room for stride variety
            )
            stepSize = max(
                1,
                int(windowLen * (1 - self.windowOverlap)),
            )

            numAnnotated = len(annotatedImages)
            if numAnnotated <= windowLen:
                # Whole video fits in one window
                windows.append((videoId, annotatedImages))
            else:
                start = 0
                while start < numAnnotated:
                    end = min(start + windowLen, numAnnotated)
                    windows.append(
                        (videoId, annotatedImages[start:end])
                    )
                    if end == numAnnotated:
                        break
                    start += stepSize

        return windows

    # ------------------------------------------------------------------
    #  Frame sampling
    # ------------------------------------------------------------------
    def _sampleFrameIndices(
        self, annotatedImages: List[dict]
    ) -> List[dict]:
        """
        Sample ``numFrames`` images from the window.

        Strategy:
        - Pick a random stride (gap) drawn with geometrically decreasing
          probability: P(gap=g) ∝ 0.5^(g - minGap).  This makes small
          gaps (fast motion, adjacent frames) more common while still
          exposing the model to large temporal jumps.
        - Starting position is chosen uniformly among positions that fit
          the chosen stride.
        - During validation, use a deterministic uniform spacing.

        If fewer annotated images are available than ``numFrames``, frames
        are repeated (with different ``frameIdx`` values) to keep the
        tensor shapes consistent.
        """
        numAvailable = len(annotatedImages)

        if numAvailable <= self.numFrames:
            # Not enough frames — repeat/pad
            selected = list(annotatedImages)
            while len(selected) < self.numFrames:
                selected.append(random.choice(annotatedImages))
            selected.sort(key=lambda x: x["frame_index"])
            return selected[: self.numFrames]

        if self.imageSet == "train":
            # --- random stride sampling ---
            # Choose a stride with geometrically decaying probability
            maxPossibleGap = min(
                self.maxFrameGap,
                (numAvailable - 1) // (self.numFrames - 1),
            )
            gap = self._sampleGap(self.minFrameGap, maxPossibleGap)

            # Span in *indices* needed
            span = (self.numFrames - 1) * gap
            maxStart = numAvailable - 1 - span
            if maxStart < 0:
                # Fallback: uniform spacing
                gap = (numAvailable - 1) // (self.numFrames - 1)
                span = (self.numFrames - 1) * gap
                maxStart = numAvailable - 1 - span

            startIdx = random.randint(0, max(0, maxStart))
            indices = [startIdx + i * gap for i in range(self.numFrames)]
            return [annotatedImages[i] for i in indices]

        else:
            # --- deterministic uniform spacing ---
            step = (numAvailable - 1) / (self.numFrames - 1) if self.numFrames > 1 else 0
            indices = [
                min(int(i * step), numAvailable - 1) for i in range(self.numFrames)
            ]
            return [annotatedImages[i] for i in indices]

    @staticmethod
    def _sampleGap(minGap: int, maxGap: int) -> int:
        """
        Sample a frame gap with geometrically decreasing probability.

        P(gap = g) ∝ 0.5^(g - minGap)  for g in [minGap, maxGap].

        Returns a gap in [minGap, maxGap].
        """
        if maxGap <= minGap:
            return minGap
        weights = [0.5 ** (g - minGap) for g in range(minGap, maxGap + 1)]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for g, w in zip(range(minGap, maxGap + 1), weights):
            cumulative += w
            if r <= cumulative:
                return g
        return maxGap

    # ------------------------------------------------------------------
    #  Image / annotation loading
    # ------------------------------------------------------------------
    def _loadImage(self, imageRecord: dict) -> Image.Image:
        """Load a single frame from disk."""
        fileName = imageRecord["file_name"]
        # Prepend 'frames' as instructed
        imgPath = self.dataRoot / "frames" / fileName

        if not imgPath.exists():
            raise FileNotFoundError(
                f"Image not found: {imgPath}  (file_name={fileName})"
            )

        with Image.open(imgPath) as img:
            img.load()
            return img.convert("RGB")

    def _loadAnnotationsForImage(
        self, imageId: int, imgWidth: int, imgHeight: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """
        Load annotations for a single image.

        Returns:
            boxes_xyxy : [N, 4] absolute xyxy
            labels     : [N]    contiguous class indices
            trackIds   : [N]    track identity
            iscrowd    : [N]    crowd flag
            segments   : list of N polygon lists (each polygon is a flat
                         list of x,y coordinates) — kept for eval drawing
        """
        anns = self.imageAnnotations.get(imageId, [])

        boxes, labels, trackIds, iscrowd = [], [], [], []
        segments: List = []

        for ann in anns:
            catId = ann["category_id"]
            if catId not in self.catIdToContiguous:
                continue

            # COCO bbox format: [x, y, w, h] in pixels
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Convert to xyxy (absolute pixels)
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(imgWidth), x + w)
            y2 = min(float(imgHeight), y + h)

            boxes.append([x1, y1, x2, y2])
            labels.append(self.catIdToContiguous[catId])
            trackIds.append(ann["track_id"])
            iscrowd.append(ann.get("iscrowd", 0))
            segments.append(ann.get("segmentation", []))

        if boxes:
            boxesTensor = torch.tensor(boxes, dtype=torch.float32)
            labelsTensor = torch.tensor(labels, dtype=torch.int64)
            trackIdsTensor = torch.tensor(trackIds, dtype=torch.int64)
            iscrowdTensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxesTensor = torch.zeros((0, 4), dtype=torch.float32)
            labelsTensor = torch.zeros((0,), dtype=torch.int64)
            trackIdsTensor = torch.zeros((0,), dtype=torch.int64)
            iscrowdTensor = torch.zeros((0,), dtype=torch.int64)

        return boxesTensor, labelsTensor, trackIdsTensor, iscrowdTensor, segments

    # ------------------------------------------------------------------
    #  Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Get a clip of ``numFrames`` frames with annotations.

        Returns:
            images  : list of N image tensors [3, H, W]
            targets : list of N target dicts, each with:
                - boxes      : [M, 4] xyxy absolute (pre-transform) or
                               cxcywh normalised (post-transform)
                - labels     : [M] class indices (contiguous)
                - trackIds   : [M] track identity integers
                - iscrowd    : [M] crowd flags
                - area       : [M] box areas
                - frameIdx   : scalar tensor (clip-local index 0…N-1)
                - origSize   : [H, W]
                - size        : [H, W]
                - imageId    : scalar tensor
                - seqId      : string (video name)
        """
        videoId, windowImages = self.windows[idx]
        selectedImages = self._sampleFrameIndices(windowImages)

        images = []
        targets = []

        for clipIdx, imgRecord in enumerate(selectedImages):
            img = self._loadImage(imgRecord)
            imgWidth, imgHeight = img.size

            boxes, labels, trackIds, iscrowd, _segments = (
                self._loadAnnotationsForImage(imgRecord["id"], imgWidth, imgHeight)
            )

            numBoxes = len(boxes)
            if numBoxes > 0:
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            else:
                area = torch.zeros((0,), dtype=torch.float32)

            target: Dict[str, Any] = {
                "boxes": boxes,       # xyxy absolute — transforms handle the rest
                "labels": labels,
                "trackIds": trackIds,
                "iscrowd": iscrowd,
                "area": area,
                "frameIdx": torch.tensor([clipIdx]),
                "origSize": torch.as_tensor([imgHeight, imgWidth]),
                "size": torch.as_tensor([imgHeight, imgWidth]),
                "imageId": torch.tensor([imgRecord["id"]]),
                "seqId": self.videoInfo[videoId]["name"],
                "seqFrameIdx": imgRecord["frame_index"],
            }

            # Apply DETR-style transforms (resize, flip, normalise, …)
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            images.append(img)
            targets.append(target)

        return images, targets


# ======================================================================
#  Transforms  (shared with VideoSequenceDataset)
# ======================================================================

def makeTaoTransforms(imageSet: str, maxSize: int = 800):
    """
    Create DETR-compatible transforms for the TAO dataset.

    Same recipe used by ``makeVideoTransforms`` in ``video_dataset.py``
    so that the two datasets can be swapped transparently.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if imageSet == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=maxSize),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=maxSize),
                ]),
            ),
            normalize,
        ])

    if imageSet == "val":
        return T.Compose([
            T.RandomResize([800], max_size=maxSize),
            normalize,
        ])

    raise ValueError(f"Unknown image set: {imageSet}")


# ======================================================================
#  Collate — reuse videoCollateFn since output format is identical
# ======================================================================

def taoCollateFn(batch: List[Tuple]) -> Tuple[List[Any], List[List[Dict]]]:
    """
    Collate function for TAO batches.

    The per-sample format is identical to ``VideoSequenceDataset``, so we
    simply delegate to ``videoCollateFn``.
    """
    from vidDetr.datasets.video_dataset import videoCollateFn
    return videoCollateFn(batch)


# ======================================================================
#  Builder
# ======================================================================

def buildTaoDataset(args) -> Tuple[Dataset, Dataset]:
    """
    Build train and validation TAO datasets from ``args``.

    Expected ``args`` attributes:
        taoDataRoot       : str   – root of the TAO dataset
        numFrames         : int
        minFrameGap       : int
        maxFrameGap       : int
        maxSize           : int   – maximum image dimension
        numClasses        : int   – will be *overwritten* by the actual count
        taoMaxCategories  : int | None – keep only top-N categories

    Returns:
        (trainDataset, valDataset)
    """
    root = Path(args.taoDataRoot)
    annDir = root / "annotations"
    trainJson = annDir / "train.json"
    valJson = annDir / "validation.json"

    maxCat = getattr(args, "taoMaxCategories", None)
    maxSize = getattr(args, "maxSize", 800)

    datasetTrain = TaoDataset(
        dataRoot=str(root),
        annotationFile=str(trainJson),
        numFrames=args.numFrames,
        transforms=makeTaoTransforms("train", maxSize=maxSize),
        imageSet="train",
        minFrameGap=getattr(args, "minFrameGap", 1),
        maxFrameGap=getattr(args, "maxFrameGap", 10),
        windowOverlap=getattr(args, "taoWindowOverlap", 0.5),
        maxCategoriesUsed=maxCat,
    )

    datasetVal = TaoDataset(
        dataRoot=str(root),
        annotationFile=str(valJson),
        numFrames=args.numFrames,
        transforms=makeTaoTransforms("val", maxSize=maxSize),
        imageSet="val",
        minFrameGap=getattr(args, "minFrameGap", 1),
        maxFrameGap=getattr(args, "maxFrameGap", 10),
        windowOverlap=0.0,  # no overlap for validation
        maxCategoriesUsed=maxCat,
    )

    # Synchronise numClasses so the model and criterion agree
    assert datasetTrain.numClasses == datasetVal.numClasses, (
        f"Train ({datasetTrain.numClasses}) and val ({datasetVal.numClasses}) "
        f"have different category counts — check maxCategoriesUsed"
    )
    args.numClasses = datasetTrain.numClasses
    print(f"[buildTaoDataset] numClasses overridden to {args.numClasses}")

    return datasetTrain, datasetVal
