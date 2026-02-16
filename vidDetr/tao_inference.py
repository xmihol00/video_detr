#!/usr/bin/env python3
# Copyright (c) 2026. All Rights Reserved.
"""
TAO Inference & Visualisation Script for VideoDETR.

Runs inference on randomly selected videos from the TAO validation set,
draws ground-truth (dashed, thin) and predicted (solid, thick) bounding
boxes onto every frame, associates detections across frames using tracking
embeddings, and writes the results as MP4 video files.

Designed for headless GPU servers — no GUI required.

Output structure::

    gt_vs_pred/
    ├── <video_name_1>/
    │   ├── video.mp4
    │   └── info.txt
    ├── <video_name_2>/
    │   ├── video.mp4
    │   └── info.txt
    └── ...

Usage:
    python vidDetr/tao_inference.py \\
        --modelPath tao_weights/video_detr_best.pth \\
        --taoDataRoot /path/to/tao/dataset \\
        --numVideos 5 \\
        --confidence 0.4
"""

import time

import safe_gpu
while True:
    try:
        safe_gpu.claim_gpus()
        break
    except:
        print("Waiting for free GPU")
        time.sleep(5)
        pass

import sys
from pathlib import Path

# Add parent directory to path for imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import argparse
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from util.misc import nested_tensor_from_tensor_list
from datasets import transforms as T
from vidDetr.models import buildVideoDETR

# ImageNet normalisation (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================================
# Argument parsing
# =========================================================================
def getArgsParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "TAO VideoDETR inference & video generation", add_help=False
    )

    # Required
    parser.add_argument(
        "--modelPath",
        default="vidDetr_weights_1/checkpoint_latest.pth",
        type=str,
        help="Path to a VideoDETR checkpoint (.pth)",
    )
    parser.add_argument(
        "--taoDataRoot",
        default="/mnt/matylda5/xmihol00/tao/dataset/",
        type=str,
        help="Root directory of TAO dataset (contains annotations/ and frames/)",
    )

    # What to process
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        choices=["train", "validation"],
        help="Which annotation split to use",
    )
    parser.add_argument(
        "--numVideos",
        default=10,
        type=int,
        help="Number of videos to randomly select and process",
    )

    # Inference behaviour
    parser.add_argument(
        "--confidence",
        default=0.75,
        type=float,
        help="Minimum confidence threshold for predictions",
    )
    parser.add_argument(
        "--nmsThreshold",
        default=0.75,
        type=float,
        help="IoU threshold for per-frame NMS",
    )
    parser.add_argument(
        "--trackingThreshold",
        default=0.5,
        type=float,
        help="Cosine-similarity threshold for cross-frame track association",
    )
    parser.add_argument(
        "--maxSize",
        default=384,
        type=int,
        help="Maximum image size for inference (must match training)",
    )

    # Output
    parser.add_argument(
        "--outputDir",
        default="gt_vs_pred",
        type=str,
        help="Root output directory for generated videos",
    )
    parser.add_argument(
        "--fps",
        default=6,
        type=int,
        help="Frame rate for output videos",
    )
    parser.add_argument(
        "--saveFrames",
        action="store_true",
        default=True,
        help="Save output as individual JPG frames instead of a video",
    )
    parser.add_argument(
        "--jpgQuality",
        default=85,
        type=int,
        help="JPEG quality (0-100) when --saveFrames is used",
    )

    # Misc
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for video selection",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device for inference",
    )

    return parser


# =========================================================================
# TAO JSON helpers
# =========================================================================
class TaoAnnotations:
    """
    Lightweight read-only index over a TAO annotation JSON.

    Builds the same category mapping (original → contiguous) that the
    training pipeline uses so that predicted class IDs match GT class IDs.
    """

    def __init__(self, jsonPath: str, taoMaxCategories: Optional[int] = None):
        print(f"[TAO] Loading annotations from {jsonPath} …")
        with open(jsonPath, "r") as f:
            data = json.load(f)

        self.videos: Dict[int, dict] = {v["id"]: v for v in data["videos"]}
        self.images: Dict[int, dict] = {img["id"]: img for img in data["images"]}
        self.categories: Dict[int, dict] = {c["id"]: c for c in data["categories"]}
        self.tracks: Dict[int, dict] = {t["id"]: t for t in data["tracks"]}

        # ── Per-video image lists (sorted by frame_index) ────────────
        self.videoImages: Dict[int, List[dict]] = defaultdict(list)
        for img in data["images"]:
            self.videoImages[img["video_id"]].append(img)
        for vid in self.videoImages:
            self.videoImages[vid].sort(key=lambda x: x["frame_index"])

        # ── Per-image annotation lists ────────────────────────────────
        self.imageAnnotations: Dict[int, List[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            self.imageAnnotations[ann["image_id"]].append(ann)

        # ── Category mapping: original TAO id → contiguous 0..N-1 ────
        # Must match what TaoDataset._buildCategoryMapping does.
        # When taoMaxCategories is set, keep only the N most frequent
        # categories (same logic as the training dataset builder).
        if taoMaxCategories is not None:
            catCounts: Dict[int, int] = defaultdict(int)
            for ann in data["annotations"]:
                catCounts[ann["category_id"]] += 1
            topCats = sorted(
                catCounts, key=lambda c: catCounts[c], reverse=True
            )[:taoMaxCategories]
            topCats.sort()  # deterministic ordering
            sortedCatIds = topCats
        else:
            sortedCatIds = sorted(self.categories.keys())
        self.catIdToContiguous: Dict[int, int] = {
            origId: idx for idx, origId in enumerate(sortedCatIds)
        }
        self.contiguousToCatId: Dict[int, int] = {
            v: k for k, v in self.catIdToContiguous.items()
        }
        self.numClasses = len(self.catIdToContiguous)

        print(
            f"[TAO]   {len(self.videos)} videos, "
            f"{len(self.images)} images, "
            f"{sum(len(v) for v in self.imageAnnotations.values())} annotations, "
            f"{self.numClasses} categories"
        )

    def getCategoryName(self, contiguousId: int) -> str:
        origId = self.contiguousToCatId.get(contiguousId, -1)
        cat = self.categories.get(origId, {})
        return cat.get("name", f"cls{contiguousId}")

    def getGtForImage(
        self, imageId: int, imgW: int, imgH: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return GT boxes (xyxy absolute), contiguous labels, and track IDs
        for a single image.
        """
        anns = self.imageAnnotations.get(imageId, [])
        boxes, labels, trackIds = [], [], []

        for ann in anns:
            catId = ann["category_id"]
            if catId not in self.catIdToContiguous:
                continue
            x, y, w, h = ann["bbox"]  # COCO format
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(imgW), x + w)
            y2 = min(float(imgH), y + h)
            boxes.append([x1, y1, x2, y2])
            labels.append(self.catIdToContiguous[catId])
            trackIds.append(ann["track_id"])

        if boxes:
            return (
                np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(trackIds, dtype=np.int64),
            )
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )


# =========================================================================
# Colour palette
# =========================================================================
def _buildColourPalette(n: int) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colours via HSV spacing."""
    colours = []
    for i in range(max(n, 1)):
        hue = int(180 * i / max(n, 1))
        hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colours.append(
            (int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2]))
        )
    return colours


def _trackColour(trackId: int) -> Tuple[int, int, int]:
    """Deterministic colour per track ID for consistent visualisation."""
    rng = random.Random(trackId * 7 + 13)
    return (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))


def resizeForOutput(
    bgrImage: np.ndarray, maxSize: int
) -> Tuple[np.ndarray, float, float]:
    """
    Resize *bgrImage* so that its longest side equals *maxSize* (preserving
    aspect ratio).  Returns ``(resizedImage, scaleX, scaleY)``.
    """
    h, w = bgrImage.shape[:2]
    if max(h, w) <= maxSize:
        return bgrImage, 1.0, 1.0
    if h >= w:
        newH = maxSize
        newW = int(round(w * maxSize / h))
    else:
        newW = maxSize
        newH = int(round(h * maxSize / w))
    resized = cv2.resize(bgrImage, (newW, newH), interpolation=cv2.INTER_LINEAR)
    return resized, newW / w, newH / h


def scaleBoxes(
    boxes: np.ndarray, scaleX: float, scaleY: float
) -> np.ndarray:
    """Scale absolute xyxy boxes by (scaleX, scaleY)."""
    if len(boxes) == 0:
        return boxes
    scaled = boxes.copy()
    scaled[:, 0] *= scaleX
    scaled[:, 1] *= scaleY
    scaled[:, 2] *= scaleX
    scaled[:, 3] *= scaleY
    return scaled


# =========================================================================
# Preprocessing (must mirror validation transforms)
# =========================================================================
def makeInferenceTransform(maxSize: int = 800):
    return T.Compose([
        T.RandomResize([800], max_size=maxSize),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocessFrame(
    bgrImage: np.ndarray,
    transform: Any,
) -> Tuple[torch.Tensor, dict]:
    rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(rgbImage)
    imgW, imgH = pilImage.size
    dummyTarget = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "origSize": torch.tensor([imgH, imgW]),
        "size": torch.tensor([imgH, imgW]),
    }
    imgTensor, target = transform(pilImage, dummyTarget)
    return imgTensor, target


# =========================================================================
# NMS
# =========================================================================
def nmsPerFrame(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iouThreshold: float = 0.5,
) -> torch.Tensor:
    from torchvision.ops import batched_nms

    keep = batched_nms(boxes, scores, labels, iou_threshold=iouThreshold)
    mask = torch.zeros(len(scores), dtype=torch.bool)
    mask[keep] = True
    return mask


# =========================================================================
# Model loading  (identical logic to inference.py)
# =========================================================================
def loadModel(
    modelPath: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Any]:
    print(f"[TAO-Infer] Loading checkpoint from {modelPath} …")
    checkpoint = torch.load(modelPath, map_location="cpu", weights_only=False)

    args = checkpoint.get("args", None)
    if args is None:
        raise RuntimeError(
            "Checkpoint does not contain 'args'. "
            "Cannot reconstruct model architecture."
        )

    # Ensure compatibility attributes exist
    args.device = str(device)
    for newAttr, oldAttr, default in [
        ("lr_backbone", "lrBackbone", 1e-5),
        ("position_embedding", "positionEmbedding", "sine"),
        ("hidden_dim", "hiddenDim", 256),
        ("enc_layers", "encLayers", 6),
        ("dec_layers", "decLayers", 6),
        ("dim_feedforward", "dimFeedforward", 2048),
        ("pre_norm", "preNorm", False),
    ]:
        if not hasattr(args, newAttr):
            setattr(args, newAttr, getattr(args, oldAttr, default))
    if not hasattr(args, "masks"):
        args.masks = False

    model, _criterion, _postprocessors = buildVideoDETR(args)
    model.to(device)

    modelStateDict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(modelStateDict, strict=False)
    if missing:
        print(f"  ⚠  Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    model.eval()
    print(
        f"[TAO-Infer] Model loaded – numFrames={model.numFrames}, "
        f"queriesPerFrame={model.queriesPerFrame}, "
        f"numClasses={model.numClasses}"
    )
    return model, args


# =========================================================================
# Sliding-window inference on a full TAO video
# =========================================================================
@torch.no_grad()
def inferVideo(
    model: torch.nn.Module,
    bgrFrames: List[np.ndarray],
    transform: Any,
    device: torch.device,
    confidence: float = 0.4,
    nmsThreshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Run sliding-window inference on *all* frames of a video.

    Returns one result dict per frame with keys:
        boxes      : (K, 4) absolute xyxy  (numpy)
        scores     : (K,)                  (numpy)
        labels     : (K,)                  (numpy)
        embeddings : (K, D)                (numpy)
    """
    numFrames = model.numFrames
    queriesPerFrame = model.queriesPerFrame
    totalFrames = len(bgrFrames)

    if totalFrames == 0:
        return []

    # ── Pre-process all frames ────────────────────────────────────────
    imgTensors: List[torch.Tensor] = []
    origSizes: List[Tuple[int, int]] = []  # (H, W)

    for bgr in bgrFrames:
        h, w = bgr.shape[:2]
        origSizes.append((h, w))
        tensor, _ = preprocessFrame(bgr, transform)
        imgTensors.append(tensor)

    # ── Sliding window ────────────────────────────────────────────────
    perFrameRaw: List[List[Tuple[torch.Tensor, ...]]] = [
        [] for _ in range(totalFrames)
    ]

    stride = max(1, numFrames // 2)
    windowStarts = list(range(0, max(1, totalFrames - numFrames + 1), stride))
    if windowStarts[-1] + numFrames < totalFrames:
        windowStarts.append(totalFrames - numFrames)

    for wStart in windowStarts:
        wEnd = min(wStart + numFrames, totalFrames)
        clipLen = wEnd - wStart

        clipTensors = [imgTensors[i] for i in range(wStart, wEnd)]
        while len(clipTensors) < numFrames:
            clipTensors.append(clipTensors[-1])

        samples = [
            nested_tensor_from_tensor_list([t]).to(device) for t in clipTensors
        ]

        outputs = model(samples)

        predLogits = outputs["pred_logits"]      # [1, Q_total, C+1]
        predBoxes = outputs["pred_boxes"]         # [1, Q_total, 4]
        predTracking = outputs["pred_tracking"]   # [1, Q_total, D]

        for localF in range(clipLen):
            globalF = wStart + localF
            qStart = localF * queriesPerFrame
            qEnd = qStart + queriesPerFrame

            logits = predLogits[0, qStart:qEnd]
            boxesCxcywh = predBoxes[0, qStart:qEnd]
            embeddings = predTracking[0, qStart:qEnd]

            probs = logits.softmax(-1)[:, :-1]
            maxScores, maxLabels = probs.max(-1)

            imgH, imgW = origSizes[globalF]
            cx = boxesCxcywh[:, 0] * imgW
            cy = boxesCxcywh[:, 1] * imgH
            bw = boxesCxcywh[:, 2] * imgW
            bh = boxesCxcywh[:, 3] * imgH
            absBoxes = torch.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1
            )

            perFrameRaw[globalF].append(
                (absBoxes, maxScores, maxLabels, embeddings)
            )

    # ── Merge & NMS per frame ─────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    for fIdx in range(totalFrames):
        chunks = perFrameRaw[fIdx]
        if not chunks:
            results.append({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "embeddings": np.zeros((0, 0), dtype=np.float32),
            })
            continue

        allBoxes = torch.cat([c[0] for c in chunks])
        allScores = torch.cat([c[1] for c in chunks])
        allLabels = torch.cat([c[2] for c in chunks])
        allEmbed = torch.cat([c[3] for c in chunks])

        keep = allScores >= confidence
        allBoxes, allScores, allLabels, allEmbed = (
            allBoxes[keep], allScores[keep], allLabels[keep], allEmbed[keep]
        )

        if len(allScores) > 0:
            keepNms = nmsPerFrame(allBoxes, allScores, allLabels, nmsThreshold)
            allBoxes, allScores, allLabels, allEmbed = (
                allBoxes[keepNms], allScores[keepNms],
                allLabels[keepNms], allEmbed[keepNms],
            )

        results.append({
            "boxes": allBoxes.cpu().numpy(),
            "scores": allScores.cpu().numpy(),
            "labels": allLabels.cpu().numpy(),
            "embeddings": allEmbed.cpu().numpy(),
        })

    return results


# =========================================================================
# Cross-frame track association  (same algorithm as inference.py)
# =========================================================================
def associateTracks(
    results: List[Dict[str, Any]],
    similarityThreshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Greedy cross-frame association via cosine similarity of tracking
    embeddings.  Assigns a consistent ``trackId`` field to each detection.
    """
    nextTrackId = 0
    activeTracks: List[Dict[str, Any]] = []
    emaAlpha = 0.6

    for frameResult in results:
        nDets = len(frameResult["scores"])
        frameResult["trackId"] = np.full(nDets, -1, dtype=np.int64)

        if nDets == 0:
            continue

        embeddings = frameResult["embeddings"]
        if embeddings.ndim < 2 or embeddings.shape[1] == 0:
            for i in range(nDets):
                frameResult["trackId"][i] = nextTrackId
                nextTrackId += 1
            continue

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-8)
        embeddings = embeddings / norms

        assigned = np.zeros(nDets, dtype=bool)

        if activeTracks:
            trackEmbeds = np.stack([t["embedding"] for t in activeTracks])
            sim = embeddings @ trackEmbeds.T

            order = np.argsort(-sim.flatten())
            usedTracks = set()
            for idx in order:
                detIdx = int(idx // len(activeTracks))
                trkIdx = int(idx % len(activeTracks))
                if assigned[detIdx] or trkIdx in usedTracks:
                    continue
                if sim[detIdx, trkIdx] < similarityThreshold:
                    break
                frameResult["trackId"][detIdx] = activeTracks[trkIdx]["id"]
                activeTracks[trkIdx]["embedding"] = (
                    emaAlpha * embeddings[detIdx]
                    + (1 - emaAlpha) * activeTracks[trkIdx]["embedding"]
                )
                norm = np.linalg.norm(activeTracks[trkIdx]["embedding"]).clip(1e-8)
                activeTracks[trkIdx]["embedding"] /= norm
                activeTracks[trkIdx]["age"] = 0
                assigned[detIdx] = True
                usedTracks.add(trkIdx)

        for i in range(nDets):
            if not assigned[i]:
                tid = nextTrackId
                nextTrackId += 1
                frameResult["trackId"][i] = tid
                activeTracks.append({
                    "id": tid,
                    "embedding": embeddings[i].copy(),
                    "age": 0,
                })

        for t in activeTracks:
            t["age"] += 1
        activeTracks = [t for t in activeTracks if t["age"] <= 30]

    return results


# =========================================================================
# Drawing helpers
# =========================================================================
def _drawDashedRect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    colour: Tuple[int, int, int],
    thickness: int = 1,
    dashLen: int = 10,
    gapLen: int = 6,
):
    """Draw a dashed rectangle on *img*."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in edges:
        length = max(abs(ex - sx), abs(ey - sy))
        if length == 0:
            continue
        dx = (ex - sx) / length
        dy = (ey - sy) / length
        drawn = 0
        draw = True
        while drawn < length:
            segLen = dashLen if draw else gapLen
            segEnd = min(drawn + segLen, length)
            if draw:
                p1 = (int(sx + dx * drawn), int(sy + dy * drawn))
                p2 = (int(sx + dx * segEnd), int(sy + dy * segEnd))
                cv2.line(img, p1, p2, colour, thickness, cv2.LINE_AA)
            drawn = segEnd
            draw = not draw


def drawFrame(
    bgrImage: np.ndarray,
    predResult: Dict[str, Any],
    gtBoxes: np.ndarray,
    gtLabels: np.ndarray,
    gtTrackIds: np.ndarray,
    taoAnns: TaoAnnotations,
    palette: List[Tuple[int, int, int]],
    videoName: str,
    frameIndex: int,
    framePos: int,
    totalFrames: int,
    hasAnnotation: bool,
) -> np.ndarray:
    """
    Draw GT (thin dashed) and predictions (thick solid) on a copy of the
    image and return it.
    """
    vis = bgrImage.copy()
    imgH, imgW = vis.shape[:2]
    numPalette = len(palette)

    # ── Ground truth (thin dashed) ────────────────────────────────────
    if len(gtBoxes) > 0:
        for i in range(len(gtBoxes)):
            x1, y1, x2, y2 = gtBoxes[i].astype(int)
            tid = int(gtTrackIds[i])
            colour = _trackColour(tid)
            _drawDashedRect(vis, (x1, y1), (x2, y2), colour,
                            thickness=2, dashLen=8, gapLen=5)
            clsName = taoAnns.getCategoryName(int(gtLabels[i]))
            label = f"GT {clsName} T{tid}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
            )
            cv2.rectangle(
                vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1
            )
            cv2.putText(
                vis, label, (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

    # ── Predictions (thick solid) ─────────────────────────────────────
    boxes = predResult["boxes"]
    scores = predResult["scores"]
    labels = predResult["labels"]
    trackIds = predResult.get("trackId", np.full(len(scores), -1))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        tid = int(trackIds[i])
        colour = _trackColour(tid + 10000)  # offset to avoid GT colour clash
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)

        clsName = taoAnns.getCategoryName(int(labels[i]))
        label = f"{clsName} {scores[i]:.2f} T{tid}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        cv2.rectangle(
            vis, (x1, y2), (x1 + tw + 4, y2 + th + 6), colour, -1
        )
        cv2.putText(
            vis, label, (x1 + 2, y2 + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    # ── HUD ───────────────────────────────────────────────────────────
    annTag = "" if hasAnnotation else " [no GT ann]"
    hud = (
        f"{videoName}  |  frame {frameIndex}  "
        f"({framePos + 1}/{totalFrames})  |  "
        f"Preds: {len(boxes)}  GT: {len(gtBoxes)}{annTag}"
    )
    cv2.putText(
        vis, hud, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
    )

    return vis


# =========================================================================
# Load all frames for a TAO video from disk
# =========================================================================
def loadVideoFrames(
    taoDataRoot: Path,
    videoRecord: dict,
    allImageRecords: List[dict],
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Load *every* frame listed in the annotation JSON for a given video.

    The annotation JSON only lists frames that were sent for annotation;
    however many videos have thousands of raw frames on disk.  To give a
    complete picture we scan the video directory and load all `frame*.jpg`
    files.

    Returns:
        bgrFrames  – list of BGR numpy arrays (one per frame)
        frameRecords – list of dicts with at least 'frame_index' and 'id'
                       (id = -1 for unannotated frames)
    """
    videoName = videoRecord["name"]  # e.g. "train/YFCC100M/v_…"
    videoDir = taoDataRoot / "frames" / videoName

    if not videoDir.exists():
        print(f"  [WARN] Video directory not found: {videoDir}")
        return [], []

    # Build a lookup from frame_index → image record (annotated frames)
    annotatedByFrame: Dict[int, dict] = {}
    for rec in allImageRecords:
        annotatedByFrame[rec["frame_index"]] = rec

    # Discover all frame files on disk
    frameFiles: List[Tuple[int, Path]] = []
    for p in sorted(videoDir.iterdir()):
        if not p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            continue
        # Parse frame index from e.g. "frame0391.jpg"
        # NOTE: filenames are 1-based (frame0001.jpg is the first frame)
        # but TAO JSON frame_index is 0-based (first frame = 0).
        stem = p.stem  # "frame0391"
        if stem.startswith("frame"):
            try:
                fileIdx = int(stem[5:])           # 1-based from filename
                fIdx = fileIdx - 1                 # 0-based to match JSON
            except ValueError:
                continue
            frameFiles.append((fIdx, p))

    frameFiles.sort(key=lambda x: x[0])

    bgrFrames: List[np.ndarray] = []
    frameRecords: List[dict] = []

    for fIdx, fPath in frameFiles:
        bgr = cv2.imread(str(fPath))
        if bgr is None:
            continue
        bgrFrames.append(bgr)

        if fIdx in annotatedByFrame:
            frameRecords.append(annotatedByFrame[fIdx])
        else:
            # Unannotated frame — we still run inference on it
            frameRecords.append({
                "id": -1,
                "frame_index": fIdx,
                "width": bgr.shape[1],
                "height": bgr.shape[0],
            })

    return bgrFrames, frameRecords


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        "TAO VideoDETR Inference", parents=[getArgsParser()]
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[TAO-Infer] Running on {device}")

    # ── Load model (first, so we can read taoMaxCategories) ───────────
    model, modelArgs = loadModel(args.modelPath, device)
    numClasses = model.numClasses

    # ── Load TAO annotations ──────────────────────────────────────────
    taoRoot = Path(args.taoDataRoot)
    annPath = taoRoot / "annotations" / f"{args.split}.json"
    assert annPath.exists(), f"Annotation file not found: {annPath}"
    taoMaxCategories = getattr(modelArgs, "taoMaxCategories", None)
    taoAnns = TaoAnnotations(str(annPath), taoMaxCategories=taoMaxCategories)

    # Verify category counts match
    if numClasses != taoAnns.numClasses:
        print(
            f"  ⚠  Model has {numClasses} classes but TAO annotations report "
            f"{taoAnns.numClasses} categories (taoMaxCategories="
            f"{taoMaxCategories}). Class predictions may be misaligned."
        )

    palette = _buildColourPalette(max(numClasses, taoAnns.numClasses))

    # ── Select random videos ──────────────────────────────────────────
    allVideoIds = sorted(taoAnns.videos.keys())
    numToSelect = min(args.numVideos, len(allVideoIds))
    selectedIds = random.sample(allVideoIds, numToSelect)
    print(f"\n[TAO-Infer] Selected {numToSelect} videos: {selectedIds}")

    # ── Inference transform ───────────────────────────────────────────
    transform = makeInferenceTransform(maxSize=args.maxSize)

    # ── Output root ───────────────────────────────────────────────────
    outputRoot = Path(args.outputDir)
    outputRoot.mkdir(parents=True, exist_ok=True)

    # ── Process each video ────────────────────────────────────────────
    for vidIdx, videoId in enumerate(selectedIds):
        videoRecord = taoAnns.videos[videoId]
        videoName = videoRecord["name"]
        # Sanitise for filesystem (replace path separators)
        safeName = videoName.replace("/", "__")

        print(f"\n{'='*65}")
        print(
            f"[{vidIdx+1}/{numToSelect}] Video {videoId}: {videoName}  "
            f"({videoRecord.get('width', '?')}×{videoRecord.get('height', '?')})"
        )
        print(f"{'='*65}")

        # Load all frames from disk
        imageRecords = taoAnns.videoImages.get(videoId, [])
        bgrFrames, frameRecords = loadVideoFrames(
            taoRoot, videoRecord, imageRecords
        )

        if not bgrFrames:
            print("  No frames found – skipping.")
            continue

        print(f"  Loaded {len(bgrFrames)} frames from disk")

        # Run inference
        print("  Running sliding-window inference …")
        results = inferVideo(
            model,
            bgrFrames,
            transform,
            device,
            confidence=args.confidence,
            nmsThreshold=args.nmsThreshold,
        )

        # Associate tracks
        print("  Associating tracks …")
        results = associateTracks(results, args.trackingThreshold)

        # ── Write output (video or individual frames) ─────────────
        vidDir = outputRoot / safeName
        vidDir.mkdir(parents=True, exist_ok=True)

        # Compute output resolution (resize first frame to get dims)
        firstResized, _, _ = resizeForOutput(bgrFrames[0], args.maxSize)
        outH, outW = firstResized.shape[:2]

        writer = None
        framesDir = None
        outVideoPath = None

        if args.saveFrames:
            framesDir = vidDir / "frames"
            framesDir.mkdir(parents=True, exist_ok=True)
        else:
            outVideoPath = vidDir / "video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(outVideoPath), fourcc, args.fps, (outW, outH)
            )

        totalPreds = 0
        totalGt = 0
        annotatedCount = 0

        for fPos, (bgr, rec) in enumerate(zip(bgrFrames, frameRecords)):
            imgH, imgW = bgr.shape[:2]
            imageId = rec["id"]
            frameIndex = rec["frame_index"]
            hasAnnotation = imageId >= 0

            if hasAnnotation:
                gtBoxes, gtLabels, gtTrackIds = taoAnns.getGtForImage(
                    imageId, imgW, imgH
                )
                annotatedCount += 1
            else:
                gtBoxes = np.zeros((0, 4), dtype=np.float32)
                gtLabels = np.zeros((0,), dtype=np.int64)
                gtTrackIds = np.zeros((0,), dtype=np.int64)

            # Resize frame to output size and scale boxes accordingly
            resizedBgr, scaleX, scaleY = resizeForOutput(bgr, args.maxSize)
            gtBoxesScaled = scaleBoxes(gtBoxes, scaleX, scaleY)
            predResultScaled = results[fPos].copy()
            predResultScaled["boxes"] = scaleBoxes(
                results[fPos]["boxes"], scaleX, scaleY
            )

            vis = drawFrame(
                resizedBgr,
                predResultScaled,
                gtBoxesScaled,
                gtLabels,
                gtTrackIds,
                taoAnns,
                palette,
                videoName,
                frameIndex,
                fPos,
                len(bgrFrames),
                hasAnnotation,
            )

            if args.saveFrames:
                framePath = framesDir / f"frame_{frameIndex:06d}.jpg"
                cv2.imwrite(
                    str(framePath), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpgQuality],
                )
            else:
                writer.write(vis)

            totalPreds += len(results[fPos]["boxes"])
            totalGt += len(gtBoxes)

        if writer is not None:
            writer.release()

        # ── Write info file ───────────────────────────────────────────
        infoPath = vidDir / "info.txt"
        outputMode = "frames (JPG)" if args.saveFrames else "video (MP4)"
        with open(infoPath, "w") as f:
            f.write(f"Video ID       : {videoId}\n")
            f.write(f"Video name     : {videoName}\n")
            f.write(f"Resolution     : {outW}×{outH} (maxSize={args.maxSize})\n")
            f.write(f"Total frames   : {len(bgrFrames)}\n")
            f.write(f"Annotated      : {annotatedCount}\n")
            f.write(f"Total GT boxes : {totalGt}\n")
            f.write(f"Total preds    : {totalPreds}\n")
            f.write(f"Confidence     : {args.confidence}\n")
            f.write(f"NMS threshold  : {args.nmsThreshold}\n")
            f.write(f"Track threshold: {args.trackingThreshold}\n")
            f.write(f"Model          : {args.modelPath}\n")
            f.write(f"Output mode    : {outputMode}\n")
            if args.saveFrames:
                f.write(f"JPG quality    : {args.jpgQuality}\n")
            else:
                f.write(f"FPS            : {args.fps}\n")

        if args.saveFrames:
            print(
                f"  ✓ Saved {len(bgrFrames)} frames to {framesDir}  "
                f"({totalPreds} preds, {totalGt} GT)"
            )
        else:
            print(
                f"  ✓ Saved {outVideoPath}  "
                f"({len(bgrFrames)} frames, {totalPreds} preds, {totalGt} GT)"
            )

    print(f"\n✓ All done. Results in {outputRoot.resolve()}")


if __name__ == "__main__":
    main()
