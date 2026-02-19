#!/usr/bin/env python3
# Copyright (c) 2026. All Rights Reserved.
"""
Inference script for VideoDETR.

Runs inference on test video sequences, associates detections across frames
using tracking embeddings, and visualises predictions vs. ground-truth with
OpenCV.

Usage:
    python vidDetr/inference.py \
        --modelPath vidDetr_weights/video_detr_best.pth \
        --testDir test \
        --dataConfig vidDetr/data.yaml \
        --confidence 0.5

Keys during visualisation:
    →  / d / SPACE  – next frame
    ←  / a          – previous frame
    n               – next sequence
    p               – previous sequence
    q / ESC         – quit
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import argparse
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from datasets import transforms as T
from vidDetr.models import buildVideoDETR

# ---------------------------------------------------------------------------
# Filename pattern (must match video_dataset.py)
# ---------------------------------------------------------------------------
FILENAME_PATTERN = re.compile(r"seq_(\d{6})_frame_(\d{4})")

# ImageNet normalisation (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================================
# Argument parsing
# =========================================================================
def getArgsParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "VideoDETR inference & visualisation", add_help=False
    )

    # Required
    parser.add_argument(
        "--modelPath",
        default="checkpoint_latest.pth",
        type=str,
        help="Path to a VideoDETR checkpoint (.pth)",
    )
    parser.add_argument(
        "--testDir",
        default="test",
        type=str,
        help="Root directory containing test/images and test/labels",
    )
    parser.add_argument(
        "--dataConfig",
        default="data.yaml",
        type=str,
        help="Path to data.yaml (used for class names)",
    )

    # Inference behaviour
    parser.add_argument(
        "--confidence",
        default=0.4,
        type=float,
        help="Minimum confidence threshold for displaying predictions",
    )
    parser.add_argument(
        "--nmsThreshold",
        default=0.5,
        type=float,
        help="IoU threshold for per-frame NMS",
    )
    parser.add_argument(
        "--trackingThreshold",
        default=0.4,
        type=float,
        help="Cosine-similarity threshold for cross-frame track association",
    )
    parser.add_argument(
        "--maxSize",
        default=384,
        type=int,
        help="Maximum image size for inference (must match training)",
    )

    # Display
    parser.add_argument(
        "--saveDir",
        default="",
        type=str,
        help="If set, save annotated frames to this directory instead of "
             "displaying them interactively",
    )
    parser.add_argument(
        "--windowName",
        default="VideoDETR Inference",
        type=str,
        help="OpenCV window title",
    )

    return parser


# =========================================================================
# Colour palette  –  one distinct colour per class
# =========================================================================
def _buildColourPalette(numClasses: int) -> List[Tuple[int, int, int]]:
    """Generate *numClasses* visually distinct BGR colours using HSV spacing."""
    colours = []
    for i in range(numClasses):
        hue = int(180 * i / max(numClasses, 1))  # OpenCV hue range 0-179
        hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
        colours.append((b, g, r))
    return colours


# =========================================================================
# Dataset discovery
# =========================================================================
def discoverTestSequences(
    testDir: str,
) -> Dict[str, List[int]]:
    """
    Scan *testDir*/images for all sequences and their frame indices.

    Returns
    -------
    sequences : dict
        ``{seqId: sorted_list_of_frame_indices}``
    """
    imagesDir = Path(testDir) / "images"
    assert imagesDir.exists(), f"Images directory not found: {imagesDir}"

    sequences: Dict[str, List[int]] = defaultdict(list)
    imageExtensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for imgFile in imagesDir.iterdir():
        if imgFile.suffix.lower() not in imageExtensions:
            continue
        match = FILENAME_PATTERN.match(imgFile.stem)
        if match is None:
            continue
        seqId = match.group(1)
        frameIdx = int(match.group(2))
        sequences[seqId].append(frameIdx)

    # Sort frames within each sequence
    for seqId in sequences:
        sequences[seqId] = sorted(sequences[seqId])

    print(f"[Inference] Discovered {len(sequences)} test sequences "
          f"in {imagesDir}")
    return dict(sequences)


# =========================================================================
# Image / label I/O helpers
# =========================================================================
def loadImage(testDir: str, seqId: str, frameIdx: int) -> np.ndarray:
    """Load an image as a BGR numpy array (for cv2) *and* an RGB PIL Image."""
    base = Path(testDir) / "images" / f"seq_{seqId}_frame_{frameIdx:04d}"
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = base.with_suffix(ext)
        if p.exists():
            bgr = cv2.imread(str(p))
            assert bgr is not None, f"Failed to read {p}"
            return bgr
    raise FileNotFoundError(f"No image found for seq {seqId} frame {frameIdx}")


def loadGtLabels(
    testDir: str, seqId: str, frameIdx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ground-truth labels in YOLO format.

    Returns
    -------
    boxes_cxcywh : (N, 4)  normalised cx, cy, w, h
    labels       : (N,)    class indices
    trackIds     : (N,)    line-number-based track IDs
    """
    labelPath = (
        Path(testDir) / "labels" / f"seq_{seqId}_frame_{frameIdx:04d}.txt"
    )
    boxes, labels, trackIds = [], [], []

    if labelPath.exists():
        with open(labelPath, "r") as f:
            for lineIdx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                classId = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h])
                labels.append(classId)
                trackIds.append(lineIdx)

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
# Preprocessing (must mirror validation transforms from video_dataset.py)
# =========================================================================
def makeInferenceTransform(maxSize: int = 800):
    """Deterministic val-style transform: resize + normalise."""
    return T.Compose([
        T.RandomResize([800], max_size=maxSize),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocessFrame(
    bgrImage: np.ndarray,
    transform: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convert a BGR cv2 image to a transformed tensor + dummy target dict.

    Returns the transformed image tensor and a target dict with ``size`` and
    ``orig_size`` entries (needed by the post-processor).
    """
    rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(rgbImage)
    imgW, imgH = pilImage.size

    # The DETR transforms expect a target dict; we provide a minimal one
    dummyTarget = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "origSize": torch.tensor([imgH, imgW]),
        "size": torch.tensor([imgH, imgW]),
    }

    imgTensor, target = transform(pilImage, dummyTarget)
    return imgTensor, target


# =========================================================================
# Post-processing utilities
# =========================================================================
def cxcywhToXyxy(boxes: np.ndarray, imgW: int, imgH: int) -> np.ndarray:
    """Normalised cxcywh → absolute xyxy."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * imgW
    y1 = (cy - h / 2) * imgH
    x2 = (cx + w / 2) * imgW
    y2 = (cy + h / 2) * imgH
    return np.stack([x1, y1, x2, y2], axis=1)


def nmsPerFrame(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iouThreshold: float = 0.5,
) -> torch.Tensor:
    """
    Class-aware NMS.  Returns a boolean keep-mask.
    """
    from torchvision.ops import batched_nms

    keep = batched_nms(boxes, scores, labels, iou_threshold=iouThreshold)
    mask = torch.zeros(len(scores), dtype=torch.bool)
    mask[keep] = True
    return mask


# =========================================================================
# Model loading
# =========================================================================
def loadModel(
    modelPath: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a VideoDETR checkpoint and reconstruct the model.

    Returns the model (in eval mode) and the stored ``args`` namespace.
    """
    print(f"[Inference] Loading checkpoint from {modelPath} …")
    checkpoint = torch.load(modelPath, map_location="cpu", weights_only=False)

    # The checkpoint stores the full args namespace used during training
    args = checkpoint.get("args", None)
    if args is None:
        raise RuntimeError(
            "Checkpoint does not contain 'args'. "
            "Cannot reconstruct model architecture."
        )

    # Ensure compatibility attributes exist
    args.device = str(device)
    if not hasattr(args, "lr_backbone"):
        args.lr_backbone = getattr(args, "lrBackbone", 1e-5)
    if not hasattr(args, "position_embedding"):
        args.position_embedding = getattr(args, "positionEmbedding", "sine")
    if not hasattr(args, "hidden_dim"):
        args.hidden_dim = getattr(args, "hiddenDim", 256)
    if not hasattr(args, "masks"):
        args.masks = False
    if not hasattr(args, "enc_layers"):
        args.enc_layers = getattr(args, "encLayers", 6)
    if not hasattr(args, "dec_layers"):
        args.dec_layers = getattr(args, "decLayers", 6)
    if not hasattr(args, "dim_feedforward"):
        args.dim_feedforward = getattr(args, "dimFeedforward", 2048)
    if not hasattr(args, "pre_norm"):
        args.pre_norm = getattr(args, "preNorm", False)

    # Build model
    model, _criterion, _postprocessors = buildVideoDETR(args)
    model.to(device)

    # Load state dict (with tolerance for missing / unexpected keys)
    modelStateDict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(modelStateDict, strict=False)
    if missing:
        print(f"  ⚠  Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    model.eval()
    print(f"[Inference] Model loaded – numFrames={model.numFrames}, "
          f"queriesPerFrame={model.queriesPerFrame}, "
          f"numClasses={model.numClasses}")
    return model, args


# =========================================================================
# Core inference loop for one sequence
# =========================================================================
@torch.no_grad()
def inferSequence(
    model: torch.nn.Module,
    testDir: str,
    seqId: str,
    frameIndices: List[int],
    transform: Any,
    device: torch.device,
    confidence: float = 0.5,
    nmsThreshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Run inference on a full video sequence using a sliding window of
    ``model.numFrames`` frames.

    For frames that appear in multiple windows the highest-confidence
    detection is kept (after NMS).

    Returns a list of result dicts (one per frame in *frameIndices*), each
    containing:
        - ``boxes``      : (K, 4) absolute xyxy
        - ``scores``     : (K,)
        - ``labels``     : (K,)
        - ``embeddings`` : (K, D) tracking embeddings
    """
    numFrames = model.numFrames
    queriesPerFrame = model.queriesPerFrame
    numClasses = model.numClasses
    totalFrames = len(frameIndices)

    # Pre-load and transform all frames
    bgrImages: List[np.ndarray] = []
    imgTensors: List[torch.Tensor] = []
    origSizes: List[Tuple[int, int]] = []  # (H, W)

    for fIdx in frameIndices:
        bgr = loadImage(testDir, seqId, fIdx)
        bgrImages.append(bgr)
        h, w = bgr.shape[:2]
        origSizes.append((h, w))

        tensor, _tgt = preprocessFrame(bgr, transform)
        imgTensors.append(tensor)

    # ---- Sliding-window inference ----
    # Accumulate per-frame detections; later merge overlapping windows.
    # Each entry: list of (boxes_xyxy, scores, labels, embeddings)
    perFrameRaw: List[List[Tuple[torch.Tensor, ...]]] = [
        [] for _ in range(totalFrames)
    ]

    # Build window start positions so every frame is covered at least once
    stride = max(1, numFrames // 2)  # 50% overlap for robustness
    windowStarts = list(range(0, max(1, totalFrames - numFrames + 1), stride))
    # Ensure last window covers the tail
    if windowStarts[-1] + numFrames < totalFrames:
        windowStarts.append(totalFrames - numFrames)

    for wStart in windowStarts:
        wEnd = min(wStart + numFrames, totalFrames)
        clipLen = wEnd - wStart

        # If the remaining tail is shorter than numFrames, pad by repeating
        # the last frame (the model requires exactly numFrames inputs).
        clipTensors = [imgTensors[i] for i in range(wStart, wEnd)]
        while len(clipTensors) < numFrames:
            clipTensors.append(clipTensors[-1])

        # Build NestedTensor list (one per frame, batch-size 1)
        samples = [
            nested_tensor_from_tensor_list([t]).to(device)
            for t in clipTensors
        ]

        outputs = model(samples)

        # --- Decode outputs per frame in the window ---
        predLogits = outputs["pred_logits"]  # [1, numQueries, C+1]
        predBoxes = outputs["pred_boxes"]    # [1, numQueries, 4]
        predTracking = outputs["pred_tracking"]  # [1, numQueries, D]

        for localF in range(clipLen):
            globalF = wStart + localF
            qStart = localF * queriesPerFrame
            qEnd = qStart + queriesPerFrame

            logits = predLogits[0, qStart:qEnd]      # [Q, C+1]
            boxesCxcywh = predBoxes[0, qStart:qEnd]   # [Q, 4]
            embeddings = predTracking[0, qStart:qEnd]  # [Q, D]

            # Class probabilities (softmax, drop no-object column)
            probs = logits.softmax(-1)[:, :-1]         # [Q, C]
            maxScores, maxLabels = probs.max(-1)        # [Q], [Q]

            # Convert normalised cxcywh → absolute xyxy
            imgH, imgW = origSizes[globalF]
            cx = boxesCxcywh[:, 0] * imgW
            cy = boxesCxcywh[:, 1] * imgH
            bw = boxesCxcywh[:, 2] * imgW
            bh = boxesCxcywh[:, 3] * imgH
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            absBoxes = torch.stack([x1, y1, x2, y2], dim=-1)

            perFrameRaw[globalF].append(
                (absBoxes, maxScores, maxLabels, embeddings)
            )

    # ---- Merge windows & apply NMS per frame ----
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

        allBoxes = torch.cat([c[0] for c in chunks], dim=0)
        allScores = torch.cat([c[1] for c in chunks], dim=0)
        allLabels = torch.cat([c[2] for c in chunks], dim=0)
        allEmbed = torch.cat([c[3] for c in chunks], dim=0)

        # Confidence filter
        keep = allScores >= confidence
        allBoxes = allBoxes[keep]
        allScores = allScores[keep]
        allLabels = allLabels[keep]
        allEmbed = allEmbed[keep]

        # NMS
        if len(allScores) > 0:
            keepNms = nmsPerFrame(allBoxes, allScores, allLabels, nmsThreshold)
            allBoxes = allBoxes[keepNms]
            allScores = allScores[keepNms]
            allLabels = allLabels[keepNms]
            allEmbed = allEmbed[keepNms]

        results.append({
            "boxes": allBoxes.cpu().numpy(),
            "scores": allScores.cpu().numpy(),
            "labels": allLabels.cpu().numpy(),
            "embeddings": allEmbed.cpu().numpy(),
        })

    return results


# =========================================================================
# Cross-frame track association using tracking embeddings
# =========================================================================
def associateTracks(
    results: List[Dict[str, Any]],
    similarityThreshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Greedy cross-frame association based on cosine similarity of tracking
    embeddings.  Assigns a consistent ``trackId`` to each detection.

    Modifies *results* in-place (adds ``trackId`` array) and returns it.
    """
    nextTrackId = 0
    # Active tracks: list of dicts {id, embedding (running avg)}
    activeTracks: List[Dict[str, Any]] = []
    emaAlpha = 0.6  # weight for new embedding in exponential moving average

    for frameResult in results:
        nDets = len(frameResult["scores"])
        frameResult["trackId"] = np.full(nDets, -1, dtype=np.int64)

        if nDets == 0:
            continue

        embeddings = frameResult["embeddings"]  # (K, D)
        if embeddings.shape[1] == 0:
            # No tracking head output; assign unique IDs
            for i in range(nDets):
                frameResult["trackId"][i] = nextTrackId
                nextTrackId += 1
            continue

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-8)
        embeddings = embeddings / norms

        assigned = np.zeros(nDets, dtype=bool)

        if activeTracks:
            # Build matrix of cosine similarities
            trackEmbeds = np.stack(
                [t["embedding"] for t in activeTracks], axis=0
            )  # (T, D)
            sim = embeddings @ trackEmbeds.T  # (K, T)

            # Greedy matching: highest similarity first
            flat = sim.flatten()
            order = np.argsort(-flat)

            usedTracks = set()
            for idx in order:
                detIdx = int(idx // len(activeTracks))
                trkIdx = int(idx % len(activeTracks))
                if assigned[detIdx] or trkIdx in usedTracks:
                    continue
                if sim[detIdx, trkIdx] < similarityThreshold:
                    break  # remaining are even lower
                # Assign
                frameResult["trackId"][detIdx] = activeTracks[trkIdx]["id"]
                # Update track embedding (EMA)
                activeTracks[trkIdx]["embedding"] = (
                    emaAlpha * embeddings[detIdx]
                    + (1 - emaAlpha) * activeTracks[trkIdx]["embedding"]
                )
                norm = np.linalg.norm(activeTracks[trkIdx]["embedding"]).clip(1e-8)
                activeTracks[trkIdx]["embedding"] /= norm
                activeTracks[trkIdx]["age"] = 0

                assigned[detIdx] = True
                usedTracks.add(trkIdx)

        # Start new tracks for unassigned detections
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

        # Age-out stale tracks (not seen for a while)
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

    # Four edges as line segments
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
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
    classNames: List[str],
    palette: List[Tuple[int, int, int]],
    seqId: str,
    frameIdx: int,
    framePos: int,
    totalFrames: int,
) -> np.ndarray:
    """
    Draw predictions (thick, solid) and GT (thin, dashed) on a copy of
    *bgrImage* and return it.
    """
    vis = bgrImage.copy()
    imgH, imgW = vis.shape[:2]
    numClasses = len(classNames) if classNames else len(palette)

    # --- Ground truth (thin dashed) ---
    if len(gtBoxes) > 0:
        gtXyxy = cxcywhToXyxy(gtBoxes, imgW, imgH)
        for i in range(len(gtXyxy)):
            x1, y1, x2, y2 = gtXyxy[i].astype(int)
            cls = int(gtLabels[i]) % len(palette)
            colour = palette[cls]
            _drawDashedRect(vis, (x1, y1), (x2, y2), colour, thickness=2, dashLen=8, gapLen=5)
            # Label text
            clsName = classNames[cls] if cls < len(classNames) else str(cls)
            label = f"GT {clsName} t{int(gtTrackIds[i])}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1)
            cv2.putText(vis, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Predictions (thick solid) ---
    boxes = predResult["boxes"]
    scores = predResult["scores"]
    labels = predResult["labels"]
    trackIds = predResult.get("trackId", np.full(len(scores), -1))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls = int(labels[i]) % len(palette)
        colour = palette[cls]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)

        clsName = classNames[cls] if cls < len(classNames) else str(cls)
        tid = int(trackIds[i])
        label = f"{clsName} {scores[i]:.2f} t{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1, y2), (x1 + tw + 4, y2 + th + 6), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y2 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # --- HUD ---
    hud = (f"Seq {seqId}  |  Frame {frameIdx}  "
           f"({framePos + 1}/{totalFrames})  |  "
           f"Preds: {len(boxes)}  GT: {len(gtBoxes)}")
    cv2.putText(vis, hud, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    helpText = "[Space/d]->  [a]<-  [n]ext seq  [p]rev seq  [q]uit"
    cv2.putText(vis, helpText, (10, imgH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        "VideoDETR Inference", parents=[getArgsParser()]
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"[Inference] Running on {device}")

    # ---- Load class names from data.yaml ----
    dataConfigPath = Path(args.dataConfig)
    if dataConfigPath.exists():
        with open(dataConfigPath, "r") as f:
            dataCfg = yaml.safe_load(f)
        classNames = list(dataCfg.get("names", {}).values())
    else:
        print(f"[Warning] data.yaml not found at {args.dataConfig}; "
              "using numeric class IDs")
        classNames = []

    # ---- Load model ----
    model, modelArgs = loadModel(args.modelPath, device)
    numClasses = model.numClasses
    if not classNames:
        classNames = [str(i) for i in range(numClasses)]
    palette = _buildColourPalette(max(numClasses, len(classNames)))

    # ---- Discover sequences ----
    sequences = discoverTestSequences(args.testDir)
    if not sequences:
        print("[Inference] No sequences found – exiting.")
        return
    seqIds = sorted(sequences.keys())

    # ---- Transforms (same as val) ----
    transform = makeInferenceTransform(maxSize=args.maxSize)

    # ---- Prepare save directory (if requested) ----
    saveDir = Path(args.saveDir) if args.saveDir else None
    if saveDir:
        saveDir.mkdir(parents=True, exist_ok=True)
        print(f"[Inference] Saving annotated frames to {saveDir}")

    # ---- Interactive visualisation loop ----
    seqIdx = 1

    while 0 <= seqIdx < len(seqIds):
        seqId = seqIds[seqIdx]
        frameIndices = sequences[seqId]
        print(f"\n{'='*60}")
        print(f"[Inference] Sequence {seqId}  ({len(frameIndices)} frames)")
        print(f"{'='*60}")

        # Run inference on the whole sequence
        results = inferSequence(
            model,
            args.testDir,
            seqId,
            frameIndices,
            transform,
            device,
            confidence=args.confidence,
            nmsThreshold=args.nmsThreshold,
        )

        # Associate tracks across frames
        results = associateTracks(results, args.trackingThreshold)

        # --- Display / save ---
        fPos = 0
        while 0 <= fPos < len(frameIndices):
            fIdx = frameIndices[fPos]
            bgr = loadImage(args.testDir, seqId, fIdx)
            gtBoxes, gtLabels, gtTrackIds = loadGtLabels(
                args.testDir, seqId, fIdx
            )
            vis = drawFrame(
                bgr, results[fPos],
                gtBoxes, gtLabels, gtTrackIds,
                classNames, palette,
                seqId, fIdx, fPos, len(frameIndices),
            )

            if saveDir:
                outPath = saveDir / f"seq_{seqId}_frame_{fIdx:04d}.jpg"
                cv2.imwrite(str(outPath), vis)
                fPos += 1
                continue

            cv2.imshow(args.windowName, vis)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):  # q / ESC
                cv2.destroyAllWindows()
                print("[Inference] Quit.")
                return
            elif key in (ord("d"), ord(" "), 83):  # d / space / right arrow
                fPos += 1
            elif key in (ord("a"), 81):  # a / left arrow
                fPos = max(0, fPos - 1)
            elif key == ord("n"):  # next sequence
                break
            elif key == ord("p"):  # previous sequence
                seqIdx -= 2  # will be incremented below
                break
            else:
                fPos += 1  # default: advance

        if saveDir:
            print(f"  Saved {len(frameIndices)} frames for seq {seqId}")

        seqIdx += 1

    cv2.destroyAllWindows()
    print("\n[Inference] All sequences processed.")


if __name__ == "__main__":
    main()
