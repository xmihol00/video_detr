#!/usr/bin/env python3
# Copyright (c) 2026. All Rights Reserved.
"""
Video Dataset Visualizer for VideoDETR.

Calls the VideoDataset dataloader 10 times, draws GT bounding boxes onto
the images using OpenCV, and saves the results to disk.

Usage:
    python -m vidDetr.video_dataset_visualizer
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vidDetr.datasets.video_dataset import VideoDataset, makeVideoDatasetTransforms

# ── Constants ────────────────────────────────────────────────────────
# Default dataset root (same as --videoDataRoot in main.py)
DEFAULT_DATA_ROOT = "/mnt/matylda5/xmihol00/datasets/climbing_videos"
OUTPUT_DIR = "video_dataset_debug_frames"
NUM_SAMPLES = 50
NUM_FRAMES = 4
MAX_SIZE = 512  # same as --maxSize default in main.py

# ImageNet normalisation constants (for de-normalising tensors)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Colour palette for different track IDs (BGR)
COLOURS = [
    (0, 220, 0),    # green
    (0, 0, 230),    # red
    (230, 180, 0),  # cyan-ish
    (0, 200, 255),  # yellow
    (200, 0, 200),  # magenta
    (255, 128, 0),  # blue-ish
    (128, 255, 0),  # lime
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink-blue
    (128, 0, 255),  # purple
]


def denormalizeTensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised [3, H, W] image tensor to a BGR uint8 ndarray."""
    img = tensor.detach().cpu().float().numpy()       # [3, H, W]
    img = img.transpose(1, 2, 0)                      # [H, W, 3] RGB
    img = img * IMAGENET_STD + IMAGENET_MEAN          # undo normalise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cxcywhToXyxy(boxes: np.ndarray, imgW: int, imgH: int) -> np.ndarray:
    """Normalised cxcywh (0-1) → absolute xyxy pixel coords."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx = boxes[:, 0] * imgW
    cy = boxes[:, 1] * imgH
    bw = boxes[:, 2] * imgW
    bh = boxes[:, 3] * imgH
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def drawBoxes(
    bgr: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    trackIds: np.ndarray,
) -> np.ndarray:
    """Draw bounding boxes with labels and track IDs onto a BGR image."""
    imgH, imgW = bgr.shape[:2]
    xyxy = cxcywhToXyxy(boxes, imgW, imgH)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        tid = int(trackIds[i]) if len(trackIds) > i else 0
        colour = COLOURS[tid % len(COLOURS)]

        cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        label = f"c{int(labels[i])} t{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(bgr, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            bgr, label, (x1 + 2, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return bgr


def main():
    dataRoot = Path(DEFAULT_DATA_ROOT) / "train"
    outputDir = Path(OUTPUT_DIR)
    outputDir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outputDir.resolve()}")

    # Build dataset with the same transforms used during training
    transforms = makeVideoDatasetTransforms("train", maxSize=MAX_SIZE)

    dataset = VideoDataset(
        dataRoot=str(dataRoot),
        numFrames=NUM_FRAMES,
        transforms=transforms,
        imageSet="train",
        batchesPerEpoch=100,
        batchSize=1,
        maxFrameOffset=10,
        samplingStrategy="mixed",
        numClasses=2,
        useCache=True,
        minBoxSize=0.0,
    )

    print(f"Dataset has {dataset.totalFrames} frames, "
          f"drawing {NUM_SAMPLES} samples × {NUM_FRAMES} frames each ...")

    for sampleIdx in range(NUM_SAMPLES):
        images, targets = dataset[sampleIdx]

        for frameIdx in range(NUM_FRAMES):
            imgTensor = images[frameIdx]     # [3, H, W] normalised tensor
            tgt = targets[frameIdx]

            # De-normalise to BGR uint8
            bgr = denormalizeTensor(imgTensor)
            imgH, imgW = bgr.shape[:2]

            # Extract boxes (normalised cxcywh after Normalize transform)
            boxes = tgt["boxes"].detach().cpu().numpy()    # [N, 4]
            labels = tgt["labels"].detach().cpu().numpy()  # [N]
            trackIds = tgt.get("trackIds", torch.zeros(len(labels), dtype=torch.int64))
            trackIds = trackIds.detach().cpu().numpy()

            # Draw GT boxes
            bgr = drawBoxes(bgr, boxes, labels, trackIds)

            # HUD info
            numGt = len(boxes)
            hud = (f"sample {sampleIdx}  frame {frameIdx}  "
                   f"size {imgW}x{imgH}  GT: {numGt}")
            cv2.putText(
                bgr, hud, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA,
            )

            # Save
            outPath = outputDir / f"sample_{sampleIdx:02d}_frame_{frameIdx:02d}.jpg"
            cv2.imwrite(str(outPath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"  Sample {sampleIdx:2d}: saved {NUM_FRAMES} frames "
              f"({len(targets[0]['boxes'])} boxes in frame 0)")

    print(f"\nDone. {NUM_SAMPLES * NUM_FRAMES} images saved to {outputDir.resolve()}")


if __name__ == "__main__":
    main()
