#!/usr/bin/env python3
# Copyright (c) 2026. All Rights Reserved.
"""
Evaluation / visualisation script for the TAO dataset loader.

This script:
1. Instantiates the TaoDataset for train and validation splits.
2. Randomly samples one clip from each split.
3. Draws bounding boxes, segmentation polygons, and track numbers onto
   every frame of the sampled clip.
4. Saves the annotated frames to ``tao_dataset_evaluation/``.

Directory layout of the output::

    tao_dataset_evaluation/
    ├── train/
    │   ├── clip_info.txt
    │   ├── frame_00.jpg
    │   ├── frame_01.jpg
    │   └── ...
    └── val/
        ├── clip_info.txt
        ├── frame_00.jpg
        ├── frame_01.jpg
        └── ...

Usage:
    python -m vidDetr.eval_tao_dataset --taoDataRoot /path/to/tao

    or from the repo root:

    python vidDetr/eval_tao_dataset.py --taoDataRoot /path/to/tao
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_parentDir = Path(__file__).resolve().parent.parent
if str(_parentDir) not in sys.path:
    sys.path.insert(0, str(_parentDir))

import argparse
import random
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from vidDetr.datasets.tao_dataset import TaoDataset


# ── Colour palette ────────────────────────────────────────────────────
def _colourForTrackId(trackId: int) -> tuple:
    """Deterministic, visually distinct colour for a given track ID."""
    rng = random.Random(trackId * 7 + 13)
    return (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))


# ── Drawing helpers ───────────────────────────────────────────────────
def _tryLoadFont(size: int = 16):
    """Try to load a TrueType font; fall back to the default bitmap font."""
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def drawAnnotationsOnFrame(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    trackIds: torch.Tensor,
    iscrowd: torch.Tensor,
    segments: List,
    categoryNames: Optional[Dict[int, str]] = None,
    lineWidth: int = 3,
    fontSize: int = 16,
) -> Image.Image:
    """
    Draw bounding boxes, segmentation polygons, and track labels onto *image*.

    Args:
        image:         PIL RGB image.
        boxes:         [N, 4] tensor in xyxy absolute pixels.
        labels:        [N] contiguous class indices.
        trackIds:      [N] track identity integers.
        iscrowd:       [N] crowd flags.
        segments:      list of N polygon lists (each is [[x1,y1,x2,y2,…], …]).
        categoryNames: optional mapping contiguous-id → human name.
        lineWidth:     box outline width.
        fontSize:      font size for labels.

    Returns:
        Annotated copy of the image.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = _tryLoadFont(fontSize)

    numBoxes = len(boxes)
    for i in range(numBoxes):
        tid = int(trackIds[i])
        colour = _colourForTrackId(tid)
        x1, y1, x2, y2 = boxes[i].tolist()

        # ── Segmentation polygon (semi-transparent fill) ──────────
        if i < len(segments) and segments[i]:
            for poly in segments[i]:
                if len(poly) >= 6:
                    coords = list(zip(poly[0::2], poly[1::2]))
                    fillColour = colour + (50,)      # RGBA with alpha
                    outlineColour = colour + (180,)
                    draw.polygon(coords, fill=fillColour, outline=outlineColour)

        # ── Bounding box ──────────────────────────────────────────
        for offset in range(lineWidth):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=colour,
            )

        # ── Label text ────────────────────────────────────────────
        classIdx = int(labels[i])
        if categoryNames and classIdx in categoryNames:
            catName = categoryNames[classIdx]
        else:
            catName = f"cls{classIdx}"

        crowdStr = " [crowd]" if int(iscrowd[i]) else ""
        label = f"T{tid} {catName}{crowdStr}"

        # Background box behind text
        textBbox = draw.textbbox((0, 0), label, font=font)
        tw = textBbox[2] - textBbox[0]
        th = textBbox[3] - textBbox[1]
        textX = max(0, x1)
        textY = max(0, y1 - th - 4)
        draw.rectangle(
            [textX, textY, textX + tw + 6, textY + th + 4],
            fill=colour + (180,),
        )
        draw.text((textX + 3, textY + 1), label, fill=(255, 255, 255), font=font)

    return img


# ── Main evaluation routine ──────────────────────────────────────────
def evaluateOneDataset(
    dataset: TaoDataset,
    outputDir: Path,
    splitName: str,
) -> None:
    """
    Sample a random clip from *dataset*, draw annotations, and save.
    """
    outDir = outputDir / splitName
    outDir.mkdir(parents=True, exist_ok=True)

    idx = random.randint(0, len(dataset) - 1)
    videoId, windowImages = dataset.windows[idx]
    selectedImages = dataset._sampleFrameIndices(windowImages)

    # Build contiguous → category name mapping
    catNames: Dict[int, str] = {}
    for contId, origId in dataset.contiguousToCatId.items():
        cat = dataset.categories.get(origId, {})
        catNames[contId] = cat.get("name", f"id{origId}")

    # ── Write clip info ───────────────────────────────────────────
    videoInfo = dataset.videoInfo[videoId]
    infoLines = [
        f"Split         : {splitName}",
        f"Dataset index : {idx}",
        f"Video ID      : {videoId}",
        f"Video name    : {videoInfo['name']}",
        f"Resolution    : {videoInfo.get('width', '?')}x{videoInfo.get('height', '?')}",
        f"Frames sampled: {len(selectedImages)}",
        f"Frame indices : {[img['frame_index'] for img in selectedImages]}",
        f"Num classes   : {dataset.numClasses}",
        "",
    ]

    for clipIdx, imgRecord in enumerate(selectedImages):
        img = dataset._loadImage(imgRecord)
        imgW, imgH = img.size

        boxes, labels, trackIds, iscrowd, segments = (
            dataset._loadAnnotationsForImage(imgRecord["id"], imgW, imgH)
        )

        infoLines.append(
            f"Frame {clipIdx:02d}: frame_index={imgRecord['frame_index']}, "
            f"image_id={imgRecord['id']}, "
            f"{len(boxes)} objects, "
            f"tracks={trackIds.tolist()}"
        )

        annotatedImg = drawAnnotationsOnFrame(
            img, boxes, labels, trackIds, iscrowd, segments,
            categoryNames=catNames,
        )
        savePath = outDir / f"frame_{clipIdx:02d}.jpg"
        annotatedImg.save(savePath, quality=95)
        print(f"  Saved {savePath}")

    infoPath = outDir / "clip_info.txt"
    with open(infoPath, "w") as f:
        f.write("\n".join(infoLines) + "\n")
    print(f"  Saved {infoPath}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate / visualise the TAO dataset loader"
    )
    parser.add_argument(
        "--taoDataRoot", default="/mnt/matylda5/xmihol00/tao/dataset/", type=str,
        help="Root directory of the TAO dataset",
    )
    parser.add_argument(
        "--numFrames", default=10, type=int,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--minFrameGap", default=1, type=int,
        help="Minimum stride between sampled frames",
    )
    parser.add_argument(
        "--maxFrameGap", default=10, type=int,
        help="Maximum stride between sampled frames",
    )
    parser.add_argument(
        "--outputDir", default="tao_dataset_evaluation", type=str,
        help="Output directory for visualisations",
    )
    parser.add_argument(
        "--taoMaxCategories", default=None, type=int,
        help="Keep only top-N most frequent categories (None = all)",
    )
    parser.add_argument(
        "--seed", default=random.randint(0, 2**32 - 1), type=int,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.taoDataRoot)
    annDir = root / "annotations"
    outputDir = Path(args.outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    maxCat = args.taoMaxCategories

    # ── Train split ───────────────────────────────────────────────
    trainJson = annDir / "train.json"
    if trainJson.exists():
        print(f"\n{'='*60}")
        print("Loading TRAIN dataset ...")
        print(f"{'='*60}")
        trainDataset = TaoDataset(
            dataRoot=str(root),
            annotationFile=str(trainJson),
            numFrames=args.numFrames,
            transforms=None,  # no transforms — we want raw images for drawing
            imageSet="train",
            minFrameGap=args.minFrameGap,
            maxFrameGap=args.maxFrameGap,
            maxCategoriesUsed=maxCat,
        )
        evaluateOneDataset(trainDataset, outputDir, "train")
    else:
        print(f"[WARN] Train annotations not found: {trainJson}")

    # ── Val split ─────────────────────────────────────────────────
    valJson = annDir / "validation.json"
    if valJson.exists():
        print(f"\n{'='*60}")
        print("Loading VAL dataset ...")
        print(f"{'='*60}")
        valDataset = TaoDataset(
            dataRoot=str(root),
            annotationFile=str(valJson),
            numFrames=args.numFrames,
            transforms=None,
            imageSet="val",
            minFrameGap=args.minFrameGap,
            maxFrameGap=args.maxFrameGap,
            maxCategoriesUsed=maxCat,
        )
        evaluateOneDataset(valDataset, outputDir, "val")
    else:
        print(f"[WARN] Validation annotations not found: {valJson}")

    print(f"\n✓ Results saved to {outputDir.resolve()}")


if __name__ == "__main__":
    main()
