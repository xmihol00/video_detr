#!/usr/bin/env python3
# Copyright (c) 2026. All Rights Reserved.
"""
VideoDETR baseline evaluation for simulated video datasets.

This script evaluates a trained VideoDETR model on a video-like dataset stored
in YOLO label format (class cx cy w h, track_id inferred from line index) or
the extended format (track_id class_id cx cy w h).  It mirrors the interface of
yolo_baseline_evaluation.py so the two reports can be compared side-by-side.

Supported metrics
-----------------
1) Detection evaluation with COCO metrics (mAP@[.50:.95], AP50, AP75, etc.)
2) Tracking evaluation with common MOT metrics (MOTA, MOTP, IDF1, ID switches)

Tracking uses the model's own tracking embeddings (L2-normalised output of the
TrackingHead) to associate detections across frames via greedy cosine-similarity
matching.  This is the same algorithm used in vidDetr/inference.py.

Dataset assumptions
-------------------
- Image files follow: seq_XXXXXX_frame_XXXX.<ext>  (primary)
  OR any filename (generic fallback for non-sequence datasets)
- Label files are YOLO format (5 cols) or extended format (6 cols):
    5 cols:  class_id  cx  cy  w  h         (track_id = line index)
    6 cols:  track_id  class_id  cx  cy  w  h
- Matching data.yaml structure (same as yolo_baseline_evaluation.py)

Usage
-----
    python vidDetr/viddetr_evaluation.py \\
        --model /path/to/video_detr_best.pth \\
        --data-config vidDetr/data.yaml \\
        --split val
"""

from __future__ import annotations

DEVICE = "cuda"
if DEVICE == "cuda":
    import safe_gpu
    import time as _time
    while True:
        try:
            safe_gpu.claim_gpus(1)
            break
        except Exception:
            print("Waiting for free GPU")
            _time.sleep(5)

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as exc:
    raise RuntimeError(
        "pycocotools is required for detection metrics. "
        "Install it via requirements or pip."
    ) from exc

try:
    import motmetrics as mm  # type: ignore[import-not-found]
except Exception as exc:
    raise RuntimeError(
        "motmetrics is required for tracking metrics. "
        "Install with: pip install motmetrics"
    ) from exc

try:
    from torchvision.ops import batched_nms  # type: ignore[import-not-found]
except Exception as exc:
    raise RuntimeError(
        "torchvision is required. Install with: pip install torchvision"
    ) from exc

from vidDetr.models import buildVideoDETR
from vidDetr.util.misc import nested_tensor_from_tensor_list
from vidDetr.datasets import transforms as T


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_PATTERN = re.compile(r"seq_(\d{6})_frame_(\d{4})")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    sequence_id: str
    frame_idx: int
    image_path: Path
    label_path: Path
    image_id: int
    width: int
    height: int


@dataclass
class GroundTruth:
    boxes_xyxy: np.ndarray   # (N, 4) absolute xyxy
    labels: np.ndarray       # (N,) int
    track_ids: np.ndarray    # (N,) int


@dataclass
class Detections:
    boxes_xyxy: np.ndarray   # (K, 4) absolute xyxy
    scores: np.ndarray       # (K,)
    labels: np.ndarray       # (K,) int
    embeddings: np.ndarray   # (K, D) L2-normalised tracking embeddings


@dataclass
class TrackedDetections:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    track_ids: np.ndarray


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, level: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("viddetr_eval")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "viddetr_evaluation.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("VideoDETR detection + tracking evaluation")

    parser.add_argument(
        "--model",
        default="/homes/eva/xm/xmihol00/video_detr/weights_2026-02-26/video_detr_best.pth",
        type=str,
        help="Path to a VideoDETR checkpoint (.pth).",
    )
    parser.add_argument(
        "--data-config",
        default="/homes/eva/xm/xmihol00/video_detr/vidDetr/data.yaml",
        type=str,
        help="Path to data YAML file with train/val image roots and class names.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--device", default=DEVICE, type=str, help="Device, e.g. 'cpu', 'cuda'.")
    parser.add_argument("--max-size", default=640, type=int, help="Inference image size (max side).")
    parser.add_argument(
        "--conf",
        default=0.35,
        type=float,
        help="Detector confidence threshold (applied before NMS).",
    )
    parser.add_argument(
        "--nms-iou",
        default=0.5,
        type=float,
        help="Per-frame NMS IoU threshold.",
    )
    parser.add_argument(
        "--tracking-threshold",
        default=0.4,
        type=float,
        help="Cosine-similarity threshold for cross-frame track association.",
    )
    parser.add_argument(
        "--tracking-ema-alpha",
        default=0.6,
        type=float,
        help="EMA weight for new embedding in track update (0 = no update, 1 = replace).",
    )
    parser.add_argument(
        "--max-track-age",
        default=30,
        type=int,
        help="Maximum frames a track can remain without a detection before deletion.",
    )
    parser.add_argument(
        "--max-sequences",
        default=None,
        type=int,
        help="Optional cap on number of sequences (for quick experiments).",
    )
    parser.add_argument(
        "--class-aware-tracking",
        action="store_true",
        help="Only associate detections/tracks of the same class.",
    )
    parser.add_argument(
        "--tracking-eval-iou-threshold",
        default=0.5,
        type=float,
        help="IoU threshold for GT-pred assignment in MOT metric computation.",
    )
    parser.add_argument("--output-dir", default="runs/viddetr_eval", type=str)
    parser.add_argument(
        "--report-file",
        default=None,
        type=str,
        help="Optional explicit report path. Defaults to output-dir/report_<timestamp>.json",
    )
    parser.add_argument("--save-tracks", action="store_true", help="Save per-frame tracking outputs to JSON.")
    parser.add_argument("--log-level", default="INFO", type=str)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data config helpers
# ---------------------------------------------------------------------------

def _read_data_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_split_paths(data_cfg: Dict[str, Any], split: str) -> Tuple[Path, Path]:
    images_root = Path(data_cfg[split])

    if images_root.name == "images":
        labels_root = images_root.parent / "labels"
    else:
        images_root = images_root / "images"
        labels_root = images_root.parent / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    return images_root, labels_root


def _load_class_names(data_cfg: Dict[str, Any]) -> Dict[int, str]:
    names = data_cfg.get("names", {})
    return {int(k): str(v) for k, v in names.items()}


# ---------------------------------------------------------------------------
# Frame indexing
# ---------------------------------------------------------------------------

def _index_frames(
    images_root: Path,
    labels_root: Path,
    logger: logging.Logger,
) -> Dict[str, List[FrameRecord]]:
    """
    Index frames from images_root/labels_root.

    Frames matching FILENAME_PATTERN are grouped by sequence ID.
    Frames that do NOT match are placed in a single synthetic sequence
    so the script still works on non-standard datasets.
    Image dimensions are read from the actual files (unlike the YOLO baseline
    which hard-codes 800×800 for the simulated dataset).
    """
    frames_by_seq: Dict[str, List[FrameRecord]] = {}
    image_id = 1

    image_by_stem: Dict[str, Path] = {
        p.stem: p for p in images_root.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    }

    logger.info("Indexing %d labels", len(list(labels_root.iterdir())))

    for i, label_path in enumerate(labels_root.iterdir()):
        if i > 0 and i % 10000 == 0:
            logger.info("Indexed %d labels", i)
        if label_path.suffix.lower() != ".txt":
            continue

        img_path = image_by_stem.get(label_path.stem)
        if img_path is None:
            logger.warning("Missing image for label, skipping: %s", label_path)
            continue

        match = FILENAME_PATTERN.match(label_path.stem)
        if match:
            seq_id = match.group(1)
            frame_idx = int(match.group(2))
        else:
            # Fallback: treat the whole directory as one sequence, sort lexically.
            seq_id = "000000"
            frame_idx = i

        # Read actual image size
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            logger.warning("Cannot read image size for %s, skipping", img_path)
            continue

        rec = FrameRecord(
            sequence_id=seq_id,
            frame_idx=frame_idx,
            image_path=img_path,
            label_path=label_path,
            image_id=image_id,
            width=img_w,
            height=img_h,
        )
        image_id += 1
        frames_by_seq.setdefault(seq_id, []).append(rec)

    for seq_id in list(frames_by_seq.keys()):
        frames_by_seq[seq_id].sort(key=lambda x: x.frame_idx)

    return frames_by_seq


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def _xywhn_to_xyxy_abs(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> List[float]:
    x1 = float(np.clip((cx - w / 2.0) * img_w, 0.0, img_w))
    y1 = float(np.clip((cy - h / 2.0) * img_h, 0.0, img_h))
    x2 = float(np.clip((cx + w / 2.0) * img_w, 0.0, img_w))
    y2 = float(np.clip((cy + h / 2.0) * img_h, 0.0, img_h))
    return [x1, y1, x2, y2]


def load_ground_truth(frame: FrameRecord) -> GroundTruth:
    """
    Load GT labels.  Supports two formats:
      5 cols:  class_id cx cy w h            (YOLO format – track_id = line_idx)
      6 cols:  track_id class_id cx cy w h   (extended video_dataset format)
    """
    boxes: List[List[float]] = []
    labels: List[int] = []
    track_ids: List[int] = []

    with open(frame.label_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                # YOLO format
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    tid = line_idx
                except ValueError:
                    continue
            elif len(parts) >= 6:
                # Extended format
                try:
                    tid = int(parts[0])
                    cls = int(parts[1])
                    cx, cy, w, h = map(float, parts[2:6])
                except ValueError:
                    continue
            else:
                continue

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue

            boxes.append(_xywhn_to_xyxy_abs(cx, cy, w, h, frame.width, frame.height))
            labels.append(cls)
            track_ids.append(tid)

    if boxes:
        return GroundTruth(
            boxes_xyxy=np.asarray(boxes, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            track_ids=np.asarray(track_ids, dtype=np.int64),
        )
    return GroundTruth(
        boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        track_ids=np.zeros((0,), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, Any]:
    """
    Load a VideoDETR checkpoint and reconstruct the model.

    Returns the model (eval mode) and the stored args namespace.
    """
    logger.info("Loading checkpoint from %s", model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    args = checkpoint.get("args", None)
    if args is None:
        raise RuntimeError(
            "Checkpoint does not contain 'args'. Cannot reconstruct model architecture."
        )

    # Ensure compatibility attributes needed by buildVideoDETR
    args.device = str(device)
    for attr, fallback in [
        ("lr_backbone", "lrBackbone"),
        ("position_embedding", "positionEmbedding"),
        ("hidden_dim", "hiddenDim"),
        ("enc_layers", "encLayers"),
        ("dec_layers", "decLayers"),
        ("dim_feedforward", "dimFeedforward"),
        ("pre_norm", "preNorm"),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, getattr(args, fallback, None))
    if not hasattr(args, "masks"):
        args.masks = False

    # dropout may be a list (per-epoch schedule); buildVideoDETR handles it
    model, _criterion, _postprocessors = buildVideoDETR(args)
    model.to(device)

    # Try EMA state first (usually better for inference)
    if "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
        logger.info("Loading EMA state dict")
    else:
        state_dict = checkpoint.get("model", checkpoint)
        logger.info("Loading model state dict (no EMA found)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s%s", len(missing), missing[:5], "…" if len(missing) > 5 else "")
    if unexpected:
        logger.warning("Unexpected keys (%d): %s%s", len(unexpected), unexpected[:5], "…" if len(unexpected) > 5 else "")

    model.eval()
    logger.info(
        "Model loaded — numFrames=%d, queriesPerFrame=%d, numClasses=%d",
        model.numFrames, model.queriesPerFrame, model.numClasses,
    )
    return model, args


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def make_transform(max_size: int) -> Any:
    """Deterministic val-style transform: resize + normalise."""
    return T.Compose([
        T.RandomResize([800], max_size=max_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess_image(bgr_path: Path, transform: Any) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load an image from *bgr_path* and apply transforms.

    Returns (tensor, (orig_h, orig_w)).
    """
    from PIL import Image as PILImage
    import cv2

    bgr = cv2.imread(str(bgr_path))
    if bgr is None:
        raise IOError(f"Cannot read image: {bgr_path}")
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    dummy_target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "origSize": torch.tensor([orig_h, orig_w]),
        "size": torch.tensor([orig_h, orig_w]),
    }
    tensor, _ = transform(pil_img, dummy_target)
    return tensor, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Sliding-window inference on a full sequence
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sequence(
    model: torch.nn.Module,
    frames: List[FrameRecord],
    transform: Any,
    device: torch.device,
    conf: float,
    nms_iou: float,
    use_focal_loss: bool,
    logger: logging.Logger,
) -> List[Detections]:
    """
    Run sliding-window inference on a full sequence.

    Returns one Detections object per frame, with boxes in absolute xyxy
    and L2-normalised tracking embeddings.
    """
    num_model_frames = model.numFrames
    queries_per_frame = model.queriesPerFrame
    total_frames = len(frames)

    # Pre-load and transform all frames
    tensors: List[torch.Tensor] = []
    orig_sizes: List[Tuple[int, int]] = []

    for rec in frames:
        tensor, (orig_h, orig_w) = preprocess_image(rec.image_path, transform)
        tensors.append(tensor)
        orig_sizes.append((orig_h, orig_w))

    # Accumulate raw detections (before NMS merge)
    # Each entry is a list of (boxes_xyxy, scores, labels, embeddings) chunks
    per_frame_chunks: List[List[Tuple[torch.Tensor, ...]]] = [[] for _ in range(total_frames)]

    stride = max(1, num_model_frames // 2)
    window_starts = list(range(0, max(1, total_frames - num_model_frames + 1), stride))
    # Ensure the last window covers the tail of the sequence
    if not window_starts or window_starts[-1] + num_model_frames < total_frames:
        window_starts.append(max(0, total_frames - num_model_frames))

    for w_start in window_starts:
        w_end = min(w_start + num_model_frames, total_frames)
        clip_len = w_end - w_start

        clip_tensors = list(tensors[w_start:w_end])
        # Pad short tail by repeating the last frame
        while len(clip_tensors) < num_model_frames:
            clip_tensors.append(clip_tensors[-1])

        samples = [
            nested_tensor_from_tensor_list([t]).to(device)
            for t in clip_tensors
        ]

        outputs = model(samples)

        pred_logits = outputs["pred_logits"]   # [1, numQueries, C+1]
        pred_boxes = outputs["pred_boxes"]     # [1, numQueries, 4]
        pred_tracking = outputs["pred_tracking"]  # [1, numQueries, D]

        for local_f in range(clip_len):
            global_f = w_start + local_f
            q_start = local_f * queries_per_frame
            q_end = q_start + queries_per_frame

            logits = pred_logits[0, q_start:q_end]        # [Q, C+1]
            boxes_cxcywh = pred_boxes[0, q_start:q_end]   # [Q, 4]
            embeddings = pred_tracking[0, q_start:q_end]  # [Q, D]

            # Compute per-query confidence scores
            if use_focal_loss:
                # Sigmoid per class; take max over real classes (exclude last no-obj dim)
                scores, pred_labels = logits[:, :-1].sigmoid().max(dim=-1)
            else:
                # Softmax; drop no-object column; take max
                probs = logits.softmax(-1)[:, :-1]  # [Q, C]
                scores, pred_labels = probs.max(dim=-1)

            # Convert normalised cxcywh → absolute xyxy
            orig_h, orig_w = orig_sizes[global_f]
            cx = boxes_cxcywh[:, 0] * orig_w
            cy = boxes_cxcywh[:, 1] * orig_h
            bw = boxes_cxcywh[:, 2] * orig_w
            bh = boxes_cxcywh[:, 3] * orig_h
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [Q, 4]

            per_frame_chunks[global_f].append(
                (abs_boxes, scores, pred_labels, embeddings)
            )

    # Merge windows & apply confidence filter + NMS per frame
    results: List[Detections] = []

    for f_idx in range(total_frames):
        chunks = per_frame_chunks[f_idx]
        if not chunks:
            embed_dim = model.trackingHead.outputDim if hasattr(model, 'trackingHead') else 128
            results.append(Detections(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int64),
                embeddings=np.zeros((0, embed_dim), dtype=np.float32),
            ))
            continue

        all_boxes = torch.cat([c[0] for c in chunks], dim=0)
        all_scores = torch.cat([c[1] for c in chunks], dim=0)
        all_labels = torch.cat([c[2] for c in chunks], dim=0)
        all_embs = torch.cat([c[3] for c in chunks], dim=0)

        # Confidence filter
        keep = all_scores >= conf
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_labels = all_labels[keep]
        all_embs = all_embs[keep]

        # Per-frame class-aware NMS
        if len(all_scores) > 0:
            keep_nms = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=nms_iou)
            all_boxes = all_boxes[keep_nms]
            all_scores = all_scores[keep_nms]
            all_labels = all_labels[keep_nms]
            all_embs = all_embs[keep_nms]

        # L2-normalise embeddings
        all_embs = F.normalize(all_embs, p=2, dim=-1)

        results.append(Detections(
            boxes_xyxy=all_boxes.cpu().numpy().astype(np.float32),
            scores=all_scores.cpu().numpy().astype(np.float32),
            labels=all_labels.cpu().numpy().astype(np.int64),
            embeddings=all_embs.cpu().numpy().astype(np.float32),
        ))

    return results


# ---------------------------------------------------------------------------
# Embedding-based tracker (same algorithm as inference.py::associateTracks)
# ---------------------------------------------------------------------------

def associate_tracks(
    detections_per_frame: List[Detections],
    similarity_threshold: float = 0.4,
    ema_alpha: float = 0.6,
    max_track_age: int = 30,
    class_aware: bool = False,
) -> List[TrackedDetections]:
    """
    Greedy cross-frame association using cosine similarity of tracking embeddings.

    Returns a list of TrackedDetections (one per frame) with consistent track IDs.
    """
    next_track_id = 0
    active_tracks: List[Dict[str, Any]] = []  # {id, embedding, age, class_id}

    tracked_results: List[TrackedDetections] = []

    for det in detections_per_frame:
        n_dets = len(det.scores)
        track_ids = np.full(n_dets, -1, dtype=np.int64)

        if n_dets > 0 and det.embeddings.shape[0] > 0:
            embs = det.embeddings.copy()  # (K, D), already normalised
            assigned = np.zeros(n_dets, dtype=bool)

            if active_tracks:
                track_embs = np.stack([t["embedding"] for t in active_tracks], axis=0)  # (T, D)

                if class_aware:
                    track_cls = np.array([t["class_id"] for t in active_tracks], dtype=np.int64)
                    det_cls = det.labels

                sim = embs @ track_embs.T  # (K, T)

                if class_aware:
                    class_mask = det_cls[:, None] == track_cls[None, :]  # (K, T)
                    sim = np.where(class_mask, sim, -1.0)

                flat = sim.flatten()
                order = np.argsort(-flat)

                used_tracks: set = set()
                for idx in order:
                    det_idx = int(idx // len(active_tracks))
                    trk_idx = int(idx % len(active_tracks))
                    if assigned[det_idx] or trk_idx in used_tracks:
                        continue
                    if sim[det_idx, trk_idx] < similarity_threshold:
                        break
                    # Assign
                    track_ids[det_idx] = active_tracks[trk_idx]["id"]
                    # EMA update
                    new_emb = ema_alpha * embs[det_idx] + (1 - ema_alpha) * active_tracks[trk_idx]["embedding"]
                    norm = max(np.linalg.norm(new_emb), 1e-8)
                    active_tracks[trk_idx]["embedding"] = new_emb / norm
                    active_tracks[trk_idx]["age"] = 0
                    active_tracks[trk_idx]["class_id"] = int(det.labels[det_idx])

                    assigned[det_idx] = True
                    used_tracks.add(trk_idx)

            # New tracks for unassigned detections
            for i in range(n_dets):
                if not assigned[i]:
                    tid = next_track_id
                    next_track_id += 1
                    track_ids[i] = tid
                    active_tracks.append({
                        "id": tid,
                        "embedding": embs[i].copy(),
                        "age": 0,
                        "class_id": int(det.labels[i]),
                    })

        elif n_dets > 0:
            # No embeddings — assign unique IDs to all detections
            for i in range(n_dets):
                track_ids[i] = next_track_id
                next_track_id += 1
                active_tracks.append({
                    "id": int(track_ids[i]),
                    "embedding": np.zeros(1),
                    "age": 0,
                    "class_id": int(det.labels[i]),
                })

        # Age out stale tracks
        for t in active_tracks:
            t["age"] += 1
        active_tracks = [t for t in active_tracks if t["age"] <= max_track_age]

        tracked_results.append(TrackedDetections(
            boxes_xyxy=det.boxes_xyxy,
            scores=det.scores,
            labels=det.labels,
            track_ids=track_ids,
        ))

    return tracked_results


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0, None], boxes_a[:, 1, None], boxes_a[:, 2, None], boxes_a[:, 3, None]
    bx1, by1, bx2, by2 = boxes_b[None, :, 0], boxes_b[None, :, 1], boxes_b[None, :, 2], boxes_b[None, :, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def _xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# COCO detection metrics
# ---------------------------------------------------------------------------

def _evaluate_detection_coco(
    coco_gt_dataset: Dict[str, Any],
    coco_pred_dets: List[Dict[str, Any]],
    logger: logging.Logger,
) -> Dict[str, float]:
    coco_gt = COCO()
    coco_gt.dataset = cast(Any, coco_gt_dataset)
    coco_gt.createIndex()

    if len(coco_pred_dets) == 0:
        logger.warning("No detections produced. Detection metrics are all zeros.")
        return {k: 0.0 for k in ["AP@[0.50:0.95]", "AP50", "AP75", "AP_small",
                                   "AP_medium", "AP_large", "AR1", "AR10",
                                   "AR100", "AR_small", "AR_medium", "AR_large"]}

    coco_dt = coco_gt.loadRes(cast(Any, coco_pred_dets))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats.tolist()
    keys = [
        "AP@[0.50:0.95]", "AP50", "AP75",
        "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100",
        "AR_small", "AR_medium", "AR_large",
    ]
    return {k: float(v) for k, v in zip(keys, stats)}


# ---------------------------------------------------------------------------
# MOT tracking metrics
# ---------------------------------------------------------------------------

def _evaluate_tracking_mot(
    accumulators: List[Any],
    seq_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    if not accumulators:
        return {}, {}

    metrics = [
        "idf1", "idp", "idr", "recall", "precision",
        "num_objects", "mostly_tracked", "partially_tracked", "mostly_lost",
        "num_false_positives", "num_misses", "num_switches", "num_fragmentations",
        "mota", "motp",
    ]
    mh = mm.metrics.create()
    summary_df = mh.compute_many(
        accumulators,
        names=seq_names,
        metrics=metrics,
        generate_overall=True,
    )

    per_sequence: Dict[str, Dict[str, float]] = {}
    for seq in seq_names:
        row = summary_df.loc[seq]
        per_sequence[seq] = {k: float(row[k]) for k in metrics}

    overall_row = summary_df.loc["OVERALL"]
    overall = {k: float(overall_row[k]) for k in metrics}
    return overall, per_sequence


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(output_dir, args.log_level)

    device = torch.device(args.device)

    # ---- Load dataset config ----
    data_cfg_path = Path(args.data_config)
    data_cfg = _read_data_config(data_cfg_path)
    class_names = _load_class_names(data_cfg)
    images_root, labels_root = _resolve_split_paths(data_cfg, args.split)

    logger.info("Loading dataset split '%s' from %s", args.split, images_root)
    frames_by_seq = _index_frames(images_root, labels_root, logger)

    seq_ids = sorted(frames_by_seq.keys())
    if args.max_sequences is not None:
        seq_ids = seq_ids[: args.max_sequences]
        frames_by_seq = {k: frames_by_seq[k] for k in seq_ids}

    total_frames = sum(len(v) for v in frames_by_seq.values())
    logger.info("Indexed %d sequences and %d frames", len(seq_ids), total_frames)

    if total_frames == 0:
        raise RuntimeError("No valid frames found for evaluation.")

    # ---- Load model ----
    model, model_args = load_model(args.model, device, logger)
    use_focal_loss = bool(getattr(model_args, "useFocalLoss", False))
    logger.info("use_focal_loss=%s", use_focal_loss)

    transform = make_transform(args.max_size)

    # ---- COCO GT containers ----
    coco_gt = {
        "info": {"description": "VideoDETR evaluation"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": class_names.get(cid, f"class_{cid}")}
            for cid in sorted(class_names.keys())
        ],
    }
    coco_pred: List[Dict[str, Any]] = []
    ann_id = 1

    accumulators: List[Any] = []
    mot_seq_names: List[str] = []
    tracker_outputs: Dict[str, Any] = {}

    total_gt_boxes = 0
    total_pred_boxes = 0
    eval_start = time.perf_counter()
    infer_time = 0.0

    # ---- Per-sequence loop ----
    for seq_idx, seq_id in enumerate(seq_ids, start=1):
        frames = frames_by_seq[seq_id]
        logger.info(
            "[%d/%d] Evaluating sequence %s (%d frames)",
            seq_idx, len(seq_ids), seq_id, len(frames),
        )

        # Sliding-window inference
        t0 = time.perf_counter()
        detections_per_frame = infer_sequence(
            model=model,
            frames=frames,
            transform=transform,
            device=device,
            conf=args.conf,
            nms_iou=args.nms_iou,
            use_focal_loss=use_focal_loss,
            logger=logger,
        )
        infer_time += time.perf_counter() - t0

        # Embedding-based tracking
        tracked_per_frame = associate_tracks(
            detections_per_frame,
            similarity_threshold=args.tracking_threshold,
            ema_alpha=args.tracking_ema_alpha,
            max_track_age=args.max_track_age,
            class_aware=args.class_aware_tracking,
        )

        acc = mm.MOTAccumulator(auto_id=True)
        if args.save_tracks:
            tracker_outputs[seq_id] = []

        # ---- Per-frame accumulation ----
        for frame, det, tracked in zip(frames, detections_per_frame, tracked_per_frame):
            gt = load_ground_truth(frame)

            # COCO GT image entry
            coco_gt["images"].append({
                "id": frame.image_id,
                "file_name": str(frame.image_path),
                "width": frame.width,
                "height": frame.height,
            })

            # COCO GT annotations
            gt_xywh = _xyxy_to_xywh(gt.boxes_xyxy)
            for i in range(len(gt_xywh)):
                bbox = gt_xywh[i]
                area = float(max(0.0, bbox[2]) * max(0.0, bbox[3]))
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": frame.image_id,
                    "category_id": int(gt.labels[i]),
                    "bbox": [float(x) for x in bbox.tolist()],
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

            # COCO predictions
            pred_xywh = _xyxy_to_xywh(det.boxes_xyxy)
            for i in range(len(pred_xywh)):
                coco_pred.append({
                    "image_id": frame.image_id,
                    "category_id": int(det.labels[i]),
                    "bbox": [float(x) for x in pred_xywh[i].tolist()],
                    "score": float(det.scores[i]),
                })

            # MOT accumulator update
            iou = _bbox_iou_matrix(gt.boxes_xyxy, tracked.boxes_xyxy)
            if args.class_aware_tracking and len(gt.labels) > 0 and len(tracked.labels) > 0:
                class_mask = gt.labels[:, None] == tracked.labels[None, :]
                iou = np.where(class_mask, iou, 0.0)

            dist = 1.0 - iou
            dist[iou < args.tracking_eval_iou_threshold] = np.nan

            acc.update(
                gt.track_ids.tolist(),
                tracked.track_ids.tolist(),
                dist,
            )

            if args.save_tracks:
                tracker_outputs[seq_id].append({
                    "frame_idx": frame.frame_idx,
                    "image_path": str(frame.image_path),
                    "boxes_xyxy": tracked.boxes_xyxy.tolist(),
                    "scores": tracked.scores.tolist(),
                    "labels": tracked.labels.tolist(),
                    "track_ids": tracked.track_ids.tolist(),
                })

            total_gt_boxes += int(len(gt.boxes_xyxy))
            total_pred_boxes += int(len(det.boxes_xyxy))

        accumulators.append(acc)
        mot_seq_names.append(seq_id)

    total_time = time.perf_counter() - eval_start

    # ---- Metrics ----
    logger.info("Computing COCO detection metrics...")
    detection_metrics = _evaluate_detection_coco(coco_gt, coco_pred, logger)

    logger.info("Computing MOT tracking metrics...")
    tracking_overall, tracking_per_sequence = _evaluate_tracking_mot(accumulators, mot_seq_names)

    timing = {
        "total_eval_seconds": float(total_time),
        "inference_seconds": float(infer_time),
        "postprocess_seconds": float(max(0.0, total_time - infer_time)),
        "frames": int(total_frames),
        "overall_fps": float(total_frames / total_time) if total_time > 0 else 0.0,
        "inference_fps": float(total_frames / infer_time) if infer_time > 0 else 0.0,
    }

    report: Dict[str, Any] = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "tracker": "embedding_cosine",
            "data_config": str(data_cfg_path),
            "split": args.split,
            "use_focal_loss": use_focal_loss,
        },
        "dataset": {
            "images_root": str(images_root),
            "labels_root": str(labels_root),
            "num_sequences": len(seq_ids),
            "num_frames": int(total_frames),
            "num_gt_boxes": int(total_gt_boxes),
            "class_count": len(class_names),
        },
        "detection": {
            "num_predictions": int(total_pred_boxes),
            "metrics_coco": detection_metrics,
        },
        "tracking": {
            "num_tracked_detections": int(total_pred_boxes),
            "metrics_overall": tracking_overall,
            "metrics_per_sequence": tracking_per_sequence,
            "matching_iou_threshold": float(args.tracking_eval_iou_threshold),
        },
        "timing": timing,
        "config": {
            k: v for k, v in vars(args).items()
            if not k.startswith("_")
        },
    }

    if args.save_tracks:
        tracks_path = output_dir / f"tracks_{timestamp}.json"
        with open(tracks_path, "w") as f:
            json.dump(tracker_outputs, f)
        logger.info("Saved per-frame tracks to %s", tracks_path)
        report["tracking"]["tracks_file"] = str(tracks_path)

    report_path = (
        Path(args.report_file)
        if args.report_file is not None
        else output_dir / f"report_{timestamp}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Report written to %s", report_path)
    logger.info("Detection AP50-95:  %.4f", detection_metrics.get("AP@[0.50:0.95]", 0.0))
    logger.info("Detection AP50:     %.4f", detection_metrics.get("AP50", 0.0))
    logger.info("Tracking IDF1:      %.4f", tracking_overall.get("idf1", 0.0))
    logger.info("Tracking MOTA:      %.4f", tracking_overall.get("mota", 0.0))
    logger.info("Tracking ID switches: %d", int(tracking_overall.get("num_switches", 0)))
    logger.info("Overall FPS:        %.2f", timing["overall_fps"])

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
