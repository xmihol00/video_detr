#!/usr/bin/env python3
# Copyright (c) 2026. All Rights Reserved.
"""
YOLO baseline evaluation for simulated video datasets.

This script evaluates a YOLO detector trained on COCO on a video-like dataset
stored in YOLO label format. It supports:

1) Detection evaluation with COCO metrics (mAP@[.50:.95], AP50, AP75, etc.)
2) Tracking evaluation with common MOT metrics (MOTA, MOTP, IDF1, ID switches)
3) Two tracking options:
   - Simple IoU tracker (implemented here)
   - ByteTrack-style tracker (IoU-based implementation inspired by ByteTrack)

Dataset assumptions:
- Image files follow: seq_XXXXXX_frame_XXXX.<ext>
- Label files are YOLO format: class cx cy w h
- Track identity is inferred from label line index across frames in a sequence.
"""

from __future__ import annotations

DEVICE = "cuda"
if DEVICE == "cuda":
    import safe_gpu
    import time
    while True:
        try:
            safe_gpu.claim_gpus(1)
            break
        except:
            print("Waiting for free GPU")
            time.sleep(5)
            pass

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import yaml
from PIL import Image

try:
	from pycocotools.coco import COCO
	from pycocotools.cocoeval import COCOeval
except Exception as exc:  # pragma: no cover - import error is environment-specific
	raise RuntimeError(
		"pycocotools is required for detection metrics. "
		"Install it via requirements or pip."
	) from exc

try:
	import motmetrics as mm  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - import error is environment-specific
	raise RuntimeError(
		"motmetrics is required for tracking metrics. "
		"Install with: pip install motmetrics"
	) from exc

try:
	from ultralytics import YOLO  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - import error is environment-specific
	raise RuntimeError(
		"ultralytics is required to run YOLO models. "
		"Install with: pip install ultralytics"
	) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_PATTERN = re.compile(r"seq_(\d{6})_frame_(\d{4})")
DATASET_IMAGE_WIDTH = 800
DATASET_IMAGE_HEIGHT = 800


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
	boxes_xyxy: np.ndarray
	labels: np.ndarray
	track_ids: np.ndarray


@dataclass
class Detections:
	boxes_xyxy: np.ndarray
	scores: np.ndarray
	labels: np.ndarray


@dataclass
class TrackedDetections:
	boxes_xyxy: np.ndarray
	scores: np.ndarray
	labels: np.ndarray
	track_ids: np.ndarray


@dataclass
class _TrackState:
	track_id: int
	bbox_xyxy: np.ndarray
	score: float
	class_id: int
	hits: int = 1
	age: int = 1
	time_since_update: int = 0


def setup_logging(output_dir: Path, level: str) -> logging.Logger:
	output_dir.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("yolo_baseline_eval")
	logger.setLevel(getattr(logging, level.upper()))
	logger.propagate = False
	logger.handlers.clear()

	fmt = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setFormatter(fmt)
	logger.addHandler(stream_handler)

	file_handler = logging.FileHandler(output_dir / "yolo_baseline_evaluation.log", mode="a")
	file_handler.setFormatter(fmt)
	logger.addHandler(file_handler)

	return logger


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser("YOLO baseline detection + tracking evaluation")

	parser.add_argument(
		"--model",
		default="/mnt/matylda5/xmihol00/yolov8/yolo11n.pt",
		type=str,
		help="Path to YOLO model weights (e.g. yolov8x.pt or custom .pt).",
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
	parser.add_argument(
		"--tracker",
		default="iou",
		choices=["iou", "bytetrack"],
		help="Tracker type: simple IoU tracker or ByteTrack-style tracker.",
	)
	parser.add_argument("--device", default=DEVICE, type=str, help="YOLO device, e.g. 'cpu', '0'.")
	parser.add_argument("--imgsz", default=640, type=int, help="Inference image size for YOLO.")
	parser.add_argument("--batch-size", default=25, type=int, help="YOLO batch size.")
	parser.add_argument("--conf", default=0.35, type=float, help="Detector confidence threshold.")
	parser.add_argument("--nms-iou", default=0.7, type=float, help="Detector NMS IoU threshold.")
	parser.add_argument(
		"--max-sequences",
		default=None,
		type=int,
		help="Optional cap on number of sequences for quick experiments.",
	)
	parser.add_argument(
		"--class-aware-tracking",
		action="store_true",
		help="Only associate detections/tracks of the same class.",
	)

	# Simple IoU tracker params
	parser.add_argument("--iou-track-threshold", default=0.35, type=float)
	parser.add_argument("--iou-max-age", default=4, type=int)
	parser.add_argument("--iou-min-hits", default=1, type=int)

	# ByteTrack-style params
	parser.add_argument("--bt-high-threshold", default=0.5, type=float)
	parser.add_argument("--bt-low-threshold", default=0.1, type=float)
	parser.add_argument("--bt-match-threshold", default=0.3, type=float)
	parser.add_argument("--bt-second-match-threshold", default=0.2, type=float)
	parser.add_argument("--bt-max-age", default=4, type=int)
	parser.add_argument("--bt-min-hits", default=1, type=int)

	parser.add_argument(
		"--tracking-eval-iou-threshold",
		default=0.5,
		type=float,
		help="IoU threshold for GT-pred assignment in MOT metric computation.",
	)

	parser.add_argument("--output-dir", default="runs/yolo_baseline_eval", type=str)
	parser.add_argument(
		"--report-file",
		default=None,
		type=str,
		help="Optional explicit report path. Defaults to output-dir/report_<timestamp>.json",
	)
	parser.add_argument("--save-tracks", action="store_true", help="Save per-frame tracking outputs to JSON.")
	parser.add_argument("--log-level", default="INFO", type=str)

	return parser.parse_args()


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
	out: Dict[int, str] = {}
	for k, v in names.items():
		out[int(k)] = str(v)
	return out


def _index_frames_constant_size(
	images_root: Path,
	labels_root: Path,
	logger: logging.Logger,
	image_width: int = DATASET_IMAGE_WIDTH,
	image_height: int = DATASET_IMAGE_HEIGHT,
) -> Dict[str, List[FrameRecord]]:
	"""Fast frame indexing using filename conventions and fixed image dimensions."""
	frames_by_seq: Dict[str, List[FrameRecord]] = {}
	image_id = 1

	image_by_stem: Dict[str, Path] = {}
	for img_path in images_root.iterdir():
		if img_path.suffix.lower() in IMAGE_EXTS:
			image_by_stem[img_path.stem] = img_path

	logger.info("Indexing %d labels against %d images", len(list(labels_root.iterdir())), len(image_by_stem))
	for i, label_path in enumerate(labels_root.iterdir()):
		if i > 0 and i % 10000 == 0:
			logger.info("Indexed %d labels", i)
		if label_path.suffix.lower() != ".txt":
			continue

		match = FILENAME_PATTERN.match(label_path.stem)
		if not match:
			continue

		img_path = image_by_stem.get(label_path.stem)
		if img_path is None:
			logger.warning("Missing image file for label, skipping: %s", label_path)
			continue

		seq_id = match.group(1)
		frame_idx = int(match.group(2))

		rec = FrameRecord(
			sequence_id=seq_id,
			frame_idx=frame_idx,
			image_path=img_path,
			label_path=label_path,
			image_id=image_id,
			width=image_width,
			height=image_height,
		)
		image_id += 1
		frames_by_seq.setdefault(seq_id, []).append(rec)

	for seq_id in list(frames_by_seq.keys()):
		frames_by_seq[seq_id].sort(key=lambda x: x.frame_idx)

	return frames_by_seq


def _xywhn_to_xyxy_abs(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
	x1 = (cx - w / 2.0) * img_w
	y1 = (cy - h / 2.0) * img_h
	x2 = (cx + w / 2.0) * img_w
	y2 = (cy + h / 2.0) * img_h
	x1 = float(np.clip(x1, 0.0, img_w))
	y1 = float(np.clip(y1, 0.0, img_h))
	x2 = float(np.clip(x2, 0.0, img_w))
	y2 = float(np.clip(y2, 0.0, img_h))
	return [x1, y1, x2, y2]


def load_ground_truth(frame: FrameRecord) -> GroundTruth:
	boxes: List[List[float]] = []
	labels: List[int] = []
	track_ids: List[int] = []

	with open(frame.label_path, "r") as f:
		for line_idx, line in enumerate(f):
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) < 5:
				continue
			try:
				cls = int(parts[0])
				cx, cy, w, h = map(float, parts[1:5])
			except ValueError:
				continue

			if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
				continue

			boxes.append(_xywhn_to_xyxy_abs(cx, cy, w, h, frame.width, frame.height))
			labels.append(cls)
			track_ids.append(line_idx)

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


def _bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
	if len(boxes_a) == 0 or len(boxes_b) == 0:
		return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

	ax1 = boxes_a[:, 0][:, None]
	ay1 = boxes_a[:, 1][:, None]
	ax2 = boxes_a[:, 2][:, None]
	ay2 = boxes_a[:, 3][:, None]

	bx1 = boxes_b[:, 0][None, :]
	by1 = boxes_b[:, 1][None, :]
	bx2 = boxes_b[:, 2][None, :]
	by2 = boxes_b[:, 3][None, :]

	ix1 = np.maximum(ax1, bx1)
	iy1 = np.maximum(ay1, by1)
	ix2 = np.minimum(ax2, bx2)
	iy2 = np.minimum(ay2, by2)

	iw = np.maximum(0.0, ix2 - ix1)
	ih = np.maximum(0.0, iy2 - iy1)
	inter = iw * ih

	area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
	area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))

	union = area_a + area_b - inter
	iou = np.where(union > 0.0, inter / union, 0.0)
	return iou.astype(np.float32)


def _greedy_iou_match(
	track_boxes: np.ndarray,
	det_boxes: np.ndarray,
	iou_threshold: float,
	track_labels: Optional[np.ndarray] = None,
	det_labels: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
	if len(track_boxes) == 0:
		return [], [], list(range(len(det_boxes)))
	if len(det_boxes) == 0:
		return [], list(range(len(track_boxes))), []

	iou = _bbox_iou_matrix(track_boxes, det_boxes)

	if track_labels is not None and det_labels is not None:
		class_mask = track_labels[:, None] == det_labels[None, :]
		iou = np.where(class_mask, iou, 0.0)

	matches: List[Tuple[int, int]] = []
	used_tracks = np.zeros((len(track_boxes),), dtype=bool)
	used_dets = np.zeros((len(det_boxes),), dtype=bool)

	flat_indices = np.argsort(iou.reshape(-1))[::-1]
	n_cols = iou.shape[1]

	for flat_idx in flat_indices:
		trk_idx = int(flat_idx // n_cols)
		det_idx = int(flat_idx % n_cols)
		if used_tracks[trk_idx] or used_dets[det_idx]:
			continue
		if iou[trk_idx, det_idx] < iou_threshold:
			break
		matches.append((trk_idx, det_idx))
		used_tracks[trk_idx] = True
		used_dets[det_idx] = True

	unmatched_tracks = [i for i, u in enumerate(used_tracks) if not u]
	unmatched_dets = [i for i, u in enumerate(used_dets) if not u]
	return matches, unmatched_tracks, unmatched_dets


class IoUTracker:
	"""Simple IoU-based online tracker."""

	def __init__(
		self,
		iou_threshold: float = 0.3,
		max_age: int = 30,
		min_hits: int = 1,
		class_aware: bool = False,
	):
		self.iou_threshold = iou_threshold
		self.max_age = max_age
		self.min_hits = min_hits
		self.class_aware = class_aware
		self._tracks: List[_TrackState] = []
		self._next_track_id = 0

	def _new_track(self, bbox: np.ndarray, score: float, cls: int) -> None:
		self._tracks.append(
			_TrackState(
				track_id=self._next_track_id,
				bbox_xyxy=bbox.copy(),
				score=float(score),
				class_id=int(cls),
			)
		)
		self._next_track_id += 1

	def update(self, detections: Detections) -> TrackedDetections:
		for trk in self._tracks:
			trk.age += 1
			trk.time_since_update += 1

		track_boxes = np.asarray([t.bbox_xyxy for t in self._tracks], dtype=np.float32) if self._tracks else np.zeros((0, 4), dtype=np.float32)
		track_labels = np.asarray([t.class_id for t in self._tracks], dtype=np.int64) if self._tracks else np.zeros((0,), dtype=np.int64)

		matches, unmatched_tracks, unmatched_dets = _greedy_iou_match(
			track_boxes=track_boxes,
			det_boxes=detections.boxes_xyxy,
			iou_threshold=self.iou_threshold,
			track_labels=track_labels if self.class_aware else None,
			det_labels=detections.labels if self.class_aware else None,
		)

		for trk_idx, det_idx in matches:
			trk = self._tracks[trk_idx]
			trk.bbox_xyxy = detections.boxes_xyxy[det_idx].copy()
			trk.score = float(detections.scores[det_idx])
			trk.class_id = int(detections.labels[det_idx])
			trk.hits += 1
			trk.time_since_update = 0

		for det_idx in unmatched_dets:
			self._new_track(
				bbox=detections.boxes_xyxy[det_idx],
				score=float(detections.scores[det_idx]),
				cls=int(detections.labels[det_idx]),
			)

		# Keep reasonably recent tracks only.
		self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

		visible = [
			t
			for t in self._tracks
			if t.time_since_update == 0 and t.hits >= self.min_hits
		]

		if visible:
			return TrackedDetections(
				boxes_xyxy=np.asarray([t.bbox_xyxy for t in visible], dtype=np.float32),
				scores=np.asarray([t.score for t in visible], dtype=np.float32),
				labels=np.asarray([t.class_id for t in visible], dtype=np.int64),
				track_ids=np.asarray([t.track_id for t in visible], dtype=np.int64),
			)

		return TrackedDetections(
			boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
			scores=np.zeros((0,), dtype=np.float32),
			labels=np.zeros((0,), dtype=np.int64),
			track_ids=np.zeros((0,), dtype=np.int64),
		)


class ByteTrackStyleTracker:
	"""
	ByteTrack-style tracker with two-stage IoU association.

	This is an IoU-only implementation inspired by ByteTrack:
	high-confidence detections are matched first, then unmatched tracks are
	recovered using lower-confidence detections.
	"""

	def __init__(
		self,
		high_threshold: float = 0.5,
		low_threshold: float = 0.1,
		match_threshold: float = 0.3,
		second_match_threshold: float = 0.2,
		max_age: int = 30,
		min_hits: int = 1,
		class_aware: bool = False,
	):
		self.high_threshold = high_threshold
		self.low_threshold = low_threshold
		self.match_threshold = match_threshold
		self.second_match_threshold = second_match_threshold
		self.max_age = max_age
		self.min_hits = min_hits
		self.class_aware = class_aware
		self._tracks: List[_TrackState] = []
		self._next_track_id = 0

	def _new_track(self, bbox: np.ndarray, score: float, cls: int) -> None:
		self._tracks.append(
			_TrackState(
				track_id=self._next_track_id,
				bbox_xyxy=bbox.copy(),
				score=float(score),
				class_id=int(cls),
			)
		)
		self._next_track_id += 1

	def update(self, detections: Detections) -> TrackedDetections:
		for trk in self._tracks:
			trk.age += 1
			trk.time_since_update += 1

		high_mask = detections.scores >= self.high_threshold
		low_mask = (detections.scores >= self.low_threshold) & (~high_mask)

		high_dets = Detections(
			boxes_xyxy=detections.boxes_xyxy[high_mask],
			scores=detections.scores[high_mask],
			labels=detections.labels[high_mask],
		)
		low_dets = Detections(
			boxes_xyxy=detections.boxes_xyxy[low_mask],
			scores=detections.scores[low_mask],
			labels=detections.labels[low_mask],
		)

		track_boxes = np.asarray([t.bbox_xyxy for t in self._tracks], dtype=np.float32) if self._tracks else np.zeros((0, 4), dtype=np.float32)
		track_labels = np.asarray([t.class_id for t in self._tracks], dtype=np.int64) if self._tracks else np.zeros((0,), dtype=np.int64)

		# Stage 1: existing tracks with high-confidence detections
		matches1, unmatched_tracks_idx, unmatched_high_idx = _greedy_iou_match(
			track_boxes=track_boxes,
			det_boxes=high_dets.boxes_xyxy,
			iou_threshold=self.match_threshold,
			track_labels=track_labels if self.class_aware else None,
			det_labels=high_dets.labels if self.class_aware else None,
		)

		for trk_idx, det_idx in matches1:
			trk = self._tracks[trk_idx]
			trk.bbox_xyxy = high_dets.boxes_xyxy[det_idx].copy()
			trk.score = float(high_dets.scores[det_idx])
			trk.class_id = int(high_dets.labels[det_idx])
			trk.hits += 1
			trk.time_since_update = 0

		# Stage 2: unmatched tracks with low-confidence detections (recovery)
		if unmatched_tracks_idx and len(low_dets.boxes_xyxy) > 0:
			rem_track_boxes = np.asarray([self._tracks[i].bbox_xyxy for i in unmatched_tracks_idx], dtype=np.float32)
			rem_track_labels = np.asarray([self._tracks[i].class_id for i in unmatched_tracks_idx], dtype=np.int64)

			matches2, unmatched_tracks_local, _ = _greedy_iou_match(
				track_boxes=rem_track_boxes,
				det_boxes=low_dets.boxes_xyxy,
				iou_threshold=self.second_match_threshold,
				track_labels=rem_track_labels if self.class_aware else None,
				det_labels=low_dets.labels if self.class_aware else None,
			)

			matched_global_tracks = set()
			for local_trk_idx, low_det_idx in matches2:
				global_trk_idx = unmatched_tracks_idx[local_trk_idx]
				trk = self._tracks[global_trk_idx]
				trk.bbox_xyxy = low_dets.boxes_xyxy[low_det_idx].copy()
				trk.score = float(low_dets.scores[low_det_idx])
				trk.class_id = int(low_dets.labels[low_det_idx])
				trk.hits += 1
				trk.time_since_update = 0
				matched_global_tracks.add(global_trk_idx)

			unmatched_tracks_idx = [
				unmatched_tracks_idx[i] for i in unmatched_tracks_local
			]

		# Create tracks from unmatched high-confidence detections only.
		for det_idx in unmatched_high_idx:
			self._new_track(
				bbox=high_dets.boxes_xyxy[det_idx],
				score=float(high_dets.scores[det_idx]),
				cls=int(high_dets.labels[det_idx]),
			)

		self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

		visible = [
			t
			for t in self._tracks
			if t.time_since_update == 0 and t.hits >= self.min_hits
		]

		if visible:
			return TrackedDetections(
				boxes_xyxy=np.asarray([t.bbox_xyxy for t in visible], dtype=np.float32),
				scores=np.asarray([t.score for t in visible], dtype=np.float32),
				labels=np.asarray([t.class_id for t in visible], dtype=np.int64),
				track_ids=np.asarray([t.track_id for t in visible], dtype=np.int64),
			)

		return TrackedDetections(
			boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
			scores=np.zeros((0,), dtype=np.float32),
			labels=np.zeros((0,), dtype=np.int64),
			track_ids=np.zeros((0,), dtype=np.int64),
		)


def _xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
	if len(boxes_xyxy) == 0:
		return np.zeros((0, 4), dtype=np.float32)
	x1 = boxes_xyxy[:, 0]
	y1 = boxes_xyxy[:, 1]
	x2 = boxes_xyxy[:, 2]
	y2 = boxes_xyxy[:, 3]
	return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)


def _run_yolo_batch(
	model: YOLO,
	image_paths: Sequence[Path],
	device: Optional[str],
	imgsz: int,
	conf: float,
	nms_iou: float,
) -> List[Detections]:
	results = model.predict(
		source=[str(p) for p in image_paths],
		device=device,
		imgsz=imgsz,
		conf=conf,
		iou=nms_iou,
		verbose=False,
		stream=False,
	)

	out: List[Detections] = []
	for r in results:
		if r.boxes is None or len(r.boxes) == 0:
			out.append(
				Detections(
					boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
					scores=np.zeros((0,), dtype=np.float32),
					labels=np.zeros((0,), dtype=np.int64),
				)
			)
			continue

		out.append(
			Detections(
				boxes_xyxy=r.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
				scores=r.boxes.conf.detach().cpu().numpy().astype(np.float32),
				labels=r.boxes.cls.detach().cpu().numpy().astype(np.int64),
			)
		)
	return out


def _make_tracker(args: argparse.Namespace):
	if args.tracker == "iou":
		return IoUTracker(
			iou_threshold=args.iou_track_threshold,
			max_age=args.iou_max_age,
			min_hits=args.iou_min_hits,
			class_aware=args.class_aware_tracking,
		)
	return ByteTrackStyleTracker(
		high_threshold=args.bt_high_threshold,
		low_threshold=args.bt_low_threshold,
		match_threshold=args.bt_match_threshold,
		second_match_threshold=args.bt_second_match_threshold,
		max_age=args.bt_max_age,
		min_hits=args.bt_min_hits,
		class_aware=args.class_aware_tracking,
	)


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
		return {
			"AP@[0.50:0.95]": 0.0,
			"AP50": 0.0,
			"AP75": 0.0,
			"AP_small": 0.0,
			"AP_medium": 0.0,
			"AP_large": 0.0,
			"AR1": 0.0,
			"AR10": 0.0,
			"AR100": 0.0,
			"AR_small": 0.0,
			"AR_medium": 0.0,
			"AR_large": 0.0,
		}

	coco_dt = coco_gt.loadRes(cast(Any, coco_pred_dets))
	coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	stats = coco_eval.stats.tolist()
	keys = [
		"AP@[0.50:0.95]",
		"AP50",
		"AP75",
		"AP_small",
		"AP_medium",
		"AP_large",
		"AR1",
		"AR10",
		"AR100",
		"AR_small",
		"AR_medium",
		"AR_large",
	]
	return {k: float(v) for k, v in zip(keys, stats)}


def _evaluate_tracking_mot(
	accumulators: List[Any],
	seq_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
	if len(accumulators) == 0:
		return {}, {}

	metrics = [
		"idf1",
		"idp",
		"idr",
		"recall",
		"precision",
		"num_objects",
		"mostly_tracked",
		"partially_tracked",
		"mostly_lost",
		"num_false_positives",
		"num_misses",
		"num_switches",
		"num_fragmentations",
		"mota",
		"motp",
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


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
	output_dir = Path(args.output_dir)
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	logger = setup_logging(output_dir, args.log_level)

	data_cfg_path = Path(args.data_config)
	data_cfg = _read_data_config(data_cfg_path)
	class_names = _load_class_names(data_cfg)
	images_root, labels_root = _resolve_split_paths(data_cfg, args.split)

	logger.info("Loading dataset split '%s' from %s", args.split, images_root)
	frames_by_seq = _index_frames_constant_size(images_root, labels_root, logger)

	seq_ids = sorted(frames_by_seq.keys())
	if args.max_sequences is not None:
		seq_ids = seq_ids[: args.max_sequences]
		frames_by_seq = {k: frames_by_seq[k] for k in seq_ids}

	total_frames = sum(len(v) for v in frames_by_seq.values())
	logger.info("Indexed %d sequences and %d frames", len(seq_ids), total_frames)

	if total_frames == 0:
		raise RuntimeError("No valid frames were found for evaluation.")

	logger.info("Loading YOLO model: %s", args.model)
	model = YOLO(args.model)

	# COCO GT containers
	coco_gt = {
		"info": {"description": "YOLO video split converted to COCO for evaluation"},
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
	total_gt_boxes = 0
	total_pred_boxes = 0
	tracker_outputs: Dict[str, Any] = {}

	accumulators: List[Any] = []
	mot_seq_names: List[str] = []

	eval_start = time.perf_counter()
	infer_time = 0.0

	for seq_idx, seq_id in enumerate(seq_ids, start=1):
		frames = frames_by_seq[seq_id]
		tracker = _make_tracker(args)
		# We do not pass an explicit frameid in acc.update(...), so auto_id must be enabled.
		acc = mm.MOTAccumulator(auto_id=True)

		if args.save_tracks:
			tracker_outputs[seq_id] = []

		logger.info("[%d/%d] Evaluating sequence %s (%d frames)", seq_idx, len(seq_ids), seq_id, len(frames))

		for batch_start in range(0, len(frames), args.batch_size):
			batch_frames = frames[batch_start : batch_start + args.batch_size]
			image_paths = [fr.image_path for fr in batch_frames]

			t0 = time.perf_counter()
			batch_dets = _run_yolo_batch(
				model=model,
				image_paths=image_paths,
				device=args.device,
				imgsz=args.imgsz,
				conf=args.conf,
				nms_iou=args.nms_iou,
			)
			infer_time += time.perf_counter() - t0

			for frame, det in zip(batch_frames, batch_dets):
				gt = load_ground_truth(frame)

				coco_gt["images"].append(
					{
						"id": frame.image_id,
						"file_name": str(frame.image_path),
						"width": frame.width,
						"height": frame.height,
					}
				)

				gt_xywh = _xyxy_to_xywh(gt.boxes_xyxy)
				for i in range(len(gt_xywh)):
					bbox = gt_xywh[i]
					area = float(max(0.0, bbox[2]) * max(0.0, bbox[3]))
					coco_gt["annotations"].append(
						{
							"id": ann_id,
							"image_id": frame.image_id,
							"category_id": int(gt.labels[i]),
							"bbox": [float(x) for x in bbox.tolist()],
							"area": area,
							"iscrowd": 0,
						}
					)
					ann_id += 1

				pred_xywh = _xyxy_to_xywh(det.boxes_xyxy)
				for i in range(len(pred_xywh)):
					coco_pred.append(
						{
							"image_id": frame.image_id,
							"category_id": int(det.labels[i]),
							"bbox": [float(x) for x in pred_xywh[i].tolist()],
							"score": float(det.scores[i]),
						}
					)

				tracked = tracker.update(det)

				# Tracking metric association with IoU thresholding.
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
					tracker_outputs[seq_id].append(
						{
							"frame_idx": frame.frame_idx,
							"image_path": str(frame.image_path),
							"boxes_xyxy": tracked.boxes_xyxy.tolist(),
							"scores": tracked.scores.tolist(),
							"labels": tracked.labels.tolist(),
							"track_ids": tracked.track_ids.tolist(),
						}
					)

				total_gt_boxes += int(len(gt.boxes_xyxy))
				total_pred_boxes += int(len(det.boxes_xyxy))

		accumulators.append(acc)
		mot_seq_names.append(seq_id)

	total_time = time.perf_counter() - eval_start

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
			"tracker": args.tracker,
			"data_config": str(data_cfg_path),
			"split": args.split,
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
		"config": vars(args),
	}

	if args.save_tracks:
		tracks_path = output_dir / f"tracks_{timestamp}.json"
		with open(tracks_path, "w") as f:
			json.dump(tracker_outputs, f)
		logger.info("Saved per-frame tracks to %s", tracks_path)
		report["tracking"]["tracks_file"] = str(tracks_path)

	if args.report_file is not None:
		report_path = Path(args.report_file)
		report_path.parent.mkdir(parents=True, exist_ok=True)
	else:
		report_path = output_dir / f"report_{timestamp}.json"

	with open(report_path, "w") as f:
		json.dump(report, f, indent=2)

	logger.info("Report written to %s", report_path)
	logger.info("Detection AP50-95: %.4f", detection_metrics.get("AP@[0.50:0.95]", 0.0))
	logger.info("Tracking IDF1: %.4f", tracking_overall.get("idf1", 0.0))
	logger.info("Tracking MOTA: %.4f", tracking_overall.get("mota", 0.0))
	logger.info("Overall FPS: %.2f", timing["overall_fps"])

	return report


def main() -> None:
	args = parse_args()
	evaluate(args)


if __name__ == "__main__":
	main()
