"""
Model Evaluation Toolkit for ONNX classification models.

This script evaluates a quantized ONNX classification model (e.g., exported via
Model Compression Toolkit) on a folder-structured dataset:

dataset/
  class_a/
	img1.jpg
  class_b/
	img2.png

It reports results to the command line and saves them to:
<model_name>_evaluation.txt
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
	from mct_quantizers import get_ort_session_options
except Exception:
	get_ort_session_options = None


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class DatasetEntry:
	image_path: str
	class_name: str
	class_index: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a quantized ONNX classification model on an image dataset."
	)
	parser.add_argument(
		"--model",
		type=str,
		default="quantized_model.onnx",
		help="Path to quantized ONNX model (default: quantized_model.onnx)",
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default="dataset/",
		help="Path to dataset root with one-level class subfolders (default: dataset/)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1,
		help="Batch size for inference (default: 1)",
	)
	parser.add_argument(
		"--input-height",
		type=int,
		default=256,
		help="Fallback input height if model input is dynamic (default: 256)",
	)
	parser.add_argument(
		"--input-width",
		type=int,
		default=256,
		help="Fallback input width if model input is dynamic (default: 256)",
	)
	parser.add_argument(
		"--mean",
		type=float,
		nargs=3,
		default=[0.485, 0.456, 0.406],
		help="Normalization mean (default: 0.485 0.456 0.406)",
	)
	parser.add_argument(
		"--std",
		type=float,
		nargs=3,
		default=[0.229, 0.224, 0.225],
		help="Normalization std (default: 0.229 0.224 0.225)",
	)
	return parser.parse_args()


def discover_dataset(dataset_root: str) -> Tuple[List[DatasetEntry], List[str]]:
	if not os.path.isdir(dataset_root):
		raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

	class_names = sorted(
		[
			d
			for d in os.listdir(dataset_root)
			if os.path.isdir(os.path.join(dataset_root, d))
		]
	)
	if not class_names:
		raise ValueError(
			f"No class subdirectories found in dataset root: {dataset_root}"
		)

	entries: List[DatasetEntry] = []
	for class_index, class_name in enumerate(class_names):
		class_dir = os.path.join(dataset_root, class_name)
		for file_name in sorted(os.listdir(class_dir)):
			if file_name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
				entries.append(
					DatasetEntry(
						image_path=os.path.join(class_dir, file_name),
						class_name=class_name,
						class_index=class_index,
					)
				)

	if not entries:
		raise ValueError(f"No supported images found under dataset: {dataset_root}")

	return entries, class_names


def resolve_input_size(
	input_shape: Sequence[object], fallback_height: int, fallback_width: int
) -> Tuple[int, int]:
	if len(input_shape) < 4:
		return fallback_height, fallback_width

	height = input_shape[2]
	width = input_shape[3]

	height = height if isinstance(height, int) and height > 0 else fallback_height
	width = width if isinstance(width, int) and width > 0 else fallback_width
	return int(height), int(width)


def preprocess_image(
	image_path: str,
	input_height: int,
	input_width: int,
	mean: np.ndarray,
	std: np.ndarray,
) -> np.ndarray:
	image = Image.open(image_path).convert("RGB")
	image = image.resize((input_width, input_height), Image.BILINEAR)
	image_array = np.asarray(image, dtype=np.float32) / 255.0
	image_array = (image_array - mean) / std
	image_array = np.transpose(image_array, (2, 0, 1))
	return image_array


def batched(items: Sequence[DatasetEntry], batch_size: int) -> List[List[DatasetEntry]]:
	return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def format_metrics_report(
	model_path: str,
	dataset_path: str,
	class_names: Sequence[str],
	total_images: int,
	correct_predictions: int,
	class_totals: Dict[str, int],
	class_correct: Dict[str, int],
	elapsed_seconds: float,
) -> str:
	model_name = os.path.splitext(os.path.basename(model_path))[0]
	accuracy = 100.0 * correct_predictions / total_images if total_images else 0.0

	lines: List[str] = []
	lines.append("=" * 80)
	lines.append("ONNX CLASSIFICATION EVALUATION REPORT")
	lines.append("=" * 80)
	lines.append(f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append(f"Model              : {model_path}")
	lines.append(f"Model name         : {model_name}")
	lines.append(f"Dataset            : {dataset_path}")
	lines.append(f"Number of classes  : {len(class_names)}")
	lines.append(f"Number of images   : {total_images}")
	lines.append(f"Elapsed time (s)   : {elapsed_seconds:.3f}")
	lines.append("-" * 80)
	lines.append(f"Top-1 Accuracy     : {accuracy:.2f}% ({correct_predictions}/{total_images})")
	lines.append("-" * 80)
	lines.append("Per-class accuracy")
	lines.append("-" * 80)
	lines.append(f"{'Class':<30} {'Correct/Total':<20} {'Accuracy':>10}")
	lines.append("-" * 80)

	for class_name in class_names:
		total = class_totals.get(class_name, 0)
		correct = class_correct.get(class_name, 0)
		class_accuracy = 100.0 * correct / total if total else 0.0
		lines.append(f"{class_name:<30} {f'{correct}/{total}':<20} {class_accuracy:>9.2f}%")

	lines.append("=" * 80)
	return "\n".join(lines)


def evaluate_classification_model(args: argparse.Namespace) -> str:
	entries, class_names = discover_dataset(args.dataset)
	class_totals = {name: 0 for name in class_names}
	class_correct = {name: 0 for name in class_names}

	available_providers = ort.get_available_providers()
	preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
	execution_providers = [p for p in preferred_providers if p in available_providers]
	if not execution_providers:
		execution_providers = available_providers

	session_options = None
	if get_ort_session_options is not None:
		session_options = get_ort_session_options()

	try:
		session = ort.InferenceSession(
			args.model,
			sess_options=session_options,
			providers=execution_providers,
		)
	except Exception as exc:
		error_message = str(exc)
		if "mct_quantizers" in error_message or "ActivationPOTQuantizer" in error_message:
			raise RuntimeError(
				"Failed to load model due to missing MCT custom ops in ONNX Runtime. "
				"Install/import `mct_quantizers` in the same environment and use "
				"its session options registration."
			) from exc
		raise
	input_meta = session.get_inputs()[0]
	output_meta = session.get_outputs()[0]
	input_name = input_meta.name
	output_name = output_meta.name

	input_height, input_width = resolve_input_size(
		input_meta.shape, args.input_height, args.input_width
	)
	print(f"Using execution providers: {execution_providers}")
	print(f"Model input tensor: {input_name}, shape={input_meta.shape}")
	print(f"Model output tensor: {output_name}, shape={output_meta.shape}")
	print(f"Preprocessing size: {input_height}x{input_width}")

	mean = np.array(args.mean, dtype=np.float32).reshape(1, 1, 3)
	std = np.array(args.std, dtype=np.float32).reshape(1, 1, 3)

	total_images = len(entries)
	correct_predictions = 0
	start_time = time.perf_counter()

	all_batches = batched(entries, max(1, args.batch_size))
	for batch_index, batch_entries in enumerate(all_batches, start=1):
		batch_images = [
			preprocess_image(
				entry.image_path,
				input_height=input_height,
				input_width=input_width,
				mean=mean,
				std=std,
			)
			for entry in batch_entries
		]

		input_tensor = np.stack(batch_images, axis=0).astype(np.float32)
		logits = session.run([output_name], {input_name: input_tensor})[0]
		logits = np.asarray(logits)
		logits = logits.reshape(logits.shape[0], -1)
		predictions = np.argmax(logits, axis=1)

		for entry, predicted_index in zip(batch_entries, predictions):
			class_totals[entry.class_name] += 1
			if int(predicted_index) == entry.class_index:
				class_correct[entry.class_name] += 1
				correct_predictions += 1

		if batch_index % 20 == 0 or batch_index == len(all_batches):
			print(
				f"Processed {batch_index}/{len(all_batches)} batches "
				f"({min(batch_index * args.batch_size, total_images)}/{total_images} images)"
			)

	elapsed_seconds = time.perf_counter() - start_time

	report_text = format_metrics_report(
		model_path=args.model,
		dataset_path=args.dataset,
		class_names=class_names,
		total_images=total_images,
		correct_predictions=correct_predictions,
		class_totals=class_totals,
		class_correct=class_correct,
		elapsed_seconds=elapsed_seconds,
	)

	return report_text


def save_report(model_path: str, report_text: str) -> str:
	model_name = os.path.splitext(os.path.basename(model_path))[0]
	output_path = os.path.join(os.getcwd(), f"{model_name}_evaluation.txt")
	with open(output_path, "w", encoding="utf-8") as report_file:
		report_file.write(report_text)
		report_file.write("\n")
	return output_path


def main() -> None:
	args = parse_args()
	report_text = evaluate_classification_model(args)
	print("\n" + report_text)
	output_path = save_report(args.model, report_text)
	print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
	main()
