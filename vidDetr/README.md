# VideoDETR — Video Object Detection and Tracking with Transformers

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
</p>

VideoDETR extends Facebook's
[DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) from
single-image object detection to **joint video object detection and
multi-object tracking** in a single end-to-end architecture. The model
processes short video clips, detects objects in every frame, and
simultaneously learns tracking embeddings that associate the same object
across frames — all without hand-crafted post-processing such as NMS or
IoU-based trackers.

---

## Table of Contents

1. [Background — DETR in a Nutshell](#1-background--detr-in-a-nutshell)
2. [What VideoDETR Changes](#2-what-videodetr-changes)
3. [Architecture Overview](#3-architecture-overview)
4. [Repository Structure](#4-repository-structure)
5. [Dependencies](#5-dependencies)
6. [Installation](#6-installation)
7. [Dataset Preparation](#7-dataset-preparation)
8. [Training](#8-training)
9. [Inference & Visualisation](#9-inference--visualisation)
10. [Testing](#10-testing)
11. [File-by-File Reference](#11-file-by-file-reference)
12. [Extending VideoDETR](#12-extending-videodetr)
13. [Key Areas for Improvement](#13-key-areas-for-improvement)
14. [Configuration Reference](#14-configuration-reference)
15. [FAQ & Troubleshooting](#15-faq--troubleshooting)
16. [Citation](#16-citation)
17. [License](#17-license)

---

## 1. Background — DETR in a Nutshell

DETR (Carion et al., ECCV 2020) reframes object detection as a direct
**set prediction** problem:

1. A **CNN backbone** (e.g. ResNet-50) extracts a spatial feature map from
   the input image.
2. **Sinusoidal positional encodings** are added to the feature map so
   the transformer knows *where* each feature token is.
3. A standard **Transformer encoder** refines the feature tokens through
   global self-attention.
4. A **Transformer decoder** takes a fixed set of *learnable object
   queries* and, via cross-attention into the encoder memory, produces
   one output embedding per query.
5. Two lightweight heads on each output embedding predict (a) the class
   distribution (including a *no-object* ∅ class) and (b) the bounding
   box coordinates (centre-x, centre-y, width, height — all normalised).
6. **Hungarian matching** finds the optimal bipartite assignment between
   predictions and ground-truth objects, and the loss (classification +
   L1 + GIoU) is computed only on matched pairs. No NMS or anchor tuning
   is needed.

### Limitations of DETR for Video

DETR processes frames **independently**. To track objects across a video
one must bolt on a separate tracker (e.g. DeepSORT, ByteTrack) as a
post-processing step, which:

* breaks end-to-end differentiability,
* cannot leverage temporal context during detection,
* introduces additional hyper-parameters and failure modes.

---

## 2. What VideoDETR Changes

VideoDETR modifies DETR to natively handle **multi-frame video clips**.
The key innovations are:

| Feature | DETR | VideoDETR |
|---------|------|-----------|
| Input | Single image | Clip of *N* frames |
| Queries | *Q* queries | *N × Q* queries (partitioned per frame) |
| Positional encoding | Spatial only | Spatial + **temporal** encoding |
| Decoder paths | Single path | **Split decoder** — shared layers + separate detection & tracking paths |
| Tracking | None (post-hoc) | **Tracking head** trained with supervised contrastive loss |
| Training signal | Class + box | Class + box + **contrastive tracking** + **label denoising** (DN-DETR) |
| Duplicate suppression | None | **IoU-based duplicate suppression loss** |
| Regularisation | Dropout | Dropout + **EMA** + **head dropout** + **stochastic depth** + per-epoch hyper-parameter scheduling |

### Novel Components in Detail

**Temporal Position Encoding** (`models/temporal_encoding.py`).
Sinusoidal or learned embeddings indexed by frame position are *added*
to the spatial positional encoding from the backbone. This lets the
transformer distinguish tokens from different frames after they are
concatenated into a single sequence.

**Split Decoder Architecture** (`models/video_detr.py`).
The transformer decoder is split into *shared layers* (serving both
tasks) and *dedicated layers* (separate parameters for detection vs.
tracking). The tracking path receives **detached** hidden states from
the shared layers so that tracking gradients do *not* flow into the
shared representation or the detection path, preventing task
interference.

**Tracking Head** (`models/tracking_head.py`).
A 3-layer MLP with L2-normalised output that produces tracking
embeddings. Same-object embeddings across frames are pulled together by
a **supervised contrastive loss** (SupCon, Khosla et al., NeurIPS 2020).

**Label Denoising** (`models/denoising.py`).
Following DN-DETR / DINO, extra "denoising queries" are prepended to the
decoder input during training. These queries carry noised versions of
ground-truth labels and boxes, and their loss is computed *directly*
(no Hungarian matching), providing a stronger gradient signal that
accelerates convergence.

**Duplicate Suppression Loss** (`losses/video_criterion.py`).
An auxiliary loss that penalises pairs of same-frame predictions with
high IoU and high confidence, directly discouraging the model from
placing multiple boxes on the same object.

---

## 3. Architecture Overview

```
Input: N frames × [3, H, W]
         │
         ▼
┌─────────────────────┐
│  Shared CNN Backbone │  (ResNet-50, frozen BN)
│  + Spatial Pos. Enc. │
└────────┬────────────┘
         │  N × [B, C, H', W']
         ▼
┌──────────────────────┐
│  + Temporal Pos. Enc.│  (sine or learned, per-frame)
└────────┬─────────────┘
         │  Concatenate across frames → [B, N·H'·W', C]
         ▼
┌──────────────────────┐
│  Transformer Encoder │  (6 layers, self-attention)
└────────┬─────────────┘
         │  Memory: [B, N·H'·W', C]
         ▼
┌──────────────────────────────────────────────────────┐
│  Shared Decoder Layers (L − K layers)                │
│  Queries: N·Q matching + DN denoising (training)     │
└──────────────┬───────────────┬───────────────────────┘
               │               │ (detach)
        ┌──────┴──────┐  ┌────┴────────┐
        │  Detection  │  │  Tracking   │
        │  Dec. Path  │  │  Dec. Path  │
        │  (K layers) │  │  (K layers) │
        └──────┬──────┘  └─────┬───────┘
               │               │
        ┌──────┴──────┐  ┌────┴────────┐
        │ Class Head  │  │ Tracking    │
        │ Box Head    │  │ Head (MLP)  │
        └─────────────┘  └─────────────┘

Outputs per frame:
  • pred_logits  [B, N·Q, C+1]
  • pred_boxes   [B, N·Q, 4]
  • pred_tracking [B, N·Q, D_track]
```

---

## 4. Repository Structure

```
vidDetr/
├── __init__.py                  # Package entry point; exports buildVideoDETR, buildVideoDataset
├── main.py                      # Training script (args, loop, checkpointing)
├── engine.py                    # trainOneEpoch, evaluate, ModelEMA, debug visualisation
├── inference.py                 # Interactive inference & visualisation on YOLO-format test data
├── tao_inference.py             # Headless inference on TAO dataset (writes MP4 videos)
├── eval_tao_dataset.py          # Visualise TAO dataset loader samples
├── test_video_detr.py           # Unit tests for all components (CPU, synthetic data)
├── logging_utils.py             # setupLogging + MetricTracker (CSV-based metric persistence)
├── data.yaml                    # Example dataset config (COCO 80 classes, YOLO layout)
│
├── models/
│   ├── __init__.py              # Exports: VideoDETR, buildVideoDETR, TemporalPositionEncoding, …
│   ├── video_detr.py            # Core VideoDETR model (backbone → encoder → split decoder → heads)
│   ├── temporal_encoding.py     # Sinusoidal & learned temporal positional encodings
│   ├── tracking_head.py         # Tracking MLP + TrackingHeadWithMemory (experimental)
│   ├── denoising.py             # DN-DETR / DINO label denoising generator
│   └── detr/                    # Vendored DETR base components (self-contained)
│       ├── __init__.py
│       ├── backbone.py          # ResNet backbone with FrozenBatchNorm2d + positional encoding
│       ├── position_encoding.py # Sine & learned 2-D spatial positional encodings
│       └── transformer.py       # Vanilla Transformer encoder-decoder
│
├── datasets/
│   ├── __init__.py              # Exports: VideoSequenceDataset, TaoDataset, build*, collate*
│   ├── video_dataset.py         # YOLO-format video sequence dataset with cache & transforms
│   ├── tao_dataset.py           # TAO benchmark dataset with multi-window sampling
│   └── transforms.py            # DETR-compatible image + bbox augmentation transforms
│
├── losses/
│   ├── __init__.py              # Exports: VideoCriterion, buildVideoCriterion, PostProcess
│   ├── video_criterion.py       # Full loss: Hungarian matching, focal/CE, L1, GIoU, DN, dup-suppression
│   └── contrastive_loss.py      # SupCon loss + HardNegativeContrastiveLoss variant
│
└── util/
    ├── __init__.py
    ├── box_ops.py               # Box format conversions, IoU, Generalized IoU
    └── misc.py                  # NestedTensor, distributed helpers, reduce_dict, accuracy, …
```

---

## 5. Dependencies

### Core (required)

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| Python | 3.8 | Language runtime |
| PyTorch | ≥ 1.9.0 | Deep learning framework |
| torchvision | ≥ 0.10.0 | ResNet backbone, image transforms |
| scipy | any | `linear_sum_assignment` for Hungarian matching |
| Pillow | any | Image I/O |
| PyYAML | any | Dataset config parsing (`data.yaml`) |
| NumPy | any | Array operations |
| OpenCV (`cv2`) | any | Debug frame visualisation, inference drawing |

### Optional

| Package | Purpose |
|---------|---------|
| `safe_gpu` | Auto-claim free GPUs on shared clusters (used in `main.py` and `tao_inference.py`) |
| `motmetrics` | Full MOT evaluation (MOTA, IDF1) — stub exists in `engine.py` |
| `pycocotools` | COCO-style evaluation (if integrating COCO mAP) |

### Install

```bash
pip install torch torchvision scipy pillow pyyaml numpy opencv-python
# Optional
pip install safe_gpu motmetrics pycocotools
```

---

## 6. Installation

```bash
# Clone the repository
git clone https://github.com/xmihol00/video_detr.git
cd video_detr

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision scipy pillow pyyaml numpy opencv-python

# Verify installation
python -m vidDetr.test_video_detr
```

The `vidDetr/` package is **self-contained** — it has no import
dependencies on files outside the `vidDetr/` directory and can be
extracted, published, or pip-installed independently.

---

## 7. Dataset Preparation

VideoDETR supports two dataset formats:

### 7.1 YOLO-format Video Sequences (default)

Organise your data as:

```
dataset_root/
├── train/
│   ├── images/
│   │   ├── seq_000001_frame_0000.jpg
│   │   ├── seq_000001_frame_0001.jpg
│   │   └── ...
│   └── labels/
│       ├── seq_000001_frame_0000.txt
│       ├── seq_000001_frame_0001.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

**Label format** (one line per object, YOLO normalised coordinates):
```
class_id  centre_x  centre_y  width  height
```

**Tracking correspondence**: Objects on the **same line number** across
label files of a sequence represent the **same physical object**. This
implicit line-based tracking ID is the supervision signal for the
contrastive tracking loss.

Create a `data.yaml` pointing to your dataset:

```yaml
train: /path/to/train/images
val:   /path/to/val/images

names:
  0: person
  1: car
  # ...
```

### 7.2 TAO Dataset

The [TAO (Tracking Any Object)](https://taodataset.org/) benchmark is
supported natively. Point `--taoDataRoot` to the TAO root directory:

```
<taoDataRoot>/
├── annotations/
│   ├── train.json
│   └── validation.json
└── frames/
    ├── train/<dataset>/<video>/<frame>.jpg
    └── val/<dataset>/<video>/<frame>.jpg
```

---

## 8. Training

### Single GPU

```bash
python -m vidDetr.main \
    --dataConfig vidDetr/data.yaml \
    --numFrames 4 \
    --queriesPerFrame 30 \
    --epochs 100 \
    --batchSize 2 \
    --lr 1e-4 \
    --lrBackbone 1e-5 \
    --backbone resnet50 \
    --useFocalLoss \
    --useDnDenoising \
    --useEma \
    --outputDir vidDetr_weights/
```

### Multi-GPU (Distributed Data Parallel)

```bash
torchrun --nproc_per_node=4 -m vidDetr.main \
    --dataConfig vidDetr/data.yaml \
    --numFrames 4 \
    --batchSize 2 \
    --outputDir vidDetr_weights/
```

### TAO Dataset Training

```bash
python -m vidDetr.main \
    --taoDataRoot /path/to/tao/dataset \
    --taoMaxCategories 200 \
    --numFrames 5 \
    --epochs 50 \
    --useFocalLoss \
    --useDnDenoising
```

### Transfer Learning from Pretrained DETR

VideoDETR can initialise from standard DETR weights. The loading logic
automatically remaps decoder layer keys to accommodate the split
decoder architecture:

```bash
python -m vidDetr.main \
    --pretrainedDetr detr-r50-e632da11.pth \
    --freezePretrained \
    --unfreezeAfterEpochs 3 \
    --dataConfig vidDetr/data.yaml
```

### Resuming Training

```bash
python -m vidDetr.main \
    --resume vidDetr_weights/checkpoint_latest.pth \
    --dataConfig vidDetr/data.yaml
```

### Key Training Features

* **Per-epoch hyper-parameter scheduling**: Most loss coefficients,
  dropout rates, and noise scales accept a list of values
  (`--eosCoef 0.15 0.20 0.25 0.30`). The scheduler picks
  `schedule[min(epoch, len-1)]`.
* **Gradient accumulation**: `--accumSteps N` accumulates gradients over
  *N* batches before each optimiser step, effectively multiplying the
  batch size.
* **EMA**: Exponential moving average of model weights
  (`--useEma --emaDecay 0.9997`), used at evaluation time.
* **Freeze/unfreeze strategy**: `--freezePretrained` freezes loaded
  weights for the first few epochs, then unfreezes for full fine-tuning.
* **Debug frames**: Saved each batch to `debug_frames/` showing GT
  (green dashed) vs predicted (red solid) boxes.
* **Train+Val merge**: `--mergeTrainVal` combines both splits for
  maximum data utilisation (checkpoint saved every epoch, no validation
  loop).

---

## 9. Inference & Visualisation

### Interactive YOLO-format Inference

```bash
python -m vidDetr.inference \
    --modelPath vidDetr_weights/video_detr_best.pth \
    --testDir test \
    --dataConfig vidDetr/data.yaml \
    --confidence 0.4 \
    --trackingThreshold 0.4
```

Keyboard controls: `→`/`d`/`Space` next frame, `←`/`a` previous,
`n`/`p` next/previous sequence, `q`/`Esc` quit.

### Headless TAO Inference (MP4 output)

```bash
python -m vidDetr.tao_inference \
    --modelPath tao_weights/video_detr_best.pth \
    --taoDataRoot /path/to/tao/dataset \
    --numVideos 10 \
    --confidence 0.4
```

Writes annotated MP4 videos to `gt_vs_pred/`.

### Saving Frames to Disk (no GUI)

```bash
python -m vidDetr.inference \
    --modelPath video_detr_best.pth \
    --testDir test \
    --saveDir inference_results/
```

---

## 10. Testing

Run the built-in test suite (CPU, synthetic data — no dataset needed):

```bash
python -m vidDetr.test_video_detr
```

This tests:
* Temporal position encoding (sine and learned)
* Tracking head forward pass
* Label denoising generator + attention mask validity
* Supervised contrastive loss
* Full VideoDETR model forward pass with split decoder
* Full pipeline integration (model + criterion + backprop)

---

## 11. File-by-File Reference

### Top-Level Scripts

| File | Description |
|------|-------------|
| `__init__.py` | Package entry point. Exports `buildVideoDETR` and `buildVideoDataset`. |
| `main.py` | Complete training script: argument parsing (70+ hyperparameters with per-epoch scheduling), model/dataset/optimizer construction, distributed training setup, checkpointing with best-model tracking, pretrained weight loading with decoder key remapping, freeze/unfreeze strategy. |
| `engine.py` | Training and evaluation loops. `trainOneEpoch()` handles gradient accumulation, EMA updates, debug frame saving, and structured CSV logging. `evaluate()` runs loss-only validation. Also provides `ModelEMA`, `associateDetectionsAcrossFrames()` (greedy embedding-based tracker), and `computeTrackingMetrics()` (MOT metrics stub). |
| `inference.py` | Interactive inference script for YOLO-format test sequences. Loads a checkpoint, runs sliding-window inference, performs NMS and cross-frame track association using cosine similarity of tracking embeddings, and visualises GT vs predictions via OpenCV. |
| `tao_inference.py` | Headless TAO inference script. Randomly samples TAO validation videos, runs the model, associates tracks, and writes annotated MP4 videos with GT (dashed green) and predictions (solid coloured). |
| `eval_tao_dataset.py` | TAO dataset loader evaluation. Samples random clips and saves annotated frames with bounding boxes and segmentation polygons. |
| `test_video_detr.py` | Comprehensive unit tests for all model components using synthetic data on CPU. |
| `logging_utils.py` | `setupLogging()` configures a named Python logger with console + file handlers. `MetricTracker` accumulates per-batch metrics and writes epoch summaries to CSV files. |
| `data.yaml` | Example dataset configuration with COCO 80 classes and YOLO-format paths. |

### `models/`

| File | Description |
|------|-------------|
| `video_detr.py` | **Core model**. `VideoDETR` processes *N* frames through a shared backbone, adds temporal encoding, concatenates features, runs the transformer encoder, then splits the decoder into detection and tracking paths. Detection heads predict class logits and cxcywh boxes; the tracking head produces L2-normalised embeddings. The `buildVideoDETR()` factory constructs the full model + criterion + post-processors. |
| `temporal_encoding.py` | `TemporalPositionEncodingSine` (fixed sinusoidal) and `TemporalPositionEncodingLearned` (trainable embedding table). The wrapper `TemporalPositionEncoding` combines temporal and spatial positional encodings by addition. |
| `tracking_head.py` | `TrackingHead`: 3-layer MLP → L2 normalisation. `TrackingHeadWithMemory`: experimental variant with a multi-head attention memory bank for temporal consistency during inference. |
| `denoising.py` | `DenoisingGenerator`: creates noised copies of GT labels and boxes, generates content/positional embeddings for denoising queries, and builds an attention mask preventing information leakage between DN groups and matching queries. Per-frame, per-group structure respects VideoDETR's frame-based query organisation. |
| `detr/` | Vendored (self-contained) copies of the original DETR components: `backbone.py` (ResNet + FrozenBatchNorm2d), `position_encoding.py` (2-D sine/learned spatial encoding), `transformer.py` (standard encoder-decoder). |

### `datasets/`

| File | Description |
|------|-------------|
| `video_dataset.py` | `VideoSequenceDataset`: loads YOLO-format video sequences. Parses `seq_XXXXXX_frame_XXXX` filenames, discovers sequences, caches index to disk, uses stratified frame sampling (training) or uniform sampling (validation). `makeVideoTransforms()` builds augmentation pipelines. `videoCollateFn()` reorganises batches from `[B, N]` to `[N, B]` and creates `NestedTensor`s. |
| `tao_dataset.py` | `TaoDataset`: loads the TAO benchmark. Parses COCO-style JSON annotations, builds a multi-window index (overlapping temporal windows per video), supports category filtering (`maxCategoriesUsed`), variable-stride frame sampling, and disk caching. |
| `transforms.py` | DETR-compatible augmentation transforms that jointly transform images and bounding boxes: `RandomHorizontalFlip`, `RandomResize`, `RandomSizeCrop`, `ColorJitter`, `RandomGrayscale`, `RandomErasing`, `Normalize`, `Compose`, etc. |

### `losses/`

| File | Description |
|------|-------------|
| `video_criterion.py` | `VideoCriterion`: the complete training loss. Performs per-frame Hungarian matching via `VideoHungarianMatcher`, then computes: sigmoid focal loss (or cross-entropy), L1 + GIoU box loss, supervised contrastive tracking loss, duplicate suppression loss, and DN-DETR denoising losses. Auxiliary losses are computed at each intermediate decoder layer. `PostProcess` converts outputs to evaluation format. `buildVideoCriterion()` is the factory function. |
| `contrastive_loss.py` | `SupervisedContrastiveLoss`: pulls same-track embeddings together, pushes different-track embeddings apart. `HardNegativeContrastiveLoss`: variant that mines the hardest negatives for more informative gradients. |

### `util/`

| File | Description |
|------|-------------|
| `box_ops.py` | Bounding box utilities: `box_cxcywh_to_xyxy`, `box_xyxy_to_cxcywh`, `box_area`, `box_iou`, `generalized_box_iou`. |
| `misc.py` | General utilities: `NestedTensor` (tensor + padding mask), `nested_tensor_from_tensor_list`, distributed training helpers (`reduce_dict`, `get_world_size`, `init_distributed_mode`, etc.), `accuracy`, `interpolate`. |

---

## 12. Extending VideoDETR

### 12.1 Adding a New Backbone

1. Create a new backbone module in `models/detr/backbone.py` or a new file
   under `models/`.
2. Ensure it returns a list of `(NestedTensor, position_encoding)` tuples
   (matching the `Joiner` interface in `backbone.py`).
3. Expose a `num_channels` attribute with the output channel count.
4. Update `build_backbone()` to accept your new architecture name.
5. Pass `--backbone <your_name>` at training time.

**Example**: To add a Swin Transformer backbone:
```python
# models/detr/backbone.py  (or a new file models/swin_backbone.py)
class SwinBackbone(nn.Module):
    def __init__(self, ...):
        ...
        self.num_channels = 768  # output channels of last stage
    def forward(self, tensor_list: NestedTensor):
        # Return features and mask
        ...
```

### 12.2 Adding a New Loss Function

1. Implement your loss as a method on `VideoCriterion` in
   `losses/video_criterion.py`, following the signature:
   ```python
   def lossMyNewLoss(self, outputs, targets, indices, numBoxes, **kwargs) -> Dict[str, Tensor]:
   ```
2. Register it in the `lossMap` dict inside `getLoss()`.
3. Add its weight to `buildVideoCriterion()`:
   ```python
   weightDict['loss_my_new'] = myNewLossCoef
   ```
4. Append `'my_new_loss'` to the `losses` list.
5. Add a `--myNewLossCoef` argument to `main.py`.

### 12.3 Adding a New Dataset

1. Create a new file `datasets/my_dataset.py` with a PyTorch `Dataset`
   class.
2. Each `__getitem__` must return `(images, targets)` where:
   * `images`: list of *N* image tensors `[3, H, W]`
   * `targets`: list of *N* dicts, each with keys `boxes` (xyxy absolute),
     `labels`, `trackIds`, `iscrowd`, `area`, `size`, `origSize`.
3. Implement a `buildMyDataset(args)` factory and a `myCollateFn`.
4. Export them from `datasets/__init__.py`.
5. Add a CLI flag in `main.py` and wire it into the dataset-selection
   logic.

### 12.4 Adding a New Tracking Association Strategy

The current greedy cosine-similarity tracker is in
`engine.py::associateDetectionsAcrossFrames()`. To improve it:

1. Create a new file `tracking/association.py`.
2. Implement your algorithm (e.g. Hungarian on a cost matrix combining
   embedding similarity, IoU, and motion prediction).
3. Call it from `inference.py` instead of the existing greedy tracker.

### 12.5 Adding Per-Epoch Scheduled Hyperparameters

1. Add the argument in `main.py::getArgsParser()` with `nargs='+'`:
   ```python
   parser.add_argument('--myParam', default=[1.0], nargs='+', type=float)
   ```
2. Add `'myParam'` to the `SCHEDULED_PARAM_NAMES` list.
3. Read it in `criterion.updateEpochParams()` or
   `model.updateEpochParams()`.

---

## 13. Key Areas for Improvement

### High Impact

| Area | Current State | Suggested Improvement |
|------|--------------|----------------------|
| **Multi-scale features** | Single feature level (last backbone layer) | Use a Feature Pyramid Network (FPN) or deformable attention (Deformable DETR) to leverage multi-scale features for detecting small objects. |
| **Deformable attention** | Standard O(N²) attention | Replace with deformable attention (Zhu et al.) to reduce memory and enable higher-resolution inputs, critical for video. |
| **Tracking association** | Greedy cosine similarity | Implement Hungarian matching on a combined cost matrix (embedding similarity + IoU + Kalman motion prediction) for robust multi-object tracking. |
| **MOT evaluation** | Stub in `engine.py` | Integrate `motmetrics` or `TrackEval` for proper MOTA, IDF1, HOTA computation. |
| **COCO mAP evaluation** | Not implemented | Add `pycocotools`-based mAP computation in the validation loop. |

### Medium Impact

| Area | Suggested Improvement |
|------|----------------------|
| **Memory-efficient training** | Implement gradient checkpointing for the backbone and encoder to reduce GPU memory, enabling larger batches or higher resolutions. |
| **Temporal attention** | Add explicit temporal self-attention layers in the decoder (attend across frame tokens) instead of relying solely on the encoder's global attention. |
| **Online inference** | Implement a streaming mode that maintains a memory buffer of past-frame features, processing one new frame at a time instead of full clips. |
| **Data augmentation** | Add Mosaic, MixUp, and CutMix augmentations adapted for video (applied consistently across clip frames). |
| **Mixed precision** | Add `torch.cuda.amp` automatic mixed precision for ~2× training speedup. |

### Code Quality

| Area | Suggested Improvement |
|------|----------------------|
| **Type annotations** | Add comprehensive type hints and enable `mypy` / Pylance strict mode. |
| **Configuration management** | Replace argparse with a structured config system (Hydra or dataclasses-based YAML configs). |
| **Unit test coverage** | Add tests for the criterion, dataset loading, augmentation pipeline, and distributed training. |
| **CI/CD** | Add GitHub Actions for linting, testing, and packaging. |
| **Packaging** | Add `pyproject.toml` / `setup.py` for `pip install -e .` support. |
| **CSV metric tracker** | The `_appendCsv` method currently has a `return` statement disabling writes — remove it when ready for production. |

---

## 14. Configuration Reference

Below are the most important command-line arguments. Most numeric
parameters accept a **list** for per-epoch scheduling.

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `resnet50` | CNN backbone (`resnet50`, `resnet101`) |
| `--hiddenDim` | `256` | Transformer hidden dimension |
| `--encLayers` | `6` | Encoder layers |
| `--decLayers` | `6` | Decoder layers |
| `--nheads` | `8` | Attention heads |
| `--dimFeedforward` | `2048` | FFN intermediate dimension |
| `--numFrames` | `4` | Frames per video clip |
| `--queriesPerFrame` | `30` | Object queries per frame |
| `--numTrackingDecLayers` | `1` | Decoder layers dedicated to tracking path |
| `--temporalEncoding` | `sine` | Temporal encoding type (`sine` / `learned`) |
| `--trackingEmbedDim` | `128` | Tracking embedding dimension |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `1e-4` | Base learning rate |
| `--lrBackbone` | `1e-5` | Backbone learning rate |
| `--batchSize` | `32` | Batch size per GPU |
| `--epochs` | `100` | Total epochs |
| `--lrDrop` | `80` | Epoch for LR step decay |
| `--warmupEpochs` | `0` | Linear warmup epochs |
| `--accumSteps` | `1` | Gradient accumulation steps |
| `--clipMaxNorm` | `[0.1]` | Gradient clipping (schedulable) |
| `--dropout` | `[0.15]` | Dropout rate (schedulable) |

### Loss

| Argument | Default | Description |
|----------|---------|-------------|
| `--useFocalLoss` | `False` | Use sigmoid focal loss |
| `--focalAlpha` | `[0.25]` | Focal loss α (schedulable) |
| `--focalGamma` | `[2.0]` | Focal loss γ (schedulable) |
| `--bboxLossCoef` | `[5.0]` | L1 box loss weight (schedulable) |
| `--giouLossCoef` | `[2.0]` | GIoU loss weight (schedulable) |
| `--eosCoef` | `[0.15 ... 0.3]` | No-object weight (schedulable) |
| `--trackingLossCoef` | `[1.0]` | Contrastive tracking loss weight (schedulable) |
| `--contrastiveTemp` | `[0.07]` | Contrastive loss temperature (schedulable) |
| `--dupLossCoef` | `[0.25 ... 0.0]` | Duplicate suppression weight (schedulable) |

### Label Denoising

| Argument | Default | Description |
|----------|---------|-------------|
| `--useDnDenoising` | `False` | Enable DN-DETR denoising |
| `--numDnGroups` | `5` | Denoising groups per batch |
| `--labelNoiseRatio` | `[0.5]` | Label flip probability (schedulable) |
| `--boxNoiseScale` | `[0.4]` | Box noise scale (schedulable) |
| `--dnLossCoef` | `[1.0]` | DN loss multiplier (schedulable) |

### Dataset

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataConfig` | `vidDetr/data.yaml` | Path to dataset YAML |
| `--numClasses` | `80` | Number of object classes |
| `--maxSize` | `384` | Max image dimension |
| `--taoDataRoot` | `None` | TAO dataset root (overrides `--dataConfig`) |

---

## 15. FAQ & Troubleshooting

**Q: I get `RuntimeError: Too many open files` during training.**
A: VideoDETR already sets `torch.multiprocessing.set_sharing_strategy('file_system')`.
If the problem persists, reduce `--numWorkers` or increase your system's
`ulimit -n`.

**Q: Training is very slow / runs out of GPU memory.**
A: Reduce `--maxSize` (e.g. 256), `--numFrames` (e.g. 2), or
`--queriesPerFrame` (e.g. 15). Use `--accumSteps` to maintain effective
batch size with smaller per-GPU batches.

**Q: How do I add a new object class?**
A: Update `data.yaml` with the new class name, set `--numClasses`
accordingly, and re-train. If fine-tuning, the classification head
dimensions will change so the pretrained head weights will be skipped
during loading.

**Q: Can I use this for real-time inference?**
A: The current architecture processes fixed-length clips. For real-time
use, implement a sliding-window or streaming mode (see
[§13 — Key Areas for Improvement](#13-key-areas-for-improvement)).

**Q: How does the `safe_gpu` import work?**
A: `safe_gpu` is an optional cluster utility that claims free GPUs. If
you don't have it installed, remove or comment out the `import safe_gpu`
block in `main.py` and `tao_inference.py`.

---

## 16. Citation

If you use VideoDETR in your research, please cite:

```bibtex
@misc{videodetr2026,
  author       = {David Mihola},
  title        = {{VideoDETR}: Video Object Detection and Tracking with Transformers},
  year         = {2026},
  url          = {https://github.com/xmihol00/video_detr}
}
```

Also cite the original DETR and the techniques VideoDETR builds upon:

```bibtex
@inproceedings{carion2020detr,
  title   = {End-to-End Object Detection with Transformers},
  author  = {Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle = {ECCV},
  year    = {2020}
}

@inproceedings{li2022dndetr,
  title   = {{DN-DETR}: Accelerate {DETR} Training by Introducing Query Denoising},
  author  = {Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
  booktitle = {CVPR},
  year    = {2022}
}

@inproceedings{khosla2020supcon,
  title   = {Supervised Contrastive Learning},
  author  = {Khosla, Prannay and others},
  booktitle = {NeurIPS},
  year    = {2020}
}
```

---

## 17. License

This project is released under the [Apache 2.0 License](../LICENSE).

The vendored DETR components in `models/detr/` and `util/` are derived
from [facebookresearch/detr](https://github.com/facebookresearch/detr),
also licensed under Apache 2.0.
