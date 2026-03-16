# PROGRESS

## 2026-03-16

- Enforced **strict equal-weight fusion** across all stage paths in `cnnSearch/models/supernet.py` (removed preferred-path weighting behavior).
- Added a new **multi-path stem selector** (`SlimStemSelector` + `SlimStemPath`) with three fused stem operator families.
- Extended architecture schema and JSON IO for stem selection metadata:
  - `SearchSpaceConfig.stemPathOptions`,
  - `ArchitectureConfig.stemPathIndex`,
  - backward-compatible loading in `cnnSearch/architecture_io.py`.
- Updated subnet materialization in `cnnSearch/models/subnet.py`:
  - standalone subnet now builds a concrete stem variant using `stemPathIndex`,
  - extraction copies weights from the corresponding supernet stem path.
- Fixed script compatibility after schema extension:
  - added `stemPathOptions` to search-space reconstruction in `cnnSearch/train_supernet.py` and `cnnSearch/evaluate_candidate.py`,
  - propagated sampled `stemPathIndex` through `cnnSearch/trainer.py`.
- Updated architecture documentation in `cnnSearch/models/SUPERNET.md` for equal path fusion and multi-path stem behavior.
- Validation completed:
  - `pytest -q tests/test_cnn_search_supernet.py` → `2 passed`,
  - two-step dummy optimization smoke run with random tensors executed successfully.

## 2026-03-15 (iteration 2)

- Updated supernet stage mechanism from single selected path to **all-path fused execution** per stage.
- Expanded path families to five profiles including exotic operators:
  - large-kernel + SE path,
  - dilated + SE path.
- Extended search-space metadata with path operator controls (`pathDilations`, `pathUseSE`, `pathMinKernelSizes`, `pathNames`) and broader kernel options (`3/5/7`).
- Updated supernet block internals with dilation-aware dynamic convolutions and optional squeeze-excitation.
- Updated subnet extraction and standalone subnet construction to preserve selected path dilation/SE/min-kernel behavior.
- Improved metric robustness for small-class datasets by clamping top-k calculations.
- Added `--disableCheckpointing` to training script to support smoke runs in low-disk environments.
- Performed required runtime validation:
  - unit smoke tests: `pytest -q tests/test_cnn_search_supernet.py` (passed),
  - training script smoke run on dummy ImageFolder dataset for multiple batches on CPU (completed with checkpointing disabled).

## 2026-03-15

- Expanded `cnnSearch` supernet from single-path slimmable ResNet to a **multi-path stage selector** architecture.
- Added path-level search controls to `ArchitectureConfig` and `SearchSpaceConfig`:
  - `stagePathIndices`, `stageKernelSizes`, `stageExtraStrides`, `enableAuxiliaryHeads`.
  - search-space options for path IDs, kernel sizes, extra strides, and path depth/width multipliers.
- Implemented dynamic multi-kernel support in `SlimConv2d` using center-cropped kernel slicing.
- Added `SlimStageSelector` and `SlimStagePath` in `cnnSearch/models/supernet.py`.
- Added stage-level auxiliary classification heads and forward output of auxiliary logits for deep supervision.
- Updated training stack:
  - auxiliary-loss support in `cnnSearch/trainer.py`.
  - `--auxiliaryLossWeight` option and updated search-space construction in `cnnSearch/train_supernet.py`.
- Updated candidate evaluation config construction in `cnnSearch/evaluate_candidate.py`.
- Refactored subnet extraction (`cnnSearch/models/subnet.py`) to extract selected path/kernel/stride/depth into a standalone subnet with stage projection.
- Updated architecture JSON loading to be backward-compatible with old configs (`cnnSearch/architecture_io.py`).
- Updated tests for new forward signature and auxiliary outputs.
- Executed tests successfully:
  - `pytest -q tests/test_cnn_search_supernet.py`
  - Result: 2 passed.

## 2026-03-12

- Implemented `cnnSearch` supernet training stack for ResNet-family NAS.
- Added search-space definition with tunable depth, channel multipliers, input resolution, and output stride.
- Implemented dynamic ResNet supernet (`cnnSearch/models/supernet.py`) with weight sharing across candidate subnetworks.
- Implemented subnet extraction utility (`cnnSearch/models/subnet.py`) to materialize standalone models from supernet weights.
- Added ImageFolder dataset pipeline and train/eval augmentation (`cnnSearch/data.py`, `cnnSearch/augmentations.py`).
- Added single-node multi-GPU DDP training entrypoint (`cnnSearch/train_supernet.py`).
- Added training/evaluation utilities, checkpointing, and JSONL metric logging (`cnnSearch/trainer.py`).
- Added candidate evaluation script and resource profiling for fitness metrics (`cnnSearch/evaluate_candidate.py`, `cnnSearch/profiling.py`).
- Added architecture I/O helper for JSON serialization (`cnnSearch/architecture_io.py`).
- Added smoke tests for supernet forward, subnet extraction, and ImageFolder auto-split loader (`tests/test_cnn_search_supernet.py`).
- Executed tests successfully:
  - `pytest -q tests/test_cnn_search_supernet.py`
  - Result: 2 passed.
