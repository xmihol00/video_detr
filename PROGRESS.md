# PROGRESS

## 2026-03-30

- Improved IMX500 compilation search flow in `cnnSearch/search_compilable_subnets.py`:
  - added `--dv` argument to select/continue a DB file,
  - when `--dv` is empty, the script now creates a timestamped DB file: `compilation_search_<YYYYMMDD_HHMMSS>.json`,
  - added companion summary output file `<db_stem>_verified_candidates.json`.
- Reworked search strategy to better estimate compilable architecture envelope:
  - binary search for largest compilable architecture,
  - binary search for smallest compilable architecture,
  - dense checks around both boundaries,
  - explicit envelope summary using parameter-memory proxy (bytes/MiB from parameter count).
- Added similarity-guided exploration to improve sparse sampling:
  - generated near-neighbor candidates around verified compilable seeds,
  - scored candidates using architecture similarity + memory proximity,
  - injected likely candidates into DB with `source="SIMILARITY"`,
  - added threshold-focused compilation checks near upper boundary to refine practical limit.
- Extended `cnnSearch/search_space.py` with new utilities:
  - `architectureDistance(...)`,
  - `architectureSimilarityScore(...)`,
  - `generateSimilarArchitectures(...)`.
- Added tests in `tests/test_search_space_similarity.py` for similarity helpers and mutation generation.
- Updated documentation:
  - added full IMX500 compilable-search workflow description and usage examples to `cnnSearch/README.md`,
  - added top-level project note in `README.md` linking to the detailed cnnSearch search docs.
- Validation run:
  - `pytest -q tests/test_search_space_similarity.py tests/test_search_space_normalization.py` → `4 passed`.

## 2026-03-23

- Fixed a validation-label indexing bug in `cnnSearch/data.py` for explicit `valDir` usage:
  - validation class IDs are now remapped into the training class index space,
  - loader now raises a clear error if validation contains classes absent from training.
- Fixed a major supernet validation reliability issue in `cnnSearch/trainer.py`:
  - added BatchNorm statistics recalibration on a configurable number of train batches before evaluation,
  - integrated calibration into `evaluate()` via `bnCalibrationLoader` and `bnCalibrationSteps`.
- Improved architecture consistency between train/eval paths:
  - `trainOneEpoch()` now samples from the active search space (including complex paths when enabled),
  - evaluation architecture in `cnnSearch/train_supernet.py` is now built from the active search space instead of hardcoded default-space assumptions.
- Added regression tests:
  - explicit train/val class-index remapping test in `tests/test_cnn_search_supernet.py`,
  - BatchNorm recalibration running-stats update test in `tests/test_trainer_validation.py`.
- Validation run:
  - `pytest -q tests/test_cnn_search_supernet.py tests/test_trainer_validation.py` → `4 passed`.
- Fixed compilation candidate materialization to be explicitly static before export:
  - added `normalizeArchitectureForSearchSpace()` in `cnnSearch/search_space.py` to clamp every architecture field to legal options for the active search space,
  - removed duplicate intermediate `COMPLEX_SEARCH_SPACE` declaration and kept a single canonical complex-space definition.
- Updated `cnnSearch/search_compilable_subnets.py`:
  - candidate population now normalizes configs before hashing and subnet extraction,
  - compilation path normalizes DB-loaded configs again before export,
  - enforced `eval()` mode on supernet and extracted subnet before quantization/export.
- Updated `cnnSearch/export_utils.py` to enforce `quantized_model.eval()` before ONNX serialization to avoid train-mode export behavior during compilation search.
- Added robust ONNX export hardening in `cnnSearch/export_utils.py` for MCT + newer PyTorch:
  - temporarily patches `torch.onnx.export` to force `dynamo=False` during MCT ONNX export,
  - adds fallback static ONNX export (`dynamic_axes=None`) when MCT exporter still fails due to dynamic-shape validation mismatches.
- Added regression test `tests/test_search_space_normalization.py` to guarantee normalization emits only valid static architecture choices.
- Fixed validation schedule behavior in `cnnSearch/train_supernet.py`:
  - replaced zero-based `epochIndex % evalEveryEpochs == 0` trigger with one-based scheduling helper,
  - validation now runs every N epochs in one-based counting and always runs on the final epoch.
- Added `shouldEvaluateOnEpoch()` in `cnnSearch/trainer.py` and regression tests in `tests/test_training_schedule.py`.
- Added periodic checkpoint cadence control:
  - new CLI argument `--save-every-epoch` in `cnnSearch/train_supernet.py` (default `0` means never write periodic checkpoints),
  - new helper `shouldSaveCheckpointOnEpoch()` in `cnnSearch/trainer.py` for one-based checkpoint cadence,
  - best model saving is now independent and always happens on validation improvement,
  - `--disableCheckpointing` now disables periodic checkpoints only.
- Extended `tests/test_training_schedule.py` with checkpoint cadence tests.
- Validation run:
  - `pytest -q tests/test_training_schedule.py tests/test_search_space_normalization.py tests/test_cnn_search_supernet.py tests/test_trainer_validation.py` → `11 passed`.
- Validation run:
  - `pytest -q tests/test_search_space_normalization.py tests/test_cnn_search_supernet.py tests/test_trainer_validation.py` → `5 passed`.

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
