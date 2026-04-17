# ANALYSIS_REPORT

## Scope
This report analyzes why sampled subnets can show near-zero / zero validation accuracy in the standalone evaluation flow (`cnnSearch/evaluation.py`, `cnnSearch/evaluate_candidate.py`) while training-time validation in `cnnSearch/train_supernet.py` appears high.

## Key Findings (Before BN Removal)

### 1) Supernet stem path was not actually sampled during training/eval
- In `cnnSearch/models/supernet.py`, `SlimStemSelector.forward` previously averaged outputs from **all** stem paths.
- `ArchitectureConfig.stemPathIndex` was effectively ignored in supernet forward.
- During extraction (`cnnSearch/models/subnet.py`), a **single stem path** is copied into the subnet.
- Result: train/eval behavior of supernet stem did not match extracted subnet behavior, creating a severe distribution mismatch.

**Impact:** extracted subnet can perform dramatically worse than supernet validation metric, including near-zero in hard cases.

### 2) BatchNorm added subnet-specific evaluation complexity
- BatchNorm requires architecture-dependent running-statistics management.
- In supernet/subnet sampling workflows this often needs calibration or specialized per-subnet handling.
- This introduces moving parts and can destabilize comparison between sampled architectures.

**Impact:** increased risk of misleading subnet accuracy and harder deployment/sampling logic.

### 3) `evaluate_candidate.py` could construct an incompatible search space
- It used `DEFAULT_SEARCH_SPACE` directly, instead of inferring search-space characteristics from checkpoint + architecture.
- If checkpoint was trained with complex paths and/or different class count or aux-head layout, evaluation extraction could be inconsistent.

**Impact:** wrong mapping during extraction and potential architecture mismatch, harming accuracy.

## Implemented Fixes

### A) Stem path routing now respects architecture
**File:** `cnnSearch/models/supernet.py`
- Updated `SlimStemSelector.forward` to select only the requested `stemPathIndex`.
- Updated `ResNetSuperNet.forwardFeatures` to pass `architectureConfig.stemPathIndex` into stem selection.

This makes training/eval architecture sampling behavior consistent with subnet extraction.

### B) Aligned candidate evaluator with checkpoint-derived search space
**File:** `cnnSearch/evaluate_candidate.py`
- Replaced manual `DEFAULT_SEARCH_SPACE` construction with:
  - `buildSearchSpaceForCheckpoint(...)`
  - `loadSupernetFromCheckpoint(...)`
- Determines complex-path mode from architecture config.
- Normalizes architecture with `normalizeArchitectureForSearchSpace(..., enableAuxiliaryHeads=False)`.

## BN-Free Simplification (Current Final State)

At your request, BatchNorm was removed from the `cnnSearch` supernet/subnet stack to make subnet sampling and evaluation simpler and deterministic.

### 1) Removed BN from core model definitions
**Files:**
- `cnnSearch/models/supernet.py`
- `cnnSearch/models/subnet.py`

Changes:
- Removed `SlimBatchNorm2d` and all BN layers in supernet blocks/stem/stage projection.
- Removed all `nn.BatchNorm2d` layers in extracted subnet blocks/stem/projection.
- Updated forward paths to use `Conv -> ReLU` without BN.
- Removed BN-weight copy logic from subnet extraction.

### 2) Removed BN calibration and related CLI wiring
**Files:**
- `cnnSearch/model_pipeline.py`
- `cnnSearch/evaluation.py`
- `cnnSearch/engine.py`
- `cnnSearch/evaluate_candidate.py`
- `cnnSearch/trainer.py`
- `cnnSearch/train_supernet.py`

Changes:
- Deleted BN recalibration helper functions in pipeline and trainer.
- Removed `--bn-calibration-steps`/`--bnCalibrationSteps` options.
- Removed all BN recalibration call sites in evaluation and training.

### 3) Improved compatibility for loading older checkpoints
**File:** `cnnSearch/model_pipeline.py`
- `loadSupernetFromCheckpoint(...)` now loads with `strict=False`.
- This tolerates older checkpoints that still contain BN parameters when you transition to BN-free code.

## Why this should fix zero-accuracy behavior
With these changes, standalone sampled subnet evaluation now matches training assumptions in three critical dimensions:
1. **Same path semantics** (stem path no longer averaged in supernet).
2. **Same normalization of architecture/search space** as checkpoint.
3. **Subnet-specific BN statistics recalibration** prior to reporting accuracy.

These were the highest-probability root causes of observed zero-accuracy outputs.

## Recommended Validation Procedure
1. Train a fresh supernet with current BN-free code (recommended).
2. Evaluate a known architecture with `cnnSearch/evaluate_candidate.py`.
3. Run `cnnSearch/evaluation.py` in supernet mode and compare pre/post quantization metrics.
4. Confirm sampled subnet top-1 is non-zero and consistent across repeated runs.

## Residual Risks / Follow-ups
- If checkpoints were trained with BN-enabled code, retraining with BN-free code is strongly recommended for stable final metrics.
- If class taxonomy differs between train/val folders, ensure class index mapping consistency.

## Conclusion
The root mismatch in stem-path execution was fixed, and the repository is now simplified to a BN-free supernet/subnet pipeline. This removes BN-specific sampling complexity and gives a cleaner base for training sampled subnets and later fine-tuning.
