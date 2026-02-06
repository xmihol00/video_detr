# VideoDETR Changelog

All notable changes to this implementation will be documented in this file.

---

## [0.1.0] - 2026-02-06 - Initial Implementation

### Added
- Initial planning document (PLAN.md)
- This changelog file
- [x] Package structure with __init__.py files
- [x] VideoSequenceDataset (datasets/video_dataset.py)
  - YOLO-style dataset loading
  - Sequence-aware frame sampling with temporal spread
  - Track ID extraction from label line numbers
- [x] Temporal positional encoding (models/temporal_encoding.py)
  - Sinusoidal and learned temporal embeddings
  - Combined spatial-temporal position encoding
- [x] VideoDETR model (models/video_detr.py)
  - Multi-frame processing through shared backbone
  - Frame-specific query embeddings
  - Detection heads (classification, bounding box)
  - Tracking embedding head for cross-frame association
- [x] Tracking head (models/tracking_head.py)
  - MLP-based embedding projection
  - L2-normalized outputs for contrastive learning
- [x] Supervised contrastive loss (losses/contrastive_loss.py)
  - Temperature-scaled contrastive learning
  - Support for same-track positive pairs across frames
- [x] Video criterion (losses/video_criterion.py)
  - Per-frame Hungarian matching
  - Combined detection + tracking losses
  - Auxiliary losses at each decoder layer
- [x] Training script (main.py)
  - Full argument parsing
  - Distributed training support
  - Checkpoint saving and resuming
- [x] Engine functions (engine.py)
  - trainOneEpoch with gradient clipping
  - evaluate with loss logging
  - Basic track association utility

### TODO (Future Work)
- [ ] Full MOT metrics evaluation (MOTA, IDF1)
- [ ] Inference script for video processing
- [ ] Visualization utilities
- [ ] Integration tests with real data
- [ ] Memory-efficient training for long sequences

---

## Progress Notes

### 2026-02-06: Project Initialization
- Analyzed existing DETR codebase structure
- Identified key components to extend:
  - `models/detr.py`: DETR model and SetCriterion
  - `models/transformer.py`: Transformer architecture
  - `models/backbone.py`: ResNet backbone with position encoding
  - `models/position_encoding.py`: Spatial positional encoding
  - `models/matcher.py`: Hungarian matching
  - `datasets/coco.py`: Dataset loading
  - `util/misc.py`: Utilities including NestedTensor
- Created detailed implementation plan

### 2026-02-06: All Components Implemented and Tested
- Completed all core components:
  - Video dataset with YOLO-style loading
  - Temporal positional encoding (learned + sinusoidal)
  - VideoDETR model with tracking head
  - Supervised contrastive loss
  - Video criterion with Hungarian matching
  - Training script and engine functions
- Created comprehensive test suite (test_video_detr.py)
- Fixed temporal encoding dimension bug (output now correctly 256)
- All 6 test suites pass on CPU:
  - ✓ Temporal encoding tests
  - ✓ Tracking head tests
  - ✓ Contrastive loss tests
  - ✓ Matcher tests
  - ✓ Criterion tests
  - ✓ Full VideoDETR model tests

### Ready for GPU Training
- Run: `python vidDetr/main.py --dataConfig vidDetr/data.yaml --numFrames 5 --epochs 100`
- Dataset expected at: `/mnt/ssd/xmihol00/simulated_video_coco/`
- Structure: `train/images/`, `train/labels/`, `val/images/`, `val/labels/`
