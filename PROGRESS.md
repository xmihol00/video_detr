# PROGRESS

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
