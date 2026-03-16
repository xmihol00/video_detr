# cnnSearch

ResNet-family supernet training stack for neural architecture search targeting edge deployment constraints.

## What is implemented

- ResNet-style **SuperNet** with tunable:
  - stage depth,
  - stage channel widths,
  - input resolution,
  - output feature-map stride (`8/16/32`),
  - stage path metadata over five path families (short-wide / balanced / deep-narrow / large-kernel+SE / dilated+SE),
  - stage kernel size (`3` or `5` or `7`),
  - stem path selection metadata (`stemPathIndex` for subnet extraction),
  - stage extra stride options.
- **Auxiliary classification heads** after multiple stages for deep supervision during supernet training.
- **All-path stage fusion**: each stage runs all paths and fuses outputs with equal-weight summation.
- **All-path stem fusion**: three stem paths run in parallel and are equally fused during supernet training.
- **Subnetwork extraction** from supernet weights to standalone ResNet subnet models.
- ImageFolder data pipeline for 1000-class classification datasets.
- Data augmentation for train/eval.
- Multi-GPU **DDP** training entrypoint.
- Candidate evaluation tooling for fitness metrics:
  - top-1 / top-5 accuracy,
  - parameter memory footprint,
  - latency measurement.

## Dataset format

The training pipeline expects torchvision `ImageFolder` layout.

Example:

- `path/to/train/class_000/...jpg`
- `path/to/train/class_001/...jpg`
- ...
- `path/to/val/class_000/...jpg` (optional)

If `--valDir` is not provided, validation is split from `--trainDir` using `--valSplitRatio`.

## Single-GPU training

```bash
cd /home/david/projs/video_detr
/home/david/projs/video_detr/.venv/bin/python -m cnnSearch.train_supernet \
  --trainDir /path/to/imagenet_train \
  --valDir /path/to/imagenet_val \
  --epochs 90 \
  --batchSize 128 \
  --numWorkers 8 \
  --imageSize 224 \
  --auxiliaryLossWeight 0.3 \
  --amp \
  --saveDir cnnSearch/outputs/supernet_run_01
```

## Multi-GPU training (single node)

```bash
cd /home/david/projs/video_detr
torchrun --nproc_per_node=4 -m cnnSearch.train_supernet \
  --trainDir /path/to/imagenet_train \
  --valDir /path/to/imagenet_val \
  --epochs 90 \
  --batchSize 128 \
  --numWorkers 8 \
  --imageSize 224 \
  --auxiliaryLossWeight 0.3 \
  --amp \
  --saveDir cnnSearch/outputs/supernet_ddp_run_01
```

`--batchSize` is per process (per GPU).

## Evaluate one candidate architecture

1. Create an architecture JSON matching `ArchitectureConfig` fields.
2. Run evaluation against a trained supernet checkpoint.

Example architecture JSON:

```json
{
  "inputResolution": 224,
  "outputStride": 16,
  "stageDepths": [2, 3, 4, 2],
  "stageWidthMultipliers": [0.75, 1.0, 0.75, 0.5],
  "stemChannels": 48,
  "stemPathIndex": 1,
  "stagePathIndices": [0, 1, 2, 1],
  "stageKernelSizes": [3, 5, 5, 3],
  "stageExtraStrides": [1, 1, 1, 1],
  "enableAuxiliaryHeads": false
}
```

```bash
cd /home/david/projs/video_detr
/home/david/projs/video_detr/.venv/bin/python -m cnnSearch.evaluate_candidate \
  --supernetCheckpoint cnnSearch/outputs/supernet_run_01/best_model.pth \
  --architectureJson /path/to/candidate_architecture.json \
  --valDir /path/to/imagenet_val \
  --batchSize 128 \
  --numWorkers 8 \
  --amp \
  --outputJson cnnSearch/outputs/candidate_eval.json
```

## Key modules

- `cnnSearch/search_space.py`: architecture search-space and sampling.
- `cnnSearch/models/supernet.py`: dynamic multi-path ResNet supernet with auxiliary heads.
- `cnnSearch/models/subnet.py`: standalone subnet and extraction for selected path/kernel/stride/depth config.
- `cnnSearch/data.py`: ImageFolder loaders + distributed samplers.
- `cnnSearch/train_supernet.py`: DDP training entrypoint.
- `cnnSearch/evaluate_candidate.py`: candidate accuracy/latency/memory evaluation.
- `cnnSearch/profiling.py`: latency and memory metrics utilities.

## Current scope and next planned step

This implementation provides the full supernet training substrate and candidate evaluation flow.
The next step is integrating the genetic algorithm loop that proposes architecture configs and consumes evaluation metrics as fitness values.
