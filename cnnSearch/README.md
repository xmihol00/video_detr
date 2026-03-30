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
- Configurable training/evaluation logging with:
  - text or JSON output,
  - optional log file sink,
  - deduplicated and throttled event logging for iterative loops.

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
  --logLevel INFO \
  --logFormat text \
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
  --logLevel INFO \
  --logFormat json \
  --logFile cnnSearch/outputs/supernet_ddp_run_01/train.jsonl \
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
  --logLevel INFO \
  --logFormat text \
  --amp \
  --outputJson cnnSearch/outputs/candidate_eval.json
```

## IMX500 compilable-subnet search

The script `cnnSearch/search_compilable_subnets.py` is a resumable architecture search utility focused on a practical question: **which subnet configurations can be quantized and compiled for Sony IMX500**.

### How the search works end-to-end

1. **Candidate population**
   - Reads/creates a DB JSON file with candidate architectures.
   - In exhaustive mode (`--num-samples` not provided), iterates the full Cartesian product from `search_space.py`.
   - In sampling mode, adds random unique candidates until the target count is reached.
   - Every sampled candidate is normalized to static legal search-space options via `normalizeArchitectureForSearchSpace(...)`.
   - Each candidate stores:
     - architecture config,
     - `param_count` (PyTorch parameter count),
     - status (`PENDING|SUCCESS|FAILED`),
     - error message if compilation fails.

2. **Compilation check for one candidate**
   - Materializes subnet from supernet weights.
   - Quantizes to ONNX with `RepresentativeDataGenerator` and `Imx500Exporter`.
   - Runs `imxconv-pt` compiler.
   - Stores pass/fail in DB and keeps the process resumable.

3. **Two boundary binary searches**
   - Sorts candidates by `param_count`.
   - Runs binary search for:
     - the **largest** compilable architecture,
     - the **smallest** compilable architecture.
   - This gives an initial compilable-memory envelope in terms of parameter memory (FP32 proxy).

4. **Dense refinement around boundaries**
   - Runs extra checks in dense windows around both boundary indices.
   - Reduces error from sparse sampling and sharpens the envelope.

5. **Similarity-guided expansion**
   - Uses verified compilable architectures as seeds.
   - Generates nearby configs with `generateSimilarArchitectures(...)` from `search_space.py` (few controlled mutations per seed).
   - Scores each candidate using:
     - architecture similarity (`architectureSimilarityScore(...)`),
     - memory proximity to upper compilable threshold.
   - Keeps only candidates inside the current compilable memory envelope.

6. **Threshold-focused validation of likely candidates**
   - Adds likely candidates to the main DB (`source: "SIMILARITY"`).
   - Compiles a budgeted subset near the upper threshold band to tighten the practical max-size boundary.

7. **Writes two output JSON artifacts**
   - **Main DB**: full history of sampled and similarity-generated candidates with statuses.
   - **Verified/Likely summary DB**: compact summary containing:
     - compilable envelope,
     - all verified compilable architectures,
     - similarity-generated likely-compilable candidates (including predicted scores and any verified outcomes).

### `--dv`: choose DB file or start a timestamped run

- `--dv <path>`: continue search from the given JSON DB.
- `--dv ""` (default): create a new DB named
  - `compilation_search_<YYYYMMDD_HHMMSS>.json`.

For each DB file, a companion summary file is produced:
- `<db_stem>_verified_candidates.json`

### Search-space role (`search_space.py`)

`cnnSearch/search_space.py` defines both the searchable dimensions and the utilities used by this script:

- `SearchSpaceConfig`: valid option sets for resolution, strides, depths, widths, paths, kernels, and extra strides.
- `ArchitectureConfig`: concrete architecture instance schema.
- `normalizeArchitectureForSearchSpace(...)`: clamps architecture values to nearest valid static options.
- `sampleRandomArchitecture(...)`: random architecture generator.
- `iterateAllArchitectures(...)`: exhaustive architecture iterator.
- `architectureDistance(...)`: normalized distance over all architecture choices.
- `architectureSimilarityScore(...)`: similarity value in `[0, 1]`.
- `generateSimilarArchitectures(...)`: local mutation generator for neighborhood exploration.

### Example commands

Start a new timestamped DB run:

```bash
cd /home/david/projs/video_detr
/home/david/projs/video_detr/.venv/bin/python -m cnnSearch.search_compilable_subnets \
  --num-samples 2000
```

Resume from an existing DB:

```bash
cd /home/david/projs/video_detr
/home/david/projs/video_detr/.venv/bin/python -m cnnSearch.search_compilable_subnets \
  --num-samples 2000 \
  --dv cnnSearch/outputs/compilation_search_20260330_173000.json
```

Enable complex stage paths during search:

```bash
cd /home/david/projs/video_detr
/home/david/projs/video_detr/.venv/bin/python -m cnnSearch.search_compilable_subnets \
  --num-samples 2000 \
  --enable-complex-paths
```

## Logging configuration

Both `train_supernet.py` and `evaluate_candidate.py` support runtime logging configuration:

- `--logLevel`: `DEBUG|INFO|WARNING|ERROR|CRITICAL`
- `--logFormat`: `text|json`
- `--logFile`: optional path for persistent logs

The logger implementation (`cnnSearch/logging_utils.py`) uses event-aware helpers to avoid repetitive spam:

- `logOnce(key, ...)`: emit only one time for a key.
- `logEveryN(key, n, ...)`: emit every `n`-th event in loops.
- `logInterval(key, seconds, ...)`: emit at most once per interval.

This enables deeper observability for data loading and training dynamics while keeping logs compact.
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
