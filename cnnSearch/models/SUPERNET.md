# SuperNet Architecture (`cnnSearch/models/supernet.py`)

This file documents the **current** expanded supernet implementation used by `cnnSearch`.

## 1) What changed in the expanded design

Compared to the original simple slimmable ResNet, the supernet now includes:

1. **Multiple stage paths** (five per stage, guided by `stagePathIndices`):
   - path `0`: short/wider profile,
   - path `1`: balanced profile,
  - path `2`: deeper/narrower profile,
  - path `3`: large-kernel + SE profile,
  - path `4`: dilated + SE profile.
2. **Per-stage kernel choice** (`stageKernelSizes`, currently `3`, `5`, or `7`) using center-cropped kernels from a larger shared kernel tensor.
3. **Per-stage extra stride** (`stageExtraStrides`) to increase architectural diversity in downsampling behavior.
4. **Auxiliary classification heads** after selected stages for deep supervision (`searchSpace.auxiliaryHeadStages`).
5. **All-path fusion**: every stage now executes all paths and combines their outputs by strict equal-weight summation.
6. **Multi-path stem fusion**: the stem now has three operator paths that are also fused with equal weights.

This creates a much larger and more expressive search space while preserving easy extraction of selected subnet paths.

---

## 2) High-level graph

```text
Input
  |
  v
Stem Selector (three paths, equal-weight fused)
  |- path 0: Conv7x7 -> BN -> ReLU -> MaxPool
  |- path 1: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> MaxPool
  |- path 2: Conv5x5 -> BN -> ReLU -> AvgPool -> Conv1x1 -> BN -> ReLU
  |
  v
Stage1 Selector (fused paths + kernel + depth + extra stride)
  |\
  | +--> Aux Head (optional)
  |
  v
Stage2 Selector (fused paths + kernel + depth + extra stride)
  |\
  | +--> Aux Head (optional)
  |
  v
Stage3 Selector (fused paths + kernel + depth + extra stride)
  |\
  | +--> Aux Head (optional)
  |
  v
Stage4 Selector (fused paths + kernel + depth + extra stride)
  |\
  | +--> Aux Head (optional)
  |
  v
AdaptiveAvgPool -> Flatten -> Main SlimLinear classifier
```

Forward returns:
- main logits,
- architecture config,
- list of auxiliary logits (empty when disabled).

---

## 3) Stage selector internals

Each stage is represented by `SlimStageSelector`, which holds several `SlimStagePath` branches.

### ASCII structure of one stage selector

```text
                +---------------- Path 0 (short / wider) ------------------+
input ----------+---------------- Path 1 (balanced) -----------------------+
                +---------------- Path 2 (deep / narrower) ----------------+
                +---------------- Path 3 (large-kernel + SE) --------------+--> weighted sum --> stage output
                +---------------- Path 4 (dilated + SE) -------------------+
```

All paths are active for every stage forward pass.
`stagePathIndices` no longer changes stage fusion weights during supernet forward.
It is retained for architecture metadata and deterministic subnet extraction.

### Per-path execution

```text
for each path p:
  block[0]  stride = stageStride * stageExtraStride
  block[1:] stride = 1
  kernel size for convs = max(selected stage kernel, pathMinKernelSize[p])
  dilation = pathDilation[p]
  optional squeeze-excitation = pathUseSE[p]
  active depth = round(baseDepth * pathDepthMultiplier)
  active channels = round(canonicalChannels * pathWidthMultiplier), aligned to 8
  final projection to canonical stage channels

fusion:
  w[p] = 1 / numPaths
  stageOutput = sum_p w[p] * pathOutput[p]
```

This keeps gradient flow through all paths and removes path-priority bias during supernet training.

---

## 4) Stem selector internals

The stem is implemented as `SlimStemSelector` containing three `SlimStemPath` branches.

```text
                +-- Stem Path 0 (7x7 conv stem) ----------------+
input ----------+-- Stem Path 1 (stacked 3x3 stem) -------------+--> equal-weight sum --> stem output
                +-- Stem Path 2 (5x5 + 1x1 with avgpool stem) --+
```

All stem paths are always executed during supernet forward and averaged with equal weights.
For subnet export, `ArchitectureConfig.stemPathIndex` chooses the concrete stem variant to materialize.

## 5) Block and layer behavior

### `SlimConv2d`
- Stores one maximal tensor: `[Cout_max, Cin_max, Kmax, Kmax]`.
- Slices channels and center-crops spatial kernel at runtime.
- Supports runtime stride, padding, and dilation overrides.

### `SlimBasicBlock`
- Residual block with two dynamic convolutions and optional projection shortcut.
- Uses selected runtime kernel size for both convs.
- Supports path-specific dilation and optional squeeze-excitation.

### `SlimBatchNorm2d`
- Shared max BN parameters/buffers with active slicing by channel count.

### `SlimLinear`
- Shared classifier with dynamic input feature slicing.

---

## 6) Search controls used by architecture config

`ArchitectureConfig` now drives all main decisions:

```text
inputResolution
outputStride
stageDepths[4]
stageWidthMultipliers[4]
stemChannels
stemPathIndex
stagePathIndices[4]
stageKernelSizes[4]
stageExtraStrides[4]
enableAuxiliaryHeads
```

Helper logic in `search_space.py`:
- `decodeStageChannels(...)`
- `decodeStagePathChannels(...)`
- `resolveStagePathDepth(...)`

---

## 7) Auxiliary heads

Auxiliary heads are stage-level classifiers (`SlimLinear`) connected after stage outputs.

```text
stage feature -> GAP -> flatten -> aux linear -> aux logits
```

Training uses these heads for deep supervision (controlled in trainer by `auxiliaryLossWeight`).
Evaluation typically disables them via `enableAuxiliaryHeads=False` in architecture config.

---

## 8) ResNet18 / ResNet34 comparison

| Property | ResNet18 | ResNet34 | Expanded SuperNet |
|---|---|---|---|
| Stage depths | fixed `2-2-2-2` | fixed `3-4-6-3` | per-stage sampled + path depth multipliers |
| Stage channels | fixed | fixed | base width multipliers + path width multipliers |
| Kernel size | fixed 3x3 blocks | fixed 3x3 blocks | per-stage runtime `3/5/7` + path min kernel constraints |
| Stride pattern | fixed | fixed | base output-stride schedule + extra stage stride |
| Path execution | N/A | N/A | all stage paths executed and equally fused |
| Stem | single 7x7 stem | single 7x7 stem | 3-path stem equally fused in supernet; selected on export |
| Exotic operators | none | none | SE path and dilated-SE path |
| Heads | main head only | main head only | main + auxiliary heads |
| Architecture count | 1 | 1 | combinatorial supernet |

ResNet18-like config can still be represented by selecting:
- balanced path for all stages,
- stage depths close to `[2,2,2,2]`,
- kernel `3`, extra stride `1`, output stride `32`, width multipliers `1.0`.

ResNet34-like config can still be represented by selecting:
- balanced path,
- stage depths close to `[3,4,6,3]`,
- kernel `3`, extra stride `1`, output stride `32`, width multipliers `1.0`.

---

## 9) Why this design helps subnet sampling

1. **Deterministic architecture decoding**: each sampled field maps to deterministic path attributes and stem/export selections.
2. **Easy extraction**: `models/subnet.py` materializes a standalone subnet using `stemPathIndex` and `stagePathIndices`.
3. **Large but structured space**: path/depth/width/kernel/stride/SE/dilation knobs create diversity without uncontrolled graph chaos.
4. **All-path training** reduces dead-path risk since every path receives gradients each step.
5. **Auxiliary supervision** improves optimization stability for deep/wide variants during supernet training.

---

## 10) Practical caveats

1. Channel and kernel slicing can bias training toward frequently sampled prefixes/options.
2. Aggressive `stageExtraStrides` may downsample too early for some tasks; use constraints in GA sampling.
3. Equal fusion improves fairness, but it may under-express path specialization unless export-time sampling is diverse.
4. For fair architecture ranking, sampling policy and coverage statistics should be monitored.

This version is intentionally designed to balance **search-space richness** and **extractable subnet clarity**.

