# SuperNet Architecture Documentation (`cnnSearch/models/supernet.py`)

## 1) Purpose and Design Intent

`ResNetSuperNet` is a **weight-sharing supernet** built around a ResNet-like backbone for classification.

The key idea is to instantiate one maximal network and activate different subnetworks at runtime by selecting:
- stem channel width,
- per-stage depth,
- per-stage width multiplier (mapped to channels),
- output feature-map stride (`8`, `16`, or `32`).

This lets one training process optimize shared parameters for many candidate architectures, which can later be extracted as standalone subnet models.

---

## 2) High-Level Graph

At a high level, `ResNetSuperNet` behaves like this:

```text
Input [B,3,H,W]
	|
	v
Stem: SlimConv7x7(s=2, Cout=stemChannels) -> SlimBN -> ReLU -> MaxPool3x3(s=2)
	|
	v
Stage1: up to D1 SlimBasicBlocks, channels C1, first-block stride S1
	|
	v
Stage2: up to D2 SlimBasicBlocks, channels C2, first-block stride S2
	|
	v
Stage3: up to D3 SlimBasicBlocks, channels C3, first-block stride S3
	|
	v
Stage4: up to D4 SlimBasicBlocks, channels C4, first-block stride S4
	|
	v
AdaptiveAvgPool(1x1) -> Flatten -> SlimLinear(C4 -> numClasses)
	|
	v
Logits [B,numClasses]
```

Where:
- `D1..D4` are selected from `architectureConfig.stageDepths`.
- `C1..C4` are decoded from width multipliers in `architectureConfig.stageWidthMultipliers`.
- `S1..S4` are selected from `_resolveStageStrides(outputStride)`.

---

## 3) Search Space Interface and Runtime Activation

The supernet consumes `ArchitectureConfig` from `cnnSearch/search_space.py`:

```text
ArchitectureConfig:
  inputResolution: int
  outputStride: int            # 8,16,32 currently supported
  stageDepths: [d1,d2,d3,d4]
  stageWidthMultipliers: [m1,m2,m3,m4]
  stemChannels: int
```

### Channel decoding

`decodeStageChannels` computes each stage channel count:

```text
channel_i = floor_to_multiple_of_8(round(baseChannels_i * multiplier_i))
channel_i >= 8
```

Default `baseChannelsPerStage = [64, 128, 256, 512]`.

### Important runtime note

`inputResolution` is part of architecture metadata, but `supernet.py` itself does not resize inputs.
The resizing/sampling policy is handled by the training/evaluation pipeline (for example in `cnnSearch/trainer.py`).

---

## 4) Building Blocks in Detail

## 4.1 `SlimConv2d`

`SlimConv2d` stores a maximal weight tensor:

```text
W_full: [maxOutChannels, maxInChannels, k, k]
```

At forward, it slices only the active region:

```text
W_active = W_full[:outChannels, :inChannels, :, :]
```

Then applies `F.conv2d` with optional stride override.

This is the core weight-sharing mechanism for variable-width subnetworks.

## 4.2 `SlimBatchNorm2d`

Holds maximal BN params and buffers:
- trainable: `weight`, `bias`
- buffers: `runningMean`, `runningVar`

At forward, it slices all BN vectors to `activeChannels` and calls functional batch norm.

## 4.3 `SlimBasicBlock`

Each block is a ResNet BasicBlock variant with always-instantiated projection path:

```text
Main path:
  Conv3x3(stride = effectiveStride) -> BN -> ReLU
  Conv3x3(stride = 1)               -> BN

Identity path:
  if (effectiveStride != 1) or (Cin != Cout):
		Conv1x1(stride = effectiveStride) -> BN
  else:
		identity = input

Output:
  ReLU(main + identity)
```

`effectiveStride` is either block default stride (set at construction) or runtime override for the first block in a stage.

## 4.4 `SlimLinear`

Stores full classifier weight `[numClasses, maxInFeatures]` and slices active input columns:

```text
W_active = W[:, :inFeatures]
logits = Linear(x, W_active, b)
```

---

## 5) Stage Construction and Dynamic Depth

Each stage is created at its **maximum depth** (from search space), then truncated at forward using `activeDepth`.

### Stage generation (`_makeStage`)

For stage with maximum depth `Dmax`:
- block `0` uses `firstStride` and `inputChannels = previous stage channels`.
- blocks `1..Dmax-1` use stride `1` and `inputChannels = stage channels`.

### Stage execution (`_forwardStage`)

Only first `activeDepth` blocks are executed:

```text
for blockIndex in range(activeDepth):
	 if blockIndex == 0:
		  block(..., strideOverride = selectedStageStride)
	 else:
		  block(...)
```

So depth is dynamically selected without re-instantiating modules.

---

## 6) Output Stride Routing (Feature Map Size Control)

The supernet maps `outputStride` to stage strides as follows:

```text
outputStride = 8  -> [S1,S2,S3,S4] = [1,2,1,1]
outputStride = 16 -> [S1,S2,S3,S4] = [1,2,2,1]
otherwise         -> [S1,S2,S3,S4] = [1,2,2,2]   # effectively stride 32 path
```

Given stem downsampling by `4` (`conv7 s=2` + `maxpool s=2`):
- stage product for OS8 is `2` → overall `4*2 = 8`
- stage product for OS16 is `4` → overall `4*4 = 16`
- stage product for OS32 is `8` → overall `4*8 = 32`

This directly controls output feature map spatial size for downstream latency/accuracy tradeoffs.

---

## 7) Connectivity ASCII Diagrams

## 7.1 Full network connectivity

```text
								 +---------------------------------------------+
Input [B,3,H,W] -------->| Stem: Conv7x7(s=2,C=stem) -> BN -> ReLU    |
								 |       -> MaxPool3x3(s=2)                    |
								 +----------------------+----------------------+
																|
																v
								 +----------------------+----------------------+
								 | Stage1: d1 blocks, C1, first stride S1      |
								 +----------------------+----------------------+
																|
																v
								 +----------------------+----------------------+
								 | Stage2: d2 blocks, C2, first stride S2      |
								 +----------------------+----------------------+
																|
																v
								 +----------------------+----------------------+
								 | Stage3: d3 blocks, C3, first stride S3      |
								 +----------------------+----------------------+
																|
																v
								 +----------------------+----------------------+
								 | Stage4: d4 blocks, C4, first stride S4      |
								 +----------------------+----------------------+
																|
																v
								 AdaptiveAvgPool(1x1) -> Flatten -> SlimLinear(C4->K)
																|
																v
														 Logits [B,K]
```

## 7.2 Single `SlimBasicBlock` connectivity

```text
							 (identity branch)
Input x --------------------+----------------------------------+
									 |                                  |
									 | if stride!=1 or Cin!=Cout       |
									 |   Conv1x1(stride=s) -> BN       |
									 | else                             |
									 |   Identity                       |
									 v                                  |
Main branch:      Conv3x3(stride=s) -> BN -> ReLU             |
											|                             |
											v                             |
								Conv3x3(stride=1) -> BN --------------+
														|
														v
												  Add + ReLU
														|
														v
													Output
```

## 7.3 Weight slicing behavior inside Slim layers

```text
Full supernet tensor (stored):
  Conv weight: [Cout_max, Cin_max, k, k]

Active subnetwork view:
  Conv weight: [Cout_active, Cin_active, k, k]

This allows many subnetworks to reuse overlapping parameter prefixes.
```

---

## 8) Default Maximum Topology Implied by Search Space

From `DEFAULT_SEARCH_SPACE`:
- depth max per stage: `[3, 4, 6, 3]`
- width max multipliers: `[1.0, 1.0, 1.0, 1.0]`
- base channels: `[64, 128, 256, 512]`
- max stem channels: `64`

So the maximal instantiated graph resembles a ResNet-34 depth profile in stages (`3-4-6-3`), but remains dynamic at runtime due to configurable depth/width/stride.

---

## 9) Comparison with ResNet18 and ResNet34

## 9.1 Canonical architectures

### ResNet18 (BasicBlock)

```text
Stem: Conv7x7/64,s2 -> BN -> ReLU -> MaxPool,s2
Stage1: 2 blocks, 64 channels, first stride 1
Stage2: 2 blocks, 128 channels, first stride 2
Stage3: 2 blocks, 256 channels, first stride 2
Stage4: 2 blocks, 512 channels, first stride 2
Head: GlobalAvgPool -> FC(512->numClasses)
```

### ResNet34 (BasicBlock)

```text
Stem: Conv7x7/64,s2 -> BN -> ReLU -> MaxPool,s2
Stage1: 3 blocks, 64 channels, first stride 1
Stage2: 4 blocks, 128 channels, first stride 2
Stage3: 6 blocks, 256 channels, first stride 2
Stage4: 3 blocks, 512 channels, first stride 2
Head: GlobalAvgPool -> FC(512->numClasses)
```

## 9.2 SuperNet vs fixed ResNets

| Aspect | ResNet18 | ResNet34 | This SuperNet |
|---|---:|---:|---|
| Stage depths | fixed `2-2-2-2` | fixed `3-4-6-3` | variable per stage (from search space) |
| Stage channels | fixed `64,128,256,512` | fixed `64,128,256,512` | variable via multipliers + 8-channel rounding |
| Stem width | fixed 64 | fixed 64 | selectable (`32/48/64` by default) |
| Output stride | fixed 32 | fixed 32 | selectable (`8/16/32`) |
| Parameters | one model | one model | shared across many subnetworks |
| Use case | direct training | direct training | NAS supernet training + later extraction |

### How to emulate ResNet18/34 in this supernet

Using default base channels and stem `64`:
- ResNet18-like config: `stageDepths=[2,2,2,2]`, multipliers all `1.0`, `outputStride=32`.
- ResNet34-like config: `stageDepths=[3,4,6,3]`, multipliers all `1.0`, `outputStride=32`.

These are structurally comparable in depth/width pattern, with the caveat that layer parameterization comes from slim sliced weights in shared tensors.

---

## 10) Technical Notes and Caveats

1. **Prefix-weight sharing bias**
	- Channel slicing uses prefixes (`[:activeChannels]`).
	- Early channels receive more updates across many subnetworks than later channels.
	- This is common in slimmable/supernet approaches and should be considered during training policy design.

2. **Output stride handling**
	- `_resolveStageStrides` explicitly supports `8` and `16`; any other value routes to the stride-32 schedule.
	- In practice, search space defaults to `8/16/32`.

3. **Input resolution is metadata, not in-model transform**
	- The model expects already-sized tensors.
	- Data pipeline is responsible for producing chosen resolutions.

4. **Classifier dimensionality**
	- `SlimLinear` slices input feature columns dynamically, so reduced final-stage channels are supported without redefining the head module.

5. **BN behavior under dynamic widths**
	- `SlimBatchNorm2d` slices running stats and affine params per active width.
	- This is necessary for mixed-width training but requires careful training schedule for fair performance across candidate widths.

---

## 11) Practical Summary

This implementation provides a clean ResNet-family supernet with:
- dynamic depth,
- dynamic per-stage channels,
- dynamic output feature map stride,
- dynamic stem width,
- shared weights enabling efficient NAS pretraining,
- compatibility with extraction into standalone subnet models.

Architecturally, it spans and extends the design envelope between ResNet18 and ResNet34, while adding additional width and feature-map-scale flexibility required for edge deployment optimization.

