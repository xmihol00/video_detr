# VideoDETR Implementation Plan

## Overview

VideoDETR extends the original DETR (DEtection TRansformer) architecture to handle video object detection and tracking in an end-to-end manner. The key insight is to process N consecutive video frames through a shared backbone, add temporal positional embeddings, and use a tracking head with supervised contrastive loss to associate detections across frames.

## Architecture Design

### 1. Input Processing
- **Input**: N frames from a video sequence (configurable, default N=5)
- **Backbone**: Shared ResNet50 backbone processes each frame independently
- **Feature Maps**: Each frame produces feature maps of shape [C, H', W'] where C=2048

### 2. Positional Encoding (Extended)
The original DETR uses 2D sinusoidal positional encoding for spatial positions. We extend this with:

#### 2.1 Spatial Position Encoding (unchanged)
- Sine/cosine encoding for x and y coordinates in the feature map
- Shape: [batch, hidden_dim, H', W']

#### 2.2 Temporal Position Encoding (NEW)
- Additional learned or sinusoidal encoding for frame index
- Each frame gets a unique temporal embedding
- Combined with spatial encoding: `pos = spatial_pos + temporal_pos`
- Temporal embedding can be added to both encoder inputs and decoder queries

### 3. Transformer Architecture

#### 3.1 Encoder
- Input: Concatenated features from all N frames [N*H'*W', batch, hidden_dim]
- The encoder can attend across all spatial locations AND across all frames
- This enables temporal reasoning at the feature level

#### 3.2 Decoder
- **Query Design**: N * 75 = N * queries_per_frame learnable queries
- Queries are organized: queries 0-74 for frame 0, 75-149 for frame 1, etc.
- Each query group is assigned to a specific frame via frame-specific query embeddings
- Output: [num_decoder_layers, batch, N*75, hidden_dim]

### 4. Output Heads

#### 4.1 Classification Head (unchanged)
- Linear layer: hidden_dim -> num_classes + 1
- Output: [batch, N*75, num_classes + 1]

#### 4.2 Bounding Box Head (unchanged)
- MLP: hidden_dim -> 4 (cx, cy, w, h)
- Output: [batch, N*75, 4]

#### 4.3 Tracking Embedding Head (NEW)
- MLP: hidden_dim -> tracking_embed_dim (default 128)
- Output: [batch, N*75, tracking_embed_dim]
- Used for associating detections across frames via contrastive learning

### 5. Loss Functions

#### 5.1 Original DETR Losses (per-frame)
- **Classification Loss**: Cross-entropy with class weights
- **Box L1 Loss**: L1 distance between predicted and GT boxes
- **GIoU Loss**: Generalized IoU for better box regression
- Hungarian matching is performed per-frame

#### 5.2 Tracking Loss (NEW - Supervised Contrastive)
```
L_track = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```
Where:
- z_i, z_j are embeddings of the same object in different frames (positive pairs)
- z_k are embeddings of different objects (negative pairs)
- τ is temperature (default 0.07)
- Similarity is computed using cosine similarity

The tracking supervision comes from label line correspondences across frames in the sequence.

### 6. Dataset Format

Based on the YOLO-style dataset structure:
```
train/
├── images/
│   ├── seq_000001_frame_0000.jpg
│   ├── seq_000001_frame_0001.jpg
│   └── ...
└── labels/
    ├── seq_000001_frame_0000.txt
    ├── seq_000001_frame_0001.txt
    └── ...
```

Label file format (YOLO style):
```
class_id center_x center_y width height
```

**Important**: Objects on the same line number across label files in a sequence represent the same object (tracking correspondence).

### 7. Sampling Strategy

For training:
1. Randomly select a sequence
2. Sample N frames with controlled spread (not just consecutive)
3. Ensure temporal diversity while maintaining reasonable motion
4. Frame sampling options:
   - Random with minimum gap
   - Uniform sampling
   - Random within segments

## File Structure

```
vidDetr/
├── PLAN.md                    # This file
├── CHANGELOG.md               # Progress tracking
├── __init__.py                # Package init
├── data.yaml                  # Dataset configuration
├── main.py                    # Training entry point
├── engine.py                  # Training/evaluation loops
├── models/
│   ├── __init__.py
│   ├── video_detr.py          # VideoDETR model
│   ├── video_transformer.py   # Extended transformer (optional)
│   ├── temporal_encoding.py   # Temporal positional encoding
│   ├── tracking_head.py       # Tracking embedding MLP
│   └── video_matcher.py       # Extended Hungarian matcher
├── datasets/
│   ├── __init__.py
│   ├── video_dataset.py       # Video sequence dataset
│   └── video_transforms.py    # Video-aware transforms
└── losses/
    ├── __init__.py
    ├── video_criterion.py     # Extended SetCriterion
    └── contrastive_loss.py    # Supervised contrastive loss
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [x] Analyze existing DETR codebase
- [ ] Create file structure
- [ ] Implement video dataset loader
- [ ] Implement temporal positional encoding

### Phase 2: Model Architecture
- [ ] Implement VideoDETR model
- [ ] Extend transformer for multi-frame processing
- [ ] Implement tracking embedding head

### Phase 3: Training Pipeline
- [ ] Implement video-aware Hungarian matching
- [ ] Implement supervised contrastive loss
- [ ] Extend SetCriterion for tracking loss
- [ ] Create training script with all hyperparameters

### Phase 4: Evaluation & Testing
- [ ] Implement video evaluation metrics
- [ ] Create inference pipeline
- [ ] Run sanity checks on CPU

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_frames | 5 | Number of frames per clip |
| queries_per_frame | 75 | Detection queries per frame |
| tracking_embed_dim | 128 | Dimension of tracking embeddings |
| contrastive_temp | 0.07 | Temperature for contrastive loss |
| loss_track_coef | 1.0 | Weight for tracking loss |
| frame_sample_gap | 1-10 | Min/max gap between sampled frames |
| hidden_dim | 256 | Transformer hidden dimension |
| num_classes | 80 | COCO classes |

## Key Design Decisions

1. **Shared vs Separate Backbones**: Use shared backbone for efficiency
2. **Query Organization**: Frame-specific queries with temporal embeddings
3. **Tracking Supervision**: Line-number correspondence in label files
4. **Loss Weighting**: Balance detection and tracking losses
5. **Memory Efficiency**: Process frames in mini-batches if needed

## References

- DETR: https://arxiv.org/abs/2005.12872
- Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
- TransTrack: https://arxiv.org/abs/2012.15460
- MOTR: https://arxiv.org/abs/2105.03247
