# AGENTS.md
> This file provides context and instructions for AI coding agents working in this repository. Read it fully before planning or making any changes.

## Project Overview
VideoDETR is a PyTorch framework that extends the original DETR (Detection Transformer) architecture to perform multi-object detection and tracking in video. Rather than processing single images, VideoDETR feeds multi-frame clips through a shared ResNet CNN backbone and enriches the resulting feature maps with temporal positional encodings before passing them through DETR's encoder-decoder transformer. Each decoder query produces not only a class prediction and bounding box, but also a low-dimensional tracking embedding trained via supervised contrastive loss to cluster the same object across frames — enabling data-association without explicit post-hoc trackers. Training stability and query-matching efficiency are improved through DN-DETR/DINO-style denoising queries, which inject noised ground-truth boxes as auxiliary inputs and use causal attention masking to prevent leakage into the main detection stream. The framework supports three dataset formats out of the box — a synthetic YOLO-format simulated dataset, the large-scale TAO benchmark, and arbitrary single-sequence real video — and ships with utilities for distributed training, exponential model averaging (EMA), and interactive inference visualisation.

## Dev environment tips
Use the `.venv` Python virtual environment for development. To activate it, run:
`source .venv/bin/activate`

## Hardware Stack
The system is designed to run on a local development machines with a small or no GPU and then on a GPU server where each node can have up to 8 GPUs. The scripts must be executable on all architectures.

## Software Stack
Mainly Python with common libraries for DNN training and general work with AI and ML.

If library is a good fit for a particular purpose, install it using `pip install`.

## Repository structure
```
cnnSearch			       # TBD: python scripts that will implement the search of the best CNN architecture using genetic algorithms
vidDetr
├── datasets
│   ├── __init__.py                    # Re-exports all three dataset classes and their collate functions
│   ├── simulated_video_dataset.py     # Simulated/synthetic video dataset with YOLO-format frames
│   ├── tao_dataset.py                 # TAO dataset: parses COCO-style JSON, splits videos into clips
│   ├── transforms.py                  # Data augmentation pipeline (crop, flip, resize, normalize, etc.)
│   └── video_dataset.py               # Real single-sequence video dataset with temporal sampling strategies
├── losses
│   ├── __init__.py                    # Re-exports VideoCriterion, PostProcess, SupervisedContrastiveLoss
│   ├── contrastive_loss.py            # Supervised contrastive loss pulling same-object embeddings across frames
│   └── video_criterion.py             # Main criterion: Hungarian matching, focal/L1/GIoU/tracking losses
├── models
│   ├── detr
│   │   ├── __init__.py                # Empty sub-package marker
│   │   ├── backbone.py                # CNN backbone: ResNet extractor with FrozenBatchNorm and Joiner
│   │   ├── position_encoding.py       # Spatial positional encoding: sinusoidal and learned variants
│   │   └── transformer.py             # DETR encoder-decoder transformer with auxiliary loss support
│   ├── __init__.py                    # Re-exports VideoDETR, TrackingHead, DenoisingGenerator, etc.
│   ├── denoising.py                   # DN-DETR style denoising query generator with attention masking
│   ├── temporal_encoding.py           # Temporal positional encoding: sinusoidal and learned variants
│   ├── tracking_head.py               # MLP head mapping decoder outputs to 128-D tracking embeddings
│   └── video_detr.py                  # Top-level model: backbone → temporal encoding → transformer → heads
├── util
│   ├── __init__.py                    # Re-exports box_ops and misc symbols
│   ├── box_ops.py                     # Bounding box format conversion, IoU, and GIoU utilities
│   └── misc.py                        # NestedTensor, distributed helpers, MetricLogger, SmoothedValue
├── CONTRIBUTING.md                    # Contribution guidelines: coding style and PR process
├── README.md                          # Architecture overview, training commands, and hyperparameter reference
├── __init__.py                        # Empty package marker
├── data.yaml                          # Dataset config: train/val paths and COCO 80-class name list
├── engine.py                          # Core training loop, evaluation loop, and ModelEMA
├── eval_tao_dataset.py                # TaoDataset verification: renders boxes/polygons and saves frames
├── inference.py                       # Interactive inference with OpenCV GUI and keyboard navigation
├── logging_utils.py                   # Logging setup and MetricTracker for CSV metric logging
├── main.py                            # Training entry point: args, model, dataloaders, training loop
├── pyproject.toml                     # Package metadata: version, license, dependency groups
├── tao_inference.py                   # Headless TAO inference writing annotated MP4s to gt_vs_pred/
├── test_video_detr.py                 # Forward-pass smoke tests for all major modules (run directly)
├── vidDETR_train.pbs                  # PBS/Torque HPC job submission script
└── video_dataset_visualizer.py        # VideoDataset debug visualiser saving annotated frames to disk
```

## Coding Guidelines
- Use the cammelCase (CammelCase) naming convention for classes, variables, functions, file names, etc., i.e., all implemented code will use this convention, which will distinguish it from library code and make it clear which code is part of the project.
- Name files starting with lowercase letters, use the cammelCase convention. Files with the `dev_` prefix are meant to run on the development machine for testing and debugging purposes.
- Comment the code extensively, especially in complex or non-obvious sections to make it readable for other developers and for future reference. Do not comment obvious code. Describe the purpose of functions, classes, and important logic decisions in comments or docstrings.
- Use **Python type annotations** on all function signatures (`typing`).
- Use `traceback` for error reporting in exception handlers.
- Do not write overly defensive code with too many checks and error handling. We will test the code thoroughly and fix bugs as they come up. Performance and code readability are more important than trying to predict every possible error case in advance. 

## Development Workflow
1. Review the `BACKLOG.md` file to understand the current tasks and priorities.
2. Before starting work on a task, check the `PROGRESS.md` file to see if there are any recent changes that might affect your work.
3. Implement the feature or fix the bug according to the coding guidelines and architecture rules outlined in this document.
4. Write unit tests for your changes and run all tests to ensure nothing is broken.
5. Summarize the updates into a concise description and add it to the `PROGRESS.md` file with the date of the change.
6. Update the `BACKLOG.md` file.
7. Be thorough, analuze ecisting code deeply, thing about the changes critically.
8. Do not write defensive code, this is not an app that would be published somewhere and tested for edge cases.
7. Suggest a commit message that clearly describes the change if the change is significant enough to warrant a commit. For smaller changes, you can skip this step. Do not commit by yourself.

## Testing
Implement unit tests using `pytest`. Tests should be organized by module and functionality. Use mocks to simulate hardware interactions and TCP communication. The testing does not have to be exhaustive, since the code will have to be tested on physical hardware eventually, but it should cover critical logic paths and edge cases.
- All tests live under `tests/`.
- Tests should not on training data, use some dummy values.

