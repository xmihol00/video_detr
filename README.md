# VideoDETR: Video Object Detection and Tracking

## What Is This Project?

VideoDETR is an intelligent system that watches video clips and simultaneously performs two critical tasks:

1. **Object Detection** — Identifying and locating every object in every frame of the video
2. **Multi-Object Tracking** — Recognizing which objects are the same entity as they move between frames

Imagine watching a crowd of people walking through a street: the system not only needs to draw boxes around each person and label them as "person," but it also needs to remember that the person on the left in frame 1 is the same person who moved to the center in frame 2. VideoDETR does both of these things together in a single, unified system without needing separate post-processing steps.

## How Does It Work?

### The Core Approach

The system takes inspiration from a groundbreaking method called DETR (Detection Transformer) that treats object detection as a prediction problem. VideoDETR extends this approach to handle video instead of single images.

The process works like this:

1. **Visual Feature Extraction** — A pre-trained neural network (ResNet) processes the video frames and extracts important visual patterns (edges, shapes, textures)

2. **Understanding Position in Space** — The system adds information about where each visual feature is located in the frame (positional encoding)

3. **Understanding Position in Time** — The system adds information about which frame each feature comes from, enabling it to connect the same object across frames (temporal encoding)

4. **Intelligent Analysis** — A Transformer neural network (the same technology used in modern language models) processes all the visual information and learns to recognize objects and track them

5. **Generating Outputs** — The system produces three key outputs for each detected object:
   - **Class prediction**: What type of object is it? (person, car, dog, etc.)
   - **Bounding box**: Where is it located in the frame?
   - **Tracking embedding**: A unique numerical "fingerprint" that remains consistent for the same object across frames

### The Clever Trick: Tracking Without Post-Processing

Most video analysis systems work in two separate stages:
- First stage: Detect objects in each frame independently
- Second stage: Use a separate tracker to match objects between frames

VideoDETR is different. It learns a special tracking "fingerprint" for each object during training, using a technique called supervised contrastive learning. This means the model naturally learns to produce similar fingerprints for the same object across different frames, without needing a separate matching algorithm afterward.

## What Makes It Special?

### Technical Innovations

- **Split Architecture** — The system has separate neural network pathways for detection and tracking. The tracking pathway receives information from the detection pathway but doesn't send gradients back, preventing the two tasks from interfering with each other.

- **Label Denoising** — During training, the system deliberately introduces incorrect labels mixed with correct ones to create a stronger learning signal. This technique (inspired by DINO and DN-DETR) helps the model converge faster and become more robust.

- **Smart Duplicate Suppression** — The system learns to avoid placing multiple boxes on the same object through a specialized loss function that penalizes duplicate detections.

- **Sophisticated Training Techniques** — The system uses exponential moving average (EMA) for model stability, dropout variants for regularization, and adaptive learning rate scheduling throughout training.

## What Can It Be Used For?

- **Crowd Monitoring** — Tracking people in busy public spaces
- **Traffic Analysis** — Counting and tracking vehicles on roads
- **Sports Analytics** — Following athletes or ball movements
- **Wildlife Monitoring** — Tracking animals in natural environments
- **Video Surveillance** — Understanding activity in security footage
- **Autonomous Systems** — Providing perception capabilities for robots and self-driving vehicles
- **Animal Behavior Research** — Studying individual animal movements in groups

## Project Organization

The codebase is organized into logical modules:

### Datasets
Three different video dataset formats are supported:
- **Simulated Videos** — Synthetic YOLO-format datasets for quick testing
- **TAO Benchmark** — Large-scale, publicly available video dataset
- **Real Videos** — Custom single-sequence video files with flexible frame sampling

### Models
The neural network architecture is split into components:
- **CNN Backbone** — ResNet for extracting visual features from images
- **Positional Encoders** — Add spatial and temporal position information
- **Transformer** — The core attention-based encoder-decoder
- **Detection & Tracking Heads** — Specialized layers that produce final predictions

### Loss Functions
Multiple objectives guide the training:
- **Detection losses** — Focal loss, L1, and GIoU for boxes
- **Tracking losses** — Supervised contrastive loss for embeddings
- **Label denoising losses** — Auxiliary training signal
- **Duplicate suppression** — Prevents multiple boxes on same object

### Training & Evaluation
- **Training engine** — Handles the learning loop, checkpointing, and logging
- **Evaluation** — Measures detection and tracking performance
- **Inference** — Interactive visualization and batch processing tools

## Getting Started

### Installation
The project requires Python 3.8 or newer and PyTorch 1.9 or later. Additional dependencies include common computer vision libraries like torchvision, OpenCV, and others for data loading and visualization.

### Training
Users can train the model on their own video data by:
1. Preparing videos in one of the supported formats
2. Creating a configuration file specifying dataset locations and class names
3. Running the training script with desired hyperparameters

### Inference
Once trained, the model can:
- Process video files interactively with visualization
- Generate annotated video output showing boxes and tracks
- Export tracking data for downstream analysis

## Hardware Considerations

The system is designed to be flexible:
- **Development**: Can run on laptops with small or no GPU
- **Production**: Scales to multi-GPU training on servers with up to 8 GPUs
- **Future**: The architecture is designed to eventually distribute the CNN backbone and transformer components to specialized hardware accelerators (Sony IMX500 for CNN, Hailo accelerators for transformer)

## Code Quality & Maintenance

The project follows consistent coding standards:
- All functions have type annotations for clarity
- Code is extensively commented, especially in complex sections
- Functions are organized logically within modules
- The codebase uses consistent naming conventions throughout
- Comprehensive unit tests validate all major components

## Testing

The framework includes a test suite that validates:
- Forward passes through all major modules
- Correct shapes of intermediate tensors
- Integration between components
- Loss computation and optimization

## Future Directions

The project has several planned improvements:
- Architecture search utilities for optimizing models for edge devices
- Enhanced MOT (Multiple Object Tracking) metrics integration
- Support for additional dataset formats
- Deployment utilities for edge devices
- Performance optimizations for real-time inference

## IMX500 Compilation Envelope Search (cnnSearch)

The repository includes `cnnSearch/search_compilable_subnets.py`, a resumable utility that searches for subnet architectures that can be quantized and compiled to Sony IMX500.

At a high level, it:
- populates a candidate architecture DB (random sampling or exhaustive enumeration),
- runs binary searches for smallest/largest compilable subnet by parameter-memory proxy,
- densifies checks around boundaries,
- generates similarity-guided nearby architectures,
- validates a threshold-focused subset near the upper compilable boundary,
- writes both a full DB and a verified/likely summary JSON.

The script now supports `--dv`:
- pass a DB JSON file to continue an existing run,
- leave it empty to create a new `compilation_search_<YYYYMMDD_HHMMSS>.json` file.

For full operational details and command examples, see `cnnSearch/README.md`.

## Key Concepts Explained Simply

**Transformer** — A type of neural network that can look at all parts of the input simultaneously and understand relationships between distant parts. Originally developed for language, it's now widely used for vision tasks.

**Positional Encoding** — Additional information added to tell the network where things are. Without it, the network wouldn't know if a visual feature is on the left or right side of the frame.

**Hungarian Matching** — A classical algorithm that finds the best way to pair predicted objects with ground-truth objects during training.

**Contrastive Learning** — A training technique where the network learns by pushing similar items closer together and pulling dissimilar items apart in a learned representation space.

**GIoU Loss** — A loss function that measures how well predicted bounding boxes match ground-truth boxes, considering both overlap and overall size.

---

**VideoDETR** brings the power of modern transformer-based deep learning to the challenging problem of video understanding, making it practical to detect and track multiple objects in video streams with a single, end-to-end trained system.
