# VideoDETR: Video Object Detection and Tracking with Transformers
#
# This package extends DETR for video object detection with end-to-end tracking.
# Key components:
# - VideoSequenceDataset: Loads video sequences with tracking annotations
# - VideoDETR: Multi-frame DETR with temporal encoding and tracking head
# - VideoCriterion: Loss with supervised contrastive learning for tracking

from .models import buildVideoDETR
from .datasets import buildVideoDataset

__all__ = ['buildVideoDETR', 'buildVideoDataset']
