from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def buildTrainTransform(imageSize: int) -> transforms.Compose:
    """Build a strong but standard augmentation pipeline for ImageNet-style classification."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(imageSize, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def buildEvalTransform(imageSize: int) -> transforms.Compose:
    """Build deterministic evaluation transforms for stable validation metrics."""
    resizeSize = int(round(imageSize * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(resizeSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
