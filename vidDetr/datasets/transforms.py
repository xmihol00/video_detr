# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from vidDetr.util.box_ops import box_xyxy_to_cxcywh
from vidDetr.util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]
    
    # Add trackIds for VideoDETR compatibility if present
    if "trackIds" in target:
        fields.append("trackIds")

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class ColorJitter(object):
    """
    Apply random color jitter to the image.
    
    This augmentation changes brightness, contrast, saturation, and hue
    randomly, helping the model generalise across different lighting
    conditions and camera characteristics.
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img, target):
        return self.jitter(img), target


class RandomGrayscale(object):
    """Randomly convert image to grayscale."""

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            # Convert to grayscale but keep 3 channels for model compatibility
            img = img.convert('L').convert('RGB')
        return img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class SmartSquareCrop(object):
    """
    Crop a wide (e.g. 16:9) image toward a 1:1 aspect ratio without
    losing any annotated objects.

    Algorithm:
    1. Compute the tight bounding rectangle of **all** GT boxes (absolute
       xyxy) plus a configurable ``margin`` (fraction of the box span).
    2. Determine the target crop size: ``min(imgW, imgH)`` (i.e. the
       shorter side) — this would give a perfect square.
    3. Along the longer axis, try to centre the crop on the GT centroid.
       If the GT extent (with margin) is wider than the target crop, the
       crop is enlarged to cover the full GT extent — the result will not
       be perfectly square, but no objects are lost.
    4. Clamp the crop window to image boundaries.
    5. Along the shorter axis the full extent is always kept (no crop).

    The crop is applied using the existing ``crop()`` function which
    correctly adjusts boxes, area, masks, etc.

    If there are **no** GT boxes the image is centre-cropped to a square.

    Args:
        margin: Fraction of the GT span to add as padding on each side
                (default 0.15 — 15 % breathing room).
        randomise_pos: If ``True`` (default) the crop position along the
                       long axis is jittered randomly within the valid
                       range (between the position that just covers all
                       GT boxes on the left and the one that just covers
                       them on the right).  If ``False`` the crop is
                       centred on the GT centroid.
    """

    def __init__(self, margin: float = 0.15, randomise_pos: bool = True):
        self.margin = margin
        self.randomise_pos = randomise_pos

    def __call__(self, img, target):
        imgW, imgH = img.size  # PIL: (width, height)

        # Already square or taller-than-wide → nothing to do on width axis
        if imgW <= imgH:
            return img, target

        # Target crop width = image height (perfect square)
        targetW = imgH

        # --- Compute GT extent along x (boxes are absolute xyxy) --------
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]  # [N, 4] absolute xyxy
            gtLeft = boxes[:, 0].min().item()
            gtRight = boxes[:, 2].max().item()
            gtSpanX = gtRight - gtLeft
            gtCentreX = (gtLeft + gtRight) / 2.0

            # Add margin (fraction of GT span, at least 10 px each side)
            pad = max(gtSpanX * self.margin, 10.0)
            neededLeft = gtLeft - pad
            neededRight = gtRight + pad
            neededW = neededRight - neededLeft

            # If GT span + margin is wider than targetW, enlarge the crop
            cropW = max(targetW, int(neededW + 0.5))
            # But never exceed image width
            cropW = min(cropW, imgW)

            # Determine valid horizontal offset range
            # The crop window [cropLeft, cropLeft + cropW] must contain
            # [neededLeft, neededRight] and be within [0, imgW].
            # cropLeft <= neededLeft  and  cropLeft + cropW >= neededRight
            maxCropLeft = max(0, int(neededLeft))
            minCropLeft = max(0, min(int(neededRight - cropW), maxCropLeft))
            # Also clamp so crop doesn't go past image right edge
            maxCropLeft = min(maxCropLeft, imgW - cropW)
            minCropLeft = min(minCropLeft, maxCropLeft)

            if self.randomise_pos and maxCropLeft > minCropLeft:
                cropLeft = random.randint(minCropLeft, maxCropLeft)
            else:
                # Centre on GT centroid, then clamp
                cropLeft = int(gtCentreX - cropW / 2.0)
                cropLeft = max(0, min(cropLeft, imgW - cropW))
                # Ensure all GT boxes are inside
                cropLeft = min(cropLeft, max(0, int(neededLeft)))
                cropLeft = max(cropLeft, int(neededRight - cropW))
                cropLeft = max(0, min(cropLeft, imgW - cropW))
        else:
            # No GT boxes — centre crop
            cropW = targetW
            cropLeft = (imgW - cropW) // 2

        # Crop height = full image height (no vertical crop for wide images)
        cropTop = 0
        cropH = imgH

        # region = (top, left, height, width) as expected by F.crop / crop()
        region = (cropTop, cropLeft, cropH, cropW)
        return crop(img, target, region)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"margin={self.margin}, randomise_pos={self.randomise_pos})")
