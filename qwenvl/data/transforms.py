# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import copy


def crop(image, target, region):
    original_size = image.size
    cropped_image = F.crop(image, *region)

    target = copy.deepcopy(target)
    i, j, h, w = region
    # bbox_scale = torch.tensor([original_size[1], original_size[0], original_size[1], original_size[0], original_size[1], original_size[0], original_size[1], original_size[0]]).to(dtype=torch.float).unsqueeze(0)
    bbox_scale = torch.tensor([original_size[0], original_size[1], original_size[0], original_size[1], original_size[0], original_size[1], original_size[0], original_size[1]]).to(dtype=torch.float).unsqueeze(0)
    fields = ["triplet_boxes", "gt_bboxes"]
    for key in fields:
        if key not in target:
            continue
        bboxes = copy.deepcopy(target[key])
        bboxes = bboxes * bbox_scale.to(bboxes.dtype)
        cropped_boxes = bboxes - torch.as_tensor([j, i, j, i, j, i, j, i]).unsqueeze(0)

        max_size = torch.as_tensor([w, h, w, h, w, h, w, h], dtype=cropped_boxes.dtype)
        cropped_boxes = torch.min(cropped_boxes, max_size.unsqueeze(0))
        cropped_boxes = cropped_boxes.clamp(min=0)

        keep_boxes = []
        for b in cropped_boxes:
            if b[2] > b[0] and b[3] > b[1] and b[6] > b[4] and b[7] > b[5]:
                keep_boxes.append(1)
            else:
                keep_boxes.append(0)
        target[key] = cropped_boxes / max_size.unsqueeze(0)
        target[key+"_keep"] = keep_boxes

    fields = ["single_gt_bboxes", "single_detect_bboxes"]
    for key in fields:
        if key not in target:
            continue
        gt_bboxes = copy.deepcopy(target[key])
        gt_bboxes = gt_bboxes * bbox_scale[:, :4]
        cropped_boxes = gt_bboxes - torch.as_tensor([j, i, j, i]).unsqueeze(0)
        max_size = torch.as_tensor([w, h, w, h], dtype=cropped_boxes.dtype)
        cropped_boxes = torch.min(cropped_boxes, max_size.unsqueeze(0))
        cropped_boxes = cropped_boxes.clamp(min=0)
        keep_boxes = []
        cropped_boxes = cropped_boxes / max_size.unsqueeze(0)
        for b in cropped_boxes:
            if b[2] > (b[0] + 0.005) and b[3] > (b[1] + 0.005):
                keep_boxes.append(1)
            else:
                keep_boxes.append(0)
        
        target[key] = cropped_boxes
        target[key+'_keep'] = keep_boxes

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = copy.deepcopy(target)

    fields = ["triplet_boxes", "gt_bboxes"]
    for key in fields:
        if key not in target:
            continue
        bboxes = copy.deepcopy(target[key])

        bboxes1, bboxes2 = bboxes[:, :4], bboxes[:, 4:]
        bboxes1 = bboxes1[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([1, 0, 1, 0])
        bboxes2 = bboxes2[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([1, 0, 1, 0])
        target[key] = torch.cat((bboxes1, bboxes2), dim=1)

    fields = ["single_gt_bboxes", "single_detect_bboxes"]
    for key in fields:
        if key not in target:
            continue
        gt_bboxes = copy.deepcopy(target[key])
        gt_bboxes = gt_bboxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([1, 0, 1, 0])
        target[key] = gt_bboxes
        keep_boxes = []
        for b in gt_bboxes:
            if b[2] > (b[0] + 0.005) and b[3] > (b[1] + 0.005):
                keep_boxes.append(1)
            else:
                keep_boxes.append(0)

        target[key+'_keep'] = keep_boxes

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

    return rescaled_image, target


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

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturatio=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturatio, hue)

    def __call__(self, img, target):
        return self.color_jitter(img), target
