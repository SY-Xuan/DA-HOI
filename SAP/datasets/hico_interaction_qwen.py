"""
HICO detection dataset.
"""
from pathlib import Path

import torchvision.transforms
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data

import datasets.transforms as T
from .hico_text_label import hico_text_label, hico_unseen_index
from qwenvl.data.hico_class import *
from transformers import AutoProcessor


class HICODetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, transforms, zero_shot_type="", qwen_path=""):
        self.img_set = img_set
        self.img_folder = img_folder
        self.num_queries = 100
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms
        self.unseen_index = hico_unseen_index.get(zero_shot_type, [])
        print(self.unseen_index)

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                new_img_anno = []
                skip_pair = []
                for hoi in img_anno['hoi_annotation']:
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        skip_pair.append((hoi['subject_id'], hoi['object_id']))
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        new_img_anno = []
                        break
                    if (hoi['category_id'] - 1) == 57:
                        continue
                    if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
                        new_img_anno.append(hoi)
                if len(new_img_anno) > 0:
                    self.ids.append(idx)
                    img_anno['hoi_annotation'] = new_img_anno
        else:
            self.ids = list(range(len(self.annotations)))
        print("{} contains {} images".format(img_set, len(self.ids)))

        objectid2index = {}
        for index, object_item in enumerate(HICO_OBJECTS):
            objectid2index[object_item['id']] = index
        self.objectid2index = objectid2index
        self.qwen_image_processor = AutoProcessor.from_pretrained(qwen_path).image_processor
        self.qwen_image_processor.max_pixels = 401408
        self.qwen_image_processor.min_pixels = 784

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self.objectid2index[obj['category_id']]) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self.objectid2index[obj['category_id']] for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img, target = self._transforms[1](img_0, target_0)
            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], [] 
            sub_obj_pairs = []
            hoi_labels = []
            for hoi in img_anno['hoi_annotation']:
                # print('hoi: ', hoi)
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue

                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][hoi['category_id'] - 1] = 1
                    continue
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(117)]
                    verb_label[hoi['category_id'] - 1] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
                    verb_labels.append(verb_label)

            target['filename'] = img_anno['file_name']
            # print('sub_obj_pairs: ', sub_obj_pairs)
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, 117), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img_0, _ = self._transforms[0](img, None)
                img, _ = self._transforms[1](img_0, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], hoi['category_id'] - 1))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
        qwen_image_tensor = self.qwen_image_processor(img_0, return_tensors='pt')

        return img, qwen_image_tensor['pixel_values'], qwen_image_tensor['image_grid_thw'], target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]))]
            ),
            normalize
            ]

    if image_set == 'val':
        return [T.Compose([
            T.RandomResize([800], max_size=1333),
        ]),
        normalize]

    raise ValueError(f'unknown {image_set}')


def build(image_set="train", root_path="./datasets/hico_20150920", zero_shot_type="", qwen_path=""):
    root = Path(root_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / "annotations" / 'trainval_hico_ann.json',
                  ),
        'val': (
            root / 'images' / 'test2015', root / "annotations" / 'test_hico_ann.json',
            )
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file,
                            transforms=make_hico_transforms(image_set), zero_shot_type=zero_shot_type, qwen_path=qwen_path
                            )

    return dataset
