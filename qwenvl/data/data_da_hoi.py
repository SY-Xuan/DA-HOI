import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
# from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .hico_class import VERB_MAPPER, HICO_ACTIONS, HICO_OBJECTS, HICO_INTERACTIONS, HOIINDEX2OBJECTVERB, hico_unseen_index
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_25_with_region
from torchvision.transforms import Compose, ColorJitter, Resize, ToTensor, Normalize, RandomChoice
from torchvision.ops import batched_nms, box_iou
from transformers.image_transforms import center_to_corners_format
import qwenvl.data.transforms as T
from qwenvl.model.grounding_model.DETR import build
import torch.nn.functional as F


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
HOI_START_TOKEN = "<hoi_start>"
HOI_END_TOKeN = "<hoi_end>"
HOI_EMBED_TOKEN0 = "<embed0>"
HOI_EMBED_TOKEN1 = "<embed1>"
HOI_EMBED_TOKEN2 = "<embed2>"
HOI_EMBED_TOKEN3 = "<embed3>"
HOI_EMBED_TOKEN4 = "<embed4>"
HOI_EMBED_TOKEN5 = "<embed5>"
HOI_EMBED_TOKEN6 = "<embed6>"
HOI_EMBED_TOKEN7 = "<embed7>"
HOI_SPECIAL_TOKEN = "<|box_end|>"
HOI_FEATURES_TOKEN = "<|object_ref_end|>"
HOI_TOKEN = "<|box_start|>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]



def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset_list = data_args.dataset_use.split(",")
        # dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        self.get_rope_index = get_rope_index_25

        list_data_dict = []
        self.ids = []
        self.unseen_index = hico_unseen_index.get(data_args.zero_shot_type, [])
        print(self.unseen_index)

        for data in dataset_list:
            self.annotations = json.load(open(data, "r"))
            rank0_print(f"dataset name: {data}")
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
                    self.annotations[idx]['hoi_annotation'] = new_img_anno

        rank0_print(f"Total training samples: {len(self.ids)}")

        random.shuffle(self.ids)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        HICO_OBJECT2HOI = {}
        for interaction in HICO_INTERACTIONS:
            if "unseen_verb" == data_args.zero_shot_type:
                if interaction['interaction_id'] in hico_unseen_index['unseen_verb']:
                    continue
            elif "non_rare_first" == data_args.zero_shot_type:
                if interaction['interaction_id'] in hico_unseen_index['non_rare_first']:
                    continue
            elif "rare_first" == data_args.zero_shot_type:
                if interaction['interaction_id'] in hico_unseen_index['rare_first']:
                    continue

            if interaction['object'] not in HICO_OBJECT2HOI.keys():
                HICO_OBJECT2HOI[interaction['object']] = []
            if interaction['action'] == "no_interaction":
                continue
                # HICO_OBJECT2HOI[interaction['object']].append(f"A person is not interacting with {interaction['object']}")
            else:
                act = interaction['action'].split("_")
                act[0] = VERB_MAPPER[act[0]]
                act = " ".join(act)
                # s = f"A person is {act} a {interaction['object']}"
                s = f"{act} a {interaction['object']}"
                HICO_OBJECT2HOI[interaction['object']].append(s)
        self.HICO_OBJECT2HOI = HICO_OBJECT2HOI
        self.object_name_list = [item['name'] for item in HICO_OBJECTS]
        objectid2index = {}
        for index, object_item in enumerate(HICO_OBJECTS):
            objectid2index[object_item['id']] = index
        self.objectid2index = objectid2index

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        self.qwen_augmentation = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]))]
            )

    def __len__(self):
        return len(self.ids)

    @torch.no_grad()
    def get_detection_results(self, pixel_values):
        outputs = self.grounding_module(
                    pixel_values)
        hidden_states = outputs['last_hidden_state'][0] # B, num_query, dim
        pred_logits = outputs['pred_logits'][0] # B, num_query, 256
        pred_boxes = outputs['pred_boxes'][0] # B, num_query, 4

        probs = F.softmax(pred_logits, -1)
        scores, labels = probs[..., :-1].max(-1)

        pred_boxes = center_to_corners_format(pred_boxes)

        keep = batched_nms(pred_boxes.float(), scores.float(), labels, 0.5)

        hidden_states = hidden_states[keep].view(-1, hidden_states.shape[-1])
        pred_logits = pred_logits[keep].view(-1, 81)
        pred_boxes = pred_boxes[keep].view(-1, 4)
        scores = scores[keep]
        probs = probs[keep]
        labels = labels[keep]

        box_threshold = 0.2

        keep = torch.nonzero(scores >= box_threshold).squeeze(1)

        is_human = labels == 0
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human[keep].sum()
        n_object = len(keep) - n_human

        min_instances = 3
        max_instances = 15

        # Keep the number of human and object instances in a specified interval
        if n_human < min_instances:
            keep_h = scores[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = scores[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            keep_h = keep[keep_h]

        if n_object < min_instances:
            keep_o = scores[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = scores[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            keep_o = keep[keep_o]

        keep = torch.cat([keep_h, keep_o])

        results = []

        score = scores[keep]
        box = pred_boxes[keep]
        select_probs = probs[keep]
        select_logit = pred_logits[keep]
        select_hidden_state = hidden_states[keep]
        group = labels[keep]

        results.append({"scores": score, "logits": select_logit, "boxes": box, "hidden_state": select_hidden_state, "labels_id": group})
        return results

    def process_image_unified(self, image):
        processor = copy.deepcopy(self.data_args.image_processor)

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]


        return image_tensor, grid_thw

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30
        while True:
            sample, success = self._get_item(i)
            if success:
                break
            else:
                i = random.randint(0, len(self.ids) - 1)
        return sample

    def _jitter_bbox(self, box):
        # xyxy to xywh
        scale_jitter_factor = 0.25
        center_jitter_factor = 2.0
        threshold = 0.7

        original_dtype = box.dtype
        while True:
            xywh_box = torch.tensor((box[0], box[1], box[2] - box[0], box[3] - box[1])).float()
            jittered_size = xywh_box[2:4] * torch.exp(torch.randn(2) * scale_jitter_factor)
            max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor).float())
            jittered_center = xywh_box[0:2] + 0.5 * xywh_box[2:4] + max_offset * (torch.rand(2) - 0.5)

            jitter_bbox = torch.cat((jittered_center - 0.5 * jittered_size, jittered_center + 0.5 * jittered_size), dim=0).clamp(0, 1.0)
            if box_iou(box.unsqueeze(0), jitter_bbox.unsqueeze(0))[0, 0] > threshold:
                break

        return jitter_bbox.to(original_dtype)

    def merge_gt(self, boxes, classes, kept_box_indices):
        kept_box_indices = torch.as_tensor(kept_box_indices).bool()
        if kept_box_indices.float().sum() == 0:
            return None, None, None, None, False
        boxes = boxes[kept_box_indices]
        classes = classes[kept_box_indices]
        scores = torch.ones(classes.shape[0]).to(classes.device)

        keep = batched_nms(boxes.float(), scores.float(), classes, 0.5)

        boxes = boxes[keep].view(-1, 4)
        classes = classes[keep].view(-1)

        is_human = classes == 0
        hum = torch.nonzero(is_human).squeeze(1)
        n_human = is_human.sum()
        if n_human == 0:
            return None, None, None, None, False
        obj = torch.nonzero(is_human == 0).squeeze(1)

        keep = torch.cat([hum, obj])

        box = boxes[keep]
        labels = classes[keep]

        x, y = torch.meshgrid(
                torch.arange(box.shape[0]),
                torch.arange(box.shape[0])
            )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_human)).unbind(1)

        merge_box = torch.cat((box[x_keep], box[y_keep]), dim=1)
        merge_classes = labels[y_keep]
        return box, labels, merge_box, merge_classes, True

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        img_anno = self.annotations[self.ids[i]]
        # define some variables
        grid_thw_merged = None
        grid_thw = None

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = [self.objectid2index[obj['category_id']] for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        image = Image.open(os.path.join(self.data_args.image_path, img_anno['file_name'])).convert('RGB')
        w, h = image.size
        boxes = boxes / torch.as_tensor([w, h, w, h], dtype=boxes.dtype)
        boxes.clamp_(min=0, max=1.0)

        target = {"single_gt_bboxes": boxes}
        image, target = self.qwen_augmentation(image, target)
        boxes = target['single_gt_bboxes']
        if "single_gt_bboxes_keep" in target:
            kept_box_indices = target['single_gt_bboxes_keep']
        else:
            kept_box_indices = []
            for b in boxes:
                if b[2] > (b[0] + 0.005) and b[3] > (b[1] + 0.005):
                    kept_box_indices.append(1)
                else:
                    kept_box_indices.append(0)

        image, grid_thw = self.process_image_unified(image)
        image = [image]
        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // self.data_args.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]

        gt_object_names = []
        sub_obj_pairs = {}
        pairs_list = []

        sub_boxes = []
        obj_boxes = []

        for hoi in img_anno['hoi_annotation']:
            if kept_box_indices[hoi['subject_id']] != 1 or kept_box_indices[hoi['object_id']] != 1:
                continue
            sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
            action = HICO_ACTIONS[hoi['category_id'] - 1]['name']
            object = self.object_name_list[classes[hoi['object_id']]]
            if sub_obj_pair not in sub_obj_pairs:
                sub_obj_pairs[sub_obj_pair] = []
                sub_boxes.append(boxes[hoi['subject_id']])
                obj_boxes.append(boxes[hoi['object_id']])
                gt_object_names.append(object)
                pairs_list.append(sub_obj_pair)

            if action != "no_interaction":
                act = action.split("_")
                act[0] = VERB_MAPPER[act[0]]
                act = " ".join(act)
                s = f"{act} a {object}"
            else:
                s = "no"
            if s not in sub_obj_pairs[sub_obj_pair]:
                sub_obj_pairs[sub_obj_pair].append(s)
        assert len(sub_boxes) == len(obj_boxes) == len(pairs_list) == len(gt_object_names)
        if len(sub_boxes) == 0:
            return {}, False
        sub_boxes = torch.stack(sub_boxes)
        obj_boxes = torch.stack(obj_boxes)

        gt_bboxes = torch.cat((sub_boxes, obj_boxes), dim=1)
        assert gt_bboxes.shape[0] == len(gt_object_names) == len(sub_obj_pairs.keys())

        sap_boxes, sap_labels, merge_boxes, merge_labels, success = self.merge_gt(boxes, classes, kept_box_indices)

        if not success:
            return {}, False
        for b in sap_boxes:
            if b[3] <= b[1] or b[2] <= b[0]:
                print(b)
                print(boxes)
                print(kept_box_indices)

        for b in merge_boxes:
            if b[3] <= b[1] or b[2] <= b[0] or b[6] <= b[4] or b[7] <= b[5]:
                print(b)
                print(boxes)
                print(merge_boxes)
                print(kept_box_indices)

        new_sources = []

        positive_selected = 4
        size = grid_thw[0]

        w_index = [0, 2]
        h_index = [1, 3]

        gt_match_labels = torch.zeros(merge_boxes.shape[0], gt_bboxes.shape[0])
        x, y = torch.nonzero(torch.min(box_iou(merge_boxes[:, :4], gt_bboxes[:, :4]), box_iou(merge_boxes[:, 4:], gt_bboxes[:, 4:])) >= 0.5).unbind(1)
        gt_match_labels[x, y] = 1.0

        match_rol_indices = torch.nonzero(gt_match_labels.sum(dim=1))
        perm = torch.randperm(match_rol_indices.shape[0])
        select_rol_indices = []
        response_labels = []

        for rol_index in match_rol_indices[perm]:
            rol_index = rol_index[0]
            select_bboxes = merge_boxes[rol_index].float().clamp(0.0, 1.0)

            object_names = self.object_name_list[merge_labels[rol_index]]
            potential_list = self.HICO_OBJECT2HOI[object_names]
            potential_list = copy.deepcopy(potential_list)
            random.shuffle(potential_list)

            person_bbox = select_bboxes[:4]
            object_bbox = select_bboxes[4:]

            merge_sentence = []

            for col_index in torch.nonzero(gt_match_labels[rol_index]):
                col_index = col_index[0]
                if gt_object_names[col_index] != object_names:
                    continue
                merge_sentence += sub_obj_pairs[pairs_list[col_index]]

            if len(merge_sentence) == 0:
                continue

            pred_person_bbox = []
            pred_object_bbox = []

            # person_bbox = self._jitter_bbox(person_bbox.clamp(0, 1.0))
            # object_bbox = self._jitter_bbox(object_bbox.clamp(0, 1.0))

            for index, r in enumerate(person_bbox.clamp(0, 1.0)):
                if index in w_index:
                    pred_person_bbox.append(int(r * (size[2] * 14)))
                elif index in h_index:
                    pred_person_bbox.append(int(r * (size[1] * 14)))
            for index, r in enumerate(object_bbox.clamp(0, 1.0)):
                if index in w_index:
                    pred_object_bbox.append(int(r * (size[2] * 14)))
                if index in h_index:
                    pred_object_bbox.append(int(r * (size[1] * 14)))

            question = "<image>\nThe coordinates of the person are {}. The coordinates of the object are {}. The features of the interaction are {}. Select the correct interaction between the given person and object from the list: {}."
            question = question.format(f"{pred_person_bbox[0]},{pred_person_bbox[1]},{pred_person_bbox[2]},{pred_person_bbox[3]}", f"{pred_object_bbox[0]},{pred_object_bbox[1]},{pred_object_bbox[2]},{pred_object_bbox[3]}", f"<|vision_start|>{HOI_TOKEN}<|vision_end|>", HOI_SPECIAL_TOKEN.join(potential_list)+HOI_SPECIAL_TOKEN)

            response = (", ".join(merge_sentence)).split(", ")

            response_label = torch.zeros((len(potential_list)))
            for r in response:
                response_label[potential_list.index(r)] = 1.0
            new_sources.append([[{"from": "human", "value": question}]])
            select_rol_indices.append(rol_index)
            response_labels.append(response_label)

            if len(new_sources) == positive_selected:
                break

        if len(new_sources) == 0:
            return {}, False

        merge_input_ids = []
        merge_labels = []
        merge_image_grid_thw = []

        for source in new_sources:
            new_inputs = preprocess_qwen_2_visual(source, self.tokenizer, grid_thw_image=grid_thw_merged if grid_thw_merged else None)
            merge_input_ids.append(new_inputs['input_ids'])
            merge_labels.append(new_inputs['labels'])
            merge_image_grid_thw.append(grid_thw)

        data_dict = dict()
        data_dict["input_ids"] = merge_input_ids
        data_dict['labels'] = merge_labels
        data_dict['select_rol_indices'] = select_rol_indices
        data_dict['response_labels'] = response_labels
        data_dict['sap_boxes'] = sap_boxes
        data_dict['sap_labels'] = sap_labels
        data_dict['interaction_labels'] = (gt_match_labels.sum(dim=1) > 0).to(gt_match_labels.dtype)

        position_ids = []
        for single_input_ids, single_image_grid_thw in zip(merge_input_ids, merge_image_grid_thw):
            position_id, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                single_input_ids,
                image_grid_thw=torch.stack(single_image_grid_thw, dim=0) if single_image_grid_thw else None,
            )
            position_ids.append(position_id)

        data_dict["position_ids"] = position_ids

        # if "image" in self.list_data_dict[i]:
        data_dict["pixel_values"] = torch.cat(image, dim=0)
        data_dict["image_grid_thw"] = torch.cat(
            [thw[0].unsqueeze(0) for thw in merge_image_grid_thw], dim=0
        )

        return data_dict, True


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids[0]]
        labels = [ids.squeeze(0) for ids in labels[0]]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids[0])
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = instances[0]['image_grid_thw']
        else:
            concat_images = None
            grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["position_ids"] = position_ids

        batch['response_labels'] = instances[0]['response_labels']
        batch['sap_boxes'] = instances[0]['sap_boxes'].numpy()
        batch['sap_labels'] = instances[0]['sap_labels']
        batch['select_rol_indices'] = instances[0]['select_rol_indices']
        batch['interaction_labels'] = instances[0]['interaction_labels']

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
