import argparse
from tqdm import tqdm

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
import json
import math
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from conversation import conv_templates, SeparatorStyle
from qwenvl.model.adapter import adapter
import transformers
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from qwenvl.model.utils import xywh2xyxy
from qwenvl.model.qwen_vl_da_hoi import Qwen2_5_VLRegionForConditionalGeneration
from qwenvl.model.grounding_model.processing_grounding_dino import NewGroundingDinoProcessor
from qwenvl.data.hico_class import *
import random
from torchvision.ops import roi_align


warnings.filterwarnings("ignore")
HOI_TOKEN = "<|box_start|>"
HOI_SPECIAL_TOKEN = "<|box_end|>"
HOI_FEATURES_TOKEN = "<|object_ref_end|>"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


HICO_OBJECT2HOI = {}
HOI2OBJECTNAME = {}
for interaction in HICO_INTERACTIONS:
    if interaction['object'] not in HICO_OBJECT2HOI.keys():
        HICO_OBJECT2HOI[interaction['object']] = []
    if interaction['action'] == "no_interaction":
        continue
    else:
        act = interaction['action'].split("_")
        act[0] = VERB_MAPPER[act[0]]
        act = " ".join(act)
        s = f"{act} a {interaction['object']}"
        HICO_OBJECT2HOI[interaction['object']].append(s)
        HOI2OBJECTNAME[s] = interaction['object']
object_name_list = [item['name'] for item in HICO_OBJECTS]
objectid2index = {item['id']: index for index, item in enumerate(HICO_OBJECTS)}


def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    device = "cuda"
    device_map = "auto"

    with adapter(enabled=True, hidden_dim=8, mlp=True, vision_enabled=False, non_linear=False):
        model = Qwen2_5_VLRegionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device, attn_implementation="flash_attention_2", use_cache=False
        )

    for n, p in model.named_parameters():
        if "tune_adapter_b" in n:
            print(f"{n}: {p}")

    processor = AutoProcessor.from_pretrained(model_path)

    transform = Compose([
        Resize(768, max_size=1333),
        # RandomResize(scales, max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    model.eval()
    model.to(device=device, dtype=torch.float16)

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    results_array = []

    for line_id, line in enumerate(tqdm(questions)):
        idx = line["img_id"]

        image_path = line["file_name"]

        filepath = "{}/{}".format(args.image_folder, image_path)
        image = Image.open(filepath).convert("RGB")
        ori_w, ori_h = image.size

        gt_boxes = []
        labels_id = []
        for bbox_anno in line['annotations']:
            gt_boxes.append(torch.tensor(bbox_anno['bbox']) / torch.tensor([ori_w, ori_h, ori_w, ori_h]))
            labels_id.append(objectid2index[bbox_anno['category_id']])
        gt_boxes = torch.stack(gt_boxes).to(device).view(-1, 4)
        labels_id = torch.tensor(labels_id).view(-1).to(device)
        scores = torch.ones_like(labels_id).to(device)

        pred_text_outputs = []
        pred_bboxes = []
        # TODO: should consider which classes be selected
        max_scores = []
        merge_pred_logits = []
        merge_potential_list = []
        pred_object_names = []

        with torch.inference_mode():
            visual_processed = processor.image_processor.preprocess(image, return_tensors="pt")
            size = visual_processed["image_grid_thw"][0]

            sap_boxes, sap_labels, merge_box, merge_classes, merge_scores = model.get_grounding_dino_results(
                scores, gt_boxes, labels_id,
            )
            if merge_box.shape[0] == 0:
                results_array.append({
                        "img_id": idx,
                        "pred_object_names": pred_object_names,
                        "bboxes": [],
                        "max_scores": max_scores,
                        "merge_potential_list": merge_potential_list,
                        "merge_pred_logits": merge_pred_logits,
                        })
                continue

            # NOTE: extract roi features
            image_embeds = model.visual(visual_processed['pixel_values'].to(dtype=torch.float16, device=device), grid_thw=visual_processed["image_grid_thw"].to(device=device))
            pairwise_tokens, interaction_scores = model.interaction_head(image_embeds, sap_boxes.to(device), sap_labels.to(device), visual_processed["image_grid_thw"][0].to(device=device))
            interaction_scores = interaction_scores.sigmoid()
            pairwise_tokens = pairwise_tokens.to(dtype=torch.float16, device=device)

            # select_interaction = torch.nonzero((interaction_scores) > 0.15).unbind(-1)[0]
            select_interaction = torch.nonzero((interaction_scores) > 0.15).unbind(-1)[0]

            w_index = [0, 2]
            h_index = [1, 3]

            for i in select_interaction:
                object_names = object_name_list[merge_classes[i]]
                potential_list = HICO_OBJECT2HOI[object_names]
                potential_list = copy.deepcopy(potential_list)
                random.shuffle(potential_list)

                bboxes = merge_box[i].float()
                person_bbox = bboxes[:4]
                object_bbox = bboxes[4:]

                pred_person_bbox = []
                pred_object_bbox = []
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

                question = "<|vision_start|><|image_pad|><|vision_end|>\nThe coordinates of the person are {}. The coordinates of the object are {}. The features of the interaction are {}. Select the correct interaction between the given person and object from the list: {}."
                question = question.format(f"{pred_person_bbox[0]},{pred_person_bbox[1]},{pred_person_bbox[2]},{pred_person_bbox[3]}", f"{pred_object_bbox[0]},{pred_object_bbox[1]},{pred_object_bbox[2]},{pred_object_bbox[3]}", f"<|vision_start|>{HOI_TOKEN}<|vision_end|>", HOI_SPECIAL_TOKEN.join(potential_list)+HOI_SPECIAL_TOKEN)

                args.conv_mode = "qwen_2"

                conv = copy.deepcopy(conv_templates[args.conv_mode])
                conv.append_message(conv.roles[0], question)
                # conv.append_message(conv.roles[1], HOI_SPECIAL_TOKEN)
                prompt_question = conv.get_prompt()

                inputs = processor(
                    text=[prompt_question],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    image_grid_thw=inputs.image_grid_thw,
                    pixel_values=inputs.pixel_values.to(dtype=torch.float16, device=device),
                    select_pooling_features=pairwise_tokens[i].unsqueeze(0),
                    output_hidden_states=True
                )
                assert outputs.hidden_states[-1].shape[0] == 1
                for hidden_state, input in zip(outputs.hidden_states[-1], inputs.input_ids):
                    token_mask = input == model.config.hoi_special_token_id # L
                    select_hidden_state = hidden_state[token_mask] # N, C
                    select_hidden_state = select_hidden_state / select_hidden_state.norm(dim=-1, keepdim=True)
                    answer_hidden_state = select_hidden_state

                    response_hidden_state = hidden_state[input == model.config.hoi_token_id]
                    response_hidden_state = response_hidden_state / response_hidden_state.norm(dim=-1, keepdim=True)

                    pred_logits = torch.einsum("bc,nc->bn", response_hidden_state, answer_hidden_state) * model.logit_scale.exp()
                    
                    verb_features = model.sap_verb_projector(pairwise_tokens[i].unsqueeze(0))
                    verb_features = verb_features / verb_features.norm(dim=-1, keepdim=True)
                    sap_pred_logits = torch.einsum("bc,nc->bn", verb_features, answer_hidden_state) * model.logit_scale.exp()

                    merge_pred_logits.append((pred_logits.sigmoid().cpu(), sap_pred_logits.sigmoid().cpu()))

                merge_potential_list.append(potential_list)
                pred_bboxes.append(merge_box[i])
                pred_object_names.append(object_names)
                max_scores.append([float(merge_scores[i, 0]), float(merge_scores[i, 1]), float(interaction_scores[i])])

            results_array.append({
                        "img_id": idx,
                        "pred_object_names": pred_object_names,
                        "bboxes": [bboxes[:4].clamp(min=0.0, max=1.0).cpu().tolist() + bboxes[4:].clamp(min=0.0, max=1.0).cpu().tolist() for bboxes in pred_bboxes],
                        "max_scores": max_scores,
                        "merge_potential_list": merge_potential_list,
                        "merge_pred_logits": merge_pred_logits,
                        })
        if (line_id + 1) % 20 == 0:
            torch.save(results_array, answers_file)
    torch.save(results_array, answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="tables/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--potential-list", action='store_true')
    args = parser.parse_args()

    eval_model(args)
