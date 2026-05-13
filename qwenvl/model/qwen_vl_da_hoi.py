from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModel
from torchvision.ops import batched_nms, box_iou, roi_align

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from transformers.generation.utils import GenerateOutput

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from qwenvl.model.grounding_model.modeling_grounding_dino_region import get_sine_pos_embed
from qwenvl.model.grounding_model.DETR import build
from transformers.image_transforms import center_to_corners_format
from qwenvl.model.grounding_model.matcher import HungarianRegionMatcher
from qwenvl.model.grounding_model.processing_grounding_dino import NewGroundingDinoProcessor, build_label_maps
from qwenvl.model.utils import xywh2xyxy, xyxy2xywh, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import numpy as np
from qwenvl.data.hico_class import HICO_ACTIONS, HICO_INTERACTIONS, HICO_OBJECTS, HOIINDEX2OBJECTVERB, VERB_MAPPER
import math
from qwenvl.model.interaction_head import HumanObjectMatcher
from qwenvl.model.adapter import AdapterLayer


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def token_sigmoid_binary_focal_loss(inputs, targets, batch_size, alpha=0.25, gamma=2.0, text_mask=None, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256
    bs, num_queries, max_text_len = inputs.size()
    text_mask = text_mask.unsqueeze(1).repeat(1, num_queries, 1).view(-1, max_text_len) # B, 1, 256

    inputs = inputs.view(-1, max_text_len)
    targets = targets.view(-1, max_text_len)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    loss = loss * text_mask
    if reduction:
        return loss.sum()
    else:
        return loss


def tokenwise_sigmoid_binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256

    # TODO: maybe try dual prompt with 2 classes

    num_queries, num_verb = inputs.size()

    inputs = inputs.view(1, num_verb)
    targets = targets.view(1, num_verb)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    accuracy = ((p > 0.5).to(targets.dtype) * targets).sum() / (targets.sum() + 1e-6)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if loss.sum() < 0:
        print(targets)
        print(alpha_t)
    if reduction:
        return loss.sum(), accuracy
    else:
        return loss.mean(), accuracy


def tokenwise_asymmetric_loss(inputs, targets, gamma_neg=4, gamma_pos=1, clip=0.05, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256

    # TODO: maybe try dual prompt with 2 classes

    num_queries, num_verb = inputs.size()

    inputs = inputs.view(1, num_verb)
    targets = targets.view(1, num_verb)

    p = torch.sigmoid(inputs)

    p_neg = 1 - p
    p_neg = (p_neg + clip).clamp(max=1)

    los_pos = targets * torch.log(p.clamp(min=1e-8))
    los_neg = (1 - targets) * torch.log(p_neg.clamp(min=1e-8))

    loss = los_pos + los_neg

    pt0 = p * targets
    pt1 = p_neg * (1 - targets)
    pt = pt0 + pt1

    one_sided_gamma = gamma_pos * targets + gamma_neg * (1 - targets)
    one_sided_w = torch.pow(1 - pt, one_sided_gamma)

    loss = loss * one_sided_w

    accuracy = ((p > 0.5).to(targets.dtype) * targets).sum() / (targets.sum() + 1e-6)

    if reduction:
        return - loss.sum() / (targets.sum() + 1e-6), accuracy
    else:
        return - loss.mean(), accuracy


def sigmoid_binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256
    num_queries, _ = inputs.size()

    inputs = inputs.view(-1, 1)
    targets = targets.view(-1, 1)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    accuracy = ((p > 0.5).to(targets.dtype) * targets).sum() / targets.sum()

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if loss.sum() < 0:
        print(targets)
        print(alpha_t)
    if reduction:
        return loss.sum(), accuracy
    else:
        return loss.mean(), accuracy


class Qwen2_5_VLRegionConfig(Qwen2_5_VLConfig):
    model_type = "qwen_vl_region"


class Qwen2_5_VLRegionForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLRegionConfig

    def __init__(self, config):
        super(Qwen2_5_VLRegionForConditionalGeneration, self).__init__(config)
        config.model_type = "qwen_vl_region"
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        self.matcher = HungarianRegionMatcher()
        self.already_load_grounding = False
        self.initialize_grounding_modules()
        self.log_losses = {}
        self.post_init()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def initialize_grounding_modules(self):
        # TODO: remember to set config
        if getattr(self.config, "grounding_module", None):
            module_path = self.config.grounding_module
            interaction_head_path = self.config.interaction_head_path
        else:
            return
        llm_hidden_dim = self.config.hidden_size
        self.grounding_module = build()
        self.interaction_head = HumanObjectMatcher(512)

        self.sap_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, llm_hidden_dim)
        )

        self.sap_verb_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, llm_hidden_dim)
        )
        
        # truncated
        self.num_text_token = 256
        self.num_queries = 900
        self.num_select_object = 20

        state_dict = torch.load(module_path)['model_state_dict']
        self.grounding_module.load_state_dict(state_dict)
        for n, m in self.grounding_module.named_modules():
            m._is_hf_initialized = True

        if interaction_head_path != "":
            state_dict = torch.load(interaction_head_path, map_location="cpu", weights_only=False)['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if "interaction_proj" in k:
                    new_state_dict[k[len("interaction_proj."):]] = v
            msg = self.interaction_head.load_state_dict(new_state_dict, strict=False)
            print(msg)
            for n, m in self.interaction_head.named_modules():
                m._is_hf_initialized = True

        HICO_OBJECT2HOI = {}
        for interaction in HICO_INTERACTIONS:
            if interaction['object'] not in HICO_OBJECT2HOI.keys():
                HICO_OBJECT2HOI[interaction['object']] = []
            if interaction['action'] == "no_interaction":
                continue
            else:
                act = interaction['action'].split("_")
                act[0] = VERB_MAPPER[act[0]]
                act = " ".join(act)
                s = f"A person is {act} {interaction['object']}"
                HICO_OBJECT2HOI[interaction['object']].append(s)
        self.HICO_OBJECT2HOI = HICO_OBJECT2HOI
        self.object_name_list = [item['name'] for item in HICO_OBJECTS]
        self.id2objectname = {index: item['name'] for index, item in enumerate(HICO_OBJECTS)}

        self.hoi_embedding_token = nn.Embedding(1, llm_hidden_dim)
        input_embeddings_avg = self.get_input_embeddings().weight.data.mean(dim=0, keepdim=True)
        self.hoi_embedding_token.weight.data.copy_(input_embeddings_avg)
        for n, m in self.hoi_embedding_token.named_modules():
            m._is_hf_initialized = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AdapterLayer):
            nn.init.xavier_uniform_(module.tune_adapter_a.weight)
            module.tune_adapter_a.bias.data.zero_()
            module.tune_adapter_b.weight.data.zero_()
            module.tune_adapter_b.bias.data.zero_()

    def get_grounding_module(self):
        grounding_module = getattr(self, "grounding_module", None)
        return grounding_module

    @torch.no_grad()
    def get_grounding_dino_results(self, scores, boxes, labels):
        keep = batched_nms(boxes, scores, labels, 0.5)
        boxes = boxes[keep].view(-1, 4)
        scores = scores[keep].view(-1)
        labels = labels[keep].view(-1)
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

        score = scores[keep]
        box = boxes[keep]
        labels = labels[keep]

        x, y = torch.meshgrid(
                torch.arange(box.shape[0], device=labels.device),
                torch.arange(box.shape[0], device=labels.device)
            )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < torch.sum(labels == 0))).unbind(1)

        merge_box = torch.cat((box[x_keep], box[y_keep]), dim=1)
        merge_classes = labels[y_keep]
        merge_scores = torch.cat((score[x_keep].unsqueeze(1), score[y_keep].unsqueeze(1)), dim=1)

        return box, labels, merge_box, merge_classes, merge_scores

    @torch.no_grad()
    def get_dino_results(self, dino_pixel_values):
        outputs = self.grounding_module(
                    dino_pixel_values)
        pred_logits = outputs['pred_logits'][0] # B, num_query, 256
        pred_boxes = outputs['pred_boxes'][0] # B, num_query, 4

        probs = F.softmax(pred_logits, -1)
        scores, labels = probs[..., :-1].max(-1)

        pred_boxes = center_to_corners_format(pred_boxes)

        keep = batched_nms(pred_boxes.float(), scores.float(), labels, 0.5)

        pred_logits = pred_logits[keep].view(-1, 81)
        pred_boxes = pred_boxes[keep].view(-1, 4)
        scores = scores[keep].view(-1)
        probs = probs[keep]
        labels = labels[keep].view(-1)

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

        score = scores[keep]
        box = pred_boxes[keep]
        select_logit = pred_logits[keep]
        labels = labels[keep]

        x, y = torch.meshgrid(
                torch.arange(box.shape[0], device=labels.device),
                torch.arange(box.shape[0], device=labels.device)
            )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < torch.sum(labels == 0))).unbind(1)

        merge_box = torch.cat((box[x_keep], box[y_keep]), dim=1)
        merge_classes = labels[y_keep]
        merge_scores = torch.cat((score[x_keep].unsqueeze(1), score[y_keep].unsqueeze(1)), dim=1)

        return box, labels, merge_box, merge_classes, merge_scores

    def construct_hoi_triplet_cache(self, results):
        grouped_results = []
        object_name_list = [item['name'] for item in HICO_OBJECTS]
        for r in results:
            labels_id = r['labels_id']
            logits = r['logits']
            boxes = r['boxes']
            hidden_state = r['hidden_state']
            scores = r['scores']

            if (labels_id == 0).sum() > 0:
                person_mask = labels_id == 0
            else:
                # NOTE: if the max label is not person, we select the highest person score as the person
                person_index = logits.softmax(-1)[:, 0].argmax() # 1
                person_mask = torch.arange(labels_id.shape[0], device=person_index.device) == person_index

            person_boxes = boxes[person_mask] # num, 4
            num_person = person_boxes.shape[0]
            assert num_person > 0
            object_boxes = boxes # num, 4
            num_object = object_boxes.shape[0]
            merge_object_name = []

            for i in range(labels_id.shape[0]):
                if person_mask[i]:
                    merge_object_name += [self.id2objectname[int(name_indexes)] for (object_name_id, name_indexes) in enumerate(labels_id) if object_name_id != i]

            person_hidden_state = hidden_state[person_mask].unsqueeze(1).repeat(1, num_object - 1, 1)
            person_score = scores[person_mask].unsqueeze(1).repeat(1, num_object - 1)

            select_mask = torch.arange(num_object).unsqueeze(0).to(labels_id.device) != torch.nonzero(person_mask)

            object_hidden_state = hidden_state.unsqueeze(0).repeat(num_person, 1, 1)[select_mask].view(num_person, num_object - 1, -1)
            object_logits = logits.unsqueeze(0).repeat(num_person, 1, 1)[select_mask]
            object_score = scores.unsqueeze(0).repeat(num_person, 1)[select_mask]

            triplet_boxes = torch.cat((person_boxes.unsqueeze(1).repeat(1, num_object - 1, 1), object_boxes.unsqueeze(0).repeat(num_person, 1, 1)[select_mask].view(num_person, num_object - 1, -1)), dim=-1).view(num_person * (num_object-1), 8)  # num, 1, 4 - 1, num, 4 -> num_person, num_object, 8
            grouped_results.append({"person_score": person_score, "object_score": object_score, "person_hidden_state": person_hidden_state, "object_hidden_state": object_hidden_state, "triplet_boxes": triplet_boxes, "object_logits": object_logits, "merge_object_name": merge_object_name})
        return grouped_results

    @torch.no_grad()
    def extract_pre_compute_match_result(
        self,
        dino_pixel_values,
        bboxes,
        objects
    ):
        gdino_hoi_results = self.get_dino_results(dino_pixel_values)
        grouped_hoi_triplets = self.construct_hoi_triplet_cache(gdino_hoi_results)
        assert len(bboxes) == 1
        for batch_id, (bbox, hoi_triplets, object) in enumerate(zip(bboxes, grouped_hoi_triplets, objects)):
            triplet_boxes = hoi_triplets['triplet_boxes'].float()
            object_logits = hoi_triplets['object_logits']
            sample_bboxes = torch.Tensor(bbox).to(object_logits)

            n = triplet_boxes.shape[0]
            match_labels = torch.zeros(n, sample_bboxes.shape[0], device=triplet_boxes.device)

            merge_object_names = hoi_triplets['merge_object_name']
            assert len(merge_object_names) == triplet_boxes.shape[0]

            x, y = torch.nonzero(torch.min(box_iou(triplet_boxes[:, :4], sample_bboxes[:, :4]), box_iou(triplet_boxes[:, 4:], sample_bboxes[:, 4:])) >= 0.5).unbind(1)
            match_labels[x, y] = 1

        return grouped_hoi_triplets[0], match_labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        select_rol_indices = None,
        response_labels = None,
        interaction_labels = None,
        sap_boxes = None,
        sap_labels = None,
        select_pooling_features=None
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw[:1])
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                if select_pooling_features is None:
                    sap_boxes = torch.as_tensor(sap_boxes).to(inputs_embeds.device)
                    sap_boxes = sap_boxes.to(inputs_embeds.device)

                    pairwise_tokens, pred_interaction = self.interaction_head(image_embeds.to(inputs_embeds.dtype), sap_boxes, sap_labels, image_grid_thw[0])

                    select_features = []
                    for rol_index in select_rol_indices:
                        select_features.append(pairwise_tokens[rol_index])
                    select_features = torch.stack(select_features, dim=0)
                    if select_features.isnan().any():
                        print(select_features)
                    select_verb_features = self.sap_verb_projector(select_features)
                    select_features = self.sap_projector(select_features)
                else:
                    select_features = self.sap_projector(select_pooling_features)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]

                # TODO: image batch = 1
                assert n_image_features * inputs_embeds.shape[0] == n_image_tokens

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1).view(-1, image_embeds.shape[-1]).contiguous()
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                hoi_mask = input_ids == self.config.hoi_token_id
                hoi_mask_unsqueezed = hoi_mask.unsqueeze(-1)
                hoi_mask_expanded = hoi_mask_unsqueezed.expand_as(inputs_embeds)
                hoi_mask_used = hoi_mask_expanded.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(hoi_mask_used, select_features)

                n_hoi_special_tokens = (input_ids == self.config.hoi_special_token_id).sum().item()
                hoi_special_mask = input_ids == self.config.hoi_special_token_id
                hoi_special_mask_unsqueezed = hoi_special_mask.unsqueeze(-1)
                hoi_special_mask_expanded = hoi_special_mask_unsqueezed.expand_as(inputs_embeds)
                hoi_special_mask = hoi_special_mask_expanded.to(inputs_embeds.device)

                inputs_embeds = inputs_embeds.masked_scatter(hoi_special_mask, self.hoi_embedding_token.weight.repeat(n_hoi_special_tokens, 1))

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        loss = 0
        sum_loss = None
        if labels is not None:
            binary_loss, accuracy = sigmoid_binary_focal_loss(pred_interaction, interaction_labels)
            sum_verb_accuracy = 0
            sum_sap_verb_accuracy = 0
            sum_sap_loss = 0
            for hidden_state, response_label, input, verb_features in zip(hidden_states, response_labels, input_ids, select_verb_features):
                token_mask = input == self.config.hoi_special_token_id # L
                select_hidden_state = hidden_state[token_mask] # N, C
                select_hidden_state = select_hidden_state / select_hidden_state.norm(dim=-1, keepdim=True)
                answer_hidden_state = select_hidden_state

                response_hidden_state = hidden_state[input == self.config.hoi_token_id]
                response_hidden_state = response_hidden_state / response_hidden_state.norm(dim=-1, keepdim=True)

                verb_features = verb_features.unsqueeze(0)
                verb_features = verb_features / verb_features.norm(dim=-1, keepdim=True)

                pred_logits = torch.einsum("bc,nc->bn", response_hidden_state, answer_hidden_state) * self.logit_scale.exp()
                verb_loss, verb_accuracy = tokenwise_sigmoid_binary_focal_loss(pred_logits, response_label)

                sap_pred_logits = torch.einsum("bc,nc->bn", verb_features, answer_hidden_state) * self.logit_scale.exp()
                sap_verb_loss, sap_verb_accuracy = tokenwise_sigmoid_binary_focal_loss(sap_pred_logits, response_label)

                loss = loss + verb_loss
                sum_verb_accuracy = sum_verb_accuracy + verb_accuracy

                sum_sap_loss = sum_sap_loss + sap_verb_loss
                sum_sap_verb_accuracy = sum_sap_verb_accuracy + sap_verb_accuracy

            loss = loss / hidden_states.shape[0]
            sum_verb_accuracy = sum_verb_accuracy / hidden_states.shape[0]

            sum_sap_loss = sum_sap_loss / hidden_states.shape[0]
            sum_sap_verb_accuracy = sum_sap_verb_accuracy / hidden_states.shape[0]

            self.log_losses['ce_loss'] = loss.item()
            self.log_losses['verb_accuracy'] = sum_verb_accuracy.item()
            self.log_losses['sap_loss'] = sum_sap_loss.item()
            self.log_losses['sap_verb_accuracy'] = sum_sap_verb_accuracy.item()
            self.log_losses['interaction_loss'] = binary_loss.item()
            self.log_losses['interaction_accuracy'] = accuracy.item()

            sum_loss = loss + binary_loss + sum_sap_loss

        if not return_dict:
            output = (None,) + outputs[1:]
            return (sum_loss,) + output if sum_loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=sum_loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    @torch.no_grad()
    def generate_bbox(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        detr_samples = None,
        **kwargs,
    ):
        outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_states = outputs.hidden_states[-1]

        sample_mean_features = []

        if isinstance(detr_samples, (list, torch.Tensor)):
            detr_samples = nested_tensor_from_tensor_list([detr_samples.to(dtype=hidden_states.dtype, device=hidden_states.device)])
        detr_pixel_value, detr_pixel_mask = detr_samples.decompose()
        detr_pixel_mask = ~detr_pixel_mask

        for batch_id, (input_id, hidden_state) in enumerate(zip(input_ids, hidden_states)):
            embed_token_index = torch.where(input_id == self.config.embed_token_index0)[0].tolist()

            per_sample_mean_features = []

            for embed_index in embed_token_index:
                text_features = hidden_state[embed_index:embed_index+4]
                per_sample_mean_features.append(text_features.mean(dim=0))
                text_features = hidden_state[embed_index+4:embed_index+8]
                per_sample_mean_features.append(text_features.mean(dim=0))

            sample_mean_features.append(torch.stack(per_sample_mean_features, dim=0)) # N, C

        sample_mean_features = torch.stack(sample_mean_features, dim=0) # B, N, C
        assert sample_mean_features.shape[1] < self.num_text_token
        if sample_mean_features.shape[1] < self.num_text_token:
            padded_text_features = torch.cat((sample_mean_features, torch.zeros((sample_mean_features.size(0), self.num_text_token - sample_mean_features.shape[1], sample_mean_features.shape[2])).to(sample_mean_features)), dim=1)
            text_masks = torch.cat((torch.ones((sample_mean_features.size(0), sample_mean_features.size(1)), dtype=torch.bool, device=sample_mean_features.device), torch.ones((sample_mean_features.size(0), self.num_text_token - sample_mean_features.size(1)), dtype=torch.bool, device=sample_mean_features.device)), dim=1)

        query_text_features = self.grounding_text_proj(padded_text_features)
        cls_sim, pred_bboxes_p, pred_bboxes_o, encoder_cls_sim, encoder_pred_bboxes_p = self.grounding_module(pixel_values=detr_pixel_value, text_query=query_text_features, text_query_masks=text_masks, pixel_mask=detr_pixel_mask)
        
        cls_sim = cls_sim.sigmoid() * text_masks.to(cls_sim.dtype)

        output_bboxes = torch.randn((sample_mean_features.shape[1], 4))
        output_prob = torch.randn((sample_mean_features.shape[1]))
        # only support batch=1
        select_sim = cls_sim[0][:, :sample_mean_features.shape[1]] # 64 * N
        C = - select_sim.cpu().detach()
        row_indices, col_indices = linear_sum_assignment(C)

        for i in range(sample_mean_features.shape[1]):
            output_bboxes[col_indices[i]] = pred_bboxes_p[0][row_indices[i]]
            output_prob[col_indices[i]] = select_sim[row_indices[i], i]
        output_bboxes = output_bboxes.view(sample_mean_features.shape[1] // 2, 8)
        output_prob = output_prob.view(sample_mean_features.shape[1] // 2, 2).mean(dim=-1)
        return {"bbox": output_bboxes, "prob": output_prob}

AutoConfig.register("qwen_vl_region", Qwen2_5_VLRegionConfig)
AutoModel.register(Qwen2_5_VLRegionConfig, Qwen2_5_VLRegionForConditionalGeneration)
