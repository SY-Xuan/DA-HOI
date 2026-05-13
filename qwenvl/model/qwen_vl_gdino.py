from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModel

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from transformers.generation.utils import GenerateOutput

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from qwenvl.model.grounding_model.modeling_grounding_dino import GroundingDinoForObjectDetection
from qwenvl.model.grounding_model.matcher import HungarianGDINOMatcher
from qwenvl.model.utils import box_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import numpy as np


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


class Qwen2_5_VLGDINOConfig(Qwen2_5_VLConfig):
    model_type = "qwen_vl_gdino"


class Qwen2_5_VLGDINOForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLGDINOConfig

    def __init__(self, config):
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        config.model_type = "qwen_vl_gdino"
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        self.matcher = HungarianGDINOMatcher()
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
        else:
            return
        llm_hidden_dim = self.config.hidden_size
        self.grounding_module = GroundingDinoForObjectDetection.from_pretrained(module_path)
        
        # truncated
        self.num_text_token = 256
        self.num_queries = 900
        self.grounding_text_proj = nn.Linear(llm_hidden_dim, self.grounding_module.config.d_model)

    def get_grounding_module(self):
        grounding_module = getattr(self, "grounding_module", None)
        return grounding_module

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
        detr_samples = None,
        bboxes = None,
        hoi_ids = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        if self.training:
            outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    second_per_grid_ts=second_per_grid_ts,
                )
            hidden_states = outputs.hidden_states[-1]

            if isinstance(detr_samples, (list, torch.Tensor)):
                detr_samples = nested_tensor_from_tensor_list(detr_samples)
            detr_pixel_value, detr_pixel_mask = detr_samples.decompose()
            detr_pixel_mask = ~detr_pixel_mask

            sample_bboxes = []
            sample_hoi_ids = []
            sample_mean_features = []

            for batch_id, (input_id, hidden_state, bbox, hoi_id) in enumerate(zip(input_ids, hidden_states, bboxes, hoi_ids)):
                embed_token_index = torch.where(input_id == self.config.embed_token_index0)[0].tolist()
                assert len(bbox) == (input_id == self.config.embed_token_index0).sum()
                sample_bboxes.append(torch.Tensor(bbox).to(hidden_state)) # N * 8
                sample_hoi_ids.append(hoi_id.to(hidden_state.device))

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
            sample_bboxes = torch.stack(sample_bboxes, dim=0) # B, N, 8
            cls_sim, pred_bboxes_p, pred_bboxes_o, encoder_cls_sim, encoder_pred_bboxes_p = self.grounding_module(pixel_values=detr_pixel_value, text_query=query_text_features, text_query_masks=text_masks, pixel_mask=detr_pixel_mask)

            sample_hoi_ids = torch.cat(sample_hoi_ids, dim=0)
            sum_giou_loss = 0
            sum_l1_loss = 0
            sum_iou = 0
            sum_cls_loss = 0

            sum_aux_giou_loss = 0
            sum_aux_l1_loss = 0
            sum_aux_cls_loss = 0

            indices = self.matcher({"pred_cls": cls_sim, "pred_boxes": pred_bboxes_p}, 
                                            {"person_boxes": sample_bboxes[:, :, :4], "object_boxes": sample_bboxes[:, :, 4:], "text_mask": text_masks})

            merge_sample_bboxes = sample_bboxes.view(sample_bboxes.shape[0], sample_bboxes.shape[1] * 2, 4)

            idx = self._get_src_permutation_idx(indices)
            src_boxes = pred_bboxes_p[idx]
            target_boxes = torch.cat([t[i] for t, (_, i) in zip(merge_sample_bboxes, indices)], dim=0)

            loss_bbox = F.l1_loss(src_boxes, xyxy2xywh(target_boxes), reduction='none').sum(dim=1)

            loss_giou = 1 - torch.diag(generalized_box_iou(xywh2xyxy(src_boxes), target_boxes))

            iou = torch.diag(box_iou(xywh2xyxy(src_boxes), target_boxes)[0])

            target_classes = torch.zeros(cls_sim.shape,
                                dtype=cls_sim.dtype, device=src_boxes.device)
            target_labels = []
            for bid in range(merge_sample_bboxes.shape[0]):
                num_bbox = merge_sample_bboxes[bid].shape[0]
                tgt_label = torch.eye(cls_sim.shape[2]).to(merge_sample_bboxes.device, dtype=merge_sample_bboxes.dtype)
                tgt_label = tgt_label[:num_bbox]
                target_labels.append(tgt_label)
            target_labels = torch.stack(target_labels, dim=0)

            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_labels, indices)])
            target_classes[idx] = target_classes_o

            cls_loss = token_sigmoid_binary_focal_loss(cls_sim, target_classes, pred_bboxes_p.size(0), text_mask=text_masks)

            sum_giou_loss = sum_giou_loss + loss_giou.sum().to(cls_sim.dtype) / (merge_sample_bboxes.size(0) * merge_sample_bboxes.size(1))
            sum_l1_loss = sum_l1_loss + loss_bbox.sum() / (merge_sample_bboxes.size(0) * merge_sample_bboxes.size(1))
            sum_iou = iou.mean()
            sum_cls_loss = sum_cls_loss + cls_loss.sum() / (merge_sample_bboxes.size(0) * merge_sample_bboxes.size(1))

            indices = self.matcher({"pred_cls": encoder_cls_sim, "pred_boxes": encoder_pred_bboxes_p},
                            {"person_boxes": sample_bboxes[:, :, :4], "object_boxes": sample_bboxes[:, :, 4:], "text_mask": text_masks})

            idx = self._get_src_permutation_idx(indices)
            src_boxes = encoder_pred_bboxes_p[idx]
            target_boxes = torch.cat([t[i] for t, (_, i) in zip(merge_sample_bboxes, indices)], dim=0)

            encoder_loss_bbox_p = F.l1_loss(src_boxes, xyxy2xywh(target_boxes), reduction='none').sum(dim=1)

            encoder_loss_giou_p = 1 - torch.diag(generalized_box_iou(xywh2xyxy(src_boxes), target_boxes))

            target_classes = torch.zeros(encoder_cls_sim.shape,
                                dtype=encoder_cls_sim.dtype, device=src_boxes.device)
            target_labels = []
            for bid in range(merge_sample_bboxes.shape[0]):
                num_bbox = merge_sample_bboxes[bid].shape[0]
                tgt_label = torch.eye(encoder_cls_sim.shape[2]).to(merge_sample_bboxes.device, dtype=merge_sample_bboxes.dtype)
                tgt_label = tgt_label[:num_bbox]
                target_labels.append(tgt_label)
            target_labels = torch.stack(target_labels, dim=0)

            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_labels, indices)])
            target_classes[idx] = target_classes_o

            encoder_cls_loss = token_sigmoid_binary_focal_loss(encoder_cls_sim, target_classes, encoder_pred_bboxes_p.size(0), text_mask=text_masks)

            sum_aux_giou_loss = sum_aux_giou_loss + encoder_loss_giou_p.sum().to(encoder_cls_sim.dtype) / (merge_sample_bboxes.size(0) * merge_sample_bboxes.size(1))
            sum_aux_l1_loss = sum_aux_l1_loss + encoder_loss_bbox_p.sum() / (merge_sample_bboxes.size(0) * merge_sample_bboxes.size(1))
            sum_aux_cls_loss = sum_aux_cls_loss + encoder_cls_loss.sum() / (merge_sample_bboxes.size(0) * merge_sample_bboxes.size(1))

            self.log_losses['giou_loss'] = sum_giou_loss.item()
            self.log_losses['l1_loss'] = sum_l1_loss.item()
            self.log_losses['iou'] = sum_iou.item()
            self.log_losses['ce_loss'] = outputs.loss.item()
            self.log_losses['cls_loss'] = sum_cls_loss.item()
            self.log_losses['aux_giou_loss'] = sum_aux_giou_loss.item()
            self.log_losses['aux_l1_loss'] = sum_aux_l1_loss.item()
            self.log_losses['aux_cls_loss'] = sum_aux_cls_loss.item()
            sum_loss = outputs.loss + sum_giou_loss * 2.0 + sum_l1_loss * 5.0 + 2.0 * sum_cls_loss + sum_aux_giou_loss * 2.0 + sum_aux_l1_loss * 5.0 + 2.0 * sum_aux_cls_loss
            if not return_dict:
                output = (outputs.logits, ) + outputs[1:]
                return (sum_loss, ) + output if sum_loss is not None else output
            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=sum_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas)

        else:
            return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    second_per_grid_ts=second_per_grid_ts,
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
        
        # pred_bboxes = torch.cat((pred_bboxes_p, pred_bboxes_o), dim=2) # B, N, 8

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

AutoConfig.register("qwen_vl_gdino", Qwen2_5_VLGDINOConfig)
AutoModel.register(Qwen2_5_VLGDINOConfig, Qwen2_5_VLGDINOForConditionalGeneration)
