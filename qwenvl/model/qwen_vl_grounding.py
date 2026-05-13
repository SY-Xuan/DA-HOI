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

from qwenvl.model.grounding_model.vl_transformer import build_vl_transformer, ImagesEncoder
from qwenvl.model.grounding_model.matcher import HungarianMatcher
from qwenvl.model.grounding_model.backbone import build_backbone, NestedTensor
from qwenvl.model.utils import box_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import numpy as np


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


class Qwen2_5_VLGroundingConfig(Qwen2_5_VLConfig):
    model_type = "qwen_vl_grounding"


class Qwen2_5_VLGroundingForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLGroundingConfig

    def __init__(self, config):
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        config.model_type = "qwen_vl_grounding"
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        self.matcher = HungarianMatcher()
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

        self.grounding_backbone = build_backbone()
        self.grounding_images_encoder = ImagesEncoder()
        self.grounding_module = build_vl_transformer()
        self.images_encoder_proj = nn.Conv2d(self.grounding_backbone.num_channels, self.grounding_images_encoder.d_model, kernel_size=1)
        # fix now
        self.num_visu_token = 400
        # truncated
        self.num_text_token = 32
        self.num_total = self.num_visu_token + self.num_text_token + 2
        self.vl_pos_embed = nn.Embedding(self.num_total, self.grounding_module.d_model)
        self.reg_token = nn.Embedding(2, self.grounding_module.d_model)

        self.grounding_visual_proj = nn.Linear(self.grounding_images_encoder.d_model, self.grounding_module.d_model)
        self.grounding_text_proj = nn.Linear(llm_hidden_dim, self.grounding_module.d_model)

        self.person_bbox_embed = nn.Sequential(nn.Linear(self.grounding_module.d_model, self.grounding_module.d_model),
                                        nn.ReLU(),
                                        nn.Linear(self.grounding_module.d_model, self.grounding_module.d_model),
                                        nn.ReLU(),
                                        nn.Linear(self.grounding_module.d_model, 4))

        self.object_bbox_embed = nn.Sequential(nn.Linear(self.grounding_module.d_model, self.grounding_module.d_model),
                                        nn.ReLU(),
                                        nn.Linear(self.grounding_module.d_model, self.grounding_module.d_model),
                                        nn.ReLU(),
                                        nn.Linear(self.grounding_module.d_model, 4))

        state_dict = torch.load(module_path, map_location='cpu')['model']
        backbone_state_dict = {}
        transformers_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone"):
                backbone_state_dict[k[len('backbone.'): ]] = v
            elif k.startswith("transformer"):
                transformers_state_dict[k[len('transformer.'): ]] = v
        self.images_encoder_proj.weight.data.copy_(state_dict['input_proj.weight'])
        self.images_encoder_proj.bias.data.copy_(state_dict['input_proj.bias'])
        self.images_encoder_proj._is_hf_initialized = True

        msg = self.grounding_backbone.load_state_dict(backbone_state_dict, strict=False)
        print("backbone load: {}".format(msg))
        for n, m in self.grounding_backbone.named_modules():
            m._is_hf_initialized = True
        msg = self.grounding_images_encoder.load_state_dict(transformers_state_dict, strict=False)
        for n, m in self.grounding_images_encoder.named_modules():
            m._is_hf_initialized = True
        self.alreay_load_grounding = True

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

            detr_features, detr_pos = self.grounding_backbone(detr_samples)
            detr_src, detr_mask = detr_features[-1].decompose()

            detr_src = self.images_encoder_proj(detr_src)
            detr_src = detr_src.flatten(2).permute(2, 0, 1)
            detr_pos = detr_pos[-1].flatten(2).permute(2, 0, 1)
            detr_mask = detr_mask.flatten(1)
            memory = self.grounding_images_encoder(detr_src, mask=detr_mask, pos_embed=detr_pos) # L, B, C
            memory = memory.permute(1, 0, 2) # B, L, C

            sample_img_features = []
            sample_text_features = []
            sample_hois = []
            sample_bboxes = []
            sample_text_masks = []
            sample_hoi_ids = []
            sample_batch_ids = []

            for batch_id, (input_id, hidden_state, bbox, global_image_features, hoi_id) in enumerate(zip(input_ids, hidden_states, bboxes, memory, hoi_ids)):
                sample_hois.append(len(bbox))
                sample_bboxes.append(torch.Tensor(bbox).to(global_image_features)) # N * 8
                sample_hoi_ids.append(hoi_id.to(global_image_features.device))

                embed_token_index = torch.where(input_id == self.config.embed_token_index0)[0].tolist()
                for embed_token_start_index in embed_token_index:
                    sample_img_features.append(global_image_features.unsqueeze(0)) # B, L, C
                    # text_features = hidden_state[start_index:end_index+1]
                    text_features = hidden_state[embed_token_start_index:embed_token_start_index+8]
                    # zero paddings
                    if text_features.shape[0] < self.num_text_token:
                        padded_text_features = torch.cat((text_features, torch.zeros((self.num_text_token - text_features.shape[0], text_features.shape[1])).to(text_features)), dim=0)
                        text_masks = torch.cat((torch.zeros((1, text_features.shape[0]), dtype=torch.bool, device=text_features.device), torch.ones((1, self.num_text_token - text_features.shape[0]), dtype=torch.bool, device=text_features.device)), dim=1)
                    else:
                        padded_text_features = text_features[:self.num_text_token]
                        text_masks = torch.zeros((1, self.num_text_token), dtype=torch.bool, device=text_features.device)
                    sample_text_features.append(padded_text_features.unsqueeze(0)) # B, L, C
                    sample_text_masks.append(text_masks)
                    sample_batch_ids.append(batch_id)

            sample_img_features = torch.cat(sample_img_features, dim=0).permute(1, 0, 2) # L, B, C
            sample_img_features = self.grounding_visual_proj(sample_img_features)

            sample_text_features = torch.cat(sample_text_features, dim=0).permute(1, 0, 2)
            sample_text_features = self.grounding_text_proj(sample_text_features)

            sample_text_masks = torch.cat(sample_text_masks, dim=0)
            sample_bboxes = torch.cat(sample_bboxes, dim=0)
            sample_img_mask = torch.zeros((sample_img_features.shape[1], self.num_visu_token), dtype=torch.bool, device=sample_img_features.device)
            reg_token = self.reg_token.weight.unsqueeze(1).repeat(1, sample_img_features.shape[1], 1)
            reg_mask = torch.zeros((sample_img_features.shape[1], 2)).to(reg_token.device).to(torch.bool)

            vl_src = torch.cat([reg_token, sample_text_features, sample_img_features], dim=0)
            vl_mask = torch.cat([reg_mask, sample_text_masks, sample_img_mask], dim=1)
            vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, sample_img_features.shape[1], 1)
            vg_hs = self.grounding_module(vl_src, vl_mask, vl_pos)
            vg_hs_p = vg_hs[0]
            vg_hs_o = vg_hs[1]
            pred_bboxes_p = self.person_bbox_embed(vg_hs_p).sigmoid()
            pred_bboxes_o = self.object_bbox_embed(vg_hs_o).sigmoid()
            pred_bboxes = torch.cat((pred_bboxes_p, pred_bboxes_o), dim=1)

            sum_giou_loss = 0
            sum_iou_object = 0
            sum_iou_person = 0
            sum_l1_loss = 0

            sample_batch_ids = torch.Tensor(sample_batch_ids).to(global_image_features.device)
            sample_hoi_ids = torch.cat(sample_hoi_ids, dim=0)

            match_rol_indices, match_col_indices = self.matcher({"pred_logits": sample_hoi_ids, "pred_person_boxes": pred_bboxes[:, :4], "pred_object_boxes": pred_bboxes[:, 4:], "pred_batch_id": sample_batch_ids}, 
                                            {"labels": sample_hoi_ids, "person_boxes": sample_bboxes[:, :4], "object_boxes": sample_bboxes[:, 4:], "batch_id": sample_batch_ids})

            match_pred_bboxes = pred_bboxes[match_rol_indices]
            match_target_bboxes = sample_bboxes[match_col_indices]

            match_batch_ids = sample_batch_ids[match_rol_indices]
            pred_bbox_p, pred_bbox_o = match_pred_bboxes[:, :4], match_pred_bboxes[:, 4:]
            target_bbox_p, target_bbox_o = match_target_bboxes[:, :4], match_target_bboxes[:, 4:]

            loss_bbox_p = F.l1_loss(pred_bbox_p, xyxy2xywh(target_bbox_p), reduction='none').sum(dim=1)
            loss_bbox_o = F.l1_loss(pred_bbox_o, xyxy2xywh(target_bbox_o), reduction='none').sum(dim=1)

            loss_giou_p = 1 - torch.diag(generalized_box_iou(xywh2xyxy(pred_bbox_p), target_bbox_p))
            loss_giou_o = 1 - torch.diag(generalized_box_iou(xywh2xyxy(pred_bbox_o), target_bbox_o))

            iou_p = torch.diag(box_iou(xywh2xyxy(pred_bbox_p), target_bbox_p)[0])
            iou_o = torch.diag(box_iou(xywh2xyxy(pred_bbox_o), target_bbox_o)[0])

            for batch_id in torch.unique(match_batch_ids):
                sum_giou_loss = sum_giou_loss + (loss_giou_o[batch_id == match_batch_ids] + loss_giou_p[batch_id == match_batch_ids]).sum() / (batch_id == match_batch_ids).float().sum()
                sum_l1_loss = sum_l1_loss + (loss_bbox_p[batch_id == match_batch_ids] + loss_bbox_o[batch_id == match_batch_ids]).sum() / (batch_id == match_batch_ids).float().sum()
                sum_iou_object = sum_iou_object + iou_o[batch_id == match_batch_ids].sum() / (batch_id == match_batch_ids).float().sum()
                sum_iou_person = sum_iou_person + iou_p[batch_id == match_batch_ids].sum() / (batch_id == match_batch_ids).float().sum()

            sum_giou_loss = sum_giou_loss / hidden_states.shape[0]
            sum_l1_loss = sum_l1_loss / hidden_states.shape[0]
            sum_iou_object = sum_iou_object / hidden_states.shape[0]
            sum_iou_person = sum_iou_person / hidden_states.shape[0]

            self.log_losses['giou_loss'] = sum_giou_loss.item()
            self.log_losses['l1_loss'] = sum_l1_loss.item()
            self.log_losses['iou_object'] = sum_iou_object.item()
            self.log_losses['iou_person'] = sum_iou_person.item()
            self.log_losses['ce_loss'] = outputs.loss.item()

            sum_loss = outputs.loss + sum_giou_loss * 2.0 + sum_l1_loss * 5.0
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


        if isinstance(detr_samples, (list, torch.Tensor)):
            detr_samples = nested_tensor_from_tensor_list([detr_samples.to(dtype=hidden_states.dtype, device=hidden_states.device)])

        detr_features, detr_pos = self.grounding_backbone(detr_samples)
        detr_src, detr_mask = detr_features[-1].decompose()

        detr_src = self.images_encoder_proj(detr_src)
        detr_src = detr_src.flatten(2).permute(2, 0, 1)
        detr_pos = detr_pos[-1].flatten(2).permute(2, 0, 1)
        detr_mask = detr_mask.flatten(1)
        memory = self.grounding_images_encoder(detr_src, mask=detr_mask, pos_embed=detr_pos) # L, B, C
        memory = memory.permute(1, 0, 2) # B, L, C

        sample_img_features = []
        sample_text_features = []
        sample_text_masks = []

        for batch_id, (input_id, hidden_state, global_image_features) in enumerate(zip(input_ids, hidden_states, memory)):

            embed_token_index = torch.where(input_id == self.config.embed_token_index0)[0].tolist()
            for embed_token_start_index in embed_token_index:
                sample_img_features.append(global_image_features.unsqueeze(0)) # B, L, C
                # text_features = hidden_state[start_index:end_index+1]
                text_features = hidden_state[embed_token_start_index:embed_token_start_index+8]
                # zero paddings
                if text_features.shape[0] < self.num_text_token:
                    padded_text_features = torch.cat((text_features, torch.zeros((self.num_text_token - text_features.shape[0], text_features.shape[1])).to(text_features)), dim=0)
                    text_masks = torch.cat((torch.zeros((1, text_features.shape[0]), dtype=torch.bool, device=text_features.device), torch.ones((1, self.num_text_token - text_features.shape[0]), dtype=torch.bool, device=text_features.device)), dim=1)
                else:
                    padded_text_features = text_features[:self.num_text_token]
                    text_masks = torch.zeros((1, self.num_text_token), dtype=torch.bool, device=text_features.device)
                sample_text_features.append(padded_text_features.unsqueeze(0)) # B, L, C
                sample_text_masks.append(text_masks)

        sample_img_features = torch.cat(sample_img_features, dim=0).permute(1, 0, 2) # L, B, C
        sample_img_features = self.grounding_visual_proj(sample_img_features)

        sample_text_features = torch.cat(sample_text_features, dim=0).permute(1, 0, 2)
        sample_text_features = self.grounding_text_proj(sample_text_features)

        sample_text_masks = torch.cat(sample_text_masks, dim=0)
        sample_img_mask = torch.zeros((sample_img_features.shape[1], self.num_visu_token), dtype=torch.bool, device=sample_img_features.device)
        reg_token = self.reg_token.weight.unsqueeze(1).repeat(1, sample_img_features.shape[1], 1)
        reg_mask = torch.zeros((sample_img_features.shape[1], 2)).to(reg_token.device).to(torch.bool)

        vl_src = torch.cat([reg_token, sample_text_features, sample_img_features], dim=0)
        vl_mask = torch.cat([reg_mask, sample_text_masks, sample_img_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, sample_img_features.shape[1], 1)
        vg_hs = self.grounding_module(vl_src, vl_mask, vl_pos)
        vg_hs_p = vg_hs[0]
        vg_hs_o = vg_hs[1]
        pred_bboxes_p = self.person_bbox_embed(vg_hs_p).sigmoid()
        pred_bboxes_o = self.object_bbox_embed(vg_hs_o).sigmoid()
        pred_bboxes = torch.cat((pred_bboxes_p, pred_bboxes_o), dim=1)

        return {"bbox": pred_bboxes, "prob": 1.0}

AutoConfig.register("qwen_vl_grounding", Qwen2_5_VLGroundingConfig)
AutoModel.register(Qwen2_5_VLGroundingConfig, Qwen2_5_VLGroundingForConditionalGeneration)
