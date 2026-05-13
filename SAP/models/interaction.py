from qwenvl.model.grounding_model.DETR import build
from torchvision.ops import batched_nms, box_iou
from transformers.image_transforms import center_to_corners_format

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math
from .interaction_head import HumanObjectMatcher
import torch.distributed as dist
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor
from torchvision.ops import roi_align


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


class InteractionModel(nn.Module):
    def __init__(self, detector_path=None, qwen_path=None):
        super(InteractionModel, self).__init__()
        self.grounding_module = build()
        self.interaction_proj = HumanObjectMatcher(512)

        state_dict = torch.load(detector_path)['model_state_dict']
        self.grounding_module.load_state_dict(state_dict)
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
        self.qwen_visual = qwen.visual

    @torch.no_grad()
    def get_detection_results_gt(self, pixel_values, qwen_pixel_values, image_grid_thw, targets):
        qwen_visual_hidden_states = self.qwen_visual(hidden_states=qwen_pixel_values.to(torch.bfloat16), grid_thw=image_grid_thw)
        split_grid_thw_merged = [merged_thw.prod() // 4
                for merged_thw in image_grid_thw]
        qwen_visual_hidden_states = torch.split(qwen_visual_hidden_states, split_grid_thw_merged, dim=0)

        results = []
        global_features = []
        for target, qwen_hidden_state, grid_thw in zip(targets, qwen_visual_hidden_states, image_grid_thw):
            labels = target['labels']
            gt_boxes = target['boxes']
            scores = torch.ones(labels.shape[0]).to(labels.device)

            keep = batched_nms(gt_boxes.float(), scores.float(), labels, 0.5)

            gt_boxes = gt_boxes[keep].view(-1, 4)
            scores = scores[keep].view(-1)
            labels = labels[keep].view(-1)

            is_human = labels == 0
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human.sum()
            n_object = gt_boxes.shape[0] - n_human

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
                keep_h = hum

            if n_object < min_instances:
                keep_o = scores[obj].argsort(descending=True)[:min_instances]
                keep_o = obj[keep_o]
            elif n_object > max_instances:
                keep_o = scores[obj].argsort(descending=True)[:max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = obj

            keep = torch.cat([keep_h, keep_o])

            score = scores[keep]
            box = gt_boxes[keep]
            labels = labels[keep]

            img_height, img_width = grid_thw[1] // 2, grid_thw[2] // 2
            reshape_image_embeds = qwen_hidden_state.reshape(1, img_height, img_width, -1).permute(0, 3, 1, 2).to(box.dtype)
            bbox_scale = torch.tensor([img_width, img_height, img_width, img_height]).unsqueeze(0).to(device=box.device, dtype=box.dtype)
            roi_features = roi_align(reshape_image_embeds, [box * bbox_scale], output_size=(7, 7), aligned=True, sampling_ratio=2).mean(dim=(2,3))

            results.append({"scores": score, "logits": scores, "boxes": box, "hidden_states": roi_features, "labels": labels})
            global_features.append(reshape_image_embeds[0])
        return results, global_features

    @torch.no_grad()
    def get_detection_results(self, pixel_values, qwen_pixel_values, image_grid_thw):
        outputs = self.grounding_module(pixel_values)
        qwen_visual_hidden_states = self.qwen_visual(hidden_states=qwen_pixel_values.to(torch.bfloat16), grid_thw=image_grid_thw).to(outputs['last_hidden_state'].dtype)
        split_grid_thw_merged = [merged_thw.prod() // 4
                for merged_thw in image_grid_thw]
        qwen_visual_hidden_states = torch.split(qwen_visual_hidden_states, split_grid_thw_merged, dim=0)

        results = []
        global_features = []
        for hidden_states, pred_logits, pred_boxes, qwen_hidden_state, grid_thw in zip(outputs['last_hidden_state'], outputs['pred_logits'], outputs['pred_boxes'], qwen_visual_hidden_states, image_grid_thw):
            probs = F.softmax(pred_logits, -1)
            scores, labels = probs[..., :-1].max(-1)

            pred_boxes = center_to_corners_format(pred_boxes)

            keep = batched_nms(pred_boxes.float(), scores.float(), labels, 0.5)

            hidden_states = hidden_states[keep].view(-1, hidden_states.shape[-1])
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
            select_probs = probs[keep]
            select_logit = pred_logits[keep]
            select_hidden_state = hidden_states[keep]
            labels = labels[keep]

            img_height, img_width = grid_thw[1] // 2, grid_thw[2] // 2
            reshape_image_embeds = qwen_hidden_state.reshape(1, img_height, img_width, -1).permute(0, 3, 1, 2)
            bbox_scale = torch.tensor([img_width, img_height, img_width, img_height]).unsqueeze(0).to(device=box.device, dtype=box.dtype)
            roi_features = roi_align(reshape_image_embeds, [box * bbox_scale], output_size=(7, 7), aligned=True, sampling_ratio=2).mean(dim=(2,3))
            
            results.append({"scores": score, "logits": select_logit, "boxes": box, "hidden_states": roi_features, "labels": labels})
            global_features.append(reshape_image_embeds[0])
        return results, global_features

    @torch.no_grad()
    def construct_hoi_triplet(self, results):
        grouped_results = []
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

            person_hidden_state = hidden_state[person_mask].unsqueeze(1).repeat(1, num_object - 1, 1)

            select_mask = torch.arange(num_object).unsqueeze(0).to(labels_id.device) != torch.nonzero(person_mask)

            object_hidden_state = hidden_state.unsqueeze(0).repeat(num_person, 1, 1)[select_mask].view(num_person, num_object - 1, -1)
            object_logits = logits.unsqueeze(0).repeat(num_person, 1, 1)[select_mask]
            object_score = scores.unsqueeze(0).repeat(num_person, 1)[select_mask]

            triplet_boxes = torch.cat((person_boxes.unsqueeze(1).repeat(1, num_object - 1, 1), object_boxes.unsqueeze(0).repeat(num_person, 1, 1)[select_mask].view(num_person, num_object - 1, -1)), dim=-1).view(num_person * (num_object-1), 8)  # num, 1, 4 - 1, num, 4 -> num_person, num_object, 8

            grouped_results.append({"person_hidden_state": person_hidden_state, "object_hidden_state": object_hidden_state, "triplet_boxes": triplet_boxes, "object_logits": object_logits,})
        return grouped_results

    @torch.no_grad()
    def inference(self, logits, prior, bh, bo, objects, attn_maps, boxes, targets):
        results = []
        logits = logits.split([p.shape[1] for p in prior], dim=0)

        for batch_id, (lg, pr, h, o, obj, bx, target) in enumerate(zip(logits, prior, bh, bo, objects, boxes, targets)):
            size = target['orig_size']

            scale_fct = torch.tensor([size[1], size[0], size[1], size[0]]).unsqueeze(0).to(lg.device)
            
            sub_boxes = bx[h] * scale_fct
            obj_boxes = bx[o] * scale_fct
 
            sl = torch.full_like(obj, 0)
            l = torch.cat((sl, obj))
            b = torch.cat((sub_boxes, obj_boxes))

            results.append({"label": l.to('cpu'), "boxes": b.to('cpu')})
            ids = torch.arange(b.shape[0])
            results[-1].update({'hoi_scores': (lg.sigmoid() * pr.prod(0)).to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results

    def forward(self, pixel_values, qwen_pixel_values, image_grid_thw, targets):
        if self.training:
            # results, detr_output_features = self.get_detection_results_gt(pixel_values, qwen_pixel_values, image_grid_thw, targets)
            results, detr_output_features = self.get_detection_results(pixel_values, qwen_pixel_values, image_grid_thw)
        else:
            results, detr_output_features = self.get_detection_results(pixel_values, qwen_pixel_values, image_grid_thw)

        logits, interaction_logits, prior, bh, bo, objects = self.interaction_proj(results, detr_output_features)
        boxes = [r['boxes'] for r in results]

        if self.training:

            batch_match_labels = []
            batch_interaction_labels = []
            batch_interaction_scores = []
            for batch_id, (bx, h, o, target) in enumerate(zip(boxes, bh, bo, targets)):
                sub_boxes = target['sub_boxes']
                obj_boxes = target['obj_boxes']
                verb_logits = target['verb_labels']

                n = h.shape[0]
                match_labels = torch.zeros(n, 117, device=h.device)
                interaction_labels = torch.zeros(n, sub_boxes.shape[0], device=h.device)

                x, y = torch.nonzero(torch.min(box_iou(bx[h], sub_boxes), box_iou(bx[o], obj_boxes)) >= 0.5).unbind(1)
                interaction_labels[x, y] = 1
                for x_id, y_id in zip(x, y):
                    match_labels[x_id][verb_logits[y_id].bool()] = 1
                batch_match_labels.append(match_labels)
                batch_interaction_labels.append((interaction_labels.sum(1, keepdim=True) > 0).to(match_labels.dtype))
            batch_match_labels = torch.cat(batch_match_labels)
            batch_interaction_labels = torch.cat(batch_interaction_labels)

            prior = torch.cat(prior, dim=1).prod(0)
            x, y = torch.nonzero(prior).unbind(1)

            logits = logits[x, y]
            prior = prior[x, y]
            batch_match_labels = batch_match_labels[x, y]

            n_p = len(torch.nonzero(batch_match_labels))
            n_interaction = len(torch.nonzero(batch_interaction_labels))
            if dist.is_initialized():
                world_size = dist.get_world_size()
                n_p = torch.as_tensor([n_p], device='cuda')
                n_interaction = torch.as_tensor([n_interaction], device='cuda')
                dist.barrier()
                dist.all_reduce(n_p)
                dist.all_reduce(n_interaction)
                n_p = (n_p / world_size).item()
                n_interaction = (n_interaction / world_size).item()

            loss = binary_focal_loss_with_logits(torch.log(prior / (1 + torch.exp(-logits) - prior) + 1e-8), batch_match_labels, reduction='sum')
            interaction_loss = binary_focal_loss_with_logits(interaction_logits, batch_interaction_labels, reduction='sum')

            return loss / n_p, interaction_loss / n_interaction
        else:
            return logits, prior, bh, bo, objects, boxes, boxes


def binary_focal_loss_with_logits(
    x, y,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: float = 1e-6
):
    loss = (1 - y - alpha).abs() * ((y-torch.sigmoid(x)).abs() + eps) ** gamma * \
        torch.nn.functional.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))
