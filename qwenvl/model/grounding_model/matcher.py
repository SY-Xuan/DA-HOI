# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from qwenvl.model.utils import box_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size * N, num_classes] with the classification logits
                 "pred_object_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_person_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_batch_id": batch_size * N

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "object_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "person_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "batch_id": batch_size * N

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        N = outputs["pred_logits"].shape[0]

        # We flatten to compute the cost matrices in a batch
        out_label = outputs["pred_logits"]  # [N, num_classes]
        out_object_bbox = outputs["pred_object_boxes"]  # [N, 4]
        out_person_bbox = outputs["pred_person_boxes"]  # [N, 4]
        out_batch_id = outputs["pred_batch_id"]  # [N]

        # Also concat the target labels and boxes
        tgt_ids = targets['labels']
        tgt_object_bbox = targets['object_boxes']
        tgt_person_bbox = targets['person_boxes']
        tgt_batch_id = targets['batch_id']

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # We only match with same label use otherwise
        cost_class = torch.cdist(out_label.float(), tgt_ids.float(), p=0) * 100 + (out_batch_id[:, None] != tgt_batch_id[None, :]).float() * 1000000

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_object_bbox.float(), xyxy2xywh(tgt_object_bbox).float(), p=1) + torch.cdist(out_person_bbox.float(), xyxy2xywh(tgt_person_bbox).float(), p=1)

        # Compute the giou cost betwen boxes
        cost_giou = - generalized_box_iou(xywh2xyxy(out_object_bbox).float(), tgt_object_bbox.float()) - generalized_box_iou(xywh2xyxy(out_person_bbox).float(), tgt_person_bbox.float())

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.cpu()

        row_indices, col_indices = linear_sum_assignment(C)
        return torch.as_tensor(row_indices, dtype=torch.int64).to(out_object_bbox.device), torch.as_tensor(col_indices, dtype=torch.int64).to(out_object_bbox.device)

class HungarianDETRMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size * N, num_classes] with the classification logits
                 "pred_object_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_person_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_batch_id": batch_size * N

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "object_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "person_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "batch_id": batch_size * N

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        B, num_queries, num_cls = outputs["pred_object"].shape

        # We flatten to compute the cost matrices in a batch
        out_embed = outputs["pred_object"].flatten(0, 1).softmax(-1)  # [num_queries, C]
        out_verb_prob = outputs["pred_verb"].flatten(0, 1).sigmoid()
        out_object_bbox = outputs["pred_object_boxes"].flatten(0, 1)  # [num_classes, 4]
        out_person_bbox = outputs["pred_person_boxes"].flatten(0, 1) # [num_classes, 4]

        # Also concat the target labels and boxes
        tgt_labels = targets["gt_objects"].flatten(0, 1)
        tgt_verbs = targets["gt_verbs"].flatten(0, 1)

        tgt_object_bbox = targets['object_boxes'].flatten(0, 1)
        tgt_person_bbox = targets['person_boxes'].flatten(0, 1)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # We only match with same label use otherwise
        cost_class = - out_embed[:, tgt_labels] # B * num_query x B * num_target

        tgt_verb_labels_permute = tgt_verbs.permute(1, 0)
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                            ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

         # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_object_bbox.float(), xyxy2xywh(tgt_object_bbox).float(), p=1) + torch.cdist(out_person_bbox.float(), xyxy2xywh(tgt_person_bbox).float(), p=1)

        # Compute the giou cost betwen boxes
        cost_giou = - generalized_box_iou(xywh2xyxy(out_object_bbox).float(), tgt_object_bbox.float()) - generalized_box_iou(xywh2xyxy(out_person_bbox).float(), tgt_person_bbox.float())
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + cost_verb_class
        C = C.view(B, num_queries, -1).cpu()

        sizes = [v.shape[0] for v in targets['object_boxes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianGDINOMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size * N, num_classes] with the classification logits
                 "pred_object_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_person_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_batch_id": batch_size * N

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "object_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "person_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "batch_id": batch_size * N

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        B, num_queries, num_cls = outputs["pred_cls"].shape
        indices = []

        for i in range(B):
            out_embed = outputs["pred_cls"][i].sigmoid()  # [num_queries, C]
            num_text_tokens = out_embed.size(-1)
            out_bbox = outputs["pred_boxes"][i]  # [num_classes, 4]

            num_target = targets['object_boxes'][i].shape[0] * 2
            tgt_label = torch.eye(num_text_tokens).to(targets['object_boxes'].device, dtype=targets['object_boxes'].dtype)
            tgt_label = tgt_label[:num_target]
            tgt_label = tgt_label / tgt_label.sum(dim=-1, keepdim=True)
            tgt_mask = targets['text_mask'][i]

            tgt_object_bbox = targets['object_boxes'][i] # num_target, 4
            tgt_person_bbox = targets['person_boxes'][i] # num_target, 4

            # we need to merge them and resize
            tgt_bbox = torch.cat((tgt_person_bbox, tgt_object_bbox), dim=1).view(num_target, 4)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # We only match with same label use otherwise
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_embed ** gamma) * (-(1 - out_embed + 1e-8).log()) # num_query x 256
            pos_cost_class = alpha * ((1 - out_embed) ** gamma) * (-(out_embed + 1e-8).log())
            cost_class = (pos_cost_class - neg_cost_class) @ tgt_label.t()

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox.float(), xyxy2xywh(tgt_bbox).float(), p=1)

            # Compute the giou cost betwen boxes
            cost_giou = - generalized_box_iou(xywh2xyxy(out_bbox).float(), tgt_bbox.float())
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # C = C.view(B, num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C.cpu()))

        # sizes = [v.shape[0] for v in targets['object_boxes']]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianRegionMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size * N, num_classes] with the classification logits
                 "pred_object_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_person_boxes": Tensor of dim [batch_size * N, 4] with the predicted box coordinates
                 "pred_batch_id": batch_size * N

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "object_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "person_boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "batch_id": batch_size * N
                 "object": num_target, obj_id
                 "label_maps"

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        num_queries, num_cls = outputs["object_logits"].shape
        indices = []

        out_embed = outputs["object_logits"].sigmoid()  # [num_queries, C]
        num_text_tokens = out_embed.size(-1)
        out_bbox = outputs["pred_boxes"]  # [num_classes, 4]

        tgt_object_bbox = targets['object_boxes'] # num_target, 4
        tgt_person_bbox = targets['person_boxes'] # num_target, 4

        tgt_object_id = targets['objects'] # num target
        label_maps = targets['label_maps']

        tgt_label = label_maps[tgt_object_id].to(out_bbox.dtype) # num target, 256

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # We only match with same label use otherwise
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_embed ** gamma) * (-(1 - out_embed + 1e-8).log()) # num_query x 256
        pos_cost_class = alpha * ((1 - out_embed) ** gamma) * (-(out_embed + 1e-8).log())
        cost_class = (pos_cost_class - neg_cost_class) @ tgt_label.t()

        # Compute the L1 cost between boxes
        cost_bbox = (torch.cdist(out_bbox[:, :4].float(), tgt_person_bbox.float(), p=1) + torch.cdist(out_bbox[:, 4:].float(), tgt_object_bbox.float(), p=1)) * 0.5

        # Compute the giou cost betwen boxes
        cost_giou = - (generalized_box_iou(out_bbox[:, :4].float(), tgt_person_bbox.float()) + generalized_box_iou(out_bbox[:, 4:].float(), tgt_object_bbox.float())) * 0.5
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # C = C.view(B, num_queries, -1).cpu()
        row_indices, col_indices = linear_sum_assignment(C.cpu())
        return torch.as_tensor(row_indices, dtype=torch.int64).to(cost_giou.device), torch.as_tensor(col_indices, dtype=torch.int64).to(cost_giou.device)
