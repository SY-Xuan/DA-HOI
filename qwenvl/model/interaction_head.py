import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict
from torchvision.ops import box_iou, roi_align
import math
from qwenvl.data.hico_class import *
from qwenvl.model.attention import TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


def compute_spatial_encodings(
    boxes_1, boxes_2, eps: float = 1e-10
):
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        paired bounding boxes (M, 8)
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (M, 36)
    """
    features = []
    for b1, b2 in zip(boxes_1, boxes_2):
        b1 = b1.float()
        b2 = b2.float()

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x, c1_y, c2_x, c2_y,
            # Relative box width and height
            b1_w, b1_h, b2_w, b2_h,
            # Relative box area
            b1_w * b1_h, b2_w * b2_h,
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(torch.cat([f, torch.log(f + eps)], 1))
        return torch.cat(features)


def compute_sinusoidal_pe(pos_tensor: Tensor, temperature: float = 10000.) -> Tensor:
    """
    Compute positional embeddings for points or bounding boxes

    Parameters:
    -----------
    pos_tensor: Tensor
        Coordinates of 2d points (x, y) normalised to (0, 1). The shape is (n_q, bs, 2).
    temperature: float, Default: 10000.
        The temperature parameter in sinusoidal functions.

    Returns:
    --------
    pos: Tensor
        Sinusoidal positional embeddings of shape (n_q, bs, 256).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(256, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 256)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        c, h, w = x.shape
        not_mask = torch.ones_like(x)[:1]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).view(h * w, 1, -1)
        return pos.to(x.dtype)


class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z


class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.human_idx = human_idx

        self.encoder_proj = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 512))
        self.decoder_proj = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 512))

        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
        )
        # self.encoder = TransformerEncoder(hidden_size=512, num_heads=8, num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(1024, 512, repr_size)

        decoder_layer = TransformerDecoderLayer(512, 512, num_heads=8, ffn_interm_dim=2048)
        self.triplet_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=2, return_intermediate=False)
        self.memory_pos = PositionEmbeddingSine(256, 20, normalize=True)
        self.num_classes = 117
        self.object_class_to_target_class = [[] for _ in range(80)]
        for item in HICO_INTERACTIONS:
            self.object_class_to_target_class[objectname2id[item['object']]].append(actionname2id[item['action']])
        self.box_pair_predictor = nn.Linear(512, 117)
        self.interaction_predictor = nn.Linear(512, 1)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def compute_box_pe(self, boxes, embeds):
        bx_c = (boxes[:, :2] + boxes[:, 2:]) / 2
        b_wh = boxes[:, 2:] - boxes[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        return box_pe.to(embeds.dtype), c_pe.to(embeds.dtype)

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def forward(self, qwen_hidden_state, sap_boxes, sap_labels, grid_thw):
        device = qwen_hidden_state.device
        img_height, img_width = grid_thw[1] // 2, grid_thw[2] // 2
        reshape_image_embeds = qwen_hidden_state.reshape(1, img_height, img_width, -1).permute(0, 3, 1, 2)

        memory = reshape_image_embeds[0]

        bbox_scale = torch.tensor([img_width, img_height, img_width, img_height]).unsqueeze(0).to(device=device)
        roi_features = roi_align(reshape_image_embeds.float(), [sap_boxes.float() * bbox_scale], output_size=(7, 7), aligned=True, sampling_ratio=2).mean(dim=(2,3)).to(dtype=qwen_hidden_state.dtype)

        is_human = sap_labels == 0
        n_h = torch.sum(is_human)
        n = sap_labels.shape[0]
        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        if len(x_keep) == 0:
            # Should never happen, just to be safe
            raise ValueError("There are no valid human-object pairs")
        x = x.flatten(); y = y.flatten()

        pairwise_spatial = compute_spatial_encodings(
            [sap_boxes[x],], [sap_boxes[y],]
        )
        pairwise_spatial = self.spatial_head(pairwise_spatial.to(roi_features.dtype))
        pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

        box_pe, c_pe = self.compute_box_pe(sap_boxes, roi_features)
        embeds = self.encoder_proj(roi_features)
        # Compute human-object queries
        ho_q = self.mmf(
            torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
            pairwise_spatial_reshaped[x_keep, y_keep]
        )
        pairwise_tokens = self.triplet_decoder(
            ho_q.unsqueeze(1),
            self.decoder_proj(memory.permute(1, 2, 0).reshape(-1, 2048).unsqueeze(1)),
            q_pos={"center": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1), "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)},
            k_pos=self.memory_pos(memory),
        ).squeeze(1)

        match_logits = self.interaction_predictor(pairwise_tokens)

        return pairwise_tokens, match_logits
