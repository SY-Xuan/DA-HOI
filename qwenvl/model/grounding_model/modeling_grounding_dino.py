import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from transformers import GroundingDinoPreTrainedModel, GroundingDinoConfig, AutoModel
from transformers.models.grounding_dino.modeling_grounding_dino import MultiScaleDeformableAttention, GroundingDinoConvEncoder, build_position_encoding, GroundingDinoConvModel, GroundingDinoMLPPredictionHead, get_sine_pos_embed
from transformers.pytorch_utils import meshgrid
from transformers.file_utils import ModelOutput
from transformers.activations import ACT2FN


class GroundingDinoContrastiveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_text_len = config.max_text_len

    def forward(
        self,
        vision_hidden_state: torch.FloatTensor,
        text_hidden_state: torch.FloatTensor,
        text_token_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        output = vision_hidden_state @ text_hidden_state.transpose(-1, -2)
        output = output.masked_fill(~text_token_mask[:, None, :], torch.finfo(output.dtype).min)
        # NOTE
        # output = output.masked_fill(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        # NOTE
        # new_output = torch.full((*output.shape[:-1], self.max_text_len), float("-inf"), device=output.device)
        new_output = torch.full((*output.shape[:-1], self.max_text_len), torch.finfo(output.dtype).min, device=output.device, dtype=output.dtype)
        new_output[..., : output.shape[-1]] = output

        return new_output


class GroundingDinoMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: GroundingDinoConfig, num_heads: int, n_points: int):
        super().__init__()

        self.attn = MultiScaleDeformableAttention()

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in GroundingDinoMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        if self.sampling_offsets.bias.dtype != hidden_states.dtype:
            self.sampling_offsets.bias.data = self.sampling_offsets.bias.data.to(hidden_states.dtype)

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        if value.dtype != torch.float32:
            value = value.to(torch.float32)
            attention_weights = attention_weights.to(torch.float32)
        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = output.to(self.output_proj.weight.dtype)
        output = self.output_proj(output)

        return output, attention_weights


class GroundingDinoTextEnhancerLayer(nn.Module):
    """Vanilla Transformer with text embeddings as input"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = GroundingDinoMultiheadAttention(
            config, num_attention_heads=config.encoder_attention_heads // 2
        )

        # Implementation of Feedforward model
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim // 2)
        self.fc2 = nn.Linear(config.encoder_ffn_dim // 2, config.d_model)

        self.layer_norm_before = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_after = nn.LayerNorm(config.d_model, config.layer_norm_eps)

        self.activation = ACT2FN[config.activation_function]
        self.num_heads = config.encoder_attention_heads // 2
        self.dropout = config.text_enhancer_dropout

    def with_pos_embed(self, hidden_state: Tensor, position_embeddings: Optional[Tensor]):
        return hidden_state if position_embeddings is None else hidden_state + position_embeddings

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_masks: Optional[torch.BoolTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Text self-attention to enhance projection of text features generated by
        the text encoder (AutoModel based on text_config) within GroundingDinoEncoderLayer

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Text features generated by the text encoder.
            attention_masks (`torch.BoolTensor`, *optional*):
                Attention mask for text self-attention. False for real tokens and True for padding tokens.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings to be added to the hidden states.

        Returns:
            `tuple(torch.FloatTensor)` comprising two elements:
            - **hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) --
                Output of the text self-attention layer.
            - **attention_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length,
              sequence_length)`) --
                Attention weights of the text self-attention layer.
        """

        # repeat attn mask
        if attention_masks.dim() == 3 and attention_masks.shape[0] == hidden_states.shape[0]:
            # batch_size, num_queries, num_keys
            attention_masks = attention_masks[:, None, :, :]
            attention_masks = attention_masks.repeat(1, self.num_heads, 1, 1)

            dtype = hidden_states.dtype
            attention_masks = attention_masks.to(dtype=dtype)  # fp16 compatibility
            attention_masks = (1.0 - attention_masks) * torch.finfo(dtype).min

        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        attention_output, attention_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=hidden_states,
            attention_mask=attention_masks,
            output_attentions=True,
        )
        attention_output = nn.functional.dropout(attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + attention_output
        hidden_states = self.layer_norm_before(hidden_states)

        residual = hidden_states
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.layer_norm_after(hidden_states)

        return hidden_states, attention_weights


class GroundingDinoBiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        vision_dim = text_dim = config.d_model
        embed_dim = config.encoder_ffn_dim // 2
        num_heads = config.encoder_attention_heads // 2
        dropout = config.fusion_dropout

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.values_vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.values_text_proj = nn.Linear(self.text_dim, self.embed_dim)

        self.out_vision_proj = nn.Linear(self.embed_dim, self.vision_dim)
        self.out_text_proj = nn.Linear(self.embed_dim, self.text_dim)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image-to-text and text-to-image cross-attention

        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            vision_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            text_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an attention
            output and weights:
            - **vision_attn_output** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_din)`)
              --
                Output of the image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_attn_output** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`) --
                Output of the text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        batch_size, tgt_len, _ = vision_features.size()

        vision_query_states = self.vision_proj(vision_features) * self.scale
        vision_query_states = self._reshape(vision_query_states, tgt_len, batch_size)

        text_key_states = self.text_proj(text_features)
        text_key_states = self._reshape(text_key_states, -1, batch_size)

        vision_value_states = self.values_vision_proj(vision_features)
        vision_value_states = self._reshape(vision_value_states, -1, batch_size)

        text_value_states = self.values_text_proj(text_features)
        text_value_states = self._reshape(text_value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        vision_query_states = vision_query_states.view(*proj_shape)
        text_key_states = text_key_states.view(*proj_shape)
        vision_value_states = vision_value_states.view(*proj_shape)
        text_value_states = text_value_states.view(*proj_shape)

        src_len = text_key_states.size(1)
        attn_weights = torch.bmm(vision_query_states, text_key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        attn_weights = attn_weights - attn_weights.max()
        # Do not increase -50000/50000, data type half has quite limited range
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_transposed = attn_weights.transpose(1, 2)
        text_attn_weights = attn_weights_transposed - torch.max(attn_weights_transposed, dim=-1, keepdim=True)[0]

        # Do not increase -50000/50000, data type half has quite limited range
        text_attn_weights = torch.clamp(text_attn_weights, min=-50000, max=50000)

        # mask vision for language
        if vision_attention_mask is not None:
            vision_attention_mask = (
                vision_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            # NOTE
            # text_attn_weights.masked_fill_(vision_attention_mask, float("-inf"))
            text_attn_weights.masked_fill_(vision_attention_mask, torch.finfo(text_attn_weights.dtype).min)

        text_attn_weights = text_attn_weights.softmax(dim=-1)

        # mask language for vision
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            # NOTE
            # attn_weights.masked_fill_(text_attention_mask, float("-inf"))
            attn_weights.masked_fill_(text_attention_mask, torch.finfo(attn_weights.dtype).min)
        vision_attn_weights = attn_weights.softmax(dim=-1)

        vision_attn_probs = F.dropout(vision_attn_weights, p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_attn_weights, p=self.dropout, training=self.training)

        vision_attn_output = torch.bmm(vision_attn_probs, text_value_states)
        text_attn_output = torch.bmm(text_attn_probs, vision_value_states)

        if vision_attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`vision_attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {vision_attn_output.size()}"
            )

        if text_attn_output.size() != (batch_size * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`text_attn_output` should be of size {(batch_size, self.num_heads, src_len, self.head_dim)}, but is {text_attn_output.size()}"
            )

        vision_attn_output = vision_attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        vision_attn_output = vision_attn_output.transpose(1, 2)
        vision_attn_output = vision_attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        text_attn_output = text_attn_output.view(batch_size, self.num_heads, src_len, self.head_dim)
        text_attn_output = text_attn_output.transpose(1, 2)
        text_attn_output = text_attn_output.reshape(batch_size, src_len, self.embed_dim)

        vision_attn_output = self.out_vision_proj(vision_attn_output)
        text_attn_output = self.out_text_proj(text_attn_output)

        return (vision_attn_output, vision_attn_weights), (text_attn_output, text_attn_weights)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->GroundingDino
class GroundingDinoDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GroundingDinoFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        drop_path = config.fusion_droppath

        # pre layer norm
        self.layer_norm_vision = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_text = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.attn = GroundingDinoBiMultiHeadAttention(config)

        # add layer scale for training stability
        self.drop_path = GroundingDinoDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        init_values = 1e-4
        self.vision_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)
        self.text_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        attention_mask_vision: Optional[torch.BoolTensor] = None,
        attention_mask_text: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image and text features fusion

        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            attention_mask_vision (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            attention_mask_text (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an enhanced
            feature and attention output and weights:
            - **vision_features** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, vision_dim)`) --
                Updated vision features with attention output from image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_features** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, text_dim)`) --
                Updated text features with attention output from text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        vision_features = self.layer_norm_vision(vision_features)
        text_features = self.layer_norm_text(text_features)
        (delta_v, vision_attn), (delta_t, text_attn) = self.attn(
            vision_features,
            text_features,
            vision_attention_mask=attention_mask_vision,
            text_attention_mask=attention_mask_text,
        )
        vision_features = vision_features + self.drop_path(self.vision_param * delta_v)
        text_features = text_features + self.drop_path(self.text_param * delta_t)

        return (vision_features, vision_attn), (text_features, text_attn)


class GroundingDinoDeformableLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = GroundingDinoMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


class GroundingDinoEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.d_model = config.d_model

        self.text_enhancer_layer = GroundingDinoTextEnhancerLayer(config)
        self.fusion_layer = GroundingDinoFusionLayer(config)
        self.deformable_layer = GroundingDinoDeformableLayer(config)

    def get_text_position_embeddings(
        self,
        text_features: Tensor,
        text_position_embedding: Optional[torch.Tensor],
        text_position_ids: Optional[torch.Tensor],
    ) -> Tensor:
        batch_size, seq_length, _ = text_features.shape
        if text_position_embedding is None and text_position_ids is None:
            text_position_embedding = torch.arange(seq_length, device=text_features.device)
            text_position_embedding = text_position_embedding.float()
            text_position_embedding = text_position_embedding.unsqueeze(0).unsqueeze(-1)
            text_position_embedding = text_position_embedding.repeat(batch_size, 1, 1)
            text_position_embedding = get_sine_pos_embed(
                text_position_embedding, num_pos_feats=self.d_model, exchange_xy=False
            )
        if text_position_ids is not None:
            text_position_embedding = get_sine_pos_embed(
                text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False
            )

        return text_position_embedding

    def forward(
        self,
        vision_features: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        level_start_index: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
    ):
        text_position_embedding = self.get_text_position_embeddings(
            text_features, text_position_embedding, text_position_ids
        ).to(vision_features.dtype)

        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            vision_features=vision_features,
            text_features=text_features,
            attention_mask_vision=key_padding_mask,
            attention_mask_text=text_attention_mask,
        )

        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            hidden_states=text_features,
            attention_masks=~text_self_attention_masks,  # note we use ~ for mask here
            position_embeddings=(text_position_embedding if text_position_embedding is not None else None),
        )

        (vision_features, vision_deformable_attn) = self.deformable_layer(
            hidden_states=vision_features,
            attention_mask=~key_padding_mask,
            position_embeddings=vision_position_embedding,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

        return (
            (vision_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn),
        )


class GroundingDinoMultiheadAttention(nn.Module):
    """Equivalent implementation of nn.MultiheadAttention with `batch_first=True`."""

    def __init__(self, config, num_attention_heads=None):
        super().__init__()
        if config.hidden_size % num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(queries))
        key_layer = self.transpose_for_scores(self.key(keys))
        value_layer = self.transpose_for_scores(self.value(values))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GroundingDinoModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class GroundingDinoDecoderLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = GroundingDinoMultiheadAttention(config, num_attention_heads=config.decoder_attention_heads)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention text
        self.encoder_attn_text = GroundingDinoMultiheadAttention(
            config, num_attention_heads=config.decoder_attention_heads
        )
        self.encoder_attn_text_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = GroundingDinoMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        vision_encoder_hidden_states: Optional[torch.Tensor] = None,
        vision_encoder_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_hidden_states: Optional[torch.Tensor] = None,
        text_encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        # Self Attention
        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, self_attn_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=hidden_states,
            attention_mask=self_attn_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention Text
        queries = self.with_pos_embed(hidden_states, position_embeddings)

        hidden_states, text_cross_attn_weights = self.encoder_attn_text(
            queries=queries,
            keys=text_encoder_hidden_states,
            values=text_encoder_hidden_states,
            attention_mask=text_encoder_attention_mask, # TODO fix cross-attention mask here
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_text_layer_norm(hidden_states)

        third_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=vision_encoder_attention_mask,
            encoder_hidden_states=vision_encoder_hidden_states,
            encoder_attention_mask=vision_encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = third_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, text_cross_attn_weights, cross_attn_weights)

        return outputs


@dataclass
class GroundingDinoDecoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    intermediate_object_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoEncoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoEncoder. This class extends BaseModelOutput, due to:
    - vision and text last hidden states
    - vision and text intermediate hidden states

    Args:
        last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the vision encoder.
        last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        text_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads.
    """

    last_hidden_state_vision: torch.FloatTensor = None
    last_hidden_state_text: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoModelOutput(ModelOutput):
    """
    Base class for outputs of the Grounding DINO encoder-decoder model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        encoder_last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        encoder_text_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads. attention softmax, used to compute the weighted average in the
            bi-attention heads.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.num_queries` scoring bounding boxes are picked as
            region proposals in the first stage. Output of bounding box binary classification (i.e. foreground and
            background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    last_hidden_state: torch.FloatTensor = None
    init_reference_points: torch.FloatTensor = None
    object_init_reference_points: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    intermediate_object_reference_points: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state_vision: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None
    encoder_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    object_enc_outputs_coord_logits: Optional[torch.FloatTensor] = None
    encoder_logits: Optional[torch.FloatTensor] = None
    encoder_pred_boxes: Optional[torch.FloatTensor] = None


def generate_masks_with_text_query_masks(text_query_masks):
    """
    Args:
        text_query_masks (bool): [bs, max_num_patches,], valid is 1
    Returns:
        self_attention_mask (bool): [bs, max_num_patches, max_num_patches], padding is 1
        position_ids (long): [bs, max_num_patches]
    """
    batch_size, num_token = text_query_masks.shape
    self_attention_mask = torch.eye(num_token, device=text_query_masks.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), dtype=torch.long, device=text_query_masks.device)

    for batch_idx in range(batch_size):
        num_valid = text_query_masks[batch_idx].sum()  # [num_patches_i,]
        self_attention_mask[batch_idx, :num_valid, :num_valid] = True
        position_ids[batch_idx, :num_valid] = torch.arange(0, num_valid, device=text_query_masks.device)

    return self_attention_mask, position_ids


class GroundingDinoEncoder(GroundingDinoPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`GroundingDinoEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([GroundingDinoEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        level_start_index: Tensor,
        valid_ratios=None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 0 for pixel features that are real (i.e. **not masked**),
                - 1 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            vision_position_embedding (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Flattened text features that are passed to the encoder.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            text_position_embedding (`torch.FloatTensor` of shape `(batch_size, text_seq_len)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            text_self_attention_masks (`torch.BoolTensor` of shape `(batch_size, text_seq_len, text_seq_len)`):
                Masks to avoid performing attention between padding text features. Mask values selected in `[0, 1]`:
                - 1 for text features that are real (i.e. **not masked**),
                - 0 for text features that are padding (i.e. **masked**).
            text_position_ids (`torch.LongTensor` of shape `(batch_size, num_queries)`):
                Position ids for text features.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=vision_features.device)

        encoder_vision_states = () if output_hidden_states else None
        encoder_text_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_attn_fused_text = () if output_attentions else None
        all_attn_fused_vision = () if output_attentions else None
        all_attn_enhanced_text = () if output_attentions else None
        all_attn_deformable = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_vision_states += (vision_features,)
                encoder_text_states += (text_features,)

            (vision_features, text_features), attentions = encoder_layer(
                vision_features=vision_features,
                vision_position_embedding=vision_position_embedding,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                key_padding_mask=vision_attention_mask,
                reference_points=reference_points,
                text_features=text_features,
                text_attention_mask=text_attention_mask,
                text_position_embedding=text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
                text_position_ids=text_position_ids,
            )

            if output_attentions:
                all_attn_fused_vision += (attentions[0],)
                all_attn_fused_text += (attentions[1],)
                all_attn_enhanced_text += (attentions[2],)
                all_attn_deformable += (attentions[3],)

        if output_hidden_states:
            encoder_vision_states += (vision_features,)
            encoder_text_states += (text_features,)

        if output_attentions:
            all_attns = (all_attn_fused_vision, all_attn_fused_text, all_attn_enhanced_text, all_attn_deformable)

        if not return_dict:
            enc_outputs = [vision_features, text_features, encoder_vision_states, encoder_text_states, all_attns]
            return tuple(v for v in enc_outputs if v is not None)
        return GroundingDinoEncoderOutput(
            last_hidden_state_vision=vision_features,
            last_hidden_state_text=text_features,
            vision_hidden_states=encoder_vision_states,
            text_hidden_states=encoder_text_states,
            attentions=all_attns,
        )


class GroundingDinoDecoder(GroundingDinoPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`GroundingDinoDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Grounding DINO:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layers = nn.ModuleList([GroundingDinoDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = GroundingDinoMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement as in two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.object_bbox_embed = None
        self.query_scale = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds,
        vision_encoder_hidden_states,
        vision_encoder_attention_mask=None,
        text_encoder_hidden_states=None,
        text_encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        self_attn_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            vision_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last hidden state from encoder related to vision feature map.
            vision_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
            text_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Last hidden state from encoder related to text features.
            text_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            self_attn_mask (`torch.BoolTensor` of shape `(batch_size, text_seq_len)`):
                Masks to avoid performing self-attention between vision hidden state. Mask values selected in `[0, 1]`:
                - 1 for queries that are real (i.e. **not masked**),
                - 0 for queries that are padding (i.e. **masked**).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attns = () if output_attentions else None
        all_cross_attns_vision = () if (output_attentions and vision_encoder_hidden_states is not None) else None
        all_cross_attns_text = () if (output_attentions and text_encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_object_reference_points = ()

        if text_encoder_attention_mask is not None:
            dtype = text_encoder_hidden_states.dtype

            text_encoder_attention_mask = text_encoder_attention_mask[:, None, None, :]
            text_encoder_attention_mask = text_encoder_attention_mask.repeat(
                1, self.config.decoder_attention_heads, self.config.num_queries, 1
            )
            text_encoder_attention_mask = text_encoder_attention_mask.to(dtype=dtype)
            text_encoder_attention_mask = text_encoder_attention_mask * torch.finfo(dtype).min

        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif num_coordinates == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError("Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
            query_pos = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2).to(text_encoder_hidden_states.dtype)
            query_pos = self.reference_points_head(query_pos)

            # In original implementation they apply layer norm before outputting intermediate hidden states
            # Though that's not through between layers so the layers use as input the output of the previous layer
            # withtout layer norm
            if output_hidden_states:
                all_hidden_states += (self.layer_norm(hidden_states),)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    query_pos,
                    reference_points_input,
                    spatial_shapes,
                    spatial_shapes_list,
                    level_start_index,
                    vision_encoder_hidden_states,
                    vision_encoder_attention_mask,
                    text_encoder_hidden_states,
                    text_encoder_attention_mask,
                    self_attn_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=query_pos,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_list=spatial_shapes_list,
                    level_start_index=level_start_index,
                    vision_encoder_hidden_states=vision_encoder_hidden_states,
                    vision_encoder_attention_mask=vision_encoder_attention_mask,
                    text_encoder_hidden_states=text_encoder_hidden_states,
                    text_encoder_attention_mask=text_encoder_attention_mask,
                    self_attn_mask=self_attn_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                assert self.object_bbox_embed is not None
                tmp = self.bbox_embed[idx](hidden_states)
                object_tmp = self.object_bbox_embed[idx](hidden_states)
                num_coordinates = reference_points.shape[-1]
                if num_coordinates == 4:
                    new_reference_points = tmp + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                    object_new_reference_points = object_tmp + torch.special.logit(reference_points, eps=1e-5)
                    object_new_reference_points = object_new_reference_points.sigmoid()
                elif num_coordinates == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                    object_new_reference_points = object_tmp
                    object_new_reference_points[..., :2] = object_tmp[..., :2] + torch.special.logit(reference_points, eps=1e-5)
                    object_new_reference_points = object_new_reference_points.sigmoid()
                else:
                    raise ValueError(
                        f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
                    )
                reference_points = new_reference_points.detach().to(hidden_states.dtype)
                object_reference_points = object_new_reference_points.detach().to(hidden_states.dtype)

            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)
            intermediate_object_reference_points += (object_reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if text_encoder_hidden_states is not None:
                    all_cross_attns_text += (layer_outputs[2],)

                if vision_encoder_hidden_states is not None:
                    all_cross_attns_vision += (layer_outputs[3],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        intermediate_object_reference_points = torch.stack(intermediate_object_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    intermediate_object_reference_points,
                    all_hidden_states,
                    all_attns,
                ]
                if v is not None
            )
        return GroundingDinoDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            intermediate_object_reference_points=intermediate_object_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class GroundingDinoHOIModel(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = GroundingDinoConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = GroundingDinoConvModel(backbone, position_embeddings)

        # Create input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            self.input_proj_vision = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_vision = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = GroundingDinoEncoder(config)
        self.decoder = GroundingDinoDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
            if (
                config.two_stage_bbox_embed_share
                and config.decoder_bbox_embed_share
                and self.decoder.bbox_embed is not None
            ):
                self.encoder_output_bbox_embed = self.decoder.bbox_embed
                self.encoder_output_object_bbox_embed = self.decoder.object_bbox_embed
            else:
                self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                self.encoder_output_object_bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )

            self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(config)
        else:
            self.reference_points = nn.Embedding(config.num_queries, 4)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def generate_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (`torch.Tensor[batch_size, sequence_length, hidden_size]`): Output of the encoder.
            padding_mask (`torch.Tensor[batch_size, sequence_length]`): Padding mask for `enc_output`.
            spatial_shapes (`torch.Tensor[num_feature_levels, 2]`): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        current_position = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, current_position : (current_position + height * width)]
            mask_flatten_ = mask_flatten_.view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            current_position += height * width

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        # NOTE
        # output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        # output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), torch.finfo(output_proposals.dtype).max)
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, torch.finfo(output_proposals.dtype).max)

        # assign each pixel as an object query
        object_query = enc_output
        # NOTE
        # object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        # object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), 0)
        object_query = object_query.masked_fill(~output_proposals_valid, 0)
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    def forward(
        self,
        pixel_values: Tensor,
        # input_ids: Tensor,
        # token_type_ids: Optional[Tensor] = None,
        # attention_mask: Optional[Tensor] = None,
        text_query=None,
        text_query_masks=None,
        pixel_mask: Optional[Tensor] = None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        >>> model = AutoModel.from_pretrained("IDEA-Research/grounding-dino-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # NOTE: True for nomask, False for mask
        text_token_mask = text_query_masks.bool()  # just to avoid renaming everywhere
        text_self_attention_masks, position_ids = generate_masks_with_text_query_masks(text_query_masks.bool())

        # NOTE: we do padding outside the gdino model
        text_features = text_query

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        vision_features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_maps = []
        masks = []
        for level, (source, mask) in enumerate(vision_features):
            feature_maps.append(self.input_proj_vision[level](source))
            masks.append(mask)

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(feature_maps):
            _len_sources = len(feature_maps)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj_vision[level](vision_features[-1][0])
                else:
                    source = self.input_proj_vision[level](feature_maps[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Create queries
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        spatial_shapes_list = []
        for level, (source, mask, pos_embed) in enumerate(zip(feature_maps, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                vision_features=source_flatten,
                vision_attention_mask=~mask_flatten,
                vision_position_embedding=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                text_features=text_features,
                text_attention_mask=~text_token_mask,
                text_position_embedding=None,
                text_self_attention_masks=~text_self_attention_masks,
                text_position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a GroundingDinoEncoderOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, GroundingDinoEncoderOutput):
            encoder_outputs = GroundingDinoEncoderOutput(
                last_hidden_state_vision=encoder_outputs[0],
                last_hidden_state_text=encoder_outputs[1],
                vision_hidden_states=encoder_outputs[2] if output_hidden_states else None,
                text_hidden_states=encoder_outputs[3] if output_hidden_states else None,
                attentions=encoder_outputs[-1] if output_attentions else None,
            )

        # Fifth, prepare decoder inputs
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.generate_encoder_output_proposals(
                encoder_outputs[0], ~mask_flatten, spatial_shapes
            )

            # hack implementation as in two-stage Deformable DETR
            # apply a detection head to each pixel (A.4 in paper)
            # linear projection for bounding box binary classification (i.e. foreground and background)
            enc_outputs_class = self.encoder_output_class_embed(
                object_query_embedding, encoder_outputs[1], text_token_mask
            )
            # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
            delta_bbox = self.encoder_output_bbox_embed(object_query_embedding)
            object_delta_bbox = self.encoder_output_object_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals
            object_enc_outputs_coord_logits = object_delta_bbox + output_proposals

            # only keep top scoring `config.num_queries` proposals
            topk = self.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            object_topk_coords_logits = torch.gather(
                object_enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            object_topk_coords_logits = object_topk_coords_logits.detach()
            object_init_reference_points = object_topk_coords_logits.sigmoid()
            init_reference_points = reference_points.to(source_flatten.dtype)
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                target = torch.gather(
                    object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                ).detach()
            # Set intermediate topk proposals (coords and class) for loss computation
            encoder_pred_boxes = reference_points
            # encoder_logits = self.encoder_output_class_embed(target, text_features, text_token_mask)
            # TODO: check the implementation of mm gdino
            # encoder_logits = self.encoder_output_class_embed(target, text_features, text_token_mask)
            encoder_logits = torch.gather(
                    enc_outputs_class, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
                ).detach()

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            vision_encoder_hidden_states=encoder_outputs[0],
            vision_encoder_attention_mask=mask_flatten,
            text_encoder_hidden_states=encoder_outputs[1],
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            self_attn_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits, encoder_logits, encoder_pred_boxes,] if value is not None)
            tuple_outputs = (
                (decoder_outputs[0], init_reference_points) + decoder_outputs[1:] + encoder_outputs + enc_outputs
            )

            return tuple_outputs

        return GroundingDinoModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            init_reference_points=init_reference_points,
            object_init_reference_points=object_init_reference_points,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_object_reference_points=decoder_outputs.intermediate_object_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state_vision=encoder_outputs.last_hidden_state_vision,
            encoder_last_hidden_state_text=encoder_outputs.last_hidden_state_text,
            encoder_vision_hidden_states=encoder_outputs.vision_hidden_states,
            encoder_text_hidden_states=encoder_outputs.text_hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            object_enc_outputs_coord_logits=object_enc_outputs_coord_logits,
            encoder_logits=encoder_logits,
            encoder_pred_boxes=encoder_pred_boxes,
        )


class GroundingDinoForObjectDetection(GroundingDinoPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # the bbox_embed in the decoder are all clones though
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"model\.decoder\.bbox_embed\.[0-9]\d*"]

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.model = GroundingDinoHOIModel(config)
        _class_embed = GroundingDinoContrastiveEmbedding(config)

        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
            _object_bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.object_bbox_embed = nn.ModuleList([_object_bbox_embed for _ in range(config.decoder_layers)])

        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        # hack for box-refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.object_bbox_embed = self.object_bbox_embed
        # hack implementation for two-stage
        self.model.decoder.class_embed = self.class_embed

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        # input_ids: torch.LongTensor,
        # token_type_ids: torch.LongTensor = None,
        # attention_mask: torch.LongTensor = None,
        text_query=None,
        text_query_masks=None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
    ):
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, GroundingDinoForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        >>> model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to COCO API
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = processor.image_processor.post_process_object_detection(
        ...     outputs, threshold=0.35, target_sizes=target_sizes
        ... )[0]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 1) for i in box.tolist()]
        ...     print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
        Detected 1 with confidence 0.45 at location [344.8, 23.2, 637.4, 373.8]
        Detected 1 with confidence 0.41 at location [11.9, 51.6, 316.6, 472.9]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_token_mask = text_query_masks.bool()  # just to avoid renaming everywhere
        # First, sent images through Grounding DINO base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values=pixel_values,
            text_query=text_query,
            text_query_masks=text_query_masks,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        enc_text_hidden_state = outputs.encoder_last_hidden_state_text
        hidden_states = outputs.intermediate_hidden_states
        init_reference_points = outputs.init_reference_points
        inter_references_points = outputs.intermediate_reference_points
        object_init_reference_points = outputs.object_init_reference_points
        object_inter_references_points = outputs.intermediate_object_reference_points

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []
        outputs_object_coords = []

        # hidden_states are of shape (batch_size, num_stages, height, width)
        # predict class and bounding box deltas for each stage
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=text_token_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            object_delta_bbox = self.object_bbox_embed[level](hidden_states[:, level])

            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
                outputs_object_coord_logits = object_delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
                object_delta_bbox[..., :2] += reference
                outputs_object_coord_logits = object_delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_object_coord = outputs_object_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_object_coords.append(outputs_object_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_object_coord = torch.stack(outputs_object_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        pred_object_boxes = outputs_object_coord[-1]

        return logits, pred_boxes, pred_object_boxes, outputs.encoder_logits, outputs.encoder_pred_boxes
