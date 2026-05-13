import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from typing import Dict, List, Callable, Any

import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2MLP, Qwen2_5_VLMLP, Qwen2_5_VLVisionFlashAttention2, Qwen2_5_VLConfig, Qwen2RMSNorm, Qwen2_5_VLDecoderLayer, QWEN2_5_VL_ATTENTION_CLASSES, Qwen2_5_VLVisionBlock, apply_rotary_pos_emb_vision, logger

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.cuda.amp import autocast


class AdapterLayer(nn.Module):
    def __init__(
        self, 
        in_features,
        hidden_dim=8, 
        scale=1,
        dropout=0.1,
        non_linear=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.in_features = in_features
        self.tune_adapter_a = nn.Linear(self.in_features, hidden_dim, bias=True)
        self.tune_adapter_b = nn.Linear(hidden_dim, self.in_features, bias=True)
        self.dropout = nn.Dropout(dropout)

        if non_linear:
            self.activate = nn.GELU()
        else:
            self.activate = nn.Identity()

    def train(self, mode: bool = True):
        self.tune_adapter_a.train(mode)
        self.tune_adapter_b.train(mode)
        self.dropout.train(mode)

    def forward(self, x):
        previous_dtype = x.dtype
        weight_dtype = self.tune_adapter_a.weight.data.dtype
        down_x = self.tune_adapter_a(x.to(weight_dtype))
        down_x = self.activate(down_x)
        up_x = self.tune_adapter_b(self.dropout(down_x))
        result = up_x.to(previous_dtype) + x
        return result

@dataclass
class AdapterConfig:
    hidden_dim: int = 16
    scale: float = 1.0
    dropout: float = 0.1
    adapter_attn: bool = True
    adapter_mlp: bool = False
    non_linear: bool = False


class AdapterQwen2_5_VLDecoderLayer(nn.Module):
    adapter_config = None
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
        #     logger.warning_once(
        #         f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
        #         "unexpected results may be encountered."
        #     )
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                self.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )
        if self.adapter_config.adapter_mlp:
            self.adapter_mlp = AdapterLayer(
                self.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.adapter_config.adapter_attn:
            hidden_states = self.adapter_attn(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.adapter_config.adapter_mlp:
            hidden_states = self.adapter_mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class NewQwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_output = []
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]

            q_window = q[:, start:end, :]
            k_window = k[:, start:end, :]
            v_window = v[:, start:end, :]

            attn_weights = torch.matmul(q_window, k_window.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output_window = torch.matmul(attn_weights, v_window)
            attn_output.append(attn_output_window)

        attn_output = torch.cat(attn_output, dim=1).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)

        attn_output = self.proj(attn_output)

        return attn_output


class NewQwen2_5_VLVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_output = []
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]

            q_window = q[:, start:end, :]
            k_window = k[:, start:end, :]
            v_window = v[:, start:end, :]

            attn_output_window = F.scaled_dot_product_attention(q_window, k_window, v_window, dropout_p=0.0)
            attn_output.append(attn_output_window)

        attn_output = torch.cat(attn_output, dim=1).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)

        return attn_output



QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": NewQwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
    "sdpa": NewQwen2_5_VLVisionSdpaAttention,
}


class AdapterQwen2_5_VLVisionBlock(nn.Module):
    adapter_config = None
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)
        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                config.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )
        if self.adapter_config.adapter_mlp:
            self.adapter_mlp = AdapterLayer(
                config.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.adapter_config.adapter_attn:
            hidden_states = self.adapter_attn(hidden_states)

        hidden_states = residual + self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        if self.adapter_config.adapter_mlp:
            hidden_states = self.adapter_mlp(hidden_states)

        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@contextmanager
def adapter(hidden_dim=16, scale=1, dropout=0.05, enabled:bool = True, vision_enabled:bool = False, non_linear=False, attn=True, mlp=False):

    AdapterQwen2_5_VLDecoderLayer.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, non_linear=non_linear, adapter_attn=attn and enabled, adapter_mlp=mlp and enabled)
    original_layer = Qwen2_5_VLDecoderLayer
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer = AdapterQwen2_5_VLDecoderLayer

    AdapterQwen2_5_VLVisionBlock.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, non_linear=non_linear, adapter_attn=attn and vision_enabled, adapter_mlp=mlp and vision_enabled)
    original_vision_layer = Qwen2_5_VLVisionBlock
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock = AdapterQwen2_5_VLVisionBlock
    yield
    # when exiting context manager - restore link to original causal self-attention class
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer = original_layer
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock = original_vision_layer
