import os
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_utils import seed_worker
from qwenvl.model.adapter import AdapterLayer


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def log(self, logs, start_time=None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        model = unwrap_model(self.model)
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if hasattr(model, "log_losses"):
            log_losses = model.log_losses
            for k, v in log_losses.items():
                logs[k] = v
        output = {**logs, **{"step": self.state.global_step}}

        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
        vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
        grounding_parameters = [
            name for name, _ in opt_model.named_parameters() if "grounding_backbone" in name or "grounding_images_encoder" in name or "interaction_proj" in name or "hoi_embedding_token" in name or "logit_scale" in name
            ]

        if self.args.mm_projector_lr is None or self.args.mm_projector_lr != 0:
            mm_projector_lr = self.args.learning_rate
        else:
            mm_projector_lr = self.args.mm_projector_lr
        if self.args.vision_tower_lr is None or self.args.vision_tower_lr != 0:
            vision_tower_lr = self.args.learning_rate
        else:
            vision_tower_lr = self.args.vision_tower_lr
        if self.args.grounding_module_lr is None or self.args.grounding_module_lr != 0:
            grounding_module_lr = self.args.learning_rate
        else:
            grounding_module_lr = self.args.grounding_module_lr
                
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n in decay_parameters
                        and n not in projector_parameters
                        and n not in vision_tower_parameters
                        and n not in grounding_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n in decay_parameters
                        and n not in projector_parameters
                        and n in vision_tower_parameters
                        and n not in grounding_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": self.args.weight_decay,
                "lr": vision_tower_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n in decay_parameters
                        and n in projector_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": self.args.weight_decay,
                "lr": mm_projector_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n in decay_parameters
                        and n in grounding_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": self.args.weight_decay,
                "lr": grounding_module_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n not in decay_parameters
                        and n not in projector_parameters
                        and n not in vision_tower_parameters
                        and n not in grounding_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n not in decay_parameters
                        and n not in projector_parameters
                        and n in vision_tower_parameters
                        and n not in grounding_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": 0.0,
                "lr": vision_tower_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n not in decay_parameters
                        and n in projector_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": 0.0,
                "lr": mm_projector_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n not in decay_parameters
                        and n in grounding_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": 0.0,
                "lr": grounding_module_lr,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


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


# Apply monkey patches
Trainer.create_optimizer = create_optimizer
Trainer.log = log

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen2_5_VisionTransformerPretrainedModel._init_weights = _init_weights
