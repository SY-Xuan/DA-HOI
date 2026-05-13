# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from qwenvl.model.qwen_vl_da_hoi import Qwen2_5_VLRegionForConditionalGeneration
from qwenvl.data.data_da_hoi import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from qwenvl.model.adapter import adapter

local_rank = None


HOI_START_TOKEN = "<hoi_start>"
HOI_END_TOKEN = "<hoi_end>"
HOI_EMBED_TOKEN0 = "<embed0>"
HOI_EMBED_TOKEN1 = "<embed1>"
HOI_EMBED_TOKEN2 = "<embed2>"
HOI_EMBED_TOKEN3 = "<embed3>"
HOI_EMBED_TOKEN4 = "<embed4>"
HOI_EMBED_TOKEN5 = "<embed5>"
HOI_EMBED_TOKEN6 = "<embed6>"
HOI_EMBED_TOKEN7 = "<embed7>"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["visual"]
        
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # names = name.split(".")
            lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        if model_args.adapter_vision_enable:
            for n, p in model.visual.named_parameters():
                if "lora" in n or "adapter" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for n, p in model.visual.named_parameters():
                p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        if model_args.adapter_enable:
            for n, p in model.model.named_parameters():
                if "lora" in n or "adapter" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for n, p in model.model.named_parameters():
                p.requires_grad = True
            model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    if model_args.tune_grounding_module:
        for n, p in model.grounding_module.named_parameters():
            p.requires_grad = False
        for n, p in model.interaction_head.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.grounding_module.named_parameters():
            p.requires_grad = False
        for n, p in model.interaction_head.named_parameters():
            p.requires_grad = False

    for n, p in model.sap_projector.named_parameters():
        p.requires_grad = True
    for n, p in model.sap_verb_projector.named_parameters():
        p.requires_grad = True
    for n, p in model.hoi_embedding_token.named_parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad = True


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    with adapter(enabled=training_args.adapter_enable, vision_enabled=training_args.adapter_vision_enable, hidden_dim=8, mlp=True, non_linear=False):
        model = Qwen2_5_VLRegionForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    model_args.adapter_enable = training_args.adapter_enable
    model_args.adapter_vision_enable = training_args.adapter_vision_enable

    model.config.adapter_enable = training_args.adapter_enable
    model.config.adapter_vision_enable = training_args.adapter_vision_enable

    model.config.hoi_token_id = 151648
    model.config.hoi_special_token_id = 151649
    model.config.hoi_features_token_id = 151647

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model.config.grounding_module = model_args.grounding_module
    model.config.interaction_head_path = model_args.interaction_head_path
    model.initialize_grounding_modules()

    set_model(model_args, model)

    trainable_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params.append(n)
    rank0_print(trainable_params)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if model_args.adapter_vision_enable or model_args.adapter_enable:
        training_args.label_names = ["labels"]
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
