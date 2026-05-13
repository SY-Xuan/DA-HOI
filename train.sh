#!/bin/bash
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm="./Qwen2.5-VL-3B-Instruct"  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-4
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train.py

# Dataset configuration (replace with public dataset names)
datasets="./data/hico/annotations/trainval_hico_ann.json"

# Output configuration
run_name="da-hoi"
output_dir=./checkpoints/${run_name}
grounding_module=""
SAP_path="./checkpoints/interaction/SAP/checkpoint_last.pth"
image_path="./data/hico/images/train2015"

# Training arguments
args="--model_name_or_path "${llm}" \
    --deepspeed ${deepspeed} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --tune_grounding_module False \
    --grounding_module ${grounding_module} \
    --interaction_head_path ${SAP_path} \
    --image_path ${image_path} \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 16 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 401408 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 90000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --grounding_module_lr 5e-4 \
    --weight_decay 0.02 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --adapter_enable True \
    --report_to tensorboard"

# Launch training
TOKENIZERS_PARALLELISM=false python -m torch.distributed.run --nproc_per_node=4 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee logs/${run_name}.log

bash eval.sh
