#!/usr/bin/env bash
export PYTHONPATH=./:$PYTHONPATH

set -x
EXP_DIR=checkpoints/interaction/SAP
QWEN_PATH=./
DETECTOR_PATH=./
DATA_ROOT_PATH=./

python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        --master_port 15961 \
        SAP/main.py \
        --output_dir ${EXP_DIR} \
        --batch_size 8 \
        --dataset_file hico \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --num_queries 100 \
        --dec_layers 6 \
        --epochs 20 \
        --lr_drop 10 \
        --use_nms_filter \
        --qwen_path ${QWEN_PATH} \
        --detector_path ${DETECTOR_PATH} \
        --zero_shot_type unseen_verb \
        --root_path ${DATA_ROOT_PATH}
        2>&1 | tee baseline_SAP.log
