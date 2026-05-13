#!/bin/bash
export PYTHONPATH=./qwenvl:$PYTHONPATH
export PYTHONPATH=./SAP:$PYTHONPATH

# Dataset configuration (replace with public dataset names)
datasets="./data/hico/annotations/test_hico_ann.json"
IMAGE_FOLDER="./data/hico/images/test2015"
CONV_MODE="qwen-2"

# Output configuration
output_dir=./checkpoints/da-hoi/

CUDA_VISIBLE_DEVICES=0 python qwenvl/eval/evaluate.py --model-path $output_dir --image-folder $IMAGE_FOLDER --question-file $datasets --answer-file ${output_dir}answer0.pkl --conv-mode $CONV_MODE --num-chunks 4 --chunk-idx 0 --potential-list 2>&1 | tee ${output_dir}answer0.log &
CUDA_VISIBLE_DEVICES=1 python qwenvl/eval/evaluate.py --model-path $output_dir --image-folder $IMAGE_FOLDER --question-file $datasets --answer-file ${output_dir}answer1.pkl --conv-mode $CONV_MODE --num-chunks 4 --chunk-idx 1 --potential-list 2>&1 | tee ${output_dir}answer1.log &
CUDA_VISIBLE_DEVICES=2 python qwenvl/eval/evaluate.py --model-path $output_dir --image-folder $IMAGE_FOLDER --question-file $datasets --answer-file ${output_dir}answer2.pkl --conv-mode $CONV_MODE --num-chunks 4 --chunk-idx 2 --potential-list 2>&1 | tee ${output_dir}answer2.log &
CUDA_VISIBLE_DEVICES=3 python qwenvl/eval/evaluate.py --model-path $output_dir --image-folder $IMAGE_FOLDER --question-file $datasets --answer-file ${output_dir}answer3.pkl --conv-mode $CONV_MODE --num-chunks 4 --chunk-idx 3 --potential-list 2>&1 | tee ${output_dir}answer3.log

# python qwenvl/eval/calculate_hico_map.py --question-file ${datasets} --answer-file ${output_dir} --group 2>&1 | tee ${output_dir}results_map.log