#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python vslice/vslice_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_summe.txt
# CUDA_VISIBLE_DEVICES=0 python vslice/vslice_dpo.py --dataset tvsum --split_file ./dataset/tvsum_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=10 --beta=0.1 --learning_rate=3e-4 > log_tvsum.txt

# CUDA_VISIBLE_DEVICES=6 python vslice/vslice_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type qwen2_vl_3b --batch_size=2 --clip_length=4 --num_epochs=5 --beta=0.1 --learning_rate=5e-5 --model_path=models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 > log_summe_dpo_qwen.txt
# CUDA_VISIBLE_DEVICES=6 python vslice/vslice_dpo.py --dataset summe --split_file ./dataset/summe_splits.json --model_type minicpm --batch_size=2 --clip_length=4 --num_epochs=5 --beta=0.1 --learning_rate=5e-5 --loss_type="DPO" --use_boost=False > log_summe_dpo.txt

GPU_A=1
GPU_B=2
DATASET="summe"

echo "Starting ${DATASET} on least-used GPU: $GPU_A"
CUDA_VISIBLE_DEVICES=$GPU_A python vslice/vslice_dpo.py \
    --dataset "$DATASET" \
    --split_file "./dataset/${DATASET}_splits.json" \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=3 \
    --beta=0.1 \
    --learning_rate=5e-5 \
    --use_boost=False \
    --loss_type="DPO" > "log_${DATASET}_dpo.txt" &

echo "Starting ${DATASET} on second least-used GPU: $GPU_B"
CUDA_VISIBLE_DEVICES=$GPU_B python vslice/vslice_dpo.py \
    --dataset "$DATASET" \
    --split_file "./dataset/${DATASET}_splits.json" \
    --model_type minicpm \
    --batch_size=2 \
    --clip_length=4 \
    --num_epochs=3 \
    --beta=0.1 \
    --learning_rate=5e-5 \
    --use_boost=False \
    --loss_type="MPO" > "log_${DATASET}_mpo.txt" &

wait
echo "All training jobs completed!"
