#!/bin/bash

# ==============================================================================
# VSLICE Moondream2 DPO Training Script
# ==============================================================================
# Examples:
#   1. Default single run on SumMe:
#      bash vslice/run_moondream.sh
#
#   2. Override parameters via environment variables:
#      DATASET=tvsum GPU_ID=0 NUM_EPOCHS=3 LOSS_TYPE=MPO bash vslice/run_moondream.sh
# ==============================================================================

DATASET=${DATASET:-"summe"}
GPU_ID=${GPU_ID:-6}
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
CLIP_LENGTH=${CLIP_LENGTH:-4}
BETA=${BETA:-0.1}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE:-16}
LOSS_TYPE=${LOSS_TYPE:-"DPO"}

# Check for local cached snapshot or fall back to HuggingFace model ID
LOCAL_SNAPSHOT="models--vikhyatk--moondream2/snapshots/92d3d73b6fd61ab84d9fe093a9c7fd8c04bf2c0d"
if [ -d "$LOCAL_SNAPSHOT" ]; then
    MODEL_PATH="$LOCAL_SNAPSHOT"
else
    MODEL_PATH="vikhyatk/moondream2"
fi

echo "=========================================================="
echo "Starting VSLICE Moondream training"
echo "  Dataset    : ${DATASET}"
echo "  GPU ID     : ${GPU_ID}"
echo "  Loss Type  : ${LOSS_TYPE}"
echo "  Model Path : ${MODEL_PATH}"
echo "  Epochs     : ${NUM_EPOCHS}"
echo "  Batch Size : ${BATCH_SIZE}"
echo "  Learning Rt: ${LEARNING_RATE}"
echo "  Chunk Size : ${EVAL_CHUNK_SIZE}"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python vslice/vslice_moondream.py \
    --dataset "${DATASET}" \
    --split_file "./dataset/${DATASET}_splits.json" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --clip_length "${CLIP_LENGTH}" \
    --num_epochs "${NUM_EPOCHS}" \
    --beta "${BETA}" \
    --learning_rate "${LEARNING_RATE}" \
    --eval_chunk_size "${EVAL_CHUNK_SIZE}" \
    --loss_type "${LOSS_TYPE}" > "log_${DATASET}_${LOSS_TYPE,,}_moondream.txt" 2>&1

echo "Moondream training completed! Log saved to log_${DATASET}_${LOSS_TYPE,,}_moondream.txt"
