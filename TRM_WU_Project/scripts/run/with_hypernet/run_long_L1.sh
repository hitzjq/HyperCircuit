#!/bin/bash
set -x
export OMP_NUM_THREADS=4

# ============================================================
# 长线训练 L1: 激进冲击型
# lr=2e-5, puzzle_emb_lr=2e-3, warmup=2000, epochs=100000
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=4096
LR=2e-5
PUZZLE_LR=2e-3
WARMUP=2000
EPOCHS=100000
EVAL_INTERVAL=5000
CONDITION_MODE="full_trm"
DATASET_PATH="/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
BASE_CKPT_PATH="/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_155422"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
LOG_DIR="logs/logs0413"
mkdir -p "$LOG_DIR"

run_name="WU4TRM_L1_aggressive_lr2e5_plr2e3_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
CKPT_DIR="checkpoints/${run_name}"
mkdir -p "${CKPT_DIR}"

echo "=========================================================="
echo "🚀 长线训练 L1: 激进冲击型"
echo "   lr=${LR}, puzzle_emb_lr=${PUZZLE_LR}, warmup=${WARMUP}"
echo "   epochs=${EPOCHS}, eval_interval=${EVAL_INTERVAL}"
echo "   run_name=${run_name}"
echo "=========================================================="

torchrun --nproc-per-node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    meta_train.py \
    --config-name=$CFG \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    lr=$LR \
    puzzle_emb_lr=$PUZZLE_LR \
    lr_warmup_steps=$WARMUP \
    epochs=$EPOCHS \
    eval_interval=$EVAL_INTERVAL \
    condition_mode="${CONDITION_MODE}" \
    data_paths="['${DATASET_PATH}']" \
    +checkpoint_path="${CKPT_DIR}" \
    +load_checkpoint=$BASE_CKPT_PATH \
    +project_name="trm-long-train" \
    +run_name="${run_name}" \
    skip_baseline_eval=${SKIP_BASELINE_EVAL} \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"

echo "✅ L1 长线训练完成。"
