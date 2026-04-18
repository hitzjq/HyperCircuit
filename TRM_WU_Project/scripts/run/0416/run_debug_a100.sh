#!/bin/bash
set -x
export OMP_NUM_THREADS=4

# ============================================================
# A100 调试脚本: 验证 meta_train.py + HyperNet 管线能跑通
# ============================================================
# - 小 batch (16), 少 epoch (100), 快速 eval
# - 不需要真 ckpt, 随机初始化即可
# - 目的: 确认代码不报错, 不验证效果
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=16
WARMUP=10
EPOCHS=100
EVAL_INTERVAL=50
DATASET_PATH="data/arc1concept-aug-1000"
CFG="cfg_wu4trm"
LOG_DIR="logs/logs_debug"
mkdir -p "$LOG_DIR"

TAG="debug_a100"
run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
CKPT_DIR="checkpoints/${run_name}"
mkdir -p "${CKPT_DIR}"

echo "=========================================================="
echo "  A100 Debug Run"
echo "  Tag: ${TAG}"
echo "  Config: r=8, alpha=16, NO RoPE, full_trm, lr=1e-5"
echo "  Epochs: ${EPOCHS}, eval_interval: ${EVAL_INTERVAL}"
echo "  ⚠️  随机初始化, 仅验证管线不报错"
echo "=========================================================="

torchrun --nproc-per-node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    meta_train.py \
    --config-name=$CFG \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    lr=1e-5 \
    puzzle_emb_lr=1e-3 \
    lr_warmup_steps=$WARMUP \
    epochs=$EPOCHS \
    eval_interval=$EVAL_INTERVAL \
    condition_mode="full_trm" \
    pg_num_blocks=2 \
    pg_d_model=256 \
    pg_use_rope=False \
    lora_r=8 \
    lora_alpha=16 \
    head_lora=True \
    weight_decay=0.1 \
    skip_baseline_eval=True \
    data_paths="['${DATASET_PATH}']" \
    +checkpoint_path="${CKPT_DIR}" \
    +project_name="trm-hp-debug" \
    +run_name="${run_name}" \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"

echo "Debug run done!"
