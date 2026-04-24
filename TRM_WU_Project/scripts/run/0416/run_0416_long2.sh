#!/bin/bash
set -x
export OMP_NUM_THREADS=4

# ============================================================
# 长线脚本 2/2: L2 (r=8 + RoPE 延长至100k)
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=2048
WARMUP=5000
EPOCHS=100000
EVAL_INTERVAL=5000
DATASET_PATH="/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
BASE_CKPT_PATH="/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
LOG_DIR="logs/logs0416"
mkdir -p "$LOG_DIR"

TAG="L2_long_r8_rope"
run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
CKPT_DIR="checkpoints/${run_name}"
mkdir -p "${CKPT_DIR}"

echo "=========================================================="
echo "Running LONG: ${TAG}"
echo "  Config: r=8, alpha=16, RoPE=ON, full_trm, lr=1e-5"
echo "  Epochs: ${EPOCHS}, eval_interval: ${EVAL_INTERVAL}"
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
    pg_use_rope=True \
    lora_r=8 \
    lora_alpha=16 \
    head_lora=True \
    weight_decay=0.1 \
    data_paths="['${DATASET_PATH}']" \
    +checkpoint_path="${CKPT_DIR}" \
    +load_checkpoint=$BASE_CKPT_PATH \
    +project_name="trm-hp-0416-long" \
    +run_name="${run_name}" \
    skip_baseline_eval=${SKIP_BASELINE_EVAL} \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"

echo "Long run L2 done!"
