#!/bin/bash
set -x
export OMP_NUM_THREADS=4

# ============================================================
# 短线探索脚本 1/2: S1–S4 (PG架构探索)
# S1: pg_num_blocks=3
# S2: pg_num_blocks=4
# S3: pg_d_model=384
# S4: pg_d_model=512
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=4096
LR=2e-5
PUZZLE_LR=2e-3
WARMUP=2000
EPOCHS=20000
EVAL_INTERVAL=2000
CONDITION_MODE="full_trm"
DATASET_PATH="/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
BASE_CKPT_PATH="/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_155422"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
LOG_DIR="logs/logs0413"
mkdir -p "$LOG_DIR"

# 格式: "标签|pg_num_blocks|pg_d_model|lora_r|lora_alpha|weight_decay"
EXPERIMENTS=(
    "S1_pgblocks3|3|256|16|32|0.1"
    "S2_pgblocks4|4|256|16|32|0.1"
    "S3_pgdim384|2|384|16|32|0.1"
    "S4_pgdim512|2|512|16|32|0.1"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG PG_BLOCKS PG_DIM LORA_R LORA_ALPHA WD <<< "$EXP"

    run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
    CKPT_DIR="checkpoints/${run_name}"
    mkdir -p "${CKPT_DIR}"

    echo "=========================================================="
    echo "🚀 短线探索: ${TAG}"
    echo "   pg_num_blocks=${PG_BLOCKS}, pg_d_model=${PG_DIM}"
    echo "   lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}, weight_decay=${WD}"
    echo "   lr=${LR}, puzzle_emb_lr=${PUZZLE_LR}"
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
        pg_num_blocks=$PG_BLOCKS \
        pg_d_model=$PG_DIM \
        lora_r=$LORA_R \
        lora_alpha=$LORA_ALPHA \
        weight_decay=$WD \
        data_paths="['${DATASET_PATH}']" \
        +checkpoint_path="${CKPT_DIR}" \
        +load_checkpoint=$BASE_CKPT_PATH \
        +project_name="trm-hp-explore" \
        +run_name="${run_name}" \
        skip_baseline_eval=${SKIP_BASELINE_EVAL} \
        2>&1 | tee "${LOG_DIR}/${run_name}.log"

    echo "✅ ${TAG} 完成。"
    sleep 10
done

echo "🎉 短线脚本 1/2 (S1–S4) 全部完成！"
