#!/bin/bash
set -x

export OMP_NUM_THREADS=4

# ============================================================
# 超参搜索脚本 2/2
# 实验 C: lr=1e-5, puzzle_emb_lr=1e-4, warmup=2000
# 实验 D: lr=5e-6, puzzle_emb_lr=1e-4, warmup=3000
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 固定参数 ----
NUM_GPUS=8
GLOBAL_BATCH_SIZE=1024
CONDITION_MODE="embedding_only"
DATASET_PATH="data/arc1concept-aug-1000"
BASE_CKPT_PATH="pretrained_base_ckpt/ARC-AGI-1/step_155718"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"

# ---- 实验组定义 ----
# 格式: "实验标签|lr|puzzle_emb_lr|warmup"
EXPERIMENTS=(
    "lr1e5_plr1e4_wu2k|1e-5|1e-4|2000"
    "lr5e6_plr1e4_wu3k|5e-6|1e-4|3000"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG LR PUZZLE_LR WARMUP <<< "$EXP"

    run_name="WU4TRM_hp_${TAG}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
    CKPT_DIR="checkpoints/${run_name}"
    mkdir -p "${CKPT_DIR}"
    mkdir -p "logs"

    echo "=========================================================="
    echo "🚀 超参搜索实验: ${TAG}"
    echo "   lr=${LR}, puzzle_emb_lr=${PUZZLE_LR}, warmup=${WARMUP}"
    echo "   condition_mode=${CONDITION_MODE}"
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
        condition_mode="${CONDITION_MODE}" \
        data_paths="['${DATASET_PATH}']" \
        +checkpoint_path="${CKPT_DIR}" \
        +load_checkpoint=$BASE_CKPT_PATH \
        +project_name="trm-hp-search" \
        +run_name="${run_name}" \
        skip_baseline_eval=${SKIP_BASELINE_EVAL} \
        2>&1 | tee "logs/${run_name}.log"

    echo "✅ 实验 ${TAG} 完成。"
    sleep 10
done

echo "🎉 脚本 2/2 全部实验完成！"
