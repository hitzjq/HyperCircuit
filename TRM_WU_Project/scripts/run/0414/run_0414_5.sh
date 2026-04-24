#!/bin/bash
set -x
export OMP_NUM_THREADS=4

# ============================================================
# 短线脚本 5/7: D1 (lr5e6) + D2 (lr2e5)
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=2048
WARMUP=2000
EPOCHS=20000
EVAL_INTERVAL=2000
CONDITION_MODE="full_trm"
DATASET_PATH="/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
BASE_CKPT_PATH="/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
LOG_DIR="logs/logs0414"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
    "D1_lr5e6|2|256|16|32|0.1|5e-6|5e-4"
    "D2_lr2e5|2|256|16|32|0.1|2e-5|2e-3"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG PG_BLOCKS PG_DIM LORA_R LORA_ALPHA WD LR PUZZLE_LR <<< "$EXP"
    run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
    CKPT_DIR="checkpoints/${run_name}"
    mkdir -p "${CKPT_DIR}"
    echo "Running: ${TAG} | pgb=${PG_BLOCKS} pgd=${PG_DIM} r=${LORA_R} a=${LORA_ALPHA} wd=${WD} lr=${LR} plr=${PUZZLE_LR}"
    torchrun --nproc-per-node=${NUM_GPUS} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        meta_train.py --config-name=$CFG \
        global_batch_size=$GLOBAL_BATCH_SIZE lr=$LR puzzle_emb_lr=$PUZZLE_LR \
        lr_warmup_steps=$WARMUP epochs=$EPOCHS eval_interval=$EVAL_INTERVAL \
        condition_mode="${CONDITION_MODE}" pg_num_blocks=$PG_BLOCKS pg_d_model=$PG_DIM \
        lora_r=$LORA_R lora_alpha=$LORA_ALPHA weight_decay=$WD \
        data_paths="['${DATASET_PATH}']" +checkpoint_path="${CKPT_DIR}" \
        +load_checkpoint=$BASE_CKPT_PATH +project_name="trm-hp-0414" +run_name="${run_name}" \
        skip_baseline_eval=${SKIP_BASELINE_EVAL} \
        2>&1 | tee "${LOG_DIR}/${run_name}.log"
    echo "${TAG} done."
    sleep 10
done
echo "Script 5/7 (D1+D2) done!"
