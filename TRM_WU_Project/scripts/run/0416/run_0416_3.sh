#!/bin/bash
set -x
export OMP_NUM_THREADS=4

# ============================================================
# 短线脚本 3/4: R5 (norope_r8_emb) + R6 (rope_r8_lr5e6)
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=2048
WARMUP=2000
EPOCHS=20000
EVAL_INTERVAL=2000
DATASET_PATH="/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
BASE_CKPT_PATH="/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
LOG_DIR="logs/logs0416"
mkdir -p "$LOG_DIR"

# 格式: "标签|pg_blocks|pg_dim|lora_r|lora_alpha|wd|lr|puzzle_lr|condition_mode|pg_use_rope|head_lora"
EXPERIMENTS=(
    "R5_norope_r8_emb|2|256|8|16|0.1|1e-5|1e-3|embedding_only|False|True"
    "R6_rope_r8_lr5e6|2|256|8|16|0.1|5e-6|1e-3|full_trm|True|True"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG PG_BLOCKS PG_DIM LORA_R LORA_ALPHA WD LR PUZZLE_LR COND_MODE USE_ROPE HEAD_LORA <<< "$EXP"

    run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
    CKPT_DIR="checkpoints/${run_name}"
    mkdir -p "${CKPT_DIR}"

    echo "=========================================================="
    echo "Running: ${TAG}"
    echo "  pg_blocks=${PG_BLOCKS}, pg_d_model=${PG_DIM}, pg_use_rope=${USE_ROPE}"
    echo "  lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}, head_lora=${HEAD_LORA}"
    echo "  condition_mode=${COND_MODE}, lr=${LR}, wd=${WD}"
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
        condition_mode="${COND_MODE}" \
        pg_num_blocks=$PG_BLOCKS \
        pg_d_model=$PG_DIM \
        pg_use_rope=$USE_ROPE \
        lora_r=$LORA_R \
        lora_alpha=$LORA_ALPHA \
        head_lora=$HEAD_LORA \
        weight_decay=$WD \
        data_paths="['${DATASET_PATH}']" \
        +checkpoint_path="${CKPT_DIR}" \
        +load_checkpoint=$BASE_CKPT_PATH \
        +project_name="trm-hp-0416" \
        +run_name="${run_name}" \
        skip_baseline_eval=${SKIP_BASELINE_EVAL} \
        2>&1 | tee "${LOG_DIR}/${run_name}.log"

    echo "${TAG} done."
    sleep 10
done

echo "Script 3/4 (R5+R6) done!"
