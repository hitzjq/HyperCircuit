#!/bin/bash
set -euo pipefail
set -x

export OMP_NUM_THREADS=4

# ============================================================
# 0423 circuit-conditioned HyperNet training
#
# Uses train-only circuit features, skips test evaluation, and saves
# checkpoints every EVAL_INTERVAL. Formal evaluation should be run later
# with a circuit file that includes test queries.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=2048
WARMUP=2000
EPOCHS=20000
EVAL_INTERVAL=2000
DATASET_PATH="/mnt/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
BASE_CKPT_PATH="/mnt/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071"
CIRCUIT_FEATURES_PATH="${CIRCUIT_FEATURES_PATH:-/mnt/kbei/HyperCircuit/TRM_WU_Project/CodeCircuit_TRM_Arc1/runs/prod_0421_1742/cc_advanced_features_train.pt}"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
SKIP_EVAL="True"
LOG_DIR="logs/logs0423"
mkdir -p "$LOG_DIR"

if [[ ! -f "$CIRCUIT_FEATURES_PATH" ]]; then
    echo "Circuit feature file not found: $CIRCUIT_FEATURES_PATH" >&2
    exit 1
fi

# Format:
#   tag|pg_blocks|pg_dim|lora_r|lora_alpha|weight_decay|lr|puzzle_lr|condition_mode|pg_use_rope|head_lora
EXPERIMENTS=(
    "C1_lorar8_circuit|2|256|8|16|0.1|1e-5|1e-3|full_trm|False|True"
    "C2_lorar32_circuit|2|256|32|64|0.1|1e-5|1e-3|full_trm|False|True"
    "R8_rope_r8_nohead_emb_circuit|2|256|8|16|0.1|1e-5|1e-3|embedding_only|True|False"
    "R2_rope_r8_emb_circuit|2|256|8|16|0.1|1e-5|1e-3|embedding_only|True|True"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r TAG PG_BLOCKS PG_DIM LORA_R LORA_ALPHA WD LR PUZZLE_LR COND_MODE USE_ROPE HEAD_LORA <<< "$EXP"

    run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
    CKPT_DIR="checkpoints/${run_name}"
    mkdir -p "$CKPT_DIR"

    echo "=========================================================="
    echo "Running: ${TAG}"
    echo "  pg_blocks=${PG_BLOCKS}, pg_d_model=${PG_DIM}, pg_use_rope=${USE_ROPE}"
    echo "  lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}, head_lora=${HEAD_LORA}"
    echo "  condition_mode=${COND_MODE}, lr=${LR}, wd=${WD}"
    echo "  circuit_features_path=${CIRCUIT_FEATURES_PATH}"
    echo "  skip_eval=${SKIP_EVAL}; checkpoints saved every ${EVAL_INTERVAL} epochs"
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
        checkpoint_every_eval=True \
        skip_eval=${SKIP_EVAL} \
        condition_mode="${COND_MODE}" \
        pg_num_blocks=$PG_BLOCKS \
        pg_d_model=$PG_DIM \
        pg_use_rope=$USE_ROPE \
        lora_r=$LORA_R \
        lora_alpha=$LORA_ALPHA \
        head_lora=$HEAD_LORA \
        weight_decay=$WD \
        circuit_features_path="${CIRCUIT_FEATURES_PATH}" \
        data_paths="['${DATASET_PATH}']" \
        +checkpoint_path="${CKPT_DIR}" \
        +load_checkpoint=$BASE_CKPT_PATH \
        +project_name="trm-hp-0423-circuit" \
        +run_name="${run_name}" \
        skip_baseline_eval=${SKIP_BASELINE_EVAL} \
        2>&1 | tee "${LOG_DIR}/${run_name}.log"

    echo "${TAG} done."
    sleep 10
done

echo "0423 circuit training done!"
