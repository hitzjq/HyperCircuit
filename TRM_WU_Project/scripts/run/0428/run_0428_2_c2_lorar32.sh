#!/bin/bash
set -euo pipefail
set -x

export OMP_NUM_THREADS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
cd "$PROJECT_ROOT"

CACHE_ROOT="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${CACHE_ROOT}/inductor" "${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"

NUM_GPUS=8
GLOBAL_BATCH_SIZE=2048
WARMUP=2000
EPOCHS=20000
EVAL_INTERVAL=2000
DATASET_PATH="${DATASET_PATH:-/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000}"
BASE_CKPT_PATH="${BASE_CKPT_PATH:-/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071}"
CIRCUIT_FEATURES_PATH="${CIRCUIT_FEATURES_PATH:-/volume/safety/kbei/HyperCircuit/TRM_WU_Project/CodeCircuit_TRM_Arc1/runs/prod_0421_1742/cc_advanced_features.pt}"
CFG="cfg_wu4trm"
SKIP_BASELINE_EVAL="True"
SKIP_EVAL="False"
LOG_DIR="logs/logs0428"
mkdir -p "$LOG_DIR"

TAG="C2_lorar32_circuit"
PG_BLOCKS=2
PG_DIM=256
LORA_R=32
LORA_ALPHA=64
WD=0.1
LR=1e-5
PUZZLE_LR=1e-3
COND_MODE="full_trm"
USE_ROPE=False
HEAD_LORA=True

if [[ ! -f "$CIRCUIT_FEATURES_PATH" ]]; then
    echo "Circuit feature file not found: $CIRCUIT_FEATURES_PATH" >&2
    exit 1
fi

run_name="WU4TRM_${TAG}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
CKPT_DIR="checkpoints/${run_name}"
mkdir -p "$CKPT_DIR"

echo "=========================================================="
echo "Running: ${TAG}"
echo "  pg_blocks=${PG_BLOCKS}, pg_d_model=${PG_DIM}, pg_use_rope=${USE_ROPE}"
echo "  lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}, head_lora=${HEAD_LORA}"
echo "  condition_mode=${COND_MODE}, lr=${LR}, wd=${WD}"
echo "  circuit_features_path=${CIRCUIT_FEATURES_PATH}"
echo "  skip_eval=${SKIP_EVAL}; evaluation + checkpoints every ${EVAL_INTERVAL} epochs"
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
    +project_name="trm-hp-0428-circuit" \
    +run_name="${run_name}" \
    skip_baseline_eval=${SKIP_BASELINE_EVAL} \
    2>&1 | tee "${LOG_DIR}/${run_name}.log"

echo "${TAG} done."
