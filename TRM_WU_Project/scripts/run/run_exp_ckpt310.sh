#!/bin/bash

# =========================================================
# 需要改：
# 1. 下面的BASE_CKPT，设置成step_310843的路径
# 2. 下面的DATA_PATH应该是这个不用改（大概？
# ======================================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

BASE_CKPT="checkpoints/ARC-AGI-1/step_310843"
mkdir -p "logs"

# ---------------------------------------------------------
# Run 1: Unfreeze Embed Tokens = True
# ---------------------------------------------------------
RUN_NAME="LoRA_r64_ckpt310_unfreezeTrue_$(date +%m%d_%H%M)"
LOG_FILE="logs/${RUN_NAME}.log"
LOG_DIR="checkpoints/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

echo "🚀 Starting Run 1: ${RUN_NAME} (Unfreeze=True)"
torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
    arch=trm \
    data_paths="[data/arc1concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name="${RUN_NAME}" \
    +checkpoint_path="${LOG_DIR}" \
    +load_checkpoint="${BASE_CKPT}" \
    lora_r=64 \
    lora_alpha=32 \
    lr=1e-3 \
    lr_warmup_steps=800 \
    global_batch_size=768 \
    epochs=10000 \
    +unfreeze_embed_tokens=true \
    2>&1 | tee "${LOG_FILE}"

# ---------------------------------------------------------
# Run 2: Unfreeze Embed Tokens = False
# ---------------------------------------------------------
echo "冷却 10 秒后开始下一项任务..."
sleep 10

RUN_NAME="LoRA_r64_ckpt310_unfreezeFalse_$(date +%m%d_%H%M)"
LOG_FILE="logs/${RUN_NAME}.log"
LOG_DIR="checkpoints/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

echo "🚀 Starting Run 2: ${RUN_NAME} (Unfreeze=False)"
torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
    arch=trm \
    data_paths="[data/arc1concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name="${RUN_NAME}" \
    +checkpoint_path="${LOG_DIR}" \
    +load_checkpoint="${BASE_CKPT}" \
    lora_r=64 \
    lora_alpha=32 \
    lr=1e-3 \
    lr_warmup_steps=800 \
    global_batch_size=768 \
    epochs=10000 \
    +unfreeze_embed_tokens=false \
    2>&1 | tee "${LOG_FILE}"

echo "🎉 All experiments in run_exp_ckpt310.sh completed."
