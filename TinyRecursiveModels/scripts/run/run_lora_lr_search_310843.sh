#!/bin/bash

# ==========================================
# TRM LoRA LR Search Script (Checkpoint 310843)
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "logs"

# ---- 控制变量设置 ----
UNFREEZE_EMBED_TOKENS=true   # 是否解冻 token embedding 参与微调
# !! 注意 !! 请将下方的 r 和 alpha 设为你之前 Rank search 得到的【最优 Rank】
FIXED_RANK=32
FIXED_ALPHA=64
# --------------------

# Learning Rate 列表: 测试经典的收敛区间 (从大到小)
LRS=("5e-3" "1e-3" "5e-4" "1e-4" "5e-5" "1e-5")

for lr in "${LRS[@]}"; do
    run_name="LoRA_lr${lr}_ckpt310843_$(date +%m%d_%H%M)"
    LOG_DIR="checkpoints/${run_name}"
    mkdir -p "${LOG_DIR}"

    echo "=========================================================="
    echo "Starting training for LR ${lr} (Checkpoint 310843, Rank ${FIXED_RANK})"
    echo "Log: logs/${run_name}.log"
    echo "=========================================================="
    
    # 保持所有其他变量绝对不变
    torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
        arch=trm \
        data_paths="[data/arc1concept-aug-1000]" \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=4 \
        +run_name="${run_name}" \
        +checkpoint_path="${LOG_DIR}" \
        +load_checkpoint=checkpoints/ARC-AGI-1/step_310843 \
        lora_r=${FIXED_RANK} \
        lora_alpha=${FIXED_ALPHA} \
        lr=${lr} \
        lr_warmup_steps=800 \
        global_batch_size=1024 \
        epochs=10000 \
        eval_interval=10000 \
        +unfreeze_embed_tokens=${UNFREEZE_EMBED_TOKENS} \
        2>&1 | tee "logs/${run_name}.log"
done

echo "所有 Learning Rate 测试完毕 (Checkpoint 310843)。"
