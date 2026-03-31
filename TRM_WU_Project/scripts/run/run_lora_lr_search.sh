#!/bin/bash

# ==========================================
# TRM LoRA LR Search Script (All Checkpoints)
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "logs/2048_lr_test"

# ---- 控制变量设置 ----
UNFREEZE_EMBED_TOKENS=true   # 是否解冻 token embedding 参与微调
# !! 注意 !! 请将下方的 r 和 alpha 设为你之前 Rank search 得到的【最优 Rank】
FIXED_RANK=16
FIXED_ALPHA=32
# --------------------

# 待测试的 Checkpoint 列表
CKPTS=("155422" "207229" "259036" "310843" "362650" "414457" "466264")

# Learning Rate 列表: 测试经典的收敛区间 (从大到小)
LRS=("2e-3" "3e-3" "4e-3" "5e-3" "6e-3" "7e-3" "8e-3" "9e-3" "1e-2")

for ckpt in "${CKPTS[@]}"; do
    for lr in "${LRS[@]}"; do
        run_name="LoRA_lr${lr}_ckpt${ckpt}_$(date +%m%d_%H%M)"
        LOG_DIR="checkpoints/${run_name}"
        mkdir -p "${LOG_DIR}"

        echo "=========================================================="
        echo "Starting training for LR ${lr} (Checkpoint ${ckpt}, Rank ${FIXED_RANK})"
        echo "Log: logs/2048_lr_test/${run_name}.log"
        echo "=========================================================="
        
        # 保持所有其他变量绝对不变
        torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
            arch=trm \
            data_paths="[data/arc1concept-aug-1000]" \
            arch.L_layers=2 \
            arch.H_cycles=3 arch.L_cycles=4 \
            +run_name="${run_name}" \
            +checkpoint_path="${LOG_DIR}" \
            +load_checkpoint=checkpoints/ARC-AGI-1/step_${ckpt} \
            lora_r=${FIXED_RANK} \
            lora_alpha=${FIXED_ALPHA} \
            lr=${lr} \
            lr_warmup_steps=800 \
            global_batch_size=2048 \
            epochs=10000 \
            eval_interval=10000 \
            +unfreeze_embed_tokens=${UNFREEZE_EMBED_TOKENS} \
            2>&1 | tee "logs/2048_lr_test/${run_name}.log"
    done
done

echo "所有 Checkpoints 的 Learning Rate 测试完毕！"
