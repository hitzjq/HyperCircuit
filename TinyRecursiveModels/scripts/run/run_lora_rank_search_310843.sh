#!/bin/bash

# ==========================================
# TRM LoRA Rank Search Script (Checkpoint 310843)
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "logs"

# ---- 可选开关 ----
UNFREEZE_EMBED_TOKENS=false   # 是否解冻 token embedding 参与微调

# Rank列表
RANKS=(8 16 32 64 128 256)

for r in "${RANKS[@]}"; do
    # 计算 alpha，让 alpha = 2 * r 是常规的缩放比例
    alpha=$((r * 2))
    
    run_name="LoRA_r${r}_ckpt310843_$(date +%m%d_%H%M)"
    LOG_DIR="checkpoints/${run_name}"
    mkdir -p "${LOG_DIR}"

    echo "=========================================================="
    echo "Starting training for Rank ${r} (Checkpoint 310843)"
    echo "Log: logs/${run_name}.log"
    echo "=========================================================="
    
    # eval_interval 设置和 epochs 相同，保证只在开头和结尾评估
    torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
        arch=trm \
        data_paths="[data/arc1concept-aug-1000]" \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=4 \
        +run_name="${run_name}" \
        +checkpoint_path="${LOG_DIR}" \
        +load_checkpoint=checkpoints/ARC-AGI-1/step_310843 \
        lora_r=${r} \
        lora_alpha=${alpha} \
        lr=1e-3 \
        lr_warmup_steps=800 \
        global_batch_size=1024 \
        epochs=10000 \
        eval_interval=10000 \
        +unfreeze_embed_tokens=${UNFREEZE_EMBED_TOKENS} \
        2>&1 | tee "logs/${run_name}.log"
done

echo "所有 Rank 测试完毕 (Checkpoint 310843)。"
