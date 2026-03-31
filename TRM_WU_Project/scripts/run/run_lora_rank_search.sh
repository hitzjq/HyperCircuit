#!/bin/bash

# ==========================================
# TRM LoRA Rank Search Script (Checkpoints 362650, 414457, 466264)
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "logs/2048_rank_test"

# ---- 可选开关 ----
UNFREEZE_EMBED_TOKENS=true   # 是否解冻 token embedding 参与微调

# 待测试的 Checkpoint 列表
CKPTS=("155422" "207229" "259036" "310843" "362650" "414457" "466264")
# Rank 列表
RANKS=(8 16 32 64 128 256)

for ckpt in "${CKPTS[@]}"; do
    for r in "${RANKS[@]}"; do
        # 计算 alpha，让 alpha = 2 * r 是常规的缩放比例
        alpha=$((r * 2))
        
        run_name="LoRA_r${r}_ckpt${ckpt}_$(date +%m%d_%H%M)"
        LOG_DIR="checkpoints/${run_name}"
        mkdir -p "${LOG_DIR}"

        echo "=========================================================="
        echo "Starting training for Rank ${r} (Checkpoint ${ckpt})"
        echo "Log: logs/2048_rank_test/${run_name}.log"
        echo "=========================================================="
        
        # eval_interval 设置和 epochs 相同，保证只在开头和结尾评估
        torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
            arch=trm \
            data_paths="[data/arc1concept-aug-1000]" \
            arch.L_layers=2 \
            arch.H_cycles=3 arch.L_cycles=4 \
            +run_name="${run_name}" \
            +checkpoint_path="${LOG_DIR}" \
            +load_checkpoint=checkpoints/ARC-AGI-1/step_${ckpt} \
            lora_r=${r} \
            lora_alpha=${alpha} \
            lr=5e-3 \
            lr_warmup_steps=800 \
            global_batch_size=2048 \
            epochs=10000 \
            eval_interval=10000 \
            +unfreeze_embed_tokens=${UNFREEZE_EMBED_TOKENS} \
            2>&1 | tee "logs/2048_rank_test/${run_name}.log"
    done
done

echo "所有指定 Checkpoints 的 Rank 测试完毕！"
