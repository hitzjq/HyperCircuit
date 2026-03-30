#!/bin/bash

# ==========================================
# TRM SHINE LoRA 微调一键启动脚本
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

run_name="SHINE_LoRA_r64_lr1e-3_8gpus_$(date +%m%d_%H%M)"
LOG_DIR="checkpoints/${run_name}"
mkdir -p "${LOG_DIR}"
mkdir -p "logs"

# ---- 可选开关 ----
UNFREEZE_EMBED_TOKENS=true   # 是否解冻 token embedding 参与微调

torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 meta_train_trm.py \
    --config-name cfg_shine_lora \
    arch=trm \
    data_paths="[data/arc1concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    run_name="${run_name}" \
    checkpoint_path="${LOG_DIR}" \
    load_checkpoint=checkpoints/ARC-AGI-1/step_310843 \
    lora_r=64 \
    lora_alpha=32 \
    hypernet_num_layers=3 \
    hypernet_num_heads=4 \
    hypernet_scale=1.0 \
    lr=1e-3 \
    lr_warmup_steps=800 \
    global_batch_size=1024 \
    epochs=10000 \
    eval_interval=10000 \
    unfreeze_embed_tokens=${UNFREEZE_EMBED_TOKENS} \
    2>&1 | tee "logs/${run_name}.log"

echo "训练已完成。日志保存在: logs/${run_name}.log"
