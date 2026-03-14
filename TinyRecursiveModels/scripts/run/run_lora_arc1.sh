#!/bin/bash

# ==========================================
# TRM LoRA 微调一键启动脚本 (遵循官方精简版)
# ==========================================

run_name="LoRA_r64_lr1e-3_8gpus"
LOG_DIR="checkpoints/${run_name}"
mkdir -p "${LOG_DIR}"

nohup torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 lora_finetune.py \
    arch=trm \
    data_paths="[data/arc1concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name="${run_name}" \
    +checkpoint_path="${LOG_DIR}" \
    +load_checkpoint=checkpoints/ARC-AGI-1/step_362650 \
    lora_r=64 \
    lora_alpha=32 \
    lr=1e-3 \
    global_batch_size=256 \
    epochs=20000 \
    > "${LOG_DIR}/terminal_output.log" 2>&1 &

echo "进程已在后台运行。查看日志: tail -f ${LOG_DIR}/terminal_output.log"
