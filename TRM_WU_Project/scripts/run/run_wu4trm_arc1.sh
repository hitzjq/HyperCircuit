#!/bin/bash
set -x

export OMP_NUM_THREADS=4

# ==========================================
# TRM_WU_Project 双趟前向训练一键启动脚本
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 配置执行参数 ----
SEQ_LEN=1024
BATCH_SIZE=8  # Micro-batch size per GPU
LR=1e-4

# ---- 日志与权重保存路径配置 ----
run_name="WU4TRM_r16_lr1e-4_$(date +%m%d_%H%M)"
CKPT_DIR="checkpoints/${run_name}"
mkdir -p "${CKPT_DIR}"
mkdir -p "logs"

# 1. Base TRM checkpoint
BASE_CKPT_PATH="pretrained_base_ckpt/ARC-AGI-1/step_155718" 

# 2. Dataset paths
ARCH="trm"
CFG="cfg_wu4trm" 

echo "🚀 开始超网络动态 LoRA 训练..."
echo "📂 权重将保存在: ${CKPT_DIR}"
echo "📝 终端日志将保存在: logs/${run_name}.log"

# We use meta_train.py which contains the Two-Pass Forward loop
accelerate launch \
    --num_processes=1 \
    meta_train.py \
    --config-name=$CFG \
    arch.name=$ARCH \
    global_batch_size=$BATCH_SIZE \
    lr=$LR \
    +checkpoint_path="${CKPT_DIR}" \
    load_checkpoint=$BASE_CKPT_PATH \
    project_name="trm-hypernetwork-integration" \
    +run_name="${run_name}" \
    2>&1 | tee "logs/${run_name}.log"

echo "✅ 训练已完成。终端日志保存在: logs/${run_name}.log"
