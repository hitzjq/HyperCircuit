#!/bin/bash

# ==========================================
# TRM End-to-End PRETRAIN Pipeline
# Data Generation -> Full Pretrain -> Evaluation
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

# 生成时间戳，保证所有文件命名不重合
TIMESTAMP=$(date +%m%d_%H%M)

# ---------------------------------------------------------
# 1. 自动生成增强数据集
# ---------------------------------------------------------
# 新的数据集名字，自带时间戳防止重名
NEW_DATA_DIR="data/arc1concept-aug-1000_${TIMESTAMP}"

echo "=========================================================="
echo "阶段 1: 开始生成 ARC-AGI-1 增强数据集 -> ${NEW_DATA_DIR}"
echo "=========================================================="
python -m dataset.build_arc_dataset_mem \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir "${NEW_DATA_DIR}" \
    --subsets training evaluation concept \
    --test-set-name evaluation

echo "数据集生成完毕。"

# ---------------------------------------------------------
# 2. 全量预训练 (Pretrain) + 评估 (Eval)
# ---------------------------------------------------------
# 定义全新的 run_name 和日志文件
RUN_NAME="E2E_Pretrain_${TIMESTAMP}"
LOG_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="logs/${RUN_NAME}.log"

mkdir -p "logs"
mkdir -p "${LOG_DIR}"

echo ""
echo "=========================================================="
echo "阶段 2 & 3: 使用新数据集进行全量 Pretrain 及评估"
echo "Run Name: ${RUN_NAME}"
echo "Log: ${LOG_FILE}"
echo "Checkpoint Dir: ${LOG_DIR}"
echo "=========================================================="

# 启动全量预训练模式
# 这里的 epochs 和 eval_interval 可以根据你的实际需求修改
torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
    arch=trm \
    data_paths="[${NEW_DATA_DIR}]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name="${RUN_NAME}" \
    +checkpoint_path="${LOG_DIR}" \
    global_batch_size=768 \
    epochs=100000 \
    eval_interval=10000 \
    lr=1e-4 \
    lr_warmup_steps=2000 \
    ema=True \
    2>&1 | tee "${LOG_FILE}"

echo "=========================================================="
echo "🎉 全量 Pretrain 端到端流程均已执行完毕！"
echo "日志保存在: ${LOG_FILE}"
echo "Checkpoints 保存在: ${LOG_DIR}"
echo "=========================================================="
