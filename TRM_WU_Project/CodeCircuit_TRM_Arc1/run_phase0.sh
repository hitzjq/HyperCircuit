#!/bin/bash
set -x

# ==========================================
# CodeCircuit TRM: Phase 0 收集激活与 SAE 训练流水线脚本
# ==========================================

# 获取本脚本的执行目录，并强制对齐到外侧工程目录，以保证 python path 一致
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 用户配置区 ----
DATASET_PATH="../data"
CKPT_PATH="../pretrained_base_ckpt/ARC-AGI-1"

# 运行模式选择: "DEBUG" (A100秒跑防雷) 或 "PROD" (H200真实开采)
RUN_MODE="DEBUG" 

# ========================================

# 逻辑分辨
if [ "$RUN_MODE" = "DEBUG" ]; then
    echo "🚨 启动【调试模式】: 绕过全量真实权重保护，跑 5 个批次看内存和逻辑"
    MAX_BATCHES=5
    CKPT_ARG=""
else
    echo "🔥 启动【生产开矿模式】: 强加载 H200 真实 Checkpoint，全量跑透"
    MAX_BATCHES=-1
    CKPT_ARG="$CKPT_PATH"
fi

echo "=========================================="
echo ">> [STEP 1/2] 开始前向推理并采集全层激活"
echo "=========================================="
python CodeCircuit_TRM_Arc1/transcoder/collect_activations.py \
    --dataset_paths "$DATASET_PATH" \
    --ckpt_path "$CKPT_ARG" \
    --max_batches $MAX_BATCHES

echo "=========================================="
echo ">> [STEP 2/2] 加载落地数据并训练 SAE Transcoder"
echo "=========================================="
python CodeCircuit_TRM_Arc1/transcoder/train_transcoder.py

echo "✅ Phase 0 脚本流水线执行完毕！请检查 CodeCircuit_TRM_Arc1/transcoder 下的 trm_transcoder_4096.pt 字典是否成功保存。"
