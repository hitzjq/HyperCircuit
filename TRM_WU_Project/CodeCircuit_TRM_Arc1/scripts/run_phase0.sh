#!/bin/bash
set -x

# ==========================================
# CodeCircuit TRM: Phase 0 收集激活与 SAE 训练流水线脚本
# ==========================================

# 获取本脚本的执行目录，并强制对齐到外侧工程目录，以保证 python path 一致
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 用户配置区 ----
DATASET_PATH="data/arc1concept-aug-1000"
CKPT_PATH="../TinyRecursiveModels/checkpoints/ARC-AGI-1"

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

# ========================================
# ---- 日志机制配置 ----
# ========================================
RUN_TIME=$(date +%m%d_%H%M)
LOG_FILE="CodeCircuit_TRM_Arc1/logs/cc_phase0_${RUN_MODE}_${RUN_TIME}.log"
mkdir -p CodeCircuit_TRM_Arc1/logs

echo "📝 终端正在启动运行！详细输出将实时保存在: $LOG_FILE"

# 将后续所有执行过程及其报错全部包装进管道，同步写盘
{
    echo "=========================================="
    echo ">> [STEP 1/2] 开始前向推理并采集全层激活"
    echo "=========================================="
    python CodeCircuit_TRM_Arc1/src/collect_activations.py \
        --dataset_paths "$DATASET_PATH" \
        --ckpt_path "$CKPT_ARG" \
        --max_batches $MAX_BATCHES

    echo "=========================================="
    echo ">> [STEP 2/2] 加载落地数据并训练 SAE Transcoder"
    echo "=========================================="
    python CodeCircuit_TRM_Arc1/src/train_transcoder.py

    echo "✅ Phase 0 脚本流水线执行完毕！请检查 CodeCircuit_TRM_Arc1/checkpoints/ 下的 trm_transcoder_4096.pt 字典是否成功保存。"
} 2>&1 | tee "$LOG_FILE"
