#!/bin/bash
# ==========================================================
# Phase 1 执行脚本：归因图提取
# ==========================================================
# 用法：bash CodeCircuit_TRM_Arc1/run_phase1.sh
#
# 模式：
#   DEBUG  → A100 上用随机权重快速排雷（5 条 query）
#   PROD   → H200 上用真实权重全量提图
# ==========================================================
set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 用户配置区 ----
DATASET_PATH="data/arc1concept-aug-1000"
CKPT_PATH="../TinyRecursiveModels/checkpoints/ARC-AGI-1"
SAE_PATH="CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096.pt"
CONFIG_PATH="config/cfg_wu4trm.yaml"

# 运行模式选择: "DEBUG" (A100秒跑防雷) 或 "PROD" (H200真实开采)
RUN_MODE="DEBUG" 

# ========================================
# ---- 按模式设定参数 ----
# ========================================
if [ "$RUN_MODE" = "DEBUG" ]; then
    MAX_QUERIES=5
    CKPT_ARG=""
    SPLIT="train"
elif [ "$RUN_MODE" = "PROD" ]; then
    MAX_QUERIES=-1
    CKPT_ARG="$CKPT_PATH"
    SPLIT="test"
else
    echo "❌ Error: RUN_MODE must be 'DEBUG' or 'PROD'"
    exit 1
fi

# ========================================
# ---- 日志机制配置 ----
# ========================================
RUN_TIME=$(date +%m%d_%H%M)
LOG_FILE="CodeCircuit_TRM_Arc1/logs/cc_phase1_${RUN_MODE}_${RUN_TIME}.log"
mkdir -p CodeCircuit_TRM_Arc1/logs

echo "📝 终端正在启动运行！详细输出将实时保存在: $LOG_FILE"

# ========================================
# ---- 执行归因提图 ----
# ========================================
{
    echo "=========================================="
    echo ">> [Phase 1] 归因图提取"
    echo "=========================================="
    python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
        --config_path "$CONFIG_PATH" \
        --ckpt_path "$CKPT_ARG" \
        --sae_path "$SAE_PATH" \
        --dataset_paths "$DATASET_PATH" \
        --max_queries "$MAX_QUERIES" \
        --split "$SPLIT"
    
    echo "✅ Phase 1 脚本执行完毕！请检查 CodeCircuit_TRM_Arc1/results/attribution_graphs/ 下的 .pt 文件。"
} 2>&1 | tee "$LOG_FILE"
