#!/bin/bash
# ==========================================================
# Phase 1 执行脚本：归因图提取
# ==========================================================
set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 用户配置区 ----
DATASET_PATH="data/arc1concept-aug-1000"
CKPT_PATH="../TinyRecursiveModels/checkpoints/ARC-AGI-1"
SAE_PATH_0="CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096_block_0.pt"
SAE_PATH_1="CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096_block_1.pt"
CONFIG_PATH="config/cfg_wu4trm.yaml"

# 运行模式: "DEBUG" (A100) 或 "PROD" (H200)
RUN_MODE="DEBUG" 

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

RUN_TIME=$(date +%m%d_%H%M)
LOG_FILE="CodeCircuit_TRM_Arc1/logs/cc_phase1_${RUN_MODE}_${RUN_TIME}.log"
mkdir -p CodeCircuit_TRM_Arc1/logs

echo "📝 终端正在启动运行！详细输出将实时保存在: $LOG_FILE"

{
    echo "=========================================="
    echo ">> [Phase 1] 归因图提取 (last-step VJP, 双 SAE)"
    echo "=========================================="
    python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
        --config_path "$CONFIG_PATH" \
        --ckpt_path "$CKPT_ARG" \
        --sae_path_0 "$SAE_PATH_0" \
        --sae_path_1 "$SAE_PATH_1" \
        --dataset_paths "$DATASET_PATH" \
        --max_queries "$MAX_QUERIES" \
        --split "$SPLIT" \
        --use_last_step
    
    echo "✅ Phase 1 脚本执行完毕！"
} 2>&1 | tee "$LOG_FILE"

