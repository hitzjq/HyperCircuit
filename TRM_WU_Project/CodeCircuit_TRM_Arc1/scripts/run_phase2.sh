#!/bin/bash

# ==============================================================================
# CodeCircuit TRM Attribution Pipeline
# Phase 2: Graph Vectorization (图 -> 特征向量转换)
# ==============================================================================

# Ensure we're in the right directory
PROJECT_ROOT=$(cd "$(dirname "$0")/../../" && pwd)
cd "$PROJECT_ROOT"

# Configuration
INPUT_DIR="CodeCircuit_TRM_Arc1/results/attribution_graphs/*.pt"
OUTPUT_PATH="CodeCircuit_TRM_Arc1/results/cc_advanced_features.pt"

# Create logs directory if missing
mkdir -p CodeCircuit_TRM_Arc1/logs
RUN_TIME=$(date +"%m%d_%H%M")
LOG_FILE="CodeCircuit_TRM_Arc1/logs/cc_phase2_${RUN_TIME}.log"

echo "📝 终端正在启动运行！详细输出将实时保存在: $LOG_FILE"

# Start the pipeline
(
    echo "=========================================="
    echo ">> [Phase 2] 高级拓扑特征向量提取"
    echo "=========================================="
    
    python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
        --input_dir "$INPUT_DIR" \
        --output_path "$OUTPUT_PATH"
        
    echo "=========================================="
    echo "✅ Phase 2 脚本执行完毕！"
    echo "特征已定型并保存于: $OUTPUT_PATH"
    
) 2>&1 | tee "$LOG_FILE"
