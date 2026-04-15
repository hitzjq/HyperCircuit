#!/bin/bash
# ==========================================================
# H200 全量生产脚本
# ==========================================================
# 用法: bash CodeCircuit_TRM_Arc1/scripts/run_pipeline_prod.sh
# ==========================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 配置 ----
RUN_NAME="prod_$(date +%m%d_%H%M)"
DATASET_PATH="/mnt/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
CONFIG_PATH="config/cfg_wu4trm.yaml"
CKPT_PATH="/mnt/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  H200 Production Pipeline"
echo "  Run: $RUN_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LOG_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"

{

echo "━━━ Step 1/4: 收集激活 (全量) ━━━"
python CodeCircuit_TRM_Arc1/src/collect_activations.py \
    --run_name "$RUN_NAME" \
    --dataset_paths "$DATASET_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --max_batches -1

echo "━━━ Step 2/4: 训练双 SAE ━━━"
python CodeCircuit_TRM_Arc1/src/train_transcoder.py \
    --run_name "$RUN_NAME" \
    --epochs 10

echo "━━━ Step 3/4: 提取归因图 (全量) ━━━"
python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
    --run_name "$RUN_NAME" \
    --config_path "$CONFIG_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --dataset_paths "$DATASET_PATH" \
    --max_queries -1 \
    --split test \
    --use_last_step

echo "━━━ Step 4/4: 归因图 → 53维向量 ━━━"
python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
    --run_name "$RUN_NAME"

echo "✅ 完成! 所有数据在: CodeCircuit_TRM_Arc1/runs/$RUN_NAME/"

} 2>&1 | tee "$LOG_FILE"
