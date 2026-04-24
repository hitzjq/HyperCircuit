#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"
SHARD_INDEX="${SHARD_INDEX:?Set SHARD_INDEX before launching this worker}"

DATASET_PATH="${DATASET_PATH:-/volume/safety/kbei/HyperCircuit_Data/data/arc1concept-aug-1000}"
CONFIG_PATH="${CONFIG_PATH:-config/cfg_wu4trm.yaml}"
CKPT_PATH="${CKPT_PATH:-/volume/safety/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071}"
MAX_QUERIES="${MAX_QUERIES:--1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

printf -v SHARD_TAG "shard_%02d_of_%02d" "$SHARD_INDEX" "$NUM_SHARDS"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARD_ROOT="$RUN_DIR/test_shards/$SHARD_TAG"
GRAPH_DIR="$SHARD_ROOT/attribution_graphs"
FEATURE_PATH="$SHARD_ROOT/cc_advanced_features_test_${SHARD_TAG}.pt"
LOG_DIR="$SHARD_ROOT/logs"
LOG_FILE="$LOG_DIR/run.log"

mkdir -p "$GRAPH_DIR" "$LOG_DIR"

for required in \
    "$RUN_DIR/checkpoints/sae_block_0.pt" \
    "$RUN_DIR/checkpoints/sae_block_1.pt" \
    "$RUN_DIR/cc_advanced_features_train.pt"; do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

{
START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
START_EPOCH="$(date +%s)"

echo "=========================================================="
echo "  Test attribution worker"
echo "  Run: $RUN_NAME"
echo "  Shard: $SHARD_INDEX/$NUM_SHARDS ($SHARD_TAG)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Started at: $START_TS"
echo "  Graph dir: $GRAPH_DIR"
echo "  Feature path: $FEATURE_PATH"
echo "=========================================================="

python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
    --run_name "$RUN_NAME" \
    --config_path "$CONFIG_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --dataset_paths "$DATASET_PATH" \
    --max_queries "$MAX_QUERIES" \
    --split test \
    --use_last_step \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$SHARD_INDEX" \
    --graph_dir "$GRAPH_DIR" \
    --skip_config_save

python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
    --run_name "$RUN_NAME" \
    --input_dir "$GRAPH_DIR" \
    --output_path "$FEATURE_PATH" \
    --skip_config_save

END_TS="$(date '+%Y-%m-%d %H:%M:%S')"
END_EPOCH="$(date +%s)"
ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))

echo "=========================================================="
echo "  Shard complete: $SHARD_INDEX/$NUM_SHARDS ($SHARD_TAG)"
echo "  Finished at: $END_TS"
echo "  Elapsed seconds: $ELAPSED_SECONDS"
echo "  Elapsed hours: $(python - <<PY
print(f"{int('$ELAPSED_SECONDS') / 3600:.2f}")
PY
)"
echo "  Output: $FEATURE_PATH"
echo "=========================================================="
} 2>&1 | tee "$LOG_FILE"
