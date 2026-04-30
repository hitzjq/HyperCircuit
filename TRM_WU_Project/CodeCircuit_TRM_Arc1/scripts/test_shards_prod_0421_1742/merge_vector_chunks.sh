#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"
NUM_VECTOR_CHUNKS="${NUM_VECTOR_CHUNKS:-64}"
SHARD_INDEX="${SHARD_INDEX:--1}"
FORCE_MERGE="${FORCE_MERGE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
LOG_FILE="$SHARDS_ROOT/merge_vector_chunks.log"

mkdir -p "$SHARDS_ROOT"

cmd=(
  python CodeCircuit_TRM_Arc1/src/merge_vector_chunks.py
  --run_name "$RUN_NAME"
  --num_shards "$NUM_SHARDS"
  --num_chunks "$NUM_VECTOR_CHUNKS"
  --shard_index "$SHARD_INDEX"
  --shards_root "$SHARDS_ROOT"
)

if [ "$FORCE_MERGE" = "1" ]; then
  cmd+=(--force)
fi

{
echo "=========================================================="
echo "  Merge vector chunks"
echo "  Run: $RUN_NAME"
echo "  Shards root: $SHARDS_ROOT"
echo "  Num shards: $NUM_SHARDS"
echo "  Num vector chunks: $NUM_VECTOR_CHUNKS"
echo "  Shard index: $SHARD_INDEX"
echo "  Force merge: $FORCE_MERGE"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

"${cmd[@]}"

echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
} 2>&1 | tee "$LOG_FILE"
