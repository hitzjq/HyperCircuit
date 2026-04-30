#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
OUTPUT_LOG="$SHARDS_ROOT/combined_test_shard_run_logs.log"

mkdir -p "$SHARDS_ROOT"

{
echo "=========================================================="
echo "  Combined test shard run logs"
echo "  Run: $RUN_NAME"
echo "  Shards root: $SHARDS_ROOT"
echo "  Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUTPUT_LOG"
echo "=========================================================="

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  printf -v shard_tag "shard_%02d_of_%02d" "$shard_index" "$NUM_SHARDS"
  shard_dir="$SHARDS_ROOT/$shard_tag"
  log_file="$shard_dir/logs/run.log"

  echo
  echo "##########################################################"
  echo "# $shard_tag"
  echo "# Source: $log_file"
  echo "##########################################################"
  echo

  if [ -f "$log_file" ]; then
    cat "$log_file"
    echo
    echo "#################### END $shard_tag ####################"
  else
    echo "MISSING LOG: $log_file"
  fi
done
} 2>&1 | tee "$OUTPUT_LOG"

echo
echo "Combined log written to: $OUTPUT_LOG"
