#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
LOG_FILE="$SHARDS_ROOT/list_test_shards.log"

mkdir -p "$SHARDS_ROOT"

{
echo "=========================================================="
echo "  List test shard directories"
echo "  Run: $RUN_NAME"
echo "  Shards root: $SHARDS_ROOT"
echo "  Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  printf -v shard_tag "shard_%02d_of_%02d" "$shard_index" "$NUM_SHARDS"
  shard_dir="$SHARDS_ROOT/$shard_tag"
  graph_dir="$shard_dir/attribution_graphs"
  feature_path="$shard_dir/feat.pt"
  legacy_feature_path="$shard_dir/cc_advanced_features_test_${shard_tag}.pt"

  echo
  echo "----------------------------------------------------------"
  echo "$shard_tag"
  echo "----------------------------------------------------------"

  if [ ! -d "$shard_dir" ]; then
    echo "MISSING DIRECTORY: $shard_dir"
    continue
  fi

  ls -lah "$shard_dir"

  if [ -d "$graph_dir" ]; then
    graph_count="$(find "$graph_dir" -maxdepth 1 -type f -name 'graph_*.pt' | wc -l | tr -d ' ')"
    echo "graph_count=$graph_count"
  else
    echo "graph_count=0 (no attribution_graphs dir)"
  fi

  if [ -f "$feature_path" ]; then
    echo "feature_file=$feature_path"
    ls -lah "$feature_path"
  elif [ -f "$legacy_feature_path" ]; then
    echo "legacy_feature_file=$legacy_feature_path"
    ls -lah "$legacy_feature_path"
  else
    echo "feature_file=MISSING"
  fi

  if [ -d "$shard_dir/logs" ]; then
    echo "logs_dir_contents:"
    ls -lah "$shard_dir/logs"
  fi
done
} 2>&1 | tee "$LOG_FILE"
