#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"
NUM_NODES="${NUM_NODES:-1}"
NODE_INDEX="${NODE_INDEX:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
NODE_ROOT="$SHARDS_ROOT/cpu_vectorize_node_${NODE_INDEX}_of_${NUM_NODES}"
NODE_LOG_DIR="$NODE_ROOT/logs"
NODE_LOG="$NODE_LOG_DIR/node.log"

mkdir -p "$NODE_LOG_DIR"

assigned_shards=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  if [ $((shard_index % NUM_NODES)) -eq "$NODE_INDEX" ]; then
    assigned_shards+=("$shard_index")
  fi
done

vectorize_shard() {
  local shard_index="$1"
  local shard_tag
  local shard_root
  local graph_dir
  local feature_path
  local legacy_feature_path
  local log_dir
  local log_file
  local graph_count

  printf -v shard_tag "shard_%02d_of_%02d" "$shard_index" "$NUM_SHARDS"
  shard_root="$SHARDS_ROOT/$shard_tag"
  graph_dir="$shard_root/attribution_graphs"
  feature_path="$shard_root/feat.pt"
  legacy_feature_path="$shard_root/cc_advanced_features_test_${shard_tag}.pt"
  log_dir="$shard_root/logs"
  log_file="$log_dir/vectorize_cpu.log"

  mkdir -p "$log_dir"

  {
  echo "=========================================================="
  echo "  CPU vectorize shard"
  echo "  Run: $RUN_NAME"
  echo "  Shard: $shard_index/$NUM_SHARDS ($shard_tag)"
  echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  Graph dir: $graph_dir"
  echo "  Feature path: $feature_path"
  echo "=========================================================="

  if [ ! -f "$feature_path" ] && [ -f "$legacy_feature_path" ]; then
    echo "Found legacy feature file, migrating to short name."
    mv "$legacy_feature_path" "$feature_path"
  fi

  if [ -f "$feature_path" ]; then
    echo "Feature already exists, skipping."
    exit 0
  fi

  if [ ! -d "$graph_dir" ]; then
    echo "Missing graph dir: $graph_dir" >&2
    exit 1
  fi

  graph_count="$(find "$graph_dir" -maxdepth 1 -type f -name 'graph_*.pt' | wc -l | tr -d ' ')"
  echo "Graph count: $graph_count"
  if [ "$graph_count" -eq 0 ]; then
    echo "No graph files found in $graph_dir" >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="" python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
    --run_name "$RUN_NAME" \
    --input_dir "$graph_dir" \
    --output_path "$feature_path" \
    --skip_config_save

  echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Output: $feature_path"
  echo "=========================================================="
  } 2>&1 | tee "$log_file"
}

{
echo "=========================================================="
echo "  CPU vectorize node"
echo "  Run: $RUN_NAME"
echo "  Node: $NODE_INDEX/$NUM_NODES"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Assigned shards: ${assigned_shards[*]}"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

batch=()
failed=0
for shard_index in "${assigned_shards[@]}"; do
  vectorize_shard "$shard_index" &
  batch+=("$!")

  if [ "${#batch[@]}" -ge "$MAX_PARALLEL" ]; then
    for pid in "${batch[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    batch=()
  fi
done

for pid in "${batch[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
exit "$failed"
} 2>&1 | tee "$NODE_LOG"
