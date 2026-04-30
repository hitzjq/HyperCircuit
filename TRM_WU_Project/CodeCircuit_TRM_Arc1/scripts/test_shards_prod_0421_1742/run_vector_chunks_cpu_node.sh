#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"
NUM_VECTOR_CHUNKS="${NUM_VECTOR_CHUNKS:-64}"
NUM_NODES="${NUM_NODES:-1}"
NODE_INDEX="${NODE_INDEX:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-1}"
GPU_KEEPALIVE_MATRIX_SIZE="${GPU_KEEPALIVE_MATRIX_SIZE:-4096}"
GPU_KEEPALIVE_STEPS="${GPU_KEEPALIVE_STEPS:-4}"
GPU_KEEPALIVE_SLEEP="${GPU_KEEPALIVE_SLEEP:-1.0}"
GPU_KEEPALIVE_LOG_INTERVAL="${GPU_KEEPALIVE_LOG_INTERVAL:-60}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
NODE_ROOT="$SHARDS_ROOT/vector_chunk_nodes/node_${NODE_INDEX}_of_${NUM_NODES}"
NODE_LOG_DIR="$NODE_ROOT/logs"
NODE_LOG="$NODE_LOG_DIR/node.log"
TOTAL_TASKS=$((NUM_SHARDS * NUM_VECTOR_CHUNKS))

mkdir -p "$NODE_LOG_DIR"

keepalive_pids=()

start_gpu_keepalive() {
  local gpu_id
  local log_file

  if [ "$GPU_KEEPALIVE" != "1" ]; then
    echo "GPU keepalive disabled."
    return 0
  fi

  echo "Starting GPU keepalive on $GPUS_PER_NODE visible GPUs."
  for gpu_id in $(seq 0 $((GPUS_PER_NODE - 1))); do
    log_file="$NODE_LOG_DIR/gpu_keepalive_${gpu_id}.log"
    CUDA_VISIBLE_DEVICES="$gpu_id" python CodeCircuit_TRM_Arc1/src/gpu_keepalive.py \
      --matrix-size "$GPU_KEEPALIVE_MATRIX_SIZE" \
      --steps "$GPU_KEEPALIVE_STEPS" \
      --sleep "$GPU_KEEPALIVE_SLEEP" \
      --log-interval "$GPU_KEEPALIVE_LOG_INTERVAL" \
      > "$log_file" 2>&1 &
    keepalive_pids+=("$!")
    echo "GPU keepalive gpu=$gpu_id pid=${keepalive_pids[-1]} log=$log_file"
  done
}

stop_gpu_keepalive() {
  if [ "${#keepalive_pids[@]}" -eq 0 ]; then
    return 0
  fi

  echo "Stopping GPU keepalive processes: ${keepalive_pids[*]}"
  kill "${keepalive_pids[@]}" 2>/dev/null || true
  wait "${keepalive_pids[@]}" 2>/dev/null || true
  keepalive_pids=()
}

run_chunk_task() {
  local task_id="$1"
  local shard_index=$((task_id / NUM_VECTOR_CHUNKS))
  local chunk_index=$((task_id % NUM_VECTOR_CHUNKS))
  local shard_tag
  local chunk_tag
  local chunks_tag
  local shard_root
  local graph_dir
  local feature_path
  local legacy_feature_path
  local chunk_root
  local chunk_path
  local log_dir
  local log_file
  local status

  printf -v shard_tag "shard_%02d_of_%02d" "$shard_index" "$NUM_SHARDS"
  printf -v chunk_tag "%03d" "$chunk_index"
  printf -v chunks_tag "%03d" "$NUM_VECTOR_CHUNKS"

  shard_root="$SHARDS_ROOT/$shard_tag"
  graph_dir="$shard_root/attribution_graphs"
  feature_path="$shard_root/feat.pt"
  legacy_feature_path="$shard_root/cc_advanced_features_test_${shard_tag}.pt"
  chunk_root="$shard_root/vector_chunks"
  chunk_path="$chunk_root/chunk_${chunk_tag}_of_${chunks_tag}.pt"
  log_dir="$chunk_root/logs"
  log_file="$log_dir/chunk_${chunk_tag}_of_${chunks_tag}.log"

  mkdir -p "$log_dir"
  echo "START task=$task_id shard=$shard_tag chunk=$chunk_index/$NUM_VECTOR_CHUNKS log=$log_file"

  set +e
  {
  echo "=========================================================="
  echo "  CPU vectorize chunk"
  echo "  Run: $RUN_NAME"
  echo "  Shard: $shard_index/$NUM_SHARDS ($shard_tag)"
  echo "  Chunk: $chunk_index/$NUM_VECTOR_CHUNKS"
  echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  Graph dir: $graph_dir"
  echo "  Chunk path: $chunk_path"
  echo "  Feature path: $feature_path"
  echo "=========================================================="

  if [ -f "$feature_path" ]; then
    echo "Shard feature already exists, skipping chunk."
    exit 0
  fi

  if [ -f "$legacy_feature_path" ]; then
    echo "Legacy shard feature already exists, skipping chunk."
    echo "Legacy path: $legacy_feature_path"
    exit 0
  fi

  if [ -f "$chunk_path" ]; then
    echo "Chunk already exists, skipping."
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
    --output_path "$chunk_path" \
    --num_chunks "$NUM_VECTOR_CHUNKS" \
    --chunk_index "$chunk_index" \
    --skip_config_save

  echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Output: $chunk_path"
  echo "=========================================================="
  } > "$log_file" 2>&1
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    echo "DONE task=$task_id shard=$shard_tag chunk=$chunk_index/$NUM_VECTOR_CHUNKS"
  else
    echo "FAIL task=$task_id shard=$shard_tag chunk=$chunk_index/$NUM_VECTOR_CHUNKS status=$status"
  fi
  return "$status"
}

{
echo "=========================================================="
echo "  CPU vectorize chunk node"
echo "  Run: $RUN_NAME"
echo "  Node: $NODE_INDEX/$NUM_NODES"
echo "  Shards: $NUM_SHARDS"
echo "  Chunks per shard: $NUM_VECTOR_CHUNKS"
echo "  Total chunk tasks: $TOTAL_TASKS"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Thread env: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
echo "  GPU keepalive: enabled=$GPU_KEEPALIVE gpus=$GPUS_PER_NODE matrix_size=$GPU_KEEPALIVE_MATRIX_SIZE steps=$GPU_KEEPALIVE_STEPS sleep=$GPU_KEEPALIVE_SLEEP"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

trap stop_gpu_keepalive EXIT INT TERM
start_gpu_keepalive

batch=()
batch_tasks=()
failed=0
assigned=0

for task_id in $(seq 0 $((TOTAL_TASKS - 1))); do
  if [ $((task_id % NUM_NODES)) -ne "$NODE_INDEX" ]; then
    continue
  fi

  assigned=$((assigned + 1))
  run_chunk_task "$task_id" &
  batch+=("$!")
  batch_tasks+=("$task_id")

  if [ "${#batch[@]}" -ge "$MAX_PARALLEL" ]; then
    for idx in "${!batch[@]}"; do
      if ! wait "${batch[$idx]}"; then
        failed=1
        echo "Task ${batch_tasks[$idx]} failed."
      fi
    done
    batch=()
    batch_tasks=()
  fi
done

for idx in "${!batch[@]}"; do
  if ! wait "${batch[$idx]}"; then
    failed=1
    echo "Task ${batch_tasks[$idx]} failed."
  fi
done

echo "Assigned tasks: $assigned"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
exit "$failed"
} 2>&1 | tee "$NODE_LOG"
