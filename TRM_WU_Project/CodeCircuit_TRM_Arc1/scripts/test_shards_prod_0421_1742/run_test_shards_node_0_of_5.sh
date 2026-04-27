#!/bin/bash
set -euo pipefail

NODE_INDEX=0
NUM_NODES="${NUM_NODES:-5}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_SHARDS="${NUM_SHARDS:-40}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
NODE_LOG_DIR="$RUN_DIR/test_shards/node_${NODE_INDEX}_of_${NUM_NODES}/logs"
NODE_LOG="$NODE_LOG_DIR/node_launcher.log"
mkdir -p "$NODE_LOG_DIR"

{
echo "Launching node $NODE_INDEX/$NUM_NODES with $GPUS_PER_NODE GPU workers and $NUM_SHARDS total shards"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

pids=()
shard_ids=()
for gpu_id in $(seq 0 $((GPUS_PER_NODE - 1))); do
    shard_index=$((NODE_INDEX * GPUS_PER_NODE + gpu_id))
    if [ "$shard_index" -ge "$NUM_SHARDS" ]; then
        continue
    fi
    printf -v shard_tag "shard_%02d_of_%02d" "$shard_index" "$NUM_SHARDS"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    RUN_NAME="$RUN_NAME" \
    NUM_SHARDS="$NUM_SHARDS" \
    SHARD_INDEX="$shard_index" \
    bash "$SCRIPT_DIR/run_test_shard_worker.sh" > /dev/null 2>&1 &
    pids+=("$!")
    shard_ids+=("$shard_index")
    echo "GPU $gpu_id -> shard $shard_index/$NUM_SHARDS ($shard_tag), pid=${pids[-1]}"
done

failed=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    shard_index="${shard_ids[$idx]}"
    if ! wait "$pid"; then
        failed=1
        echo "Worker shard $shard_index/$NUM_SHARDS failed (pid=$pid)"
    else
        echo "Worker shard $shard_index/$NUM_SHARDS finished successfully (pid=$pid)"
    fi
done

echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
exit "$failed"
} 2>&1 | tee "$NODE_LOG"
