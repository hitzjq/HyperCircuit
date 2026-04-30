#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"
NUM_VECTOR_CHUNKS="${NUM_VECTOR_CHUNKS:-64}"
SMOKE_SHARD_INDEX="${SMOKE_SHARD_INDEX:-0}"
SMOKE_CHUNK_INDEX="${SMOKE_CHUNK_INDEX:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
LOG_FILE="$SHARDS_ROOT/smoke_vector_chunks.log"

printf -v SHARD_TAG "shard_%02d_of_%02d" "$SMOKE_SHARD_INDEX" "$NUM_SHARDS"
printf -v CHUNK_TAG "%03d" "$SMOKE_CHUNK_INDEX"
printf -v CHUNKS_TAG "%03d" "$NUM_VECTOR_CHUNKS"

SHARD_ROOT="$SHARDS_ROOT/$SHARD_TAG"
GRAPH_DIR="$SHARD_ROOT/attribution_graphs"
CHUNK_ROOT="$SHARD_ROOT/vector_chunks"
CHUNK_PATH="$CHUNK_ROOT/chunk_${CHUNK_TAG}_of_${CHUNKS_TAG}.pt"

mkdir -p "$CHUNK_ROOT" "$SHARDS_ROOT"

{
echo "=========================================================="
echo "  Smoke test vector chunks"
echo "  Run: $RUN_NAME"
echo "  Shard: $SMOKE_SHARD_INDEX/$NUM_SHARDS ($SHARD_TAG)"
echo "  Chunk: $SMOKE_CHUNK_INDEX/$NUM_VECTOR_CHUNKS"
echo "  Graph dir: $GRAPH_DIR"
echo "  Output: $CHUNK_PATH"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

if [ ! -d "$GRAPH_DIR" ]; then
  echo "ERROR: Missing graph dir: $GRAPH_DIR" >&2
  exit 1
fi

graph_count="$(find "$GRAPH_DIR" -maxdepth 1 -type f -name 'graph_*.pt' | wc -l | tr -d ' ')"
echo "Graph count: $graph_count"
if [ "$graph_count" -eq 0 ]; then
  echo "ERROR: No graph files found in $GRAPH_DIR" >&2
  exit 1
fi

python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
  --run_name "$RUN_NAME" \
  --input_dir "$GRAPH_DIR" \
  --output_path "$CHUNK_PATH" \
  --num_chunks "$NUM_VECTOR_CHUNKS" \
  --chunk_index "$SMOKE_CHUNK_INDEX" \
  --skip_config_save

python - <<PY
import torch

path = "$CHUNK_PATH"
data = torch.load(path, map_location="cpu", weights_only=False)
features = data["features"]
n_queries = data["n_queries"]
feature_dim = data["feature_dim"]
query_mapping = data["query_mapping"]

print("==========================================================")
print("  Smoke output check")
print(f"  path: {path}")
print(f"  features_shape: {tuple(features.shape)}")
print(f"  n_queries: {n_queries}")
print(f"  feature_dim: {feature_dim}")
print(f"  chunk: {data.get('chunk_index')} / {data.get('num_chunks')}")
print(f"  source_n_graphs: {data.get('source_n_graphs')}")
print(f"  first_graph: {query_mapping[0]['graph_file'] if query_mapping else 'none'}")
print(f"  last_graph: {query_mapping[-1]['graph_file'] if query_mapping else 'none'}")

assert features.ndim == 2, features.shape
assert features.shape[0] == n_queries == len(query_mapping)
assert features.shape[1] == feature_dim == 53
assert data.get("chunk_index") == $SMOKE_CHUNK_INDEX
assert data.get("num_chunks") == $NUM_VECTOR_CHUNKS
assert data.get("source_n_graphs") == $graph_count
print("  result: PASS")
print("==========================================================")
PY

echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Smoke test passed."
echo "=========================================================="
} 2>&1 | tee "$LOG_FILE"

echo
echo "Smoke log written to: $LOG_FILE"
