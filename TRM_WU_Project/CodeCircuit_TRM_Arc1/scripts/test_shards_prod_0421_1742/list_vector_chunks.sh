#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
NUM_SHARDS="${NUM_SHARDS:-40}"
NUM_VECTOR_CHUNKS="${NUM_VECTOR_CHUNKS:-64}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
LOG_FILE="$SHARDS_ROOT/list_vector_chunks.log"

mkdir -p "$SHARDS_ROOT"

{
echo "=========================================================="
echo "  List vector chunks"
echo "  Run: $RUN_NAME"
echo "  Shards root: $SHARDS_ROOT"
echo "  Num shards: $NUM_SHARDS"
echo "  Num vector chunks: $NUM_VECTOR_CHUNKS"
echo "  Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

total_done=0
total_expected=$((NUM_SHARDS * NUM_VECTOR_CHUNKS))

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  printf -v shard_tag "shard_%02d_of_%02d" "$shard_index" "$NUM_SHARDS"
  printf -v chunks_tag "%03d" "$NUM_VECTOR_CHUNKS"

  shard_root="$SHARDS_ROOT/$shard_tag"
  chunk_root="$shard_root/vector_chunks"
  feature_path="$shard_root/feat.pt"
  legacy_feature_path="$shard_root/cc_advanced_features_test_${shard_tag}.pt"

  done_count=0
  missing=()
  for chunk_index in $(seq 0 $((NUM_VECTOR_CHUNKS - 1))); do
    printf -v chunk_tag "%03d" "$chunk_index"
    chunk_path="$chunk_root/chunk_${chunk_tag}_of_${chunks_tag}.pt"
    if [ -f "$chunk_path" ]; then
      done_count=$((done_count + 1))
    else
      missing+=("$chunk_tag")
    fi
  done
  total_done=$((total_done + done_count))

  echo
  echo "----------------------------------------------------------"
  echo "$shard_tag"
  echo "----------------------------------------------------------"
  echo "chunks_done=$done_count/$NUM_VECTOR_CHUNKS"
  if [ -f "$feature_path" ]; then
    echo "feature_file=$feature_path"
    ls -lh "$feature_path"
  elif [ -f "$legacy_feature_path" ]; then
    echo "legacy_feature_file=$legacy_feature_path"
    ls -lh "$legacy_feature_path"
  else
    echo "feature_file=MISSING"
  fi

  if [ -d "$chunk_root" ]; then
    echo "chunk_dir=$chunk_root"
    echo "chunk_dir_size=$(du -sh "$chunk_root" 2>/dev/null | awk '{print $1}')"
  else
    echo "chunk_dir=MISSING"
  fi

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "missing_chunks=${missing[*]}"
  else
    echo "missing_chunks=none"
  fi
done

echo
echo "=========================================================="
echo "Overall chunks done: $total_done/$total_expected"
echo "=========================================================="
} 2>&1 | tee "$LOG_FILE"
