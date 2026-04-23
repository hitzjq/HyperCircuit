#!/bin/bash
set -euo pipefail

RUN_NAME="prod_0421_1742"
NUM_SHARDS="${NUM_SHARDS:-40}"
EXPECTED_TEST_QUERIES="${EXPECTED_TEST_QUERIES:-385815}"
ALLOW_PARTIAL_MERGE="${ALLOW_PARTIAL_MERGE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
LOG_FILE="$SHARDS_ROOT/merge_test_shards.log"

mkdir -p "$SHARDS_ROOT"

{
echo "=========================================================="
echo "  Merge test shards"
echo "  Run: $RUN_NAME"
echo "  Shards root: $SHARDS_ROOT"
echo "=========================================================="

RUN_DIR="$RUN_DIR" \
NUM_SHARDS="$NUM_SHARDS" \
EXPECTED_TEST_QUERIES="$EXPECTED_TEST_QUERIES" \
ALLOW_PARTIAL_MERGE="$ALLOW_PARTIAL_MERGE" \
python - <<'PY'
import os
import torch

run_dir = os.environ["RUN_DIR"]
num_shards = int(os.environ["NUM_SHARDS"])
expected = int(os.environ.get("EXPECTED_TEST_QUERIES", "0"))
allow_partial = os.environ.get("ALLOW_PARTIAL_MERGE", "0") == "1"

train_path = os.path.join(run_dir, "cc_advanced_features_train.pt")
test_path = os.path.join(run_dir, "cc_advanced_features_test.pt")
all_path = os.path.join(run_dir, "cc_advanced_features_all.pt")
canonical_path = os.path.join(run_dir, "cc_advanced_features.pt")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Missing train features: {train_path}")

entries = []
for shard_index in range(num_shards):
    shard_tag = f"shard_{shard_index:02d}_of_{num_shards:02d}"
    shard_path = os.path.join(
        run_dir,
        "test_shards",
        shard_tag,
        f"cc_advanced_features_test_{shard_tag}.pt",
    )
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing shard feature file: {shard_path}")

    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    features = shard["features"]
    mapping = shard["query_mapping"]
    if len(mapping) != features.shape[0]:
        raise ValueError(f"Feature/mapping length mismatch in {shard_path}")

    for row_index, meta in enumerate(mapping):
        graph_index = int(meta.get("graph_index", row_index))
        entries.append((graph_index, features[row_index], meta))

entries.sort(key=lambda item: item[0])

seen = set()
duplicates = []
for graph_index, _, _ in entries:
    if graph_index in seen:
        duplicates.append(graph_index)
    seen.add(graph_index)
if duplicates:
    raise ValueError(f"Duplicate graph_index values found, first duplicates: {duplicates[:10]}")

if expected and len(entries) != expected:
    message = f"Merged {len(entries)} test rows, expected {expected}"
    if allow_partial:
        print(f"WARNING: {message}")
    else:
        raise RuntimeError(message + ". Set ALLOW_PARTIAL_MERGE=1 to merge partial results.")

test_features = torch.stack([feature for _, feature, _ in entries], dim=0)
test_mapping = [meta for _, _, meta in entries]
test_data = {
    "features": test_features,
    "query_mapping": test_mapping,
    "feature_dim": test_features.shape[1],
    "n_queries": len(test_mapping),
}
torch.save(test_data, test_path)
print(f"Saved merged test features: {test_path}")
print(f"Test feature shape: {list(test_features.shape)}")

train = torch.load(train_path, map_location="cpu", weights_only=False)
merged = {
    "features": torch.cat([train["features"], test_features], dim=0),
    "query_mapping": list(train["query_mapping"]) + test_mapping,
}
merged["feature_dim"] = merged["features"].shape[1]
merged["n_queries"] = len(merged["query_mapping"])

torch.save(merged, all_path)
torch.save(merged, canonical_path)
print(f"Saved merged all features: {all_path}")
print(f"Updated canonical features: {canonical_path}")
print(f"All feature shape: {list(merged['features'].shape)}")
PY
} 2>&1 | tee "$LOG_FILE"
