#!/bin/bash
# ==========================================================
# H200 full production pipeline
# Usage: bash CodeCircuit_TRM_Arc1/scripts/run_pipeline_prod.sh
# Output:
#   - cc_advanced_features_train.pt
#   - cc_advanced_features_test.pt
#   - cc_advanced_features_all.pt
#   - cc_advanced_features.pt      (canonical merged "all" file)
# ==========================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- Config ----
RUN_NAME="prod_$(date +%m%d_%H%M)"
DATASET_PATH="/mnt/kbei/HyperCircuit_Data/data/arc1concept-aug-1000"
CONFIG_PATH="config/cfg_wu4trm.yaml"
CKPT_PATH="/mnt/kbei/HyperCircuit_Data/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071"
COLLECT_MAX_BATCHES="${COLLECT_MAX_BATCHES:--1}"
SAE_EPOCHS="${SAE_EPOCHS:-10}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-4096}"
SAE_D_SAE="${SAE_D_SAE:-4096}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"
TRAIN_MAX_QUERIES="${TRAIN_MAX_QUERIES:--1}"
TEST_MAX_QUERIES="${TEST_MAX_QUERIES:--1}"
RUN_TEST_SPLIT="${RUN_TEST_SPLIT:-1}"
STOP_AFTER_SAE="${STOP_AFTER_SAE:-0}"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
GRAPH_DIR="$RUN_DIR/attribution_graphs"
FEATURES_PATH="$RUN_DIR/cc_advanced_features.pt"
TRAIN_FEATURES_PATH="$RUN_DIR/cc_advanced_features_train.pt"
TEST_FEATURES_PATH="$RUN_DIR/cc_advanced_features_test.pt"
ALL_FEATURES_PATH="$RUN_DIR/cc_advanced_features_all.pt"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"

run_extract_and_vectorize() {
    local split="$1"
    local output_copy="$2"
    local max_queries="$3"

    echo "=========================================================="
    echo "Extracting attribution graphs for split: $split"
    echo "=========================================================="

    # The graph filenames are reused on each extraction pass.
    # Clear the previous pass first so graph_to_vector only sees the current split.
    rm -f "$GRAPH_DIR"/*.pt

    python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
        --run_name "$RUN_NAME" \
        --config_path "$CONFIG_PATH" \
        --ckpt_path "$CKPT_PATH" \
        --dataset_paths "$DATASET_PATH" \
        --max_queries "$max_queries" \
        --split "$split" \
        --use_last_step

    echo "=========================================================="
    echo "Converting graphs to feature vectors for split: $split"
    echo "=========================================================="

    python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
        --run_name "$RUN_NAME"

    cp "$FEATURES_PATH" "$output_copy"
    echo "Saved split feature file: $output_copy"
}

{
echo "=========================================================="
echo "  H200 Production Pipeline"
echo "  Run: $RUN_NAME"
echo "  Config: collect_max_batches=$COLLECT_MAX_BATCHES, sae_epochs=$SAE_EPOCHS, sae_batch_size=$SAE_BATCH_SIZE, sae_d_sae=$SAE_D_SAE, train_max_queries=$TRAIN_MAX_QUERIES, test_max_queries=$TEST_MAX_QUERIES"
echo "=========================================================="

echo "Step 1/4: collect activations"
python CodeCircuit_TRM_Arc1/src/collect_activations.py \
    --run_name "$RUN_NAME" \
    --dataset_paths "$DATASET_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --max_batches "$COLLECT_MAX_BATCHES"

echo "Step 2/4: train SAE"
TRAIN_TRANSCODER_ARGS=(
    --run_name "$RUN_NAME"
    --epochs "$SAE_EPOCHS"
    --batch_size "$SAE_BATCH_SIZE"
    --d_sae "$SAE_D_SAE"
)
if [ "$SAVE_EVERY_EPOCH" = "1" ]; then
    TRAIN_TRANSCODER_ARGS+=(--save_every_epoch)
fi
python CodeCircuit_TRM_Arc1/src/train_transcoder.py "${TRAIN_TRANSCODER_ARGS[@]}"

if [ "$STOP_AFTER_SAE" = "1" ]; then
    echo "Stopping after Step 2 because STOP_AFTER_SAE=1"
    exit 0
fi

echo "Step 3/4: extract + vectorize train split"
run_extract_and_vectorize "train" "$TRAIN_FEATURES_PATH" "$TRAIN_MAX_QUERIES"

if [ "$RUN_TEST_SPLIT" = "1" ]; then
echo "Step 4/4: extract + vectorize test split"
run_extract_and_vectorize "test" "$TEST_FEATURES_PATH" "$TEST_MAX_QUERIES"

echo "Merging train + test features into all"
FEATURES_TRAIN="$TRAIN_FEATURES_PATH" \
FEATURES_TEST="$TEST_FEATURES_PATH" \
FEATURES_ALL="$ALL_FEATURES_PATH" \
FEATURES_CANONICAL="$FEATURES_PATH" \
python - <<'PY'
import os
import torch

train_path = os.environ["FEATURES_TRAIN"]
test_path = os.environ["FEATURES_TEST"]
all_path = os.environ["FEATURES_ALL"]
canonical_path = os.environ["FEATURES_CANONICAL"]

train = torch.load(train_path, map_location="cpu", weights_only=False)
test = torch.load(test_path, map_location="cpu", weights_only=False)

merged = {
    "features": torch.cat([train["features"], test["features"]], dim=0),
    "query_mapping": list(train["query_mapping"]) + list(test["query_mapping"]),
}
merged["feature_dim"] = merged["features"].shape[1]
merged["n_queries"] = len(merged["query_mapping"])

torch.save(merged, all_path)
torch.save(merged, canonical_path)

print(f"Merged feature tensor shape: {list(merged['features'].shape)}")
print(f"Saved merged all file: {all_path}")
print(f"Updated canonical file: {canonical_path}")
PY
else
echo "Skipping test split because RUN_TEST_SPLIT=$RUN_TEST_SPLIT"
cp "$TRAIN_FEATURES_PATH" "$ALL_FEATURES_PATH"
cp "$TRAIN_FEATURES_PATH" "$FEATURES_PATH"
echo "Using train features as temporary all/main outputs"
fi

echo "Done. Outputs are under: $RUN_DIR"
echo "  train: $TRAIN_FEATURES_PATH"
echo "  test : $TEST_FEATURES_PATH"
echo "  all  : $ALL_FEATURES_PATH"
echo "  main : $FEATURES_PATH"

} 2>&1 | tee "$LOG_FILE"
