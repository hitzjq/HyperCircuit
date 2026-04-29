#!/bin/bash
set -u

RUN_NAME="${RUN_NAME:-prod_0421_1742}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"
SHARDS_ROOT="$RUN_DIR/test_shards"
LOG_FILE="$SHARDS_ROOT/diagnose_merge_paths.log"

mkdir -p "$SHARDS_ROOT"

{
echo "=========================================================="
echo "  Diagnose merge paths"
echo "  Run: $RUN_NAME"
echo "  Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
echo

echo "[pwd]"
pwd
echo

echo "[project root listing]"
ls -ld CodeCircuit_TRM_Arc1 Codecircuit_TRM_Arc1 2>&1
echo

echo "[merge script path]"
ls -lah CodeCircuit_TRM_Arc1/scripts/test_shards_prod_0421_1742/merge_test_shards.sh 2>&1
echo

echo "[merge script key lines]"
rg -n "Codecircuit|CodeCircuit_TRM_Arc1|RUN_DIR|SHARDS_ROOT|feat\\.pt|regenerating from|Diagnostics for" \
  CodeCircuit_TRM_Arc1/scripts/test_shards_prod_0421_1742/merge_test_shards.sh 2>&1
echo

echo "[expected run dirs]"
ls -ld \
  CodeCircuit_TRM_Arc1/runs \
  CodeCircuit_TRM_Arc1/runs/"$RUN_NAME" \
  CodeCircuit_TRM_Arc1/runs/"$RUN_NAME"/test_shards 2>&1
echo

echo "[unexpected case-variant run dirs]"
ls -ld \
  Codecircuit_TRM_Arc1 \
  Codecircuit_TRM_Arc1/runs \
  Codecircuit_TRM_Arc1/runs/"$RUN_NAME" \
  Codecircuit_TRM_Arc1/runs/"$RUN_NAME"/test_shards 2>&1
echo

echo "[test_shards top-level sample]"
find "$SHARDS_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | head -n 60 2>&1
echo

echo "[shard_00 expected contents]"
ls -lah "$SHARDS_ROOT/shard_00_of_40" 2>&1
echo

echo "[shard_00 graph count]"
if [ -d "$SHARDS_ROOT/shard_00_of_40/attribution_graphs" ]; then
  find "$SHARDS_ROOT/shard_00_of_40/attribution_graphs" -maxdepth 1 -type f -name 'graph_*.pt' | wc -l
else
  echo "missing attribution_graphs directory"
fi
echo

echo "[shard_00 graph sample]"
find "$SHARDS_ROOT/shard_00_of_40/attribution_graphs" -maxdepth 1 -type f -name 'graph_*.pt' | sort | head -n 10 2>&1
echo

echo "[list_test_shards.log head]"
head -n 80 "$SHARDS_ROOT/list_test_shards.log" 2>&1
echo

echo "[merge_test_shards.log tail]"
tail -n 160 "$SHARDS_ROOT/merge_test_shards.log" 2>&1
echo

echo "[done]"
echo "Wrote diagnostics to: $LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
