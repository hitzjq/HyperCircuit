#!/bin/bash
# ==========================================================
# 一键调试脚本 (A100)
# ==========================================================
# 用法: bash CodeCircuit_TRM_Arc1/scripts/run_debug_a100.sh
#
# 所有 4 步共享同一个 run_name，数据保存在:
#   CodeCircuit_TRM_Arc1/runs/<run_name>/
# ==========================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 生成带时间的 run_name ----
RUN_NAME="debug_$(date +%m%d_%H%M)"
DATASET_PATH="data/arc1concept-aug-1000"
CONFIG_PATH="config/cfg_wu4trm.yaml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  A100 Debug Pipeline"
echo "  Run: $RUN_NAME"
echo "  Dir: CodeCircuit_TRM_Arc1/runs/$RUN_NAME/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 日志
LOG_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"

{

# ====== Step 1: 收集激活 ======
echo ""
echo "━━━ Step 1/4: 收集激活 (2 batches) ━━━"
python CodeCircuit_TRM_Arc1/src/collect_activations.py \
    --run_name "$RUN_NAME" \
    --dataset_paths "$DATASET_PATH" \
    --max_batches 2

# ====== Step 2: 训练双 SAE ======
echo ""
echo "━━━ Step 2/4: 训练双 SAE (2 epochs) ━━━"
python CodeCircuit_TRM_Arc1/src/train_transcoder.py \
    --run_name "$RUN_NAME" \
    --epochs 2

# ====== Step 3: 提取归因图 ======
echo ""
echo "━━━ Step 3/4: 提取归因图 (2 queries) ━━━"
python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
    --run_name "$RUN_NAME" \
    --config_path "$CONFIG_PATH" \
    --dataset_paths "$DATASET_PATH" \
    --max_queries 2 \
    --split train \
    --use_last_step

# ====== Step 4: 转向量 ======
echo ""
echo "━━━ Step 4/4: 归因图 → 53维向量 ━━━"
python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
    --run_name "$RUN_NAME"

# ====== 验证 ======
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证输出"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RUN_DIR="CodeCircuit_TRM_Arc1/runs/$RUN_NAME"

python -c "
import torch, os

run_dir = '$RUN_DIR'

# Check activations
b0 = len(os.listdir(os.path.join(run_dir, 'activations/block_0'))) if os.path.isdir(os.path.join(run_dir, 'activations/block_0')) else 0
b1 = len(os.listdir(os.path.join(run_dir, 'activations/block_1'))) if os.path.isdir(os.path.join(run_dir, 'activations/block_1')) else 0
print(f'  [Step 1] block_0: {b0} files, block_1: {b1} files', '✅' if b0 > 0 and b1 > 0 else '❌')

# Check SAEs
s0 = os.path.exists(os.path.join(run_dir, 'checkpoints/sae_block_0.pt'))
s1 = os.path.exists(os.path.join(run_dir, 'checkpoints/sae_block_1.pt'))
print(f'  [Step 2] SAE_0: {s0}, SAE_1: {s1}', '✅' if s0 and s1 else '❌')

# Check graphs
graph_dir = os.path.join(run_dir, 'attribution_graphs')
n_g = len([f for f in os.listdir(graph_dir) if f.endswith('.pt')]) if os.path.isdir(graph_dir) else 0
print(f'  [Step 3] {n_g} attribution graphs', '✅' if n_g > 0 else '❌')
if n_g > 0:
    d = torch.load(os.path.join(graph_dir, 'graph_000000.pt'), map_location='cpu', weights_only=False)
    print(f'           adj={list(d[\"adjacency_matrix\"].shape)}, feats={d[\"n_selected_features\"]}, errors={d[\"n_error_nodes\"]}, tokens={d[\"n_token_nodes\"]}, logits={d[\"n_logit_nodes\"]}')
    print(f'           has query_meta: {\"query_meta\" in d}')

# Check features
feat_path = os.path.join(run_dir, 'cc_advanced_features.pt')
if os.path.exists(feat_path):
    data = torch.load(feat_path, map_location='cpu', weights_only=False)
    print(f'  [Step 4] features shape: {list(data[\"features\"].shape)}, query_mapping: {data[\"n_queries\"]} entries ✅')
    print(f'           非零率: {(data[\"features\"].abs() > 1e-6).float().mean().item()*100:.1f}%')
else:
    print(f'  [Step 4] features not found ❌')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ 完成! 所有数据在: CodeCircuit_TRM_Arc1/runs/$RUN_NAME/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

} 2>&1 | tee "$LOG_FILE"
