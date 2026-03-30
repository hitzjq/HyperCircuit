#!/bin/bash
# =====================================================
# 批量评估 pretrain checkpoints 的 ARC 分数
# =====================================================
# 需要改：
# 1. 下面的CHECKPOINT_DIR，设置成之前pretrain出来的那些ckpt的根路径
# 2. 下面的DATA_PATH，设置成数据集路径
# =====================================================

# ============ 在这里修改参数 ============
CHECKPOINT_DIR="/path/to/checkpoints/ARC-AGI-1"     # 存放 step_xxxx 的目录（绝对路径）
DATA_PATH="/path/to/data/arc1concept-aug-1000"       # 评估数据集路径（绝对路径）
NUM_GPUS=8                                           # GPU 数量
BATCH_SIZE=768                                       # 评估 batch size
L_CYCLES=4                                           # L_cycles（ARC-AGI-1 用 4）
# =======================================

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 日志保存在 ./logs 文件夹下 (相对于 TinyRecursiveModels)
mkdir -p "logs"
LOG_FILE="logs/eval_$(date +%m%d_%H%M).log"

echo "📊 Checkpoint Dir : ${CHECKPOINT_DIR}"
echo "📂 Data Path      : ${DATA_PATH}"
echo "🖥  GPUs           : ${NUM_GPUS}"
echo "📝 Log File        : ${LOG_FILE}"
echo ""

cd "${PROJECT_ROOT}"

DISABLE_COMPILE=1 torchrun \
    --nproc-per-node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    scripts/eval/eval_checkpoints.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --data_path "${DATA_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --L_cycles ${L_CYCLES} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✅ Done! Results: ${CHECKPOINT_DIR}/eval_results.json"
echo "📝 Log: ${LOG_FILE}"
