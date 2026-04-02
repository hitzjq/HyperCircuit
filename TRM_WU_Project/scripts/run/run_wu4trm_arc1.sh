#!/bin/bash
set -x

export OMP_NUM_THREADS=4

# ==========================================
# TRM_WU_Project 双趟前向训练一键启动脚本
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 配置执行参数 ----
SEQ_LEN=1024
# ---- 硬件与规模配置 ----
NUM_GPUS=8           # 在这里填写你想用多少张显卡 (例如改成 4 或 8)
GLOBAL_BATCH_SIZE=256  # 这是**全局总 Batch Size**，它会自动平分给你设定的 NUM_GPUS
LR=1e-4

# ---- 日志与权重保存路径配置 ----
run_name="WU4TRM_r16_lr1e-4_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
CKPT_DIR="checkpoints/${run_name}"
mkdir -p "${CKPT_DIR}"
mkdir -p "logs"

# 1. Base TRM checkpoint
BASE_CKPT_PATH="pretrained_base_ckpt/ARC-AGI-1/step_155718" 

# 2. Dataset paths
ARCH="trm"
CFG="cfg_wu4trm" 

# ========================================
# ⚙️ 运行模式切换 (切换调试/正式运行)
# ========================================
# True  = 调试模式：不作评测，秒开训练，飞速看日志和 loss (低配机器调试时必开)
# False = 正式运行模式：在第0步完整正向评估近30万个样例（跑大盘分数时开）
SKIP_BASELINE_EVAL="True"

echo "🚀 开始超网络动态 LoRA 训练..."
echo "卡数配置: ${NUM_GPUS} GPUs"
echo "调试模式 (跳过第一轮评估): ${SKIP_BASELINE_EVAL}"
echo "全局批次: ${GLOBAL_BATCH_SIZE} (每张卡分配 $((GLOBAL_BATCH_SIZE / NUM_GPUS)) 个样本)"
echo "📂 权重将保存在: ${CKPT_DIR}"
echo "📝 终端日志将保存在: logs/${run_name}.log"

# 使用 PyTorch 原生的一键分布式引擎
torchrun --nproc-per-node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    meta_train.py \
    --config-name=$CFG \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    lr=$LR \
    +checkpoint_path="${CKPT_DIR}" \
    +load_checkpoint=$BASE_CKPT_PATH \
    +project_name="trm-hypernetwork-integration" \
    +run_name="${run_name}" \
    skip_baseline_eval=${SKIP_BASELINE_EVAL} \
    2>&1 | tee "logs/${run_name}.log"

echo "✅ 训练已完成。终端日志保存在: logs/${run_name}.log"
