#!/bin/bash
set -x

export OMP_NUM_THREADS=4

# ==========================================
# TRM_WU_Project 串行消融对比一键启动脚本
# ==========================================

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT"

# ---- 配置执行参数 ----
SEQ_LEN=1024
# ---- 硬件与规模配置 ----
NUM_GPUS=8           # 集群 8张 H200
GLOBAL_BATCH_SIZE=1024  # 全局 Batch Size 1024，拉满算力
LR=1e-4

# ---- 数据集与权重配置 ----
DATASET_PATH="data/arc1concept-aug-1000"
BASE_CKPT_PATH="pretrained_base_ckpt/ARC-AGI-1/step_155718" 

ARCH="trm"
CFG="cfg_wu4trm" 
SKIP_BASELINE_EVAL="True"

# 我们要在循环里分别跑这两种模式
# 第一遍跑极致压缩时间的 embedding_only
# 第二遍跑提取深层思维特征的 full_trm
MODES=("embedding_only" "full_trm")

for MODE in "${MODES[@]}"; do
    echo "=========================================================="
    echo "🚀 [ 消融对比实验 ] 即将启动 Condition 模式 -> 【 ${MODE} 】"
    echo "=========================================================="

    # 根据当前模式自动定制保存路径和 wandb名字
    run_name="WU4TRM_ablation_${MODE}_bs${GLOBAL_BATCH_SIZE}_${NUM_GPUS}gpus_$(date +%m%d_%H%M)"
    CKPT_DIR="checkpoints/${run_name}"
    mkdir -p "${CKPT_DIR}"
    mkdir -p "logs"

    echo "卡数配置: ${NUM_GPUS} GPUs"
    echo "正在使用的数据集: ${DATASET_PATH}"
    echo "全局批次: ${GLOBAL_BATCH_SIZE}"
    echo "目标存档目录: ${CKPT_DIR}"
    echo "终端监控日志: logs/${run_name}.log"

    # 使用 PyTorch 分布式引擎启动训练
    torchrun --nproc-per-node=${NUM_GPUS} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:0 \
        --nnodes=1 \
        meta_train.py \
        --config-name=$CFG \
        global_batch_size=$GLOBAL_BATCH_SIZE \
        lr=$LR \
        data_paths="['${DATASET_PATH}']" \
        condition_mode="${MODE}" \
        +checkpoint_path="${CKPT_DIR}" \
        +load_checkpoint=$BASE_CKPT_PATH \
        +project_name="trm-hypernetwork-ablation" \
        +run_name="${run_name}" \
        skip_baseline_eval=${SKIP_BASELINE_EVAL} \
        2>&1 | tee "logs/${run_name}.log"

    echo "✅ 模式 [ ${MODE} ] 训练运行完毕或已中断。"
    echo ">>> 即将清理显存并休息 10 秒钟准备运行下一个实验 <<<"
    sleep 10
done

echo "🎉🎉🎉 所有消融实验全跑完啦！请去 logs/ 目录收取对比战报！"
