#!/bin/bash

# ==========================================
# TRM LoRA 微调一键启动脚本 (带本地日志监控版)
# ==========================================

# 1. 实验超参数设定区 (在此处直接修改所有关键参数)
NUM_GPUS=4                                 # 使用的 GPU 数量
R_VALUE=64                                 # LoRA Rank
ALPHA_VALUE=32                             # LoRA Alpha
LR_VALUE="1e-3"                            # 学习率
BATCH_SIZE=256                             # 全局 Batch Size (会被 num_gpus 平分)
EPOCHS=20                                  # 调试用：最大训练轮次 (填个具体的数字比如 20 跑完自动停下并保存，如果要训全量则改成 20000)

# 2. 实验名称与输出路径结构
EXP_NAME="LoRA_r${R_VALUE}_lr${LR_VALUE}_${NUM_GPUS}gpus"
BASE_CHECKPOINT_DIR="checkpoints/LoRA_Experiments"
CURRENT_EXP_DIR="${BASE_CHECKPOINT_DIR}/${EXP_NAME}"
LOG_FILE="${CURRENT_EXP_DIR}/terminal_output.log"

# 创建当前实验的专属日志与权重文件夹
mkdir -p "${CURRENT_EXP_DIR}"
echo "🌟 准备启动实验: ${EXP_NAME}"
echo "📂 日志文件将保存在: ${LOG_FILE}"

# 3. 启动命令 (使用 nohup 后台静默运行并收集终端字符)
# nohup 可以让程序在您同事断开 SSH 连接后继续在后台跑。
# > ${LOG_FILE} 2>&1 把所有标准输出和报错都重定向到了专属结果文件夹的 log 里。

nohup torchrun --standalone --nproc_per_node=${NUM_GPUS} lora_finetune.py \
    checkpoint_path="${CURRENT_EXP_DIR}" \
    run_name="${EXP_NAME}" \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=4 \
    data_paths="['data/arc1concept-aug-1000/train']" \
    data_paths_test="['data/arc1concept-aug-1000/evaluation']" \
    +load_checkpoint=checkpoints/ARC-AGI-1/step_155718 \
    lora_r=${R_VALUE} \
    lora_alpha=${ALPHA_VALUE} \
    lr=${LR_VALUE} \
    global_batch_size=${BATCH_SIZE} \
    epochs=${EPOCHS} \
    > "${LOG_FILE}" 2>&1 &


