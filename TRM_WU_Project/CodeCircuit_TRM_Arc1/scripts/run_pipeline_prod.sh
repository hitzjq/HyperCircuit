#!/bin/bash

# ==============================================================================
# CodeCircuit TRM Attribution Pipeline (全量生产环境 PROD)
# ==============================================================================
# 此脚本设计在 H200 等真实集群上运行，无任何截断限制，提取全量真实拓扑特征。

# 1. 确保在项目根目录运行
PROJECT_ROOT=$(cd "$(dirname "$0")/../../" && pwd)
cd "$PROJECT_ROOT"

# 2. 基础路径配置
DATASET_PATH="data/arc1concept-aug-1000"
CKPT_PATH="../TinyRecursiveModels/checkpoints/ARC-AGI-1" # H200真实权重路径
CONFIG_PATH="config/cfg_wu4trm.yaml"

ACTIVATIONS_DIR="CodeCircuit_TRM_Arc1/results/activations"
SAE_PATH="CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096.pt"
GRAPHS_DIR="CodeCircuit_TRM_Arc1/results/attribution_graphs"
FEATURES_OUT="CodeCircuit_TRM_Arc1/results/cc_advanced_features.pt"

# 日志设置
mkdir -p CodeCircuit_TRM_Arc1/logs
RUN_TIME=$(date +"%m%d_%H%M")
LOG_FILE="CodeCircuit_TRM_Arc1/logs/pipeline_PROD_${RUN_TIME}.log"

echo "🚀 [TRM CodeCircuit] 正在启动全量生产管线..."
echo "📝 日志实时写入: $LOG_FILE"
echo "──────────────────────────────────────────"

(
    # ==========================================================
    # [PHASE 0] 数据收集 & 训练核心编解码器 (SAE)
    # ==========================================================
    echo ">> [STEP 1/3] 执行 Phase 0: 全量激活数据收集与 SAE 训练"
    
    # 清理并重建激活存储（防止混入上一次的测试脏数据）
    rm -rf "$ACTIVATIONS_DIR"
    mkdir -p "$ACTIVATIONS_DIR"
    mkdir -p "$(dirname "$SAE_PATH")"
    
    # 真正收集全集激活 (移除 --max_batches)
    python CodeCircuit_TRM_Arc1/src/collect_activations.py \
        --dataset_paths "$DATASET_PATH" \
        --ckpt_path "$CKPT_PATH"
        
    echo ">> [STEP 1.5/3] 训练真实 SAE Transcoder (挂载 PROD 参数)"
    # PROD 级别 SAE 训练：训练 30 个 Epoch
    python CodeCircuit_TRM_Arc1/src/train_transcoder.py \
        --activations_dir "$ACTIVATIONS_DIR" \
        --save_path "$SAE_PATH" \
        --d_in 512 \
        --d_sae 4096 \
        --epochs 30 \
        --batch_size 8192 \
        --lr 5e-4
        
    echo "──────────────────────────────────────────"

    # ==========================================================
    # [PHASE 1] VJP 全量拓扑成图
    # ==========================================================
    echo ">> [STEP 2/3] 执行 Phase 1: 拓扑高配归因图阵列化"
    
    # 清空可能存在的旧图
    rm -rf "$GRAPHS_DIR"
    mkdir -p "$GRAPHS_DIR"
    
    # 全量 VJP 制图 (移除 --max_queries，真实跑全集)
    python CodeCircuit_TRM_Arc1/src/extract_attribution.py \
        --config_path "$CONFIG_PATH" \
        --ckpt_path "$CKPT_PATH" \
        --sae_path "$SAE_PATH" \
        --dataset_paths "$DATASET_PATH" \
        --split "train"
        
    echo "──────────────────────────────────────────"

    # ==========================================================
    # [PHASE 2] 一键压制最终供 PG 调用的超强向量集
    # ==========================================================
    echo ">> [STEP 3/3] 执行 Phase 2: 高级拓扑特征降维压缩"
    
    python CodeCircuit_TRM_Arc1/src/graph_to_vector.py \
        --input_dir "${GRAPHS_DIR}/*.pt" \
        --output_path "$FEATURES_OUT"
        
    echo "──────────────────────────────────────────"
    echo "🎉🎉🎉 ALL DONE! 完整流水线结束。 🎉🎉🎉"
    echo "最终投喂特征文件就绪: $FEATURES_OUT"

) 2>&1 | tee "$LOG_FILE"
