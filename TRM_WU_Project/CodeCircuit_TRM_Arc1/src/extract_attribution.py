"""
Phase 1: Attribution Graph Extraction for TRM
==============================================
对每个 ARC query 计算归因图并保存。

流程：
  1. 加载 TRM 模型 + SAE 字典 → 组装 UnrolledTRMWrapper
  2. 对每个 query 跑全展开 forward，收集所有虚拟层的 SAE feature 激活
  3. 计算 Cross-Entropy Loss → .backward()
  4. 计算归因分数：attribution = activation * gradient
  5. 组装邻接矩阵，剪枝，保存
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 项目路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from trm_wrapper import UnrolledTRMWrapper, AttributionOutput


def load_sae(sae_path, device):
    """加载已训练的 SAE Transcoder"""
    # 复用 Phase 0 的 SparseAutoencoder 定义（同目录下）
    from train_transcoder import SparseAutoencoder
    
    sae = SparseAutoencoder(d_in=512, d_sae=4096)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.to(device)
    sae.eval()
    
    # SAE 参数不需要梯度
    for p in sae.parameters():
        p.requires_grad = False
    
    return sae


def load_trm_model(config_path, ckpt_path, device):
    """加载 TRM 基座模型"""
    import yaml
    from utils.functions import load_model_class
    
    # 手动声明 TRM 结构的配置（避开 Hydra 的 defaults 解析问题）
    # 获取 vocab_size (对齐 dataset 定义)
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    dataset_cfg = PuzzleDatasetConfig(
        seed=42, dataset_paths=["data/arc1concept-aug-1000"], global_batch_size=1,
        test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1
    )
    dataset = PuzzleDataset(dataset_cfg, split="train")
    metadata = dataset.metadata

    model_cfg = {
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4,
        "puzzle_emb_ndim": 512,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
        "halt_exploration_prob": 0.1,
        "halt_max_steps": 16,
        "batch_size": 2,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
    }

    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    model = model_cls(model_cfg).to(device)
    model.eval()
    
    # 加载 checkpoint（如有）
    if ckpt_path and os.path.exists(ckpt_path):
        best_ckpt = None
        for fname in sorted(os.listdir(ckpt_path)):
            if fname.startswith("step_") and not fname.endswith("_pg"):
                best_ckpt = os.path.join(ckpt_path, fname)
        
        if best_ckpt:
            print(f"Loading checkpoint: {best_ckpt}")
            state_dict = torch.load(best_ckpt, map_location=device)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, assign=True, strict=False)
    else:
        print("⚠️ No checkpoint provided. Using random weights for debugging.")
    
    return model


def compute_attribution_scores(wrapper_output: AttributionOutput, labels: torch.Tensor):
    """
    计算归因分数。
    
    1. 计算 Cross-Entropy Loss
    2. backward() 让梯度流回所有虚拟层
    3. 对每个虚拟层：attribution = activation * gradient
    
    Args:
        wrapper_output: UnrolledTRMWrapper 的输出
        labels: (batch, seq_len) ground truth token ids
    
    Returns:
        attributions: List[Tensor] 每个虚拟层的归因分数 (batch, seq, d_sae)
        loss_value: float 损失值
    """
    logits = wrapper_output.logits
    
    # Cross-Entropy Loss（对齐总方案决策 D2）
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,  # 忽略 padding
    )
    
    # 反向传播 — 梯度将沿全展开链路畅通流回每一层
    loss.backward()
    
    # 计算归因分数：activation × gradient（Taylor 一阶展开）
    attributions = []
    for feat_act in wrapper_output.feature_activations:
        if feat_act.grad is not None:
            attr = (feat_act * feat_act.grad).detach()
        else:
            attr = torch.zeros_like(feat_act)
        attributions.append(attr)
    
    return attributions, loss.item()


def build_layer_histogram(attributions, max_virtual_layers=42):
    """
    构建层分布直方图。
    
    对每个虚拟层，统计有多少个非零 feature 被激活。
    未执行的层（halt 提前终止）填 0。
    
    Args:
        attributions: List[Tensor] 每个虚拟层的归因分数
        max_virtual_layers: 直方图固定长度
    
    Returns:
        histogram: Tensor (max_virtual_layers,)
    """
    histogram = torch.zeros(max_virtual_layers)
    for i, attr in enumerate(attributions):
        if i >= max_virtual_layers:
            break
        # 统计该层的非零归因特征数（取绝对值阈值）
        histogram[i] = (attr.abs() > 1e-6).float().sum().item()
    return histogram


def extract_query_graph(wrapper, batch, device):
    """
    对单个 query 提取归因图。
    
    Returns:
        graph_data: dict 包含邻接矩阵、特征激活、归因分数等
    """
    # 清零梯度
    wrapper.zero_grad()
    
    # 需要 feature_activations 保留梯度
    batch_gpu = {k: v.to(device) for k, v in batch.items()}
    
    # 前向传播
    output = wrapper(batch_gpu)
    
    # 确保 feature_activations 需要梯度
    for feat in output.feature_activations:
        feat.retain_grad()
    
    # 计算归因
    attributions, loss_val = compute_attribution_scores(output, batch_gpu["labels"])
    
    # 构建层分布直方图
    n_virtual_layers = len(attributions)
    histogram = build_layer_histogram(attributions)
    
    # 提取每层的 top-K 特征节点（按归因分数绝对值排序）
    top_k = 100  # 每层最多保留 100 个特征
    active_features = []   # (layer, position, feature_idx) 三元组
    activation_values = [] # 对应的激活值
    attribution_scores = [] # 对应的归因分数
    
    for layer_idx, (feat_act, attr) in enumerate(zip(
        output.feature_activations, attributions
    )):
        # feat_act: (batch, seq, d_sae)
        # attr: (batch, seq, d_sae)
        # 取 batch 维度的平均
        mean_attr = attr.mean(dim=0).abs()  # (seq, d_sae)
        mean_act = feat_act.detach().mean(dim=0)  # (seq, d_sae)
        
        # 找非零位置
        nonzero_mask = mean_attr > 1e-6
        if nonzero_mask.any():
            positions, feature_indices = nonzero_mask.nonzero(as_tuple=True)
            scores = mean_attr[positions, feature_indices]
            
            # 按分数排序取 top-K
            if len(scores) > top_k:
                topk_idx = scores.topk(top_k).indices
                positions = positions[topk_idx]
                feature_indices = feature_indices[topk_idx]
                scores = scores[topk_idx]
            
            for pos, feat_idx, score in zip(positions, feature_indices, scores):
                active_features.append([layer_idx, pos.item(), feat_idx.item()])
                activation_values.append(mean_act[pos, feat_idx].item())
                attribution_scores.append(score.item())
    
    graph_data = {
        "loss": loss_val,
        "n_virtual_layers": n_virtual_layers,
        "layer_histogram": histogram,
        "active_features": torch.tensor(active_features) if active_features else torch.zeros(0, 3, dtype=torch.long),
        "activation_values": torch.tensor(activation_values),
        "attribution_scores": torch.tensor(attribution_scores),
    }
    
    return graph_data


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Extract Attribution Graphs")
    parser.add_argument("--config_path", type=str, default="config/cfg_wu4trm.yaml")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--sae_path", type=str, 
                        default="CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096.pt")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        default=["data/arc1concept-aug-1000"])
    parser.add_argument("--output_dir", type=str, 
                        default="CodeCircuit_TRM_Arc1/results/attribution_graphs")
    parser.add_argument("--max_queries", type=int, default=-1,
                        help="最大处理 query 数，设 -1 则处理全部")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集 split (train/test)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ====== 加载模型 ======
    print("Loading SAE Transcoder...")
    sae = load_sae(args.sae_path, device)
    
    print("Loading TRM Base Model...")
    trm_model = load_trm_model(args.config_path, args.ckpt_path, device)
    
    print("Assembling UnrolledTRMWrapper...")
    wrapper = UnrolledTRMWrapper(trm_model.inner, sae)
    wrapper.to(device)
    wrapper.eval()
    
    # ====== 加载数据 ======
    print("Loading Dataset...")
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    
    dataset_cfg = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=args.dataset_paths,
        global_batch_size=1,   # 归因逐条处理
        test_set_mode=(args.split == "test"),
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_cfg, split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    
    # ====== 提图循环 ======
    print(f"Starting Attribution Extraction (split={args.split})...")
    count = 0
    
    for set_name, batch, effective_batch_size in tqdm(dataloader, desc="Extracting"):
        if args.max_queries > 0 and count >= args.max_queries:
            break
        
        try:
            graph_data = extract_query_graph(wrapper, batch, device)
            
            # 保存
            save_path = os.path.join(args.output_dir, f"graph_{count:06d}.pt")
            torch.save(graph_data, save_path)
            
            if count % 50 == 0:
                n_features = len(graph_data["active_features"])
                print(f"\n  Query {count}: loss={graph_data['loss']:.4f}, "
                      f"active_features={n_features}, "
                      f"layers={graph_data['n_virtual_layers']}")
            
            count += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM at query {count}. Skipping...")
                torch.cuda.empty_cache()
                continue
            raise
    
    print(f"\n✅ Attribution extraction complete! {count} graphs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
