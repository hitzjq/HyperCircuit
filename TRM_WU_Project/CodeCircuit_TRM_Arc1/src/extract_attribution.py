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
    # 将整个 SAE 网络与字典统一切换为 BFloat16 精度，完美对齐 TRM
    sae = sae.to(torch.bfloat16)
    
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
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1).long(),
        ignore_index=-100,
    )
    
    # 【改动】：保留计算图，为接下来的多次独立 VJP 提取邻接线做准备
    loss.backward(retain_graph=True)
    
    # 计算一阶节点全局归因分数：activation × gradient
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
    对单个 query 提取归因图：构建符合原版 CodeCircuit 的边特征矩阵 (adjacency_matrix)。
    为了获取节点间偏导（边），将在找到重要节点后，基于它们各自的张量反打多次 VJP！
    """
    print(f"\n  [TRACE] Starting Query Graph Extraction...", flush=True)
    wrapper.zero_grad()
    batch_gpu = {k: v.to(device) for k, v in batch.items()}
    
    print(f"  [TRACE] Forward pass...", flush=True)
    output = wrapper(batch_gpu)
    
    print(f"  [TRACE] Global Backward passed starting...", flush=True)
    # 计算全局归因，同时 retain_graph
    attributions, loss_val = compute_attribution_scores(output, batch_gpu["labels"])
    print(f"  [TRACE] Global Backward Finished! attributions found.", flush=True)
    
    n_virtual_layers = len(attributions)
    histogram = build_layer_histogram(attributions)
    
    print(f"  [TRACE] Node Filtering...", flush=True)
    
    # ====== Step 1: 捞出活跃节点的全局影响力，用张量向量化防止 Python 循环死锁 ======
    # 调整为更犀利的 Circuit 剪枝节点数（保留图最重要的 250 个枢纽节点，剔除冗余噪音，速度可直降至 3s！）
    max_nodes = 250
    node_pool = []
    
    for layer_idx, (feat_act, attr) in enumerate(zip(output.feature_activations, attributions)):
        act_0 = feat_act[0].detach()  # (seq, d_sae)
        attr_0 = attr[0].abs()        # (seq, d_sae)
        
        # PyTorch 端直接取层内 TopK（即使单层，最多也只可能贡献 max_nodes 个前排）
        k = min(max_nodes, attr_0.numel())
        flat_attr = attr_0.flatten()
        top_scores, top_indices = torch.topk(flat_attr, k)
        
        # 过滤掉底噪，减少不必要的 Python 循环
        valid_mask = top_scores > 1e-6
        top_scores = top_scores[valid_mask]
        top_indices = top_indices[valid_mask]
        
        if len(top_scores) == 0:
            continue
            
        # GPU 张量操作还原坐标
        seq_indices = torch.div(top_indices, attr_0.size(1), rounding_mode='floor')
        feat_indices = top_indices % attr_0.size(1)
        
        acts = act_0[seq_indices, feat_indices]
        
        # 现在这个 list 推导式最多只有几百个元素，瞬间完成
        for s, seq_i, f_i, a_val in zip(top_scores.tolist(), seq_indices.tolist(), feat_indices.tolist(), acts.tolist()):
            node_pool.append({
                "layer": layer_idx,
                "pos": seq_i,
                "feat": f_i,
                "act_val": a_val,
                "score": s
            })
    
    # 按照全局影响力倒序排序，截取总网的 Top N
    node_pool = sorted(node_pool, key=lambda x: x["score"], reverse=True)[:max_nodes]
    # 截取后再按推理的时序（层级递进，然后 Token 位置递进）进行拓扑排序！这是邻接图计算基石。
    node_pool = sorted(node_pool, key=lambda x: (x["layer"], x["pos"]))
    
    N = len(node_pool)
    adjacency_matrix = torch.zeros((N, N), dtype=torch.float32, device=device)
    
    active_features = []
    activation_values = []
    attribution_scores = []
    
    for i, node in enumerate(node_pool):
        active_features.append([node["layer"], node["pos"], node["feat"]])
        activation_values.append(node["act_val"])
        attribution_scores.append(node["score"])
        
    print(f"  [TRACE] Generating VJP Edges...", flush=True)
    from tqdm import tqdm
    tqdm.write(f"\n  [VJP Profiler] Computing Adjacency Edges for {N} topologically sorted active nodes...")
    
    # ====== Step 2: VJP (Vector-Jacobian Product) 组装边网络！======
    # [核心定论]：刚才的 Batched VJP 需要扩容内存并产生 187 次极度缓慢的前向重连（导致35秒）。
    # 返回最初被证明是神级设计的最简连环策略：1 次 Forward + max_nodes 次串行极速 Backward！
    # 依赖 retain_graph=True 直接在已有 C++ 原生计算图高速滑行，把 35 秒降维打击回极速！
    from tqdm import tqdm
    for target_idx in tqdm(range(1, N), desc="VJP Edge Tracing", leave=False, dynamic_ncols=True):
        target = node_pool[target_idx]
        b_layer = target["layer"]
        
        # 定位目标节点的张量标量值 (原本的唯一图)
        b_tensor_val = output.feature_activations[b_layer][0, target["pos"], target["feat"]]
        
        if not b_tensor_val.requires_grad:
            continue
            
        # 发送串行高频短脉冲，直接借助 1 棵保留的庞大前戏图复用 250 次，榨干显卡
        grads = torch.autograd.grad(
            outputs=b_tensor_val, 
            inputs=output.feature_activations[:b_layer+1],  # 截断传递路径，后续层必定无连接
            retain_graph=True, 
            allow_unused=True
        )
        
        # 收集来自 B 之前的所有 Source 节点的连接
        for source_idx in range(target_idx):
            source = node_pool[source_idx]
            s_layer = source["layer"]
            s_grad_tensor = grads[s_layer]
            if s_grad_tensor is not None:
                grad_a_to_b = s_grad_tensor[0, source["pos"], source["feat"]].item()
                edge_weight = source["act_val"] * grad_a_to_b
                adjacency_matrix[target_idx, source_idx] = edge_weight
                
    # 释放显存树
    wrapper.zero_grad()
    
    # ====== Step 3: 打包成仿 Graph 对象 ======
    # CodeCircuit 的 _extract_advanced_features 主要探测 "adjacency_matrix", "active_features" 等字段
    graph_data = {
        "loss": loss_val,
        "n_virtual_layers": n_virtual_layers,
        "layer_histogram": histogram,
        "active_features": torch.tensor(active_features, dtype=torch.long),
        "activation_values": torch.tensor(activation_values, dtype=torch.float32),
        "attribution_scores": torch.tensor(attribution_scores, dtype=torch.float32),
        "adjacency_matrix": adjacency_matrix.cpu(),
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
