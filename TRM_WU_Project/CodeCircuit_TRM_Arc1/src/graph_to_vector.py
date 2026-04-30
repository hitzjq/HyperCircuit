"""
Graph-to-Vector Feature Extraction
===================================
对齐 CodeCircuit graph_dataset.py _extract_advanced_features:

特征向量结构 (固定长度):
  Level 1: 高层统计 (5 维)
  Level 2: 节点统计 (6 维) + 层直方图 (n_layers 维)
  Level 3: 拓扑特征 (12 维)
  
  总维度 = 5 + 6 + n_layers + 12 = 23 + n_layers
  对 TRM (n_layers=30): 53 维
"""

import os
import argparse
import glob
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from run_config import RunConfig

N_VIRTUAL_LAYERS = 30  # TRM: H_cycles(3) × (L_cycles(4)+1) × L_layers(2)


def normalize_matrix(matrix):
    """对齐 CodeCircuit graph.py normalize_matrix"""
    normalized = np.abs(matrix)
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-10, None)
    return normalized / row_sums


def compute_influence(A, logit_weights, max_iter=1000):
    """对齐 CodeCircuit graph.py compute_influence: B = A + A^2 + ..."""
    current = logit_weights @ A
    influence = current.copy()
    for _ in range(max_iter):
        current = current @ A
        if not np.any(current):
            break
        influence += current
    return influence


def compute_node_influence(adj, logit_weights):
    return compute_influence(normalize_matrix(adj), logit_weights)


def prune_graph_simple(adj, n_features, n_errors, n_tokens, n_logits, 
                       logit_probs, node_threshold=0.8):
    """
    简化版 prune_graph (对齐 CodeCircuit graph.py prune_graph 的核心逻辑)
    
    Returns:
        node_mask: (total_nodes,) bool
        cumulative_scores: (total_nodes,) float
    """
    total_nodes = adj.shape[0]
    
    # 构造 logit_weights
    logit_weights = np.zeros(total_nodes)
    logit_weights[-n_logits:] = logit_probs
    
    # 计算节点 influence
    node_influence = compute_node_influence(adj, logit_weights)
    
    # 按 influence 排序, 保留 top threshold
    sorted_scores = np.sort(node_influence)[::-1]
    cumulative = np.cumsum(sorted_scores) / (np.sum(sorted_scores) + 1e-10)
    threshold_idx = int(np.searchsorted(cumulative, node_threshold))
    threshold_idx = min(threshold_idx, len(cumulative) - 1)
    threshold_val = sorted_scores[threshold_idx]
    
    node_mask = node_influence >= threshold_val
    # 总是保留 token 和 logit 节点
    node_mask[-(n_tokens + n_logits):] = True
    
    # 计算 cumulative scores
    sorted_indices = np.argsort(node_influence)[::-1]
    sorted_vals = node_influence[sorted_indices]
    cumulative_scores = np.zeros(total_nodes)
    cum_vals = np.cumsum(sorted_vals) / (np.sum(sorted_vals) + 1e-10)
    cumulative_scores[sorted_indices] = cum_vals
    
    return node_mask, cumulative_scores


def extract_advanced_features(graph_data, node_threshold=0.8):
    """
    对齐 CodeCircuit graph_dataset.py _extract_advanced_features
    
    返回固定长度的特征向量 (5 + 6 + n_layers + 12 = 53 维)
    """
    adj = graph_data["adjacency_matrix"].numpy()
    n_features = graph_data["n_selected_features"]
    n_errors = graph_data["n_error_nodes"]
    n_tokens = graph_data["n_token_nodes"]
    n_logits = graph_data["n_logit_nodes"]
    n_layers = graph_data["n_layers"]
    n_pos = graph_data["n_pos"]
    active_features_tensor = graph_data["active_features"]  # (n_features, 3)
    activation_values = graph_data["activation_values"]      # (n_features,)
    logit_probs = graph_data["logit_probabilities"].numpy()  # (k,)
    
    # 剪枝
    node_mask, cumulative_scores = prune_graph_simple(
        adj, n_features, n_errors, n_tokens, n_logits, logit_probs, node_threshold
    )
    
    features = []
    total_nodes = adj.shape[0]
    
    # 确定各类节点的 index 范围
    error_start = n_features
    error_end = error_start + n_errors
    token_start = error_end
    token_end = token_start + n_tokens
    
    pruned_indices = np.where(node_mask)[0].tolist()
    pruned_feature_nodes = [i for i in pruned_indices if i < n_features]
    pruned_error_nodes = [i for i in pruned_indices if error_start <= i < error_end]
    
    # --- Level 1: 高层统计 (5 维) ---
    features.append(float(n_features))                     # 总活跃特征数
    features.append(float(len(pruned_feature_nodes)))      # 剪枝后特征节点数
    features.append(float(len(pruned_error_nodes)))        # 剪枝后 error 节点数
    
    if len(logit_probs) > 0:
        features.append(float(logit_probs[0]))             # Top logit probability
        entropy = -np.sum(logit_probs * np.log(logit_probs + 1e-8))
        features.append(float(entropy))                     # Logit entropy
    else:
        features.extend([0.0, 0.0])
    
    # --- Level 2: 节点统计 (6 维) ---
    pruned_influence = cumulative_scores[node_mask]
    features.append(float(np.mean(pruned_influence)) if len(pruned_influence) > 0 else 0.0)
    
    if len(pruned_error_nodes) > 0:
        error_influence = cumulative_scores[pruned_error_nodes]
        features.append(float(np.sum(error_influence)))
        features.append(float(np.mean(error_influence)))
    else:
        features.extend([0.0, 0.0])
    
    if len(pruned_feature_nodes) > 0:
        pruned_acts = activation_values[pruned_feature_nodes].numpy()
        features.append(float(np.mean(pruned_acts)))
        features.append(float(np.max(pruned_acts)))
        features.append(float(np.std(pruned_acts)))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # --- 层直方图 (n_layers 维) ---
    layer_counts = [0] * n_layers
    for node_idx in pruned_feature_nodes:
        if node_idx < len(active_features_tensor):
            layer = active_features_tensor[node_idx, 0].item()
            if layer < n_layers:
                layer_counts[layer] += 1
    features.extend([float(c) for c in layer_counts])
    
    # --- Level 3: 拓扑特征 (12 维) ---
    topo = extract_topological_features(pruned_indices, adj, 
                                         n_features, n_errors, n_tokens, n_logits, n_layers, n_pos)
    features.extend(topo)
    
    return np.array(features, dtype=np.float32)


def extract_topological_features(pruned_indices, adj, 
                                  n_features, n_errors, n_tokens, n_logits, n_layers, n_pos):
    """
    对齐 CodeCircuit _extract_topological_and_edge_features (12 维)
    """
    num_nodes = len(pruned_indices)
    
    defaults = [0.0] * 12  # 12 维: sum/mean/std/n_edges/density/components/deg_mean/deg_max/bet_mean/bet_max/avg_spl/input_logit_spl
    
    if num_nodes < 2:
        return defaults
    
    # 构建剪枝子图
    pruned_adj = adj[np.ix_(pruned_indices, pruned_indices)]
    G = nx.from_numpy_array(pruned_adj, create_using=nx.DiGraph)
    
    # 边权统计
    edge_weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
    
    features = []
    if edge_weights.size > 0:
        features.append(float(np.sum(edge_weights)))
        features.append(float(np.mean(edge_weights)))
        features.append(float(np.std(edge_weights)))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    features.append(float(G.number_of_edges()))
    features.append(float(nx.density(G)))
    features.append(float(nx.number_weakly_connected_components(G)))
    
    try:
        dc = nx.degree_centrality(G)
        features.append(float(np.mean(list(dc.values()))))
        features.append(float(np.max(list(dc.values()))) if dc else 0.0)
    except:
        features.extend([0.0, 0.0])
    
    try:
        bc = nx.betweenness_centrality(G, weight='weight')
        features.append(float(np.mean(list(bc.values()))))
        features.append(float(np.max(list(bc.values()))) if bc else 0.0)
    except:
        features.extend([0.0, 0.0])
    
    # 平均最短路径
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    if len(largest_cc) > 1:
        try:
            sub = G.subgraph(largest_cc)
            features.append(float(nx.average_shortest_path_length(sub, weight='weight')))
        except:
            features.append(-1.0)
    else:
        features.append(-1.0)
    
    # Input → Logit 最短路径
    error_start = n_features
    error_end = error_start + n_errors
    token_start = error_end
    token_end = token_start + n_tokens
    logit_start = token_end
    
    global_to_local = {g: l for l, g in enumerate(pruned_indices)}
    
    local_tokens = [global_to_local[i] for i in pruned_indices if token_start <= i < token_end]
    local_logits = [global_to_local[i] for i in pruned_indices if i >= logit_start]
    
    min_path = float('inf')
    if local_tokens and local_logits:
        for s in local_tokens:
            for t in local_logits:
                if nx.has_path(G, source=s, target=t):
                    try:
                        pl = nx.shortest_path_length(G, source=s, target=t, weight='weight')
                        min_path = min(min_path, pl)
                    except:
                        pass
    
    features.append(float(min_path) if min_path != float('inf') else -1.0)
    
    return features


def process_graphs(input_pattern, output_path):
    print(f"Loading graphs from: {input_pattern}")
    graph_files = sorted(glob.glob(input_pattern))
    
    if len(graph_files) == 0:
        print("未找到任何 .pt 文件，请检查路径。")
        return
    
    print(f"Found {len(graph_files)} graph files.")
    
    all_features = []
    query_mapping = []  # 保存 index → query 的映射关系
    
    for g_path in tqdm(graph_files, desc="Extracting Features"):
        data = torch.load(g_path, map_location="cpu", weights_only=False)
        feat_vec = extract_advanced_features(data)
        all_features.append(feat_vec)
        
        # 收集 query 身份信息
        meta = data.get("query_meta", {})
        query_mapping.append({
            "graph_file": os.path.basename(g_path),
            "graph_index": meta.get("graph_index", len(query_mapping)),
            "set_name": meta.get("set_name", ""),
            "puzzle_identifiers": meta.get("puzzle_identifiers", None),
            "inputs": meta.get("inputs", None),
            "labels": meta.get("labels", None),
        })
    
    final_tensor = torch.tensor(np.stack(all_features), dtype=torch.float32)
    print(f"\n✅ Feature Extraction Complete!")
    print(f"📊 Shape: {final_tensor.shape} (n_queries × feature_dims)")
    print(f"   Feature breakdown: 5 (high-level) + 6 (node stats) + {N_VIRTUAL_LAYERS} (layer hist) + 12 (topology) = {final_tensor.shape[1]}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存完整的 dataset: features + query mapping
    output_data = {
        "features": final_tensor,           # (n_queries, 53) — 53维特征向量
        "query_mapping": query_mapping,      # list of dicts: 每个 query 的身份
        "feature_dim": final_tensor.shape[1],
        "n_queries": len(query_mapping),
    }
    torch.save(output_data, output_path)
    print(f"💾 Saved to: {output_path}")
    print(f"   包含: features tensor + {len(query_mapping)} 条 query 映射")
    
    # 使用示例
    print(f"\n📌 HyperNet 使用方式:")
    print(f"   data = torch.load('{output_path}')")
    print(f"   circuit_vec = data['features'][i]          # 第 i 个 query 的 53 维电路特征")
    print(f"   query_input = data['query_mapping'][i]['inputs']  # 对应的原始输入")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Convert attribution graphs to feature vectors")
    RunConfig.add_run_args(parser)
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing graph_*.pt files. Defaults to the run attribution_graphs directory.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Feature .pt output path. Defaults to the run cc_advanced_features.pt path.")
    parser.add_argument("--skip_config_save", action="store_true",
                        help="Do not update the run config.json. Useful for concurrent shard workers.")
    args = parser.parse_args()
    
    rc = RunConfig(run_name=args.run_name)
    rc.print_summary()
    
    input_dir = args.input_dir or rc.graphs_dir
    output_path = args.output_path or rc.features_path
    input_pattern = os.path.join(input_dir, "*.pt")
    process_graphs(input_pattern, output_path)
    
    if not args.skip_config_save:
        rc.save_config(extra_info={
            "step": "4_graph_to_vector",
            "input_dir": input_dir,
            "output_path": output_path,
        })
