import os
import argparse
import glob
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

def extract_advanced_features_from_adj(adj_matrix: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    基于归因邻接矩阵提取高阶拓扑和统计特征（对齐 CodeCircuit GraphDataset 设计）
    Args:
        adj_matrix: (N, N) Numpy 数组，代表节点间的梯度影响权重
        threshold: 绝对值小于该值的边被认为是底噪，予以剔除
    Returns:
        1D numpy array 包含诸多图论和统计学特征。
    """
    N = adj_matrix.shape[0]
    
    # 构建 NetworkX 有向图 (忽略底噪边)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    
    edge_weights = []
    for i in range(N):
        for j in range(N):
            w = adj_matrix[i, j]
            if abs(w) > threshold:
                G.add_edge(i, j, weight=w)
                edge_weights.append(w)
                
    # --- 1. 基本边统计 ---
    features = []
    
    # sum, mean, std
    if len(edge_weights) > 0:
        features.append(float(np.sum(edge_weights)))
        features.append(float(np.mean(edge_weights)))
        features.append(float(np.std(edge_weights)))
    else:
        features.extend([0.0, 0.0, 0.0])
        
    # n_edges_pruned (有效边数) 和 graph_density
    features.append(float(len(edge_weights)))
    features.append(nx.density(G))
    
    # 连通分量 (对于弱连通)
    num_components = nx.number_weakly_connected_components(G)
    features.append(float(num_components))
    
    # --- 2. 节点中心性分析 ---
    # Degree Centrality (度中心性，只算入度或出度)
    try:
        in_degree = list(nx.in_degree_centrality(G).values())
        features.append(float(np.mean(in_degree)) if in_degree else 0.0)
        features.append(float(np.max(in_degree)) if in_degree else 0.0)
    except:
        features.extend([0.0, 0.0])
        
    # Betweenness Centrality (介数中心性)
    try:
        betweenness = list(nx.betweenness_centrality(G, weight='weight').values())
        features.append(float(np.mean(betweenness)) if betweenness else 0.0)
        features.append(float(np.max(betweenness)) if betweenness else 0.0)
    except:
        features.extend([0.0, 0.0])
        
    # --- 3. 最短路径 ---
    try:
        # 平均最短路径长度（如果图不连通，NetworkX 会报错，故需捕获并在连通子图内算）
        if nx.is_weakly_connected(G):
            avg_spl = nx.average_shortest_path_length(G, weight='weight', method='dijkstra')
        else:
            components = list(nx.weakly_connected_components(G))
            lengths = []
            for c in components:
                sub_G = G.subgraph(c)
                if len(sub_G) > 1:
                    lengths.append(nx.average_shortest_path_length(sub_G, weight='weight'))
            avg_spl = np.mean(lengths) if lengths else 0.0
        features.append(float(avg_spl))
    except Exception as e:
        features.append(0.0)
        
    return np.array(features, dtype=np.float32)

def process_graphs(input_pattern, output_path):
    print(f"Loading graphs from: {input_pattern}")
    graph_files = sorted(glob.glob(input_pattern))
    
    if len(graph_files) == 0:
        print("未找到任何 .pt 文件，请检查路径。")
        return
        
    all_features = []
    
    for g_path in tqdm(graph_files, desc="Converting Graph -> Vector"):
        data = torch.load(g_path, map_location="cpu", weights_only=False)
        
        adj = data["adjacency_matrix"].numpy()
        
        # 获取图高维统计特征
        adv_features = extract_advanced_features_from_adj(adj)
        
        # [可选补充] 接入节点自身的 activation_values 和 attribution_scores 平均值
        act_mean = float(data["activation_values"].mean())
        attr_mean = float(data["attribution_scores"].mean())
        n_layers = float(data["n_virtual_layers"])
        
        # 拼接组成最终给 PG 的 1D 向量
        combined = np.concatenate([[act_mean, attr_mean, n_layers], adv_features])
        all_features.append(combined)
        
    final_tensor = torch.tensor(np.stack(all_features), dtype=torch.float32)
    print(f"\n✅ All Graphs Processed!")
    print(f"📊 Final Dataset Shape: {final_tensor.shape} (Num_Queries x Feature_Dims)")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_tensor, output_path)
    print(f"💾 Saved strictly formatted dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="CodeCircuit_TRM_Arc1/results/attribution_graphs/*.pt")
    parser.add_argument("--output_path", type=str, default="CodeCircuit_TRM_Arc1/results/cc_advanced_features.pt")
    args = parser.parse_args()
    
    process_graphs(args.input_dir, args.output_path)
