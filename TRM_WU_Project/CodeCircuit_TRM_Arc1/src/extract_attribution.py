"""
Phase 1: Attribution Graph Extraction for TRM
==============================================
对每个 ARC query 计算归因图并保存。

对齐 CodeCircuit 原版 Graph 结构:
  节点类型: [active_features, error_nodes, embed_nodes, logit_nodes]
  
  - active_features: 被选中的 SAE 特征节点 (layer, pos, feat_idx)
  - error_nodes: 每层每位置的 SAE 重构误差 (n_layers × n_active_pos)
  - embed_nodes: 输入 token embedding (n_active_pos 个)
  - logit_nodes: Top-k 输出 logit (k 个)

流程：
  1. 加载 TRM 模型 + 2个 SAE → 组装 UnrolledTRMWrapper
  2. 先跑 15 步 ACT (无梯度) → 得到深度推理后的 z_H, z_L
  3. 用 Wrapper 带梯度跑第 16 步 Inner forward (30 虚拟层)
  4. 计算 Cross-Entropy Loss → backward(retain_graph=True)
  5. 筛选 Top-N 活跃特征节点
  6. 对每个 source 节点, 注入 encoder_vector 做 VJP → 收集对所有 target 的边
  7. 组装完整邻接矩阵 [features, errors, tokens, logits]
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
from run_config import RunConfig

# ==================== 常量 ====================
N_VIRTUAL_LAYERS = 30    # H_cycles(3) × (L_cycles(4)+1) × L_layers(2)
MAX_FEATURE_NODES = 250  # 最多保留的 SAE 特征节点数
MAX_N_LOGITS = 10        # Top-k logit 节点数


def load_dual_sae(sae_path_0, sae_path_1, device):
    """加载 2 个已训练的 SAE Transcoder"""
    from train_transcoder import SparseAutoencoder
    
    sae_0 = SparseAutoencoder(d_in=512, d_sae=4096)
    sae_0.load_state_dict(torch.load(sae_path_0, map_location=device))
    sae_0.to(device).to(torch.bfloat16)
    for p in sae_0.parameters():
        p.requires_grad = False
    
    sae_1 = SparseAutoencoder(d_in=512, d_sae=4096)
    sae_1.load_state_dict(torch.load(sae_path_1, map_location=device))
    sae_1.to(device).to(torch.bfloat16)
    for p in sae_1.parameters():
        p.requires_grad = False
    
    return [sae_0, sae_1]


def load_trm_model(config_path, ckpt_path, device):
    """加载 TRM 基座模型"""
    from utils.functions import load_model_class
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    
    dataset_cfg = PuzzleDatasetConfig(
        seed=42, dataset_paths=["data/arc1concept-aug-1000"], global_batch_size=1,
        test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1
    )
    dataset = PuzzleDataset(dataset_cfg, split="train")
    metadata = dataset.metadata

    model_cfg = {
        "H_cycles": 3, "L_cycles": 4, "H_layers": 0, "L_layers": 2,
        "hidden_size": 512, "num_heads": 8, "expansion": 4,
        "puzzle_emb_ndim": 512, "pos_encodings": "rope",
        "forward_dtype": "bfloat16", "mlp_t": False,
        "puzzle_emb_len": 16, "no_ACT_continue": True,
        "halt_exploration_prob": 0.1, "halt_max_steps": 16,
        "batch_size": 2,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
    }

    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    model = model_cls(model_cfg).to(device)
    model.eval()
    
    if ckpt_path and os.path.exists(ckpt_path):
        if os.path.isfile(ckpt_path):
            # 直接指定文件
            actual_ckpt = ckpt_path
        else:
            # 目录：找最新的 step_* 文件
            actual_ckpt = None
            for fname in sorted(os.listdir(ckpt_path)):
                if fname.startswith("step_") and not fname.endswith("_pg"):
                    actual_ckpt = os.path.join(ckpt_path, fname)
        
        if actual_ckpt:
            print(f"Loading checkpoint: {actual_ckpt}")
            state_dict = torch.load(actual_ckpt, map_location=device)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, assign=True, strict=False)
    else:
        print("⚠️ No checkpoint provided. Using random weights for debugging.")
    
    return model


def run_act_preflight(model, batch, device, halt_max_steps=16):
    """先跑 15 步 ACT (无梯度), 返回最后一步的初始 z_H, z_L"""
    with torch.no_grad():
        with torch.device(device):
            carry = model.initial_carry(batch)
        
        preflight_steps = halt_max_steps - 1
        for step in range(preflight_steps):
            carry, outputs = model(carry, batch)
            if carry.halted.all():
                break
        
        reset_carry = model.inner.reset_carry(carry.halted, carry.inner_carry)
        z_H = reset_carry.z_H.clone()
        z_L = reset_carry.z_L.clone()
    
    return z_H, z_L


def compute_salient_logits(logits_at_last_pos, lm_head_weight, max_n_logits=10):
    """
    对齐 CodeCircuit attribute.py compute_salient_logits:
    选择 top-k logit 节点, 计算 logit probability 和 demeaned unembedding vector。
    
    Args:
        logits_at_last_pos: (vocab_size,) 最后一个 position 的 logits
        lm_head_weight: (vocab_size, d_model) — lm_head 的权重矩阵
        max_n_logits: top-k
    
    Returns:
        logit_indices: (k,) token ids
        logit_probs: (k,) softmax probabilities
        logit_vecs: (k, d_model) demeaned unembedding vectors
    """
    probs = torch.softmax(logits_at_last_pos.float(), dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    
    # 选到 cumulative prob >= 0.95 为止
    cumsum = torch.cumsum(top_p, 0)
    cutoff = int(torch.searchsorted(cumsum, 0.95).item()) + 1
    cutoff = min(cutoff, max_n_logits)
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
    
    # demeaned unembedding vectors
    cols = lm_head_weight[top_idx]  # (k, d_model)
    demeaned = cols - lm_head_weight.mean(dim=0, keepdim=True)
    
    return top_idx, top_p, demeaned


def extract_query_graph(wrapper, sae_models, batch, device,
                        init_z_H=None, init_z_L=None, query_meta=None):
    """
    对单个 query 提取完整归因图 (对齐 CodeCircuit Graph 结构)。
    
    邻接矩阵结构:
      [selected_features | error_nodes | embed_nodes | logit_nodes]
      
    其中:
      - selected_features: MAX_FEATURE_NODES 个被选中的 SAE 特征
      - error_nodes: N_VIRTUAL_LAYERS × n_active_pos (SAE 重构误差)
      - embed_nodes: n_active_pos 个 (输入 token)
      - logit_nodes: top-k 个 (输出预测)
    
    Args:
        query_meta: dict, 保存 query 身份 {"set_name", "puzzle_id", "inputs", "labels"}
    """
    wrapper.zero_grad()
    batch_gpu = {k: v.to(device) for k, v in batch.items()}
    
    # ====== Forward ======
    output = wrapper(batch_gpu, init_z_H=init_z_H, init_z_L=init_z_L)
    
    # ====== Global backward (retain graph for VJP) ======
    logits = output.logits
    labels = batch_gpu["labels"]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1).long(),
        ignore_index=-100,
    )
    loss.backward(retain_graph=True)
    loss_val = loss.item()
    
    # ====== 全局归因分数: |activation × gradient| ======
    attributions = []
    for feat_act in output.feature_activations:
        if feat_act.grad is not None:
            attr = (feat_act * feat_act.grad).abs().detach()
        else:
            attr = torch.zeros_like(feat_act)
        attributions.append(attr)
    
    # ====== 筛选 Top-N 活跃特征 → active_features ======
    node_pool = []
    for layer_idx, (feat_act, attr) in enumerate(zip(output.feature_activations, attributions)):
        act_0 = feat_act[0].detach()    # (seq, d_sae)
        attr_0 = attr[0]                # (seq, d_sae)
        
        k = min(MAX_FEATURE_NODES, attr_0.numel())
        flat_attr = attr_0.flatten()
        top_scores, top_indices = torch.topk(flat_attr, k)
        
        valid_mask = top_scores > 1e-6
        top_scores = top_scores[valid_mask]
        top_indices = top_indices[valid_mask]
        
        if len(top_scores) == 0:
            continue
            
        seq_indices = torch.div(top_indices, attr_0.size(1), rounding_mode='floor')
        feat_indices = top_indices % attr_0.size(1)
        acts = act_0[seq_indices, feat_indices]
        block_idx = output.block_indices[layer_idx]
        
        for s, seq_i, f_i, a_val in zip(
            top_scores.tolist(), seq_indices.tolist(), 
            feat_indices.tolist(), acts.tolist()
        ):
            node_pool.append({
                "layer": layer_idx, "pos": seq_i, "feat": f_i,
                "act_val": a_val, "score": s, "block_idx": block_idx,
            })
    
    # 全局 Top-N
    node_pool = sorted(node_pool, key=lambda x: x["score"], reverse=True)[:MAX_FEATURE_NODES]
    node_pool = sorted(node_pool, key=lambda x: (x["layer"], x["pos"]))
    n_selected = len(node_pool)
    
    # ====== 构建 active_features 张量 ======
    active_features = torch.tensor(
        [[n["layer"], n["pos"], n["feat"]] for n in node_pool], dtype=torch.long
    )
    activation_values = torch.tensor(
        [n["act_val"] for n in node_pool], dtype=torch.float32
    )
    attribution_scores = torch.tensor(
        [n["score"] for n in node_pool], dtype=torch.float32
    )
    
    # selected_features 在我们的简化中 = 0..n_selected-1 (因为已经筛选过了)
    selected_features = torch.arange(n_selected, dtype=torch.long)
    
    # ====== 确定 active positions ======
    active_positions = sorted(set(n["pos"] for n in node_pool))
    n_active_pos = len(active_positions)
    pos_to_local = {p: i for i, p in enumerate(active_positions)}
    
    # ====== 计算 logit 节点 ======
    # 获取 lm_head 权重
    lm_head = wrapper.trm_inner.lm_head
    if hasattr(lm_head, 'weight'):
        lm_head_weight = lm_head.weight.detach().float()  # (vocab_size, d_model)
    else:
        # Linear 层默认 weight shape = (out_features, in_features)
        lm_head_weight = list(lm_head.parameters())[0].detach().float()
    
    # 取最后一个 position 的 logits
    logits_last = logits[0, -1].detach().float()  # (vocab_size,)
    logit_indices, logit_probs, logit_vecs = compute_salient_logits(
        logits_last, lm_head_weight, MAX_N_LOGITS
    )
    n_logits = len(logit_indices)
    
    # ====== 邻接矩阵布局 ======
    # [selected_features(N) | error_nodes(30×P) | embed_nodes(P) | logit_nodes(K)]
    n_errors = N_VIRTUAL_LAYERS * n_active_pos
    n_tokens = n_active_pos
    total_nodes = n_selected + n_errors + n_tokens + n_logits
    
    adj = torch.zeros(total_nodes, total_nodes, dtype=torch.float32, device=device)
    
    # 偏移量
    error_offset = n_selected
    token_offset = error_offset + n_errors
    logit_offset = token_offset + n_tokens
    
    # ====== VJP: Feature → Feature + Token → Feature edges ======
    # 对每个 source feature, 注入 encoder_vec → backward → 收集 target 处的梯度
    
    for src_idx in tqdm(range(n_selected), desc="VJP Attribution", leave=False, dynamic_ncols=True):
        src = node_pool[src_idx]
        s_layer = src["layer"]
        s_block = src["block_idx"]
        
        if s_layer >= len(output.resid_activations):
            continue
        
        resid_at_source = output.resid_activations[s_layer]
        if resid_at_source is None or not resid_at_source.requires_grad:
            continue
        
        # encoder vector for this source
        source_sae = sae_models[s_block]
        enc_vec = source_sae.encoder.weight[src["feat"]].detach()  # (d_model,)
        
        # 构造注入梯度: 只在 source position 注入
        inject_grad = torch.zeros_like(resid_at_source)
        inject_grad[0, src["pos"]] = enc_vec.to(inject_grad.dtype)
        
        # 收集 inputs = 所有 layer 的 resid + input_embeddings
        inputs_list = []
        input_layer_map = []  # 每个 input 对应的虚拟层 index
        
        # 所有虚拟层的 resid (target features 和 error 的来源)
        for layer_idx in range(len(output.resid_activations)):
            r = output.resid_activations[layer_idx]
            if r is not None and r.requires_grad:
                inputs_list.append(r)
                input_layer_map.append(('resid', layer_idx))
        
        # input_embeddings (token nodes 的来源)
        if output.input_embeddings.requires_grad:
            inputs_list.append(output.input_embeddings)
            input_layer_map.append(('embed', -1))
        
        if not inputs_list:
            continue
        
        # VJP
        try:
            grads = torch.autograd.grad(
                outputs=resid_at_source,
                inputs=inputs_list,
                grad_outputs=inject_grad,
                retain_graph=True,
                allow_unused=True
            )
        except RuntimeError:
            continue
        
        # 收集 source → target edges
        for grad_val, (node_type, layer_idx) in zip(grads, input_layer_map):
            if grad_val is None:
                continue
            
            if node_type == 'resid':
                # Feature edges: source → target_feature
                for tgt_idx, tgt in enumerate(node_pool):
                    if tgt["layer"] != layer_idx:
                        continue
                    
                    grad_at_pos = grad_val[0, tgt["pos"]]
                    tgt_sae = sae_models[tgt["block_idx"]]
                    dec_vec = tgt_sae.decoder.weight[:, tgt["feat"]].detach()
                    dec_vec = dec_vec * tgt["act_val"]
                    
                    edge = torch.dot(dec_vec.to(grad_at_pos.dtype), grad_at_pos).item()
                    # 邻接矩阵: [target, source]
                    adj[tgt_idx, src_idx] = edge
                
                # Error edges: source → error_node at (layer_idx, pos)
                for pos in active_positions:
                    local_pos = pos_to_local[pos]
                    error_node_idx = error_offset + layer_idx * n_active_pos + local_pos
                    
                    if error_node_idx < total_nodes and layer_idx < len(output.error_vectors):
                        err_vec = output.error_vectors[layer_idx][0, pos]  # (d_model,)
                        grad_at_pos = grad_val[0, pos]
                        edge = torch.dot(err_vec.to(grad_at_pos.dtype), grad_at_pos).item()
                        adj[error_node_idx, src_idx] = edge
            
            elif node_type == 'embed':
                # Token edges: source → embed_node
                for pos in active_positions:
                    local_pos = pos_to_local[pos]
                    token_node_idx = token_offset + local_pos
                    
                    tok_vec = output.input_embeddings[0, pos].detach()
                    grad_at_pos = grad_val[0, pos]
                    edge = torch.dot(tok_vec.to(grad_at_pos.dtype), grad_at_pos).item()
                    adj[token_node_idx, src_idx] = edge
    
    # ====== Logit edges: logit → feature ======
    # 对每个 logit node, 注入 demeaned unembedding vec 做 VJP
    # logit 节点的 layer = n_layers (最后一层之后), position = last_pos
    last_resid = output.resid_activations[-1] if output.resid_activations else None
    
    if last_resid is not None and last_resid.requires_grad and n_logits > 0:
        for logit_i in range(n_logits):
            logit_vec = logit_vecs[logit_i].to(last_resid.dtype).to(device)  # (d_model,)
            
            inject_grad = torch.zeros_like(last_resid)
            # 注入在最后一个 position
            inject_grad[0, -1] = logit_vec
            
            # 收集所有 resid 的梯度
            resid_inputs = []
            resid_layers = []
            for li in range(len(output.resid_activations)):
                r = output.resid_activations[li]
                if r is not None and r.requires_grad:
                    resid_inputs.append(r)
                    resid_layers.append(li)
            
            # 加入 input_embeddings
            if output.input_embeddings.requires_grad:
                resid_inputs.append(output.input_embeddings)
                resid_layers.append(-1)  # sentinel for embed
            
            if not resid_inputs:
                continue
            
            try:
                grads = torch.autograd.grad(
                    outputs=last_resid,
                    inputs=resid_inputs,
                    grad_outputs=inject_grad,
                    retain_graph=True,
                    allow_unused=True,
                )
            except RuntimeError:
                continue
            
            logit_node_idx = logit_offset + logit_i
            
            for grad_val, layer_idx in zip(grads, resid_layers):
                if grad_val is None:
                    continue
                
                if layer_idx == -1:
                    # logit → token edges
                    for pos in active_positions:
                        local_pos = pos_to_local[pos]
                        tok_idx = token_offset + local_pos
                        tok_vec = output.input_embeddings[0, pos].detach()
                        edge = torch.dot(tok_vec.to(grad_val.dtype), grad_val[0, pos]).item()
                        adj[logit_node_idx, tok_idx] = edge
                else:
                    # logit → feature edges
                    for feat_idx, feat in enumerate(node_pool):
                        if feat["layer"] != layer_idx:
                            continue
                        feat_sae = sae_models[feat["block_idx"]]
                        dec_vec = feat_sae.decoder.weight[:, feat["feat"]].detach()
                        dec_vec = dec_vec * feat["act_val"]
                        edge = torch.dot(dec_vec.to(grad_val.dtype), grad_val[0, feat["pos"]]).item()
                        adj[logit_node_idx, feat_idx] = edge
                    
                    # logit → error edges
                    for pos in active_positions:
                        local_pos = pos_to_local[pos]
                        err_idx = error_offset + layer_idx * n_active_pos + local_pos
                        if err_idx < total_nodes and layer_idx < len(output.error_vectors):
                            err_vec = output.error_vectors[layer_idx][0, pos]
                            edge = torch.dot(err_vec.to(grad_val.dtype), grad_val[0, pos]).item()
                            adj[logit_node_idx, err_idx] = edge
    
    # 释放显存
    wrapper.zero_grad()
    
    # ====== 打包 (对齐 CodeCircuit Graph 字段) ======
    graph_data = {
        # CodeCircuit-compatible fields
        "adjacency_matrix": adj.cpu(),
        "active_features": active_features,           # (n_selected, 3): (layer, pos, feat_idx)
        "selected_features": selected_features,        # (n_selected,): indices into active_features
        "activation_values": activation_values,         # (n_selected,)
        "logit_tokens": logit_indices.cpu(),            # (k,) vocab ids
        "logit_probabilities": logit_probs.cpu(),       # (k,)
        # Metadata
        "n_layers": N_VIRTUAL_LAYERS,
        "n_pos": n_active_pos,
        "n_selected_features": n_selected,
        "n_error_nodes": n_errors,
        "n_token_nodes": n_tokens,
        "n_logit_nodes": n_logits,
        "active_positions": torch.tensor(active_positions, dtype=torch.long),
        # Extra
        "loss": loss_val,
        "attribution_scores": attribution_scores,
    }
    
    # 保存 query 身份标识 (用于与 HyperNet 输入对齐)
    if query_meta is not None:
        graph_data["query_meta"] = query_meta
    
    return graph_data


def main():
    parser = argparse.ArgumentParser(description="Step 3: Extract Attribution Graphs")
    RunConfig.add_run_args(parser)
    parser.add_argument("--config_path", type=str, default="config/cfg_wu4trm.yaml")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        default=["data/arc1concept-aug-1000"])
    parser.add_argument("--max_queries", type=int, default=-1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--use_last_step", action="store_true", default=True)
    args = parser.parse_args()
    
    rc = RunConfig(run_name=args.run_name)
    rc.print_summary()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(rc.graphs_dir, exist_ok=True)
    
    print("Loading Dual SAE Transcoders...")
    sae_models = load_dual_sae(rc.sae_block_0_path, rc.sae_block_1_path, device)
    
    print("Loading TRM Base Model...")
    trm_model = load_trm_model(args.config_path, args.ckpt_path, device)
    
    print("Assembling UnrolledTRMWrapper...")
    wrapper = UnrolledTRMWrapper(trm_model.inner, sae_models)
    wrapper.to(device)
    wrapper.eval()
    
    print("Loading Dataset...")
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    dataset_cfg = PuzzleDatasetConfig(
        seed=42, dataset_paths=args.dataset_paths,
        global_batch_size=1, test_set_mode=(args.split == "test"),
        epochs_per_iter=1, rank=0, num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_cfg, split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    
    halt_max_steps = trm_model.config.halt_max_steps
    print(f"\nStarting Attribution (split={args.split}, use_last_step={args.use_last_step})...")
    count = 0
    
    for set_name, batch, effective_batch_size in tqdm(dataloader, desc="Extracting"):
        if args.max_queries > 0 and count >= args.max_queries:
            break
        
        try:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            
            # 构造 query 身份标识 (供 HyperNet 对齐用)
            query_meta = {
                "set_name": set_name,
                "graph_index": count,
                "puzzle_identifiers": batch["puzzle_identifiers"].cpu(),
                "inputs": batch["inputs"].cpu(),
                "labels": batch["labels"].cpu(),
            }
            
            init_z_H, init_z_L = None, None
            if args.use_last_step:
                init_z_H, init_z_L = run_act_preflight(
                    trm_model, batch_gpu, device, halt_max_steps
                )
            
            graph_data = extract_query_graph(
                wrapper, sae_models, batch, device,
                init_z_H=init_z_H, init_z_L=init_z_L,
                query_meta=query_meta,
            )
            
            save_path = os.path.join(rc.graphs_dir, f"graph_{count:06d}.pt")
            torch.save(graph_data, save_path)
            
            if count % 50 == 0:
                print(f"\n  Query {count} [{set_name}]: loss={graph_data['loss']:.4f}, "
                      f"features={graph_data['n_selected_features']}, "
                      f"errors={graph_data['n_error_nodes']}, "
                      f"tokens={graph_data['n_token_nodes']}, "
                      f"logits={graph_data['n_logit_nodes']}, "
                      f"adj={list(graph_data['adjacency_matrix'].shape)}")
            
            count += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM at query {count}. Skipping...")
                torch.cuda.empty_cache()
                continue
            raise
    
    rc.save_config(extra_info={
        "step": "3_extract_attribution",
        "ckpt_path": args.ckpt_path,
        "dataset_paths": args.dataset_paths,
        "max_queries": args.max_queries,
        "split": args.split,
        "n_graphs": count,
    })
    print(f"\n✅ Attribution extraction complete! {count} graphs saved to {rc.graphs_dir}")


if __name__ == "__main__":
    main()
