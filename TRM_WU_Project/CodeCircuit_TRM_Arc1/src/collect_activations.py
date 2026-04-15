import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from run_config import RunConfig

# Dynamic path resolution to import TRM_WU_Project and TinyRecursiveModels
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Points to TRM_WU_Project
trm_root = project_root.parent / "TinyRecursiveModels"

sys.path.append(str(project_root))
sys.path.append(str(trm_root))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# TRM 架构参数（来自 ARC-AGI-1 checkpoint）
# H_cycles = 3, L_cycles = 4, L_layers = 2
# 每次 L_level 调用经过 2 个 Block（Block 0 和 Block 1）
# 每次 Inner forward: 15 次 L_level × 2 个 Block = 30 次 MLP 调用
# ACT 最多 16 步: 16 × 30 = 480 次 MLP 调用（全部收集，其中按Block分存）
# -------------------------------------------------------------------

# Global buffers — 按 Block 分类收集
block_0_buffer = []  # Block 0 (偶数 index) 的 MLP 输出
block_1_buffer = []  # Block 1 (奇数 index) 的 MLP 输出
save_chunk_idx_0 = 0
save_chunk_idx_1 = 0
call_counter = 0
BUFFER_MAX_ITEMS = 3000  # 每积累 ~3000 个 MLP 输出就 flush 一次（可调）
_run_config = None  # 全局 RunConfig, main() 中初始化


def make_block_hooks(model):
    """
    为 Block 0 和 Block 1 的 mlp.down_proj 分别注册 hook。
    
    TRM 的 L_level 是一个包含 L_layers=2 个 Block 的 ReasoningModule。
    layers[0] = Block 0, layers[1] = Block 1。
    每次 L_level 被调用时，依次经过 Block 0 → Block 1。
    """
    hooks = []
    
    for block_idx, layer in enumerate(model.inner.L_level.layers):
        def hook_fn(module, input, output, _block_idx=block_idx):
            global call_counter
            flat_out = output.detach().flatten(0, 1).to(torch.bfloat16)
            
            if _block_idx == 0:
                block_0_buffer.append(flat_out.cpu())
            else:
                block_1_buffer.append(flat_out.cpu())
            
            call_counter += 1
        
        h = layer.mlp.down_proj.register_forward_hook(hook_fn)
        hooks.append(h)
    
    return hooks


def flush_buffer(block_idx, force=False):
    """Flush 指定 Block 的 buffer 到磁盘"""
    global save_chunk_idx_0, save_chunk_idx_1
    
    if block_idx == 0:
        buf = block_0_buffer
        chunk_idx = save_chunk_idx_0
        out_subdir = "block_0"
    else:
        buf = block_1_buffer
        chunk_idx = save_chunk_idx_1
        out_subdir = "block_1"
    
    if not force and len(buf) < BUFFER_MAX_ITEMS:
        return
    
    if len(buf) == 0:
        return
    
    merged_tensor = torch.cat(buf, dim=0)  # [N_total_tokens, hidden_size]
    
    # Shuffle to break temporal correlations
    perm = torch.randperm(merged_tensor.size(0))
    merged_tensor = merged_tensor[perm]
    
    out_dir = _run_config.block_0_dir if block_idx == 0 else _run_config.block_1_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"chunk_{chunk_idx:04d}.safetensors")
    save_file({"activations": merged_tensor}, out_path)
    
    print(f"  [Block {block_idx}] Saved {merged_tensor.shape[0]} tokens to {out_path}")
    
    buf.clear()
    
    if block_idx == 0:
        save_chunk_idx_0 += 1
    else:
        save_chunk_idx_1 += 1


def main():
    global _run_config
    parser = argparse.ArgumentParser(description="Step 1: Collect TRM internal activations for SAE training")
    RunConfig.add_run_args(parser)
    parser.add_argument("--dataset_paths", nargs='+', default=['data/arc1concept-aug-1000'], help="Paths to dataset folders")
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to TRM base checkpoint. Leave empty for random init (A100 debug).")
    parser.add_argument("--max_batches", type=int, default=-1, help="Max batches to evaluate. -1 for full dataset (H200).")
    args = parser.parse_args()
    
    _run_config = RunConfig(run_name=args.run_name)
    _run_config.create_dirs()
    _run_config.print_summary()
    _run_config.save_config(extra_info={
        "step": "1_collect_activations",
        "ckpt_path": args.ckpt_path,
        "dataset_paths": args.dataset_paths,
        "max_batches": args.max_batches,
    })

    print("Loading Dataset from:", args.dataset_paths)
    dataset_cfg = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=args.dataset_paths,
        global_batch_size=2,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_cfg, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    metadata = dataset.metadata

    print("Loading Base TRM Config...")
    model_cfg = {
        "H_cycles": 3,
        "L_cycles": 4,
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

    print("Initializing Model...")
    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    model = model_cls(model_cfg).to(device)
    
    # Load base checkpoint if provided
    base_ckpt_path = args.ckpt_path
    actual_load_file = base_ckpt_path

    if base_ckpt_path and os.path.isdir(base_ckpt_path):
        candidates = [f for f in os.listdir(base_ckpt_path) if f.startswith("step_")]
        if candidates:
            actual_load_file = os.path.join(base_ckpt_path, candidates[0])

    if actual_load_file and os.path.exists(actual_load_file):
        print(f"✅ Loading TRUE Base Weights from {actual_load_file}...")
        sd = torch.load(actual_load_file, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
    else:
        print(f"⚠️ Warning: No valid checkpoint found at '{base_ckpt_path}'. Using RANDOM weights for debugging!")

    model.eval()

    print("Registering Hooks on mlp.down_proj (Block 0 and Block 1 separately)...")
    hooks = make_block_hooks(model)

    print("Starting Multi-Step ACT Inference and Collection...")
    global call_counter
    halt_max_steps = model_cfg["halt_max_steps"]  # 16
    
    with torch.no_grad():
        for i, (set_name, batch, global_batch_size) in enumerate(tqdm(dataloader, desc="Collecting")):
            if args.max_batches > 0 and i >= args.max_batches: 
                print(f"\nReached max_batches limit ({args.max_batches}). Stopping collection.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 初始化 carry
            with torch.device(device):
                carry = model.initial_carry(batch)
            
            # 多步 ACT 推理循环 — 每步触发 30 次 hook（15 L_level × 2 Block）
            for act_step in range(halt_max_steps):
                carry, outputs = model(carry, batch)
                
                # 检查是否所有序列都已 halt
                if carry.halted.all():
                    break
            
            # Flush buffer if needed
            flush_buffer(0)
            flush_buffer(1)

    # Save remaining
    flush_buffer(0, force=True)
    flush_buffer(1, force=True)
    
    for h in hooks:
        h.remove()
        
    print(f"\nFinished. Total hooks triggered: {call_counter}")
    print(f"  Block 0 chunks: {save_chunk_idx_0}")
    print(f"  Block 1 chunks: {save_chunk_idx_1}")


if __name__ == "__main__":
    main()
