import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file

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

# Global counter and buffers
call_counter = 0
SLICES_PER_FORWARD = 30 # H_cycles(3) * (L_cycles(4)+1) * L_layers(2) = 30
activation_buffer = [] # format: list of [total_tokens, hidden_size]
save_chunk_idx = 0
BUFFER_MAX_SIZE = 100 * SLICES_PER_FORWARD # Adjust to fit your RAM/VRAM

def mlp_hook(module, input, output):
    """
    Hook to capture the output of mlp.down_proj.
    output shape: [batch_size, seq_len, hidden_size]
    """
    global call_counter
    # Flatten across batch and seq_len -> [batch * seq_len, hidden_size]
    flat_out = output.detach().flatten(0, 1).to(torch.bfloat16)
    
    # Optional: we can keep memory of which slice it came from if we want separate SAEs,
    # but CodeCircuit trains a Universal SAE across all layers by mixing them.
    # We will mix all slice activations together for the SAE.
    activation_buffer.append(flat_out.cpu())
    
    call_counter += 1

def flush_buffer(force=False):
    global activation_buffer, save_chunk_idx
    if not force and len(activation_buffer) < BUFFER_MAX_SIZE:
        return
    
    if len(activation_buffer) == 0:
        return
        
    print(f"Flushing chunk {save_chunk_idx} to disk...")
    # Concatenate all gathered flatten activations
    merged_tensor = torch.cat(activation_buffer, dim=0) # [N_total_tokens, hidden_size]
    
    # Shuffle the tokens to break temporal correlations for SAE training
    perm = torch.randperm(merged_tensor.size(0))
    merged_tensor = merged_tensor[perm]
    
    os.makedirs("CodeCircuit_TRM_Arc1/results/activations", exist_ok=True)
    out_path = f"CodeCircuit_TRM_Arc1/results/activations/chunk_{save_chunk_idx}.safetensors"
    save_file({"activations": merged_tensor}, out_path)
    
    print(f"Saved {merged_tensor.shape[0]} tokens to {out_path}.")
    
    activation_buffer.clear()
    save_chunk_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Collect TRM internal activations for SAE training")
    parser.add_argument("--dataset_paths", nargs='+', default=['data/arc1concept-aug-1000'], help="Paths to dataset folders")
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to TRM base checkpoint. Leave empty for random init (A100 debug).")
    parser.add_argument("--max_batches", type=int, default=-1, help="Max batches to evaluate. -1 for full dataset (H200).")
    args = parser.parse_args()

    print("Loading Dataset from:", args.dataset_paths)
    # Setup dataset mimicking TRM config
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
    # Hardcoded base parameters reflecting `trm.yaml`
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

    # If the user passed a directory like ARC-AGI-1/, find the 'step_*' file inside it.
    if base_ckpt_path and os.path.isdir(base_ckpt_path):
        candidates = [f for f in os.listdir(base_ckpt_path) if f.startswith("step_")]
        if candidates:
            # Taking the first one if multiple exist, though typically there's only one.
            actual_load_file = os.path.join(base_ckpt_path, candidates[0])

    if actual_load_file and os.path.exists(actual_load_file):
        print(f"✅ Loading TRUE Base Weights from {actual_load_file}...")
        sd = torch.load(actual_load_file, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
    else:
        print(f"⚠️ Warning: No valid checkpoint found at '{base_ckpt_path}'. Using RANDOM weights for debugging!")

    model.eval()

    print("Registering Hooks on mlp.down_proj...")
    hooks = []
    # In TRM, SwiGLU down_proj is what we target
    for layer in model.inner.L_level.layers:
        h = layer.mlp.down_proj.register_forward_hook(mlp_hook)
        hooks.append(h)

    print("Starting Inference and Collection...")
    global call_counter
    
    with torch.no_grad():
        for i, (set_name, batch, global_batch_size) in enumerate(tqdm(dataloader, desc="Collecting")):
            # Halt early if running a quick debug on A100
            if args.max_batches > 0 and i >= args.max_batches: 
                print(f"\nReached max_batches limit ({args.max_batches}). Stopping collection.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 使用 Device Context 强行逼迫 TRM 内部所有自动申请的张量都长在显卡上，彻底断绝 CPU 内漏
            with torch.device(device):
                carry = model.initial_carry(batch)
            
            # Forward pass (this will automatically hit the hooks 30 times per sample).
            model(carry, batch)
            
            # Check if buffer is full and save
            flush_buffer()

    # Save remaining
    flush_buffer(force=True)
    
    for h in hooks:
        h.remove()
        
    print(f"Finished. Total hooks triggered: {call_counter}")


if __name__ == "__main__":
    main()
