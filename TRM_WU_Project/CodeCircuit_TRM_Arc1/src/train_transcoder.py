"""
Train 2 Sparse Autoencoders (SAE) for TRM — one per physical Block.

Block 0 (layers[0]) → SAE_0
Block 1 (layers[1]) → SAE_1

Each SAE learns to decompose its Block's MLP output into sparse features.
After training, both SAEs are exported in CodeCircuit-compatible CLT format:
  - Even virtual layers (0,2,4,...) use SAE_0's weights
  - Odd virtual layers (1,3,5,...) use SAE_1's weights
"""

import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import argparse
from tqdm import tqdm
from run_config import RunConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivationDataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.safetensors")))
        if not self.chunk_files:
            raise ValueError(f"No .safetensors files found in {data_dir}")
            
        print(f"Found {len(self.chunk_files)} activation chunks in {data_dir}")
        
        import safetensors
        self.chunk_sizes = []
        for file in self.chunk_files:
            with safetensors.safe_open(file, framework="pt") as f:
                self.chunk_sizes.append(f.get_slice("activations").get_shape()[0])
                
        self.total_samples = sum(self.chunk_sizes)
        
        import itertools
        self.cumulative_sizes = list(itertools.accumulate(self.chunk_sizes))

        self.current_chunk_idx = -1
        self.current_chunk_data = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        import bisect
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if chunk_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[chunk_idx - 1]

        if chunk_idx != self.current_chunk_idx:
            self.current_chunk_data = load_file(self.chunk_files[chunk_idx])["activations"].to(torch.float32)
            self.current_chunk_idx = chunk_idx

        return self.current_chunk_data[local_idx]


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in=512, d_sae=4096):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        
        # W_enc: (d_in, d_sae)
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        # W_dec: (d_sae, d_in)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, f):
        return self.decoder(f) + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        return x_reconstructed, f


def train_single_sae(data_dir, save_path, args, block_name="block"):
    """Train one SAE on the given activation directory."""
    print(f"\n{'='*60}")
    print(f"Training SAE for {block_name}")
    print(f"  Data: {data_dir}")
    print(f"  Save: {save_path}")
    print(f"{'='*60}")
    
    dataset = ActivationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    sae = SparseAutoencoder(d_in=args.d_in, d_sae=args.d_sae).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    
    l1_coeff = args.l1_coeff
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(args.epochs):
        sae.train()
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        
        pbar = tqdm(dataloader, desc=f"[{block_name}] Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
                
            x_hat, f = sae(batch) 
            
            mse_loss = torch.nn.functional.mse_loss(x_hat, batch)
            l1_loss = f.abs().sum(dim=-1).mean()
            loss = mse_loss + l1_coeff * l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            
            pbar.set_postfix({"mse": f"{mse_loss.item():.4f}", "l1": f"{l1_loss.item():.2f}"})
            
            if (step + 1) % 1000 == 0:
                print(f"  [{block_name}] Step {step+1} | L1: {l1_loss.item():.4f} | MSE: {mse_loss.item():.4f}")
        
        num_steps = max(len(dataloader), 1)
        print(
            f"  [{block_name}] Epoch {epoch + 1}/{args.epochs} complete | "
            f"avg_loss={total_loss / num_steps:.4f} | "
            f"avg_mse={total_mse / num_steps:.4f} | "
            f"avg_l1={total_l1 / num_steps:.4f}"
        )
        
        if args.save_every_epoch:
            torch.save(sae.state_dict(), save_path)
            print(f"  [{block_name}] Saved latest checkpoint after epoch {epoch + 1}: {save_path}")

    torch.save(sae.state_dict(), save_path)
    print(f"✅ {block_name} SAE saved to {save_path}")
    
    return sae


def export_dual_clt_safetensors(sae_0, sae_1, base_dir, n_layers=30):
    """
    Export 2 SAEs in CodeCircuit-compatible CLT format.
    
    Virtual layer mapping:
      - Even layers (0, 2, 4, ...) → Block 0 → SAE_0
      - Odd layers  (1, 3, 5, ...) → Block 1 → SAE_1
    """
    from safetensors.torch import save_file
    
    out_dir = os.path.join(base_dir, "trm_cross_layer_transcoder")
    os.makedirs(out_dir, exist_ok=True)
    
    # Pre-extract weights
    saes = [sae_0, sae_1]
    w_encs = [sae.encoder.weight.detach().cpu() for sae in saes]  # (4096, 512)
    b_encs = [sae.encoder.bias.detach().cpu() for sae in saes]    # (4096,)
    w_dec_bases = [sae.decoder.weight.detach().cpu().t() for sae in saes]  # (4096, 512)
    b_decs = [sae.b_dec.detach().cpu() for sae in saes]           # (512,)
    
    for i in range(n_layers):
        # Block 0 → even layers, Block 1 → odd layers
        block_idx = i % 2
        
        enc_dict = {
            f"b_dec_{i}": b_decs[block_idx],
            f"b_enc_{i}": b_encs[block_idx],
            f"W_enc_{i}": w_encs[block_idx]
        }
        save_file(enc_dict, os.path.join(out_dir, f"W_enc_{i}.safetensors"))
        
        w_dec_i = w_dec_bases[block_idx].unsqueeze(1).repeat(1, n_layers - i, 1)
        dec_dict = {
            f"W_dec_{i}": w_dec_i
        }
        save_file(dec_dict, os.path.join(out_dir, f"W_dec_{i}.safetensors"))
        
    print(f"✅ Dual-SAE CLT exported to {out_dir}/ with {n_layers} virtual layers")
    print(f"   Even layers → SAE_0 (Block 0), Odd layers → SAE_1 (Block 1)")


def train_sae(args):
    """Train 2 SAEs (one per Block) and export as CLT."""
    rc = RunConfig(run_name=args.run_name)
    rc.print_summary()
    
    block_0_dir = rc.block_0_dir
    block_1_dir = rc.block_1_dir
    
    # Validate directories exist
    if not os.path.isdir(block_0_dir):
        raise FileNotFoundError(f"Block 0 activations not found: {block_0_dir}")
    if not os.path.isdir(block_1_dir):
        raise FileNotFoundError(f"Block 1 activations not found: {block_1_dir}")
    
    # Train SAE for Block 0
    sae_0 = train_single_sae(block_0_dir, rc.sae_block_0_path, args, block_name="Block_0")
    
    # Train SAE for Block 1
    sae_1 = train_single_sae(block_1_dir, rc.sae_block_1_path, args, block_name="Block_1")
    
    # Export dual CLT
    export_dual_clt_safetensors(sae_0, sae_1, base_dir=rc.checkpoints_dir, n_layers=args.n_layers)
    
    # 更新 config.json
    rc.save_config(extra_info={
        "step": "2_train_transcoder",
        "d_sae": args.d_sae,
        "epochs": args.epochs,
        "lr": args.lr,
        "l1_coeff": args.l1_coeff,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Train Dual Sparse Autoencoder (SAE) for TRM")
    RunConfig.add_run_args(parser)
    parser.add_argument("--d_in", type=int, default=512, help="Input feature dimension")
    parser.add_argument("--d_sae", type=int, default=4096, help="SAE projected sparse dimension")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 penalty coefficient for sparsity")
    parser.add_argument("--n_layers", type=int, default=30, help="Number of virtual layers")
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="Overwrite the latest SAE checkpoint after each epoch so long runs can be stopped early.",
    )
    
    args = parser.parse_args()
    train_sae(args)
