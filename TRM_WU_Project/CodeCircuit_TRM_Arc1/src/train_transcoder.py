import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActivationDataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.safetensors")))
        if not self.chunk_files:
            raise ValueError(f"No .safetensors files found in {data_dir}")
            
        print(f"Found {len(self.chunk_files)} activation chunks.")
        
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
            # We assume sequential loading is mostly handled by keeping one chunk in RAM
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


def train_sae(args):
    dataset = ActivationDataset(args.activations_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    sae = SparseAutoencoder(d_in=args.d_in, d_sae=args.d_sae).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    
    l1_coeff = args.l1_coeff
    
    for epoch in range(args.epochs):
        sae.train()
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
                
            x_centered = batch - sae.b_dec
                
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
                print(f"Step {step+1} | L1: {l1_loss.item():.4f} | MSE: {mse_loss.item():.4f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(sae.state_dict(), args.save_path)
    print(f"✅ Basic SAE dictionary saved to {args.save_path}")
    
    # 适配 CodeCircuit 的原生 VJP 必须伪装成 CLT 落盘
    base_dir = os.path.dirname(args.save_path)
    export_clt_safetensors(sae, base_dir=base_dir, n_layers=42)

def export_clt_safetensors(sae, base_dir, n_layers=42):
    from safetensors.torch import save_file
    
    out_dir = os.path.join(base_dir, "trm_cross_layer_transcoder")
    os.makedirs(out_dir, exist_ok=True)
    
    w_enc = sae.encoder.weight.detach().cpu()
    b_enc = sae.encoder.bias.detach().cpu()
    w_dec_base = sae.decoder.weight.detach().cpu().t() # (4096, 512)
    b_dec = sae.b_dec.detach().cpu()
    
    for i in range(n_layers):
        enc_dict = {
            f"b_dec_{i}": b_dec,
            f"b_enc_{i}": b_enc,
            f"W_enc_{i}": w_enc
        }
        save_file(enc_dict, os.path.join(out_dir, f"W_enc_{i}.safetensors"))
        
        w_dec_i = w_dec_base.unsqueeze(1).repeat(1, n_layers - i, 1)
        dec_dict = {
            f"W_dec_{i}": w_dec_i
        }
        save_file(dec_dict, os.path.join(out_dir, f"W_dec_{i}.safetensors"))
        
    print(f"✅ CodeCircuit-Compatible CLT exported to {out_dir}/ with {n_layers} virtual layers!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE) for TRM")
    parser.add_argument("--activations_dir", type=str, default="CodeCircuit_TRM_Arc1/results/activations", help="Directory containing activation safetensors")
    parser.add_argument("--save_path", type=str, default="CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096.pt", help="Path to save the SAE model")
    parser.add_argument("--d_in", type=int, default=512, help="Input feature dimension")
    parser.add_argument("--d_sae", type=int, default=4096, help="SAE projected sparse dimension")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 penalty coefficient for sparsity")
    
    args = parser.parse_args()
    train_sae(args)

