import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActivationDataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.safetensors")))
        if not self.chunk_files:
            raise ValueError(f"No .safetensors files found in {data_dir}")
            
        print(f"Found {len(self.chunk_files)} activation chunks.")
        # Load the first chunk just to get the shape
        sample = load_file(self.chunk_files[0])["activations"]
        self.chunk_size = sample.shape[0]
        self.total_samples = len(self.chunk_files) * self.chunk_size

        self.current_chunk_idx = -1
        self.current_chunk_data = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if chunk_idx != self.current_chunk_idx:
            # We assume sequential loading is mostly handled by keeping one chunk in RAM
            # In a true dataloader you'd want an IterableDataset or chunk caching.
            # This is simplified for Phase 0 MVP.
            self.current_chunk_data = load_file(self.chunk_files[chunk_idx])["activations"].to(torch.float32)
            self.current_chunk_idx = chunk_idx

        return self.current_chunk_data[local_idx]


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in=512, d_sae=4096):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        
        # SAE traditionally uses tied or untied weights. We use untied as per CodeCircuit defaults.
        # W_enc: (d_in, d_sae)
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        # W_dec: (d_sae, d_in)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)  # Decoder often has no bias
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Initialize
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def encode(self, x):
        """ Returns the activated SAE features (f(x)) """
        return torch.relu(self.encoder(x))

    def decode(self, f):
        """ Reconstructs x from features f """
        return self.decoder(f) + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        return x_reconstructed, f


def train_sae():
    data_dir = "CodeCircuit_TRM_Arc1/results/activations"
    dataset = ActivationDataset(data_dir)
    # Use a large batch size for SAE training
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
    
    sae = SparseAutoencoder(d_in=512, d_sae=4096).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    # Loss coefficients
    l1_coeff = 1e-3  # Adjust sparsity penalty. Increase if graph is too dense.
    
    epochs = 2
    for epoch in range(epochs):
        sae.train()
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
                
            # Mean-center the inputs (optional but recommended in SAEs)
            x_centered = batch - sae.b_dec
                
            x_hat, f = sae(batch) # Original uncentered as input
            
            # 1. MSE (Reconstruction Loss)
            mse_loss = torch.nn.functional.mse_loss(x_hat, batch)
            
            # 2. L1 (Sparsity Loss)
            l1_loss = f.abs().sum(dim=-1).mean()
            
            loss = mse_loss + l1_coeff * l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            
            pbar.set_postfix({"mse": f"{mse_loss.item():.4f}", "l1": f"{l1_loss.item():.2f}"})
            
    # Save the transcoder exactly where the rest of the pipeline expects it
    os.makedirs("CodeCircuit_TRM_Arc1/checkpoints", exist_ok=True)
    out_path = "CodeCircuit_TRM_Arc1/checkpoints/trm_transcoder_4096.pt"
    torch.save(sae.state_dict(), out_path)
    print(f"\nSAE Training complete! Transcoder saved to: {out_path}")


if __name__ == "__main__":
    train_sae()
