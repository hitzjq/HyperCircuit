import torch
import torch.nn as nn
import math
from einops import rearrange
from models.layers import CastedLinear

class PGTokenizer(nn.Module):
    def __init__(
        self,
        module_specs: list,  # List of (module_name, out_features, in_features)
        rank: int = 16,
        token_dim: int = 512,
        dim_wo_acc: int = 64,
        dim_acc: int = 4,
        lora_B_zero_init: bool = True
    ):
        super().__init__()
        self.module_specs = module_specs
        self.rank = rank
        self.token_dim = token_dim
        self.dim_wo_acc = dim_wo_acc
        self.dim_acc = dim_acc
        self.d_model = dim_wo_acc * dim_acc
        self.token_area = rank * token_dim

        self.mapping_A = []
        self.mapping_B = []
        self.total_virtual_tokens = 0
        
        self._build_mapping()

        # Learnable empty positions
        self.token_pos_emb = nn.Embedding(self.total_virtual_tokens * self.rank, self.dim_wo_acc)
        # Using trunc_normal usually works, but standard init is okay too
        nn.init.normal_(self.token_pos_emb.weight, std=0.02)

        # Projections out
        self.proj_out_A = CastedLinear(self.dim_wo_acc, self.token_dim, bias=False)
        self.proj_out_B = CastedLinear(self.dim_wo_acc, self.token_dim, bias=False)
        
        if lora_B_zero_init:
            nn.init.zeros_(self.proj_out_B.weight)

    def _build_mapping(self):
        """
        Dynamically calculates the number of virtual tokens needed for each module's
        lora_A (in_features * rank) and lora_B (out_features * rank).
        """
        for name, out_features, in_features in self.module_specs:
            # lora_A: [rank, in_features]
            params_A = self.rank * in_features
            tokens_A = math.ceil(params_A / self.token_area)
            self.mapping_A.append({
                "name": name,
                "shape": (self.rank, in_features),
                "num_params": params_A,
                "token_start": self.total_virtual_tokens,
                "token_count": tokens_A
            })
            self.total_virtual_tokens += tokens_A

            # lora_B: [out_features, rank]
            params_B = out_features * self.rank
            tokens_B = math.ceil(params_B / self.token_area)
            self.mapping_B.append({
                "name": name,
                "shape": (out_features, self.rank),
                "num_params": params_B,
                "token_start": self.total_virtual_tokens,
                "token_count": tokens_B
            })
            self.total_virtual_tokens += tokens_B

    def get_initial_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns the initial packed Transformer tokens: [B, num_transformer_tokens, d_model]
        """
        # [total_virtual_tokens * rank, dim_wo_acc]
        all_embs = self.token_pos_emb.weight.to(dtype=dtype)
        # Pack to d_model: [total_virtual_tokens * (rank // dim_acc), dim_wo_acc * dim_acc]
        # rank must be divisible by dim_acc
        packed = rearrange(all_embs, '(v r_groups acc) d -> (v r_groups) (acc d)', 
                           v=self.total_virtual_tokens,
                           acc=self.dim_acc, 
                           d=self.dim_wo_acc)
        
        # repeat for batch
        packed = packed.unsqueeze(0).expand(batch_size, -1, -1)
        return packed

    def detokenize(self, transformer_out: torch.Tensor, scale=2.0) -> dict:
        """
        Unpacks the transformer output and slices it into LoRA state dict.
        transformer_out: [B, num_transformer_tokens, d_model]
        """
        B = transformer_out.shape[0]

        # 1. Unpack
        # [B, num_transformer_tokens, dim_acc * dim_wo_acc] -> [B, total_virtual, rank, dim_wo_acc]
        unpacked = rearrange(transformer_out, 'b (v r_groups) (acc d) -> b v (r_groups acc) d',
                             v=self.total_virtual_tokens,
                             acc=self.dim_acc,
                             d=self.dim_wo_acc)

        # 2. Project
        out_A = self.proj_out_A(unpacked) # [B, total_virtual, rank, token_dim]
        out_B = self.proj_out_B(unpacked) # [B, total_virtual, rank, token_dim]

        # 3. Slice and Reshape into LoRA matrices
        lora_dict = {"scale": scale}

        def process_mapping(mapping, out_tensor, is_lora_a):
            for m in mapping:
                start = m["token_start"]
                count = m["token_count"]
                
                # slice tokens specific to this module
                token_slice = out_tensor[:, start : start + count, :, :] # [B, count, rank, token_dim]
                
                # flatten the slice entirely keeping batch dimension
                flat_slice = token_slice.reshape(B, -1)
                
                # take only the needed elements to remove padding
                needed_elements = m["num_params"]
                flat_slice = flat_slice[:, :needed_elements]
                
                # reshape back to desired LoRA matrix size
                reshaped = flat_slice.view(B, *m["shape"])
                
                # Register in dict
                key_prefix = m["name"]
                key_suffix = "lora_A" if is_lora_a else "lora_B"
                full_key = f"{key_prefix}.{key_suffix}"
                lora_dict[full_key] = reshaped

        process_mapping(self.mapping_A, out_A, is_lora_a=True)
        process_mapping(self.mapping_B, out_B, is_lora_a=False)

        return lora_dict
