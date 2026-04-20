import torch
import torch.nn as nn
from typing import Optional
from .pg_tokenizer import PGTokenizer
from .pg_transformer import PGTransformer
from models.layers import CastedLinear

class ParameterGenerator(nn.Module):
    def __init__(
        self,
        module_specs: list,
        d_model: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        cond_dim: int = 512,
        rank: int = 16,
        token_dim: int = 512,
        dim_acc: int = 4,
        lora_B_zero_init: bool = True,
        use_rope: bool = False,
        circuit_dim: int = 0,  # 0 = 不使用电路特征, 53 = 使用 CodeCircuit 53维特征
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_wo_acc = d_model // dim_acc
        self.cond_dim = cond_dim

        # 1. Condition Projection
        # Project base TRM hidden_size (e.g. 512) to PG d_model (256)
        if cond_dim != d_model:
            self.cond_proj = CastedLinear(cond_dim, d_model, bias=False)
        else:
            self.cond_proj = nn.Identity()

        # 1.5 Circuit Feature Projection (可训练，随 PG 一起更新)
        self.use_circuit = circuit_dim > 0
        if self.use_circuit:
            self.circuit_proj = nn.Sequential(
                CastedLinear(circuit_dim, d_model, bias=False),
                nn.SiLU(),
                CastedLinear(d_model, d_model, bias=False),
            )

        # 2. Tokenizer (handles positional embeddings, packing, unpacking, detokenizing)
        self.tokenizer = PGTokenizer(
            module_specs=module_specs,
            rank=rank,
            token_dim=token_dim,
            dim_wo_acc=self.dim_wo_acc,
            dim_acc=dim_acc,
            lora_B_zero_init=lora_B_zero_init
        )

        # 3. Backbone Transformer (pure self/cross attention)
        self.transformer = PGTransformer(
            d_model=d_model,
            num_blocks=num_blocks,
            num_heads=num_heads,
            cond_dim=d_model, # condition is projected to d_model before cross_attn
            expansion=4.0,
            use_rope=use_rope
        )

    def forward(self, z_H: torch.Tensor, scale: float = 2.0, circuit_feat: Optional[torch.Tensor] = None) -> dict:
        """
        z_H: Context hidden states from Pass 1, shape [B, seq_len+16, cond_dim=512]
        circuit_feat: 离线提取的电路特征, shape [B, circuit_dim] (e.g. [B, 53]), or None
        """
        B = z_H.shape[0]

        # Step 1: Condition Projection
        cond = self.cond_proj(z_H) # [B, S, 256]

        # Step 1.5: 拼接电路特征 token (模拟在线提取)
        if self.use_circuit and circuit_feat is not None:
            circuit_token = self.circuit_proj(circuit_feat)    # [B, d_model]
            circuit_token = circuit_token.unsqueeze(1)         # [B, 1, d_model]
            cond = torch.cat([circuit_token, cond], dim=1)     # [B, S+1, d_model]

        # Step 2: Initialize Embeddings and Pack into Tokens
        # tokens: [B, 152, 256] (for 38 virtual tokens, rank 16, dim_acc 4)
        x = self.tokenizer.get_initial_tokens(batch_size=B, device=z_H.device, dtype=z_H.dtype)

        # Step 3: Transformer Reasoning
        x = self.transformer(x, cond) # [B, 152, 256]

        # Step 4 & 5: Unpack, Project, Slice and Reshape to State Dict
        lora_dict = self.tokenizer.detokenize(x, scale=scale)

        return lora_dict
