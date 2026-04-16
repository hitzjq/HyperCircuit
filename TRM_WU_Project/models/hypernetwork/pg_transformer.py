import torch
import torch.nn as nn
from models.layers import CastedLinear, SwiGLU, Attention, RotaryEmbedding, rms_norm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class PGCrossAttention(nn.Module):
    """
    Standard Cross-Attention for PG Transformer.
    Query comes from PG tokens, Key/Value come from Condition.
    """
    def __init__(self, d_model: int, condition_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = CastedLinear(d_model, d_model, bias=False)
        self.k_proj = CastedLinear(condition_dim, d_model, bias=False)
        self.v_proj = CastedLinear(condition_dim, d_model, bias=False)
        self.o_proj = CastedLinear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, S_x, _ = x.shape
        _, S_c, _ = cond.shape

        q = self.q_proj(x).view(B, S_x, self.num_heads, self.head_dim)
        k = self.k_proj(cond).view(B, S_c, self.num_heads, self.head_dim)
        v = self.v_proj(cond).view(B, S_c, self.num_heads, self.head_dim)

        # Transpose for scaled_dot_product_attention: B S H D -> B H S D
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # B H S D -> B S H D -> B S D
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_x, -1)
        return self.o_proj(attn_output)

class PGTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, cond_dim: int, expansion: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        # We reuse the TRM Attention for self-attention, but we don't pass RoPE (cos_sin=None)
        head_dim = d_model // num_heads
        self.self_attn = Attention(
            hidden_size=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = PGCrossAttention(d_model=d_model, condition_dim=cond_dim, num_heads=num_heads)
        
        self.norm3 = RMSNorm(d_model)
        self.mlp = SwiGLU(hidden_size=d_model, expansion=expansion)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, cos_sin=None) -> torch.Tensor:
        # 1. Self-Attention (with optional RoPE)
        h = x + self.self_attn(cos_sin=cos_sin, hidden_states=self.norm1(x))
        # 2. Cross-Attention
        h = h + self.cross_attn(self.norm2(h), cond)
        # 3. FFN (SwiGLU)
        out = h + self.mlp(self.norm3(h))
        return out

class PGTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        cond_dim: int = 256,
        expansion: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 256
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PGTransformerBlock(d_model, num_heads, cond_dim, expansion)
            for _ in range(num_blocks)
        ])
        self.norm_final = RMSNorm(d_model)

        # Optional 1D RoPE for self-attention positional awareness
        self.use_rope = use_rope
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=max_seq_len,
                base=10000
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cos_sin = None
        if self.use_rope:
            cos, sin = self.rotary_emb()
            cos_sin = (cos[:x.shape[1]], sin[:x.shape[1]])

        for block in self.blocks:
            x = block(x, cond, cos_sin=cos_sin)
        return self.norm_final(x)
