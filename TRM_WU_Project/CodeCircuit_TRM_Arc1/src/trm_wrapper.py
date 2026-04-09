"""
Phase 1: Unrolled TRM Attribution Wrapper
=========================================
非侵入式包装器，将 TRM 的递归循环完全展开为全可导链路。

核心改动：
  - 原版 trm.py 第 208 行：with torch.no_grad(): 前 H_cycles-1 步
  - 本 Wrapper：移除 no_grad，所有 H_cycles 步全部保留梯度图

设计对齐 CodeCircuit (replacement_model.py)：
  1. 只在 MLP 输出处注入 SAE (encode → decode)
  2. 冻结 Attention Pattern 的梯度（让 V 投影保持可导）
  3. 冻结 RMSNorm 的 scale 因子的梯度
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class AttributionOutput:
    """归因前向传播的输出结构"""
    logits: torch.Tensor                      # (batch, seq_len, vocab_size)
    feature_activations: List[torch.Tensor]   # 每个虚拟层的 SAE 编码特征 (batch, seq, d_sae)
    mlp_outputs: List[torch.Tensor]           # 每个虚拟层的原始 MLP 输出 (用于计算 error)
    sae_reconstructions: List[torch.Tensor]   # 每个虚拟层的 SAE 重构值 (用于计算 error)


class FrozenAttention(nn.Module):
    """
    包裹原版 Attention 模块。
    
    让 QKV 投影和 O 投影保留梯度（线性操作），
    但将 attention pattern (softmax 输出) 冻结为常量。
    
    对齐 CodeCircuit: replacement_model.py 第 194-198 行
    block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
    """
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn
    
    def forward(self, cos_sin, hidden_states):
        import einops
        from models.layers import apply_rotary_pos_emb
        
        attn = self.original_attn
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV 投影 — 保留梯度
        qkv = attn.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 
                        attn.num_heads + 2 * attn.num_key_value_heads, 
                        attn.head_dim)
        query = qkv[:, :, :attn.num_heads]
        key   = qkv[:, :, attn.num_heads: attn.num_heads + attn.num_key_value_heads]
        value = qkv[:, :, attn.num_heads + attn.num_key_value_heads:]
        
        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # 转换为 (B, H, S, D) 格式
        query, key, value = map(
            lambda t: einops.rearrange(t, 'B S H D -> B H S D'), 
            (query, key, value)
        )
        
        # ===== 核心冻结点 =====
        # 手动计算 attention pattern 并 detach
        scale = query.shape[-1] ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = attn_weights.detach()  # ← 冻结 attention pattern
        
        # Value 加权 — 保留 value 的梯度
        attn_output = torch.matmul(attn_weights, value)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, attn.output_size)
        
        # O 投影 — 保留梯度
        return attn.o_proj(attn_output)


def frozen_rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """
    RMSNorm，但 scale 因子（即 rsqrt(variance)）被冻结。
    
    对齐 CodeCircuit: replacement_model.py 第 199-205 行
    block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
    """
    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    
    variance = hidden_states_f32.square().mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + variance_epsilon).detach()  # ← 冻结 scale
    
    return (hidden_states_f32 * scale).to(input_dtype)


class UnrolledBlock(nn.Module):
    """
    对 TinyRecursiveReasoningModel_ACTV1Block 的归因适配版本。
    
    - 使用 FrozenAttention 替代原版 Attention
    - 使用 frozen_rms_norm 替代原版 rms_norm
    - MLP 保持原样（梯度自由流过）
    """
    def __init__(self, original_block):
        super().__init__()
        self.config = original_block.config
        self.norm_eps = original_block.norm_eps
        
        # MLP 保持原样引用（共享权重）
        self.mlp = original_block.mlp
        
        # Attention: 包裹为冻结版本
        if self.config.mlp_t:
            self.mlp_t = original_block.mlp_t
        else:
            self.frozen_attn = FrozenAttention(original_block.self_attn)
    
    def forward(self, cos_sin, hidden_states):
        # Post Norm (与原版 trm.py Block.forward 完全一致，仅替换 norm 和 attn)
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = frozen_rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = frozen_rms_norm(
                hidden_states + self.frozen_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps
            )
        
        # MLP — 梯度自由流过
        out = self.mlp(hidden_states)
        hidden_states = frozen_rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states, out  # 额外返回 mlp 原始输出，用于 SAE 注入


class UnrolledReasoningModule(nn.Module):
    """
    对 TinyRecursiveReasoningModel_ACTV1ReasoningModule 的归因适配版本。
    收集每一层的 MLP 输出用于 SAE 注入。
    """
    def __init__(self, original_module):
        super().__init__()
        self.unrolled_layers = nn.ModuleList([
            UnrolledBlock(layer) for layer in original_module.layers
        ])
    
    def forward(self, hidden_states, input_injection, sae_model=None, 
                feature_collector=None, mlp_collector=None, recon_collector=None, **kwargs):
        hidden_states = hidden_states + input_injection
        
        for layer in self.unrolled_layers:
            hidden_states, mlp_out = layer(hidden_states=hidden_states, **kwargs)
            
            # SAE 注入点：在每层 MLP 输出后进行 encode → decode
            if sae_model is not None and feature_collector is not None:
                features = sae_model.encode(mlp_out)      # (batch, seq, d_sae) 稀疏
                reconstructed = sae_model.decode(features) # (batch, seq, d_in)
                
                feature_collector.append(features)
                mlp_collector.append(mlp_out.detach())
                recon_collector.append(reconstructed.detach())
        
        return hidden_states


class UnrolledTRMWrapper(nn.Module):
    """
    TRM 归因全展开包装器。
    
    将 TinyRecursiveReasoningModel_ACTV1_Inner 的 forward 逻辑完全复刻，
    但移除 torch.no_grad() 屏障，让梯度在所有 H_cycles 中畅通无阻。
    
    对齐 CodeCircuit 的 ReplacementModel：
    - 只在 MLP 处注入 SAE（不动 Attention）
    - 冻结 Attention Pattern（detach softmax 输出）
    - 冻结 RMSNorm Scale（detach rsqrt 因子）
    """
    
    def __init__(self, trm_inner, sae_model):
        """
        Args:
            trm_inner: TinyRecursiveReasoningModel_ACTV1_Inner 实例
            sae_model: 已训练的 SparseAutoencoder (d_in=512, d_sae=4096)
        """
        super().__init__()
        self.config = trm_inner.config
        self.trm_inner = trm_inner
        self.sae_model = sae_model
        
        # 构建归因版本的 L_level（共享原版权重，但替换 Attention 和 Norm）
        self.unrolled_L_level = UnrolledReasoningModule(trm_inner.L_level)
        
        # 直接引用原版的 I/O 组件（不修改）
        self.embed_tokens = trm_inner.embed_tokens
        self.lm_head = trm_inner.lm_head
        
        # 冻结所有原版参数（我们不训练模型，只做归因）
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> AttributionOutput:
        """
        全展开前向传播。
        
        复刻 trm.py TinyRecursiveReasoningModel_ACTV1_Inner.forward() 第 196-226 行，
        但移除 with torch.no_grad()。
        """
        # ====== 输入编码（复刻第 197-202 行）======
        seq_info = dict(
            cos_sin=self.trm_inner.rotary_emb() if hasattr(self.trm_inner, 'rotary_emb') else None,
        )
        input_embeddings = self.trm_inner._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )
        
        # ====== 初始化隐藏状态 ======
        batch_size = batch["inputs"].shape[0]
        seq_len = input_embeddings.shape[1]
        
        z_H = self.trm_inner.H_init.expand(batch_size, seq_len, -1).clone()
        z_L = self.trm_inner.L_init.expand(batch_size, seq_len, -1).clone()
        
        # 开启梯度追踪
        z_H.requires_grad_(True)
        z_L.requires_grad_(True)
        input_embeddings.requires_grad_(True)
        
        # ====== 收集器 ======
        all_features = []        # 每个虚拟层的 SAE 编码特征
        all_mlp_outputs = []     # 每个虚拟层的原始 MLP 输出
        all_reconstructions = [] # 每个虚拟层的 SAE 重构值
        
        # ====== 全展开循环（复刻第 207-216 行，移除 no_grad）======
        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                z_L = self.unrolled_L_level(
                    z_L, z_H + input_embeddings,
                    sae_model=self.sae_model,
                    feature_collector=all_features,
                    mlp_collector=all_mlp_outputs,
                    recon_collector=all_reconstructions,
                    **seq_info
                )
            # H_step 结束时的 z_H 更新
            z_H = self.unrolled_L_level(
                z_H, z_L,
                sae_model=self.sae_model,
                feature_collector=all_features,
                mlp_collector=all_mlp_outputs,
                recon_collector=all_reconstructions,
                **seq_info
            )
        
        # ====== 输出 Logits（复刻第 224 行）======
        puzzle_emb_len = self.trm_inner.puzzle_emb_len if hasattr(self.trm_inner, 'puzzle_emb_len') else 0
        logits = self.trm_inner.lm_head(z_H)[:, puzzle_emb_len:]
        
        return AttributionOutput(
            logits=logits,
            feature_activations=all_features,
            mlp_outputs=all_mlp_outputs,
            sae_reconstructions=all_reconstructions,
        )
