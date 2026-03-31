# TRM4HY-WU 设计方案

## 目标
将 HY-WU 的 Parameter Generation (PG) 架构思想应用到 TRM (TinyRecursiveModels) 上，为 ARC-AGI 任务生成实例级的 LoRA 适配器。

---

## 已敲定的设计决策

### 1. Parameter Embedding 方案（方案 B：Module Embedding）

**核心思想**：为每个 LoRA 目标模块分配一个可学习的"身份 token"，通过 Self-Attention + Cross-Attention 生成 LoRA 参数。

#### 1.1 超网络生成的 LoRA 目标（10 个 module token）

**所有原本加了 LoRA 的 CastedLinear 模块，全部由超网络动态生成 LoRA 参数：**

| Token ID | 模块路径 | in_dim | out_dim |
|----------|---------|--------|---------|
| 0 | `lm_head` | 512 | vocab_size |
| 1 | `q_head` | 512 | 2 |
| 2 | `layers[0].self_attn.qkv_proj` | 512 | 1536 |
| 3 | `layers[0].self_attn.o_proj` | 512 | 512 |
| 4 | `layers[0].mlp.gate_up_proj` | 512 | 3072 |
| 5 | `layers[0].mlp.down_proj` | 1536 | 512 |
| 6 | `layers[1].self_attn.qkv_proj` | 512 | 1536 |
| 7 | `layers[1].self_attn.o_proj` | 512 | 512 |
| 8 | `layers[1].mlp.gate_up_proj` | 512 | 3072 |
| 9 | `layers[1].mlp.down_proj` | 1536 | 512 |

#### 1.2 超网络架构

```
module_emb = nn.Embedding(10, d_model)   # 10 个 LoRA 目标的身份 embedding

tokens = module_emb.weight  →  [10, d_model]

for block in transformer_blocks:
    tokens = self_attn(tokens)          # 模块间协调
    tokens = cross_attn(tokens, z_H)    # 从 condition 提取信息
    tokens = ffn(tokens)

# 每个 token 投影成对应模块的 LoRA A + B
for i in range(10):
    lora_A[i] = proj_A_i(tokens[i])     # → [r, in_dim_i]
    lora_B[i] = proj_B_i(tokens[i])     # → [out_dim_i, r]
```

#### 1.3 不通过超网络生成的模块

| 模块 | 处理方式 | 原因 |
|------|---------|------|
| `puzzle_emb` | SignSGD 独立训练，不冻结 | per-puzzle 身份信息，不适合超网络 |
| `embed_tokens` | 直接训练（始终解冻） | 是 CastedEmbedding，不是 CastedLinear，不加 LoRA，全量参数直接更新 |
| `H_init / L_init` | 冻结 | nn.Buffer |
| `rotary_emb` | 冻结 | 固定位置编码 |

---

## TRM 完整模块结构参考

配置: `hidden_size=512, num_heads=8, expansion=4, L_layers=2`

```
TinyRecursiveReasoningModel_ACTV1
└── inner
    ├── embed_tokens (CastedEmbedding, 512)       [不加LoRA, 始终解冻直接训练]
    ├── puzzle_emb (CastedSparseEmbedding)         [不加LoRA, SignSGD独立训练]
    ├── rotary_emb (RotaryEmbedding)               [冻结]
    ├── H_init, L_init (nn.Buffer)                 [冻结]
    ├── lm_head (CastedLinear, 512→V)              [动态LoRA, Token#0]
    ├── q_head (CastedLinear, 512→2)               [动态LoRA, Token#1]
    └── L_level (ReasoningModule)
        ├── layers[0] (Block)
        │   ├── self_attn.qkv_proj  512→1536       [动态LoRA, Token#2]
        │   ├── self_attn.o_proj    512→512        [动态LoRA, Token#3]
        │   ├── mlp.gate_up_proj    512→3072       [动态LoRA, Token#4]
        │   └── mlp.down_proj       1536→512       [动态LoRA, Token#5]
        └── layers[1] (Block)
            ├── self_attn.qkv_proj  512→1536       [动态LoRA, Token#6]
            ├── self_attn.o_proj    512→512        [动态LoRA, Token#7]
            ├── mlp.gate_up_proj    512→3072       [动态LoRA, Token#8]
            └── mlp.down_proj       1536→512       [动态LoRA, Token#9]
```

SwiGLU inter 维度计算: `ceil_to_256(round(4 × 512 × 2/3)) = 1536`

### LoRA 参数量 (r=16)

| 模块 | lora_A | lora_B | 合计 |
|------|--------|--------|------|
| lm_head (×1) | [16, 512] | [V, 16] | 8,192 + 16V |
| q_head (×1) | [16, 512] | [2, 16] | 8,224 |
| qkv_proj (×2) | [16, 512] | [1536, 16] | 32,768 × 2 |
| o_proj (×2) | [16, 512] | [512, 16] | 16,384 × 2 |
| gate_up_proj (×2) | [16, 512] | [3072, 16] | 57,344 × 2 |
| down_proj (×2) | [16, 1536] | [512, 16] | 32,768 × 2 |
| **动态 LoRA 合计** | | | **~295K + 16V** |

---

## 待敲定的设计决策

- [ ] Condition (z_H) 的来源：Pass 1 具体怎么做？用 train examples 还是整个 batch？
- [ ] 超网络的具体超参数：d_model, num_heads, num_blocks
- [ ] LoRA rank r 的选择
- [ ] 投影头的设计：统一维度还是每个模块独立投影
- [ ] 训练策略：joint training vs freeze base + train hypernet
