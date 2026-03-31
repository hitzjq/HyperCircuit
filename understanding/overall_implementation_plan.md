# TRM4HY-WU 设计方案（合并版）

## 目标

将 HY-WU 的 Parameter Generation (PG) 架构应用到 TRM，为 ARC-AGI 任务生成实例级 LoRA 适配器。

---

## 1. 已敲定的设计决策

### 1.1 Parameter Embedding 方案

**方案 B：10 个 Module Embedding Token + Self-Attention + Cross-Attention**

所有 10 个 CastedLinear 模块由超网络动态生成 LoRA 参数。

### 1.2 Condition 来源

**方案 2B：深层 hidden states**

Pass 1 跑完整 TRM forward (no_grad) → z_H: `[B, seq_len+16, 512]`

### 1.3 LoRA Rank

**r = 16**

### 1.4 投影头

**每个模块独立投影头**（实际中通过 tokenization + 共享 proj_out 实现）

### 1.5 训练策略

**Joint Training**：PG + embed_tokens + puzzle_emb 一起训练

---

## 2. 超参数完整配置

### 2.1 由基座 TRM 决定的参数（不可调）

| 参数 | 值 | 说明 |
|------|-----|------|
| `rank` | 16 | LoRA 的秩，= lora_r |
| `lora_alpha` | 32 | LoRA 缩放因子 → scale = alpha/rank = 2.0 |
| `base_hidden_size` | 512 | 基座 TRM 的隐藏维度，决定 condition 维度 |
| `base_vocab_size` | V (~16) | ARC 数据集词表大小 |
| `base_seq_len` | 取决于数据 | 输入序列长度 |
| `puzzle_emb_len` | 16 | puzzle embedding 占的 token 位置数 |

### 2.2 PG 自身设计的参数（可调）

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `token_dim` | 512 | 每个 token 在每个 rank 位置输出多少个值。匹配 base_hidden_size。决定 token_area。 |
| `token_area` | 16×512 = 8,192 | 一个 token 能承载多少个 LoRA 参数。= rank × token_dim。 |
| `dim_accumulation` | 4 | 打包倍数。把 4 个 rank 位置的向量拼接成 1 个 Transformer token。减少 token 数。 |
| `dim_wo_acc` | 64 | 打包前每个 rank 位置的内部表示维度。learnable embedding 的实际维度。 |
| `d_model` | 256 | PG Transformer 的隐藏维度。= dim_wo_acc × dim_accumulation = 64 × 4。 |
| `num_heads` | 4 | PG Transformer 的注意力头数。head_dim = 256/4 = 64。 |
| `num_pg_blocks` | 2 | PG Transformer 的深度。为匹配 7.3M 的基座，调浅一点防止过拟合。 |
| `ffn_expansion` | 4 | PG FFN 的膨胀倍数。FFN 宽度 = d_model × ffn_expansion = 1024。 |
| `pg_dropout` | 0.0 | PG Transformer 的 dropout，训练初期建议 0。 |
| `norm_type` | RMSNorm | 归一化方式，与 TRM 基座保持一致。 |
| `pos_encoding` | 无 | PG tokens 不使用位置编码（靠 learnable embedding 区分身份）。 |
| `proj_out_bias` | False | proj_out 是否用 bias。 |
| `lora_B_zero_init` | True | proj_out_B 是否零初始化。保证初始时 LoRA 输出为 0，不影响基座。 |

### 2.3 训练超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `pg_lr` | 1e-3 ~ 5e-3 | PG 的学习率 |
| `embed_tokens_lr` | 与 pg_lr 相同或更低 | embed_tokens 的学习率 |
| `puzzle_emb_lr` | 沿用原设置 | puzzle_emb 的 SignSGD 学习率 |
| `lr_warmup_steps` | 800 | 学习率预热步数 |
| `optimizer` | AdamATan2 | 沿用 TRM 原版优化器 |

---

## 3. Tokenization 设计

### 3.1 Token 数量

| # | 模块 | A params | A tokens | B params | B tokens |
|---|------|----------|----------|----------|----------|
| 0 | lm_head | 8,192 | 1 | 16V (~256) | 1 |
| 1 | q_head | 8,192 | 1 | 32 | 1 |
| 2 | L0.qkv_proj | 8,192 | 1 | 24,576 | 3 |
| 3 | L0.o_proj | 8,192 | 1 | 8,192 | 1 |
| 4 | L0.gate_up_proj | 8,192 | 1 | 49,152 | 6 |
| 5 | L0.down_proj | 24,576 | 3 | 8,192 | 1 |
| 6 | L1.qkv_proj | 8,192 | 1 | 24,576 | 3 |
| 7 | L1.o_proj | 8,192 | 1 | 8,192 | 1 |
| 8 | L1.gate_up_proj | 8,192 | 1 | 49,152 | 6 |
| 9 | L1.down_proj | 24,576 | 3 | 8,192 | 1 |
| | **合计** | | **14** | | **24** |

**38 virtual tokens → 每个展开 rank=16 个位置 → 608 个原始位置 → dim_accumulation=4 打包 → 152 Transformer tokens**

### 3.1.1 具体例子：L0.gate_up_proj 的 lora_B 如何变成 tokens

以最大的 LoRA 矩阵为例：`L0.gate_up_proj` 的 `lora_B: [3072, 16] = 49,152 个参数`

**第 1 步：展平**

```
lora_B 矩阵: [3072行, 16列]
展平成一条线: [w₀, w₁, w₂, ..., w₄₉₁₅₁]  共 49,152 个值
```

**第 2 步：切分成 virtual tokens**

```
每个 token 装 token_area = rank × token_dim = 16 × 512 = 8,192 个值
49,152 ÷ 8,192 = 6 个 virtual token（刚好整除）

Token B₀: [w₀     ~ w₈₁₉₁]     8,192 个值
Token B₁: [w₈₁₉₂  ~ w₁₆₃₈₃]    8,192 个值
Token B₂: [w₁₆₃₈₄ ~ w₂₄₅₇₅]    8,192 个值
Token B₃: [w₂₄₅₇₆ ~ w₃₂₇₆₇]    8,192 个值
Token B₄: [w₃₂₇₆₈ ~ w₄₀₉₅₉]    8,192 个值
Token B₅: [w₄₀₉₆₀ ~ w₄₉₁₅₁]    8,192 个值
```

**第 3 步：每个 virtual token 展开为 rank 个位置**

每个 virtual token 的 8,192 个值被看作 `[rank=16, token_dim=512]` 的 2D 结构，
也就是 16 个 `dim_wo_acc=64` 维的向量：

```
Token B₀ 展开后: 16 个 64 维向量
  rank 0:  [64 个值]
  rank 1:  [64 个值]
  ...
  rank 15: [64 个值]
```

**第 4 步：dim_accumulation=4 打包**

把每 4 个相邻 rank 位置的 64 维向量拼成 1 个 256 维 Transformer token：

```
  rank 0~3:   4 × [64维] 拼接 → 1 个 Transformer token [256维]
  rank 4~7:   4 × [64维] 拼接 → 1 个 Transformer token [256维]
  rank 8~11:  4 × [64维] 拼接 → 1 个 Transformer token [256维]
  rank 12~15: 4 × [64维] 拼接 → 1 个 Transformer token [256维]


→ 1 个 virtual token 变成 16/4 = 4 个 Transformer tokens
→ gate_up_proj lora_B 的 6 个 virtual tokens → 6 × 4 = 24 个 Transformer tokens
```

**第 5 步（推理后）：Detokenize 还原**

PG Transformer 处理完后，反向操作：解包 → proj_out → flatten → 截取 49,152 个值 → reshape 回 `[3072, 16]`

### 3.1.2 打包到底在减少什么

```
起点: 38 virtual tokens × rank 16 = 608 个原始位置（每个 dim_wo_acc=64 维）

不打包 (dim_acc=1):  608 个 Transformer tokens（每个 64 维）  ← 太多
打包×4  (dim_acc=4):  152 个 Transformer tokens（每个 256 维）  ← 减少 4 倍 ✓
打包×8  (dim_acc=8):   76 个 Transformer tokens（每个 512 维）
打包×16 (dim_acc=16):  38 个 Transformer tokens（每个 1024 维） ← 最紧凑

→ 打包是用更宽的向量换更少的 token 数
→ 38 不是起点，608 才是起点
```

### 3.2 不通过 PG 生成的模块

| 模块 | 处理方式 |
|------|---------|
| `embed_tokens` | 始终解冻直接训练（不加 LoRA） |
| `puzzle_emb` | SignSGD 独立训练（不加 LoRA） |
| `H_init / L_init` | 冻结 |
| `rotary_emb` | 冻结 |

---

## 4. 完整端到端演示（以最新 2.3M PG 参数为例）

为了让你完全看清“参数是怎么被切分、如何变成 LoRA 的”，我们端到端跑一遍数据流！
当前配置：`d_model=256`, `dim_wo_acc=64`, `dim_accumulation=4`, `rank=16`, `token_dim=512`。

### 场景准备
我们要为 10 个模块生成 LoRA。其中：
*   **模块1**：`q_head` 的 `lora_B` (极小，需要 32 个参数 → 向上取整需要 1 个 virtual token)
*   **模块2**：`L0.gate_up_proj` 的 `lora_B` (极大，需要 49,152 个参数 → 刚好需要 6 个 virtual token)
*   所有 14 个 A + 24 个 B 合计：**38 个 virtual tokens**。

### Step 1: 提取 Condition 进行降维
*   基座跑完第一遍，拿到深层特征 `z_H` 原本形状：`[B, seq_len+16, 512]`
*   通过 `condition_proj` (Linear 512→256) 降维映射到 PG 的思维维度：
*   得到 `cond`: **`[B, seq_len+16, 256]`**

### Step 2: 初始化并“打包” Tokens
*   38 个 virtual token，每个 token 都硬性要求生成 16 个 rank 位置的数据。
*   总计产生：38 × 16 = **608 个独立位置**。
*   PG 为这 608 个位置初始化可学习参数 `[608, dim_wo_acc=64]`：
    ```
    原始 Embedding: [B, 608, 64]
    ```
*   **打包 (dim_accumulation=4)**：把每 4 个相邻的 64 维向量，首尾相接拼成 256 维。
    ```
    打包后 Tokens: [B, 608/4, 64×4] = [B, 152, 256]
    ```
*   这 152 个 `256维` 的向量就是送进 Transformer 的真正的 “词(tokens)”。

### Step 3: PG Transformer 思考过程 (2层)
*   **Self-Attention**: 152 个 token 在内部互相看（包含 A 和 B 的 token，以及跨层/跨模块的 token）。大家互相协调互相影响。
*   **Cross-Attention**: 152 个 token 向 `cond [B, seq_len+16, 256]` 进行 Query（交叉注意力），从 ARC 题目的视觉/逻辑隐状态中主动提取信息。
*   **FFN**: 通过 `256 → 1024 → 256` 进行非线性映射。
*   经过 2 层 Transformer 后输出形状不变，依然是：**`[B, 152, 256]`**。

### Step 4: 解包与独立投影
怎么把结合了上下文思想的 256 维向量变回特定的参数值？
*   **解包**：将 256 维拆回 4 个 64 维，变回 608 个位置：
    ```
    拆解后: [B, 38, 16, 64]   (38个virtual token，每个16个rank，内部64维)
    ```
*   **分流 A 和 B**：按照固定索引进行切片（前 14 个给 A，后 24 个给 B）：
    ```
    A tokens: [B, 14, 16, 64]
    B tokens: [B, 24, 16, 64]
    ```
*   **投影到目标宽度 (proj_out)**：
    我们基座的 `hidden_size=512`（也就是 `token_dim`），所以需要用 Linear 把宽度 64 投影拉伸到 512：
    ```
    proj_out_B (Linear 64→512):
    → [B, 24, 16, 512]  (这就是所有 B tokens 生成的最终数值！)
    ```

### Step 5: 提取并重组为 LoRA 矩阵 (Detokenize)
每个 virtual token 目前包含了 `16 × 512 = 8,192` 个参数值。

**对 模块1 (`q_head` lora_B)：只需 32 个参数**
1. 在 24 个 B token 中，按索引拿到它专属的第 1 个 token。
2. 形状分离出 `[B, 1, 16, 512]`。
3. 把这块数据拍扁 (Flatten)：变成 `[B, 8192]` 长度的一条线。
4. **截取**：因为只需要 32 个参数，我们只取前 32 个数值，扔掉后面 8160 个冗余值（这就是 HY-WU 中提到的 Padding 浪费）。
5. **重组** (Reshape)：将这 32 个值重组成 `q_head` 目标矩阵的形状 `[B, 2, 16]`。✅ 搞定！

**对 模块2 (`L0.gate_up_proj` lora_B)：需 49,152 个参数**
1. 按索引拿到它专属的 6 个连续 tokens。
2. 形状分离出 `[B, 6, 16, 512]`。
3. 拍扁 (Flatten)：变成正好是 `[B, 49152]` 长度的一条线！(6×16×512=49152，没有任何浪费)。
4. **截取**：正好 49152 个，全都要。
5. **重组** (Reshape)：将这根横线重组成实际需要的权重矩阵形状 `[B, 3072, 16]`。✅ 搞定！

通过这套 `展平 → 切token → 网络处理 → 还原 → 截断 → 变形` 的机制，无论模块的形状多么千奇百怪，PG 输出的统一样式最终都会完美对齐到所有需要的 `lora_A` 和 `lora_B` 矩阵中。

---

## 5. 参数量估算校准 (L_layers=2)

### 5.1 基座 TRM 参数量 (精确计算)

| 模块 | 形状 / 计算式 | 参数量 |
|------|--------------|-------|
| `embed_tokens` | `32 × 512` (假设 V=32) | ~16K |
| `puzzle_emb` | `1024 × 512` (假设 1024 个题目) | ~524K |
| `lm_head` | `512 × 32` | ~16K |
| `q_head` | `512 × 2` | ~1K |
| **单层 Transformer** | `qkv(786K) + o(262K) + gate_up(1.57M) + down(786K)` | **~3.4M** |
| 2层 L_level | 2 × 3.4M | ~6.8M |
| **基座总计** | | **~7.36M** |

### 5.2 PG 参数量 (缩小版 d_model=256, num_blocks=2)

| 组件 | 参数量 |
|------|-------|
| token_pos_emb (608 × 64) | 39K |
| condition_proj (Linear 512→256) | 131K |
| norm_tokens (RMSNorm 256) | 0.5K |
| 2 × self_attn (Q/K/V/O 各 Linear 256→256) | 524K |
| 2 × cross_attn (Q/K/V/O 各 Linear 256→256) | 524K |
| 2 × FFN (Linear 256→1024 + Linear 1024→256) | 1.05M |
| 2 × norms (4 个 RMSNorm per block) | 4K |
| proj_out_A (Linear 64→512) | 33K |
| proj_out_B (Linear 64→512) | 33K |
| norm_final (RMSNorm 256) | 0.5K |
| **PG 总计** | **~2.33M** |

| 对比 | 参数量 |
|------|-------|
| 基座 TRM | ~7.36M |
| PG | ~2.33M |
| PG/基座 | **~31.6%** (非常合理的超网络比例) |

---

## 6. TRM 完整模块结构参考

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

---

## 7. 完整推理管线

```
Pass 1 (no_grad):
  inputs [B, seq_len] → embed_tokens + puzzle_emb → [B, seq_len+16, 512]
  递归推理 (3H × 4L) → z_H: [B, seq_len+16, 512]

PG Forward:
  z_H → tokens init [B, 152, 512]
  4 × (self_attn + cross_attn + FFN)
  → 解包 + proj → detokenize → loradict

LoRA 注入:
  set_loradict → 修改每个 CastedLinear 的 forward

Pass 2 (有梯度):
  inputs → embed_tokens + puzzle_emb → [B, seq_len+16, 512]
  递归推理 + LoRA → z_H → lm_head → logits [B, seq_len, V]
  loss = CE(logits, labels)

Backward:
  loss.backward() → 更新 PG 参数 + embed_tokens + puzzle_emb
```

---

## 8. 待确认

> [!IMPORTANT]
> 1. 新方案修改为 `d_model=256`, `num_pg_blocks=2`，整体占比 `~31.6%`。是否 OK？
> 2. `dim_accumulation=4` 是否 OK？（也可以选 2）
> 3. `lora_B_zero_init=True`：PG 输出初始为零，保证训练初期基座行为不变——是否同意？
> 4. 以上训练超参数（lr 等）是否需要调整？
