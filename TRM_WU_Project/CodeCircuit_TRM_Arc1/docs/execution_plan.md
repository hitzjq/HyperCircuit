# 实施方案与执行路径 (Execution Plan)

所有代码放置于 `CodeCircuit_TRM_Arc1/` 目录，不侵入 `TinyRecursiveModels/` 和 `TRM_WU_Project/`。

## Phase 0: 训练 TRM-Transcoder (SAE)
**目标**：获得一个能把 TRM 隐藏层激活分解为稀疏可解释 Feature 的字典。
**步骤**：
1. 在 8xH200 上运行全量 ARC-AGI-1 puzzles 的前向推理。
2. 在每个虚拟层（共计 42 个切面）的 `mlp.down_proj` 输出处挂 Hook，收集所有激活路线。
3. 把收集的 `[Total_Tokens, Hidden_Size=512]` 隐状态进行切块打散并存入 `.safetensors`。
4. 训练一层的 Sparse Autoencoder (512 -> 4096)，使用 MSE + L1 稀疏约束。

**生成文件**：
- `CodeCircuit_TRM_Arc1/transcoder/collect_activations.py`
- `CodeCircuit_TRM_Arc1/transcoder/train_transcoder.py`
- 产出产物: `trm_transcoder_4096.pt`

---

## Phase 1: TRM 旁路归因包装器 (Bypass Forward Wrapper)
**目标**：在不修改 `trm.py` 的前提下，构建一个**全程保留梯度**的全微分网络。
**机制**：
- 实例化时接收原 `TRMInner` 权重。
- 利用 Python `for` 循环完全拆解 42 层切面的调用树，彻底剥离掉限制梯度的 `torch.no_grad()` 边界。
- 注入 Phase 0 pre-trained 的 Transcoder。利用类似 CodeCircuit 的 `ReplacementMLP` 思路，在反向传播时将激活相加操作替换为通过 SAE Decoder，以便后续直接提取一阶 Influence 梯度。

**生成文件**：
- `CodeCircuit_TRM_Arc1/adapters/trm_wrapper.py`

---

## Phase 2: Attribution Graph 构建引擎
**目标**：为每个 ARC query 自动剥离一张稀疏归因图。
**机制**：
- 驱动 Phase 1 的微导 Wrapper 跑 42 层全微分激活 `Forward`。
- 从头部定义 `Target = Cross-Entropy Loss (全格点监督交叉熵)` 发起 `Backward`。
- **Taylor 权重 (Influence) = 激活值 $\times$ 梯度**。
- 将特征过滤后的 Taylor 节点构成多层级电路连通图落盘。

**生成文件**：
- `CodeCircuit_TRM_Arc1/graph/attribute_trm.py`
- 产出产物: `graph_query_{id}.pt`

---

## Phase 3: 图特征到向量的连续化融合
**目标**：由于下游是神经网络 PG (而非随机森林)，需要将任意大小图的连通结构转化成**可微的稠密张量矩阵**。
**机制**：
- 加载提取的 `.pt` 稀疏图集。
- 舍弃掉原版中的 “度中心性” 等纯手工离散指标。
- 按空间 / 时间（深度）轴对所有的激活 SAE 节点权重执行 `Mean-Pooling` 和 `Max-Pooling`。
- 连续化压平至 `[batch, fixed_seq_len, dense_dim]` 并存入 Dataset，保证 HyperNetwork 注意力机理及梯度的传递不断裂。

**生成文件**：
- `CodeCircuit_TRM_Arc1/graph/graph_to_vector.py`

---

## 硬件调度与算力规划

合理的设备调配以防止极大张量收集时的 OOM (Out Of Memory)。

| 阶段 | 设备要求 | 说明 |
|------|------|------|
| **Phase 0 提取** | **8×H200** | 大规模显存刚需：42切面推理全量数据落盘 |
| **Phase 0 训练** | **8×A100 (DDP)** | 训练 SAE，并行读取本地磁盘分块的 Safetensors |
| **Phase 1~2 调试**| **1×A100** | Wrapper 的功能验收、梯度连通验证以及小图剥离验证 |
| **Phase 2 流水线**| **8×H200** | 并发大批量跑 Backward 计算 Target，生成数十万张 Circuit .pt 图 |
| **Phase 3 融合** | **CPU / 1×A100** | 内存与显存组合。做 Pooling Tensor 归一并合成 Dataset |
