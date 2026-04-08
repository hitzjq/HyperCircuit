# TRM Circuit 提图与编码：概念与架构设计方案 (Design Concept)

## 任务背景

在 8xH200 (运行) + 8xA100 (调试) 的算力支持下，将 CodeCircuit 的 Attribution Graph 提取与降维方法迁移到 TRM (TinyRecursiveModels) 上，面向 ARC-AGI-1 数据集。

**最终目标**：为每个 ARC query 提取一张推理电路图 (Circuit Graph)，并自动降维为固定长度的拓扑与信号特征向量 (Feature Vector)，供下游分类 / 打分使用。

---

## 三大核心难点

1. **模型类型不匹配**：CodeCircuit 面向 HuggingFace 标准 Transformer（如 Gemma），使用 `ReplacementModel` + `TranscoderSet`。TRM 是自定义递归模块，有 `H_cycles / L_cycles` 和 `torch.no_grad()` 屏障。
2. **缺乏 Transcoder（SAE）**：CodeCircuit 需要预训练好的 Transcoder 来生产可解释的 Feature 节点。TRM 没有现成的。
3. **输出多维性**：CodeCircuit 的 Target 是最后一个 token 的 next-token logit（标量）。ARC 的输出是整张网格（多维），需要重新定义归因起点。

---

## 已确认的设计决策 ✅

### 决策 1：SAE (Transcoder) 安放位置 ✅

**方案：按迭代次数展开为多层 (Unroll)，每层的 `mlp.down_proj` 输出处统一安装 SAE**

TRM 本质上是递归网络。假设设定 `max_H_cycles=3`，`L_steps=4`（即 `L_level` 有 2 层 Block，每层跑 2 次 `L_cycles`），我们在构建归因时把整个递归完全展开：

```
H_cycle 1: L_step 1 → L_step 2 → L_step 3 → L_step 4
H_cycle 2: L_step 1 → L_step 2 → L_step 3 → L_step 4
H_cycle 3: L_step 1 → L_step 2 → L_step 3 → L_step 4
```

变成一个有 **最多 12 个"虚拟层"** 的深层模型。同一个底层权重（如 `L_level.layers[0].mlp`）在第 1 次循环和第 3 次循环时，被当作**图中处于不同"时间层"的独立节点**。

SAE 统一挂在每个虚拟层的 `mlp.down_proj` 输出处（即每个 Block 运算收尾后的残差流），与 CodeCircuit 对 Gemma 每层 MLP 输出挂 Transcoder 的做法完全一致。

### 决策 2：归因 Target ✅

**方案：Cross-Entropy Loss 作为归因起点**

```python
target = F.cross_entropy(predicted_logits, ground_truth_labels)
target.backward()
```

- `predicted_logits`：模型对 query output 网格每个位置的颜色预测 logits
- `ground_truth_labels`：正确答案的颜色编号

交叉熵自带自适应加权：模型有把握的位置（如不变的背景色）贡献小梯度；模型不确定的位置（关键推理位置）贡献大梯度。既避免了"全格 logit 和"的噪声，也避免了"只选 Delta"在全变换题目上的退化问题。

### 决策 3：Halt 不定长处理 ✅

**方案 α：尊重原生 Halt，不等深图 + 归一化特征提取**

- 每道题**按实际 halt 次数展开**，电路图深度各不相同。
- 在 Phase 3（特征提取）时，使用与 CodeCircuit 完全一致的策略：**不直接用邻接矩阵，而是把任意大小的图压成固定长度的统计向量**。
- 路径类特征除以实际层数做归一化；层分布直方图固定长度为 `max_H_cycles × L_steps`，超出部分填 0。
- `n_actual_H_cycles`（实际执行次数）作为特征向量中的一个额外维度。

> **理由**：归因的本质是解释模型的真实决策过程。不应强制模型跑它不会执行的额外步骤，否则会引入幻象信号。

---

## 代码架构设计与实现逻辑 (Code Architecture & Logic)

为了保证代码清晰且不侵入原项目，下面是每个核心脚本的代码级规划：

### 1. 激活值收集 (`transcoder/collect_activations.py`)
- **Hook 策略设计**：利用 `PyTorch` 的 `register_forward_hook` 挂载到 `model.inner.L_level.layers[i].mlp` (`SwiGLU`) 的输出处。
- **动态切面追踪 (Slice Tracker)**：TRM 同一个物理 MLP 在一轮前向中会被调用 42 次 (3 H\_cycles $\times$ 7 步 $\times$ 2 层)。我们将设计一个 Wrapper 或在 Hook 内部维护一个 `call_counter` 状态机。每次 Hook 触发时，打上 `slice_id = call_counter % 42` 的时间戳标签。
- **持久化分发**：8xH200 每秒生成的隐状态张量极大。我们将实现一个基于 `Queue` 的异步落盘机制，凑满 N 个 batch 就展平为 `[Total_Tokens, Hidden_Size=512]` 后转存 `safetensors` 文件，彻底释放显存。

### 2. SAE 模型及训练 (`transcoder/train_transcoder.py`)
- **网络结构设计**：实现独立的 `class SparseAutoencoder(nn.Module)`，含 Encoder (`Linear(512, 4096)` + `ReLU`) 和 Decoder (`Linear(4096, 512)` 且权重与 Encoder 无绑定)。
- **Loss 体系**：标准组合 `Loss = MSE(重构损失) + λ * L1(Encoder输出)`。
- **并行调度**：采用 `torch.nn.parallel.DistributedDataParallel` 跑在 8xA100 上，使用随机打乱的磁盘 chunk `DataLoader` 进行批量消费。

### 3. 无侵入归因包装器 (`adapters/trm_wrapper.py`)
- **问题**：原生TRM 的 `H_cycles - 1` 步全部包在 `with torch.no_grad():` 内，这对于我们跑反向传播归因是致命的（梯度全断）。
- **完全展开逻辑 (Unroll Re-implementation)**：建一个新的 `class TRMAttributionWrapper(nn.Module)`，实例化时接收原生 `TRMInner`。我们将其 `forward()` 中的 `for` 循环逻辑**逐行复刻**，但**彻底移除 `no_grad()`** 屏障！这实现了一个全微导 (fully differentiable) 的 42 层超级网络。
- **挂载 Transcoder (Feature Injection)**：不仅展开，由于后续要进行归因，我们要劫持 `L_level`，使用类似 CodeCircuit 的 `ReplacementMLP`，在反向传播时将激活路线直接桥接到预训练好的 SAE Decoder 权重上去，以便拿到各个稀疏 Feature 节点的准确一阶梯度。

### 4.图生成与超网端适配 (`graph/attribute_trm.py` & `graph_to_vector.py`)
- **Taylor 提图引擎**：对全微分 Wrapper 发起 `forward`，计算 Cross-Entropy Loss 后发回 `backward`。`节点重要性 = 激活值(正向) * 梯度(反向)`。
- **连续化提取 (Dense Pooling)**：为满足 PG HyperNetwork 所需的输入形式（原本 CodeCircuit 的输出只是离散值），我们将在 `graph_to_vector.py` 里将 42 个切面的有效图节点重要性做 Tensor 化处理，通过时空域上的 `Mean/Max Pooling` 合成出 `[batch, fixed_seq_len, dense_dim]` 的稠密连续向量返回。

---

> [!NOTE]
> 为保证文档权责明确，“实施阶段 (Execution Phases)”以及“硬件算力调度”的逐步操作命令清单已经被拆分。如需查看具体落地的 4 个 Phase 的代码路径与硬件安排，请参阅 [执行方案 (Execution Plan)](file:///C:/Users/11152/.gemini/antigravity/brain/7c823eb3-a885-4abf-bb9a-fca06c44edb4/execution_plan.md)
> 

---

## 决策日志 (Decision Log) 📋

> 本节记录每一个关键决策点的选择过程和备选方案。如果后续实验效果不理想，可以回溯到这里，了解当时为何做了某个选择，并快速切换到备选方案。
> 
> *注：基于最近的讨论，所有初始的开放性问题均已得出结论，并作为 D6, D7, D8 记录在案。*

---

### D1：SAE (Transcoder) 安放位置

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | 按迭代次数展开为多层 (Unroll)，每层的 `mlp.down_proj` 输出处 (残差流) 安装 SAE |
| **决策理由** | 与 CodeCircuit 对 Gemma 的做法完全一致（每层 MLP 输出处挂 Transcoder）。展开为虚拟层可以让同一权重在不同循环时间步成为图中的独立节点，完美表达多步推理路径。 |

**备选方案 A：只在 `L_level` 的综合输出处挂一个 SAE（即每个 H_cycle 结束时的"汇总层"）**
- 优点：图更小，计算快
- 缺点：丢失了"同一个 H_cycle 内部不同 L_step 之间的精细推理路径"
- **何时考虑切换**：如果图太大导致 Phase 2 跑不动（OOM 或时间不可接受），可以先退到这个粗粒度版本

**备选方案 B：在 Attention 输出处也挂 SAE（不仅是 MLP）**
- 优点：捕获注意力模式的可解释特征
- 缺点：CodeCircuit 原版也没这样做（它冻结了 attention pattern 的梯度），且 SAE 训练数据量翻倍
- **何时考虑切换**：如果 Phase 3 产出的特征向量对下游任务区分度不够，可以尝试增加注意力层面的特征节点

**CodeCircuit 原始做法参考**：`replacement_model.py` 第 177-178 行，对每层 `block.mlp` 包裹 `ReplacementMLP`，在 `hook_in` 和 `hook_out` 上操作。`_configure_gradient_flow()` 第 197-198 行冻结了 `hook_pattern`（attention pattern），只让梯度通过 MLP 路径流动。

---

### D2：归因 Target（反向传播的起点标量）

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | Cross-Entropy Loss（模型预测 logits 与 ground truth labels 之间的交叉熵） |
| **决策理由** | 自带自适应加权——模型有把握的位置梯度自然小，不确定的位置梯度大。通用性强，对"部分变化"和"全部变化"的 ARC 题型都适用。 |

**备选方案 A：全格 Logit 和（所有输出位置的正确颜色 logit 之和）**
- `target = sum(logit[i][gt[i]] for i in all_positions)`
- 优点：最简单直接
- 缺点：背景色的"复制噪声"会淹没真正的推理路径
- **何时考虑切换**：如果发现 Cross-Entropy 的梯度过于集中在极少数位置（长尾效应），可以试试这个"均匀"版本

**备选方案 B：Delta 变异位置的 Logit 和（只选 input ≠ output 的格子）**
- `target = sum(logit[pos][gt_color] for pos in delta_positions)`
- 优点：极其干净，只追踪"变化推理"电路
- 缺点：对于"旋转整个网格"等全变化题型，退化为方案 A
- **何时考虑切换**：如果 Cross-Entropy 效果一般，且数据集中大部分题型是"局部变化"型，这个方案可能更好

**备选方案 C：Top-K 不确定位置的 Logit 和**
- 先跑一遍前向，找出模型预测置信度最低的 K 个位置，只对这些位置求导
- 优点：自适应聚焦，不需要 ground truth
- 缺点：引入额外的前向推理成本；且"不确定"不等于"重要"
- **何时考虑切换**：如果需要在没有 ground truth 的场景（如 test set）下提取电路

**CodeCircuit 原始做法参考**：`attribute.py` 第 204-209 行 `compute_salient_logits()`，选最后一个位置的 top-k logits（累积概率 ≥ 95%），用它们的 unembedding 向量作为归因起点。这对 TRM 不直接适用，因为 TRM 不是 next-token prediction。

---

### D3：Halt 不定长处理

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | 方案 α：尊重原生 Halt，图深度不等，Phase 3 用归一化统计消除深度差异 |
| **决策理由** | 归因应反映模型的真实决策过程。强行跑额外步骤会引入模型不会执行的幻象信号。CodeCircuit 本身也产生不等大的图并用统计量消解。 |

**备选方案 β：强制跑满最大 H_cycles，后用 Halt Mask 剪枝**
- 所有题统一跑满 3 个 H_cycle → 统一 12 层深度图
- Phase 3 剪掉 halt 之后的 Dead Nodes
- 优点：工程简单，图结构统一
- 缺点：halt 之后强行跑出的梯度会**污染** halt 之前那些层的归因值（因为前向值变了），即使后来砍掉节点，影响已经发生
- **何时考虑切换**：如果方案 α 导致特征向量不可比问题严重到归一化都救不了（例如 1-cycle 和 3-cycle 的特征分布完全不同类），可以试试方案 β

**备选方案 γ：按 H_cycle 数分组处理**
- 把所有题目按实际 H_cycle 数分桶（1-cycle 一桶、2-cycle 一桶、3-cycle 一桶）
- 每桶内部独立训 classifier
- 优点：桶内图深度一致，无需归一化
- 缺点：样本被拆分为多个子集，每桶样本量可能不够
- **何时考虑切换**：如果方案 α 和 β 都不理想，且发现不同 cycle 数的题目确实需要完全不同的分类策略

**TRM 原生行为参考**：`trm.py` 第 207-216 行，前 `H_cycles-1` 次在 `torch.no_grad()` 内运行，只有最后一次有梯度。第 267-287 行，`q_halt_logits > 0` 触发 halt，且评估时固定用最大步数。

---

### D4：与 CodeCircuit 的偏离点汇总

以下是我们的方案与 CodeCircuit 原始实现不同的所有地方：

| # | CodeCircuit 原始做法 | 我们的 TRM 适配 | 偏离原因 | 风险等级 |
|---|-------------------|---------------|---------|---------|
| 1 | 使用 `HookedTransformer` (TransformerLens) | 自定义 `TRM_Attribution_Wrapper` | TRM 不是 HF 模型，无法转换为 HookedTransformer | ⚠️ 中 — Wrapper 实现复杂度高 |
| 2 | 使用预训练好的 Gemma Transcoder | 自己从头训练 TRM-Transcoder | 没有现成的 TRM SAE | ⚠️ 中 — SAE 质量直接决定图质量 |
| 3 | Target 是最后一个位置的 top-k logit | Target 是全格 Cross-Entropy Loss | ARC 输出是多维网格而非单 token | 🟢 低 — 数学上等价于多起点归因 |
| 4 | 图深度固定 = 模型层数 (如 26 层) | 图深度可变 = 实际 halt 步数 × L_steps | TRM 递归 + 自适应 halt | ⚠️ 中 — 需要归一化处理 |
| 5 | 冻结 attention pattern 梯度 | 同样冻结（在 Wrapper 中实现） | 与 CodeCircuit 一致 | 🟢 低 |
| 6 | 冻结 LayerNorm scale 梯度 | TRM 用 RMSNorm，同样冻结 | 与 CodeCircuit 一致 | 🟢 低 |
| 7 | `selected_features` 按 influence 排序选 top-K | 相同策略 | 直接复用 | 🟢 低 |
| 8 | 图节点包含 Token、Feature、Error、Logit | 替换为 Input Grid Token、Feature、Error、Output Grid Token | ARC 的输入输出语义不同 | 🟢 低 |

> [!WARNING]
> **最高风险点**：D4 中第 1 项（Wrapper）和第 2 项（自训 SAE）。如果 SAE 重建质量差（completeness score 低），整个图就不可信。建议在 Phase 0 结束后先用 `compute_graph_scores()`（来自 `graph.py` 第 250-297 行）检查 replacement score 和 completeness score，确认 > 0.7 再继续。

---

### D5：特征提取策略

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | 完全化用 CodeCircuit 的 `_extract_advanced_features()` 策略，补充 TRM 特有维度（`n_actual_H_cycles`、路径归一化） |
| **决策理由** | CodeCircuit 的图本身就不等大（不同 prompt 的 Feature 节点数差异很大），它已经验证了"统计量压缩"策略的可行性 |

**备选方案 A：直接用 GNN (图神经网络) 编码整张图**
- 用 GCN/GAT 直接把图编码为向量，免去手工特征设计
- 优点：端到端，不丢信息
- 缺点：需要额外训练 GNN，样本量可能不够
- **何时考虑切换**：如果手工特征的分类效果见顶（如 AUC < 0.75），可以试试 GNN

**备选方案 B：用 Graph Kernel (WL Kernel) 做相似度**
- 不提取向量，直接算图之间的核相似度，输入 SVM
- 优点：不需要设计特征
- 缺点：大规模计算慢
- **何时考虑切换**：如果想做图之间的相似性分析而非分类

**CodeCircuit 原始做法参考**：`graph_dataset.py` 第 152-358 行 `_extract_advanced_features()` + `_extract_topological_and_edge_features()`，产出约 22 + n_layers 维的特征向量。`5_classification.py` 用 RandomForest 做分类。

---

### D6：SAE 字典大小设计

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | `d_sae = 4096` (即 TRM hidden_size 512 的 8 倍扩展) |
| **决策理由** | CodeCircuit 中 Gemma 的 hidden_size=2048，字典范围在 16K~65K (8x~32x扩展)。由于 TRM 是轻量模型，保守选择 8x 扩展 (4096) 可以兼顾显存开销与稀疏表达能力，且跑 Phase 0 的收集和训练均非常快。 |

**备选方案 A：更宽的字典 (16384 即 32x 扩展)**
- 优点：更高的稀疏度，特征也许更加互相解耦，具有更精细的可解释性。
- 缺点：训练时间和推理建图时的 Taylor influence 计算开销呈指数级增长；且 TRM 简单推理可能不需要几万种特征。
- **何时考虑切换**：如果 4096 维的 SAE reconstruction loss 一直居高不下，或者重构出来的 Circuit graph "一片糊" (全是连在一起没有清晰结构的 dense features)。

---

### D7：MVP 测试挂钩 (Hook) 切面数量

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | 在跑收集脚本 (collect_activations.py) 时，一上来就获取全部的虚拟切面 (42 个切面：3 H_cycles × (6 L_cycles + 1) × 2 L_layers) |
| **决策理由** | 基于当前的配置 (`cfg_wu4trm.yaml` 中配置 `H=3, L=6, L_layers=2`)，全切面总量为 42。考虑到我们在 8xH200 上进行推理搜集，42 个切面完全在极高算力设备的并发容载范围之内，无需为了"MVP"去手动缩减层数，能够一次性拿到无偏差的、包含时间序列的完整分布。 |

**备选方案 A：仅挂少部分切面 (如只挂最后一个 H_cycle 的层)**
- 优点：收集时显存占用极小。
- 缺点：因为在完整的推理流程中，不同深度的特征激活分布截然不同（深层往往更加收敛专注），只收集后半部分会导致前几层推理建图时特征无法被 SAE 识别出来 (OOD)。
- **何时考虑切换**：如果在 8xH200 上跑满全数据全切面导致极度的 Out Of Memory，可考虑先跑前一小部分进行代码测试。

---

### D8：下游任务接口适配 (为了接入 HyperNetwork PG)

| 项目 | 内容 |
|------|------|
| **决策时间** | 2026-04-07 |
| **当前选择** | 特征提取阶段需舍弃/调整纯离散的图网络拓扑指标，转而针对 HyperNetwork 需求输出“连续可微的统计特征 (Dense embedding)” |
| **决策理由** | 原版 CodeCircuit 后接 RandomForest 分类器接收离散值即可。而我们的下游是 PG HyperNetwork (Transformer 的 Cross-Attention Condition 输入)。它的要求是输入的张量具有强连续性，这就要求我们将提取到的 Attribution 图通过 Pooling 转化为包含持续信号的连续矩阵。 |

**备选方案 A：直接输出离散值并给 HyperNetwork 强行套 Embedding 层**
- 优点：不需要大改 Phase 3 代码，提取的指标具有强人工可解释性（比如 "该张图密度为0.25" 直接 embedding 进去）。
- 缺点：难以和模型深度连续的 `hidden states` 特质吻合，可能会阻断 PG 通过链式法则学习图特征梯度的能力。
- **何时考虑切换**：如果在连续性 Pooling 处理上碰到非常大的工程难度，并且证明 HyperNetwork 的 Tokenizer 能很好的捕捉低维度离散结构。
