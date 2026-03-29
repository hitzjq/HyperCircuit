# SHINE x TRM (ARC-AGI-1) 融合技术方案设计

本项目旨在利用 **SHINE** 框架的能力，为 **TRM (TinyRecursiveModels)** 基座模型在推理每个 ARC-AGI-1 任务 (Query) 时，动态生成专属的 LoRA 权重 (Query-Specific LoRA) 以辅助其递归推理能力。

这里结合 `SHINE` 的 MetaNetwork 设计理念和 `TRM` `run_lora_arc1.sh` 中的微调配置，给出一份完整的逻辑方案与数据流分析。

---

## 1. 核心理论对接

### SHINE 的核心思想
SHINE (Scalable In-Context Hypernetwork) 的本质是：
`Context (证据/提示) -> 提取 Memory States -> MetaNetwork -> 生成 LoRA 权重矩阵 -> 注入 Base Model 提升当前 Query 表现`。

### TRM 端到端映射
在 ARC-AGI-1 任务中，结合 TRM 的特点，可以做如下映射：
- **Context (Evidence)**: ARC 的原始网络/题目属性。由于 TRM 自带了对谜题标识符 (puzzle_identifiers) 到 embedding 的映射，我们可以直接视 `puzzle_emb(puzzle_identifiers)` 为 Context 表征。
- **Memory States**: 输入给 SHINE `MetaNetwork` 的状态张量。可以使用 `puzzle_emb` 的输出直接作为 Memory States。
- **MetaNetwork**: 采用 SHINE 提供的 `MetanetworkLinear` 或 `MetanetworkTransformer` 作为超网络评估器 (HyperModulator)。
- **Base Model (Frozen)**: TRM 模型本身（冻结不参与全局训练）。
- **LoRA 目标层**: TRM 中的 `qkv_proj, o_proj, gate_up_proj, down_proj` 等 CastedLinear。

---

## 2. 详细数据流与计算链路 (Data Flow)

按照 `run_lora_arc1.sh` (Batch Size = 1024, L_layers = 2) 为例。

### Step 1: 提取 Context Embedding 作为 Memory States
在 `lora_finetune.py` 的每一个 Step 中：
通过 `batch["puzzle_identifiers"]` 从冻结或可带度量学习的 `puzzle_emb` 字典中查出当前批次的任务表征：
```python
# memory_states shape: [1024, 1, 64] (假设 puzzle_emb_ndim = 64)
memory_states = model.model.inner.puzzle_emb(batch["puzzle_identifiers"]).unsqueeze(1)
```

### Step 2: MetaNetwork 前向推断生成 LoRA
调用 SHINE 的 `Metanetwork`：
```python
# plain_tensor 是一串大的一维打平的权重参数 [1024, total_lora_params]
plain_tensor = metanetwork(memory_states)

# 利用 SHINE 的 generate_lora_dict 根据结构将一维平铺张量切割
# 得到针对每一层每一个样例专属的 lora_A: [1024, in_dim, r] 和 lora_B: [1024, r, out_dim]
loradict = metamodel.generate_lora_dict(lora_r=16, plain_tensor=plain_tensor)
```

### Step 3: 带有 Hook 的 TRM 递归推理
TRM 是一个不断循环的动态图模型。SHINE 原有做法是将 `loradict` 直接传入 Base 模型推理。
为了不硬改 TRM 极度优化的底层递归逻辑（特别是 `halted` 状态机制），我们采用 **Forward Hook 注入策略**：
1.在每一层的目标线性层（如 `qkv_proj`）注册 `register_forward_hook`。
2.当 TRM 执行到目标模块时，Hook 被唤醒。拦截输入张量 `x`（shape 为 `[1024, seq_len, hidden_dim]`）。
3.计算动态 LoRA 残差并加持回激活值中：
```python
# Hook 内部利用 BMM 逐样本计算动态 LoRA
delta_x = torch.bmm(torch.bmm(x_flat, lora_A[layer_idx]), lora_B[layer_idx])
output = original_output + delta_x * scaling
```
4.由于 TRM 的 `halted` 掩码机制，即便不同的 sample 在不同 cycle 中退出，batch 维度始终固定在 `global_batch_size=1024`，BMM 尺寸永远对齐。

### Step 4: 梯度回播与模型更新 (Loss Backward)
- 损失函数（原有的交叉熵或其它 metrics）正常对输出计算。
- Loss 只会在 TRM 的 Hook 中产生梯度路径。TRM 全量冻结，梯度计算将会沿着 `delta_x` 流回 `lora_A/B`，最后流入 `MetaNetwork`。
- Optimizer `step()` 仅更新 SHINE `MetaNetwork` 和可选的 `puzzle_emb`。

---

## 3. 需实施的代码改造路径

1. **依赖引入**: 把 `SHINE/metanetwork_family.py` 里的结构引用到 TRM 的工作流中。
2. **LoRA 降级替换**: 在 `run_lora_arc1.sh` 脚本和 TRM 的 `lora_finetune.py` 代码中，增加 `--use_shine_hypernet` 标记，禁用默认的 TRM 静态 LoRA，转而初始化 SHINE MetaNetwork。
3. **编写 Hook Injector**: 写一个专门的 Python Context Manager 或 wrapper 类 `SHINEHookInjector`，负责在前向发生前根据当前 batch 计算出 `loradict`，然后为 TRM 的目标层挂上基于 batch_size 的 Hook。
4. **训练循环适配**: 确保 `train_batch` 后清理所有的 hook handle！防止显存泄漏。

---

## 4. 关键验证点

目前这套方案的理论依据十分清晰。主要改动点将集中在：
1. 不修改 TRM 核心内部结构，靠 Hook 外挂进行针对每个 Query 不同的 LoRA 参数注入。
2. 选择 `puzzle_emb` 作为 SHINE 的 Memory States 的直接输入。
