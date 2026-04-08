# PRD：将 CodeCircuit 迁移到 TRM 的 query 级 attribution graph 方案

## 任务
为 TinyRecursiveModels（TRM）在 ARC-AGI-1 上提取 **CodeCircuit 风格** 的 query 级 attribution graph，产出一份详细的迁移与分析方案；本阶段**只做规划，不实现代码，不跑实验**。

## 硬约束
- 未来如果实现，只能写入 `CodeCircuit_TRM_Arc1/` 与 `.omx/`。
- 不得修改 `TinyRecursiveModels/` 或 `TRM_WU_Project/`。
- 目标对象保持为：**每个 ARC query 一张图，解释最终预测**。
- 如果要尽量忠实于 CodeCircuit，**canonical 路线必须建立在 TRM-transcoder 的稀疏特征上**。直接在 hidden state / channel 上做 attribution 只能算探索性 baseline，不能算 canonical 的 CodeCircuit-style circuit。

## 证据基线
- CodeCircuit 不是直接对原始神经元抽图；它是通过 `ReplacementModel.from_pretrained(...)` 加载 transcoder set，再在这个 sparse feature basis 上做 attribution（`CodeCircuit/circuit_tracer/replacement_model.py:119-214`；`CodeCircuit/data/3_generate_graph.py:8-18,51-106`）。
- CodeCircuit 的 attribution 目标是 final-position salient logits，并估计 feature 到 logits 的 direct linear effects（`CodeCircuit/circuit_tracer/attribution/attribute.py:1-20,37-66,171-239`）。
- TRM 是递归 ARC 模型：输入为 grid tokens + puzzle identifiers，内部有 `H_cycles/L_cycles`、`z_L/z_H`、逐位置 logits 与 halt logits（`TinyRecursiveModels/models/recursive_reasoning/trm.py:32-63,118-222,249-297`）。
- TRM 当前 shipped forward 有明显的 attribution 边界：早期 recurrence 在 `torch.no_grad()` 下运行，carry 也会被 detach（`TinyRecursiveModels/models/recursive_reasoning/trm.py:207-222`）。
- TRM 不像 Gemma 那样只有一个天然的“MLP 替换点”；可能的 feature-bearing surface 至少包括 block `mlp` 输出、attention 输出、`L_level` 输出、`z_L`、`z_H`（`TinyRecursiveModels/models/recursive_reasoning/trm.py:65-115,149-150,184-220`）。
- ARC 的 query-level 语义来自：逐位置 logits -> `argmax(preds)` -> crop / inverse-aug -> 后续 vote aggregation（`TinyRecursiveModels/models/losses.py:61-64`; `TinyRecursiveModels/evaluators/arc.py:69-177`; `TinyRecursiveModels/pretrain.py:374-405`）。

## 结论
采用 **transcoder-first semantic port**：
1. 保留 CodeCircuit 的 sparse-feature circuit 思路；
2. 先把 TRM 的 query-level target 定义清楚；
3. 先设计并训练 TRM-transcoder；
4. 再基于这个 sparse feature space 去做 canonical circuit extraction。

## 备选方案
### 方案 A —— 推荐：transcoder-first semantic port
- 先为选定的 TRM reasoning surface 训练 transcoder；
- 再在稀疏特征空间里做 query-level circuit extraction；
- 这是最接近 CodeCircuit 原方法的路线。

### 方案 B —— 给 TRM 套一层 `ReplacementModel` 兼容壳
- 表面接口像 CodeCircuit；
- 但底层架构差异太深，不可信。

### 方案 C —— 先做不带 transcoder 的 hidden-state attribution
- 可以更快得到一个图；
- 但只能当 baseline，不能当 canonical 方案。

## Canonical query graph 定义
- **对象**：一个 ARC query 在 **evaluator vote aggregation 之前** 的一次最终推理结果，也就是 **pre-vote**。
- **target scope**：所有 cropped、inverse-aug 后的 canonical output cells。
- **每个 cell 的目标标量**：predicted-token logit margin = `logit(预测类) - max(其它类 logits)`。
- **并列最大值规则**：继承 `torch.argmax(outputs["logits"], dim=-1)` 的语义，即取 vocab 维度上第一个最大值索引。
- **metadata 必须包含**：seq index、cropped cell 坐标、canonical cell 坐标、预测 token/color、target margin。
- `q_halt_logits` 只能是辅助 metadata 或 side-target，不能是主目标。

## Canonical feature-space 定义
- Canonical source node 必须是 **TRM-transcoder 产生的稀疏特征**。
- 因此，canonical 的 CodeCircuit-style circuit extraction 必须在**先训练好 TRM-transcoder**之后才能开始。
- 第一个 transcoder 的关键问题是**挂载在哪个 surface**，而不是默认“直接替换 MLP”。
- MVP 的首选候选 surface：`L_level` block `mlp` / block 后 hidden update。
- 必须显式比较的备选：attention 输出、`L_level` 输出、`z_L`、`z_H`。
- 若单一 surface 不够，后续可以扩展到 multi-transcoder，但 MVP 必须先选定一个 canonical surface。
- 没有 transcoder 的 hidden-state/channel attribution 只能是 non-canonical baseline。

## 聚合规则
- 先对每个 canonical target cell 分别构建一张 subgraph。
- 再通过 seq-index / cell mapping 把它们映射回统一坐标系。
- 最终 query graph 的聚合方式固定为：**edge-wise sum**。
- 不允许 average、normalize、vote-weight。
- metadata 必须包含逐 cell contribution table。

## 相对原始 CodeCircuit 的必要偏离
1. 不再解释 final-position next-token，而是解释 **所有 canonical output-cell 的 predicted-token logit margin**。
2. 不再是单目标 attribution，而是多目标子图的 **edge-wise sum**。
3. Gemma 的既有 sparse feature basis 不能直接复用，必须先建立 **TRM-native transcoder**。
4. “TRM-transcoder 挂在哪里”本身就是一级设计问题，不能偷换成“先把 MLP 换掉就行”。
5. 必须在 `CodeCircuit_TRM_Arc1/` 中做 TRM-native analysis stack，而不是直接复用 `ReplacementModel`。
6. direct-effect 的诚实表述只能是两种之一：
   - 只覆盖最后一个可微 recurrent cycle；
   - 或在 `CodeCircuit_TRM_Arc1/` 中重建 analysis-only forward。

## 按优先级排列的路线
### Phase 0 —— 先锁定语义
- 锁定 target 定义、metadata 契约、聚合规则，以及“canonical circuit 必须基于 sparse transcoder features”这条原则。

### Phase 1 —— 先设计第一个 TRM-transcoder
- 比较 candidate surfaces；
- 选定一个 MVP canonical surface；
- 明确 transcoder 的训练目标、稀疏性目标、重建对象与质量门槛；
- 明确 transcoder 特征如何进入后续 attribution 路径。

### Phase 2 —— 锁定工作区与复用边界
- 后续模块只允许出现在 `CodeCircuit_TRM_Arc1/analysis`、`adapters`、`graph`、`specs`、`transcoders`。
- 保持对 `TinyRecursiveModels/` 与 `TRM_WU_Project/` 的 denylist。

### Phase 3 —— 解决 recurrence honesty
- 二选一：
  1. 只解释最后一个可微 recurrent cycle；
  2. 单独实现 analysis-only forward。
- 且所选 honesty mode 必须与 transcoder surface 相兼容。

### Phase 4 —— 设计 TRM 原生 attribution backend
- 定义 adapter 输入、target 抽取、metadata 映射、sparse feature activation readout 与 backward 路径。

### Phase 5 —— 未来 MVP（当前不执行）
- 一个 checkpoint、一个 ARC query、一个训练好的 TRM-transcoder、一张 pre-vote query graph。
- 输出 graph + 完整 metadata + transcoder surface + honesty mode。

## MVP 定义
MVP 不是“先搞出一张图再说”，而是：
- 一个 query；
- 一张 pre-vote 图；
- 全部 canonical output-cell margins；
- 一个选定并训练好的 TRM-transcoder；
- 基于稀疏特征的 extraction；
- edge-wise-sum 聚合；
- 完整 metadata；
- 不触碰受保护目录。

## 验收标准
- Canonical target 精确定义为：all canonical output cells + predicted-token logit margin + pre-vote。
- Canonical feature space 精确定义为：一个明确选定 surface 上训练出来的 TRM-transcoder sparse features。
- 文档明确指出：no-transcoder attribution 是 non-canonical。
- 聚合规则精确定义为 edge-wise sum。
- metadata 必须包含 cell mapping、逐 cell contribution、transcoder surface、honesty mode。
- transcoder 设计/训练必须先于 canonical circuit extraction。
- 全程仍是 planning-only，并遵守工作区限制。

## 主要风险
- 第一个 transcoder surface 选错。
- Transcoder 质量不够，学不到有意义的 sparse features。
- 在 recurrence / no-grad 边界上把 direct effect 说大了。
- query-level 聚合掩盖逐 cell 因果。
- 误写受保护目录。

## ADR 摘要
**决策**：transcoder-first semantic port。
**原因**：CodeCircuit 的 circuit 不只是“有一张图”，而是建立在稀疏特征基底上的图；如果不先建立 TRM-transcoder，就只是在模仿外形，而不是复用方法核心。
**后果**：未来工作的第一优先级是 transcoder surface 选择与质量门槛，而不是直接进入 canonical circuit extraction。
