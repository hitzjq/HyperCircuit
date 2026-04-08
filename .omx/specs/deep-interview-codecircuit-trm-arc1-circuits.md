# Deep Interview Spec: CodeCircuit-style circuits for TRM on ARC-AGI-1

## Metadata
- Profile: standard
- Rounds: 6
- Final ambiguity: 17.3%
- Threshold: 20%
- Context type: brownfield
- Context snapshot: `.omx/context/codecircuit-trm-arc1-circuits-20260406T095851Z.md`
- Transcript: `.omx/interviews/codecircuit-trm-arc1-circuits-20260406T101848Z.md`

## Clarity Breakdown
| Dimension | Score |
|---|---:|
| Intent | 0.90 |
| Outcome | 0.85 |
| Scope | 0.84 |
| Constraints | 0.68 |
| Success | 0.78 |
| Context | 0.84 |

## Intent
用户希望借用 `CodeCircuit` 的方法，为 `TinyRecursiveModels` 在 `ARC-AGI-1` 上的每个 query 提取类似 `attribution graph` 的 circuit，并提前识别哪些组件/假设不能直接复用，再给出一份可执行的迁移方案。

## Desired Outcome
产出一份详细方案，明确：
1. `CodeCircuit` 中哪些模块/思路可以直接复用；
2. 哪些地方不能直接用于 TRM；
3. 每个不兼容点的原因；
4. 一个按优先级排序的迁移方案；
5. 一个最小可运行版本（MVP）路线。

## In-Scope
- 比较 `CodeCircuit` 与 `TinyRecursiveModels` 在模型接口、输入表示、输出目标、graph 构造、hook 点上的异同。
- 将 CodeCircuit 的“prompt-specific attribution graph”定义迁移到 TRM/ARC 场景。
- 以 `ARC-AGI-1` 为第一阶段目标。
- 把目标对象定义为：**每个 query 一张 graph，解释该 query 的最终输出预测**。
- 若底层需要按多个输出位置/输出 logits 做 attribution 再合并，允许作为实现细节提出。
- 方案中明确说明任何必须调整的地方，并说明理由。

## Out-of-Scope / Non-goals
- 当前不直接修改代码。
- 当前不跑实验。
- 当前不训练新模型。
- 当前不扩展到 `ARC-AGI-2`。
- 当前不把任务转交给 Ralph 执行实现。

## Decision Boundaries
- 我可以主动完成差距分析与技术方案细化。
- 如果发现必须偏离 CodeCircuit 的原始定义或 pipeline，我必须显式向用户说明调整点。
- 只要最终交付仍保持“尽量类似 CodeCircuit attribution graph”的目标形式，我可以在实现层建议必要的适配（如多输出位置 attribution 汇总为 query-level graph）。

## Constraints
- 结果形式要尽量贴近 CodeCircuit 的 attribution graph，而不是完全自由重定义 circuit。
- 分析对象不是 halt-only，也不是天然的一格一图，而是优先面向 query-level final prediction。
- 交付停留在详细方案层，不进入实现。

## Testable Acceptance Criteria
以下内容全部出现则本轮任务达标：
1. 明确列出 `CodeCircuit` 中可直接复用的模块/思路；
2. 明确列出不能直接复用的模块/假设；
3. 对每个“不兼容点”给出为什么不行；
4. 给出按优先级排序的迁移方案；
5. 给出 MVP 路线；
6. 任何必须调整 CodeCircuit 定义/流程的地方都被显式标注。

## Assumptions Exposed + Resolutions
- 假设 1：CodeCircuit 的 circuit 可以原样迁移到 TRM。
  - Resolution: 大概率不能原样迁移，因为 CodeCircuit 面向 HuggingFace 文本生成模型与 final-position salient logits，而 TRM 是自定义递归、非自回归、grid 输出模型。
- 假设 2：最像 CodeCircuit 的 target 可能是 halt 决策或单 cell。
  - Resolution: 否。最像的是“每个 query 一张 graph，目标是该 query 的最终输出预测”。
- 假设 3：当前应直接进入实现。
  - Resolution: 否。本轮只做详细差距分析与迁移方案。

## Pressure-pass Findings
- 初始请求中的“每个 query 的 circuit”是模糊的；经过追问后，已压实为 query-level final prediction graph。
- 初始请求没有说清当前是否允许实现；经过追问后，已明确本轮只做方案，不实施。
- 用户一开始没有给出方案粒度；经过追问后，已明确需要“可复用点 / 不兼容点 / 原因 / 优先级路线 / MVP”。

## Brownfield Evidence vs Inference
### Evidence
- `CodeCircuit/data/3_generate_graph.py` 使用 `ReplacementModel.from_pretrained(...)` 和 `attribute(prompt=...)`，输入是文本 prompt，输出 graph `.pt`。
- `CodeCircuit/circuit_tracer/attribution/attribute.py` 围绕 final-position salient logits 构建 prompt-specific attribution graph。
- `TinyRecursiveModels/models/recursive_reasoning/trm.py` 是自定义递归推理模型，输入为 ARC grid token 序列与 `puzzle_identifiers`，含 `H_cycles/L_cycles`、`q_halt_logits`，输出非自回归 logits。
- `TinyRecursiveModels/evaluators/arc.py` 对 query/puzzle 的预测与 halt 分数做聚合。
- `TinyRecursiveModels/dataset/build_arc_dataset.py` 将 ARC 数据编码为固定 30x30 grid 序列。

### Inference
- 若要保持 CodeCircuit 风格的 query-level graph，TRM 很可能需要把多个输出位置的 attribution 聚合成一张 graph。
- `circuit_tracer` 当前面向 HuggingFace/ReplacementModel 的抽象大概率无法直接套到 TRM，需要适配层或重写局部接口。

## Technical Context Findings
- 关键 CodeCircuit 入口：
  - `CodeCircuit/data/3_generate_graph.py`
  - `CodeCircuit/circuit_tracer/attribution/attribute.py`
  - `CodeCircuit/circuit_tracer/replacement_model.py`
- 关键 TRM 入口：
  - `TinyRecursiveModels/models/recursive_reasoning/trm.py`
  - `TinyRecursiveModels/dataset/build_arc_dataset.py`
  - `TinyRecursiveModels/puzzle_dataset.py`
  - `TinyRecursiveModels/evaluators/arc.py`

## Recommended Execution Bridge
**Recommended:** `$ralplan`
- Why: 需求已经足够清晰，但还需要把“差距分析 + 迁移方案”正式整理成可执行计划与测试规范；这比直接进 Ralph 更合适。
- Suggested next invocation:
  - `$ralplan .omx/specs/deep-interview-codecircuit-trm-arc1-circuits.md`

## Handoff Note
下游规划必须保留以下约束：
- 目标形式尽量贴近 CodeCircuit attribution graph；
- 图的语义对象是“每个 ARC query 的最终输出预测”；
- 任何偏离原始 CodeCircuit 定义的适配都必须被显式说明；
- 当前阶段不实现、不跑实验。
- D:\Project with Jiefu\HyperCircuit\TinyRecursiveModels和D:\Project with Jiefu\HyperCircuit\TRM_WU_Project这两个下面的文件不要做任何改动，如果需要的话，可以在D:\Project with Jiefu\HyperCircuit下面创建一个副本，再进行改动
