# Test Spec：CodeCircuit 到 TRM 的 query 级 attribution 规划验证

## 目的
验证这份方案是否精确、证据充分、以 transcoder-first 为核心，并且仍然严格停留在规划阶段。

## 精确验收项
- [ ] 文档明确写出：当前阶段只做规划，不做实现/实验。
- [ ] 未来写入范围被限制在 `CodeCircuit_TRM_Arc1/` 与 `.omx/`。
- [ ] `TinyRecursiveModels/` 与 `TRM_WU_Project/` 被明确列为禁止写入目录。
- [ ] 文档明确区分概念复用与代码复用。
- [ ] 文档明确指出：CodeCircuit 的 canonical 方法依赖 transcoder 派生的稀疏特征基底。
- [ ] Canonical query object 被定义为 evaluator vote aggregation 之前的一次最终推理。
- [ ] Canonical target scope 是所有 cropped inverse-aug canonical output cells。
- [ ] Canonical target scalar 是 predicted-token logit margin。
- [ ] 文档要求精确的 seq-index / cell mapping metadata。
- [ ] Canonical feature space 被定义为：一个明确选定 surface 上训练出的 TRM-transcoder sparse features。
- [ ] 文档明确说明：不带 transcoder 的 hidden-state/channel attribution 是 non-canonical baseline。
- [ ] 文档明确要求先做 transcoder-surface 选择，而不是默认“替换 MLP 就够了”。
- [ ] 首选 MVP surface 以及备选 surface（attention、`L_level` output、`z_L`、`z_H`）都被点名。
- [ ] 聚合规则是 edge-wise sum。
- [ ] metadata 必须包含逐 cell contribution table。
- [ ] recurrence / no-grad 边界被明确指出。
- [ ] direct-effect 的诚实表述被限制为两种：最后一个可微 cycle，或 analysis-only forward。
- [ ] 阶段顺序中，transcoder 设计/训练先于 canonical circuit extraction。
- [ ] MVP 明确包含：训练好的 transcoder、sparse-feature extraction、完整 metadata，以及不触碰受保护目录。

## 证据检查
主要论断应能回指到：
- `CodeCircuit/data/3_generate_graph.py`
- `CodeCircuit/circuit_tracer/attribution/attribute.py`
- `CodeCircuit/circuit_tracer/replacement_model.py`
- `TinyRecursiveModels/models/recursive_reasoning/trm.py`
- `TinyRecursiveModels/dataset/build_arc_dataset.py`
- `TinyRecursiveModels/models/losses.py`
- `TinyRecursiveModels/evaluators/arc.py`
- `TinyRecursiveModels/pretrain.py`

## 审查流程
1. 确认推荐方案仍然是 Option A。
2. 确认方案是 transcoder-first，而不是 hidden-state-first。
3. 确认 canonical feature space 建立在 sparse features 上，并且 no-transcoder baseline 被标成 non-canonical。
4. 确认 target、aggregation、metadata、honesty mode 都是精确定义，而不是示意描述。
5. 确认工作区 denylist 仍然有效。

## 退出条件
只有当以上验收项全部通过，且下列文件存在时，本次 planning 才算完成：
- `.omx/plans/prd-codecircuit-trm-arc1-circuits.md`
- `.omx/plans/test-spec-codecircuit-trm-arc1-circuits.md`
