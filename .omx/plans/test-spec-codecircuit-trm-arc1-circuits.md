# Test Spec: CodeCircuit-to-TRM Query-Level Attribution Plan

## Purpose
Verify that the plan is precise, evidence-grounded, transcoder-first, and still planning-only.

## Precision checks
- [ ] The plan is explicitly planning-only.
- [ ] Future writes are confined to `CodeCircuit_TRM_Arc1/` plus `.omx/`.
- [ ] `TinyRecursiveModels/` and `TRM_WU_Project/` are explicitly denied write targets.
- [ ] The plan separates concept reuse from code reuse.
- [ ] The plan states that CodeCircuit’s canonical method depends on a transcoder-derived sparse feature basis.
- [ ] The canonical query object is one final inference pass before evaluator vote aggregation.
- [ ] The canonical targets are all cropped inverse-aug canonical output cells.
- [ ] The target scalar is predicted-token logit margin.
- [ ] The plan requires exact seq-index/cell mapping metadata.
- [ ] The canonical feature space is trained TRM-transcoder sparse features.
- [ ] The plan explicitly marks no-transcoder hidden-state/channel attribution as non-canonical.
- [ ] The plan requires an explicit transcoder-surface choice instead of assuming “replace the MLP” is automatically enough.
- [ ] The preferred MVP surface and the alternative surfaces (`attention`, `L_level` output, `z_L`, `z_H`) are named.
- [ ] The aggregation operator is edge-wise sum.
- [ ] Metadata must include per-cell contribution tables.
- [ ] The recurrence/no-grad boundary is explicitly called out.
- [ ] The plan limits direct-effect claims to exactly two honesty modes: last differentiable cycle only, or analysis-only forward.
- [ ] Transcoder design/training precedes canonical circuit extraction in the phase order.
- [ ] The MVP includes a trained transcoder, sparse-feature extraction, full metadata, and no protected-directory writes.

## Evidence checks
The plan should ground major claims in:
- `CodeCircuit/data/3_generate_graph.py`
- `CodeCircuit/circuit_tracer/attribution/attribute.py`
- `CodeCircuit/circuit_tracer/replacement_model.py`
- `TinyRecursiveModels/models/recursive_reasoning/trm.py`
- `TinyRecursiveModels/dataset/build_arc_dataset.py`
- `TinyRecursiveModels/models/losses.py`
- `TinyRecursiveModels/evaluators/arc.py`
- `TinyRecursiveModels/pretrain.py`

## Review procedure
1. Confirm Option A remains the favored option.
2. Confirm the plan is transcoder-first rather than hidden-state-first.
3. Confirm the canonical feature space is sparse-feature-based and non-transcoder baselines are marked non-canonical.
4. Confirm the target contract, aggregation rule, metadata contract, and honesty modes are exact.
5. Confirm the workspace denylist remains intact.

## Exit criteria
Planning is complete only if all precision checks pass and both files exist:
- `.omx/plans/prd-codecircuit-trm-arc1-circuits.md`
- `.omx/plans/test-spec-codecircuit-trm-arc1-circuits.md`
