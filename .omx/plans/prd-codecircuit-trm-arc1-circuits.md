# PRD: CodeCircuit-to-TRM Query-Level Attribution Graph Migration Plan

## Task
Produce a planning-only migration plan for extracting CodeCircuit-style query-level attribution graphs for TinyRecursiveModels (TRM) on ARC-AGI-1. No code changes or experiments in this stage.

## Hard constraints
- Future implementation must live only in `CodeCircuit_TRM_Arc1/` plus `.omx/`.
- No writes under `TinyRecursiveModels/` or `TRM_WU_Project/`.
- Target object remains: **one graph per ARC query explaining final query prediction**.
- To stay faithful to CodeCircuit, the **canonical** path must use **TRM-transcoder sparse features**. Raw hidden-state/channel attribution may exist only as a non-canonical exploratory baseline.

## Evidence baseline
- CodeCircuit graph generation loads `ReplacementModel.from_pretrained(...)` and calls `attribute(prompt=...)` over text prompts (`CodeCircuit/data/3_generate_graph.py:8-18,51-106`).
- `ReplacementModel.from_pretrained(...)` loads a transcoder set, so CodeCircuit attribution is built on a learned sparse feature basis, not raw neurons (`CodeCircuit/circuit_tracer/replacement_model.py:119-214`).
- CodeCircuit attribution targets final-position salient logits and computes direct linear effects between features and logits (`CodeCircuit/circuit_tracer/attribution/attribute.py:1-20,37-66,171-239`).
- TRM is a recurrent ARC model with grid-token inputs, puzzle identifiers, `H_cycles/L_cycles`, per-position logits, and halt logits (`TinyRecursiveModels/models/recursive_reasoning/trm.py:32-63,118-222,249-297`).
- TRM recurrence is only partially differentiable in the shipped forward path because early cycles run under `torch.no_grad()` and carry is detached (`TinyRecursiveModels/models/recursive_reasoning/trm.py:207-222`).
- TRM has multiple possible feature-bearing surfaces, not one obvious Gemma-like MLP-only replacement point: block-local `mlp`, attention outputs, `L_level` outputs, `z_L`, `z_H` (`TinyRecursiveModels/models/recursive_reasoning/trm.py:65-115,149-150,184-220`).
- ARC prediction/eval semantics are query-level grid prediction followed by crop/inverse-aug handling and later vote aggregation (`TinyRecursiveModels/models/losses.py:61-64`; `TinyRecursiveModels/evaluators/arc.py:69-177`; `TinyRecursiveModels/pretrain.py:374-405`).

## Decision
Adopt a **transcoder-first semantic port**:
1. keep CodeCircuit’s sparse-feature circuit idea,
2. define TRM query targets explicitly,
3. design/train a TRM-transcoder before canonical circuit extraction,
4. then build a TRM-native attribution backend around that sparse feature space.

## Options considered
### Option A — Favored: transcoder-first semantic port
- Train/select a TRM-transcoder on a chosen TRM reasoning surface first.
- Then extract query-level circuits over sparse features.
- Best fidelity to CodeCircuit.

### Option B — compatibility shim around `ReplacementModel`
- Force TRM into CodeCircuit’s existing API shape.
- Rejected because the mismatch is architectural, not superficial.

### Option C — no-transcoder prototype first
- Start with hidden-state/channel attribution, postpone sparse features.
- Allowed only as a debugging baseline, not as the canonical path.

## Canonical query-graph definition
- **Object:** one final inference pass **before evaluator vote aggregation** for one ARC query.
- **Target scope:** all cropped, inverse-aug canonical output cells.
- **Target scalar per cell:** predicted-token logit margin = `logit(predicted token) - max(other logits)`.
- **Tie-break rule:** inherit `torch.argmax(outputs["logits"], dim=-1)` semantics from `TinyRecursiveModels/models/losses.py:61-64`.
- **Required metadata:** sequence index, cropped cell coordinate, canonical cell coordinate, predicted token/color, target margin.
- `q_halt_logits` may be auxiliary metadata only.

## Canonical feature-space definition
- Canonical source nodes are **TRM-transcoder sparse features**.
- Canonical CodeCircuit-style extraction therefore requires a **trained TRM-transcoder before graph extraction**.
- The first design step is **surface selection**, not blindly “replace the MLP”.
- Preferred MVP candidate: `L_level` block `mlp` / post-block hidden update.
- Must explicitly compare against: attention outputs, `L_level` outputs, `z_L`, `z_H`.
- If one surface is insufficient, later phases may add multiple transcoders.
- Hidden-state/channel attribution without a transcoder is non-canonical.

## Deterministic aggregation rule
- Build one subgraph per canonical target cell.
- Map each subgraph back through seq-index/cell metadata.
- Aggregate by **edge-wise sum** only.
- No averaging, normalization, or vote-weighting in the canonical object.
- Metadata must include per-cell contribution tables.

## Key deviations from original CodeCircuit
1. Replace final-position next-token targets with **all canonical output-cell margins**.
2. Replace single-target attribution with **multi-target edge-wise-sum aggregation**.
3. Replace Gemma’s existing sparse feature basis with a **TRM-native transcoder**.
4. Treat transcoder placement as a first-class design decision; TRM is not “just replace the MLP”.
5. Replace `ReplacementModel` with a TRM-native analysis stack in `CodeCircuit_TRM_Arc1/`.
6. Constrain direct-effect claims to either:
   - last differentiable recurrent cycle only, or
   - a separate analysis-only forward in `CodeCircuit_TRM_Arc1/`.

## Prioritized migration plan
### Phase 0 — lock semantics
- Freeze target definition, metadata contract, aggregation rule, and the rule that canonical circuits use sparse transcoder features.

### Phase 1 — design the first TRM-transcoder
- Compare candidate surfaces.
- Choose one canonical MVP surface.
- Specify transcoder training objective, sparsity objective, reconstruction target, and quality gates.
- Define how transcoder features participate in later analysis.

### Phase 2 — lock workspace and reuse boundaries
- Keep all future work under `CodeCircuit_TRM_Arc1/analysis`, `adapters`, `graph`, `specs`, `transcoders`.
- Preserve the denylist against `TinyRecursiveModels/` and `TRM_WU_Project/`.

### Phase 3 — solve recurrence honesty
- Pick exactly one honesty mode:
  1. last differentiable recurrent cycle only, or
  2. analysis-only forward without the shipped `no_grad` boundary.
- Ensure the chosen mode is compatible with the transcoder surface.

### Phase 4 — build attribution design
- Define adapter inputs, target extraction, metadata mapping, sparse feature activation readout, and backward path.

### Phase 5 — MVP later
- One TRM checkpoint, one ARC query, one trained TRM-transcoder, one pre-vote query graph.
- Output graph + full metadata + transcoder surface + honesty mode.

## MVP definition
The MVP is **not** “first graph at any cost”. It is:
- one query,
- one pre-vote graph,
- all canonical output-cell margins,
- one chosen trained TRM-transcoder,
- sparse-feature-based extraction,
- edge-wise-sum aggregation,
- full metadata,
- zero writes to protected directories.

## Acceptance criteria
- Canonical target is exact: all canonical output cells, predicted-token logit margin, pre-vote.
- Canonical feature space is exact: trained TRM-transcoder sparse features on one chosen surface.
- No-transcoder attribution is explicitly marked non-canonical.
- Aggregation rule is exact: edge-wise sum.
- Metadata includes cell mapping, per-cell contributions, transcoder surface, and honesty mode.
- Transcoder design/training precedes canonical extraction.
- Plan remains planning-only and respects workspace restrictions.

## Main risks
- Wrong first transcoder surface.
- Weak transcoder quality.
- Overclaiming direct effects across the no-grad recurrence boundary.
- Query-level aggregation hiding per-cell causality.
- Accidental writes into protected directories.

## ADR summary
**Decision:** transcoder-first semantic port.
**Why:** CodeCircuit circuits depend on sparse features; reproducing only the graph shell without the sparse feature basis would not be method-faithful.
**Consequence:** future work starts with transcoder surface selection and quality gates before canonical circuit extraction.
