# Bounded rw/simp gold templates, V1

This milestone extracts typed **gold targets** and validates their rendering.
It does not add a prediction head, train a model, alter existing prepared
labels or measure proof-search success. The removed library-query experiment
is not part of this work.

## Scope fixed from training evidence

- `rw`: one to three bare local/library references, each with forward or
  backward direction (`←` and ASCII `<-` canonicalize to the same flag).
- `simp`: zero to three bare references, with optional `only`.
- Location: goal (default or explicit `at ⊢`) or one existing local declaration.
- Public global names resolve fully qualified; local names use context IDs.
- Single-child `tacticSeq`/`tacticSeq1Indented`/list wrappers are removed only
  along a single-tactic root path. Multi-tactic blocks remain unsupported.

Rule count, direction and location are distinct fields. Three rules plus a
local location can contain **four reference occurrences**; this gold schema
is not the legacy decoder's three-total-argument format. No trained decoder
has been extended to consume it yet. Repeated references remain legal.

Deferred: applied/parenthesized rules, field/method notation, tuples, fresh
binders, config/discharger syntax, wildcard or multiple local locations,
combined goal-plus-local locations, reverse/erasing simp arguments, and
other tactic families. Old flat rendering remains the default and is reused
for other families in template mode. Within rw/simp, unsupported structures
are rejected rather than silently dropping syntax.

## Target structure

`RwSimpTemplate.to_target()` emits a versioned target such as:

```json
{
  "schema": "rw_simp_template_v1",
  "family": "rw",
  "rules": [
    {"reference": {"source": "library", "identity": "Nat.add_comm"}, "backward": true}
  ],
  "only": false,
  "location": {"kind": "local", "reference": {"source": "local", "identity": 4}},
  "rule_count": 1
}
```

Library identities are fully qualified names; locals are source-context
indices, **not graph-node indices**. Future training conversion must map
them through the bound vocabulary/corpus and preserve scope. Modifier flags
are never relabeled as premise references. These targets use annotated
source-tree roles; unannotated names are not inferred from spelling.

Parsing first calls existing source-trace compiler to validate context
references and exclude missing/unannotated/new-scope syntax. It then checks
exact pinned syntax constructors, child counts, keywords and punctuation.
Template extraction and reference rendering are separate: a private global
may be a valid typed target but still need a resolver before rendering.

## Gold replay

Pull published code, then from server repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_template_v1 \
  --renderer template-v1 --split val --num-theorems 20 --seed 42 --file-timeout 600
```

Same source-native snapshots, pinned toolchain and row-selection policy as
[gold replay](native_gold_replay.md); original and rendered branches are
restored independently and every active goal is compared. Use a new report
directory: renderer, schema and parser-code hashes are resume-bound.

The report includes `template_target`, `template_error_code`, `render_path`
and `flat_renderable` per row. Summary adds:

- `template_v1_parsed_steps`: typed target extracted, before name rendering.
- `flat_renderable_steps`: old renderer could produce text on this cohort.
- `newly_renderable_vs_flat`: new mode produced text where old mode did not.
- `rendered_transition_match`: final rendered action reproduces reference
  normalized goal transition in native Lean.

Static renderability is not execution accuracy. All unsupported/failing
rows remain in the selected denominator. Typed gold structures are supplied
from recorded syntax, not predicted by a model. Full prepared-train V1
coverage and server native replay remain unmeasured at implementation time.

## Verification and next gate

**50 focused tests pass** with real pinned-backend integration enabled.
Native fixtures cover directions, local locations, `simp only`, empty simp
lists, mixed directions, repeated references, trailing commas and three
rules plus a fourth location reference. Combined locations, applied rules
and `simp [*]` are rejected; original source tactics still replay correctly.
Unit tests check wrapper boundaries, role/scope validation, serialized
targets, separate resolution failures and renderer-mode resume isolation.

Evaluate native replay and exact grammar coverage before exporting training
targets or adding a template-aware prediction head. Keep old flat cohort and
newly covered cases separate; no model/whole-proof improvement is claimed.
