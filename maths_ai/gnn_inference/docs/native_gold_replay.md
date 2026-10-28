# Source-native gold-action replay

First integration milestone: check recorded actions and their flat rendering
against actual Lean tactic snapshots. This runs no GNN, retrieval, training
or proof search, and does **not** measure complete-theorem proof success.

## What is checked

1. Select theorem/file pairs deterministically from the requested split,
   before checking cache availability or action support. Include every
   recorded row of each selected theorem; failed/unsupported rows remain
   in the report denominator.
2. Validate original source-invocation caches against dataset state/tactic/
   target hashes. Require normalized and v3 source-syntax sidecars bound to
   those exact raw records.
3. Compile original source files using the pinned frontend. Locate each
   invocation by its original unit/index and UTF-8 byte range.
4. Capture `GoalState` from `mctxBefore` and all active goals, running inside
   original `ContextInfo` (environment, namespace, open declarations and name
   generator). Independently restore that same native snapshot for original
   and rendered tactics. No pretty-printed goal reconstruction or `sorry`
   insertion is used.
5. Compare normalized input target/local types, indices, roles and names
   with cached model inputs. Compare all ordered resulting goals with the
   original invocation, including local let values. This is normalized
   model-input/transition parity, not equality of complete kernel states.

Renderer supports already-resolved flat local/global references, with the
same three-reference budget as current decoder. List tactics get brackets
(`rw [h]`, `simp [h]`); term tactics keep application syntax (`exact f x`).
Public global names are fully qualified; locals resolve by context index.
Ambiguous/inaccessible locals, private/numeric declaration names requiring
additional resolution, unannotated leaves, compound syntax and fresh-name
arguments remain explicit unsupported cases. Proof placeholders are not
rendered as proof actions. This is not a full source-syntax renderer.

## Backend and scope

Use existing extraction environment:

- Mathlib `29dcec074de168ac2bf835a77ef68bbe069194c5`.
- Pantograph `73781c2d58456e4bf369dadd4a501e1b78a0b177`.
- Lean `leanprover/lean4:v4.10.0-rc1`.

The pinned protocol exports source invocation data but no command to restore
an arbitrary recorded invocation. `lean/ReplayGold.lean` uses its existing
frontend/GoalState APIs directly; no backend fork modification is needed.
Both projects' `lake env` search paths are resolved before running the Lean
helper. Source files and cached traces are read only; temporary request/
response files are deleted automatically. Original-source compilation
errors invalidate that file's replay results.

An invocation's active goals can be only part of a larger proof. Even an
empty resulting goal array is **not** counted as a completed theorem.
Full-proof reconstruction/rechecking and live model inference are later
milestones. Existing hybrid search is not changed by this runner.

## First server run

Pull published code and run from repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.replay_gold_actions \
  --source-root maths_ai/_support_files/sexpr_environment/mathlib4 \
  --pantograph-root maths_ai/_support_files/sexpr_environment/Pantograph \
  --cache-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-dir maths_ai/gnn_inference/runs/native_gold_replay_smoke \
  --split val --num-theorems 3 --seed 42 --file-timeout 600
```

No environment rebuild, new dataset extraction or model checkpoint is
needed if the pinned extraction environment and sidecars already exist.
Missing dependencies or wrong/dirty source pins fail explicitly.

Report is `OUTPUT_DIR/report.json`, checkpointed after each source file.
Repeat identical options with `--resume` after interruption. Resume binds
selected rows, source/cache hashes, backend pins, helper/runner/renderer
code and timeout. Use a new directory for a changed cohort or implementation.

Read final summary counts:

- `selected_steps`: every selected dataset row.
- `renderable_steps`: renderer produced a candidate string, before Lean checking.
- `input_model_match`: live normalized inputs agree with cached model inputs.
- `original_transition_match`: original tactic replays to recorded normalized goals.
- `rendered_transition_match`: rendered gold references reproduce that transition.
- `status_counts`: cache, capture, input-parity, original/rendered failures or unsupported syntax.
- `proof_success_measured`: always false in this milestone.

Exit 0 means every selected row either matches or is explicitly unsupported
by the renderer after successful original replay. Exit 2 means operational,
cache, parity or replay failures; inspect per-row details. Neither is a
proof-search score. First evaluate this smoke run before increasing cohort.

## Verification

**38 focused tests passed**, including a real pinned-backend integration
check against existing extraction API coordinates. Native cases cover
local/global names, rewrites, function application, let bindings, local
instances, preserving two sibling goals and independent branch restoration.
A later theorem is unavailable in an earlier source snapshot. Unit checks
cover unsupported references, deterministic selection, digest binding,
failure denominators and report resume.

Native integration can be repeated with:

```sh
MATHS_AI_PANTOGRAPH_ROOT=/path/to/pinned/Pantograph \
  python -m pytest maths_ai/gnn_inference/tests/test_gold_replay.py -q
```

Without that variable, the real-backend integration test is skipped while
unit tests still run. No server Mathlib replay result has been measured yet.
