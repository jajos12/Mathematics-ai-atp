# Accepted GNN proof search

## Submission scope and provenance

This branch extends upstream `cf97980b6ace56b2783c45020da2764a9dda9d70`
with the accepted flat policy, matched text retriever, native best-first search,
independent proof verification, and a fixed miniF2F validation runner.

| Component | Implementation |
| --- | --- |
| Graph/model/index | Extended existing `atp_lean_gnn/{graph,parser,pyg,model,argument_selector,lemma_index,premise_pool}.py` |
| Accepted bundle and flat inference | `joint_bundle.py`, `joint_source_decoder.py`, `text_retriever.py`, `live_policy.py` |
| Normalized input and reference formatting | `model_graph.py`, `flat_actions.py` |
| Native execution and original-target verification | `native_session.py`, `proof_tasks.py`, `proof_verification.py`, `lean/GnnSearch.lean` |
| Search, cohort, and reporting | `proof_search.py`, `benchmark.py`, existing `reporting.py` |

Model packaging follows `ea72894` and the published accepted release. Extraction
reference `83118df10a8a2d4113c2ab9d4ce9295b000d2b0e` precedes the first template
commit, `f1c8eb50dcdfbf6af19271cbe265fed299e0253c`. Search semantics were reviewed
from `facf6b7`; its structured-supervision changes were not carried over.
Shared upstream API behavior and dependency pins are retained.

The model's existing GRU/count/source/local/library decoder predicts zero to
three flat references. The newer template grammar, source-program renderers,
structured decoder, training/export pipelines, and template-native validation
experiments are outside this submission. Normalized S-expressions here describe
**model input**, not generated action templates.

## Frozen model

- Repository: `iCogLabs-Dev/Mathlib-Sexpr-GNN`.
- Revision: `8aba62625dfeb833e4591498700f351f3bc60af9`.
- Format: `joint_source_system_v1`; policy: `joint_source_flat_decoder_v2`.
- Pair: joint epoch 3, local refinement epoch 8, retriever/index epoch 5.
- Policy tensor SHA-256: `a16d7f63ddde721c35611dcc2c604096cbe31d17490407669e2516c72d38e2ce`.
- Retriever tensor SHA-256: `48ac6ab921bd8a6a7c2c49b3fdc307b97293a9f3737f33484673141f456aa5fd`.
- Bundle manifest SHA-256: `ebc7620282838e7a0d26250ddbf159ec02c859700c17217d9a43db3ef4145107`.
- GAT: hidden 128, four layers, four heads, state/mean/attention readout.
- Eleven trained node types; bidirectional edges; retrieval K=200 over 318,953 lemmas.
- Default profile: `accepted_local_cues`; explicit comparison profile: `base`.

The loader verifies the pinned manifest, every component file, tensor hashes,
tokenizer, corpus identities, normalized FAISS rows, architecture, and runtime
contract. Keep the downloaded bundle intact, including its corpus and index.

## Installation and native environment

Use Python 3.11+ and the submission checkout. The checked-in lock includes
Transformers and preserves upstream PyPantograph's revision. For a fresh Python
environment:

```bash
uv sync --locked
source .venv/bin/activate
```

An existing CUDA environment can install the project after its declared runtime
dependencies are present with `python -m pip install --no-deps -e .`. Its actual
package versions, device, GPU properties, precision settings, and thread counts
are bound into the benchmark report. Use the same environment for resume.

Existing server artifact paths can be reused from an isolated submission
worktree. From the submission root, define:

```bash
export RESEARCH_ROOT=/home/jovyan/yonatan/Mathematics-ai-atp
export BUNDLE="$RESEARCH_ROOT/maths_ai/_support_files/artifacts/hf_joint_model"
export SOURCE_ROOT="$RESEARCH_ROOT/maths_ai/_support_files/sexpr_environment/mathlib4"
export PANTOGRAPH_ROOT="$RESEARCH_ROOT/maths_ai/_support_files/sexpr_environment/Pantograph"
```

For a new bundle download, use an empty destination:

```bash
python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("iCogLabs-Dev/Mathlib-Sexpr-GNN", revision="8aba62625dfeb833e4591498700f351f3bc60af9", local_dir=os.environ["BUNDLE"])'
```

Required native pins:

| Dependency | Revision |
| --- | --- |
| Lean | `leanprover/lean4:v4.10.0-rc1` |
| Mathlib | `29dcec074de168ac2bf835a77ef68bbe069194c5` |
| Pantograph model-S-expression fork | `73781c2d58456e4bf369dadd4a501e1b78a0b177` |

For fresh native directories, install `elan`, then run:

```bash
elan toolchain install leanprover/lean4:v4.10.0-rc1
git clone --filter=blob:none --no-checkout https://github.com/leanprover-community/mathlib4.git "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" fetch --depth=1 origin 29dcec074de168ac2bf835a77ef68bbe069194c5
git -C "$SOURCE_ROOT" checkout --detach 29dcec074de168ac2bf835a77ef68bbe069194c5
(cd "$SOURCE_ROOT" && lake exe cache get)
git clone --filter=blob:none --no-checkout https://github.com/jajos12/Pantograph.git "$PANTOGRAPH_ROOT"
git -C "$PANTOGRAPH_ROOT" fetch --depth=1 origin 73781c2d58456e4bf369dadd4a501e1b78a0b177
git -C "$PANTOGRAPH_ROOT" checkout --detach 73781c2d58456e4bf369dadd4a501e1b78a0b177
(cd "$PANTOGRAPH_ROOT" && lake build)
```

The runtime checks clean tracked native sources, toolchain/revisions, compiler
version, package manifest, helper hash, and compiled backend entry points. It
combines Mathlib and Pantograph `LEAN_PATH` entries automatically. The helper is
included in installed Python wheels. Every run includes a native/independent
verifier preflight; its tiny fixture is infrastructure-only and is not a solve.

## Fixed validation cohort

Repository `yangky11/miniF2F-lean4`, commit
`5746b7d6c47855ce1294bed87329618ff7f1bc31`; **all 244 validation tasks**,
including AMC and mathd. Order is ascending SHA-256 of `42:<name>`, exactly the
full selection used by the separate DeepSeek baseline. The official test split
is untouched. Selection is frozen before parsing or compatibility results.

```bash
python -B -u -m maths_ai.gnn_inference.scripts.run_gnn_benchmark \
  --cohort-dir maths_ai/gnn_inference/artifacts/minif2f_valid_244 \
  --prepare-only
```

Preparation saves the pinned index, original sources and hashes, theorem
prefix/header boundaries, and cohort metadata. Interrupted downloads resume
from an owned cache. A complete baseline cohort cache can be read without
modification; incomplete external caches require a new destination. Only source
tasks are reused, never baseline proofs or previous model responses.

The pinned miniF2F source targets Lean **4.24.0** and Mathlib
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. This model's native environment is
Lean **4.10.0-rc1**, so these runs are explicitly labeled **`development_port`**.
The original theorem text is retained. Parsing/elaboration incompatibilities
remain failures in the 244-task denominator.

## Launch and resume

Run from this branch's root in the chosen Python environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -u -m maths_ai.gnn_inference.scripts.run_gnn_benchmark \
  --bundle "$BUNDLE" --source-root "$SOURCE_ROOT" \
  --pantograph-root "$PANTOGRAPH_ROOT" \
  --cohort-dir maths_ai/gnn_inference/artifacts/minif2f_valid_244 \
  --output-dir maths_ai/gnn_inference/runs/gnn_valid_244_accepted \
  --device cuda --profile accepted_local_cues \
  --model-calls 32 --tactic-trials 96 --nodes 64 \
  --depth 16 --top-k 8 --seconds 120 --seed 42
```

Use `--device cpu` for CPU evaluation. The initial output directory must be new.
For task-boundary resume, repeat the **identical command** with `--resume`.
Code, cohort, source context, bundle, environment, device/settings, and budgets
must match the saved binding. A different profile/budget requires a new run.
An output-directory lock prevents concurrent writers from spending duplicate
task budgets.

Completed tasks retain their costs. Successful proofs are independently checked
again on resume. An interrupted in-flight task becomes a terminal failure;
its counted model requests/trials remain, and it receives no fresh search
budget. After an abrupt process kill, elapsed costs are explicitly lower bounds;
that label persists through subsequent resumes. Frontier-level recovery is not
implemented. A missing/tampered completed task file is an error, not permission
to rerun the task.

## Search and proof acceptance

The frontier contains parent/action edges ordered by cumulative negative log
probability. Each expanded state receives one model call and at most eight
top-ranked tactic families, with one greedy argument sequence per family.
Full scores include tactic, argument count, and each source/reference choice.
The auxiliary stop head does not decide action length.

All native states preserve original context, local let/instance information,
hidden context indices, metavariables, and sibling goals. OOV local identities
are resolved from original DAG labels rather than vocabulary IDs. Local names
and global references are quoted; ambiguous/inaccessible references and binding
arguments are rejected. Bare `intro` remains executable. The target theorem and
native helper declarations are excluded from retrieved premises; targets already
available through imports/context are rejected.

A native zero-goal snapshot is only a proof candidate. A fresh Lean process
replays its focused actions against the original prefix/header, checks all
goals, and audits the theorem's actual axioms. Only `propext`, `Classical.choice`,
and `Quot.sound` are allowed. `sorryAx` and other assumptions fail verification.

There is no proof LLM, repair, generation fallback, PLN ranking, value model, or
hand-written benchmark automation pass. Automation tactics proposed by the GNN
are normal model actions.

## Budgets, costs, and output

- Per task: 32 attempted model calls, 96 native trials, 64 nodes **including
  root and verification-rejected children**, depth 16, top 8 tactic families.
- Active budget: 120 seconds covering inference, search, checkpointing, and
  independent verification; excludes model loading and initial native startup.
  Synchronous model calls are checked at operation boundaries and cannot be
  preempted mid-inference, so elapsed time can exceed this soft wall limit.
- Native trial timeout: 20 seconds; initial startup: 120 seconds; independent
  verification: at most 30 seconds and remaining active budget.
- Native defaults: 1,000,000 heartbeats and recursion depth 2048.

`report.json` contains bindings, immutable cohort, task digests, status/solve/cost
totals, and invocation histories. Invocations record model loading, preparation,
environment validation, preflight, runner time, and successful-proof rechecks.
`tasks/<id>.json` records actions, nodes, model requests/responses, trials,
verification source/diagnostics, native startup, and end-to-end task timing.
`summary.csv` exposes task groups, statuses, verified flags, counts, and timing.

Report verified solves divided by **244**, all input/runtime/budget/interruption
failures, model calls and trials, active seconds, startup/end-to-end seconds,
loading/preflight/recheck costs, profile, hardware, and environment label.
The separate DeepSeek baseline uses its own generator, automation, compute,
and timing protocol: same cohort does not imply a matched-cost comparison.
This validation result is not official-test or pretraining-held-out accuracy.

## Diagnostic and acceptance commands

Single original-target diagnostic:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -u -m maths_ai.gnn_inference.scripts.run_gnn_search \
  --bundle "$BUNDLE" --source-root "$SOURCE_ROOT" \
  --pantograph-root "$PANTOGRAPH_ROOT" \
  --header 'theorem GnnSubmissionSmoke (P : Prop) (h : P) : P' \
  --output maths_ai/gnn_inference/runs/submission_smoke.json --device cpu
```

Use an existing output parent directory and a new file. `--prefix-file` preserves
trusted original imports/namespaces/options/notation/declarations. `--goal` is
also available for a new standalone target.

```bash
MATHS_AI_PANTOGRAPH_ROOT="$PANTOGRAPH_ROOT" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -m pytest maths_ai/gnn_inference/tests maths_ai/api/tests -q
```

Additional numerical acceptance against the preserved research/pre-template
checkout, imported in a separate subprocess:

```bash
MATHS_AI_GNN_BUNDLE="$BUNDLE" MATHS_AI_GNN_REFERENCE="$RESEARCH_ROOT" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -m pytest maths_ai/gnn_inference/tests/test_accepted_model_parity.py -q
```

Local acceptance covers existing GNN/API behavior, strict normalized/OOV
mapping, saved-parent best-first ordering, independent-verifier rejection,
resource limits, native context/notation/let/instance/sibling handling, timeout
cleanup, task interruption/resume/tampering, cost totals, and CSV output.
Accepted-weight comparison checks graph features, node/state embeddings,
retrieval IDs/vectors/scores, tactic/count/source/reference scores, decoded
arguments, actions, and full log probabilities within `1e-6` on three fixtures.
Wheel installation/import and packaged-helper checks also passed.

Release acceptance on November 13, 2026: **135 tests passed** with all three
acceptance environment variables set (including numerical parity and a real
accepted-weight native cohort/resume fixture). An independently installed wheel
using lock-selected Transformers **5.12.1** proved the identity diagnostic with
one model call, one native trial, two nodes, and fresh axiom-audited verification.
`uv lock --check --offline` passed. These fixtures are acceptance checks, not
miniF2F solve-rate measurements.

All 244 pinned sources were downloaded, parsed, and cache-revalidated locally.
Local native acceptance uses the pinned backend with Init; full Mathlib is
available on the research server rather than this local checkout. The full
244-task solving run and its solve rate remain the benchmark to execute.
