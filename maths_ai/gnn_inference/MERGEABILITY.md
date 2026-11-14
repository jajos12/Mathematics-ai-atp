# Prover mergeability and persistent-atp integration audit

## Pinned scope

This is an integration audit and implementation handoff, not a declaration that
the combined persistent service is production-ready.

| Component | Inspected revision |
| --- | --- |
| Mathematics-ai-atp main | `4b0ce1e1a76100ce485b801befc084143090032e` |
| Accepted GNN submission | `1315dca67529590e8729b65e27308490951afae9` |
| MCTS-AC PR #84 | `7d1b750b8b752117d997fc471befdeb79ababb29` |
| Existing v2-serving PR #85 | `6a719f3bf506f7f8fcb65e90d6e8dc246d9dfb8a` |
| persistent-atp dev | `b9ebd964110d6bc219d25fccea03966e18c28627` |

PR #84 is a substantial alternative runtime/API implementation: its diff from
the common ancestor contains 463 files, 81,614 insertions, and 5,209 deletions.
Its successful focused tests do not establish compatibility with main's API.

## General system picture

```text
Persistent ATP / coordinator / context compiler
  - durable proof store, research strategy, context/premise selection
  - claims/declarations, leases/fencing, journal revisions, commit gate
  - asks for bounded formal attempts; independently admits proof evidence
                     |
              FormalATPAdapter
         start / status / resume-or-retry / cancel / replay
                     |
         one Mathematics-ai-atp public service boundary
         request identity, admission, budgets, persistence,
         environment/source binding, artifacts, result conversion
                     |
         explicitly selected internal prover backend
         + accepted flat GNN + matched retrieval + best-first
         + MCTS actor-critic + PUCT/best-first + optional PLN guidance
                     |
         compatible native Lean context and proof verification
                     |
         bound candidate/certificate + diagnostics + actual costs
                     |
         persistent-atp replay/commit decision
```

Our bounded prover is the inner loop. Long-horizon knowledge persistence and
commit authority belong to persistent-atp. A local MCTS evaluation job and a
coordinator-owned formal run are different objects, even when both are durable.

The GNN and MCTS policies both use flat source-aware actions. MCTS adds an actor
residual and critic, sampling/PUCT, worker ownership and certification machinery;
this is separate from the excluded template/structured-decoder research.

## Existing contracts

### Persistent boundary

`persistent-atp/src/mathproof/formal_atp.py` defines the five boundary methods.
The canonical request schema binds proof, claim, declaration, run, revision,
lease, fencing token, Lean source artifact, environment identity/hash, goal,
optional hypotheses, search policy, model/corpus references, and budget.

There are three distinct implementation paths:

1. `MathsAIFormalATP` imports the older Python `HybridReasoner` directly and
   converts its hypergraph. It does not automatically select our new prover or
   the MCTS service. Its resume method performs another `prove` call; persistent
   Thompson-sampler state is not a saved native search frontier.
2. `MathsAIHTTPFormalATP` translates canonical requests into main's provider
   envelope and translates its asynchronous snapshots/hypergraph results back.
3. `HTTPFormalATP` expects a provider to implement the canonical JSON protocol
   natively. It is not interchangeable with the current provider translator.

Main's HTTP envelope is `run_id/problem/environment/budget`, not the canonical
request verbatim. A direct canonical POST returning 422 is expected when the
translation adapter is bypassed.

### Mathematics main and accepted submission

Main exposes `/prove`, `/formal-search/start`, status/retry/cancel routes,
`/formal-replay`, component prediction/scoring, and readiness in `/health`.
Its default builder still constructs the older hybrid reasoner. Adding our
CLI/native modules does **not** wire the measured GNN-only search into this
default service. The 58/244 benchmark result applies to the frozen CLI backend.

### MCTS PR #84

The PR defines another `maths_ai.api` application with `/prove`, `/evaluations`,
evaluation status/result/cancel, `/health`, and `/ready`. It has a process
supervisor, bounded queue, typed resource outcomes, SQLite evaluation storage,
and kernel-certified proof artifacts. Its `/prove` response is
`certified/candidate/outcome/...`, rather than main's
`solved/exhausted/termination_reason/proof_trace/summary`.

Those capabilities are useful implementation components, but the application
does not implement persistent-atp's current formal lifecycle routes.

## Verified blockers

### 1. API replacement breaks the existing caller

`PR84:maths_ai/api/app.py:69` defines the competing app. Actual router probes
show all five existing formal lifecycle/replay routes missing; a formal start
returns 404. A legacy `/prove` request containing `timeout` returns 422 because
the MCTS request uses `timeout_s` and rejects unknown fields.

Resolution: retain the public contract and place search engines behind an
explicit backend/profile seam. Version or namespace incompatible MCTS-native
request/response routes rather than replacing main's handlers during conflict
resolution. Keep formal runs distinct from dataset evaluation jobs.

### 2. Replay attestation is incomplete across repositories

`main:maths_ai/api/service.py:180` returns only `verified/status`.
`persistent-atp:src/mathproof/maths_ai_http.py:552-566` requires successful
replay to attest `run_id`, `certificate_hash`, and `environment_hash`.

The real adapter rejects the provider response with:

```text
replay verification unavailable: provider did not attest the run, certificate,
and actual environment
```

Resolution: validate the actual source/runtime binding, replay the stored proof,
derive the certificate identity from that stored evidence, and return the bound
attestation. Echoing a client-supplied environment hash is insufficient.

### 3. Resource exhaustion is translated into an internal error

`main:maths_ai/hybrid_reasoner/hypergraph.py:310-311` defines exhaustion as
either node-budget exhaustion or a dead root. The persistent HTTP adapter at
`src/mathproof/maths_ai_http.py:172-179` assumes exhaustion means a dead root.

A real provider graph with an open root and `mark_budget_exhausted()` becomes
`internal-error: formal-search exhausted flag contradicts the root status`.

Resolution: translate the explicit termination reason and budget flags.
Budget/depth/time limits with live obligations remain bounded-search outcomes;
frontier exhaustion maps to stagnation, and protocol/runtime faults retain their
separate error categories.

### 4. Canonical step limits are dropped

`persistent-atp:src/mathproof/maths_ai_http.py:298-316` maps
`wall_clock_ms` to `timeout`, but drops `budget.max_steps` and does not forward
the selected search policy/model/corpus references. The probe's request for
three steps and 5,000 ms becomes only `{"timeout": 5.0}`.

Resolution: define enforceable backend-neutral limits and acknowledge the
resolved policy. A tactic-trial cap, node cap, and PUCT simulation cap are
different quantities. Unsupported limits/policies must be rejected explicitly.
Apply any agreed `search_policy.max_tactic_steps` bound too. Report actual costs.

### 5. Source and environment metadata do not establish actual execution binding

The current HTTP translator places the source artifact and environment hash in
`environment.variables`. Main's `_request_args` uses only goal/hypotheses/budget;
it does not resolve that artifact or select/validate the requested environment.
Our original-prefix/header native path already provides useful execution and
verification machinery, but is not connected to this provider envelope.
Main's `_verify_declaration_sync` also reconstructs a theorem with only
`import Init`; it does not preserve the original Mathlib imports/prefix used by
the GNN backend. Its replay path cannot simply substitute for that verifier.

Resolution: agree on an artifact resolver and environment manifest/registry.
Search and replay must use the same original declaration/context and pinned
runtime. Preserve all coordinator identity fields for result association; the
coordinator remains responsible for lease/fencing and commit authorization.

### 6. Shared model and native-client refactors need explicit reconciliation

Taking PR #84's shared modules alongside our accepted loader reproduces:

```text
cannot import name 'PublishedPointerClassifier' from
'maths_ai.gnn_inference.atp_lean_gnn.argument_selector'
```

The MCTS branch has refactored the encoder/model/decoder APIs. Its checkpoint
converter is explicit and requires original checkpoint bindings, prepared
metadata, candidate manifest, and retrieval deployment metadata. A published HF
directory is not a drop-in format-4 actor-critic checkpoint. Conversion adds a
new critic; it does not establish critic training or improved proof performance.

Use one canonical graph/model/retrieval implementation with a checked accepted-v2
compatibility/conversion layer and numerical parity gates. Include PR #85's
owner in this reconciliation: it touches the same accepted-model and live-state
serving modules. Merging all implementations independently duplicates ownership.

Native contracts also differ despite sharing Lean 4.10 and Mathlib `29dcec07`:

| Path | Native/client binding |
| --- | --- |
| Frozen GNN baseline | direct Lean helper; Pantograph provenance `73781c2d...` |
| PR #85 serving | PyPantograph `3322696...`; compatible Pantograph `0dd5d222...` |
| PR #84 | async client `2b326508...`; contract revision 7; companion base `0dd5d222...`, patched source `46a33e1c...` |

Do not substitute runtime directories or compare hashes as though these were
identical deployments. Validate/rebind a shared deployment explicitly, or retain
separate declared backend runtime profiles/processes.

## Mergeability and tests performed

- Accepted submission + main: clean trial merge, including dependencies/lock.
- Combined GNN/API/hybrid regression suite with native and numerical acceptance:
  **155 passed**, zero skipped; three dependency deprecation warnings.
- Combined `uv lock --check --offline`: passed.
- PR #84 vs main: **25 conflicted paths**, including the public API, model core,
  hybrid reasoner, PLN integration, Docker, and dependency lock.
- PR #84 vs the frozen GNN submission: **25 conflicted paths**, with additional
  graph/index/training reconciliation in that pairing.
- PR #84 API/config tests with its pinned async client: **23 passed**. These use
  fake inference in real runtime processes, not trained-model native acceptance.
- Persistent adapter/HTTP tests at the pinned dev revision: **52 passed** plus
  five subtests. Their success alone does not test the actual provider pair.
- Cross-repository probes used the real routers, formal-run persistence,
  hypergraph serialization and external translation/validation. Search/replay
  themselves were injected fixtures; these probes make no Lean proof claim.

`scripts/probe_formal_atp_contract.py` reproduces the cross-repo observations:

```bash
python scripts/probe_formal_atp_contract.py \
  --provider-root /path/to/combined-main-and-gnn \
  --persistent-root /path/to/persistent-atp --mode main

python scripts/probe_formal_atp_contract.py \
  --provider-root /path/to/MCTS-AC-checkout \
  --persistent-root /path/to/persistent-atp --mode mcts
```

Run in an environment containing the provider's declared imports; the MCTS probe
needs its pinned async client. The script imports the selected provider in a
fresh process, reports revisions/pending merge, labels all data `fixtures_only`,
and describes observed gaps rather than declaring deployment acceptance.

## Recommended implementation and merge order

1. Integrate latest main into our submission using the verified clean merge;
   retain the frozen benchmark commit as provenance. Preserve main's recent
   syntax-error classification, Lean-only proof closure, and CPU/GPU deployment
   fixes.
2. Agree on one public formal-run contract with the persistent-atp maintainers:
   identity/source/environment binding, backend policy selection, budget units,
   certificate hashing/attestation, result dispositions, and retry capabilities.
3. Expose the accepted GNN search as a backend of that service. Reuse its bundle,
   normalized live policy, native session, verifier, and search; do not invoke
   the benchmark CLI per HTTP request or create another search implementation.
4. Reconcile the MCTS shared core and accepted-weight adapter with PR #85, then
   integrate its scheduler/supervisor/worker and evaluation capabilities behind
   compatible routes. Keep PLN readiness specific to profiles that enable PLN.
5. Add end-to-end tests that call the real persistent adapter against both
   selected backends: bounded start/poll, actual cancellation, honest retry,
   budget exhaustion, unavailable/mismatched environments, source identity,
   proof artifacts and independently verified replay attestation.
6. Run a small trained-model native integration suite and fresh deployment
   smoke before merging the combined service. Model conversion/native contract
   parity and HTTP contract parity are separate gates.

Current restart semantics must remain explicit: main retries start the original
request again; the GNN benchmark resumes at completed-task boundaries; MCTS's
evaluation jobs become interrupted on restart. None of these establishes native
frontier continuation under a persistent-atp `resume` call. Declare capabilities
and charge cumulative costs instead of silently replenishing budgets.
