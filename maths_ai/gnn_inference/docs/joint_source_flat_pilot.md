# Normalized flat-action joint-source pilot

This is an **offline argument experiment**, not full Lean tactic syntax or a
publishable bundle. The normalized GNN provides local-node vectors; frozen
MiniLM epoch 5 provides library IDs, cosine scores and 384-D lemma vectors.
A learned adapter feeds lemma vectors to the 128-D GRU. Source choice precedes
within-source identity at **every** argument position, including mixed actions.
Library ordering starts as frozen text ranking, with a zero-initialized
GRU-conditioned residual for later argument positions; no text-parser GNN
lemma vectors are mixed with normalized local nodes.

After the first pilot stopped after at most one argument, the revised decoder
predicts **0–3 argument slots explicitly** from state and tactic. It trains
this count on *all* supported actions, even when a cited lemma is absent from
top 200. The old binary stop head remains an auxiliary diagnostic, not a
greedy-length decision. Length classes receive smoothed inverse-frequency
weights from training rows; local-source errors receive a training-only
source weight. On retrieval misses, a reachable gold prefix can still teach
its source, but missing identity and subsequent GRU/stop targets are masked.

The corrected flat-action cohort contains 16,821 train / 322 val local and
42,116 train / 802 val library positions (5,301 / 97 mixed-source actions).
Unsupported syntax receives no argument or stop supervision. Retrieval-absent
gold cannot receive fabricated stop or candidate supervision. Validation never
injects gold; source/local/library ranks are teacher-forced diagnostics, while
complete-action counts use predicted choices. No Lean proof success is implied.

Commands below run on server from repository root after syncing code. Confirm
disk space and available GPU first. Candidate preparation may take substantial
time for 218,727 train states; `--resume` reuses committed row batches. The
table stores IDs/scores only (~350 MB for train top-200), not per-row text
vectors. FAISS index stays in `index_epoch_5`.

```sh
python -u -m maths_ai.gnn_inference.scripts.prepare_text_library_candidates \
  --checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_5_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_5 \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --output-dir maths_ai/gnn_inference/runs/sexpr_text_library_epoch_5 \
  --device cuda:0 --batch-size 32 --faiss-threads 4
```

Review `val_metrics.json`: any/all-citation@200 and micro recall are distinct.
The earlier 69.26% is any-citation retrieval on 2,765 cited validation rows;
this cache must reproduce that cohort before training. New train/val rows are
bound to their corrected action and citation labels and the exact retriever.

```sh
python -u -m maths_ai.gnn_inference.scripts.train_joint_source_decoder \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --candidate-dir maths_ai/gnn_inference/runs/sexpr_text_library_epoch_5 \
  --text-checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_5_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_5 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --baseline-checkpoint maths_ai/gnn_inference/runs/sexpr_available_old_settings/run_20261015_084134/best.pt \
  --pointer-config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --run-dir maths_ai/gnn_inference/runs/sexpr_joint_source_flat_epoch_5_count_pilot \
  --device cuda:0 --epochs 1 --batch-size 32 --max-train-batches 500
```

Use this **new** run directory: the previous `joint_source_flat_decoder_v1`
checkpoint has no count head and is not resumable as the revised v2 model.
This pilot freezes graph backbone and text retriever, training tactic
embeddings, local pointer head, count head, auxiliary stop head, source gate
and lemma adapter.
It resumes within an epoch using `--resume` with identical settings. To
extend a completed run, keep every setting but increase `--epochs` and pass
`--resume`. Its `best.pt` is selected by nonempty oracle-tactic action match,
then count macro recall and reachable loss, **not** by stop-only matches.
Report retrieval-absent rows, both-sources-available source accuracy,
auxiliary stop diagnostics, **per-length count recall**, local/library source
recall and candidate ranks, nonempty actions, and all supported actions
separately. Compare library top-1 to frozen text ordering on the same
reachable argument positions; reject a learned rerank if it worsens that baseline.
Training does not greedily decode actions, so exact-action scores are only
reported for validation. Compare this **same-500-batch** run against v1 before
starting a separate larger-data run (for example, 2,000 batches).

For a **controlled count-loss ablation** after V2's three epochs, repeat the
same 500-batch/epoch command and seed with `--epochs 3`, add
`--count-weight-exponent 1.0`, and use a **new** run directory such as
`runs/sexpr_joint_source_flat_epoch_5_count_inverse_pilot`. Default `0.5`
remains the original square-root inverse-frequency policy; `1.0` weights
rare lengths more strongly from *training* counts only. It cannot resume the
old V2 run because that would switch objectives mid-run. Compare count-2/3
recall, zero/one losses, oracle/predicted nonempty exact, and source/identity
on the same validation cohort; weight changes alone are not a proved fix.

The library pool currently searches the full corpus: declaration accessibility
at the historical proof position remains unverified. Serving integration,
full syntax generation, same-live-state Pantograph parity and Lean-checked
proof search are separate gates before a composite bundle can be exported.

### Optional local type cue (offline experiment only)

`local_type_signals` compares normalized goal/hypothesis S-expressions: literal
type equality for `exact`, and Eq/Iff/HEq side-in-goal for the **first** `rw`,
`simp` or `subst` argument. Soft strengths selected on training rows: **8** for
`exact`/first `rw`/first `subst`, **4** for first `simp`. They are not general Lean-validity
rules. Populate `SourceCandidates.local_type_signal` using the oracle tactic
for teacher-forced diagnostics, or the **predicted** tactic for greedy decode;
set `local_type_tactic` to that same family. Finalized optional offline
settings are:

```python
decoder.local_type_bonus_strengths = {
    "exact": 8.0, "rw": 8.0, "simp": 4.0, "subst": 8.0,
}
```

Unlisted families and missing signals keep baseline scoring when scalar
`local_type_bonus_strength` is left at its default zero.
The earlier scalar `local_type_bonus_strength = 8` remains available for the
two-family comparison. Default **0** and empty map preserve existing V2
checkpoints and scoring.
The source gate receives the *original* local maxima, so the bonus changes
within-local ranking only at position zero; later `rw` positions may use a
different goal. Paired offline validation (same epoch-8 weights, bonus 0 vs 8):
matched local top-1 **152→165/277**; oracle-tactic nonempty flat actions
**64→72/759**, predicted-tactic nonempty **19→27/759**. Predicted-tactic
features used the predicted family, not the gold family. These offline
sidecars have not been verified against live Pantograph state fields (#42).

### Local candidate signals by tactic

One shared local scorer remains the default. Type signals are *soft features*
for its candidates, with no-match fallback—not separate hard filters for each
of the 214 tactic classes. First four rows have measured narrow-cohort gains:

| Tactic/role | Candidate relationship to examine | Status |
|---|---|---|
| `exact` with a local proof | Hypothesis type literally equals goal; later test definitional equality | **Measured** for one-local actions; 13/26 val have a unique literal match, all 13 recorded gold |
| First local `rw` | Eq/Iff/HEq side occurs within initial goal | **Measured**; first-local val rank 29→32/38 with soft bonus; 25/2,052 train rows would lose gold under hard filtering |
| First local `simp` | Eq/Iff/HEq side occurs within initial goal for a subset of simplification lemmas | **Measured**; train-selected bonus 4: 1,226→1,351/1,877 train (133 rescued, 8 harmed), 29→31/38 val (2 rescued, 0 harmed observed), retrieval-ready val 28→30/35; never Eq-only filter |
| First local `subst` | Eq/Iff/HEq side occurs within initial goal for equality-proof references | **Measured**; train-selected bonus 8: 390→426/623 train (57 rescued, 21 harmed), 6→8/13 val (2 rescued, 0 harmed observed); adds 2 oracle complete actions on full val, no predicted-action gain |
| Later `rw`, `rewrite`, `rwa`, `simp_rw` | Equality relevant to goal *after preceding rewrites* | Hypothesis; initial-goal filter is unsafe; needs prefix/Lean-state evidence |
| Later `simp`, `simpa`, `dsimp` | Rewrite/simplification relation, potentially conditional or implicit | Hypothesis; first-slot Eq-side cue does not cover simplifier behavior in general |
| `apply`, `refine` | Candidate conclusion can elaborate/unify with goal, perhaps generating subgoals | Hypothesis; not literal equality; complex `refine` terms need syntax generation |
| `cases`, `rcases`, later `subst` | Destructible type or equality/variable role useful to current goal | Hypothesis; `subst` may take a variable or equality proof; no one rigid mask |
| `use`, `exists` | Candidate term fits an existential witness or constructor argument | Hypothesis; needs goal structure and elaboration, not proof-type equality |
| `intro`, `have`, `let` | Often introduce new names/goals instead of selecting an input local | Out of this flat local-pointer scope; requires syntax/binding generation |

Paired read-only server comparison used the same epoch-8 local projection,
joint V2 count/source checkpoint, all 4,157 val states and frozen top-200
candidate IDs. The server's existing `rw` cue at half strength implements the
same first-`simp` Eq-side bonus 4; no checkpoints or files were changed.

| Validation metric | Off | `exact`/`rw` 8 | Plus `simp` 4 |
|---|---:|---:|---:|
| Matched local top-1, oracle tactic/prefix | 152/277 | 165/277 | **167/277** |
| Nonempty exact flat action, oracle tactic | 64/759 | 72/759 | **73/759** |
| Nonempty exact flat action, predicted tactic | 19/759 | 27/759 | 27/759 |
| All supported exact, oracle tactic | 490/1,207 | 498/1,207 | 499/1,207 |
| All supported exact, predicted tactic | 198/1,207 | 206/1,207 | 206/1,207 |

Features came from normalized `val/model_sexpr_v2` sidecars using oracle
tactics for teacher-forced ranks and predicted tactics for greedy actions.
The extra `simp` cue gains two reachable local hits and one oracle complete
action over 8/8, but no predicted-tactic complete action. This is still
offline trace imitation, not Lean execution. An oracle-prefix audit of the
same matched local cohort found **15 rescued / 0 harmed** on val (10 `exact`,
3 `rw`, 2 `simp`); training had harms, so zero observed val harms is not a
guarantee. Largest remaining errors: `exact` **28/74** (12/36 first and
16/38 later), `apply` **13/21** (7/13 first, 6/8 later), `rw` **10/42**,
`subst` **9/15**, and `simp` **7/44**. Later positions receive no cue.

`flat_reference_syntax` treats `exact f x` and `apply f x` as whitespace-
separated term references, while `rw [h₁, h₂]` and `simp [h₁, h₂]` are
list-argument tactics. Later `exact`/`apply` operands may need the preceding
function's **type/domain** (and elaboration), not a changed intermediate
goal. Read-only all-supported position audit (not limited to reachable pool):
`exact` has **3,798 train / 53 val** later local positions; `apply` **610 / 13**.
Among second-slot locals, first reference is library for `exact` **2,155/
2,700 train, 29/39 val** and `apply` **474/507 train, 9/10 val**. The
normalized type of that library function is unavailable. A directly
comparable explicit outer `:forall` domain of a first **local** function
exists for `exact` only **291/3,798 train**, **7/53 val** later locals;
`apply` **21/610 train**, **1/13 val**. Gold local type literally matches
the domain in **280/291 train, 6/7 val** (`exact`) and **20/21 train, 1/1
val** (`apply`). Unique matching local is gold **189/189 train, 4/4 val**
for `exact`, **9/9 train, 1/1 val** for `apply`, but one `exact` val row has
a domain-matching *other* local: a hard filter would exclude gold. This
narrow, mostly second-slot cue cannot fix most later errors. The subsequent
first-local `apply` audit stripped all outer `:forall` binders from each
eligible FV type and compared the literal conclusion with the initial goal.
Only **20/585 train** and **1/13 val** gold references matched. Each match
was a unique gold candidate; no conflicting matches appeared in this audit.
The cue covers too little `apply` to justify a new score setting; type
unification/elaboration remains untested. First-local `subst` audit:
equality-like recorded gold **476/623 train, 9/13 val**; its Eq/Iff/HEq side
occurs in initial goal for **399/623 train, 7/13 val**. But a different
local matches on **138/623 train, 4/13 val** rows whose gold does not, so
this cannot be a hard filter. Unique Eq-side match is gold **337/459 train,
5/7 val**. FV occurrence in goal is worse: gold **120/623 train, 3/13 val**,
while a different FV matches without gold on **481/623 train, 9/13 val**.
Discard FV-occurrence as a role filter. A read-only Eq-side soft rank test
selected strength **8** on all **623** train rows from
`(0, .25, .5, 1, 2, 4, 8, 16, 32)`, preferring the smaller strength on ties.
Train top-1 **390→426/623**, with **57 rescued / 21 harmed**; val
**6→8/13**, with **2 rescued / 0 harmed** observed. No oversized rows were
skipped. These are first-local ranks, not complete actions or Lean validity.
Later `rw` needs an updated goal. Confirm live type parity and actual Lean
validity before serving. Library top-200 ranking
remains a separate task; no normalized type graph for every library
declaration is assumed here.

### Closed local-identity experiment cycle

Final paired acceptance run covered all **4,157** val states with the same
epoch-8 checkpoint and candidate IDs. The three-family reference reproduced
its prior counts. Adding train-selected `subst=8` gave:

| Metric | `exact=8, rw=8, simp=4` | Plus `subst=8` | Rescued / harmed |
|---|---:|---:|---:|
| Matched local top-1, oracle tactic/prefix | 167/277 | **169/277** | 2 / 0 |
| Nonempty exact, oracle tactic | 73/759 | **75/759** | 2 / 0 |
| Nonempty exact, predicted tactic | 27/759 | 27/759 | 0 / 0 |
| All supported exact, oracle tactic | 499/1,207 | **501/1,207** | 2 / 0 |
| All supported exact, predicted tactic | 206/1,207 | 206/1,207 | 0 / 0 |

Decision: retain `subst=8` in the optional experimental configuration. No
strength was retuned on val. Compared with the same epoch-8 weights without
cues, final local identity is **152→169/277 (54.9→61.0%)**, oracle nonempty
exact **64→75/759 (8.43→9.88%)**, predicted nonempty exact **19→27/759
(2.50→3.56%)**. The last `subst` addition has no predicted-action gain;
21 train harms and small tactic cohorts limit generalization claims.
Source-gate evidence remains unadjusted; all cues remain default-off and
restricted to first position. Joint-decoder/local-identity checks: **19 passed**.

This offline experiment cycle is complete. Later operands require prefix/type evidence
and often missing library-function types; `apply` requires stronger
unification/elaboration. Live normalized-state parity and Lean-checked
execution are follow-up work before serving, not claims of this experiment.

### Next workstream: global lemma identity in the existing decoder

Keep frozen epoch-5 top-200 candidate IDs and the finalized local baseline.
The historical reference is **62/408** library identities, teacher-forced on
actions whose entire gold sequence is reachable, equal to frozen text order.
The denominator is positions, not 408 whole actions; missing retrieval is a
separate limitation. Before modifying the library component, remeasure it
with the epoch-8 local checkpoint: local training changed the shared
`argument_selector.out_proj`, which also scores adapted library vectors.

Current library logits are `text_score * exp(log_inverse_text_temperature)`
plus `tanh(library_residual_scale) * score_candidates(GRU_hidden, adapted_text)`.
GRU hidden already contains tactic and selected-prefix context. Library
identity loss is trained only on fully reachable actions; retrieval misses
provide source-prefix supervision but no library identity loss. The original
joint pilot used three epochs of 500 train batches, not full library-only
supervision. A small scalar alone does not establish a negligible residual:
measure actual per-pool contribution and resulting rank changes.

Read-only validation audit compares frozen text, original joint V2 and
epoch-8 local checkpoints on the same **408** reachable positions. Report
top-1/top-5/MRR by tactic and slot, rank gains/losses, residual/prior score
spread, and whether gold-prefix context changes library scores or favors
already-used lemmas. Hold tactic and previous references at gold only for
this diagnostic. This does not measure greedy sequence accuracy.

Use the findings to modify the existing decoder's library component, not a
separate standalone model. Any trainable library changes must preserve the
local projection and fixed graph/tactic/count/GRU components initially, and
be selected on training/validation without changing candidate policy. Final
acceptance includes local **169/277**, oracle nonempty **75/759**, predicted
nonempty **27/759**, library ranks, and source choice with finalized local
cues. The opt-in independent library query and frozen-component trainer are
implemented; server training is pending. See the
[library identity probe](library_identity_probe.md) for the measured audit,
architecture, candidate/prefix denominators and resumable pilot command.
