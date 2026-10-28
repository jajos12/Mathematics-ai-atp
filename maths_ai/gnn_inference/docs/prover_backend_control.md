# DeepSeek inference backend control

Purpose: compare fresh, single-turn generation through local Ollama GGUF
and official unquantized Transformers weights. This is a **public-example
compatibility diagnostic**, not agent evaluation, training or a held-out
theorem benchmark.

## Fixed contract

- Reference: `deepseek-ai/DeepSeek-Prover-V2-7B`, revision
  `a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b`.
- One immutable arithmetic statement, matching the published example.
- Fresh user-only prompt per seed; no repair, retrieval or chat history.
- Seeds **0, 1, 2**; context **2,048**; output cap **1,024**; temperature
  **0.6**; top-p **0.95**; top-k **50**; repetition penalty **1.0**; min-p **0**.
- Preserve raw output, token counts, hashes, runtime versions and truncation.
  Matching seeds do not guarantee identical samples across implementations.
- Transformers loads official BF16/FP16 weights entirely onto one CUDA GPU;
  no quantization, adapters or CPU offload. No Ollama server or sudo needed.

Both backends use the reference's serialized tokenizer and chat template.
The runner explicitly loads `PreTrainedTokenizerFast`, checks exact text
round-trip and records the canonical input-token hash. On the inspected
local Transformers 5.14.1 setup, `AutoTokenizer` rebuilt this BPE tokenizer
with a Metaspace pre-tokenizer, losing whitespace and producing 168 instead
of 193 tokens. Such a control is invalid and is now rejected.

Ollama receives raw rendered text without a duplicate BOS when GGUF adds
one automatically. Otherwise the observed count was 194 versus reference
193. After compensation, all local trials report **193**, matching reference.
Matching counts are a necessary check, not proof of identical internal IDs.
The GGUF metadata/runtime also did not report the reference's YaRN scaling;
this is a recorded discrepancy, **not a confirmed cause** of bad output.

## Server: official full-precision control

Pull the published branch, activate the existing user-owned `atp` environment,
and work from repository root. First check dependencies, CUDA, available
VRAM and cache disk **without downloading model weights**:

```sh
git pull --ff-only jajos gnn-issue-fixes
python -B -m maths_ai.gnn_inference.scripts.compare_prover_backends \
  --backend transformers --device cuda:0 --dtype bfloat16 --preflight
```

Requirements: CUDA-enabled PyTorch, Transformers, Accelerate; approximately
18 GiB free GPU memory and 20 GiB free cache disk for the first download.
Official weights are about 13.8 GB. If dependencies are missing, use a
writable Python environment; do not modify system packages or install a
model daemon. Inspect preflight failures before changing the environment.

If preflight passes:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.compare_prover_backends \
  --backend transformers \
  --output-dir maths_ai/gnn_inference/runs/deepseek_transformers_control \
  --device cuda:0 --dtype bfloat16 \
  --seeds 0 1 2 --context-tokens 2048 --max-new-tokens 1024
```

Use `--device cuda:1` if that is the free 3090. `--hf-cache` can select a
user-owned cache on a volume with space. Reports checkpoint each completed
seed; `--resume` rejects changed prompt, weights, runtime or sampling policy.
Completed resume skips model loading when no seeds remain.

## Local GGUF generation and proof validation

```sh
python -B -u -m maths_ai.gnn_inference.scripts.compare_prover_backends \
  --backend ollama --output-dir /tmp/opencode/deepseek_ollama_control_v2 \
  --seeds 0 1 2 --context-tokens 2048 --max-new-tokens 1024 \
  --unload-ollama-before-validation
```

The explicit unload flag releases this model after generation, before Lean
checking on a low-memory machine. Existing model files/service stay intact.
Validate either backend's report against the same compiled Lean project:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.compare_prover_backends \
  --validate-report /path/to/generated/report.json \
  --lean-project maths_ai/lean_mathlib \
  --output-dir /path/to/new/checked-report
```

Only a generated tactic body enters validation; target/imports are fixed.
Final theorem compiles with axiom inspection, rejecting `sorryAx` and fresh
axioms outside `propext`, `Classical.choice`, `Quot.sound`. Missing/truncated
final code, Lean rejection, validator timeout and backend failure are distinct
statuses. A sketch with holes cannot substitute for a truncated final block.
No empty-visible-goal shortcut is used. Checker limits are 200,000 heartbeats
and 120 seconds by default; the actual toolchain is reported.

Local corrected control: seeds **0 and 1 compile**, seed **2 hits output cap**
with an unterminated final proof block. All three match 193 input tokens.
This means **2/3 samples on one known example**, not 2/3 theorem solve rate.
Reference results and their limitations follow below.
Verification: **32 focused diagnostic/joint/local tests pass**, including
reference full-precision loader/sampling contracts, tokenizer text fidelity,
single-BOS handling, immutable target/axiom checks, operational timeout and
resume-policy rejection. The reference subsequently executed on the server.

Compare report contracts after both runs:

```sh
python -B -m maths_ai.gnn_inference.scripts.compare_prover_backends \
  --compare /path/to/ollama/report.json /path/to/transformers/report.json
```

Different tokenizer/prompt/settings hashes fail comparison. Backend stacks,
sampling implementations and precision differ, so even a successful matched
contract is not an isolated quantization ablation. Proof validity must use
the same Lean environment; generation-only reports explicitly say unchecked.

## Official BF16 reference result

Server loaded the pinned official weights on one RTX 3090 in BF16, reporting
only `torch.bfloat16` parameters. Runtime: PyTorch **2.7.1+cu118**, Transformers
**5.18.0**, tokenizers **0.23.2**, Accelerate **1.15.0**. First load took
**583.11 s**, including the one-time ~13.8 GB download and weight loading;
this is not per-proof generation latency.

Three unique seeds all report **193 input tokens**. Generation times:
**31.30 / 26.27 / 26.52 s**, output lengths **1,024 / 996 / 999** tokens.
Seed 0 ends at the cap before a completed final proof. Seeds 1 and 2 contain
complete candidates. Their exact final proof blocks, pasted by the user,
were rechecked locally with the same fixed-target validator, full
Mathlib/Aesop and Lean **4.29.1**: both compiler exit 0, standard allowed
axioms, no `sorryAx`. Check times **49.53 / 5.16 s**. Pasted duplicate seed-0
log text is not counted as another sample. Full server report/hash manifest
has not been imported, so the automated comparison-key check remains pending.

| Backend | Seed 0 | Seed 1 | Seed 2 | Verified complete samples |
|---|---|---|---|---:|
| Local Q4_K_M Ollama | Pass | Pass | Output cap | 2/3 |
| Official BF16 Transformers | Output cap | Pass | Pass | 2/3 |

This single known example shows both paths can produce checked proofs under
fresh formal-completion conditions. It does **not** establish equal model
quality, isolate quantization effects, or explain the user's degenerate
interactive response. The control explicitly requests a proof plan, so prose
here is expected and is not a code-only instruction-following test. Seed
matching is not token-sequence matching across different samplers.

The server's final `lean_passed: 0` was generation-only (`not_requested`),
not three rejected proofs. Deprecation warnings did not prevent inference.
Future work should separate output-budget truncation from proof errors and
use a fixed small task set before agent-quality claims. Keep this compatibility
control distinct from source-native replay and theorem-solving benchmarks.
