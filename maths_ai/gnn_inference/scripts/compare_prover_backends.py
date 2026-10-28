"""Controlled single-turn inference for a public Lean compatibility example.

No agent, retrieval, repair, training, or benchmark-success claim. Ollama
receives the exact prompt rendered by the pinned reference tokenizer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import time


REFERENCE = "deepseek-ai/DeepSeek-Prover-V2-7B"
REVISION = "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b"
OLLAMA_MODEL = "hf.co/unsloth/DeepSeek-Prover-V2-7B-GGUF:Q4_K_M"
SCHEMA = "prover_backend_control_v1"
TARGET = "abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10"
THEOREM = "backend_control_mathd_algebra_10"
DECLARATION = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
    r"/-- What is the positive difference between $120\%$ of 30 and $130\%$ of 20? Show that it is 10.-/"
    "\ntheorem mathd_algebra_10 : " + TARGET + " := by\n  sorry"
)
PLAN = (
    "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan "
    "outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate "
    "lemmas, and proof structures that will guide the construction of the final formal proof."
)
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _prompt(style):
    code = "```lean4\n" + DECLARATION + "\n```" if style == "fenced" else DECLARATION
    return "Complete the following Lean 4 code:\n\n" + code + "\n\n" + PLAN


def _generation(args):
    return {
        "seeds": args.seeds, "context_tokens": args.context_tokens,
        "max_new_tokens": args.max_new_tokens, "temperature": 0.6,
        "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0,
        "min_p": 0.0, "single_turn": True,
    }


def _extract_body(raw):
    if raw.count("```") % 2:
        raise ValueError("Unterminated code block; final output may be truncated")
    blocks = re.findall(r"```(?:lean4?|Lean)?[ \t]*\n(.*?)```", raw, flags=re.DOTALL)
    code = textwrap.dedent(blocks[-1]).strip() if blocks else raw.strip()
    declaration = re.search(r"\b(?:theorem|lemma|example)\b.*?:=\s*by\b(.*)", code, re.DOTALL)
    if declaration:
        code = textwrap.dedent(declaration.group(1)).strip()
    elif code.startswith("by\n") or code.startswith("by "):
        code = textwrap.dedent(code[2:]).strip()
    if not code:
        raise ValueError("No tactic body found")
    return code


def _lean_source(body):
    # Only the body is model-provided. Statement/imports remain fixed even if
    # the generated declaration changed its theorem type or added assumptions.
    return ("import Mathlib\nimport Aesop\nset_option maxHeartbeats 200000\n"
            "open BigOperators Real Nat Topology Rat\n"
            f"theorem {THEOREM} : {TARGET} := by\n"
            + "\n".join("  " + line for line in body.splitlines())
            + f"\n#print axioms {THEOREM}\n")


def _lean_context(project):
    project = project.resolve()
    toolchain = (project / "lean-toolchain").read_text().strip()
    installed = subprocess.run(["elan", "toolchain", "list"], capture_output=True, text=True,
                               check=True, timeout=15).stdout.splitlines()
    if toolchain not in {line.split()[0] for line in installed if line.strip()}:
        raise RuntimeError(f"Lean toolchain not installed: {toolchain}; no automatic install attempted")
    result = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=project,
                            capture_output=True, text=True, check=True, timeout=60)
    return project, toolchain, result.stdout.strip()


def _check_source(code, theorem_name, context, timeout):
    """Compile host-constructed source and audit one frozen target theorem."""
    project, toolchain, lean_path = context
    env = {**os.environ, "LEAN_PATH": lean_path}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lean-backend-control-") as directory:
        source = Path(directory) / "Candidate.lean"
        source.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(["elan", "run", toolchain, "lean", str(source)],
                                    cwd=project, env=env, capture_output=True,
                                    text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "validator_timeout", "seconds": round(time.monotonic() - started, 3)}
    output = result.stdout + result.stderr
    pattern = r"^'" + re.escape(theorem_name) + r"' depends on axioms:\s*\[([^\]]*)\]"
    match = re.search(pattern, output, re.MULTILINE)
    no_axioms = re.search(r"^'" + re.escape(theorem_name) + r"' does not depend on any axioms$",
                          output, re.MULTILINE) is not None
    axioms = [item.strip() for item in match.group(1).split(",") if item.strip()] if match else []
    passed = (result.returncode == 0 and (bool(match) or no_axioms)
              and set(axioms) <= ALLOWED_AXIOMS and "sorryAx" not in output)
    return {"status": "passed" if passed else "lean_rejected", "toolchain": toolchain,
            "exit_code": result.returncode, "axioms": axioms,
            "seconds": round(time.monotonic() - started, 3), "diagnostics": output}


def _validate(raw, context, timeout):
    try:
        body = _extract_body(raw)
    except ValueError as error:
        return {"status": "no_complete_body", "error": str(error)}
    return {**_check_source(_lean_source(body), THEOREM, context, timeout), "proof_body": body}


def _runtime():
    result = {}
    for name in ("torch", "transformers", "tokenizers", "accelerate", "requests"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _preflight(args):
    result = {"runtime": _runtime()}
    if args.backend == "transformers":
        missing = [name for name in ("torch", "transformers", "accelerate") if result["runtime"][name] is None]
        if missing:
            raise RuntimeError("Missing Python dependencies: " + ", ".join(missing)
                               + "; use a writable Python environment, without sudo")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch required; inspect the user environment before installing")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("Reference control requires an explicit CUDA device, without CPU offload")
        free, total = torch.cuda.mem_get_info(device)
        result.update(gpu=torch.cuda.get_device_name(device), free_vram_gib=free / 2**30,
                      total_vram_gib=total / 2**30, dtype=args.dtype)
        if free < 18 * 2**30:
            raise RuntimeError(f"Reference control needs ~18 GiB free VRAM; available {free / 2**30:.2f} GiB")
        with torch.cuda.device(device):
            if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 unsupported on selected setup; use --dtype float16")
        cache = args.hf_cache or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        while not cache.exists():
            cache = cache.parent
        disk = shutil.disk_usage(cache).free
        result["free_cache_disk_gib"] = disk / 2**30
        if disk < args.min_free_disk_gib * 2**30:
            raise RuntimeError("Insufficient cache disk for reference weights; inspect space before downloading")
    else:
        import requests
        result["ollama_version"] = requests.get(args.ollama_url + "/api/version", timeout=10).json()
        response = requests.post(args.ollama_url + "/api/show", json={"model": args.ollama_model}, timeout=30)
        response.raise_for_status()
        data = response.json()
        info = data.get("model_info", {})
        parent = info.get("general.base_model.0.repo_url")
        if parent and parent.rstrip("/") != "https://huggingface.co/" + REFERENCE:
            raise ValueError("Ollama model declares a different reference checkpoint")
        result["gguf_declared_reference"] = parent
        result["ollama_add_bos_token"] = info.get("tokenizer.ggml.add_bos_token", False)
        result["ollama_details"] = data.get("details")
        result["ollama_model_info"] = {key: value for key, value in info.items()
                                       if "rope" in key or key.startswith("tokenizer.ggml.")
                                       and not isinstance(value, list)}
        digest = re.search(r"sha256-([0-9a-f]{64})", data.get("modelfile", ""))
        result["gguf_blob_sha256"] = digest.group(1) if digest else None
        result["ollama_template_sha256"] = _sha(data.get("template", ""))
    return result


def _ollama(args, prompt, expected_tokens, seed):
    import requests
    started = time.monotonic()
    response = requests.post(args.ollama_url + "/api/generate", json={
        "model": args.ollama_model, "prompt": prompt, "raw": True,
        "stream": False, "keep_alive": "5m",
        "options": {"num_ctx": args.context_tokens, "num_predict": args.max_new_tokens,
                    "temperature": 0.6, "top_p": 0.95, "top_k": 50,
                    "repeat_penalty": 1.0, "min_p": 0.0, "seed": seed},
    }, timeout=(10, args.request_timeout))
    response.raise_for_status()
    data = response.json()
    return {"seed": seed, "status": "generated", "raw_text": data["response"],
            "wall_seconds": round(time.monotonic() - started, 3),
            "input_tokens": data.get("prompt_eval_count"),
            "input_token_count_matches": data.get("prompt_eval_count") == expected_tokens,
            "output_tokens": data.get("eval_count"), "done_reason": data.get("done_reason"),
            "load_seconds": data.get("load_duration", 0) / 1e9,
            "prompt_seconds": data.get("prompt_eval_duration", 0) / 1e9,
            "generation_seconds": data.get("eval_duration", 0) / 1e9}


class ReferenceGenerator:
    """One frozen reference load, reusable across independent prompts/tasks."""

    def __init__(self, args, tokenizer):
        self.args, self.tokenizer, self.model = args, tokenizer, None

    def __enter__(self):
        import torch
        from packaging.version import Version
        from transformers import AutoModelForCausalLM
        started = time.monotonic()
        precision_key = ("dtype" if Version(importlib.metadata.version("transformers")) >= Version("4.56")
                         else "torch_dtype")
        self.model = AutoModelForCausalLM.from_pretrained(
            REFERENCE, revision=REVISION, cache_dir=self.args.hf_cache,
            **{precision_key: getattr(torch, self.args.dtype)}, device_map={"": self.args.device},
            attn_implementation="sdpa", trust_remote_code=False,
            local_files_only=getattr(self.args, "local_files_only", False),
        ).eval()
        try:
            if getattr(self.model, "is_loaded_in_4bit", False) or getattr(self.model, "is_loaded_in_8bit", False):
                raise RuntimeError("Reference model must not be quantized")
            if any(p.device.type != "cuda" for p in self.model.parameters()):
                raise RuntimeError("Reference model unexpectedly offloaded")
        except Exception:
            self.close()
            raise
        print(json.dumps({"reference_load_seconds": round(time.monotonic() - started, 3),
                          "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()})}), flush=True)
        return self

    def generate(self, input_ids, seed):
        import torch
        from transformers import GenerationConfig, set_seed
        args, tokenizer = self.args, self.tokenizer
        inputs = torch.tensor([input_ids], device=args.device)
        generation = GenerationConfig(
            do_sample=True, num_beams=1, max_new_tokens=args.max_new_tokens,
            temperature=0.6, top_p=0.95, top_k=50, min_p=0.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True, max_time=args.request_timeout,
        )
        set_seed(seed)
        torch.cuda.synchronize(args.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         generation_config=generation)
        torch.cuda.synchronize(args.device)
        tokens = output[0, len(input_ids):].tolist()
        ended = bool(tokens) and tokens[-1] == tokenizer.eos_token_id
        return {"seed": seed, "status": "generated",
                "raw_text": tokenizer.decode(tokens, skip_special_tokens=True),
                "wall_seconds": round(time.monotonic() - started, 3),
                "input_tokens": len(input_ids), "input_token_count_matches": True,
                "output_tokens": len(tokens),
                "done_reason": "stop" if ended else "length" if len(tokens) >= args.max_new_tokens else "time_limit"}

    def close(self):
        import torch
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, *args):
        self.close()


def _reference(args, tokenizer, input_ids, seeds):
    with ReferenceGenerator(args, tokenizer) as model:
        for seed in seeds:
            yield model.generate(input_ids, seed)


def _compare(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    if any(report["comparison_key"] != reports[0]["comparison_key"] for report in reports):
        raise ValueError("Prompt/tokenizer/settings differ; reports are not a matched control")
    return [{"backend": report["backend"], "runtime": report["runtime"],
             "backend_error": report.get("backend_error"),
             "trials": [{key: trial.get(key) for key in
                         ("seed", "status", "input_tokens", "input_token_count_matches", "output_tokens",
                          "done_reason", "wall_seconds", "lean")}
                        for trial in report["trials"]]}
            for report in reports]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ollama", "transformers"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-style", choices=("fenced", "plain"), default="fenced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--min-free-disk-gib", type=float, default=20)
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--lean-project", type=Path)
    parser.add_argument("--lean-timeout", type=float, default=120)
    parser.add_argument("--unload-ollama-before-validation", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.request_timeout) or args.request_timeout <= 0
            or not math.isfinite(args.lean_timeout) or args.lean_timeout <= 0
            or not math.isfinite(args.min_free_disk_gib) or args.min_free_disk_gib < 0):
        parser.error("Timeouts must be positive and free-disk guard nonnegative")
    if args.compare:
        print(json.dumps(_compare(args.compare), indent=2, ensure_ascii=False))
        return 0
    if args.validate_report:
        if args.lean_project is None:
            parser.error("--validate-report requires --lean-project")
        report = json.loads(args.validate_report.read_text())
        if report.get("schema") != SCHEMA or report.get("target_sha256") != _sha(TARGET):
            raise ValueError("Unknown report/target; refusing to change validation specification")
    else:
        if args.backend is None:
            parser.error("--backend is required")
        runtime = _preflight(args)
        print(json.dumps({"preflight": runtime}, ensure_ascii=False), flush=True)
        if args.preflight:
            return 0
        if (len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0
                or args.context_tokens < 1 or args.max_new_tokens < 1):
            raise ValueError("Invalid seeds or context/generation limits")
        from transformers import AutoConfig, PreTrainedTokenizerFast
        # This checkpoint's serialized tokenizer is BPE despite its Llama
        # class hint. Some AutoTokenizer versions rebuild it as Metaspace.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            REFERENCE, revision=REVISION, cache_dir=args.hf_cache, trust_remote_code=False,
        )
        config = AutoConfig.from_pretrained(REFERENCE, revision=REVISION,
                                            cache_dir=args.hf_cache, trust_remote_code=False)
        messages = [{"role": "user", "content": _prompt(args.prompt_style)}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # Transformers 5 may return BatchEncoding; older releases return a list.
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if not isinstance(ids, list) or not all(isinstance(token, int) for token in ids):
            raise ValueError("Expected one flat reference token sequence")
        if tokenizer.encode(rendered, add_special_tokens=False) != ids:
            raise ValueError("Rendered prompt does not round-trip to reference token IDs")
        if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != rendered:
            raise ValueError("Reference tokenizer changes prompt text; refusing an invalid control")
        if len(ids) + args.max_new_tokens > args.context_tokens:
            raise ValueError("Prompt plus generation budget exceeds context; no silent truncation allowed")
        settings = _generation(args)
        key = _sha({"reference": REFERENCE, "revision": REVISION,
                    "rendered_sha256": _sha(rendered), "input_ids_sha256": _sha(ids), "settings": settings})
        report = {"schema": SCHEMA, "backend": args.backend, "reference": REFERENCE,
                  "reference_revision": REVISION, "target_sha256": _sha(TARGET),
                  "prompt_style": args.prompt_style,
                  "comparison_key": key, "prompt": messages[0]["content"], "rendered_prompt": rendered,
                  "rendered_prompt_sha256": _sha(rendered), "input_ids_sha256": _sha(ids),
                  "expected_input_tokens": len(ids), "generation": settings,
                  "reference_rope_scaling": config.rope_scaling, "runtime": runtime, "trials": []}
        transport = rendered
        if args.backend == "ollama" and runtime.get("ollama_add_bos_token"):
            # Ollama's GGUF runner adds BOS even when raw text starts with one.
            # Send the remainder so both backends see exactly one BOS.
            if not tokenizer.bos_token or not transport.startswith(tokenizer.bos_token):
                raise ValueError("Cannot compensate automatic BOS without the reference prefix")
            transport = transport[len(tokenizer.bos_token):]
            report["ollama_bos_policy"] = "gguf_automatic_single_bos"
        elif args.backend == "ollama":
            report["ollama_bos_policy"] = "reference_rendered_bos"
        if args.backend == "ollama":
            report["ollama_transport_prompt_sha256"] = _sha(transport)
        report["backend_key"] = _sha({
            "backend": args.backend, "model": args.ollama_model if args.backend == "ollama" else REFERENCE,
            "gguf_blob_sha256": runtime.get("gguf_blob_sha256"),
            "ollama_template_sha256": runtime.get("ollama_template_sha256"),
            "ollama_version": runtime.get("ollama_version"), "packages": runtime.get("runtime"),
            "dtype": args.dtype if args.backend == "transformers" else None,
        })
    if args.output_dir is None:
        parser.error("--output-dir is required")
    root = args.output_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"Output directory exists: {root}; use a new directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous["comparison_key"] != report["comparison_key"]
                or previous["backend_key"] != report["backend_key"]):
            raise ValueError("Resume uses a changed prompt, weights, runtime, tokenizer or generation policy")
        report = previous
    _write(path, report)
    if not args.validate_report:
        completed = {trial["seed"] for trial in report["trials"] if trial["status"] == "generated"}
        seeds = [seed for seed in args.seeds if seed not in completed]
        generator = (iter(()) if not seeds else _reference(args, tokenizer, ids, seeds)
                     if args.backend == "transformers"
                     else (_ollama(args, transport, len(ids), seed) for seed in seeds))
        try:
            for trial in generator:
                trial["lean"] = {"status": "not_requested"}
                report["trials"].append(trial)
                report.pop("backend_error", None)
                _write(path, report)
                print(json.dumps({"backend": args.backend, **trial}, ensure_ascii=False), flush=True)
        except Exception as error:
            report["backend_error"] = {"type": type(error).__name__, "message": str(error)}
            _write(path, report)
            print(json.dumps({"backend_error": report["backend_error"], "report": str(path)}), flush=True)
            return 2
        if args.backend == "ollama" and args.unload_ollama_before_validation:
            import requests
            response = requests.post(args.ollama_url + "/api/generate",
                                     json={"model": args.ollama_model, "keep_alive": 0}, timeout=30)
            response.raise_for_status()
    if args.lean_project:
        context = _lean_context(args.lean_project)
        for trial in report["trials"]:
            trial["lean"] = _validate(trial["raw_text"], context, args.lean_timeout)
            _write(path, report)
            print(json.dumps({"seed": trial["seed"], "lean": trial["lean"]}, ensure_ascii=False), flush=True)
    print(json.dumps({"report": str(path), "backend": report["backend"],
                      "generated": len(report["trials"]),
                      "lean_passed": sum(t["lean"]["status"] == "passed" for t in report["trials"]),
                      "note": "Public compatibility example; not a theorem benchmark or isolated quantization ablation."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
