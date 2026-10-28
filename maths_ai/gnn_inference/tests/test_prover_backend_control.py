"""Matched prompt contracts and immutable Lean validation; no model downloads."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.scripts import compare_prover_backends as runner


def test_final_code_block_extraction_preserves_nested_proof_indentation():
    raw = """A sketch:
```lean4
theorem sample : True := by sorry
```
Final proof:
```lean4
theorem changed_name : True := by
  have h : True := by
    trivial
  exact h
```
"""
    assert runner._extract_body(raw) == "have h : True := by\n  trivial\nexact h"
    source = runner._lean_source(runner._extract_body(raw))
    assert f"theorem {runner.THEOREM} : {runner.TARGET} := by" in source
    assert "theorem changed_name" not in source
    assert "open BigOperators Real Nat Topology Rat" in source
    with pytest.raises(ValueError, match="No tactic body"):
        runner._extract_body("")
    with pytest.raises(ValueError, match="Unterminated code block"):
        runner._extract_body(raw + "\n```lean4\ntruncated final proof")


@pytest.mark.parametrize("axioms,exit_code,expected", [
    ("propext, Classical.choice, Quot.sound", 0, "passed"),
    ("sorryAx", 0, "lean_rejected"),
    ("propext, fabricated_assumption", 0, "lean_rejected"),
    ("propext", 1, "lean_rejected"),
])
def test_validation_checks_fixed_statement_and_allowed_axioms(tmp_path, monkeypatch, axioms, exit_code, expected):
    captured = []

    def compile_(command, **kwargs):
        path = Path(command[-1])
        captured.append(path)
        code = path.read_text()
        assert runner.TARGET in code
        assert "(h : False)" not in code  # Added model assumptions are discarded.
        assert f"#print axioms {runner.THEOREM}" in code
        return SimpleNamespace(returncode=exit_code,
                               stdout=f"'{runner.THEOREM}' depends on axioms: [{axioms}]\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", compile_)
    result = runner._validate("theorem bad (h : False) : True := by\n  norm_num",
                              (tmp_path, "leanprover/lean4:v4.29.1", "test-path"), 1)
    assert result["status"] == expected
    assert not captured[0].exists()  # Generated candidate file is temporary.


def test_validator_timeout_is_operational_failure_not_bad_proof(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner._validate("norm_num", (tmp_path, "leanprover/lean4:v4.29.1", ""), 1)
    assert result["status"] == "validator_timeout"


def test_ollama_receives_exact_raw_prompt_and_explicit_sampling(monkeypatch):
    import requests
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "response": "by norm_num", "prompt_eval_count": 17,
            "eval_count": 3, "done_reason": "stop",
        })

    monkeypatch.setattr(requests, "post", post)
    args = SimpleNamespace(ollama_url="http://localhost:11434", ollama_model="test-model",
                           context_tokens=2048, max_new_tokens=1024, request_timeout=10)
    result = runner._ollama(args, "<BOS>EXACT PROMPT", 17, 42)
    payload = calls[0][1]["json"]
    assert payload["raw"] is True and payload["prompt"] == "<BOS>EXACT PROMPT"
    assert "messages" not in payload and "context" not in payload
    assert payload["options"]["repeat_penalty"] == 1.0
    assert payload["options"]["top_k"] == 50
    assert payload["options"]["seed"] == 42
    assert result["input_token_count_matches"] is True
    assert runner._ollama(args, "same", 18, 43)["input_token_count_matches"] is False


@pytest.mark.parametrize("return_encoding", [False, True])
def test_matched_reports_and_resume_reject_changes(tmp_path, monkeypatch, return_encoding):
    class Tokenizer:
        chat_template = "fixed-template"
        bos_token = "<BOS>"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert len(messages) == 1 and messages[0]["role"] == "user"
            assert add_generation_prompt
            rendered = self.bos_token + messages[0]["content"]
            if not tokenize:
                return rendered
            ids = list(rendered.encode())
            return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_encoding else ids

        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return list(text.encode())

        def decode(self, ids, **kwargs):
            return bytes(ids).decode()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        PreTrainedTokenizerFast=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(rope_scaling={"type": "yarn"})),
    ))
    runtime = {"runtime": {"transformers": "test"}, "gguf_blob_sha256": "original-weights",
               "ollama_add_bos_token": True}
    monkeypatch.setattr(runner, "_preflight", lambda args: runtime.copy())
    calls = []

    def generate(args, prompt, expected_tokens, seed):
        calls.append(seed)
        assert prompt.startswith("Complete")  # GGUF supplies the one BOS itself.
        return {"seed": seed, "status": "generated", "raw_text": "norm_num",
                "input_tokens": expected_tokens, "input_token_count_matches": True}

    monkeypatch.setattr(runner, "_ollama", generate)
    output = tmp_path / "ollama"
    argv = ["--backend", "ollama", "--output-dir", str(output), "--seeds", "0", "1", "2"]
    assert runner.main(argv) == 0
    assert calls == [0, 1, 2]
    first = json.loads((output / "report.json").read_text())
    assert first["reference_revision"] == runner.REVISION
    assert first["rendered_prompt"].startswith("<BOS>")
    assert first["ollama_bos_policy"] == "gguf_automatic_single_bos"
    assert first["generation"]["repetition_penalty"] == 1.0
    assert runner.main([*argv, "--resume"]) == 0
    assert calls == [0, 1, 2]  # No duplicate inference on completed resume.
    with pytest.raises(ValueError, match="changed prompt"):
        runner.main([*argv, "--prompt-style", "plain", "--resume"])
    runtime["gguf_blob_sha256"] = "changed-weights"
    with pytest.raises(ValueError, match="changed prompt"):
        runner.main([*argv, "--resume"])

    counterpart = tmp_path / "reference.json"
    reference = {**first, "backend": "transformers", "backend_key": "different", "runtime": {"dtype": "bf16"}}
    counterpart.write_text(json.dumps(reference))
    comparison = runner._compare([output / "report.json", counterpart])
    assert [item["backend"] for item in comparison] == ["ollama", "transformers"]
    reference["comparison_key"] = "different-prompt"
    counterpart.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="not a matched control"):
        runner._compare([output / "report.json", counterpart])


def test_backend_error_is_saved_without_fabricating_a_generation(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            return [1] if tokenize else "prompt"

        def encode(self, *args, **kwargs):
            return [1]

        def decode(self, *args, **kwargs):
            return "prompt"

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        PreTrainedTokenizerFast=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(rope_scaling={})),
    ))
    monkeypatch.setattr(runner, "_preflight", lambda args: {"runtime": {}})

    def fail(*args):
        raise ConnectionError("backend unavailable")

    monkeypatch.setattr(runner, "_ollama", fail)
    output = tmp_path / "failed"
    assert runner.main(["--backend", "ollama", "--output-dir", str(output)]) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["backend_error"]["type"] == "ConnectionError"
    assert report["trials"] == []


def test_whitespace_changing_tokenizer_fails_before_inference(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            return [1] if tokenize else "original prompt"

        def encode(self, *args, **kwargs):
            return [1]

        def decode(self, *args, **kwargs):
            return "originalprompt"

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        PreTrainedTokenizerFast=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(rope_scaling={})),
    ))
    monkeypatch.setattr(runner, "_preflight", lambda args: {"runtime": {}})
    output = tmp_path / "bad-tokenizer"
    with pytest.raises(ValueError, match="tokenizer changes prompt text"):
        runner.main(["--backend", "ollama", "--output-dir", str(output)])
    assert not output.exists()


def test_reference_preflight_fails_before_download_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        runner._preflight(SimpleNamespace(backend="transformers"))


def test_reference_requests_full_precision_and_explicit_sampling(monkeypatch):
    import torch
    from transformers import GenerationConfig
    loaded, sampled, seeded = [], [], []
    tensor = torch.tensor

    class Model:
        def eval(self):
            return self

        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16)])

        def generate(self, **kwargs):
            sampled.append(kwargs)
            return torch.cat([kwargs["input_ids"], tensor([[100001]])], dim=1)

    def load(*args, **kwargs):
        loaded.append((args, kwargs))
        return Model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        GenerationConfig=GenerationConfig, set_seed=seeded.append,
    ))
    # Exercise loader/generation contracts without claiming a real GPU test.
    monkeypatch.setattr(torch, "tensor", lambda data, *, device: tensor(data))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = SimpleNamespace(hf_cache=None, dtype="bfloat16", device="cuda:0",
                           max_new_tokens=1024, request_timeout=300)
    tokenizer = SimpleNamespace(bos_token_id=100000, eos_token_id=100001,
                                decode=lambda *a, **k: "norm_num")
    results = list(runner._reference(args, tokenizer, [100000, 1, 2], [0, 1]))
    assert seeded == [0, 1]
    kwargs = loaded[0][1]
    assert kwargs["revision"] == runner.REVISION
    assert kwargs.get("dtype", kwargs.get("torch_dtype")) == torch.bfloat16
    assert kwargs["device_map"] == {"": "cuda:0"}
    assert "quantization_config" not in kwargs and "load_in_4bit" not in kwargs
    config = sampled[0]["generation_config"]
    assert (config.temperature, config.top_p, config.top_k, config.repetition_penalty) == (0.6, 0.95, 50, 1.0)
    assert config.max_new_tokens == 1024 and config.num_beams == 1
    assert config.max_time == 300 and "max_time" not in sampled[0]
    assert [row["done_reason"] for row in results] == ["stop", "stop"]
