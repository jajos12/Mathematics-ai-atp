"""Ten curated proof-completion tasks with bounded verifier-feedback repair.

Compatibility smoke only: tasks are not a held-out benchmark. No training,
retrieval, source replay, or model download is performed.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import textwrap

from maths_ai.gnn_inference.scripts import compare_prover_backends as control


TASKS = (
    ("and_assoc", "logic", "∀ (P Q R : Prop), (P ∧ Q) ∧ R → P ∧ (Q ∧ R)"),
    ("or_elim", "logic", "∀ (P Q R : Prop), (P → R) → (Q → R) → P ∨ Q → R"),
    ("nat_square", "nat", "∀ n : Nat, (n + 1)^2 = n^2 + 2*n + 1"),
    ("nat_cancel", "nat", "∀ a b c : Nat, a + b = a + c → b = c"),
    ("int_linear", "int", "∀ x y : Int, x ≤ y → 2*x + 3 ≤ 2*y + 3"),
    ("real_square", "real", "∀ x y : ℝ, (x-y)^2 = x^2 - 2*x*y + y^2"),
    ("real_nonneg", "real", "∀ x : ℝ, 0 ≤ x^2 + 1"),
    ("list_length", "list", "∀ (α : Type) (xs ys : List α), (xs ++ ys).length = xs.length + ys.length"),
    ("list_reverse", "list", "∀ (α : Type) (xs : List α), xs.reverse.reverse = xs"),
    ("set_distrib", "set", "∀ (α : Type) (A B C : Set α), A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)"),
)
AUTOMATION = "intros\nsolve\n| rfl\n| assumption\n| aesop (config := { maxRuleApplications := 50 })\n| omega\n| ring\n| nlinarith"


def _task_source(task, body="sorry"):
    return ("import Mathlib\nimport Aesop\nset_option maxHeartbeats 200000\n"
            f"theorem smoke_{task[0]} : {task[2]} := by\n"
            + "\n".join("  " + line for line in body.splitlines()))


def _candidate_source(task, raw):
    if raw.count("```") % 2:
        raise ValueError("Unterminated final code block")
    blocks = re.findall(r"```(?:lean4?|Lean)?[ \t]*\n(.*?)```", raw, re.DOTALL)
    code = textwrap.dedent(blocks[-1]).strip() if blocks else raw.strip()
    declarations = list(re.finditer(r"^[ \t]*(?:theorem|lemma)\s+([A-Za-z_][\w']*)", code, re.MULTILINE))
    expected = next((m for m in declarations if m.group(1) == "smoke_" + task[0]), None)
    if declarations:
        chosen = expected or declarations[0]
        candidate_name = chosen.group(1)
        candidate = code[declarations[0].start():]
    else:
        # Also accept a tactic-only response under the exact supplied header.
        body = control._extract_body(raw)
        candidate_name = "smoke_" + task[0]
        candidate = f"theorem {candidate_name} : {task[2]} := by\n" + "\n".join(
            "  " + line for line in body.splitlines()
        )
    spec, checked, namespace = "spec_" + task[0], "checked_" + task[0], "Candidate_" + task[0]
    # Elaborate the specification BEFORE generated code. Full candidate
    # declarations may rearrange equivalent binders, but cannot change target.
    source = ("import Mathlib\nimport Aesop\nset_option maxHeartbeats 200000\n"
              f"def {spec} : Prop := {task[2]}\nnamespace {namespace}\n"
              + candidate + f"\nend {namespace}\nset_option maxHeartbeats 200000\n"
              f"theorem {checked} : {spec} := by\n  exact {namespace}.{candidate_name}\n"
              f"#print axioms {checked}\n")
    return source, checked


def _validate(task, raw, context, timeout):
    try:
        source, checked = _candidate_source(task, raw)
    except ValueError as error:
        return {"status": "no_complete_body", "error": str(error)}
    return control._check_source(source, checked, context, timeout)


def _messages(task, previous=None):
    prompt = "Complete the following Lean 4 code:\n```lean4\n" + _task_source(task) + "\n```\n" + control.PLAN
    messages = [{"role": "user", "content": prompt}]
    if previous:
        # Do not carry a long reasoning transcript into the next task/attempt.
        raw = previous["raw_text"]
        blocks = re.findall(r"```(?:lean4?)?[ \t]*\n(.*?)```", raw, re.DOTALL)
        failed = blocks[-1] if blocks and raw.count("```") % 2 == 0 else raw[-1200:]
        feedback = previous["lean"].get("diagnostics") or previous["lean"].get("error") or previous["lean"]["status"]
        failed = failed if len(failed) <= 4000 else failed[:4000] + "\n[prior output elided]"
        feedback = feedback if len(feedback) <= 1800 else feedback[:1800] + "\n[diagnostics elided]"
        messages.extend([
            {"role": "assistant", "content": "```lean4\n" + failed + "\n```"},
            {"role": "user", "content": "Verifier feedback:\n" + feedback
             + "\nReturn a corrected complete proof of the original statement. Keep the statement unchanged; "
               "give a brief plan and then the final complete Lean code. No sorry or admit."},
        ])
    return messages


def _render(tokenizer, messages, args):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if (tokenizer.encode(text, add_special_tokens=False) != ids
            or tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text):
        raise ValueError("Tokenizer changes rendered prompt")
    if len(ids) + args.max_new_tokens > args.context_tokens:
        raise ValueError("Prompt plus output budget exceeds context; no silent truncation")
    return text, ids


def _fit_prompt(tokenizer, messages, args):
    """Only compact marked failure history; never truncate the task message."""
    for limit in (None, 1200, 600, 300):
        actual = [dict(message) for message in messages]
        if limit is not None:
            for message in actual[1:]:
                if len(message["content"]) > limit:
                    message["content"] = message["content"][:limit] + "\n[history elided; original statement unchanged]"
        try:
            text, ids = _render(tokenizer, actual, args)
            return actual, text, ids
        except ValueError as error:
            if "exceeds context" not in str(error) or len(messages) == 1:
                raise
    raise ValueError("Repair context cannot fit without truncating original task")


def _cached_weights(cache):
    from huggingface_hub import hf_hub_download
    try:
        index = Path(hf_hub_download(control.REFERENCE, "model.safetensors.index.json",
                                    revision=control.REVISION, cache_dir=cache, local_files_only=True))
        payload = json.loads(index.read_text())
        paths = [Path(hf_hub_download(control.REFERENCE, name, revision=control.REVISION,
                                      cache_dir=cache, local_files_only=True))
                 for name in sorted(set(payload["weight_map"].values()))]
        if sum(path.stat().st_size for path in paths) < payload["metadata"]["total_size"]:
            raise ValueError("Incomplete cached shards")
        return {"snapshot": str(index.parent), "shards": len(paths)}
    except Exception as error:
        raise RuntimeError("Official pinned checkpoint is not fully cached; no download attempted. "
                           "Use the same HF cache as the successful backend control.") from error


def _summary(report):
    rows = report["tasks"]
    passed = lambda attempt: attempt.get("lean", {}).get("status") == "passed"
    solved = [any(passed(a) for a in row["attempts"]) for row in rows]
    baseline = [row.get("automation", {}).get("status") == "passed" for row in rows]
    first = [bool(row["attempts"]) and passed(row["attempts"][0]) for row in rows]
    return {"selected_tasks": len(rows), "completed_tasks": sum(r.get("complete", False) for r in rows),
            "first_try_solved": sum(first), "solved_with_repairs": sum(solved),
            "newly_solved_by_repair": sum(s and not f for s, f in zip(solved, first)),
            "automation_solved": sum(baseline),
            "model_only_solved": sum(s and not b for s, b in zip(solved, baseline)),
            "automation_only_solved": sum(b and not s for s, b in zip(solved, baseline)),
            "model_calls": sum(len(r["attempts"]) for r in rows),
            "generation_requests_started": report.get("generation_requests_started", 0),
            "note": "Curated compatibility smoke, not held-out accuracy or causal feedback ablation."}


def _run(report, generate, validate, save, max_attempts, seed, *,
         tasks=None, messages_factory=None, automation_first=False):
    tasks = TASKS if tasks is None else tasks
    messages_factory = _messages if messages_factory is None else messages_factory
    for index, (task, row) in enumerate(zip(tasks, report["tasks"])):
        if row.get("complete") or any(a.get("lean", {}).get("status") == "passed" for a in row["attempts"]):
            row["complete"] = True
            save()
            continue
        if "automation" not in row:
            row["automation"] = validate(task, AUTOMATION)
            save()
        if automation_first and row["automation"]["status"] == "passed":
            row["complete"] = True
            save()
            continue
        while len(row["attempts"]) < max_attempts or (
                row["attempts"] and "lean" not in row["attempts"][-1]):
            if row["attempts"] and "lean" not in row["attempts"][-1]:
                attempt = row["attempts"][-1]  # Resume validation, no duplicate model call.
            else:
                previous = row["attempts"][-1] if row["attempts"] else None
                attempt_seed = seed + index * 10 + len(row["attempts"])
                report["generation_requests_started"] = report.get("generation_requests_started", 0) + 1
                save()
                attempt = generate(messages_factory(task, previous), attempt_seed)
                row["attempts"].append(attempt)
                save()  # Preserve raw completion before compiler execution.
            attempt["lean"] = validate(task, attempt["raw_text"])
            save()
            print(json.dumps({"task": row["id"], "attempt": len(row["attempts"]),
                              "status": attempt["lean"]["status"], "done_reason": attempt.get("done_reason"),
                              "output_tokens": attempt.get("output_tokens")}), flush=True)
            if attempt["lean"]["status"] == "passed":
                break
        row["complete"] = True
        save()
    return _summary(report)


def _recheck_automation(args):
    original = args.recheck_automation.read_text()
    report = json.loads(original)
    manifest = report["manifest"]
    if (manifest.get("schema") != "ten_proof_smoke_v1"
            or control._sha(manifest["tasks"]) != control._sha(TASKS)
            or [row["id"] for row in report["tasks"]] != [task[0] for task in TASKS]):
        raise ValueError("Unknown smoke cohort; refusing to reassign saved model results")
    context = control._lean_context(args.lean_project)
    if (context[1] != manifest["toolchain"]
            or control._sha((context[0] / "lake-manifest.json").read_text()) != manifest["lake_manifest_sha256"]):
        raise ValueError("Recheck must use the original Lean toolchain and manifest")
    if args.lean_timeout != manifest["lean_timeout"]:
        raise ValueError("Recheck must retain the original validator time budget")
    path = args.output_dir / "report.json"
    if path.resolve() == args.recheck_automation.resolve():
        raise ValueError("Use a separate output directory; original report stays unchanged")
    policy = {"source_report_sha256": control._sha(original), "automation_sha256": control._sha(AUTOMATION),
              "runner_sha256": control._sha(Path(__file__).read_text()), "model_inference": False}
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Recheck exists: {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("automation_recheck") != policy:
            raise ValueError("Recheck resume changes source report or automation policy")
        report = previous
    else:
        report["automation_recheck"] = policy
        for row in report["tasks"]:
            row["automation_before_recheck"] = row.get("automation")
            row["automation_recheck_complete"] = False
    control._write(path, report)
    for task, row in zip(TASKS, report["tasks"]):
        if row["automation_recheck_complete"]:
            continue
        row["automation"] = _validate(task, AUTOMATION, context, args.lean_timeout)
        row["automation_recheck_complete"] = True
        control._write(path, report)
        print(json.dumps({"task": task[0],
                          "automation_before": (row["automation_before_recheck"] or {}).get("status"),
                          "automation_after": row["automation"]["status"]}), flush=True)
    report["summary"] = _summary(report)
    control._write(path, report)
    print(json.dumps({"report": str(path), "new_model_calls": 0, **report["summary"]}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--lean-timeout", type=float, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recheck-automation", type=Path)
    args = parser.parse_args(argv)
    if (args.context_tokens <= args.max_new_tokens or args.max_new_tokens < 1 or args.seed < 0
            or not math.isfinite(args.request_timeout) or args.request_timeout <= 0
            or not math.isfinite(args.lean_timeout) or args.lean_timeout <= 0):
        raise ValueError("Invalid generation/context budget or seed")
    if args.recheck_automation:
        return _recheck_automation(args)
    args.backend, args.min_free_disk_gib, args.local_files_only = "transformers", 1, True
    cache = _cached_weights(args.hf_cache)
    runtime = control._preflight(args)
    context = control._lean_context(args.lean_project)
    # Stop before inference if the existing Lean environment cannot check proofs.
    probe = control._check_source("import Mathlib\nimport Aesop\ntheorem smoke_preflight : True := by trivial\n"
                                  "#print axioms smoke_preflight\n", "smoke_preflight", context, args.lean_timeout)
    if probe["status"] != "passed":
        raise RuntimeError("Lean preflight failed: " + json.dumps(probe))
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(control.REFERENCE, revision=control.REVISION,
                                                        cache_dir=args.hf_cache, local_files_only=True)
    initial_hashes = [control._sha(_render(tokenizer, _messages(task), args)[1]) for task in TASKS]
    manifest = {"schema": "ten_proof_smoke_v1", "tasks": TASKS,
                "reference": control.REFERENCE, "revision": control.REVISION,
                "dtype": args.dtype, "seed": args.seed, "max_attempts": args.max_attempts,
                "context_tokens": args.context_tokens, "max_new_tokens": args.max_new_tokens,
                "request_timeout": args.request_timeout, "lean_timeout": args.lean_timeout,
                "lean_project": str(context[0]), "toolchain": context[1],
                "lake_manifest_sha256": control._sha((context[0] / "lake-manifest.json").read_text()),
                "runtime": runtime["runtime"], "initial_input_hashes": initial_hashes,
                "runner_sha256": control._sha(Path(__file__).read_text()),
                "backend_sha256": control._sha(Path(control.__file__).read_text())}
    path = args.output_dir / "report.json"
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run exists: {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = control._sha(manifest)
    report = {"manifest": manifest, "binding": key, "preflight": runtime, "cached_weights": cache,
              "tasks": [{"id": task[0], "attempts": []} for task in TASKS]}
    if path.exists():
        report = json.loads(path.read_text())
        if report["binding"] != key:
            raise ValueError("Resume changes tasks, checkpoint, code, environment or budgets")
    save = lambda: control._write(path, report)
    save()
    with ExitStack() as stack:
        model = None

        def generate(messages, seed):
            nonlocal model
            messages, text, ids = _fit_prompt(tokenizer, messages, args)
            if model is None:
                model = stack.enter_context(control.ReferenceGenerator(args, tokenizer))
            result = model.generate(ids, seed)
            result.update(messages=messages, prompt_sha256=control._sha(text), input_ids_sha256=control._sha(ids))
            return result

        def validate(task, raw):
            return _validate(task, raw, context, args.lean_timeout)

        try:
            summary = _run(report, generate, validate, save, args.max_attempts, args.seed)
        except Exception as error:
            report["operational_error"] = {"type": type(error).__name__, "message": str(error)}
            save()
            raise
    report.pop("operational_error", None)
    report["summary"] = summary
    save()
    print(json.dumps({"report": str(path), **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
