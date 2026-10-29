"""Pinned, stratified miniF2F validation development run; no official test use."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re

from maths_ai.gnn_inference.scripts import compare_prover_backends as control
from maths_ai.gnn_inference.scripts import run_proof_smoke as smoke


REPOSITORY = "yangky11/miniF2F-lean4"
COMMIT = "5746b7d6c47855ce1294bed87329618ff7f1bc31"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/"
GROUPS = ("imo", "aime", "algebra", "induction", "numbertheory")


def _fetch(path):
    import requests
    response = requests.get(RAW + path, timeout=60)
    response.raise_for_status()
    return response.text


def _select(index, seed, per_group):
    names = re.findall(r"^import MiniF2F\.Valid\.([A-Za-z0-9_]+)\s*$", index, re.MULTILINE)
    if len(names) != len(set(names)) or not names:
        raise ValueError("Invalid or duplicate validation index entries")
    selected = []
    for group in GROUPS:
        pool = [name for name in names if name.lower().startswith(group)]
        if len(pool) < per_group:
            raise ValueError(f"Not enough {group} validation tasks for fixed selection")
        selected.extend((name, group) for name in sorted(
            pool, key=lambda name: control._sha(f"{seed}:{name}"))[:per_group])
    return selected


def _masked(source):
    """Mask comments/strings while preserving offsets for header boundaries."""
    chars = list(source)
    i, depth = 0, 0
    while i < len(source):
        if depth:
            if source.startswith("/-", i):
                depth += 1
                chars[i:i+2] = "  "
                i += 2
            elif source.startswith("-/", i):
                depth -= 1
                chars[i:i+2] = "  "
                i += 2
            else:
                if source[i] != "\n":
                    chars[i] = " "
                i += 1
        elif source.startswith("/-", i):
            depth = 1
            chars[i:i+2] = "  "
            i += 2
        elif source.startswith("--", i):
            while i < len(source) and source[i] != "\n":
                chars[i] = " "
                i += 1
        elif source[i] == '"':
            chars[i] = " "
            i += 1
            while i < len(source):
                char = source[i]
                if char != "\n":
                    chars[i] = " "
                i += 1
                if char == "\\" and i < len(source):
                    chars[i] = " "
                    i += 1
                elif char == '"':
                    break
        else:
            i += 1
    if depth:
        raise ValueError("Unterminated source comment")
    return "".join(chars)


def _parse(name, source):
    masked = _masked(source)
    declaration = re.search(r"^[ \t]*theorem\s+" + re.escape(name) + r"(?=\s|\(|\{|:)", masked, re.MULTILINE)
    if declaration is None:
        raise ValueError("Expected standalone named theorem not found")
    prefix = source[:declaration.start()]
    for line in _masked(prefix).splitlines():
        line = line.strip()
        if line and not (line in {"import Mathlib", "import Aesop"}
                         or line.startswith("open ") or line.startswith("set_option ")):
            raise ValueError("Unsupported pre-theorem context; no dependency/proof guessing")
    depth = 0
    for i in range(declaration.start(), len(masked) - 1):
        if masked[i] in "([{":
            depth += 1
        elif masked[i] in ")]}":
            depth -= 1
        elif depth == 0 and masked[i:i+2] == ":=":
            return {"prefix": prefix, "header": source[declaration.start():i].rstrip()}
    raise ValueError("Top-level proof assignment not found")


def _prepare(root, seed, per_group):
    path = root / "cohort.json"
    if path.exists():
        cohort = json.loads(path.read_text())
        if cohort["commit"] != COMMIT or cohort["seed"] != seed or cohort["per_group"] != per_group:
            raise ValueError("Existing cohort uses a different revision/selection")
        index = (root / "valid_index.lean").read_text()
        if (control._sha(index) != cohort["index_sha256"]
                or [(task["id"], task["group"]) for task in cohort["tasks"]] != _select(index, seed, per_group)):
            raise ValueError("Saved cohort selection changed")
        for task in cohort["tasks"]:
            source = (root / task["file"]).read_text()
            if control._sha(source) != task["source_sha256"]:
                raise ValueError("Saved benchmark source changed")
            if task["ingest_status"] == "parsed" and _parse(task["id"], source) != {
                    "prefix": task["prefix"], "header": task["header"]}:
                raise ValueError("Saved declaration differs from original source")
        return cohort
    root.mkdir(parents=True, exist_ok=True)
    index = _fetch("MiniF2F/Valid.lean")
    (root / "valid_index.lean").write_text(index)
    selected = _select(index, seed, per_group)  # Freeze names before source/compatibility checks.
    toolchain = _fetch("lean-toolchain").strip()
    upstream = json.loads(_fetch("lake-manifest.json"))
    mathlib = next(item["rev"] for item in upstream["packages"] if item["name"] == "mathlib")
    tasks = []
    for name, group in selected:
        relative = f"sources/{name}.lean"
        source = _fetch(f"MiniF2F/Valid/{name}.lean")
        (root / "sources").mkdir(exist_ok=True)
        (root / relative).write_text(source)
        task = {"id": name, "group": group, "file": relative, "source_sha256": control._sha(source)}
        try:
            task.update(_parse(name, source), ingest_status="parsed")
        except ValueError as error:
            task.update(ingest_status="unsupported", ingest_error=str(error))
        tasks.append(task)
    cohort = {"repository": REPOSITORY, "commit": COMMIT, "split": "valid", "seed": seed,
              "per_group": per_group, "index_sha256": control._sha(index),
              "upstream_toolchain": toolchain, "upstream_mathlib": mathlib, "tasks": tasks}
    control._write(path, cohort)
    return cohort


def _spec(task):
    name = task["id"]
    stub, spec = "Expected_" + name, "Spec_" + name
    header = re.sub(r"^theorem\s+" + re.escape(name), "axiom " + stub, task["header"].lstrip(), count=1)
    # Stub is only used to elaborate the original header's type. The final
    # axiom audit forbids using this stub (or sorry) as a proof premise.
    return (task["prefix"] + "\nset_option maxHeartbeats 200000\n"
            f"run_elab do\n  if (← Lean.getEnv).contains `{name} then\n"
            '    throwError "Benchmark target already available in imports"\n'
            + header + "\n"
            "run_elab do\n"
            f"  let info ← Lean.getConstInfo `{stub}\n"
            '  if !info.levelParams.isEmpty then throwError "Universe-polymorphic benchmark target unsupported"\n'
            "  Lean.addDecl (.defnDecl {\n"
            f"    name := `{spec}\n"
            "    levelParams := []\n    type := Lean.mkSort .zero\n    value := info.type\n"
            "    hints := .abbrev\n    safety := .safe\n  })\n")


def _validate(task, raw, context, timeout):
    if task["ingest_status"] != "parsed":
        return {"status": "input_unsupported", "error": task["ingest_error"]}
    if raw.count("```") % 2:
        return {"status": "no_complete_body", "error": "Unterminated final code block"}
    blocks = re.findall(r"```(?:lean4?)?[ \t]*\n(.*?)```", raw, re.DOTALL)
    code = blocks[-1].strip() if blocks else raw.strip()
    decls = list(re.finditer(r"^[ \t]*(?:theorem|lemma)\s+([A-Za-z_][\w']*)", code, re.MULTILINE))
    if decls:
        chosen = next((m for m in decls if m.group(1) == task["id"]), decls[0])
        candidate = code[decls[0].start():]
        candidate_name = chosen.group(1)
    else:
        try:
            body = control._extract_body(raw)
        except ValueError as error:
            return {"status": "no_complete_body", "error": str(error)}
        candidate_name = task["id"]
        candidate = task["header"] + " := by\n" + "\n".join("  " + line for line in body.splitlines())
    namespace, checked = "Candidate_" + task["id"], "Checked_" + task["id"]
    source = (_spec(task) + f"namespace {namespace}\n" + candidate
              + f"\nend {namespace}\nset_option maxHeartbeats 200000\n"
              f"theorem {checked} : Spec_{task['id']} := by exact {namespace}.{candidate_name}\n"
              f"#print axioms {checked}\n")
    return control._check_source(source, checked, context, timeout)


def _messages(task, previous=None):
    source = task["prefix"] + task["header"] + " := by\n  sorry\n"
    messages = [{"role": "user", "content": "Complete the following Lean 4 code:\n```lean4\n"
                 + source + "```\n" + control.PLAN}]
    if previous:
        raw = previous["raw_text"]
        blocks = re.findall(r"```(?:lean4?)?[ \t]*\n(.*?)```", raw, re.DOTALL)
        failed = blocks[-1] if blocks and raw.count("```") % 2 == 0 else raw[-1200:]
        feedback = previous["lean"].get("diagnostics") or previous["lean"].get("error") or previous["lean"]["status"]
        messages.extend([
            {"role": "assistant", "content": "```lean4\n" + failed + "\n```"},
            {"role": "user", "content": "Verifier feedback:\n" + feedback
             + "\nProvide a corrected complete proof of the original statement without sorry or new assumptions."},
        ])
    return messages


def _summary(report):
    result = smoke._summary(report)
    result.update(compatible_tasks=sum(row.get("input_status") == "passed" for row in report["tasks"]),
                  input_failures=sum(row.get("input_status") != "passed" for row in report["tasks"]),
                  baseline_checked_tasks=sum("automation" in row for row in report["tasks"]),
                  model_eligible_tasks=sum(row.get("input_status") == "passed"
                                           and row.get("automation", {}).get("status") != "passed"
                                           for row in report["tasks"]),
                  model_attempted_tasks=sum(bool(row["attempts"]) for row in report["tasks"]),
                  combined_solved=result["automation_solved"] + result["model_only_solved"],
                  note="miniF2F-valid development subset; not official test, not pretraining-held-out accuracy.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--allow-environment-port", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--lean-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.per_group < 1 or args.seed < 0 or args.context_tokens <= args.max_new_tokens
            or args.max_new_tokens < 1 or not math.isfinite(args.lean_timeout) or args.lean_timeout <= 0
            or not math.isfinite(args.request_timeout) or args.request_timeout <= 0):
        raise ValueError("Invalid cohort or token budget")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume or a new directory")
    cohort = _prepare(args.output_dir, args.seed, args.per_group)
    context = control._lean_context(args.lean_project)
    lake = json.loads((context[0] / "lake-manifest.json").read_text())
    mathlib = next((item["rev"] for item in lake.get("packages", []) if item["name"] == "mathlib"), None)
    if mathlib is None:
        import subprocess
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=context[0],
                               capture_output=True, text=True, check=True).stdout.strip()
        if Path(root).resolve() == context[0]:
            mathlib = subprocess.run(["git", "rev-parse", "HEAD"], cwd=context[0],
                                     capture_output=True, text=True, check=True).stdout.strip()
    exact = context[1] == cohort["upstream_toolchain"] and mathlib == cohort["upstream_mathlib"]
    if not exact and not args.allow_environment_port:
        raise ValueError("Benchmark pins differ; use exact environment or explicitly --allow-environment-port")
    tasks = cohort["tasks"]
    manifest = {"schema": "minif2f_valid_dev_v1", "cohort_sha256": control._sha(cohort),
                "environment_mode": "upstream_pins" if exact else "explicit_development_port",
                "toolchain": context[1], "mathlib_revision": mathlib,
                "lake_manifest_sha256": control._sha(lake), "seed": args.seed,
                "reference": control.REFERENCE, "reference_revision": control.REVISION,
                "context_tokens": args.context_tokens, "max_new_tokens": args.max_new_tokens,
                "max_attempts": args.max_attempts, "lean_timeout": args.lean_timeout,
                "request_timeout": args.request_timeout, "dtype": args.dtype, "device": args.device,
                "runtime": control._runtime(), "runner_sha256": control._sha(Path(__file__).read_text()),
                "smoke_sha256": control._sha(Path(smoke.__file__).read_text()),
                "backend_sha256": control._sha(Path(control.__file__).read_text())}
    path = args.output_dir / "report.json"
    binding = control._sha(manifest)
    report = {"manifest": manifest, "binding": binding, "cohort": cohort,
              "tasks": [{"id": task["id"], "attempts": []} for task in tasks]}
    if path.exists():
        report = json.loads(path.read_text())
        if report["binding"] != binding:
            raise ValueError("Resume changes selection, sources, environment, code or budgets")
    save = lambda: control._write(path, report)
    save()
    for task, row in zip(tasks, report["tasks"]):
        if "input_status" not in row:
            if task["ingest_status"] != "parsed":
                row["input_status"] = "unsupported"
                row["input_error"] = task["ingest_error"]
            else:
                preflight = _spec(task) + f"theorem Input_{task['id']} : True := by trivial\n#print axioms Input_{task['id']}\n"
                row["input_check"] = control._check_source(preflight, "Input_" + task["id"], context, args.lean_timeout)
                row["input_status"] = row["input_check"]["status"]
            if row["input_status"] != "passed":
                row["complete"] = True
            save()
        if row["input_status"] == "passed" and "automation" not in row:
            row["automation"] = _validate(task, smoke.AUTOMATION, context, args.lean_timeout)
            if row["automation"]["status"] == "passed":
                row["complete"] = True
            save()
            print(json.dumps({"task": task["id"], "input": row["input_status"],
                              "automation": row["automation"]["status"]}), flush=True)
    if not args.baseline_only and any(not row.get("complete") for row in report["tasks"]):
        args.backend, args.min_free_disk_gib, args.local_files_only = "transformers", 1, True
        report["cached_weights"] = smoke._cached_weights(args.hf_cache)
        report["preflight"] = control._preflight(args)
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(control.REFERENCE, revision=control.REVISION,
                                                            cache_dir=args.hf_cache, local_files_only=True)
        with ExitStack() as stack:
            model = None

            def generate(messages, seed):
                nonlocal model
                messages, rendered, ids = smoke._fit_prompt(tokenizer, messages, args)
                if model is None:
                    model = stack.enter_context(control.ReferenceGenerator(args, tokenizer))
                result = model.generate(ids, seed)
                result.update(messages=messages, prompt_sha256=control._sha(rendered), input_ids_sha256=control._sha(ids))
                return result

            try:
                smoke._run(report, generate, lambda task, raw: _validate(task, raw, context, args.lean_timeout),
                           save, args.max_attempts, args.seed, tasks=tasks, messages_factory=_messages, automation_first=True)
            except Exception as error:
                report["operational_error"] = {"type": type(error).__name__, "message": str(error)}
                report["summary"] = _summary(report)
                save()
                raise
    report.pop("operational_error", None)
    report["summary"] = _summary(report)
    save()
    print(json.dumps({"report": str(path), **report["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
