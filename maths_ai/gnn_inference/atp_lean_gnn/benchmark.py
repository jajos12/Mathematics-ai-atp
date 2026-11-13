"""Fixed miniF2F validation ingestion and resumable GNN-only cohort evaluation."""

import csv
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
from datetime import datetime, timezone
from urllib.request import urlopen

from .bundle import file_sha256
from .native_session import LiveLeanSession, LiveStateError
from .proof_search import search_gnn
from .proof_tasks import checked_header, parse_problem, split_prefix
from .proof_verification import LeanProofVerifier
from .reporting import atomic_json, payload_sha256


REPOSITORY = "yangky11/miniF2F-lean4"
COMMIT = "5746b7d6c47855ce1294bed87329618ff7f1bc31"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/"
REPORT_SCHEMA = "gnn_minif2f_validation_v1"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch(path):
    with urlopen(RAW + path, timeout=60) as response:
        return response.read().decode("utf-8")


def selected_tasks(index, seed):
    names = re.findall(r"^import MiniF2F\.Valid\.([A-Za-z0-9_]+)\s*$", index, re.MULTILINE)
    if not names or len(names) != len(set(names)):
        raise ValueError("invalid validation index")
    return [(name, "_".join(name.split("_")[:2]) if name.startswith("mathd_") else name.split("_")[0])
            for name in sorted(names, key=lambda name: _sha(f"{seed}:{name}"))]


def prepare_cohort(root, *, seed=42, fetch=_fetch, expected_count=244):
    root = Path(root).resolve()
    path, owner = root / "cohort.json", root / "gnn_cache_owner.json"
    ownership = {"repository": REPOSITORY, "commit": COMMIT, "seed": seed, "selection": "full"}
    if path.is_file():
        cohort = json.loads(path.read_text())
        if (cohort.get("repository") != REPOSITORY or cohort.get("commit") != COMMIT
                or cohort.get("split") != "valid" or cohort.get("selection") != "full" or cohort.get("seed") != seed):
            raise ValueError("cohort differs from fixed validation protocol")
        index = (root / "valid_index.lean").read_text()
        if cohort["index_sha256"] != _sha(index):
            raise ValueError("validation index changed")
    else:
        if root.exists() and any(root.iterdir()) and not owner.is_file():
            raise ValueError("unrecognized nonempty cohort directory")
        root.mkdir(parents=True, exist_ok=True)
        if owner.is_file() and json.loads(owner.read_text()) != ownership:
            raise ValueError("cohort cache ownership differs")
        atomic_json(owner, ownership)
        index = fetch("MiniF2F/Valid.lean")
        tasks = selected_tasks(index, seed)
        if len(tasks) != expected_count:
            raise ValueError("validation population differs from protocol")
        (root / "valid_index.lean").write_text(index)
        upstream = json.loads(fetch("lake-manifest.json"))
        cohort = {"repository": REPOSITORY, "commit": COMMIT, "split": "valid", "seed": seed,
            "per_group": 5, "selection": "full", "index_sha256": _sha(index),
            "upstream_toolchain": fetch("lean-toolchain").strip(),
            "upstream_mathlib": next(package["rev"] for package in upstream["packages"] if package["name"] == "mathlib"),
            "tasks": [{"id": name, "group": group, "file": f"sources/{name}.lean", "ingest_status": "pending"}
                      for name, group in tasks]}
        atomic_json(path, cohort)
    tasks = selected_tasks(index, seed)
    if (len(tasks) != expected_count or [(task["id"], task["group"]) for task in cohort["tasks"]] != tasks):
        raise ValueError("cohort identities/denominator differ from fixed selection")
    pending = any(task["ingest_status"] == "pending" for task in cohort["tasks"])
    if pending and (not owner.is_file() or json.loads(owner.read_text()) != ownership):
        raise ValueError("incomplete external cohort is read-only; use a new GNN cache")
    for number, task in enumerate(cohort["tasks"], 1):
        expected_file = f"sources/{task['id']}.lean"
        if task["file"] != expected_file:
            raise ValueError("unsafe cohort source path")
        source_path = root / expected_file
        if task["ingest_status"] == "pending":
            source = fetch("MiniF2F/Valid/" + task["id"] + ".lean")
            source_path.parent.mkdir(exist_ok=True)
            if source_path.exists() and source_path.read_text() != source:
                raise ValueError("partial source differs from pinned upstream")
            source_path.write_text(source)
            task["source_sha256"] = _sha(source)
            try:
                task.update(parse_problem(task["id"], source), ingest_status="parsed")
            except ValueError as error:
                task.update(ingest_status="unsupported", ingest_error=str(error))
            atomic_json(path, cohort)
        else:
            source = source_path.read_text()
            if _sha(source) != task["source_sha256"]:
                raise ValueError("saved original theorem source changed")
            if task["ingest_status"] == "parsed" and parse_problem(task["id"], source) != {
                    "prefix": task["prefix"], "header": task["header"]}:
                raise ValueError("saved theorem boundary changed")
            if task["ingest_status"] not in {"parsed", "unsupported"}:
                raise ValueError("invalid ingestion status")
        if number % 25 == 0 or number == len(tasks):
            print(f"validation sources ready: {number}/{len(tasks)}", flush=True)
    return cohort


def implementation_binding():
    root = Path(__file__).parent
    versions = {}
    for distribution in ("torch", "torch-geometric", "numpy", "transformers", "tokenizers", "huggingface-hub", "faiss-cpu", "safetensors"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    files = {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*.py"))}
    for name in ("run_gnn_benchmark.py", "run_gnn_search.py"):
        files["../scripts/" + name] = file_sha256(root.parent / "scripts" / name)
    return {"files": files, "versions": versions}


def inference_binding(policy):
    import faiss
    import torch
    device = next(policy.system.policy.parameters()).device
    hardware = {"device": str(device), "machine": platform.machine(), "platform": platform.platform(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")}}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=properties.name,
            memory_bytes=properties.total_memory, capability=list(torch.cuda.get_device_capability(device)),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    return hardware


def summarize(records, selected):
    statuses = {}
    solved, calls, responses, trials, nodes, active = 0, 0, 0, 0, 0, 0.0
    startup, end_to_end, model, verification = 0.0, 0.0, 0.0, 0.0
    for record in records:
        status = record["status"]
        statuses[status] = statuses.get(status, 0) + 1
        search = record.get("search", {})
        passed = status == "solved" and search.get("proof_verified") is True
        if status == "solved" and not passed:
            raise ValueError("unverified solved task")
        solved += int(passed)
        calls += search.get("model_calls", 0)
        responses += search.get("model_responses", 0)
        trials += search.get("tactic_trials", 0)
        nodes += len(search.get("nodes", []))
        active += search.get("seconds", 0)
        startup += record.get("native_startup_seconds", 0)
        end_to_end += record.get("end_to_end_seconds", 0)
        model += search.get("model_seconds", 0)
        verification += sum(check.get("seconds", 0) for check in search.get("verifications", []))
    completed = sum(record["status"] != "running" for record in records)
    return {"selected_tasks": selected, "completed_tasks": completed, "pending_tasks": selected - completed,
        "verified_solved": solved, "solve_rate": solved / max(1, selected), "status_counts": statuses,
        "model_calls": calls, "model_responses": responses, "tactic_trials": trials, "nodes": nodes,
        "summed_active_seconds": active, "summed_native_startup_seconds": startup,
        "summed_task_end_to_end_seconds": end_to_end, "summed_model_seconds": model,
        "summed_verification_seconds": verification,
        "elapsed_cost_is_lower_bound": any(record.get("elapsed_cost_is_lower_bound", False) or record["status"] == "running" for record in records),
        "model": "accepted_flat_gnn", "deepseek_used": False,
        "proof_success_measured": bool(records)}


async def run_cohort(policy, cohort, output, *, environment, budget, resume=False, session_factory=None,
                     verifier_factory=None, preflight=True, invocation_costs=None):
    """Keep one writer per output, including across separate resume processes."""
    import fcntl
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Retain the lock inode: unlinking it could let a third writer bypass a
    # process that already opened the previous lock file.
    with (output.parent / f".{output.name}.gnn.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("benchmark output already has an active writer") from error
        return await _run_cohort(policy, cohort, output, environment=environment, budget=budget, resume=resume,
            session_factory=session_factory, verifier_factory=verifier_factory, preflight=preflight, invocation_costs=invocation_costs)


async def _run_cohort(policy, cohort, output, *, environment, budget, resume, session_factory,
                      verifier_factory, preflight, invocation_costs):
    output = Path(output).resolve()
    base_imports = ("Init", "Mathlib")
    source_root, pantograph_root = environment["source_root"], environment["pantograph_root"]
    manifest = {"schema": REPORT_SCHEMA, "cohort_sha256": payload_sha256(cohort), "environment": environment,
        "model_bundle_sha256": file_sha256(policy.system.root / "bundle.json"),
        "policy_state_sha256": policy.system.manifest["policy_state_sha256"],
        "retriever_state_sha256": policy.system.manifest["retriever_state_sha256"],
        "profile": policy.system.profile, "budget": budget.__dict__, "implementation": implementation_binding(),
        "inference": inference_binding(policy),
        "native_limits": {"step_seconds": 20, "startup_seconds": 120, "heartbeats": 1000000, "rec_depth": 2048},
        "timing": "active_search_and_verification_excludes_loading_and_native_startup",
        "resume": "completed_task_boundary_interrupted_tasks_remain_failures_no_fresh_budget",
        "premise_filter": "exclude_target_and_native_runtime_declarations_v1",
        "environment_mode": "upstream" if cohort["upstream_toolchain"] == environment["toolchain"]
            and cohort["upstream_mathlib"] == environment["revisions"]["source"] else "development_port"}
    binding = payload_sha256(manifest)
    report_path = output / "report.json"
    if output.exists():
        if not resume or not report_path.is_file():
            raise FileExistsError("new benchmark output required; use --resume for an existing bound run")
        report = json.loads(report_path.read_text())
        if report["manifest"] != manifest or report["binding"] != binding or report["cohort"] != cohort:
            raise ValueError("benchmark resume changed cohort/model/environment/code/budgets")
    else:
        output.mkdir(parents=True)
        (output / "tasks").mkdir()
        report = {"manifest": manifest, "binding": binding, "cohort": cohort, "complete": False,
                  "task_records": {}, "summary": summarize([], len(cohort["tasks"]))}
        atomic_json(report_path, report)
    if set(report["task_records"]) - {task["id"] for task in cohort["tasks"]}:
        raise ValueError("unknown task in saved report")
    records = {}
    for identity, entry in report["task_records"].items():
        if entry["file"] != f"tasks/{identity}.json":
            raise ValueError("unsafe saved task record path")
        task_path = output / entry["file"]
        if not task_path.is_file():
            raise ValueError("saved task file missing; refusing fresh budget")
        if entry["status"] != "running" and entry["sha256"] != file_sha256(task_path):
            raise ValueError("completed task record changed")
        records[identity] = json.loads(task_path.read_text())

    def save(record):
        record["binding"] = binding
        task_path = output / "tasks" / f"{record['id']}.json"
        atomic_json(task_path, record)
        report["task_records"][record["id"]] = {"file": str(task_path.relative_to(output)), "sha256": file_sha256(task_path),
                                               "status": record["status"]}
        records[record["id"]] = record
        report["summary"] = summarize(list(records.values()), len(cohort["tasks"]))
        atomic_json(report_path, report)

    session_factory = session_factory or (lambda task: LiveLeanSession(source_root=source_root,
        pantograph_root=pantograph_root, imports=base_imports, prefix=task["prefix"]))
    verifier_factory = verifier_factory or (lambda task: LeanProofVerifier(source_root=Path(source_root),
        imports=base_imports, prefix=task["prefix"]))
    invocation = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "resume": resume, "costs": invocation_costs or {}, "successful_proof_rechecks": []}
    for previous in report.get("invocations", []):
        if previous["status"] == "running":
            previous.update(status="interrupted", elapsed_cost_is_lower_bound=True)
    report.setdefault("invocations", []).append(invocation)
    invocation_started = time.monotonic()

    async def execute():
        if preflight:
            started = time.monotonic()
            task = {"prefix": "", "header": "theorem GnnRuntimePreflight (P : Prop) (h : P) : P"}
            async with session_factory(task) as session:
                root = await session.start_declaration(task["header"])
                from .model_graph import normalized_goal_record
                normalized_goal_record(root.goals[0])
                closed = await session.step(root, "exact h")
                check = await verifier_factory(task).verify(task["header"], ["exact h"], timeout=budget.verification_seconds)
                if not closed.closed or check["status"] != "passed":
                    raise RuntimeError("native runtime/independent verifier preflight failed")
            invocation["preflight_seconds"] = time.monotonic() - started
            report["preflight_passed"] = True
            atomic_json(report_path, report)
        for task in cohort["tasks"]:
            await execute_task(task)

    async def execute_task(task):
        task_path = output / "tasks" / f"{task['id']}.json"
        task_sha = payload_sha256(task)
        if task_path.is_file():
            record = json.loads(task_path.read_text())
            if record.get("binding") != binding or record.get("task_sha256") != task_sha:
                raise ValueError("saved task source or protocol differs")
            previous = report["task_records"].get(task["id"])
            if (previous and previous["status"] != "running"
                    and previous["sha256"] != file_sha256(task_path)):
                raise ValueError("completed task record changed")
            if record.get("id") != task["id"]:
                raise ValueError("saved task identity differs")
            records[task["id"]] = record
            if record["status"] in {"running", "interrupted"}:
                record["elapsed_cost_is_lower_bound"] = record.get("elapsed_cost_is_lower_bound", False) or record["status"] == "running"
                record["status"] = "interrupted"
                if record.get("search", {}).get("status") == "running":
                    record["search"]["status"] = "interrupted"
                save(record)
            elif record["status"] == "solved":
                check = await verifier_factory(task).verify(task["header"], record["search"]["accepted_path"],
                    timeout=budget.verification_seconds)
                if check["status"] != "passed":
                    raise ValueError("saved successful proof failed independent resume recheck")
                invocation["successful_proof_rechecks"].append({"id": task["id"], **check})
            return
        if task["id"] in report["task_records"]:
            raise ValueError("saved task file missing; refusing fresh budget")
        record = {"id": task["id"], "group": task["group"], "task_sha256": task_sha, "status": "running", "search": {}}
        save(record)
        if task["ingest_status"] != "parsed":
            record.update(status="input_unsupported", error=task.get("ingest_error", "unsupported source"))
        else:
            started = time.monotonic()
            try:
                _, target_name = checked_header(task["header"], prefix=task["prefix"])
                async with session_factory(task) as session:
                    await session.start_declaration(task["header"])
                    record["native_startup_seconds"] = time.monotonic() - started

                    def checkpoint(search):
                        record["search"] = search
                        # The task remains in flight until context cleanup and
                        # final timing are durably saved, even after a solve.
                        record["status"] = "running"
                        save(record)

                    search = await search_gnn(policy.excluding({target_name}), session, task["header"], verifier_factory(task),
                                              budget=budget, checkpoint=checkpoint)
                    record.update(status=search["status"], search=search)
            except (LiveStateError, ValueError, RuntimeError, OSError) as error:
                record.update(status="input_error" if "native_startup_seconds" not in record else "runtime_error", error=str(error))
            except (KeyboardInterrupt, asyncio.CancelledError):
                record["status"] = "interrupted"
                raise
            finally:
                record["end_to_end_seconds"] = time.monotonic() - started
                save(record)
        save(record)
        print(f"tasks={len(records)}/{len(cohort['tasks'])} solved={report['summary']['verified_solved']} "
               f"last={task['id']}:{record['status']}", flush=True)
    try:
        atomic_json(report_path, report)
        await execute()
        report["complete"] = True
        invocation["status"] = "completed"
    except BaseException as error:
        invocation.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "failed",
                          error=f"{type(error).__name__}: {error}")
        raise
    finally:
        invocation["runner_seconds"] = time.monotonic() - invocation_started
        invocation["accounted_end_to_end_seconds"] = invocation["runner_seconds"] + sum(invocation["costs"].values())
        # Include untouched records on a failed/interrupted resume, never shrink
        # its previous cost totals to only tasks visited during this invocation.
        for identity, entry in report["task_records"].items():
            if identity not in records:
                records[identity] = json.loads((output / entry["file"]).read_text())
        report["summary"] = summarize(list(records.values()), len(cohort["tasks"]))
        report["summary"]["elapsed_cost_is_lower_bound"] |= any(previous.get("elapsed_cost_is_lower_bound", False) for previous in report["invocations"])
        atomic_json(report_path, report)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "group", "status", "verified", "model_calls", "model_responses", "trials", "nodes",
            "active_seconds", "native_startup_seconds", "end_to_end_seconds", "model_seconds", "verification_seconds", "elapsed_cost_is_lower_bound"))
        writer.writeheader()
        for record in records.values():
            search = record.get("search", {})
            writer.writerow({"id": record["id"], "group": record["group"], "status": record["status"], "verified": search.get("proof_verified", False),
                "model_calls": search.get("model_calls", 0), "model_responses": search.get("model_responses", 0),
                "trials": search.get("tactic_trials", 0), "nodes": len(search.get("nodes", [])), "active_seconds": search.get("seconds", 0),
                "native_startup_seconds": record.get("native_startup_seconds", 0), "end_to_end_seconds": record.get("end_to_end_seconds", 0),
                "model_seconds": search.get("model_seconds", 0), "verification_seconds": sum(check.get("seconds", 0) for check in search.get("verifications", [])),
                "elapsed_cost_is_lower_bound": record.get("elapsed_cost_is_lower_bound", False)})
    return report
