"""Run fixed, resumable miniF2F validation with the accepted flat GNN only."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time

from maths_ai.gnn_inference.atp_lean_gnn.benchmark import prepare_cohort, run_cohort
from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy
from maths_ai.gnn_inference.atp_lean_gnn.native_session import environment_binding
from maths_ai.gnn_inference.atp_lean_gnn.proof_search import GnnSearchBudget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--pantograph-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=("base", "accepted_local_cues"), default="accepted_local_cues")
    parser.add_argument("--model-calls", type=int, default=32)
    parser.add_argument("--tactic-trials", type=int, default=96)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only and any(getattr(args, name) is None for name in ("bundle", "source_root", "pantograph_root", "output_dir")):
        parser.error("benchmark requires --bundle, --source-root, --pantograph-root and --output-dir")
    budget = GnnSearchBudget(args.model_calls, args.tactic_trials, args.nodes, args.depth, args.top_k, args.seconds)
    started = time.monotonic()
    cohort = prepare_cohort(args.cohort_dir, seed=args.seed)
    preparation = time.monotonic() - started
    if args.prepare_only:
        print(json.dumps({"selected_tasks": len(cohort["tasks"]), "ingest_status": dict(Counter(task["ingest_status"] for task in cohort["tasks"])),
            "cohort": str(args.cohort_dir / "cohort.json"), "upstream_toolchain": cohort["upstream_toolchain"],
            "upstream_mathlib": cohort["upstream_mathlib"]}, indent=2))
        return 0
    started = time.monotonic()
    environment = environment_binding(args.source_root, args.pantograph_root)
    environment_seconds = time.monotonic() - started
    started = time.monotonic()
    print("Loading pinned accepted GNN/retriever once", flush=True)
    policy = LiveJointPolicy.from_bundle(args.bundle, device=args.device, profile=args.profile)
    loading = time.monotonic() - started
    report = asyncio.run(run_cohort(policy, cohort, args.output_dir, environment=environment, budget=budget, resume=args.resume,
        invocation_costs={"model_loading_seconds": loading, "cohort_preparation_seconds": preparation,
                          "environment_validation_seconds": environment_seconds}))
    print(json.dumps({"summary": report["summary"], "model_loading_seconds_this_invocation": loading,
                      "report": str(args.output_dir / "report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
