"""Single-target accepted GNN proof-search smoke/diagnostic entrypoint."""

import argparse
import asyncio
import json
from pathlib import Path
import time

from maths_ai.gnn_inference.atp_lean_gnn.benchmark import implementation_binding, inference_binding
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy
from maths_ai.gnn_inference.atp_lean_gnn.native_session import LiveLeanSession, environment_binding
from maths_ai.gnn_inference.atp_lean_gnn.proof_search import GnnSearchBudget, search_gnn
from maths_ai.gnn_inference.atp_lean_gnn.proof_tasks import checked_header, split_prefix
from maths_ai.gnn_inference.atp_lean_gnn.proof_verification import LeanProofVerifier
from maths_ai.gnn_inference.atp_lean_gnn.reporting import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-root", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--header")
    target.add_argument("--goal")
    parser.add_argument("--prefix-file", type=Path)
    parser.add_argument("--imports", nargs="+", default=["Init", "Mathlib"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=("base", "accepted_local_cues"), default="accepted_local_cues")
    parser.add_argument("--model-calls", type=int, default=32)
    parser.add_argument("--tactic-trials", type=int, default=96)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    prefix = args.prefix_file.read_text() if args.prefix_file else ""
    header, name = checked_header(args.header if args.header is not None else f"theorem GnnSearchChecked : {args.goal}", prefix=prefix)
    budget = GnnSearchBudget(args.model_calls, args.tactic_trials, args.nodes, args.depth, args.top_k, args.seconds)
    prefix_imports, _ = split_prefix(prefix)
    imports = tuple(dict.fromkeys((*args.imports, *prefix_imports)))
    environment = environment_binding(args.source_root, args.pantograph_root, imports=imports)
    started = time.monotonic()
    policy = LiveJointPolicy.from_bundle(args.bundle, device=args.device, profile=args.profile).excluding({name})
    binding = {"environment": environment, "bundle_sha256": file_sha256(args.bundle / "bundle.json"),
               "profile": args.profile, "implementation": implementation_binding(), "inference": inference_binding(policy)}
    timing = {"model_loading_seconds": time.monotonic() - started}

    def save(report):
        report["manifest"] = binding
        report["timing"] = timing
        atomic_json(args.output, report)

    async def run():
        started = time.monotonic()
        async with LiveLeanSession(source_root=args.source_root, pantograph_root=args.pantograph_root,
                                   imports=imports, prefix=prefix) as session:
            await session.start_declaration(header)
            timing["native_startup_seconds"] = time.monotonic() - started
            verifier = LeanProofVerifier(source_root=args.source_root, imports=imports, prefix=prefix)
            try:
                return await search_gnn(policy, session, header, verifier, budget=budget, checkpoint=save)
            finally:
                timing["native_and_search_seconds"] = time.monotonic() - started

    result = asyncio.run(run())
    save(result)
    print(f"{result['status']}: calls={result['model_calls']} trials={result['tactic_trials']} "
          f"nodes={len(result['nodes'])} seconds={result['seconds']:.2f}; report={args.output}")
    return 0 if result["proof_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
