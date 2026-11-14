"""Characterize pinned persistent-atp/provider interoperability with fake search.

Uses real routers, provider persistence, graph serialization, and external
adapter validation. No Lean proof or model-quality claim comes from fixtures.
"""

import argparse
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("main", "mcts"), required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.provider_root), str(args.persistent_root / "src")]

    from fastapi.testclient import TestClient
    from maths_ai.api.app import create_app
    from mathproof.maths_ai_http import MathsAIAPIError, MathsAIHTTPFormalATP

    request = {
        "proof_id": "integration", "claim_id": "integration/c-1",
        "formal_declaration_id": "integration/fd-1", "run_id": "integration/fr-1",
        "base_revision": 0, "lease_id": "integration-lease", "fencing_token": 1,
        "lean_source_artifact": "sha256:" + "b2" * 32,
        "environment_id": "fixture", "environment_hash": "sha256:" + "c3" * 32,
        "goal_text": "P", "hypotheses": ["P : Prop", "h : P"],
        "budget": {"max_steps": 3, "wall_clock_ms": 5000},
        "search_policy": {"deny_sorry": True, "policy_name": "accepted-gnn"},
    }

    class Service:
        mode = "solved"

        async def shutdown(self):
            pass

        async def stop(self):
            pass

        async def prove(self, goal, hypotheses=None, **kwargs):
            from maths_ai.data_models.proof_components import Goal, TacticCandidate
            from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph
            graph = ProofHypergraph(Goal(expression=goal, hypotheses=hypotheses or []))
            if self.mode == "solved":
                graph.add_edge(graph.root_id, TacticCandidate(tactic_name="exact", arguments=["h"], probability=1), [])
            elif self.mode == "budget":
                graph.mark_budget_exhausted()
            else:
                graph.mark_node_exhausted(graph.root_id, note="fixture frontier exhausted")
            return graph

        async def replay(self, *args):
            return {"verified": True, "status": "verified"}

    service = Service()

    async def factory():
        return service

    def revision(root, name):
        process = subprocess.run(["git", "rev-parse", "--verify", name], cwd=root,
                                 capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None

    result = {"mode": args.mode, "fixtures_only": True,
        "provider_head": revision(args.provider_root, "HEAD"),
        "provider_pending_merge": revision(args.provider_root, "MERGE_HEAD"),
        "persistent_head": revision(args.persistent_root, "HEAD")}
    with tempfile.TemporaryDirectory() as directory:
        app = create_app(factory, formal_runs_dir=Path(directory)) if args.mode == "main" else create_app(factory)
        with TestClient(app) as client:
            paths = {path.replace("{run_id:path}", "{run_id}")
                     for path in client.get("/openapi.json").json()["paths"]}
            result["routes"] = sorted(paths)
            result["missing_formal_routes"] = sorted({"/formal-search/start", "/formal-search/{run_id}",
                "/formal-search/{run_id}/retry", "/formal-search/{run_id}/cancel", "/formal-replay"} - paths)

            def bridge(method, path, payload=None, *, expected_statuses=frozenset({200})):
                response = client.request(method, path, json=payload)
                if response.status_code not in expected_statuses:
                    raise MathsAIAPIError(response.text, status_code=response.status_code)
                return response.json()

            result["canonical_start_direct_http"] = client.post("/formal-search/start", json=request).status_code
            if args.mode == "mcts":
                result["legacy_prove_request_http"] = client.post("/prove", json={"goal": "P", "timeout": 5}).status_code
                print(json.dumps(result, indent=2))
                return

            for serial, mode in enumerate(("solved", "budget", "dead"), 1):
                service.mode = mode
                current = {**request, "run_id": f"integration/fr-{serial}"}
                adapter = MathsAIHTTPFormalATP("http://testserver")
                adapter._call = bridge
                adapter.formal_search_start(current)
                deadline = time.monotonic() + 3
                while True:
                    translated = adapter.formal_search_status(current["run_id"])
                    if translated["disposition"] != "searching":
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("fixture did not finish")
                    time.sleep(.01)
                row = {"disposition": translated["disposition"], "artifacts": translated.get("artifacts"),
                    "mapped_budget": adapter._envelopes[current["run_id"]]["budget"]}
                if mode == "solved":
                    try:
                        row["replay"] = adapter.formal_replay(translated["certificate"], current["environment_hash"])
                    except MathsAIAPIError as error:
                        row["replay_error"] = str(error)
                result[mode] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
