
"""Run tactic inference on a single proof state interactively."""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from atp_lean_gnn.cli import DEMO_STATE
from atp_lean_gnn.bundle import (
    load_baseline_weights_into_pointer,
    load_pointer_bundle,
    load_state_dict_checked,
)
from atp_lean_gnn.inference import InferencePipeline
from atp_lean_gnn.lemma_index import LemmaIndex
from atp_lean_gnn.training import (
    PreparedMetadata,
    detect_state_dict_model_type,
    load_baseline_config,
    load_prepared_metadata,
)
from atp_lean_gnn.premise_scoring import PremiseScorer
from atp_lean_gnn.lemma_corpus import load_lemma_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Tactic Inference")
    parser.add_argument("--bundle", type=str, default=None, help="Path to an exported model bundle directory (preferred: it carries and verifies its own vocabularies, so no prepared dataset is needed)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json (e.g. from runs/baseline_gnn/run_*/config.json). Not needed with --bundle.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best.pt checkpoint (backbone + tactic model). Not needed with --bundle.")
    parser.add_argument("--scorer-checkpoint", type=str, default=None, help="Path to a premise scorer checkpoint (best.pt from a premise_selection run). If omitted, the scorer uses random weights.")
    parser.add_argument("--index-path", type=str, help="Path to FAISS index. If missing, retrieval will return nothing.")
    parser.add_argument("--corpus-path", type=str, help="Path to lemmas.jsonl for decoding retrieved lemma IDs to names.")
    parser.add_argument("--k", type=int, default=500, help="Number of lemmas to retrieve")
    parser.add_argument("--state", type=str, default=DEMO_STATE, help="Raw Lean proof state string")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading on {device}...")

    scorer: PremiseScorer | None = None
    if args.bundle:
        bundle_dir = Path(args.bundle)
        if not bundle_dir.exists():
            print(f"ERROR: bundle directory not found: {bundle_dir}")
            return 1
        # A baseline bundle is wrapped into a pointer shell here, because
        # InferencePipeline reaches through model.backbone.
        loaded = load_pointer_bundle(bundle_dir, device=device)
        model = loaded.model
        metadata = loaded.metadata
        hidden_dim = int(loaded.config.model.hidden_dim)
        scorer = loaded.scorer
        if loaded.randomly_initialized:
            print(
                f"WARNING: {bundle_dir} has no trained argument head; "
                f"{', '.join(loaded.randomly_initialized)} is randomly initialized."
            )
    else:
        if not args.config or not args.checkpoint:
            print("ERROR: pass --bundle, or both --config and --checkpoint.")
            return 1

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: config file not found: {config_path}")
            return 1

        config = load_baseline_config(config_path)
        hidden_dim = int(config.model.hidden_dim)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Vocabularies come from the checkpoint when it has them; the config's
        # prepared_root is a path from the training machine, which may be absent
        # here or may now hold a rebuilt corpus whose vocabulary is a different
        # mapping of the same size -- that loads without error and mislabels
        # every prediction.
        embedded_node_vocab = ckpt.get("node_vocab") if isinstance(ckpt, dict) else None
        embedded_tactic_vocab = ckpt.get("tactic_vocab") if isinstance(ckpt, dict) else None
        if isinstance(embedded_node_vocab, dict) and isinstance(embedded_tactic_vocab, dict):
            metadata = PreparedMetadata.from_vocabs(
                node_vocab={str(k): int(v) for k, v in embedded_node_vocab.items()},
                tactic_vocab={str(k): int(v) for k, v in embedded_tactic_vocab.items()},
            )
        else:
            try:
                metadata = load_prepared_metadata(config.prepared_root)
            except (FileNotFoundError, ValueError) as exc:
                print(
                    f"ERROR: checkpoint '{args.checkpoint}' does not carry its "
                    "vocabularies and they could not be read from the prepared dataset "
                    f"at '{config.prepared_root}': {exc}\n"
                    "Export a bundle with scripts/export_model_bundle.py and pass "
                    "--bundle instead."
                )
                return 1

        from atp_lean_gnn.argument_selector import TacticWithArgsClassifier

        model = TacticWithArgsClassifier(
            num_node_labels=len(metadata.node_vocab),
            num_tactics=len(metadata.tactic_vocab),
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            use_node_type=config.use_node_type,
            max_args=getattr(config, "max_args", 3),
            gnn_type=config.gnn_type,
            heads=getattr(config.model, "heads", 8),
            readout=getattr(config.model, "readout", "state"),
        )

        # Decide from the weights' own key structure whether this is a baseline
        # to be wrapped or a pointer to be loaded as-is, then load strictly. The
        # previous version prefixed whatever did not look like a pointer key and
        # loaded with strict=False, which turned an architecture mismatch, a
        # renaming bug, and an intended partial transfer into the same silent
        # success.
        try:
            if detect_state_dict_model_type(state_dict) == "baseline":
                randomly_initialized = load_baseline_weights_into_pointer(model, state_dict)
                if randomly_initialized:
                    print(
                        f"WARNING: '{args.checkpoint}' is a baseline checkpoint; "
                        f"{', '.join(randomly_initialized)} is randomly initialized, so "
                        "argument selection is not a prediction."
                    )
            else:
                load_state_dict_checked(model, state_dict)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
        model = model.to(device)

    model.eval()

    # Build scorer and load trained weights
    if scorer is None:
        if args.scorer_checkpoint:
            scorer_ckpt_path = Path(args.scorer_checkpoint)
            if scorer_ckpt_path.exists():
                scorer_ckpt = torch.load(scorer_ckpt_path, map_location=device, weights_only=False)
                # Checkpoints from train_scorer save under 'scorer_state_dict'
                scorer_state = scorer_ckpt.get("scorer_state_dict", scorer_ckpt)
                # The scoring mode is determined by the weights: only mode="mlp"
                # creates a "scorer" submodule. Reading it beats taking it as a
                # flag, which can disagree with the file it is describing.
                scorer_mode = (
                    "mlp"
                    if any(str(key).startswith("scorer.") for key in scorer_state)
                    else "dot"
                )
                scorer = PremiseScorer(hidden_dim=hidden_dim, mode=scorer_mode)
                try:
                    load_state_dict_checked(scorer, scorer_state)
                except ValueError as exc:
                    print(f"ERROR: {exc}")
                    return 1
                scorer = scorer.to(device)
                print(f"Loaded trained {scorer_mode} scorer weights from {scorer_ckpt_path}")
            else:
                print(f"WARNING: scorer checkpoint not found at {scorer_ckpt_path} — using random weights.")
                scorer = PremiseScorer(hidden_dim=hidden_dim, mode="dot").to(device)
        else:
            print("WARNING: No --scorer-checkpoint provided. PremiseScorer is using RANDOM weights.")
            scorer = PremiseScorer(hidden_dim=hidden_dim, mode="dot").to(device)
    scorer.eval()
    # Load index if provided
    lemma_index = None
    if args.index_path:
        index_path = Path(args.index_path)
        if index_path.exists():
            from atp_lean_gnn.lemma_index import (
                read_index_manifest,
                load_index_for_encoder,
            )

            manifest = read_index_manifest(index_path)
            if manifest.get("encoder_state_sha256"):
                # Verify the index was built by this model's encoder before
                # retrieval runs: same dimensions but a different vector
                # space returns plausible wrong premises with no error.
                try:
                    lemma_index = load_index_for_encoder(
                        index_path,
                        encoder_state_dict=model.state_dict(),
                        node_vocab=metadata.node_vocab,
                        tactic_vocab=metadata.tactic_vocab,
                    )
                except ValueError as exc:
                    print(f"ERROR: {exc}")
                    return 1
            else:
                # Pre-binding index: load, but say it is unverified.
                print(
                    "WARNING: index predates encoder binding (no "
                    "encoder_state_sha256 in its manifest); loading it "
                    "unverified. Rebuild it with build_lemma_index.py."
                )
                lemma_index = LemmaIndex.load(index_path)
            print(f"Loaded index with {len(lemma_index.lemma_ids)} lemmas.")
        else:
            print(f"WARNING: index path {index_path} not found.")
            
    if lemma_index is None:
        # Create an empty index as fallback
        import faiss
        import numpy as np
        d = hidden_dim
        lemma_index = LemmaIndex(
            index=faiss.IndexFlatL2(d),
            lemma_ids=[],
            lemma_vectors=np.empty((0, d), dtype=np.float32)
        )

    lemma_corpus = None
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        if corpus_path.exists():
            records = load_lemma_corpus(corpus_path)
            lemma_corpus = {record.lemma_id: record for record in records}
            print(f"Loaded corpus with {len(lemma_corpus)} lemmas.")
        else:
            print(f"WARNING: corpus path {corpus_path} not found.")

    # Initialize Pipeline
    pipeline = InferencePipeline(
        model=model,
        scorer=scorer,
        lemma_index=lemma_index,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        device=device,
        k=args.k,
        lemma_corpus=lemma_corpus,
    )

    print("\n--- Input State ---")
    print(args.state)
    print("-------------------\n")

    prediction = pipeline.predict_tactic(args.state)
    
    print(f"Predicted Tactic:  \033[1;32m{prediction}\033[0m")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
