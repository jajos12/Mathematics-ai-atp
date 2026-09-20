from pathlib import Path
from typing import List, Optional, cast

import torch

from maths_ai.data_models.proof_components import StableArgumentRef, TacticCandidate
from maths_ai.gnn_inference.atp_lean_gnn.bundle import (
    load_baseline_weights_into_pointer,
    load_pointer_bundle,
    load_state_dict_checked,
)
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import load_retriever_checkpoint
from maths_ai.gnn_inference.atp_lean_gnn.unified_reranking import (
    load_unified_reranker_weights,
    verify_unified_reranker_dependencies,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PreparedMetadata,
    detect_state_dict_model_type,
    load_baseline_config,
    load_prepared_metadata,
)

from .model import GNNPredictor


class GNNModelEngine:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        tactic_predictor_model_path: Optional[Path] = None,
        argument_predictor_model_path: Optional[Path] = None,
        *,
        bundle_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
        retriever_checkpoint_path: Optional[Path] = None,
        scorer_mode: Optional[str] = None,
        k: int = 500,
        device: str = "cuda",
    ):
        """Load the tactic model and premise scorer for serving.

        Prefer ``bundle_dir``: a bundle carries the vocabularies the weights were
        trained against and verifies them by hash, so it loads correctly with no
        prepared dataset on the machine.  The vocabularies are not incidental
        metadata -- node IDs index the label embedding and tactic IDs index the
        classifier, and both are assigned by sorted position, so serving weights
        against a vocabulary rebuilt from a different set of rows produces a
        model that loads cleanly and names the wrong tactic for every goal.

        Args:
            config_path: legacy path to a training ``config.json``. Used only
                when ``bundle_dir`` is not given.
            tactic_predictor_model_path: legacy checkpoint (.pt) holding
                ``model_state_dict`` for the trained model.
            argument_predictor_model_path: legacy checkpoint (.pt) holding
                ``scorer_state_dict`` for the trained ``PremiseScorer``.
            bundle_dir: an exported model bundle, as written by
                ``scripts/export_model_bundle.py``.
            index_path: optional path to a FAISS lemma index directory.
            corpus_path: optional path to lemmas.jsonl for decoding retrieved
                lemma IDs.
            retriever_checkpoint_path: optional trained proof-state query tower.
                Its frozen lemma tower must match ``index_path``.
            scorer_mode: legacy override for the scorer's scoring mode. Left
                unset it is recovered from the scorer's own weights, which is
                always correct; passing it wrong silently changes the
                architecture the weights are loaded into.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if bundle_dir is not None:
            tactic_model, argument_model, metadata, hidden_dim, edge_mode = self._load_from_bundle(
                Path(bundle_dir),
                argument_predictor_model_path=argument_predictor_model_path,
                scorer_mode=scorer_mode,
            )
        else:
            if config_path is None or tactic_predictor_model_path is None:
                raise ValueError(
                    "GNNModelEngine needs either bundle_dir, or both config_path and "
                    "tactic_predictor_model_path."
                )
            tactic_model, argument_model, metadata, hidden_dim, edge_mode = self._load_from_checkpoints(
                config_path=Path(config_path),
                tactic_predictor_model_path=Path(tactic_predictor_model_path),
                argument_predictor_model_path=(
                    None
                    if argument_predictor_model_path is None
                    else Path(argument_predictor_model_path)
                ),
                scorer_mode=scorer_mode,
            )

        reranker_checkpoint = None
        if argument_predictor_model_path is not None:
            reranker_checkpoint = load_unified_reranker_weights(
                argument_predictor_model_path,
                model=tactic_model,
                node_vocab=metadata.node_vocab,
                tactic_vocab=metadata.tactic_vocab,
                expected_edge_mode=edge_mode,
                device=self.device,
            )

        retriever = (
            None
            if retriever_checkpoint_path is None
            else load_retriever_checkpoint(
                retriever_checkpoint_path,
                pointer_backbone=tactic_model.backbone,
                hidden_dim=hidden_dim,
                node_vocab=metadata.node_vocab,
                tactic_vocab=metadata.tactic_vocab,
                device=self.device,
            )
        )
        lemma_index = self._load_lemma_index(
            index_path,
            hidden_dim,
            encoder_state_dict=(
                tactic_model.state_dict()
                if retriever is None
                else retriever.lemma_encoder.state_dict()
            ),
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
            corpus_path=corpus_path,
            expected_edge_mode=edge_mode,
        )
        lemma_corpus = self._load_lemma_corpus(corpus_path)
        verify_unified_reranker_dependencies(
            reranker_checkpoint,
            index_manifest=lemma_index.manifest,
            retriever=retriever,
        )

        self.gnn_inference = GNNPredictor(
            tactic_model=tactic_model,
            argument_model=argument_model,
            lemma_index=lemma_index,
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
            device=self.device,
            k=k,
            lemma_corpus=lemma_corpus,
            retriever=retriever,
            edge_mode=edge_mode,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_from_bundle(
        self,
        bundle_dir: Path,
        *,
        argument_predictor_model_path: Optional[Path],
        scorer_mode: Optional[str],
    ) -> tuple[TacticWithArgsClassifier, PremiseScorer, PreparedMetadata, int, str]:
        # load_pointer_bundle wraps a baseline bundle into a pointer shell,
        # because InferencePipeline reaches through model.backbone and so cannot
        # accept a bare baseline.
        loaded = load_pointer_bundle(bundle_dir, device=self.device)
        tactic_model = cast(TacticWithArgsClassifier, loaded.model)
        hidden_dim = int(loaded.config.model.hidden_dim)

        if loaded.randomly_initialized:
            print(
                "WARNING: bundle "
                f"{bundle_dir} has no trained argument head; "
                f"{', '.join(loaded.randomly_initialized)} is randomly initialized, so "
                "argument selection is not a prediction."
            )

        if loaded.scorer is not None:
            argument_model = loaded.scorer
        elif argument_predictor_model_path is not None:
            argument_model = self._load_scorer_checkpoint(
                Path(argument_predictor_model_path),
                hidden_dim=hidden_dim,
                scorer_mode=scorer_mode,
            )
        else:
            # An untrained scorer scores every premise identically, so premise
            # ranking degenerates to the retrieval order rather than being wrong
            # in a way that looks trained.
            print(
                f"WARNING: bundle {bundle_dir} carries no premise scorer and none was "
                "supplied; premise scoring is untrained."
            )
            argument_model = PremiseScorer(
                hidden_dim=hidden_dim, mode=scorer_mode or "dot"
            ).to(self.device)
            argument_model.eval()

        return tactic_model, argument_model, loaded.metadata, hidden_dim, loaded.config.edge_mode

    def _load_from_checkpoints(
        self,
        *,
        config_path: Path,
        tactic_predictor_model_path: Path,
        argument_predictor_model_path: Optional[Path],
        scorer_mode: Optional[str],
    ) -> tuple[TacticWithArgsClassifier, PremiseScorer, PreparedMetadata, int, str]:
        config = load_baseline_config(config_path)
        tactic_checkpoint = torch.load(
            tactic_predictor_model_path, map_location=self.device, weights_only=False
        )
        state_dict = (
            tactic_checkpoint.get("model_state_dict", tactic_checkpoint)
            if isinstance(tactic_checkpoint, dict)
            else tactic_checkpoint
        )

        # Take the vocabularies from the checkpoint when it has them. The
        # config's prepared_root is a path recorded on the training machine: it
        # is usually absent here, and when a path of that name does exist it may
        # hold a rebuilt corpus whose vocabulary is a different mapping of the
        # same size, which loads without complaint and mislabels everything.
        embedded_node_vocab = (
            tactic_checkpoint.get("node_vocab")
            if isinstance(tactic_checkpoint, dict)
            else None
        )
        embedded_tactic_vocab = (
            tactic_checkpoint.get("tactic_vocab")
            if isinstance(tactic_checkpoint, dict)
            else None
        )
        if isinstance(embedded_node_vocab, dict) and isinstance(embedded_tactic_vocab, dict):
            metadata = PreparedMetadata.from_vocabs(
                node_vocab={str(k): int(v) for k, v in embedded_node_vocab.items()},
                tactic_vocab={str(k): int(v) for k, v in embedded_tactic_vocab.items()},
            )
        else:
            try:
                metadata = load_prepared_metadata(config.prepared_root)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"Checkpoint '{tactic_predictor_model_path}' does not carry its "
                    "vocabularies and they could not be read from the prepared dataset "
                    f"at '{config.prepared_root}': {exc}. Export a model bundle with "
                    "scripts/export_model_bundle.py and pass bundle_dir instead -- "
                    "weights without their vocabularies cannot be served correctly."
                ) from exc

        tactic_model = TacticWithArgsClassifier(
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
        # A baseline checkpoint has to be wrapped rather than loaded directly:
        # InferencePipeline reaches through model.backbone, so it needs a pointer
        # model even when only a baseline was trained. Deciding this from the
        # weights' own key structure -- rather than remapping whatever fails to
        # match -- is what keeps the load itself strict.
        if detect_state_dict_model_type(state_dict) == "baseline":
            randomly_initialized = load_baseline_weights_into_pointer(
                tactic_model, state_dict
            )
            if randomly_initialized:
                print(
                    f"WARNING: '{tactic_predictor_model_path}' is a baseline checkpoint; "
                    f"{', '.join(randomly_initialized)} is randomly initialized, so "
                    "argument selection is not a prediction."
                )
        else:
            load_state_dict_checked(tactic_model, state_dict)
        tactic_model = tactic_model.to(self.device)
        tactic_model.eval()

        hidden_dim = int(config.model.hidden_dim)
        if argument_predictor_model_path is not None:
            argument_model = self._load_scorer_checkpoint(
                argument_predictor_model_path,
                hidden_dim=hidden_dim,
                scorer_mode=scorer_mode,
            )
        else:
            argument_model = PremiseScorer(
                hidden_dim=hidden_dim, mode=scorer_mode or "dot"
            ).to(self.device)
            argument_model.eval()

        return tactic_model, argument_model, metadata, hidden_dim, config.edge_mode

    def _load_scorer_checkpoint(
        self,
        path: Path,
        *,
        hidden_dim: int,
        scorer_mode: Optional[str],
    ) -> PremiseScorer:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint.get("scorer_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        # The scoring mode is determined by the weights: only mode="mlp" creates
        # a "scorer" submodule. Reading it beats accepting it as an argument,
        # since a wrong argument builds a different architecture and then fails
        # (or, with a lenient load, silently drops the head).
        detected_mode = (
            "mlp" if any(str(key).startswith("scorer.") for key in state_dict) else "dot"
        )
        if scorer_mode is not None and scorer_mode != detected_mode:
            raise ValueError(
                f"scorer_mode='{scorer_mode}' was requested but the weights in '{path}' "
                f"are a '{detected_mode}' scorer."
            )
        scorer = PremiseScorer(hidden_dim=hidden_dim, mode=detected_mode)
        load_state_dict_checked(scorer, state_dict)
        scorer = scorer.to(self.device)
        scorer.eval()
        return scorer

    @staticmethod
    def _load_lemma_index(
        index_path: Optional[Path],
        hidden_dim: int,
        *,
        encoder_state_dict,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        corpus_path: Optional[Path],
        expected_edge_mode: str,
    ) -> LemmaIndex:
        if index_path is not None and Path(index_path).exists():
            from .atp_lean_gnn.lemma_index import load_index_for_encoder

            return load_index_for_encoder(
                index_path,
                encoder_state_dict=encoder_state_dict,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                corpus_path=corpus_path,
                expected_edge_mode=expected_edge_mode,
            )

        import faiss
        import numpy as np

        return LemmaIndex(
            index=faiss.IndexFlatL2(hidden_dim),
            lemma_ids=[],
            lemma_vectors=np.empty((0, hidden_dim), dtype=np.float32),
        )

    @staticmethod
    def _load_lemma_corpus(corpus_path: Optional[Path]):
        if corpus_path is None or not Path(corpus_path).exists():
            return None

        records = load_lemma_corpus(corpus_path)
        return {record.lemma_id: record for record in records}

    def inference(self, goal_expression: str, top_k: int = 3) -> List[TacticCandidate]:
        """Predict ranked tactic candidates for ``goal_expression``.

        Contract (depended on by ``HybridReasoner.predict_next_tactic``):
        return up to ``top_k`` candidates, each a ``TacticCandidate``
        carrying the tactic family name, its selected argument/premise
        names, and the model's predicted probability — sorted by
        probability, descending.
        """
        predictions = self.gnn_inference.predict_tactics_with_arguments(goal_expression, top_k=top_k)
        print(*predictions, "\n")
        result = []
        for prediction in predictions:
            details = cast(list, prediction.get("selected_argument_details", []))
            result.append(
                TacticCandidate(
                    tactic_name=str(prediction["tactic_name"]),
                    arguments=[
                        str(argument)
                        for argument in cast(list, prediction["selected_arguments"])
                    ],
                    argument_refs=[
                        StableArgumentRef(
                            source=str(detail.source),
                            candidate_id=int(detail.candidate_id),
                            graph_id=(0 if str(detail.source) in {"local", "fresh"} else None),
                        )
                        for detail in details
                    ],
                    probability=float(cast(float, prediction["probability"])),
                )
            )

        for res in result:
            if res.tactic_name == "rw":
                res.arguments = ["[" + ",".join(res.arguments) + "]"]

        return result
