"""Inference pipeline for end-to-end tactic prediction.

This module provides the ``InferencePipeline`` which integrates graph conversion,
tactic prediction, premise retrieval, and candidate scoring to produce a final
tactic string.
"""

from __future__ import annotations

import re
import torch
from torch_geometric.data import Batch

from .argument_selector import TacticWithArgsClassifier
from .graph import DAGBuilder, GraphNode, proof_state_to_dag, goal_state_to_proof_state
from .lemma_corpus import LemmaRecord
from .lemma_index import LemmaIndex
from .premise_pool import build_unified_pools
from .premise_scoring import PremiseScorer
from .pyg import build_premise_mask, dag_to_pyg
from .state import ProofState, parse_state
from .training import transform_edge_index

# ── Tactic-aware argument filtering rules ──────────────────────────────────
# Fresh name only: generate a new identifier, reject all candidates
_FRESH_NAME_TACTICS = frozenset({"intro", "rintro", "introV2"})
# Local only: accept only local context nodes, reject library lemmas
_LOCAL_ONLY_TACTICS = frozenset({"cases", "rcases", "rcases_pattern", "obtain"})
# All non-fresh tactics decode against the unified pool; the stop head handles
# actions whose learned sequence length is zero.


_FV_LABEL_RE = re.compile(r"^FV(\d+)$")


def _local_names_by_fv_label(dag: DAGBuilder) -> dict[str, str]:
    """Map each ``FV{i}`` pointer label to the Lean name of the local it denotes.

    Structured hypothesis nodes are ``Hyp(FV{i}, name, HypRole:role, type)``, so
    the graph already carries the correspondence and no side table has to be
    threaded in from the caller.  Legacy two-child ``Hyp(name, type)`` nodes have
    no ``FV{i}`` child and contribute nothing.
    """
    names: dict[str, str] = {}
    for node in dag.nodes:
        if node.label != "Hyp" or len(node.children) != 4:
            continue
        context_label = dag.nodes[node.children[0]].label
        if _FV_LABEL_RE.match(context_label):
            names[context_label] = dag.nodes[node.children[1]].label
    return names


def _resolve_local_node_name(
    node: GraphNode,
    dag: DAGBuilder,
    local_names: dict[str, str] | None = None,
) -> str:
    """Extract a Lean-usable name for a selected local-context node.

    Both shapes of hypothesis node appear, and the name sits at a different
    child index in each: structured nodes are ``Hyp(FV{i}, name, HypRole:role,
    type)`` and legacy ones are ``Hyp(name, type)``.  Reading ``children[0]``
    unconditionally returns ``"FV3"`` for a structured node, which is not Lean.

    The pointer head can also select a bare ``FV{i}`` node directly, since those
    are premise-selectable in their own right.  ``local_names`` — from
    ``_local_names_by_fv_label`` — resolves those back to a name.

    A name the model cannot use still comes back as-is: an anonymous local is
    ``"_"``, and a local whose inaccessible marker was sanitized upstream may be
    ``"p✝"``.  Neither is valid in a tactic, and neither is filtered here,
    because dropping arguments silently would change the arity a caller gets
    back.
    """
    if node.label == "Hyp" and node.children:
        name_index = 1 if len(node.children) == 4 else 0
        return dag.nodes[node.children[name_index]].label
    if local_names and _FV_LABEL_RE.match(node.label):
        return local_names.get(node.label, node.label)
    return node.label


def _extract_fresh_names_from_dag(dag: DAGBuilder) -> list[str]:
    """Walk the DAG and collect fresh variable names from ∀-bound leaf nodes.

    Only returns variables at binder_depth == 1 (the outermost forall),
    excluding variables nested inside type annotations like ``Set α``.
    """
    from .graph import BINDER_KIND_FORALL
    return [
        node.label for node in dag.nodes
        if node.is_bound == 1 and node.binder_kind == BINDER_KIND_FORALL
        and node.binder_depth == 1
        and not node.children
    ]


def _top_tactic_candidates(
    tactic_probs: torch.Tensor,
    id_to_tactic: dict[int, str],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Return the top-k tactic candidates sorted by probability."""
    if top_k <= 0:
        return []

    top_k = min(int(top_k), int(tactic_probs.size(-1)))
    if top_k <= 0:
        return []

    topk = torch.topk(tactic_probs, k=top_k, dim=-1)
    candidates: list[dict[str, object]] = []
    for tactic_id, probability in zip(topk.indices.tolist(), topk.values.tolist(), strict=False):
        candidates.append(
            {
                "tactic_id": int(tactic_id),
                "tactic_name": id_to_tactic.get(int(tactic_id), "<UNK>"),
                "probability": round(float(probability), 6),
            }
        )
    return candidates


class ArgumentPrediction:
    """Details for a single selected argument."""

    def __init__(
        self,
        source: str,
        candidate_id: int,
        label: str,
        score: float,
    ) -> None:
        self.source = source
        self.candidate_id = candidate_id
        self.label = label
        self.score = score

    def __repr__(self) -> str:
        return f"ArgumentPrediction(source={self.source!r}, candidate_id={self.candidate_id}, label={self.label!r}, score={self.score:.4f})"


class InferenceResult:
    """Structured inference result for tactic and argument prediction."""

    def __init__(
        self,
        predicted_tactic: str,
        tactic_name: str,
        tactic_id: int,
        tactic_probabilities: list[tuple[str, float]],
        selected_arguments: list[str],
        selected_argument_details: list[ArgumentPrediction],
        *,
        top_tactic_predictions: list[dict[str, object]] | None = None,
    ) -> None:
        self.predicted_tactic = predicted_tactic
        self.tactic_name = tactic_name
        self.tactic_id = tactic_id
        self.tactic_probabilities = tactic_probabilities
        self.selected_arguments = selected_arguments
        self.selected_argument_details = selected_argument_details
        self.top_tactic_predictions = top_tactic_predictions or []


class InferencePipeline:
    """End-to-end tactic prediction pipeline."""

    def __init__(
        self,
        model: TacticWithArgsClassifier,
        scorer: PremiseScorer,
        lemma_index: LemmaIndex,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        device: torch.device,
        k: int = 500,
        lemma_corpus: dict[int, LemmaRecord] | None = None,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.lemma_index = lemma_index
        self.node_vocab = node_vocab
        self.tactic_vocab = tactic_vocab
        self.device = device
        self.k = k
        self.lemma_corpus = lemma_corpus

        # Invert tactic vocab for decoding
        self.id_to_tactic = {idx: name for name, idx in tactic_vocab.items()}

        self.model.eval()
        self.scorer.eval()

    @torch.no_grad()
    def predict_tactic(self, state_str: str) -> str:
        """Predict a full tactic string given a Lean proof state."""
        return self.predict_tactic_result(state_str).predicted_tactic

    @torch.no_grad()
    def predict_tactic_result(self, state_str: str, *, top_k: int = 1) -> InferenceResult:
        """Predict tactics and return detailed inference information for the top-k candidates."""
        state = parse_state(state_str)
        
        # 1. Graph construction
        dag = proof_state_to_dag(state)
        return self._predict_from_dag(dag, top_k=top_k)

    @torch.no_grad()
    def predict_from_goal_state(self, goal_state, *, top_k: int = 1) -> InferenceResult:
        """Predict tactics from a Pantograph ``GoalState``.

        The DAG is built from the goal's and hypotheses' model S-expressions
        rather than from parsed text, which is what the model was trained on.
        Requires ``patch_pantograph_for_sexp()`` to have been called before the
        state was parsed, and the server to have been created with
        ``MODEL_SEXPR_SERVER_OPTIONS``.

        ``goal_state`` must come from a tactic application; see
        ``goal_state_to_proof_state`` for why a state from ``goal_start`` does
        not carry the fields this needs.
        """
        text_state, hyp_sexps, goal_sexp = goal_state_to_proof_state(goal_state)
        dag = proof_state_to_dag(text_state, goal_sexp=goal_sexp, hyp_sexps=hyp_sexps)
        return self._predict_from_dag(dag, top_k=top_k)

    def _predict_from_dag(self, dag: DAGBuilder, *, top_k: int = 1) -> InferenceResult:
        """Core prediction logic from a pre-built DAG."""
        data = dag_to_pyg(dag, self.node_vocab)
        
        try:
            state_idx = next(i for i, n in enumerate(dag.nodes) if n.label == "State")
        except StopIteration:
            state_idx = 0
        data.state_node_index = torch.tensor([state_idx], dtype=torch.long)
        
        premise_mask = build_premise_mask(dag)
        data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)
        
        data = data.to(self.device)
        data.state_node_index = data.state_node_index.to(self.device)
        data.premise_mask = data.premise_mask.to(self.device)
        # Apply bidirectional edges to match training edge_mode
        data.edge_index = transform_edge_index(data.edge_index, edge_mode="bidirectional")
        batch = Batch.from_data_list([data])

        node_embeddings = self.model.backbone.encode_nodes(batch)
        state_emb = self.model.backbone.readout(node_embeddings, batch)
        
        tactic_logits = self.model.backbone.classifier(state_emb)
        tactic_probs = torch.softmax(tactic_logits.squeeze(0), dim=-1)
        top_candidates = _top_tactic_candidates(tactic_probs, self.id_to_tactic, top_k=top_k)

        tactic_distribution = [
            (item["tactic_name"], float(item["probability"]))
            for item in top_candidates
        ]

        pools = build_unified_pools(
            state_emb,
            node_embeddings,
            batch.premise_mask,
            batch.batch,
            lemma_index=self.lemma_index,
            k=self.k,
        )
        pool = pools[0]
        # Built once: the pointer head may select the same `FV{i}` node for
        # several tactic candidates, and the walk is over every node in the DAG.
        local_names = _local_names_by_fv_label(dag)

        top_tactic_predictions: list[dict[str, object]] = []
        for candidate in top_candidates:
            tactic_id = int(candidate["tactic_id"])
            tactic_name = str(candidate["tactic_name"])
            tactic_id_tensor = torch.tensor([tactic_id], dtype=torch.long, device=self.device)
            tactic_emb = self.model.tactic_embedding(tactic_id_tensor)

            if not pool.candidate_ids:
                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": [],
                        "selected_argument_details": [],
                    }
                )
                continue

            # ── Tactic-aware argument filtering ──────────────────────────
            if tactic_name in _FRESH_NAME_TACTICS:
                fresh_names = _extract_fresh_names_from_dag(dag)
                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": fresh_names,
                        "selected_argument_details": [
                            ArgumentPrediction(source="fresh", candidate_id=0, label=name, score=0.0)
                            for name in fresh_names
                        ],
                    }
                )
                continue

            candidate_mask = torch.tensor(
                [src == "local" for src in pool.candidate_sources],
                device=self.device,
                dtype=torch.bool,
            ) if tactic_name in _LOCAL_ONLY_TACTICS else torch.ones(
                len(pool.candidate_ids), device=self.device, dtype=torch.bool
            )
            candidate_indices = torch.where(candidate_mask)[0]
            if candidate_indices.numel() == 0:
                top_tactic_predictions.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "probability": float(candidate["probability"]),
                        "selected_arguments": [],
                        "selected_argument_details": [],
                    }
                )
                continue
            candidate_vectors = pool.candidate_vectors[candidate_indices]
            decoder_state = self.model.argument_selector.initial_state(state_emb, tactic_emb)
            selected_positions = torch.zeros(
                candidate_vectors.size(0), dtype=torch.bool, device=self.device
            )
            top_indices: list[int] = []
            selected_scores: list[float] = []
            for step in range(self.model.max_args + 1):
                if float(self.model.stop_head(decoder_state).item()) >= 0:
                    break
                if step == self.model.max_args:
                    break
                scores = self.model.argument_selector.score_candidates(
                    decoder_state, candidate_vectors
                )
                scores = scores.masked_fill(selected_positions, float("-inf"))
                selected_position = int(scores.argmax().item())
                selected_score = float(scores[selected_position].item())
                selected_positions[selected_position] = True
                top_indices.append(int(candidate_indices[selected_position].item()))
                selected_scores.append(selected_score)
                selected_embedding = candidate_vectors[selected_position].unsqueeze(0)
                decoder_state = self.model.argument_selector.gru(
                    selected_embedding, decoder_state
                )

            arguments = []
            selected_argument_details = []
            for idx, score_value in zip(top_indices, selected_scores):
                source = pool.candidate_sources[idx]
                cid = pool.candidate_ids[idx]

                if source == "local":
                    node = dag.nodes[cid]
                    arg_str = _resolve_local_node_name(node, dag, local_names)
                else:
                    if self.lemma_corpus and cid in self.lemma_corpus:
                        arg_str = self.lemma_corpus[cid].name
                    else:
                        arg_str = f"<lemma_{cid}>"

                arguments.append(arg_str)
                selected_argument_details.append(
                    ArgumentPrediction(
                        source=source,
                        candidate_id=cid,
                        label=arg_str,
                        score=score_value,
                    )
                )

            top_tactic_predictions.append(
                {
                    "tactic_id": tactic_id,
                    "tactic_name": tactic_name,
                    "probability": float(candidate["probability"]),
                    "selected_arguments": arguments,
                    "selected_argument_details": selected_argument_details,
                }
            )

        top1 = top_tactic_predictions[0] if top_tactic_predictions else None
        predicted_tactic = str(top1["tactic_name"]) if top1 else "<UNK>"
        if top1 and top1["selected_arguments"]:
            predicted_tactic = f"{predicted_tactic} {' '.join(str(item) for item in top1['selected_arguments'])}"

        return InferenceResult(
            predicted_tactic=predicted_tactic,
            tactic_name=str(top1["tactic_name"]) if top1 else "<UNK>",
            tactic_id=int(top1["tactic_id"]) if top1 else -1,
            tactic_probabilities=tactic_distribution,
            selected_arguments=list(top1["selected_arguments"]) if top1 else [],
            selected_argument_details=list(top1["selected_argument_details"]) if top1 else [],
            top_tactic_predictions=top_tactic_predictions,
        )
