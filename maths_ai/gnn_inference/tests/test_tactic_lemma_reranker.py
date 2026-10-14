"""Tactic-aware training rows and resumable gold-present ranking metrics."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.gnn_inference.atp_lean_gnn.tactic_lemma_reranker import (
    RerankRow, TacticLemmaReranker, multi_positive_loss, pair_prompt,
    rank_metrics, sample_training_candidates,
)
from maths_ai.gnn_inference.scripts import train_tactic_lemma_reranker as runner


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_training_pairs_use_corrected_citations_and_never_raw_tactic():
    graph = Data(y=torch.tensor([1]), citation_lemma_ids=torch.tensor([11, 12]))
    graph.row_index = 0
    graph.tactic_name = "rw"
    graph.tactic_raw = "rw [<a>Secret.gold</a>]"
    table = {"rows": np.asarray([0]), "ids": np.asarray([[10, 11, 12, 13]])}
    rows, stats = runner._build_rows(
        [graph], table, [("Goal: P", [999])], split="train", unknown_tactic_id=0,
    )
    assert stats["positive_pairs"] == 2
    assert rows[0].positive_ids == (11, 12)
    assert "Secret.gold" not in pair_prompt(
        rows[0].state_text, rows[0].tactic_name, "Demo.other", "Q"
    )[0]
    ids, positives = sample_training_candidates(
        table["ids"][0].tolist(), rows[0].positive_ids,
        max_positives=2, negatives=2, rng=random.Random(42),
    )
    assert {ids[i] for i in positives} == {11, 12}
    assert set(ids) == {10, 11, 12, 13}


def test_multi_positive_loss_trains_all_observed_citations():
    logits = torch.tensor([2.0, 0.0, 1.0], requires_grad=True)
    loss = multi_positive_loss(logits, [0, 2])
    loss.backward()
    # High-scoring positive may be reduced to distribute probability across
    # *both* positives; the weaker positive must receive an upward gradient.
    assert logits.grad[2] < 0
    assert logits.grad[1] > 0
    assert loss.detach().item() < multi_positive_loss(logits.detach(), [1, 2]).item()
    with pytest.raises(ValueError, match="positive"):
        multi_positive_loss(logits, [])


def test_cross_encoder_head_and_backbone_receive_gradients():
    model = TacticLemmaReranker(_TinyEncoder())
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    multi_positive_loss(model(tokens), [0]).backward()
    assert model.encoder.embedding.weight.grad is not None
    assert model.relevance.weight.grad is not None


def test_validation_ranks_share_gold_present_population_and_resume(tmp_path, monkeypatch):
    rows = [RerankRow(0, "Goal: P", "rw", (11,), 0),
            RerankRow(1, "Goal: Q", "exact", (20, 21), 1)]
    table = {"ids": np.asarray([[10, 11, 12], [20, 22, 21]])}
    corpus = {
        i: LemmaRecord(i, f"Demo.{i}", f"statement{i}", "Demo", "")
        for i in (10, 11, 12, 20, 21, 22)
    }

    def scored(model, tokenizer, row, ids, records, **kwargs):
        # Static non-gold-based model output; second gold is already ranked first.
        ranking = {10: 0.0, 11: 2.0, 12: 1.0, 20: 1.5, 22: 0.5, 21: 2.0}
        return torch.tensor([ranking[i] for i in ids])

    monkeypatch.setattr(runner, "_score_group", scored)
    report = runner._evaluate(
        SimpleNamespace(eval=lambda: None), None, rows, table, corpus,
        device=torch.device("cpu"), max_length=32, pair_batch_size=2,
        output_dir=tmp_path / "eval", model_sha="model", candidate_sha="candidates",
    )
    assert report["text_retriever"]["gold_present_states"] == 2
    assert report["text_retriever"]["any_positive_at_1"] == 0.5
    assert report["cross_encoder"]["any_positive_at_1"] == 1.0
    assert report["cross_encoder"]["known_in_pool_citations"] == 3
    monkeypatch.setattr(runner, "_score_group", lambda *a, **k: pytest.fail("must reuse finished scores"))
    assert runner._evaluate(
        SimpleNamespace(eval=lambda: None), None, rows, table, corpus,
        device=torch.device("cpu"), max_length=32, pair_batch_size=2,
        output_dir=tmp_path / "eval", model_sha="model", candidate_sha="candidates",
    ) == report
    with pytest.raises(ValueError, match="another model"):
        runner._evaluate(
            SimpleNamespace(eval=lambda: None), None, rows, table, corpus,
            device=torch.device("cpu"), max_length=32, pair_batch_size=2,
            output_dir=tmp_path / "eval", model_sha="changed", candidate_sha="candidates",
        )
