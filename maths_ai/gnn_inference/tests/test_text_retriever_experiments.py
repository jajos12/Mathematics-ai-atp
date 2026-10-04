"""Training signal and row provenance for the text retrieval candidate."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.gnn_inference.atp_lean_gnn.text_retriever import (
    TextDualEncoder, goal_first_text, matched_text_rows, tokenize_texts,
)
from maths_ai.gnn_inference.scripts import train_text_retriever
from maths_ai.gnn_inference.scripts.train_text_retriever import train_text_epoch


class _ToyHFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)

    def forward(self, *, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()


def _tokenizer(texts, **kwargs):
    lengths = [min(len(text), kwargs["max_length"]) for text in texts]
    width = max(lengths)
    ids = torch.zeros(len(texts), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, text in enumerate(texts):
        ids[row, :lengths[row]] = torch.tensor([ord(char) % 63 + 1 for char in text[:width]])
        mask[row, :lengths[row]] = 1
    return {"input_ids": ids, "attention_mask": mask}


def test_text_towers_both_receive_retrieval_gradients() -> None:
    model = TextDualEncoder(_ToyHFEncoder())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    records = {
        record.lemma_id: record for record in (
            LemmaRecord(10, "Nat.add_comm", "x + y = y + x", "Nat", ""),
            LemmaRecord(11, "Nat.zero_add", "0 + x = x", "Nat", ""),
            LemmaRecord(12, "Nat.add_zero", "x + 0 = x", "Nat", ""),
        )
    }
    metrics = train_text_epoch(
        model, _tokenizer,
        [("goal x + y", [10]), ("goal 0 + x", [11, 12])],
        records, None, optimizer=optimizer, device=torch.device("cpu"),
        batch_size=2, max_length=32, random_negatives=1, hard_negatives=0,
        max_batches=1, temperature=0.05, seed=42,
    )
    assert metrics["batches"] == 1
    assert metrics["labeled_examples"] == 2
    assert model.state_encoder.embedding.weight.grad is not None
    assert model.lemma_encoder.embedding.weight.grad is not None


def test_text_pairs_use_prepared_external_labels_and_source_row_identity() -> None:
    data = Data(
        y=torch.tensor([1]),
        arg_lemma_ids=torch.tensor([10, -1]),
        arg_node_indices=torch.tensor([-1, 0]),
    )
    data.arg_count = 2
    data.row_index = 7
    data.theorem = "Nat.demo"
    data.tactic_raw = "rw [Nat.add_comm, h]"
    data.dataset_name = "dataset"
    data.split = "val"
    upstream = DatasetRow(
        state="h : Nat\n⊢ x + y = y + x", theorem="Nat.demo",
        tactic="rw [Nat.add_comm, h]", split="val", row_index=7,
        dataset_name="dataset",
    )
    assert matched_text_rows([data], [upstream], name_to_id={"Nat.add_comm": 10}) == [
        ("Goal: x + y = y + x\nContext: h : Nat", [10])
    ]
    assert goal_first_text("plain text") == "plain text"
    wrong = DatasetRow(
        state=upstream.state, theorem=upstream.theorem, tactic="exact h",
        split="val", row_index=7, dataset_name="dataset",
    )
    with pytest.raises(ValueError, match="differs"):
        matched_text_rows([data], [wrong], name_to_id={"Nat.add_comm": 10})


def test_interrupted_text_epoch_resumes_without_replaying_optimizer_steps() -> None:
    torch.manual_seed(4)
    initial = TextDualEncoder(_ToyHFEncoder())
    records = {
        i: LemmaRecord(i, f"lemma{i}", f"statement{i}", "", "")
        for i in (10, 11, 12)
    }
    pairs = [("first goal", [10]), ("second goal", [11])]

    def run(model, optimizer, *, max_batches, start_batch=0, rng_state=None, progress=None):
        return train_text_epoch(
            model, _tokenizer, pairs, records, None,
            optimizer=optimizer, device=torch.device("cpu"),
            batch_size=1, max_length=32, random_negatives=1,
            hard_negatives=0, max_batches=max_batches,
            temperature=0.05, seed=9, start_batch=start_batch,
            rng_state=rng_state, progress=progress, progress_every=1,
        )

    uninterrupted = copy.deepcopy(initial)
    full_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)
    run(uninterrupted, full_optimizer, max_batches=2)

    paused = copy.deepcopy(initial)
    paused_optimizer = torch.optim.AdamW(paused.parameters(), lr=1e-3)
    progress = []
    run(
        paused, paused_optimizer, max_batches=1,
        progress=lambda batch, rng: progress.append((batch, rng.getstate())),
    )
    resumed = copy.deepcopy(initial)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed.load_state_dict(paused.state_dict())
    resumed_optimizer.load_state_dict(paused_optimizer.state_dict())
    assert progress[0][0] == 1
    run(resumed, resumed_optimizer, max_batches=2, start_batch=1, rng_state=progress[0][1])
    for key, full_weight in uninterrupted.state_dict().items():
        assert torch.allclose(full_weight, resumed.state_dict()[key], atol=1e-6)


def test_real_transformer_towers_load_from_local_pretrained_files(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    (tmp_path / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nnat\nadd\ncomm\n",
        encoding="utf-8",
    )
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    tokenizer.save_pretrained(tmp_path)
    backbone = transformers.BertModel(transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2,
    ))
    # Avoid save_pretrained's optional accelerate/deepspeed integration: this
    # local test needs only ordinary model config and state tensors.
    backbone.config.to_json_file(str(tmp_path / "config.json"))
    torch.save(backbone.state_dict(), tmp_path / "pytorch_model.bin")
    model = TextDualEncoder.from_pretrained(str(tmp_path))
    goal = model.encode_states(tokenize_texts(
        tokenizer, ["nat add"], device=torch.device("cpu"), max_length=32,
    ))
    lemma = model.encode_lemmas(tokenize_texts(
        tokenizer, ["nat add comm"], device=torch.device("cpu"), max_length=32,
    ))
    assert torch.allclose(goal.norm(dim=-1), torch.ones(1), atol=1e-5)
    (goal * lemma).sum().backward()
    assert model.state_encoder.embeddings.word_embeddings.weight.grad is not None
    assert model.lemma_encoder.embeddings.word_embeddings.weight.grad is not None


def test_cached_text_pairs_refuse_changed_prepared_data(tmp_path, monkeypatch) -> None:
    root = tmp_path / "prepared"
    packed = root / "packed" / "bidirectional"
    packed.mkdir(parents=True)
    (packed / "manifest.json").write_text("{}", encoding="utf-8")
    def graph(split):
        data = Data(
            y=torch.tensor([1]), arg_lemma_ids=torch.tensor([10]),
            arg_node_indices=torch.tensor([-1]),
        )
        data.arg_count = 1
        data.row_index = 0
        data.theorem = f"Nat.{split}"
        data.tactic_raw = "exact Nat.add_comm"
        data.dataset_name = "dataset"
        data.split = split
        return data

    class FakePrepared(list):
        packed_cache_loaded = True

        def __init__(self, metadata, *, split, **kwargs):
            super().__init__([graph(split)])

    monkeypatch.setattr(train_text_retriever, "load_prepared_metadata", lambda path: SimpleNamespace())
    monkeypatch.setattr(train_text_retriever, "PreparedGraphDataset", FakePrepared)
    monkeypatch.setattr(train_text_retriever, "iter_dataset_rows", lambda *, dataset_name, split: iter([
        DatasetRow(
            state=f"⊢ Nat.{split}", theorem=f"Nat.{split}",
            tactic="exact Nat.add_comm", split=split, row_index=0,
            dataset_name=dataset_name,
        )
    ]))
    run = tmp_path / "run"
    run.mkdir()
    expected = train_text_retriever._pairs(
        root, "dataset", {"Nat.add_comm": 10}, run, resume=False, corpus_sha="corpus"
    )
    assert expected[1] == [("Goal: Nat.val", [10])]
    assert train_text_retriever._pairs(
        root, "dataset", {"Nat.add_comm": 10}, run, resume=True, corpus_sha="corpus"
    ) == expected
    (packed / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="cached text pairs"):
        train_text_retriever._pairs(
            root, "dataset", {"Nat.add_comm": 10}, run, resume=True, corpus_sha="corpus"
        )
