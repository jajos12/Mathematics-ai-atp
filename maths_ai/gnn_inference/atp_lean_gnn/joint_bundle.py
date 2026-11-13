"""Strict clean-root loader for the pinned accepted flat-policy release.

Runtime extracted from ea72894. Configuration construction reproduces the
pre-template PointerConfig defaults without importing offline training code.
"""

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .bundle import file_sha256
from .argument_selector import PublishedPointerClassifier
from .joint_source_decoder import JointSourceDecoder, candidate_groups_from_batch, local_type_signals
from .lemma_index import LemmaIndex, state_dict_sha256
from .pyg import MODEL_NODE_TYPE_TO_ID as NODE_TYPE_TO_ID
from .text_retriever import TextDualEncoder, goal_first_text, tokenize_texts


FORMAT = "joint_source_system_v1"
GRAPH_FORMAT = "lean-model-sexp-v2"
POLICY_SHA256 = "a16d7f63ddde721c35611dcc2c604096cbe31d17490407669e2516c72d38e2ce"
RETRIEVER_SHA256 = "48ac6ab921bd8a6a7c2c49b3fdc307b97293a9f3737f33484673141f456aa5fd"
BUNDLE_SHA256 = "ebc7620282838e7a0d26250ddbf159ec02c859700c17217d9a43db3ef4145107"
INDEX_FILES = ("faiss.index", "lemma_vectors.npy", "lemma_ids.json", "lemma_names.json", "manifest.json")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _vocab(path):
    value = _json(path)
    if (not isinstance(value, dict) or not value or any(type(index) is not int for index in value.values())
            or sorted(value.values()) != list(range(len(value)))):
        raise ValueError(f"vocabulary must have unique contiguous IDs: {path}")
    return value


def _tokenizer_hash(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _build_policy(config, node_vocab, tactic_vocab, text_dim):
    pointer, model = config["pointer_config"], config["pointer_config"].get("model", {})
    kind = str(pointer.get("gnn_type", "sage")).lower().strip()
    if kind != "gat" or model.get("readout", "state_mean_attention") != "state_mean_attention":
        raise ValueError("accepted model requires its published GAT/state-mean-attention architecture")
    policy = PublishedPointerClassifier(num_node_labels=len(node_vocab), num_tactics=len(tactic_vocab),
        num_node_types=len(NODE_TYPE_TO_ID), hidden_dim=int(model.get("hidden_dim", 128)),
        num_layers=int(model.get("num_layers", 4)), dropout=float(model.get("dropout", 0.2)),
        use_node_type=bool(pointer.get("use_node_type", True)), max_args=int(pointer.get("max_args", 3)),
        heads=int(model.get("heads", 8)))
    return JointSourceDecoder(policy, text_dim=text_dim)


def _build_retriever(config):
    from transformers import AutoConfig, AutoModel
    encoder = copy.deepcopy(config["encoder_config"])
    kind = encoder.pop("model_type")
    return TextDualEncoder(AutoModel.from_config(AutoConfig.for_model(kind, **encoder)))


def _validate_index(index, retriever, tokenizer, config, corpus_path):
    import faiss
    expected = {"model_type": "pretrained_text_dual_encoder_index_v1", "model_name": config["model_name"],
        "tokenizer_sha256": _tokenizer_hash(tokenizer), "state_text_policy": "goal_first_v1",
        "corpus_sha256": file_sha256(corpus_path),
        "lemma_encoder_sha256": state_dict_sha256(retriever.lemma_encoder.state_dict()),
        "max_length": config["max_length"], "normalize": True}
    if index.manifest != expected:
        raise ValueError("retriever/tokenizer/corpus/index manifest binding mismatch")
    size = len(index.lemma_ids)
    if (not size or len(set(index.lemma_ids)) != size or len(index.lemma_names) != size
            or len(set(index.lemma_names)) != size or index.index.ntotal != size
            or index.index.metric_type != faiss.METRIC_INNER_PRODUCT or index.lemma_vectors.dtype != np.float32
            or index.lemma_vectors.shape != (size, config["embedding_dim"])):
        raise ValueError("invalid FAISS/vector/ID/name population or metric")
    for start in range(0, size, 8192):
        vectors = index.lemma_vectors[start:start + 8192]
        if (not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
                or not np.array_equal(index.index.reconstruct_n(start, len(vectors)), vectors)):
            raise ValueError("FAISS rows differ from normalized lemma vectors")
    names = dict(zip(index.lemma_ids, index.lemma_names))
    seen, corpus_names = set(), set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            identity, name = record["lemma_id"], record["name"]
            if (type(identity) is not int or identity in seen or name in corpus_names or names.get(identity) != name
                    or not isinstance(record.get("statement"), str) or not record["statement"].strip()):
                raise ValueError("lemma corpus differs from index ID/name mapping")
            seen.add(identity)
            corpus_names.add(name)
    if seen != set(names):
        raise ValueError("lemma corpus/index population mismatch")


@dataclass
class LoadedJointSourceBundle:
    root: Path
    policy: JointSourceDecoder
    retriever: TextDualEncoder
    tokenizer: object
    lemma_index: LemmaIndex
    node_vocab: dict
    tactic_vocab: dict
    manifest: dict
    retrieval_config: dict
    inference_config: dict
    profile: str

    @torch.no_grad()
    def retrieve(self, states, *, k=None):
        k = self.inference_config["retrieval_k"] if k is None else k
        if (not isinstance(states, (list, tuple)) or not states
                or not all(isinstance(state, str) and state.strip() for state in states)):
            raise ValueError("nonempty original proof-state texts required")
        if type(k) is not int or not 1 <= k <= len(self.lemma_index.lemma_ids):
            raise ValueError("retrieval K outside corpus bounds")
        device = next(self.retriever.parameters()).device
        tokens = tokenize_texts(self.tokenizer, [goal_first_text(state) for state in states],
            device=device, max_length=self.retrieval_config["max_length"])
        return self.lemma_index.search(self.retriever.encode_states(tokens), k=k)

    @torch.no_grad()
    def predict(self, batch, states, *, graph_format, normalized_records=None, k=None):
        if graph_format != GRAPH_FORMAT or not isinstance(states, (list, tuple)):
            raise ValueError("normalized model-S-expression graphs and ordered state texts required")
        count = int(batch.num_graphs)
        if len(states) != count:
            raise ValueError("one original state text required per graph")
        if self.profile == "accepted_local_cues" and (normalized_records is None or len(normalized_records) != count):
            raise ValueError("local cue profile requires normalized records")
        batch = batch.clone().to(next(self.policy.parameters()).device)
        batch.y = torch.zeros(count, dtype=torch.long, device=batch.x.device)
        nodes = self.policy.pointer.backbone.encode_nodes(batch)
        states_encoded = self.policy.pointer.backbone.readout(nodes, batch)
        logits = self.policy.pointer.backbone.classifier(states_encoded)
        ids, vectors, scores = self.retrieve(states, k=k)
        pools = candidate_groups_from_batch(batch, nodes, ids, vectors, scores)
        id_to_tactic = {identity: name for name, identity in self.tactic_vocab.items()}
        id_to_node = {identity: name for name, identity in self.node_vocab.items()}
        result = []
        for row, pool in enumerate(pools):
            tactic_id = int(logits[row].argmax())
            tactic = id_to_tactic[tactic_id]
            if self.profile == "accepted_local_cues":
                start, end = int(batch.ptr[row]), int(batch.ptr[row + 1])
                signal = local_type_signals(pool, batch.x[start:end], id_to_node,
                    tactic_name=tactic, **normalized_records[row])
                pool = replace(pool, local_type_signal=signal, local_type_tactic=tactic)
            arguments, complete = self.policy.decode(states_encoded[row], tactic_id, pool)
            result.append({"tactic_id": tactic_id, "tactic": tactic, "complete": complete,
                "arguments": [{"source": arg.source.value, "identity": arg.stable_id} for arg in arguments]})
        return result


def load_joint_source_bundle(root, *, device="cpu", profile="accepted_local_cues", require_accepted=True):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    root = Path(root).resolve()
    if require_accepted and file_sha256(root / "bundle.json") != BUNDLE_SHA256:
        raise ValueError("bundle manifest differs from the pinned Hugging Face release")
    manifest = _json(root / "bundle.json")
    if manifest.get("format") != FORMAT or manifest.get("model_type") != "joint_source_flat_decoder_v2":
        raise ValueError("not an accepted flat joint-source bundle")
    if require_accepted and (manifest.get("policy_state_sha256") != POLICY_SHA256
                             or manifest.get("retriever_state_sha256") != RETRIEVER_SHA256):
        raise ValueError("model is not the pinned accepted policy/retriever pair")
    for relative, digest in manifest["files"].items():
        member = PurePosixPath(relative)
        if member.is_absolute() or ".." in member.parts or file_sha256(root / relative) != digest:
            raise ValueError(f"bundle file missing, unsafe or changed: {relative}")
    required = {"policy/model.safetensors", "policy/config.json", "policy/inference.json", "vocab/node_vocab.json",
        "vocab/tactic_vocab.json", "retriever/model.safetensors", "retriever/config.json", "retriever/corpus/lemmas.jsonl",
        *("retriever/index/" + name for name in INDEX_FILES)}
    if not required.issubset(manifest["files"]):
        raise ValueError("bundle manifest omits required components")
    tokenizer_files = {str(path.relative_to(root)) for path in (root / "retriever/tokenizer").rglob("*") if path.is_file()}
    if not tokenizer_files or tokenizer_files != {name for name in manifest["files"] if name.startswith("retriever/tokenizer/")}:
        raise ValueError("unmanifested/missing tokenizer files")
    node_vocab, tactic_vocab = (_vocab(root / "vocab" / name) for name in ("node_vocab.json", "tactic_vocab.json"))
    policy_config, retrieval = _json(root / "policy/config.json"), _json(root / "retriever/config.json")
    if (policy_config["input_contract"]["graph_format"] != GRAPH_FORMAT
            or policy_config["input_contract"]["node_type_to_id"] != NODE_TYPE_TO_ID
            or policy_config["input_contract"]["edge_mode"] not in {"forward", "bidirectional"}):
        raise ValueError("unsupported normalized graph contract")
    if (retrieval.get("model_type") != "pretrained_text_dual_encoder" or retrieval.get("pooling") != "attention_mask_mean"
            or retrieval.get("normalize") is not True or type(retrieval.get("max_length")) is not int
            or retrieval["max_length"] < 1):
        raise ValueError("unsupported text retrieval contract")
    retriever = _build_retriever(retrieval)
    retriever.load_state_dict(load_file(str(root / "retriever/model.safetensors")), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "retriever/tokenizer", local_files_only=True, trust_remote_code=False)
    index = LemmaIndex.load(root / "retriever/index")
    _validate_index(index, retriever, tokenizer, retrieval, root / "retriever/corpus/lemmas.jsonl")
    policy = _build_policy(policy_config, node_vocab, tactic_vocab, index.lemma_vectors.shape[1])
    policy.load_state_dict(load_file(str(root / "policy/model.safetensors")), strict=True)
    inference = _json(root / "policy/inference.json")
    if profile not in inference["profiles"] or inference["argument_count_policy"] != "predicted_count":
        raise ValueError("unsupported inference profile/count policy")
    if (inference["max_args"] != policy.pointer.max_args or inference["max_args"] != 3
            or type(inference["retrieval_k"]) is not int or not 1 <= inference["retrieval_k"] <= len(index.lemma_ids)
            or inference.get("allow_repeated_arguments") is not True
            or manifest["policy_state_sha256"] != state_dict_sha256(policy.state_dict())
            or manifest["retriever_state_sha256"] != state_dict_sha256(retriever.state_dict())):
        raise ValueError("bundle architecture/runtime/tensor binding mismatch")
    policy.local_type_bonus_strengths = inference["profiles"][profile]["local_type_bonus_strengths"]
    policy.to(device).eval().requires_grad_(False)
    retriever.to(device).eval().requires_grad_(False)
    return LoadedJointSourceBundle(root, policy, retriever, tokenizer, index, node_vocab, tactic_vocab,
        manifest, retrieval, inference, profile)
