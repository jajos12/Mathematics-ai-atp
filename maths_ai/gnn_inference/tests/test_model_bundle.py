from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from maths_ai.gnn_inference.atp_lean_gnn import (
    BaselineConfig,
    DatasetRow,
    GraphSAGEClassifierConfig,
    PreparedMetadata,
    TrainingLoopConfig,
    build_tactic_vocab,
    encode_tactic_name,
    label_example,
    train_baseline,
)
from maths_ai.gnn_inference.atp_lean_gnn.labels import UNKNOWN_TACTIC
from maths_ai.gnn_inference.atp_lean_gnn.bundle import (
    NODE_VOCAB_NAME,
    SHARED_VOCAB_DIRNAME,
    TACTIC_VOCAB_NAME,
    export_model_bundle,
    load_model_bundle,
    load_pointer_bundle,
    load_state_dict_checked,
    resolve_vocab_paths,
)
from maths_ai.gnn_inference.atp_lean_gnn.cache import (
    SplitReport,
    prepare_output_root,
    write_manifest,
    write_pyg_artifact,
    write_vocab,
)
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels, dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PointerConfig,
    build_baseline_model,
    build_pointer_model,
    detect_state_dict_model_type,
    load_prepared_metadata,
)
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
    TacticWithArgsClassifier,
    TacticWithArgsConfig,
)


class StateDictModelTypeTests(unittest.TestCase):
    def test_legacy_pointer_and_baseline_are_told_apart_by_the_backbone_prefix(self) -> None:
        # Legacy pointer weights have the backbone prefix but no GRU decoder.
        self.assertEqual(
            detect_state_dict_model_type({"backbone.classifier.weight": torch.zeros(2)}),
            "pointer",
        )
        self.assertEqual(
            detect_state_dict_model_type({"classifier.weight": torch.zeros(2)}),
            "baseline",
        )

    def test_detects_gru_pointer_state_dict_separately(self) -> None:
        self.assertEqual(
            detect_state_dict_model_type({"argument_selector.gru.weight_ih": torch.zeros(2)}),
            "pointer_gru",
        )


class LoadStateDictCheckedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(3, 2)

    def test_an_exact_state_dict_loads(self) -> None:
        missing = load_state_dict_checked(self.model, dict(self.model.state_dict()))
        self.assertEqual(missing, [])

    def test_an_unexpected_key_is_rejected(self) -> None:
        state = dict(self.model.state_dict())
        state["not_a_real_parameter"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "unexpected key"):
            load_state_dict_checked(self.model, state)

    def test_a_missing_key_is_rejected_unless_declared(self) -> None:
        state = dict(self.model.state_dict())
        del state["bias"]
        with self.assertRaisesRegex(ValueError, "missing key"):
            load_state_dict_checked(self.model, state)
        # A declared gap is the one case strict=False was ever meant for.
        self.assertEqual(
            load_state_dict_checked(self.model, state, allow_missing_prefixes=("bias",)),
            ["bias"],
        )

    def test_a_size_mismatch_is_reported_as_bad_input_not_as_a_crash(self) -> None:
        # Torch raises RuntimeError for a size mismatch even with strict=False,
        # because the keys line up and only the tensors disagree. Callers treat a
        # failed load as bad input and catch ValueError, so it has to arrive as
        # one or it escapes as a traceback.
        state = dict(self.model.state_dict())
        state["weight"] = torch.zeros(2, 5)
        with self.assertRaisesRegex(ValueError, "do not fit the model"):
            load_state_dict_checked(self.model, state)


class ModelBundleTests(unittest.TestCase):
    """Export/load round trips for published model bundles.

    The property under test throughout is that a bundle is enough on its own.
    Every test that loads a bundle first deletes the prepared dataset, because a
    bundle that quietly depends on the training corpus still being on disk is
    exactly the failure the bundle format exists to prevent.
    """

    def setUp(self) -> None:
        self.prepared_root = Path("tests") / "_tmp_prepared_bundle"
        self.run_root = Path("tests") / "_tmp_runs_bundle"
        self.export_root = Path("tests") / "_tmp_bundle_repo"
        for path in (self.prepared_root, self.run_root, self.export_root):
            if path.exists():
                shutil.rmtree(path)
        self._build_fake_prepared_root()

    def tearDown(self) -> None:
        for path in (self.prepared_root, self.run_root, self.export_root):
            if path.exists():
                shutil.rmtree(path)

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def _split_rows(self) -> dict[str, list[DatasetRow]]:
        return {
            "train": [
                DatasetRow(
                    state="n : Nat\n|- Even n",
                    theorem="demo.train.even",
                    tactic="simp only [h1]",
                    split="train",
                    row_index=0,
                    dataset_name="fake/dataset",
                ),
                DatasetRow(
                    state="State : Nat\n|- State = State",
                    theorem="demo.train.eq",
                    tactic="rw [foo]",
                    split="train",
                    row_index=1,
                    dataset_name="fake/dataset",
                ),
            ],
            "val": [
                DatasetRow(
                    state="m : Nat\n|- Even m",
                    theorem="demo.val.known",
                    tactic="simp",
                    split="val",
                    row_index=0,
                    dataset_name="fake/dataset",
                ),
            ],
            "test": [
                DatasetRow(
                    state="z : Nat\n|- z = z",
                    theorem="demo.test.known",
                    tactic="rw",
                    split="test",
                    row_index=0,
                    dataset_name="fake/dataset",
                ),
            ],
        }

    def _build_fake_prepared_root(self) -> None:
        split_rows = self._split_rows()
        prepare_output_root(self.prepared_root, splits=["train", "val", "test"], force=True)

        node_labels: set[str] = set()
        train_tactic_names: list[str] = []
        dags_by_split: dict[str, list[tuple[DatasetRow, object, str]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for split, rows in split_rows.items():
            for row in rows:
                dag = proof_state_to_dag(row.state)
                tactic_name = str(label_example(row.tactic)["tactic_name"])
                dags_by_split[split].append((row, dag, tactic_name))
                if split == "train":
                    node_labels.update(node.label for node in dag.nodes)
                    train_tactic_names.append(tactic_name)

        self.node_vocab = build_vocab_from_labels(node_labels)
        self.tactic_vocab = build_tactic_vocab(train_tactic_names)
        write_vocab(self.prepared_root, name="node_vocab.json", vocab=self.node_vocab)
        write_vocab(self.prepared_root, name="tactic_vocab.json", vocab=self.tactic_vocab)

        for split in ("train", "val", "test"):
            report = SplitReport(split=split)
            for row, dag, tactic_name in dags_by_split[split]:
                data = dag_to_pyg(dag, self.node_vocab)
                data.y = torch.tensor(
                    [encode_tactic_name(tactic_name, self.tactic_vocab)],
                    dtype=torch.long,
                )
                data.split = split
                data.row_index = row.row_index
                data.dataset_name = row.dataset_name
                data.theorem = row.theorem
                data.tactic_raw = row.tactic
                data.tactic_name = tactic_name
                write_pyg_artifact(
                    self.prepared_root, split=split, row_index=row.row_index, data=data
                )
                report.record_success(dag=dag, tactic_name=tactic_name)

            manifest = report.to_manifest(
                dataset_name="fake/dataset",
                output_root=self.prepared_root,
                vocab_source="train",
                sample_limit=None,
            )
            write_manifest(self.prepared_root, split=split, manifest=manifest)

    def _tiny_baseline_config(self) -> BaselineConfig:
        return BaselineConfig(
            prepared_root=self.prepared_root,
            run_root=self.run_root,
            seed=7,
            device="cpu",
            edge_mode="bidirectional",
            use_node_type=True,
            model=GraphSAGEClassifierConfig(hidden_dim=16, num_layers=2, dropout=0.1),
            training=TrainingLoopConfig(
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip=1.0,
                log_every_batches=1,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
                use_amp=False,
            ),
        ).normalized()

    def _train_tiny_baseline(self) -> Path:
        summary = train_baseline(self._tiny_baseline_config())
        return Path(str(summary["best_checkpoint"]))

    def _write_pointer_checkpoint(self, path: Path) -> tuple[PointerConfig, dict]:
        """Save a pointer checkpoint by hand, the way _save_checkpoint does.

        Training a real pointer needs argument targets the fixture does not have,
        and the export path cares only about the checkpoint's shape.
        """
        config = PointerConfig(
            prepared_root=self.prepared_root,
            run_root=self.run_root,
            seed=7,
            device="cpu",
            edge_mode="bidirectional",
            use_node_type=True,
            gnn_type="sage",
            max_args=3,
            model=TacticWithArgsConfig(hidden_dim=16, num_layers=2, dropout=0.1, max_args=3),
            training=TrainingLoopConfig(batch_size=2, epochs=1, num_workers=0),
        ).normalized()
        metadata = load_prepared_metadata(self.prepared_root)
        model = build_pointer_model(metadata, config)
        payload = {
            "epoch": 3,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"fake": "optimizer state"},
            "val_metrics": {"top1_accuracy": 0.5},
            "node_vocab": metadata.node_vocab,
            "tactic_vocab": metadata.tactic_vocab,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return config, payload

    def _delete_prepared_root(self) -> None:
        shutil.rmtree(self.prepared_root)

    # ------------------------------------------------------------------
    # Checkpoints carry their vocabularies
    # ------------------------------------------------------------------

    def test_training_checkpoints_embed_their_vocabularies_and_hashes(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = load_prepared_metadata(self.prepared_root)

        self.assertEqual(checkpoint["node_vocab"], metadata.node_vocab)
        self.assertEqual(checkpoint["tactic_vocab"], metadata.tactic_vocab)
        # The hash is over the canonical JSON of the mapping, so it identifies
        # the vocabulary rather than the file that happened to hold it.
        self.assertEqual(len(checkpoint["node_vocab_sha256"]), 64)
        self.assertEqual(len(checkpoint["tactic_vocab_sha256"]), 64)

    # ------------------------------------------------------------------
    # Round trip with no dataset present
    # ------------------------------------------------------------------

    def test_bundle_loads_and_predicts_identically_with_no_prepared_dataset(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        metadata = load_prepared_metadata(self.prepared_root)
        config = self._tiny_baseline_config()

        reference = build_baseline_model(metadata, config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
        reference.eval()

        dataset_sample = dag_to_pyg(
            proof_state_to_dag("n : Nat\n|- Even n"), metadata.node_vocab
        )
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([dataset_sample])
        batch.state_node_index = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            expected_logits = reference(batch)

        bundle_dir = self.export_root / "tactic-baseline"
        manifest = export_model_bundle(
            checkpoint_path=checkpoint_path,
            output_dir=bundle_dir,
            dataset="fake/dataset",
        )
        self.assertEqual(manifest["model_type"], "baseline")
        self.assertFalse(manifest["optimizer_state_included"])

        # The point of the format: the corpus is gone, the model still loads.
        self._delete_prepared_root()
        loaded = load_model_bundle(bundle_dir, device="cpu")

        self.assertEqual(loaded.node_vocab, metadata.node_vocab)
        self.assertEqual(loaded.tactic_vocab, metadata.tactic_vocab)
        with torch.no_grad():
            actual_logits = loaded.model(batch)
        self.assertTrue(torch.equal(expected_logits, actual_logits))

    def test_vocabularies_are_shared_between_bundles_by_default(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        first = self.export_root / "tactic-baseline"
        second = self.export_root / "tactic-baseline-copy"
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=first)
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=second)

        shared_dir = self.export_root / SHARED_VOCAB_DIRNAME
        self.assertTrue((shared_dir / NODE_VOCAB_NAME).exists())
        self.assertTrue((shared_dir / TACTIC_VOCAB_NAME).exists())
        # One copy makes drift between bundles unrepresentable rather than merely
        # detectable: there is no second file to disagree with.
        self.assertFalse((first / NODE_VOCAB_NAME).exists())
        self.assertEqual(
            resolve_vocab_paths(first), (shared_dir / NODE_VOCAB_NAME, shared_dir / TACTIC_VOCAB_NAME)
        )

        self._delete_prepared_root()
        self.assertIsNotNone(load_model_bundle(first, device="cpu").model)
        self.assertIsNotNone(load_model_bundle(second, device="cpu").model)

    def test_a_self_contained_bundle_loads_after_being_moved_alone(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "repo" / "tactic-baseline"
        export_model_bundle(
            checkpoint_path=checkpoint_path,
            output_dir=bundle_dir,
            self_contained=True,
        )
        self.assertTrue((bundle_dir / NODE_VOCAB_NAME).exists())

        moved = self.export_root / "elsewhere"
        shutil.move(str(bundle_dir), str(moved))
        self._delete_prepared_root()

        self.assertEqual(load_model_bundle(moved, device="cpu").model_type, "baseline")

    # ------------------------------------------------------------------
    # The vocabulary binding is verified, not trusted
    # ------------------------------------------------------------------

    def test_a_renumbered_vocabulary_of_the_same_size_is_refused(self) -> None:
        # This is the failure the whole format exists for. Swapping two IDs keeps
        # every tensor shape valid, so load_state_dict succeeds and the model
        # then names the wrong tactic for every goal, with no diagnostic anywhere.
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=bundle_dir)

        tactic_vocab_path = self.export_root / SHARED_VOCAB_DIRNAME / TACTIC_VOCAB_NAME
        vocab = json.loads(tactic_vocab_path.read_text(encoding="utf-8"))
        names = [name for name, index in sorted(vocab.items(), key=lambda item: item[1])]
        self.assertGreaterEqual(len(names), 3)
        vocab[names[1]], vocab[names[2]] = vocab[names[2]], vocab[names[1]]
        tactic_vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

        self._delete_prepared_root()
        with self.assertRaisesRegex(ValueError, "does not match the one these weights"):
            load_model_bundle(bundle_dir, device="cpu")

    def test_a_vocabulary_of_the_wrong_size_is_named_before_it_becomes_a_shape_error(
        self,
    ) -> None:
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=bundle_dir)

        tactic_vocab_path = self.export_root / SHARED_VOCAB_DIRNAME / TACTIC_VOCAB_NAME
        vocab = json.loads(tactic_vocab_path.read_text(encoding="utf-8"))
        vocab["a_tactic_that_was_not_trained_on"] = max(vocab.values()) + 1
        tactic_vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

        self._delete_prepared_root()
        with self.assertRaisesRegex(ValueError, "the manifest declares"):
            load_model_bundle(bundle_dir, device="cpu")

    def test_tampered_weights_are_refused(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        manifest = export_model_bundle(
            checkpoint_path=checkpoint_path, output_dir=bundle_dir
        )
        weights_path = bundle_dir / manifest["weights"]
        weights_path.write_bytes(weights_path.read_bytes() + b"\x00")

        self._delete_prepared_root()
        with self.assertRaisesRegex(ValueError, "has been modified or truncated"):
            load_model_bundle(bundle_dir, device="cpu")

    def test_missing_vocabularies_name_both_places_that_were_searched(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=bundle_dir)
        shutil.rmtree(self.export_root / SHARED_VOCAB_DIRNAME)

        with self.assertRaises(FileNotFoundError) as caught:
            load_model_bundle(bundle_dir, device="cpu")
        message = str(caught.exception)
        self.assertIn(str(bundle_dir), message)
        self.assertIn(SHARED_VOCAB_DIRNAME, message)

    # ------------------------------------------------------------------
    # Publication hygiene
    # ------------------------------------------------------------------

    def test_published_config_does_not_leak_training_machine_paths(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=bundle_dir)

        published = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
        # prepared_root and run_root are absolute paths on the training machine,
        # so publishing them would leak the account name and cluster layout while
        # being useless to a consumer, who needs no corpus at all.
        self.assertEqual(published["prepared_root"], ".")
        self.assertEqual(published["run_root"], ".")
        self.assertNotIn(str(self.prepared_root.resolve()), json.dumps(published))

    def test_copied_run_reports_do_not_leak_training_machine_paths(self) -> None:
        # summary.json records the absolute location of the prepared root, the
        # run directory, and every checkpoint. Redacting config.json while
        # copying this file verbatim next to it would publish the same
        # directories anyway, so the copy is redacted too.
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        manifest = export_model_bundle(
            checkpoint_path=checkpoint_path,
            output_dir=bundle_dir,
            run_dir=checkpoint_path.parent,
        )
        self.assertIn("summary.json", manifest["copied_run_files"])

        published = (bundle_dir / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn(str(self.prepared_root.resolve()), published)
        self.assertNotIn(str(checkpoint_path.parent.resolve()), published)
        # The basename survives, so the report still says which checkpoint its
        # numbers belong to.
        summary = json.loads(published)
        self.assertEqual(summary["best_checkpoint"], "best.pt")
        self.assertEqual(summary["prepared_root"], self.prepared_root.name)

    def test_bundle_weights_load_without_executing_pickle(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        manifest = export_model_bundle(
            checkpoint_path=checkpoint_path,
            output_dir=bundle_dir,
            weights_format="torch",
        )
        # The fallback format must still be safe to read: a dict of nothing but
        # tensors is loadable with weights_only=True, which is the property that
        # matters for a file other people download.
        tensors = torch.load(
            bundle_dir / manifest["weights"], map_location="cpu", weights_only=True
        )
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in tensors.values()))
        self.assertNotIn("optimizer_state_dict", tensors)

    def test_a_checkpoint_without_vocabularies_needs_the_prepared_root(self) -> None:
        checkpoint_path = self._train_tiny_baseline()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("node_vocab", "tactic_vocab", "node_vocab_sha256", "tactic_vocab_sha256"):
            checkpoint.pop(key, None)
        legacy_path = self.run_root / "legacy.pt"
        torch.save(checkpoint, legacy_path)

        bundle_dir = self.export_root / "tactic-baseline"
        with self.assertRaisesRegex(ValueError, "does not carry its vocabularies"):
            export_model_bundle(checkpoint_path=legacy_path, output_dir=bundle_dir)

        # The escape hatch for checkpoints trained before the vocabularies were
        # embedded: name the prepared root explicitly.
        manifest = export_model_bundle(
            checkpoint_path=legacy_path,
            output_dir=bundle_dir,
            prepared_root=self.prepared_root,
        )
        self.assertEqual(manifest["vocab_source"], str(self.prepared_root))

    # ------------------------------------------------------------------
    # Pointer bundles, baseline wrapping, and the scorer companion
    # ------------------------------------------------------------------

    def test_a_gru_pointer_checkpoint_round_trips_as_a_gru_pointer_bundle(self) -> None:
        checkpoint_path = self.run_root / "pointer" / "best.pt"
        _, payload = self._write_pointer_checkpoint(checkpoint_path)

        bundle_dir = self.export_root / "pointer"
        manifest = export_model_bundle(
            checkpoint_path=checkpoint_path, output_dir=bundle_dir
        )
        self.assertEqual(manifest["model_type"], "pointer_gru")
        self.assertEqual(manifest["source_epoch"], 3)

        self._delete_prepared_root()
        loaded = load_model_bundle(bundle_dir, device="cpu")

        self.assertIsInstance(loaded.model, TacticWithArgsClassifier)
        self.assertEqual((), loaded.randomly_initialized)
        for key, expected in payload["model_state_dict"].items():
            self.assertTrue(
                torch.equal(expected, loaded.model.state_dict()[key]),
                msg=f"parameter '{key}' changed across the bundle round trip",
            )

    def test_legacy_pointer_model_type_remains_supported(self) -> None:
        """The new bundle identity must not invalidate published legacy bundles."""
        from maths_ai.gnn_inference.atp_lean_gnn.bundle import VALID_MODEL_TYPES

        self.assertIn("pointer", VALID_MODEL_TYPES)
        self.assertIn("pointer_gru", VALID_MODEL_TYPES)

    def test_a_baseline_bundle_is_wrapped_into_a_pointer_and_says_what_is_random(
        self,
    ) -> None:
        # InferencePipeline reaches through model.backbone, so serving a baseline
        # requires wrapping it. The wrap has to be explicit about the untrained
        # argument head rather than presenting its output as a prediction.
        checkpoint_path = self._train_tiny_baseline()
        bundle_dir = self.export_root / "tactic-baseline"
        export_model_bundle(checkpoint_path=checkpoint_path, output_dir=bundle_dir)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._delete_prepared_root()
        loaded = load_pointer_bundle(bundle_dir, device="cpu")

        self.assertEqual(loaded.model_type, "pointer")
        self.assertIsInstance(loaded.model, TacticWithArgsClassifier)
        self.assertEqual(("argument_selector",), loaded.randomly_initialized)
        # The backbone is the baseline verbatim, and the tactic embedding is
        # seeded from its classifier, exactly as training's transfer does it.
        for key, expected in checkpoint["model_state_dict"].items():
            self.assertTrue(
                torch.equal(expected, loaded.model.state_dict()[f"backbone.{key}"]),
                msg=f"backbone parameter '{key}' was not transferred verbatim",
            )
        self.assertTrue(
            torch.equal(
                loaded.model.tactic_embedding.weight,
                loaded.model.backbone.classifier.weight,
            )
        )

    def test_a_scorer_rides_inside_the_pointer_bundle(self) -> None:
        # A scorer checkpoint holds both the fine-tuned pointer and the scorer,
        # and a scorer is meaningless apart from the pointer it was trained
        # against, so it travels in the same bundle rather than a directory of
        # its own that could be paired with the wrong weights.
        checkpoint_path = self.run_root / "premise" / "best.pt"
        config, payload = self._write_pointer_checkpoint(checkpoint_path)
        scorer = PremiseScorer(hidden_dim=config.model.hidden_dim, mode="mlp")
        payload["scorer_state_dict"] = scorer.state_dict()
        torch.save(payload, checkpoint_path)

        bundle_dir = self.export_root / "pointer"
        manifest = export_model_bundle(
            checkpoint_path=checkpoint_path, output_dir=bundle_dir
        )
        # Both constructor arguments are recovered from the weights themselves,
        # so nothing has to be told what its own scorer is.
        self.assertEqual(manifest["scorer"]["hidden_dim"], config.model.hidden_dim)
        self.assertEqual(manifest["scorer"]["scoring_mode"], "mlp")

        self._delete_prepared_root()
        loaded = load_model_bundle(bundle_dir, device="cpu")
        self.assertIsInstance(loaded.scorer, PremiseScorer)
        for key, expected in scorer.state_dict().items():
            self.assertTrue(torch.equal(expected, loaded.scorer.state_dict()[key]))

    def test_a_config_that_disagrees_with_its_weights_fails_before_anything_is_written(
        self,
    ) -> None:
        checkpoint_path = self._train_tiny_baseline()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["config"]["model"]["hidden_dim"] = 32
        broken_path = self.run_root / "broken.pt"
        torch.save(checkpoint, broken_path)

        bundle_dir = self.export_root / "broken"
        with self.assertRaises(ValueError):
            export_model_bundle(checkpoint_path=broken_path, output_dir=bundle_dir)
        # Failing while the output is still empty is what keeps a broken bundle
        # from reaching whoever downloads it.
        self.assertFalse((bundle_dir / "bundle.json").exists())


class PreparedMetadataFromVocabsTests(unittest.TestCase):
    def test_metadata_can_be_built_without_a_dataset(self) -> None:
        node_vocab = {"<UNK>": 0, "State": 1, "Nat": 2}
        tactic_vocab = {UNKNOWN_TACTIC: 0, "simp": 1}

        metadata = PreparedMetadata.from_vocabs(
            node_vocab=node_vocab, tactic_vocab=tactic_vocab
        )

        self.assertEqual(metadata.state_label_id, 1)
        self.assertEqual(metadata.unknown_tactic_id, 0)
        # manifests is read only by the training-side split accessors, so an
        # inference-only metadata legitimately has none.
        self.assertEqual(metadata.manifests, {})

    def test_a_vocabulary_missing_its_required_token_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "'State'"):
            PreparedMetadata.from_vocabs(
                node_vocab={"<UNK>": 0}, tactic_vocab={UNKNOWN_TACTIC: 0}
            )
        with self.assertRaisesRegex(ValueError, UNKNOWN_TACTIC):
            PreparedMetadata.from_vocabs(
                node_vocab={"<UNK>": 0, "State": 1}, tactic_vocab={"simp": 0}
            )


if __name__ == "__main__":
    unittest.main()
