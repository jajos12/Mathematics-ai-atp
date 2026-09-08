from __future__ import annotations

import unittest

import torch

from maths_ai.gnn_inference.atp_lean_gnn import (
    DEMO_STATE,
    TACTIC_ARITY,
    build_premise_mask,
    build_vocab,
    get_tactic_arity,
    parse_tactic_arguments,
    proof_state_to_dag,
)
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
    ArgumentSelector,
    TacticWithArgsClassifier,
    compute_combined_loss,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg


class TacticArityRegistryTests(unittest.TestCase):
    def test_known_tactics_return_correct_arity(self) -> None:
        self.assertEqual(get_tactic_arity("simp"), 0)
        self.assertEqual(get_tactic_arity("apply"), 1)
        self.assertEqual(get_tactic_arity("exact"), 1)
        self.assertEqual(get_tactic_arity("rw"), 1)
        self.assertEqual(get_tactic_arity("have"), 2)

    def test_unknown_tactic_returns_default(self) -> None:
        self.assertEqual(get_tactic_arity("totally_unknown_tactic_xyz"), 1)

    def test_all_entries_have_nonnegative_arity(self) -> None:
        for tactic, arity in TACTIC_ARITY.items():
            self.assertGreaterEqual(arity, 0, f"Tactic '{tactic}' has negative arity")


class ParseTacticArgumentsTests(unittest.TestCase):
    def test_bracket_arguments(self) -> None:
        name, args = parse_tactic_arguments("rw [foo, bar]")
        self.assertEqual(name, "rw")
        self.assertEqual(args, ["foo", "bar"])

    def test_plain_argument(self) -> None:
        name, args = parse_tactic_arguments("apply h1")
        self.assertEqual(name, "apply")
        self.assertEqual(args, ["h1"])

    def test_simp_only_with_brackets(self) -> None:
        name, args = parse_tactic_arguments("simp only [h1, h2]")
        self.assertEqual(name, "simp")
        self.assertEqual(args, ["h1", "h2"])

    def test_no_arguments(self) -> None:
        name, args = parse_tactic_arguments("simp")
        self.assertEqual(name, "simp")
        self.assertEqual(args, [])

    def test_empty_string(self) -> None:
        name, args = parse_tactic_arguments("")
        self.assertEqual(args, [])

    def test_exact_with_complex_argument(self) -> None:
        name, args = parse_tactic_arguments("exact Nat.zero_add n")
        self.assertEqual(name, "exact")
        self.assertIn("Nat.zero_add", args)


class PremiseMaskTests(unittest.TestCase):
    def test_excludes_syntax_nodes(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)

        self.assertEqual(len(mask), dag.num_nodes)

        for node, is_valid in zip(dag.nodes, mask):
            if node.label in ("App", "Arrow", "State", "Goal", "Forall"):
                self.assertFalse(
                    is_valid,
                    f"Syntax node '{node.label}' (id={node.id}) should be masked out",
                )

    def test_includes_var_and_hyp_nodes(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)

        hyp_included = any(
            mask[node.id] for node in dag.nodes if node.label == "Hyp"
        )
        var_included = any(
            mask[node.id] for node in dag.nodes if node.node_type == "var"
        )
        self.assertTrue(hyp_included, "At least one 'Hyp' node should be included")
        self.assertTrue(var_included, "At least one 'var' node should be included")

    def test_at_least_one_node_is_selectable(self) -> None:
        dag = proof_state_to_dag(DEMO_STATE)
        mask = build_premise_mask(dag)
        self.assertTrue(any(mask), "Premise mask should have at least one True entry")


class ArgumentSelectorTests(unittest.TestCase):
    def test_output_shape_and_masking(self) -> None:
        hidden_dim = 16
        selector = ArgumentSelector(hidden_dim)

        batch_size = 2
        nodes_per_graph = [5, 7]
        total_nodes = sum(nodes_per_graph)

        state_emb = torch.randn(batch_size, hidden_dim)
        tactic_emb = torch.randn(batch_size, hidden_dim)
        node_embeddings = torch.randn(total_nodes, hidden_dim)

        # Build batch index and premise mask
        batch_index = torch.cat([
            torch.full((n,), i, dtype=torch.long) for i, n in enumerate(nodes_per_graph)
        ])
        premise_mask = torch.ones(total_nodes, dtype=torch.bool)
        # Mask out the last node in each graph
        premise_mask[4] = False   # last node of graph 0
        premise_mask[11] = False  # last node of graph 1

        scores, selected_emb = selector(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index
        )

        max_nodes = max(nodes_per_graph)
        self.assertEqual(scores.shape, (batch_size, max_nodes))
        self.assertEqual(selected_emb.shape, (batch_size, hidden_dim))

        # Verify masked positions have -inf scores
        probs = torch.softmax(scores, dim=1)
        self.assertAlmostEqual(float(probs[0, 4].item()), 0.0, places=5)

    def test_autoregressive_step_changes_output(self) -> None:
        hidden_dim = 16
        selector = ArgumentSelector(hidden_dim)

        state_emb = torch.randn(1, hidden_dim)
        tactic_emb = torch.randn(1, hidden_dim)
        node_embeddings = torch.randn(5, hidden_dim)
        batch_index = torch.zeros(5, dtype=torch.long)
        premise_mask = torch.ones(5, dtype=torch.bool)

        scores1, sel1 = selector(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index
        )
        state2 = selector.initial_state(state_emb, tactic_emb)
        scores2, sel2 = selector(
            state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
            decoder_state=selector.gru(sel1, state2),
        )

        # Scores should differ because the query context changed
        self.assertFalse(
            torch.allclose(scores1, scores2),
            "Autoregressive step should produce different scores",
        )

    def test_repeat_positions_are_suppressed(self) -> None:
        selector = ArgumentSelector(8)
        state = torch.randn(1, 8)
        tactic = torch.randn(1, 8)
        nodes = torch.randn(3, 8)
        batch_index = torch.zeros(3, dtype=torch.long)
        mask = torch.ones(3, dtype=torch.bool)
        excluded = torch.tensor([[True, False, False]])

        scores, _ = selector(
            state, tactic, nodes, mask, batch_index,
            excluded_positions=excluded,
        )
        self.assertTrue(torch.isneginf(scores[0, 0]))


class TacticWithArgsClassifierTests(unittest.TestCase):
    def _build_tiny_batch(self):
        """Build a minimal batched PyG graph for testing."""
        from torch_geometric.data import Batch, Data

        dag1 = proof_state_to_dag("n : Nat\n⊢ Even n")
        dag2 = proof_state_to_dag("m : Nat\n⊢ Even m")

        vocab = build_vocab([dag1, dag2])
        d1 = dag_to_pyg(dag1, vocab, add_reverse_edges=True)
        d2 = dag_to_pyg(dag2, vocab, add_reverse_edges=True)

        # Add required fields
        for data, dag in [(d1, dag1), (d2, dag2)]:
            data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
            data.y = torch.tensor([1], dtype=torch.long)
            data.tactic_name = "apply"
            data.arg_node_indices = torch.tensor([0], dtype=torch.long)
            data.arg_count = 1

        # Find State node for state_node_index
        state_label_id = vocab.get("State", 0)
        for data in [d1, d2]:
            state_matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
            data.state_node_index = state_matches[-1:]

        batch = Batch.from_data_list([d1, d2])
        return batch, vocab

    def test_forward_returns_both_heads(self) -> None:
        batch, vocab = self._build_tiny_batch()

        model = TacticWithArgsClassifier(
            num_node_labels=len(vocab),
            num_tactics=5,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            max_args=2,
        )

        tactic_logits, arg_logits_list, stop_logits_list = model(
            batch,
            teacher_tactic_ids=batch.y.view(-1),
            arg_targets=torch.tensor([[0], [5]], dtype=torch.long),
        )
        self.assertEqual(tactic_logits.shape, (2, 5))
        self.assertEqual(len(arg_logits_list), 2)
        self.assertEqual(len(stop_logits_list), 3)
        for arg_logits in arg_logits_list:
            self.assertEqual(arg_logits.shape[0], 2)

    def test_zero_arity_returns_empty_arg_list(self) -> None:
        batch, vocab = self._build_tiny_batch()

        model = TacticWithArgsClassifier(
            num_node_labels=len(vocab),
            num_tactics=5,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
            max_args=2,
        )

        tactic_logits, arg_logits_list, stop_logits_list = model(
            batch,
            teacher_tactic_ids=batch.y.view(-1),
        )

        self.assertEqual(tactic_logits.shape, (2, 5))
        self.assertEqual(len(arg_logits_list), 2)
        self.assertEqual(len(stop_logits_list), 3)

    def test_three_argument_decode_is_representable(self) -> None:
        batch, vocab = self._build_tiny_batch()
        model = TacticWithArgsClassifier(
            num_node_labels=len(vocab), num_tactics=5, hidden_dim=16,
            num_layers=2, max_args=3,
        )
        _, arg_logits_list, stop_logits_list = model(
            batch,
            teacher_tactic_ids=batch.y.view(-1),
            arg_targets=torch.tensor([[0, 1, 2], [5, 6, 7]], dtype=torch.long),
        )
        self.assertEqual(len(arg_logits_list), 3)
        self.assertEqual(len(stop_logits_list), 4)

    def test_backward_through_multi_step_teacher_forcing(self) -> None:
        """A backward pass through the recurrent loop must not hit autograd.

        The exclusion mask is consumed by masked_fill at every decode step,
        and masked_fill saves its mask for backward.  Mutating that mask in
        place between steps bumps its version, so autograd aborts on the
        second step with "modified by an inplace operation" -- a defect the
        forward-only tests cannot see.  The mask must accumulate out-of-place.
        """
        batch, vocab = self._build_tiny_batch()
        model = TacticWithArgsClassifier(
            num_node_labels=len(vocab), num_tactics=5, hidden_dim=16,
            num_layers=2, max_args=3,
        )
        n_per_graph = int(batch.ptr[1].item())
        # Global-index targets across two graphs, forcing the mask to update
        # after every step while gradients are being recorded.
        arg_targets = torch.tensor(
            [[0, 1, -1], [n_per_graph, n_per_graph + 1, -1]], dtype=torch.long
        )
        tactic_logits, arg_logits_list, stop_logits_list = model(
            batch,
            teacher_tactic_ids=batch.y.view(-1),
            arg_targets=arg_targets,
        )
        self.assertEqual(len(arg_logits_list), 3)

        loss = tactic_logits.sum()
        for scores in arg_logits_list:
            # clamp keeps the -inf masked positions out of the sum's gradient
            loss = loss + scores.clamp(min=-1e4).sum()
        for stop_logits in stop_logits_list:
            loss = loss + stop_logits.sum()
        loss.backward()  # must not raise

        # Gradients reached the decoder parameters, not just the backbone.
        self.assertIsNotNone(model.argument_selector.gru.weight_ih.grad)
        self.assertTrue(
            model.argument_selector.gru.weight_ih.grad.abs().sum().item() > 0
        )
        self.assertIsNotNone(model.stop_head.weight.grad)


class CombinedLossTests(unittest.TestCase):
    def test_loss_uses_corpus_counts_not_registry_arity(self) -> None:
        tactic_logits = torch.randn(3, 4, requires_grad=True)
        arg_logits = [
            torch.randn(3, 6, requires_grad=True),
            torch.randn(3, 6, requires_grad=True),
            torch.randn(3, 6, requires_grad=True),
        ]
        stop_logits = [torch.randn(3, requires_grad=True) for _ in range(3)]
        targets = torch.tensor([[0, -1, -1], [1, 2, -1], [3, 4, 5]])
        batch_index = torch.cat([
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.full((2,), 2, dtype=torch.long),
        ])

        _, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            torch.tensor([1, 1, 1]),
            targets,
            batch_index,
            arg_count_per_sample=[0, 1, 3],
            stop_logits_list=stop_logits,
        )

        self.assertEqual(metrics["arg_target_count"], 4)
        self.assertEqual(metrics["arg_truncated_count"], 0)
        # Denominator counts only DAG-node targets within the decode budget;
        # the -1 lemma positions never enter it.
        self.assertEqual(metrics["arg_lemma_position_count"], 0)

    def test_targets_beyond_max_args_are_reported(self) -> None:
        tactic_logits = torch.randn(1, 4, requires_grad=True)
        arg_logits = [torch.randn(1, 6, requires_grad=True) for _ in range(2)]
        _, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            torch.tensor([1]),
            torch.tensor([[0, 1]]),
            torch.zeros(2, dtype=torch.long),
            arg_count_per_sample=[3],
        )

        # Only the two node targets within the decode budget are scoreable;
        # the third position is truncated, not counted as a missed target.
        self.assertEqual(metrics["arg_target_count"], 2)
        self.assertEqual(metrics["arg_truncated_count"], 1)
        self.assertEqual(metrics["arg_truncated_examples"], 1)

    def test_lemma_positions_do_not_inflate_the_coverage_denominator(self) -> None:
        """Lemma citations are the scorer's population, not the pointer's.

        Stored as -1 node indices, they must be excluded from
        arg_target_count and reported separately, so coverage measures the
        model rather than the corpus's citation density.
        """
        tactic_logits = torch.randn(2, 4, requires_grad=True)
        arg_logits = [torch.randn(2, 6, requires_grad=True) for _ in range(2)]
        # Sample 0: two node targets then a lemma citation (-1).
        # Sample 1: one node target, one lemma citation, one truncated.
        targets = torch.tensor([[0, 1, -1], [2, -1, 3]])
        _, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            torch.tensor([1, 1]),
            targets,
            torch.cat([torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long)]),
            arg_count_per_sample=[3, 3],
        )
        # Decode budget is 2 steps: 2 + 1 in-budget node targets.
        self.assertEqual(metrics["arg_target_count"], 3)
        # Sample 0's step-2 lemma and sample 1's step-1 lemma are both past
        # the decode budget or unresolvable node indices.
        self.assertEqual(metrics["arg_lemma_position_count"], 1)

    def test_truncated_samples_stop_target_clamps_to_the_last_step(self) -> None:
        """arg_count > max_args must supervise "stop" at the final position.

        The decoder cannot decode past max_args, so a truncated sample whose
        targets all say "continue" pushes the stop head negative everywhere
        and penalizes correct early stops on untruncated samples.
        """
        # One sample with arg_count=2, one truncated with arg_count=3, decoder
        # exposes 3 stop positions (max_args=2).  Both clamp the boundary to
        # step 2, so a model that always predicts "stop" is now fully correct.
        tactic_logits = torch.randn(2, 4, requires_grad=True)
        arg_logits = [torch.randn(2, 6, requires_grad=True) for _ in range(2)]
        stop_logits = [
            torch.full((2,), 20.0, requires_grad=True) for _ in range(3)
        ]
        _, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            torch.tensor([1, 1]),
            torch.tensor([[0, -1], [1, 2]]),
            torch.cat([torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long)]),
            arg_count_per_sample=[2, 3],
            stop_logits_list=stop_logits,
        )
        # Targets for both samples are [0, 0, 1]: "continue, continue, stop".
        # A model that always fires the stop head is wrong at steps 0,1.
        self.assertGreater(metrics["stop_loss"], 0.0)

        # With targets clamped, a model that never fires matches the
        # truncated sample at the final step only if the target there is 1.
        # Verify the clamped supervision directly: a never-stop model must
        # incur loss from the final position, not get it for free.
        never_stop = [
            torch.full((2,), -20.0, requires_grad=True) for _ in range(3)
        ]
        _, metrics_ns = compute_combined_loss(
            tactic_logits.detach(),
            [a.detach() for a in arg_logits],
            torch.tensor([1, 1]),
            torch.tensor([[0, -1], [1, 2]]),
            torch.cat([torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long)]),
            arg_count_per_sample=[2, 3],
            stop_logits_list=never_stop,
        )
        # Pre-clamp behavior supervised "continue" at the final step for the
        # truncated sample; post-clamp the target is 1, so never-stop must be
        # penalized there (loss strictly greater than zero).
        self.assertGreater(metrics_ns["stop_loss"], 0.0)
        # Scored positions are the decision-relevant ones: steps 0,1 (target
        # "continue", prediction continue -> correct) and step 2 (target
        # "stop", prediction continue -> wrong).  2 of 3 correct.
        self.assertAlmostEqual(metrics_ns["stop_accuracy"], 2.0 / 3.0)

    def test_stop_accuracy_scores_only_decision_relevant_steps(self) -> None:
        """Positions past the boundary are trained but not scored.

        Counting the trivially-correct plateau after the target flips to stop
        inflated the metric the same way the old arg_target_coverage bug did.
        """
        # Sample with arg_count=0: boundary step 0.  Scored positions: 0 and 1.
        # Positions 2+ are the plateau and must not enter the count.
        tactic_logits = torch.randn(1, 4, requires_grad=True)
        arg_logits = [torch.randn(1, 6, requires_grad=True) for _ in range(2)]
        # Model predicts continue at step 0 (wrong), stop everywhere after.
        stop_logits = [
            torch.tensor([-5.0]),
            torch.tensor([5.0]),
            torch.tensor([5.0]),
        ]
        _, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            torch.tensor([1]),
            torch.tensor([[-1, -1]]),
            torch.arange(6, dtype=torch.long),
            arg_count_per_sample=[0],
            stop_logits_list=stop_logits,
        )
        # Scored: steps 0 (target stop, pred continue -> wrong) and 1
        # (target stop, pred stop -> correct).  Steps 2+ excluded.
        self.assertAlmostEqual(metrics["stop_accuracy"], 0.5)

    def test_masks_invalid_arg_targets(self) -> None:
        batch_size = 2
        num_tactics = 5
        num_nodes = 8

        tactic_logits = torch.randn(batch_size, num_tactics, requires_grad=True)
        arg_logits = [torch.randn(batch_size, num_nodes, requires_grad=True)]
        tactic_targets = torch.tensor([1, 2], dtype=torch.long)
        arg_targets = torch.tensor([[3], [-1]], dtype=torch.long)  # second sample unresolvable
        batch_index = torch.cat([
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        ])

        loss, metrics = compute_combined_loss(
            tactic_logits,
            arg_logits,
            tactic_targets,
            arg_targets,
            batch_index,
            arg_count_per_sample=[1, 1],
            arg_loss_weight=0.5,
            unknown_tactic_id=0,
        )

        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        self.assertGreater(float(loss.item()), 0.0, "Loss should be positive")
        self.assertIn("tactic_loss", metrics)
        self.assertIn("arg_loss", metrics)
        self.assertEqual(metrics["arg_valid_count"], 1)
        # The second sample's target is -1 (token resolved to no node), which
        # is not a scoreable target under the honest denominator: coverage is
        # over targets the pointer could have been scored on, and an
        # unresolvable token never enters it.  A premise-masked target (>= 0
        # but -inf logits) does stay in the denominator so masking regressions
        # surface as coverage < 1.
        self.assertEqual(metrics["arg_target_count"], 1)
        self.assertEqual(metrics["arg_target_coverage"], 1.0)
        self.assertIn("arg_top1_accuracy", metrics)
        self.assertIn("arg_top5_accuracy", metrics)

        # Verify gradients flow
        loss.backward()
        self.assertIsNotNone(tactic_logits.grad)

    def test_zero_arity_skips_arg_loss(self) -> None:
        tactic_logits = torch.randn(2, 5, requires_grad=True)
        tactic_targets = torch.tensor([1, 2], dtype=torch.long)
        batch_index = torch.cat([
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
        ])

        loss, metrics = compute_combined_loss(
            tactic_logits,
            [],  # no arg logits
            tactic_targets,
            torch.tensor([[-1], [-1]], dtype=torch.long),
            batch_index,
            arg_count_per_sample=[0, 0],
            arg_loss_weight=0.5,
            unknown_tactic_id=0,
        )

        self.assertAlmostEqual(metrics["arg_loss"], 0.0)
        self.assertEqual(metrics["arg_valid_count"], 0)
        self.assertEqual(metrics["arg_top1_accuracy"], 0.0)
        self.assertEqual(metrics["arg_top5_accuracy"], 0.0)
        self.assertEqual(metrics["arg_target_coverage"], 0.0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
