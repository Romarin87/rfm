from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rfm.features.mrto_full import MRTOFullReactionInputAdapter
from rfm.features.mrto_v1 import MRTOv1ReactionEncoder, MRTOv1ReactionInputAdapter
from rfm.models.task_models import (
    MaskedEditPretrainingModel,
    ReactionPropertyRegressor,
    SuirenFusionPropertyRegressor,
    load_stage_a_checkpoint,
    load_stage_b_checkpoint,
)


def pair_mask(batch_size: int, n_atoms: int) -> torch.Tensor:
    atom_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    valid = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    return valid & ~torch.eye(n_atoms, dtype=torch.bool).unsqueeze(0)


def symmetric(value: torch.Tensor) -> torch.Tensor:
    value = torch.triu(value, diagonal=1)
    return value + value.transpose(1, 2)


def property_batch(*, geometry: bool = False) -> dict[str, torch.Tensor]:
    batch_size, n_atoms = 2, 5
    valid = pair_mask(batch_size, n_atoms)
    bo_r = symmetric(torch.randint(0, 3, (batch_size, n_atoms, n_atoms)).float())
    delta = symmetric(torch.randint(-1, 2, (batch_size, n_atoms, n_atoms)).float())
    bo_p = (bo_r + delta).clamp_min(0.0)
    channels = [bo_r, bo_p]
    if geometry:
        d_r = symmetric(torch.rand(batch_size, n_atoms, n_atoms) * 5.0)
        d_p = symmetric(torch.rand(batch_size, n_atoms, n_atoms) * 5.0)
        channels.extend([d_r, d_p])
    return {
        "z": torch.randint(1, 10, (batch_size, n_atoms)),
        "atom_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool),
        "pair_valid": valid,
        "pair_input": torch.stack(channels, dim=-1),
    }


def reverse_property_batch(batch: dict[str, torch.Tensor], *, geometry: bool = False) -> dict[str, torch.Tensor]:
    reverse = {key: value.clone() for key, value in batch.items()}
    order = [1, 0, 3, 2] if geometry else [1, 0]
    reverse["pair_input"] = batch["pair_input"][..., order]
    for key in ("suiren_3d_atom_features", "suiren_3d_graph_features"):
        if key not in batch:
            continue
        h_r, h_p, delta_h, abs_delta_h = batch[key].chunk(4, dim=-1)
        reverse[key] = torch.cat([h_p, h_r, -delta_h, abs_delta_h], dim=-1)
    return reverse


def add_suiren_features(batch: dict[str, torch.Tensor], state_dim: int = 4) -> None:
    batch_size, n_atoms = batch["atom_mask"].shape
    atom_r = torch.randn(batch_size, n_atoms, state_dim)
    atom_p = torch.randn(batch_size, n_atoms, state_dim)
    atom_delta = atom_p - atom_r
    batch["suiren_3d_atom_features"] = torch.cat([atom_r, atom_p, atom_delta, atom_delta.abs()], dim=-1)
    graph_r = torch.randn(batch_size, state_dim)
    graph_p = torch.randn(batch_size, state_dim)
    graph_delta = graph_p - graph_r
    batch["suiren_3d_graph_features"] = torch.cat(
        [graph_r, graph_p, graph_delta, graph_delta.abs()], dim=-1
    )


def add_masked_edit_view(batch: dict[str, torch.Tensor]) -> None:
    bo_r = batch["pair_input"][..., 0]
    bo_p = batch["pair_input"][..., 1]
    d_r_normalized = (batch["pair_input"][..., 2] / 10.0).clamp(0.0, 1.0)
    delta = bo_p - bo_r
    visibility = torch.ones_like(delta)
    visibility[:, 0, 1] = 0.0
    visibility[:, 1, 0] = 0.0
    batch["masked_pair_input"] = torch.stack(
        [bo_r, d_r_normalized, delta * visibility, visibility],
        dim=-1,
    )
    batch["masked_pair_valid"] = batch["pair_valid"].clone()


def masked_batch() -> dict[str, torch.Tensor]:
    batch_size, n_atoms = 2, 5
    valid = pair_mask(batch_size, n_atoms)
    bo_r = symmetric(torch.randint(0, 2, (batch_size, n_atoms, n_atoms)).float())
    delta = symmetric(torch.randint(-1, 2, (batch_size, n_atoms, n_atoms)).float())
    visibility = torch.ones_like(bo_r)
    visibility[:, 0, 1] = 0.0
    visibility[:, 1, 0] = 0.0
    visibility[:, 2, 3] = 0.0
    visibility[:, 3, 2] = 0.0
    return {
        "z": torch.randint(1, 10, (batch_size, n_atoms)),
        "atom_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool),
        "pair_valid": valid,
        "pair_input": torch.stack([bo_r, delta * visibility, visibility], dim=-1),
    }


class MRTOv1TrainingContractTest(unittest.TestCase):
    def test_parity_aligned_suiren_atom_input_is_single_location_and_swap_equivariant(self) -> None:
        torch.manual_seed(31)
        batch = property_batch()
        add_suiren_features(batch)
        batch.pop("suiren_3d_graph_features")
        adapter = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            suiren_atom_dims={"3d": 16},
            enable_suiren_gates=True,
            suiren_gate_init=0.0,
            suiren_injection_mode="parity_initial_only",
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            gate_logit = torch.atanh(torch.tensor(0.4))
            adapter.suiren_atom_parity_even_gate_logits["3d"].copy_(gate_logit)
            adapter.suiren_atom_parity_odd_gate_logits["3d"].copy_(gate_logit)

        forward = adapter(batch)
        reverse = adapter(reverse_property_batch(batch))
        torch.testing.assert_close(
            forward.metadata["mrto_atom_fields"]["a_plus"],
            reverse.metadata["mrto_atom_fields"]["a_plus"],
        )
        torch.testing.assert_close(
            forward.metadata["mrto_atom_fields"]["a_minus"],
            -reverse.metadata["mrto_atom_fields"]["a_minus"],
        )
        self.assertFalse(bool(forward.metadata["mrto_suiren_atom_present"].any()))
        self.assertFalse(forward.metadata["suiren_cached_derived_blocks_used"])

        changed_derived = {key: value.clone() for key, value in batch.items()}
        state_dim = changed_derived["suiren_3d_atom_features"].shape[-1] // 4
        changed_derived["suiren_3d_atom_features"][..., 2 * state_dim :] = torch.randn_like(
            changed_derived["suiren_3d_atom_features"][..., 2 * state_dim :]
        )
        changed = adapter(changed_derived)
        torch.testing.assert_close(forward.atom_tokens, changed.atom_tokens)
        torch.testing.assert_close(forward.reaction_token, changed.reaction_token)

    def test_stage_b_checkpoint_zero_gate_exactly_matches_parity_suiren_model(self) -> None:
        torch.manual_seed(37)
        common = {
            "encoder_type": "mrto_full",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 4,
            "mrto_event_topk": 4,
            "mrto_endpoint_layers": 1,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
        }
        stage_b = ReactionPropertyRegressor(
            2,
            16,
            1,
            0.0,
            "property_rp2d_bo",
            **common,
        ).eval()
        stage_c = SuirenFusionPropertyRegressor(
            2,
            16,
            1,
            0.0,
            "property_rp2d_bo",
            suiren_atom_dims={"3d": 16},
            enable_suiren_input_gates=True,
            suiren_input_gate_init=0.0,
            suiren_injection_mode="parity_initial_only",
            **common,
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "model": stage_b.state_dict(),
                    "config": {"encoder_type": "mrto_full"},
                    "model_class": "ReactionPropertyRegressor",
                },
                checkpoint,
            )
            restored = load_stage_b_checkpoint(stage_c, str(checkpoint), torch.device("cpu"))
        self.assertEqual(restored, checkpoint)
        self.assertEqual(
            stage_c.suiren_input_gate_values(),
            {"atom_plus": {"3d": 0.0}, "atom_minus": {"3d": 0.0}, "graph": {}},
        )

        batch = property_batch()
        add_suiren_features(batch)
        batch.pop("suiren_3d_graph_features")
        with torch.no_grad():
            expected = stage_b(batch)
            actual = stage_c(batch)
        torch.testing.assert_close(expected["y"], actual["y"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(expected["reaction_h"], actual["reaction_h"], atol=0.0, rtol=0.0)

        stage_c.zero_grad(set_to_none=True)
        stage_c(batch)["y"].square().sum().backward()
        for gates in (
            stage_c.input_adapter.suiren_atom_parity_even_gate_logits,
            stage_c.input_adapter.suiren_atom_parity_odd_gate_logits,
        ):
            self.assertIsNotNone(gates["3d"].grad)
            self.assertGreater(float(gates["3d"].grad.abs()), 0.0)

    def test_mrto_full_single_location_suiren_injection_contract(self) -> None:
        torch.manual_seed(17)
        batch = property_batch()
        add_suiren_features(batch)
        batch.pop("suiren_3d_graph_features")
        perturbed = {key: value.clone() for key, value in batch.items()}
        perturbed["suiren_3d_atom_features"] = torch.randn_like(
            perturbed["suiren_3d_atom_features"]
        )

        initial_only = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            suiren_atom_dims={"3d": 16},
            enable_suiren_gates=True,
            suiren_gate_init=0.0,
            suiren_injection_mode="initial_only",
            dropout=0.0,
        ).eval()
        original_input = initial_only(batch)
        perturbed_input = initial_only(perturbed)
        torch.testing.assert_close(original_input.atom_tokens, perturbed_input.atom_tokens)
        torch.testing.assert_close(original_input.reaction_token, perturbed_input.reaction_token)
        self.assertFalse(bool(original_input.metadata["mrto_suiren_atom_present"].any()))
        self.assertEqual(initial_only.suiren_input_gate_values(), {"atom": {"3d": 0.0}, "graph": {}})

        probe = torch.randn_like(original_input.atom_tokens)
        (original_input.atom_tokens * probe).sum().backward()
        gate_grad = initial_only.suiren_atom_gate_logits["3d"].grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(float(gate_grad.abs()), 0.0)

        conditioner_only = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            suiren_atom_dims={"3d": 16},
            suiren_injection_mode="conditioner_only",
            dropout=0.0,
        ).eval()
        original_input = conditioner_only(batch)
        perturbed_input = conditioner_only(perturbed)
        torch.testing.assert_close(original_input.atom_tokens, perturbed_input.atom_tokens)
        torch.testing.assert_close(original_input.reaction_token, perturbed_input.reaction_token)
        self.assertTrue(bool(original_input.metadata["mrto_suiren_atom_present"].all()))
        self.assertFalse(
            torch.allclose(
                original_input.metadata["mrto_suiren_atom_plus"],
                perturbed_input.metadata["mrto_suiren_atom_plus"],
            )
        )

    def _assert_parity(self, geometry: bool) -> None:
        schema = "property_irc_rp_bo" if geometry else "property_rp2d_bo"
        pair_dim = 4 if geometry else 2
        adapter = MRTOv1ReactionInputAdapter(
            hidden_dim=32,
            pair_input_dim=pair_dim,
            input_schema=schema,
            task_name="property",
            endpoint_layers=1,
            dropout=0.0,
        ).eval()
        encoder = MRTOv1ReactionEncoder(
            hidden_dim=32,
            layers=2,
            dropout=0.0,
            attention_heads=4,
            event_slots=3,
            triangle_layers=1,
            triangle_dim=8,
        ).eval()
        forward_batch = property_batch(geometry=geometry)
        reverse_batch = reverse_property_batch(forward_batch, geometry=geometry)
        with torch.no_grad():
            forward_input = adapter(forward_batch)
            reverse_input = adapter(reverse_batch)
            forward = encoder(forward_input)
            reverse = encoder(reverse_input)

        for field in ("a_plus",):
            torch.testing.assert_close(
                forward_input.metadata["mrto_atom_fields"][field],
                reverse_input.metadata["mrto_atom_fields"][field],
                atol=2e-5,
                rtol=2e-5,
            )
        for field in ("a_minus",):
            torch.testing.assert_close(
                forward_input.metadata["mrto_atom_fields"][field],
                -reverse_input.metadata["mrto_atom_fields"][field],
                atol=2e-5,
                rtol=2e-5,
            )
        for field in ("p_plus",):
            torch.testing.assert_close(
                forward_input.metadata["mrto_pair_fields"][field],
                reverse_input.metadata["mrto_pair_fields"][field],
                atol=2e-5,
                rtol=2e-5,
            )
        for field in ("p_minus",):
            torch.testing.assert_close(
                forward_input.metadata["mrto_pair_fields"][field],
                -reverse_input.metadata["mrto_pair_fields"][field],
                atol=2e-5,
                rtol=2e-5,
            )
        for name in ("mrto_atom_plus", "mrto_pair_plus", "mrto_event_plus", "mrto_reaction_plus"):
            torch.testing.assert_close(forward[name], reverse[name], atol=3e-5, rtol=3e-5)
        for name in ("mrto_atom_minus", "mrto_pair_minus", "mrto_event_minus", "mrto_reaction_minus"):
            torch.testing.assert_close(forward[name], -reverse[name], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(
            forward["mrto_event_pair_weights"],
            reverse["mrto_event_pair_weights"],
            atol=3e-5,
            rtol=3e-5,
        )

    def test_exact_2d_forward_reverse_parity(self) -> None:
        self._assert_parity(geometry=False)

    def test_exact_3d_forward_reverse_parity(self) -> None:
        self._assert_parity(geometry=True)

    def test_suiren_atom_and_graph_inputs_preserve_forward_reverse_parity(self) -> None:
        adapter = MRTOv1ReactionInputAdapter(
            hidden_dim=32,
            pair_input_dim=4,
            input_schema="property_irc_rp_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            dropout=0.0,
            suiren_atom_dims={"3d": 16},
            suiren_graph_dims={"3d": 16},
        ).eval()
        encoder = MRTOv1ReactionEncoder(
            hidden_dim=32,
            layers=2,
            dropout=0.0,
            attention_heads=4,
            event_slots=3,
            triangle_layers=1,
            triangle_dim=8,
        ).eval()
        forward_batch = property_batch(geometry=True)
        add_suiren_features(forward_batch)
        reverse_batch = reverse_property_batch(forward_batch, geometry=True)
        with torch.no_grad():
            forward_input = adapter(forward_batch)
            reverse_input = adapter(reverse_batch)
            forward = encoder(forward_input)
            reverse = encoder(reverse_input)

        torch.testing.assert_close(
            forward_input.metadata["mrto_context_minus"],
            -reverse_input.metadata["mrto_context_minus"],
            atol=2e-5,
            rtol=2e-5,
        )
        for field in ("a_plus",):
            torch.testing.assert_close(
                forward_input.metadata["mrto_atom_fields"][field],
                reverse_input.metadata["mrto_atom_fields"][field],
                atol=2e-5,
                rtol=2e-5,
            )
        for field in ("a_minus",):
            torch.testing.assert_close(
                forward_input.metadata["mrto_atom_fields"][field],
                -reverse_input.metadata["mrto_atom_fields"][field],
                atol=2e-5,
                rtol=2e-5,
            )
        for name in ("mrto_atom_plus", "mrto_pair_plus", "mrto_event_plus", "mrto_reaction_plus"):
            torch.testing.assert_close(forward[name], reverse[name], atol=3e-5, rtol=3e-5)
        for name in ("mrto_atom_minus", "mrto_pair_minus", "mrto_event_minus", "mrto_reaction_minus"):
            torch.testing.assert_close(forward[name], -reverse[name], atol=3e-5, rtol=3e-5)

    def test_raw_projection_keeps_single_bond_distinct_from_no_bond(self) -> None:
        adapter = MRTOv1ReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="property",
            endpoint_layers=0,
        ).eval()
        raw = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]]])
        batch = {
            "z": torch.tensor([[6, 6]]),
            "atom_mask": torch.ones(1, 2, dtype=torch.bool),
            "pair_valid": torch.tensor([[[False, True], [True, False]]]),
            "pair_input": raw,
        }
        with torch.no_grad():
            encoded_input = adapter(batch)
        bonded = encoded_input.metadata["mrto_pair_fields"]["p_plus"][0, 0, 1]
        diagonal = encoded_input.metadata["mrto_pair_fields"]["p_plus"][0, 0, 0]
        self.assertGreater(float((bonded - diagonal).abs().max()), 1e-4)

    def test_stage_a_shapes_and_all_parameters_receive_gradients(self) -> None:
        model = MaskedEditPretrainingModel(
            3,
            32,
            2,
            0.0,
            "masked_edit_2d",
            encoder_type="mrto_v1",
            mrto_attention_heads=4,
            mrto_event_slots=3,
            mrto_endpoint_layers=1,
            mrto_triangle_layers=1,
            mrto_triangle_dim=8,
        )
        out = model(masked_batch())
        self.assertEqual(tuple(out["reaction_h"].shape), (2, 32))
        self.assertEqual(tuple(out["mrto_event_pair_weights"].shape), (2, 3, 5, 5))
        loss = sum(
            out[key].float().square().mean()
            for key in ("delta_bo", "changed_logits", "edit_logits", "core_logits", "reaction_h")
        )
        loss = loss + out["mrto_event_pair_weights"].square().mean()
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(missing, [])

    def test_stage_a_checkpoint_restores_v1_operator_and_masked_adapter(self) -> None:
        kwargs = {
            "encoder_type": "mrto_v1",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 3,
            "mrto_endpoint_layers": 1,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
        }
        stage_a = MaskedEditPretrainingModel(3, 32, 2, 0.0, "masked_edit_2d", **kwargs)
        stage_b = ReactionPropertyRegressor(2, 32, 2, 0.0, "property_rp2d_bo", **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model": stage_a.state_dict(), "config": {"encoder_type": "mrto_v1"}}, checkpoint)
            restored = load_stage_a_checkpoint(stage_b, str(checkpoint), torch.device("cpu"))
        self.assertEqual(restored, checkpoint)
        for source, target in (
            (stage_a.encoder.state_dict(), stage_b.encoder.state_dict()),
            (stage_a.adapter.state_dict(), stage_b.masked_adapter.state_dict()),
            (stage_a.masked_edit_heads.state_dict(), stage_b.masked_edit_heads.state_dict()),
        ):
            self.assertEqual(source.keys(), target.keys())
            for key in source:
                torch.testing.assert_close(source[key], target[key])

    def test_stage_a_checkpoint_restores_into_suiren_v1_model(self) -> None:
        kwargs = {
            "encoder_type": "mrto_v1",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 3,
            "mrto_endpoint_layers": 1,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
        }
        stage_a = MaskedEditPretrainingModel(4, 32, 2, 0.0, "masked_edit_irc_rp", **kwargs)
        stage_c = SuirenFusionPropertyRegressor(
            4,
            32,
            2,
            0.0,
            "property_irc_rp_bo",
            suiren_atom_dims={"3d": 16},
            **kwargs,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model": stage_a.state_dict(), "config": {"encoder_type": "mrto_v1"}}, checkpoint)
            load_stage_a_checkpoint(stage_c, str(checkpoint), torch.device("cpu"))
        batch = property_batch(geometry=True)
        add_suiren_features(batch)
        batch.pop("suiren_3d_graph_features")
        with torch.no_grad():
            output = stage_c(batch)
        self.assertEqual(tuple(output["y"].shape), (2, 2))
        self.assertEqual(tuple(output["reaction_h"].shape), (2, 32))
        self.assertEqual(tuple(output["mrto_event_pair_weights"].shape), (2, 3, 5, 5))

    def test_joint_encoder_pass_matches_two_sequential_passes(self) -> None:
        kwargs = {
            "encoder_type": "mrto_v1",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 3,
            "mrto_endpoint_layers": 1,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
        }
        sequential = ReactionPropertyRegressor(4, 32, 2, 0.0, "property_irc_rp_bo", **kwargs).eval()
        joint = ReactionPropertyRegressor(
            4,
            32,
            2,
            0.0,
            "property_irc_rp_bo",
            joint_encoder_pass=True,
            **kwargs,
        ).eval()
        joint.load_state_dict(sequential.state_dict())
        batch = property_batch(geometry=True)
        add_masked_edit_view(batch)
        with torch.no_grad():
            expected = sequential(batch)
            actual = joint(batch)
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], atol=3e-5, rtol=3e-5)

    def test_suiren_joint_encoder_pass_matches_two_sequential_passes(self) -> None:
        kwargs = {
            "encoder_type": "mrto_v1",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 3,
            "mrto_endpoint_layers": 1,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
            "suiren_atom_dims": {"3d": 16},
        }
        sequential = SuirenFusionPropertyRegressor(4, 32, 2, 0.0, "property_irc_rp_bo", **kwargs).eval()
        joint = SuirenFusionPropertyRegressor(
            4,
            32,
            2,
            0.0,
            "property_irc_rp_bo",
            joint_encoder_pass=True,
            **kwargs,
        ).eval()
        joint.load_state_dict(sequential.state_dict())
        batch = property_batch(geometry=True)
        add_suiren_features(batch)
        batch.pop("suiren_3d_graph_features")
        add_masked_edit_view(batch)
        with torch.no_grad():
            expected = sequential(batch)
            actual = joint(batch)
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], atol=3e-5, rtol=3e-5)

    def test_suiren_masked_outputs_preserve_edit_set_predictions(self) -> None:
        model = SuirenFusionPropertyRegressor(
            4,
            32,
            1,
            0.0,
            "property_irc_rp_bo",
            encoder_type="mrto_full",
            mrto_attention_heads=4,
            mrto_event_slots=4,
            mrto_event_topk=4,
            mrto_endpoint_layers=1,
            mrto_triangle_layers=1,
            mrto_triangle_dim=8,
            mrto_equiformer_layers=1,
            mrto_equiformer_channels=8,
            mrto_equiformer_lmax=1,
        )
        batch_size, n_atoms, event_slots = 2, 5, 4
        encoded = {
            "atom_h": torch.randn(batch_size, n_atoms, 32),
            "pair_h": torch.randn(batch_size, n_atoms, n_atoms, 32),
            "reaction_h": torch.randn(batch_size, 32),
            "mrto_center_atom_logits": torch.randn(batch_size, n_atoms),
            "mrto_center_pair_logits": torch.randn(batch_size, n_atoms, n_atoms),
            "mrto_event_pair_weights": torch.randn(batch_size, event_slots, n_atoms, n_atoms),
            "mrto_event_presence_logits": torch.randn(batch_size, event_slots),
            "mrto_event_delta_bo": torch.randn(batch_size, event_slots),
        }
        output = model._masked_outputs(encoded)
        for key in (
            "mrto_event_pair_weights",
            "mrto_event_presence_logits",
            "mrto_event_delta_bo",
        ):
            self.assertIs(output[key], encoded[key])


if __name__ == "__main__":
    unittest.main()
