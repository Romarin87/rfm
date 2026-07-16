from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from rfm.features.mrto_full import (
    EquiformerV2EndpointAdapter,
    MRTOFullReactionEncoder,
    MRTOFullReactionInputAdapter,
)
from rfm.models.task_models import (
    MRTOParityEnergyHead,
    MaskedEditPretrainingModel,
    ReactionPropertyRegressor,
    SuirenFusionPropertyRegressor,
    load_stage_a_checkpoint,
)
from rfm.tasks.masked_edit import compute_loss, edit_set_matching_loss


def pair_valid(batch: int, atoms: int) -> torch.Tensor:
    atom_mask = torch.ones(batch, atoms, dtype=torch.bool)
    valid = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    return valid & ~torch.eye(atoms, dtype=torch.bool).unsqueeze(0)


def rotation() -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3))
    matrix[:, 0] *= torch.linalg.det(matrix)
    return matrix


def property_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch, atoms = 2, 5
    coordinates_r = torch.randn(batch, atoms, 3)
    coordinates_p = torch.randn(batch, atoms, 3)
    bo_r = torch.zeros(batch, atoms, atoms)
    bo_p = torch.zeros_like(bo_r)
    for index in range(atoms - 1):
        bo_r[:, index, index + 1] = bo_r[:, index + 1, index] = 1.0
        bo_p[:, index, index + 1] = bo_p[:, index + 1, index] = 1.0
    bo_p[:, 0, 1] = bo_p[:, 1, 0] = 0.0
    bo_p[:, 0, 2] = bo_p[:, 2, 0] = 1.0
    return {
        "z": torch.randint(1, 10, (batch, atoms)),
        "atom_mask": torch.ones(batch, atoms, dtype=torch.bool),
        "pair_valid": pair_valid(batch, atoms),
        "pair_input": torch.stack(
            [bo_r, bo_p, torch.cdist(coordinates_r, coordinates_r), torch.cdist(coordinates_p, coordinates_p)],
            dim=-1,
        ),
        "coordinates_R": coordinates_r,
        "coordinates_P": coordinates_p,
    }


def reverse_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    reverse = {key: value.clone() for key, value in batch.items()}
    reverse["pair_input"] = batch["pair_input"][..., [1, 0, 3, 2]]
    reverse["coordinates_R"] = batch["coordinates_P"] @ rotation() + torch.tensor([2.0, -1.0, 3.0])
    reverse["coordinates_P"] = batch["coordinates_R"] @ rotation() + torch.tensor([-2.0, 4.0, 1.0])
    for key in ("suiren_3d_atom_features", "suiren_3d_graph_features"):
        if key in batch:
            h_r, h_p, delta_h, abs_delta_h = batch[key].chunk(4, dim=-1)
            reverse[key] = torch.cat([h_p, h_r, -delta_h, abs_delta_h], dim=-1)
    return reverse


class MRTOFullContractTest(unittest.TestCase):
    def test_equiformer_endpoint_readout_is_rotation_translation_invariant(self) -> None:
        torch.manual_seed(1)
        batch, atoms = 1, 5
        z = torch.randint(1, 10, (batch, atoms))
        atom_mask = torch.ones(batch, atoms, dtype=torch.bool)
        valid = pair_valid(batch, atoms)
        coordinates = torch.randn(batch, atoms, 3)
        transformed = coordinates @ rotation() + torch.tensor([3.0, -2.0, 1.0])
        adapter = EquiformerV2EndpointAdapter(16, 1, 8, 1, 5.0, 64, 8, 0.0).eval()
        with torch.no_grad():
            atom_a, pair_a = adapter(z, coordinates, atom_mask, valid)
            atom_b, pair_b = adapter(z, transformed, atom_mask, valid)
        torch.testing.assert_close(atom_a, atom_b, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(pair_a, pair_b, atol=3e-5, rtol=3e-5)

    def test_full_operator_preserves_swap_parity_under_independent_endpoint_frames(self) -> None:
        adapter = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=4,
            input_schema="property_irc_rp_bo",
            task_name="property",
            endpoint_layers=1,
            equiformer_layers=1,
            equiformer_channels=8,
            equiformer_lmax=1,
            dropout=0.0,
        ).eval()
        encoder = MRTOFullReactionEncoder(
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            attention_heads=4,
            event_slots=4,
            triangle_layers=1,
            triangle_dim=8,
            event_topk=4,
        ).eval()
        forward_batch = property_batch()
        backward_batch = reverse_batch(forward_batch)
        with torch.no_grad():
            forward = encoder(adapter(forward_batch))
            backward = encoder(adapter(backward_batch))
        for key in ("mrto_atom_plus", "mrto_pair_plus", "mrto_event_plus", "mrto_reaction_plus"):
            torch.testing.assert_close(forward[key], backward[key], atol=7e-5, rtol=7e-5)
        for key in ("mrto_atom_minus", "mrto_pair_minus", "mrto_event_minus", "mrto_reaction_minus"):
            torch.testing.assert_close(forward[key], -backward[key], atol=7e-5, rtol=7e-5)
        torch.testing.assert_close(
            forward["mrto_event_pair_weights"],
            backward["mrto_event_pair_weights"],
            atol=7e-5,
            rtol=7e-5,
        )
        torch.testing.assert_close(
            forward["mrto_event_delta_bo"],
            -backward["mrto_event_delta_bo"],
            atol=7e-5,
            rtol=7e-5,
        )
        energy_head = MRTOParityEnergyHead(16, 0.0).eval()
        y_forward = energy_head(forward)
        y_backward = energy_head(backward)
        torch.testing.assert_close(y_backward[:, 0], -y_forward[:, 0], atol=7e-5, rtol=7e-5)
        torch.testing.assert_close(
            y_backward[:, 1],
            y_forward[:, 1] - y_forward[:, 0],
            atol=7e-5,
            rtol=7e-5,
        )

    def test_edit_set_matching_is_slot_permutation_invariant(self) -> None:
        batch, slots, atoms = 1, 4, 4
        weights = torch.full((batch, slots, atoms, atoms), 1e-8)
        weights[0, 0, 0, 1] = 1.0
        weights[0, 1, 2, 3] = 1.0
        presence = torch.tensor([[8.0, 8.0, -8.0, -8.0]])
        delta = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        changed = torch.zeros(batch, atoms, atoms)
        delta_target = torch.zeros_like(changed)
        loss_mask = torch.zeros_like(changed)
        for i, j, value in ((0, 1, 1.0), (2, 3, -1.0)):
            changed[0, i, j] = changed[0, j, i] = 1.0
            delta_target[0, i, j] = delta_target[0, j, i] = value
            loss_mask[0, i, j] = loss_mask[0, j, i] = 1.0
        batch_data = {
            "changed": changed,
            "delta_bo": delta_target,
            "loss_mask": loss_mask,
            "pair_valid": pair_valid(batch, atoms),
        }
        expected = edit_set_matching_loss(weights, presence, delta, batch_data)
        order = torch.tensor([1, 0, 3, 2])
        permuted = edit_set_matching_loss(weights[:, order], presence[:, order], delta[:, order], batch_data)
        torch.testing.assert_close(expected, permuted)
        wrong = edit_set_matching_loss(weights.roll(1, dims=-1), presence, delta, batch_data)
        self.assertGreater(float(wrong), float(expected) + 0.5)

    def test_suiren_prior_conditions_blocks_without_breaking_swap_parity(self) -> None:
        batch = property_batch()
        atom_r = torch.randn(2, 5, 4)
        atom_p = torch.randn(2, 5, 4)
        graph_r = torch.randn(2, 4)
        graph_p = torch.randn(2, 4)
        batch["suiren_3d_atom_features"] = torch.cat(
            [atom_r, atom_p, atom_p - atom_r, (atom_p - atom_r).abs()], dim=-1
        )
        batch["suiren_3d_graph_features"] = torch.cat(
            [graph_r, graph_p, graph_p - graph_r, (graph_p - graph_r).abs()], dim=-1
        )
        reverse = reverse_batch(batch)
        adapter = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=4,
            input_schema="property_irc_rp_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            suiren_atom_dims={"3d": 16},
            suiren_graph_dims={"3d": 16},
            equiformer_layers=1,
            equiformer_channels=8,
            equiformer_lmax=1,
            dropout=0.0,
        ).eval()
        encoder = MRTOFullReactionEncoder(
            16,
            1,
            0.0,
            attention_heads=4,
            event_slots=4,
            triangle_layers=1,
            triangle_dim=8,
            event_topk=4,
        ).eval()
        for block in encoder.layers:
            block.prior_conditioner.atom_gate.data.fill_(0.5)
            block.prior_conditioner.event_gate.data.fill_(0.5)
        with torch.no_grad():
            forward_output = encoder(adapter(batch))
            reverse_output = encoder(adapter(reverse))
        torch.testing.assert_close(
            forward_output["mrto_reaction_plus"],
            reverse_output["mrto_reaction_plus"],
            atol=8e-5,
            rtol=8e-5,
        )
        torch.testing.assert_close(
            forward_output["mrto_reaction_minus"],
            -reverse_output["mrto_reaction_minus"],
            atol=8e-5,
            rtol=8e-5,
        )

    def test_suiren_atom_features_enter_initial_tokens_and_receive_first_step_gradients(self) -> None:
        torch.manual_seed(13)
        batch = property_batch()
        batch["pair_input"] = batch["pair_input"][..., :2]
        batch.pop("coordinates_R")
        batch.pop("coordinates_P")
        atom_r = torch.randn(2, 5, 4)
        atom_p = torch.randn(2, 5, 4)
        batch["suiren_3d_atom_features"] = torch.cat(
            [atom_r, atom_p, atom_p - atom_r, (atom_p - atom_r).abs()], dim=-1
        )
        zero_batch = dict(batch)
        zero_batch["suiren_3d_atom_features"] = torch.zeros_like(batch["suiren_3d_atom_features"])
        adapter = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            suiren_atom_dims={"3d": 16},
            dropout=0.0,
        )
        encoder = MRTOFullReactionEncoder(
            16,
            1,
            0.0,
            attention_heads=4,
            event_slots=4,
            triangle_layers=1,
            triangle_dim=8,
            event_topk=4,
            prior_gate_init=0.1,
        )

        actual_input = adapter(batch)
        zero_input = adapter(zero_batch)
        self.assertGreater(
            float(
                (
                    actual_input.metadata["mrto_atom_fields"]["a_plus"]
                    - zero_input.metadata["mrto_atom_fields"]["a_plus"]
                )
                .abs()
                .max()
            ),
            1e-5,
        )
        output = encoder(actual_input)
        loss = output["reaction_h"].square().mean() + output["atom_h"].square().mean()
        loss.backward()
        projection_grad = adapter.suiren_atom_state_projection["3d"][0].weight.grad
        conditioner_grad = encoder.layers[0].prior_conditioner.atom_even[1].weight.grad
        self.assertIsNotNone(projection_grad)
        self.assertIsNotNone(conditioner_grad)
        self.assertGreater(float(projection_grad.norm()), 0.0)
        self.assertGreater(float(conditioner_grad.norm()), 0.0)

    def test_atom_only_suiren_does_not_enable_graph_event_conditioner(self) -> None:
        torch.manual_seed(17)
        batch = property_batch()
        batch["pair_input"] = batch["pair_input"][..., :2]
        batch.pop("coordinates_R")
        batch.pop("coordinates_P")
        atom_r = torch.randn(2, 5, 4)
        atom_p = torch.randn(2, 5, 4)
        batch["suiren_3d_atom_features"] = torch.cat(
            [atom_r, atom_p, atom_p - atom_r, (atom_p - atom_r).abs()], dim=-1
        )
        adapter = MRTOFullReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="suiren_fusion_property",
            endpoint_layers=1,
            suiren_atom_dims={"3d": 16},
            dropout=0.0,
        ).eval()
        encoder = MRTOFullReactionEncoder(
            16,
            1,
            0.0,
            attention_heads=4,
            event_slots=4,
            triangle_layers=1,
            triangle_dim=8,
            event_topk=4,
            prior_gate_init=0.2,
        ).eval()
        clean_input = adapter(batch)
        perturbed_input = adapter(batch)
        self.assertTrue(bool(clean_input.metadata["mrto_suiren_atom_present"].all()))
        self.assertFalse(bool(clean_input.metadata["mrto_suiren_graph_present"].any()))
        perturbed_input.metadata["mrto_suiren_graph_plus"] = torch.randn_like(
            perturbed_input.metadata["mrto_suiren_graph_plus"]
        )
        perturbed_input.metadata["mrto_suiren_graph_minus"] = torch.randn_like(
            perturbed_input.metadata["mrto_suiren_graph_minus"]
        )
        with torch.no_grad():
            clean = encoder(clean_input)
            perturbed = encoder(perturbed_input)
        torch.testing.assert_close(clean["mrto_event_plus"], perturbed["mrto_event_plus"])
        torch.testing.assert_close(clean["mrto_event_minus"], perturbed["mrto_event_minus"])

    def test_stage_a_checkpoint_does_not_overwrite_stage_c_suiren_conditioner(self) -> None:
        common = {
            "encoder_type": "mrto_full",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 4,
            "mrto_event_topk": 4,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
        }
        stage_a = MaskedEditPretrainingModel(
            3,
            16,
            1,
            0.0,
            "masked_edit_2d",
            mrto_prior_gate_init=0.0,
            **common,
        )
        stage_c = SuirenFusionPropertyRegressor(
            2,
            16,
            1,
            0.0,
            "property_rp2d_bo",
            suiren_atom_dims={"3d": 16},
            mrto_prior_gate_init=0.2,
            joint_encoder_pass=True,
            **common,
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "model": stage_a.state_dict(),
                    "config": {"encoder_type": "mrto_full"},
                },
                checkpoint,
            )
            load_stage_a_checkpoint(stage_c, str(checkpoint), torch.device("cpu"))
        gates = stage_c.suiren_prior_gate_values()
        self.assertAlmostEqual(gates[0]["atom"], 0.2, places=6)
        self.assertAlmostEqual(gates[0]["graph_event"], 0.2, places=6)

    def test_stage_a_full_model_uses_all_trainable_parameters(self) -> None:
        torch.manual_seed(5)
        batch, atoms = 2, 5
        coordinates = torch.randn(batch, atoms, 3)
        atom_mask = torch.ones(batch, atoms, dtype=torch.bool)
        pair_input = torch.zeros(batch, atoms, atoms, 4)
        pair_input[..., 1] = torch.cdist(coordinates, coordinates) / 10.0
        pair_input[..., 3] = 1.0
        model = MaskedEditPretrainingModel(
            4,
            32,
            1,
            0.0,
            "masked_edit_irc_rp",
            encoder_type="mrto_full",
            mrto_attention_heads=4,
            mrto_event_slots=4,
            mrto_event_topk=4,
            mrto_triangle_layers=1,
            mrto_triangle_dim=8,
            mrto_equiformer_layers=1,
            mrto_equiformer_channels=8,
            mrto_equiformer_lmax=1,
        )
        output = model(
            {
                "z": torch.randint(1, 10, (batch, atoms)),
                "atom_mask": atom_mask,
                "pair_valid": pair_valid(batch, atoms),
                "pair_input": pair_input,
                "coordinates_R": coordinates,
            }
        )
        valid = pair_valid(batch, atoms)
        delta_bo = torch.zeros(batch, atoms, atoms)
        delta_bo[:, 0, 1] = delta_bo[:, 1, 0] = 1.0
        training_batch = {
            "delta_bo": delta_bo,
            "changed": delta_bo.abs(),
            "edit_class": delta_bo.long(),
            "core_atom": torch.tensor([[1, 1, 0, 0, 0]] * batch, dtype=torch.float32),
            "loss_mask": valid.float(),
            "pair_valid": valid,
            "atom_mask": atom_mask,
        }
        args = SimpleNamespace(
            delta_bo_weight=1.0,
            changed_weight=0.5,
            edit_weight=0.5,
            core_weight=0.2,
            mrto_event_set_weight=0.0,
            mrto_event_diversity_weight=0.0,
            mrto_edit_set_weight=0.1,
        )
        loss, _ = compute_loss(output, training_batch, args, torch.ones(4))
        loss.backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_stage_b_views_share_one_endpoint_equiformer(self) -> None:
        model = ReactionPropertyRegressor(
            4,
            16,
            1,
            0.0,
            "property_irc_rp_bo",
            encoder_type="mrto_full",
            mrto_attention_heads=4,
            mrto_event_slots=4,
            mrto_event_topk=4,
            mrto_triangle_layers=1,
            mrto_triangle_dim=8,
            mrto_equiformer_layers=1,
            mrto_equiformer_channels=8,
            mrto_equiformer_lmax=1,
            joint_encoder_pass=True,
        )
        self.assertIs(
            model.adapter.equivariant_geometry,
            model.masked_adapter.equivariant_geometry,
        )

    def test_stage_a_checkpoint_restores_shared_stage_b_geometry(self) -> None:
        common = {
            "encoder_type": "mrto_full",
            "mrto_attention_heads": 4,
            "mrto_event_slots": 4,
            "mrto_event_topk": 4,
            "mrto_triangle_layers": 1,
            "mrto_triangle_dim": 8,
            "mrto_equiformer_layers": 1,
            "mrto_equiformer_channels": 8,
            "mrto_equiformer_lmax": 1,
        }
        stage_a = MaskedEditPretrainingModel(
            4,
            16,
            1,
            0.0,
            "masked_edit_irc_rp",
            **common,
        )
        stage_b = ReactionPropertyRegressor(
            4,
            16,
            1,
            0.0,
            "property_irc_rp_bo",
            joint_encoder_pass=True,
            **common,
        )
        source_parameter = next(stage_a.adapter.equivariant_geometry.parameters()).detach().clone()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "model": stage_a.state_dict(),
                    "config": {"encoder_type": "mrto_full"},
                },
                checkpoint,
            )
            load_stage_a_checkpoint(stage_b, str(checkpoint), torch.device("cpu"))
        restored_parameter = next(stage_b.adapter.equivariant_geometry.parameters()).detach()
        torch.testing.assert_close(source_parameter, restored_parameter)


if __name__ == "__main__":
    unittest.main()
