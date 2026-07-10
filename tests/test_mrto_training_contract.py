from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rfm.features.mrto import MRTOReactionInputAdapter
from rfm.models.task_models import (
    MaskedEditPretrainingModel,
    ReactionPropertyRegressor,
    load_stage_a_checkpoint,
)


def synthetic_batch(pair_dim: int = 3) -> dict[str, torch.Tensor]:
    batch_size, n_atoms = 2, 5
    atom_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    pair_valid = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    pair_valid &= ~torch.eye(n_atoms, dtype=torch.bool).unsqueeze(0)
    pair_input = torch.rand(batch_size, n_atoms, n_atoms, pair_dim)
    pair_input = 0.5 * (pair_input + pair_input.transpose(1, 2))
    return {
        "z": torch.randint(1, 10, (batch_size, n_atoms)),
        "atom_mask": atom_mask,
        "pair_valid": pair_valid,
        "pair_input": pair_input,
    }


class MRTOTrainingContractTest(unittest.TestCase):
    def test_stage_a_outputs_have_fixed_contract_and_event_weights(self) -> None:
        model = MaskedEditPretrainingModel(
            3,
            32,
            2,
            0.0,
            "masked_edit_2d",
            encoder_type="mrto",
            mrto_attention_heads=4,
            mrto_event_slots=3,
        ).eval()
        batch = synthetic_batch()
        with torch.no_grad():
            encoded = model.encoder(model.adapter(batch))
        self.assertEqual(tuple(encoded["atom_h"].shape), (2, 5, 32))
        self.assertEqual(tuple(encoded["pair_h"].shape), (2, 5, 5, 32))
        self.assertEqual(tuple(encoded["reaction_h"].shape), (2, 32))
        self.assertEqual(tuple(encoded["mrto_event_pair_weights"].shape), (2, 3, 5, 5))
        valid_mass = encoded["mrto_event_pair_weights"].reshape(2, 3, -1).sum(dim=-1)
        torch.testing.assert_close(valid_mass, torch.ones_like(valid_mass), atol=1e-5, rtol=1e-5)

    def test_stage_a_losses_reach_every_trainable_parameter(self) -> None:
        model = MaskedEditPretrainingModel(
            3,
            32,
            2,
            0.0,
            "masked_edit_2d",
            encoder_type="mrto",
            mrto_attention_heads=4,
            mrto_event_slots=3,
        )
        out = model(synthetic_batch())
        loss = sum(
            out[key].float().square().mean()
            for key in ("delta_bo", "changed_logits", "edit_logits", "core_logits")
        )
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(missing, [])

    def test_even_odd_endpoint_fields_have_expected_swap_parity_without_transition_basis(self) -> None:
        adapter = MRTOReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=2,
            input_schema="property_rp2d_bo",
            task_name="property",
            use_odd_field=False,
        ).eval()
        forward = synthetic_batch(pair_dim=2)
        reverse = {key: value.clone() for key, value in forward.items()}
        reverse["pair_input"] = forward["pair_input"][..., [1, 0]]
        with torch.no_grad():
            f = adapter(forward).metadata
            r = adapter(reverse).metadata
        torch.testing.assert_close(f["mrto_atom_fields"]["a_plus"], r["mrto_atom_fields"]["a_plus"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(f["mrto_pair_fields"]["p_plus"], r["mrto_pair_fields"]["p_plus"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(f["mrto_atom_fields"]["a_minus"], -r["mrto_atom_fields"]["a_minus"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(f["mrto_pair_fields"]["p_minus"], -r["mrto_pair_fields"]["p_minus"], atol=1e-5, rtol=1e-5)

    def test_stage_a_checkpoint_restores_mrto_encoder_masked_adapter_and_heads(self) -> None:
        stage_a = MaskedEditPretrainingModel(
            3,
            32,
            2,
            0.0,
            "masked_edit_2d",
            encoder_type="mrto",
            mrto_attention_heads=4,
            mrto_event_slots=3,
        )
        stage_b = ReactionPropertyRegressor(
            2,
            32,
            2,
            0.0,
            "property_rp2d_bo",
            encoder_type="mrto",
            mrto_attention_heads=4,
            mrto_event_slots=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "model": stage_a.state_dict(),
                    "config": {"encoder_type": "mrto"},
                },
                checkpoint,
            )
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


if __name__ == "__main__":
    unittest.main()

