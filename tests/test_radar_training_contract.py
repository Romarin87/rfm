from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rfm.features.radar import RADARReactionInputAdapter, _soft_pool
from rfm.models.task_models import (
    MaskedEditPretrainingModel,
    ReactionPropertyRegressor,
    load_stage_a_checkpoint,
)


def synthetic_batch(pair_dim: int = 4, hidden_suiren_dim: int = 0) -> dict[str, torch.Tensor]:
    batch_size, n_atoms = 2, 5
    atom_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    pair_valid = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    pair_valid &= ~torch.eye(n_atoms, dtype=torch.bool).unsqueeze(0)
    pair_input = torch.randn(batch_size, n_atoms, n_atoms, pair_dim)
    pair_input = 0.5 * (pair_input + pair_input.transpose(1, 2))
    out = {
        "z": torch.randint(1, 10, (batch_size, n_atoms)),
        "atom_mask": atom_mask,
        "pair_valid": pair_valid,
        "pair_input": pair_input,
    }
    if hidden_suiren_dim:
        out["suiren_3d_atom_features"] = torch.randn(batch_size, n_atoms, hidden_suiren_dim)
    return out


class RADARTrainingContractTest(unittest.TestCase):
    def test_stage_a_losses_reach_every_trainable_parameter(self) -> None:
        model = MaskedEditPretrainingModel(
            4,
            32,
            2,
            0.0,
            "masked_edit_irc_rp",
            encoder_type="radar",
            radar_attention_heads=4,
        )
        out = model(synthetic_batch())
        loss = sum(value.float().square().mean() for key, value in out.items() if key != "reaction_h")
        loss.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(missing, [])

    def test_changed_and_core_predictions_are_router_logits(self) -> None:
        model = MaskedEditPretrainingModel(
            4,
            32,
            2,
            0.0,
            "masked_edit_irc_rp",
            encoder_type="radar",
            radar_attention_heads=4,
        ).eval()
        batch = synthetic_batch()
        with torch.no_grad():
            encoded = model.encoder(model.adapter(batch))
            out = model(batch)
        torch.testing.assert_close(out["changed_logits"], encoded["radar_center_pair_logits"])
        torch.testing.assert_close(out["core_logits"], encoded["radar_center_atom_logits"])

    def test_soft_pool_stays_normalized_for_very_negative_logits(self) -> None:
        values = torch.tensor([[[2.0], [4.0]]])
        logits = torch.full((1, 2), -20.0)
        mask = torch.ones((1, 2), dtype=torch.bool)
        pooled = _soft_pool(values, logits, mask)
        torch.testing.assert_close(pooled, torch.tensor([[3.0]]))

    def test_suiren_blocks_are_routed_to_matching_state_streams(self) -> None:
        adapter = RADARReactionInputAdapter(
            hidden_dim=16,
            pair_input_dim=4,
            input_schema="property_irc_rp_bo",
            task_name="suiren_fusion_property",
            suiren_atom_dims={"3d": 8},
        ).eval()
        baseline = synthetic_batch(hidden_suiren_dim=8)
        baseline["suiren_3d_atom_features"].zero_()
        changed = {key: value.clone() for key, value in baseline.items()}
        changed["suiren_3d_atom_features"][..., 0] = 1.0
        changed["suiren_3d_atom_features"][..., 1] = -1.0
        with torch.no_grad():
            base_streams = adapter(baseline).metadata["radar_state_streams"]
            changed_streams = adapter(changed).metadata["radar_state_streams"]
        self.assertFalse(torch.allclose(base_streams["a_r"], changed_streams["a_r"]))
        torch.testing.assert_close(base_streams["a_p"], changed_streams["a_p"])
        torch.testing.assert_close(base_streams["a_delta"], changed_streams["a_delta"])

    def test_stage_a_checkpoint_restores_encoder_masked_adapter_and_heads(self) -> None:
        stage_a = MaskedEditPretrainingModel(
            4,
            32,
            2,
            0.0,
            "masked_edit_irc_rp",
            encoder_type="radar",
            radar_attention_heads=4,
        )
        stage_b = ReactionPropertyRegressor(
            4,
            32,
            2,
            0.0,
            "property_irc_rp_bo",
            encoder_type="radar",
            radar_attention_heads=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "model": stage_a.state_dict(),
                    "config": {"encoder_type": "radar"},
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
