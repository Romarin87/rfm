from __future__ import annotations

import torch
import torch.nn as nn

from rfm.models.conditional_residual import ConditionalResidualProbe


class DummyStageB(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(0.9)
        self.anchor = nn.Parameter(torch.ones(()))
        self.hidden_dim = hidden_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["z"].shape[0]
        plus = self.dropout(torch.ones((batch_size, self.hidden_dim), device=batch["z"].device))
        minus = self.dropout(torch.full_like(plus, 2.0))
        prediction = torch.stack([plus.mean(dim=-1), minus.mean(dim=-1)], dim=-1)
        return {
            "y": prediction,
            "mrto_reaction_plus": plus,
            "mrto_reaction_minus": minus,
        }


def make_probe(mode: str) -> ConditionalResidualProbe:
    return ConditionalResidualProbe(
        stage_b=DummyStageB(8),
        hidden_dim=8,
        suiren_atom_dim=32,
        y_mean=torch.tensor([1.0, 2.0]),
        y_std=torch.tensor([3.0, 4.0]),
        mode=mode,
        residual_hidden_dim=8,
        dropout=0.0,
    )


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "z": torch.ones((2, 3), dtype=torch.long),
        "atom_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "pair_input": torch.zeros((2, 3, 3, 4)),
        "pair_valid": torch.ones((2, 3, 3), dtype=torch.bool),
        "suiren_3d_atom_features": torch.randn((2, 3, 32)),
    }


def test_zero_initialization_preserves_frozen_stage_b_prediction() -> None:
    probe = make_probe("stageb_suiren")
    probe.train()
    out = probe(make_batch())
    assert not probe.stage_b.training
    assert torch.equal(out["y"], out["baseline_y"])
    assert torch.count_nonzero(out["residual_norm"]) == 0
    assert all(not parameter.requires_grad for parameter in probe.stage_b.parameters())


def test_control_and_suiren_modes_have_identical_trainable_capacity() -> None:
    control = make_probe("stageb_only")
    suiren = make_probe("stageb_suiren")
    assert control.trainable_parameter_count() == suiren.trainable_parameter_count()
    assert control.residual_state_dict().keys() == suiren.residual_state_dict().keys()


def test_suiren_values_only_affect_suiren_mode_after_output_head_is_enabled() -> None:
    control = make_probe("stageb_only")
    suiren = make_probe("stageb_suiren")
    suiren.load_state_dict(control.state_dict())
    with torch.no_grad():
        control.residual_head[-1].weight.fill_(0.1)
        suiren.residual_head[-1].weight.fill_(0.1)
    first = make_batch()
    changed_features = first["suiren_3d_atom_features"].clone()
    changed_features[0, 0, 0] += 5.0
    second = {**first, "suiren_3d_atom_features": changed_features}
    control_first = control(first)["y"]
    control_second = control(second)["y"]
    suiren_first = suiren(first)["y"]
    suiren_second = suiren(second)["y"]
    assert torch.allclose(control_first, control_second)
    assert not torch.allclose(suiren_first, suiren_second)
