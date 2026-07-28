from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from rfm.cli.train_reaction_property import stage_b_parameter_group
from rfm.cli.train_suiren_fusion_property import stage_c_parameter_group
from rfm.optim import build_optimizer


class _FineTuningModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_adapter = nn.Module()
        self.input_adapter.suiren_projection = nn.Linear(4, 4)
        self.encoder = nn.Module()
        self.encoder.prior_conditioner = nn.Linear(4, 4)
        self.reg_head = nn.Linear(4, 2)


class OptimizerBuilderTest(unittest.TestCase):
    def test_stage_c_adamw_uses_discriminative_learning_rates(self) -> None:
        model = _FineTuningModel()
        args = SimpleNamespace(optimizer="adamw", lr=2e-4, weight_decay=1e-4)
        optimizer = build_optimizer(
            model.named_parameters(),
            args,
            parameter_group_resolver=lambda name, parameter: stage_c_parameter_group(
                name,
                parameter,
                0.1,
            ),
        )

        groups = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(groups), {"new_suiren", "stage_b_inherited"})
        self.assertAlmostEqual(groups["new_suiren"]["lr"], 2e-4)
        self.assertAlmostEqual(groups["stage_b_inherited"]["lr"], 2e-5)

        new_ids = {id(parameter) for parameter in groups["new_suiren"]["params"]}
        expected_new_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("input_adapter.suiren_") or ".prior_conditioner." in name
        }
        self.assertEqual(new_ids, expected_new_ids)

        metadata = optimizer.metadata["parameter_groups"]
        metadata_by_name = {group["name"]: group for group in metadata}
        self.assertAlmostEqual(metadata_by_name["new_suiren"]["adamw_lr"], 2e-4)
        self.assertAlmostEqual(metadata_by_name["stage_b_inherited"]["adamw_lr"], 2e-5)
        self.assertEqual(
            sum(group["parameters"] for group in metadata),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_stage_c_parameter_group_ignores_ddp_module_prefix(self) -> None:
        parameter = nn.Parameter(torch.ones(1))
        self.assertEqual(
            stage_c_parameter_group("module.input_adapter.suiren_gate", parameter, 0.1),
            ("new_suiren", 1.0),
        )
        self.assertEqual(
            stage_c_parameter_group("module.reg_head.weight", parameter, 0.1),
            ("stage_b_inherited", 0.1),
        )

    def test_stage_b_parameter_group_scales_all_parameters(self) -> None:
        parameter = nn.Parameter(torch.ones(1))
        self.assertEqual(
            stage_b_parameter_group("module.encoder.layers.0.weight", parameter, 0.1),
            ("stage_b_inherited", 0.1),
        )
        self.assertEqual(
            stage_b_parameter_group("module.reg_head.weight", parameter, 0.1),
            ("stage_b_inherited", 0.1),
        )


if __name__ == "__main__":
    unittest.main()
