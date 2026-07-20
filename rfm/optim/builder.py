"""Optimizer construction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import torch

from .muon import Muon


ParameterGroupResolver = Callable[[str, torch.nn.Parameter], tuple[str, float]]


class CompositeOptimizer:
    """Small wrapper that steps multiple PyTorch optimizers together."""

    def __init__(self, optimizers: list[torch.optim.Optimizer], metadata: dict[str, Any]):
        self.optimizers = optimizers
        self.metadata = metadata
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "metadata": self.metadata,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer, sub_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(sub_state)
        self.metadata = state_dict.get("metadata", self.metadata)


def _is_muon_matrix(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim != 2:
        return False
    lower = name.lower()
    if any(skip in lower for skip in ("embed", "embedding", "norm", "bias")):
        return False
    return True


def _resolve_parameter_groups(
    params: list[tuple[str, torch.nn.Parameter]],
    resolver: ParameterGroupResolver | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, param in params:
        group_name, lr_scale = (
            resolver(name, param) if resolver is not None else ("default", 1.0)
        )
        lr_scale = float(lr_scale)
        if lr_scale <= 0.0:
            raise ValueError(
                f"optimizer lr scale must be positive, got {lr_scale} "
                f"for group {group_name!r}"
            )
        if group_name in grouped and grouped[group_name]["lr_scale"] != lr_scale:
            raise ValueError(
                f"optimizer group {group_name!r} resolved to inconsistent lr scales: "
                f"{grouped[group_name]['lr_scale']} and {lr_scale}"
            )
        group = grouped.setdefault(
            group_name,
            {"name": group_name, "lr_scale": lr_scale, "named_params": []},
        )
        group["named_params"].append((name, param))
    return list(grouped.values())


def _group_metadata(
    groups: list[dict[str, Any]],
    base_lr: float,
    muon_lr: float | None = None,
) -> list[dict[str, Any]]:
    metadata = []
    for group in groups:
        named_params = group["named_params"]
        item = {
            "name": group["name"],
            "lr_scale": group["lr_scale"],
            "adamw_lr": base_lr * group["lr_scale"],
            "param_tensors": len(named_params),
            "parameters": sum(param.numel() for _, param in named_params),
        }
        if muon_lr is not None:
            item["muon_lr"] = muon_lr * group["lr_scale"]
        metadata.append(item)
    return metadata


def build_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    args: Any,
    parameter_group_resolver: ParameterGroupResolver | None = None,
) -> torch.optim.Optimizer | CompositeOptimizer:
    params = [(name, param) for name, param in named_parameters if param.requires_grad]
    groups = _resolve_parameter_groups(params, parameter_group_resolver)
    optimizer_name = getattr(args, "optimizer", "adamw")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [param for _, param in group["named_params"]],
                    "lr": args.lr * group["lr_scale"],
                    "group_name": group["name"],
                }
                for group in groups
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizer.metadata = {
            "optimizer": "adamw",
            "base_lr": args.lr,
            "weight_decay": args.weight_decay,
            "parameter_groups": _group_metadata(groups, args.lr),
        }
        return optimizer
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    adamw_groups = []
    muon_groups = []
    for group in groups:
        adamw_params = [
            param for name, param in group["named_params"] if not _is_muon_matrix(name, param)
        ]
        muon_params = [
            param for name, param in group["named_params"] if _is_muon_matrix(name, param)
        ]
        if adamw_params:
            adamw_groups.append(
                {
                    "params": adamw_params,
                    "lr": args.lr * group["lr_scale"],
                    "group_name": group["name"],
                }
            )
        if muon_params:
            muon_groups.append(
                {
                    "params": muon_params,
                    "lr": args.muon_lr * group["lr_scale"],
                    "group_name": group["name"],
                }
            )
    optimizers: list[torch.optim.Optimizer] = []
    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW(adamw_groups, lr=args.lr, weight_decay=args.weight_decay)
        )
    if muon_groups:
        ns_dtype_name = getattr(args, "muon_ns_dtype", "bfloat16")
        ns_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[ns_dtype_name]
        optimizers.append(
            Muon(
                muon_groups,
                lr=args.muon_lr,
                momentum=args.muon_momentum,
                weight_decay=args.muon_weight_decay,
                ns_steps=args.muon_ns_steps,
                ns_dtype=ns_dtype,
                distributed=getattr(args, "muon_distributed", True),
            )
        )
    metadata = {
        "optimizer": "muon",
        "adamw_param_tensors": sum(len(group["params"]) for group in adamw_groups),
        "muon_param_tensors": sum(len(group["params"]) for group in muon_groups),
        "muon_lr": args.muon_lr,
        "muon_momentum": args.muon_momentum,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_ns_dtype": getattr(args, "muon_ns_dtype", "bfloat16"),
        "muon_distributed": getattr(args, "muon_distributed", True),
        "muon_variant": "quintic_newton_schulz_distributed",
        "non_matrix_fallback": "adamw",
        "parameter_groups": _group_metadata(groups, args.lr, args.muon_lr),
    }
    return CompositeOptimizer(optimizers, metadata)
