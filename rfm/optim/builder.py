"""Optimizer construction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .muon import Muon


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


def build_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    args: Any,
) -> torch.optim.Optimizer | CompositeOptimizer:
    params = [(name, param) for name, param in named_parameters if param.requires_grad]
    optimizer_name = getattr(args, "optimizer", "adamw")
    if optimizer_name == "adamw":
        return torch.optim.AdamW((param for _, param in params), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    muon_params = [param for name, param in params if _is_muon_matrix(name, param)]
    adamw_params = [param for name, param in params if not _is_muon_matrix(name, param)]
    optimizers: list[torch.optim.Optimizer] = []
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay))
    if muon_params:
        ns_dtype_name = getattr(args, "muon_ns_dtype", "bfloat16")
        ns_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[ns_dtype_name]
        optimizers.append(
            Muon(
                muon_params,
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
        "adamw_param_tensors": len(adamw_params),
        "muon_param_tensors": len(muon_params),
        "muon_lr": args.muon_lr,
        "muon_momentum": args.muon_momentum,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_ns_dtype": getattr(args, "muon_ns_dtype", "bfloat16"),
        "muon_distributed": getattr(args, "muon_distributed", True),
        "muon_variant": "quintic_newton_schulz_distributed",
        "non_matrix_fallback": "adamw",
    }
    return CompositeOptimizer(optimizers, metadata)
