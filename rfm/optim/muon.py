"""Muon optimizer implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist


def zeropower_via_newton_schulz(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    ns_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Approximate the polar factor of a 2D update matrix.

    Muon applies momentum in parameter space and then orthogonalizes matrix
    updates. This uses the quintic Newton-Schulz polynomial commonly used in
    public Muon implementations.
    """

    if g.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D tensor.")
    original_dtype = g.dtype
    compute_dtype = ns_dtype or torch.bfloat16
    if g.device.type == "cpu" and compute_dtype == torch.bfloat16:
        compute_dtype = torch.float32
    x = g.to(dtype=compute_dtype)
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False
    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D matrix parameters.

    The implementation supports DDP compute sharding: every rank owns a stable
    subset of matrix parameters for the orthogonalization work, then broadcasts
    the resulting update so all model replicas remain identical.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.005,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
        ns_dtype: torch.dtype | None = None,
        distributed: bool = True,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
            "ns_dtype": ns_dtype,
            "distributed": distributed,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            ns_dtype = group["ns_dtype"]
            use_distributed = bool(group["distributed"]) and world_size > 1
            for param_index, param in enumerate(group["params"]):
                grad = param.grad
                if grad is None:
                    continue
                if grad.ndim != 2:
                    raise RuntimeError("Muon received a non-2D parameter; use hybrid parameter grouping.")
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                owner = param_index % world_size
                if not use_distributed or owner == rank:
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buffer = state["momentum_buffer"]
                    buffer.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    update = grad.lerp(buffer, momentum) if nesterov else buffer
                    update = zeropower_via_newton_schulz(update, steps=ns_steps, ns_dtype=ns_dtype)
                else:
                    update = torch.empty_like(param)
                if use_distributed:
                    dist.broadcast(update, src=owner)
                scale = max(1.0, (param.size(0) / max(1, param.size(1))) ** 0.5)
                param.add_(update, alpha=-lr * scale)
        return loss

