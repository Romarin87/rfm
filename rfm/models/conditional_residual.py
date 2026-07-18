"""Frozen Stage B conditional residual probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .task_models import ReactionPropertyRegressor


def _config_value(config: dict[str, Any], name: str, default: Any) -> Any:
    return config.get(name, default)


def load_frozen_stage_b(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ReactionPropertyRegressor, dict[str, Any], torch.Tensor, torch.Tensor]:
    """Rebuild and strictly restore a complete Stage B property model."""

    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage B checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Stage B checkpoint must contain a complete model state: {checkpoint}")
    config = dict(payload.get("config", {}))
    if not config:
        raise ValueError(f"Stage B checkpoint is missing its training config: {checkpoint}")
    if config.get("geometry_mode") == "irc_rp":
        input_schema, pair_raw_dim = "property_irc_rp_bo", 4
    elif config.get("geometry_mode") == "2d":
        input_schema, pair_raw_dim = "property_rp2d_bo", 2
    else:
        raise ValueError(f"unsupported Stage B geometry_mode={config.get('geometry_mode')!r}")

    model = ReactionPropertyRegressor(
        pair_raw_dim=pair_raw_dim,
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
        input_schema=input_schema,
        dynamic_pair_update=bool(_config_value(config, "enable_dynamic_pair_update", False)),
        dynamic_pair_update_scale=float(_config_value(config, "dynamic_pair_update_scale", 1.0)),
        dynamic_pair_update_dropout=_config_value(config, "dynamic_pair_update_dropout", None),
        attention_readout=bool(_config_value(config, "enable_attention_readout", False)),
        directional_3d_adapter=bool(_config_value(config, "enable_directional_3d_adapter", False)),
        encoder_type=str(config["encoder_type"]),
        radar_attention_heads=int(_config_value(config, "radar_attention_heads", 8)),
        radar_center_router=bool(_config_value(config, "radar_center_router", True)),
        radar_delta_stream=bool(_config_value(config, "radar_delta_stream", True)),
        radar_pair_update_scale=float(_config_value(config, "radar_pair_update_scale", 0.75)),
        radar_reaction_update_scale=float(_config_value(config, "radar_reaction_update_scale", 1.0)),
        radar_router_gate_init=float(_config_value(config, "radar_router_gate_init", 0.05)),
        mrto_attention_heads=int(_config_value(config, "mrto_attention_heads", 8)),
        mrto_event_slots=int(_config_value(config, "mrto_event_slots", 4)),
        mrto_use_event_slots=bool(_config_value(config, "mrto_use_event_slots", True)),
        mrto_use_odd_field=bool(_config_value(config, "mrto_use_odd_field", True)),
        mrto_pair_update_scale=float(_config_value(config, "mrto_pair_update_scale", 1.0)),
        mrto_reaction_update_scale=float(_config_value(config, "mrto_reaction_update_scale", 1.0)),
        mrto_endpoint_layers=int(_config_value(config, "mrto_endpoint_layers", 2)),
        mrto_triangle_layers=int(_config_value(config, "mrto_triangle_layers", 2)),
        mrto_triangle_dim=int(_config_value(config, "mrto_triangle_dim", 16)),
        mrto_triangle_scale=float(_config_value(config, "mrto_triangle_scale", 0.5)),
        mrto_event_topk=int(_config_value(config, "mrto_event_topk", 0)),
        mrto_event_feedback_scale=float(_config_value(config, "mrto_event_feedback_scale", 0.5)),
        mrto_geometry_rbf_bins=int(_config_value(config, "mrto_geometry_rbf_bins", 16)),
        mrto_equiformer_layers=int(_config_value(config, "mrto_equiformer_layers", 2)),
        mrto_equiformer_channels=int(_config_value(config, "mrto_equiformer_channels", 32)),
        mrto_equiformer_lmax=int(_config_value(config, "mrto_equiformer_lmax", 2)),
        mrto_equiformer_radius=float(_config_value(config, "mrto_equiformer_radius", 5.0)),
        mrto_equiformer_max_neighbors=int(_config_value(config, "mrto_equiformer_max_neighbors", 64)),
        joint_encoder_pass=False,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.eval()

    y_mean = torch.as_tensor(payload["target_mean"], dtype=torch.float32, device=device)
    y_std = torch.as_tensor(payload["target_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return model, config, y_mean, y_std


class ConditionalResidualProbe(nn.Module):
    """Predict a normalized correction on top of a frozen Stage B model.

    Both probe modes have exactly the same trainable modules. The control mode
    replaces Suiren atom states with zeros before the shared projector.
    """

    MODES = ("stageb_only", "stageb_suiren")

    def __init__(
        self,
        stage_b: nn.Module,
        hidden_dim: int,
        suiren_atom_dim: int,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        mode: str,
        residual_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported residual probe mode={mode!r}")
        if suiren_atom_dim <= 0 or suiren_atom_dim % 4:
            raise ValueError("Suiren atom features must contain [h_R, h_P, Delta_h, abs_Delta_h]")
        self.stage_b = stage_b
        self.stage_b.requires_grad_(False)
        self.stage_b.eval()
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.suiren_state_dim = int(suiren_atom_dim // 4)
        residual_hidden_dim = int(residual_hidden_dim or hidden_dim)

        self.base_plus_norm = nn.LayerNorm(hidden_dim)
        self.base_minus_norm = nn.LayerNorm(hidden_dim)
        self.suiren_state_norm = nn.LayerNorm(self.suiren_state_dim, elementwise_affine=False)
        self.suiren_project = nn.Linear(self.suiren_state_dim, hidden_dim, bias=False)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, residual_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden_dim, 2),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("target_mean", y_mean.detach().clone().float())
        self.register_buffer("target_std", y_std.detach().clone().float().clamp_min(1e-6))

    def train(self, mode: bool = True) -> ConditionalResidualProbe:
        super().train(mode)
        self.stage_b.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def residual_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("stage_b.")
        }

    def load_residual_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)

    def _pool_suiren_states(
        self,
        features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = self.suiren_state_dim
        h_r = features[..., :state_dim]
        h_p = features[..., state_dim : 2 * state_dim]
        if self.mode == "stageb_only":
            h_r = torch.zeros_like(h_r)
            h_p = torch.zeros_like(h_p)
        mask = atom_mask.bool().unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1).to(features.dtype)

        def pool(state: torch.Tensor) -> torch.Tensor:
            projected = self.suiren_project(self.suiren_state_norm(state))
            return projected.masked_fill(~mask, 0.0).sum(dim=1) / denominator

        return pool(h_r), pool(h_p)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "suiren_3d_atom_features" not in batch:
            raise KeyError("conditional residual probe requires suiren_3d_atom_features")
        property_batch = {
            key: value
            for key, value in batch.items()
            if not key.startswith("suiren_") and not key.startswith("masked_")
        }
        with torch.no_grad():
            base = self.stage_b(property_batch)
        if "mrto_reaction_plus" not in base or "mrto_reaction_minus" not in base:
            raise ValueError("conditional residual probe requires MRTO parity reaction states")

        suiren_r, suiren_p = self._pool_suiren_states(
            batch["suiren_3d_atom_features"],
            batch["atom_mask"],
        )
        context = torch.cat(
            [
                self.base_plus_norm(base["mrto_reaction_plus"].detach()),
                self.base_minus_norm(base["mrto_reaction_minus"].detach()),
                suiren_r,
                suiren_p,
            ],
            dim=-1,
        )
        residual_norm = self.residual_head(context)
        baseline_y = base["y"].detach()
        prediction = baseline_y + residual_norm * self.target_std
        return {
            "y": prediction,
            "baseline_y": baseline_y,
            "residual_norm": residual_norm,
            "y_is_raw": True,
        }
