"""Task models for the current RFM restart path."""

from __future__ import annotations

import torch
import torch.nn as nn

from rfm.data.reaction_samples import EDIT_CLASSES, ENERGY_TARGETS
from rfm.features.token_space import MaskedEditAdapter, RPairPropertyAdapter, TokenSpaceReactionEncoder, UnifiedReactionInputAdapter


class MaskedEditHeads(nn.Module):
    """Masked Delta_BO, changed pair, edit class, and core atom heads."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.delta_head = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.changed_head = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.edit_head = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(EDIT_CLASSES)))
        self.core_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, atom_h: torch.Tensor, pair_h: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, n_atoms, hidden = atom_h.shape
        h_i = atom_h.unsqueeze(2).expand(batch, n_atoms, n_atoms, hidden)
        h_j = atom_h.unsqueeze(1).expand(batch, n_atoms, n_atoms, hidden)
        pair_repr = torch.cat([h_i, h_j, pair_h], dim=-1)
        return {
            "delta_bo": self.delta_head(pair_repr).squeeze(-1),
            "changed_logits": self.changed_head(pair_repr).squeeze(-1),
            "edit_logits": self.edit_head(pair_repr),
            "core_logits": self.core_head(atom_h).squeeze(-1),
        }


class MaskedEditPretrainingModel(nn.Module):
    """Masked edit model with strict raw inputs: Z + masked BO/edit basis + optional D_R."""

    def __init__(self, pair_raw_dim: int, hidden_dim: int, layers: int, dropout: float, input_schema: str):
        super().__init__()
        self.adapter = MaskedEditAdapter(
            hidden_dim=hidden_dim,
            pair_input_dim=pair_raw_dim,
            input_schema=input_schema,
        )
        self.encoder = TokenSpaceReactionEncoder(hidden_dim=hidden_dim, layers=layers, dropout=dropout)
        self.masked_edit_heads = MaskedEditHeads(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoder_input = self.adapter(batch)
        encoded = self.encoder(encoder_input)
        out = self.masked_edit_heads(encoded["atom_h"], encoded["pair_h"])
        out["reaction_h"] = encoded["reaction_h"]
        return out


class ReactionPropertyRegressor(nn.Module):
    """Reaction property regression model.

    Inputs are Z + BO_R + BO_P for 2D, and additionally IRC R/P endpoint
    D_R/D_P for the 3D variant.
    """

    def __init__(self, pair_raw_dim: int, hidden_dim: int, layers: int, dropout: float, input_schema: str):
        super().__init__()
        masked_schema, masked_pair_raw_dim = _masked_edit_schema_for_property(input_schema)
        self.adapter = RPairPropertyAdapter(
            hidden_dim=hidden_dim,
            pair_input_dim=pair_raw_dim,
            input_schema=input_schema,
        )
        self.masked_adapter = MaskedEditAdapter(
            hidden_dim=hidden_dim,
            pair_input_dim=masked_pair_raw_dim,
            input_schema=masked_schema,
        )
        self.encoder = TokenSpaceReactionEncoder(hidden_dim=hidden_dim, layers=layers, dropout=dropout)
        self.masked_edit_heads = MaskedEditHeads(hidden_dim)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ENERGY_TARGETS)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoder_input = self.adapter(batch)
        encoded = self.encoder(encoder_input)
        atom_h = encoded["atom_h"]
        atom_mask = batch["atom_mask"].bool()
        masked_h = atom_h.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        mean_pool = masked_h.sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
        y = self.reg_head(torch.cat([encoded["reaction_h"], mean_pool], dim=-1))

        out = {
            "y": y,
            "reaction_h": encoded["reaction_h"],
        }
        if "masked_pair_input" in batch:
            masked_out = self._masked_forward(batch)
            out.update({f"masked_{key}": value for key, value in masked_out.items()})
        return out

    def _masked_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        masked_batch = {
            "z": batch["z"],
            "atom_mask": batch["atom_mask"],
            "pair_input": batch["masked_pair_input"],
            "pair_valid": batch["masked_pair_valid"],
        }
        encoded = self.encoder(self.masked_adapter(masked_batch))
        out = self.masked_edit_heads(encoded["atom_h"], encoded["pair_h"])
        out["reaction_h"] = encoded["reaction_h"]
        return out


class SuirenFusionPropertyRegressor(nn.Module):
    """Reaction property regression with frozen Suiren graph/atom cache inputs."""

    def __init__(
        self,
        pair_raw_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        input_schema: str,
        suiren_atom_dim: int = 0,
        suiren_graph_dim: int = 0,
        suiren_atom_dims: dict[str, int] | None = None,
        suiren_graph_dims: dict[str, int] | None = None,
    ):
        super().__init__()
        masked_schema, masked_pair_raw_dim = _masked_edit_schema_for_property(input_schema)
        self.input_adapter = UnifiedReactionInputAdapter(
            hidden_dim=hidden_dim,
            pair_input_dim=pair_raw_dim,
            input_schema=input_schema,
            suiren_atom_dim=suiren_atom_dim,
            suiren_graph_dim=suiren_graph_dim,
            suiren_atom_dims=suiren_atom_dims,
            suiren_graph_dims=suiren_graph_dims,
        )
        self.masked_adapter = MaskedEditAdapter(
            hidden_dim=hidden_dim,
            pair_input_dim=masked_pair_raw_dim,
            input_schema=masked_schema,
        )
        self.encoder = TokenSpaceReactionEncoder(hidden_dim=hidden_dim, layers=layers, dropout=dropout)
        self.masked_edit_heads = MaskedEditHeads(hidden_dim)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ENERGY_TARGETS)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in batch.items():
            if key.startswith("suiren_") and key.endswith("_failed") and bool(value.bool().any().item()):
                raise ValueError(f"Suiren cache contains failed rows in {key}; rebuild the cache before training or evaluation")

        encoder_input = self.input_adapter(batch)
        encoded = self.encoder(encoder_input)
        atom_h = encoded["atom_h"]
        atom_mask = batch["atom_mask"].bool()
        masked_h = atom_h.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        mean_pool = masked_h.sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
        y = self.reg_head(torch.cat([encoded["reaction_h"], mean_pool], dim=-1))

        out = {
            "y": y,
            "reaction_h": encoded["reaction_h"],
        }
        if "masked_pair_input" in batch:
            masked_out = self._masked_forward(batch)
            out.update({f"masked_{key}": value for key, value in masked_out.items()})
        return out

    def _masked_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        masked_batch = {
            "z": batch["z"],
            "atom_mask": batch["atom_mask"],
            "pair_input": batch["masked_pair_input"],
            "pair_valid": batch["masked_pair_valid"],
        }
        encoded = self.encoder(self.masked_adapter(masked_batch))
        out = self.masked_edit_heads(encoded["atom_h"], encoded["pair_h"])
        out["reaction_h"] = encoded["reaction_h"]
        return out


def _masked_edit_schema_for_property(input_schema: str) -> tuple[str, int]:
    if input_schema.startswith("property_irc_rp"):
        return "masked_edit_irc_rp", 4
    return "masked_edit_2d", 3


def load_pretrained_encoder(model: nn.Module, path: str, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["encoder"] if isinstance(payload, dict) and "encoder" in payload else payload
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if not key.startswith("reaction_update") and not key.startswith("reaction_norm")]
    if bad_missing or unexpected:
        raise RuntimeError(f"pretrained encoder load mismatch: missing={bad_missing[:10]} unexpected={unexpected[:10]}")
