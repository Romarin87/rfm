#!/usr/bin/env python
"""Canonical token-space interfaces for RFM.

Different tasks keep their own raw inputs, but every task-facing adapter must
produce the same `RFMEncoderInput` shape before the shared Reaction Encoder
trunk or task heads consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


DEFAULT_MODALITIES = (
    "has_R",
    "has_P",
    "has_R_3D",
    "has_P_3D",
    "has_partial_Delta_BO",
    "has_Delta_BO",
    "has_Suiren_R",
    "has_Suiren_P",
)


def pairwise_distance(coords: np.ndarray, clip: float | None = None, normalize: bool = False) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=-1).astype(np.float32)
    if clip is not None:
        distances = np.clip(distances, 0.0, float(clip))
        if normalize:
            distances = distances / max(float(clip), 1e-6)
    return distances


def edit_class_from_bo(bo_r: np.ndarray, bo_p: np.ndarray, bond_eps: float = 1e-6) -> np.ndarray:
    delta_bo = bo_p - bo_r
    cls = np.zeros_like(delta_bo, dtype=np.int64)
    cls[(bo_r <= bond_eps) & (bo_p > bond_eps)] = 1
    cls[(bo_r > bond_eps) & (bo_p <= bond_eps)] = 2
    cls[(bo_r > bond_eps) & (bo_p > bond_eps) & (np.abs(delta_bo) > bond_eps)] = 3
    return cls


def changed_pairs(delta_bo: np.ndarray, bond_eps: float = 1e-6) -> np.ndarray:
    return (np.abs(delta_bo) > bond_eps).astype(np.float32)


def reaction_core_from_delta(delta_bo: np.ndarray, bond_eps: float = 1e-6) -> np.ndarray:
    return (np.abs(delta_bo).sum(axis=1) > bond_eps).astype(np.float32)


def generate_visibility_mask(
    delta_bo: np.ndarray,
    *,
    strategy: str,
    pair_mask_ratio: float,
    changed_pair_mask_ratio: float,
    seed: int,
) -> np.ndarray:
    """Return `visibility`: 1 means visible, 0 means masked target."""
    n_atoms = delta_bo.shape[0]
    pair_valid = np.ones((n_atoms, n_atoms), dtype=bool)
    np.fill_diagonal(pair_valid, False)
    upper = np.triu(pair_valid, k=1)
    changed = (np.abs(delta_bo) > 1e-6) & upper
    rng = np.random.default_rng(seed)

    target = np.zeros((n_atoms, n_atoms), dtype=bool)
    if strategy == "reaction_center":
        target |= changed
        if pair_mask_ratio > 0:
            target |= (rng.random((n_atoms, n_atoms)) < pair_mask_ratio) & upper & (~changed)
    elif strategy == "changed_enriched":
        target |= (rng.random((n_atoms, n_atoms)) < pair_mask_ratio) & upper
        target |= changed & (rng.random((n_atoms, n_atoms)) < changed_pair_mask_ratio)
    elif strategy == "random_pair":
        target |= (rng.random((n_atoms, n_atoms)) < pair_mask_ratio) & upper
    else:
        raise ValueError(f"unsupported mask strategy: {strategy}")

    if not target.any():
        choices = np.argwhere(changed if changed.any() else upper)
        if len(choices):
            i, j = choices[rng.integers(0, len(choices))]
            target[i, j] = True
    target = target | target.T
    return (~target).astype(np.float32)


def pair_valid_matrix(n_atoms: int) -> np.ndarray:
    pair_valid = np.ones((n_atoms, n_atoms), dtype=bool)
    np.fill_diagonal(pair_valid, False)
    return pair_valid


@dataclass(frozen=True)
class ReactionPairBasisSpec:
    schema: str
    names: tuple[str, ...]
    has_r: bool = True
    has_p: bool = False
    has_r_3d: bool = False
    has_p_3d: bool = False
    has_partial_delta_bo: bool = False
    has_delta_bo: bool = False


PAIR_BASIS_SPECS: dict[str, ReactionPairBasisSpec] = {
    "masked_edit_2d": ReactionPairBasisSpec(
        schema="masked_edit_2d",
        names=(
            "BO_R",
            "BO_P_visible",
            "Delta_BO_visible",
            "abs_Delta_BO_visible",
            "visibility",
            "A_R",
            "A_P_visible",
            "Delta_A_P_visible",
            "formed_visible",
            "broken_visible",
            "order_changed_visible",
        ),
        has_partial_delta_bo=True,
    ),
    "masked_edit_irc_rp": ReactionPairBasisSpec(
        schema="masked_edit_irc_rp",
        names=(
            "BO_R",
            "BO_P_visible",
            "Delta_BO_visible",
            "abs_Delta_BO_visible",
            "visibility",
            "A_R",
            "A_P_visible",
            "Delta_A_P_visible",
            "formed_visible",
            "broken_visible",
            "order_changed_visible",
            "D_R",
        ),
        has_r_3d=True,
        has_partial_delta_bo=True,
    ),
    "property_r2d_bo": ReactionPairBasisSpec(
        schema="property_r2d_bo",
        names=("BO_R", "A_R"),
    ),
    "property_r3d_bo": ReactionPairBasisSpec(
        schema="property_r3d_bo",
        names=("BO_R", "A_R", "D_R"),
        has_r_3d=True,
    ),
    "product_edit_r2d_bo": ReactionPairBasisSpec(
        schema="product_edit_r2d_bo",
        names=("BO_R", "A_R"),
    ),
    "product_edit_r3d_bo": ReactionPairBasisSpec(
        schema="product_edit_r3d_bo",
        names=("BO_R", "A_R", "D_R"),
        has_r_3d=True,
    ),
    "property_rp2d_bo": ReactionPairBasisSpec(
        schema="property_rp2d_bo",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
        ),
        has_p=True,
        has_delta_bo=True,
    ),
    "property_rp2d_bo_delta": ReactionPairBasisSpec(
        schema="property_rp2d_bo_delta",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
        ),
        has_p=True,
        has_delta_bo=True,
    ),
    "property_rp2d_r_delta": ReactionPairBasisSpec(
        schema="property_rp2d_r_delta",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
        ),
        has_p=True,
        has_delta_bo=True,
    ),
    "property_rp2d_p_delta": ReactionPairBasisSpec(
        schema="property_rp2d_p_delta",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
        ),
        has_p=True,
        has_delta_bo=True,
    ),
    "property_irc_rp_bo": ReactionPairBasisSpec(
        schema="property_irc_rp_bo",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
            "D_R",
            "D_P",
            "Delta_D",
            "abs_Delta_D",
        ),
        has_p=True,
        has_r_3d=True,
        has_p_3d=True,
        has_delta_bo=True,
    ),
    "property_irc_rp_bo_delta": ReactionPairBasisSpec(
        schema="property_irc_rp_bo_delta",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
            "D_R",
            "D_P",
            "Delta_D",
            "abs_Delta_D",
        ),
        has_p=True,
        has_r_3d=True,
        has_p_3d=True,
        has_delta_bo=True,
    ),
    "property_irc_rp_r_delta": ReactionPairBasisSpec(
        schema="property_irc_rp_r_delta",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
            "D_R",
            "D_P",
            "Delta_D",
            "abs_Delta_D",
        ),
        has_p=True,
        has_r_3d=True,
        has_p_3d=True,
        has_delta_bo=True,
    ),
    "property_irc_rp_p_delta": ReactionPairBasisSpec(
        schema="property_irc_rp_p_delta",
        names=(
            "BO_R",
            "BO_P",
            "Delta_BO",
            "abs_Delta_BO",
            "A_R",
            "A_P",
            "Delta_A",
            "formed",
            "broken",
            "order_changed",
            "D_R",
            "D_P",
            "Delta_D",
            "abs_Delta_D",
        ),
        has_p=True,
        has_r_3d=True,
        has_p_3d=True,
        has_delta_bo=True,
    ),
}


def infer_pair_basis_schema(pair_input_dim: int, task_name: str) -> str:
    """Infer a raw-input schema for compact CLI arguments.

    Task modules should prefer explicit schema names. This helper exists only
    to keep adapter construction concise inside the current package.
    """
    if task_name == "masked_edit":
        if pair_input_dim == 3:
            return "masked_edit_2d"
        if pair_input_dim == 4:
            return "masked_edit_irc_rp"
    if task_name == "property":
        if pair_input_dim == 1:
            return "property_r2d_bo"
        if pair_input_dim == 2:
            return "property_rp2d_bo"
        if pair_input_dim == 3:
            return "property_rp2d_bo_delta"
        if pair_input_dim == 4:
            return "property_irc_rp_bo"
        if pair_input_dim == 5:
            return "property_irc_rp_bo_delta"
    raise ValueError(f"Cannot infer pair basis schema for task={task_name!r}, pair_input_dim={pair_input_dim}")


class ReactionInputFeaturizer(nn.Module):
    """Canonical deterministic raw-input-to-basis featurizer.

    This module does not add new information. It rewrites visible raw inputs
    into a common reaction-friendly basis, such as `Delta_BO = BO_P - BO_R`,
    adjacency indicators, and optional normalized R/P distance features.
    Masked edit inputs only use visible `Delta_BO` values, so masked targets are
    not reconstructed.
    """

    def __init__(self, schema: str, distance_clip: float = 10.0, bond_eps: float = 1e-6):
        super().__init__()
        if schema not in PAIR_BASIS_SPECS:
            raise ValueError(f"Unsupported reaction input schema: {schema}")
        self.schema = schema
        self.distance_clip = float(distance_clip)
        self.bond_eps = float(bond_eps)

    @property
    def spec(self) -> ReactionPairBasisSpec:
        return PAIR_BASIS_SPECS[self.schema]

    @property
    def output_dim(self) -> int:
        return len(self.spec.names)

    def _adjacency(self, bo: torch.Tensor) -> torch.Tensor:
        return (bo > self.bond_eps).to(dtype=bo.dtype)

    def _distance_basis(self, d_raw: torch.Tensor, already_normalized: bool = False) -> torch.Tensor:
        if already_normalized:
            d_norm = d_raw.clamp(0.0, 1.0)
        else:
            d_angstrom = d_raw.clamp(0.0, self.distance_clip)
            d_norm = d_angstrom / max(self.distance_clip, 1e-6)
        return d_norm

    def _bo_edit_basis(self, bo_r: torch.Tensor, bo_p: torch.Tensor, delta_bo: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        if delta_bo is None:
            delta_bo = bo_p - bo_r
        a_r = self._adjacency(bo_r)
        a_p = self._adjacency(bo_p)
        delta_a = a_p - a_r
        formed = ((a_r <= 0.0) & (a_p > 0.0)).to(dtype=bo_r.dtype)
        broken = ((a_r > 0.0) & (a_p <= 0.0)).to(dtype=bo_r.dtype)
        order_changed = ((a_r > 0.0) & (a_p > 0.0) & (delta_bo.abs() > self.bond_eps)).to(dtype=bo_r.dtype)
        return bo_r, bo_p, delta_bo, delta_bo.abs(), a_r, a_p, delta_a, formed, broken, order_changed

    def forward(self, pair_raw: torch.Tensor) -> torch.Tensor:
        if pair_raw.ndim != 4:
            raise ValueError(f"pair_raw must be [B,N,N,C], got {tuple(pair_raw.shape)}")

        if self.schema == "masked_edit_2d":
            bo_r = pair_raw[..., 0]
            visible_delta = pair_raw[..., 1]
            visible = pair_raw[..., 2].clamp(0.0, 1.0)
            bo_p_visible = bo_r + visible_delta
            basis = self._bo_edit_basis(bo_r, bo_p_visible)
            return torch.stack([basis[0], basis[1], basis[2], basis[3], visible, *basis[4:]], dim=-1)

        if self.schema == "masked_edit_irc_rp":
            bo_r = pair_raw[..., 0]
            d_r = pair_raw[..., 1]
            visible_delta = pair_raw[..., 2]
            visible = pair_raw[..., 3].clamp(0.0, 1.0)
            bo_p_visible = bo_r + visible_delta
            basis = self._bo_edit_basis(bo_r, bo_p_visible)
            d_r_norm = self._distance_basis(d_r, already_normalized=True)
            return torch.stack([basis[0], basis[1], basis[2], basis[3], visible, *basis[4:], d_r_norm], dim=-1)

        if self.schema == "property_r2d_bo":
            bo_r = pair_raw[..., 0]
            return torch.stack([bo_r, self._adjacency(bo_r)], dim=-1)

        if self.schema == "property_r3d_bo":
            bo_r = pair_raw[..., 0]
            d_r_norm = self._distance_basis(pair_raw[..., 1], already_normalized=False)
            return torch.stack([bo_r, self._adjacency(bo_r), d_r_norm], dim=-1)

        if self.schema == "product_edit_r2d_bo":
            bo_r = pair_raw[..., 0]
            return torch.stack([bo_r, self._adjacency(bo_r)], dim=-1)

        if self.schema == "product_edit_r3d_bo":
            bo_r = pair_raw[..., 0]
            d_r_norm = self._distance_basis(pair_raw[..., 1], already_normalized=False)
            return torch.stack([bo_r, self._adjacency(bo_r), d_r_norm], dim=-1)

        if self.schema == "property_rp2d_bo":
            bo_r = pair_raw[..., 0]
            bo_p = pair_raw[..., 1]
            return torch.stack(self._bo_edit_basis(bo_r, bo_p), dim=-1)

        if self.schema == "property_rp2d_bo_delta":
            bo_r = pair_raw[..., 0]
            bo_p = pair_raw[..., 1]
            delta_bo = pair_raw[..., 2]
            return torch.stack(self._bo_edit_basis(bo_r, bo_p, delta_bo), dim=-1)

        if self.schema == "property_rp2d_r_delta":
            bo_r = pair_raw[..., 0]
            delta_bo = pair_raw[..., 1]
            bo_p = bo_r + delta_bo
            return torch.stack(self._bo_edit_basis(bo_r, bo_p, delta_bo), dim=-1)

        if self.schema == "property_rp2d_p_delta":
            bo_p = pair_raw[..., 0]
            delta_bo = pair_raw[..., 1]
            bo_r = bo_p - delta_bo
            return torch.stack(self._bo_edit_basis(bo_r, bo_p, delta_bo), dim=-1)

        if self.schema == "property_irc_rp_bo":
            bo_r = pair_raw[..., 0]
            bo_p = pair_raw[..., 1]
            d_r = self._distance_basis(pair_raw[..., 2], already_normalized=False)
            d_p = self._distance_basis(pair_raw[..., 3], already_normalized=False)
            delta_d = d_p - d_r
            return torch.stack([*self._bo_edit_basis(bo_r, bo_p), d_r, d_p, delta_d, delta_d.abs()], dim=-1)

        if self.schema == "property_irc_rp_bo_delta":
            bo_r = pair_raw[..., 0]
            bo_p = pair_raw[..., 1]
            delta_bo = pair_raw[..., 2]
            d_r = self._distance_basis(pair_raw[..., 3], already_normalized=False)
            d_p = self._distance_basis(pair_raw[..., 4], already_normalized=False)
            delta_d = d_p - d_r
            return torch.stack([*self._bo_edit_basis(bo_r, bo_p, delta_bo), d_r, d_p, delta_d, delta_d.abs()], dim=-1)

        if self.schema == "property_irc_rp_r_delta":
            bo_r = pair_raw[..., 0]
            delta_bo = pair_raw[..., 1]
            bo_p = bo_r + delta_bo
            d_r = self._distance_basis(pair_raw[..., 2], already_normalized=False)
            d_p = self._distance_basis(pair_raw[..., 3], already_normalized=False)
            delta_d = d_p - d_r
            return torch.stack([*self._bo_edit_basis(bo_r, bo_p, delta_bo), d_r, d_p, delta_d, delta_d.abs()], dim=-1)

        if self.schema == "property_irc_rp_p_delta":
            bo_p = pair_raw[..., 0]
            delta_bo = pair_raw[..., 1]
            bo_r = bo_p - delta_bo
            d_r = self._distance_basis(pair_raw[..., 2], already_normalized=False)
            d_p = self._distance_basis(pair_raw[..., 3], already_normalized=False)
            delta_d = d_p - d_r
            return torch.stack([*self._bo_edit_basis(bo_r, bo_p, delta_bo), d_r, d_p, delta_d, delta_d.abs()], dim=-1)

        raise AssertionError(f"Unhandled schema: {self.schema}")


@dataclass
class RFMEncoderInput:
    """Canonical input object shared by RFM adapters.

    Shapes:
      atom_tokens:      [B, N, H]
      pair_tokens:      [B, N, N, H]
      reaction_token:   [B, H]
      atom_valid_mask:  [B, N]
      pair_valid_mask:  [B, N, N]
      modality_mask:    modality name -> [B] bool tensor
    """

    atom_tokens: torch.Tensor
    pair_tokens: torch.Tensor
    reaction_token: torch.Tensor
    atom_valid_mask: torch.Tensor
    pair_valid_mask: torch.Tensor
    modality_mask: dict[str, torch.Tensor] = field(default_factory=dict)
    task_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.atom_tokens.ndim != 3:
            raise ValueError(f"atom_tokens must be [B,N,H], got {tuple(self.atom_tokens.shape)}")
        if self.pair_tokens.ndim != 4:
            raise ValueError(f"pair_tokens must be [B,N,N,H], got {tuple(self.pair_tokens.shape)}")
        if self.reaction_token.ndim != 2:
            raise ValueError(f"reaction_token must be [B,H], got {tuple(self.reaction_token.shape)}")
        batch, n_atoms, hidden = self.atom_tokens.shape
        if self.pair_tokens.shape != (batch, n_atoms, n_atoms, hidden):
            raise ValueError(
                "pair_tokens shape must match atom_tokens as [B,N,N,H], "
                f"got pair={tuple(self.pair_tokens.shape)} atom={tuple(self.atom_tokens.shape)}"
            )
        if self.reaction_token.shape != (batch, hidden):
            raise ValueError(f"reaction_token must be [B,H], got {tuple(self.reaction_token.shape)}")
        if self.atom_valid_mask.shape != (batch, n_atoms):
            raise ValueError(f"atom_valid_mask must be [B,N], got {tuple(self.atom_valid_mask.shape)}")
        if self.pair_valid_mask.shape != (batch, n_atoms, n_atoms):
            raise ValueError(f"pair_valid_mask must be [B,N,N], got {tuple(self.pair_valid_mask.shape)}")
        for name, mask in self.modality_mask.items():
            if mask.shape != (batch,):
                raise ValueError(f"modality_mask[{name}] must be [B], got {tuple(mask.shape)}")

    def to(self, device: torch.device | str) -> "RFMEncoderInput":
        return RFMEncoderInput(
            atom_tokens=self.atom_tokens.to(device),
            pair_tokens=self.pair_tokens.to(device),
            reaction_token=self.reaction_token.to(device),
            atom_valid_mask=self.atom_valid_mask.to(device),
            pair_valid_mask=self.pair_valid_mask.to(device),
            modality_mask={key: value.to(device) for key, value in self.modality_mask.items()},
            task_name=self.task_name,
            metadata=self.metadata,
        )


def valid_pair_mask(atom_valid_mask: torch.Tensor) -> torch.Tensor:
    """Build off-diagonal pair mask from atom mask."""
    pair = atom_valid_mask.unsqueeze(1) & atom_valid_mask.unsqueeze(2)
    n_atoms = atom_valid_mask.shape[1]
    diag = torch.eye(n_atoms, dtype=torch.bool, device=atom_valid_mask.device).unsqueeze(0)
    return pair & ~diag


def batch_bool(batch_size: int, value: bool, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), bool(value), dtype=torch.bool, device=device)


class TokenPairMessageLayer(nn.Module):
    """Message-passing layer over canonical atom/pair tokens."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        dynamic_pair_update: bool = True,
        dynamic_pair_update_scale: float = 0.75,
        dynamic_pair_update_dropout: float | None = None,
    ):
        super().__init__()
        self.dynamic_pair_update = bool(dynamic_pair_update)
        self.dynamic_pair_update_scale = float(dynamic_pair_update_scale)
        if self.dynamic_pair_update:
            pair_dropout = dropout if dynamic_pair_update_dropout is None else float(dynamic_pair_update_dropout)
            self.pair_update = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Dropout(pair_dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pair_norm = nn.LayerNorm(hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atom_h: torch.Tensor,
        pair_h: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        pair_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_atoms, hidden = atom_h.shape
        h_i = atom_h.unsqueeze(2).expand(batch, n_atoms, n_atoms, hidden)
        h_j = atom_h.unsqueeze(1).expand(batch, n_atoms, n_atoms, hidden)
        if self.dynamic_pair_update:
            pair_repr = torch.cat([h_i, h_j, pair_h], dim=-1)
            pair_h = self.pair_norm(pair_h + self.dynamic_pair_update_scale * self.pair_update(pair_repr))
            pair_h = pair_h * pair_valid_mask.unsqueeze(-1)
        msg = self.message(torch.cat([h_i, h_j, pair_h], dim=-1))
        msg = msg * pair_valid_mask.unsqueeze(-1)
        denom = pair_valid_mask.sum(dim=2).clamp_min(1).float().unsqueeze(-1)
        agg = msg.sum(dim=2) / denom
        atom_h = self.norm(atom_h + self.update(torch.cat([atom_h, agg], dim=-1)))
        return atom_h * atom_valid_mask.unsqueeze(-1), pair_h


class TokenSpaceReactionEncoder(nn.Module):
    """Shared trunk that consumes `RFMEncoderInput`, independent of raw modality."""

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        dropout: float,
        dynamic_pair_update: bool = True,
        dynamic_pair_update_scale: float = 0.75,
        dynamic_pair_update_dropout: float | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TokenPairMessageLayer(
                    hidden_dim,
                    dropout,
                    dynamic_pair_update=dynamic_pair_update,
                    dynamic_pair_update_scale=dynamic_pair_update_scale,
                    dynamic_pair_update_dropout=dynamic_pair_update_dropout,
                )
                for _ in range(layers)
            ]
        )
        self.reaction_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reaction_norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_input: RFMEncoderInput) -> dict[str, torch.Tensor]:
        encoder_input.validate()
        atom_h = encoder_input.atom_tokens * encoder_input.atom_valid_mask.unsqueeze(-1)
        pair_h = encoder_input.pair_tokens * encoder_input.pair_valid_mask.unsqueeze(-1)
        for layer in self.layers:
            atom_h, pair_h = layer(atom_h, pair_h, encoder_input.atom_valid_mask, encoder_input.pair_valid_mask)
        denom = encoder_input.atom_valid_mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
        pooled = (atom_h * encoder_input.atom_valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        reaction_h = self.reaction_norm(
            encoder_input.reaction_token + self.reaction_update(torch.cat([encoder_input.reaction_token, pooled], dim=-1))
        )
        return {"atom_h": atom_h, "pair_h": pair_h, "reaction_h": reaction_h}


class BaseRFMAdapter(nn.Module):
    """Common utilities for raw-input adapters."""

    def __init__(self, hidden_dim: int, task_name: str, modalities: tuple[str, ...] = DEFAULT_MODALITIES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_name = task_name
        self.modalities = modalities
        self.task_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.modality_embedding = nn.ParameterDict({name: nn.Parameter(torch.zeros(hidden_dim)) for name in modalities})
        nn.init.normal_(self.task_embedding, mean=0.0, std=0.02)
        for value in self.modality_embedding.values():
            nn.init.normal_(value, mean=0.0, std=0.02)

    def modality_dict(self, batch_size: int, device: torch.device, **values: bool | torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.modalities:
            value = values.get(name, False)
            if torch.is_tensor(value):
                mask = value.to(device=device, dtype=torch.bool)
                if mask.ndim == 0:
                    mask = mask.expand(batch_size)
                if mask.shape != (batch_size,):
                    raise ValueError(f"modality {name} tensor must be scalar or [B], got {tuple(mask.shape)}")
                out[name] = mask
            else:
                out[name] = batch_bool(batch_size, bool(value), device)
        return out

    def modality_token(self, modality_mask: dict[str, torch.Tensor]) -> torch.Tensor:
        token: torch.Tensor | None = None
        for name, mask in modality_mask.items():
            emb = self.modality_embedding[name].unsqueeze(0) * mask.float().unsqueeze(-1)
            token = emb if token is None else token + emb
        if token is None:
            raise ValueError("empty modality mask")
        return token

    def reaction_from_atoms(self, atom_tokens: torch.Tensor, atom_valid_mask: torch.Tensor, modality_mask: dict[str, torch.Tensor]) -> torch.Tensor:
        denom = atom_valid_mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
        pooled = (atom_tokens * atom_valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled + self.task_embedding.unsqueeze(0) + self.modality_token(modality_mask)


class MaskedEditAdapter(BaseRFMAdapter):
    """Adapter for masked Delta_BO edit/core pretraining."""

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        max_z: int = 36,
        input_schema: str | None = None,
        distance_clip: float = 10.0,
    ):
        super().__init__(hidden_dim, task_name="masked_edit")
        schema = input_schema or infer_pair_basis_schema(pair_input_dim, self.task_name)
        self.featurizer = ReactionInputFeaturizer(schema, distance_clip=distance_clip)
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.pair_projection = nn.Sequential(nn.LayerNorm(self.featurizer.output_dim), nn.Linear(self.featurizer.output_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        z = batch["z"].clamp(0, self.z_embedding.num_embeddings - 1)
        atom_valid = batch["atom_mask"].bool()
        pair_valid = batch.get("pair_valid", batch.get("pair_mask"))
        if pair_valid is None:
            pair_valid = valid_pair_mask(atom_valid)
        pair_valid = pair_valid.bool()
        atom_tokens = self.z_embedding(z)
        atom_tokens = atom_tokens * atom_valid.unsqueeze(-1)
        pair_basis = self.featurizer(batch["pair_input"])
        pair_tokens = self.pair_projection(pair_basis) * pair_valid.unsqueeze(-1)
        spec = self.featurizer.spec
        modality = self.modality_dict(
            z.shape[0],
            z.device,
            has_R=spec.has_r,
            has_P=spec.has_p,
            has_R_3D=spec.has_r_3d,
            has_P_3D=spec.has_p_3d,
            has_partial_Delta_BO=spec.has_partial_delta_bo,
            has_Delta_BO=spec.has_delta_bo,
        )
        reaction_token = self.reaction_from_atoms(atom_tokens, atom_valid, modality)
        out = RFMEncoderInput(
            atom_tokens,
            pair_tokens,
            reaction_token,
            atom_valid,
            pair_valid,
            modality,
            self.task_name,
            metadata={"pair_basis_schema": self.featurizer.schema, "pair_basis_names": self.featurizer.spec.names},
        )
        out.validate()
        return out


class RPairPropertyAdapter(BaseRFMAdapter):
    """Adapter for R/P property prediction with reaction-delta pair features."""

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        max_z: int = 36,
        has_r_3d: bool = False,
        has_p_3d: bool = False,
        input_schema: str | None = None,
        distance_clip: float = 10.0,
    ):
        super().__init__(hidden_dim, task_name="property")
        schema = input_schema or infer_pair_basis_schema(pair_input_dim, self.task_name)
        self.featurizer = ReactionInputFeaturizer(schema, distance_clip=distance_clip)
        self.has_r_3d = has_r_3d or self.featurizer.spec.has_r_3d
        self.has_p_3d = has_p_3d or self.featurizer.spec.has_p_3d
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.pair_projection = nn.Sequential(nn.LayerNorm(self.featurizer.output_dim), nn.Linear(self.featurizer.output_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        z = batch["z"].clamp(0, self.z_embedding.num_embeddings - 1)
        atom_valid = batch["atom_mask"].bool()
        pair_valid = batch.get("pair_valid", batch.get("pair_mask"))
        if pair_valid is None:
            pair_valid = valid_pair_mask(atom_valid)
        pair_valid = pair_valid.bool()
        pair_raw = batch["pair_input"] if "pair_input" in batch else batch["pair_feats"]
        atom_tokens = self.z_embedding(z)
        atom_tokens = atom_tokens * atom_valid.unsqueeze(-1)
        pair_basis = self.featurizer(pair_raw)
        pair_tokens = self.pair_projection(pair_basis) * pair_valid.unsqueeze(-1)
        spec = self.featurizer.spec
        modality = self.modality_dict(
            z.shape[0],
            z.device,
            has_R=spec.has_r,
            has_P=spec.has_p,
            has_Delta_BO=spec.has_delta_bo,
            has_partial_Delta_BO=spec.has_partial_delta_bo,
            has_R_3D=self.has_r_3d,
            has_P_3D=self.has_p_3d,
        )
        reaction_token = self.reaction_from_atoms(atom_tokens, atom_valid, modality)
        out = RFMEncoderInput(
            atom_tokens,
            pair_tokens,
            reaction_token,
            atom_valid,
            pair_valid,
            modality,
            self.task_name,
            metadata={"pair_basis_schema": self.featurizer.schema, "pair_basis_names": self.featurizer.spec.names},
        )
        out.validate()
        return out


class UnifiedReactionInputAdapter(BaseRFMAdapter):
    """Unified Reaction Encoder input adapter for raw R/P inputs plus Suiren features.

    Frozen Suiren features are input features here, not a post-encoder fusion
    branch. Graph-level features update the initial reaction token, atom-level
    features update atom tokens after atom-map alignment, and pair tokens remain
    derived only from BO/D basis features in the first version.
    """

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        input_schema: str,
        suiren_atom_dim: int = 0,
        suiren_graph_dim: int = 0,
        suiren_atom_dims: dict[str, int] | None = None,
        suiren_graph_dims: dict[str, int] | None = None,
        max_z: int = 36,
        distance_clip: float = 10.0,
    ):
        super().__init__(hidden_dim, task_name="suiren_fusion_property")
        self.featurizer = ReactionInputFeaturizer(input_schema, distance_clip=distance_clip)
        self.suiren_atom_dims = dict(suiren_atom_dims or {})
        self.suiren_graph_dims = dict(suiren_graph_dims or {})
        if suiren_atom_dim > 0 and not self.suiren_atom_dims:
            self.suiren_atom_dims["generic"] = suiren_atom_dim
        if suiren_graph_dim > 0 and not self.suiren_graph_dims:
            self.suiren_graph_dims["generic"] = suiren_graph_dim
        self.pair_input_dim = pair_input_dim
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(self.featurizer.output_dim),
            nn.Linear(self.featurizer.output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.suiren_atom_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_atom_dims.items())
                if dim > 0
            }
        )
        self.suiren_graph_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_graph_dims.items())
                if dim > 0
            }
        )

    @staticmethod
    def suiren_feature_key(stream: str, level: str) -> str:
        if stream == "generic":
            return f"suiren_{level}_features"
        return f"suiren_{stream}_{level}_features"

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        z = batch["z"].clamp(0, self.z_embedding.num_embeddings - 1)
        atom_valid = batch["atom_mask"].bool()
        pair_valid = batch.get("pair_valid", batch.get("pair_mask"))
        if pair_valid is None:
            pair_valid = valid_pair_mask(atom_valid)
        pair_valid = pair_valid.bool()
        batch_size, n_atoms = atom_valid.shape

        pair_raw = batch["pair_input"] if "pair_input" in batch else batch["pair_feats"]
        pair_basis = self.featurizer(pair_raw)
        pair_tokens = self.pair_projection(pair_basis) * pair_valid.unsqueeze(-1)

        atom_tokens = self.z_embedding(z)
        has_atom = False
        for stream, projection in self.suiren_atom_projection.items():
            key = self.suiren_feature_key(stream, "atom")
            if key not in batch:
                continue
            has_atom = True
            atom_features = batch[key]
            if atom_features.shape[:2] != (batch_size, n_atoms):
                raise ValueError(f"suiren_atom_features must be [B,N,C], got {tuple(atom_features.shape)}")
            atom_tokens = atom_tokens + projection(atom_features)
        atom_tokens = atom_tokens * atom_valid.unsqueeze(-1)

        spec = self.featurizer.spec
        has_graph = any(self.suiren_feature_key(stream, "graph") in batch for stream in self.suiren_graph_projection)
        has_suiren = bool(has_atom or has_graph)
        modality = self.modality_dict(
            batch_size,
            z.device,
            has_R=spec.has_r,
            has_P=spec.has_p,
            has_Delta_BO=spec.has_delta_bo,
            has_partial_Delta_BO=spec.has_partial_delta_bo,
            has_R_3D=spec.has_r_3d,
            has_P_3D=spec.has_p_3d,
            has_Suiren_R=has_suiren,
            has_Suiren_P=has_suiren,
        )
        reaction_token = self.reaction_from_atoms(atom_tokens, atom_valid, modality)
        for stream, projection in self.suiren_graph_projection.items():
            key = self.suiren_feature_key(stream, "graph")
            if key not in batch:
                continue
            graph_features = batch[key]
            if graph_features.shape[0] != batch_size:
                raise ValueError(f"suiren_graph_features must be [B,C], got {tuple(graph_features.shape)}")
            reaction_token = reaction_token + projection(graph_features)

        out = RFMEncoderInput(
            atom_tokens,
            pair_tokens,
            reaction_token,
            atom_valid,
            pair_valid,
            modality,
            self.task_name,
            metadata={
                "pair_basis_schema": self.featurizer.schema,
                "pair_basis_names": self.featurizer.spec.names,
                "suiren_atom_dims": self.suiren_atom_dims,
                "suiren_graph_dims": self.suiren_graph_dims,
                "suiren_pair_tokens": False,
            },
        )
        out.validate()
        return out


class ProductEditInputAdapter(BaseRFMAdapter):
    """R-only input adapter for product graph-edit prediction.

    Product-side graph, Delta_BO, product-side geometry, and product-side
    Suiren features are labels or prohibited information for this task. This
    adapter therefore consumes only R graph/geometry plus optional R-only
    frozen Suiren features.
    """

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        input_schema: str,
        suiren_atom_dim: int = 0,
        suiren_graph_dim: int = 0,
        suiren_atom_dims: dict[str, int] | None = None,
        suiren_graph_dims: dict[str, int] | None = None,
        max_z: int = 36,
        distance_clip: float = 10.0,
    ):
        super().__init__(hidden_dim, task_name="product_edit")
        self.featurizer = ReactionInputFeaturizer(input_schema, distance_clip=distance_clip)
        self.suiren_atom_dims = dict(suiren_atom_dims or {})
        self.suiren_graph_dims = dict(suiren_graph_dims or {})
        if suiren_atom_dim > 0 and not self.suiren_atom_dims:
            self.suiren_atom_dims["generic"] = suiren_atom_dim
        if suiren_graph_dim > 0 and not self.suiren_graph_dims:
            self.suiren_graph_dims["generic"] = suiren_graph_dim
        self.pair_input_dim = pair_input_dim
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(self.featurizer.output_dim),
            nn.Linear(self.featurizer.output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.suiren_atom_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_atom_dims.items())
                if dim > 0
            }
        )
        self.suiren_graph_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_graph_dims.items())
                if dim > 0
            }
        )

    @staticmethod
    def suiren_feature_key(stream: str, level: str) -> str:
        if stream == "generic":
            return f"suiren_{level}_features"
        return f"suiren_{stream}_{level}_features"

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        z = batch["z"].clamp(0, self.z_embedding.num_embeddings - 1)
        atom_valid = batch["atom_mask"].bool()
        pair_valid = batch.get("pair_valid", batch.get("pair_mask"))
        if pair_valid is None:
            pair_valid = valid_pair_mask(atom_valid)
        pair_valid = pair_valid.bool()
        batch_size, n_atoms = atom_valid.shape

        pair_raw = batch["pair_input"] if "pair_input" in batch else batch["pair_feats"]
        pair_basis = self.featurizer(pair_raw)
        pair_tokens = self.pair_projection(pair_basis) * pair_valid.unsqueeze(-1)

        atom_tokens = self.z_embedding(z)
        has_atom = False
        for stream, projection in self.suiren_atom_projection.items():
            key = self.suiren_feature_key(stream, "atom")
            if key not in batch:
                continue
            has_atom = True
            atom_features = batch[key]
            if atom_features.shape[:2] != (batch_size, n_atoms):
                raise ValueError(f"{key} must be [B,N,C], got {tuple(atom_features.shape)}")
            atom_tokens = atom_tokens + projection(atom_features)
        atom_tokens = atom_tokens * atom_valid.unsqueeze(-1)

        spec = self.featurizer.spec
        has_graph = any(self.suiren_feature_key(stream, "graph") in batch for stream in self.suiren_graph_projection)
        has_suiren = bool(has_atom or has_graph)
        modality = self.modality_dict(
            batch_size,
            z.device,
            has_R=spec.has_r,
            has_P=False,
            has_Delta_BO=False,
            has_partial_Delta_BO=False,
            has_R_3D=spec.has_r_3d,
            has_P_3D=False,
            has_Suiren_R=has_suiren,
            has_Suiren_P=False,
        )
        reaction_token = self.reaction_from_atoms(atom_tokens, atom_valid, modality)
        for stream, projection in self.suiren_graph_projection.items():
            key = self.suiren_feature_key(stream, "graph")
            if key not in batch:
                continue
            graph_features = batch[key]
            if graph_features.shape[0] != batch_size:
                raise ValueError(f"{key} must be [B,C], got {tuple(graph_features.shape)}")
            reaction_token = reaction_token + projection(graph_features)

        out = RFMEncoderInput(
            atom_tokens,
            pair_tokens,
            reaction_token,
            atom_valid,
            pair_valid,
            modality,
            self.task_name,
            metadata={
                "pair_basis_schema": self.featurizer.schema,
                "pair_basis_names": self.featurizer.spec.names,
                "suiren_atom_dims": self.suiren_atom_dims,
                "suiren_graph_dims": self.suiren_graph_dims,
                "suiren_pair_tokens": False,
                "product_side_inputs": False,
            },
        )
        out.validate()
        return out
