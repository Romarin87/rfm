"""Parity-preserving mapped reaction transition operator.

MRTO-v1 keeps endpoint state encoding parameter-shared, carries explicit even
and odd atom/pair/event fields through every block, and forms the public
``atom_h / pair_h / reaction_h`` interface only at the output boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rfm.features.token_space import (
    BaseRFMAdapter,
    RFMEncoderInput,
    ReactionInputFeaturizer,
    infer_pair_basis_schema,
    valid_pair_mask,
)


def _feature_projection(in_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Project raw scientific features without normalizing them away first."""

    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    return nn.Sequential(*layers)


def _even_mlp(in_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    return nn.Sequential(*layers)


class OddRMSNorm(nn.Module):
    """RMS normalization with scale only, so f(-x) = -f(x)."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * scale * self.weight


class OddMLP(nn.Module):
    """Bias-free odd map used on sign-changing transition fields."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = OddRMSNorm(in_dim)
        self.in_projection = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(self.in_projection(self.norm(value)))
        return self.out_projection(self.dropout(value))


class GaussianRBF(nn.Module):
    """Fixed Gaussian basis for normalized endpoint distances in [0, 1]."""

    def __init__(self, bins: int):
        super().__init__()
        if bins < 2:
            raise ValueError(f"RBF bins must be >=2, got {bins}")
        centers = torch.linspace(0.0, 1.0, bins)
        self.register_buffer("centers", centers, persistent=False)
        spacing = float(centers[1] - centers[0])
        self.gamma = 0.5 / max(spacing * spacing, 1e-8)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        delta = distance.clamp(0.0, 1.0).unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * delta.square())


def _masked_pair_mean(pair_h: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    weights = pair_mask.to(dtype=pair_h.dtype).unsqueeze(-1)
    return (pair_h * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


def _masked_global_pair_mean(pair_h: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    weights = pair_mask.to(dtype=pair_h.dtype).unsqueeze(-1)
    dims = (1, 2)
    return (pair_h * weights).sum(dim=dims) / weights.sum(dim=dims).clamp_min(1.0)


def _masked_atom_mean(atom_h: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    weights = atom_mask.to(dtype=atom_h.dtype).unsqueeze(-1)
    return (atom_h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _channel(pair_basis: torch.Tensor, names: tuple[str, ...], *candidates: str) -> torch.Tensor | None:
    lookup = {name.lower(): index for index, name in enumerate(names)}
    for candidate in candidates:
        index = lookup.get(candidate.lower())
        if index is not None:
            return pair_basis[..., index]
    return None


class EndpointStateBlock(nn.Module):
    """Parameter-shared local processing applied independently to R and P."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.atom_update = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.pair_update = _even_mlp(hidden_dim * 3, hidden_dim, dropout)
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_h: torch.Tensor,
        pair_h: torch.Tensor,
        atom_mask: torch.Tensor,
        local_pair_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_atoms, hidden = atom_h.shape
        h_i = atom_h.unsqueeze(2).expand(batch, n_atoms, n_atoms, hidden)
        h_j = atom_h.unsqueeze(1).expand(batch, n_atoms, n_atoms, hidden)
        messages = self.message(torch.cat([h_j, pair_h], dim=-1))
        weights = local_pair_mask.to(dtype=messages.dtype).unsqueeze(-1)
        aggregate = (messages * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        atom_h = self.atom_norm(atom_h + self.dropout(self.atom_update(torch.cat([atom_h, aggregate], dim=-1))))
        pair_delta = self.pair_update(torch.cat([pair_h, h_i + h_j, (h_i - h_j).abs()], dim=-1))
        pair_h = self.pair_norm(pair_h + self.dropout(pair_delta))
        return atom_h * atom_mask.unsqueeze(-1), pair_h * pair_mask.unsqueeze(-1)


class MRTOv1ReactionInputAdapter(BaseRFMAdapter):
    """Build shared-endpoint and parity-typed fields from canonical raw inputs."""

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        input_schema: str | None,
        task_name: str,
        max_z: int = 36,
        distance_clip: float = 10.0,
        use_odd_field: bool = True,
        endpoint_layers: int = 2,
        geometry_rbf_bins: int = 16,
        geometry_neighbor_cutoff: float = 0.6,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, task_name=task_name)
        schema = input_schema or infer_pair_basis_schema(
            pair_input_dim,
            "property" if "property" in task_name else "masked_edit",
        )
        self.featurizer = ReactionInputFeaturizer(schema, distance_clip=distance_clip)
        self.use_odd_field = bool(use_odd_field)
        self.geometry_neighbor_cutoff = float(geometry_neighbor_cutoff)
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)

        # BO and adjacency are deliberately projected before normalization. In
        # the v0 adapter, LayerNorm([BO, A]) erased the distinction between the
        # common [0,0] and [1,1] states.
        self.endpoint_pair_projection = _feature_projection(2, hidden_dim, dropout)
        has_geometry = self.featurizer.spec.has_r_3d or self.featurizer.spec.has_p_3d
        self.geometry_rbf = GaussianRBF(geometry_rbf_bins) if has_geometry else None
        self.geometry_projection = (
            _feature_projection(geometry_rbf_bins, hidden_dim, dropout) if has_geometry else None
        )
        self.endpoint_blocks = nn.ModuleList(
            [EndpointStateBlock(hidden_dim, dropout) for _ in range(endpoint_layers)]
        )
        self.even_pair_projection = _feature_projection(5, hidden_dim, dropout)
        self.odd_pair_projection = OddMLP(4, hidden_dim, dropout)
        self.even_atom_projection = _feature_projection(5, hidden_dim, dropout)
        self.odd_atom_projection = OddMLP(4, hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        z = batch["z"].clamp(0, self.z_embedding.num_embeddings - 1)
        atom_mask = batch["atom_mask"].bool()
        pair_mask = batch.get("pair_valid", batch.get("pair_mask"))
        if pair_mask is None:
            pair_mask = valid_pair_mask(atom_mask)
        pair_mask = pair_mask.bool()
        pair_raw = batch["pair_input"] if "pair_input" in batch else batch["pair_feats"]
        pair_basis = self.featurizer(pair_raw)
        names = self.featurizer.spec.names
        zero = pair_basis[..., 0].new_zeros(pair_basis.shape[:-1])

        bo_r = _channel(pair_basis, names, "BO_R")
        if bo_r is None:
            raise ValueError(f"MRTO-v1 requires BO_R, schema={self.featurizer.schema}")
        bo_p = _channel(pair_basis, names, "BO_P", "BO_P_visible")
        if bo_p is None:
            bo_p = bo_r
        a_r = _channel(pair_basis, names, "A_R")
        a_p = _channel(pair_basis, names, "A_P", "A_P_visible")
        if a_r is None:
            a_r = (bo_r > 1e-6).to(dtype=bo_r.dtype)
        if a_p is None:
            a_p = (bo_p > 1e-6).to(dtype=bo_p.dtype)

        delta_bo = _channel(pair_basis, names, "Delta_BO", "Delta_BO_visible")
        if delta_bo is None:
            delta_bo = bo_p - bo_r
        abs_delta_bo = _channel(pair_basis, names, "abs_Delta_BO", "abs_Delta_BO_visible")
        if abs_delta_bo is None:
            abs_delta_bo = delta_bo.abs()
        delta_a = _channel(pair_basis, names, "Delta_A", "Delta_A_P_visible")
        if delta_a is None:
            delta_a = a_p - a_r
        formed = _channel(pair_basis, names, "formed", "formed_visible")
        broken = _channel(pair_basis, names, "broken", "broken_visible")
        order_changed = _channel(pair_basis, names, "order_changed", "order_changed_visible")
        visibility = _channel(pair_basis, names, "visibility")
        if formed is None:
            formed = ((a_r <= 0) & (a_p > 0)).to(dtype=bo_r.dtype)
        if broken is None:
            broken = ((a_r > 0) & (a_p <= 0)).to(dtype=bo_r.dtype)
        if order_changed is None:
            order_changed = ((a_r > 0) & (a_p > 0) & (delta_bo.abs() > 1e-6)).to(dtype=bo_r.dtype)
        if visibility is None:
            visibility = torch.ones_like(bo_r)

        d_r = _channel(pair_basis, names, "D_R")
        d_p = _channel(pair_basis, names, "D_P")
        delta_d = _channel(pair_basis, names, "Delta_D")
        abs_delta_d = _channel(pair_basis, names, "abs_Delta_D")
        if delta_d is None:
            delta_d = d_p - d_r if d_r is not None and d_p is not None else zero
        if abs_delta_d is None:
            abs_delta_d = delta_d.abs()

        even_raw = torch.stack(
            [abs_delta_bo, visibility, formed + broken, order_changed, abs_delta_d],
            dim=-1,
        )
        odd_raw = torch.stack([delta_bo, delta_a, formed - broken, delta_d], dim=-1)

        endpoint_r = torch.stack([bo_r, a_r], dim=-1)
        endpoint_p = torch.stack([bo_p, a_p], dim=-1)
        pair_r = self.endpoint_pair_projection(endpoint_r)
        pair_p = self.endpoint_pair_projection(endpoint_p)
        if d_r is not None:
            if self.geometry_rbf is None or self.geometry_projection is None:
                raise RuntimeError("D_R is present but the MRTO-v1 geometry adapter was not initialized")
            pair_r = pair_r + self.geometry_projection(self.geometry_rbf(d_r))
        if d_p is not None:
            if self.geometry_rbf is None or self.geometry_projection is None:
                raise RuntimeError("D_P is present but the MRTO-v1 geometry adapter was not initialized")
            pair_p = pair_p + self.geometry_projection(self.geometry_rbf(d_p))
        pair_r = pair_r * pair_mask.unsqueeze(-1)
        pair_p = pair_p * pair_mask.unsqueeze(-1)

        endpoint_r_mask = (a_r > 0) & pair_mask
        endpoint_p_mask = (a_p > 0) & pair_mask
        if d_r is not None:
            endpoint_r_mask |= (d_r < self.geometry_neighbor_cutoff) & pair_mask
        if d_p is not None:
            endpoint_p_mask |= (d_p < self.geometry_neighbor_cutoff) & pair_mask

        atom_r = self.z_embedding(z) * atom_mask.unsqueeze(-1)
        atom_p = self.z_embedding(z) * atom_mask.unsqueeze(-1)
        for block in self.endpoint_blocks:
            atom_r, pair_r = block(atom_r, pair_r, atom_mask, endpoint_r_mask, pair_mask)
            atom_p, pair_p = block(atom_p, pair_p, atom_mask, endpoint_p_mask, pair_mask)

        even_pair = self.even_pair_projection(even_raw) * pair_mask.unsqueeze(-1)
        odd_pair = self.odd_pair_projection(odd_raw) * pair_mask.unsqueeze(-1)
        pair_plus = 0.5 * (pair_r + pair_p) + even_pair
        pair_minus = 0.5 * (pair_p - pair_r) + odd_pair

        even_atom_raw = _masked_pair_mean(even_raw, pair_mask)
        odd_atom_raw = _masked_pair_mean(odd_raw, pair_mask)
        atom_plus = 0.5 * (atom_r + atom_p) + self.even_atom_projection(even_atom_raw)
        atom_minus = 0.5 * (atom_p - atom_r) + self.odd_atom_projection(odd_atom_raw)
        if not self.use_odd_field:
            atom_minus = torch.zeros_like(atom_plus)
            pair_minus = torch.zeros_like(pair_plus)
        atom_plus = atom_plus * atom_mask.unsqueeze(-1)
        atom_minus = atom_minus * atom_mask.unsqueeze(-1)
        pair_plus = pair_plus * pair_mask.unsqueeze(-1)
        pair_minus = pair_minus * pair_mask.unsqueeze(-1)

        operator_pair_mask = ((a_r > 0) | (a_p > 0) | (visibility < 0.5)) & pair_mask
        if d_r is not None:
            operator_pair_mask |= (d_r < self.geometry_neighbor_cutoff) & pair_mask
        if d_p is not None:
            operator_pair_mask |= (d_p < self.geometry_neighbor_cutoff) & pair_mask

        spec = self.featurizer.spec
        modality = self.modality_dict(
            z.shape[0],
            z.device,
            has_R=spec.has_r,
            has_P=spec.has_p,
            has_Delta_BO=spec.has_delta_bo,
            has_partial_Delta_BO=spec.has_partial_delta_bo,
            has_R_3D=spec.has_r_3d,
            has_P_3D=spec.has_p_3d,
            has_Suiren_R=False,
            has_Suiren_P=False,
        )
        context_token = self.task_embedding.unsqueeze(0) + self.modality_token(modality)
        out = RFMEncoderInput(
            atom_tokens=(atom_plus + atom_minus) * atom_mask.unsqueeze(-1),
            pair_tokens=(pair_plus + pair_minus) * pair_mask.unsqueeze(-1),
            reaction_token=context_token,
            atom_valid_mask=atom_mask,
            pair_valid_mask=pair_mask,
            modality_mask=modality,
            task_name=self.task_name,
            metadata={
                "pair_basis_schema": self.featurizer.schema,
                "pair_basis_names": names,
                "encoder_family": "mrto_v1",
                "mrto_atom_fields": {"a_plus": atom_plus, "a_minus": atom_minus},
                "mrto_pair_fields": {"p_plus": pair_plus, "p_minus": pair_minus},
                "mrto_operator_pair_mask": operator_pair_mask,
                "mrto_use_odd_field": self.use_odd_field,
            },
        )
        out.validate()
        return out


class LowRankParityTriangle(nn.Module):
    """Low-rank triangle multiplication respecting R/P swap parity."""

    def __init__(self, hidden_dim: int, triangle_dim: int, dropout: float, scale: float):
        super().__init__()
        if triangle_dim < 1:
            raise ValueError(f"triangle_dim must be positive, got {triangle_dim}")
        self.plus_projection = nn.Linear(hidden_dim, triangle_dim)
        self.minus_projection = nn.Linear(hidden_dim, triangle_dim, bias=False)
        self.plus_out = _even_mlp(triangle_dim, hidden_dim, dropout)
        self.minus_out = OddMLP(triangle_dim, hidden_dim, dropout)
        self.plus_norm = nn.LayerNorm(hidden_dim)
        self.minus_norm = OddRMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(scale)

    def forward(
        self,
        pair_plus: torch.Tensor,
        pair_minus: torch.Tensor,
        operator_pair_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge = operator_pair_mask.to(dtype=pair_plus.dtype).unsqueeze(-1)
        plus = self.plus_projection(pair_plus) * edge
        minus = self.minus_projection(pair_minus) * edge
        triangle_plus = torch.einsum("bikd,bkjd->bijd", plus, plus) + torch.einsum(
            "bikd,bkjd->bijd", minus, minus
        )
        triangle_minus = torch.einsum("bikd,bkjd->bijd", plus, minus) + torch.einsum(
            "bikd,bkjd->bijd", minus, plus
        )
        path_count = torch.einsum(
            "bik,bkj->bij",
            operator_pair_mask.to(dtype=pair_plus.dtype),
            operator_pair_mask.to(dtype=pair_plus.dtype),
        ).clamp_min(1.0)
        triangle_plus = triangle_plus / path_count.unsqueeze(-1)
        triangle_minus = triangle_minus / path_count.unsqueeze(-1)
        triangle_plus = 0.5 * (triangle_plus + triangle_plus.transpose(1, 2))
        triangle_minus = 0.5 * (triangle_minus + triangle_minus.transpose(1, 2))
        mask = pair_mask.unsqueeze(-1)
        pair_plus = self.plus_norm(pair_plus + self.scale * self.dropout(self.plus_out(triangle_plus))) * mask
        pair_minus = self.minus_norm(pair_minus + self.scale * self.dropout(self.minus_out(triangle_minus))) * mask
        return pair_plus, pair_minus


class ParityAtomAttention(nn.Module):
    """Even attention connectivity with separate even/odd value transport."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pair_bias = nn.Linear(hidden_dim * 2, heads, bias=False)
        self.plus_value = nn.Linear(hidden_dim, hidden_dim)
        self.minus_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.odd_to_plus_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.plus_to_odd_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.plus_out = nn.Linear(hidden_dim, hidden_dim)
        self.minus_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.plus_update = _even_mlp(hidden_dim * 3, hidden_dim, dropout)
        self.minus_update = OddMLP(hidden_dim * 3, hidden_dim, dropout)
        self.plus_norm = nn.LayerNorm(hidden_dim)
        self.minus_norm = OddRMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_plus: torch.Tensor,
        atom_minus: torch.Tensor,
        pair_plus: torch.Tensor,
        pair_minus: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_atoms, hidden = atom_plus.shape
        q = self.query(atom_plus).view(batch, n_atoms, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(atom_plus).view(batch, n_atoms, self.heads, self.head_dim).transpose(1, 2)
        even_pair = torch.cat([pair_plus, pair_minus.square()], dim=-1)
        bias = self.pair_bias(even_pair).permute(0, 3, 1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale + bias
        logits = logits.masked_fill(~atom_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        odd_gate = torch.tanh(self.odd_to_plus_gate(pair_minus))
        plus_values = self.plus_value(atom_plus).unsqueeze(1).expand(-1, n_atoms, -1, -1)
        minus_values = self.minus_value(atom_minus).unsqueeze(1).expand(-1, n_atoms, -1, -1)
        value_plus = plus_values + odd_gate * minus_values
        plus_gate = torch.tanh(self.plus_to_odd_gate(pair_minus))
        value_minus = minus_values + plus_gate * plus_values
        value_plus = value_plus.view(batch, n_atoms, n_atoms, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        value_minus = value_minus.view(batch, n_atoms, n_atoms, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        context_plus = (weights.unsqueeze(-1) * value_plus).sum(dim=3).transpose(1, 2).reshape(batch, n_atoms, hidden)
        context_minus = (weights.unsqueeze(-1) * value_minus).sum(dim=3).transpose(1, 2).reshape(batch, n_atoms, hidden)
        context_plus = self.plus_out(context_plus)
        context_minus = self.minus_out(context_minus)

        even_input = torch.cat([atom_plus, context_plus, atom_minus.square()], dim=-1)
        odd_input = torch.cat([atom_minus, context_minus, atom_plus * atom_minus], dim=-1)
        atom_plus = self.plus_norm(atom_plus + self.dropout(self.plus_update(even_input)))
        atom_minus = self.minus_norm(atom_minus + self.dropout(self.minus_update(odd_input)))
        mask = atom_mask.unsqueeze(-1)
        return atom_plus * mask, atom_minus * mask


class ParityEventInteraction(nn.Module):
    """Extract sparse events and broadcast them back to pair and atom fields."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        event_slots: int,
        dropout: float,
        event_topk: int,
        feedback_scale: float,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.event_slots = int(event_slots)
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.event_topk = int(event_topk)
        self.feedback_scale = float(feedback_scale)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.plus_value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.minus_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.center_bias = nn.Linear(hidden_dim * 2, heads, bias=False)
        self.event_plus_update = _even_mlp(hidden_dim * 3, hidden_dim, dropout)
        self.event_minus_update = OddMLP(hidden_dim * 3, hidden_dim, dropout)
        self.event_plus_norm = nn.LayerNorm(hidden_dim)
        self.event_minus_norm = OddRMSNorm(hidden_dim)
        self.pair_plus_feedback = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.pair_minus_feedback = OddMLP(hidden_dim * 2, hidden_dim, dropout)
        self.atom_plus_feedback = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.atom_minus_feedback = OddMLP(hidden_dim * 2, hidden_dim, dropout)
        self.pair_plus_norm = nn.LayerNorm(hidden_dim)
        self.pair_minus_norm = OddRMSNorm(hidden_dim)
        self.atom_plus_norm = nn.LayerNorm(hidden_dim)
        self.atom_minus_norm = OddRMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        event_plus: torch.Tensor,
        event_minus: torch.Tensor,
        atom_plus: torch.Tensor,
        atom_minus: torch.Tensor,
        pair_plus: torch.Tensor,
        pair_minus: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        event_pair_mask: torch.Tensor,
        pair_center_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_atoms, _, hidden = pair_plus.shape
        pair_even = torch.cat([pair_plus, pair_minus.square()], dim=-1)
        flat_even = pair_even.reshape(batch, n_atoms * n_atoms, hidden * 2)
        flat_minus = pair_minus.reshape(batch, n_atoms * n_atoms, hidden)
        flat_mask = event_pair_mask.reshape(batch, n_atoms * n_atoms).bool()
        q = self.query(event_plus).view(batch, self.event_slots, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(flat_even).view(batch, n_atoms * n_atoms, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        center = self.center_bias(flat_even).transpose(1, 2).unsqueeze(2)
        logits = logits + center + pair_center_logits.reshape(batch, 1, 1, n_atoms * n_atoms)
        logits = logits.masked_fill(~flat_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        if self.event_topk > 0 and self.event_topk < n_atoms * n_atoms:
            topk = min(self.event_topk, n_atoms * n_atoms)
            indices = logits.topk(topk, dim=-1).indices
            keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
            keep &= flat_mask[:, None, None, :]
            logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * flat_mask[:, None, None, :].to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        plus_value = self.plus_value(flat_even).view(batch, n_atoms * n_atoms, self.heads, self.head_dim).transpose(1, 2)
        minus_value = self.minus_value(flat_minus).view(batch, n_atoms * n_atoms, self.heads, self.head_dim).transpose(1, 2)
        context_plus = torch.matmul(self.dropout(weights), plus_value).transpose(1, 2).reshape(batch, self.event_slots, hidden)
        context_minus = torch.matmul(self.dropout(weights), minus_value).transpose(1, 2).reshape(batch, self.event_slots, hidden)
        event_plus_input = torch.cat([event_plus, context_plus, event_minus.square()], dim=-1)
        event_minus_input = torch.cat([event_minus, context_minus, event_plus * event_minus], dim=-1)
        event_plus = self.event_plus_norm(event_plus + self.dropout(self.event_plus_update(event_plus_input)))
        event_minus = self.event_minus_norm(event_minus + self.dropout(self.event_minus_update(event_minus_input)))

        event_pair_weights = weights.mean(dim=1).reshape(batch, self.event_slots, n_atoms, n_atoms)
        symmetric_weights = event_pair_weights + event_pair_weights.transpose(2, 3)
        pair_event_plus = torch.einsum("bkij,bkh->bijh", symmetric_weights, event_plus)
        pair_event_minus = torch.einsum("bkij,bkh->bijh", symmetric_weights, event_minus)
        pair_plus = self.pair_plus_norm(
            pair_plus
            + self.feedback_scale
            * self.dropout(self.pair_plus_feedback(torch.cat([pair_plus, pair_event_plus], dim=-1)))
        )
        pair_minus = self.pair_minus_norm(
            pair_minus
            + self.feedback_scale
            * self.dropout(self.pair_minus_feedback(torch.cat([pair_minus, pair_event_minus], dim=-1)))
        )
        pair_plus = pair_plus * pair_mask.unsqueeze(-1)
        pair_minus = pair_minus * pair_mask.unsqueeze(-1)

        atom_event_plus = _masked_pair_mean(pair_event_plus, pair_mask)
        atom_event_minus = _masked_pair_mean(pair_event_minus, pair_mask)
        atom_plus = self.atom_plus_norm(
            atom_plus
            + self.feedback_scale
            * self.dropout(self.atom_plus_feedback(torch.cat([atom_plus, atom_event_plus], dim=-1)))
        )
        atom_minus = self.atom_minus_norm(
            atom_minus
            + self.feedback_scale
            * self.dropout(self.atom_minus_feedback(torch.cat([atom_minus, atom_event_minus], dim=-1)))
        )
        atom_plus = atom_plus * atom_mask.unsqueeze(-1)
        atom_minus = atom_minus * atom_mask.unsqueeze(-1)
        return event_plus, event_minus, atom_plus, atom_minus, pair_plus, pair_minus, event_pair_weights


@dataclass
class MRTOv1CenterState:
    atom_logits: torch.Tensor
    pair_logits: torch.Tensor
    event_pair_weights: torch.Tensor


class MRTOv1TransitionBlock(nn.Module):
    """Triangle, parity-aware atom attention, pair update, and event broadcast."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        heads: int,
        event_slots: int,
        triangle_dim: int,
        triangle_scale: float,
        use_triangle: bool,
        pair_update_scale: float,
        event_topk: int,
        event_feedback_scale: float,
    ):
        super().__init__()
        self.pair_update_scale = float(pair_update_scale)
        self.triangle = (
            LowRankParityTriangle(hidden_dim, triangle_dim, dropout, triangle_scale)
            if use_triangle
            else None
        )
        self.atom_attention = ParityAtomAttention(hidden_dim, heads, dropout)
        self.pair_plus_update = _even_mlp(hidden_dim * 4, hidden_dim, dropout)
        self.pair_minus_update = OddMLP(hidden_dim * 3, hidden_dim, dropout)
        self.pair_plus_norm = nn.LayerNorm(hidden_dim)
        self.pair_minus_norm = OddRMSNorm(hidden_dim)
        self.pair_center = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.pair_center_out = nn.Linear(hidden_dim, 1)
        self.events = ParityEventInteraction(
            hidden_dim,
            heads,
            event_slots,
            dropout,
            event_topk,
            event_feedback_scale,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        atom_plus: torch.Tensor,
        atom_minus: torch.Tensor,
        pair_plus: torch.Tensor,
        pair_minus: torch.Tensor,
        event_plus: torch.Tensor,
        event_minus: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        operator_pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MRTOv1CenterState]:
        if self.triangle is not None:
            pair_plus, pair_minus = self.triangle(pair_plus, pair_minus, operator_pair_mask, pair_mask)
        atom_plus, atom_minus = self.atom_attention(atom_plus, atom_minus, pair_plus, pair_minus, atom_mask)

        plus_i = atom_plus.unsqueeze(2)
        plus_j = atom_plus.unsqueeze(1)
        minus_i = atom_minus.unsqueeze(2)
        minus_j = atom_minus.unsqueeze(1)
        even_pair_input = torch.cat(
            [pair_plus, plus_i + plus_j, minus_i * minus_j, (plus_i - plus_j).abs()],
            dim=-1,
        )
        odd_pair_input = torch.cat(
            [pair_minus, minus_i + minus_j, plus_i * minus_j + minus_i * plus_j],
            dim=-1,
        )
        pair_plus = self.pair_plus_norm(
            pair_plus + self.pair_update_scale * self.dropout(self.pair_plus_update(even_pair_input))
        ) * pair_mask.unsqueeze(-1)
        pair_minus = self.pair_minus_norm(
            pair_minus + self.pair_update_scale * self.dropout(self.pair_minus_update(odd_pair_input))
        ) * pair_mask.unsqueeze(-1)

        pair_even = torch.cat([pair_plus, pair_minus.square()], dim=-1)
        pair_logits = self.pair_center_out(self.pair_center(pair_even)).squeeze(-1)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))
        pair_logits = pair_logits.masked_fill(~pair_mask, torch.finfo(pair_logits.dtype).min)
        event_pair_mask = torch.triu(operator_pair_mask, diagonal=1)
        event_plus, event_minus, atom_plus, atom_minus, pair_plus, pair_minus, event_weights = self.events(
            event_plus,
            event_minus,
            atom_plus,
            atom_minus,
            pair_plus,
            pair_minus,
            atom_mask,
            pair_mask,
            event_pair_mask,
            pair_logits,
        )

        pair_probability = torch.sigmoid(pair_logits) * pair_mask.to(dtype=pair_logits.dtype)
        atom_probability = 1.0 - (1.0 - pair_probability.clamp(max=1.0 - 1e-6)).prod(dim=2)
        atom_probability = atom_probability.clamp(1e-6, 1.0 - 1e-6)
        atom_logits = torch.logit(atom_probability).masked_fill(~atom_mask, torch.finfo(pair_logits.dtype).min)
        center = MRTOv1CenterState(atom_logits, pair_logits, event_weights)
        return atom_plus, atom_minus, pair_plus, pair_minus, event_plus, event_minus, center


class MRTOv1ReactionEncoder(nn.Module):
    """Full parity-typed MRTO operator core with sparse event bottleneck."""

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        dropout: float,
        attention_heads: int = 8,
        event_slots: int = 4,
        triangle_layers: int = 2,
        triangle_dim: int = 16,
        triangle_scale: float = 0.5,
        pair_update_scale: float = 0.75,
        event_topk: int = 0,
        event_feedback_scale: float = 0.5,
    ):
        super().__init__()
        if event_slots < 1:
            raise ValueError(f"event_slots must be positive, got {event_slots}")
        self.event_slots = int(event_slots)
        self.event_plus_init = nn.Parameter(torch.randn(event_slots, hidden_dim) * 0.02)
        self.context_to_event = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MRTOv1TransitionBlock(
                    hidden_dim,
                    dropout,
                    heads=attention_heads,
                    event_slots=event_slots,
                    triangle_dim=triangle_dim,
                    triangle_scale=triangle_scale,
                    use_triangle=index < triangle_layers,
                    pair_update_scale=pair_update_scale,
                    event_topk=event_topk,
                    event_feedback_scale=event_feedback_scale,
                )
                for index in range(layers)
            ]
        )
        self.atom_plus_out = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.atom_minus_out = OddMLP(hidden_dim, hidden_dim, dropout)
        self.pair_plus_out = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.pair_minus_out = OddMLP(hidden_dim, hidden_dim, dropout)
        self.reaction_plus_out = _even_mlp(hidden_dim * 4, hidden_dim, dropout)
        self.reaction_minus_out = OddMLP(hidden_dim * 3, hidden_dim, dropout)

    def forward(self, encoder_input: RFMEncoderInput) -> dict[str, torch.Tensor]:
        encoder_input.validate()
        atom_fields = encoder_input.metadata.get("mrto_atom_fields")
        pair_fields = encoder_input.metadata.get("mrto_pair_fields")
        operator_pair_mask = encoder_input.metadata.get("mrto_operator_pair_mask")
        if not atom_fields or not pair_fields or operator_pair_mask is None:
            raise ValueError("MRTOv1ReactionEncoder requires MRTOv1ReactionInputAdapter metadata")
        atom_mask = encoder_input.atom_valid_mask
        pair_mask = encoder_input.pair_valid_mask
        atom_plus = atom_fields["a_plus"] * atom_mask.unsqueeze(-1)
        atom_minus = atom_fields["a_minus"] * atom_mask.unsqueeze(-1)
        pair_plus = pair_fields["p_plus"] * pair_mask.unsqueeze(-1)
        pair_minus = pair_fields["p_minus"] * pair_mask.unsqueeze(-1)
        batch = atom_plus.shape[0]
        event_plus = self.event_plus_init.unsqueeze(0).expand(batch, -1, -1)
        event_plus = event_plus + self.context_to_event(encoder_input.reaction_token).unsqueeze(1)
        event_minus = torch.zeros_like(event_plus)
        center: MRTOv1CenterState | None = None
        for block in self.layers:
            atom_plus, atom_minus, pair_plus, pair_minus, event_plus, event_minus, center = block(
                atom_plus,
                atom_minus,
                pair_plus,
                pair_minus,
                event_plus,
                event_minus,
                atom_mask,
                pair_mask,
                operator_pair_mask,
            )

        atom_plus_out = self.atom_plus_out(torch.cat([atom_plus, atom_minus.square()], dim=-1))
        atom_minus_out = self.atom_minus_out(atom_minus)
        pair_plus_out = self.pair_plus_out(torch.cat([pair_plus, pair_minus.square()], dim=-1))
        pair_minus_out = self.pair_minus_out(pair_minus)
        atom_h = (atom_plus_out + atom_minus_out) * atom_mask.unsqueeze(-1)
        pair_h = (pair_plus_out + pair_minus_out) * pair_mask.unsqueeze(-1)

        reaction_plus = self.reaction_plus_out(
            torch.cat(
                [
                    event_plus.mean(dim=1),
                    event_minus.square().mean(dim=1),
                    _masked_atom_mean(atom_plus, atom_mask),
                    _masked_global_pair_mean(pair_plus, pair_mask),
                ],
                dim=-1,
            )
        )
        reaction_minus = self.reaction_minus_out(
            torch.cat(
                [
                    event_minus.mean(dim=1),
                    _masked_atom_mean(atom_minus, atom_mask),
                    _masked_global_pair_mean(pair_minus, pair_mask),
                ],
                dim=-1,
            )
        )
        reaction_h = reaction_plus + reaction_minus
        out = {
            "atom_h": atom_h,
            "pair_h": pair_h,
            "reaction_h": reaction_h,
            "mrto_atom_plus": atom_plus,
            "mrto_atom_minus": atom_minus,
            "mrto_pair_plus": pair_plus,
            "mrto_pair_minus": pair_minus,
            "mrto_event_plus": event_plus,
            "mrto_event_minus": event_minus,
            "mrto_reaction_plus": reaction_plus,
            "mrto_reaction_minus": reaction_minus,
        }
        if center is not None:
            out["mrto_center_atom_logits"] = center.atom_logits
            out["mrto_center_pair_logits"] = center.pair_logits
            out["mrto_event_pair_weights"] = center.event_pair_weights
        return out
