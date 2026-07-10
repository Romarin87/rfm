"""Mapped Reaction Transition Operator components.

MRTO keeps the public task-head contract unchanged while changing the internal
representation from R/P feature fusion to mapped transition fields:
atom_h [B,N,H], pair_h [B,N,N,H], reaction_h [B,H].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rfm.features.radar import PairBiasedSelfAttention, _soft_pool
from rfm.features.token_space import (
    BaseRFMAdapter,
    RFMEncoderInput,
    ReactionInputFeaturizer,
    infer_pair_basis_schema,
    valid_pair_mask,
)


def _channel_indices(names: tuple[str, ...], group: str) -> tuple[int, ...]:
    indices: list[int] = []
    for index, name in enumerate(names):
        lower = name.lower()
        if group == "r":
            keep = lower in {"bo_r", "a_r", "d_r"} or lower.endswith("_r")
        elif group == "p":
            keep = lower in {"bo_p", "a_p", "d_p", "bo_p_visible", "a_p_visible"} or lower.endswith("_p")
        elif group == "transition":
            keep = (
                "delta" in lower
                or "abs" in lower
                or "formed" in lower
                or "broken" in lower
                or "order_changed" in lower
                or lower == "visibility"
            )
        else:
            raise ValueError(group)
        if keep:
            indices.append(index)
    return tuple(indices or range(len(names)))


def _masked_pair_mean(pair_basis: torch.Tensor, pair_valid: torch.Tensor) -> torch.Tensor:
    weights = pair_valid.float().unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (pair_basis * weights).sum(dim=2) / denom


def _mlp(in_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    return nn.Sequential(*layers)


class MRTOReactionInputAdapter(BaseRFMAdapter):
    """Build even/odd atom and pair transition fields from mapped R/P inputs."""

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        input_schema: str | None,
        task_name: str,
        max_z: int = 36,
        distance_clip: float = 10.0,
        use_odd_field: bool = True,
    ):
        super().__init__(hidden_dim, task_name=task_name)
        schema = input_schema or infer_pair_basis_schema(pair_input_dim, "property" if "property" in task_name else "masked_edit")
        self.featurizer = ReactionInputFeaturizer(schema, distance_clip=distance_clip)
        self.use_odd_field = bool(use_odd_field)

        basis_names = self.featurizer.spec.names
        r_indices = _channel_indices(basis_names, "r")
        p_indices = _channel_indices(basis_names, "p")
        t_indices = _channel_indices(basis_names, "transition")
        self.register_buffer("r_channel_indices", torch.tensor(r_indices, dtype=torch.long), persistent=False)
        self.register_buffer("p_channel_indices", torch.tensor(p_indices, dtype=torch.long), persistent=False)
        self.register_buffer("transition_channel_indices", torch.tensor(t_indices, dtype=torch.long), persistent=False)

        self.shared_endpoint_shape = len(r_indices) == len(p_indices)
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.atom_r_projection = _mlp(len(r_indices), hidden_dim)
        self.atom_p_projection = self.atom_r_projection if self.shared_endpoint_shape else _mlp(len(p_indices), hidden_dim)
        self.pair_r_projection = _mlp(len(r_indices), hidden_dim)
        self.pair_p_projection = self.pair_r_projection if self.shared_endpoint_shape else _mlp(len(p_indices), hidden_dim)
        self.atom_transition_projection = _mlp(len(t_indices), hidden_dim)
        self.pair_transition_projection = _mlp(len(t_indices), hidden_dim)
        self.atom_token_projection = _mlp(hidden_dim * 2, hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        z = batch["z"].clamp(0, self.z_embedding.num_embeddings - 1)
        atom_valid = batch["atom_mask"].bool()
        pair_valid = batch.get("pair_valid", batch.get("pair_mask"))
        if pair_valid is None:
            pair_valid = valid_pair_mask(atom_valid)
        pair_valid = pair_valid.bool()

        pair_raw = batch["pair_input"] if "pair_input" in batch else batch["pair_feats"]
        pair_basis = self.featurizer(pair_raw)
        r_basis = pair_basis.index_select(-1, self.r_channel_indices)
        p_basis = pair_basis.index_select(-1, self.p_channel_indices)
        transition_basis = pair_basis.index_select(-1, self.transition_channel_indices)

        z_token = self.z_embedding(z)
        r_summary = _masked_pair_mean(r_basis, pair_valid)
        p_summary = _masked_pair_mean(p_basis, pair_valid)
        transition_summary = _masked_pair_mean(transition_basis, pair_valid)

        a_r = z_token + self.atom_r_projection(r_summary)
        a_p = z_token + self.atom_p_projection(p_summary)
        a_plus = 0.5 * (a_r + a_p)
        a_minus = 0.5 * (a_p - a_r)
        if self.use_odd_field:
            a_minus = a_minus + self.atom_transition_projection(transition_summary)
        else:
            a_minus = torch.zeros_like(a_plus)

        p_r = self.pair_r_projection(r_basis)
        p_p = self.pair_p_projection(p_basis)
        p_plus = 0.5 * (p_r + p_p)
        p_minus = 0.5 * (p_p - p_r)
        if self.use_odd_field:
            p_minus = p_minus + self.pair_transition_projection(transition_basis)
        else:
            p_minus = torch.zeros_like(p_plus)

        a_plus = a_plus * atom_valid.unsqueeze(-1)
        a_minus = a_minus * atom_valid.unsqueeze(-1)
        p_plus = p_plus * pair_valid.unsqueeze(-1)
        p_minus = p_minus * pair_valid.unsqueeze(-1)

        atom_tokens = self.atom_token_projection(torch.cat([a_plus, a_minus], dim=-1)) * atom_valid.unsqueeze(-1)
        pair_tokens = (p_plus + p_minus) * pair_valid.unsqueeze(-1)
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
        reaction_token = self.reaction_from_atoms(atom_tokens, atom_valid, modality)
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
                "encoder_family": "mrto",
                "mrto_atom_fields": {
                    "a_plus": a_plus,
                    "a_minus": a_minus,
                },
                "mrto_pair_fields": {
                    "p_plus": p_plus,
                    "p_minus": p_minus,
                },
                "mrto_endpoint_projection_shared": self.shared_endpoint_shape,
                "mrto_use_odd_field": self.use_odd_field,
            },
        )
        out.validate()
        return out


@dataclass
class MRTOCenterState:
    atom_logits: torch.Tensor
    pair_logits: torch.Tensor
    event_pair_weights: torch.Tensor | None


class EventSlotReadout(nn.Module):
    """Cross-attention from sparse reaction event slots to the pair field."""

    def __init__(self, hidden_dim: int, num_heads: int, event_slots: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if event_slots < 1:
            raise ValueError(f"event_slots must be positive, got {event_slots}")
        self.num_heads = int(num_heads)
        self.event_slots = int(event_slots)
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.slots = nn.Parameter(torch.randn(event_slots, hidden_dim) * 0.02)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def initial_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.slots.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward(
        self,
        event_h: torch.Tensor,
        pair_h: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_atoms, _, hidden = pair_h.shape
        pair_flat = pair_h.reshape(batch, n_atoms * n_atoms, hidden)
        mask_flat = pair_mask.reshape(batch, n_atoms * n_atoms).bool()
        q = self.query(event_h).view(batch, self.event_slots, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(pair_flat).view(batch, n_atoms * n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(pair_flat).view(batch, n_atoms * n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~mask_flat[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask_flat[:, None, None, :].float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(batch, self.event_slots, hidden)
        event_h = self.norm(event_h + self.dropout(self.out(context)))
        event_pair_weights = weights.mean(dim=1).reshape(batch, self.event_slots, n_atoms, n_atoms)
        return event_h, event_pair_weights


class MRTOTransitionBlock(nn.Module):
    """One even/odd transition-field block."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        attention_heads: int,
        use_event_slots: bool,
        event_slots: int,
        pair_update_scale: float,
        reaction_update_scale: float,
    ):
        super().__init__()
        self.use_event_slots = bool(use_event_slots)
        self.pair_update_scale = float(pair_update_scale)
        self.reaction_update_scale = float(reaction_update_scale)
        self.plus_attention = PairBiasedSelfAttention(hidden_dim, attention_heads, dropout)
        self.minus_attention = PairBiasedSelfAttention(hidden_dim, attention_heads, dropout)
        self.pair_plus_update = _mlp(hidden_dim * 5, hidden_dim, dropout)
        self.pair_minus_update = _mlp(hidden_dim * 5, hidden_dim, dropout)
        self.pair_plus_norm = nn.LayerNorm(hidden_dim)
        self.pair_minus_norm = nn.LayerNorm(hidden_dim)
        self.pair_mix = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.atom_center = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pair_center = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.event_readout = EventSlotReadout(hidden_dim, attention_heads, event_slots, dropout) if use_event_slots else None
        self.reaction_update = _mlp(hidden_dim * 5, hidden_dim, dropout)
        self.reaction_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        a_plus: torch.Tensor,
        a_minus: torch.Tensor,
        p_plus: torch.Tensor,
        p_minus: torch.Tensor,
        event_h: torch.Tensor | None,
        reaction_h: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, MRTOCenterState]:
        a_plus = self.plus_attention(a_plus, p_plus, atom_mask)
        a_minus = self.minus_attention(a_minus, p_plus, atom_mask)

        plus_pair = a_plus.unsqueeze(2) + a_plus.unsqueeze(1)
        minus_pair = a_minus.unsqueeze(2) + a_minus.unsqueeze(1)
        minus_pair_diff = a_minus.unsqueeze(2) - a_minus.unsqueeze(1)
        pair_input = torch.cat([p_plus, p_minus, plus_pair, minus_pair, minus_pair_diff], dim=-1)
        p_plus = self.pair_plus_norm(
            p_plus + self.pair_update_scale * self.dropout(self.pair_plus_update(pair_input))
        ) * pair_mask.unsqueeze(-1)
        p_minus = self.pair_minus_norm(
            p_minus + self.pair_update_scale * self.dropout(self.pair_minus_update(pair_input))
        ) * pair_mask.unsqueeze(-1)
        pair_h = self.pair_mix(torch.cat([p_plus, p_minus], dim=-1)) * pair_mask.unsqueeze(-1)

        atom_logits = self.atom_center(torch.cat([a_plus, a_minus.abs()], dim=-1)).squeeze(-1)
        pair_logits = self.pair_center(pair_h).squeeze(-1)
        atom_logits = atom_logits.masked_fill(~atom_mask.bool(), torch.finfo(atom_logits.dtype).min)
        pair_logits = pair_logits.masked_fill(~pair_mask.bool(), torch.finfo(pair_logits.dtype).min)

        event_pair_weights = None
        if self.event_readout is not None:
            layer_slots = self.event_readout.initial_slots(a_plus.shape[0], a_plus.device)
            if event_h is None:
                event_h = layer_slots
            else:
                event_h = event_h + layer_slots
            event_h, event_pair_weights = self.event_readout(event_h, pair_h, pair_mask)
            event_summary = event_h.mean(dim=1)
        else:
            event_summary = _soft_pool(pair_h, pair_logits, pair_mask)

        atom_mix = a_plus + a_minus
        center_atom = _soft_pool(atom_mix, atom_logits, atom_mask)
        env_atom = _soft_pool(atom_mix, atom_logits, atom_mask, invert=True)
        center_pair = _soft_pool(pair_h, pair_logits, pair_mask)
        reaction_input = torch.cat([reaction_h, event_summary, center_atom, env_atom, center_pair], dim=-1)
        reaction_h = self.reaction_norm(
            reaction_h + self.reaction_update_scale * self.dropout(self.reaction_update(reaction_input))
        )
        center_state = MRTOCenterState(atom_logits=atom_logits, pair_logits=pair_logits, event_pair_weights=event_pair_weights)
        return a_plus, a_minus, p_plus, p_minus, event_h, reaction_h, center_state


class MRTOReactionEncoder(nn.Module):
    """Mapped Reaction Transition Operator with even/odd fields and event slots."""

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        dropout: float,
        attention_heads: int = 8,
        event_slots: int = 4,
        use_event_slots: bool = True,
        pair_update_scale: float = 1.0,
        reaction_update_scale: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MRTOTransitionBlock(
                    hidden_dim,
                    dropout,
                    attention_heads=attention_heads,
                    use_event_slots=use_event_slots,
                    event_slots=event_slots,
                    pair_update_scale=pair_update_scale,
                    reaction_update_scale=reaction_update_scale,
                )
                for _ in range(layers)
            ]
        )
        self.atom_out = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.pair_out = _mlp(hidden_dim * 2, hidden_dim, dropout)
        self.reaction_to_atom = nn.Linear(hidden_dim, hidden_dim)
        self.reaction_to_pair = nn.Linear(hidden_dim, hidden_dim)
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.reaction_norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_input: RFMEncoderInput) -> dict[str, torch.Tensor]:
        encoder_input.validate()
        atom_fields = encoder_input.metadata.get("mrto_atom_fields")
        if not atom_fields:
            raise ValueError("MRTOReactionEncoder requires MRTOReactionInputAdapter metadata")
        pair_fields = encoder_input.metadata.get("mrto_pair_fields")
        if not pair_fields:
            raise ValueError("MRTOReactionEncoder requires pair transition-field metadata")
        a_plus = atom_fields["a_plus"] * encoder_input.atom_valid_mask.unsqueeze(-1)
        a_minus = atom_fields["a_minus"] * encoder_input.atom_valid_mask.unsqueeze(-1)
        p_plus = pair_fields["p_plus"] * encoder_input.pair_valid_mask.unsqueeze(-1)
        p_minus = pair_fields["p_minus"] * encoder_input.pair_valid_mask.unsqueeze(-1)
        event_h: torch.Tensor | None = None
        reaction_h = encoder_input.reaction_token
        center_state: MRTOCenterState | None = None
        for layer in self.layers:
            a_plus, a_minus, p_plus, p_minus, event_h, reaction_h, center_state = layer(
                a_plus,
                a_minus,
                p_plus,
                p_minus,
                event_h,
                reaction_h,
                encoder_input.atom_valid_mask,
                encoder_input.pair_valid_mask,
            )

        reaction_h = self.reaction_norm(reaction_h)
        atom_h = self.atom_norm(
            self.atom_out(torch.cat([a_plus, a_minus], dim=-1))
            + self.reaction_to_atom(reaction_h).unsqueeze(1)
        ) * encoder_input.atom_valid_mask.unsqueeze(-1)
        pair_h = self.pair_norm(
            self.pair_out(torch.cat([p_plus, p_minus], dim=-1))
            + self.reaction_to_pair(reaction_h).unsqueeze(1).unsqueeze(1)
        ) * encoder_input.pair_valid_mask.unsqueeze(-1)
        out = {
            "atom_h": atom_h,
            "pair_h": pair_h,
            "reaction_h": reaction_h,
            "mrto_atom_plus": a_plus,
            "mrto_atom_minus": a_minus,
            "mrto_pair_plus": p_plus,
            "mrto_pair_minus": p_minus,
        }
        if center_state is not None:
            out["mrto_center_atom_logits"] = center_state.atom_logits
            out["mrto_center_pair_logits"] = center_state.pair_logits
            if center_state.event_pair_weights is not None:
                out["mrto_event_pair_weights"] = center_state.event_pair_weights
        return out
