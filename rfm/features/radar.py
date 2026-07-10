"""RADAR reaction encoder components.

RADAR is a reaction-specific encoder for atom-mapped R/P states. It keeps the
public task-head contract unchanged: atom_h [B,N,H], pair_h [B,N,N,H], and
reaction_h [B,H].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rfm.features.token_space import (
    BaseRFMAdapter,
    RFMEncoderInput,
    ReactionInputFeaturizer,
    batch_bool,
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
        elif group == "delta":
            keep = (
                "delta" in lower
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


class RADARReactionInputAdapter(BaseRFMAdapter):
    """Build explicit R/P/delta atom streams and a pair reaction-edit stream."""

    def __init__(
        self,
        hidden_dim: int,
        pair_input_dim: int,
        input_schema: str | None,
        task_name: str,
        suiren_atom_dim: int = 0,
        suiren_graph_dim: int = 0,
        suiren_atom_dims: dict[str, int] | None = None,
        suiren_graph_dims: dict[str, int] | None = None,
        max_z: int = 36,
        distance_clip: float = 10.0,
        enable_delta_stream: bool = True,
        enable_suiren_gates: bool = False,
    ):
        super().__init__(hidden_dim, task_name=task_name)
        schema = input_schema or infer_pair_basis_schema(pair_input_dim, "property" if "property" in task_name else "masked_edit")
        self.featurizer = ReactionInputFeaturizer(schema, distance_clip=distance_clip)
        self.enable_delta_stream = bool(enable_delta_stream)
        self.enable_suiren_gates = bool(enable_suiren_gates)
        self.suiren_atom_dims = dict(suiren_atom_dims or {})
        self.suiren_graph_dims = dict(suiren_graph_dims or {})
        if suiren_atom_dim > 0 and not self.suiren_atom_dims:
            self.suiren_atom_dims["generic"] = suiren_atom_dim
        if suiren_graph_dim > 0 and not self.suiren_graph_dims:
            self.suiren_graph_dims["generic"] = suiren_graph_dim

        basis_names = self.featurizer.spec.names
        r_indices = _channel_indices(basis_names, "r")
        p_indices = _channel_indices(basis_names, "p")
        delta_indices = _channel_indices(basis_names, "delta")
        self.register_buffer("r_channel_indices", torch.tensor(r_indices, dtype=torch.long), persistent=False)
        self.register_buffer("p_channel_indices", torch.tensor(p_indices, dtype=torch.long), persistent=False)
        self.register_buffer("delta_channel_indices", torch.tensor(delta_indices, dtype=torch.long), persistent=False)

        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.atom_r_projection = nn.Sequential(nn.LayerNorm(len(r_indices)), nn.Linear(len(r_indices), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.atom_p_projection = nn.Sequential(nn.LayerNorm(len(p_indices)), nn.Linear(len(p_indices), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.atom_delta_projection = nn.Sequential(nn.LayerNorm(len(delta_indices)), nn.Linear(len(delta_indices), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_r_projection = nn.Sequential(nn.LayerNorm(len(r_indices)), nn.Linear(len(r_indices), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_p_projection = nn.Sequential(nn.LayerNorm(len(p_indices)), nn.Linear(len(p_indices), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_delta_projection = nn.Sequential(nn.LayerNorm(len(delta_indices)), nn.Linear(len(delta_indices), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_combine = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.atom_combine = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        invalid_suiren_dims = {name: dim for name, dim in self.suiren_atom_dims.items() if dim <= 0 or dim % 4}
        if invalid_suiren_dims:
            raise ValueError(
                "RADAR atom-level Suiren features require [h_R,h_P,Delta_h,abs_Delta_h] "
                f"with four equal-width blocks, got {invalid_suiren_dims}"
            )
        self.suiren_atom_state_dims = {name: dim // 4 for name, dim in self.suiren_atom_dims.items()}
        self.suiren_atom_r_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_atom_state_dims.items())
            }
        )
        self.suiren_atom_p_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_atom_state_dims.items())
            }
        )
        self.suiren_atom_delta_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_atom_state_dims.items())
            }
        )
        self.suiren_graph_projection = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for name, dim in sorted(self.suiren_graph_dims.items())
                if dim > 0
            }
        )
        self.suiren_atom_gate_logits = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.suiren_atom_r_projection})
        self.suiren_graph_gate_logits = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.suiren_graph_projection})

    @staticmethod
    def suiren_feature_key(stream: str, level: str) -> str:
        if stream == "generic":
            return f"suiren_{level}_features"
        return f"suiren_{stream}_{level}_features"

    def _suiren_scale(self, stream: str, level: str) -> torch.Tensor:
        if not self.enable_suiren_gates:
            return torch.ones((), device=self.r_channel_indices.device)
        logits = self.suiren_atom_gate_logits[stream] if level == "atom" else self.suiren_graph_gate_logits[stream]
        return 2.0 * torch.sigmoid(logits)

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
        delta_basis = pair_basis.index_select(-1, self.delta_channel_indices)
        z_token = self.z_embedding(z)
        r_summary = _masked_pair_mean(r_basis, pair_valid)
        p_summary = _masked_pair_mean(p_basis, pair_valid)
        delta_summary = _masked_pair_mean(delta_basis, pair_valid)

        a_r = z_token + self.atom_r_projection(r_summary)
        a_p = z_token + self.atom_p_projection(p_summary)
        if self.enable_delta_stream:
            a_delta = self.atom_delta_projection(delta_summary)
        else:
            a_delta = torch.zeros_like(a_r)

        has_atom = False
        for stream, projection in self.suiren_atom_r_projection.items():
            key = self.suiren_feature_key(stream, "atom")
            if key not in batch:
                continue
            has_atom = True
            feature = batch[key]
            if feature.shape[:2] != a_r.shape[:2]:
                raise ValueError(f"{key} must be [B,N,C], got {tuple(feature.shape)}")
            state_dim = self.suiren_atom_state_dims[stream]
            if feature.shape[-1] != state_dim * 4:
                raise ValueError(f"{key} must have four {state_dim}-wide blocks, got {tuple(feature.shape)}")
            h_r, h_p, delta_h, abs_delta_h = feature.split(state_dim, dim=-1)
            scale = self._suiren_scale(stream, "atom")
            a_r = a_r + scale * projection(h_r)
            a_p = a_p + scale * self.suiren_atom_p_projection[stream](h_p)
            if self.enable_delta_stream:
                delta_feature = torch.cat([delta_h, abs_delta_h], dim=-1)
                a_delta = a_delta + scale * self.suiren_atom_delta_projection[stream](delta_feature)

        a_r = a_r * atom_valid.unsqueeze(-1)
        a_p = a_p * atom_valid.unsqueeze(-1)
        a_delta = a_delta * atom_valid.unsqueeze(-1)
        pair_r = self.pair_r_projection(r_basis) * pair_valid.unsqueeze(-1)
        pair_p = self.pair_p_projection(p_basis) * pair_valid.unsqueeze(-1)
        pair_delta = self.pair_delta_projection(delta_basis) * pair_valid.unsqueeze(-1)
        pair_tokens = self.pair_combine(torch.cat([pair_r, pair_p, pair_delta], dim=-1)) * pair_valid.unsqueeze(-1)
        atom_tokens = self.atom_combine(torch.cat([a_r, a_p, a_delta], dim=-1)) * atom_valid.unsqueeze(-1)

        spec = self.featurizer.spec
        has_graph = any(self.suiren_feature_key(stream, "graph") in batch for stream in self.suiren_graph_projection)
        has_suiren = bool(has_atom or has_graph)
        modality = self.modality_dict(
            z.shape[0],
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
            feature = batch[key]
            if feature.shape[0] != z.shape[0]:
                raise ValueError(f"{key} must be [B,C], got {tuple(feature.shape)}")
            reaction_token = reaction_token + self._suiren_scale(stream, "graph") * projection(feature)

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
                "encoder_family": "radar",
                "radar_state_streams": {
                    "a_r": a_r,
                    "a_p": a_p,
                    "a_delta": a_delta,
                },
                "radar_pair_state_streams": {
                    "pair_r": pair_r,
                    "pair_p": pair_p,
                    "pair_delta": pair_delta,
                },
                "suiren_atom_dims": self.suiren_atom_dims,
                "suiren_atom_state_dims": self.suiren_atom_state_dims,
                "suiren_atom_layout": ("h_R", "h_P", "Delta_h", "abs_Delta_h"),
                "suiren_graph_dims": self.suiren_graph_dims,
                "suiren_input_gates": self.enable_suiren_gates,
                "delta_atom_stream": self.enable_delta_stream,
            },
        )
        out.validate()
        return out


class PairBiasedSelfAttention(nn.Module):
    """Multi-head atom self-attention with scalar pair bias per head."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.pair_bias = nn.Linear(hidden_dim, self.num_heads, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h: torch.Tensor, pair_h: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        batch, n_atoms, hidden = atom_h.shape
        qkv = self.qkv(atom_h).view(batch, n_atoms, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores + self.pair_bias(pair_h).permute(0, 3, 1, 2)
        key_mask = atom_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(batch, n_atoms, hidden)
        return self.norm(atom_h + self.dropout(self.out(context))) * atom_mask.unsqueeze(-1)


@dataclass
class CenterState:
    atom_logits: torch.Tensor
    pair_logits: torch.Tensor


def _soft_pool(values: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, invert: bool = False) -> torch.Tensor:
    """Return a normalized center/environment pool without a zero-weight escape."""

    scores = -logits if invert else logits
    if values.ndim == 4:
        batch, n_atoms, _, hidden = values.shape
        flat_mask = mask.reshape(batch, n_atoms * n_atoms)
        flat_scores = scores.reshape(batch, n_atoms * n_atoms)
        flat_values = values.reshape(batch, n_atoms * n_atoms, hidden)
        masked_scores = flat_scores.masked_fill(~flat_mask, torch.finfo(flat_scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=1) * flat_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (flat_values * weights.unsqueeze(-1)).sum(dim=1)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=1) * mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (values * weights.unsqueeze(-1)).sum(dim=1)


class RADARStateTransitionBlock(nn.Module):
    """One dual-state reaction transition block."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        attention_heads: int,
        center_router: bool,
        pair_update_scale: float,
        reaction_update_scale: float,
    ):
        super().__init__()
        self.center_router = bool(center_router)
        self.pair_update_scale = float(pair_update_scale)
        self.reaction_update_scale = float(reaction_update_scale)
        self.r_state_attention = PairBiasedSelfAttention(hidden_dim, attention_heads, dropout)
        self.p_state_attention = PairBiasedSelfAttention(hidden_dim, attention_heads, dropout)
        self.delta_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.delta_norm = nn.LayerNorm(hidden_dim)
        self.pair_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.atom_center = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pair_center = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        if not self.center_router:
            for module in (self.atom_center, self.pair_center):
                for param in module.parameters():
                    param.requires_grad_(False)
        self.reaction_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reaction_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        a_r: torch.Tensor,
        a_p: torch.Tensor,
        a_delta: torch.Tensor,
        pair_r_bias: torch.Tensor,
        pair_p_bias: torch.Tensor,
        pair_h: torch.Tensor,
        reaction_h: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CenterState]:
        a_r = self.r_state_attention(a_r, pair_r_bias, atom_mask)
        a_p = self.p_state_attention(a_p, pair_p_bias, atom_mask)
        atom_pair_summary = (pair_h * pair_mask.unsqueeze(-1)).sum(dim=2) / pair_mask.sum(dim=2).clamp_min(1).float().unsqueeze(-1)
        delta_input = torch.cat([a_delta, a_p - a_r, (a_p - a_r).abs(), atom_pair_summary], dim=-1)
        a_delta = self.delta_norm(a_delta + self.dropout(self.delta_update(delta_input))) * atom_mask.unsqueeze(-1)

        batch, n_atoms, hidden = a_r.shape
        ar_pair = a_r.unsqueeze(2) + a_r.unsqueeze(1)
        ap_pair = a_p.unsqueeze(2) + a_p.unsqueeze(1)
        ad_pair = a_delta.unsqueeze(2) + a_delta.unsqueeze(1)
        ad_diff = (a_delta.unsqueeze(2) - a_delta.unsqueeze(1)).abs()
        pair_input = torch.cat([pair_h, ar_pair, ap_pair, ad_pair, ad_diff], dim=-1)
        pair_h = self.pair_norm(
            pair_h + self.pair_update_scale * self.dropout(self.pair_update(pair_input))
        ) * pair_mask.unsqueeze(-1)

        if self.center_router:
            atom_logits = self.atom_center(torch.cat([a_delta, atom_pair_summary], dim=-1)).squeeze(-1)
            pair_logits = self.pair_center(pair_h).squeeze(-1)
        else:
            atom_logits = torch.zeros_like(atom_mask, dtype=pair_h.dtype)
            pair_logits = torch.zeros_like(pair_mask, dtype=pair_h.dtype)
        atom_logits = atom_logits.masked_fill(~atom_mask.bool(), torch.finfo(atom_logits.dtype).min)
        pair_logits = pair_logits.masked_fill(~pair_mask.bool(), torch.finfo(pair_logits.dtype).min)

        atom_mix = (a_r + a_p + a_delta) / 3.0
        center_atom = _soft_pool(atom_mix, atom_logits, atom_mask)
        env_atom = _soft_pool(atom_mix, atom_logits, atom_mask, invert=True)
        center_pair = _soft_pool(pair_h, pair_logits, pair_mask)
        env_pair = _soft_pool(pair_h, pair_logits, pair_mask, invert=True)
        reaction_input = torch.cat([reaction_h, center_atom, env_atom, center_pair, env_pair], dim=-1)
        reaction_h = self.reaction_norm(
            reaction_h + self.reaction_update_scale * self.dropout(self.reaction_update(reaction_input))
        )
        return a_r, a_p, a_delta, pair_h, reaction_h, CenterState(atom_logits=atom_logits, pair_logits=pair_logits)


class RADARReactionEncoder(nn.Module):
    """Reaction-aligned dual-state encoder with adaptive pair routing."""

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        dropout: float,
        attention_heads: int = 8,
        center_router: bool = True,
        pair_update_scale: float = 0.75,
        reaction_update_scale: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RADARStateTransitionBlock(
                    hidden_dim,
                    dropout,
                    attention_heads,
                    center_router=center_router,
                    pair_update_scale=pair_update_scale,
                    reaction_update_scale=reaction_update_scale,
                )
                for _ in range(layers)
            ]
        )
        self.atom_out = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.reaction_to_atom = nn.Linear(hidden_dim, hidden_dim)
        self.reaction_to_pair = nn.Linear(hidden_dim, hidden_dim)
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.reaction_norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_input: RFMEncoderInput) -> dict[str, torch.Tensor]:
        encoder_input.validate()
        streams = encoder_input.metadata.get("radar_state_streams")
        if not streams:
            raise ValueError("RADARReactionEncoder requires RADARReactionInputAdapter metadata")
        pair_streams = encoder_input.metadata.get("radar_pair_state_streams")
        if not pair_streams:
            raise ValueError("RADARReactionEncoder requires state-specific pair-bias streams")
        a_r = streams["a_r"] * encoder_input.atom_valid_mask.unsqueeze(-1)
        a_p = streams["a_p"] * encoder_input.atom_valid_mask.unsqueeze(-1)
        a_delta = streams["a_delta"] * encoder_input.atom_valid_mask.unsqueeze(-1)
        pair_r_bias = pair_streams["pair_r"] * encoder_input.pair_valid_mask.unsqueeze(-1)
        pair_p_bias = pair_streams["pair_p"] * encoder_input.pair_valid_mask.unsqueeze(-1)
        pair_h = encoder_input.pair_tokens * encoder_input.pair_valid_mask.unsqueeze(-1)
        reaction_h = encoder_input.reaction_token
        center_state: CenterState | None = None
        for layer in self.layers:
            a_r, a_p, a_delta, pair_h, reaction_h, center_state = layer(
                a_r,
                a_p,
                a_delta,
                pair_r_bias,
                pair_p_bias,
                pair_h,
                reaction_h,
                encoder_input.atom_valid_mask,
                encoder_input.pair_valid_mask,
            )
        reaction_h = self.reaction_norm(reaction_h)
        atom_h = self.atom_norm(
            self.atom_out(torch.cat([a_r, a_p, a_delta], dim=-1))
            + self.reaction_to_atom(reaction_h).unsqueeze(1)
        ) * encoder_input.atom_valid_mask.unsqueeze(-1)
        pair_h = self.pair_norm(
            pair_h + self.reaction_to_pair(reaction_h).unsqueeze(1).unsqueeze(1)
        ) * encoder_input.pair_valid_mask.unsqueeze(-1)
        out = {
            "atom_h": atom_h,
            "pair_h": pair_h,
            "reaction_h": reaction_h,
        }
        if center_state is not None:
            out["radar_center_atom_logits"] = center_state.atom_logits
            out["radar_center_pair_logits"] = center_state.pair_logits
        return out
