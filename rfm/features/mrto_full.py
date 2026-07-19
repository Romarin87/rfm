"""Complete mapped reaction transition operator used by the Stage A track.

This module keeps the exact R/P even-odd contract from MRTO-v1 and replaces
independent event-slot attention with competitive sparse routing.  The final
event states also decode an unordered edit set through pair location, event
presence, and signed bond-order change predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from rfm.features.mrto_v1 import (
    GaussianRBF,
    LowRankParityTriangle,
    MRTOv1ReactionInputAdapter,
    OddMLP,
    OddRMSNorm,
    ParityAtomAttention,
    _even_mlp,
    _feature_projection,
    _masked_atom_mean,
    _masked_global_pair_mean,
    _masked_pair_mean,
)
from rfm.features.token_space import RFMEncoderInput


class EquiformerV2EndpointAdapter(nn.Module):
    """Shared endpoint-local EquiformerV2 adapter with invariant readout."""

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        sphere_channels: int,
        lmax: int,
        max_radius: float,
        max_neighbors: int,
        pair_rbf_bins: int,
        dropout: float,
    ):
        super().__init__()
        try:
            from suiren_models.model.EST_eqv2 import EST_Eqv2
        except ImportError as exc:
            raise ImportError(
                "mrto_full requires the Suiren EquiformerV2 package; add "
                "Suiren-Foundation-Model/src to PYTHONPATH"
            ) from exc
        self.max_radius = float(max_radius)
        self.backbone = EST_Eqv2(
            use_pbc=False,
            regress_forces=False,
            output_module=False,
            otf_graph=True,
            max_neighbors=max_neighbors,
            max_radius=max_radius,
            max_num_elements=90,
            num_layers=layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=max(32, sphere_channels),
            num_heads=4,
            attn_alpha_channels=max(16, sphere_channels // 2),
            attn_value_channels=max(8, sphere_channels // 4),
            ffn_hidden_channels=max(128, sphere_channels * 4),
            norm_type="layer_norm_sh",
            lmax_list=[lmax],
            mmax_list=[min(2, lmax)],
            edge_channels=max(32, sphere_channels),
            use_atom_edge_embedding=True,
            use_grid_mlp=True,
            alpha_drop=dropout,
            drop_path_rate=0.0,
            proj_drop=dropout,
            num_experts_steerable=1,
            num_experts_spherical=1,
            weight_init="uniform",
        )
        # EST_Eqv2 constructs its energy decoder even when output_module=False.
        # It is outside the endpoint adapter contract and must not enter DDP.
        for parameter in self.backbone.energy_block.parameters():
            parameter.requires_grad_(False)
        self.atom_projection = _feature_projection(sphere_channels, hidden_dim, dropout)
        self.pair_rbf = GaussianRBF(pair_rbf_bins)
        self.pair_projection = _feature_projection(hidden_dim * 2 + pair_rbf_bins, hidden_dim, dropout)

    @staticmethod
    def _as_graph_batch(z: torch.Tensor, coordinates: torch.Tensor, atom_mask: torch.Tensor):
        from torch_geometric.data import Data

        counts = atom_mask.sum(dim=1)
        flat_mask = atom_mask.reshape(-1)
        batch_index = torch.arange(z.shape[0], device=z.device).unsqueeze(1).expand_as(z).reshape(-1)[flat_mask]
        graph = Data(
            x=z.reshape(-1)[flat_mask],
            pos=coordinates.reshape(-1, 3)[flat_mask],
        )
        graph.batch = batch_index
        graph.ptr = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)])
        return graph

    def forward(
        self,
        z: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self._as_graph_batch(z, coordinates, atom_mask)
        # Suiren's EST spherical expert injects a CUDA-only random frame while
        # in training mode.  The deterministic EquiformerV2 path remains fully
        # differentiable and avoids introducing an arbitrary endpoint frame.
        self.backbone.eval()
        node_embedding = self.backbone(graph)["node_embedding"][:, 0, :]
        flat_atom = self.atom_projection(node_embedding)
        atom_h = flat_atom.new_zeros((*atom_mask.shape, flat_atom.shape[-1]))
        atom_h[atom_mask] = flat_atom

        distance = torch.cdist(coordinates.float(), coordinates.float())
        distance_basis = self.pair_rbf(distance / self.max_radius)
        atom_i = atom_h.unsqueeze(2).expand(-1, -1, atom_h.shape[1], -1)
        atom_j = atom_h.unsqueeze(1).expand(-1, atom_h.shape[1], -1, -1)
        pair_h = self.pair_projection(
            torch.cat([atom_i + atom_j, (atom_i - atom_j).abs(), distance_basis], dim=-1)
        )
        return atom_h * atom_mask.unsqueeze(-1), pair_h * pair_mask.unsqueeze(-1)


class MRTOFullReactionInputAdapter(MRTOv1ReactionInputAdapter):
    """Canonical fields plus a shared EquiformerV2 endpoint geometry adapter."""

    def __init__(
        self,
        *args,
        equiformer_layers: int = 2,
        equiformer_channels: int = 32,
        equiformer_lmax: int = 2,
        equiformer_radius: float = 5.0,
        equiformer_max_neighbors: int = 64,
        equiformer_adapter: EquiformerV2EndpointAdapter | None = None,
        suiren_injection_mode: str = "both",
        **kwargs,
    ):
        if suiren_injection_mode not in {
            "both",
            "initial_only",
            "conditioner_only",
            "parity_initial_only",
        }:
            raise ValueError(
                "suiren_injection_mode must be one of "
                "('both', 'initial_only', 'conditioner_only', 'parity_initial_only'), "
                f"got {suiren_injection_mode!r}"
            )
        self.suiren_injection_mode = suiren_injection_mode
        kwargs["enable_distance_geometry"] = False
        kwargs["inject_suiren_initial"] = suiren_injection_mode in {
            "both",
            "initial_only",
            "parity_initial_only",
        }
        kwargs["include_suiren_modality_tokens"] = suiren_injection_mode == "both"
        kwargs["parity_aligned_suiren_atom"] = suiren_injection_mode == "parity_initial_only"
        super().__init__(*args, **kwargs)
        has_geometry = self.featurizer.spec.has_r_3d or self.featurizer.spec.has_p_3d
        if equiformer_adapter is not None and not has_geometry:
            raise ValueError("an endpoint Equiformer cannot be attached to a 2D-only input schema")
        self.equivariant_geometry = equiformer_adapter
        if has_geometry and self.equivariant_geometry is None:
            self.equivariant_geometry = EquiformerV2EndpointAdapter(
                self.hidden_dim,
                equiformer_layers,
                equiformer_channels,
                equiformer_lmax,
                equiformer_radius,
                equiformer_max_neighbors,
                16,
                float(kwargs.get("dropout", 0.0)),
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> RFMEncoderInput:
        encoder_input = super().forward(batch)
        encoder_input.metadata["encoder_family"] = "mrto_full"
        encoder_input.metadata["suiren_injection_mode"] = self.suiren_injection_mode
        if self.suiren_injection_mode in {"initial_only", "parity_initial_only"}:
            encoder_input.metadata["mrto_suiren_atom_present"] = torch.zeros_like(
                encoder_input.metadata["mrto_suiren_atom_present"]
            )
            encoder_input.metadata["mrto_suiren_graph_present"] = torch.zeros_like(
                encoder_input.metadata["mrto_suiren_graph_present"]
            )
        if self.equivariant_geometry is None:
            return encoder_input
        if "coordinates_R" not in batch:
            raise ValueError("mrto_full IRC-3D input requires coordinates_R")

        atom_mask = encoder_input.atom_valid_mask
        pair_mask = encoder_input.pair_valid_mask
        geo_atom_r, geo_pair_r = self.equivariant_geometry(
            batch["z"], batch["coordinates_R"], atom_mask, pair_mask
        )
        if "coordinates_P" in batch:
            geo_atom_p, geo_pair_p = self.equivariant_geometry(
                batch["z"], batch["coordinates_P"], atom_mask, pair_mask
            )
        else:
            geo_atom_p = torch.zeros_like(geo_atom_r)
            geo_pair_p = torch.zeros_like(geo_pair_r)

        atom_fields = encoder_input.metadata["mrto_atom_fields"]
        pair_fields = encoder_input.metadata["mrto_pair_fields"]
        atom_fields["a_plus"] = atom_fields["a_plus"] + 0.5 * (geo_atom_r + geo_atom_p)
        atom_fields["a_minus"] = atom_fields["a_minus"] + 0.5 * (geo_atom_p - geo_atom_r)
        pair_fields["p_plus"] = pair_fields["p_plus"] + 0.5 * (geo_pair_r + geo_pair_p)
        pair_fields["p_minus"] = pair_fields["p_minus"] + 0.5 * (geo_pair_p - geo_pair_r)
        encoder_input.atom_tokens = (atom_fields["a_plus"] + atom_fields["a_minus"]) * atom_mask.unsqueeze(-1)
        encoder_input.pair_tokens = (pair_fields["p_plus"] + pair_fields["p_minus"]) * pair_mask.unsqueeze(-1)
        encoder_input.metadata["mrto_equivariant_geometry"] = {
            "backbone": "Suiren EST_Eqv2 / EquiformerV2",
            "endpoint_weight_sharing": True,
            "separate_endpoint_frames": True,
            "invariant_readout": "l=0 atom scalar + invariant pair projection",
        }
        encoder_input.validate()
        return encoder_input


class CompetitiveParityEventInteraction(nn.Module):
    """Extract sparse events with pair-to-slot competition and broadcast them."""

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
        if event_topk < 1:
            raise ValueError("complete MRTO requires mrto_event_topk >= 1")
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

    def _competitive_sparse_weights(
        self,
        logits: torch.Tensor,
        flat_mask: torch.Tensor,
        pair_center_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Route each candidate pair competitively, then normalize each slot."""

        candidate_count = logits.shape[-1]
        topk = min(self.event_topk, candidate_count)
        topk_indices = logits.topk(topk, dim=-1).indices
        keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_indices, True)
        keep &= flat_mask[:, None, None, :]

        # Slot softmax makes event slots compete for a candidate pair.  Entries
        # excluded by every top-k mask remain exactly zero after normalization.
        finite_logits = logits.masked_fill(~keep, -1.0e4)
        pair_to_slot = torch.softmax(finite_logits, dim=2) * keep.to(dtype=logits.dtype)
        center_gate = torch.sigmoid(pair_center_logits).unsqueeze(1).unsqueeze(1)
        weights = pair_to_slot * center_gate
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

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
        logits = logits + self.center_bias(flat_even).transpose(1, 2).unsqueeze(2)
        logits = logits + pair_center_logits.reshape(batch, 1, 1, n_atoms * n_atoms)
        weights = self._competitive_sparse_weights(
            logits,
            flat_mask,
            pair_center_logits.reshape(batch, n_atoms * n_atoms),
        )

        plus_value = self.plus_value(flat_even).view(batch, n_atoms * n_atoms, self.heads, self.head_dim).transpose(1, 2)
        minus_value = self.minus_value(flat_minus).view(batch, n_atoms * n_atoms, self.heads, self.head_dim).transpose(1, 2)
        context_plus = torch.matmul(self.dropout(weights), plus_value).transpose(1, 2).reshape(batch, self.event_slots, hidden)
        context_minus = torch.matmul(self.dropout(weights), minus_value).transpose(1, 2).reshape(batch, self.event_slots, hidden)
        event_plus = self.event_plus_norm(
            event_plus
            + self.dropout(self.event_plus_update(torch.cat([event_plus, context_plus, event_minus.square()], dim=-1)))
        )
        event_minus = self.event_minus_norm(
            event_minus
            + self.dropout(self.event_minus_update(torch.cat([event_minus, context_minus, event_plus * event_minus], dim=-1)))
        )

        event_pair_weights = weights.mean(dim=1).reshape(batch, self.event_slots, n_atoms, n_atoms)
        symmetric_weights = event_pair_weights + event_pair_weights.transpose(2, 3)
        pair_event_plus = torch.einsum("bkij,bkh->bijh", symmetric_weights, event_plus)
        pair_event_minus = torch.einsum("bkij,bkh->bijh", symmetric_weights, event_minus)
        pair_plus = self.pair_plus_norm(
            pair_plus
            + self.feedback_scale
            * self.dropout(self.pair_plus_feedback(torch.cat([pair_plus, pair_event_plus], dim=-1)))
        ) * pair_mask.unsqueeze(-1)
        pair_minus = self.pair_minus_norm(
            pair_minus
            + self.feedback_scale
            * self.dropout(self.pair_minus_feedback(torch.cat([pair_minus, pair_event_minus], dim=-1)))
        ) * pair_mask.unsqueeze(-1)

        atom_event_plus = _masked_pair_mean(pair_event_plus, pair_mask)
        atom_event_minus = _masked_pair_mean(pair_event_minus, pair_mask)
        atom_plus = self.atom_plus_norm(
            atom_plus
            + self.feedback_scale
            * self.dropout(self.atom_plus_feedback(torch.cat([atom_plus, atom_event_plus], dim=-1)))
        ) * atom_mask.unsqueeze(-1)
        atom_minus = self.atom_minus_norm(
            atom_minus
            + self.feedback_scale
            * self.dropout(self.atom_minus_feedback(torch.cat([atom_minus, atom_event_minus], dim=-1)))
        ) * atom_mask.unsqueeze(-1)
        return event_plus, event_minus, atom_plus, atom_minus, pair_plus, pair_minus, event_pair_weights


@dataclass
class MRTOFullCenterState:
    atom_logits: torch.Tensor
    pair_logits: torch.Tensor
    event_pair_weights: torch.Tensor


class ParityPriorConditioner(nn.Module):
    """AdaLN/FiLM-style Suiren prior conditioning inside an MRTO block."""

    def __init__(self, hidden_dim: int, dropout: float, gate_init: float):
        super().__init__()
        if not 0.0 <= gate_init < 1.0:
            raise ValueError(f"Suiren prior gate_init must be in [0,1), got {gate_init}")
        self.atom_even = _even_mlp(hidden_dim * 2, hidden_dim * 2, dropout)
        self.atom_odd = OddMLP(hidden_dim * 2, hidden_dim, dropout)
        self.atom_plus_norm = nn.LayerNorm(hidden_dim)
        self.atom_minus_norm = OddRMSNorm(hidden_dim)
        self.graph_even = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.graph_odd = OddMLP(hidden_dim * 2, hidden_dim, dropout)
        raw_gate_init = math.atanh(gate_init)
        self.atom_gate = nn.Parameter(torch.tensor(raw_gate_init))
        self.event_gate = nn.Parameter(torch.tensor(raw_gate_init))

    def forward(
        self,
        atom_plus: torch.Tensor,
        atom_minus: torch.Tensor,
        event_plus: torch.Tensor,
        event_minus: torch.Tensor,
        prior_atom_plus: torch.Tensor,
        prior_atom_minus: torch.Tensor,
        prior_graph_plus: torch.Tensor,
        prior_graph_minus: torch.Tensor,
        prior_atom_present: torch.Tensor,
        prior_graph_present: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_even_context = torch.cat([prior_atom_plus, prior_atom_minus.square()], dim=-1)
        gamma, beta = self.atom_even(atom_even_context).chunk(2, dim=-1)
        odd_shift = self.atom_odd(
            torch.cat([prior_atom_minus, prior_atom_plus * prior_atom_minus], dim=-1)
        )
        present_atom = prior_atom_present.to(dtype=atom_plus.dtype).view(-1, 1, 1)
        atom_scale = torch.tanh(self.atom_gate) * present_atom
        atom_plus = atom_plus + atom_scale * (
            torch.tanh(gamma) * self.atom_plus_norm(atom_plus) + beta
        )
        atom_minus = atom_minus + atom_scale * (
            torch.tanh(gamma) * self.atom_minus_norm(atom_minus) + odd_shift
        )

        graph_even_context = torch.cat([prior_graph_plus, prior_graph_minus.square()], dim=-1)
        graph_odd_context = torch.cat(
            [prior_graph_minus, prior_graph_plus * prior_graph_minus], dim=-1
        )
        present_event = prior_graph_present.to(dtype=event_plus.dtype).view(-1, 1, 1)
        event_scale = torch.tanh(self.event_gate) * present_event
        event_plus = event_plus + event_scale * self.graph_even(graph_even_context).unsqueeze(1)
        event_minus = event_minus + event_scale * self.graph_odd(graph_odd_context).unsqueeze(1)
        mask = atom_mask.unsqueeze(-1)
        return atom_plus * mask, atom_minus * mask, event_plus, event_minus


class MRTOFullTransitionBlock(nn.Module):
    """One complete pair-triangle, atom, event, and feedback operator block."""

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
        prior_gate_init: float,
    ):
        super().__init__()
        self.pair_update_scale = float(pair_update_scale)
        self.prior_conditioner = ParityPriorConditioner(hidden_dim, dropout, prior_gate_init)
        self.triangle = LowRankParityTriangle(hidden_dim, triangle_dim, dropout, triangle_scale) if use_triangle else None
        self.atom_attention = ParityAtomAttention(hidden_dim, heads, dropout)
        self.pair_plus_update = _even_mlp(hidden_dim * 4, hidden_dim, dropout)
        self.pair_minus_update = OddMLP(hidden_dim * 3, hidden_dim, dropout)
        self.pair_plus_norm = nn.LayerNorm(hidden_dim)
        self.pair_minus_norm = OddRMSNorm(hidden_dim)
        self.pair_center = _even_mlp(hidden_dim * 2, hidden_dim, dropout)
        self.pair_center_out = nn.Linear(hidden_dim, 1)
        self.events = CompetitiveParityEventInteraction(
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
        prior_atom_plus: torch.Tensor,
        prior_atom_minus: torch.Tensor,
        prior_graph_plus: torch.Tensor,
        prior_graph_minus: torch.Tensor,
        prior_atom_present: torch.Tensor,
        prior_graph_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MRTOFullCenterState]:
        atom_plus, atom_minus, event_plus, event_minus = self.prior_conditioner(
            atom_plus,
            atom_minus,
            event_plus,
            event_minus,
            prior_atom_plus,
            prior_atom_minus,
            prior_graph_plus,
            prior_graph_minus,
            prior_atom_present,
            prior_graph_present,
            atom_mask,
        )
        if self.triangle is not None:
            pair_plus, pair_minus = self.triangle(pair_plus, pair_minus, operator_pair_mask, pair_mask)
        atom_plus, atom_minus = self.atom_attention(atom_plus, atom_minus, pair_plus, pair_minus, atom_mask)

        plus_i = atom_plus.unsqueeze(2)
        plus_j = atom_plus.unsqueeze(1)
        minus_i = atom_minus.unsqueeze(2)
        minus_j = atom_minus.unsqueeze(1)
        even_pair_input = torch.cat(
            [pair_plus, plus_i + plus_j, minus_i * minus_j, (plus_i - plus_j).abs()], dim=-1
        )
        odd_pair_input = torch.cat(
            [pair_minus, minus_i + minus_j, plus_i * minus_j + minus_i * plus_j], dim=-1
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
        atom_logits = torch.logit(atom_probability.clamp(1e-6, 1.0 - 1e-6))
        atom_logits = atom_logits.masked_fill(~atom_mask, torch.finfo(pair_logits.dtype).min)
        center = MRTOFullCenterState(atom_logits, pair_logits, event_weights)
        return atom_plus, atom_minus, pair_plus, pair_minus, event_plus, event_minus, center


class MRTOFullReactionEncoder(nn.Module):
    """Full parity-typed reaction operator with an explicit edit-set bottleneck."""

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
        event_topk: int = 16,
        event_feedback_scale: float = 0.5,
        prior_gate_init: float = 0.1,
    ):
        super().__init__()
        if event_slots < 1:
            raise ValueError(f"event_slots must be positive, got {event_slots}")
        self.event_slots = int(event_slots)
        self.event_plus_init = nn.Parameter(torch.randn(event_slots, hidden_dim) * 0.02)
        self.context_to_event = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MRTOFullTransitionBlock(
                    hidden_dim,
                    dropout,
                    attention_heads,
                    event_slots,
                    triangle_dim,
                    triangle_scale,
                    index < triangle_layers,
                    pair_update_scale,
                    event_topk,
                    event_feedback_scale,
                    prior_gate_init,
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
        self.event_presence = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.event_delta_hidden = OddMLP(hidden_dim * 2, hidden_dim, dropout)
        self.event_delta_out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_input: RFMEncoderInput) -> dict[str, torch.Tensor]:
        encoder_input.validate()
        atom_fields = encoder_input.metadata.get("mrto_atom_fields")
        pair_fields = encoder_input.metadata.get("mrto_pair_fields")
        operator_pair_mask = encoder_input.metadata.get("mrto_operator_pair_mask")
        if not atom_fields or not pair_fields or operator_pair_mask is None:
            raise ValueError("MRTOFullReactionEncoder requires MRTOFullReactionInputAdapter metadata")
        atom_mask = encoder_input.atom_valid_mask
        pair_mask = encoder_input.pair_valid_mask
        atom_plus = atom_fields["a_plus"] * atom_mask.unsqueeze(-1)
        atom_minus = atom_fields["a_minus"] * atom_mask.unsqueeze(-1)
        pair_plus = pair_fields["p_plus"] * pair_mask.unsqueeze(-1)
        pair_minus = pair_fields["p_minus"] * pair_mask.unsqueeze(-1)
        batch = atom_plus.shape[0]
        event_plus = self.event_plus_init.unsqueeze(0).expand(batch, -1, -1)
        event_plus = event_plus + self.context_to_event(encoder_input.reaction_token).unsqueeze(1)
        context_minus = encoder_input.metadata.get("mrto_context_minus")
        event_minus = context_minus.unsqueeze(1).expand_as(event_plus) if context_minus is not None else torch.zeros_like(event_plus)
        prior_atom_plus = encoder_input.metadata.get("mrto_suiren_atom_plus", torch.zeros_like(atom_plus))
        prior_atom_minus = encoder_input.metadata.get("mrto_suiren_atom_minus", torch.zeros_like(atom_minus))
        prior_graph_plus = encoder_input.metadata.get(
            "mrto_suiren_graph_plus", torch.zeros_like(encoder_input.reaction_token)
        )
        prior_graph_minus = encoder_input.metadata.get(
            "mrto_suiren_graph_minus", torch.zeros_like(encoder_input.reaction_token)
        )
        prior_atom_present = encoder_input.metadata.get(
            "mrto_suiren_atom_present",
            torch.zeros(batch, dtype=torch.bool, device=atom_plus.device),
        )
        prior_graph_present = encoder_input.metadata.get(
            "mrto_suiren_graph_present",
            torch.zeros(batch, dtype=torch.bool, device=atom_plus.device),
        )

        center: MRTOFullCenterState | None = None
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
                prior_atom_plus,
                prior_atom_minus,
                prior_graph_plus,
                prior_graph_minus,
                prior_atom_present,
                prior_graph_present,
            )

        atom_plus_out = self.atom_plus_out(torch.cat([atom_plus, atom_minus.square()], dim=-1))
        atom_minus_out = self.atom_minus_out(atom_minus)
        pair_plus_out = self.pair_plus_out(torch.cat([pair_plus, pair_minus.square()], dim=-1))
        pair_minus_out = self.pair_minus_out(pair_minus)
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
        event_presence_logits = self.event_presence(torch.cat([event_plus, event_minus.square()], dim=-1)).squeeze(-1)
        event_delta_bo = self.event_delta_out(
            self.event_delta_hidden(torch.cat([event_minus, event_plus * event_minus], dim=-1))
        ).squeeze(-1)
        out = {
            "atom_h": (atom_plus_out + atom_minus_out) * atom_mask.unsqueeze(-1),
            "pair_h": (pair_plus_out + pair_minus_out) * pair_mask.unsqueeze(-1),
            "reaction_h": reaction_plus + reaction_minus,
            "mrto_atom_plus": atom_plus,
            "mrto_atom_minus": atom_minus,
            "mrto_pair_plus": pair_plus,
            "mrto_pair_minus": pair_minus,
            "mrto_event_plus": event_plus,
            "mrto_event_minus": event_minus,
            "mrto_reaction_plus": reaction_plus,
            "mrto_reaction_minus": reaction_minus,
            "mrto_event_presence_logits": event_presence_logits,
            "mrto_event_delta_bo": event_delta_bo,
        }
        if center is not None:
            out["mrto_center_atom_logits"] = center.atom_logits
            out["mrto_center_pair_logits"] = center.pair_logits
            out["mrto_event_pair_weights"] = center.event_pair_weights
        return out

    def suiren_prior_gate_values(self) -> list[dict[str, float | int]]:
        return [
            {
                "layer": index,
                "atom": float(torch.tanh(block.prior_conditioner.atom_gate).detach().cpu()),
                "graph_event": float(torch.tanh(block.prior_conditioner.event_gate).detach().cpu()),
            }
            for index, block in enumerate(self.layers)
        ]
