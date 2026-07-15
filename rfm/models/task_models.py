"""Task models for the current RFM restart path."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from rfm.data.reaction_samples import EDIT_CLASSES, ENERGY_TARGETS
from rfm.features.mrto import MRTOReactionEncoder, MRTOReactionInputAdapter
from rfm.features.mrto_full import MRTOFullReactionEncoder, MRTOFullReactionInputAdapter
from rfm.features.mrto_v1 import MRTOv1ReactionEncoder, MRTOv1ReactionInputAdapter
from rfm.features.radar import RADARReactionEncoder, RADARReactionInputAdapter
from rfm.features.token_space import (
    MaskedEditAdapter,
    RFMEncoderInput,
    RPairPropertyAdapter,
    TokenSpaceReactionEncoder,
    UnifiedReactionInputAdapter,
)


def _concat_mrto_v1_inputs(first: RFMEncoderInput, second: RFMEncoderInput) -> RFMEncoderInput:
    """Batch two parity-typed MRTO inputs without changing their semantics."""

    def concat_field(group: str) -> dict[str, torch.Tensor]:
        first_fields = first.metadata[group]
        second_fields = second.metadata[group]
        if first_fields.keys() != second_fields.keys():
            raise ValueError(f"MRTO input field mismatch for {group}")
        return {key: torch.cat([first_fields[key], second_fields[key]], dim=0) for key in first_fields}

    batch_first = first.atom_tokens.shape[0]
    batch_second = second.atom_tokens.shape[0]
    modality_names = set(first.modality_mask) | set(second.modality_mask)
    modality = {}
    for name in modality_names:
        first_mask = first.modality_mask.get(name)
        second_mask = second.modality_mask.get(name)
        if first_mask is None:
            first_mask = torch.zeros(batch_first, dtype=torch.bool, device=first.atom_tokens.device)
        if second_mask is None:
            second_mask = torch.zeros(batch_second, dtype=torch.bool, device=second.atom_tokens.device)
        modality[name] = torch.cat([first_mask, second_mask], dim=0)

    context_first = first.metadata.get("mrto_context_minus")
    context_second = second.metadata.get("mrto_context_minus")
    if context_first is None:
        context_first = torch.zeros_like(first.reaction_token)
    if context_second is None:
        context_second = torch.zeros_like(second.reaction_token)

    def concat_optional(key: str, first_default: torch.Tensor, second_default: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [first.metadata.get(key, first_default), second.metadata.get(key, second_default)],
            dim=0,
        )

    first_atom_zero = torch.zeros_like(first.metadata["mrto_atom_fields"]["a_plus"])
    second_atom_zero = torch.zeros_like(second.metadata["mrto_atom_fields"]["a_plus"])
    first_graph_zero = torch.zeros_like(first.reaction_token)
    second_graph_zero = torch.zeros_like(second.reaction_token)
    out = RFMEncoderInput(
        atom_tokens=torch.cat([first.atom_tokens, second.atom_tokens], dim=0),
        pair_tokens=torch.cat([first.pair_tokens, second.pair_tokens], dim=0),
        reaction_token=torch.cat([first.reaction_token, second.reaction_token], dim=0),
        atom_valid_mask=torch.cat([first.atom_valid_mask, second.atom_valid_mask], dim=0),
        pair_valid_mask=torch.cat([first.pair_valid_mask, second.pair_valid_mask], dim=0),
        modality_mask=modality,
        task_name="joint_property_masked_edit",
        metadata={
            "encoder_family": first.metadata.get("encoder_family", "mrto_v1"),
            "mrto_atom_fields": concat_field("mrto_atom_fields"),
            "mrto_pair_fields": concat_field("mrto_pair_fields"),
            "mrto_operator_pair_mask": torch.cat(
                [first.metadata["mrto_operator_pair_mask"], second.metadata["mrto_operator_pair_mask"]],
                dim=0,
            ),
            "mrto_context_minus": torch.cat([context_first, context_second], dim=0),
            "mrto_suiren_atom_plus": concat_optional(
                "mrto_suiren_atom_plus", first_atom_zero, second_atom_zero
            ),
            "mrto_suiren_atom_minus": concat_optional(
                "mrto_suiren_atom_minus", first_atom_zero, second_atom_zero
            ),
            "mrto_suiren_graph_plus": concat_optional(
                "mrto_suiren_graph_plus", first_graph_zero, second_graph_zero
            ),
            "mrto_suiren_graph_minus": concat_optional(
                "mrto_suiren_graph_minus", first_graph_zero, second_graph_zero
            ),
            "mrto_suiren_present": concat_optional(
                "mrto_suiren_present",
                torch.zeros(batch_first, dtype=torch.bool, device=first.atom_tokens.device),
                torch.zeros(batch_second, dtype=torch.bool, device=second.atom_tokens.device),
            ),
        },
    )
    out.validate()
    return out


def _split_encoded_batch(
    encoded: dict[str, torch.Tensor],
    first_batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    first: dict[str, torch.Tensor] = {}
    second: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        first[key] = value[:first_batch_size]
        second[key] = value[first_batch_size:]
    return first, second


class AtomAttentionReadout(nn.Module):
    """Masked attention pooling over atom states with the reaction token as query."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

    def forward(self, atom_h: torch.Tensor, atom_mask: torch.Tensor, reaction_h: torch.Tensor) -> torch.Tensor:
        query = self.query(reaction_h).unsqueeze(1)
        key = self.key(atom_h)
        scores = (query * key).sum(dim=-1) * self.scale
        scores = scores.masked_fill(~atom_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        value = self.value(atom_h)
        return (weights.unsqueeze(-1) * value).sum(dim=1)


class MRTOParityEnergyHead(nn.Module):
    """Predict odd reaction energy and even symmetric barrier in raw units."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.energy_odd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.symmetric_barrier_even = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        d_e = self.energy_odd(encoded["mrto_reaction_minus"]).squeeze(-1)
        symmetric_barrier = self.symmetric_barrier_even(encoded["mrto_reaction_plus"]).squeeze(-1)
        d_e_dagger = symmetric_barrier + 0.5 * d_e
        return torch.stack([d_e, d_e_dagger], dim=-1)


class MaskedEditHeads(nn.Module):
    """Masked Delta_BO, changed pair, edit class, and core atom heads."""

    def __init__(
        self,
        hidden_dim: int,
        router_residual: bool = False,
        router_gate_init: float = 0.05,
        reaction_context: bool = False,
    ):
        super().__init__()
        self.router_residual = bool(router_residual)
        self.reaction_context = bool(reaction_context)
        pair_dim = hidden_dim * (4 if self.reaction_context else 3)
        atom_dim = hidden_dim * (2 if self.reaction_context else 1)
        self.delta_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.changed_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.edit_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(EDIT_CLASSES)))
        self.core_head = nn.Sequential(nn.LayerNorm(atom_dim), nn.Linear(atom_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        if self.router_residual:
            if not 0.0 < router_gate_init < 1.0:
                raise ValueError(f"router_gate_init must be in (0,1), got {router_gate_init}")
            gate_logit = math.log(router_gate_init / (1.0 - router_gate_init))
            self.router_pair_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
            self.router_atom_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    def router_gate_values(self) -> dict[str, float]:
        if not self.router_residual:
            return {}
        return {
            "pair": float(torch.sigmoid(self.router_pair_gate_logit).detach().cpu()),
            "atom": float(torch.sigmoid(self.router_atom_gate_logit).detach().cpu()),
        }

    def forward(
        self,
        atom_h: torch.Tensor,
        pair_h: torch.Tensor,
        *,
        reaction_h: torch.Tensor | None = None,
        router_atom_logits: torch.Tensor | None = None,
        router_pair_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, n_atoms, hidden = atom_h.shape
        h_i = atom_h.unsqueeze(2).expand(batch, n_atoms, n_atoms, hidden)
        h_j = atom_h.unsqueeze(1).expand(batch, n_atoms, n_atoms, hidden)
        pair_repr = torch.cat([h_i, h_j, pair_h], dim=-1)
        atom_repr = atom_h
        if self.reaction_context:
            if reaction_h is None:
                raise ValueError("reaction-context masked-edit heads require reaction_h")
            pair_context = reaction_h[:, None, None, :].expand(batch, n_atoms, n_atoms, hidden)
            atom_context = reaction_h[:, None, :].expand(batch, n_atoms, hidden)
            pair_repr = torch.cat([pair_repr, pair_context], dim=-1)
            atom_repr = torch.cat([atom_h, atom_context], dim=-1)
        changed_logits = self.changed_head(pair_repr).squeeze(-1)
        core_logits = self.core_head(atom_repr).squeeze(-1)
        out: dict[str, torch.Tensor] = {
            "delta_bo": self.delta_head(pair_repr).squeeze(-1),
            "changed_logits": changed_logits,
            "edit_logits": self.edit_head(pair_repr),
            "core_logits": core_logits,
        }
        if self.router_residual:
            if router_atom_logits is None or router_pair_logits is None:
                raise ValueError("RADAR masked-edit heads require atom and pair router logits")
            pair_gate = torch.sigmoid(self.router_pair_gate_logit)
            atom_gate = torch.sigmoid(self.router_atom_gate_logit)
            out["changed_logits"] = changed_logits + pair_gate * torch.tanh(router_pair_logits)
            out["core_logits"] = core_logits + atom_gate * torch.tanh(router_atom_logits)
            out["router_pair_logits"] = router_pair_logits
            out["router_atom_logits"] = router_atom_logits
        return out


def _router_logits(encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    atom_logits = encoded.get("radar_center_atom_logits")
    pair_logits = encoded.get("radar_center_pair_logits")
    if atom_logits is None:
        atom_logits = encoded.get("mrto_center_atom_logits")
    if pair_logits is None:
        pair_logits = encoded.get("mrto_center_pair_logits")
    return atom_logits, pair_logits


def _masked_edit_outputs(heads: MaskedEditHeads, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    router_atom_logits, router_pair_logits = _router_logits(encoded)
    return heads(
        encoded["atom_h"],
        encoded["pair_h"],
        reaction_h=encoded["reaction_h"],
        router_atom_logits=router_atom_logits,
        router_pair_logits=router_pair_logits,
    )


class MaskedEditPretrainingModel(nn.Module):
    """Masked edit model with strict raw inputs: Z + masked BO/edit basis + optional D_R."""

    def __init__(
        self,
        pair_raw_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        input_schema: str,
        dynamic_pair_update: bool = False,
        dynamic_pair_update_scale: float = 1.0,
        dynamic_pair_update_dropout: float | None = None,
        encoder_type: str = "token_space",
        radar_attention_heads: int = 8,
        radar_center_router: bool = True,
        radar_delta_stream: bool = True,
        radar_pair_update_scale: float = 0.75,
        radar_reaction_update_scale: float = 1.0,
        radar_router_gate_init: float = 0.05,
        mrto_attention_heads: int = 8,
        mrto_event_slots: int = 4,
        mrto_use_event_slots: bool = True,
        mrto_use_odd_field: bool = True,
        mrto_pair_update_scale: float = 1.0,
        mrto_reaction_update_scale: float = 1.0,
        mrto_endpoint_layers: int = 2,
        mrto_triangle_layers: int = 2,
        mrto_triangle_dim: int = 16,
        mrto_triangle_scale: float = 0.5,
        mrto_event_topk: int = 0,
        mrto_event_feedback_scale: float = 0.5,
        mrto_geometry_rbf_bins: int = 16,
        mrto_equiformer_layers: int = 2,
        mrto_equiformer_channels: int = 32,
        mrto_equiformer_lmax: int = 2,
        mrto_equiformer_radius: float = 5.0,
        mrto_equiformer_max_neighbors: int = 64,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type in {"mrto_v1", "mrto_full"}:
            if not mrto_use_event_slots:
                raise ValueError(f"encoder_type={encoder_type!r} requires event slots")
            adapter_class = MRTOFullReactionInputAdapter if encoder_type == "mrto_full" else MRTOv1ReactionInputAdapter
            encoder_class = MRTOFullReactionEncoder if encoder_type == "mrto_full" else MRTOv1ReactionEncoder
            full_adapter_kwargs = (
                {
                    "equiformer_layers": mrto_equiformer_layers,
                    "equiformer_channels": mrto_equiformer_channels,
                    "equiformer_lmax": mrto_equiformer_lmax,
                    "equiformer_radius": mrto_equiformer_radius,
                    "equiformer_max_neighbors": mrto_equiformer_max_neighbors,
                }
                if encoder_type == "mrto_full"
                else {}
            )
            self.adapter = adapter_class(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="masked_edit",
                use_odd_field=mrto_use_odd_field,
                endpoint_layers=mrto_endpoint_layers,
                geometry_rbf_bins=mrto_geometry_rbf_bins,
                dropout=dropout,
                **full_adapter_kwargs,
            )
            self.encoder = encoder_class(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=mrto_attention_heads,
                event_slots=mrto_event_slots,
                triangle_layers=mrto_triangle_layers,
                triangle_dim=mrto_triangle_dim,
                triangle_scale=mrto_triangle_scale,
                pair_update_scale=mrto_pair_update_scale,
                event_topk=mrto_event_topk,
                event_feedback_scale=mrto_event_feedback_scale,
            )
        elif encoder_type == "mrto":
            self.adapter = MRTOReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="masked_edit",
                use_odd_field=mrto_use_odd_field,
            )
            self.encoder = MRTOReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=mrto_attention_heads,
                event_slots=mrto_event_slots,
                use_event_slots=mrto_use_event_slots,
                pair_update_scale=mrto_pair_update_scale,
                reaction_update_scale=mrto_reaction_update_scale,
            )
        elif encoder_type == "radar":
            self.adapter = RADARReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="masked_edit",
                enable_delta_stream=radar_delta_stream,
            )
            self.encoder = RADARReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=radar_attention_heads,
                center_router=radar_center_router,
                pair_update_scale=radar_pair_update_scale,
                reaction_update_scale=radar_reaction_update_scale,
            )
        elif encoder_type == "token_space":
            self.adapter = MaskedEditAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
            )
            self.encoder = TokenSpaceReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                dynamic_pair_update=dynamic_pair_update,
                dynamic_pair_update_scale=dynamic_pair_update_scale,
                dynamic_pair_update_dropout=dynamic_pair_update_dropout,
            )
        else:
            raise ValueError(f"unsupported encoder_type={encoder_type!r}")
        self.masked_edit_heads = MaskedEditHeads(
            hidden_dim,
            router_residual=(encoder_type == "radar" and radar_center_router)
            or encoder_type in {"mrto", "mrto_v1", "mrto_full"},
            router_gate_init=radar_router_gate_init,
            reaction_context=encoder_type == "mrto_full",
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoder_input = self.adapter(batch)
        encoded = self.encoder(encoder_input)
        out = _masked_edit_outputs(self.masked_edit_heads, encoded)
        out["reaction_h"] = encoded["reaction_h"]
        for key in (
            "mrto_event_pair_weights",
            "mrto_event_plus",
            "mrto_event_minus",
            "mrto_reaction_plus",
            "mrto_reaction_minus",
            "mrto_event_presence_logits",
            "mrto_event_delta_bo",
        ):
            if key in encoded:
                out[key] = encoded[key]
        return out


class ReactionPropertyRegressor(nn.Module):
    """Reaction property regression model.

    Inputs are Z + BO_R + BO_P for 2D, and additionally IRC R/P endpoint
    D_R/D_P for the 3D variant.
    """

    def __init__(
        self,
        pair_raw_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        input_schema: str,
        dynamic_pair_update: bool = False,
        dynamic_pair_update_scale: float = 1.0,
        dynamic_pair_update_dropout: float | None = None,
        attention_readout: bool = False,
        directional_3d_adapter: bool = False,
        encoder_type: str = "token_space",
        radar_attention_heads: int = 8,
        radar_center_router: bool = True,
        radar_delta_stream: bool = True,
        radar_pair_update_scale: float = 0.75,
        radar_reaction_update_scale: float = 1.0,
        radar_router_gate_init: float = 0.05,
        mrto_attention_heads: int = 8,
        mrto_event_slots: int = 4,
        mrto_use_event_slots: bool = True,
        mrto_use_odd_field: bool = True,
        mrto_pair_update_scale: float = 1.0,
        mrto_reaction_update_scale: float = 1.0,
        mrto_endpoint_layers: int = 2,
        mrto_triangle_layers: int = 2,
        mrto_triangle_dim: int = 16,
        mrto_triangle_scale: float = 0.5,
        mrto_event_topk: int = 0,
        mrto_event_feedback_scale: float = 0.5,
        mrto_geometry_rbf_bins: int = 16,
        mrto_equiformer_layers: int = 2,
        mrto_equiformer_channels: int = 32,
        mrto_equiformer_lmax: int = 2,
        mrto_equiformer_radius: float = 5.0,
        mrto_equiformer_max_neighbors: int = 64,
        joint_encoder_pass: bool = False,
    ):
        super().__init__()
        masked_schema, masked_pair_raw_dim = _masked_edit_schema_for_property(input_schema)
        self.encoder_type = encoder_type
        self.joint_encoder_pass = bool(joint_encoder_pass)
        if self.joint_encoder_pass and encoder_type not in {"mrto_v1", "mrto_full"}:
            raise ValueError("joint_encoder_pass requires encoder_type='mrto_v1' or 'mrto_full'")
        if encoder_type in {"mrto_v1", "mrto_full"}:
            if directional_3d_adapter:
                raise ValueError("directional_3d_adapter is only supported by encoder_type='token_space'")
            if not mrto_use_event_slots:
                raise ValueError(f"encoder_type={encoder_type!r} requires event slots")
            adapter_class = MRTOFullReactionInputAdapter if encoder_type == "mrto_full" else MRTOv1ReactionInputAdapter
            encoder_class = MRTOFullReactionEncoder if encoder_type == "mrto_full" else MRTOv1ReactionEncoder
            adapter_kwargs = {
                "hidden_dim": hidden_dim,
                "use_odd_field": mrto_use_odd_field,
                "endpoint_layers": mrto_endpoint_layers,
                "geometry_rbf_bins": mrto_geometry_rbf_bins,
                "dropout": dropout,
            }
            if encoder_type == "mrto_full":
                adapter_kwargs.update(
                    equiformer_layers=mrto_equiformer_layers,
                    equiformer_channels=mrto_equiformer_channels,
                    equiformer_lmax=mrto_equiformer_lmax,
                    equiformer_radius=mrto_equiformer_radius,
                    equiformer_max_neighbors=mrto_equiformer_max_neighbors,
                )
            self.adapter = adapter_class(
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="property",
                **adapter_kwargs,
            )
            masked_adapter_kwargs = dict(adapter_kwargs)
            if encoder_type == "mrto_full":
                masked_adapter_kwargs["equiformer_adapter"] = self.adapter.equivariant_geometry
            self.masked_adapter = adapter_class(
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
                task_name="masked_edit",
                **masked_adapter_kwargs,
            )
            self.encoder = encoder_class(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=mrto_attention_heads,
                event_slots=mrto_event_slots,
                triangle_layers=mrto_triangle_layers,
                triangle_dim=mrto_triangle_dim,
                triangle_scale=mrto_triangle_scale,
                pair_update_scale=mrto_pair_update_scale,
                event_topk=mrto_event_topk,
                event_feedback_scale=mrto_event_feedback_scale,
            )
        elif encoder_type == "mrto":
            if directional_3d_adapter:
                raise ValueError("directional_3d_adapter is only supported by encoder_type='token_space'")
            self.adapter = MRTOReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="property",
                use_odd_field=mrto_use_odd_field,
            )
            self.masked_adapter = MRTOReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
                task_name="masked_edit",
                use_odd_field=mrto_use_odd_field,
            )
            self.encoder = MRTOReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=mrto_attention_heads,
                event_slots=mrto_event_slots,
                use_event_slots=mrto_use_event_slots,
                pair_update_scale=mrto_pair_update_scale,
                reaction_update_scale=mrto_reaction_update_scale,
            )
        elif encoder_type == "radar":
            if directional_3d_adapter:
                raise ValueError("directional_3d_adapter is only supported by encoder_type='token_space'")
            self.adapter = RADARReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="property",
                enable_delta_stream=radar_delta_stream,
            )
            self.masked_adapter = RADARReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
                task_name="masked_edit",
                enable_delta_stream=radar_delta_stream,
            )
            self.encoder = RADARReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=radar_attention_heads,
                center_router=radar_center_router,
                pair_update_scale=radar_pair_update_scale,
                reaction_update_scale=radar_reaction_update_scale,
            )
        elif encoder_type == "token_space":
            self.adapter = RPairPropertyAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                directional_3d_adapter=directional_3d_adapter,
            )
            self.masked_adapter = MaskedEditAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
            )
            self.encoder = TokenSpaceReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                dynamic_pair_update=dynamic_pair_update,
                dynamic_pair_update_scale=dynamic_pair_update_scale,
                dynamic_pair_update_dropout=dynamic_pair_update_dropout,
            )
        else:
            raise ValueError(f"unsupported encoder_type={encoder_type!r}")
        self.masked_edit_heads = MaskedEditHeads(
            hidden_dim,
            router_residual=(encoder_type == "radar" and radar_center_router)
            or encoder_type in {"mrto", "mrto_v1", "mrto_full"},
            router_gate_init=radar_router_gate_init,
            reaction_context=encoder_type == "mrto_full",
        )
        self.atom_readout = AtomAttentionReadout(hidden_dim) if attention_readout else None
        self.property_output_is_raw = encoder_type == "mrto_full"
        self.reg_head = (
            MRTOParityEnergyHead(hidden_dim, dropout)
            if self.property_output_is_raw
            else nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(ENERGY_TARGETS)),
            )
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoder_input = self.adapter(batch)
        masked_encoded: dict[str, torch.Tensor] | None = None
        if self.joint_encoder_pass and "masked_pair_input" in batch:
            masked_input = self.masked_adapter(self._masked_batch(batch))
            joint_encoded = self.encoder(_concat_mrto_v1_inputs(encoder_input, masked_input))
            encoded, masked_encoded = _split_encoded_batch(joint_encoded, encoder_input.atom_tokens.shape[0])
        else:
            encoded = self.encoder(encoder_input)
        atom_h = encoded["atom_h"]
        atom_mask = batch["atom_mask"].bool()
        if self.atom_readout is None:
            masked_h = atom_h.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
            atom_pool = masked_h.sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
        else:
            atom_pool = self.atom_readout(atom_h, atom_mask, encoded["reaction_h"])
        y = (
            self.reg_head(encoded)
            if self.property_output_is_raw
            else self.reg_head(torch.cat([encoded["reaction_h"], atom_pool], dim=-1))
        )

        out = {
            "y": y,
            "reaction_h": encoded["reaction_h"],
        }
        if self.property_output_is_raw:
            out["y_is_raw"] = True
        for key in (
            "mrto_event_pair_weights",
            "mrto_event_plus",
            "mrto_event_minus",
            "mrto_reaction_plus",
            "mrto_reaction_minus",
            "mrto_event_presence_logits",
            "mrto_event_delta_bo",
        ):
            if key in encoded:
                out[key] = encoded[key]
        if "masked_pair_input" in batch:
            masked_out = (
                self._masked_outputs(masked_encoded)
                if masked_encoded is not None
                else self._masked_forward(batch)
            )
            out.update({f"masked_{key}": value for key, value in masked_out.items()})
        return out

    @staticmethod
    def _masked_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {
            "z": batch["z"],
            "atom_mask": batch["atom_mask"],
            "pair_input": batch["masked_pair_input"],
            "pair_valid": batch["masked_pair_valid"],
        }
        if "coordinates_R" in batch:
            out["coordinates_R"] = batch["coordinates_R"]
        return out

    def _masked_outputs(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = _masked_edit_outputs(self.masked_edit_heads, encoded)
        out["reaction_h"] = encoded["reaction_h"]
        for key in (
            "mrto_event_pair_weights",
            "mrto_event_plus",
            "mrto_event_minus",
            "mrto_reaction_plus",
            "mrto_reaction_minus",
            "mrto_event_presence_logits",
            "mrto_event_delta_bo",
        ):
            if key in encoded:
                out[key] = encoded[key]
        return out

    def _masked_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.masked_adapter(self._masked_batch(batch)))
        return self._masked_outputs(encoded)


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
        enable_suiren_input_gates: bool = False,
        dynamic_pair_update: bool = False,
        dynamic_pair_update_scale: float = 1.0,
        dynamic_pair_update_dropout: float | None = None,
        attention_readout: bool = False,
        directional_3d_adapter: bool = False,
        encoder_type: str = "token_space",
        radar_attention_heads: int = 8,
        radar_center_router: bool = True,
        radar_delta_stream: bool = True,
        radar_pair_update_scale: float = 0.75,
        radar_reaction_update_scale: float = 1.0,
        radar_router_gate_init: float = 0.05,
        mrto_attention_heads: int = 8,
        mrto_event_slots: int = 4,
        mrto_use_event_slots: bool = True,
        mrto_use_odd_field: bool = True,
        mrto_pair_update_scale: float = 1.0,
        mrto_reaction_update_scale: float = 1.0,
        mrto_endpoint_layers: int = 2,
        mrto_triangle_layers: int = 2,
        mrto_triangle_dim: int = 16,
        mrto_triangle_scale: float = 0.5,
        mrto_event_topk: int = 0,
        mrto_event_feedback_scale: float = 0.5,
        mrto_geometry_rbf_bins: int = 16,
        mrto_equiformer_layers: int = 2,
        mrto_equiformer_channels: int = 32,
        mrto_equiformer_lmax: int = 2,
        mrto_equiformer_radius: float = 5.0,
        mrto_equiformer_max_neighbors: int = 64,
        joint_encoder_pass: bool = False,
    ):
        super().__init__()
        masked_schema, masked_pair_raw_dim = _masked_edit_schema_for_property(input_schema)
        self.encoder_type = encoder_type
        self.joint_encoder_pass = bool(joint_encoder_pass)
        if self.joint_encoder_pass and encoder_type not in {"mrto_v1", "mrto_full"}:
            raise ValueError("joint_encoder_pass requires encoder_type='mrto_v1' or 'mrto_full'")
        if encoder_type in {"mrto_v1", "mrto_full"}:
            if directional_3d_adapter:
                raise ValueError("directional_3d_adapter is only supported by encoder_type='token_space'")
            if not mrto_use_event_slots:
                raise ValueError(f"encoder_type={encoder_type!r} requires event slots")
            adapter_class = MRTOFullReactionInputAdapter if encoder_type == "mrto_full" else MRTOv1ReactionInputAdapter
            encoder_class = MRTOFullReactionEncoder if encoder_type == "mrto_full" else MRTOv1ReactionEncoder
            adapter_kwargs = {
                "hidden_dim": hidden_dim,
                "use_odd_field": mrto_use_odd_field,
                "endpoint_layers": mrto_endpoint_layers,
                "geometry_rbf_bins": mrto_geometry_rbf_bins,
                "dropout": dropout,
            }
            if encoder_type == "mrto_full":
                adapter_kwargs.update(
                    equiformer_layers=mrto_equiformer_layers,
                    equiformer_channels=mrto_equiformer_channels,
                    equiformer_lmax=mrto_equiformer_lmax,
                    equiformer_radius=mrto_equiformer_radius,
                    equiformer_max_neighbors=mrto_equiformer_max_neighbors,
                )
            self.input_adapter = adapter_class(
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="suiren_fusion_property",
                suiren_atom_dim=suiren_atom_dim,
                suiren_graph_dim=suiren_graph_dim,
                suiren_atom_dims=suiren_atom_dims,
                suiren_graph_dims=suiren_graph_dims,
                enable_suiren_gates=enable_suiren_input_gates,
                **adapter_kwargs,
            )
            masked_adapter_kwargs = dict(adapter_kwargs)
            if encoder_type == "mrto_full":
                masked_adapter_kwargs["equiformer_adapter"] = self.input_adapter.equivariant_geometry
            self.masked_adapter = adapter_class(
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
                task_name="masked_edit",
                **masked_adapter_kwargs,
            )
            self.encoder = encoder_class(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=mrto_attention_heads,
                event_slots=mrto_event_slots,
                triangle_layers=mrto_triangle_layers,
                triangle_dim=mrto_triangle_dim,
                triangle_scale=mrto_triangle_scale,
                pair_update_scale=mrto_pair_update_scale,
                event_topk=mrto_event_topk,
                event_feedback_scale=mrto_event_feedback_scale,
            )
        elif encoder_type == "radar":
            if directional_3d_adapter:
                raise ValueError("directional_3d_adapter is only supported by encoder_type='token_space'")
            self.input_adapter = RADARReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                task_name="suiren_fusion_property",
                suiren_atom_dim=suiren_atom_dim,
                suiren_graph_dim=suiren_graph_dim,
                suiren_atom_dims=suiren_atom_dims,
                suiren_graph_dims=suiren_graph_dims,
                enable_delta_stream=radar_delta_stream,
                enable_suiren_gates=enable_suiren_input_gates,
            )
            self.masked_adapter = RADARReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
                task_name="masked_edit",
                enable_delta_stream=radar_delta_stream,
            )
            self.encoder = RADARReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                attention_heads=radar_attention_heads,
                center_router=radar_center_router,
                pair_update_scale=radar_pair_update_scale,
                reaction_update_scale=radar_reaction_update_scale,
            )
        elif encoder_type == "token_space":
            self.input_adapter = UnifiedReactionInputAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=pair_raw_dim,
                input_schema=input_schema,
                suiren_atom_dim=suiren_atom_dim,
                suiren_graph_dim=suiren_graph_dim,
                suiren_atom_dims=suiren_atom_dims,
                suiren_graph_dims=suiren_graph_dims,
                enable_suiren_gates=enable_suiren_input_gates,
                directional_3d_adapter=directional_3d_adapter,
            )
            self.masked_adapter = MaskedEditAdapter(
                hidden_dim=hidden_dim,
                pair_input_dim=masked_pair_raw_dim,
                input_schema=masked_schema,
            )
            self.encoder = TokenSpaceReactionEncoder(
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                dynamic_pair_update=dynamic_pair_update,
                dynamic_pair_update_scale=dynamic_pair_update_scale,
                dynamic_pair_update_dropout=dynamic_pair_update_dropout,
            )
        else:
            raise ValueError(f"unsupported encoder_type={encoder_type!r}")
        self.masked_edit_heads = MaskedEditHeads(
            hidden_dim,
            router_residual=(encoder_type == "radar" and radar_center_router)
            or encoder_type in {"mrto_v1", "mrto_full"},
            router_gate_init=radar_router_gate_init,
            reaction_context=encoder_type == "mrto_full",
        )
        self.atom_readout = AtomAttentionReadout(hidden_dim) if attention_readout else None
        self.property_output_is_raw = encoder_type == "mrto_full"
        self.reg_head = (
            MRTOParityEnergyHead(hidden_dim, dropout)
            if self.property_output_is_raw
            else nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(ENERGY_TARGETS)),
            )
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in batch.items():
            if key.startswith("suiren_") and key.endswith("_failed") and bool(value.bool().any().item()):
                raise ValueError(f"Suiren cache contains failed rows in {key}; rebuild the cache before training or evaluation")

        encoder_input = self.input_adapter(batch)
        masked_encoded: dict[str, torch.Tensor] | None = None
        if self.joint_encoder_pass and "masked_pair_input" in batch:
            masked_input = self.masked_adapter(self._masked_batch(batch))
            joint_encoded = self.encoder(_concat_mrto_v1_inputs(encoder_input, masked_input))
            encoded, masked_encoded = _split_encoded_batch(joint_encoded, encoder_input.atom_tokens.shape[0])
        else:
            encoded = self.encoder(encoder_input)
        atom_h = encoded["atom_h"]
        atom_mask = batch["atom_mask"].bool()
        if self.atom_readout is None:
            masked_h = atom_h.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
            atom_pool = masked_h.sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
        else:
            atom_pool = self.atom_readout(atom_h, atom_mask, encoded["reaction_h"])
        y = (
            self.reg_head(encoded)
            if self.property_output_is_raw
            else self.reg_head(torch.cat([encoded["reaction_h"], atom_pool], dim=-1))
        )

        out = {
            "y": y,
            "reaction_h": encoded["reaction_h"],
        }
        if self.property_output_is_raw:
            out["y_is_raw"] = True
        for key in (
            "mrto_event_pair_weights",
            "mrto_event_plus",
            "mrto_event_minus",
            "mrto_reaction_plus",
            "mrto_reaction_minus",
            "mrto_event_presence_logits",
            "mrto_event_delta_bo",
        ):
            if key in encoded:
                out[key] = encoded[key]
        if "masked_pair_input" in batch:
            masked_out = (
                self._masked_outputs(masked_encoded)
                if masked_encoded is not None
                else self._masked_forward(batch)
            )
            out.update({f"masked_{key}": value for key, value in masked_out.items()})
        return out

    @staticmethod
    def _masked_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {
            "z": batch["z"],
            "atom_mask": batch["atom_mask"],
            "pair_input": batch["masked_pair_input"],
            "pair_valid": batch["masked_pair_valid"],
        }
        if "coordinates_R" in batch:
            out["coordinates_R"] = batch["coordinates_R"]
        return out

    def _masked_outputs(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = _masked_edit_outputs(self.masked_edit_heads, encoded)
        out["reaction_h"] = encoded["reaction_h"]
        for key in (
            "mrto_event_pair_weights",
            "mrto_event_plus",
            "mrto_event_minus",
            "mrto_reaction_plus",
            "mrto_reaction_minus",
            "mrto_event_presence_logits",
            "mrto_event_delta_bo",
        ):
            if key in encoded:
                out[key] = encoded[key]
        return out

    def _masked_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.masked_adapter(self._masked_batch(batch)))
        return self._masked_outputs(encoded)


def _masked_edit_schema_for_property(input_schema: str) -> tuple[str, int]:
    if input_schema.startswith("property_irc_rp"):
        return "masked_edit_irc_rp", 4
    return "masked_edit_2d", 3


def load_pretrained_encoder(model: nn.Module, path: str, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["encoder"] if isinstance(payload, dict) and "encoder" in payload else payload
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("reaction_update", "reaction_norm")
    allowed_missing_fragments = (".pair_update.", ".pair_norm.")
    bad_missing = [
        key
        for key in missing
        if not (
            key.startswith(allowed_missing_prefixes)
            or (key.startswith("layers.") and any(fragment in key for fragment in allowed_missing_fragments))
        )
    ]
    if bad_missing or unexpected:
        raise RuntimeError(f"pretrained encoder load mismatch: missing={bad_missing[:10]} unexpected={unexpected[:10]}")


def load_stage_a_checkpoint(model: nn.Module, path: str, device: torch.device) -> Path:
    """Restore the complete Stage A state used by Stage B/C multi-task training."""

    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage A checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Stage A checkpoint must contain a full model state: {checkpoint}")
    state = payload["model"]
    source_encoder_type = payload.get("config", {}).get("encoder_type")
    target_encoder_type = getattr(model, "encoder_type", None)
    if source_encoder_type and target_encoder_type and source_encoder_type != target_encoder_type:
        raise ValueError(
            f"Stage A encoder_type mismatch: source={source_encoder_type} target={target_encoder_type}"
        )

    def component(prefix: str) -> dict[str, torch.Tensor]:
        start = f"{prefix}."
        values = {key[len(start) :]: value for key, value in state.items() if key.startswith(start)}
        if not values:
            raise ValueError(f"Stage A checkpoint is missing component {prefix!r}: {checkpoint}")
        return values

    if not hasattr(model, "masked_adapter") or not hasattr(model, "masked_edit_heads"):
        raise TypeError("target model must expose masked_adapter and masked_edit_heads")
    model.encoder.load_state_dict(component("encoder"), strict=True)
    model.masked_adapter.load_state_dict(component("adapter"), strict=True)
    model.masked_edit_heads.load_state_dict(component("masked_edit_heads"), strict=True)
    return checkpoint
