"""ReactionDeltaSample tensor construction for the current RFM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np

from rfm.features.token_space import (
    changed_pairs,
    edit_class_from_bo,
    generate_visibility_mask,
    pair_valid_matrix,
    pairwise_distance,
    reaction_core_from_delta,
)


EDIT_CLASSES = ("unchanged", "formed", "broken", "order_changed")
ENERGY_TARGETS = ("dE", "dE_dagger")
MAX_Z = 36


@dataclass(frozen=True)
class LoadedSplit:
    path: str
    split: str | None
    samples: Sequence[dict[str, Any]]


def load_split(path: str, limit: int = 0) -> LoadedSplit:
    from .hdf5 import load_reaction_delta_sample_store, split_path

    data_path, split = split_path(path)
    return LoadedSplit(data_path, split, load_reaction_delta_sample_store(data_path, split=split, limit=limit))


def bond_order_matrices(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = len(sample["atomic_numbers"])
    atom_index = {atom_map: idx for idx, atom_map in enumerate(sample["atom_map_order"])}
    bo_r = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    bo_p = np.zeros((n_atoms, n_atoms), dtype=np.float32)

    def fill(bonds: list[dict[str, Any]], matrix: np.ndarray) -> None:
        for bond in bonds:
            i = atom_index[bond["i"]]
            j = atom_index[bond["j"]]
            matrix[i, j] = matrix[j, i] = float(bond["bo"])

    fill(sample["bonds_R"], bo_r)
    fill(sample["bonds_P"], bo_p)
    return bo_r, bo_p


def coordinates(sample: dict[str, Any], side: str) -> np.ndarray:
    try:
        return np.asarray(sample["coordinates"][side], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"sample {sample.get('reaction_id', '<unknown>')} has no IRC R/P endpoint coordinates.{side}") from exc


def masked_edit_pair_input(sample: dict[str, Any], sample_index: int, args: Any) -> dict[str, np.ndarray]:
    bo_r, bo_p = bond_order_matrices(sample)
    delta_bo = (bo_p - bo_r).astype(np.float32)
    visibility = generate_visibility_mask(
        delta_bo,
        strategy=args.mask_strategy,
        pair_mask_ratio=args.pair_mask_ratio,
        changed_pair_mask_ratio=args.changed_pair_mask_ratio,
        seed=args.seed + sample_index * 1009,
    )
    channels = [bo_r, delta_bo * visibility, visibility]
    if args.geometry_mode == "irc_rp":
        d_r = pairwise_distance(coordinates(sample, "R"), clip=args.distance_clip, normalize=True)
        channels.insert(1, d_r)
    pair_input = np.stack(channels, axis=-1).astype(np.float32)
    return {
        "pair_input": pair_input,
        "pair_valid": pair_valid_matrix(delta_bo.shape[0]),
        "visibility": visibility,
        "loss_mask": (1.0 - visibility).astype(np.float32),
        "delta_bo": delta_bo,
        "changed": changed_pairs(delta_bo),
        "edit_class": edit_class_from_bo(bo_r, bo_p),
        "core_atom": reaction_core_from_delta(delta_bo),
    }


def reaction_property_pair_input(sample: dict[str, Any], args: Any) -> dict[str, np.ndarray]:
    bo_r, bo_p = bond_order_matrices(sample)
    delta_bo = (bo_p - bo_r).astype(np.float32)
    channels = [bo_r, bo_p]
    if args.geometry_mode == "irc_rp":
        channels.extend(
            [
                pairwise_distance(coordinates(sample, "R"), clip=None, normalize=False),
                pairwise_distance(coordinates(sample, "P"), clip=None, normalize=False),
            ]
        )
    pair_input = np.stack(channels, axis=-1).astype(np.float32)
    return {
        "pair_input": pair_input,
        "pair_valid": pair_valid_matrix(delta_bo.shape[0]),
        "changed": changed_pairs(delta_bo),
        "core_atom": reaction_core_from_delta(delta_bo),
    }


def target_vector(sample: dict[str, Any], targets: tuple[str, ...] = ENERGY_TARGETS) -> np.ndarray:
    return np.asarray([float(sample["targets"][name]) for name in targets], dtype=np.float32)
