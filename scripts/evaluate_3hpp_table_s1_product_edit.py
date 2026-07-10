#!/usr/bin/env python
"""Evaluate R-only WLDN product-edit models on 3HPP Table S1 structures.

The literature data contains one reactant, multiple products, and TS xyz files
for most reactions. Product xyz atom order is not guaranteed to match the
reactant. This script aligns every R/P/TS structure into one common reactant
atom order before building RFM ReactionDeltaSample rows.

TS coordinates are used only to establish ground-truth atom correspondence;
they are not written to the processed samples or passed to the R-only model.
Formal results must review the generated skipped-row and alignment-RMSD audit.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from rfm.data.hdf5 import write_processed_hdf5
from rfm.data.reaction_samples import bond_order_matrices
from rfm.models import WLDNProductEditPredictor
from rfm.tasks import product_edit, product_edit_wldn
from rfm.utils.runtime import move_to_device


BO_BY_TYPE = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}


@dataclass(frozen=True)
class XYZ:
    symbols: list[str]
    coords: np.ndarray
    comment: str
    text: str


@dataclass(frozen=True)
class ParsedMol:
    mol: Chem.Mol
    charge: int
    bonds: list[tuple[int, int, float]]
    smiles: str


@dataclass(frozen=True)
class MappingResult:
    coords_in_ref_order: np.ndarray
    target_index_for_ref_index: list[int]
    rmsd: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--suiren-3d-atom-cache", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-pool-size", type=int, default=32)
    parser.add_argument("--candidate-beam-size", type=int, default=64)
    parser.add_argument("--center-top-m", type=int, default=6)
    parser.add_argument("--edit-class-top-k", type=int, default=2)
    parser.add_argument("--top-k-values", default="1,3,5,10,32")
    parser.add_argument("--max-group-products", type=int, default=128)
    parser.add_argument("--delta-vocab-decimals", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=-1.0)
    parser.add_argument("--dynamic-pair-update-scale", type=float, default=math.nan)
    parser.add_argument("--distance-clip", type=float, default=10.0)
    return parser.parse_args()


def read_xyz(path: Path) -> XYZ:
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        raise ValueError(f"empty xyz: {path}")
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip() if len(lines) > 1 else ""
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad xyz atom line in {path}: {line!r}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) != n_atoms:
        raise ValueError(f"xyz atom count mismatch in {path}: declared={n_atoms} parsed={len(symbols)}")
    return XYZ(symbols, np.asarray(coords, dtype=np.float64), comment, text)


def xyz_charge_hint(xyz: XYZ) -> int | None:
    stripped = xyz.comment.strip()
    if not stripped:
        return None
    try:
        return int(stripped.split()[0])
    except Exception:
        return None


def parse_mol_from_xyz(xyz: XYZ, charge_candidates: list[int] | None = None) -> ParsedMol:
    hint = xyz_charge_hint(xyz)
    candidates: list[int] = []
    if hint is not None:
        candidates.append(hint)
    candidates.extend(charge_candidates or [0, -1, 1, -2, 2, -3, 3])
    seen: set[int] = set()
    errors: list[str] = []
    for charge in candidates:
        if charge in seen:
            continue
        seen.add(charge)
        mol = Chem.MolFromXYZBlock(xyz.text)
        if mol is None:
            raise ValueError("RDKit MolFromXYZBlock returned None")
        try:
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=int(charge),
                allowChargedFragments=True,
                embedChiral=False,
            )
            Chem.SanitizeMol(mol, catchErrors=True)
            bonds: list[tuple[int, int, float]] = []
            for bond in mol.GetBonds():
                bo = BO_BY_TYPE.get(bond.GetBondType(), float(bond.GetBondTypeAsDouble()))
                bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), float(bo)))
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return ParsedMol(mol, int(charge), bonds, smiles)
        except Exception as exc:  # noqa: BLE001 - record fallback failures.
            errors.append(f"charge={charge}: {type(exc).__name__}: {exc}")
    raise ValueError("; ".join(errors[-5:]))


def kabsch_rmsd(ref: np.ndarray, target_in_ref_order: np.ndarray) -> float:
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    target_centered = target_in_ref_order - target_in_ref_order.mean(axis=0, keepdims=True)
    cov = target_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(cov)
    det = np.linalg.det(u @ vt)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if det >= 0 else -1.0
    rot = u @ correction @ vt
    aligned = target_centered @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - ref_centered) ** 2, axis=1))))


def element_groups(symbols: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, symbol in enumerate(symbols):
        groups[symbol].append(idx)
    return dict(groups)


def best_mapping_to_ref(ref_symbols: list[str], ref_coords: np.ndarray, target: XYZ) -> MappingResult:
    target_groups = element_groups(target.symbols)
    ref_groups = element_groups(ref_symbols)
    if set(ref_groups) != set(target_groups):
        raise ValueError(f"element sets differ: ref={ref_groups} target={target_groups}")
    for symbol in ref_groups:
        if len(ref_groups[symbol]) != len(target_groups[symbol]):
            raise ValueError(f"element count differs for {symbol}: ref={len(ref_groups[symbol])} target={len(target_groups[symbol])}")

    symbols = sorted(ref_groups)
    n_atoms = len(ref_symbols)

    def pairwise(coords: np.ndarray) -> np.ndarray:
        diff = coords[:, None, :] - coords[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    ref_dist = pairwise(ref_coords)
    target_dist = pairwise(target.coords)

    def fingerprint(dist: np.ndarray, symbols_for_dist: list[str], atom_idx: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        for symbol in symbols:
            indices = [idx for idx, item in enumerate(symbols_for_dist) if item == symbol]
            parts.append(np.sort(dist[atom_idx, indices]))
        return np.concatenate(parts)

    ref_fp = [fingerprint(ref_dist, ref_symbols, idx) for idx in range(n_atoms)]
    target_fp = [fingerprint(target_dist, target.symbols, idx) for idx in range(n_atoms)]
    target_for_ref = [-1] * n_atoms
    for symbol in symbols:
        ref_indices = ref_groups[symbol]
        target_indices = target_groups[symbol]
        cost = np.zeros((len(ref_indices), len(target_indices)), dtype=np.float64)
        for row, ref_idx in enumerate(ref_indices):
            for col, target_idx in enumerate(target_indices):
                cost[row, col] = float(np.linalg.norm(ref_fp[ref_idx] - target_fp[target_idx]))
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols, strict=True):
            target_for_ref[ref_indices[int(row)]] = int(target_indices[int(col)])

    if any(idx < 0 for idx in target_for_ref):
        raise ValueError("incomplete Hungarian atom mapping")

    def current_rmsd(mapping: list[int]) -> float:
        return kabsch_rmsd(ref_coords, target.coords[np.asarray(mapping, dtype=np.int64)])

    best_rmsd = current_rmsd(target_for_ref)
    improved = True
    passes = 0
    while improved and passes < 4:
        improved = False
        passes += 1
        for symbol in symbols:
            ref_indices = ref_groups[symbol]
            current_targets = [target_for_ref[idx] for idx in ref_indices]
            best_local = list(current_targets)
            for perm in itertools.permutations(current_targets):
                trial = list(target_for_ref)
                for ref_idx, target_idx in zip(ref_indices, perm, strict=True):
                    trial[ref_idx] = int(target_idx)
                rmsd = current_rmsd(trial)
                if rmsd + 1.0e-9 < best_rmsd:
                    best_rmsd = rmsd
                    best_local = list(perm)
                    improved = True
            for ref_idx, target_idx in zip(ref_indices, best_local, strict=True):
                target_for_ref[ref_idx] = int(target_idx)

    best_target_for_ref = target_for_ref
    return MappingResult(
        coords_in_ref_order=target.coords[np.asarray(best_target_for_ref, dtype=np.int64)].astype(np.float32),
        target_index_for_ref_index=best_target_for_ref,
        rmsd=best_rmsd,
    )


def remap_bonds_to_ref(bonds: list[tuple[int, int, float]], target_for_ref: list[int]) -> list[dict[str, Any]]:
    ref_for_target = {target_idx: ref_idx for ref_idx, target_idx in enumerate(target_for_ref)}
    out: list[dict[str, Any]] = []
    for i, j, bo in bonds:
        if i not in ref_for_target or j not in ref_for_target:
            raise ValueError(f"bond atom index not covered by mapping: {(i, j)}")
        ref_i = ref_for_target[i] + 1
        ref_j = ref_for_target[j] + 1
        if ref_i == ref_j:
            continue
        if ref_i > ref_j:
            ref_i, ref_j = ref_j, ref_i
        out.append({"i": int(ref_i), "j": int(ref_j), "bo": float(bo)})
    return sorted(out, key=lambda row: (row["i"], row["j"], row["bo"]))


def reaction_core_mask_from_bonds(sample: dict[str, Any]) -> list[int]:
    bo_r, bo_p = bond_order_matrices(sample)
    delta = bo_p - bo_r
    return [int(v) for v in (np.abs(delta).sum(axis=1) > 1.0e-6)]


def build_3hpp_samples(xyz_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_reactant = read_xyz(xyz_dir / "r1_reactant.xyz")
    ref_symbols = first_reactant.symbols
    ref_coords = first_reactant.coords
    periodic = Chem.GetPeriodicTable()

    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for reaction_no in range(1, 76):
        rid = f"3hpp_r{reaction_no}"
        r_path = xyz_dir / f"r{reaction_no}_reactant.xyz"
        p_path = xyz_dir / f"r{reaction_no}_product.xyz"
        ts_path = xyz_dir / f"r{reaction_no}_ts.xyz"
        missing = [str(path.name) for path in (r_path, p_path, ts_path) if not path.exists()]
        if missing:
            skipped.append({"reaction_id": rid, "reason": "missing_xyz", "missing": missing})
            continue
        try:
            r_xyz = read_xyz(r_path)
            p_xyz = read_xyz(p_path)
            ts_xyz = read_xyz(ts_path)
            r_mol = parse_mol_from_xyz(r_xyz)
            p_mol = parse_mol_from_xyz(p_xyz)
            if r_xyz.symbols != ref_symbols:
                raise ValueError("reactant atom symbol order differs from r1_reactant")
            if ts_xyz.symbols != ref_symbols:
                raise ValueError("TS atom symbol order differs from r1_reactant")
            # Table S1 keeps reactant and TS xyz files in one common atom order.
            # Only product files need remapping; using raw R/TS order avoids
            # symmetry-driven H/O swaps from splitting the one-reactant group.
            identity_map = list(range(len(ref_symbols)))
            r_coords = r_xyz.coords.astype(np.float32)
            ts_coords = ts_xyz.coords.astype(np.float32)
            p_map = best_mapping_to_ref(ref_symbols, ts_xyz.coords, p_xyz)
            sample: dict[str, Any] = {
                "reaction_id": rid,
                "source_reaction_no": reaction_no,
                "source": "3HPP Table S1 dft_structures",
                "atomic_numbers": [int(periodic.GetAtomicNumber(symbol)) for symbol in ref_symbols],
                "atom_map_order": list(range(1, len(ref_symbols) + 1)),
                "bonds_R": remap_bonds_to_ref(r_mol.bonds, identity_map),
                "bonds_P": remap_bonds_to_ref(p_mol.bonds, p_map.target_index_for_ref_index),
                "coordinates": {
                    "R": r_coords.tolist(),
                    "P": p_map.coords_in_ref_order.tolist(),
                },
                "targets": {
                    "dE": float("nan"),
                    "dE_dagger": float("nan"),
                    "dH": float("nan"),
                    "dH_dagger": float("nan"),
                    "dG": float("nan"),
                    "dG_dagger": float("nan"),
                },
                "n_reactants": 1,
                "n_products": 1,
                "coordinate_metadata": {
                    "coordinate_source": "TableS1 xyz",
                    "common_atom_order": "r1_reactant geometry/order",
                    "product_alignment_reference": "per-reaction TS aligned to common atom order",
                    "r_alignment_rmsd": 0.0,
                    "p_to_ts_alignment_rmsd": p_map.rmsd,
                    "ts_alignment_rmsd": 0.0,
                    "rdkit_charge_R": r_mol.charge,
                    "rdkit_charge_P": p_mol.charge,
                    "smiles_R": r_mol.smiles,
                    "smiles_P": p_mol.smiles,
                },
            }
            sample["reaction_core_mask"] = reaction_core_mask_from_bonds(sample)
            bo_r, bo_p = bond_order_matrices(sample)
            delta_values = sorted(float(v) for v in np.unique(np.round(bo_p - bo_r, 6)) if abs(float(v)) > 1.0e-8)
            records.append(
                {
                    "reaction_id": rid,
                    "status": "ok",
                    "charge_R": r_mol.charge,
                    "charge_P": p_mol.charge,
                    "r_alignment_rmsd": 0.0,
                    "p_to_ts_alignment_rmsd": p_map.rmsd,
                    "ts_alignment_rmsd": 0.0,
                    "n_changed_pairs": int(np.count_nonzero(np.triu(np.abs(bo_p - bo_r) > 1.0e-6, k=1))),
                    "delta_values": delta_values,
                    "smiles_R": r_mol.smiles,
                    "smiles_P": p_mol.smiles,
                }
            )
            samples.append(sample)
        except Exception as exc:  # noqa: BLE001 - external data audit should continue.
            skipped.append({"reaction_id": rid, "reason": type(exc).__name__, "error": str(exc)})

    metadata = {
        "dataset": "3HPP Table S1",
        "source": str(xyz_dir),
        "n_samples": len(samples),
        "skipped": skipped,
        "records": records,
        "alignment_policy": {
            "global_reference": "raw Table S1 reactant/TS atom order, verified identical across reactions",
            "reactant_mapping": "identity mapping; reactant xyz atom order is shared",
            "ts_mapping": "identity mapping; TS xyz atom order is shared",
            "product_mapping": "element-constrained distance-fingerprint Hungarian initialization plus local Kabsch refinement to per-reaction TS",
        },
    }
    return samples, metadata


def make_args_for_dataset(base: argparse.Namespace, checkpoint_config: dict[str, Any]) -> argparse.Namespace:
    hidden_dim = int(base.hidden_dim or checkpoint_config.get("hidden_dim", 256))
    layers = int(base.layers or checkpoint_config.get("layers", 8))
    dropout = float(base.dropout if base.dropout >= 0 else checkpoint_config.get("dropout", 0.1))
    dynamic_scale = float(
        base.dynamic_pair_update_scale
        if np.isfinite(base.dynamic_pair_update_scale)
        else checkpoint_config.get("dynamic_pair_update_scale", 0.75)
    )
    return argparse.Namespace(
        geometry_mode="irc_r",
        distance_clip=base.distance_clip,
        max_group_products=base.max_group_products,
        delta_vocab_decimals=base.delta_vocab_decimals,
        trust_suiren_cache=False,
        preload_suiren_graph_cache=False,
        preload_suiren_atom_cache=True,
        suiren_cache_layout=checkpoint_config.get("suiren_cache_layout", "rp_delta_abs"),
        suiren_2d_graph_cache="",
        suiren_3d_graph_cache="",
        suiren_2d_atom_cache="",
        suiren_3d_atom_cache=base.suiren_3d_atom_cache,
        center_top_m=base.center_top_m,
        edit_class_top_k=base.edit_class_top_k,
        candidate_beam_size=base.candidate_beam_size,
        candidate_pool_size=base.candidate_pool_size,
        top_k=max(parse_topks(base.top_k_values)),
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        dynamic_pair_update=bool(checkpoint_config.get("dynamic_pair_update", True)),
        dynamic_pair_update_scale=dynamic_scale,
        dynamic_pair_update_dropout=checkpoint_config.get("dynamic_pair_update_dropout", None),
    )


def parse_topks(value: str) -> tuple[int, ...]:
    out = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not out:
        raise ValueError("--top-k-values is empty")
    return out


def load_model(args: argparse.Namespace, dataset: product_edit_wldn.ProductEditDataset, device: torch.device) -> WLDNProductEditPredictor:
    input_schema = "product_edit_r3d_bo"
    model = WLDNProductEditPredictor(
        pair_raw_dim=2,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        input_schema=input_schema,
        n_delta_classes=len(dataset.delta_vocab),
        suiren_atom_dims=dataset.suiren_atom_dims,
        suiren_graph_dims=dataset.suiren_graph_dims,
        dynamic_pair_update=args.dynamic_pair_update,
        dynamic_pair_update_scale=args.dynamic_pair_update_scale,
        dynamic_pair_update_dropout=args.dynamic_pair_update_dropout,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"model state mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
    model.eval()
    return model


def candidate_exact(candidates: np.ndarray, targets: np.ndarray, pair_mask: np.ndarray, top_k: int) -> bool:
    return product_edit._candidate_exact(candidates, targets, pair_mask, top_k)


def evaluate_external(
    model: WLDNProductEditPredictor,
    loader: DataLoader,
    device: torch.device,
    delta_vocab: np.ndarray,
    args: argparse.Namespace,
    topks: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    no_change = product_edit.no_change_index(delta_vocab)
    exact_sample = {k: 0 for k in topks}
    exact_group = {k: 0 for k in topks}
    candidate_pool_sizes: list[int] = []
    center_coverage = 0
    n_samples = 0
    valid_top1 = 0
    valid_top5 = 0
    valid_top5_total = 0
    rows: list[dict[str, Any]] = []

    changed_true_top1: list[np.ndarray] = []
    changed_pred_top1: list[np.ndarray] = []
    changed_true_best: list[np.ndarray] = []
    changed_pred_best: list[np.ndarray] = []
    delta_mae_best: list[float] = []
    delta_true_top1: list[np.ndarray] = []
    delta_pred_top1: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            out = model(batch)
            candidate_class, candidate_mask = product_edit_wldn.generate_candidate_classes(out, batch, delta_vocab, args, include_target=False)
            rank_order = product_edit_wldn._rank_candidates(model, out, batch, candidate_class, candidate_mask, delta_vocab)
            ranked = torch.gather(candidate_class, 1, rank_order[:, :, None, None].expand_as(candidate_class)).detach().cpu().numpy()
            candidate_mask_np = torch.gather(candidate_mask, 1, rank_order).detach().cpu().numpy().astype(bool)
            change_np = out["change_logits"].detach().cpu().numpy()
            bo_r = batch["bo_r"].detach().cpu().numpy()
            target_class = batch["target_class"].detach().cpu().numpy()
            target_delta = batch["target_delta_bo"].detach().cpu().numpy()
            group_target = batch["group_target_class"].detach().cpu().numpy()
            group_mask = batch["group_target_mask"].detach().cpu().numpy().astype(bool)
            pair_valid = batch["pair_valid"].detach().cpu().numpy().astype(bool)

            for row_idx, reaction_id in enumerate(batch["reaction_id"]):
                n_samples += 1
                n_atoms = int(batch["n_atoms"][row_idx].detach().cpu())
                pair_mask = np.triu(pair_valid[row_idx, :n_atoms, :n_atoms], k=1)
                active_count = int(candidate_mask_np[row_idx].sum())
                candidate_pool_sizes.append(active_count)
                candidates = ranked[row_idx, :active_count, :n_atoms, :n_atoms].copy()
                if candidates.size == 0:
                    candidates = np.full((1, n_atoms, n_atoms), no_change, dtype=np.int64)
                for atom_idx in range(n_atoms):
                    candidates[:, atom_idx, atom_idx] = no_change
                row_target = target_class[row_idx, :n_atoms, :n_atoms]
                row_delta = target_delta[row_idx, :n_atoms, :n_atoms]
                active_group_targets = group_target[row_idx, group_mask[row_idx], :n_atoms, :n_atoms]
                if active_group_targets.size == 0:
                    active_group_targets = row_target[None, :, :]

                center_coverage += int(
                    product_edit_wldn._center_coverage(
                        change_np[row_idx, :n_atoms, :n_atoms],
                        row_target,
                        pair_mask,
                        no_change,
                        int(args.center_top_m),
                    )
                )
                for k in topks:
                    exact_sample[k] += int(candidate_exact(candidates, row_target[None, :, :], pair_mask, k))
                    exact_group[k] += int(candidate_exact(candidates, active_group_targets, pair_mask, k))

                pred_delta_all = delta_vocab[candidates]
                valid_top1 += int(product_edit._valid_product(bo_r[row_idx, :n_atoms, :n_atoms], pred_delta_all[0], pair_mask))
                for cand_idx in range(min(5, pred_delta_all.shape[0])):
                    valid_top5 += int(product_edit._valid_product(bo_r[row_idx, :n_atoms, :n_atoms], pred_delta_all[cand_idx], pair_mask))
                    valid_top5_total += 1

                pair_target = row_target[pair_mask]
                pair_errors = [(candidates[cand_idx][pair_mask] != pair_target).mean() for cand_idx in range(candidates.shape[0])]
                best_idx = int(np.argmin(pair_errors))
                top1_delta = pred_delta_all[0]
                best_delta = pred_delta_all[best_idx]
                changed_true_top1.append((pair_target != no_change).astype(np.int64))
                changed_pred_top1.append((candidates[0][pair_mask] != no_change).astype(np.int64))
                changed_true_best.append((pair_target != no_change).astype(np.int64))
                changed_pred_best.append((candidates[best_idx][pair_mask] != no_change).astype(np.int64))
                delta_true_top1.append(row_delta[pair_mask])
                delta_pred_top1.append(top1_delta[pair_mask])
                delta_mae_best.append(float(np.mean(np.abs(row_delta[pair_mask] - best_delta[pair_mask]))))

                max_rows = max(topks)
                for cand_idx in range(min(max_rows, candidates.shape[0])):
                    cand_delta = pred_delta_all[cand_idx]
                    rows.append(
                        {
                            "reaction_id": str(reaction_id),
                            "candidate_rank": int(cand_idx + 1),
                            "exact_sample": int(candidate_exact(candidates[cand_idx : cand_idx + 1], row_target[None, :, :], pair_mask, 1)),
                            "exact_group": int(candidate_exact(candidates[cand_idx : cand_idx + 1], active_group_targets, pair_mask, 1)),
                            "valid_product": int(product_edit._valid_product(bo_r[row_idx, :n_atoms, :n_atoms], cand_delta, pair_mask)),
                            "n_predicted_edits": int((np.abs(cand_delta[pair_mask]) > 1.0e-6).sum()),
                            "edits": product_edit._summarize_edits(cand_delta, pair_mask, limit=32),
                        }
                    )

    denom = max(n_samples, 1)
    changed_y_top1 = np.concatenate(changed_true_top1) if changed_true_top1 else np.asarray([], dtype=np.int64)
    changed_p_top1 = np.concatenate(changed_pred_top1) if changed_pred_top1 else np.asarray([], dtype=np.int64)
    changed_y_best = np.concatenate(changed_true_best) if changed_true_best else np.asarray([], dtype=np.int64)
    changed_p_best = np.concatenate(changed_pred_best) if changed_pred_best else np.asarray([], dtype=np.int64)
    delta_y = np.concatenate(delta_true_top1) if delta_true_top1 else np.asarray([], dtype=np.float32)
    delta_p = np.concatenate(delta_pred_top1) if delta_pred_top1 else np.asarray([], dtype=np.float32)
    metrics: dict[str, Any] = {
        "n_samples": int(n_samples),
        "center_coverage_topM": float(center_coverage / denom),
        "candidate_pool_size_mean": float(np.mean(candidate_pool_sizes)) if candidate_pool_sizes else 0.0,
        "valid_product_rate_top1": float(valid_top1 / denom),
        "valid_product_rate_top5": float(valid_top5 / max(valid_top5_total, 1)),
        "edit_pair_F1_top1": product_edit_wldn.binary_f1(changed_y_top1, changed_p_top1),
        "edit_pair_F1_best_of_K": product_edit_wldn.binary_f1(changed_y_best, changed_p_best),
        "Delta_BO_MAE_top1": product_edit_wldn.mean_absolute_error(delta_y, delta_p) if delta_y.size else 0.0,
        "Delta_BO_MAE_best_of_K": float(np.mean(delta_mae_best)) if delta_mae_best else 0.0,
    }
    for k in topks:
        metrics[f"exact_product_match_top{k}_sample"] = float(exact_sample[k] / denom)
        metrics[f"exact_product_match_top{k}_group"] = float(exact_group[k] / denom)
    return metrics, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, metadata = build_3hpp_samples(Path(args.xyz_dir))
    if not samples:
        raise RuntimeError("no evaluable 3HPP samples were built")
    metadata["note"] = "TS coordinates are used only for product atom-order alignment during construction and are not written into sample coordinates; product-edit model input uses R only."
    write_processed_hdf5(args.output_h5, {"test": samples}, metadata)
    (output_dir / "build_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "sample_audit.csv", metadata["records"])
    write_csv(output_dir / "skipped.csv", metadata["skipped"])

    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = raw_checkpoint.get("config", {}) if isinstance(raw_checkpoint, dict) else {}
    ckpt_delta_vocab = np.asarray(raw_checkpoint.get("delta_vocab"), dtype=np.float32)
    if ckpt_delta_vocab.size == 0:
        raise RuntimeError("checkpoint missing delta_vocab")
    dataset_args = make_args_for_dataset(args, checkpoint_config)
    dataset_args.checkpoint = args.checkpoint
    dataset = product_edit_wldn.ProductEditDataset(f"{args.output_h5}::test", limit=0, args=dataset_args, delta_vocab=ckpt_delta_vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=product_edit_wldn.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(dataset_args, dataset, device)
    topks = parse_topks(args.top_k_values)
    metrics, rows = evaluate_external(model, loader, device, dataset.delta_vocab, dataset_args, topks)
    result = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "suiren_3d_atom_cache": args.suiren_3d_atom_cache,
        "output_h5": args.output_h5,
        "settings": {
            "candidate_pool_size": args.candidate_pool_size,
            "candidate_beam_size": args.candidate_beam_size,
            "center_top_m": args.center_top_m,
            "edit_class_top_k": args.edit_class_top_k,
            "top_k_values": list(topks),
            "hidden_dim": dataset_args.hidden_dim,
            "layers": dataset_args.layers,
            "dynamic_pair_update_scale": dataset_args.dynamic_pair_update_scale,
            "suiren_atom_dims": dataset.suiren_atom_dims,
        },
        "dataset": {
            "n_samples": len(samples),
            "n_skipped": len(metadata["skipped"]),
            "skipped": metadata["skipped"],
            "group_stats": dataset.group_stats,
        },
        "metrics": metrics,
    }
    (output_dir / f"{args.run_name}_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / f"{args.run_name}_predictions.csv", rows)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
