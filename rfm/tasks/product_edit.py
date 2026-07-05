"""R-only top-K product graph-edit prediction components."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rfm.data.reaction_samples import EDIT_CLASSES, bond_order_matrices, load_split, product_edit_pair_input
from rfm.tasks.metrics import binary_f1, macro_f1, mean_absolute_error
from rfm.tasks.suiren_fusion_property import AtomFeatureStore, GraphFeatureStore, SUIREN_STREAMS
from rfm.utils.runtime import move_to_device


def canonical_reactant_key(sample: dict[str, Any], decimals: int = 6) -> str:
    """Hash atom identity and BO_R in atom-map order."""
    bo_r, _ = bond_order_matrices(sample)
    payload = {
        "z": [int(v) for v in sample["atomic_numbers"]],
        "bo_r": np.round(bo_r, decimals=decimals).tolist(),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def delta_vocab_from_samples(samples: Any, decimals: int = 6) -> np.ndarray:
    values: set[float] = set()
    for idx in range(len(samples)):
        bo_r, bo_p = bond_order_matrices(samples[idx])
        delta = np.round(bo_p - bo_r, decimals=decimals)
        values.update(float(v) for v in np.unique(delta))
    if 0.0 not in values:
        values.add(0.0)
    return np.asarray(sorted(values), dtype=np.float32)


def delta_class_matrix(delta_bo: np.ndarray, delta_vocab: np.ndarray, decimals: int = 6, tol: float = 1e-5) -> np.ndarray:
    rounded = np.round(delta_bo.astype(np.float32), decimals=decimals)
    out = np.full(rounded.shape, -1, dtype=np.int64)
    for class_idx, value in enumerate(delta_vocab):
        out[np.isclose(rounded, float(value), atol=tol, rtol=0.0)] = class_idx
    if np.any(out < 0):
        unknown = sorted(float(v) for v in np.unique(rounded[out < 0]))
        raise ValueError(f"Delta_BO values are not covered by delta_vocab: {unknown[:8]}")
    return out


def no_change_index(delta_vocab: np.ndarray) -> int:
    matches = np.where(np.isclose(delta_vocab, 0.0, atol=1e-8, rtol=0.0))[0]
    if len(matches) != 1:
        raise ValueError("delta_vocab must contain exactly one zero/no-change class")
    return int(matches[0])


def _r_only_suiren_dim(dim: int, layout: str) -> int:
    if dim <= 0:
        return 0
    if layout == "r_only":
        return int(dim)
    n_blocks = {"rp_delta_abs": 4, "rp_delta": 3}[layout]
    if dim % n_blocks != 0:
        raise ValueError(f"Suiren feature dim {dim} is incompatible with layout={layout}")
    return int(dim // n_blocks)


def _r_only_suiren_feature(feature: np.ndarray, layout: str) -> np.ndarray:
    if layout == "r_only":
        return np.asarray(feature, dtype=np.float32)
    dim = int(feature.shape[-1])
    r_dim = _r_only_suiren_dim(dim, layout)
    return np.asarray(feature[..., :r_dim], dtype=np.float32)


class ProductEditDataset(Dataset):
    """Build strict R-only product edit tensors from packed HDF5 rows."""

    def __init__(self, path: str, limit: int, args: Any, delta_vocab: np.ndarray | None = None):
        self.loaded = load_split(path, limit)
        self.samples = self.loaded.samples
        self.args = args
        self.decimals = int(getattr(args, "delta_vocab_decimals", 6))
        self.delta_vocab = np.asarray(delta_vocab if delta_vocab is not None else delta_vocab_from_samples(self.samples, self.decimals), dtype=np.float32)
        self.no_change_idx = no_change_index(self.delta_vocab)
        self.max_group_products = int(getattr(args, "max_group_products", 16))
        self.reactant_keys: list[str] = []
        self.group_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self.samples)):
            key = canonical_reactant_key(self.samples[idx], decimals=self.decimals)
            self.reactant_keys.append(key)
            self.group_to_indices[key].append(idx)

        trust_cache = bool(getattr(args, "trust_suiren_cache", False))
        preload_graph = bool(getattr(args, "preload_suiren_graph_cache", False))
        preload_atom = bool(getattr(args, "preload_suiren_atom_cache", False))
        self.suiren_cache_layout = getattr(args, "suiren_cache_layout", "rp_delta_abs")
        self.graph_stores = {
            stream: GraphFeatureStore(
                getattr(args, f"suiren_{stream}_graph_cache", ""),
                trust_cache=trust_cache,
                preload=preload_graph,
            )
            for stream in SUIREN_STREAMS
        }
        self.atom_stores = {
            stream: AtomFeatureStore(
                getattr(args, f"suiren_{stream}_atom_cache", ""),
                trust_cache=trust_cache,
                preload=preload_atom,
            )
            for stream in SUIREN_STREAMS
        }
        n = len(self.samples)
        for name, stores in (("graph", self.graph_stores), ("atom", self.atom_stores)):
            for stream, store in stores.items():
                if store.path is not None and len(store) < n:
                    raise ValueError(f"{stream} {name} cache shorter than samples for {path}: cache={len(store)} samples={n}")

    @property
    def group_stats(self) -> dict[str, Any]:
        sizes = np.asarray([len(indices) for indices in self.group_to_indices.values()], dtype=np.int64)
        if sizes.size == 0:
            return {"reactant_groups": 0, "multi_product_groups": 0, "max_products_per_group": 0, "multi_product_rows": 0}
        return {
            "reactant_groups": int(sizes.size),
            "multi_product_groups": int((sizes > 1).sum()),
            "max_products_per_group": int(sizes.max()),
            "multi_product_rows": int(sizes[sizes > 1].sum()),
        }

    @property
    def suiren_graph_dims(self) -> dict[str, int]:
        return {stream: _r_only_suiren_dim(store.dim, self.suiren_cache_layout) for stream, store in self.graph_stores.items() if store.path is not None}

    @property
    def suiren_atom_dims(self) -> dict[str, int]:
        return {stream: _r_only_suiren_dim(store.dim, self.suiren_cache_layout) for stream, store in self.atom_stores.items() if store.path is not None}

    def __len__(self) -> int:
        return len(self.samples)

    def _target_class(self, sample: dict[str, Any]) -> np.ndarray:
        bo_r, bo_p = bond_order_matrices(sample)
        return delta_class_matrix(bo_p - bo_r, self.delta_vocab, decimals=self.decimals)

    def _group_targets(self, idx: int, current_target: np.ndarray) -> list[np.ndarray]:
        indices = list(self.group_to_indices[self.reactant_keys[idx]])
        if self.max_group_products > 0 and len(indices) > self.max_group_products:
            indices = indices[: self.max_group_products]
            if idx not in indices:
                indices[-1] = idx
        out: list[np.ndarray] = []
        seen: set[bytes] = set()
        for group_idx in indices:
            target = current_target if group_idx == idx else self._target_class(self.samples[group_idx])
            signature = np.ascontiguousarray(target).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            out.append(target)
        return out or [current_target]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        pair = product_edit_pair_input(sample, self.args)
        reaction_id = str(sample["reaction_id"])
        target_class = delta_class_matrix(pair["delta_bo"], self.delta_vocab, decimals=self.decimals)
        item: dict[str, Any] = {
            "reaction_id": reaction_id,
            "reactant_key": self.reactant_keys[idx],
            "z": np.asarray(sample["atomic_numbers"], dtype=np.int64),
            "pair_input": pair["pair_input"],
            "pair_valid": pair["pair_valid"],
            "bo_r": pair["bo_r"],
            "target_delta_bo": pair["delta_bo"],
            "target_class": target_class,
            "group_target_class": self._group_targets(idx, target_class),
            "changed": pair["changed"],
            "edit_class": pair["edit_class"],
            "core_atom": pair["core_atom"],
            "n_atoms": len(sample["atomic_numbers"]),
            "group_size": len(self.group_to_indices[self.reactant_keys[idx]]),
        }
        for stream, store in self.graph_stores.items():
            features, failed = store.get(idx, reaction_id)
            if features is not None:
                item[f"suiren_{stream}_graph_features"] = _r_only_suiren_feature(features, self.suiren_cache_layout)
                item[f"suiren_{stream}_graph_failed"] = failed
        for stream, store in self.atom_stores.items():
            features, failed = store.get(idx, reaction_id, sample["atom_map_order"])
            if features is not None:
                item[f"suiren_{stream}_atom_features"] = _r_only_suiren_feature(features, self.suiren_cache_layout)
                item[f"suiren_{stream}_atom_failed"] = failed
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_n = max(item["n_atoms"] for item in batch)
    max_group = max(len(item["group_target_class"]) for item in batch)
    pair_dim = batch[0]["pair_input"].shape[-1]
    z = torch.zeros((batch_size, max_n), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    pair_input = torch.zeros((batch_size, max_n, max_n, pair_dim), dtype=torch.float32)
    pair_valid = torch.zeros((batch_size, max_n, max_n), dtype=torch.bool)
    bo_r = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    target_delta_bo = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    target_class = torch.zeros((batch_size, max_n, max_n), dtype=torch.long)
    group_target_class = torch.zeros((batch_size, max_group, max_n, max_n), dtype=torch.long)
    group_target_mask = torch.zeros((batch_size, max_group), dtype=torch.bool)
    changed = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    edit_class = torch.zeros((batch_size, max_n, max_n), dtype=torch.long)
    core_atom = torch.zeros((batch_size, max_n), dtype=torch.float32)
    group_size: list[int] = []
    n_atoms: list[int] = []
    reaction_ids: list[str] = []
    reactant_keys: list[str] = []

    for row, item in enumerate(batch):
        n = item["n_atoms"]
        z[row, :n] = torch.from_numpy(item["z"])
        atom_mask[row, :n] = True
        pair_input[row, :n, :n] = torch.from_numpy(item["pair_input"])
        pair_valid[row, :n, :n] = torch.from_numpy(item["pair_valid"])
        bo_r[row, :n, :n] = torch.from_numpy(item["bo_r"])
        target_delta_bo[row, :n, :n] = torch.from_numpy(item["target_delta_bo"])
        target_class[row, :n, :n] = torch.from_numpy(item["target_class"])
        changed[row, :n, :n] = torch.from_numpy(item["changed"])
        edit_class[row, :n, :n] = torch.from_numpy(item["edit_class"])
        core_atom[row, :n] = torch.from_numpy(item["core_atom"])
        for group_idx, target in enumerate(item["group_target_class"]):
            group_target_class[row, group_idx, :n, :n] = torch.from_numpy(target)
            group_target_mask[row, group_idx] = True
        group_size.append(int(item["group_size"]))
        n_atoms.append(n)
        reaction_ids.append(item["reaction_id"])
        reactant_keys.append(item["reactant_key"])

    out: dict[str, Any] = {
        "reaction_id": reaction_ids,
        "reactant_key": reactant_keys,
        "z": z,
        "atom_mask": atom_mask,
        "pair_input": pair_input,
        "pair_valid": pair_valid,
        "bo_r": bo_r,
        "target_delta_bo": target_delta_bo,
        "target_class": target_class,
        "group_target_class": group_target_class,
        "group_target_mask": group_target_mask,
        "changed": changed,
        "edit_class": edit_class,
        "core_atom": core_atom,
        "n_atoms": torch.tensor(n_atoms, dtype=torch.long),
        "group_size": torch.tensor(group_size, dtype=torch.long),
    }

    for stream in SUIREN_STREAMS:
        feature_key = f"suiren_{stream}_graph_features"
        failed_key = f"suiren_{stream}_graph_failed"
        graph_dim = 0
        for item in batch:
            feature = item.get(feature_key)
            if feature is not None:
                graph_dim = int(feature.shape[-1])
                break
        if graph_dim:
            graph = torch.zeros((batch_size, graph_dim), dtype=torch.float32)
            graph_failed = torch.zeros((batch_size,), dtype=torch.bool)
            for row, item in enumerate(batch):
                feature = item.get(feature_key)
                if feature is not None:
                    graph[row] = torch.from_numpy(feature)
                graph_failed[row] = bool(item.get(failed_key, False))
            out[feature_key] = graph
            out[failed_key] = graph_failed

    for stream in SUIREN_STREAMS:
        feature_key = f"suiren_{stream}_atom_features"
        failed_key = f"suiren_{stream}_atom_failed"
        atom_dim = 0
        for item in batch:
            feature = item.get(feature_key)
            if feature is not None:
                atom_dim = int(feature.shape[-1])
                break
        if atom_dim:
            atom = torch.zeros((batch_size, max_n, atom_dim), dtype=torch.float32)
            atom_failed = torch.zeros((batch_size,), dtype=torch.bool)
            for row, item in enumerate(batch):
                feature = item.get(feature_key)
                n = int(item["n_atoms"])
                if feature is not None:
                    atom[row, :n] = torch.from_numpy(feature)
                atom_failed[row] = bool(item.get(failed_key, False))
            out[feature_key] = atom
            out[failed_key] = atom_failed

    return out


def target_upper(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.triu(batch["pair_valid"].bool(), diagonal=1)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def class_weights(delta_vocab: np.ndarray, args: Any, device: torch.device) -> torch.Tensor:
    explicit = getattr(args, "delta_class_weights", "")
    if explicit:
        values = parse_float_list(explicit)
        if len(values) != len(delta_vocab):
            raise ValueError(f"--delta-class-weights must have {len(delta_vocab)} values")
        return torch.tensor(values, dtype=torch.float32, device=device)
    weights = torch.full((len(delta_vocab),), float(getattr(args, "changed_class_weight", 4.0)), dtype=torch.float32, device=device)
    weights[no_change_index(delta_vocab)] = float(getattr(args, "no_change_class_weight", 0.2))
    return weights


def compute_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: Any,
    delta_class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = out["delta_logits"]
    pair_mask = target_upper(batch).float()
    group_targets = batch["group_target_class"]
    group_mask = batch["group_target_mask"].bool()
    batch_size, n_candidates, n_atoms, _, n_classes = logits.shape
    n_group = group_targets.shape[1]
    pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    per_group_losses: list[torch.Tensor] = []
    flat_logits = logits.reshape(batch_size, n_candidates, n_atoms * n_atoms, n_classes)
    flat_pair_mask = pair_mask.reshape(batch_size, n_atoms * n_atoms)
    for group_idx in range(n_group):
        target = group_targets[:, group_idx].reshape(batch_size, n_atoms * n_atoms)
        candidate_losses: list[torch.Tensor] = []
        for cand_idx in range(n_candidates):
            ce = F.cross_entropy(
                flat_logits[:, cand_idx].reshape(batch_size * n_atoms * n_atoms, n_classes),
                target.reshape(batch_size * n_atoms * n_atoms),
                weight=delta_class_weights,
                reduction="none",
            ).reshape(batch_size, n_atoms * n_atoms)
            candidate_losses.append((ce * flat_pair_mask).sum(dim=1) / pair_count)
        per_group_losses.append(torch.stack(candidate_losses, dim=1).min(dim=1).values)
    loss_matrix = torch.stack(per_group_losses, dim=1)
    active = group_mask.float()
    loss = (loss_matrix * active).sum() / active.sum().clamp_min(1.0)
    return loss, {"product_edit_loss": float(loss.detach().cpu())}


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: Any,
    delta_class_weights: torch.Tensor,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "product_edit_loss": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(out, batch, args, delta_class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_size = batch["z"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["product_edit_loss"] += parts["product_edit_loss"] * batch_size
        totals["n"] += batch_size
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    args: Any,
    delta_class_weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "product_edit_loss": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(out, batch, args, delta_class_weights)
        batch_size = batch["z"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["product_edit_loss"] += parts["product_edit_loss"] * batch_size
        totals["n"] += batch_size
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


def _edit_class_from_delta(bo_r: np.ndarray, delta: np.ndarray, bond_eps: float = 1e-6) -> np.ndarray:
    bo_p = bo_r + delta
    cls = np.zeros_like(delta, dtype=np.int64)
    cls[(bo_r <= bond_eps) & (bo_p > bond_eps)] = 1
    cls[(bo_r > bond_eps) & (bo_p <= bond_eps)] = 2
    cls[(bo_r > bond_eps) & (bo_p > bond_eps) & (np.abs(delta) > bond_eps)] = 3
    return cls


def _valid_product(bo_r: np.ndarray, delta: np.ndarray, pair_mask: np.ndarray, max_bo: float = 3.5, tol: float = 1e-5) -> bool:
    bo_p = bo_r + delta
    values = bo_p[pair_mask]
    return bool(np.isfinite(values).all() and values.min(initial=0.0) >= -tol and values.max(initial=0.0) <= max_bo + tol)


def _candidate_exact(candidates: np.ndarray, targets: np.ndarray, pair_mask: np.ndarray, top_k: int) -> bool:
    k = min(top_k, candidates.shape[0])
    active_targets = targets[:, pair_mask]
    for cand_idx in range(k):
        pred = candidates[cand_idx][pair_mask]
        if bool((active_targets == pred[None, :]).all(axis=1).any()):
            return True
    return False


def _summarize_edits(delta: np.ndarray, pair_mask: np.ndarray, limit: int = 16) -> str:
    edits = []
    for i, j in np.argwhere(pair_mask & (np.abs(delta) > 1e-6))[:limit]:
        edits.append(f"{int(i)}-{int(j)}:{float(delta[i, j]):g}")
    return ";".join(edits)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    delta_vocab: np.ndarray,
    max_prediction_rows: int = 20000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    no_change = no_change_index(delta_vocab)
    topks = (1, 3, 5)
    exact_sample = {k: 0 for k in topks}
    exact_group = {k: 0 for k in topks}
    valid_top1 = 0
    valid_topk = 0
    valid_topk_total = 0
    n_samples = 0
    changed_true_top1: list[np.ndarray] = []
    changed_pred_top1: list[np.ndarray] = []
    changed_true_best: list[np.ndarray] = []
    changed_pred_best: list[np.ndarray] = []
    edit_true_top1: list[np.ndarray] = []
    edit_pred_top1: list[np.ndarray] = []
    edit_true_best: list[np.ndarray] = []
    edit_pred_best: list[np.ndarray] = []
    delta_true_top1: list[np.ndarray] = []
    delta_pred_top1: list[np.ndarray] = []
    delta_mae_best: list[float] = []
    group_sizes: list[int] = []
    rows: list[dict[str, Any]] = []

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        pred_classes = out["delta_logits"].argmax(dim=-1).detach().cpu().numpy()
        bo_r = batch["bo_r"].detach().cpu().numpy()
        target_class = batch["target_class"].detach().cpu().numpy()
        target_delta = batch["target_delta_bo"].detach().cpu().numpy()
        group_target = batch["group_target_class"].detach().cpu().numpy()
        group_mask = batch["group_target_mask"].detach().cpu().numpy().astype(bool)
        pair_valid = batch["pair_valid"].detach().cpu().numpy().astype(bool)
        group_sizes.extend(batch["group_size"].detach().cpu().numpy().astype(int).tolist())

        for row_idx, reaction_id in enumerate(batch["reaction_id"]):
            n_samples += 1
            n_atoms = int(batch["n_atoms"][row_idx].detach().cpu())
            pair_mask = np.triu(pair_valid[row_idx, :n_atoms, :n_atoms], k=1)
            candidates = pred_classes[row_idx, :, :n_atoms, :n_atoms].copy()
            for atom_idx in range(n_atoms):
                candidates[:, atom_idx, atom_idx] = no_change
            row_target = target_class[row_idx, :n_atoms, :n_atoms]
            row_delta = target_delta[row_idx, :n_atoms, :n_atoms]
            active_group_targets = group_target[row_idx, group_mask[row_idx], :n_atoms, :n_atoms]
            if active_group_targets.size == 0:
                active_group_targets = row_target[None, :, :]

            for k in topks:
                exact_sample[k] += int(_candidate_exact(candidates, row_target[None, :, :], pair_mask, k))
                exact_group[k] += int(_candidate_exact(candidates, active_group_targets, pair_mask, k))

            pred_delta_all = delta_vocab[candidates]
            valid_top1 += int(_valid_product(bo_r[row_idx, :n_atoms, :n_atoms], pred_delta_all[0], pair_mask))
            for cand_idx in range(min(5, pred_delta_all.shape[0])):
                valid_topk += int(_valid_product(bo_r[row_idx, :n_atoms, :n_atoms], pred_delta_all[cand_idx], pair_mask))
                valid_topk_total += 1

            pair_target = row_target[pair_mask]
            pair_errors = [(candidates[cand_idx][pair_mask] != pair_target).mean() for cand_idx in range(candidates.shape[0])]
            best_idx = int(np.argmin(pair_errors))
            top1_delta = pred_delta_all[0]
            best_delta = pred_delta_all[best_idx]
            true_edit = _edit_class_from_delta(bo_r[row_idx, :n_atoms, :n_atoms], row_delta)[pair_mask]
            top1_edit = _edit_class_from_delta(bo_r[row_idx, :n_atoms, :n_atoms], top1_delta)[pair_mask]
            best_edit = _edit_class_from_delta(bo_r[row_idx, :n_atoms, :n_atoms], best_delta)[pair_mask]

            changed_true_top1.append((pair_target != no_change).astype(np.int64))
            changed_pred_top1.append((candidates[0][pair_mask] != no_change).astype(np.int64))
            changed_true_best.append((pair_target != no_change).astype(np.int64))
            changed_pred_best.append((candidates[best_idx][pair_mask] != no_change).astype(np.int64))
            edit_true_top1.append(true_edit)
            edit_pred_top1.append(top1_edit)
            edit_true_best.append(true_edit)
            edit_pred_best.append(best_edit)
            delta_true_top1.append(row_delta[pair_mask])
            delta_pred_top1.append(top1_delta[pair_mask])
            delta_mae_best.append(float(np.mean(np.abs(row_delta[pair_mask] - best_delta[pair_mask]))))

            if len(rows) < max_prediction_rows:
                for cand_idx in range(min(5, candidates.shape[0])):
                    cand_delta = pred_delta_all[cand_idx]
                    rows.append(
                        {
                            "reaction_id": reaction_id,
                            "candidate_rank": cand_idx + 1,
                            "reactant_key": batch["reactant_key"][row_idx],
                            "exact_sample": int(_candidate_exact(candidates[cand_idx : cand_idx + 1], row_target[None, :, :], pair_mask, 1)),
                            "exact_group": int(_candidate_exact(candidates[cand_idx : cand_idx + 1], active_group_targets, pair_mask, 1)),
                            "valid_product": int(_valid_product(bo_r[row_idx, :n_atoms, :n_atoms], cand_delta, pair_mask)),
                            "n_predicted_edits": int((np.abs(cand_delta[pair_mask]) > 1e-6).sum()),
                            "edits": _summarize_edits(cand_delta, pair_mask),
                        }
                    )
                    if len(rows) >= max_prediction_rows:
                        break

    denom = max(n_samples, 1)
    changed_y_top1 = np.concatenate(changed_true_top1) if changed_true_top1 else np.asarray([], dtype=np.int64)
    changed_p_top1 = np.concatenate(changed_pred_top1) if changed_pred_top1 else np.asarray([], dtype=np.int64)
    changed_y_best = np.concatenate(changed_true_best) if changed_true_best else np.asarray([], dtype=np.int64)
    changed_p_best = np.concatenate(changed_pred_best) if changed_pred_best else np.asarray([], dtype=np.int64)
    edit_y_top1 = np.concatenate(edit_true_top1) if edit_true_top1 else np.asarray([], dtype=np.int64)
    edit_p_top1 = np.concatenate(edit_pred_top1) if edit_pred_top1 else np.asarray([], dtype=np.int64)
    edit_y_best = np.concatenate(edit_true_best) if edit_true_best else np.asarray([], dtype=np.int64)
    edit_p_best = np.concatenate(edit_pred_best) if edit_pred_best else np.asarray([], dtype=np.int64)
    delta_y = np.concatenate(delta_true_top1) if delta_true_top1 else np.asarray([], dtype=np.float32)
    delta_p = np.concatenate(delta_pred_top1) if delta_pred_top1 else np.asarray([], dtype=np.float32)

    metrics: dict[str, Any] = {
        "n_samples": int(n_samples),
        "group_size_mean": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "multi_product_row_rate": float(np.mean(np.asarray(group_sizes) > 1)) if group_sizes else 0.0,
        "valid_product_rate_top1": float(valid_top1 / denom),
        "valid_product_rate_top5": float(valid_topk / max(valid_topk_total, 1)),
        "edit_pair_F1_top1": binary_f1(changed_y_top1, changed_p_top1),
        "edit_pair_F1_best_of_K": binary_f1(changed_y_best, changed_p_best),
        "edit_macro_F1_top1": macro_f1(edit_y_top1, edit_p_top1, n_classes=len(EDIT_CLASSES)),
        "edit_macro_F1_best_of_K": macro_f1(edit_y_best, edit_p_best, n_classes=len(EDIT_CLASSES)),
        "formed_broken_F1_top1": binary_f1(np.isin(edit_y_top1, [1, 2]).astype(int), np.isin(edit_p_top1, [1, 2]).astype(int)),
        "formed_broken_F1_best_of_K": binary_f1(np.isin(edit_y_best, [1, 2]).astype(int), np.isin(edit_p_best, [1, 2]).astype(int)),
        "Delta_BO_MAE_top1": mean_absolute_error(delta_y, delta_p) if delta_y.size else 0.0,
        "Delta_BO_MAE_best_of_K": float(np.mean(delta_mae_best)) if delta_mae_best else 0.0,
    }
    for k in topks:
        metrics[f"exact_product_match_top{k}_sample"] = float(exact_sample[k] / denom)
        metrics[f"exact_product_match_top{k}_group"] = float(exact_group[k] / denom)
    return metrics, rows


def write_summary(output_dir: Path, best: dict[str, Any]) -> None:
    lines = [
        "# RFM R-only Top-K Product Graph-Edit Prediction",
        "",
        f"Best epoch: {best['epoch']}",
        "Selection: minimum valid_loss",
        f"Input schema: {best['input_schema']}",
        f"K: {best['top_k']}",
        "",
        "| Split | top1 sample | top3 sample | top5 sample | top1 group | top3 group | top5 group | edit F1 top1 | edit F1 best-K | Delta BO MAE top1 | Delta BO MAE best-K | valid top5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        metric = best[split]
        lines.append(
            f"| {split} | {metric['exact_product_match_top1_sample']:.4f} | {metric['exact_product_match_top3_sample']:.4f} | "
            f"{metric['exact_product_match_top5_sample']:.4f} | {metric['exact_product_match_top1_group']:.4f} | "
            f"{metric['exact_product_match_top3_group']:.4f} | {metric['exact_product_match_top5_group']:.4f} | "
            f"{metric['edit_pair_F1_top1']:.4f} | {metric['edit_pair_F1_best_of_K']:.4f} | "
            f"{metric['Delta_BO_MAE_top1']:.4f} | {metric['Delta_BO_MAE_best_of_K']:.4f} | {metric['valid_product_rate_top5']:.4f} |"
        )
    output_dir.joinpath("summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
