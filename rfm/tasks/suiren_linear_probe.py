"""Suiren-only linear probing utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from rfm.data.hdf5 import STRING_DTYPE, split_path
from rfm.data.reaction_samples import ENERGY_TARGETS
from rfm.tasks.metrics import regression_metrics


POOLED_SCHEMA = "rfm_suiren_atom_mean_probe_hdf5_v0.1"
POOLED_LAYOUT = ("mean_h_R", "mean_h_P", "mean_abs_Delta_h")
REPRESENTATIONS = ("state", "state_abs_delta")


@dataclass(frozen=True)
class ProbeSplit:
    reaction_ids: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    state_dim: int
    metadata: dict[str, Any]


def _decode_ids(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[()], dtype=str)


def _require_complete_atom_cache(h5: h5py.File, path: Path) -> None:
    required = {"reaction_ids", "features", "atom_ptr", "failed", "processed"}
    missing = sorted(required.difference(h5.keys()))
    if missing:
        raise ValueError(f"atom cache is missing datasets {missing}: {path}")
    failed = np.asarray(h5["failed"], dtype=bool)
    if failed.any():
        raise ValueError(f"atom cache has failed rows: {path} failed={int(failed.sum())}/{len(failed)}")
    processed = np.asarray(h5["processed"], dtype=bool)
    if not processed.all():
        raise ValueError(
            f"atom cache is incomplete: {path} processed={int(processed.sum())}/{len(processed)}"
        )


def validate_pooled_cache(path: str | Path, source_path: str | Path | None = None) -> dict[str, Any]:
    cache_path = Path(path)
    with h5py.File(cache_path, "r") as h5:
        if h5.attrs.get("schema_version", "") != POOLED_SCHEMA:
            raise ValueError(f"unexpected pooled cache schema: {cache_path}")
        required = {"reaction_ids", "features", "n_atoms", "processed"}
        missing = sorted(required.difference(h5.keys()))
        if missing:
            raise ValueError(f"pooled cache is missing datasets {missing}: {cache_path}")
        n_rows = len(h5["reaction_ids"])
        if h5["features"].shape[0] != n_rows or h5["n_atoms"].shape != (n_rows,):
            raise ValueError(f"pooled cache row counts disagree: {cache_path}")
        processed = np.asarray(h5["processed"], dtype=bool)
        if processed.shape != (n_rows,) or not processed.all():
            raise ValueError(f"pooled cache is incomplete: {cache_path}")
        metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
        if source_path is not None and metadata.get("source_atom_cache") != str(Path(source_path)):
            raise ValueError(f"pooled cache source mismatch: {cache_path}")
        if metadata.get("layout") != list(POOLED_LAYOUT):
            raise ValueError(f"unexpected pooled feature layout: {cache_path}")
        state_dim = int(metadata.get("state_dim", 0))
        if state_dim <= 0 or h5["features"].shape[1] != state_dim * len(POOLED_LAYOUT):
            raise ValueError(f"invalid pooled feature dimension: {cache_path}")
        if not np.isfinite(np.asarray(h5["features"][: min(n_rows, 1024)], dtype=np.float32)).all():
            raise ValueError(f"pooled cache contains non-finite features: {cache_path}")
        return metadata


def _mean_rows(values: np.ndarray, local_ptr: np.ndarray, counts: np.ndarray) -> np.ndarray:
    starts = local_ptr[:-1]
    sums = np.add.reduceat(values, starts, axis=0)
    return sums / counts[:, None]


def build_pooled_atom_cache(
    source_path: str | Path,
    output_path: str | Path,
    *,
    chunk_reactions: int = 2048,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Mean-pool atom-aligned h_R, h_P, and abs(Delta_h) into a reaction cache."""

    source = Path(source_path)
    output = Path(output_path)
    if chunk_reactions <= 0:
        raise ValueError("chunk_reactions must be positive")
    if output.exists() and not overwrite:
        return validate_pooled_cache(output, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    with h5py.File(source, "r") as src:
        _require_complete_atom_cache(src, source)
        n_rows = len(src["reaction_ids"])
        atom_ptr = np.asarray(src["atom_ptr"], dtype=np.int64)
        if atom_ptr.shape != (n_rows + 1,) or atom_ptr[0] != 0:
            raise ValueError(f"invalid atom_ptr in {source}")
        if atom_ptr[-1] != src["features"].shape[0]:
            raise ValueError(f"atom_ptr does not span feature rows in {source}")
        n_atoms = np.diff(atom_ptr)
        if np.any(n_atoms <= 0):
            raise ValueError(f"atom cache contains empty reactions: {source}")
        feature_dim = int(src["features"].shape[1])
        if feature_dim % 4:
            raise ValueError(f"expected [h_R,h_P,Delta_h,abs_Delta_h] cache layout: {source}")
        state_dim = feature_dim // 4
        source_metadata = json.loads(src.attrs.get("metadata_json", "{}"))
        metadata = {
            "source_atom_cache": str(source),
            "source_metadata": source_metadata,
            "source_layout": ["h_R", "h_P", "Delta_h", "abs_Delta_h"],
            "layout": list(POOLED_LAYOUT),
            "state_dim": state_dim,
            "aggregation": "unweighted_atom_mean",
            "explicit_Delta_h_excluded": True,
            "uses_reaction_encoder": False,
        }
        reaction_ids = _decode_ids(src["reaction_ids"])

        with h5py.File(temporary, "w") as dst:
            dst.attrs["schema_version"] = POOLED_SCHEMA
            dst.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            dst.create_dataset("reaction_ids", data=reaction_ids.astype(object), dtype=STRING_DTYPE)
            dst.create_dataset("n_atoms", data=n_atoms.astype(np.int32))
            processed = dst.create_dataset("processed", shape=(n_rows,), dtype=np.bool_)
            feature_chunks = (min(chunk_reactions, max(n_rows, 1)), state_dim * len(POOLED_LAYOUT))
            pooled_features = dst.create_dataset(
                "features",
                shape=(n_rows, state_dim * len(POOLED_LAYOUT)),
                dtype=np.float32,
                chunks=feature_chunks,
            )

            for row_start in range(0, n_rows, chunk_reactions):
                row_stop = min(row_start + chunk_reactions, n_rows)
                atom_start = int(atom_ptr[row_start])
                atom_stop = int(atom_ptr[row_stop])
                local_ptr = atom_ptr[row_start : row_stop + 1] - atom_start
                counts = np.diff(local_ptr).astype(np.float32)
                h_r = np.asarray(src["features"][atom_start:atom_stop, :state_dim], dtype=np.float32)
                h_p = np.asarray(
                    src["features"][atom_start:atom_stop, state_dim : 2 * state_dim],
                    dtype=np.float32,
                )
                abs_delta = np.asarray(
                    src["features"][atom_start:atom_stop, 3 * state_dim : 4 * state_dim],
                    dtype=np.float32,
                )
                pooled = np.concatenate(
                    [
                        _mean_rows(h_r, local_ptr, counts),
                        _mean_rows(h_p, local_ptr, counts),
                        _mean_rows(abs_delta, local_ptr, counts),
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                if not np.isfinite(pooled).all():
                    raise ValueError(f"non-finite pooled features at rows {row_start}:{row_stop}: {source}")
                pooled_features[row_start:row_stop] = pooled
                processed[row_start:row_stop] = True
            dst.flush()

    os.replace(temporary, output)
    validate_pooled_cache(output, source)
    return metadata


def _load_targets(data_spec: str) -> tuple[np.ndarray, np.ndarray]:
    data_path, split = split_path(data_spec)
    with h5py.File(data_path, "r") as h5:
        if split is None:
            names = list(h5.get("splits", {}).keys())
            if len(names) != 1:
                raise ValueError(f"data has splits {names}; use file.h5::split")
            split = names[0]
        group = h5["splits"][split]
        reaction_ids = _decode_ids(group["reaction_id"])
        columns = tuple(json.loads(group["targets"].attrs.get("columns", "[]")))
        column_index = {name: idx for idx, name in enumerate(columns)}
        missing = [name for name in ENERGY_TARGETS if name not in column_index]
        if missing:
            raise ValueError(f"data targets are missing {missing}: {data_spec}")
        targets = np.asarray(
            group["targets"][:, [column_index[name] for name in ENERGY_TARGETS]],
            dtype=np.float32,
        )
        return reaction_ids, targets


def load_probe_split(data_spec: str, pooled_cache: str | Path, representation: str) -> ProbeSplit:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; choose from {REPRESENTATIONS}")
    metadata = validate_pooled_cache(pooled_cache)
    state_dim = int(metadata["state_dim"])
    with h5py.File(pooled_cache, "r") as h5:
        cache_ids = _decode_ids(h5["reaction_ids"])
        stop = 2 * state_dim if representation == "state" else 3 * state_dim
        features = np.asarray(h5["features"][:, :stop], dtype=np.float32)
    data_ids, targets = _load_targets(data_spec)
    if cache_ids.shape != data_ids.shape or not np.array_equal(cache_ids, data_ids):
        mismatch = np.flatnonzero(cache_ids != data_ids) if cache_ids.shape == data_ids.shape else np.asarray([])
        detail = f" first_mismatch={int(mismatch[0])}" if mismatch.size else ""
        raise ValueError(f"pooled cache/data reaction_id mismatch:{detail}")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("pooled feature and target row counts disagree")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("probe split contains non-finite values")
    return ProbeSplit(cache_ids, features, targets, state_dim, metadata)


def _standardize(
    value: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(value).to(device=device, dtype=torch.float32).clone()
    return tensor.sub_(mean).div_(std)


def fit_ridge_probe(
    train: ProbeSplit,
    valid: ProbeSplit,
    test: ProbeSplit,
    *,
    alphas: list[float],
    device: torch.device,
) -> dict[str, Any]:
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("ridge alphas must be positive")
    feature_dim = train.features.shape[1]
    if valid.features.shape[1] != feature_dim or test.features.shape[1] != feature_dim:
        raise ValueError("probe feature dimensions disagree across splits")

    x_train_raw = torch.from_numpy(train.features).to(device=device, dtype=torch.float32)
    feature_mean = x_train_raw.mean(dim=0)
    feature_std = x_train_raw.std(dim=0, correction=0).clamp_min(1e-6)
    x_train = x_train_raw.sub_(feature_mean).div_(feature_std)
    x_valid = _standardize(valid.features, feature_mean, feature_std, device)
    x_test = _standardize(test.features, feature_mean, feature_std, device)

    y_train_raw = torch.from_numpy(train.targets).to(device=device, dtype=torch.float32)
    target_mean = y_train_raw.mean(dim=0)
    target_std = y_train_raw.std(dim=0, correction=0).clamp_min(1e-6)
    y_train = y_train_raw.sub_(target_mean).div_(target_std)
    y_valid = _standardize(valid.targets, target_mean, target_std, device)

    covariance = x_train.T @ x_train / len(train.features)
    cross_covariance = x_train.T @ y_train / len(train.features)
    identity = torch.eye(feature_dim, dtype=covariance.dtype, device=device)
    search: list[dict[str, float]] = []
    best_loss = float("inf")
    best_alpha = 0.0
    best_weight: torch.Tensor | None = None
    for alpha in sorted(set(float(value) for value in alphas)):
        weight = torch.linalg.solve(covariance + alpha * identity, cross_covariance)
        valid_prediction = x_valid @ weight
        valid_loss = float(torch.mean((valid_prediction - y_valid) ** 2).item())
        search.append({"alpha": alpha, "valid_normalized_mse": valid_loss})
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_alpha = alpha
            best_weight = weight
    if best_weight is None:
        raise RuntimeError("ridge search did not produce a model")

    valid_prediction = (x_valid @ best_weight) * target_std + target_mean
    test_prediction = (x_test @ best_weight) * target_std + target_mean
    valid_pred_np = valid_prediction.cpu().numpy()
    test_pred_np = test_prediction.cpu().numpy()
    mean_valid = np.broadcast_to(target_mean.cpu().numpy(), valid.targets.shape)
    mean_test = np.broadcast_to(target_mean.cpu().numpy(), test.targets.shape)
    return {
        "feature_dim": feature_dim,
        "selected_alpha": best_alpha,
        "selected_valid_normalized_mse": best_loss,
        "alpha_search": search,
        "feature_mean": feature_mean.cpu().numpy(),
        "feature_std": feature_std.cpu().numpy(),
        "target_mean": target_mean.cpu().numpy(),
        "target_std": target_std.cpu().numpy(),
        "weight": best_weight.cpu().numpy(),
        "valid_prediction": valid_pred_np,
        "test_prediction": test_pred_np,
        "valid_metrics": regression_metrics(valid.targets, valid_pred_np, ENERGY_TARGETS),
        "test_metrics": regression_metrics(test.targets, test_pred_np, ENERGY_TARGETS),
        "mean_baseline_valid_metrics": regression_metrics(valid.targets, mean_valid, ENERGY_TARGETS),
        "mean_baseline_test_metrics": regression_metrics(test.targets, mean_test, ENERGY_TARGETS),
    }
