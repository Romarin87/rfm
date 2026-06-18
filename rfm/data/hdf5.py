#!/usr/bin/env python
"""HDF5 I/O helpers for RFM ReactionDeltaSample records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
TARGETS = ("dE", "dE_dagger", "dH", "dH_dagger", "dG", "dG_dagger")


def _as_json(sample: dict[str, Any]) -> str:
    return json.dumps(sample, separators=(",", ":"), ensure_ascii=False)


def _decode_json(value: Any) -> dict[str, Any]:
    return json.loads(value.decode("utf-8") if isinstance(value, bytes) else str(value))


def _flatten_int_lists(values: list[list[int]], dtype: str) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(values) + 1, dtype=np.int64)
    total = 0
    for idx, row in enumerate(values):
        total += len(row)
        ptr[idx + 1] = total
    if total:
        data = np.asarray([item for row in values for item in row], dtype=dtype)
    else:
        data = np.asarray([], dtype=dtype)
    return ptr, data


def write_split(group: h5py.Group, samples: list[dict[str, Any]]) -> None:
    """Write one split group using the v0.1 packed schema."""
    group.attrs["schema_version"] = "rfm_reaction_delta_hdf5_v0.1"
    group.attrs["n_samples"] = len(samples)
    group.create_dataset("reaction_id", data=np.asarray([s["reaction_id"] for s in samples], dtype=object), dtype=STRING_DTYPE)
    group.create_dataset("sample_json", data=np.asarray([_as_json(s) for s in samples], dtype=object), dtype=STRING_DTYPE)
    group.create_dataset("n_atoms", data=np.asarray([len(s["atomic_numbers"]) for s in samples], dtype=np.int32))
    group.create_dataset("n_reactants", data=np.asarray([int(s.get("n_reactants", 0)) for s in samples], dtype=np.int16))
    group.create_dataset("n_products", data=np.asarray([int(s.get("n_products", 0)) for s in samples], dtype=np.int16))
    group.create_dataset(
        "targets",
        data=np.asarray([[float(s.get("targets", {}).get(name, np.nan)) for name in TARGETS] for s in samples], dtype=np.float32),
    )
    group["targets"].attrs["columns"] = json.dumps(TARGETS)

    atom_ptr, atom_data = _flatten_int_lists([list(map(int, s["atomic_numbers"])) for s in samples], "i2")
    map_ptr, map_data = _flatten_int_lists([list(map(int, s["atom_map_order"])) for s in samples], "i4")
    core_ptr, core_data = _flatten_int_lists([[int(v) for v in s.get("reaction_core_mask", [])] for s in samples], "i1")
    group.create_dataset("atomic_numbers_ptr", data=atom_ptr)
    group.create_dataset("atomic_numbers", data=atom_data)
    group.create_dataset("atom_map_order_ptr", data=map_ptr)
    group.create_dataset("atom_map_order", data=map_data)
    group.create_dataset("reaction_core_mask_ptr", data=core_ptr)
    group.create_dataset("reaction_core_mask", data=core_data)

    for side in ("R", "P"):
        present = [side in (s.get("coordinates") or {}) for s in samples]
        group.create_dataset(f"has_coordinates_{side}", data=np.asarray(present, dtype=np.bool_))
        if all(present):
            max_n = max((len(s["atomic_numbers"]) for s in samples), default=0)
            coords = np.full((len(samples), max_n, 3), np.nan, dtype=np.float32)
            for idx, sample in enumerate(samples):
                arr = np.asarray(sample["coordinates"][side], dtype=np.float32)
                coords[idx, : arr.shape[0], :] = arr
            group.create_dataset(f"coordinates_{side}", data=coords, compression="gzip", compression_opts=4, shuffle=True)


def write_processed_hdf5(
    output_path: str | Path,
    splits: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "rfm_reaction_delta_hdf5_v0.1"
        h5.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        split_root = h5.create_group("splits")
        for split, samples in splits.items():
            write_split(split_root.create_group(split), samples)


class Hdf5ReactionDeltaSampleStore(Sequence[dict[str, Any]]):
    """Lazy per-row access to one packed HDF5 split.

    The HDF5 file handle is opened lazily in each process. This avoids loading
    full train splits into Python memory before DataLoader workers start.
    """

    def __init__(self, path: str | Path, split: str | None = None, limit: int = 0, skip: int = 0):
        self.path = Path(path)
        self.split = split
        self.skip = max(int(skip), 0)
        self.limit = max(int(limit), 0)
        self._h5: h5py.File | None = None
        self._sample_json: h5py.Dataset | None = None
        self._targets: h5py.Dataset | None = None
        self._target_columns: tuple[str, ...] = ()
        with h5py.File(self.path, "r") as h5:
            if self.split is None:
                split_names = list(h5.get("splits", {}).keys())
                if len(split_names) != 1:
                    raise ValueError(f"HDF5 file has splits {split_names}; pass path as file.h5::split or use --*-split")
                self.split = split_names[0]
            group = h5["splits"][self.split]
            n_total = len(group["sample_json"])
            self.start = min(self.skip, n_total)
            self.stop = n_total if self.limit <= 0 else min(self.start + self.limit, n_total)
            if "targets" in group:
                columns = group["targets"].attrs.get("columns", "[]")
                self._target_columns = tuple(json.loads(columns))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_sample_json"] = None
        state["_targets"] = None
        return state

    def _ensure_open(self) -> None:
        if self._h5 is not None:
            return
        self._h5 = h5py.File(self.path, "r")
        group = self._h5["splits"][self.split]
        self._sample_json = group["sample_json"]
        self._targets = group.get("targets")

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        self._ensure_open()
        assert self._sample_json is not None
        return _decode_json(self._sample_json[self.start + idx])

    def target_matrix(self, targets: tuple[str, ...]) -> np.ndarray:
        if not targets:
            return np.zeros((len(self), 0), dtype=np.float32)
        if self._target_columns:
            self._ensure_open()
            assert self._targets is not None
            index = {name: idx for idx, name in enumerate(self._target_columns)}
            if all(name in index for name in targets):
                cols = [index[name] for name in targets]
                return np.asarray(self._targets[self.start : self.stop, cols], dtype=np.float32)
        return np.asarray([[float(self[idx]["targets"][name]) for name in targets] for idx in range(len(self))], dtype=np.float32)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._sample_json = None
        self._targets = None

    def __del__(self) -> None:
        self.close()


def load_reaction_delta_sample_store(
    path: str | Path,
    split: str | None = None,
    limit: int = 0,
    skip: int = 0,
) -> Sequence[dict[str, Any]]:
    in_path = Path(path)
    if in_path.suffix.lower() in {".h5", ".hdf5"}:
        return Hdf5ReactionDeltaSampleStore(in_path, split=split, limit=limit, skip=skip)
    return load_reaction_delta_samples(in_path, split=split, limit=limit, skip=skip)


def load_reaction_delta_samples(
    path: str | Path,
    split: str | None = None,
    limit: int = 0,
    skip: int = 0,
) -> list[dict[str, Any]]:
    """Load JSONL or packed HDF5 samples for existing RFM training code."""
    in_path = Path(path)
    if in_path.suffix.lower() not in {".h5", ".hdf5"}:
        rows: list[dict[str, Any]] = []
        with in_path.open(encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if line_idx < skip:
                    continue
                if limit and len(rows) >= limit:
                    break
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with h5py.File(in_path, "r") as h5:
        if split is None:
            split_names = list(h5.get("splits", {}).keys())
            if len(split_names) != 1:
                raise ValueError(f"HDF5 file has splits {split_names}; pass path as file.h5::split or use --*-split")
            split = split_names[0]
        ds = h5["splits"][split]["sample_json"]
        start = min(max(skip, 0), len(ds))
        stop = len(ds) if limit <= 0 else min(start + limit, len(ds))
        return [_decode_json(ds[idx]) for idx in range(start, stop)]


def split_path(path: str, default_split: str | None = None) -> tuple[str, str | None]:
    """Parse `processed.h5::train` style paths."""
    if "::" in path:
        file_path, split = path.rsplit("::", 1)
        return file_path, split
    return path, default_split


def write_feature_cache_hdf5(
    output_path: str | Path,
    reaction_ids: list[str],
    features: np.ndarray,
    failed: list[bool] | np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write graph-level frozen feature cache as HDF5."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "rfm_feature_cache_hdf5_v0.1"
        h5.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        h5.create_dataset("reaction_ids", data=np.asarray(reaction_ids, dtype=object), dtype=STRING_DTYPE)
        h5.create_dataset("features", data=np.asarray(features), compression="gzip", compression_opts=4, shuffle=True)
        h5.create_dataset("failed", data=np.asarray(failed, dtype=np.bool_))


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load `.npz` or HDF5 graph-level feature cache."""
    in_path = Path(path)
    if in_path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(in_path, "r") as h5:
            reaction_ids = h5["reaction_ids"].asstr()[()]
            if "processed" in h5 and not bool(np.asarray(h5["processed"][()], dtype=bool).all()):
                done = int(np.asarray(h5["processed"][()], dtype=bool).sum())
                raise ValueError(f"Feature cache is incomplete: {done}/{len(reaction_ids)} rows processed in {in_path}")
            return {
                "reaction_ids": np.asarray(reaction_ids, dtype=str),
                "features": np.asarray(h5["features"][()], dtype=np.float32),
                "failed": np.asarray(h5["failed"][()], dtype=bool) if "failed" in h5 else np.zeros(len(reaction_ids), dtype=bool),
            }
    payload = np.load(in_path)
    return {
        "reaction_ids": payload["reaction_ids"].astype(str),
        "features": payload["features"].astype(np.float32),
        "failed": payload["failed"].astype(bool) if "failed" in payload else np.zeros(len(payload["reaction_ids"]), dtype=bool),
    }
