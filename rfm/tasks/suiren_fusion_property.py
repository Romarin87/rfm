"""Suiren fusion reaction property regression datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rfm.data.reaction_samples import ENERGY_TARGETS, load_split, masked_edit_pair_input, reaction_property_pair_input, target_vector
from rfm.tasks import reaction_property


SUIREN_STREAMS = ("2d", "3d")


def _failed_count(h5: Any, path: Path, kind: str) -> int:
    if "failed" not in h5:
        raise ValueError(f"{kind} feature cache missing failed dataset: {path}")
    failed = np.asarray(h5["failed"][()], dtype=bool)
    count = int(failed.sum())
    if count:
        raise ValueError(f"{kind} feature cache has failed rows: {path} failed={count}/{len(failed)}")
    return count


class GraphFeatureStore:
    def __init__(self, path: str | None, *, trust_cache: bool = False, preload: bool = False):
        self.path = Path(path) if path else None
        self.trust_cache = trust_cache
        self.preload = preload
        self._h5: h5py.File | None = None
        self._ids: h5py.Dataset | None = None
        self._features: h5py.Dataset | None = None
        self._failed: h5py.Dataset | None = None
        self._ids_array: np.ndarray | None = None
        self._features_array: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.dim = 0
        self.length = 0
        if self.path is not None:
            import h5py

            with h5py.File(self.path, "r") as h5:
                self.length = len(h5["reaction_ids"])
                self.dim = int(h5["features"].shape[1])
                self.metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
                _failed_count(h5, self.path, "graph")
                if "processed" in h5 and not bool(np.asarray(h5["processed"][()], dtype=bool).all()):
                    done = int(np.asarray(h5["processed"][()], dtype=bool).sum())
                    raise ValueError(f"incomplete graph feature cache: {self.path} processed={done}/{len(h5['processed'])}")
                if self.preload:
                    self._features_array = np.asarray(h5["features"], dtype=np.float32)
                    if not self.trust_cache:
                        self._ids_array = np.asarray(h5["reaction_ids"].asstr()[()], dtype=object)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_ids"] = None
        state["_features"] = None
        state["_failed"] = None
        return state

    def _ensure_open(self) -> None:
        if self._features_array is not None:
            return
        if self.path is None or self._h5 is not None:
            return
        import h5py

        self._h5 = h5py.File(self.path, "r")
        if "processed" in self._h5 and not bool(np.asarray(self._h5["processed"][()], dtype=bool).all()):
            done = int(np.asarray(self._h5["processed"][()], dtype=bool).sum())
            raise ValueError(f"incomplete graph feature cache: {self.path} processed={done}/{len(self._h5['processed'])}")
        _failed_count(self._h5, self.path, "graph")
        if not self.trust_cache:
            self._ids = self._h5["reaction_ids"]
        self._features = self._h5["features"]
        self._failed = self._h5.get("failed")

    def __len__(self) -> int:
        return self.length

    def get(self, idx: int, reaction_id: str) -> tuple[np.ndarray | None, bool]:
        if self.path is None:
            return None, False
        self._ensure_open()
        if not self.trust_cache:
            if self._ids_array is not None:
                cached_id = str(self._ids_array[idx])
            else:
                assert self._ids is not None
                cached_id = self._ids.asstr()[idx]
            if cached_id != reaction_id:
                raise ValueError(f"graph cache reaction_id mismatch at {idx}: {cached_id} != {reaction_id}")
        if self._features_array is not None:
            feature = self._features_array[idx]
        else:
            assert self._features is not None
            feature = np.asarray(self._features[idx], dtype=np.float32)
        return feature, False

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._ids = None
        self._features = None
        self._failed = None

    def __del__(self) -> None:
        self.close()


class AtomFeatureStore:
    def __init__(self, path: str | None, *, trust_cache: bool = False, preload: bool = False):
        self.path = Path(path) if path else None
        self.trust_cache = trust_cache
        self.preload = preload
        self._h5: h5py.File | None = None
        self._ids: h5py.Dataset | None = None
        self._features: h5py.Dataset | None = None
        self._failed: h5py.Dataset | None = None
        self._atom_ptr: h5py.Dataset | None = None
        self._atom_map_order: h5py.Dataset | None = None
        self._ids_array: np.ndarray | None = None
        self._features_array: np.ndarray | None = None
        self._atom_ptr_array: np.ndarray | None = None
        self._atom_map_order_array: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.dim = 0
        self.length = 0
        if self.path is not None:
            import h5py

            with h5py.File(self.path, "r") as h5:
                self.length = len(h5["reaction_ids"])
                self.dim = int(h5["features"].shape[1])
                self.metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
                _failed_count(h5, self.path, "atom")
                if "processed" in h5 and not bool(np.asarray(h5["processed"][()], dtype=bool).all()):
                    done = int(np.asarray(h5["processed"][()], dtype=bool).sum())
                    raise ValueError(f"incomplete atom feature cache: {self.path} processed={done}/{len(h5['processed'])}")
                self._atom_ptr_array = np.asarray(h5["atom_ptr"], dtype=np.int64)
                if self.preload:
                    self._features_array = np.asarray(h5["features"], dtype=np.float32)
                if not self.trust_cache:
                    self._ids_array = np.asarray(h5["reaction_ids"].asstr()[()], dtype=object)
                    self._atom_map_order_array = np.asarray(h5["atom_map_order"], dtype=np.int32)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_ids"] = None
        state["_features"] = None
        state["_failed"] = None
        state["_atom_ptr"] = None
        state["_atom_map_order"] = None
        return state

    def _ensure_open(self) -> None:
        if self._features_array is not None:
            return
        if self.path is None or self._h5 is not None:
            return
        import h5py

        self._h5 = h5py.File(self.path, "r")
        if "processed" in self._h5 and not bool(np.asarray(self._h5["processed"][()], dtype=bool).all()):
            done = int(np.asarray(self._h5["processed"][()], dtype=bool).sum())
            raise ValueError(f"incomplete atom feature cache: {self.path} processed={done}/{len(self._h5['processed'])}")
        _failed_count(self._h5, self.path, "atom")
        if not self.trust_cache:
            self._ids = self._h5["reaction_ids"]
        self._features = self._h5["features"]
        self._failed = self._h5.get("failed")
        if self._atom_ptr_array is None:
            self._atom_ptr = self._h5["atom_ptr"]
        if not self.trust_cache and self._atom_map_order_array is None:
            self._atom_map_order = self._h5["atom_map_order"]

    def __len__(self) -> int:
        return self.length

    def get(self, idx: int, reaction_id: str, atom_map_order: list[int]) -> tuple[np.ndarray | None, bool]:
        if self.path is None:
            return None, False
        self._ensure_open()
        if self._atom_ptr_array is not None:
            start = int(self._atom_ptr_array[idx])
            end = int(self._atom_ptr_array[idx + 1])
        else:
            assert self._atom_ptr is not None
            start = int(self._atom_ptr[idx])
            end = int(self._atom_ptr[idx + 1])
        if not self.trust_cache:
            if self._ids_array is not None:
                cached_id = str(self._ids_array[idx])
            else:
                assert self._ids is not None
                cached_id = self._ids.asstr()[idx]
            if cached_id != reaction_id:
                raise ValueError(f"atom cache reaction_id mismatch at {idx}: {cached_id} != {reaction_id}")
            if self._atom_map_order_array is not None:
                cached_maps = self._atom_map_order_array[start:end]
            else:
                assert self._atom_map_order is not None
                cached_maps = np.asarray(self._atom_map_order[start:end], dtype=np.int32)
            expected_maps = np.asarray(atom_map_order, dtype=np.int32)
            if cached_maps.shape != expected_maps.shape or not np.array_equal(cached_maps, expected_maps):
                raise ValueError(f"atom cache atom_map_order mismatch at {idx}: {reaction_id}")
        if self._features_array is not None:
            feature = self._features_array[start:end]
        else:
            assert self._features is not None
            feature = np.asarray(self._features[start:end], dtype=np.float32)
        return feature, False

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._ids = None
        self._features = None
        self._failed = None
        self._atom_ptr = None
        self._atom_map_order = None

    def __del__(self) -> None:
        self.close()


class SuirenFusionPropertyDataset(Dataset):
    """Reaction property tensors plus optional frozen Suiren graph/atom caches."""

    def __init__(self, path: str, limit: int, args: Any):
        self.loaded = load_split(path, limit)
        self.samples = self.loaded.samples
        self.args = args
        trust_cache = bool(getattr(args, "trust_suiren_cache", False))
        preload_graph = bool(getattr(args, "preload_suiren_graph_cache", False))
        preload_atom = bool(getattr(args, "preload_suiren_atom_cache", False))
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
        if not any(store.path is not None for store in [*self.graph_stores.values(), *self.atom_stores.values()]):
            raise ValueError("At least one Suiren cache is required for fusion training")

    @property
    def suiren_graph_dims(self) -> dict[str, int]:
        return {stream: store.dim for stream, store in self.graph_stores.items() if store.path is not None}

    @property
    def suiren_atom_dims(self) -> dict[str, int]:
        return {stream: store.dim for stream, store in self.atom_stores.items() if store.path is not None}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        pair = reaction_property_pair_input(sample, self.args)
        masked_pair = masked_edit_pair_input(sample, idx, self.args)
        reaction_id = str(sample["reaction_id"])
        item = {
            "reaction_id": reaction_id,
            "z": np.asarray(sample["atomic_numbers"], dtype=np.int64),
            "pair_input": pair["pair_input"],
            "pair_valid": pair["pair_valid"],
            "core": pair["core_atom"],
            "changed": pair["changed"],
            "y": target_vector(sample, ENERGY_TARGETS),
            "masked_pair_input": masked_pair["pair_input"],
            "masked_pair_valid": masked_pair["pair_valid"],
            "masked_visibility": masked_pair["visibility"],
            "masked_loss_mask": masked_pair["loss_mask"],
            "masked_delta_bo": masked_pair["delta_bo"],
            "masked_changed": masked_pair["changed"],
            "masked_edit_class": masked_pair["edit_class"],
            "masked_core_atom": masked_pair["core_atom"],
            "n_atoms": len(sample["atomic_numbers"]),
            "core_size": int(pair["core_atom"].sum()),
        }
        for stream, store in self.graph_stores.items():
            features, failed = store.get(idx, reaction_id)
            if features is not None:
                item[f"suiren_{stream}_graph_features"] = features
                item[f"suiren_{stream}_graph_failed"] = failed
        for stream, store in self.atom_stores.items():
            features, failed = store.get(idx, reaction_id, sample["atom_map_order"])
            if features is not None:
                item[f"suiren_{stream}_atom_features"] = features
                item[f"suiren_{stream}_atom_failed"] = failed
        return item


def target_stats(dataset: SuirenFusionPropertyDataset) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(dataset.samples, "target_matrix"):
        y = dataset.samples.target_matrix(ENERGY_TARGETS)  # type: ignore[attr-defined]
    else:
        y = np.stack([target_vector(dataset.samples[idx], ENERGY_TARGETS) for idx in range(len(dataset))], axis=0)
    mean = torch.tensor(y.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(y.std(axis=0), dtype=torch.float32).clamp_min(1e-6)
    return mean, std


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = reaction_property.collate(batch)
    batch_size = len(batch)
    max_n = int(out["z"].shape[1])

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


train_one_epoch = reaction_property.train_one_epoch
evaluate_loss = reaction_property.evaluate_loss
evaluate = reaction_property.evaluate
predict_rows = reaction_property.predict_rows
failure_cases = reaction_property.failure_cases
write_summary = reaction_property.write_summary
