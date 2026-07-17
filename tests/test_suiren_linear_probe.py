from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from rfm.data.hdf5 import STRING_DTYPE
from rfm.tasks.suiren_linear_probe import (
    ProbeSplit,
    build_pooled_atom_cache,
    fit_ridge_probe,
    load_probe_split,
)


def _write_atom_cache(path, reaction_ids, rows):
    atom_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for idx, row in enumerate(rows):
        atom_ptr[idx + 1] = atom_ptr[idx] + len(row)
    features = np.concatenate(rows, axis=0).astype(np.float32)
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps({"feature_layout": ["h_R", "h_P", "Delta_h", "abs_Delta_h"]})
        h5.create_dataset("reaction_ids", data=np.asarray(reaction_ids, dtype=object), dtype=STRING_DTYPE)
        h5.create_dataset("features", data=features)
        h5.create_dataset("atom_ptr", data=atom_ptr)
        h5.create_dataset("failed", data=np.zeros(len(rows), dtype=bool))
        h5.create_dataset("processed", data=np.ones(len(rows), dtype=bool))


def _write_data(path, reaction_ids, targets):
    with h5py.File(path, "w") as h5:
        group = h5.create_group("splits").create_group("train")
        group.create_dataset("reaction_id", data=np.asarray(reaction_ids, dtype=object), dtype=STRING_DTYPE)
        dataset = group.create_dataset("targets", data=np.asarray(targets, dtype=np.float32))
        dataset.attrs["columns"] = json.dumps(["dE", "dE_dagger"])


def test_build_and_load_pooled_atom_cache(tmp_path):
    source = tmp_path / "atom.h5"
    pooled = tmp_path / "pooled.h5"
    data = tmp_path / "data.h5"
    ids = ["r0", "r1"]
    rows = [
        np.asarray(
            [
                [1, 2, 3, 4, 2, 2, 2, 2],
                [3, 4, 5, 6, 2, 2, 2, 2],
            ],
            dtype=np.float32,
        ),
        np.asarray([[2, 6, 4, 8, 2, 2, 2, 2]], dtype=np.float32),
    ]
    _write_atom_cache(source, ids, rows)
    _write_data(data, ids, [[1, 2], [3, 4]])

    metadata = build_pooled_atom_cache(source, pooled, chunk_reactions=1)
    assert metadata["explicit_Delta_h_excluded"] is True
    split = load_probe_split(f"{data}::train", pooled, "state_abs_delta")
    np.testing.assert_allclose(split.features[0], [2, 3, 4, 5, 2, 2])
    np.testing.assert_allclose(split.features[1], [2, 6, 4, 8, 2, 2])
    state = load_probe_split(f"{data}::train", pooled, "state")
    assert state.features.shape == (2, 4)


def test_ridge_probe_recovers_linearly_readable_signal():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(320, 8)).astype(np.float32)
    weight = rng.normal(size=(8, 2)).astype(np.float32)
    y = x @ weight

    def split(start, stop):
        return ProbeSplit(
            reaction_ids=np.asarray([f"r{i}" for i in range(start, stop)]),
            features=x[start:stop].copy(),
            targets=y[start:stop].copy(),
            state_dim=4,
            metadata={},
        )

    result = fit_ridge_probe(
        split(0, 220),
        split(220, 270),
        split(270, 320),
        alphas=[1e-8, 1e-6, 1e-4],
        device=torch.device("cpu"),
    )
    assert result["valid_metrics"]["dE"]["R2"] > 0.999
    assert result["valid_metrics"]["dE_dagger"]["R2"] > 0.999
    assert result["test_metrics"]["dE"]["R2"] > 0.999
    assert result["test_metrics"]["dE_dagger"]["R2"] > 0.999
