"""Data and optimization helpers for the conditional Suiren residual probe."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rfm.data.reaction_samples import ENERGY_TARGETS, coordinates, load_split, reaction_property_pair_input, target_vector
from rfm.tasks.metrics import regression_metrics
from rfm.tasks.suiren_fusion_property import AtomFeatureStore
from rfm.utils.runtime import move_to_device


class SuirenResidualProbeDataset(Dataset):
    """Property inputs plus one atom-map-aligned Suiren 3D atom cache."""

    def __init__(
        self,
        path: str,
        cache_path: str,
        limit: int,
        args: Any,
    ):
        self.loaded = load_split(path, limit)
        self.samples = self.loaded.samples
        self.args = args
        self.atom_store = AtomFeatureStore(
            cache_path,
            trust_cache=bool(getattr(args, "trust_suiren_cache", False)),
            preload=bool(getattr(args, "preload_suiren_atom_cache", False)),
        )
        if self.atom_store.path is None:
            raise ValueError("a Suiren 3D atom cache is required")
        if len(self.atom_store) < len(self.samples) or (not limit and len(self.atom_store) != len(self.samples)):
            raise ValueError(
                f"Suiren atom cache/sample length mismatch: cache={len(self.atom_store)} samples={len(self.samples)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def atom_counts(self) -> np.ndarray:
        if hasattr(self.samples, "n_atoms_array"):
            return self.samples.n_atoms_array()  # type: ignore[attr-defined]
        return np.asarray([len(sample["atomic_numbers"]) for sample in self.samples], dtype=np.int32)

    @property
    def suiren_atom_dim(self) -> int:
        return self.atom_store.dim

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        pair = reaction_property_pair_input(sample, self.args)
        reaction_id = str(sample["reaction_id"])
        features, failed = self.atom_store.get(idx, reaction_id, sample["atom_map_order"])
        if failed or features is None:
            raise ValueError(f"invalid Suiren cache row: {reaction_id}")
        item = {
            "reaction_id": reaction_id,
            "z": np.asarray(sample["atomic_numbers"], dtype=np.int64),
            "pair_input": pair["pair_input"],
            "pair_valid": pair["pair_valid"],
            "y": target_vector(sample, ENERGY_TARGETS),
            "suiren_3d_atom_features": features,
            "n_atoms": len(sample["atomic_numbers"]),
        }
        if self.args.geometry_mode == "irc_rp":
            item["coordinates_R"] = coordinates(sample, "R")
            item["coordinates_P"] = coordinates(sample, "P")
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_n = max(int(item["n_atoms"]) for item in batch)
    pair_dim = int(batch[0]["pair_input"].shape[-1])
    suiren_dim = int(batch[0]["suiren_3d_atom_features"].shape[-1])
    z = torch.zeros((batch_size, max_n), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    pair_valid = torch.zeros((batch_size, max_n, max_n), dtype=torch.bool)
    pair_input = torch.zeros((batch_size, max_n, max_n, pair_dim), dtype=torch.float32)
    suiren = torch.zeros((batch_size, max_n, suiren_dim), dtype=torch.float32)
    y = torch.zeros((batch_size, len(ENERGY_TARGETS)), dtype=torch.float32)
    reaction_ids: list[str] = []
    coordinates_r = torch.zeros((batch_size, max_n, 3), dtype=torch.float32) if "coordinates_R" in batch[0] else None
    coordinates_p = torch.zeros_like(coordinates_r) if coordinates_r is not None else None

    for row, item in enumerate(batch):
        n_atoms = int(item["n_atoms"])
        z[row, :n_atoms] = torch.from_numpy(item["z"])
        atom_mask[row, :n_atoms] = True
        pair_valid[row, :n_atoms, :n_atoms] = torch.from_numpy(item["pair_valid"])
        pair_input[row, :n_atoms, :n_atoms] = torch.from_numpy(item["pair_input"])
        suiren[row, :n_atoms] = torch.from_numpy(item["suiren_3d_atom_features"])
        y[row] = torch.from_numpy(item["y"])
        reaction_ids.append(str(item["reaction_id"]))
        if coordinates_r is not None and coordinates_p is not None:
            coordinates_r[row, :n_atoms] = torch.from_numpy(item["coordinates_R"])
            coordinates_p[row, :n_atoms] = torch.from_numpy(item["coordinates_P"])

    out = {
        "reaction_id": reaction_ids,
        "z": z,
        "atom_mask": atom_mask,
        "pair_valid": pair_valid,
        "pair_input": pair_input,
        "suiren_3d_atom_features": suiren,
        "y": y,
    }
    if coordinates_r is not None and coordinates_p is not None:
        out["coordinates_R"] = coordinates_r
        out["coordinates_P"] = coordinates_p
    return out


def compute_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    prediction_norm = (out["y"] - y_mean) / y_std
    target_norm = (batch["y"] - y_mean) / y_std
    return F.mse_loss(prediction_norm, target_norm)


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    log_every_steps: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    started_at = time.perf_counter()
    log_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = compute_loss(out, batch, y_mean, y_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            grad_clip,
        )
        optimizer.step()
        batch_size = int(batch["y"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        if log_rank and log_every_steps and ((step + 1) % log_every_steps == 0 or step + 1 == len(loader)):
            elapsed = time.perf_counter() - started_at
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": epoch,
                        "step": step + 1,
                        "steps": len(loader),
                        "mean_loss": total_loss / max(total_samples, 1),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": (step + 1) / max(elapsed, 1e-6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {"loss": total_loss / max(total_samples, 1)}


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: Any,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss = compute_loss(out, batch, y_mean, y_std)
        batch_size = int(batch["y"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_base: list[np.ndarray] = []
    residual_norm: list[np.ndarray] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        y_true.append(batch["y"].detach().cpu().numpy())
        y_pred.append(out["y"].detach().cpu().numpy())
        y_base.append(out["baseline_y"].detach().cpu().numpy())
        residual_norm.append(out["residual_norm"].detach().cpu().numpy())
    true = np.concatenate(y_true, axis=0)
    pred = np.concatenate(y_pred, axis=0)
    base = np.concatenate(y_base, axis=0)
    residual = np.concatenate(residual_norm, axis=0)
    return {
        "regression": regression_metrics(true, pred, ENERGY_TARGETS),
        "baseline_regression": regression_metrics(true, base, ENERGY_TARGETS),
        "residual_norm_rms": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
    }


@torch.no_grad()
def predict_rows(model: torch.nn.Module, loader: Any, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        true = batch["y"].detach().cpu().numpy()
        pred = out["y"].detach().cpu().numpy()
        base = out["baseline_y"].detach().cpu().numpy()
        for reaction_id, y_true, y_pred, y_base in zip(batch["reaction_id"], true, pred, base):
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "true_dE": float(y_true[0]),
                    "baseline_dE": float(y_base[0]),
                    "pred_dE": float(y_pred[0]),
                    "true_dE_dagger": float(y_true[1]),
                    "baseline_dE_dagger": float(y_base[1]),
                    "pred_dE_dagger": float(y_pred[1]),
                }
            )
    return rows


def write_summary(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Suiren 条件残差探针",
        "",
        f"模式：`{metrics['mode']}`",
        f"最佳 epoch：{metrics['epoch']}",
        f"最佳 valid loss：{metrics['best_valid_loss']:.6f}",
        "",
        "| split | model | dE MAE | dE_dagger MAE |",
        "|---|---|---:|---:|",
    ]
    for split in ("valid", "test"):
        result = metrics[split]
        lines.append(
            f"| {split} | frozen Stage B | {result['baseline_regression']['dE']['MAE']:.4f} | "
            f"{result['baseline_regression']['dE_dagger']['MAE']:.4f} |"
        )
        lines.append(
            f"| {split} | + residual | {result['regression']['dE']['MAE']:.4f} | "
            f"{result['regression']['dE_dagger']['MAE']:.4f} |"
        )
    output_dir.joinpath("summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
