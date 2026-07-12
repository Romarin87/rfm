"""Reaction property regression components."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rfm.data.reaction_samples import EDIT_CLASSES, ENERGY_TARGETS, load_split, masked_edit_pair_input, reaction_property_pair_input, target_vector
from rfm.tasks import masked_edit
from rfm.tasks.metrics import binary_f1, macro_f1, mean_absolute_error, regression_metrics, safe_auroc
from rfm.utils.runtime import move_to_device


class ReactionPropertyDataset(Dataset):
    """Build reaction property regression tensors from packed HDF5 rows."""

    def __init__(self, path: str, limit: int, args: Any):
        self.loaded = load_split(path, limit)
        self.samples = self.loaded.samples
        self.args = args

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def atom_counts(self) -> np.ndarray:
        if hasattr(self.samples, "n_atoms_array"):
            return self.samples.n_atoms_array()  # type: ignore[attr-defined]
        return np.asarray([len(sample["atomic_numbers"]) for sample in self.samples], dtype=np.int32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        pair = reaction_property_pair_input(sample, self.args)
        masked_pair = masked_edit_pair_input(sample, idx, self.args)
        core_size = int(pair["core_atom"].sum())
        return {
            "reaction_id": sample["reaction_id"],
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
            "core_size": core_size,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_n = max(item["n_atoms"] for item in batch)
    pair_dim = batch[0]["pair_input"].shape[-1]
    masked_pair_dim = batch[0]["masked_pair_input"].shape[-1]
    z = torch.zeros((batch_size, max_n), dtype=torch.long)
    pair_input = torch.zeros((batch_size, max_n, max_n, pair_dim), dtype=torch.float32)
    masked_pair_input = torch.zeros((batch_size, max_n, max_n, masked_pair_dim), dtype=torch.float32)
    atom_mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    pair_valid = torch.zeros((batch_size, max_n, max_n), dtype=torch.bool)
    masked_pair_valid = torch.zeros((batch_size, max_n, max_n), dtype=torch.bool)
    masked_visibility = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    masked_loss_mask = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    masked_delta_bo = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    masked_changed = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    masked_edit_class = torch.zeros((batch_size, max_n, max_n), dtype=torch.long)
    masked_core_atom = torch.zeros((batch_size, max_n), dtype=torch.float32)
    core = torch.zeros((batch_size, max_n), dtype=torch.float32)
    changed = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    y = torch.zeros((batch_size, len(ENERGY_TARGETS)), dtype=torch.float32)
    n_atoms: list[int] = []
    core_size: list[int] = []
    reaction_ids: list[str] = []

    for row, item in enumerate(batch):
        n = item["n_atoms"]
        z[row, :n] = torch.from_numpy(item["z"])
        pair_input[row, :n, :n] = torch.from_numpy(item["pair_input"])
        masked_pair_input[row, :n, :n] = torch.from_numpy(item["masked_pair_input"])
        atom_mask[row, :n] = True
        pair_valid[row, :n, :n] = torch.from_numpy(item["pair_valid"])
        masked_pair_valid[row, :n, :n] = torch.from_numpy(item["masked_pair_valid"])
        masked_visibility[row, :n, :n] = torch.from_numpy(item["masked_visibility"])
        masked_loss_mask[row, :n, :n] = torch.from_numpy(item["masked_loss_mask"])
        masked_delta_bo[row, :n, :n] = torch.from_numpy(item["masked_delta_bo"])
        masked_changed[row, :n, :n] = torch.from_numpy(item["masked_changed"])
        masked_edit_class[row, :n, :n] = torch.from_numpy(item["masked_edit_class"])
        masked_core_atom[row, :n] = torch.from_numpy(item["masked_core_atom"])
        core[row, :n] = torch.from_numpy(item["core"])
        changed[row, :n, :n] = torch.from_numpy(item["changed"])
        y[row] = torch.from_numpy(item["y"])
        n_atoms.append(n)
        core_size.append(item["core_size"])
        reaction_ids.append(item["reaction_id"])

    return {
        "reaction_id": reaction_ids,
        "z": z,
        "pair_input": pair_input,
        "masked_pair_input": masked_pair_input,
        "atom_mask": atom_mask,
        "pair_valid": pair_valid,
        "masked_pair_valid": masked_pair_valid,
        "masked_visibility": masked_visibility,
        "masked_loss_mask": masked_loss_mask,
        "masked_delta_bo": masked_delta_bo,
        "masked_changed": masked_changed,
        "masked_edit_class": masked_edit_class,
        "masked_core_atom": masked_core_atom,
        "core": core,
        "changed": changed,
        "y": y,
        "n_atoms": torch.tensor(n_atoms, dtype=torch.long),
        "core_size": torch.tensor(core_size, dtype=torch.long),
    }


def target_stats(dataset: ReactionPropertyDataset) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(dataset.samples, "target_matrix"):
        y = dataset.samples.target_matrix(ENERGY_TARGETS)  # type: ignore[attr-defined]
    else:
        y = np.stack([target_vector(dataset.samples[idx], ENERGY_TARGETS) for idx in range(len(dataset))], axis=0)
    mean = torch.tensor(y.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(y.std(axis=0), dtype=torch.float32).clamp_min(1e-6)
    return mean, std


def masked_batch_view(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "atom_mask": batch["atom_mask"],
        "pair_valid": batch["masked_pair_valid"],
        "loss_mask": batch["masked_loss_mask"],
        "delta_bo": batch["masked_delta_bo"],
        "changed": batch["masked_changed"],
        "edit_class": batch["masked_edit_class"],
        "core_atom": batch["masked_core_atom"],
    }


def masked_out_view(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    masked_out = {
        "delta_bo": out["masked_delta_bo"],
        "changed_logits": out["masked_changed_logits"],
        "edit_logits": out["masked_edit_logits"],
        "core_logits": out["masked_core_logits"],
    }
    for key in ("router_pair_logits", "router_atom_logits"):
        prefixed_key = f"masked_{key}"
        if prefixed_key in out:
            masked_out[key] = out[prefixed_key]
    if "masked_mrto_event_pair_weights" in out:
        masked_out["mrto_event_pair_weights"] = out["masked_mrto_event_pair_weights"]
    return masked_out


def edit_class_weights(args: Any, device: torch.device) -> torch.Tensor:
    values = masked_edit.parse_float_list(getattr(args, "edit_class_weights", "1.0,4.0,4.0,2.0"))
    if len(values) != len(EDIT_CLASSES):
        raise ValueError(f"--edit-class-weights must have {len(EDIT_CLASSES)} values")
    return torch.tensor(values, dtype=torch.float32, device=device)


def property_loss_weight(args: Any) -> float:
    if hasattr(args, "property_weight"):
        return float(args.property_weight)
    if hasattr(args, "stage_c_weight"):
        return float(args.stage_c_weight)
    return 1.0


def compute_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    args: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_norm = (batch["y"] - y_mean) / y_std
    property_loss = F.mse_loss(out["y"], y_norm)
    stage_a_loss, stage_a_parts = masked_edit.compute_loss(masked_out_view(out), masked_batch_view(batch), args, edit_class_weights(args, batch["y"].device))
    stage_a_weight = float(getattr(args, "stage_a_weight", 0.5))
    prop_weight = property_loss_weight(args)
    consistency_loss = forward_reverse_consistency_loss(out["y"], batch.get("reaction_id", []), y_mean, y_std)
    consistency_weight = float(getattr(args, "forward_reverse_consistency_weight", 0.0))
    loss = stage_a_weight * stage_a_loss + prop_weight * property_loss + consistency_weight * consistency_loss
    return loss, {
        "property_loss": float(property_loss.detach().cpu()),
        "stage_a_loss": float(stage_a_loss.detach().cpu()),
        "consistency_loss": float(consistency_loss.detach().cpu()),
        **stage_a_parts,
    }


def forward_reverse_consistency_loss(
    y_pred_norm: torch.Tensor,
    reaction_ids: list[str],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    if not reaction_ids:
        return y_pred_norm.new_zeros(())
    index = {str(reaction_id): idx for idx, reaction_id in enumerate(reaction_ids)}
    terms: list[torch.Tensor] = []
    y_pred_raw = y_pred_norm * y_std + y_mean
    for reaction_id, forward_idx in index.items():
        if reaction_id.endswith("__reverse"):
            continue
        reverse_idx = index.get(f"{reaction_id}__reverse")
        if reverse_idx is None:
            continue
        forward_raw = y_pred_raw[forward_idx]
        expected_reverse_raw = torch.stack(
            [
                -forward_raw[0],
                forward_raw[1] - forward_raw[0],
            ]
        )
        expected_reverse_norm = (expected_reverse_raw - y_mean) / y_std
        terms.append(F.mse_loss(y_pred_norm[reverse_idx], expected_reverse_norm))
    if not terms:
        return y_pred_norm.new_zeros(())
    return torch.stack(terms).mean()


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    args: Any,
    device: torch.device,
    epoch: int = -1,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "property_loss": 0.0,
        "stage_a_loss": 0.0,
        "consistency_loss": 0.0,
        "delta_bo_loss": 0.0,
        "changed_loss": 0.0,
        "edit_loss": 0.0,
        "core_loss": 0.0,
        "event_set_loss": 0.0,
        "event_diversity_loss": 0.0,
        "n": 0.0,
    }
    accumulation_steps = max(int(getattr(args, "gradient_accumulation_steps", 1)), 1)
    log_every_steps = max(int(getattr(args, "log_every_steps", 0)), 0)
    log_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    started_at = time.perf_counter()
    final_group = len(loader) % accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        should_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader)
        group_size = (
            final_group
            if final_group and step >= len(loader) - final_group
            else accumulation_steps
        )
        sync_context = model.no_sync() if hasattr(model, "no_sync") and not should_step else nullcontext()
        with sync_context:
            out = model(batch)
            loss, parts = compute_loss(out, batch, y_mean, y_std, args)
            (loss / group_size).backward()
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = batch["y"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        totals["n"] += batch_size
        if log_rank and log_every_steps and ((step + 1) % log_every_steps == 0 or (step + 1) == len(loader)):
            elapsed = time.perf_counter() - started_at
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": int(epoch),
                        "step": step + 1,
                        "steps": len(loader),
                        "samples": int(totals["n"]),
                        "mean_loss": totals["loss"] / max(totals["n"], 1.0),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": (step + 1) / max(elapsed, 1e-6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: Any, y_mean: torch.Tensor, y_std: torch.Tensor, args: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "property_loss": 0.0,
        "stage_a_loss": 0.0,
        "consistency_loss": 0.0,
        "delta_bo_loss": 0.0,
        "changed_loss": 0.0,
        "edit_loss": 0.0,
        "core_loss": 0.0,
        "event_set_loss": 0.0,
        "event_diversity_loss": 0.0,
        "n": 0.0,
    }
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(out, batch, y_mean, y_std, args)
        batch_size = batch["y"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        totals["n"] += batch_size
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Any, y_mean: torch.Tensor, y_std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true, y_pred = [], []
    core_labels, core_scores = [], []
    changed_labels, changed_scores = [], []
    edit_labels, edit_preds = [], []
    delta_true, delta_pred = [], []
    router_pair_labels, router_pair_scores = [], []
    router_atom_labels, router_atom_scores = [], []
    router_pair_entropies, router_atom_entropies = [], []
    n_atoms, core_size = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        pred = out["y"] * y_std + y_mean
        y_true.append(batch["y"].detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        masked_batch = masked_batch_view(batch)
        masked_out = masked_out_view(out)
        target_mask = masked_edit.target_upper(masked_batch)
        changed_labels.append(masked_batch["changed"][target_mask].detach().cpu().numpy())
        changed_scores.append(masked_out["changed_logits"][target_mask].detach().cpu().numpy())
        edit_labels.append(masked_batch["edit_class"][target_mask].detach().cpu().numpy())
        edit_preds.append(masked_out["edit_logits"][target_mask].argmax(dim=-1).detach().cpu().numpy())
        delta_true.append(masked_batch["delta_bo"][target_mask].detach().cpu().numpy())
        delta_pred.append(masked_out["delta_bo"][target_mask].detach().cpu().numpy())
        core_labels.append(masked_batch["core_atom"][batch["atom_mask"]].detach().cpu().numpy())
        core_scores.append(masked_out["core_logits"][batch["atom_mask"]].detach().cpu().numpy())
        if "router_pair_logits" in masked_out and "router_atom_logits" in masked_out:
            router_pair_mask = torch.triu(masked_batch["pair_valid"], diagonal=1)
            router_pair_labels.append(masked_batch["changed"][router_pair_mask].detach().cpu().numpy())
            router_pair_scores.append(masked_out["router_pair_logits"][router_pair_mask].detach().cpu().numpy())
            router_atom_labels.append(masked_batch["core_atom"][batch["atom_mask"]].detach().cpu().numpy())
            router_atom_scores.append(masked_out["router_atom_logits"][batch["atom_mask"]].detach().cpu().numpy())
            router_pair_entropies.extend(masked_edit.normalized_router_entropy(masked_out["router_pair_logits"], masked_batch["pair_valid"]))
            router_atom_entropies.extend(masked_edit.normalized_router_entropy(masked_out["router_atom_logits"], batch["atom_mask"]))
        n_atoms.extend(batch["n_atoms"].detach().cpu().numpy().tolist())
        core_size.extend(batch["core_size"].detach().cpu().numpy().tolist())

    y_true_arr = np.concatenate(y_true, axis=0)
    y_pred_arr = np.concatenate(y_pred, axis=0)
    metrics = {"regression": regression_metrics(y_true_arr, y_pred_arr, ENERGY_TARGETS)}
    changed_y = np.concatenate(changed_labels)
    changed_s = np.concatenate(changed_scores)
    edit_y = np.concatenate(edit_labels)
    edit_p = np.concatenate(edit_preds)
    dbo_y = np.concatenate(delta_true)
    dbo_p = np.concatenate(delta_pred)
    core_y = np.concatenate(core_labels)
    core_s = np.concatenate(core_scores)
    metrics.update(
        {
            "target_pairs": int(changed_y.shape[0]),
            "changed_positive_rate": float(changed_y.mean()) if changed_y.size else 0.0,
            "changed_pair_AUROC": safe_auroc(changed_y, changed_s),
            "changed_pair_F1": binary_f1(changed_y.astype(int), (changed_s >= 0.0).astype(int)),
            "edit_macro_F1": macro_f1(edit_y, edit_p, n_classes=len(EDIT_CLASSES)),
            "formed_broken_macro_F1": binary_f1(np.isin(edit_y, [1, 2]).astype(int), np.isin(edit_p, [1, 2]).astype(int)),
            "delta_BO_MAE": mean_absolute_error(dbo_y, dbo_p),
            "core_atom_AUROC": safe_auroc(core_y, core_s),
            "core_atom_F1": binary_f1(core_y.astype(int), (core_s >= 0.0).astype(int)),
        }
    )
    if router_pair_scores:
        router_pair_y = np.concatenate(router_pair_labels)
        router_pair_s = np.concatenate(router_pair_scores)
        router_atom_y = np.concatenate(router_atom_labels)
        router_atom_s = np.concatenate(router_atom_scores)
        metrics.update(
            {
                "router_pair_AUROC": safe_auroc(router_pair_y, router_pair_s),
                "router_atom_AUROC": safe_auroc(router_atom_y, router_atom_s),
                "router_pair_normalized_entropy": float(np.mean(router_pair_entropies)),
                "router_atom_normalized_entropy": float(np.mean(router_atom_entropies)),
            }
        )
    metrics["n_atoms_mean"] = float(np.mean(n_atoms))
    metrics["core_size_mean"] = float(np.mean(core_size))
    return metrics


@torch.no_grad()
def predict_rows(model: torch.nn.Module, loader: Any, y_mean: torch.Tensor, y_std: torch.Tensor, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        pred = (out["y"] * y_std + y_mean).detach().cpu().numpy()
        true = batch["y"].detach().cpu().numpy()
        for reaction_id, y_true, y_hat in zip(batch["reaction_id"], true, pred):
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "true_dE": float(y_true[0]),
                    "pred_dE": float(y_hat[0]),
                    "abs_error_dE": float(abs(y_true[0] - y_hat[0])),
                    "true_dE_dagger": float(y_true[1]),
                    "pred_dE_dagger": float(y_hat[1]),
                    "abs_error_dE_dagger": float(abs(y_true[1] - y_hat[1])),
                }
            )
    return rows


def failure_cases(rows: list[dict[str, Any]], top_k: int = 100) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["abs_error_dE_dagger"], reverse=True)[:top_k]
    return [
        {
            "rank": rank,
            "reaction_id": row["reaction_id"],
            "abs_error_dE_dagger": row["abs_error_dE_dagger"],
            "true_dE_dagger": row["true_dE_dagger"],
            "pred_dE_dagger": row["pred_dE_dagger"],
            "abs_error_dE": row["abs_error_dE"],
            "true_dE": row["true_dE"],
            "pred_dE": row["pred_dE"],
        }
        for rank, row in enumerate(ordered, 1)
    ]


def write_summary(output_dir: Path, best: dict[str, Any]) -> None:
    lines = [
        "# RFM Reaction Property Regression",
        "",
        f"Best epoch: {best['epoch']}",
        "Selection: minimum valid_loss",
        f"Pretrained encoder: {best.get('pretrained_encoder') or 'none'}",
        "",
        "| Split | dE MAE | dE R2 | dE_dagger MAE | dE_dagger R2 | dE_dagger Spearman | Changed AUROC | Edit macro-F1 | Delta BO MAE | Core F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        metric = best[split]
        lines.append(
            f"| {split} | {metric['regression']['dE']['MAE']:.4f} | {metric['regression']['dE']['R2']:.4f} | "
            f"{metric['regression']['dE_dagger']['MAE']:.4f} | {metric['regression']['dE_dagger']['R2']:.4f} | "
            f"{metric['regression']['dE_dagger']['Spearman']:.4f} | {metric['changed_pair_AUROC']:.4f} | "
            f"{metric['edit_macro_F1']:.4f} | {metric['delta_BO_MAE']:.4f} | {metric['core_atom_F1']:.4f} |"
        )
    output_dir.joinpath("summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
