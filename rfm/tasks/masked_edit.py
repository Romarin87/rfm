"""Masked edit/core pretraining components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rfm.data.reaction_samples import EDIT_CLASSES, load_split, masked_edit_pair_input
from rfm.tasks.metrics import binary_f1, macro_f1, mean_absolute_error, safe_auroc
from rfm.utils.runtime import move_to_device


class MaskedEditDataset(Dataset):
    """Build masked edit tensors from packed HDF5 ReactionDeltaSample rows."""

    def __init__(self, path: str, limit: int, args: Any):
        self.loaded = load_split(path, limit)
        self.samples = self.loaded.samples
        self.args = args

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        pair = masked_edit_pair_input(sample, idx, self.args)
        return {
            "reaction_id": sample["reaction_id"],
            "z": np.asarray(sample["atomic_numbers"], dtype=np.int64),
            "pair_input": pair["pair_input"],
            "pair_valid": pair["pair_valid"],
            "visibility": pair["visibility"],
            "loss_mask": pair["loss_mask"],
            "delta_bo": pair["delta_bo"],
            "changed": pair["changed"],
            "edit_class": pair["edit_class"],
            "core_atom": pair["core_atom"],
            "n_atoms": len(sample["atomic_numbers"]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_n = max(item["n_atoms"] for item in batch)
    pair_dim = batch[0]["pair_input"].shape[-1]
    z = torch.zeros((batch_size, max_n), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    core_atom = torch.zeros((batch_size, max_n), dtype=torch.float32)
    pair_input = torch.zeros((batch_size, max_n, max_n, pair_dim), dtype=torch.float32)
    pair_valid = torch.zeros((batch_size, max_n, max_n), dtype=torch.bool)
    visibility = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    loss_mask = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    delta_bo = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    changed = torch.zeros((batch_size, max_n, max_n), dtype=torch.float32)
    edit_class = torch.zeros((batch_size, max_n, max_n), dtype=torch.long)
    reaction_ids: list[str] = []

    for row, item in enumerate(batch):
        n_atoms = item["n_atoms"]
        z[row, :n_atoms] = torch.from_numpy(item["z"])
        atom_mask[row, :n_atoms] = True
        core_atom[row, :n_atoms] = torch.from_numpy(item["core_atom"])
        pair_input[row, :n_atoms, :n_atoms] = torch.from_numpy(item["pair_input"])
        pair_valid[row, :n_atoms, :n_atoms] = torch.from_numpy(item["pair_valid"])
        visibility[row, :n_atoms, :n_atoms] = torch.from_numpy(item["visibility"])
        loss_mask[row, :n_atoms, :n_atoms] = torch.from_numpy(item["loss_mask"])
        delta_bo[row, :n_atoms, :n_atoms] = torch.from_numpy(item["delta_bo"])
        changed[row, :n_atoms, :n_atoms] = torch.from_numpy(item["changed"])
        edit_class[row, :n_atoms, :n_atoms] = torch.from_numpy(item["edit_class"])
        reaction_ids.append(item["reaction_id"])

    return {
        "reaction_id": reaction_ids,
        "z": z,
        "atom_mask": atom_mask,
        "core_atom": core_atom,
        "pair_input": pair_input,
        "pair_valid": pair_valid,
        "visibility": visibility,
        "loss_mask": loss_mask,
        "delta_bo": delta_bo,
        "changed": changed,
        "edit_class": edit_class,
    }


def target_upper(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.triu((batch["loss_mask"] > 0.5) & batch["pair_valid"], diagonal=1)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def normalized_router_entropy(logits: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Return per-sample softmax entropy normalized to [0, 1]."""

    logits = logits.reshape(logits.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1).bool()
    entropies: list[float] = []
    for sample_logits, sample_mask in zip(logits, mask):
        valid_logits = sample_logits[sample_mask]
        if valid_logits.numel() <= 1:
            continue
        probabilities = torch.softmax(valid_logits.float(), dim=0)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        entropies.append(float((entropy / np.log(valid_logits.numel())).detach().cpu()))
    return entropies


def compute_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: Any,
    edit_class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = target_upper(batch)
    delta_loss = F.smooth_l1_loss(out["delta_bo"][mask], batch["delta_bo"][mask])
    changed_loss = F.binary_cross_entropy_with_logits(out["changed_logits"][mask], batch["changed"][mask])
    edit_loss = F.cross_entropy(out["edit_logits"][mask], batch["edit_class"][mask], weight=edit_class_weights)
    core_loss = F.binary_cross_entropy_with_logits(out["core_logits"][batch["atom_mask"]], batch["core_atom"][batch["atom_mask"]])
    loss = args.delta_bo_weight * delta_loss + args.changed_weight * changed_loss + args.edit_weight * edit_loss + args.core_weight * core_loss
    return loss, {
        "delta_bo_loss": float(delta_loss.detach().cpu()),
        "changed_loss": float(changed_loss.detach().cpu()),
        "edit_loss": float(edit_loss.detach().cpu()),
        "core_loss": float(core_loss.detach().cpu()),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: Any,
    edit_class_weights: torch.Tensor,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "delta_bo_loss": 0.0, "changed_loss": 0.0, "edit_loss": 0.0, "core_loss": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(out, batch, args, edit_class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_size = batch["z"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        totals["n"] += batch_size
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    args: Any,
    edit_class_weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "delta_bo_loss": 0.0, "changed_loss": 0.0, "edit_loss": 0.0, "core_loss": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(out, batch, args, edit_class_weights)
        batch_size = batch["z"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        totals["n"] += batch_size
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Any, device: torch.device, max_prediction_rows: int = 20000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    changed_labels, changed_scores = [], []
    edit_labels, edit_preds = [], []
    delta_true, delta_pred = [], []
    core_labels, core_scores = [], []
    router_pair_labels, router_pair_scores = [], []
    router_atom_labels, router_atom_scores = [], []
    router_pair_entropies, router_atom_entropies = [], []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        mask = target_upper(batch)
        changed_labels.append(batch["changed"][mask].detach().cpu().numpy())
        changed_scores.append(out["changed_logits"][mask].detach().cpu().numpy())
        edit_labels.append(batch["edit_class"][mask].detach().cpu().numpy())
        edit_preds.append(out["edit_logits"][mask].argmax(dim=-1).detach().cpu().numpy())
        delta_true.append(batch["delta_bo"][mask].detach().cpu().numpy())
        delta_pred.append(out["delta_bo"][mask].detach().cpu().numpy())
        core_labels.append(batch["core_atom"][batch["atom_mask"]].detach().cpu().numpy())
        core_scores.append(out["core_logits"][batch["atom_mask"]].detach().cpu().numpy())
        if "router_pair_logits" in out and "router_atom_logits" in out:
            router_pair_mask = torch.triu(batch["pair_valid"], diagonal=1)
            router_pair_labels.append(batch["changed"][router_pair_mask].detach().cpu().numpy())
            router_pair_scores.append(out["router_pair_logits"][router_pair_mask].detach().cpu().numpy())
            router_atom_labels.append(batch["core_atom"][batch["atom_mask"]].detach().cpu().numpy())
            router_atom_scores.append(out["router_atom_logits"][batch["atom_mask"]].detach().cpu().numpy())
            router_pair_entropies.extend(normalized_router_entropy(out["router_pair_logits"], batch["pair_valid"]))
            router_atom_entropies.extend(normalized_router_entropy(out["router_atom_logits"], batch["atom_mask"]))
        if len(rows) < max_prediction_rows:
            probs = torch.sigmoid(out["changed_logits"]).detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            for item_idx, reaction_id in enumerate(batch["reaction_id"]):
                coords = np.argwhere(np.triu(mask_np[item_idx], k=1))
                for i, j in coords[:32]:
                    rows.append(
                        {
                            "reaction_id": reaction_id,
                            "i": int(i),
                            "j": int(j),
                            "changed_true": float(batch["changed"][item_idx, i, j].detach().cpu()),
                            "changed_score": float(probs[item_idx, i, j]),
                            "edit_true": EDIT_CLASSES[int(batch["edit_class"][item_idx, i, j].detach().cpu())],
                            "edit_pred": EDIT_CLASSES[int(out["edit_logits"][item_idx, i, j].argmax().detach().cpu())],
                            "delta_bo_true": float(batch["delta_bo"][item_idx, i, j].detach().cpu()),
                            "delta_bo_pred": float(out["delta_bo"][item_idx, i, j].detach().cpu()),
                        }
                    )
                    if len(rows) >= max_prediction_rows:
                        break
                if len(rows) >= max_prediction_rows:
                    break

    y = np.concatenate(changed_labels)
    scores = np.concatenate(changed_scores)
    edit_y = np.concatenate(edit_labels)
    edit_p = np.concatenate(edit_preds)
    dbo_y = np.concatenate(delta_true)
    dbo_p = np.concatenate(delta_pred)
    core_y = np.concatenate(core_labels)
    core_s = np.concatenate(core_scores)
    metrics = {
        "target_pairs": int(y.shape[0]),
        "changed_positive_rate": float(y.mean()) if y.size else 0.0,
        "changed_pair_AUROC": safe_auroc(y, scores),
        "changed_pair_F1": binary_f1(y.astype(int), (scores >= 0.0).astype(int)),
        "edit_macro_F1": macro_f1(edit_y, edit_p, n_classes=len(EDIT_CLASSES)),
        "formed_broken_macro_F1": binary_f1(np.isin(edit_y, [1, 2]).astype(int), np.isin(edit_p, [1, 2]).astype(int)),
        "delta_BO_MAE": mean_absolute_error(dbo_y, dbo_p),
        "core_atom_AUROC": safe_auroc(core_y, core_s),
        "core_atom_F1": binary_f1(core_y.astype(int), (core_s >= 0.0).astype(int)),
    }
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
    return metrics, rows


def write_summary(output_dir: Path, best: dict[str, Any]) -> None:
    lines = [
        "# RFM Masked Edit/Core Pretraining",
        "",
        f"Best epoch: {best['epoch']}",
        "Selection: minimum valid_loss",
        "",
        "| Split | Target pairs | Changed AUROC | Changed F1 | Edit macro-F1 | Formed/Broken F1 | Delta BO MAE | Core AUROC | Core F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        metric = best[split]
        lines.append(
            f"| {split} | {metric['target_pairs']} | {metric['changed_pair_AUROC']:.4f} | {metric['changed_pair_F1']:.4f} | "
            f"{metric['edit_macro_F1']:.4f} | {metric['formed_broken_macro_F1']:.4f} | {metric['delta_BO_MAE']:.4f} | "
            f"{metric['core_atom_AUROC']:.4f} | {metric['core_atom_F1']:.4f} |"
        )
    output_dir.joinpath("summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
