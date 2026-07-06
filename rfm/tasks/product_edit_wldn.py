"""WLDN-style R-only product prediction.

The task keeps the strict R-only input contract from :mod:`rfm.tasks.product_edit`.
It replaces full-matrix pair argmax decoding with:

1. reaction-center proposal over atom pairs,
2. legal sparse Delta_BO candidate generation,
3. difference-graph candidate ranking.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rfm.tasks import product_edit
from rfm.tasks.metrics import binary_f1, macro_f1, mean_absolute_error
from rfm.utils.runtime import move_to_device

ProductEditDataset = product_edit.ProductEditDataset
collate = product_edit.collate
target_upper = product_edit.target_upper
class_weights = product_edit.class_weights
no_change_index = product_edit.no_change_index


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _parse_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _log_sigmoid(value: float) -> float:
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _log_softmax_np(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    shifted = values.astype(np.float64) - max_value
    return shifted - math.log(float(np.exp(shifted).sum()))


def _candidate_signature(matrix: np.ndarray, pair_mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix[pair_mask]).tobytes()


def _legal_classes(bo_value: float, delta_vocab: np.ndarray, no_change: int) -> list[int]:
    out: list[int] = []
    for idx, delta in enumerate(delta_vocab):
        if idx == no_change:
            continue
        bo_p = bo_value + float(delta)
        if -1e-5 <= bo_p <= 3.5 + 1e-5:
            out.append(idx)
    return out


def _enumerate_row_candidates(
    change_logits: np.ndarray,
    delta_logits: np.ndarray,
    bo_r: np.ndarray,
    pair_mask: np.ndarray,
    delta_vocab: np.ndarray,
    no_change: int,
    center_top_m: int,
    edit_class_top_k: int,
    beam_size: int,
    max_candidates: int,
) -> list[np.ndarray]:
    n_atoms = pair_mask.shape[0]
    base = np.full((n_atoms, n_atoms), no_change, dtype=np.int64)
    pair_indices = np.argwhere(pair_mask)
    if pair_indices.size == 0:
        return [base]

    scores = change_logits[pair_mask]
    order = np.argsort(scores)[::-1][: min(center_top_m, len(scores))]
    selected_pairs = [(int(pair_indices[pos, 0]), int(pair_indices[pos, 1])) for pos in order]

    beam: list[tuple[float, dict[tuple[int, int], int]]] = [(0.0, {})]
    for i, j in selected_pairs:
        row_logits = delta_logits[i, j]
        log_probs = _log_softmax_np(row_logits)
        legal = _legal_classes(float(bo_r[i, j]), delta_vocab, no_change)
        legal = sorted(legal, key=lambda cls: float(log_probs[cls]), reverse=True)[:edit_class_top_k]
        options: list[tuple[int, float]] = [(no_change, _log_sigmoid(-float(change_logits[i, j])))]
        changed_log_prob = _log_sigmoid(float(change_logits[i, j]))
        options.extend((cls, changed_log_prob + float(log_probs[cls])) for cls in legal)

        next_beam: list[tuple[float, dict[tuple[int, int], int]]] = []
        for score, edits in beam:
            for cls, option_score in options:
                new_edits = dict(edits)
                if cls == no_change:
                    new_edits.pop((i, j), None)
                else:
                    new_edits[(i, j)] = int(cls)
                next_beam.append((score + option_score, new_edits))
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:beam_size]

    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _, edits in beam:
        matrix = base.copy()
        for (i, j), cls in edits.items():
            matrix[i, j] = cls
            matrix[j, i] = cls
        signature = _candidate_signature(matrix, pair_mask)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(matrix)
        if len(candidates) >= max_candidates:
            break
    return candidates or [base]


@torch.no_grad()
def generate_candidate_classes(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    delta_vocab: np.ndarray,
    args: Any,
    include_target: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate candidate Delta_BO class matrices from proposal logits.

    If ``include_target`` is true, the row-specific ground-truth product is
    inserted at candidate index 0 for ranker training.
    """

    device = out["change_logits"].device
    no_change = no_change_index(delta_vocab)
    center_top_m = int(getattr(args, "center_top_m", 6))
    edit_class_top_k = int(getattr(args, "edit_class_top_k", 2))
    beam_size = int(getattr(args, "candidate_beam_size", 64))
    pool_size = int(getattr(args, "candidate_pool_size", max(getattr(args, "top_k", 5), 16)))
    if include_target:
        max_candidates = pool_size + 1
        generated_limit = pool_size
    else:
        max_candidates = pool_size
        generated_limit = pool_size

    change_np = out["change_logits"].detach().cpu().numpy()
    delta_np = out["delta_logits"].detach().cpu().numpy()
    bo_r_np = batch["bo_r"].detach().cpu().numpy()
    pair_valid_np = batch["pair_valid"].detach().cpu().numpy().astype(bool)
    target_np = batch["target_class"].detach().cpu().numpy()
    batch_size, n_atoms, _ = pair_valid_np.shape

    candidate = torch.full((batch_size, max_candidates, n_atoms, n_atoms), no_change, dtype=torch.long, device=device)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)

    for row in range(batch_size):
        pair_mask = np.triu(pair_valid_np[row], k=1)
        write_idx = 0
        seen: set[bytes] = set()
        if include_target:
            target = target_np[row].astype(np.int64)
            candidate[row, 0] = torch.as_tensor(target, dtype=torch.long, device=device)
            candidate_mask[row, 0] = True
            seen.add(_candidate_signature(target, pair_mask))
            write_idx = 1

        generated = _enumerate_row_candidates(
            change_np[row],
            delta_np[row],
            bo_r_np[row],
            pair_mask,
            delta_vocab,
            no_change,
            center_top_m=center_top_m,
            edit_class_top_k=edit_class_top_k,
            beam_size=beam_size,
            max_candidates=generated_limit,
        )
        for matrix in generated:
            if write_idx >= max_candidates:
                break
            signature = _candidate_signature(matrix, pair_mask)
            if signature in seen:
                continue
            seen.add(signature)
            candidate[row, write_idx] = torch.as_tensor(matrix, dtype=torch.long, device=device)
            candidate_mask[row, write_idx] = True
            write_idx += 1
    return candidate, candidate_mask


def compute_loss(
    model: torch.nn.Module,
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: Any,
    delta_vocab: np.ndarray,
    delta_class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    no_change = no_change_index(delta_vocab)
    pair_mask = target_upper(batch)
    target_class = batch["target_class"].long()
    target_changed = (target_class != no_change) & pair_mask

    pos_weight = torch.tensor(float(getattr(args, "changed_pair_pos_weight", 8.0)), dtype=torch.float32, device=out["change_logits"].device)
    center_loss_raw = F.binary_cross_entropy_with_logits(
        out["change_logits"],
        target_changed.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    center_loss = (center_loss_raw * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)

    if bool(target_changed.any().item()):
        delta_loss = F.cross_entropy(out["delta_logits"][target_changed], target_class[target_changed], weight=delta_class_weights)
    else:
        delta_loss = out["delta_logits"].sum() * 0.0

    candidate_class, candidate_mask = generate_candidate_classes(out, batch, delta_vocab, args, include_target=True)
    delta_tensor = torch.as_tensor(delta_vocab, dtype=torch.float32, device=out["change_logits"].device)
    candidate_delta = delta_tensor[candidate_class]
    scores = _raw_model(model).score_candidates(out, batch, candidate_delta)
    scores = scores.masked_fill(~candidate_mask, -1.0e9)
    rank_target = torch.zeros((scores.shape[0],), dtype=torch.long, device=scores.device)
    rank_loss = F.cross_entropy(scores, rank_target)

    center_weight = _parse_float(getattr(args, "center_loss_weight", None), 1.0)
    delta_weight = _parse_float(getattr(args, "delta_loss_weight", None), 1.0)
    rank_weight = _parse_float(getattr(args, "rank_loss_weight", None), 1.0)
    loss = center_weight * center_loss + delta_weight * delta_loss + rank_weight * rank_loss
    return loss, {
        "center_loss": float(center_loss.detach().cpu()),
        "delta_loss": float(delta_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "product_edit_loss": float(loss.detach().cpu()),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: Any,
    delta_vocab: np.ndarray,
    delta_class_weights: torch.Tensor,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "center_loss": 0.0, "delta_loss": 0.0, "rank_loss": 0.0, "product_edit_loss": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(model, out, batch, args, delta_vocab, delta_class_weights)
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
    delta_vocab: np.ndarray,
    delta_class_weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "center_loss": 0.0, "delta_loss": 0.0, "rank_loss": 0.0, "product_edit_loss": 0.0, "n": 0.0}
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch)
        loss, parts = compute_loss(model, out, batch, args, delta_vocab, delta_class_weights)
        batch_size = batch["z"].shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        totals["n"] += batch_size
    n = max(totals.pop("n"), 1.0)
    return {key: value / n for key, value in totals.items()}


def _rank_candidates(
    model: torch.nn.Module,
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    candidate_class: torch.Tensor,
    candidate_mask: torch.Tensor,
    delta_vocab: np.ndarray,
) -> torch.Tensor:
    delta_tensor = torch.as_tensor(delta_vocab, dtype=torch.float32, device=out["change_logits"].device)
    scores = _raw_model(model).score_candidates(out, batch, delta_tensor[candidate_class])
    return scores.masked_fill(~candidate_mask, -1.0e9).argsort(dim=1, descending=True)


def _center_coverage(change_logits: np.ndarray, target_class: np.ndarray, pair_mask: np.ndarray, no_change: int, top_m: int) -> bool:
    true_pos = set(tuple(map(int, row)) for row in np.argwhere(pair_mask & (target_class != no_change)))
    if not true_pos:
        return True
    pair_indices = np.argwhere(pair_mask)
    if pair_indices.size == 0:
        return False
    scores = change_logits[pair_mask]
    order = np.argsort(scores)[::-1][: min(top_m, len(scores))]
    pred_pos = set((int(pair_indices[pos, 0]), int(pair_indices[pos, 1])) for pos in order)
    return true_pos.issubset(pred_pos)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    delta_vocab: np.ndarray,
    args: Any,
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
    center_coverage = 0
    candidate_pool_sizes: list[int] = []
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
        candidate_class, candidate_mask = generate_candidate_classes(out, batch, delta_vocab, args, include_target=False)
        rank_order = _rank_candidates(model, out, batch, candidate_class, candidate_mask, delta_vocab)
        ranked = torch.gather(candidate_class, 1, rank_order[:, :, None, None].expand_as(candidate_class)).detach().cpu().numpy()
        candidate_mask_np = torch.gather(candidate_mask, 1, rank_order).detach().cpu().numpy().astype(bool)
        change_np = out["change_logits"].detach().cpu().numpy()
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

            center_coverage += int(_center_coverage(change_np[row_idx, :n_atoms, :n_atoms], row_target, pair_mask, no_change, int(getattr(args, "center_top_m", 6))))
            for k in topks:
                exact_sample[k] += int(product_edit._candidate_exact(candidates, row_target[None, :, :], pair_mask, k))
                exact_group[k] += int(product_edit._candidate_exact(candidates, active_group_targets, pair_mask, k))

            pred_delta_all = delta_vocab[candidates]
            valid_top1 += int(product_edit._valid_product(bo_r[row_idx, :n_atoms, :n_atoms], pred_delta_all[0], pair_mask))
            for cand_idx in range(min(5, pred_delta_all.shape[0])):
                valid_topk += int(product_edit._valid_product(bo_r[row_idx, :n_atoms, :n_atoms], pred_delta_all[cand_idx], pair_mask))
                valid_topk_total += 1

            pair_target = row_target[pair_mask]
            pair_errors = [(candidates[cand_idx][pair_mask] != pair_target).mean() for cand_idx in range(candidates.shape[0])]
            best_idx = int(np.argmin(pair_errors))
            top1_delta = pred_delta_all[0]
            best_delta = pred_delta_all[best_idx]
            true_edit = product_edit._edit_class_from_delta(bo_r[row_idx, :n_atoms, :n_atoms], row_delta)[pair_mask]
            top1_edit = product_edit._edit_class_from_delta(bo_r[row_idx, :n_atoms, :n_atoms], top1_delta)[pair_mask]
            best_edit = product_edit._edit_class_from_delta(bo_r[row_idx, :n_atoms, :n_atoms], best_delta)[pair_mask]

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
                            "exact_sample": int(product_edit._candidate_exact(candidates[cand_idx : cand_idx + 1], row_target[None, :, :], pair_mask, 1)),
                            "exact_group": int(product_edit._candidate_exact(candidates[cand_idx : cand_idx + 1], active_group_targets, pair_mask, 1)),
                            "valid_product": int(product_edit._valid_product(bo_r[row_idx, :n_atoms, :n_atoms], cand_delta, pair_mask)),
                            "n_predicted_edits": int((np.abs(cand_delta[pair_mask]) > 1e-6).sum()),
                            "edits": product_edit._summarize_edits(cand_delta, pair_mask),
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
        "center_coverage_topM": float(center_coverage / denom),
        "candidate_pool_size_mean": float(np.mean(candidate_pool_sizes)) if candidate_pool_sizes else 0.0,
        "valid_product_rate_top1": float(valid_top1 / denom),
        "valid_product_rate_top5": float(valid_topk / max(valid_topk_total, 1)),
        "edit_pair_F1_top1": binary_f1(changed_y_top1, changed_p_top1),
        "edit_pair_F1_best_of_K": binary_f1(changed_y_best, changed_p_best),
        "edit_macro_F1_top1": macro_f1(edit_y_top1, edit_p_top1, n_classes=len(product_edit.EDIT_CLASSES)),
        "edit_macro_F1_best_of_K": macro_f1(edit_y_best, edit_p_best, n_classes=len(product_edit.EDIT_CLASSES)),
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
        "# RFM WLDN-Style R-only Product Prediction",
        "",
        f"Best epoch: {best['epoch']}",
        "Selection: minimum valid_loss",
        f"Input schema: {best['input_schema']}",
        f"center_top_m: {best['center_top_m']}",
        f"candidate_pool_size: {best['candidate_pool_size']}",
        "",
        "| Split | top1 group | top3 group | top5 group | center coverage | edit F1 top1 | edit F1 best-K | Delta BO MAE top1 | Delta BO MAE best-K | valid top5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        metric = best[split]
        lines.append(
            f"| {split} | {metric['exact_product_match_top1_group']:.4f} | {metric['exact_product_match_top3_group']:.4f} | "
            f"{metric['exact_product_match_top5_group']:.4f} | {metric['center_coverage_topM']:.4f} | "
            f"{metric['edit_pair_F1_top1']:.4f} | {metric['edit_pair_F1_best_of_K']:.4f} | "
            f"{metric['Delta_BO_MAE_top1']:.4f} | {metric['Delta_BO_MAE_best_of_K']:.4f} | {metric['valid_product_rate_top5']:.4f} |"
        )
    output_dir.joinpath("summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
