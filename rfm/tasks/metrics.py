"""Pure-numpy metric helpers for RFM tasks."""

from __future__ import annotations

import math

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.size == 0:
        return 0.0
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = _rankdata(scores)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    pred = np.asarray(pred, dtype=np.int64).ravel()
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def macro_f1(labels: np.ndarray, pred: np.ndarray, n_classes: int | None = None) -> float:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    pred = np.asarray(pred, dtype=np.int64).ravel()
    if labels.size == 0:
        return 0.0
    classes = range(n_classes) if n_classes is not None else sorted(set(labels.tolist()) | set(pred.tolist()))
    scores = []
    for cls in classes:
        cls_labels = (labels == cls).astype(np.int64)
        cls_pred = (pred == cls).astype(np.int64)
        scores.append(binary_f1(cls_labels, cls_pred))
    return float(np.mean(scores) if scores else 0.0)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, prefix: str) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return {f"{prefix}_AUROC": 0.0, f"{prefix}_F1": 0.0}
    pred = (np.asarray(scores) >= 0.0).astype(np.int64)
    return {
        f"{prefix}_AUROC": safe_auroc(labels, scores),
        f"{prefix}_F1": binary_f1(labels, pred),
    }


def _r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 0.0 if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def _spearman(true: np.ndarray, pred: np.ndarray) -> float:
    true_rank = _rankdata(true)
    pred_rank = _rankdata(pred)
    true_std = float(true_rank.std())
    pred_std = float(pred_rank.std())
    if true_std <= 0 or pred_std <= 0:
        return 0.0
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(targets):
        true = np.asarray(y_true[:, idx], dtype=np.float64)
        pred = np.asarray(y_pred[:, idx], dtype=np.float64)
        mse = float(np.mean((true - pred) ** 2))
        out[target] = {
            "MAE": mean_absolute_error(true, pred),
            "RMSE": float(math.sqrt(mse)),
            "R2": _r2_score(true, pred),
            "Spearman": _spearman(true, pred),
        }
    return out
