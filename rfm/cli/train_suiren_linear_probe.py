"""Train a Suiren-only reaction-property ridge linear probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rfm.data.reaction_samples import ENERGY_TARGETS
from rfm.tasks import reaction_property, suiren_linear_probe
from rfm.utils.runtime import append_registry, write_csv, write_json, write_run_provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train-suiren-atom-cache", required=True)
    parser.add_argument("--valid-suiren-atom-cache", required=True)
    parser.add_argument("--test-suiren-atom-cache", required=True)
    parser.add_argument("--pooled-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--representation",
        choices=suiren_linear_probe.REPRESENTATIONS,
        default="state_abs_delta",
    )
    parser.add_argument("--ridge-alphas", default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--pool-chunk-reactions", type=int, default=2048)
    parser.add_argument("--overwrite-pooled-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--registry", default="outputs/experiment_registry.csv")
    return parser.parse_args(argv)


def _alphas(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("--ridge-alphas must contain at least one value")
    return values


def _prediction_rows(split: suiren_linear_probe.ProbeSplit, prediction: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reaction_id, true, pred in zip(split.reaction_ids, split.targets, prediction):
        rows.append(
            {
                "reaction_id": str(reaction_id),
                "true_dE": float(true[0]),
                "pred_dE": float(pred[0]),
                "abs_error_dE": float(abs(true[0] - pred[0])),
                "true_dE_dagger": float(true[1]),
                "pred_dE_dagger": float(pred[1]),
                "abs_error_dE_dagger": float(abs(true[1] - pred[1])),
            }
        )
    return rows


def _write_summary(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Suiren-only 线性探针",
        "",
        "该诊断只使用 frozen Suiren atom embedding 的反应级均值，不读取 BO、距离或 Reaction Encoder 输出。",
        "",
        f"- 表示：`{metrics['representation']}`",
        f"- 特征维度：`{metrics['feature_dim']}`",
        f"- 选择的 ridge alpha：`{metrics['selected_alpha']}`",
        f"- valid normalized MSE：`{metrics['selected_valid_normalized_mse']:.6f}`",
        "",
        "| Split | dE MAE | dE RMSE | dE R2 | dE_dagger MAE | dE_dagger RMSE | dE_dagger R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("valid", "test"):
        values = metrics[split]
        lines.append(
            f"| {split} | {values['dE']['MAE']:.4f} | {values['dE']['RMSE']:.4f} | "
            f"{values['dE']['R2']:.4f} | {values['dE_dagger']['MAE']:.4f} | "
            f"{values['dE_dagger']['RMSE']:.4f} | {values['dE_dagger']['R2']:.4f} |"
        )
    output_dir.joinpath("summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    write_run_provenance(output_dir, args, sys.argv)
    pooled_dir = Path(args.pooled_cache_dir)
    pooled_paths = {
        split: pooled_dir / f"suiren_3d_atom_{split}_mean_probe.h5"
        for split in ("train", "valid", "test")
    }
    source_paths = {
        "train": args.train_suiren_atom_cache,
        "valid": args.valid_suiren_atom_cache,
        "test": args.test_suiren_atom_cache,
    }
    for split in ("train", "valid", "test"):
        print(json.dumps({"event": "pool_start", "split": split, "output": str(pooled_paths[split])}), flush=True)
        suiren_linear_probe.build_pooled_atom_cache(
            source_paths[split],
            pooled_paths[split],
            chunk_reactions=args.pool_chunk_reactions,
            overwrite=args.overwrite_pooled_cache,
        )
        print(json.dumps({"event": "pool_complete", "split": split}), flush=True)

    train = suiren_linear_probe.load_probe_split(args.train, pooled_paths["train"], args.representation)
    valid = suiren_linear_probe.load_probe_split(args.valid, pooled_paths["valid"], args.representation)
    test = suiren_linear_probe.load_probe_split(args.test, pooled_paths["test"], args.representation)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    result = suiren_linear_probe.fit_ridge_probe(
        train,
        valid,
        test,
        alphas=_alphas(args.ridge_alphas),
        device=device,
    )

    serializable = {
        "task": "suiren_only_linear_probe",
        "representation": args.representation,
        "uses_reaction_encoder": False,
        "uses_bo_or_distance": False,
        "pooling": "unweighted_atom_mean",
        "feature_layout": list(suiren_linear_probe.POOLED_LAYOUT),
        "feature_dim": result["feature_dim"],
        "selected_alpha": result["selected_alpha"],
        "selected_valid_normalized_mse": result["selected_valid_normalized_mse"],
        "alpha_search": result["alpha_search"],
        "valid": result["valid_metrics"],
        "test": result["test_metrics"],
        "mean_baseline_valid": result["mean_baseline_valid_metrics"],
        "mean_baseline_test": result["mean_baseline_test_metrics"],
        "target_mean": result["target_mean"].tolist(),
        "target_std": result["target_std"].tolist(),
        "sizes": {"train": len(train.targets), "valid": len(valid.targets), "test": len(test.targets)},
        "device": str(device),
    }
    valid_rows = _prediction_rows(valid, result["valid_prediction"])
    test_rows = _prediction_rows(test, result["test_prediction"])
    torch.save(
        {
            "weight": torch.from_numpy(result["weight"]),
            "feature_mean": torch.from_numpy(result["feature_mean"]),
            "feature_std": torch.from_numpy(result["feature_std"]),
            "target_mean": torch.from_numpy(result["target_mean"]),
            "target_std": torch.from_numpy(result["target_std"]),
            "selected_alpha": result["selected_alpha"],
            "representation": args.representation,
            "model_class": "RidgeLinearProbe",
        },
        output_dir / "model.pt",
    )
    write_json(output_dir / "metrics.json", serializable)
    write_json(
        output_dir / "data_manifest.json",
        {
            "data": {"train": args.train, "valid": args.valid, "test": args.test},
            "source_atom_caches": source_paths,
            "pooled_caches": {key: str(value) for key, value in pooled_paths.items()},
            "strict_reaction_id_alignment": True,
            "train_only_normalization": True,
            "test_used_for_model_selection": False,
        },
    )
    write_csv(output_dir / "predictions_valid.csv", valid_rows)
    write_csv(output_dir / "predictions_test.csv", test_rows)
    write_csv(output_dir / "failure_cases.csv", reaction_property.failure_cases(test_rows))
    _write_summary(output_dir, serializable)
    append_registry(
        args,
        output_dir,
        stage="diagnostic",
        task="suiren_only_linear_probe",
        model="RidgeLinearProbe",
        status="complete",
        notes=(
            f"representation={args.representation}; pooling=atom_mean; reaction_encoder=false; "
            f"selected_alpha={result['selected_alpha']}; selection=minimum_valid_normalized_mse"
        ),
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
