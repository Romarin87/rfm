"""Runtime helpers shared by current RFM training entry points."""

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp() -> tuple[int, int, int]:
    if not ddp_enabled():
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, local_rank: int) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def write_run_provenance(output_dir: Path, args: Any, argv: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in vars(args).items()) + "\n", encoding="utf-8")
    output_dir.joinpath("command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
    commit = os.popen("git rev-parse HEAD 2>/dev/null || true").read().strip()
    output_dir.joinpath("git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    log_path = output_dir / "run_log.txt"
    if not log_path.exists():
        log_path.write_text(f"created_at: {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("reaction_id\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_registry(args: Any, output_dir: Path, *, stage: str, task: str, model: str, status: str, notes: str) -> None:
    registry = Path(args.registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    exists = registry.exists()
    fields = [
        "run_id",
        "datetime",
        "stage",
        "task",
        "model",
        "dataset",
        "split",
        "seed",
        "config_path",
        "metrics_path",
        "checkpoint_path",
        "predictions_path",
        "git_commit",
        "data_manifest",
        "status",
        "notes",
    ]
    with registry.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": args.run_id or output_dir.name,
                "datetime": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "task": task,
                "model": model,
                "dataset": args.train,
                "split": "train/valid/test",
                "seed": args.seed,
                "config_path": str(output_dir / "config.yaml"),
                "metrics_path": str(output_dir / "metrics.json"),
                "checkpoint_path": str(output_dir / "model.pt"),
                "predictions_path": str(output_dir / "predictions_test.csv"),
                "git_commit": str(output_dir / "git_commit.txt"),
                "data_manifest": str(output_dir / "data_manifest.json"),
                "status": status,
                "notes": notes,
            }
        )
