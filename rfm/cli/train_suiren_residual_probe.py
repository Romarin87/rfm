"""Train a conditional residual probe above a frozen Stage B checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rfm.data.sampling import AtomCountBatchSampler
from rfm.models.conditional_residual import ConditionalResidualProbe, load_frozen_stage_b
from rfm.tasks import suiren_residual_probe
from rfm.utils.runtime import cleanup_ddp, ddp_enabled, is_main, resolve_device, set_seed, setup_ddp, write_csv, write_json, write_run_provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train-suiren-3d-atom-cache", required=True)
    parser.add_argument("--valid-suiren-3d-atom-cache", required=True)
    parser.add_argument("--test-suiren-3d-atom-cache", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--mode", choices=ConditionalResidualProbe.MODES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batching", choices=("random", "atom_count_bucket"), default="atom_count_bucket")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--geometry-mode", choices=("irc_rp",), default="irc_rp")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-suiren-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-suiren-atom-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        values["persistent_workers"] = args.persistent_workers
        values["prefetch_factor"] = args.prefetch_factor
    return values


def make_dataset(args: argparse.Namespace, split: str) -> suiren_residual_probe.SuirenResidualProbeDataset:
    return suiren_residual_probe.SuirenResidualProbeDataset(
        getattr(args, split),
        getattr(args, f"{split}_suiren_3d_atom_cache"),
        getattr(args, f"max_{split}"),
        args,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank, world, local_rank = setup_ddp()
    set_seed(args.seed + rank)
    device = resolve_device(args.device, local_rank)
    output_dir = Path(args.output_dir)
    if is_main(rank):
        write_run_provenance(output_dir, args, sys.argv)
    if ddp_enabled():
        dist.barrier()

    train_ds = make_dataset(args, "train")
    valid_ds = make_dataset(args, "valid")
    test_ds = make_dataset(args, "test")
    if not (train_ds.suiren_atom_dim == valid_ds.suiren_atom_dim == test_ds.suiren_atom_dim):
        raise ValueError("Suiren atom dimensions differ across train/valid/test")

    if args.train_batching == "atom_count_bucket":
        train_sampler = AtomCountBatchSampler(
            train_ds.atom_counts,
            args.batch_size,
            seed=args.seed,
            num_replicas=world,
            rank=rank,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=suiren_residual_probe.collate,
            **loader_kwargs(args),
        )
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp_enabled() else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=suiren_residual_probe.collate,
            **loader_kwargs(args),
        )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=suiren_residual_probe.collate, **loader_kwargs(args))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=suiren_residual_probe.collate, **loader_kwargs(args))

    stage_b, stage_b_config, y_mean, y_std = load_frozen_stage_b(args.stage_b_checkpoint, device)
    hidden_dim = int(stage_b_config["hidden_dim"])
    model = ConditionalResidualProbe(
        stage_b=stage_b,
        hidden_dim=hidden_dim,
        suiren_atom_dim=train_ds.suiren_atom_dim,
        y_mean=y_mean,
        y_std=y_std,
        mode=args.mode,
        residual_hidden_dim=args.residual_hidden_dim or hidden_dim,
        dropout=args.dropout,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    train_model = (
        DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if ddp_enabled() and device.type == "cuda"
        else model
    )

    history: list[dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve_epochs = 0
    early_stopped = False
    stop_epoch = args.epochs - 1
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = suiren_residual_probe.train_one_epoch(
            train_model,
            train_loader,
            optimizer,
            y_mean,
            y_std,
            device,
            args.grad_clip,
            epoch,
            args.log_every_steps,
        )
        stop_now = False
        if is_main(rank):
            valid_loss = suiren_residual_probe.evaluate_loss(model, valid_loader, y_mean, y_std, device)
            row = {"epoch": epoch, "train": train_metrics, "valid_loss": valid_loss}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if valid_loss < best_valid_loss - args.early_stop_min_delta:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.residual_state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            stop_now = args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience
            if stop_now:
                early_stopped = True
                stop_epoch = epoch
                print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_valid_loss": best_valid_loss}, ensure_ascii=False), flush=True)
        if ddp_enabled():
            stop_tensor = torch.tensor([int(stop_now)], device=device)
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
            stop_now = bool(stop_tensor.item())
        if stop_now:
            break

    if is_main(rank):
        if best_state is None:
            raise RuntimeError("residual probe did not produce a checkpoint")
        model.load_residual_state_dict(best_state)
        valid_metrics = suiren_residual_probe.evaluate(model, valid_loader, device)
        test_metrics = suiren_residual_probe.evaluate(model, test_loader, device)
        metrics = {
            "mode": args.mode,
            "epoch": best_epoch,
            "seed": args.seed,
            "world_size": world,
            "best_valid_loss": best_valid_loss,
            "epochs_requested": args.epochs,
            "epochs_completed": stop_epoch + 1 if early_stopped else len(history),
            "early_stopped": early_stopped,
            "stage_b_checkpoint": args.stage_b_checkpoint,
            "stage_b_config": stage_b_config,
            "trainable_parameter_count": model.trainable_parameter_count(),
            "suiren_atom_dim": train_ds.suiren_atom_dim,
            "suiren_state_dim": model.suiren_state_dim,
            "valid": valid_metrics,
            "test": test_metrics,
            "sizes": {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)},
        }
        torch.save(
            {
                "residual_model": best_state,
                "config": vars(args),
                "stage_b_checkpoint": args.stage_b_checkpoint,
                "target_mean": y_mean.detach().cpu().tolist(),
                "target_std": y_std.detach().cpu().tolist(),
            },
            output_dir / "model.pt",
        )
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "history.json", history)
        write_json(
            output_dir / "data_manifest.json",
            {
                "train": args.train,
                "valid": args.valid,
                "test": args.test,
                "train_suiren_3d_atom_cache": args.train_suiren_3d_atom_cache,
                "valid_suiren_3d_atom_cache": args.valid_suiren_3d_atom_cache,
                "test_suiren_3d_atom_cache": args.test_suiren_3d_atom_cache,
                "mode": args.mode,
                "control_semantics": "Suiren R/P atom states are replaced with zeros before the shared projector",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_csv(output_dir / "predictions_valid.csv", suiren_residual_probe.predict_rows(model, valid_loader, device))
        write_csv(output_dir / "predictions_test.csv", suiren_residual_probe.predict_rows(model, test_loader, device))
        suiren_residual_probe.write_summary(output_dir, metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    cleanup_ddp()


if __name__ == "__main__":
    main()
