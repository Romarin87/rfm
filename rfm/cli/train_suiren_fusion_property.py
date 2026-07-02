"""CLI for Suiren fusion reaction property regression."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rfm.models import SuirenFusionPropertyRegressor, load_pretrained_encoder
from rfm.optim import build_optimizer
from rfm.tasks import suiren_fusion_property
from rfm.utils.runtime import (
    append_registry,
    cleanup_ddp,
    ddp_enabled,
    is_main,
    resolve_device,
    set_seed,
    setup_ddp,
    write_csv,
    write_json,
    write_run_provenance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    for split in ("train", "valid", "test"):
        for stream in ("2d", "3d"):
            parser.add_argument(f"--{split}-suiren-{stream}-graph-cache", default="")
            parser.add_argument(f"--{split}-suiren-{stream}-atom-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pretrained-encoder", default="")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dynamic-pair-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pair-update-scale", type=float, default=0.75)
    parser.add_argument("--dynamic-pair-update-dropout", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--geometry-mode", choices=("2d", "irc_rp"), default="2d")
    parser.add_argument("--stage-a-weight", type=float, default=0.5)
    parser.add_argument("--stage-c-weight", type=float, default=1.0)
    parser.add_argument("--mask-strategy", choices=("reaction_center", "changed_enriched", "random_pair"), default="reaction_center")
    parser.add_argument("--pair-mask-ratio", type=float, default=0.10)
    parser.add_argument("--changed-pair-mask-ratio", type=float, default=1.0)
    parser.add_argument("--distance-clip", type=float, default=10.0)
    parser.add_argument("--delta-bo-weight", type=float, default=1.0)
    parser.add_argument("--changed-weight", type=float, default=0.5)
    parser.add_argument("--edit-weight", type=float, default=0.5)
    parser.add_argument("--core-weight", type=float, default=0.2)
    parser.add_argument("--edit-class-weights", default="1.0,4.0,4.0,2.0")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-suiren-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-suiren-graph-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-suiren-atom-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--registry", default="outputs/experiment_registry.csv")
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--muon-lr", type=float, default=0.005)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-ns-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--muon-distributed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def data_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def reaction_property_schema(args: argparse.Namespace) -> tuple[str, int]:
    if args.geometry_mode == "irc_rp":
        return "property_irc_rp_bo", 4
    return "property_rp2d_bo", 2


def split_args(args: argparse.Namespace, split: str) -> argparse.Namespace:
    values = vars(args).copy()
    for stream in ("2d", "3d"):
        values[f"suiren_{stream}_graph_cache"] = getattr(args, f"{split}_suiren_{stream}_graph_cache")
        values[f"suiren_{stream}_atom_cache"] = getattr(args, f"{split}_suiren_{stream}_atom_cache")
    return argparse.Namespace(**values)


def cache_manifest(args: argparse.Namespace, train_ds: Any, valid_ds: Any, test_ds: Any) -> dict[str, Any]:
    return {
        "train": {
            "suiren_2d_graph_cache": args.train_suiren_2d_graph_cache,
            "suiren_3d_graph_cache": args.train_suiren_3d_graph_cache,
            "suiren_2d_atom_cache": args.train_suiren_2d_atom_cache,
            "suiren_3d_atom_cache": args.train_suiren_3d_atom_cache,
        },
        "valid": {
            "suiren_2d_graph_cache": args.valid_suiren_2d_graph_cache,
            "suiren_3d_graph_cache": args.valid_suiren_3d_graph_cache,
            "suiren_2d_atom_cache": args.valid_suiren_2d_atom_cache,
            "suiren_3d_atom_cache": args.valid_suiren_3d_atom_cache,
        },
        "test": {
            "suiren_2d_graph_cache": args.test_suiren_2d_graph_cache,
            "suiren_3d_graph_cache": args.test_suiren_3d_graph_cache,
            "suiren_2d_atom_cache": args.test_suiren_2d_atom_cache,
            "suiren_3d_atom_cache": args.test_suiren_3d_atom_cache,
        },
        "dims": {
            "suiren_graph_dims": train_ds.suiren_graph_dims,
            "suiren_atom_dims": train_ds.suiren_atom_dims,
        },
        "metadata": {
            "train_graph": {stream: store.metadata for stream, store in train_ds.graph_stores.items() if store.path is not None},
            "train_atom": {stream: store.metadata for stream, store in train_ds.atom_stores.items() if store.path is not None},
            "valid_graph": {stream: store.metadata for stream, store in valid_ds.graph_stores.items() if store.path is not None},
            "valid_atom": {stream: store.metadata for stream, store in valid_ds.atom_stores.items() if store.path is not None},
            "test_graph": {stream: store.metadata for stream, store in test_ds.graph_stores.items() if store.path is not None},
            "test_atom": {stream: store.metadata for stream, store in test_ds.atom_stores.items() if store.path is not None},
        },
    }


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

    train_ds = suiren_fusion_property.SuirenFusionPropertyDataset(
        args.train,
        args.max_train,
        split_args(args, "train"),
    )
    valid_ds = suiren_fusion_property.SuirenFusionPropertyDataset(
        args.valid,
        args.max_valid,
        split_args(args, "valid"),
    )
    test_ds = suiren_fusion_property.SuirenFusionPropertyDataset(
        args.test,
        args.max_test,
        split_args(args, "test"),
    )
    if train_ds.suiren_graph_dims != valid_ds.suiren_graph_dims or train_ds.suiren_graph_dims != test_ds.suiren_graph_dims:
        raise ValueError("Suiren graph feature dimensions differ across splits")
    if train_ds.suiren_atom_dims != valid_ds.suiren_atom_dims or train_ds.suiren_atom_dims != test_ds.suiren_atom_dims:
        raise ValueError("Suiren atom feature dimensions differ across splits")

    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp_enabled() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=suiren_fusion_property.collate,
        **data_loader_kwargs(args),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=suiren_fusion_property.collate,
        **data_loader_kwargs(args),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=suiren_fusion_property.collate,
        **data_loader_kwargs(args),
    )

    y_mean, y_std = suiren_fusion_property.target_stats(train_ds)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    input_schema, pair_raw_dim = reaction_property_schema(args)
    model = SuirenFusionPropertyRegressor(
        pair_raw_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        input_schema=input_schema,
        suiren_atom_dims=train_ds.suiren_atom_dims,
        suiren_graph_dims=train_ds.suiren_graph_dims,
        dynamic_pair_update=args.dynamic_pair_update,
        dynamic_pair_update_scale=args.dynamic_pair_update_scale,
        dynamic_pair_update_dropout=args.dynamic_pair_update_dropout,
    ).to(device)
    if args.pretrained_encoder:
        load_pretrained_encoder(model, args.pretrained_encoder, device)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    train_model: nn.Module = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.freeze_encoder) if ddp_enabled() and device.type == "cuda" else model
    optimizer = build_optimizer(train_model.named_parameters(), args)

    history: list[dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_valid_loss_parts: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improve_epochs = 0
    early_stopped = False
    stop_epoch = args.epochs - 1

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = suiren_fusion_property.train_one_epoch(train_model, train_loader, optimizer, y_mean, y_std, args, device)
        stop_now = False
        if is_main(rank):
            valid_loss = suiren_fusion_property.evaluate_loss(model, valid_loader, y_mean, y_std, args, device)
            valid_metrics = suiren_fusion_property.evaluate(model, valid_loader, y_mean, y_std, device)
            row = {"epoch": epoch, "train": train_metrics, "valid_loss": valid_loss, "valid": valid_metrics}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if valid_loss["loss"] < best_valid_loss - args.early_stop_min_delta:
                best_valid_loss = valid_loss["loss"]
                best_valid_loss_parts = valid_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            stop_now = args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience
            if stop_now:
                early_stopped = True
                stop_epoch = epoch
                print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_valid_loss": best_valid_loss, "patience": args.early_stop_patience}, ensure_ascii=False), flush=True)
        if ddp_enabled():
            stop_tensor = torch.tensor([1 if stop_now else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
            stop_now = bool(stop_tensor.item())
        if stop_now:
            if not is_main(rank):
                early_stopped = True
                stop_epoch = epoch
            break

    if is_main(rank):
        if best_state is not None:
            model.load_state_dict(best_state)
        valid_metrics = suiren_fusion_property.evaluate(model, valid_loader, y_mean, y_std, device)
        test_metrics = suiren_fusion_property.evaluate(model, test_loader, y_mean, y_std, device)
        valid_rows = suiren_fusion_property.predict_rows(model, valid_loader, y_mean, y_std, device)
        test_rows = suiren_fusion_property.predict_rows(model, test_loader, y_mean, y_std, device)
        cache_info = cache_manifest(args, train_ds, valid_ds, test_ds)
        best = {
            "epoch": best_epoch,
            "seed": args.seed,
            "world_size": world,
            "pretrained_encoder": args.pretrained_encoder,
            "freeze_encoder": args.freeze_encoder,
            "input_schema": input_schema,
            "optimizer": args.optimizer,
            "optimizer_metadata": getattr(optimizer, "metadata", {"optimizer": args.optimizer}),
            "epochs_requested": args.epochs,
            "epochs_completed": stop_epoch + 1 if early_stopped else len(history),
            "early_stopped": early_stopped,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "selection": {"criterion": "valid_loss", "best_valid_loss": best_valid_loss, "best_valid_loss_parts": best_valid_loss_parts},
            "valid": valid_metrics,
            "test": test_metrics,
            "target_mean": y_mean.detach().cpu().numpy().tolist(),
            "target_std": y_std.detach().cpu().numpy().tolist(),
            "sizes": {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)},
            "suiren_feature_dims": cache_info["dims"],
        }
        torch.save({"model": best_state, "config": vars(args), "target_mean": best["target_mean"], "target_std": best["target_std"], "model_class": "SuirenFusionPropertyRegressor"}, output_dir / "model.pt")
        torch.save({"input_adapter": model.input_adapter.state_dict(), "config": vars(args), "feature_dims": cache_info["dims"]}, output_dir / "unified_reaction_input_adapter.pt")
        torch.save({"adapter": model.masked_adapter.state_dict(), "config": vars(args)}, output_dir / "masked_edit_adapter.pt")
        torch.save({"encoder": model.encoder.state_dict(), "config": vars(args), "model_class": "TokenSpaceReactionEncoder"}, output_dir / "reaction_encoder.pt")
        torch.save({"masked_edit_heads": model.masked_edit_heads.state_dict(), "config": vars(args)}, output_dir / "masked_edit_heads.pt")
        torch.save({"property_head": model.reg_head.state_dict(), "config": vars(args)}, output_dir / "property_head.pt")
        write_json(output_dir / "metrics.json", best)
        write_json(output_dir / "history.json", history)
        write_json(output_dir / "cache_manifest.json", cache_info)
        write_json(
            output_dir / "data_manifest.json",
            {
                "train": args.train,
                "valid": args.valid,
                "test": args.test,
                "sizes": best["sizes"],
                "geometry_mode": args.geometry_mode,
                "input_schema": input_schema,
                "suiren_feature_dims": cache_info["dims"],
                "suiren_input_semantics": {
                    "graph_features": "initial_reaction_token_input_projection",
                    "atom_features": "atom_token_input_projection_aligned_by_atom_map_order",
                    "suiren_pair_tokens": False,
                    "encoder_output_shape_fixed": True,
                },
                "loss": {
                    "stage_a_weight": args.stage_a_weight,
                    "stage_c_weight": args.stage_c_weight,
                    "effective_property_weight": args.stage_c_weight,
                    "stage_a_components": {
                        "delta_bo_weight": args.delta_bo_weight,
                        "changed_weight": args.changed_weight,
                        "edit_weight": args.edit_weight,
                        "core_weight": args.core_weight,
                    },
                },
                "mask": {
                    "mask_strategy": args.mask_strategy,
                    "pair_mask_ratio": args.pair_mask_ratio,
                    "changed_pair_mask_ratio": args.changed_pair_mask_ratio,
                },
                "cache_manifest": str(output_dir / "cache_manifest.json"),
                "stage_c_loader": {
                    "trust_suiren_cache": args.trust_suiren_cache,
                    "preload_suiren_graph_cache": args.preload_suiren_graph_cache,
                    "preload_suiren_atom_cache": args.preload_suiren_atom_cache,
                    "num_workers": args.num_workers,
                    "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
                    "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_csv(output_dir / "predictions_valid.csv", valid_rows)
        write_csv(output_dir / "predictions_test.csv", test_rows)
        write_csv(output_dir / "failure_cases.csv", suiren_fusion_property.failure_cases(test_rows))
        suiren_fusion_property.write_summary(output_dir, best)
        append_registry(
            args,
            output_dir,
            stage="suiren_fusion_property_regression",
            task="suiren_fusion_property_regression",
            model="SuirenFusionPropertyRegressor",
            status="complete",
            notes=f"input_schema={input_schema}; geometry_mode={args.geometry_mode}; suiren=unified_input; loss=0.5*L_A+1.0*L_C; selection=minimum valid_loss; optimizer={args.optimizer}; suiren_dims={cache_info['dims']}",
        )
        print(json.dumps(best, indent=2, ensure_ascii=False), flush=True)
    cleanup_ddp()


if __name__ == "__main__":
    main()
