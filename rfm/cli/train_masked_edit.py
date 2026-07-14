"""CLI for masked edit/core pretraining."""

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

from rfm.data.reaction_samples import EDIT_CLASSES
from rfm.models import MaskedEditPretrainingModel
from rfm.optim import build_optimizer
from rfm.tasks import masked_edit
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-type", choices=("token_space", "radar", "mrto", "mrto_v1", "mrto_full"), default="token_space")
    parser.add_argument("--radar-attention-heads", type=int, default=8)
    parser.add_argument("--radar-center-router", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radar-delta-stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radar-pair-update-scale", type=float, default=0.75)
    parser.add_argument("--radar-reaction-update-scale", type=float, default=1.0)
    parser.add_argument("--radar-router-gate-init", type=float, default=0.05)
    parser.add_argument("--mrto-attention-heads", type=int, default=8)
    parser.add_argument("--mrto-event-slots", type=int, default=4)
    parser.add_argument("--mrto-use-event-slots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mrto-use-odd-field", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mrto-pair-update-scale", type=float, default=1.0)
    parser.add_argument("--mrto-reaction-update-scale", type=float, default=1.0)
    parser.add_argument("--mrto-endpoint-layers", type=int, default=2)
    parser.add_argument("--mrto-triangle-layers", type=int, default=2)
    parser.add_argument("--mrto-triangle-dim", type=int, default=16)
    parser.add_argument("--mrto-triangle-scale", type=float, default=0.5)
    parser.add_argument("--mrto-event-topk", type=int, default=0)
    parser.add_argument("--mrto-event-feedback-scale", type=float, default=0.5)
    parser.add_argument("--mrto-geometry-rbf-bins", type=int, default=16)
    parser.add_argument("--mrto-equiformer-layers", type=int, default=2)
    parser.add_argument("--mrto-equiformer-channels", type=int, default=32)
    parser.add_argument("--mrto-equiformer-lmax", type=int, default=2)
    parser.add_argument("--mrto-equiformer-radius", type=float, default=5.0)
    parser.add_argument("--mrto-equiformer-max-neighbors", type=int, default=64)
    parser.add_argument("--mrto-event-set-weight", type=float, default=0.0)
    parser.add_argument("--mrto-event-diversity-weight", type=float, default=0.0)
    parser.add_argument("--mrto-edit-set-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--geometry-mode", choices=("2d", "irc_rp"), default="2d")
    parser.add_argument("--mask-strategy", choices=("reaction_center", "changed_enriched", "random_pair"), default="reaction_center")
    parser.add_argument("--pair-mask-ratio", type=float, default=0.10)
    parser.add_argument("--changed-pair-mask-ratio", type=float, default=1.0)
    parser.add_argument("--distance-clip", type=float, default=10.0)
    parser.add_argument("--delta-bo-weight", type=float, default=1.0)
    parser.add_argument("--changed-weight", type=float, default=0.5)
    parser.add_argument("--edit-weight", type=float, default=0.5)
    parser.add_argument("--core-weight", type=float, default=0.2)
    parser.add_argument("--enable-dynamic-pair-update", action="store_true")
    parser.add_argument("--dynamic-pair-update-scale", type=float, default=1.0)
    parser.add_argument("--dynamic-pair-update-dropout", type=float, default=None)
    parser.add_argument("--edit-class-weights", default="1.0,4.0,4.0,2.0")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
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


def masked_edit_schema(args: argparse.Namespace) -> tuple[str, int]:
    if args.geometry_mode == "irc_rp":
        return "masked_edit_irc_rp", 4
    return "masked_edit_2d", 3


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.encoder_type == "mrto_full" and args.mrto_event_topk == 0:
        args.mrto_event_topk = 16
    rank, world, local_rank = setup_ddp()
    set_seed(args.seed + rank)
    device = resolve_device(args.device, local_rank)
    output_dir = Path(args.output_dir)
    if is_main(rank):
        write_run_provenance(output_dir, args, sys.argv)
    if ddp_enabled():
        dist.barrier()

    train_ds = masked_edit.MaskedEditDataset(args.train, args.max_train, args)
    valid_ds = masked_edit.MaskedEditDataset(args.valid, args.max_valid, args)
    test_ds = masked_edit.MaskedEditDataset(args.test, args.max_test, args)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp_enabled() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=masked_edit.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=masked_edit.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=masked_edit.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    input_schema, pair_raw_dim = masked_edit_schema(args)
    model = MaskedEditPretrainingModel(
        pair_raw_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        input_schema=input_schema,
        dynamic_pair_update=args.enable_dynamic_pair_update,
        dynamic_pair_update_scale=args.dynamic_pair_update_scale,
        dynamic_pair_update_dropout=args.dynamic_pair_update_dropout,
        encoder_type=args.encoder_type,
        radar_attention_heads=args.radar_attention_heads,
        radar_center_router=args.radar_center_router,
        radar_delta_stream=args.radar_delta_stream,
        radar_pair_update_scale=args.radar_pair_update_scale,
        radar_reaction_update_scale=args.radar_reaction_update_scale,
        radar_router_gate_init=args.radar_router_gate_init,
        mrto_attention_heads=args.mrto_attention_heads,
        mrto_event_slots=args.mrto_event_slots,
        mrto_use_event_slots=args.mrto_use_event_slots,
        mrto_use_odd_field=args.mrto_use_odd_field,
        mrto_pair_update_scale=args.mrto_pair_update_scale,
        mrto_reaction_update_scale=args.mrto_reaction_update_scale,
        mrto_endpoint_layers=args.mrto_endpoint_layers,
        mrto_triangle_layers=args.mrto_triangle_layers,
        mrto_triangle_dim=args.mrto_triangle_dim,
        mrto_triangle_scale=args.mrto_triangle_scale,
        mrto_event_topk=args.mrto_event_topk,
        mrto_event_feedback_scale=args.mrto_event_feedback_scale,
        mrto_geometry_rbf_bins=args.mrto_geometry_rbf_bins,
        mrto_equiformer_layers=args.mrto_equiformer_layers,
        mrto_equiformer_channels=args.mrto_equiformer_channels,
        mrto_equiformer_lmax=args.mrto_equiformer_lmax,
        mrto_equiformer_radius=args.mrto_equiformer_radius,
        mrto_equiformer_max_neighbors=args.mrto_equiformer_max_neighbors,
    ).to(device)
    train_model: nn.Module = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if ddp_enabled() and device.type == "cuda" else model
    optimizer = build_optimizer(train_model.named_parameters(), args)
    weight_values = masked_edit.parse_float_list(args.edit_class_weights)
    if len(weight_values) != len(EDIT_CLASSES):
        raise ValueError(f"--edit-class-weights must have {len(EDIT_CLASSES)} values")
    edit_class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    history: list[dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_valid_loss_parts: dict[str, float] | None = None
    no_improve_epochs = 0
    early_stopped = False
    stop_epoch = args.epochs - 1

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = masked_edit.train_one_epoch(train_model, train_loader, optimizer, device, args, edit_class_weights)
        stop_now = False
        if is_main(rank):
            valid_loss = masked_edit.evaluate_loss(model, valid_loader, device, args, edit_class_weights)
            valid_metrics, _ = masked_edit.evaluate(model, valid_loader, device, max_prediction_rows=0)
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "valid_loss": valid_loss,
                "valid": valid_metrics,
                "router_gates": model.masked_edit_heads.router_gate_values(),
            }
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
                print(
                    json.dumps(
                        {
                            "event": "early_stop",
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_valid_loss": best_valid_loss,
                            "patience": args.early_stop_patience,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
        valid_metrics, valid_rows = masked_edit.evaluate(model, valid_loader, device)
        test_metrics, test_rows = masked_edit.evaluate(model, test_loader, device)
        best = {
            "epoch": best_epoch,
            "seed": args.seed,
            "world_size": world,
            "valid": valid_metrics,
            "test": test_metrics,
            "sizes": {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)},
            "edit_classes": EDIT_CLASSES,
            "model": "MaskedEditPretrainingModel",
            "encoder_type": args.encoder_type,
            "encoder_class": type(model.encoder).__name__,
            "input_schema": input_schema,
            "optimizer": args.optimizer,
            "optimizer_metadata": getattr(optimizer, "metadata", {"optimizer": args.optimizer}),
            "epochs_requested": args.epochs,
            "epochs_completed": stop_epoch + 1 if early_stopped else len(history),
            "early_stopped": early_stopped,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "router_gates": model.masked_edit_heads.router_gate_values(),
            "selection": {"criterion": "valid_loss", "best_valid_loss": best_valid_loss, "best_valid_loss_parts": best_valid_loss_parts},
        }
        torch.save({"model": best_state, "config": vars(args), "model_class": "MaskedEditPretrainingModel", "encoder_class": type(model.encoder).__name__}, output_dir / "model.pt")
        torch.save({"adapter": model.adapter.state_dict(), "config": vars(args)}, output_dir / "masked_edit_adapter.pt")
        torch.save({"encoder": model.encoder.state_dict(), "config": vars(args), "model_class": type(model.encoder).__name__}, output_dir / "reaction_encoder.pt")
        torch.save({"masked_edit_heads": model.masked_edit_heads.state_dict(), "config": vars(args), "edit_classes": EDIT_CLASSES}, output_dir / "masked_edit_heads.pt")
        write_json(output_dir / "metrics.json", best)
        write_json(output_dir / "history.json", history)
        write_json(
            output_dir / "data_manifest.json",
            {
                "train": args.train,
                "valid": args.valid,
                "test": args.test,
                "sizes": best["sizes"],
                "geometry_mode": args.geometry_mode,
                "input_schema": input_schema,
                "encoder_type": args.encoder_type,
                "encoder_class": type(model.encoder).__name__,
                "radar": {
                    "attention_heads": args.radar_attention_heads,
                    "center_router": args.radar_center_router,
                    "delta_stream": args.radar_delta_stream,
                    "pair_update_scale": args.radar_pair_update_scale,
                    "reaction_update_scale": args.radar_reaction_update_scale,
                    "router_gate_init": args.radar_router_gate_init,
                    "router_residual": args.radar_center_router,
                },
                "mrto": {
                    "attention_heads": args.mrto_attention_heads,
                    "event_slots": args.mrto_event_slots,
                    "use_event_slots": args.mrto_use_event_slots,
                    "use_odd_field": args.mrto_use_odd_field,
                    "pair_update_scale": args.mrto_pair_update_scale,
                    "reaction_update_scale": args.mrto_reaction_update_scale,
                    "endpoint_layers": args.mrto_endpoint_layers,
                    "triangle_layers": args.mrto_triangle_layers,
                    "triangle_dim": args.mrto_triangle_dim,
                    "triangle_scale": args.mrto_triangle_scale,
                    "event_topk": args.mrto_event_topk,
                    "event_feedback_scale": args.mrto_event_feedback_scale,
                    "geometry_rbf_bins": args.mrto_geometry_rbf_bins,
                    "equiformer_layers": args.mrto_equiformer_layers,
                    "equiformer_channels": args.mrto_equiformer_channels,
                    "equiformer_lmax": args.mrto_equiformer_lmax,
                    "equiformer_radius": args.mrto_equiformer_radius,
                    "equiformer_max_neighbors": args.mrto_equiformer_max_neighbors,
                    "event_set_weight": args.mrto_event_set_weight,
                    "event_diversity_weight": args.mrto_event_diversity_weight,
                    "edit_set_weight": args.mrto_edit_set_weight,
                    "router_residual": args.encoder_type in {"mrto", "mrto_v1", "mrto_full"},
                },
                "mask_strategy": args.mask_strategy,
                "effective_batch_size_per_process": args.batch_size * args.gradient_accumulation_steps,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_csv(output_dir / "predictions_valid.csv", valid_rows)
        write_csv(output_dir / "predictions_test.csv", test_rows)
        masked_edit.write_summary(output_dir, best)
        append_registry(
            args,
            output_dir,
            stage="masked_edit_pretraining",
            task="masked_delta_bo_edit_core",
            model="MaskedEditPretrainingModel",
            status="complete",
            notes=(
                f"input_schema={input_schema}; geometry_mode={args.geometry_mode}; encoder_type={args.encoder_type}; "
                f"selection=minimum valid_loss; optimizer={args.optimizer}"
            ),
        )
        print(json.dumps(best, indent=2, ensure_ascii=False), flush=True)
    cleanup_ddp()


if __name__ == "__main__":
    main()
