"""CLI for WLDN-style R-only product graph-edit prediction."""

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

from rfm.models import WLDNProductEditPredictor, load_pretrained_encoder
from rfm.optim import build_optimizer
from rfm.tasks import product_edit_wldn
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
    parser.add_argument("--pretrained-encoder", default="")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dynamic-pair-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pair-update-scale", type=float, default=0.75)
    parser.add_argument("--dynamic-pair-update-dropout", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--geometry-mode", choices=("2d", "irc_r"), default="2d")
    parser.add_argument("--distance-clip", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--center-top-m", type=int, default=6)
    parser.add_argument("--candidate-pool-size", type=int, default=32)
    parser.add_argument("--candidate-beam-size", type=int, default=64)
    parser.add_argument("--edit-class-top-k", type=int, default=2)
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--changed-pair-pos-weight", type=float, default=8.0)
    parser.add_argument("--max-group-products", type=int, default=16)
    parser.add_argument("--delta-vocab-decimals", type=int, default=6)
    parser.add_argument("--delta-class-weights", default="")
    parser.add_argument("--no-change-class-weight", type=float, default=0.2)
    parser.add_argument("--changed-class-weight", type=float, default=4.0)
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
    parser.add_argument("--suiren-2d-graph-cache", default="")
    parser.add_argument("--suiren-3d-graph-cache", default="")
    parser.add_argument("--suiren-2d-atom-cache", default="")
    parser.add_argument("--suiren-3d-atom-cache", default="")
    parser.add_argument("--valid-suiren-2d-graph-cache", default="")
    parser.add_argument("--valid-suiren-3d-graph-cache", default="")
    parser.add_argument("--valid-suiren-2d-atom-cache", default="")
    parser.add_argument("--valid-suiren-3d-atom-cache", default="")
    parser.add_argument("--test-suiren-2d-graph-cache", default="")
    parser.add_argument("--test-suiren-3d-graph-cache", default="")
    parser.add_argument("--test-suiren-2d-atom-cache", default="")
    parser.add_argument("--test-suiren-3d-atom-cache", default="")
    parser.add_argument("--suiren-cache-layout", choices=("rp_delta_abs", "rp_delta", "r_only"), default="rp_delta_abs")
    parser.add_argument("--trust-suiren-cache", action="store_true")
    parser.add_argument("--preload-suiren-graph-cache", action="store_true")
    parser.add_argument("--preload-suiren-atom-cache", action="store_true")
    parser.add_argument("--max-prediction-rows", type=int, default=20000)
    return parser.parse_args(argv)


def product_edit_schema(args: argparse.Namespace) -> tuple[str, int]:
    if args.geometry_mode == "irc_r":
        return "product_edit_r3d_bo", 2
    return "product_edit_r2d_bo", 1


def split_cache_args(args: argparse.Namespace, split: str) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    if split not in {"valid", "test"}:
        return out
    for stream in ("2d", "3d"):
        for level in ("graph", "atom"):
            split_key = f"{split}_suiren_{stream}_{level}_cache"
            base_key = f"suiren_{stream}_{level}_cache"
            value = getattr(args, split_key, "")
            if value:
                setattr(out, base_key, value)
    return out


def _main_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


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

    valid_args = split_cache_args(args, "valid")
    test_args = split_cache_args(args, "test")
    train_ds = product_edit_wldn.ProductEditDataset(args.train, args.max_train, args)
    valid_ds = product_edit_wldn.ProductEditDataset(args.valid, args.max_valid, valid_args, delta_vocab=train_ds.delta_vocab)
    test_ds = product_edit_wldn.ProductEditDataset(args.test, args.max_test, test_args, delta_vocab=train_ds.delta_vocab)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp_enabled() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=product_edit_wldn.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=product_edit_wldn.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=product_edit_wldn.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    input_schema, pair_raw_dim = product_edit_schema(args)
    model = WLDNProductEditPredictor(
        pair_raw_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        input_schema=input_schema,
        n_delta_classes=len(train_ds.delta_vocab),
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
    delta_class_weights = product_edit_wldn.class_weights(train_ds.delta_vocab, args, device)

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
        train_metrics = product_edit_wldn.train_one_epoch(train_model, train_loader, optimizer, device, args, train_ds.delta_vocab, delta_class_weights)
        stop_now = False
        if is_main(rank):
            valid_loss = product_edit_wldn.evaluate_loss(model, valid_loader, device, args, train_ds.delta_vocab, delta_class_weights)
            valid_metrics, _ = product_edit_wldn.evaluate(model, valid_loader, device, train_ds.delta_vocab, args, max_prediction_rows=0)
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
        valid_metrics, valid_rows = product_edit_wldn.evaluate(model, valid_loader, device, train_ds.delta_vocab, args, max_prediction_rows=args.max_prediction_rows)
        test_metrics, test_rows = product_edit_wldn.evaluate(model, test_loader, device, train_ds.delta_vocab, args, max_prediction_rows=args.max_prediction_rows)
        best = {
            "epoch": best_epoch,
            "seed": args.seed,
            "world_size": world,
            "pretrained_encoder": args.pretrained_encoder,
            "freeze_encoder": args.freeze_encoder,
            "input_schema": input_schema,
            "geometry_mode": args.geometry_mode,
            "top_k": args.top_k,
            "center_top_m": args.center_top_m,
            "candidate_pool_size": args.candidate_pool_size,
            "candidate_beam_size": args.candidate_beam_size,
            "edit_class_top_k": args.edit_class_top_k,
            "loss_weights": {
                "center": args.center_loss_weight,
                "delta": args.delta_loss_weight,
                "rank": args.rank_loss_weight,
                "changed_pair_pos_weight": args.changed_pair_pos_weight,
            },
            "delta_vocab": train_ds.delta_vocab.astype(float).tolist(),
            "delta_class_weights": delta_class_weights.detach().cpu().numpy().astype(float).tolist(),
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
            "sizes": {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)},
            "group_stats": {"train": train_ds.group_stats, "valid": valid_ds.group_stats, "test": test_ds.group_stats},
            "suiren_atom_dims": train_ds.suiren_atom_dims,
            "suiren_graph_dims": train_ds.suiren_graph_dims,
        }
        main_model = _main_model(model)
        torch.save(
            {
                "model": best_state,
                "config": vars(args),
                "delta_vocab": train_ds.delta_vocab,
                "model_class": "WLDNProductEditPredictor",
            },
            output_dir / "model.pt",
        )
        torch.save({"adapter": main_model.input_adapter.state_dict(), "config": vars(args)}, output_dir / "product_edit_adapter.pt")
        torch.save({"encoder": main_model.encoder.state_dict(), "config": vars(args), "model_class": "TokenSpaceReactionEncoder"}, output_dir / "reaction_encoder.pt")
        torch.save({"proposal_head": main_model.proposal_head.state_dict(), "config": vars(args), "delta_vocab": train_ds.delta_vocab}, output_dir / "product_edit_proposal_head.pt")
        torch.save({"candidate_ranker": main_model.candidate_ranker.state_dict(), "config": vars(args)}, output_dir / "product_edit_candidate_ranker.pt")
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
                "task": "WLDN-style R-only product graph-edit prediction",
                "input_policy": {
                    "uses_BO_R": True,
                    "uses_D_R_irc": args.geometry_mode == "irc_r",
                    "uses_BO_P_as_input": False,
                    "uses_Delta_BO_as_input": False,
                    "uses_D_P_irc_as_input": False,
                    "suiren_features_are_R_only": True,
                },
                "wldn_style": {
                    "reaction_center_proposal": True,
                    "legal_sparse_candidate_generation": True,
                    "difference_graph_candidate_ranking": True,
                    "true_product_inserted_for_ranker_training": True,
                },
                "delta_vocab": best["delta_vocab"],
                "group_stats": best["group_stats"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_csv(output_dir / "predictions_valid.csv", valid_rows)
        write_csv(output_dir / "predictions_test.csv", test_rows)
        product_edit_wldn.write_summary(output_dir, best)
        append_registry(
            args,
            output_dir,
            stage="product_edit_prediction",
            task="r_only_wldn_product_graph_edit",
            model="WLDNProductEditPredictor",
            status="complete",
            notes=(
                f"input_schema={input_schema}; geometry_mode={args.geometry_mode}; "
                f"center_top_m={args.center_top_m}; candidate_pool_size={args.candidate_pool_size}; "
                f"selection=minimum valid_loss; optimizer={args.optimizer}"
            ),
        )
        print(json.dumps(best, indent=2, ensure_ascii=False), flush=True)
    cleanup_ddp()


if __name__ == "__main__":
    main()
