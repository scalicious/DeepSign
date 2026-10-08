"""Stage 1: Training classifier with optional correction and domain alignment per spec Section 7.1."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
# BUG FIX: torch.cuda.amp.* is deprecated in PyTorch 2.x; use torch.amp.*
from torch.amp import GradScaler, autocast
from aquasign.data.dataset import get_dataloaders
from aquasign.losses import domain_alignment_loss
from aquasign.metrics import compute_classification_metrics
from aquasign.models.aquasign_net import AquaSignNet
from aquasign.utils import CSVLogger, load_config, save_checkpoint, save_json, seed_everything


def train_stage1(
    config_path: str,
    corr: int = 0,
    align: int = 0,
    seed: int = 0,
    resume: bool = False,
    device_name: Optional[str] = None,
    epochs_override: Optional[int] = None,
):
    # 1. Deterministic seeding
    seed_everything(seed)
    cfg = load_config(config_path)

    # Resolve device
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    direction = cfg.get("direction", "pool2sea")
    run_dir = Path(cfg["paths"]["results_dir"]) / "runs" / f"{direction}_c{corr}a{align}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved run config
    run_cfg = dict(cfg)
    run_cfg.update({"corr": corr, "align": align, "seed": seed, "device": str(device)})
    save_json(run_cfg, str(run_dir / "config.json"))

    # 2. Data loaders
    loaders, meta = get_dataloaders(config_path, direction=direction)
    src_train_loader = loaders["src_train"]
    src_val_loader = loaders["src_val"]
    tgt_train_loader = loaders["tgt_train"]

    num_classes = meta["num_classes"]
    class_weights = meta["class_weights"].to(device)
    norm_mean = meta["norm_mean"]
    norm_std = meta["norm_std"]

    # 3. Model construction
    model = AquaSignNet(
        num_classes=num_classes,
        use_correction=(corr == 1),
        norm_mean=tuple(norm_mean.tolist()),
        norm_std=tuple(norm_std.tolist()),
    ).to(device)

    # Criterion
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    lambda_align = cfg.get("alignment", {}).get("lambda_align", 1.0)

    # Optimizer & Scheduler
    lr = float(cfg.get("lr", 1e-3))
    min_lr = float(cfg.get("min_lr", 1e-5))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    epochs = epochs_override if epochs_override is not None else int(cfg.get("epochs", 30))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    use_amp = cfg.get("amp", True) and device.type == "cuda"
    # GradScaler needs the device type explicitly in PyTorch 2.x
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # Logging
    log_file = run_dir / "log.csv"
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_ce_loss",
        "train_align_loss",
        "val_loss",
        "val_acc",
        "val_macro_f1",
    ]
    logger = CSVLogger(str(log_file), fieldnames=fieldnames)

    # Resume if requested
    start_epoch = 1
    best_val_f1 = -1.0
    last_pt = run_dir / "last.pt"
    if resume and last_pt.exists():
        print(f"Resuming from checkpoint {last_pt}...")
        ckpt = torch.load(last_pt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)

    # 4. Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_ce_total = 0.0
        train_align_total = 0.0
        n_steps = 0

        tgt_iter = iter(tgt_train_loader) if align == 1 else None

        for src_x, src_y, _ in src_train_loader:
            src_x = src_x.to(device)
            src_y = src_y.to(device)

            optimizer.zero_grad()

            if align == 1:
                # Draw target batch
                try:
                    tgt_x, _, _ = next(tgt_iter)
                except (StopIteration, TypeError):
                    tgt_iter = iter(tgt_train_loader)
                    tgt_x, _, _ = next(tgt_iter)
                tgt_x = tgt_x.to(device)

                # Concatenate and run ONE joint forward pass per spec 7.1
                joint_x = torch.cat([src_x, tgt_x], dim=0)

                with autocast(device_type=device.type, enabled=use_amp):
                    out = model(joint_x, return_features=True)
                    logits = out["logits"]
                    features = out["features"]

                    n_src = src_x.size(0)
                    logits_src = logits[:n_src]
                    f_src = features[:n_src]
                    f_tgt = features[n_src:]

                    ce_loss = ce_loss_fn(logits_src, src_y)
                    align_loss = domain_alignment_loss(f_src, f_tgt)
                    loss = ce_loss + lambda_align * align_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_ce_total += ce_loss.item()
                train_align_total += align_loss.item()
            else:
                with autocast(device_type=device.type, enabled=use_amp):
                    out = model(src_x, return_features=False)
                    loss = ce_loss_fn(out["logits"], src_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_ce_total += loss.item()
                train_align_total += 0.0

            train_loss_total += loss.item()
            n_steps += 1

        scheduler.step()

        avg_train_loss = train_loss_total / max(n_steps, 1)
        avg_ce_loss = train_ce_total / max(n_steps, 1)
        avg_align_loss = train_align_total / max(n_steps, 1)

        # Validation on source_val
        model.eval()
        val_loss_total = 0.0
        val_preds = []
        val_targets = []
        val_steps = 0

        with torch.no_grad():
            for vx, vy, _ in src_val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    vout = model(vx, return_features=False)
                    vloss = ce_loss_fn(vout["logits"], vy)

                val_loss_total += vloss.item()
                preds = torch.argmax(vout["logits"], dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(vy.cpu().numpy())
                val_steps += 1

        avg_val_loss = val_loss_total / max(val_steps, 1)
        val_metrics = compute_classification_metrics(np.array(val_targets), np.array(val_preds), num_classes=num_classes)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        current_lr = scheduler.get_last_lr()[0]
        logger.log({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": avg_train_loss,
            "train_ce_loss": avg_ce_loss,
            "train_align_loss": avg_align_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}, Align: {avg_align_loss:.4f}) "
            f"| Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "config": run_cfg,
            # BUG FIX: save norm stats so evaluate.py can load them without
            # re-scanning the entire train dataset on each evaluation call.
            "norm_mean": norm_mean.tolist(),
            "norm_std": norm_std.tolist(),
        }
        # Save last checkpoint every epoch (for resume capability)
        save_checkpoint(state, str(run_dir), filename="last.pt")

        # Save best checkpoint by source-val macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            state["best_val_f1"] = best_val_f1
            save_checkpoint(state, str(run_dir), filename="best.pt")
            print(f"  -> New best source-val macro-F1: {best_val_f1:.4f} saved to best.pt")

    print(f"Training finished. Checkpoints and logs saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaSign Stage 1 Training")
    parser.add_argument("--config", type=str, default="configs/pool2sea.yaml")
    parser.add_argument("--corr", type=int, choices=[0, 1], default=0)
    parser.add_argument("--align", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_stage1(
        config_path=args.config,
        corr=args.corr,
        align=args.align,
        seed=args.seed,
        resume=args.resume,
        epochs_override=args.epochs,
    )
