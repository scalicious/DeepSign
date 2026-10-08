"""Evaluation script: evaluate checkpoint on source-test and target-test with optional AdaBN."""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from aquasign.adabn import adapt_bn
from aquasign.data.dataset import get_dataloaders
from aquasign.metrics import compute_classification_metrics, compute_domain_gap, save_confusion_matrix
from aquasign.models.aquasign_net import AquaSignNet
from aquasign.utils import load_config, save_json


def evaluate_split(model: nn.Module, loader, device: torch.device, num_classes: int):
    """Run model on loader and return true and predicted labels."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            out = model(x)
            preds = torch.argmax(out["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
    return metrics, y_true, y_pred


def evaluate_run(
    checkpoint_path: str,
    adabn: bool = False,
    device_name: Optional[str] = None,
) -> Dict[str, float]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    direction = cfg.get("direction", "pool2sea")

    # BUG FIX: read num_classes from classes.json (via config path) instead of
    # hardcoding 15 — stays correct if the class filter threshold ever changes.
    classes_path = Path(cfg.get("paths", {}).get("classes_json", "classes.json"))
    if not classes_path.exists():
        classes_path = Path("classes.json")
    with open(classes_path, "r") as f:
        num_classes = len(json.load(f))

    # Data loaders (only need test splits; avoid expensive stat recompute)
    loaders, _ = get_dataloaders("configs/base.yaml", direction=direction)
    src_test_loader = loaders["src_test"]
    tgt_train_loader = loaders["tgt_train"]
    tgt_test_loader = loaders["tgt_test"]

    # BUG FIX: restore norm stats from checkpoint so the model's internal
    # normalization buffer matches what was used during training, without
    # re-scanning 3 000+ images.
    norm_mean = ckpt.get("norm_mean", [0.485, 0.456, 0.406])
    norm_std  = ckpt.get("norm_std",  [0.229, 0.224, 0.225])

    # Model
    model = AquaSignNet(
        num_classes=num_classes,
        use_correction=(cfg.get("corr", 0) == 1),
        norm_mean=tuple(norm_mean),
        norm_std=tuple(norm_std),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Source Test Evaluation
    src_metrics, src_true, src_pred = evaluate_split(model, src_test_loader, device, num_classes)

    # 2. Target Test Evaluation (with copy if AdaBN enabled)
    eval_model = model
    if adabn:
        eval_model = copy.deepcopy(model)
        eval_model = adapt_bn(eval_model, tgt_train_loader, device=device)

    tgt_metrics, tgt_true, tgt_pred = evaluate_split(eval_model, tgt_test_loader, device, num_classes)

    # Domain gap
    gap = compute_domain_gap(src_metrics["macro_f1"], tgt_metrics["macro_f1"])

    results = {
        "direction": direction,
        "corr": cfg.get("corr", 0),
        "align": cfg.get("align", 0),
        "adabn": bool(adabn),
        "seed": cfg.get("seed", 0),
        "src_f1": src_metrics["macro_f1"],
        "src_acc": src_metrics["accuracy"],
        "tgt_f1": tgt_metrics["macro_f1"],
        "tgt_acc": tgt_metrics["accuracy"],
        "gap": gap,
    }

    # Save metrics JSON beside checkpoint
    out_name = f"eval_adabn_{'on' if adabn else 'off'}.json"
    save_json(results, str(ckpt_path.parent / out_name))
    print(f"\n--- Evaluation Results ({direction}, AdaBN: {'ON' if adabn else 'OFF'}) ---")
    print(f"Source Test F1:  {results['src_f1']:.4f} | Acc: {results['src_acc']:.4f}")
    print(f"Target Test F1:  {results['tgt_f1']:.4f} | Acc: {results['tgt_acc']:.4f}")
    print(f"Domain Gap (F1): {results['gap']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaSign Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--adabn", action="store_true")
    args = parser.parse_args()

    evaluate_run(checkpoint_path=args.checkpoint, adabn=args.adabn)
