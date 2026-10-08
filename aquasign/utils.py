"""Utility functions: deterministic seeding, config parsing, checkpointing, and logging."""

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import yaml


def seed_everything(seed: int = 1234) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config with optional _base_ inheritance."""
    config_file = Path(config_path)
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "_base_" in cfg:
        base_rel = cfg.pop("_base_")
        if (config_file.parent / base_rel).exists():
            base_path = config_file.parent / base_rel
        elif Path(base_rel).exists():
            base_path = Path(base_rel)
        else:
            raise FileNotFoundError(f"Base config {base_rel} not found from {config_file}")
        base_cfg = load_config(str(base_path))
        merged = deep_merge(base_cfg, cfg)
        return merged
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    res = dict(base)
    for k, v in override.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = deep_merge(res[k], v)
        else:
            res[k] = v
    return res


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    """Save model checkpoint safely."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    target_path = Path(checkpoint_dir) / filename
    temp_path = Path(checkpoint_dir) / f"{filename}.tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, target_path)
    return str(target_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint into model and optionally optimizer/scheduler."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


class CSVLogger:
    """Logs training metrics to a CSV file row by row."""

    def __init__(self, log_path: str, fieldnames: list):
        self.log_path = Path(log_path)
        self.fieldnames = fieldnames
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_json(data: Any, path: str) -> None:
    """Save data to JSON with indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
