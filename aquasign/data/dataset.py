"""Dataset loading, split indexing, normalization stats, and dual-domain loaders."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from aquasign.data.augment import AquaSignAugment
from aquasign.utils import load_config


class AquaSignDataset(Dataset):
    """Dataset for AquaSign hand gestures.

    Loads from cached uint8 .npy if available, falling back to disk files.
    Returns x (float32 tensor in [0, 1]), y (int class_id), and sample_id (str).
    """

    def __init__(
        self,
        domain: str,
        split_samples: List[str],
        config: Dict[str, Any],
        is_train: bool = False,
        augment_fn: Optional[Any] = None,
    ):
        self.domain = domain
        self.is_train = is_train
        self.augment_fn = augment_fn
        self.h, self.w = config["input_hw"]

        cache_dir = Path(config["paths"]["cache_dir"])
        hw_tag = f"{self.h}x{self.w}"
        if config.get("use_roi", False):
            hw_tag += "_roi"

        npy_path = cache_dir / f"{domain}_{hw_tag}.npy"
        csv_path = cache_dir / f"index_cached_{domain}_{hw_tag}.csv"

        if npy_path.exists() and csv_path.exists():
            # Load from fast npy cache
            # BUG FIX: keep the mmap alive as self._full_npy and index at __getitem__
            # time.  Fancy-indexing a mmap (full_npy[indices]) forces a RAM copy;
            # storing just the integer index array is O(N_split) not O(N_total).
            self.cached_df = pd.read_csv(csv_path)
            split_set = set(split_samples)
            mask = self.cached_df["sample_id"].isin(split_set)
            self.df = self.cached_df[mask].copy().reset_index(drop=True)
            self._cache_indices = self.df["cache_idx"].values.astype(np.int64)
            self._full_npy = np.load(npy_path, mmap_mode="r")  # kept open, not copied
            self.use_cache = True
        else:
            # Fallback to index.csv
            index_df = pd.read_csv(config["paths"]["index_csv"])
            split_set = set(split_samples)
            self.df = index_df[(index_df["domain"] == domain) & (index_df["sample_id"].isin(split_set))].copy().reset_index(drop=True)
            self.use_cache = False
            self._full_npy = None
            self._cache_indices = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        y = int(row["class_id"])

        if self.use_cache:
            # Index the mmap at item time — no full copy into RAM
            cache_idx = int(self._cache_indices[idx])
            img_arr = self._full_npy[cache_idx]  # uint8 (3, H, W), read from disk/OS cache
            x = torch.from_numpy(np.array(img_arr, dtype=np.uint8)).float() / 255.0
        else:
            # Load from disk
            p = Path(row["left_path"])
            with Image.open(p) as img:
                img = img.convert("RGB").resize((self.w, self.h), Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
                x = torch.from_numpy(arr).float() / 255.0

        if self.is_train and self.augment_fn is not None:
            x = self.augment_fn(x)

        return x, y, sample_id


def compute_source_stats(dataset: AquaSignDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean and std over source-train split on RAW pixels (no augmentation).

    BUG FIX: must use a no-augmentation view of the dataset so the stats reflect
    the true pixel distribution, not a distribution distorted by random crops /
    brightness jitter.
    """
    # Temporarily disable augmentation for this pass
    old_is_train = dataset.is_train
    old_augment = dataset.augment_fn
    dataset.is_train = False
    dataset.augment_fn = None

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0

    for x, _, _ in loader:
        b, c, h, w = x.shape
        channel_sum += x.sum(dim=[0, 2, 3])
        channel_sq_sum += (x ** 2).sum(dim=[0, 2, 3])
        total_pixels += b * h * w

    # Restore
    dataset.is_train = old_is_train
    dataset.augment_fn = old_augment

    mean = channel_sum / total_pixels
    std = torch.sqrt(torch.clamp((channel_sq_sum / total_pixels) - (mean ** 2), min=1e-8))
    return mean, std


def compute_class_weights(dataset: AquaSignDataset, num_classes: int) -> torch.Tensor:
    """Compute class weights: w_c = (1 / n_c) / mean(1 / n_c)."""
    counts = np.zeros(num_classes, dtype=np.float32)
    labels = dataset.df["class_id"].values
    for y in labels:
        counts[y] += 1

    inv_counts = np.zeros_like(counts)
    for c in range(num_classes):
        if counts[c] > 0:
            inv_counts[c] = 1.0 / counts[c]
        else:
            inv_counts[c] = 0.0

    mean_inv = np.mean(inv_counts[counts > 0]) if np.any(counts > 0) else 1.0
    weights = inv_counts / mean_inv
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(config_path: str, direction: Optional[str] = None) -> Dict[str, Any]:
    """Construct data loaders for source and target splits based on direction config."""
    cfg = load_config(config_path)
    if direction:
        cfg["direction"] = direction
        if direction == "pool2sea":
            cfg["source_domain"] = "pool"
            cfg["target_domain"] = "sea"
        elif direction == "sea2pool":
            cfg["source_domain"] = "sea"
            cfg["target_domain"] = "pool"

    src_dom = cfg.get("source_domain", "pool")
    tgt_dom = cfg.get("target_domain", "sea")

    with open(cfg["paths"]["splits_json"], "r") as f:
        splits = json.load(f)

    with open(cfg["paths"]["classes_json"], "r") as f:
        classes = json.load(f)
    num_classes = len(classes)

    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 2)

    # Augmentation
    augment_fn = AquaSignAugment(input_hw=cfg["input_hw"])

    # Source datasets
    src_splits = splits[src_dom]["source"]
    src_train_ds = AquaSignDataset(src_dom, src_splits["train"], cfg, is_train=True, augment_fn=augment_fn)
    src_val_ds = AquaSignDataset(src_dom, src_splits["val"], cfg, is_train=False)
    src_calib_ds = AquaSignDataset(src_dom, src_splits["calib"], cfg, is_train=False)
    src_test_ds = AquaSignDataset(src_dom, src_splits["test"], cfg, is_train=False)

    # Target datasets
    # BUG FIX: target_train is used for domain alignment (C3) and AdaBN (C2).
    # It must NOT receive crop/jitter augmentation — those would distort the
    # feature distribution that CORAL / AdaBN are trying to align.  Only the
    # haze augmentation (applied inside sim/haze.py by the trainer) is allowed.
    tgt_splits = splits[tgt_dom]["target"]
    tgt_train_ds = AquaSignDataset(tgt_dom, tgt_splits["target_train"], cfg, is_train=False)
    tgt_test_ds = AquaSignDataset(tgt_dom, tgt_splits["target_test"], cfg, is_train=False)

    # Dataloaders
    loaders = {
        "src_train": DataLoader(src_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "src_val": DataLoader(src_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "src_calib": DataLoader(src_calib_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "src_test": DataLoader(src_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "tgt_train": DataLoader(tgt_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "tgt_test": DataLoader(tgt_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    # Meta
    class_weights = compute_class_weights(src_train_ds, num_classes)
    mean, std = compute_source_stats(src_train_ds)

    meta = {
        "num_classes": num_classes,
        "class_weights": class_weights,
        "norm_mean": mean,
        "norm_std": std,
        "source_domain": src_dom,
        "target_domain": tgt_dom,
    }

    return loaders, meta
