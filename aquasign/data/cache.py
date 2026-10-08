"""Cache preprocessed raw images into uint8 numpy arrays (N, 3, H, W)."""

import ast
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from aquasign.utils import load_config


def crop_roi(image: Image.Image, roi_xyxy_str: Optional[str], margin: float = 0.2) -> Image.Image:
    """Crop hand ROI with 20% margin if present."""
    if not roi_xyxy_str or pd.isna(roi_xyxy_str):
        return image
    try:
        x1, y1, x2, y2 = ast.literal_eval(str(roi_xyxy_str))
        w = x2 - x1
        h = y2 - y1
        img_w, img_h = image.size

        # Expand margin
        mx = int(w * margin)
        my = int(h * margin)
        nx1 = max(0, x1 - mx)
        ny1 = max(0, y1 - my)
        nx2 = min(img_w, x2 + mx)
        ny2 = min(img_h, y2 + my)

        if nx2 > nx1 and ny2 > ny1:
            return image.crop((nx1, ny1, nx2, ny2))
    except Exception:
        pass
    return image


def build_cache(config_path: str = "configs/base.yaml", domain: Optional[str] = None):
    cfg = load_config(config_path)
    index_csv = Path(cfg["paths"]["index_csv"])
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing {index_csv}. Run build_index.py first.")

    df = pd.read_csv(index_csv)
    h, w = cfg["input_hw"]
    use_roi = cfg.get("use_roi", False)
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    domains = [domain] if domain else ["pool", "sea"]

    for dom in domains:
        dom_df = df[df["domain"] == dom].copy().reset_index(drop=True)
        n = len(dom_df)
        print(f"Caching domain '{dom}': {n} images to ({h}x{w})...")

        cache_arr = np.empty((n, 3, h, w), dtype=np.uint8)
        valid_indices = []

        for i, row in enumerate(dom_df.itertuples()):
            img_path = Path(row.left_path)
            if not img_path.exists():
                print(f"Warning: file {img_path} not found.")
                continue

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if use_roi:
                    img = crop_roi(img, row.roi_xyxy)
                img = img.resize((w, h), Image.Resampling.BILINEAR)
                # (H, W, 3) -> (3, H, W)
                arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
                cache_arr[i] = arr
                valid_indices.append(i)

        if len(valid_indices) < n:
            cache_arr = cache_arr[valid_indices]
            dom_df = dom_df.iloc[valid_indices].reset_index(drop=True)

        hw_tag = f"{h}x{w}"
        if use_roi:
            hw_tag += "_roi"
        npy_out = cache_dir / f"{dom}_{hw_tag}.npy"
        csv_out = cache_dir / f"index_cached_{dom}_{hw_tag}.csv"

        np.save(npy_out, cache_arr)
        dom_df["cache_idx"] = np.arange(len(dom_df))
        dom_df.to_csv(csv_out, index=False)
        print(f"Saved {dom} cache to {npy_out} (shape: {cache_arr.shape}) and {csv_out}")


if __name__ == "__main__":
    build_cache()
