"""Build index.csv and classes.json from raw dataset annotations."""

import ast
import json
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from aquasign.utils import load_config, save_json

POOL_SCENARIOS = {"brodarski-A", "brodarski-B", "brodarski-C", "brodarski-D", "genova-A"}
SEA_SCENARIOS = {"biograd-A", "biograd-B", "biograd-C"}


def parse_roi(roi_str: Optional[str]) -> Optional[str]:
    """Parse '[x, y, w, h]' string to '[x1, y1, x2, y2]' string or None."""
    if not roi_str or pd.isna(roi_str) or str(roi_str).strip() == "":
        return None
    try:
        val = ast.literal_eval(str(roi_str).strip())
        if isinstance(val, (list, tuple)) and len(val) == 4:
            x, y, w, h = val
            return f"[{x},{y},{x+w},{y+h}]"
    except Exception:
        pass
    return None


def extract_order_key(filename: str) -> int:
    """Extract integer sequence index from filename (e.g. 'biograd-A_00012_left.jpg' -> 12)."""
    base = os.path.basename(filename).replace("_left.jpg", "").replace("_right.jpg", "")
    parts = base.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def build_index(config_path: str = "configs/base.yaml"):
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if not raw_dir.exists():
        raw_dir = Path("data/raw")

    tp_csv = raw_dir / "CADDY_gestures_all_true_positives_release_v2.csv"
    print(f"Reading raw positive annotations from {tp_csv}...")
    df_tp = pd.read_csv(tp_csv)
    # Filter synthetic == 0
    raw_tp = df_tp[df_tp["synthetic"] == 0].copy()

    # Determine domain
    def assign_domain(scenario):
        if scenario in POOL_SCENARIOS:
            return "pool"
        elif scenario in SEA_SCENARIOS:
            return "sea"
        return "unknown"

    raw_tp["domain"] = raw_tp["scenario"].apply(assign_domain)
    raw_tp = raw_tp[raw_tp["domain"].isin(["pool", "sea"])].copy()

    # Check class overlap with threshold
    min_per_domain = cfg.get("min_per_domain", 40)
    crosstab = pd.crosstab(raw_tp["domain"], raw_tp["label name"])
    
    surviving_classes = []
    dropped_classes = []
    for c in sorted(raw_tp["label name"].unique()):
        p_cnt = crosstab.loc["pool", c] if "pool" in crosstab.index and c in crosstab.columns else 0
        s_cnt = crosstab.loc["sea", c] if "sea" in crosstab.index and c in crosstab.columns else 0
        if p_cnt >= min_per_domain and s_cnt >= min_per_domain:
            surviving_classes.append(c)
        else:
            dropped_classes.append((c, p_cnt, s_cnt))

    print(f"Retained {len(surviving_classes)} classes:")
    for c in surviving_classes:
        print(f"  - {c} (pool: {crosstab.loc['pool', c]}, sea: {crosstab.loc['sea', c]})")
    if dropped_classes:
        print("Dropped classes failing min_per_domain:")
        for c, p, s in dropped_classes:
            print(f"  - {c} (pool: {p}, sea: {s})")

    # Map class names to 0..N-1
    class_to_id = {name: idx for idx, name in enumerate(sorted(surviving_classes))}
    save_json(class_to_id, cfg["paths"]["classes_json"])

    # Filter df to retained classes
    filtered_df = raw_tp[raw_tp["label name"].isin(class_to_id)].copy()

    # Construct records
    records = []
    for idx, row in filtered_df.iterrows():
        left_rel = row["stereo left"].lstrip("/")
        right_rel = row["stereo right"].lstrip("/")
        left_path = str(raw_dir / left_rel)
        right_path = str(raw_dir / right_rel)
        
        pair_id = os.path.basename(left_rel).replace("_left.jpg", "")
        order_key = extract_order_key(left_rel)
        roi_xyxy = parse_roi(row.get("roi left"))

        records.append({
            "sample_id": f"{row['scenario']}_{order_key:05d}",
            "pair_id": pair_id,
            "scenario": row["scenario"],
            "domain": row["domain"],
            "class_id": class_to_id[row["label name"]],
            "class_name": row["label name"],
            "left_path": left_path,
            "right_path": right_path,
            "order_key": order_key,
            "roi_xyxy": roi_xyxy,
        })

    index_df = pd.DataFrame(records)
    # Sort deterministically
    index_df = index_df.sort_values(by=["scenario", "class_id", "order_key"]).reset_index(drop=True)

    out_csv = Path(cfg["paths"]["index_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(out_csv, index=False)
    print(f"Successfully wrote {len(index_df)} rows to {out_csv}")


if __name__ == "__main__":
    build_index()
