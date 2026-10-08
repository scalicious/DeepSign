"""Create leakage-safe block-based splits (splits.json) for cross-domain experiments."""

import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List
import pandas as pd
from aquasign.utils import load_config, save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_splits(config_path: str = "configs/base.yaml", split_seed: int = 1234):
    cfg = load_config(config_path)
    index_csv = Path(cfg["paths"]["index_csv"])
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing {index_csv}. Run build_index.py first.")

    df = pd.read_csv(index_csv)
    block_size = cfg.get("block_size", 50)

    # Deterministic RNG for block splitting
    rng = random.Random(split_seed)

    # We partition each (scenario, class_id) sequence into contiguous blocks of block_size
    # Each sample belongs to a unique block_id: f"{scenario}_{class_id}_b{b_idx}"
    blocks = {}  # block_id -> list of sample_ids (and pair_ids)
    block_meta = {}  # block_id -> {"domain": domain, "scenario": scenario, "class_id": class_id}

    for (scenario, class_id), group in df.groupby(["scenario", "class_id"]):
        sorted_group = group.sort_values(by="order_key").reset_index(drop=True)
        n_samples = len(sorted_group)
        n_blocks = math.ceil(n_samples / block_size)
        domain = sorted_group["domain"].iloc[0]

        for b_idx in range(n_blocks):
            start = b_idx * block_size
            end = min(start + block_size, n_samples)
            b_slice = sorted_group.iloc[start:end]
            b_id = f"{scenario}_{class_id}_b{b_idx:03d}"
            blocks[b_id] = b_slice["sample_id"].tolist()
            block_meta[b_id] = {
                "domain": domain,
                "scenario": scenario,
                "class_id": int(class_id),
                "n_samples": len(b_slice),
            }

    # Now assign blocks to splits per domain and per class
    # To support both directions (pool2sea and sea2pool), we create split partitions for BOTH domains:
    # For a domain when used as SOURCE:
    #   train 70%, val 10%, calib 10%, test 10%
    # For a domain when used as TARGET:
    #   target_train 50%, target_test 50%

    splits_by_domain = {"pool": {}, "sea": {}}

    for domain in ["pool", "sea"]:
        dom_blocks = [b_id for b_id, meta in block_meta.items() if meta["domain"] == domain]
        
        # Group by class_id within domain to ensure class representation across splits
        dom_class_blocks = {}
        for b_id in dom_blocks:
            c = block_meta[b_id]["class_id"]
            dom_class_blocks.setdefault(c, []).append(b_id)

        src_train_b, src_val_b, src_calib_b, src_test_b = [], [], [], []
        tgt_train_b, tgt_test_b = [], []

        for c, c_b_ids in sorted(dom_class_blocks.items()):
            # Shuffle blocks deterministically
            shuffled = list(c_b_ids)
            rng.shuffle(shuffled)
            k = len(shuffled)

            # Target split: 50 / 50
            if k == 1:
                tgt_train_b.extend(shuffled)
            else:
                n_tgt_train = max(1, round(k * 0.5))
                tgt_train_b.extend(shuffled[:n_tgt_train])
                tgt_test_b.extend(shuffled[n_tgt_train:])

            # Source split: train 70 / val 10 / calib 10 / test 10
            if k < 4:
                logger.warning(
                    f"Domain {domain}, class {c} has only {k} blocks (<4). Assigning all blocks to train."
                )
                src_train_b.extend(shuffled)
            else:
                n_val = max(1, round(k * 0.1))
                n_calib = max(1, round(k * 0.1))
                n_test = max(1, round(k * 0.1))
                n_train = k - (n_val + n_calib + n_test)
                if n_train < 1:
                    n_train = 1
                    # adjust others
                    rem = k - n_train
                    n_val = max(1, rem // 3)
                    n_calib = max(1, rem // 3)
                    n_test = max(1, rem - n_val - n_calib)

                train_slice = shuffled[:n_train]
                val_slice = shuffled[n_train : n_train + n_val]
                calib_slice = shuffled[n_train + n_val : n_train + n_val + n_calib]
                test_slice = shuffled[n_train + n_val + n_calib :]

                src_train_b.extend(train_slice)
                src_val_b.extend(val_slice)
                src_calib_b.extend(calib_slice)
                src_test_b.extend(test_slice)

        # Convert block IDs to sample IDs
        def to_samples(b_list):
            res = []
            for b in b_list:
                res.extend(blocks[b])
            return sorted(list(set(res)))

        def to_block_list(b_list):
            return sorted(list(set(b_list)))

        splits_by_domain[domain] = {
            "source": {
                "train_blocks": to_block_list(src_train_b),
                "val_blocks": to_block_list(src_val_b),
                "calib_blocks": to_block_list(src_calib_b),
                "test_blocks": to_block_list(src_test_b),
                "train": to_samples(src_train_b),
                "val": to_samples(src_val_b),
                "calib": to_samples(src_calib_b),
                "test": to_samples(src_test_b),
            },
            "target": {
                "target_train_blocks": to_block_list(tgt_train_b),
                "target_test_blocks": to_block_list(tgt_test_b),
                "target_train": to_samples(tgt_train_b),
                "target_test": to_samples(tgt_test_b),
            },
        }

    # Store full output
    output = {
        "split_seed": split_seed,
        "block_size": block_size,
        "pool": splits_by_domain["pool"],
        "sea": splits_by_domain["sea"],
    }

    out_file = Path(cfg["paths"]["splits_json"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, str(out_file))
    print(f"Saved block splits to {out_file}")

    for domain in ["pool", "sea"]:
        src = splits_by_domain[domain]["source"]
        tgt = splits_by_domain[domain]["target"]
        print(f"\n[{domain.upper()} Domain Split Summary]")
        print(f"  Source Train: {len(src['train'])} samples ({len(src['train_blocks'])} blocks)")
        print(f"  Source Val:   {len(src['val'])} samples ({len(src['val_blocks'])} blocks)")
        print(f"  Source Calib: {len(src['calib'])} samples ({len(src['calib_blocks'])} blocks)")
        print(f"  Source Test:  {len(src['test'])} samples ({len(src['test_blocks'])} blocks)")
        print(f"  Target Train: {len(tgt['target_train'])} samples ({len(tgt['target_train_blocks'])} blocks)")
        print(f"  Target Test:  {len(tgt['target_test'])} samples ({len(tgt['target_test_blocks'])} blocks)")


if __name__ == "__main__":
    make_splits()
