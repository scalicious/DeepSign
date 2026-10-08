"""Unit tests for leakage-free block-based splits."""

import json
from pathlib import Path
import pandas as pd
import pytest


def test_splits_no_leak():
    splits_file = Path("data/splits.json")
    index_file = Path("data/index.csv")

    assert splits_file.exists(), "data/splits.json not found"
    assert index_file.exists(), "data/index.csv not found"

    with open(splits_file, "r") as f:
        splits = json.load(f)

    df = pd.read_csv(index_file)
    sample_to_pair = dict(zip(df["sample_id"], df["pair_id"]))
    sample_to_class = dict(zip(df["sample_id"], df["class_id"]))

    for domain in ["pool", "sea"]:
        src = splits[domain]["source"]
        tgt = splits[domain]["target"]

        # 1. Source blocks: pairwise check between train, val, calib, test
        src_block_names = ["train_blocks", "val_blocks", "calib_blocks", "test_blocks"]
        for i in range(len(src_block_names)):
            for j in range(i + 1, len(src_block_names)):
                s1, s2 = src_block_names[i], src_block_names[j]
                b1 = set(src[s1])
                b2 = set(src[s2])
                overlap = b1.intersection(b2)
                assert len(overlap) == 0, f"{domain}: {s1} and {s2} share blocks: {overlap}"

        # 2. Source samples & pairs: pairwise check between train, val, calib, test
        src_splits = ["train", "val", "calib", "test"]
        for i in range(len(src_splits)):
            for j in range(i + 1, len(src_splits)):
                s1, s2 = src_splits[i], src_splits[j]
                p1 = {sample_to_pair[s] for s in src[s1]}
                p2 = {sample_to_pair[s] for s in src[s2]}
                overlap = p1.intersection(p2)
                assert len(overlap) == 0, f"{domain}: {s1} and {s2} share stereo pairs: {overlap}"

        # 3. Target train vs Target test
        tgt_train_b = set(tgt["target_train_blocks"])
        tgt_test_b = set(tgt["target_test_blocks"])
        overlap_b = tgt_train_b.intersection(tgt_test_b)
        assert len(overlap_b) == 0, f"{domain}: target train/test share blocks: {overlap_b}"

        tgt_train_p = {sample_to_pair[s] for s in tgt["target_train"]}
        tgt_test_p = {sample_to_pair[s] for s in tgt["target_test"]}
        overlap_p = tgt_train_p.intersection(tgt_test_p)
        assert len(overlap_p) == 0, f"{domain}: target train/test share stereo pairs: {overlap_p}"
