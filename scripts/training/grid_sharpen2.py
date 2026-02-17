#!/usr/bin/env python3
"""Phase 2: Test promising sharpening combinations."""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.training.train_arbiter import (
    DATASET_DIR, CLASS_METERS, TOTAL_FEATURES, N_CLASSES,
    build_feature_matrix,
)
from scripts.training.grid_sharpen import train_and_eval


def main():
    with open(DATASET_DIR / "tuning.json") as f:
        tuning_data = json.load(f)
    with open(DATASET_DIR / "test.json") as f:
        test_data = json.load(f)

    by_meter = defaultdict(list)
    for entry in tuning_data:
        by_meter[entry["expected"]].append(entry)

    train_data, val_data = [], []
    for meter, items in sorted(by_meter.items()):
        np.random.seed(42)
        np.random.shuffle(items)
        split_idx = int(len(items) * 0.8)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])

    # Promising combos from Phase 1 winners
    configs = [
        {},  # baseline
        {"autocorr": 2.0},
        {"onset_mlp": 2.0},
        {"onset_mlp": 3.0},
        {"hcdf": 1.5},
        {"autocorr": 2.0, "onset_mlp": 2.0},
        {"autocorr": 2.0, "onset_mlp": 3.0},
        {"autocorr": 2.0, "hcdf": 1.5},
        {"onset_mlp": 2.0, "hcdf": 1.5},
        {"onset_mlp": 3.0, "hcdf": 1.5},
        {"autocorr": 2.0, "onset_mlp": 2.0, "hcdf": 1.5},
        {"autocorr": 2.0, "onset_mlp": 3.0, "hcdf": 1.5},
        {"autocorr": 2.0, "beat_this": 2.0},
        {"autocorr": 2.0, "bar_tracking": 3.0},
        {"autocorr": 2.0, "beat_this": 2.0, "onset_mlp": 2.0},
        {"autocorr": 2.0, "beat_this": 2.0, "hcdf": 1.5},
        # All sharpened
        {"autocorr": 2.0, "onset_mlp": 2.0, "hcdf": 1.5, "beat_this": 2.0},
        {"autocorr": 2.0, "onset_mlp": 3.0, "hcdf": 1.5, "bar_tracking": 3.0},
        # Multi-seed for top configs (run with 3 seeds, take average)
    ]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"\n{'='*90}")
    print(f"Testing {len(configs)} sharpening configurations")
    print(f"{'='*90}\n")

    results = []
    for i, cfg in enumerate(configs):
        label = ", ".join(f"{k}:{v}" for k, v in cfg.items()) if cfg else "baseline"
        t0 = time.perf_counter()
        ok, n, pc, vb = train_and_eval(train_data, val_data, test_data, cfg or None)
        dt = time.perf_counter() - t0
        results.append((ok, n, pc, vb, cfg, label))
        m3 = pc.get(3, (0, 0))
        m4 = pc.get(4, (0, 0))
        m5 = pc.get(5, (0, 0))
        m7 = pc.get(7, (0, 0))
        print(f"  [{i+1:2d}/{len(configs)}] {label:50s}  "
              f"{ok}/{n} ({ok/n:.1%})  "
              f"3:{m3[0]:3d}/{m3[1]}  4:{m4[0]:3d}/{m4[1]}  "
              f"5:{m5[0]:2d}/{m5[1]}  7:{m7[0]:2d}/{m7[1]}  "
              f"val={vb:.1%}  [{dt:.0f}s]", flush=True)

    results.sort(key=lambda r: r[0], reverse=True)

    print(f"\n{'='*90}")
    print(f"Results sorted by test accuracy:")
    print(f"{'='*90}")
    print(f"  {'Config':50s}  {'Total':>10s}  {'3/x':>8s}  {'4/x':>8s}  {'5/x':>6s}  {'7/x':>6s}")
    print(f"  {'-'*95}")
    for ok, n, pc, vb, cfg, label in results:
        m3 = pc.get(3, (0, 0))
        m4 = pc.get(4, (0, 0))
        m5 = pc.get(5, (0, 0))
        m7 = pc.get(7, (0, 0))
        print(f"  {label:50s}  {ok:4d}/{n} ({ok/n:.1%})  "
              f"{m3[0]:3d}/{m3[1]}  {m4[0]:3d}/{m4[1]}  "
              f"{m5[0]:2d}/{m5[1]}   {m7[0]:2d}/{m7[1]}")

    best = results[0]
    print(f"\nBEST: {best[5]}  →  {best[0]}/{best[1]} ({best[0]/best[1]:.1%})")


if __name__ == "__main__":
    main()
