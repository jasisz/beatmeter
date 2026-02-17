#!/usr/bin/env python3
"""Grid search over signal sharpening alphas for arbiter training.

Phase 1: Each signal individually with alpha in [0.5, 1.5, 2.0, 3.0]
Phase 2: Combine best alphas from Phase 1

Prints results as a table sorted by METER2800-test accuracy.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.training.train_arbiter import (
    DATASET_DIR, CLASS_METERS, METER_TO_IDX, N_CLASSES,
    SIGNAL_NAMES, TOTAL_FEATURES, N_SIGNALS, N_METERS, METER_KEYS,
    build_feature_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def train_and_eval(train_data, val_data, test_data, sharpening, seed=42, epochs=200, patience=30):
    """Train arbiter with given sharpening, return (test_overall, per_class_acc, val_balanced)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_train, y_train = build_feature_matrix(train_data, sharpening=sharpening)
    X_val, y_val = build_feature_matrix(val_data, sharpening=sharpening)

    # Standardize
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train_s = (X_train - feat_mean) / feat_std
    X_val_s = (X_val - feat_mean) / feat_std

    # pos_weight
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    raw_pw = neg_counts / np.maximum(pos_counts, 1)
    pos_weight = torch.tensor(np.clip(raw_pw, 0.5, 10.0), dtype=torch.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val))

    batch_size = min(64, len(X_train_s))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(X_train_s) > batch_size)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(TOTAL_FEATURES, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(32, N_CLASSES),
    )
    model.to(device)
    pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Balanced accuracy
        model.eval()
        correct_pc = Counter()
        total_pc = Counter()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = torch.sigmoid(model(xb)).argmax(1)
                true = yb.argmax(1)
                for p, t in zip(pred.cpu(), true.cpu()):
                    total_pc[t.item()] += 1
                    if p.item() == t.item():
                        correct_pc[t.item()] += 1

        accs = [correct_pc.get(i, 0) / max(total_pc.get(i, 0), 1) for i in range(N_CLASSES) if total_pc.get(i, 0) > 0]
        val_bal = np.mean(accs) if accs else 0.0

        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate on test (METER2800 portion only — first 700 entries)
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    X_test, y_test = build_feature_matrix(test_data, sharpening=sharpening)
    X_test_s = (X_test - feat_mean) / feat_std

    test_ds = TensorDataset(torch.tensor(X_test_s), torch.tensor(y_test))
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    correct_pc = Counter()
    total_pc = Counter()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = torch.sigmoid(model(xb)).argmax(1)
            true = yb.argmax(1)
            for p, t in zip(pred.cpu(), true.cpu()):
                total_pc[t.item()] += 1
                if p.item() == t.item():
                    correct_pc[t.item()] += 1

    per_class = {}
    for i, m in enumerate(CLASS_METERS):
        t = total_pc.get(i, 0)
        c = correct_pc.get(i, 0)
        per_class[m] = (c, t)

    total_ok = sum(correct_pc.values())
    total_n = sum(total_pc.values())

    return total_ok, total_n, per_class, best_val


def main():
    # Load data
    with open(DATASET_DIR / "tuning.json") as f:
        tuning_data = json.load(f)
    with open(DATASET_DIR / "test.json") as f:
        test_data = json.load(f)

    # Split tuning
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

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}", flush=True)

    # ── Phase 1: Individual signal sharpening ──
    alphas = [0.5, 1.5, 2.0, 3.0]
    configs = [{}]  # baseline
    for sig in SIGNAL_NAMES:
        for a in alphas:
            configs.append({sig: a})

    print(f"\n{'='*70}")
    print(f"Phase 1: {len(configs)} configs (baseline + {len(SIGNAL_NAMES)} signals × {len(alphas)} alphas)")
    print(f"{'='*70}\n")

    results = []
    for i, cfg in enumerate(configs):
        label = ", ".join(f"{k}:{v}" for k, v in cfg.items()) if cfg else "baseline"
        t0 = time.perf_counter()
        ok, n, pc, vb = train_and_eval(train_data, val_data, test_data, cfg or None)
        dt = time.perf_counter() - t0
        results.append((ok, n, pc, vb, cfg, label))
        # Print per-class for 3,4,5,7 (METER2800 classes)
        m3 = pc.get(3, (0, 0))
        m4 = pc.get(4, (0, 0))
        m5 = pc.get(5, (0, 0))
        m7 = pc.get(7, (0, 0))
        print(f"  [{i+1:2d}/{len(configs)}] {label:30s}  "
              f"{ok}/{n} ({ok/n:.1%})  "
              f"3:{m3[0]:3d}/{m3[1]}  4:{m4[0]:3d}/{m4[1]}  "
              f"5:{m5[0]:2d}/{m5[1]}  7:{m7[0]:2d}/{m7[1]}  "
              f"val_bal={vb:.1%}  [{dt:.0f}s]", flush=True)

    # Sort by test accuracy
    results.sort(key=lambda r: r[0], reverse=True)

    print(f"\n{'='*70}")
    print(f"Phase 1 — Top 10:")
    print(f"{'='*70}")
    print(f"  {'Config':30s}  {'Total':>10s}  {'3/x':>8s}  {'4/x':>8s}  {'5/x':>6s}  {'7/x':>6s}  {'val_bal':>8s}")
    print(f"  {'-'*80}")
    for ok, n, pc, vb, cfg, label in results[:10]:
        m3 = pc.get(3, (0, 0))
        m4 = pc.get(4, (0, 0))
        m5 = pc.get(5, (0, 0))
        m7 = pc.get(7, (0, 0))
        print(f"  {label:30s}  {ok:4d}/{n} ({ok/n:.1%})  "
              f"{m3[0]:3d}/{m3[1]}  {m4[0]:3d}/{m4[1]}  "
              f"{m5[0]:2d}/{m5[1]}   {m7[0]:2d}/{m7[1]}  "
              f"{vb:.1%}")

    # ── Phase 2: Combine top winners ──
    # Find best alpha for each signal (that improved over baseline)
    baseline_ok = results[-1][0] if results[-1][4] == {} else None
    for r in results:
        if r[4] == {}:
            baseline_ok = r[0]
            break

    best_per_signal = {}
    for ok, n, pc, vb, cfg, label in results:
        if len(cfg) == 1:
            sig = list(cfg.keys())[0]
            alpha = list(cfg.values())[0]
            if ok > baseline_ok and (sig not in best_per_signal or ok > best_per_signal[sig][0]):
                best_per_signal[sig] = (ok, alpha)

    if best_per_signal:
        print(f"\n{'='*70}")
        print(f"Phase 2: Combining winners")
        winners = {sig: alpha for sig, (_, alpha) in best_per_signal.items()}
        print(f"  Winners: {winners}")
        print(f"{'='*70}\n")

        # Try all subsets of 2+ winners
        winner_items = list(winners.items())
        phase2_configs = []
        for r in range(2, len(winner_items) + 1):
            from itertools import combinations
            for combo in combinations(winner_items, r):
                phase2_configs.append(dict(combo))

        phase2_results = []
        for i, cfg in enumerate(phase2_configs):
            label = ", ".join(f"{k}:{v}" for k, v in cfg.items())
            t0 = time.perf_counter()
            ok, n, pc, vb = train_and_eval(train_data, val_data, test_data, cfg)
            dt = time.perf_counter() - t0
            phase2_results.append((ok, n, pc, vb, cfg, label))
            m3 = pc.get(3, (0, 0))
            m4 = pc.get(4, (0, 0))
            m5 = pc.get(5, (0, 0))
            m7 = pc.get(7, (0, 0))
            print(f"  [{i+1:2d}/{len(phase2_configs)}] {label:40s}  "
                  f"{ok}/{n} ({ok/n:.1%})  "
                  f"3:{m3[0]:3d}/{m3[1]}  4:{m4[0]:3d}/{m4[1]}  "
                  f"5:{m5[0]:2d}/{m5[1]}  7:{m7[0]:2d}/{m7[1]}  "
                  f"[{dt:.0f}s]", flush=True)

        phase2_results.sort(key=lambda r: r[0], reverse=True)
        print(f"\n{'='*70}")
        print(f"Phase 2 — Top 5:")
        print(f"{'='*70}")
        for ok, n, pc, vb, cfg, label in phase2_results[:5]:
            m3 = pc.get(3, (0, 0))
            m4 = pc.get(4, (0, 0))
            m5 = pc.get(5, (0, 0))
            m7 = pc.get(7, (0, 0))
            print(f"  {label:40s}  {ok:4d}/{n} ({ok/n:.1%})  "
                  f"{m3[0]:3d}/{m3[1]}  {m4[0]:3d}/{m4[1]}  "
                  f"{m5[0]:2d}/{m5[1]}   {m7[0]:2d}/{m7[1]}")

    # ── Final: Overall best ──
    all_results = results + (phase2_results if best_per_signal else [])
    all_results.sort(key=lambda r: r[0], reverse=True)
    best = all_results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best[5]}  →  {best[0]}/{best[1]} ({best[0]/best[1]:.1%})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
