#!/usr/bin/env python3
"""Train a small arbiter MLP that replaces the hand-tuned weighted combination.

The arbiter takes per-signal, per-meter scores from the engine's 10 signals
(+ trust values + tempo) and learns the optimal combination to predict meter.

Multi-label output: sigmoid per class (supports polymetric predictions).

Two phases:
  1. Extract: run engine on all files, capture signal_results → JSON dataset
  2. Train: tiny MLP on the extracted features → checkpoint

Usage:
    # Extract signal features (slow first time, cached after):
    uv run python scripts/training/train_arbiter.py --extract --split tuning --workers 4

    # Extract with WIKIMETER extra data:
    uv run python scripts/training/train_arbiter.py --extract --split tuning --workers 4 --extra-data

    # Train on extracted features:
    uv run python scripts/training/train_arbiter.py --train

    # Quick test:
    uv run python scripts/training/train_arbiter.py --extract --train --limit 20 --workers 1
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.utils import resolve_audio_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical signal names (order matters for feature vector)
SIGNAL_NAMES = [
    "beatnet", "beat_this", "madmom", "autocorr",
    "accent", "periodicity", "bar_tracking", "onset_mlp",
    "resnet", "hcdf",
]

# Canonical meter keys (order matters for feature vector)
METER_KEYS = [
    "2_4", "3_4", "4_4", "5_4", "5_8", "6_8",
    "7_4", "7_8", "9_8", "10_8", "11_8", "12_8",
]

N_SIGNALS = len(SIGNAL_NAMES)
N_METERS = len(METER_KEYS)
N_SIGNAL_FEATURES = N_SIGNALS * N_METERS  # 10 × 12 = 120
N_EXTRA = 4  # beatnet_trust, beat_this_trust, madmom_trust, tempo_bpm
TOTAL_FEATURES = N_SIGNAL_FEATURES + N_EXTRA  # 124

# Output classes: meter numerator (multi-label)
CLASS_METERS = [3, 4, 5, 7, 9, 11]
N_CLASSES = len(CLASS_METERS)
METER_TO_IDX = {m: i for i, m in enumerate(CLASS_METERS)}

DATASET_DIR = PROJECT_ROOT / "data" / "arbiter_dataset"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Entry = (audio_path, primary_meter, meters_dict)
# meters_dict maps meter_numerator -> weight in [0, 1]
Entry = tuple[Path, int, dict[int, float]]


def load_meter2800_entries(data_dir: Path, split: str) -> list[Entry]:
    """Load METER2800 entries (single-label)."""
    if split == "tuning":
        tab_files = ["data_train_4_classes.tab", "data_val_4_classes.tab"]
    elif split == "test":
        tab_files = ["data_test_4_classes.tab"]
    else:
        tab_files = [f"data_{split}_4_classes.tab"]

    entries: list[Entry] = []
    for tab in tab_files:
        label_path = data_dir / tab
        if not label_path.exists():
            continue
        with open(label_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fname = row["filename"].strip('"')
                meter = int(row["meter"])
                audio_path = resolve_audio_path(fname, data_dir)
                if audio_path:
                    entries.append((audio_path, meter, {meter: 1.0}))
    return entries


LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5,
    "seven": 7, "nine": 9, "eleven": 11,
}


def load_wikimeter_entries(wikimeter_dir: Path) -> list[Entry]:
    """Load WIKIMETER entries (multi-label from meter column)."""
    tab_path = wikimeter_dir / "data_wikimeter.tab"
    if not tab_path.exists():
        print(f"  WIKIMETER not found at {tab_path}", flush=True)
        return []

    audio_dir = wikimeter_dir / "audio"
    entries: list[Entry] = []
    with open(tab_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["label"].strip('"')
            primary_meter = LABEL_TO_METER.get(label)
            if primary_meter is None:
                continue

            # Parse multi-label meter column: "3:0.7,4:0.8"
            meters_dict: dict[int, float] = {}
            meter_str = row["meter"].strip('"')
            for part in meter_str.split(","):
                part = part.strip()
                if ":" in part:
                    m_str, w_str = part.split(":", 1)
                    try:
                        meters_dict[int(m_str)] = float(w_str)
                    except ValueError:
                        continue
                else:
                    try:
                        meters_dict[int(part)] = 1.0
                    except ValueError:
                        continue

            if not meters_dict:
                meters_dict = {primary_meter: 1.0}

            fname = row["filename"].strip('"')
            basename = Path(fname).name
            audio_path = audio_dir / basename
            if audio_path.exists():
                entries.append((audio_path, primary_meter, meters_dict))
    return entries


# ---------------------------------------------------------------------------
# Feature extraction (runs engine, captures signal details)
# ---------------------------------------------------------------------------

_worker_engine = None


def _worker_init():
    global _worker_engine
    warnings.filterwarnings("ignore")
    import torch  # noqa: F401
    from beatmeter.analysis.cache import AnalysisCache
    from beatmeter.analysis.engine import AnalysisEngine
    _worker_engine = AnalysisEngine(cache=AnalysisCache())


def _worker_extract(args: tuple[str, int, dict]) -> dict | None:
    """Process a single file: run engine, capture signal details.

    Also computes disabled signals (ResNet, HCDF) that aren't in the
    engine's default pipeline but are useful for the arbiter.
    """
    import torch
    import beatmeter.analysis.meter as meter_mod

    audio_path_str, expected_meter, meters_dict = args
    fname = Path(audio_path_str).name

    try:
        with torch.inference_mode():
            result = _worker_engine.analyze_file(audio_path_str, skip_sections=True)

        details = meter_mod.last_signal_details
        if details is None:
            return None

        predicted = None
        if result and result.meter_hypotheses:
            predicted = result.meter_hypotheses[0].numerator

        sig_results = details["signal_results"]

        # Add disabled signals that engine skips (W=0.0)
        import librosa
        y, sr = librosa.load(audio_path_str, sr=22050, mono=True)
        tempo_bpm = details.get("tempo_bpm")
        beat_interval = 60.0 / tempo_bpm if tempo_bpm and tempo_bpm > 0 else None

        if "resnet" not in sig_results:
            try:
                from beatmeter.analysis.signals.resnet_meter import signal_resnet_meter
                s = signal_resnet_meter(y, sr)
                if s:
                    sig_results["resnet"] = {f"{k[0]}_{k[1]}": v for k, v in s.items()}
            except Exception:
                pass

        if "hcdf" not in sig_results:
            try:
                from beatmeter.analysis.signals.hcdf_meter import signal_hcdf_meter
                s = signal_hcdf_meter(y, sr, beat_interval=beat_interval)
                if s:
                    sig_results["hcdf"] = {f"{k[0]}_{k[1]}": v for k, v in s.items()}
            except Exception:
                pass

        return {
            "fname": fname,
            "expected": expected_meter,
            "meters": {str(k): v for k, v in meters_dict.items()},
            "predicted": predicted,
            "signal_results": sig_results,
            "trust": details["trust"],
            "tempo_bpm": tempo_bpm,
        }
    except Exception as e:
        print(f"  ERROR {fname}: {e}", file=sys.stderr, flush=True)
        return None


def extract_dataset(
    entries: list[Entry],
    workers: int,
    split_name: str,
) -> list[dict]:
    """Run engine on all files and capture signal features."""
    args_list = [(str(p), m, md) for p, m, md in entries]

    print(f"\nExtracting signal features for {len(entries)} files ({workers} workers)...", flush=True)
    t0 = time.perf_counter()

    results = []
    if workers <= 1:
        _worker_init()
        for i, args in enumerate(args_list):
            r = _worker_extract(args)
            if r is not None:
                results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(entries)}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_worker_init, maxtasksperchild=50) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker_extract, args_list)):
                if r is not None:
                    results.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(entries)}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Extracted {len(results)}/{len(entries)} files in {elapsed:.0f}s", flush=True)

    # Save dataset
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASET_DIR / f"{split_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Saved to {out_path}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(
    dataset: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset dicts to (X, y) numpy arrays.

    X: (n_samples, TOTAL_FEATURES) — signal scores + trust + tempo
    y: (n_samples, N_CLASSES) — multi-hot label vector with weights
    """
    X = np.zeros((len(dataset), TOTAL_FEATURES), dtype=np.float32)
    y = np.zeros((len(dataset), N_CLASSES), dtype=np.float32)

    for i, entry in enumerate(dataset):
        sig_results = entry["signal_results"]
        trust = entry["trust"]
        tempo = entry.get("tempo_bpm") or 0.0

        # Signal scores: 10 signals × 12 meters = 120 features
        for s_idx, sig_name in enumerate(SIGNAL_NAMES):
            if sig_name in sig_results:
                for m_idx, meter_key in enumerate(METER_KEYS):
                    feat_idx = s_idx * N_METERS + m_idx
                    X[i, feat_idx] = sig_results[sig_name].get(meter_key, 0.0)

        # Trust values + tempo: 4 features
        base = N_SIGNAL_FEATURES
        X[i, base + 0] = trust.get("beatnet", 0.0)
        X[i, base + 1] = trust.get("beat_this", 0.0)
        X[i, base + 2] = trust.get("madmom", 0.0)
        X[i, base + 3] = tempo / 200.0  # normalize to ~[0, 1.5]

        # Multi-hot labels from "meters" dict
        meters = entry.get("meters")
        if meters:
            for m_str, weight in meters.items():
                m_int = int(m_str)
                idx = METER_TO_IDX.get(m_int)
                if idx is not None:
                    y[i, idx] = float(weight)
        else:
            # Fallback: single-label from "expected"
            meter = entry["expected"]
            idx = METER_TO_IDX.get(meter)
            if idx is not None:
                y[i, idx] = 1.0

    return X, y


# ---------------------------------------------------------------------------
# Arbiter model
# ---------------------------------------------------------------------------

def train_arbiter(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict] | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
):
    """Train the arbiter MLP (multi-label with BCE loss)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_train, y_train = build_feature_matrix(train_data)
    X_val, y_val = build_feature_matrix(val_data)

    if len(train_data) < 4:
        print(f"\nToo few training samples ({len(train_data)}), need at least 4", flush=True)
        return

    print(f"\nArbiter training (multi-label):", flush=True)
    print(f"  Features: {TOTAL_FEATURES} ({N_SIGNALS} signals × {N_METERS} meters + {N_EXTRA} extra)", flush=True)
    print(f"  Classes: {CLASS_METERS}", flush=True)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}", flush=True)

    # Per-class positive counts (for pos_weight)
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    print(f"  Positive counts: {dict(zip(CLASS_METERS, pos_counts.astype(int).tolist()))}", flush=True)

    # Standardize features
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    # pos_weight: balance classes (cap at 10 to avoid instability)
    pos_weight = torch.tensor(
        np.clip(neg_counts / np.maximum(pos_counts, 1), 0.5, 10.0),
        dtype=torch.float32,
    )
    print(f"  pos_weight: {[f'{w:.1f}' for w in pos_weight.tolist()]}", flush=True)

    # Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val),
    )
    batch_size = min(64, len(X_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(X_train) > batch_size)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Model: tiny MLP
    model = nn.Sequential(
        nn.Linear(TOTAL_FEATURES, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(32, N_CLASSES),
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_metric = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_total += len(xb)
        scheduler.step()

        # Validate: primary accuracy (argmax of sigmoid == argmax of labels)
        model.eval()
        val_primary_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                probs = torch.sigmoid(model(xb))
                pred_primary = probs.argmax(1)
                true_primary = yb.argmax(1)
                val_primary_correct += (pred_primary == true_primary).sum().item()
                val_total += len(xb)

        val_acc = val_primary_correct / val_total

        is_best = val_acc > best_val_metric
        if is_best:
            best_val_metric = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or is_best:
            marker = " *" if is_best else ""
            print(f"  ep {epoch:3d}  loss {train_loss_sum/train_total:.4f}  "
                  f"val_primary {val_acc:.1%}{marker}", flush=True)

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best was ep {best_epoch})", flush=True)
            break

    print(f"\nBest val primary acc: {best_val_metric:.1%} at epoch {best_epoch}", flush=True)

    # Evaluate with per-class breakdown
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)

    def _evaluate(X_np, y_np, label):
        X_std = (X_np - feat_mean) / feat_std
        ds = TensorDataset(torch.tensor(X_std), torch.tensor(y_np))
        dl = DataLoader(ds, batch_size=256, shuffle=False)

        # Primary accuracy (argmax match)
        correct_per_class = Counter()
        total_per_class = Counter()
        # Multi-label metrics (threshold = 0.5)
        tp = np.zeros(N_CLASSES)
        fp = np.zeros(N_CLASSES)
        fn = np.zeros(N_CLASSES)

        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                probs = torch.sigmoid(model(xb))
                pred_primary = probs.argmax(1)
                true_primary = yb.argmax(1)

                for p, t in zip(pred_primary.cpu(), true_primary.cpu()):
                    total_per_class[t.item()] += 1
                    if p.item() == t.item():
                        correct_per_class[t.item()] += 1

                # Multi-label at threshold 0.5
                pred_binary = (probs > 0.5).cpu().numpy()
                true_binary = (yb > 0.5).cpu().numpy()
                tp += (pred_binary & true_binary).sum(axis=0)
                fp += (pred_binary & ~true_binary).sum(axis=0)
                fn += (~pred_binary & true_binary).sum(axis=0)

        total_ok = sum(correct_per_class.values())
        total_n = sum(total_per_class.values())
        print(f"\n{label} — Primary: {total_ok}/{total_n} = {total_ok/max(total_n,1):.1%}", flush=True)
        print(f" {'Meter':>6s}  {'Correct':>8s}  {'Total':>6s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}", flush=True)
        print(f" {'-'*56}", flush=True)
        f1_scores = []
        for cls_idx in range(N_CLASSES):
            m = CLASS_METERS[cls_idx]
            c = correct_per_class.get(cls_idx, 0)
            t = total_per_class.get(cls_idx, 0)
            acc = c / max(t, 1)
            prec = tp[cls_idx] / max(tp[cls_idx] + fp[cls_idx], 1)
            rec = tp[cls_idx] / max(tp[cls_idx] + fn[cls_idx], 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1_scores.append(f1)
            print(f" {m:>4d}/x  {c:>8d}  {t:>6d}  {acc:>5.1%}  {prec:>5.1%}  {rec:>5.1%}  {f1:>5.1%}", flush=True)
        macro_f1 = np.mean(f1_scores)
        print(f"  Macro-F1: {macro_f1:.1%}", flush=True)

    _evaluate(X_val, y_val, "Val")

    if test_data:
        X_test, y_test = build_feature_matrix(test_data)
        _evaluate(X_test, y_test, "Test")

    # Save checkpoint
    ckpt_path = PROJECT_ROOT / "data" / "meter_arbiter.pt"
    import torch
    torch.save({
        "model_state": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "signal_names": SIGNAL_NAMES,
        "meter_keys": METER_KEYS,
        "class_meters": CLASS_METERS,
        "total_features": TOTAL_FEATURES,
        "n_classes": N_CLASSES,
        "multi_label": True,
        "best_val_acc": best_val_metric,
        "best_epoch": best_epoch,
    }, ckpt_path)
    print(f"\nModel saved to {ckpt_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train meter arbiter MLP")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--wikimeter-dir", type=Path, default=Path("data/wikimeter"))
    parser.add_argument("--extract", action="store_true", help="Run extraction phase")
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--extra-data", action="store_true", help="Include WIKIMETER data")
    parser.add_argument("--split", default="tuning", help="Split to extract")
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    if args.extract:
        entries = load_meter2800_entries(data_dir, args.split)
        if args.limit > 0:
            entries = entries[:args.limit]
        print(f"Loaded {len(entries)} METER2800 entries for {args.split}", flush=True)

        if args.extra_data:
            wiki_entries = load_wikimeter_entries(args.wikimeter_dir.resolve())
            if args.limit > 0:
                wiki_entries = wiki_entries[:args.limit]
            print(f"Loaded {len(wiki_entries)} WIKIMETER entries", flush=True)
            entries.extend(wiki_entries)
            print(f"Total: {len(entries)} entries", flush=True)

        extract_dataset(entries, args.workers, args.split)

        # Also extract test split if extracting tuning
        if args.split == "tuning":
            test_entries = load_meter2800_entries(data_dir, "test")
            if args.limit > 0:
                test_entries = test_entries[:args.limit]
            if test_entries:
                print(f"\nAlso extracting test split ({len(test_entries)} files)...", flush=True)
                extract_dataset(test_entries, args.workers, "test")

    if args.train:
        # Load extracted datasets
        tuning_path = DATASET_DIR / "tuning.json"
        test_path = DATASET_DIR / "test.json"

        if not tuning_path.exists():
            print(f"ERROR: No extracted data at {tuning_path}", flush=True)
            print("  Run first:  --extract --split tuning --workers 4", flush=True)
            sys.exit(1)

        with open(tuning_path) as f:
            tuning_data = json.load(f)
        print(f"Loaded {len(tuning_data)} tuning entries", flush=True)

        test_data = None
        if test_path.exists():
            with open(test_path) as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} test entries", flush=True)

        # Split tuning into train/val (80/20 stratified by primary meter)
        by_meter = defaultdict(list)
        for entry in tuning_data:
            by_meter[entry["expected"]].append(entry)

        train_data = []
        val_data = []
        for meter, items in sorted(by_meter.items()):
            np.random.seed(42)
            np.random.shuffle(items)
            split_idx = int(len(items) * 0.8)
            train_data.extend(items[:split_idx])
            val_data.extend(items[split_idx:])

        print(f"Split: {len(train_data)} train, {len(val_data)} val", flush=True)
        meter_dist = Counter(e["expected"] for e in train_data)
        print(f"Train meters: {dict(sorted(meter_dist.items()))}", flush=True)

        train_arbiter(train_data, val_data, test_data, epochs=args.epochs)


if __name__ == "__main__":
    main()
