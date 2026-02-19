#!/usr/bin/env python3
"""Train a small arbiter MLP that replaces the hand-tuned weighted combination.

The arbiter takes per-signal, per-meter scores from the engine's 6 signals
and learns the optimal combination to predict meter.

Multi-label output: sigmoid per class (supports polymetric predictions).

WIKIMETER-primary: WIKIMETER is the primary dataset (6 classes, balanced).
METER2800 train+val are added to the training pool only.

Two phases:
  1. Extract: read signal caches → JSON dataset (train.json, val.json, test.json)
  2. Train: tiny MLP on the extracted features → checkpoint

Usage:
    # Extract signal features:
    uv run python scripts/training/train_arbiter.py --extract --meter2800 data/meter2800 --workers 4

    # Train on extracted features:
    uv run python scripts/training/train_arbiter.py --train

    # Quick test:
    uv run python scripts/training/train_arbiter.py --extract --train --meter2800 data/meter2800 --limit 20 --workers 1
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.utils import load_meter2800_entries as _load_meter2800_base, resolve_audio_path, split_by_stem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical signal names (order matters for feature vector)
# Ablation (10 runs, ±std) showed periodicity, madmom, resnet, accent
# contribute 0pp — safely dropped.
SIGNAL_NAMES = [
    "beatnet", "beat_this", "autocorr",
    "bar_tracking", "onset_mlp", "hcdf",
]

# Canonical meter keys (order matters for feature vector)
METER_KEYS = [
    "2_4", "3_4", "4_4", "5_4", "5_8", "6_8",
    "7_4", "7_8", "9_8", "10_8", "11_8", "12_8",
]

N_SIGNALS = len(SIGNAL_NAMES)
N_METERS = len(METER_KEYS)
N_SIGNAL_FEATURES = N_SIGNALS * N_METERS  # 6 × 12 = 72
TOTAL_FEATURES = N_SIGNAL_FEATURES  # 72 (signal scores only — ablation confirmed no gain from extras)

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
    """Load METER2800 entries (single-label) with corrections applied."""
    return [
        (path, meter, {meter: 1.0})
        for path, meter in _load_meter2800_base(data_dir, split)
    ]


LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5,
    "seven": 7, "nine": 9, "eleven": 11,
}


def _parse_wikimeter_row(row, audio_dir: Path) -> Entry | None:
    """Parse a single WIKIMETER row into an Entry."""
    label = row["label"].strip('"')
    primary_meter = LABEL_TO_METER.get(label)
    if primary_meter is None:
        return None

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
        return (audio_path, primary_meter, meters_dict)
    return None


def load_wikimeter_entries(
    wikimeter_dir: Path,
) -> tuple[list[Entry], list[Entry], list[Entry]]:
    """Load WIKIMETER entries split by song-level hash (80/10/10).

    Segments from the same song always go to the same split
    to prevent data leakage.

    Returns (train, val, test) entry lists.
    """
    tab_path = wikimeter_dir / "data_wikimeter.tab"
    if not tab_path.exists():
        print(f"  WIKIMETER not found at {tab_path}", flush=True)
        return [], [], []

    audio_dir = wikimeter_dir / "audio"
    train, val, test = [], [], []
    with open(tab_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry = _parse_wikimeter_row(row, audio_dir)
            if entry is None:
                continue
            split = split_by_stem(entry[0].stem)
            if split == "train":
                train.append(entry)
            elif split == "val":
                val.append(entry)
            else:
                test.append(entry)
    return train, val, test


# ---------------------------------------------------------------------------
# Feature extraction (runs engine, captures signal details)
# ---------------------------------------------------------------------------

_worker_cache = None

# Map from SIGNAL_NAMES to cache signal names
_SIGNAL_CACHE_MAP = {
    "beatnet": "beatnet_spacing",
    "beat_this": "beat_this_spacing",
    "madmom": "madmom_activation",
    "autocorr": "onset_autocorr",
    "accent": "accent_pattern",
    "periodicity": "beat_periodicity",
    "bar_tracking": "bar_tracking",
    "onset_mlp": "onset_mlp_meter",
    "resnet": "resnet_meter",
    "hcdf": "hcdf_meter",
}


def _worker_init():
    global _worker_cache
    warnings.filterwarnings("ignore")
    from beatmeter.analysis.cache import AnalysisCache
    _worker_cache = AnalysisCache()


def _onset_alignment_score(
    beat_times: np.ndarray,
    onset_event_times: np.ndarray,
    max_dist: float = 0.07,
) -> float:
    """Score how well beat positions align with detected onset events.

    Replicates AnalysisEngine._onset_alignment_score but works on numpy arrays.
    """
    if len(beat_times) == 0 or len(onset_event_times) == 0:
        return 0.0

    # Precision: do beats land on actual onsets?
    total_fwd = 0.0
    for b in beat_times:
        min_dist = float(np.min(np.abs(onset_event_times - b)))
        total_fwd += min(min_dist, max_dist)
    precision = 1.0 - total_fwd / len(beat_times) / max_dist

    # Recall: do onsets have a nearby beat?
    total_rev = 0.0
    for ot in onset_event_times:
        min_dist = float(np.min(np.abs(beat_times - ot)))
        total_rev += min(min_dist, max_dist)
    recall = 1.0 - total_rev / len(onset_event_times) / max_dist

    if precision + recall > 0:
        return 2.0 * precision * recall / (precision + recall)
    return 0.0


def _worker_extract(args: tuple[str, int, dict]) -> dict | None:
    """Read all signal scores + alignment + tempo from cache."""
    audio_path_str, expected_meter, meters_dict = args
    fname = Path(audio_path_str).name

    try:
        audio_hash = _worker_cache.audio_hash(audio_path_str)

        sig_results = {}
        for sig_name, cache_name in _SIGNAL_CACHE_MAP.items():
            cached = _worker_cache.load_signal(audio_hash, cache_name)
            if cached:
                sig_results[sig_name] = cached

        # Compute alignment scores from cached beats + onsets
        onset_data = _worker_cache.load_onsets(audio_hash)
        onset_event_times = np.array(onset_data["onset_events"]) if onset_data else np.array([])

        beatnet_alignment = 0.0
        beat_this_alignment = 0.0
        madmom_alignment = 0.0

        if len(onset_event_times) > 0:
            # BeatNet
            bn_data = _worker_cache.load_beats(audio_hash, "beatnet")
            if bn_data:
                bt = np.array([b["time"] for b in bn_data])
                beatnet_alignment = _onset_alignment_score(bt, onset_event_times)

            # Beat This!
            bth_data = _worker_cache.load_beats(audio_hash, "beat_this")
            if bth_data:
                bt = np.array([b["time"] for b in bth_data])
                beat_this_alignment = _onset_alignment_score(bt, onset_event_times)

            # madmom (best of bpb variants)
            for bpb in [3, 4, 5, 7]:
                mm_data = _worker_cache.load_beats(audio_hash, f"madmom_bpb{bpb}")
                if mm_data:
                    bt = np.array([b["time"] for b in mm_data])
                    madmom_alignment = max(madmom_alignment, _onset_alignment_score(bt, onset_event_times))

        # Read tempo from cache (try both methods)
        tempo_librosa = 0.0
        tempo_tempogram = 0.0
        td = _worker_cache.load_signal(audio_hash, "tempo_librosa")
        if td:
            tempo_librosa = td.get("bpm", 0.0)
        td = _worker_cache.load_signal(audio_hash, "tempo_tempogram")
        if td:
            tempo_tempogram = td.get("bpm", 0.0)

        return {
            "fname": fname,
            "expected": expected_meter,
            "meters": {str(k): v for k, v in meters_dict.items()},
            "predicted": None,
            "signal_results": sig_results,
            "alignment": {
                "beatnet": beatnet_alignment,
                "beat_this": beat_this_alignment,
                "madmom": madmom_alignment,
            },
            "tempo_librosa": tempo_librosa,
            "tempo_tempogram": tempo_tempogram,
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
    sharpening: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset dicts to (X, y) numpy arrays.

    X: (n_samples, TOTAL_FEATURES) — signal scores only (6 signals × 12 meters)
    y: (n_samples, N_CLASSES) — multi-hot label vector with weights

    sharpening: optional dict mapping signal name to power alpha (>1 = sharper).
        Example: {"onset_mlp": 2.0} raises onset_mlp scores to power 2
        then re-normalizes max=1. Equivalent to temperature T=1/alpha on softmax.
    """
    X = np.zeros((len(dataset), TOTAL_FEATURES), dtype=np.float32)
    y = np.zeros((len(dataset), N_CLASSES), dtype=np.float32)

    for i, entry in enumerate(dataset):
        sig_results = entry["signal_results"]

        # Signal scores: 6 signals × 12 meters = 72 features
        for s_idx, sig_name in enumerate(SIGNAL_NAMES):
            if sig_name in sig_results:
                alpha = sharpening.get(sig_name, 1.0) if sharpening else 1.0
                raw = [sig_results[sig_name].get(mk, 0.0) for mk in METER_KEYS]

                if alpha != 1.0:
                    arr = np.array(raw)
                    if arr.max() > 0:
                        arr = arr ** alpha
                        arr /= arr.max()  # re-normalize max=1
                    for m_idx, val in enumerate(arr):
                        X[i, s_idx * N_METERS + m_idx] = val
                else:
                    for m_idx, val in enumerate(raw):
                        X[i, s_idx * N_METERS + m_idx] = val

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
    boost_rare: float = 1.0,
    seed: int = 42,
    sharpening: dict[str, float] | None = None,
):
    """Train the arbiter MLP (multi-label with BCE loss)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_train, y_train = build_feature_matrix(train_data, sharpening=sharpening)
    X_val, y_val = build_feature_matrix(val_data, sharpening=sharpening)

    if len(train_data) < 4:
        print(f"\nToo few training samples ({len(train_data)}), need at least 4", flush=True)
        return

    print(f"\nArbiter training (multi-label):", flush=True)
    print(f"  Features: {TOTAL_FEATURES} ({N_SIGNALS} signals × {N_METERS} meters)", flush=True)
    print(f"  Classes: {CLASS_METERS}", flush=True)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}", flush=True)
    if sharpening:
        print(f"  Sharpening: {sharpening}", flush=True)

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
    raw_pw = neg_counts / np.maximum(pos_counts, 1)
    if boost_rare > 1.0:
        # Boost rare classes: 5, 7, 9, 11 (indices 2, 3, 4, 5)
        for idx in range(2, len(CLASS_METERS)):
            raw_pw[idx] *= boost_rare
    pos_weight = torch.tensor(
        np.clip(raw_pw, 0.5, 10.0),
        dtype=torch.float32,
    )
    print(f"  pos_weight: {[f'{w:.1f}' for w in pos_weight.tolist()]}"
          f"  (boost_rare={boost_rare})", flush=True)

    # Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val),
    )
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

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

        # Validate: balanced accuracy (macro per-class accuracy)
        # This prevents the model from sacrificing rare classes (5/x, 7/x)
        # for common classes (3/x, 4/x) that dominate overall accuracy.
        model.eval()
        val_correct_per_class = Counter()
        val_total_per_class = Counter()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                probs = torch.sigmoid(model(xb))
                pred_primary = probs.argmax(1)
                true_primary = yb.argmax(1)
                for p, t in zip(pred_primary.cpu(), true_primary.cpu()):
                    val_total_per_class[t.item()] += 1
                    if p.item() == t.item():
                        val_correct_per_class[t.item()] += 1

        # Balanced accuracy = mean of per-class accuracies
        per_class_accs = []
        for cls_idx in range(N_CLASSES):
            t = val_total_per_class.get(cls_idx, 0)
            c = val_correct_per_class.get(cls_idx, 0)
            if t > 0:
                per_class_accs.append(c / t)
        val_acc = np.mean(per_class_accs) if per_class_accs else 0.0
        val_overall = sum(val_correct_per_class.values()) / max(sum(val_total_per_class.values()), 1)

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
                  f"val_balanced {val_acc:.1%}  val_overall {val_overall:.1%}{marker}", flush=True)

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best was ep {best_epoch})", flush=True)
            break

    print(f"\nBest val balanced acc: {best_val_metric:.1%} at epoch {best_epoch}", flush=True)

    if best_state is None:
        print("WARNING: No improvement during training, using final model state", flush=True)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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
        X_test, y_test = build_feature_matrix(test_data, sharpening=sharpening)
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
        "sharpening": sharpening or {},
    }, ckpt_path)
    print(f"\nModel saved to {ckpt_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train meter arbiter MLP")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wikimeter"),
                        help="Primary data dir (WIKIMETER)")
    parser.add_argument("--meter2800", type=Path, default=None,
                        help="METER2800 data dir (train+val added to training pool)")
    parser.add_argument("--extract", action="store_true", help="Run extraction phase")
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--boost-rare", type=float, default=1.0,
                        help="Multiply pos_weight for rare classes (5,7,9,11)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds to try (picks best val)")
    parser.add_argument("--sharpen", type=str, default="",
                        help="Per-signal sharpening power, e.g. 'onset_mlp:2.0,beatnet:1.5'")
    args = parser.parse_args()

    wikimeter_dir = args.data_dir.resolve()

    if args.extract:
        # Primary: WIKIMETER → train/val/test
        wiki_train, wiki_val, wiki_test = load_wikimeter_entries(wikimeter_dir)
        if args.limit > 0:
            wiki_train = wiki_train[:args.limit]
            wiki_val = wiki_val[:args.limit]
            wiki_test = wiki_test[:args.limit]
        print(f"WIKIMETER: {len(wiki_train)} train, {len(wiki_val)} val, {len(wiki_test)} test", flush=True)

        train_entries = list(wiki_train)
        val_entries = list(wiki_val)
        test_entries = list(wiki_test)

        # Extra: METER2800 train+val → training pool only
        if args.meter2800:
            m2800_dir = args.meter2800.resolve()
            m2800_train = load_meter2800_entries(m2800_dir, "train")
            m2800_val = load_meter2800_entries(m2800_dir, "val")
            if args.limit > 0:
                m2800_train = m2800_train[:args.limit]
                m2800_val = m2800_val[:args.limit]
            print(f"METER2800: {len(m2800_train)} train + {len(m2800_val)} val → training pool", flush=True)
            train_entries.extend(m2800_train + m2800_val)

        print(f"Total: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test", flush=True)

        extract_dataset(train_entries, args.workers, "train")
        extract_dataset(val_entries, args.workers, "val")
        extract_dataset(test_entries, args.workers, "test")

    if args.train:
        # Load extracted datasets (3-file split: train/val/test)
        train_path = DATASET_DIR / "train.json"
        val_path = DATASET_DIR / "val.json"
        test_path = DATASET_DIR / "test.json"

        if not train_path.exists():
            print(f"ERROR: No extracted data at {train_path}", flush=True)
            print("  Run first:  --extract --meter2800 data/meter2800 --workers 4", flush=True)
            sys.exit(1)
        if not val_path.exists():
            print(f"ERROR: No extracted data at {val_path}", flush=True)
            print("  Run first:  --extract --meter2800 data/meter2800 --workers 4", flush=True)
            sys.exit(1)

        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
        print(f"Loaded {len(train_data)} train, {len(val_data)} val entries", flush=True)

        test_data = None
        if test_path.exists():
            with open(test_path) as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} test entries", flush=True)

        meter_dist = Counter(e["expected"] for e in train_data)
        print(f"Train meters: {dict(sorted(meter_dist.items()))}", flush=True)
        val_dist = Counter(e["expected"] for e in val_data)
        print(f"Val meters: {dict(sorted(val_dist.items()))}", flush=True)

        # Parse sharpening
        sharpening = {}
        if args.sharpen:
            for part in args.sharpen.split(","):
                sig, alpha = part.strip().split(":")
                sharpening[sig.strip()] = float(alpha.strip())
            print(f"Sharpening: {sharpening}", flush=True)

        if args.seeds > 1:
            print(f"\nMulti-seed training ({args.seeds} seeds)...", flush=True)
            for s in range(args.seeds):
                seed = 42 + s * 7
                print(f"\n{'='*60}", flush=True)
                print(f"  Seed {s+1}/{args.seeds} (seed={seed})", flush=True)
                print(f"{'='*60}", flush=True)
                train_arbiter(train_data, val_data, test_data, epochs=args.epochs,
                              boost_rare=args.boost_rare, seed=seed,
                              sharpening=sharpening or None)
        else:
            train_arbiter(train_data, val_data, test_data, epochs=args.epochs,
                          boost_rare=args.boost_rare, seed=args.seed,
                          sharpening=sharpening or None)


if __name__ == "__main__":
    main()
