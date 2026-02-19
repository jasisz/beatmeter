#!/usr/bin/env python3
"""Train MeterNet — unified meter classification from audio + beat + signal + tempo features.

MeterNet sees everything the pipeline produces:
- 1361 audio features (from onset_mlp_features.py, reuses onset_features_cache_v5)
- 42 beat tracker features (IBI stats, alignment, downbeat spacing, cross-tracker agreement)
- 60 signal scores (5 signals x 12 meters, from warm cache)
- 4 tempo features (from warm cache)
Total: 1467 dimensions.

Architecture: same Residual MLP as onset_mlp v5 (parameterized via grid search).
Loss: BCEWithLogitsLoss (multi-label, supports WIKIMETER polymetric annotations).
Val metric: balanced accuracy (macro per-class, prevents rare-class sacrifice).

Prerequisites:
- Warm cache complete (all phases: beatnet, beat_this, madmom, librosa, onsets,
  signals, tempo, onset_mlp, hcdf)
- onset_mlp features cached (data/onset_features_cache_v5/)

Usage:
    # Smoke test
    uv run python scripts/training/train_meter_net.py --meter2800 data/meter2800 --limit 3 --workers 1

    # Full training
    uv run python scripts/training/train_meter_net.py --meter2800 data/meter2800 --workers 4

    # Custom hyperparameters
    uv run python scripts/training/train_meter_net.py --meter2800 data/meter2800 --hidden 1024 --lr 3e-4
"""

import argparse
import csv
import hashlib
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from beatmeter.analysis.signals.onset_mlp_features import (
    SR,
    MAX_DURATION_S,
    FEATURE_VERSION_V5,
    TOTAL_FEATURES_V5,
    extract_features_from_path,
)
from beatmeter.analysis.signals.meter_net_features import (
    TOTAL_FEATURES,
    FEATURE_VERSION,
    N_AUDIO_FEATURES,
    N_BEAT_FEATURES,
    N_SIGNAL_FEATURES,
    N_TEMPO_FEATURES,
    extract_beat_features,
    extract_signal_scores,
    extract_tempo_features,
)
from scripts.utils import (
    load_meter2800_entries as _load_meter2800_base,
    split_by_stem,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTMIX_ALPHA = 1.0

CLASS_METERS_6 = [3, 4, 5, 7, 9, 11]

# Label mapping for WIKIMETER
LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5,
    "seven": 7, "nine": 9, "eleven": 11,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Entry = (audio_path, primary_meter, meters_dict)
Entry = tuple[Path, int, dict[int, float]]


def load_meter2800_split(
    data_dir: Path, split: str, valid_meters: set[int],
) -> list[Entry]:
    """Load METER2800 entries as single-label Entries."""
    return [
        (path, meter, {meter: 1.0})
        for path, meter in _load_meter2800_base(data_dir, split, valid_meters=valid_meters)
    ]


def load_wikimeter(data_dir: Path, valid_meters: set[int]) -> list[Entry]:
    """Load WIKIMETER entries with multi-label support."""
    tab_path = data_dir / "data_wikimeter.tab"
    if not tab_path.exists():
        return []

    audio_dir = data_dir / "audio"
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
                    m_s, w_s = part.split(":", 1)
                    try:
                        meters_dict[int(m_s)] = float(w_s)
                    except ValueError:
                        continue
                else:
                    try:
                        meters_dict[int(part)] = 1.0
                    except ValueError:
                        continue

            if not meters_dict:
                meters_dict = {primary_meter: 1.0}

            fname = Path(row["filename"].strip('"')).name
            audio_path = audio_dir / fname
            if not audio_path.exists():
                continue

            if primary_meter not in valid_meters:
                continue

            entries.append((audio_path, primary_meter, meters_dict))
    return entries


# ---------------------------------------------------------------------------
# Feature extraction with caching
# ---------------------------------------------------------------------------


def _onset_cache_key(audio_path: Path) -> str:
    """Cache key for onset_features_cache_v5 (same as train_onset_mlp.py)."""
    st = audio_path.stat()
    raw = f"{audio_path.resolve()}::{st.st_size}::{st.st_mtime_ns}::{FEATURE_VERSION_V5}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _full_cache_key(audio_path: Path) -> str:
    """Cache key for full MeterNet feature vector (1467-dim)."""
    st = audio_path.stat()
    raw = f"{audio_path.resolve()}::{st.st_size}::{st.st_mtime_ns}::meter_net_v1"
    return hashlib.sha1(raw.encode()).hexdigest()


def _build_full_features(
    audio_feat: np.ndarray,
    audio_path: Path,
    analysis_cache,
) -> np.ndarray | None:
    """Combine audio features with beat/signal/tempo from analysis cache."""
    audio_hash = analysis_cache.audio_hash(str(audio_path))
    beat_feat = extract_beat_features(analysis_cache, audio_hash)
    signal_feat = extract_signal_scores(analysis_cache, audio_hash)
    tempo_feat = extract_tempo_features(analysis_cache, audio_hash)

    full = np.concatenate([audio_feat, beat_feat, signal_feat, tempo_feat])
    if full.shape[0] != TOTAL_FEATURES:
        return None
    return full


def extract_all_features(
    entries: list[Entry],
    onset_cache_dir: Path,
    analysis_cache,
    label: str = "",
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract MeterNet features for all entries.

    Uses a two-level cache:
    1. Full 1467-dim vector in meter_net_features_cache/ (fast: 1 np.load)
    2. Fallback: audio features from onset_features_cache_v5 + 15 JSON reads

    Returns (X, y_multi_hot, primary_meters).
    """
    meter_to_idx = {m: i for i, m in enumerate(CLASS_METERS_6)}
    n_classes = len(CLASS_METERS_6)

    # Full feature cache directory
    full_cache_dir = onset_cache_dir.parent / "meter_net_features_cache"
    full_cache_dir.mkdir(parents=True, exist_ok=True)

    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    primary_meters: list[int] = []
    skipped = 0
    need_extract: list[tuple[int, Path, int, dict[int, float]]] = []
    n_full_cached = 0

    onset_cache_dir.mkdir(parents=True, exist_ok=True)

    desc = f"Loading {label}" if label else "Loading"

    # Phase 1: Load from full feature cache, or onset cache + analysis cache
    for i, (audio_path, primary_meter, meters_dict) in enumerate(
        _progress(entries, desc)
    ):
        # Try full feature cache first (1 np.load, instant)
        full_key = _full_cache_key(audio_path)
        full_cache_path = full_cache_dir / f"{full_key}.npy"
        if full_cache_path.exists():
            full = np.load(full_cache_path)
            if full.shape[0] == TOTAL_FEATURES:
                features_list.append(full)
                y = np.zeros(n_classes, dtype=np.float32)
                for m, w in meters_dict.items():
                    idx = meter_to_idx.get(m)
                    if idx is not None:
                        y[idx] = float(w)
                labels_list.append(y)
                primary_meters.append(primary_meter)
                n_full_cached += 1
                continue

        # Try onset cache + analysis cache (15 JSON reads)
        onset_key = _onset_cache_key(audio_path)
        onset_cache_path = onset_cache_dir / f"{onset_key}.npy"

        audio_feat = None
        if onset_cache_path.exists():
            audio_feat = np.load(onset_cache_path)
            if audio_feat.shape[0] != TOTAL_FEATURES_V5:
                audio_feat = None

        if audio_feat is None:
            need_extract.append((i, audio_path, primary_meter, meters_dict))
            continue

        full = _build_full_features(audio_feat, audio_path, analysis_cache)
        if full is None:
            need_extract.append((i, audio_path, primary_meter, meters_dict))
            continue

        # Save to full feature cache for next time
        np.save(full_cache_path, full)

        features_list.append(full)
        y = np.zeros(n_classes, dtype=np.float32)
        for m, w in meters_dict.items():
            idx = meter_to_idx.get(m)
            if idx is not None:
                y[idx] = float(w)
        labels_list.append(y)
        primary_meters.append(primary_meter)

    n_from_json = len(features_list) - n_full_cached
    if n_full_cached or n_from_json:
        print(f"  {n_full_cached} full-cached, {n_from_json} from analysis cache, "
              f"{len(need_extract)} need audio extraction")

    # Phase 2: Extract missing audio features
    if need_extract:
        desc2 = f"Extracting {label}" if label else "Extracting"

        def _handle_extracted(audio_feat, audio_path, primary_meter, meters_dict):
            if audio_feat is None:
                return False

            # Save onset cache
            onset_key = _onset_cache_key(audio_path)
            np.save(onset_cache_dir / f"{onset_key}.npy", audio_feat)

            # Build and save full features
            full = _build_full_features(audio_feat, audio_path, analysis_cache)
            if full is None:
                return False

            full_key = _full_cache_key(audio_path)
            np.save(full_cache_dir / f"{full_key}.npy", full)

            features_list.append(full)
            y = np.zeros(n_classes, dtype=np.float32)
            for m, w in meters_dict.items():
                idx = meter_to_idx.get(m)
                if idx is not None:
                    y[idx] = float(w)
            labels_list.append(y)
            primary_meters.append(primary_meter)
            return True

        if workers > 1 and len(need_extract) > 2:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            work = [(str(ap), "v5") for _, ap, _, _ in need_extract]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(extract_features_from_path, w[0], w[1]): idx
                    for idx, w in enumerate(work)
                }
                for future in _progress(
                    as_completed(futures), desc2, total=len(futures)
                ):
                    result_idx = futures[future]
                    _, audio_path, primary_meter, meters_dict = need_extract[
                        result_idx
                    ]
                    if not _handle_extracted(future.result(), audio_path, primary_meter, meters_dict):
                        skipped += 1
        else:
            for _, audio_path, primary_meter, meters_dict in _progress(
                need_extract, desc2
            ):
                audio_feat = extract_features_from_path(str(audio_path), "v5")
                if not _handle_extracted(audio_feat, audio_path, primary_meter, meters_dict):
                    skipped += 1

    if skipped:
        print(f"  ({skipped} files skipped — feature extraction failed)")

    X = np.stack(features_list).astype(np.float32)
    y = np.stack(labels_list).astype(np.float32)
    return X, y, primary_meters


def _progress(iterable, desc="", total=None):
    """tqdm wrapper."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, leave=False, total=total)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# Model (Residual MLP — same architecture as onset_mlp v5)
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MeterNet(nn.Module):
    """Residual MLP for unified meter classification.

    Architecture: input -> hidden -> N x hidden (residual) -> hidden*0.4 -> hidden*0.2 -> n_classes
    n_blocks controls the number of residual blocks (depth).
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 640,
        dropout_scale: float = 1.0,
        n_blocks: int = 1,
    ):
        super().__init__()
        ds = dropout_scale
        h2 = max(int(hidden * 0.4), 64)
        h4 = max(int(hidden * 0.2), 32)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(min(0.3 * ds, 0.5)),
        )
        self.residual = nn.Sequential(
            *[ResidualBlock(hidden, dropout=min(0.25 * ds, 0.5)) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(min(0.2 * ds, 0.5)),
            nn.Linear(h2, h4),
            nn.BatchNorm1d(h4),
            nn.ReLU(),
            nn.Dropout(min(0.15 * ds, 0.5)),
            nn.Linear(h4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.residual(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training augmentation (feature-level)
# ---------------------------------------------------------------------------


def _cutmix_batch(
    xb: torch.Tensor, yb: torch.Tensor, alpha: float = CUTMIX_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix: replace contiguous feature segment with another sample's."""
    if alpha <= 0:
        return xb, yb

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xb.size(0), device=xb.device)

    feat_dim = xb.size(1)
    cut_len = int(feat_dim * (1 - lam))
    cut_start = np.random.randint(0, max(1, feat_dim - cut_len))
    cut_end = cut_start + cut_len

    mixed_x = xb.clone()
    mixed_x[:, cut_start:cut_end] = xb[idx, cut_start:cut_end]

    actual_lam = 1 - cut_len / feat_dim
    mixed_y = actual_lam * yb + (1 - actual_lam) * yb[idx]

    return mixed_x, mixed_y


FOCAL_GAMMA = 2.0


def _focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """Focal loss for multi-label BCE (sigmoid).

    Down-weights well-classified examples, focusing on hard ones.
    """
    probs = torch.sigmoid(logits)
    # p_t: probability of correct class
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Standard BCE with pos_weight
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none",
    )
    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization fitted on train set."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    primary_train: list[int],
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 8e-4,
    device: str = "cpu",
    hidden: int = 640,
    dropout_scale: float = 1.0,
    seed: int = 42,
    use_focal: bool = False,
    n_blocks: int = 1,
) -> tuple[MeterNet, dict]:
    """Train MeterNet with BCE/Focal loss and balanced accuracy validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = y_train.shape[1]
    input_dim = X_train.shape[1]
    meter_to_idx = {m: i for i, m in enumerate(CLASS_METERS_6)}
    idx_to_meter = {i: m for m, i in meter_to_idx.items()}

    # Class weights: inverse frequency of primary meter
    primary_idx = np.array([meter_to_idx[m] for m in primary_train])
    counts = Counter(primary_idx)
    total = len(primary_idx)

    # pos_weight for BCE loss (balance positive/negative per class)
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    raw_pw = neg_counts / np.maximum(pos_counts, 1)
    pos_weight = torch.tensor(
        np.clip(raw_pw, 0.5, 10.0), dtype=torch.float32,
    ).to(device)

    # WeightedRandomSampler based on primary meter
    sample_weights = np.array(
        [total / (n_classes * counts[c]) for c in primary_idx],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(primary_idx),
        replacement=True,
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
    )
    val_dl = DataLoader(val_ds, batch_size=256)

    model = MeterNet(
        input_dim, n_classes, hidden=hidden, dropout_scale=dropout_scale,
        n_blocks=n_blocks,
    ).to(device)
    if use_focal:
        def criterion(logits, targets):
            return _focal_bce_loss(logits, targets, pos_weight)
        print(f"  Loss: Focal BCE (gamma={FOCAL_GAMMA})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2,
    )

    best_val_metric = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience_limit = 50

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Feature augmentation: small noise + random scaling
            noise = torch.randn_like(xb) * 0.02
            scale = 1.0 + (torch.rand(xb.size(0), 1, device=device) - 0.5) * 0.1
            xb = xb * scale + noise

            # CutMix
            xb_mixed, yb_mixed = _cutmix_batch(xb, yb)

            optimizer.zero_grad()
            logits = model(xb_mixed)
            loss = criterion(logits, yb_mixed)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = torch.sigmoid(model(xb)).argmax(1)
                targets = yb.argmax(1)
            train_correct += (preds == targets).sum().item()
            train_total += xb.size(0)

        scheduler.step()

        # Validate: balanced accuracy
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

        per_class_accs = []
        for cls_idx in range(n_classes):
            t = val_total_per_class.get(cls_idx, 0)
            c = val_correct_per_class.get(cls_idx, 0)
            if t > 0:
                per_class_accs.append(c / t)
        val_balanced = np.mean(per_class_accs) if per_class_accs else 0.0
        val_overall = sum(val_correct_per_class.values()) / max(
            sum(val_total_per_class.values()), 1
        )

        train_acc = train_correct / max(train_total, 1)

        is_best = val_balanced > best_val_metric
        if is_best:
            best_val_metric = val_balanced
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5 or is_best:
            per_class_str = "  ".join(
                f"{idx_to_meter[i]}/x:{val_correct_per_class.get(i, 0)}/{val_total_per_class.get(i, 0)}"
                for i in sorted(val_total_per_class.keys())
            )
            marker = " *" if is_best else ""
            print(
                f"  ep {epoch:3d}  "
                f"train {train_acc:.1%} loss {train_loss / train_total:.3f}  "
                f"val_balanced {val_balanced:.1%} overall {val_overall:.1%}  "
                f"[{per_class_str}]{marker}"
            )

        if patience_counter >= patience_limit:
            print(f"  Early stopping at epoch {epoch} (best was ep {best_epoch})")
            break

    if best_state is None:
        print("WARNING: No improvement, using final state")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_val_acc": best_val_metric}
    return model, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Evaluate model on test set with per-class and overall accuracy."""
    idx_to_meter = {i: m for i, m in enumerate(CLASS_METERS_6)}
    n_classes = len(CLASS_METERS_6)

    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_t))
        preds = probs.argmax(1).cpu().numpy()
        true_primary = y_t.argmax(1).cpu().numpy()

    per_class = {}
    for cls_idx in sorted(set(true_primary)):
        mask = true_primary == cls_idx
        correct = int((preds[mask] == cls_idx).sum())
        total = int(mask.sum())
        meter = idx_to_meter[cls_idx]
        per_class[meter] = {"correct": correct, "total": total}

    overall_correct = int((preds == true_primary).sum())
    overall_total = len(true_primary)

    print(f"\nTest: {overall_correct}/{overall_total} = {overall_correct / overall_total:.1%}")
    print(f"{'Meter':>6s}  {'Correct':>8s}  {'Total':>6s}  {'Acc':>6s}")
    print("-" * 36)
    for meter in sorted(per_class.keys()):
        info = per_class[meter]
        acc = info["correct"] / max(info["total"], 1)
        print(f"{meter:>4d}/x  {info['correct']:>4d}/{info['total']:<4d}  {acc:>6.1%}")
    print("-" * 36)
    print(f"{'Total':>6s}  {overall_correct:>4d}/{overall_total:<4d}  {overall_correct / overall_total:>6.1%}")

    return {
        "overall_acc": overall_correct / max(overall_total, 1),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train MeterNet")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/wikimeter"),
        help="Primary data dir (WIKIMETER)",
    )
    parser.add_argument(
        "--meter2800", type=Path, default=None,
        help="METER2800 data dir (train+val added to training pool)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden", type=int, default=640)
    parser.add_argument("--dropout-scale", type=float, default=1.0)
    parser.add_argument("--n-blocks", type=int, default=1, help="Number of residual blocks (1-3)")
    parser.add_argument("--focal", action="store_true", help="Use focal BCE loss")
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    valid_meters = set(CLASS_METERS_6)
    meter_to_idx = {m: i for i, m in enumerate(CLASS_METERS_6)}

    print(f"MeterNet (features={TOTAL_FEATURES})")
    print(f"  Audio: {N_AUDIO_FEATURES}, Beat: {N_BEAT_FEATURES}, "
          f"Signal: {N_SIGNAL_FEATURES}, Tempo: {N_TEMPO_FEATURES}")
    print(f"Classes: {CLASS_METERS_6}")
    print(f"Primary data: {args.data_dir}"
          + (f" + METER2800: {args.meter2800}" if args.meter2800 else ""))

    # Load WIKIMETER entries (split by hash)
    print("\nLoading data...")
    wiki_entries = load_wikimeter(args.data_dir, valid_meters)
    wiki_train, wiki_val, wiki_test = [], [], []
    for entry in wiki_entries:
        split = split_by_stem(entry[0].stem)
        if split == "train":
            wiki_train.append(entry)
        elif split == "val":
            wiki_val.append(entry)
        else:
            wiki_test.append(entry)
    print(f"  WIKIMETER: {len(wiki_train)} train, {len(wiki_val)} val, {len(wiki_test)} test")

    train_entries = list(wiki_train)
    val_entries = list(wiki_val)
    test_entries = list(wiki_test)

    # Extra: METER2800 — each split goes where it belongs
    if args.meter2800:
        m2800_train = load_meter2800_split(args.meter2800, "train", valid_meters)
        m2800_val = load_meter2800_split(args.meter2800, "val", valid_meters)
        m2800_test = load_meter2800_split(args.meter2800, "test", valid_meters)
        print(f"  METER2800: {len(m2800_train)} train, {len(m2800_val)} val, {len(m2800_test)} test")
        train_entries.extend(m2800_train)
        val_entries.extend(m2800_val)
        test_entries.extend(m2800_test)

    if args.limit:
        train_entries = train_entries[:args.limit]
        val_entries = val_entries[:args.limit]
        test_entries = test_entries[:args.limit]

    print(f"  Total: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test")

    for name, entries in [("Train", train_entries), ("Val", val_entries), ("Test", test_entries)]:
        dist = Counter(m for _, m, _ in entries)
        dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist.keys()))
        print(f"  {name}: {dist_str}")

    # Initialize analysis cache
    from beatmeter.analysis.cache import AnalysisCache
    analysis_cache = AnalysisCache()

    # Extract features
    onset_cache_dir = Path(f"data/onset_features_cache_{FEATURE_VERSION_V5}")
    print(f"\nExtracting features (onset cache: {onset_cache_dir})...")
    t0 = time.time()

    X_train, y_train, pm_train = extract_all_features(
        train_entries, onset_cache_dir, analysis_cache, "train", args.workers,
    )
    X_val, y_val, pm_val = extract_all_features(
        val_entries, onset_cache_dir, analysis_cache, "val",
    )
    X_test, y_test, pm_test = extract_all_features(
        test_entries, onset_cache_dir, analysis_cache, "test",
    )

    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(f"  Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # Check feature completeness (non-zero groups)
    for name, X in [("Train", X_train), ("Val", X_val)]:
        audio_ok = np.any(X[:, :N_AUDIO_FEATURES] != 0, axis=1).sum()
        beat_ok = np.any(
            X[:, N_AUDIO_FEATURES : N_AUDIO_FEATURES + N_BEAT_FEATURES] != 0,
            axis=1,
        ).sum()
        sig_ok = np.any(
            X[
                :,
                N_AUDIO_FEATURES + N_BEAT_FEATURES : N_AUDIO_FEATURES
                + N_BEAT_FEATURES
                + N_SIGNAL_FEATURES,
            ]
            != 0,
            axis=1,
        ).sum()
        print(
            f"  {name} non-zero: audio={audio_ok}/{len(X)}, "
            f"beat={beat_ok}/{len(X)}, signal={sig_ok}/{len(X)}"
        )

    # Standardize
    X_train, X_val, X_test, feat_mean, feat_std = _standardize(X_train, X_val, X_test)
    print("  Features standardized (z-score, fit on train)")

    # Train
    print(
        f"\nTraining MeterNet (input={X_train.shape[1]}, hidden={args.hidden}, "
        f"blocks={args.n_blocks}, dropout_scale={args.dropout_scale}, classes={len(CLASS_METERS_6)})..."
    )
    model, info = train_model(
        X_train, y_train, pm_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        hidden=args.hidden,
        dropout_scale=args.dropout_scale,
        seed=args.seed,
        use_focal=args.focal,
        n_blocks=args.n_blocks,
    )
    print(f"\nBest val balanced acc: {info['best_val_acc']:.1%} at epoch {info['best_epoch']}")

    # Test
    results = evaluate(model, X_test, y_test, args.device)

    # Save
    if args.save:
        save_path = args.save
    elif args.limit:
        save_path = Path("data/meter_net_test.pt")
        print(f"  (--limit active, saving to {save_path})")
    else:
        save_path = Path("data/meter_net.pt")

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_meters": CLASS_METERS_6,
            "meter_to_idx": {m: i for i, m in enumerate(CLASS_METERS_6)},
            "input_dim": X_train.shape[1],
            "n_classes": len(CLASS_METERS_6),
            "best_epoch": info["best_epoch"],
            "best_val_acc": info["best_val_acc"],
            "test_results": results,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_version": FEATURE_VERSION,
            "hidden_dim": args.hidden,
            "dropout_scale": args.dropout_scale,
            "n_blocks": args.n_blocks,
        },
        save_path,
    )
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
