#!/usr/bin/env python3
"""Train MLP on multi-tempo, multi-signal autocorrelation features for meter classification.

v5: All v4 features (876) + 4th tempo candidate (+256) + beat-position histograms (+160)
    + autocorrelation ratios (+60) + tempogram meter salience (+9) = 1361-dim features.
+ Residual MLP (640→640 with skip), CutMix, AdamW, CosineAnnealingWarmRestarts.
+ Audio-level augmentation (--augment N): random crop, time stretch, pitch shift, noise.

WIKIMETER-primary: WIKIMETER is the primary dataset (6 classes, balanced).
METER2800 splits go to proper train/val/test (not all dumped into training pool).

Usage:
    # WIKIMETER-primary + METER2800 as extra training data
    uv run python scripts/training/train_onset_mlp.py --meter2800 data/meter2800

    # With audio augmentation (4 augmented copies per rare-class file)
    uv run python scripts/training/train_onset_mlp.py --meter2800 data/meter2800 --augment 4

    # Quick test
    uv run python scripts/training/train_onset_mlp.py --meter2800 data/meter2800 --limit 20
"""

import argparse
import csv
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import librosa
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
    extract_features_v5,
    extract_features_from_path,
)
from scripts.utils import load_meter2800_entries as _load_meter2800_base, resolve_audio_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_VERSION = FEATURE_VERSION_V5
TOTAL_FEATURES = TOTAL_FEATURES_V5

LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
FOCAL_GAMMA = 2.0

CLASS_METERS_6 = [3, 4, 5, 7, 9, 11]

# Rare meters that get more augmentation
RARE_METERS = {5, 7, 9, 11}

# ---------------------------------------------------------------------------
# Audio augmentation
# ---------------------------------------------------------------------------


def _augment_audio(y: np.ndarray, sr: int, aug_type: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a single audio augmentation.

    All augmentations are fast (no STFT). Designed for meter/rhythm features:
    - random_crop: changes beat phase alignment (most useful)
    - noise: simulates recording quality variation
    - gain: random volume scaling per segment
    - time_mask: zeros random segments, simulates dropouts

    Args:
        y: Audio signal.
        sr: Sample rate.
        aug_type: One of "random_crop", "noise", "gain", "time_mask".
        rng: Random number generator.

    Returns:
        Augmented audio array (same sr).
    """
    if aug_type == "random_crop":
        max_samples = sr * MAX_DURATION_S
        if len(y) > max_samples:
            max_start = len(y) - max_samples
            start = rng.integers(0, max_start)
            y = y[start:start + max_samples]
        return y

    elif aug_type == "noise":
        snr_db = rng.uniform(20.0, 30.0)
        signal_power = np.mean(y ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(max(noise_power, 1e-10)), size=len(y))
        return y + noise.astype(y.dtype)

    elif aug_type == "gain":
        # Random gain per ~2s segment — changes RMS/onset strength patterns
        seg_len = sr * 2
        n_segs = max(1, len(y) // seg_len)
        gains = rng.uniform(0.5, 1.5, size=n_segs)
        for i, g in enumerate(gains):
            start = i * seg_len
            end = min(start + seg_len, len(y))
            y[start:end] = y[start:end] * g
        return y

    elif aug_type == "time_mask":
        # Zero out 1-3 random segments of 0.2-0.8s — simulates dropouts
        n_masks = rng.integers(1, 4)
        for _ in range(n_masks):
            mask_len = int(rng.uniform(0.2, 0.8) * sr)
            start = rng.integers(0, max(1, len(y) - mask_len))
            y[start:start + mask_len] = 0.0
        return y

    return y


def _file_rng(audio_path: Path, aug_idx: int) -> np.random.Generator:
    """Deterministic RNG per (file, augmentation index).

    Same file + same index always produces the same augmentation,
    regardless of processing order or number of workers.
    """
    seed_str = f"{audio_path.resolve()}::aug_{aug_idx}"
    seed = int(hashlib.sha1(seed_str.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def _augment_and_extract(
    audio_path: Path,
    indices: list[int],
) -> dict[int, np.ndarray]:
    """Generate augmented feature vectors for specific indices.

    Args:
        audio_path: Path to audio file.
        indices: Which augmentation indices to generate (e.g. [2, 3]).

    Returns:
        Dict mapping aug index → feature array. Missing indices = extraction failed.
    """
    if not indices:
        return {}

    try:
        y_full, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception:
        return {}

    if len(y_full) < sr * 2:
        return {}

    aug_types = ["random_crop", "noise", "gain", "time_mask"]
    results = {}

    for i in indices:
        aug_type = aug_types[i % len(aug_types)]
        rng = _file_rng(audio_path, i)
        try:
            y_aug = _augment_audio(y_full.copy(), sr, aug_type, rng)
            max_samples = sr * MAX_DURATION_S
            if len(y_aug) > max_samples:
                start = (len(y_aug) - max_samples) // 2
                y_aug = y_aug[start:start + max_samples]
            feat = extract_features_v5(y_aug, sr)
            if feat is not None:
                results[i] = feat
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_meter2800_split(
    data_dir: Path, split: str, valid_meters: set[int]
) -> list[tuple[Path, int]]:
    """Load METER2800 entries for a given split with corrections applied."""
    return _load_meter2800_base(data_dir, split, valid_meters=valid_meters)


def load_wikimeter(data_dir: Path, valid_meters: set[int]) -> list[tuple[Path, int]]:
    """Load WIKIMETER entries. Uses primary meter from multi-label string."""
    tab_path = data_dir / "data_wikimeter.tab"
    if not tab_path.exists():
        return []

    audio_dir = data_dir / "audio"
    entries = []
    with open(tab_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fname = row["filename"].strip('"')
            meter_str = row["meter"].strip('"')
            best_meter, best_weight = None, -1.0
            for part in meter_str.split(","):
                part = part.strip()
                if ":" in part:
                    m_s, w_s = part.split(":", 1)
                    m, w = int(m_s.strip()), float(w_s.strip())
                else:
                    m, w = int(part), 1.0
                if m in valid_meters and w > best_weight:
                    best_meter, best_weight = m, w
            if best_meter is None:
                continue

            stem = Path(fname).stem
            audio_path = audio_dir / f"{stem}.mp3"
            if not audio_path.exists():
                audio_path = audio_dir / f"{stem}.wav"
            if not audio_path.exists():
                continue
            entries.append((audio_path, best_meter))
    return entries


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------


def _cache_key(audio_path: Path, suffix: str = "") -> str:
    st = audio_path.stat()
    raw = f"{audio_path.resolve()}::{st.st_size}::{st.st_mtime_ns}::{FEATURE_VERSION}{suffix}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _process_one_uncached(args_tuple):
    """Extract base features and/or missing augmentations for a single file.

    Called from ProcessPoolExecutor or sequentially.
    """
    audio_path_str, meter, cache_dir_str, need_base, missing_aug_indices, base_key, aug_key_map = args_tuple
    audio_path = Path(audio_path_str)
    cache_dir = Path(cache_dir_str)

    result = {"meter": meter, "feat": None, "aug_results": {}}

    if need_base:
        feat = extract_features_from_path(audio_path, version="v5")
        if feat is None:
            return result
        np.save(cache_dir / f"{base_key}.npy", feat)
        result["feat"] = feat

    if missing_aug_indices:
        aug_results = _augment_and_extract(audio_path, missing_aug_indices)
        for ai, af in aug_results.items():
            if ai in aug_key_map:
                np.save(cache_dir / f"{aug_key_map[ai]}.npy", af)
        result["aug_results"] = aug_results

    return result


def extract_with_cache(
    entries: list[tuple[Path, int]],
    cache_dir: Path,
    label: str = "",
    n_augment: int = 0,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for all entries, with disk caching and optional parallelism.

    Incremental: loads what's cached, extracts only what's missing.
    --augment 1 then --augment 4 will reuse aug_0 and only compute aug_1..3.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_list = []
    labels_list = []
    skipped = 0
    aug_count = 0
    uncached_work = []

    # Phase 1: load cached, identify gaps
    desc = f"Loading {label}" if label else "Loading"
    for audio_path, meter in _progress(entries, desc):
        key = _cache_key(audio_path)
        cache_path = cache_dir / f"{key}.npy"

        n_aug = 0
        if n_augment > 0:
            n_aug = n_augment if meter in RARE_METERS else max(1, n_augment // 3)

        need_base = not cache_path.exists()
        base_feat = None
        if not need_base:
            base_feat = np.load(cache_path)
            if base_feat.shape[0] != TOTAL_FEATURES:
                base_feat = None
                need_base = True

        # Check which augmentations are cached
        cached_augs = {}  # index → feat
        missing_aug_indices = []
        aug_key_map = {}  # index → cache key
        for ai in range(n_aug):
            ak = _cache_key(audio_path, f"::aug_{ai}")
            aug_key_map[ai] = ak
            aug_cache = cache_dir / f"{ak}.npy"
            if aug_cache.exists():
                af = np.load(aug_cache)
                if af.shape[0] == TOTAL_FEATURES:
                    cached_augs[ai] = af
                else:
                    missing_aug_indices.append(ai)
            else:
                missing_aug_indices.append(ai)

        if need_base or missing_aug_indices:
            uncached_work.append((
                str(audio_path), meter, str(cache_dir),
                need_base, missing_aug_indices, key, aug_key_map,
            ))
            # Store what we already have — will merge after extraction
            if base_feat is not None:
                features_list.append(base_feat)
                labels_list.append(meter)
            for ai in sorted(cached_augs.keys()):
                features_list.append(cached_augs[ai])
                labels_list.append(meter)
                aug_count += 1
        else:
            # Fully cached
            features_list.append(base_feat)
            labels_list.append(meter)
            for ai in sorted(cached_augs.keys()):
                features_list.append(cached_augs[ai])
                labels_list.append(meter)
                aug_count += 1

    n_cached = len(entries) - len(uncached_work)
    if uncached_work:
        n_need_base = sum(1 for _, _, _, nb, _, _, _ in uncached_work if nb)
        n_need_aug = sum(len(mai) for _, _, _, _, mai, _, _ in uncached_work)
        print(f"  {n_cached} fully cached, {n_need_base} base + {n_need_aug} aug to extract")
    elif aug_count:
        print(f"  ({aug_count} augmented samples from cache)")

    if not uncached_work:
        X = np.stack(features_list).astype(np.float32)
        y = np.array(labels_list, dtype=np.int64)
        return X, y

    # Phase 2: extract missing (parallel if workers > 1)
    desc2 = f"Extracting {label}" if label else "Extracting"

    def _handle_result(result):
        nonlocal skipped, aug_count
        meter = result["meter"]
        if result["feat"] is not None:
            features_list.append(result["feat"])
            labels_list.append(meter)
        elif result["feat"] is None and not result["aug_results"]:
            # Base extraction failed and no augs
            skipped += 1
            return
        for ai in sorted(result["aug_results"].keys()):
            features_list.append(result["aug_results"][ai])
            labels_list.append(meter)
            aug_count += 1

    if workers > 1 and len(uncached_work) > 2:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one_uncached, item): item for item in uncached_work}
            for future in _progress(as_completed(futures), desc2, total=len(futures)):
                _handle_result(future.result())
    else:
        for item in _progress(uncached_work, desc2):
            _handle_result(_process_one_uncached(item))

    if skipped:
        print(f"  ({skipped} files skipped — extraction failed)")
    if aug_count:
        print(f"  ({aug_count} augmented samples)")

    X = np.stack(features_list).astype(np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def _progress(iterable, desc="", total=None):
    """tqdm wrapper."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, leave=False, total=total)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Single residual block: Linear → BN → ReLU → Dropout → Linear → BN + skip."""

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


class OnsetMLPv5(nn.Module):
    """Residual MLP for meter classification (v5).

    Architecture: input → hidden → hidden (residual) → hidden//2 → hidden//4 → n_classes
    dropout_scale multiplies base dropout rates (0.3, 0.25, 0.2, 0.15).
    """

    def __init__(self, input_dim: int, n_classes: int, hidden: int = 640, dropout_scale: float = 1.0):
        super().__init__()
        ds = dropout_scale
        h2 = max(int(hidden * 0.4), 64)   # 640→256, 320→128, 1024→409
        h4 = max(int(hidden * 0.2), 32)   # 640→128, 320→64,  1024→204
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(min(0.3 * ds, 0.5)),
        )
        self.residual = ResidualBlock(hidden, dropout=min(0.25 * ds, 0.5))
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



def _focal_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """Focal loss with soft targets (for mixup/cutmix) and class weights."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    focal_weight = (1.0 - (probs * targets_soft).sum(dim=1)) ** gamma
    ce = -(targets_soft * log_probs).sum(dim=1)
    sample_weight = (targets_soft * class_weights).sum(dim=1)
    return (focal_weight * ce * sample_weight).mean()


# ---------------------------------------------------------------------------
# Training augmentation (feature-level)
# ---------------------------------------------------------------------------


def _mixup_batch(
    xb: torch.Tensor, yb: torch.Tensor, n_classes: int, alpha: float = MIXUP_ALPHA
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation: interpolate random pairs of examples."""
    if alpha <= 0:
        one_hot = torch.zeros(yb.size(0), n_classes, device=yb.device)
        one_hot.scatter_(1, yb.unsqueeze(1), 1.0)
        return xb, one_hot

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xb.size(0), device=xb.device)

    mixed_x = lam * xb + (1 - lam) * xb[idx]

    y_onehot = torch.zeros(yb.size(0), n_classes, device=yb.device)
    y_onehot.scatter_(1, yb.unsqueeze(1), 1.0)
    y_onehot_shuf = torch.zeros(yb.size(0), n_classes, device=yb.device)
    y_onehot_shuf.scatter_(1, yb[idx].unsqueeze(1), 1.0)
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot_shuf

    return mixed_x, mixed_y


def _cutmix_batch(
    xb: torch.Tensor, yb: torch.Tensor, n_classes: int, alpha: float = CUTMIX_ALPHA
) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix augmentation: replace a contiguous segment of features with another sample's."""
    if alpha <= 0:
        one_hot = torch.zeros(yb.size(0), n_classes, device=yb.device)
        one_hot.scatter_(1, yb.unsqueeze(1), 1.0)
        return xb, one_hot

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xb.size(0), device=xb.device)

    # For 1D features, "cut" is a contiguous range of feature indices
    feat_dim = xb.size(1)
    cut_len = int(feat_dim * (1 - lam))
    cut_start = np.random.randint(0, max(1, feat_dim - cut_len))
    cut_end = cut_start + cut_len

    mixed_x = xb.clone()
    mixed_x[:, cut_start:cut_end] = xb[idx, cut_start:cut_end]

    # Actual lambda after cut
    actual_lam = 1 - cut_len / feat_dim

    y_onehot = torch.zeros(yb.size(0), n_classes, device=yb.device)
    y_onehot.scatter_(1, yb.unsqueeze(1), 1.0)
    y_onehot_shuf = torch.zeros(yb.size(0), n_classes, device=yb.device)
    y_onehot_shuf.scatter_(1, yb[idx].unsqueeze(1), 1.0)
    mixed_y = actual_lam * y_onehot + (1 - actual_lam) * y_onehot_shuf

    return mixed_x, mixed_y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
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
    X_val: np.ndarray,
    y_val: np.ndarray,
    meter_to_idx: dict[int, int],
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 8e-4,
    device: str = "cpu",
    hidden: int = 640,
    dropout_scale: float = 1.0,
) -> tuple[OnsetMLPv5, dict]:
    """Train the OnsetMLPv5 model."""
    n_classes = len(meter_to_idx)
    input_dim = X_train.shape[1]

    y_train_idx = np.array([meter_to_idx[m] for m in y_train])
    y_val_idx = np.array([meter_to_idx[m] for m in y_val])

    # Class weights for focal loss
    counts = Counter(y_train_idx)
    total = len(y_train_idx)
    class_weights = torch.tensor(
        [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(device)

    # WeightedRandomSampler — oversample rare classes
    sample_weights = np.array([
        total / (n_classes * counts[c]) for c in y_train_idx
    ], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(y_train_idx),
        replacement=True,
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_idx, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_idx, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = OnsetMLPv5(input_dim, n_classes, hidden=hidden, dropout_scale=dropout_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2,
    )

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience_limit = 50

    idx_to_meter = {v: k for k, v in meter_to_idx.items()}

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
            # 50/50 Mixup vs CutMix
            if np.random.random() < 0.5:
                xb_mixed, yb_mixed = _mixup_batch(xb, yb, n_classes)
            else:
                xb_mixed, yb_mixed = _cutmix_batch(xb, yb, n_classes)

            optimizer.zero_grad()
            logits = model(xb_mixed)
            loss = _focal_loss(logits, yb_mixed, class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = model(xb).argmax(1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = Counter()
        per_class_total = Counter()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)
                for pred, true in zip(preds.cpu().numpy(), yb.cpu().numpy()):
                    per_class_total[true] += 1
                    if pred == true:
                        per_class_correct[true] += 1

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if epoch % 10 == 0 or epoch <= 5 or val_acc > best_val_acc:
            per_class_str = "  ".join(
                f"{idx_to_meter[i]}/x:{per_class_correct[i]}/{per_class_total[i]}"
                for i in sorted(per_class_total.keys())
            )
            marker = " *" if val_acc > best_val_acc else ""
            print(
                f"  ep {epoch:3d}  "
                f"train {train_acc:.1%} loss {train_loss/train_total:.3f}  "
                f"val {val_acc:.1%}  [{per_class_str}]{marker}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping at epoch {epoch} (best was ep {best_epoch})")
                break

    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_val_acc": best_val_acc}
    return model, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meter_to_idx: dict[int, int],
    device: str = "cpu",
) -> dict:
    """Evaluate model on test set. Returns per-class and overall accuracy."""
    idx_to_meter = {v: k for k, v in meter_to_idx.items()}
    y_test_idx = np.array([meter_to_idx[m] for m in y_test])

    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(1).cpu().numpy()

    per_class = {}
    for cls_idx in sorted(set(y_test_idx)):
        mask = y_test_idx == cls_idx
        correct = (preds[mask] == cls_idx).sum()
        total = mask.sum()
        meter = idx_to_meter[cls_idx]
        per_class[meter] = {"correct": int(correct), "total": int(total)}

    overall_correct = (preds == y_test_idx).sum()
    overall_total = len(y_test_idx)

    print(f"\nTest: {overall_correct}/{overall_total} = {overall_correct/overall_total:.1%}")
    print(f"{'Meter':>6s}  {'Correct':>8s}  {'Total':>6s}  {'Acc':>6s}")
    print("-" * 36)
    for meter in sorted(per_class.keys()):
        info = per_class[meter]
        acc = info["correct"] / max(info["total"], 1)
        print(f"{meter:>4d}/x  {info['correct']:>4d}/{info['total']:<4d}  {acc:>6.1%}")
    print("-" * 36)
    print(f"{'Total':>6s}  {overall_correct:>4d}/{overall_total:<4d}  {overall_correct/overall_total:>6.1%}")

    return {
        "overall_acc": overall_correct / max(overall_total, 1),
        "overall_correct": int(overall_correct),
        "overall_total": overall_total,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train onset MLP v5")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wikimeter"),
                        help="Primary data dir (WIKIMETER)")
    parser.add_argument("--meter2800", type=Path, default=None,
                        help="METER2800 data dir (train+val added to training pool)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit entries per split (0=all)")
    parser.add_argument("--augment", type=int, default=0,
                        help="N augmented copies per file (0=off). Rare classes get N, common get N//3.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for feature extraction (default: 1)")
    parser.add_argument("--hidden", type=int, default=640,
                        help="Hidden layer size (default: 640)")
    parser.add_argument("--dropout-scale", type=float, default=1.0,
                        help="Dropout multiplier (default: 1.0)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save model to this path")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    class_meters = CLASS_METERS_6
    meter_to_idx = {m: i for i, m in enumerate(class_meters)}
    valid_meters = set(class_meters)

    print(f"Onset MLP v5 (features={TOTAL_FEATURES})")
    print(f"Classes: {class_meters}")
    print(f"Primary data: {args.data_dir}" + (f" + METER2800: {args.meter2800}" if args.meter2800 else ""))
    if args.augment:
        print(f"Augmentation: {args.augment}x rare, {max(1, args.augment // 3)}x common")

    # Load entries — WIKIMETER-primary
    print("\nLoading data...")
    from scripts.utils import split_by_stem
    wiki_entries = load_wikimeter(args.data_dir, valid_meters)
    wiki_train, wiki_val, wiki_test = [], [], []
    for path, meter in wiki_entries:
        split = split_by_stem(path.stem)
        if split == "train":
            wiki_train.append((path, meter))
        elif split == "val":
            wiki_val.append((path, meter))
        else:
            wiki_test.append((path, meter))
    print(f"  WIKIMETER: {len(wiki_train)} train, {len(wiki_val)} val, {len(wiki_test)} test")

    train_entries = wiki_train
    val_entries = wiki_val
    test_entries = wiki_test

    # Extra: METER2800 splits → proper train/val/test
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

    print(f"  Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")

    for name, entries in [("Train", train_entries), ("Val", val_entries), ("Test", test_entries)]:
        dist = Counter(m for _, m in entries)
        dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist.keys()))
        print(f"  {name}: {dist_str}")

    # Extract features
    cache_dir = Path(f"data/onset_features_cache_{FEATURE_VERSION}")
    print(f"\nExtracting features (cache: {cache_dir})...")
    t0 = time.time()
    X_train, y_train = extract_with_cache(train_entries, cache_dir, "train", n_augment=args.augment, workers=args.workers)
    X_val, y_val = extract_with_cache(val_entries, cache_dir, "val", workers=args.workers)
    X_test, y_test = extract_with_cache(test_entries, cache_dir, "test", workers=args.workers)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(f"  Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # Post-augmentation distribution
    if args.augment:
        dist = Counter(y_train)
        dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist))
        print(f"  Train (with aug): {dist_str}")

    # Z-score standardization
    X_train, X_val, X_test, feat_mean, feat_std = _standardize(X_train, X_val, X_test)
    print("  Features standardized (z-score, fit on train)")

    # Train
    print(f"\nTraining Residual MLP (input={X_train.shape[1]}, hidden={args.hidden}, "
          f"dropout_scale={args.dropout_scale}, classes={len(class_meters)})...")
    model, info = train_model(
        X_train, y_train, X_val, y_val,
        meter_to_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        hidden=args.hidden,
        dropout_scale=args.dropout_scale,
    )
    print(f"\nBest val acc: {info['best_val_acc']:.1%} at epoch {info['best_epoch']}")

    # Test
    results = evaluate(model, X_test, y_test, meter_to_idx, args.device)

    # Save (skip auto-save for --limit runs to avoid overwriting production checkpoint)
    if args.save:
        save_path = args.save
    elif args.limit:
        save_path = Path(f"data/meter_onset_mlp_test.pt")
        print(f"  (--limit active, saving to {save_path} instead of production checkpoint)")
    else:
        save_path = Path("data/meter_onset_mlp.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_meters": class_meters,
            "meter_to_idx": meter_to_idx,
            "input_dim": X_train.shape[1],
            "n_classes": len(class_meters),
            "best_epoch": info["best_epoch"],
            "best_val_acc": info["best_val_acc"],
            "test_results": results,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_version": FEATURE_VERSION,
            "arch_version": "v5",
            "hidden_dim": args.hidden,
            "dropout_scale": args.dropout_scale,
        },
        save_path,
    )
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
