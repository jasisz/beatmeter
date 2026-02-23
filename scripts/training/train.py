#!/usr/bin/env python3
"""Train MeterNet — unified meter classification from audio + MERT features.

MeterNet sees:
- 1449 audio features (from onset_mlp_features.py v6)
- 1536 MERT-v1-95M embedding (layer 3)
Total: 2985 dimensions.

Architecture: Residual MLP (parameterized via grid search).
Loss: BCEWithLogitsLoss (multi-label, supports WIKIMETER polymetric annotations).
Val metric: balanced accuracy (macro per-class, prevents rare-class sacrifice).

Usage:
    # Smoke test
    uv run python scripts/training/train.py --meter2800 data/meter2800 --limit 3 --workers 1

    # Full training
    uv run python scripts/training/train.py --meter2800 data/meter2800 --workers 4
"""

import argparse
import csv
import hashlib
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
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
    FEATURE_VERSION_V6,
    TOTAL_FEATURES_V6,
    extract_features_v6,
)
from beatmeter.analysis.signals.meter_net_features import (
    TOTAL_FEATURES,
    FEATURE_VERSION,
    N_AUDIO_FEATURES,
    N_MERT_FEATURES,
    MERT_LAYER,
    ALL_GROUP_NAMES,
    GROUP_DIMS,
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


# Feature dimensions
_TOTAL_FEATURES_ACTIVE = TOTAL_FEATURES
_AUDIO_FEATURES_ACTIVE = TOTAL_FEATURES_V6

# MERT configuration (set by main() based on CLI args)
_MERT_LOOKUP: dict[str, Path] | None = None
_MERT_LAYER_ACTIVE: int = MERT_LAYER
_N_MERT_ACTIVE: int = 0  # 0 = no MERT, N_MERT_FEATURES = with MERT


def _build_mert_lookup(*dirs: Path) -> dict[str, Path]:
    """Build stem -> .npy path lookup from MERT embedding directories."""
    lookup: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for npy in d.glob("*.npy"):
            lookup[npy.stem] = npy
    return lookup


def _load_mert_embedding(audio_path: Path) -> np.ndarray:
    """Load MERT embedding for an audio file. Returns zeros if not found."""
    if _MERT_LOOKUP is None:
        return np.zeros(N_MERT_FEATURES, dtype=np.float32)
    npy_path = _MERT_LOOKUP.get(audio_path.stem)
    if npy_path is None:
        return np.zeros(N_MERT_FEATURES, dtype=np.float32)
    try:
        emb = np.load(npy_path)  # shape (12, 1536)
        return emb[_MERT_LAYER_ACTIVE].astype(np.float32)
    except Exception:
        return np.zeros(N_MERT_FEATURES, dtype=np.float32)


def _extract_audio_from_path(audio_path_str: str) -> np.ndarray | None:
    """Load audio and extract v6 features. Picklable for multiprocessing."""
    import librosa

    try:
        y, _ = librosa.load(audio_path_str, sr=SR, duration=MAX_DURATION_S)
        return extract_features_v6(y, SR)
    except Exception:
        return None


def _onset_cache_key(audio_path: Path) -> str:
    """Cache key for audio features cache (v6)."""
    st = audio_path.stat()
    raw = f"{audio_path.resolve()}::{st.st_size}::{st.st_mtime_ns}::{FEATURE_VERSION_V6}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _full_cache_key(audio_path: Path) -> str:
    """Cache key for full MeterNet feature vector."""
    if _N_MERT_ACTIVE == 0:
        suffix = "meter_net_v8_slim"
    else:
        suffix = f"meter_net_v8_slim_mert{_N_MERT_ACTIVE}_L{_MERT_LAYER_ACTIVE}"
    st = audio_path.stat()
    raw = f"{audio_path.resolve()}::{st.st_size}::{st.st_mtime_ns}::{suffix}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _build_full_features(
    audio_feat: np.ndarray,
    audio_path: Path,
    feat_db=None,
) -> np.ndarray | None:
    """Combine audio + MERT features."""
    parts = [audio_feat]

    # Optional MERT embedding
    if _N_MERT_ACTIVE > 0:
        mert_feat = _load_mert_embedding(audio_path)
        parts.append(mert_feat)

    full = np.concatenate(parts)
    if full.shape[0] != _TOTAL_FEATURES_ACTIVE:
        return None
    return full


def extract_all_features(
    entries: list[Entry],
    feat_db,
    label: str = "",
    workers: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract MeterNet features for all entries.

    Returns (X, y_multi_hot, primary_meters).
    """
    meter_to_idx = {m: i for i, m in enumerate(CLASS_METERS_6)}
    n_classes = len(CLASS_METERS_6)

    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    primary_meters: list[int] = []
    skipped = 0
    need_extract: list[tuple[int, Path, int, dict[int, float]]] = []
    n_full_cached = 0

    desc = f"Loading {label}" if label else "Loading"

    # Phase 1: Load from LMDB cache
    for i, (audio_path, primary_meter, meters_dict) in enumerate(
        _progress(entries, desc)
    ):
        # Try full feature cache first
        full_key = _full_cache_key(audio_path)
        full = feat_db.load(f"full:{full_key}")

        if full is not None and full.shape[0] == _TOTAL_FEATURES_ACTIVE:
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

        # Try onset cache
        onset_key = _onset_cache_key(audio_path)
        audio_feat = feat_db.load(f"onset:v6:{onset_key}")

        if audio_feat is not None and audio_feat.shape[0] == _AUDIO_FEATURES_ACTIVE:
            full = _build_full_features(audio_feat, audio_path, feat_db)
            if full is not None:
                feat_db.save(f"full:{full_key}", full)
                features_list.append(full)
                y = np.zeros(n_classes, dtype=np.float32)
                for m, w in meters_dict.items():
                    idx = meter_to_idx.get(m)
                    if idx is not None:
                        y[idx] = float(w)
                labels_list.append(y)
                primary_meters.append(primary_meter)
                continue

        need_extract.append((i, audio_path, primary_meter, meters_dict))

    n_from_cache = len(features_list) - n_full_cached
    if verbose and (n_full_cached or n_from_cache):
        print(f"  {n_full_cached} full-cached, {n_from_cache} from onset cache, "
              f"{len(need_extract)} need audio extraction")

    # Phase 2: Extract missing audio features
    if need_extract:
        desc2 = f"Extracting {label}" if label else "Extracting"

        def _handle_extracted(audio_feat, audio_path, primary_meter, meters_dict):
            if audio_feat is None:
                return False

            # Save onset cache
            onset_key = _onset_cache_key(audio_path)
            feat_db.save(f"onset:v6:{onset_key}", audio_feat)

            # Build and save full features
            full = _build_full_features(audio_feat, audio_path, feat_db)
            if full is None:
                return False

            full_key = _full_cache_key(audio_path)
            feat_db.save(f"full:{full_key}", full)

            features_list.append(full)
            y = np.zeros(n_classes, dtype=np.float32)
            for m, w in meters_dict.items():
                idx = meter_to_idx.get(m)
                if idx is not None:
                    y[idx] = float(w)
            labels_list.append(y)
            primary_meters.append(primary_meter)
            return True

        paths = [str(audio_path) for _, audio_path, _, _ in need_extract]

        if workers > 1 and len(need_extract) > 2:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_extract_audio_from_path, p): idx
                    for idx, p in enumerate(paths)
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
            for (path_str, (_, audio_path, primary_meter, meters_dict)) in zip(
                paths,
                _progress(need_extract, desc2),
            ):
                audio_feat = _extract_audio_from_path(path_str)
                if not _handle_extracted(audio_feat, audio_path, primary_meter, meters_dict):
                    skipped += 1

    if verbose and skipped:
        print(f"  ({skipped} files skipped — feature extraction failed)")

    n_extracted_ok = max(len(need_extract) - skipped, 0)
    label_tag = label if label else "data"
    print(
        f"  [{label_tag}] ready={len(features_list)}  "
        f"full_cache={n_full_cached}  reused={n_from_cache}  "
        f"extracted={n_extracted_ok}  skipped={skipped}"
    )

    X = np.stack(features_list).astype(np.float32)
    y = np.stack(labels_list).astype(np.float32)
    return X, y, primary_meters


def _progress(iterable, desc="", total=None):
    """tqdm wrapper."""
    try:
        from tqdm import tqdm

        return tqdm(
            iterable,
            desc=desc,
            leave=False,
            total=total,
            disable=not sys.stdout.isatty(),
        )
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# Model (Residual MLP)
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
    """Residual MLP for unified meter classification."""

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


def _mixup_batch(
    xb: torch.Tensor, yb: torch.Tensor, alpha: float = CUTMIX_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup: linear interpolation of full feature vectors."""
    if alpha <= 0:
        return xb, yb

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xb.size(0), device=xb.device)

    mixed_x = lam * xb + (1 - lam) * xb[idx]
    mixed_y = lam * yb + (1 - lam) * yb[idx]

    return mixed_x, mixed_y


def _stratified_kfold(
    primary_meters: list[int], n_folds: int, seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold split. Returns list of (train_indices, val_indices)."""
    rng = np.random.RandomState(seed)

    class_indices: dict[int, list[int]] = {}
    for i, m in enumerate(primary_meters):
        class_indices.setdefault(m, []).append(i)

    for m in class_indices:
        rng.shuffle(class_indices[m])

    fold_assignments = np.zeros(len(primary_meters), dtype=int)
    for indices in class_indices.values():
        for i, idx in enumerate(indices):
            fold_assignments[idx] = i % n_folds

    folds = []
    all_indices = np.arange(len(primary_meters))
    for k in range(n_folds):
        val_mask = fold_assignments == k
        folds.append((all_indices[~val_mask], all_indices[val_mask]))

    return folds


FOCAL_GAMMA = 2.0


def _focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """Focal loss for multi-label BCE (sigmoid)."""
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

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
    verbose: bool = False,
    log_every: int = 10,
    aug_mode: str = "cutmix",
) -> tuple[nn.Module, dict]:
    """Train MeterNet with BCE/Focal loss and balanced accuracy validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = y_train.shape[1]
    input_dim = X_train.shape[1]
    meter_to_idx = {m: i for i, m in enumerate(CLASS_METERS_6)}
    idx_to_meter = {i: m for m, i in meter_to_idx.items()}

    primary_idx = np.array([meter_to_idx[m] for m in primary_train])
    counts = Counter(primary_idx)
    total = len(primary_idx)

    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    raw_pw = neg_counts / np.maximum(pos_counts, 1)
    pos_weight = torch.tensor(
        np.clip(raw_pw, 0.5, 10.0), dtype=torch.float32,
    ).to(device)

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
    print(f"  Watch: val_balanced (macro). Early stop patience={patience_limit} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            noise = torch.randn_like(xb) * 0.02
            scale = 1.0 + (torch.rand(xb.size(0), 1, device=device) - 0.5) * 0.1
            xb = xb * scale + noise

            if aug_mode == "cutmix":
                xb_mixed, yb_mixed = _cutmix_batch(xb, yb)
            elif aug_mode == "mixup":
                xb_mixed, yb_mixed = _mixup_batch(xb, yb)
            else:
                xb_mixed, yb_mixed = xb, yb

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

        log_every = max(log_every, 1)
        should_log = epoch <= 3 or epoch % log_every == 0 or epoch == epochs
        if verbose and is_best and not should_log:
            should_log = True
        if should_log:
            marker = " *BEST" if is_best else ""
            msg = (
                f"  ep {epoch:3d}/{epochs}  "
                f"train_acc {train_acc:.1%}  "
                f"val_bal {val_balanced:.1%}  "
                f"val_overall {val_overall:.1%}  "
                f"best {best_val_metric:.1%} (ep {best_epoch}){marker}"
            )
            if verbose:
                per_class_str = "  ".join(
                    f"{idx_to_meter[i]}/x:{val_correct_per_class.get(i, 0)}/{val_total_per_class.get(i, 0)}"
                    for i in sorted(val_total_per_class.keys())
                )
                msg += f"  [{per_class_str}]"
            print(msg)

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
    verbose: bool = False,
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

    overall_acc = overall_correct / max(overall_total, 1)
    print(f"\nTest overall: {overall_correct}/{overall_total} = {overall_acc:.1%}")

    class_parts = []
    for meter in sorted(per_class.keys()):
        info = per_class[meter]
        acc = info["correct"] / max(info["total"], 1)
        class_parts.append(f"{meter}/x:{info['correct']}/{info['total']} ({acc:.1%})")

    if verbose:
        print(f"{'Meter':>6s}  {'Correct':>8s}  {'Total':>6s}  {'Acc':>6s}")
        print("-" * 36)
        for meter in sorted(per_class.keys()):
            info = per_class[meter]
            acc = info["correct"] / max(info["total"], 1)
            print(f"{meter:>4d}/x  {info['correct']:>4d}/{info['total']:<4d}  {acc:>6.1%}")
        print("-" * 36)
        print(f"{'Total':>6s}  {overall_correct:>4d}/{overall_total:<4d}  {overall_acc:>6.1%}")
    else:
        print("  Per-class: " + " | ".join(class_parts))

    return {
        "overall_acc": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _prune_staging(keep: int = 5):
    """Keep only the most recent `keep` checkpoints in data/checkpoints/."""
    ckpt_dir = Path("data/checkpoints")
    if not ckpt_dir.exists():
        return
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    for old in pts[:-keep]:
        old.unlink()
        print(f"  Pruned old checkpoint: {old.name}")


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
    parser.add_argument("--aug-mode", type=str, default="cutmix",
                        choices=["cutmix", "mixup", "none"],
                        help="Sample mixing augmentation (default: cutmix)")
    parser.add_argument("--no-mert", action="store_true",
                        help="Disable MERT embeddings (train without them)")
    parser.add_argument("--mert-layer", type=int, default=MERT_LAYER,
                        help=f"MERT layer to use (default {MERT_LAYER})")

    parser.add_argument("--kfold", type=int, default=0,
                        help="K-fold CV (0=off, 5=recommended)")
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print epoch metrics every N epochs")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed diagnostics")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    valid_meters = set(CLASS_METERS_6)
    meter_to_idx = {m: i for i, m in enumerate(CLASS_METERS_6)}

    # MERT configuration
    global _MERT_LOOKUP, _MERT_LAYER_ACTIVE, _N_MERT_ACTIVE, _TOTAL_FEATURES_ACTIVE
    if not args.no_mert:
        _MERT_LAYER_ACTIVE = args.mert_layer
        _MERT_LOOKUP = _build_mert_lookup(
            Path("data/mert_embeddings/meter2800"),
            Path("data/mert_embeddings/wikimeter"),
        )
        if _MERT_LOOKUP:
            _N_MERT_ACTIVE = N_MERT_FEATURES
            _TOTAL_FEATURES_ACTIVE = N_AUDIO_FEATURES + N_MERT_FEATURES
            print(f"MERT: {len(_MERT_LOOKUP)} embeddings, layer={_MERT_LAYER_ACTIVE}, dims={N_MERT_FEATURES}")
        else:
            _TOTAL_FEATURES_ACTIVE = N_AUDIO_FEATURES
            print("MERT: no embeddings found, training without MERT")
    else:
        _TOTAL_FEATURES_ACTIVE = N_AUDIO_FEATURES
        print("MERT: disabled (--no-mert)")

    print(f"MeterNet v7-slim | classes={CLASS_METERS_6} | features={_TOTAL_FEATURES_ACTIVE}")
    mert_str = f", mert={_N_MERT_ACTIVE}" if _N_MERT_ACTIVE > 0 else ""
    print(f"  blocks: audio={_AUDIO_FEATURES_ACTIVE}{mert_str}")
    print(f"  data: {args.data_dir}"
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

    # Build entry pools based on mode
    use_kfold = args.kfold > 1
    pool_entries: list[Entry] = []
    wiki_pool: list[Entry] = []
    test_entries: list[Entry] = list(wiki_test)

    if use_kfold:
        wiki_pool = list(wiki_train) + list(wiki_val)
        if args.meter2800:
            m2800_train = load_meter2800_split(args.meter2800, "train", valid_meters)
            m2800_val = load_meter2800_split(args.meter2800, "val", valid_meters)
            m2800_test = load_meter2800_split(args.meter2800, "test", valid_meters)
            print(f"  METER2800: {len(m2800_train)} train, {len(m2800_val)} val, {len(m2800_test)} test")
            pool_entries = list(m2800_train) + list(m2800_val)
            test_entries.extend(m2800_test)
        else:
            print("  WARNING: --kfold without --meter2800 not supported")
            sys.exit(1)
        if args.limit:
            pool_entries = pool_entries[:args.limit]
            wiki_pool = wiki_pool[:args.limit]
            test_entries = test_entries[:args.limit]
        print(f"  K-fold mode: {args.kfold} folds, M2800 pool={len(pool_entries)}, "
              f"WIKI train={len(wiki_pool)} (always in train), test={len(test_entries)}")
    else:
        train_entries = list(wiki_train)
        val_entries = list(wiki_val)
        n_wiki_val = len(wiki_val)
        if args.meter2800:
            m2800_train = load_meter2800_split(args.meter2800, "train", valid_meters)
            m2800_val = load_meter2800_split(args.meter2800, "val", valid_meters)
            m2800_test = load_meter2800_split(args.meter2800, "test", valid_meters)
            print(f"  METER2800: {len(m2800_train)} train, {len(m2800_val)} val, {len(m2800_test)} test")
            train_entries.extend(wiki_val)
            val_entries = list(m2800_val)
            train_entries.extend(m2800_train)
            test_entries.extend(m2800_test)
            print(f"  Val strategy: METER2800 val only ({len(m2800_val)} files), "
                  f"WIKIMETER val ({n_wiki_val}) moved to training")
        if args.limit:
            train_entries = train_entries[:args.limit]
            val_entries = val_entries[:args.limit]
            test_entries = test_entries[:args.limit]
        print(f"  Total: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test")

    if args.verbose:
        if use_kfold:
            dist = Counter(m for _, m, _ in pool_entries)
            dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist.keys()))
            print(f"  M2800 pool: {dist_str}")
            dist2 = Counter(m for _, m, _ in wiki_pool)
            dist_str2 = "  ".join(f"{m}/x:{dist2[m]}" for m in sorted(dist2.keys()))
            print(f"  WIKI train: {dist_str2}")
        else:
            for name, entries in [("Train", train_entries), ("Val", val_entries), ("Test", test_entries)]:
                dist = Counter(m for _, m, _ in entries)
                dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist.keys()))
                print(f"  {name}: {dist_str}")

    # Initialize numpy LMDB (no analysis cache needed — just audio + MERT features)
    from beatmeter.analysis.cache import NumpyLMDB
    feat_db = NumpyLMDB("data/features.lmdb")

    # Extract features
    print("\nExtracting features (cache: data/features.lmdb)...")
    t0 = time.time()

    if use_kfold:
        X_pool, y_pool, pm_pool = extract_all_features(
            pool_entries, feat_db, "M2800 pool", args.workers, args.verbose,
        )
        X_wiki, y_wiki, pm_wiki = extract_all_features(
            wiki_pool, feat_db, "WIKI train", args.workers, args.verbose,
        )
    else:
        X_pool = None
        X_train, y_train, pm_train = extract_all_features(
            train_entries, feat_db, "train", args.workers, args.verbose,
        )
        X_val, y_val, _ = extract_all_features(
            val_entries, feat_db, "val", args.workers, args.verbose,
        )
    X_test, y_test, _ = extract_all_features(
        test_entries, feat_db, "test", args.workers, args.verbose,
    )

    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    if use_kfold:
        print(f"  Shapes: M2800 pool {X_pool.shape}, WIKI {X_wiki.shape}, test {X_test.shape}")
    else:
        print(f"  Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # Common training kwargs
    train_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        hidden=args.hidden,
        dropout_scale=args.dropout_scale,
        seed=args.seed,
        use_focal=args.focal,
        n_blocks=args.n_blocks,
        verbose=args.verbose,
        log_every=args.log_every,
        aug_mode=args.aug_mode,
    )

    # -----------------------------------------------------------------------
    # K-fold cross-validation
    # -----------------------------------------------------------------------
    if use_kfold:
        n_folds = args.kfold
        folds = _stratified_kfold(pm_pool, n_folds, seed=args.seed)
        print(f"\n{'='*60}")
        print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
        print(f"  M2800 pool: {len(pm_pool)} files (K-fold split)")
        print(f"  WIKI train: {len(X_wiki)} files (always in train)")
        print(f"{'='*60}")

        for k, (_, val_idx) in enumerate(folds):
            dist = Counter(pm_pool[i] for i in val_idx)
            dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist.keys()))
            print(f"  Fold {k+1} val: {len(val_idx)} M2800 files ({dist_str})")

        fold_val_accs: list[float] = []
        fold_test_accs: list[float] = []
        fold_test_correct: list[int] = []
        fold_epochs: list[int] = []

        for k, (train_idx, val_idx) in enumerate(folds):
            X_m2800_train_k = X_pool[train_idx]
            y_m2800_train_k = y_pool[train_idx]
            pm_m2800_train_k = [pm_pool[i] for i in train_idx]

            X_train_k = np.concatenate([X_m2800_train_k, X_wiki], axis=0)
            y_train_k = np.concatenate([y_m2800_train_k, y_wiki], axis=0)
            pm_train_k = pm_m2800_train_k + list(pm_wiki)

            X_val_k = X_pool[val_idx]
            y_val_k = y_pool[val_idx]

            X_train_k, X_val_k, X_test_k, _, _ = _standardize(
                X_train_k, X_val_k, X_test,
            )

            n_m2800_train = len(train_idx)
            n_wiki_train = len(X_wiki)
            print(
                f"\n--- Fold {k+1}/{n_folds} "
                f"(train={n_m2800_train}+{n_wiki_train}={len(X_train_k)}, "
                f"val={len(val_idx)} M2800) ---"
            )
            model_k, info_k = train_model(
                X_train_k, y_train_k, pm_train_k,
                X_val_k, y_val_k,
                **train_kwargs,
            )
            val_acc = info_k["best_val_acc"]
            fold_val_accs.append(val_acc)
            fold_epochs.append(info_k["best_epoch"])

            model_k.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_test_k, dtype=torch.float32).to(args.device)
                probs = torch.sigmoid(model_k(X_t))
                preds = probs.argmax(1).cpu().numpy()
                true_primary = y_test.argmax(axis=1)
                test_correct_k = int((preds == true_primary).sum())
                test_total_k = len(true_primary)
                test_acc_k = test_correct_k / max(test_total_k, 1)
            fold_test_accs.append(test_acc_k)
            fold_test_correct.append(test_correct_k)
            print(f"  FOLD_DONE fold={k+1}/{n_folds} val={val_acc:.4f} "
                  f"test={test_correct_k}/{test_total_k} ep={info_k['best_epoch']}")

        mean_val = float(np.mean(fold_val_accs))
        std_val = float(np.std(fold_val_accs))
        mean_test = float(np.mean(fold_test_accs))
        std_test = float(np.std(fold_test_accs))
        mean_epoch = int(np.mean(fold_epochs))
        print(f"\n{'='*60}")
        print(f"K-FOLD RESULTS")
        print(f"  Val balanced: {mean_val:.1%} ± {std_val:.1%}")
        print(f"  Test overall: {mean_test:.1%} ± {std_test:.1%}")
        print(f"{'='*60}")
        for k in range(len(fold_val_accs)):
            print(f"  Fold {k+1}: val={fold_val_accs[k]:.1%}  test={fold_test_correct[k]}/{len(y_test)} ({fold_test_accs[k]:.1%})  ep={fold_epochs[k]}")
        print(f"  Mean epoch: {mean_epoch}")

        # Retrain on ALL data
        print(f"\n--- RETRAIN on full pool (epochs={mean_epoch}) ---")
        X_full = np.concatenate([X_pool, X_wiki], axis=0)
        y_full = np.concatenate([y_pool, y_wiki], axis=0)
        pm_full = list(pm_pool) + list(pm_wiki)

        last_val_idx = folds[-1][1]
        X_retrain_val = X_pool[last_val_idx]
        y_retrain_val = y_pool[last_val_idx]

        X_full_s, X_retrain_val_s, X_test_s, feat_mean, feat_std = _standardize(
            X_full, X_retrain_val, X_test,
        )

        retrain_kwargs = dict(train_kwargs)
        retrain_kwargs["epochs"] = mean_epoch
        model, retrain_info = train_model(
            X_full_s, y_full, pm_full,
            X_retrain_val_s, y_retrain_val,
            **retrain_kwargs,
        )
        print(f"  Retrain done (epoch {retrain_info['best_epoch']}/{mean_epoch})")

        results = evaluate(model, X_test_s, y_test, args.device, verbose=args.verbose)

        info = {
            "best_val_acc": mean_val,
            "best_epoch": mean_epoch,
            "kfold_val_accs": fold_val_accs,
            "kfold_mean": mean_val,
            "kfold_std": std_val,
            "kfold_test_accs": fold_test_accs,
            "kfold_test_mean": mean_test,
            "kfold_test_std": std_test,
            "kfold_epochs": fold_epochs,
        }
        n_train_effective = len(X_full)
        n_val_effective = len(X_pool) // n_folds

    # -----------------------------------------------------------------------
    # Standard single-run training
    # -----------------------------------------------------------------------
    else:
        X_train, X_val, X_test, feat_mean, feat_std = _standardize(X_train, X_val, X_test)
        print("  Features standardized (z-score, fit on train)")

        print(
            f"\nTraining MeterNet (input={X_train.shape[1]}, hidden={args.hidden}, "
            f"blocks={args.n_blocks}, dropout_scale={args.dropout_scale}, "
            f"classes={len(CLASS_METERS_6)})..."
        )
        model, info = train_model(
            X_train, y_train, pm_train,
            X_val, y_val,
            **train_kwargs,
        )
        print(f"\nBest val balanced acc: {info['best_val_acc']:.1%} at epoch {info['best_epoch']}")

        results = evaluate(model, X_test, y_test, args.device, verbose=args.verbose)
        n_train_effective = len(X_train)
        n_val_effective = len(X_val)

    # -----------------------------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------------------------
    if args.save:
        save_path = args.save
    elif args.limit:
        save_path = Path("data/meter_net_test.pt")
        print(f"  (--limit active, saving to {save_path})")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = Path("data/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_path = ckpt_dir / f"{ts}.pt"

    from beatmeter.experiment import enrich_checkpoint, log_experiment, make_experiment_record

    ckpt = {
        "model_state": model.state_dict(),
        "class_meters": CLASS_METERS_6,
        "meter_to_idx": {m: i for i, m in enumerate(CLASS_METERS_6)},
        "input_dim": _TOTAL_FEATURES_ACTIVE,
        "n_classes": len(CLASS_METERS_6),
        "best_epoch": info["best_epoch"],
        "best_val_acc": info["best_val_acc"],
        "test_results": results,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "feature_version": FEATURE_VERSION,
        "n_audio_features": _AUDIO_FEATURES_ACTIVE,
        "hidden_dim": args.hidden,
        "dropout_scale": args.dropout_scale,
        "n_blocks": args.n_blocks,
    }
    # MERT metadata
    if _N_MERT_ACTIVE > 0:
        ckpt["mert_layer"] = _MERT_LAYER_ACTIVE
        ckpt["n_mert_features"] = _N_MERT_ACTIVE
    # K-fold metadata
    if use_kfold:
        ckpt["kfold"] = args.kfold
        ckpt["kfold_val_accs"] = info["kfold_val_accs"]
        ckpt["kfold_mean"] = info["kfold_mean"]
        ckpt["kfold_std"] = info["kfold_std"]
        ckpt["kfold_test_accs"] = info["kfold_test_accs"]
        ckpt["kfold_test_mean"] = info["kfold_test_mean"]
        ckpt["kfold_test_std"] = info["kfold_test_std"]
    ckpt = enrich_checkpoint(
        ckpt, args=args,
        train_size=n_train_effective, val_size=n_val_effective, test_size=len(X_test),
    )
    torch.save(ckpt, save_path)
    val_str = f"{info['best_val_acc']:.1%}"
    if use_kfold:
        val_str += f" ± {info['kfold_std']:.1%} ({args.kfold}-fold)"
    print(
        f"\nModel saved to {save_path}  "
        f"(val={val_str}, test={results['overall_acc']:.1%})"
    )
    if not args.save and not args.limit:
        print(f"To promote:  uv run python scripts/eval.py --promote {save_path}")
        _prune_staging(keep=5)

    log_experiment(make_experiment_record(
        type="train", model="meter_net",
        params={"hidden": args.hidden, "n_blocks": args.n_blocks, "lr": args.lr,
                "dropout_scale": args.dropout_scale,
                "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                "focal": args.focal, "no_mert": args.no_mert,
                "kfold": args.kfold if use_kfold else None,
                "mert_layer": _MERT_LAYER_ACTIVE if _N_MERT_ACTIVE > 0 else None},
        results={"best_val_acc": info["best_val_acc"], "best_epoch": info["best_epoch"],
                 "test_acc": results["overall_acc"], "test_correct": results["overall_correct"],
                 "test_total": results["overall_total"]},
        checkpoint=str(save_path),
        extra={"train_size": n_train_effective, "val_size": n_val_effective,
               "test_size": len(X_test),
               "feature_version": FEATURE_VERSION, "input_dim": _TOTAL_FEATURES_ACTIVE},
    ))


if __name__ == "__main__":
    main()
