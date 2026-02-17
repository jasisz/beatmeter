#!/usr/bin/env python3
"""Train MLP on multi-tempo, multi-signal autocorrelation features for meter classification.

v4: Multi-tempo autocorrelation (768) + tempogram profile (64) + MFCC stats (26)
    + spectral contrast (14) + onset rate stats (4) = 876-dim features.
+ focal loss, larger model (512→256→128), z-score, mixup, label smoothing.

The idea: autocorrelation features capture beat-relative periodicity but depend on
tempo estimation. Tempogram/MFCC/spectral features provide tempo-independent context.
Focal loss focuses training on hard examples (e.g. ambiguous 4/x).

Feature vector: 768 (autocorr) + 64 (tempogram) + 26 (MFCC) + 14 (contrast) + 4 (onset) = 876 dims.

Usage:
    # METER2800 only (4 classes: 3,4,5,7)
    uv run python scripts/training/train_onset_mlp.py --data-dir data/meter2800

    # METER2800 + WIKIMETER (6 classes: 3,4,5,7,9,11)
    uv run python scripts/training/train_onset_mlp.py --data-dir data/meter2800 --extra-data data/wikimeter

    # Quick test
    uv run python scripts/training/train_onset_mlp.py --data-dir data/meter2800 --limit 20
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.utils import resolve_audio_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR = 22050
HOP_LENGTH = 512
MAX_DURATION_S = 30
N_BEAT_FEATURES = 64       # autocorrelation sampled at 64 beat-relative lags
BEAT_RANGE = (0.5, 16.0)   # from 0.5 to 16 beats
WINDOW_PCT = 0.05           # ±5% window around expected lag
N_TEMPO_CANDIDATES = 3     # primary, half, double
N_SIGNALS = 4              # onset + RMS + spectral flux + chroma
N_AUTOCORR_FEATURES = N_BEAT_FEATURES * N_TEMPO_CANDIDATES * N_SIGNALS  # 768

# Tempo-independent features (v4)
N_TEMPOGRAM_BINS = 64      # tempogram profile sampled at log-spaced BPMs
N_MFCC = 13               # MFCC coefficients → mean + std = 26 dims
N_CONTRAST_BANDS = 6      # spectral contrast bands (librosa returns n_bands+1 rows incl. valley)
N_CONTRAST_DIMS = (N_CONTRAST_BANDS + 1) * 2  # 7 rows × (mean + std) = 14 dims
N_ONSET_STATS = 4          # onset rate: mean/std/median interval + count
N_EXTRA_FEATURES = N_TEMPOGRAM_BINS + N_MFCC * 2 + N_CONTRAST_DIMS + N_ONSET_STATS  # 108
TOTAL_FEATURES = N_AUTOCORR_FEATURES + N_EXTRA_FEATURES  # 876

FEATURE_VERSION = "v4"     # bump when feature extraction changes (invalidates cache)
LABEL_SMOOTHING = 0.1      # soften hard targets
MIXUP_ALPHA = 0.2          # mixup interpolation parameter
FOCAL_GAMMA = 2.0          # focal loss focusing parameter

CLASS_METERS_4 = [3, 4, 5, 7]
CLASS_METERS_6 = [3, 4, 5, 7, 9, 11]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation of a 1-D signal."""
    if len(signal) < 10:
        return np.zeros(1)
    signal = signal - signal.mean()
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return np.zeros(len(signal))
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    return autocorr


def _onset_autocorrelation(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute normalized autocorrelation of onset strength envelope."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    return _normalized_autocorrelation(onset_env)


def _rms_autocorrelation(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute normalized autocorrelation of RMS energy envelope."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    return _normalized_autocorrelation(rms)


def _spectral_flux_autocorrelation(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute normalized autocorrelation of spectral flux (timbral change rate)."""
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    return _normalized_autocorrelation(flux)


def _chroma_autocorrelation(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute normalized autocorrelation of chroma energy (harmonic periodicity)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    # Sum across pitch classes → single energy curve per frame
    chroma_energy = np.sum(chroma, axis=0)
    return _normalized_autocorrelation(chroma_energy)


def _estimate_tempos(y: np.ndarray, sr: int = SR) -> list[float]:
    """Estimate tempo and return 3 candidates: T, T/2, T×2."""
    tempo = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)
    t = float(tempo[0]) if len(tempo) > 0 else 120.0
    if t < 30 or t > 300:
        t = 120.0
    return [t, max(30.0, t / 2), min(300.0, t * 2)]


def _sample_autocorr_at_tempo(
    autocorr: np.ndarray, tempo_bpm: float, sr: int = SR
) -> np.ndarray:
    """Sample autocorrelation at beat-relative lags for a given tempo."""
    beat_period_frames = (60.0 / tempo_bpm) * (sr / HOP_LENGTH)
    beat_multiples = np.linspace(BEAT_RANGE[0], BEAT_RANGE[1], N_BEAT_FEATURES)
    features = np.zeros(N_BEAT_FEATURES)

    for i, k in enumerate(beat_multiples):
        lag = int(k * beat_period_frames)
        if 0 < lag < len(autocorr):
            window = max(1, int(lag * WINDOW_PCT))
            start = max(0, lag - window)
            end = min(len(autocorr), lag + window + 1)
            features[i] = float(np.max(autocorr[start:end]))

    return features


def _tempogram_profile(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute averaged tempogram profile at log-spaced BPMs (tempo-independent).

    Returns N_TEMPOGRAM_BINS values representing rhythmic energy at different tempi.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    # Compute tempogram (autocorrelation-based)
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    # Average across time → 1D profile
    avg_tg = tg.mean(axis=1)
    # Sample at log-spaced BPM values (30-300 BPM)
    bpm_bins = np.logspace(np.log10(30), np.log10(300), N_TEMPOGRAM_BINS)
    # Convert BPM to lag in frames, then index into avg_tg
    profile = np.zeros(N_TEMPOGRAM_BINS)
    for i, bpm in enumerate(bpm_bins):
        lag = int((60.0 / bpm) * (sr / HOP_LENGTH))
        if 0 < lag < len(avg_tg):
            # Take max in small window around target lag
            w = max(1, lag // 20)
            start = max(0, lag - w)
            end = min(len(avg_tg), lag + w + 1)
            profile[i] = float(np.max(avg_tg[start:end]))
    # Normalize
    pmax = profile.max()
    if pmax > 1e-10:
        profile /= pmax
    return profile


def _mfcc_statistics(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute mean and std of MFCC coefficients (tempo-independent timbral context).

    Returns 2 × N_MFCC = 26 values.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def _spectral_contrast_statistics(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute mean and std of spectral contrast (tempo-independent timbral texture).

    Returns 2 × N_CONTRAST_BANDS = 14 values.
    """
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_bands=N_CONTRAST_BANDS,
    )
    return np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])


def _onset_rate_statistics(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute onset timing statistics (tempo-independent rhythmic density).

    Returns 4 values: mean interval, std interval, median interval, onset rate.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(onset_frames) < 3:
        return np.zeros(N_ONSET_STATS)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    intervals = np.diff(onset_times)
    if len(intervals) == 0:
        return np.zeros(N_ONSET_STATS)
    duration = len(y) / sr
    return np.array([
        float(np.mean(intervals)),
        float(np.std(intervals)),
        float(np.median(intervals)),
        float(len(onset_frames) / max(duration, 1.0)),  # onsets per second
    ])


def extract_features(audio_path: Path) -> np.ndarray | None:
    """Extract v4 features: autocorrelation + tempogram + MFCC + contrast + onset stats.

    Returns 876-dim vector: 768 (autocorr) + 64 (tempogram) + 26 (MFCC) + 14 (contrast) + 4 (onset),
    or None on failure.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SR, duration=MAX_DURATION_S, mono=True)
    except Exception:
        return None

    if len(y) < sr * 2:  # minimum 2 seconds
        return None

    # --- Part 1: autocorrelation features (768 dims, tempo-dependent) ---
    autocorrs = [
        _onset_autocorrelation(y, sr),
        _rms_autocorrelation(y, sr),
        _spectral_flux_autocorrelation(y, sr),
        _chroma_autocorrelation(y, sr),
    ]
    if any(len(ac) < 10 for ac in autocorrs):
        return None

    tempos = _estimate_tempos(y, sr)

    parts = []
    for t in tempos:
        for ac in autocorrs:
            parts.append(_sample_autocorr_at_tempo(ac, t, sr))

    # --- Part 2: tempo-independent features (108 dims) ---
    parts.append(_tempogram_profile(y, sr))           # 64 dims
    parts.append(_mfcc_statistics(y, sr))              # 26 dims
    parts.append(_spectral_contrast_statistics(y, sr)) # 14 dims
    parts.append(_onset_rate_statistics(y, sr))        # 4 dims

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_meter2800_split(
    data_dir: Path, split: str, valid_meters: set[int]
) -> list[tuple[Path, int]]:
    """Load METER2800 entries for a given split."""
    label_path = data_dir / f"data_{split}_4_classes.tab"
    if not label_path.exists():
        return []

    entries = []
    with open(label_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fname = row["filename"].strip('"')
            meter = int(row["meter"])
            if meter not in valid_meters:
                continue
            audio_path = resolve_audio_path(fname, data_dir)
            if audio_path:
                entries.append((audio_path, meter))
    return entries


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
            # Parse primary meter: "3:0.7,4:0.8" → take highest weight
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

            # Resolve audio path
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


def _cache_key(audio_path: Path) -> str:
    st = audio_path.stat()
    raw = f"{audio_path.resolve()}::{st.st_size}::{st.st_mtime_ns}::{FEATURE_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()


def extract_with_cache(
    entries: list[tuple[Path, int]],
    cache_dir: Path,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for all entries, with disk caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_list = []
    labels_list = []
    skipped = 0

    desc = f"Extracting {label}" if label else "Extracting"
    for audio_path, meter in _progress(entries, desc):
        key = _cache_key(audio_path)
        cache_path = cache_dir / f"{key}.npy"

        if cache_path.exists():
            feat = np.load(cache_path)
        else:
            feat = extract_features(audio_path)
            if feat is not None:
                np.save(cache_path, feat)

        if feat is None:
            skipped += 1
            continue

        features_list.append(feat)
        labels_list.append(meter)

    if skipped:
        print(f"  ({skipped} files skipped — extraction failed)")

    X = np.stack(features_list).astype(np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def _progress(iterable, desc=""):
    """tqdm wrapper."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class OnsetMLP(nn.Module):
    """MLP for meter classification from multi-tempo autocorrelation features."""

    def __init__(self, input_dim: int, n_classes: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _focal_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """Focal loss with soft targets (for mixup) and class weights.

    Focuses training on hard examples by down-weighting well-classified ones.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    # Focal modulation: (1 - p_t)^gamma
    focal_weight = (1.0 - (probs * targets_soft).sum(dim=1)) ** gamma
    # Soft cross-entropy
    ce = -(targets_soft * log_probs).sum(dim=1)
    # Apply class weights via target distribution
    sample_weight = (targets_soft * class_weights).sum(dim=1)
    return (focal_weight * ce * sample_weight).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization fitted on train set."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meter_to_idx: dict[int, int],
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[OnsetMLP, dict]:
    """Train the OnsetMLP model."""
    n_classes = len(meter_to_idx)
    input_dim = X_train.shape[1]

    # Map meter labels to class indices
    y_train_idx = np.array([meter_to_idx[m] for m in y_train])
    y_val_idx = np.array([meter_to_idx[m] for m in y_val])

    # Class weights for imbalanced data
    counts = Counter(y_train_idx)
    total = len(y_train_idx)
    class_weights = torch.tensor(
        [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(device)

    # Datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_idx, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_idx, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = OnsetMLP(input_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6
    )
    # Focal loss replaces CrossEntropyLoss — focuses on hard examples
    # criterion is not used directly; _focal_loss is called in training loop

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    patience_limit = 40

    idx_to_meter = {v: k for k, v in meter_to_idx.items()}

    for epoch in range(1, epochs + 1):
        # Train
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
            # Mixup augmentation
            xb_mixed, yb_mixed = _mixup_batch(xb, yb, n_classes)
            optimizer.zero_grad()
            logits = model(xb_mixed)
            loss = _focal_loss(logits, yb_mixed, class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            # Accuracy on original (unmixed) labels for logging
            with torch.no_grad():
                preds = model(xb).argmax(1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

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
        scheduler.step(1 - val_acc)

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
    model: OnsetMLP,
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
    parser = argparse.ArgumentParser(description="Train onset autocorrelation MLP")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--extra-data", type=Path, default=None,
                        help="WIKIMETER data dir for 6-class training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit entries per split (0=all)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save model to this path")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Determine class set
    use_6_classes = args.extra_data is not None
    class_meters = CLASS_METERS_6 if use_6_classes else CLASS_METERS_4
    meter_to_idx = {m: i for i, m in enumerate(class_meters)}
    valid_meters = set(class_meters)

    print(f"Classes: {class_meters}")
    print(f"Data: {args.data_dir}" + (f" + {args.extra_data}" if args.extra_data else ""))

    # Load entries
    print("\nLoading data...")
    train_entries = load_meter2800_split(args.data_dir, "train", valid_meters)
    val_entries = load_meter2800_split(args.data_dir, "val", valid_meters)
    test_entries = load_meter2800_split(args.data_dir, "test", valid_meters)

    if args.extra_data:
        wiki_entries = load_wikimeter(args.extra_data, valid_meters)
        from scripts.utils import split_by_stem
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
        train_entries.extend(wiki_train)
        val_entries.extend(wiki_val)
        test_entries.extend(wiki_test)

    if args.limit:
        train_entries = train_entries[:args.limit]
        val_entries = val_entries[:args.limit]
        test_entries = test_entries[:args.limit]

    print(f"  Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")

    # Distribution
    for name, entries in [("Train", train_entries), ("Val", val_entries), ("Test", test_entries)]:
        dist = Counter(m for _, m in entries)
        dist_str = "  ".join(f"{m}/x:{dist[m]}" for m in sorted(dist.keys()))
        print(f"  {name}: {dist_str}")

    # Extract features
    cache_dir = Path(f"data/onset_features_cache_{FEATURE_VERSION}")
    print("\nExtracting features...")
    t0 = time.time()
    X_train, y_train = extract_with_cache(train_entries, cache_dir, "train")
    X_val, y_val = extract_with_cache(val_entries, cache_dir, "val")
    X_test, y_test = extract_with_cache(test_entries, cache_dir, "test")
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(f"  Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # Z-score standardization (fit on train)
    X_train, X_val, X_test, feat_mean, feat_std = _standardize(X_train, X_val, X_test)
    print("  Features standardized (z-score, fit on train)")

    # Train
    print(f"\nTraining MLP (input={X_train.shape[1]}, classes={len(class_meters)})...")
    model, info = train_model(
        X_train, y_train, X_val, y_val,
        meter_to_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(f"\nBest val acc: {info['best_val_acc']:.1%} at epoch {info['best_epoch']}")

    # Test
    results = evaluate(model, X_test, y_test, meter_to_idx, args.device)

    # Save
    save_path = args.save or Path("data/meter_onset_mlp.pt")
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
        },
        save_path,
    )
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
