#!/usr/bin/env python3
"""Train ResNet18 for meter (time signature) classification on METER2800.

Architecture:
    - torchvision ResNet18, random initialization
    - Input: MFCC spectrogram (13 coefficients) -> resize to (1, 224, 224)
    - Output: 4 classes (meter 3, 4, 5, 7)

Training:
    - Loss: CrossEntropyLoss with class weights (compensate 1200/1200/200/200 imbalance)
    - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
    - Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
    - Early stopping: patience=10 on val loss
    - Augmentation: time masking (0-20 frames), time shift (+/-10%), gaussian noise (0.01)
    - Epochs: 50, batch: 32
    - Device: MPS (Apple Silicon) with fallback to CPU

Usage:
    uv run python scripts/train_meter_resnet.py
    uv run python scripts/train_meter_resnet.py --data-dir data/meter2800
    uv run python scripts/train_meter_resnet.py --epochs 100 --batch-size 64
"""

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_METERS = [3, 4, 5, 7]  # ordered class labels
METER_TO_IDX = {m: i for i, m in enumerate(CLASS_METERS)}
IDX_TO_METER = {i: m for i, m in enumerate(CLASS_METERS)}

MFCC_PARAMS = {
    "n_mfcc": 13,
    "n_mels": 128,
    "hop_length": 512,
    "sr": 22050,
}

TARGET_DURATION = 30.0  # seconds — take center crop
IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _find_label_files(data_dir: Path, pattern: str = "*_4_classes*") -> list[Path]:
    """Find label CSV/TSV/TAB files in the dataset directory.

    By default looks for 4-class files (METER2800 convention).
    Falls back to any CSV/TAB if specific pattern not found.
    """
    exts = ("*.csv", "*.tab", "*.tsv")
    # First: try pattern-matched files
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(data_dir.glob(pattern + ext.lstrip("*"))))
    if files:
        return files
    # Fallback: any label files
    for ext in exts:
        files.extend(sorted(data_dir.glob(ext)))
    if not files:
        for ext in exts:
            files.extend(sorted(data_dir.rglob(ext)))
    return files


def _resolve_audio_path(raw_fname: str, data_dir: Path) -> Path | None:
    """Resolve a METER2800 .tab filename to an actual audio file on disk.

    The .tab files reference paths like "/MAG/00553.wav" but actual files
    are flattened into data_dir/audio/ as "00553.mp3" (or with collision
    prefix like "OWN_0001.mp3").
    """
    # Strip quotes
    raw_fname = raw_fname.strip('"').strip("'").strip()
    if not raw_fname:
        return None

    p = Path(raw_fname)
    src_dir = p.parent.name  # e.g. "MAG", "FMA", "GTZAN", "OWN"
    stem = p.stem            # e.g. "00553", "rock.00099"
    audio_dir = data_dir / "audio"

    # Try different extension and prefix combinations
    for ext in (".mp3", ".wav", ".ogg", ".flac", p.suffix):
        # Direct: audio/00553.mp3
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Collision prefix: audio/OWN_0001.mp3
        if src_dir:
            candidate = audio_dir / f"{src_dir}_{stem}{ext}"
            if candidate.exists():
                return candidate
            # Download artifact prefix: audio/OWN_._0001.mp3
            candidate = audio_dir / f"{src_dir}_._{stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def _parse_label_file(
    label_path: Path, data_dir: Path
) -> list[tuple[Path, int]]:
    """Parse a CSV/TSV label file from METER2800.

    Auto-detects delimiter (tab vs comma).
    Handles METER2800 .tab format: filename<TAB>label<TAB>meter<TAB>alt_meter

    Returns list of (audio_path, meter) tuples where audio_path is resolved.
    """
    import csv

    # Ordered: prefer specific names first (METER2800 has both "meter" and "label" columns)
    filename_cols = ["filename", "file", "audio_file", "audio_path", "audio", "path"]
    label_cols = ["meter", "time_signature", "ts", "time_sig", "signature", "label"]

    entries: list[tuple[Path, int]] = []
    missing = 0

    # Detect delimiter from first line
    with open(label_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "\t" if "\t" in first_line else ","

    with open(label_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            warnings.warn(f"Empty label file: {label_path}")
            return entries

        # Normalize header names to lowercase stripped (remove quotes)
        header_map = {
            h.strip().lower().strip('"').strip("'"): h
            for h in reader.fieldnames
        }

        # Find the filename column
        fname_key = None
        for candidate in filename_cols:
            if candidate in header_map:
                fname_key = header_map[candidate]
                break
        if fname_key is None:
            warnings.warn(
                f"No filename column found in {label_path}. "
                f"Headers: {list(reader.fieldnames)}"
            )
            return entries

        # Find the label column (prefer "meter" over "label" for METER2800)
        label_key = None
        for candidate in label_cols:
            if candidate in header_map:
                label_key = header_map[candidate]
                break
        if label_key is None:
            warnings.warn(
                f"No label column found in {label_path}. "
                f"Headers: {list(reader.fieldnames)}"
            )
            return entries

        for row in reader:
            raw_fname = row.get(fname_key, "").strip().strip('"').strip("'")
            raw_label = row.get(label_key, "").strip().strip('"').strip("'")
            if not raw_fname or not raw_label:
                continue

            # Parse meter — accept "3", "3/4", "4/4", etc.
            meter_str = raw_label.split("/")[0].strip()
            try:
                meter = int(meter_str)
            except ValueError:
                continue

            if meter not in METER_TO_IDX:
                continue

            # Resolve audio path
            audio_path = _resolve_audio_path(raw_fname, data_dir)
            if audio_path is None:
                missing += 1
                continue

            entries.append((audio_path, meter))

    if missing > 0:
        print(f"    ({missing} entries skipped — audio file not found)")

    return entries


def _load_split(
    data_dir: Path, split: str
) -> list[tuple[Path, int]] | None:
    """Load entries for a specific split (train/val/test) from label files.

    Looks for files matching *_{split}_4_classes.{csv,tab,tsv}.
    Returns None if no matching file found.
    """
    for ext in (".tab", ".csv", ".tsv"):
        pattern = f"data_{split}_4_classes{ext}"
        label_path = data_dir / pattern
        if label_path.exists():
            entries = _parse_label_file(label_path, data_dir)
            print(f"  Loaded {len(entries)} entries from {label_path.name}")
            return entries
    return None


def _load_entries(data_dir: Path) -> list[tuple[Path, int]]:
    """Load all (audio_path, meter) entries from label files in data_dir."""
    label_files = _find_label_files(data_dir)
    if not label_files:
        print(f"ERROR: No label files found in {data_dir}")
        sys.exit(1)

    all_entries: list[tuple[Path, int]] = []
    for label_path in label_files:
        entries = _parse_label_file(label_path, data_dir)
        print(f"  Loaded {len(entries)} entries from {label_path.name}")
        all_entries.extend(entries)

    if not all_entries:
        print(f"ERROR: No valid entries found in label files under {data_dir}")
        sys.exit(1)

    return all_entries


def _extract_mfcc(audio_path: Path) -> np.ndarray | None:
    """Extract MFCC features from an audio file.

    Returns numpy array of shape (13, T) or None on failure.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=MFCC_PARAMS["sr"], mono=True)
    except Exception as e:
        warnings.warn(f"Failed to load {audio_path}: {e}")
        return None

    if y is None or len(y) == 0:
        warnings.warn(f"Empty audio: {audio_path}")
        return None

    # Take center crop of TARGET_DURATION seconds
    target_samples = int(TARGET_DURATION * sr)
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=MFCC_PARAMS["n_mfcc"],
        n_mels=MFCC_PARAMS["n_mels"],
        hop_length=MFCC_PARAMS["hop_length"],
    )

    if mfcc is None or mfcc.size == 0:
        warnings.warn(f"Empty MFCC for {audio_path}")
        return None

    return mfcc


def _mfcc_to_tensor(mfcc: np.ndarray) -> torch.Tensor:
    """Convert MFCC (13, T) to a normalized (1, 224, 224) tensor."""
    t = torch.from_numpy(mfcc).float().unsqueeze(0)  # (1, 13, T)
    # Resize to (1, 224, 224) using bilinear interpolation
    t = F.interpolate(t.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (1, 224, 224)
    # Normalize: zero mean, unit variance per sample
    mean = t.mean()
    std = t.std()
    if std > 1e-6:
        t = (t - mean) / std
    else:
        t = t - mean
    return t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MeterDataset(Dataset):
    """METER2800 dataset with MFCC features cached in memory."""

    def __init__(
        self,
        entries: list[tuple[Path, int]],
        augment: bool = False,
    ):
        self.augment = augment
        self.features: list[torch.Tensor] = []
        self.labels: list[int] = []

        skipped = 0
        for audio_path, meter in entries:
            if not audio_path.exists():
                warnings.warn(f"Missing file, skipping: {audio_path}")
                skipped += 1
                continue

            mfcc = _extract_mfcc(audio_path)
            if mfcc is None:
                skipped += 1
                continue

            tensor = _mfcc_to_tensor(mfcc)
            self.features.append(tensor)
            self.labels.append(METER_TO_IDX[meter])

        if skipped > 0:
            print(f"  WARNING: skipped {skipped} files (missing or unreadable)")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        """Apply training augmentations to a (1, 224, 224) tensor."""
        # Time masking: zero out 0-20 consecutive time-frames (columns)
        mask_width = torch.randint(0, 21, (1,)).item()
        if mask_width > 0:
            max_start = x.shape[-1] - mask_width
            if max_start > 0:
                start = torch.randint(0, max_start, (1,)).item()
                x[:, :, start : start + mask_width] = 0.0

        # Time shift: circular shift by +/-10%
        max_shift = int(0.1 * x.shape[-1])
        if max_shift > 0:
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        # Gaussian noise
        x = x + 0.01 * torch.randn_like(x)

        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(num_classes: int = 4) -> nn.Module:
    """Build ResNet18 adapted for single-channel MFCC input."""
    model = torchvision.models.resnet18(weights=None)
    # Replace first conv to accept 1-channel input instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace final FC for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights, normalized to sum to num_classes."""
    counts = Counter(labels)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(1.0 / c)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w * num_classes / w.sum()  # normalize so weights sum to num_classes
    return w


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model. Returns (avg_loss, accuracy, all_labels, all_preds)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

        all_labels.extend(batch_y.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_labels, all_preds


def print_confusion_matrix(labels: list[int], preds: list[int]) -> None:
    """Print a readable confusion matrix with per-class accuracy."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_METERS))))
    meter_names = [f"{m}/x" for m in CLASS_METERS]

    print("\nConfusion Matrix:")
    header = "          " + "  ".join(f"{n:>6s}" for n in meter_names)
    print(header)
    print("          " + "-" * (8 * len(meter_names)))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6d}" for v in row)
        row_total = row.sum()
        row_acc = row[i] / row_total * 100 if row_total > 0 else 0.0
        print(f"  {meter_names[i]:>6s} | {row_str}   ({row_acc:5.1f}%)")

    print("\nPer-class accuracy:")
    for i, m in enumerate(CLASS_METERS):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        acc = class_correct / class_total * 100 if class_total > 0 else 0.0
        print(f"  {m}/x: {class_correct}/{class_total} = {acc:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def select_device(requested: str) -> torch.device:
    """Select compute device with automatic fallback."""
    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet18 for meter classification on METER2800"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/meter2800"),
        help="Path to METER2800 dataset directory (default: data/meter2800)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/meter_resnet18.pt"),
        help="Path to save model checkpoint (default: data/meter_resnet18.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, mps, cuda, cpu (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ---- Seed ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Device ----
    device = select_device(args.device)
    print(f"Device: {device}")

    # ---- Load entries ----
    data_dir = args.data_dir.resolve()
    print(f"\nLoading dataset from {data_dir}")

    # Try pre-defined splits first (METER2800 convention)
    train_entries = _load_split(data_dir, "train")
    val_entries = _load_split(data_dir, "val")
    test_entries = _load_split(data_dir, "test")

    if train_entries is not None and val_entries is not None and test_entries is not None:
        print(f"\n  Using pre-defined splits from dataset")
    else:
        # Fallback: load all entries and split manually
        print(f"\n  No pre-defined splits found, loading all entries and splitting...")
        entries = _load_entries(data_dir)
        labels_for_split = [METER_TO_IDX[m] for _, m in entries]
        indices = list(range(len(entries)))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=args.seed,
            stratify=labels_for_split,
        )
        temp_labels = [labels_for_split[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=args.seed,
            stratify=temp_labels,
        )

        train_entries = [entries[i] for i in train_idx]
        val_entries = [entries[i] for i in val_idx]
        test_entries = [entries[i] for i in test_idx]

    # Show class distribution
    all_entries = train_entries + val_entries + test_entries
    meter_counts = Counter(m for _, m in all_entries)
    print(f"\nTotal entries: {len(all_entries)}")
    for m in CLASS_METERS:
        print(f"  Meter {m}: {meter_counts.get(m, 0)}")

    print(f"\nSplit: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test")

    # ---- Build datasets ----
    print("\nExtracting MFCC features (this may take a while)...")
    print("  Training set:")
    train_ds = MeterDataset(train_entries, augment=True)
    print("  Validation set:")
    val_ds = MeterDataset(val_entries, augment=False)
    print("  Test set:")
    test_ds = MeterDataset(test_entries, augment=False)

    print(f"\nDataset sizes: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    if len(train_ds) == 0:
        print("ERROR: Training set is empty after loading. Check audio files.")
        sys.exit(1)

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type != "cpu"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type != "cpu"),
    )

    # ---- Model ----
    model = build_model(num_classes=len(CLASS_METERS))
    model = model.to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss with class weights ----
    class_weights = compute_class_weights(train_ds.labels, len(CLASS_METERS))
    print(f"Class weights: {class_weights.tolist()}")
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ---- Training loop ----
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10

    print(f"\n{'Epoch':>5s}  {'TrainLoss':>10s}  {'TrainAcc':>9s}  "
          f"{'ValLoss':>10s}  {'ValAcc':>9s}  {'LR':>10s}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  "
            f"{val_loss:10.4f}  {val_acc:8.1%}  {current_lr:10.6f}"
        )

        scheduler.step(val_loss)

        # Track best by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Early stopping on val loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no val loss improvement for {early_stop_patience} epochs)")
                break

    # ---- Evaluate on test set ----
    print("\n" + "=" * 65)
    print("Test set evaluation (best model by val accuracy)")
    print("=" * 65)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.1%} ({sum(1 for a, b in zip(test_labels, test_preds) if a == b)}/{len(test_labels)})")
    print(f"Best val accuracy: {best_val_acc:.1%}")

    print_confusion_matrix(test_labels, test_preds)

    # ---- Save checkpoint ----
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
        "class_map": IDX_TO_METER,
        "mfcc_params": MFCC_PARAMS,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    print(f"  Val accuracy:  {best_val_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")


if __name__ == "__main__":
    main()
