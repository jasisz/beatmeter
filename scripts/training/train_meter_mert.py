#!/usr/bin/env python3
"""Train an MLP classifier on pre-extracted MERT embeddings for meter detection.

Uses embeddings from extract_mert_embeddings.py (shape (12, 1536) per file).
Performs a layer sweep to find the best MERT layer, then trains a small MLP.

Usage:
    uv run python scripts/train_meter_mert.py
    uv run python scripts/train_meter_mert.py --data-dir data/meter2800 --embeddings-dir data/mert_embeddings/meter2800
    uv run python scripts/train_meter_mert.py --epochs 100 --hidden-dim 512
"""

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_METERS = [3, 4, 5, 7]
METER_TO_IDX = {m: i for i, m in enumerate(CLASS_METERS)}
IDX_TO_METER = {i: m for i, m in enumerate(CLASS_METERS)}
NUM_LAYERS = 12
POOLED_DIM = 1536  # 768 mean + 768 max


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MERTEmbeddingDataset(Dataset):
    """Dataset of pre-extracted MERT embeddings."""

    def __init__(
        self,
        entries: list[tuple[Path, int]],
        layer_idx: int,
        augment: bool = False,
    ):
        self.layer_idx = layer_idx
        self.augment = augment
        self.features: list[torch.Tensor] = []
        self.labels: list[int] = []

        skipped = 0
        for emb_path, meter in entries:
            if not emb_path.exists():
                skipped += 1
                continue
            try:
                emb = np.load(emb_path)  # (12, 1536)
                vec = emb[layer_idx]     # (1536,)
                self.features.append(torch.from_numpy(vec).float())
                self.labels.append(METER_TO_IDX[meter])
            except Exception:
                skipped += 1
                continue

        if skipped > 0:
            print(f"  WARNING: skipped {skipped} files (missing or unreadable)")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self.features[idx].clone()
        y = self.labels[idx]
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MeterMLP(nn.Module):
    """Simple MLP classifier: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, input_dim: int = POOLED_DIM, hidden_dim: int = 256,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_entries(
    data_dir: Path,
    embeddings_dir: Path,
    split: str,
) -> list[tuple[Path, int]] | None:
    """Load entries for a specific split, mapping audio paths to embedding paths."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.utils import parse_label_file

    valid_meters = set(METER_TO_IDX.keys())

    for ext in (".tab", ".csv", ".tsv"):
        label_path = data_dir / f"data_{split}_4_classes{ext}"
        if label_path.exists():
            audio_entries = parse_label_file(label_path, data_dir, valid_meters=valid_meters)
            # Map audio_path -> embedding_path
            entries = []
            for audio_path, meter in audio_entries:
                emb_path = embeddings_dir / f"{audio_path.stem}.npy"
                entries.append((emb_path, meter))
            print(f"  Loaded {len(entries)} entries from {label_path.name}")
            return entries
    return None


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to num_classes."""
    counts = Counter(labels)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(1.0 / c)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w * num_classes / w.sum()
    return w


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
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
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
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
    return total_loss / max(total, 1), correct / max(total, 1), all_labels, all_preds


def print_confusion_matrix(labels: list[int], preds: list[int]) -> None:
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
# Layer sweep
# ---------------------------------------------------------------------------


def quick_layer_sweep(
    train_entries: list[tuple[Path, int]],
    val_entries: list[tuple[Path, int]],
    device: torch.device,
    hidden_dim: int = 256,
    epochs: int = 15,
    batch_size: int = 64,
) -> int:
    """Try each of the 12 MERT layers, return the one with best val accuracy."""
    print("\n=== Layer Sweep (quick 15-epoch training per layer) ===")
    best_layer = -1
    best_acc = 0.0

    for layer_idx in range(NUM_LAYERS):
        train_ds = MERTEmbeddingDataset(train_entries, layer_idx, augment=True)
        val_ds = MERTEmbeddingDataset(val_entries, layer_idx, augment=False)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"  Layer {layer_idx:2d}: no data, skipping")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = MeterMLP(input_dim=POOLED_DIM, hidden_dim=hidden_dim).to(device)
        class_weights = compute_class_weights(train_ds.labels, len(CLASS_METERS)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_acc_layer = 0.0
        for _ in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc_layer:
                best_val_acc_layer = val_acc

        print(f"  Layer {layer_idx:2d}: val_acc = {best_val_acc_layer:.1%}")
        if best_val_acc_layer > best_acc:
            best_acc = best_val_acc_layer
            best_layer = layer_idx

    print(f"\n  Best layer: {best_layer} (val_acc = {best_acc:.1%})")
    return best_layer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train MLP on MERT embeddings for meter classification")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/mert_embeddings/meter2800"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/meter_mert_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--layer", type=int, default=-1,
                        help="MERT layer index (0-11). -1 = auto-select via sweep")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-layer-sweep", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"Device: {device}")

    data_dir = args.data_dir.resolve()
    embeddings_dir = args.embeddings_dir.resolve()

    print(f"\nLoading entries from {data_dir}")
    print(f"Embeddings from {embeddings_dir}")

    train_entries = load_split_entries(data_dir, embeddings_dir, "train")
    val_entries = load_split_entries(data_dir, embeddings_dir, "val")
    test_entries = load_split_entries(data_dir, embeddings_dir, "test")

    if train_entries is None or val_entries is None or test_entries is None:
        print("ERROR: Could not find train/val/test split files")
        sys.exit(1)

    # Show distribution
    all_entries = train_entries + val_entries + test_entries
    meter_counts = Counter(m for _, m in all_entries)
    print(f"\nTotal entries: {len(all_entries)}")
    for m in CLASS_METERS:
        print(f"  Meter {m}: {meter_counts.get(m, 0)}")
    print(f"\nSplit: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test")

    # Layer selection
    if args.layer >= 0:
        best_layer = args.layer
        print(f"\nUsing specified layer: {best_layer}")
    elif args.skip_layer_sweep:
        best_layer = 11  # default to last layer
        print(f"\nSkipping layer sweep, using layer {best_layer}")
    else:
        best_layer = quick_layer_sweep(
            train_entries, val_entries, device,
            hidden_dim=args.hidden_dim, batch_size=args.batch_size,
        )

    # Build datasets with best layer
    print(f"\n=== Full Training with Layer {best_layer} ===")
    print("  Building datasets...")
    train_ds = MERTEmbeddingDataset(train_entries, best_layer, augment=True)
    val_ds = MERTEmbeddingDataset(val_entries, best_layer, augment=False)
    test_ds = MERTEmbeddingDataset(test_entries, best_layer, augment=False)
    print(f"  Sizes: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    if len(train_ds) == 0:
        print("ERROR: Training set is empty. Check embeddings.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = MeterMLP(
        input_dim=POOLED_DIM,
        hidden_dim=args.hidden_dim,
        num_classes=len(CLASS_METERS),
        dropout=args.dropout,
    ).to(device)
    print(f"\nMLP parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with class weights
    class_weights = compute_class_weights(train_ds.labels, len(CLASS_METERS)).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 15

    print(f"\n{'Epoch':>5s}  {'TrainLoss':>10s}  {'TrainAcc':>9s}  "
          f"{'ValLoss':>10s}  {'ValAcc':>9s}  {'LR':>10s}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  "
              f"{val_loss:10.4f}  {val_acc:8.1%}  {current_lr:10.6f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test evaluation
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

    # Save checkpoint
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "classifier_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
        "class_map": IDX_TO_METER,
        "layer_idx": best_layer,
        "input_dim": POOLED_DIM,
        "hidden_dim": args.hidden_dim,
        "num_classes": len(CLASS_METERS),
        "dropout": args.dropout,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "model_name": "m-a-p/MERT-v1-95M",
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    print(f"  Layer:         {best_layer}")
    print(f"  Val accuracy:  {best_val_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")


if __name__ == "__main__":
    main()
