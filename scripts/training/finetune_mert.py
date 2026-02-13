#!/usr/bin/env python3
"""End-to-end LoRA fine-tuning of MERT for meter classification.

Uses MERT-v1-330M (or 95M) with LoRA adapters on attention Q/V matrices,
plus a classification head. Audio is loaded, resampled to 24 kHz,
center-cropped to 30s, and split into 5s chunks.

Gradients flow through MERT (via LoRA) and the classification head together.

Requires: pip install peft

Usage:
    uv run python scripts/training/finetune_mert.py --data-dir data/meter2800
    uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 --model m-a-p/MERT-v1-95M
    uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 --epochs 30 --lr 1e-4
    uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 --no-lora  # frozen baseline
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_METERS = [3, 4, 5, 7, 9, 11]
METER_TO_IDX = {m: i for i, m in enumerate(CLASS_METERS)}
IDX_TO_METER = {i: m for i, m in enumerate(CLASS_METERS)}

MERT_SR = 24000
CHUNK_SAMPLES = 5 * MERT_SR   # 5-second chunks
MAX_DURATION_S = 30
LABEL_SMOOTH_NEG = 0.1  # label smoothing: negative classes get 0.1 instead of 0.0

MODEL_CONFIGS = {
    "m-a-p/MERT-v1-95M":  (12, 768),
    "m-a-p/MERT-v1-330M": (24, 1024),
}


# ---------------------------------------------------------------------------
# Dataset: loads raw audio, resamples, crops
# ---------------------------------------------------------------------------


class MERTAudioDataset(Dataset):
    """Dataset that loads audio files and returns resampled+cropped waveforms.

    Supports multi-label class lists: each entry has meters like [3] or [3, 4].
    Returns a smoothed multi-hot label vector for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        entries: list[tuple[Path, list[int]]],
        augment: bool = False,
        noise_std: float = 0.01,
    ):
        self.entries: list[tuple[Path, list[int]]] = []
        self.augment = augment
        self.noise_std = noise_std

        skipped = 0
        for audio_path, meters in entries:
            if not audio_path.exists():
                skipped += 1
                continue
            valid = [m for m in meters if m in METER_TO_IDX]
            if valid:
                self.entries.append((audio_path, valid))

        if skipped > 0:
            print(f"  WARNING: skipped {skipped} files (not found)")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        path, meters = self.entries[idx]
        label = np.full(len(CLASS_METERS), LABEL_SMOOTH_NEG, dtype=np.float32)
        for m in meters:
            label[METER_TO_IDX[m]] = 1.0

        try:
            audio, _ = librosa.load(str(path), sr=MERT_SR, mono=True)
        except Exception:
            audio = np.zeros(MERT_SR, dtype=np.float32)

        # Crop to MAX_DURATION_S
        max_samples = MAX_DURATION_S * MERT_SR
        if len(audio) > max_samples:
            if self.augment:
                # Stochastic phase augmentation: random crop position each epoch
                # Model sees different phase alignments, preventing phase bias
                start = np.random.randint(0, len(audio) - max_samples)
            else:
                # Deterministic center crop for validation/test
                start = (len(audio) - max_samples) // 2
            audio = audio[start : start + max_samples]

        # Pad if too short
        if len(audio) < MERT_SR:
            audio = np.pad(audio, (0, MERT_SR - len(audio)))

        # Data augmentation
        if self.augment:
            if self.noise_std > 0:
                audio = audio + self.noise_std * np.random.randn(len(audio)).astype(np.float32)
            audio = np.roll(audio, np.random.randint(-MERT_SR // 2, MERT_SR // 2))

        return audio.astype(np.float32), label


def simple_collate(batch):
    """Simple collate: returns list of numpy arrays + multi-hot label tensor."""
    audios, labels = zip(*batch)
    return list(audios), torch.tensor(np.stack(labels), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------


class MERTClassificationHead(nn.Module):
    """Weighted-sum of all layers + MLP classifier.

    Accepts pre-pooled (batch, num_layers, pooled_dim) tensors.
    """

    def __init__(self, num_layers: int, pooled_dim: int, num_classes: int = 6,
                 head_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        """stacked: (batch, num_layers, pooled_dim) -> (batch, num_classes)"""
        weights = torch.softmax(self.layer_logits, dim=0)
        w = weights.view(1, self.num_layers, 1)
        fused = (stacked * w).sum(dim=1)  # (batch, pooled_dim)
        return self.head(fused)


# ---------------------------------------------------------------------------
# MERT forward pass with pooling (gradient-aware)
# ---------------------------------------------------------------------------


def mert_forward_pool(
    audios: list[np.ndarray],
    mert_model: nn.Module,
    processor,
    device: torch.device,
    num_layers: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Run MERT on a batch of audio arrays, return pooled (batch, num_layers, pooled_dim).

    Each audio is split into 5s chunks, processed through MERT, then
    mean+max pooled per layer. Gradients flow through for LoRA training.
    """
    pooled_dim = hidden_dim * 2
    batch_pooled = []

    for audio_np in audios:
        # Split into 5s chunks
        chunks = []
        for start in range(0, len(audio_np), CHUNK_SAMPLES):
            chunk = audio_np[start : start + CHUNK_SAMPLES]
            if len(chunk) < MERT_SR:
                continue
            chunks.append(chunk)
        if not chunks:
            chunks = [audio_np]

        # Process each chunk — accumulate pooled vectors with gradient
        chunk_layer_means = [[] for _ in range(num_layers)]
        chunk_layer_maxes = [[] for _ in range(num_layers)]

        for chunk in chunks:
            inputs = processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = mert_model(**inputs)

            # hidden_states[0] = conv features, [1..num_layers] = transformer layers
            for li in range(num_layers):
                hs = outputs.hidden_states[li + 1].squeeze(0)  # (T, hidden_dim)
                chunk_layer_means[li].append(hs.mean(dim=0))
                chunk_layer_maxes[li].append(hs.max(dim=0).values)

        # Aggregate across chunks: mean of means, max of maxes
        layer_pooled = []
        for li in range(num_layers):
            mean_agg = torch.stack(chunk_layer_means[li]).mean(dim=0)  # (hidden_dim,)
            max_agg = torch.stack(chunk_layer_maxes[li]).max(dim=0).values  # (hidden_dim,)
            layer_pooled.append(torch.cat([mean_agg, max_agg]))  # (pooled_dim,)

        # (num_layers, pooled_dim)
        batch_pooled.append(torch.stack(layer_pooled))

    # (batch, num_layers, pooled_dim)
    return torch.stack(batch_pooled)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_entries(data_dir: Path, split: str) -> list[tuple[Path, list[int]]] | None:
    """Load entries for a specific split. Returns (path, [meter]) tuples."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.utils import parse_label_file

    valid_meters = set(METER_TO_IDX.keys())
    for ext in (".tab", ".csv", ".tsv"):
        label_path = data_dir / f"data_{split}_4_classes{ext}"
        if label_path.exists():
            raw = parse_label_file(label_path, data_dir, valid_meters=valid_meters)
            entries = [(p, [m]) for p, m in raw]
            print(f"  Loaded {len(entries)} entries from {label_path.name}")
            return entries
    return None


def compute_pos_weights(entries: list[tuple[Path, list[int]]], num_classes: int) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss: ratio of negatives/positives per class."""
    pos_counts = np.zeros(num_classes, dtype=np.float32)
    for _, meters in entries:
        for m in meters:
            if m in METER_TO_IDX:
                pos_counts[METER_TO_IDX[m]] += 1
    total = len(entries)
    neg_counts = total - pos_counts
    # pos_weight = neg/pos (higher weight for rare classes); 1.0 for unseen classes
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    mert_model: nn.Module,
    head: nn.Module,
    processor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_layers: int,
    hidden_dim: int,
    grad_accum_steps: int = 1,
    use_lora: bool = True,
) -> tuple[float, float]:
    head.train()
    if use_lora:
        mert_model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train", leave=False,
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    for step, (audios, labels) in enumerate(pbar):
        labels = labels.to(device)

        # Forward through MERT (with gradients for LoRA)
        if use_lora:
            pooled = mert_forward_pool(audios, mert_model, processor, device,
                                       num_layers, hidden_dim)
        else:
            with torch.no_grad():
                pooled = mert_forward_pool(audios, mert_model, processor, device,
                                           num_layers, hidden_dim)
            pooled = pooled.detach()

        # Forward through head
        logits = head(pooled)
        loss = criterion(logits, labels) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or step == len(loader) - 1:
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + [p for p in mert_model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * len(audios)
        # Primary accuracy: argmax prediction matches argmax of label
        preds = logits.argmax(dim=1)
        true_primary = labels.argmax(dim=1)
        correct += (preds == true_primary).sum().item()
        total += len(audios)

        pbar.set_postfix_str(f"loss={total_loss/total:.3f} acc={correct/total:.0%}")

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    mert_model: nn.Module,
    head: nn.Module,
    processor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_layers: int,
    hidden_dim: int,
) -> tuple[float, float, list[int], list[int], np.ndarray, np.ndarray]:
    """Evaluate model. Returns (loss, acc, primary_labels, primary_preds, all_probs, all_labels_multihot)."""
    head.eval()
    mert_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels_idx: list[int] = []   # primary label index (argmax)
    all_preds_idx: list[int] = []    # argmax prediction index
    all_probs: list[np.ndarray] = []       # sigmoid probabilities
    all_labels_raw: list[np.ndarray] = []  # multi-hot label vectors

    for audios, labels in tqdm(loader, desc="  Eval", leave=False):
        labels = labels.to(device)
        pooled = mert_forward_pool(audios, mert_model, processor, device,
                                   num_layers, hidden_dim)
        logits = head(pooled)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(audios)
        probs = torch.sigmoid(logits)
        preds = logits.argmax(dim=1)
        true_primary = labels.argmax(dim=1)
        correct += (preds == true_primary).sum().item()
        total += len(audios)
        all_labels_idx.extend(true_primary.cpu().tolist())
        all_preds_idx.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())
        all_labels_raw.append(labels.cpu().numpy())

    probs_arr = np.concatenate(all_probs, axis=0)      # (N, num_classes)
    labels_arr = np.concatenate(all_labels_raw, axis=0)  # (N, num_classes)

    return (total_loss / max(total, 1), correct / max(total, 1),
            all_labels_idx, all_preds_idx, probs_arr, labels_arr)


def print_eval_metrics(
    labels_idx: list[int], preds_idx: list[int],
    probs: np.ndarray, labels_multihot: np.ndarray,
) -> None:
    """Print compact multi-label metrics (notebook-aligned)."""
    from sklearn.metrics import average_precision_score, f1_score

    num_classes = len(CLASS_METERS)
    labels_binary = (labels_multihot > 0.5).astype(np.float32)
    print(f"\n{'Meter':>6s}  {'Correct':>7s}  {'Total':>5s}  {'Acc':>6s}  {'AP':>6s}")
    print("-" * 40)
    total_correct = total_count = 0
    aps = []
    labels_arr = np.array(labels_idx)
    preds_arr = np.array(preds_idx)

    for i, m in enumerate(CLASS_METERS):
        mask = labels_arr == i
        n = int(mask.sum())
        if n == 0:
            print(f"{m:>4d}/x  {'—':>7s}  {0:>5d}  {'—':>6s}  {'—':>6s}")
            continue
        c = int((preds_arr[mask] == i).sum())
        acc = c / n
        total_correct += c
        total_count += n
        ap_str = "—"
        if labels_binary[:, i].sum() > 0:
            ap = average_precision_score(labels_binary[:, i], probs[:, i])
            aps.append(ap)
            ap_str = f"{ap:.3f}"
        print(f"{m:>4d}/x  {c:>5d}/{n:<5d}       {acc:>5.1%}  {ap_str:>6s}")

    if total_count:
        print("-" * 40)
        if aps:
            print(f"{'Total':>6s}  {total_correct:>5d}/{total_count:<5d}       {total_correct/total_count:>5.1%}  mAP={np.mean(aps):.3f}")
        else:
            print(f"{'Total':>6s}  {total_correct:>5d}/{total_count:<5d}       {total_correct/total_count:>5.1%}")

    preds_binary = (probs > 0.5).astype(np.int32)
    cols = [i for i in range(num_classes) if labels_binary[:, i].sum() > 0]
    if cols:
        mf1 = f1_score(labels_binary[:, cols], preds_binary[:, cols], average="macro", zero_division=0)
        print(f"Macro-F1: {mf1:.3f}")

    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    gap = sorted_p[:, 0] - sorted_p[:, 1]
    print(f"Confidence gap: mean={gap.mean():.3f}, median={np.median(gap):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MERT with LoRA for meter classification")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/meter_mert_finetuned.pt"))
    parser.add_argument("--model", type=str, default="m-a-p/MERT-v1-330M",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for classification head (auto-scaled per model if not set)")
    parser.add_argument("--lora-lr", type=float, default=None,
                        help="Learning rate for LoRA parameters (auto-scaled per model if not set)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit entries per split (0 = all). Useful for smoke tests.")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Noise augmentation std (0 to disable)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--no-lora", action="store_true",
                        help="Freeze MERT entirely, only train the head (cheaper baseline)")
    parser.add_argument("--extra-data", type=Path, nargs="+", default=[],
                        help="Extra data directories with .tab files (e.g. data/wikimeter)")
    parser.add_argument("--extra-val-ratio", type=float, default=0.1,
                        help="Fraction of extra data songs held out for val")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Auto-scale LR per model size if not explicitly set
    lr_defaults = {
        "m-a-p/MERT-v1-95M":  {"lr": 1e-3, "lora_lr": 5e-5},
        "m-a-p/MERT-v1-330M": {"lr": 5e-4, "lora_lr": 5e-5},
    }
    defaults = lr_defaults.get(args.model, lr_defaults["m-a-p/MERT-v1-330M"])
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.lora_lr is None:
        args.lora_lr = defaults["lora_lr"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    model_name = args.model
    num_layers, hidden_dim = MODEL_CONFIGS[model_name]
    pooled_dim = hidden_dim * 2
    use_lora = not args.no_lora

    print(f"Model: {model_name}", flush=True)
    print(f"  Layers: {num_layers}, hidden: {hidden_dim}, pooled: {pooled_dim}")
    print(f"Device: {device}")
    print(f"LoRA: {'disabled' if not use_lora else f'rank={args.lora_rank}, alpha={args.lora_alpha}'}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # Load data
    data_dir = args.data_dir.resolve()
    print(f"\nLoading entries from {data_dir}")

    train_entries = load_split_entries(data_dir, "train")
    val_entries = load_split_entries(data_dir, "val")
    test_entries = load_split_entries(data_dir, "test")

    if train_entries is None or val_entries is None or test_entries is None:
        print("ERROR: Could not find train/val/test split files")
        sys.exit(1)

    # Load extra data directories with per-song stratified val split.
    # Notebook-aligned behavior: parse meter classes, ignore soft weights.
    if args.extra_data:
        import random
        import re
        from collections import defaultdict
        from scripts.utils import resolve_audio_path as _resolve
        random.seed(args.seed)
        valid_meters = set(METER_TO_IDX.keys())

        for extra_dir in args.extra_data:
            extra_dir = extra_dir.resolve()
            if not extra_dir.exists():
                print(f"  WARNING: extra-data dir not found: {extra_dir}")
                continue
            tab_files = sorted(
                p for p in extra_dir.iterdir()
                if p.suffix in (".tab", ".csv", ".tsv")
            )
            for tab_file in tab_files:
                import csv as _csv
                extra_entries: list[tuple[Path, list[int]]] = []
                with open(tab_file, newline="", encoding="utf-8") as fh:
                    reader = _csv.DictReader(fh, delimiter="\t")
                    for row in reader:
                        raw_fname = row.get("filename", "").strip().strip('"')
                        raw_meter = row.get("meter", "").strip().strip('"')
                        if not raw_fname or not raw_meter:
                            continue
                        meters: list[int] = []
                        try:
                            for part in raw_meter.split(","):
                                part = part.strip()
                                if ":" in part:
                                    m = int(part.split(":", 1)[0])
                                else:
                                    m = int(part)
                                if m in valid_meters:
                                    meters.append(m)
                        except ValueError:
                            continue
                        if not meters:
                            continue
                        audio_path = _resolve(raw_fname, extra_dir)
                        if audio_path is not None:
                            extra_entries.append((audio_path, meters))

                if not extra_entries:
                    continue

                # Per-song stratified split (no segment leakage)
                song_segments: dict[str, list[tuple[Path, list[int]]]] = defaultdict(list)
                for path, meters in extra_entries:
                    song_stem = re.sub(r"_seg\d+$", "", path.stem)
                    song_segments[song_stem].append((path, meters))

                # Group songs by primary meter for stratified split
                meter_songs: dict[int, list[str]] = defaultdict(list)
                for song_stem, segs in song_segments.items():
                    primary = segs[0][1][0]
                    meter_songs[primary].append(song_stem)

                # Pick ~val_ratio songs per meter for val
                val_songs: set[str] = set()
                for meter, songs in sorted(meter_songs.items()):
                    random.shuffle(songs)
                    n_val = max(1, int(len(songs) * args.extra_val_ratio))
                    val_songs.update(songs[:n_val])

                extra_train, extra_val = [], []
                for song_stem, segs in song_segments.items():
                    if song_stem in val_songs:
                        extra_val.extend(segs)
                    else:
                        extra_train.extend(segs)

                print(f"  Extra: +{len(extra_train)} train ({len(song_segments) - len(val_songs)} songs)"
                      f" +{len(extra_val)} val ({len(val_songs)} songs) from {tab_file.name}")
                train_entries.extend(extra_train)
                val_entries.extend(extra_val)

    if args.limit > 0:
        train_entries = train_entries[:args.limit]
        val_entries = val_entries[:args.limit]
        test_entries = test_entries[:args.limit]
        print(f"  Limited to {args.limit} entries per split")

    # Show distribution
    all_entries = train_entries + val_entries + test_entries
    meter_counts: Counter = Counter()
    for _, meters in all_entries:
        for m in meters:
            meter_counts[m] += 1
    print(f"\nTotal: {len(all_entries)} entries")
    for m in CLASS_METERS:
        print(f"  Meter {m}: {meter_counts.get(m, 0)}")
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test")

    # Load MERT model
    print(f"\nLoading {model_name}...")
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    mert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
    mert_model = mert_model.to(device)

    mert_params = sum(p.numel() for p in mert_model.parameters())
    print(f"  MERT parameters: {mert_params/1e6:.0f}M", flush=True)

    # Apply LoRA if requested
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("ERROR: peft not installed. Run: uv pip install peft")
            print("  Or use --no-lora to train only the classification head.")
            sys.exit(1)

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        mert_model = get_peft_model(mert_model, lora_config)
        trainable, total_p = mert_model.get_nb_trainable_parameters()
        print(f"  LoRA trainable: {trainable:,} / {total_p:,} ({trainable/total_p:.2%})")
    else:
        for param in mert_model.parameters():
            param.requires_grad = False
        print("  MERT frozen (no LoRA)")

    # Classification head
    head = MERTClassificationHead(
        num_layers=num_layers,
        pooled_dim=pooled_dim,
        num_classes=len(CLASS_METERS),
        head_dim=args.head_dim,
        dropout=args.dropout,
    ).to(device)

    head_params = sum(p.numel() for p in head.parameters())
    print(f"  Head parameters: {head_params:,}")

    # Datasets + DataLoaders
    train_ds = MERTAudioDataset(train_entries, augment=True, noise_std=args.noise_std)
    val_ds = MERTAudioDataset(val_entries, augment=False)
    test_ds = MERTAudioDataset(test_entries, augment=False)
    print(f"\nDatasets: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=simple_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=simple_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=simple_collate)

    # Pos weights for BCEWithLogitsLoss (sigmoid multi-label)
    pos_weights = compute_pos_weights(
        train_ds.entries, len(CLASS_METERS),
    ).to(device)
    print(f"Pos weights: {pos_weights.tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer: discriminative LR
    param_groups = [
        {"params": head.parameters(), "lr": args.lr},
    ]
    if use_lora:
        lora_params = [p for p in mert_model.parameters() if p.requires_grad]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lora_lr})
            print(f"\nOptimizer: head lr={args.lr}, LoRA lr={args.lora_lr}")
        else:
            print(f"\nOptimizer: head lr={args.lr} (no LoRA params found)")
    else:
        print(f"\nOptimizer: head lr={args.lr}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6,
    )

    # Resume from checkpoint (explicit --resume wins; else auto-resume from --checkpoint)
    start_epoch = 1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    checkpoint_path = args.checkpoint.resolve()
    resume_path = args.resume.resolve() if args.resume else checkpoint_path

    if resume_path.exists():
        print(f"\nResuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "head_state_dict" in ckpt:
            head.load_state_dict(ckpt["head_state_dict"])
        if use_lora and ckpt.get("lora_state_dict"):
            for name, param_data in ckpt["lora_state_dict"].items():
                parts = name.split(".")
                obj = mert_model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                getattr(obj, parts[-1]).data.copy_(param_data.to(device))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_accuracy", -1.0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Epoch {start_epoch}, best val: {best_val_acc:.1%}")

    # Training loop (notebook-aligned)
    patience_counter = 0

    print(f"\n{'Epoch':>5s}  {'TrainLoss':>10s}  {'TrainAcc':>9s}  "
          f"{'ValLoss':>10s}  {'ValAcc':>9s}  {'LR':>10s}  {'Time':>6s}", flush=True)
    print("-" * 65, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            mert_model, head, processor, train_loader, criterion, optimizer,
            device, num_layers, hidden_dim, args.grad_accum, use_lora,
        )
        val_loss, val_acc, _, _, _, _ = evaluate(
            mert_model, head, processor, val_loader, criterion,
            device, num_layers, hidden_dim,
        )

        scheduler.step(val_acc)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_acc > best_val_acc
        marker = " *" if improved else ""
        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  "
              f"{val_loss:10.4f}  {val_acc:8.1%}  {current_lr:10.1e}  {elapsed:5.0f}s{marker}", flush=True)

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint every epoch
        ckpt_data = {
            "head_state_dict": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            "lora_state_dict": {n: p.cpu().clone() for n, p in mert_model.named_parameters() if p.requires_grad} if use_lora else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_map": IDX_TO_METER,
            "model_name": model_name,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "pooled_dim": pooled_dim,
            "head_dim": args.head_dim,
            "num_classes": len(CLASS_METERS),
            "dropout": args.dropout,
            "val_accuracy": best_val_acc,
            "val_loss": best_val_loss,
            "epoch": epoch,
            "lora_rank": args.lora_rank if use_lora else 0,
            "lora_alpha": args.lora_alpha if use_lora else 0,
            "model_type": "MERTFineTuned",
        }
        torch.save(ckpt_data, checkpoint_path)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test set evaluation")
    print("=" * 60)

    # Notebook behavior: evaluate from last saved checkpoint.
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "head_state_dict" in ckpt:
            head.load_state_dict(ckpt["head_state_dict"])
            head = head.to(device)
        if use_lora and ckpt.get("lora_state_dict"):
            for name, param_data in ckpt["lora_state_dict"].items():
                parts = name.split(".")
                obj = mert_model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                getattr(obj, parts[-1]).data.copy_(param_data.to(device))
        print(f"Loaded: epoch {ckpt.get('epoch', '?')}, val {ckpt.get('val_accuracy', 0):.1%}")

    test_loss, test_acc, test_labels, test_preds, test_probs, test_labels_mh = evaluate(
        mert_model, head, processor, test_loader, criterion,
        device, num_layers, hidden_dim,
    )
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.1%} ({sum(1 for a, b in zip(test_labels, test_preds) if a == b)}/{len(test_labels)})")
    print(f"Best val accuracy: {best_val_acc:.1%}")

    print_eval_metrics(test_labels, test_preds, test_probs, test_labels_mh)

    # Persist test accuracy into checkpoint.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt["test_accuracy"] = test_acc
        torch.save(ckpt, checkpoint_path)
    else:
        torch.save(
            {
                "head_state_dict": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                "lora_state_dict": {n: p.cpu().clone() for n, p in mert_model.named_parameters() if p.requires_grad} if use_lora else None,
                "class_map": IDX_TO_METER,
                "model_name": model_name,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "pooled_dim": pooled_dim,
                "head_dim": args.head_dim,
                "num_classes": len(CLASS_METERS),
                "dropout": args.dropout,
                "val_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "epoch": max(start_epoch - 1, 0),
                "lora_rank": args.lora_rank if use_lora else 0,
                "lora_alpha": args.lora_alpha if use_lora else 0,
                "model_type": "MERTFineTuned",
            },
            checkpoint_path,
        )

    print(f"\nCheckpoint saved to {checkpoint_path}")
    print(f"  Model: {model_name}")
    print(f"  LoRA: {'rank=' + str(args.lora_rank) if use_lora else 'none'}")
    print(f"  Val accuracy:  {best_val_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")


if __name__ == "__main__":
    main()
