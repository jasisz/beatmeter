#!/usr/bin/env python3
"""Extract MERT embeddings for METER2800 and/or local fixtures.

Saves per-file embeddings as .npy arrays of shape (num_layers, pooled_dim) — one
(mean+max)-pooled vector per hidden layer. These are used by
train_meter_mert.py to train a lightweight MLP classifier.

Supports both MERT-v1-95M (12 layers, 768 hidden) and MERT-v1-330M (24 layers, 1024 hidden).

Usage:
    uv run python scripts/training/extract_mert_embeddings.py --data-dir data/meter2800
    uv run python scripts/training/extract_mert_embeddings.py --data-dir data/meter2800 --model m-a-p/MERT-v1-330M
    uv run python scripts/training/extract_mert_embeddings.py --fixtures-dir tests/fixtures
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "m-a-p/MERT-v1-95M"
MERT_SR = 24000          # MERT expects 24 kHz
CHUNK_SAMPLES = 5 * MERT_SR  # 5-second chunks (120 000 samples)
MAX_DURATION_S = 30       # center-crop to 30 s

# Model configs: (num_transformer_layers, hidden_dim)
MODEL_CONFIGS = {
    "m-a-p/MERT-v1-95M":  (12, 768),
    "m-a-p/MERT-v1-330M": (24, 1024),
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".oga", ".opus", ".aiff", ".aif"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mert_model(model_name: str, device: torch.device):
    """Load MERT model and processor from HuggingFace."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    print(f"Loading {model_name} ...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded on {device} ({params/1e6:.0f}M params)")
    return model, processor


# ---------------------------------------------------------------------------
# Audio loading + resampling
# ---------------------------------------------------------------------------

def load_audio_24k(path: Path) -> np.ndarray | None:
    """Load audio, convert to mono float32 at 24 kHz, center-crop to 30 s."""
    try:
        # Use librosa for loading (handles mp3, ogg, flac, wav etc.)
        audio, sr = librosa.load(str(path), sr=MERT_SR, mono=True)
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None

    if audio is None or len(audio) == 0:
        warnings.warn(f"Empty audio: {path}")
        return None

    # center crop to MAX_DURATION_S
    max_samples = MAX_DURATION_S * MERT_SR
    if len(audio) > max_samples:
        start = (len(audio) - max_samples) // 2
        audio = audio[start : start + max_samples]

    if len(audio) < MERT_SR:  # skip files shorter than 1 s
        warnings.warn(f"Audio too short ({len(audio)/MERT_SR:.1f}s): {path}")
        return None

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embedding(
    audio: np.ndarray,
    model,
    processor,
    device: torch.device,
    num_layers: int,
    hidden_dim: int,
) -> np.ndarray:
    """Extract (num_layers, hidden_dim*2) embedding from audio via MERT.

    Splits audio into 5 s non-overlapping chunks, runs MERT on each,
    pools (mean + max) per layer per chunk, then aggregates across chunks.
    """
    pooled_dim = hidden_dim * 2  # mean + max

    # Split into chunks
    chunks = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if len(chunk) < MERT_SR:  # skip chunks < 1 s
            continue
        chunks.append(chunk)

    if not chunks:
        chunks = [audio]

    # Accumulate per-layer pooled vectors
    layer_means = [[] for _ in range(num_layers)]
    layer_maxes = [[] for _ in range(num_layers)]

    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        # outputs.hidden_states: tuple of (1, T, hidden_dim)
        # hidden_states[0] is the conv feature extractor output
        # hidden_states[1..num_layers] are the transformer layers
        hidden_states = outputs.hidden_states

        for layer_idx in range(num_layers):
            hs = hidden_states[layer_idx + 1].squeeze(0)  # (T, hidden_dim)
            layer_means[layer_idx].append(hs.mean(dim=0).cpu().numpy())
            layer_maxes[layer_idx].append(hs.max(dim=0).values.cpu().numpy())

    # Aggregate across chunks: mean of means, max of maxes
    embedding = np.zeros((num_layers, pooled_dim), dtype=np.float32)
    for layer_idx in range(num_layers):
        mean_vec = np.mean(layer_means[layer_idx], axis=0)   # (hidden_dim,)
        max_vec = np.max(layer_maxes[layer_idx], axis=0)      # (hidden_dim,)
        embedding[layer_idx] = np.concatenate([mean_vec, max_vec])

    return embedding


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_files(
    audio_paths: list[Path],
    output_dir: Path,
    model,
    processor,
    device: torch.device,
    num_layers: int,
    hidden_dim: int,
    resume: bool = True,
):
    """Extract embeddings for a list of audio files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(audio_paths)
    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, path in enumerate(audio_paths):
        out_path = output_dir / f"{path.stem}.npy"
        if resume and out_path.exists():
            skipped += 1
            continue

        audio = load_audio_24k(path)
        if audio is None:
            failed += 1
            continue

        embedding = extract_embedding(audio, model, processor, device, num_layers, hidden_dim)
        np.save(out_path, embedding)
        done += 1

        elapsed = time.time() - t_start
        rate = (done + skipped) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{total}] {path.name} -> {out_path.name}  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"\nDone: {done} extracted, {skipped} skipped (cached), {failed} failed")


def collect_meter2800_files(data_dir: Path) -> list[Path]:
    """Collect audio files referenced in METER2800 label files."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.utils import parse_label_file

    audio_paths = []
    seen = set()
    valid_meters = {3, 4, 5, 7}

    for ext in (".tab", ".csv", ".tsv"):
        for label_file in sorted(data_dir.glob(f"*{ext}")):
            entries = parse_label_file(label_file, data_dir, valid_meters=valid_meters)
            for path, _meter in entries:
                if path not in seen:
                    seen.add(path)
                    audio_paths.append(path)

    return audio_paths


def collect_fixture_files(fixtures_dir: Path) -> list[Path]:
    """Collect all audio files from fixtures directory."""
    files = []
    for f in sorted(fixtures_dir.iterdir()):
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file():
            files.append(f)
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract MERT embeddings")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="METER2800 dataset directory (e.g., data/meter2800)")
    parser.add_argument("--fixtures-dir", type=Path, default=None,
                        help="Local fixtures directory (e.g., tests/fixtures)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mert_embeddings"),
                        help="Output directory for .npy embeddings")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_CONFIGS.keys()),
                        help=f"MERT model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-extract even if .npy already exists")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, mps, cuda, cpu")
    args = parser.parse_args()

    if args.data_dir is None and args.fixtures_dir is None:
        parser.error("Specify at least one of --data-dir or --fixtures-dir")

    # Model config
    model_name = args.model
    if model_name not in MODEL_CONFIGS:
        parser.error(f"Unknown model: {model_name}. Known: {list(MODEL_CONFIGS.keys())}")
    num_layers, hidden_dim = MODEL_CONFIGS[model_name]
    pooled_dim = hidden_dim * 2

    # Output subdirectory suffix for non-default models
    model_suffix = ""
    if model_name != DEFAULT_MODEL:
        # e.g. "m-a-p/MERT-v1-330M" -> "330m"
        model_suffix = "_" + model_name.split("-")[-1].lower()

    print(f"Model: {model_name}")
    print(f"  Layers: {num_layers}, hidden: {hidden_dim}, pooled: {pooled_dim}")

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model once
    model, processor = load_mert_model(model_name, device)

    # Process METER2800
    if args.data_dir is not None:
        data_dir = args.data_dir.resolve()
        print(f"\n--- METER2800 ({data_dir}) ---")
        files = collect_meter2800_files(data_dir)
        print(f"Found {len(files)} audio files")
        out_dir = args.output_dir / f"meter2800{model_suffix}"
        process_files(files, out_dir, model, processor, device,
                      num_layers, hidden_dim, resume=not args.no_resume)

    # Process fixtures
    if args.fixtures_dir is not None:
        fixtures_dir = args.fixtures_dir.resolve()
        print(f"\n--- Fixtures ({fixtures_dir}) ---")
        files = collect_fixture_files(fixtures_dir)
        print(f"Found {len(files)} audio files")
        out_dir = args.output_dir / f"fixtures{model_suffix}"
        process_files(files, out_dir, model, processor, device,
                      num_layers, hidden_dim, resume=not args.no_resume)


if __name__ == "__main__":
    main()
