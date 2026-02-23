#!/usr/bin/env python3
"""Import pre-extracted MERT embeddings into analysis.lmdb as meter_net_mert.

Reads .npy files from data/mert_embeddings/{meter2800,wikimeter}/ and saves
a single layer (default: layer 3) as a float32 array in the analysis cache.

This is a one-time batch operation. After running, MeterNet inference can
load MERT features from cache without the 2GB MERT model.

Usage:
    uv run python scripts/setup/import_mert.py
    uv run python scripts/setup/import_mert.py --layer 3 --limit 5  # smoke test
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def collect_audio_files(meter2800_dir: Path, wikimeter_dir: Path) -> list[Path]:
    """Collect all audio files from both datasets."""
    files: list[Path] = []

    # METER2800
    audio_dir = meter2800_dir / "audio"
    if audio_dir.exists():
        for ext in ("*.mp3", "*.wav"):
            files.extend(audio_dir.glob(ext))

    # WIKIMETER
    audio_dir = wikimeter_dir / "audio"
    if audio_dir.exists():
        for ext in ("*.mp3", "*.wav"):
            files.extend(audio_dir.glob(ext))

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Import MERT embeddings into analysis.lmdb")
    parser.add_argument("--layer", type=int, default=3, help="MERT layer to import (default: 3)")
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    parser.add_argument("--meter2800", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--wikimeter", type=Path, default=Path("data/wikimeter"))
    parser.add_argument("--mert-dir", type=Path, default=Path("data/mert_embeddings"))
    args = parser.parse_args()

    from beatmeter.analysis.cache import AnalysisCache

    cache = AnalysisCache()

    # Build stem -> .npy lookup
    mert_lookup: dict[str, Path] = {}
    for subdir in ["meter2800", "wikimeter"]:
        d = args.mert_dir / subdir
        if d.exists():
            for npy in d.glob("*.npy"):
                mert_lookup[npy.stem] = npy
    print(f"MERT embeddings found: {len(mert_lookup)}")

    # Collect audio files (skip macOS resource forks)
    audio_files = [f for f in collect_audio_files(args.meter2800, args.wikimeter)
                   if not f.name.startswith("._")]
    if args.limit:
        audio_files = audio_files[:args.limit]
    print(f"Audio files: {len(audio_files)}")

    imported = 0
    skipped = 0
    already = 0

    for audio_path in audio_files:
        audio_hash = cache.audio_hash(str(audio_path))

        # Check if already cached
        existing = cache.load_array(audio_hash, "meter_net_mert")
        if existing is not None:
            already += 1
            continue

        # Find matching .npy
        npy_path = mert_lookup.get(audio_path.stem)
        if npy_path is None:
            skipped += 1
            continue

        try:
            emb = np.load(npy_path)  # shape (12, 1536)
            layer_emb = emb[args.layer].astype(np.float32)
            cache.save_array(audio_hash, "meter_net_mert", layer_emb)
            imported += 1
        except Exception as e:
            print(f"  ERROR {audio_path.stem}: {e}")
            skipped += 1

    print(f"\nDone: imported={imported}, already_cached={already}, skipped={skipped}")
    cache.close()


if __name__ == "__main__":
    main()
