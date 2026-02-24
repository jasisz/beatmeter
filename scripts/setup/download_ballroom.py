#!/usr/bin/env python3
"""Download and prepare the Ballroom dataset for cross-dataset evaluation.

The Ballroom dataset (Krebs et al., 2004) contains 698 ~30s dance music
excerpts across 8 genres. We use it as an independent meter benchmark:
  - 3/4: Waltz (110), VienneseWaltz (65) → 175 tracks
  - 4/4: ChaChaCha (111), Jive (60), Quickstep (82), Rumba (98),
          Samba (86), Tango (86) → 523 tracks

Source: https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html

Usage:
    uv run python scripts/setup/download_ballroom.py
"""

import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

AUDIO_URL = "https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz"
DEST_DIR = Path("data/ballroom")
AUDIO_DIR = DEST_DIR / "audio"

# Genre → meter mapping (standard ballroom dance rules)
GENRE_METER = {
    "Waltz": 3,
    "VienneseWaltz": 3,
    "ChaChaCha": 4,
    "Jive": 4,
    "Quickstep": 4,
    "Rumba": 4,
    "Rumba-American": 4,
    "Rumba-International": 4,
    "Rumba-Misc": 4,
    "Samba": 4,
    "Tango": 4,
}


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress indicator."""
    print(f"  Downloading {url}...")
    req = urllib.request.Request(url, headers={
        "User-Agent": "RhythmAnalyzerBenchmark/1.0 (research)"
    })
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.0f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded // (1024*1024)}MB", end="", flush=True)
    print()


def main():
    if AUDIO_DIR.exists() and any(AUDIO_DIR.iterdir()):
        n_files = sum(1 for f in AUDIO_DIR.rglob("*.wav"))
        if n_files >= 690:
            print(f"  Ballroom already downloaded: {n_files} files in {AUDIO_DIR}")
            print("  Delete data/ballroom/ to re-download.")
            return
        print(f"  Partial download found ({n_files} files), re-downloading...")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DEST_DIR / "data1.tar.gz"

    # Download
    if not tar_path.exists():
        download_with_progress(AUDIO_URL, tar_path)
    else:
        print(f"  Archive already exists: {tar_path}")

    # Extract
    print("  Extracting...")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(DEST_DIR / "_tmp")

    # The tar extracts to BallroomData/ with genre subdirs
    ballroom_root = DEST_DIR / "_tmp" / "BallroomData"
    if not ballroom_root.exists():
        # Try finding it
        for d in (DEST_DIR / "_tmp").rglob("*"):
            if d.is_dir() and d.name == "BallroomData":
                ballroom_root = d
                break

    if not ballroom_root.exists():
        print(f"  ERROR: Could not find BallroomData/ in archive")
        print(f"  Contents of {DEST_DIR / '_tmp'}:")
        for p in (DEST_DIR / "_tmp").iterdir():
            print(f"    {p.name}")
        sys.exit(1)

    # Flatten: genre/file.wav → audio/Genre_file.wav
    # Also build metadata TSV
    entries = []
    for genre_dir in sorted(ballroom_root.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name
        if genre not in GENRE_METER:
            print(f"  Skipping unknown genre: {genre}")
            continue

        meter = GENRE_METER[genre]
        for wav in sorted(genre_dir.glob("*.wav")):
            dest_name = f"{genre}_{wav.name}"
            dest_path = AUDIO_DIR / dest_name
            shutil.copy2(wav, dest_path)
            entries.append((dest_name, genre, meter))

    # Write metadata
    meta_path = DEST_DIR / "data_ballroom.tab"
    with open(meta_path, "w") as f:
        f.write("filename\tgenre\tmeter\n")
        for fname, genre, meter in entries:
            f.write(f"{fname}\t{genre}\t{meter}\n")

    # Cleanup
    shutil.rmtree(DEST_DIR / "_tmp")
    tar_path.unlink()

    # Summary
    from collections import Counter
    genre_counts = Counter(g for _, g, _ in entries)
    meter_counts = Counter(m for _, _, m in entries)

    print(f"\n  Ballroom dataset ready: {len(entries)} files")
    print(f"  Location: {AUDIO_DIR}")
    print(f"  Metadata: {meta_path}")
    print(f"\n  By genre:")
    for genre, count in sorted(genre_counts.items()):
        meter = GENRE_METER[genre]
        print(f"    {genre:20s} {count:4d} tracks  ({meter}/4)")
    print(f"\n  By meter:")
    for meter, count in sorted(meter_counts.items()):
        print(f"    {meter}/4: {count} tracks")


if __name__ == "__main__":
    main()
