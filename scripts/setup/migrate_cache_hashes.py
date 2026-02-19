#!/usr/bin/env python3
"""Migrate .cache files from old audio_hash (stem+size+200KB) to new (stem+size).

Old hash: SHA256(stem + file_size + first_200KB_of_content)[:16]
New hash: SHA256(stem + file_size)[:16]

Renames JSON files in .cache/ so the warm cache stays valid after the
audio_hash change.  Safe to run multiple times (skips already-migrated).

Usage:
    uv run python scripts/setup/migrate_cache_hashes.py
    uv run python scripts/setup/migrate_cache_hashes.py --dry-run
"""

import argparse
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"

# Directories with audio files
AUDIO_DIRS = [
    PROJECT_ROOT / "data" / "meter2800" / "audio",
    PROJECT_ROOT / "data" / "wikimeter" / "audio",
]


def old_audio_hash(file_path: Path) -> str:
    """Old hash: stem + size + first 200KB content."""
    h = hashlib.sha256()
    h.update(file_path.stem.encode("utf-8", errors="ignore"))
    try:
        h.update(str(file_path.stat().st_size).encode("ascii"))
    except OSError:
        pass
    try:
        with open(file_path, "rb") as f:
            h.update(f.read(200_000))
    except OSError:
        pass
    return h.hexdigest()[:16]


def new_audio_hash(file_path: Path) -> str:
    """New hash: stem + size only."""
    h = hashlib.sha256()
    h.update(file_path.stem.encode("utf-8", errors="ignore"))
    try:
        h.update(str(file_path.stat().st_size).encode("ascii"))
    except OSError:
        pass
    return h.hexdigest()[:16]


def collect_audio_files() -> list[Path]:
    """Find all audio files in known directories."""
    files = []
    for d in AUDIO_DIRS:
        if d.exists():
            files.extend(d.glob("*.mp3"))
            files.extend(d.glob("*.wav"))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Collecting audio files...")
    audio_files = collect_audio_files()
    print(f"  Found {len(audio_files)} audio files")

    # Build old_hash -> new_hash mapping
    print("Computing hash mapping...")
    mapping: dict[str, str] = {}  # old_hash -> new_hash
    same = 0
    for af in audio_files:
        oh = old_audio_hash(af)
        nh = new_audio_hash(af)
        if oh == nh:
            same += 1
            continue
        mapping[oh] = nh

    print(f"  {len(mapping)} files need migration, {same} unchanged")

    if not mapping:
        print("Nothing to migrate!")
        return

    # Walk .cache/ and rename files
    renamed = 0
    skipped = 0
    not_found = 0

    for json_file in CACHE_DIR.rglob("*.json"):
        stem = json_file.stem  # this is the audio_hash
        if stem in mapping:
            new_name = json_file.with_name(mapping[stem] + ".json")
            if new_name.exists():
                skipped += 1
                continue
            if args.dry_run:
                print(f"  RENAME {json_file.relative_to(CACHE_DIR)} -> {new_name.name}")
            else:
                json_file.rename(new_name)
            renamed += 1

    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Renamed {renamed} files, skipped {skipped} (already exist)")


if __name__ == "__main__":
    main()
