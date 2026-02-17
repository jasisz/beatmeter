#!/usr/bin/env python3
"""Find audio files where beat trackers returned empty results.

Scans the analysis cache for empty beat tracker outputs ([] = no beats detected).
Maps audio hashes back to filenames and looks up YouTube URLs for WIKIMETER songs.

Files that fail multiple trackers are likely not music (speech, applause, silence).

Usage:
    uv run python scripts/setup/find_bad_audio.py
    uv run python scripts/setup/find_bad_audio.py --min-failures 2   # only show files failing 2+ trackers
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from beatmeter.analysis.cache import AnalysisCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
METER2800_AUDIO = PROJECT_ROOT / "data" / "meter2800" / "audio"
WIKIMETER_AUDIO = PROJECT_ROOT / "data" / "wikimeter" / "audio"
WIKIMETER_JSON = PROJECT_ROOT / "scripts" / "setup" / "wikimeter.json"
BLACKLIST_PATH = PROJECT_ROOT / "scripts" / "setup" / "meter2800_blacklist.txt"


def load_blacklist() -> set[str]:
    stems: set[str] = set()
    if BLACKLIST_PATH.exists():
        for line in BLACKLIST_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                stems.add(line)
    return stems


def build_hash_to_path() -> dict[str, Path]:
    """Map audio_hash -> file path for all known audio files."""
    cache = AnalysisCache()
    mapping: dict[str, Path] = {}

    for audio_dir in [METER2800_AUDIO, WIKIMETER_AUDIO]:
        if not audio_dir.exists():
            continue
        for f in sorted(audio_dir.iterdir()):
            if f.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"}:
                ah = cache.audio_hash(str(f))
                mapping[ah] = f

    return mapping


def load_wikimeter_urls() -> dict[str, str]:
    """Map song stem (sanitized filename) -> YouTube URL."""
    sys.path.insert(0, str(PROJECT_ROOT))

    urls: dict[str, str] = {}
    if not WIKIMETER_JSON.exists():
        return urls

    from scripts.setup.download_wikimeter import sanitize_filename

    with open(WIKIMETER_JSON) as f:
        songs = json.load(f)

    for song in songs:
        artist = song.get("artist", "")
        title = song.get("title", "")
        stem = sanitize_filename(artist, title)
        for src in song.get("sources", []):
            url = src.get("url", "")
            if "youtube" in url or "youtu.be" in url:
                urls[stem] = url
                break

    return urls


def find_empty_beats() -> dict[str, list[str]]:
    """Find all empty beat cache entries. Returns {audio_hash: [tracker_names]}."""
    beats_dir = CACHE_DIR / "beats"
    if not beats_dir.exists():
        return {}

    empty: dict[str, list[str]] = {}

    for tracker_dir in sorted(beats_dir.iterdir()):
        if not tracker_dir.is_dir():
            continue
        tracker_name = tracker_dir.name

        for hash_dir in tracker_dir.iterdir():
            if not hash_dir.is_dir():
                continue
            for json_file in hash_dir.glob("*.json"):
                content = json_file.read_text().strip()
                if content == "[]":
                    ah = json_file.stem
                    if ah not in empty:
                        empty[ah] = []
                    empty[ah].append(tracker_name)

    return empty


def main():
    parser = argparse.ArgumentParser(description="Find audio files with empty tracker results")
    parser.add_argument("--min-failures", type=int, default=1,
                        help="Minimum number of failed trackers to show (default: 1)")
    args = parser.parse_args()

    print("Scanning cache for empty beat results...", flush=True)
    empty = find_empty_beats()
    if not empty:
        print("No empty results found.")
        return

    print(f"Found {len(empty)} audio hashes with empty results.\n")
    print("Building hash→filename mapping...", flush=True)
    h2p = build_hash_to_path()
    wiki_urls = load_wikimeter_urls()
    blacklist = load_blacklist()

    import re

    # Group by number of failures
    results = []
    for ah, trackers in sorted(empty.items(), key=lambda x: -len(x[1])):
        path = h2p.get(ah)
        if path is None:
            continue

        n_fail = len(trackers)
        if n_fail < args.min_failures:
            continue

        stem = path.stem
        is_blacklisted = stem in blacklist
        dataset = "wikimeter" if "wikimeter" in str(path) else "meter2800"

        # Find YouTube URL for WIKIMETER (exact stem match)
        youtube = ""
        if dataset == "wikimeter":
            song_stem = re.sub(r"_seg\d+$", "", stem)
            youtube = wiki_urls.get(song_stem, "")

        results.append((n_fail, path, trackers, dataset, is_blacklisted, youtube))

    # Sort: most failures first, then by name
    results.sort(key=lambda x: (-x[0], str(x[1])))

    print(f"\n{'='*80}")
    print(f"{'FAILURES':>8}  {'STATUS':>10}  {'DATASET':>10}  FILE")
    print(f"{'='*80}")

    for n_fail, path, trackers, dataset, is_bl, youtube in results:
        status = "BLACKLIST" if is_bl else "NEW"
        tracker_str = ", ".join(sorted(trackers))
        print(f"{n_fail:>8}  {status:>10}  {dataset:>10}  {path}")
        print(f"{'':>32}  trackers: {tracker_str}")
        if youtube:
            print(f"{'':>32}  youtube: {youtube}")
        print()

    # Summary
    new_count = sum(1 for _, _, _, _, bl, _ in results if not bl)
    bl_count = sum(1 for _, _, _, _, bl, _ in results if bl)
    print(f"Total: {len(results)} files ({new_count} new, {bl_count} already blacklisted)")


if __name__ == "__main__":
    main()
