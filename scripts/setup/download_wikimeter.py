#!/usr/bin/env python3
"""Download WIKIMETER dataset from YouTube.

Songs with Wikipedia-verified time signatures across all meter classes
(3/4, 4/4, 5/x, 7/x, 9/x, 11/x, polyrhythmic).

Each song is split into segments (default 35s). Output is a .tab file
compatible with METER2800 parse_label_file().

Song catalog: wikimeter.json (single source of truth).

DISCLAIMER: Audio downloaded by this script is intended solely for
non-commercial academic research purposes (fair use).
Only short excerpts are retained.

Requirements:
    pip install yt-dlp
    brew install ffmpeg  (or apt-get install ffmpeg)

Usage:
    uv run python scripts/setup/download_wikimeter.py --dry-run
    uv run python scripts/setup/download_wikimeter.py --limit 3
    uv run python scripts/setup/download_wikimeter.py
    uv run python scripts/setup/download_wikimeter.py --meters 5,7,9,11
    uv run python scripts/setup/download_wikimeter.py --no-poly
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "scripts" / "setup" / "wikimeter.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "wikimeter"
METER_LABELS = {3: "three", 4: "four", 5: "five", 7: "seven", 9: "nine", 11: "eleven"}
MAX_SEGMENTS = 5  # Cap per song: 5 × 35s = 2.9 min max (fair use)

Meters = dict[int, float]  # {meter: weight}, e.g. {3: 1.0} or {3: 0.7, 4: 0.8}

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def load_catalog(path: Path) -> list[dict]:
    """Load wikimeter.json. Converts meter keys from str to int."""
    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    for entry in catalog:
        entry["meters"] = {int(k): v for k, v in entry["meters"].items()}
    return catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def meters_to_list(meters: Meters) -> list[int]:
    return list(meters.keys())


def meters_to_tab_str(meters: Meters) -> str:
    """Convert meters to .tab format: {3: 1.0} → "3", {3: 0.9, 4: 0.8} → "3:0.9,4:0.8"."""
    parts = []
    for m, w in meters.items():
        parts.append(str(m) if w == 1.0 else f"{m}:{w}")
    return ",".join(parts)


def sanitize_filename(artist: str, title: str) -> str:
    name = f"{artist}_{title}".lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80]


def get_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return None


def download_audio(video_id: str, dest_dir: Path, timeout: int = 120) -> Path | None:
    """Download full audio from YouTube by video ID. Returns path or None."""
    tmp_path = dest_dir / "full.%(ext)s"
    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp", url,
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(tmp_path),
        "--no-playlist", "--quiet", "--no-warnings",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f" yt-dlp error: {stderr[:100]}")
            return None
    except subprocess.TimeoutExpired:
        print(" timeout")
        return None
    except FileNotFoundError:
        print(" ERROR: yt-dlp not found. Install: pip install yt-dlp")
        sys.exit(1)

    downloaded = list(dest_dir.glob("full.*"))
    return downloaded[0] if downloaded else None


def segment_audio(
    src: Path, stem: str, audio_dir: Path, segment_length: int = 35,
) -> list[str]:
    """Split audio into segments evenly spaced across the track."""
    duration = get_duration(src)
    if duration is None or duration < 15:
        return []

    audio_dir.mkdir(parents=True, exist_ok=True)

    margin = 10.0 if duration > 40 else 0.0
    usable_start = margin
    usable_duration = duration - 2 * margin

    if usable_duration < 15:
        usable_start = 0
        usable_duration = duration

    MIN_GAP = 5.0
    stride = segment_length + MIN_GAP
    n_segments = min(MAX_SEGMENTS, max(1, int((usable_duration + MIN_GAP) // stride)))

    if n_segments == 1:
        starts = [max(0.0, (usable_duration - segment_length) / 2)]
    elif n_segments * segment_length > usable_duration:
        total_needed = n_segments * segment_length
        center_start = max(0.0, (usable_duration - total_needed) / 2)
        starts = [center_start + i * segment_length for i in range(n_segments)]
    else:
        actual_stride = (usable_duration - segment_length) / (n_segments - 1) if n_segments > 1 else 0
        starts = [i * actual_stride for i in range(n_segments)]

    segments: list[str] = []
    for seg_idx, offset in enumerate(starts):
        seg_duration = min(segment_length, usable_duration - offset)
        if seg_duration < 15:
            continue
        start_time = usable_start + offset
        seg_stem = f"{stem}_seg{seg_idx:02d}"
        dest = audio_dir / f"{seg_stem}.mp3"

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-ss", f"{start_time:.1f}", "-t", f"{seg_duration:.1f}",
                 "-acodec", "libmp3lame", "-ab", "192k",
                 "-ar", "44100", "-ac", "1", str(dest)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                segments.append(seg_stem)
        except subprocess.TimeoutExpired:
            pass

    return segments


def download_and_segment(
    video_id: str, stem: str, audio_dir: Path, segment_length: int = 35,
) -> list[str]:
    """Download + segment. Returns list of segment stems."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = download_audio(video_id, Path(tmpdir))
        if src is None:
            return []

        duration = get_duration(src)
        if duration is None or duration < 15:
            print(f" too short ({duration}s)")
            return []

        return segment_audio(src, stem, audio_dir, segment_length)


def write_tab_file(entries: list[tuple[str, Meters]], tab_path: Path) -> None:
    """Write .tab file compatible with METER2800 format."""
    tab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tab_path, "w", encoding="utf-8") as f:
        f.write("filename\tlabel\tmeter\talt_meter\n")
        for stem, meters in entries:
            meter_list = meters_to_list(meters)
            primary = meter_list[0]
            label = METER_LABELS.get(primary, str(primary))
            meter_str = meters_to_tab_str(meters)
            alt = primary * 2
            f.write(f'"/WIKIMETER/{stem}.mp3"\t"{label}"\t{meter_str}\t{alt}\n')


def parse_meters_arg(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download WIKIMETER dataset from YouTube"
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Max songs (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--meters", type=str, default="", help="Filter: 3,4 or 5,7,9,11")
    parser.add_argument("--segment-length", type=int, default=35)
    parser.add_argument("--no-poly", action="store_true", help="Exclude polyrhythmic songs")
    parser.add_argument("--tab-only", action="store_true", help="Regenerate .tab only")
    args = parser.parse_args()

    catalog_path: Path = args.catalog.resolve()
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found: {catalog_path}")
        sys.exit(1)

    catalog = load_catalog(catalog_path)
    output_dir: Path = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    tab_path = output_dir / "data_wikimeter.tab"
    skip_existing = not args.no_skip_existing
    meter_filter = set(parse_meters_arg(args.meters)) if args.meters else None

    # Filter
    songs = list(catalog)
    if args.no_poly:
        songs = [s for s in songs if len(meters_to_list(s["meters"])) == 1]
    if meter_filter:
        songs = [s for s in songs if any(m in meter_filter for m in meters_to_list(s["meters"]))]
    if args.limit > 0:
        songs = songs[:args.limit]

    meter_counts: dict[int, int] = {}
    for s in songs:
        for m in meters_to_list(s["meters"]):
            meter_counts[m] = meter_counts.get(m, 0) + 1

    meter_summary = ", ".join(f"{meter_counts.get(m, 0)}×{m}/x" for m in sorted(meter_counts))
    print(f"WIKIMETER — {len(songs)} songs ({meter_summary})")

    # Dry run
    if args.dry_run:
        for i, song in enumerate(songs, 1):
            meters = song["meters"]
            meter_list = meters_to_list(meters)
            meters_str = "+".join(str(m) for m in meter_list)
            soft = any(w != 1.0 for w in meters.values())
            tag = " [soft]" if soft else (" [poly]" if len(meter_list) > 1 else "")
            print(f"  {i:3d}. [{meters_str}] {song['artist']} — {song['title']}{tag}  {song['video_id']}")
        return

    # Tab-only: scan existing audio
    if args.tab_only:
        successful: list[tuple[str, Meters]] = []
        for song in songs:
            stem = sanitize_filename(song["artist"], song["title"])
            existing = sorted(f for f in audio_dir.glob(f"{stem}_seg*.mp3")
                              if f.stat().st_size > 1000)
            for f in existing[:MAX_SEGMENTS]:
                successful.append((f.stem, song["meters"]))
        write_tab_file(successful, tab_path)
        print(f"  {len(successful)} segments → {tab_path}")
        return

    # Download
    audio_dir.mkdir(parents=True, exist_ok=True)
    successful: list[tuple[str, Meters]] = []
    stats = {"ok": 0, "skip": 0, "fail": 0}

    for i, song in enumerate(songs, 1):
        artist, title = song["artist"], song["title"]
        video_id = song["video_id"]
        meters = song["meters"]
        stem = sanitize_filename(artist, title)
        meters_str = "+".join(str(m) for m in meters_to_list(meters))

        # Check existing
        existing = sorted(f for f in audio_dir.glob(f"{stem}_seg*.mp3")
                          if f.stat().st_size > 1000)
        if skip_existing and existing:
            n = min(len(existing), MAX_SEGMENTS)
            for f in existing[:n]:
                successful.append((f.stem, meters))
            stats["skip"] += 1
            continue

        print(f"  [{i:3d}/{len(songs)}] {artist} — {title} ({meters_str})", end="", flush=True)

        segments = download_and_segment(video_id, stem, audio_dir, args.segment_length)

        if segments:
            print(f" — {len(segments)} segs")
            for seg_stem in segments:
                successful.append((seg_stem, meters))
            stats["ok"] += 1
        else:
            print(" — FAILED")
            stats["fail"] += 1

    # Write .tab
    write_tab_file(successful, tab_path)

    print(f"\nDone: {stats['ok']} downloaded, {stats['skip']} skipped, {stats['fail']} failed")
    print(f"Segments: {len(successful)} → {tab_path}")


if __name__ == "__main__":
    main()
