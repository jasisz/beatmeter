#!/usr/bin/env python3
"""Download WIKIMETER dataset from multiple source types.

Song catalog format (scripts/setup/wikimeter.json):
- artist/title/meters
- sources[] with prioritized entries, e.g. wikimedia + youtube fallback

Each song is split into segments (default 35s). Output is a .tab file
compatible with METER2800 parse_label_file().

DISCLAIMER: Audio downloaded by this script is intended solely for
non-commercial academic research purposes (fair use).
Only short excerpts are retained.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from wikimeter_tools import (
    DEFAULT_CATALOG,
    load_catalog,
    parse_youtube_video_id,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "wikimeter"
METER_LABELS = {3: "three", 4: "four", 5: "five", 7: "seven", 9: "nine", 11: "eleven"}
MAX_SEGMENTS = 5
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

Meters = dict[int, float]
Source = dict[str, Any]


def meters_to_list(meters: Meters) -> list[int]:
    return list(meters.keys())


def meters_to_tab_str(meters: Meters) -> str:
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
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return None


def source_label(source: Source) -> str:
    s_type = str(source.get("type", "")).strip().lower()
    if s_type == "youtube":
        video_id = str(source.get("video_id", "")).strip() or parse_youtube_video_id(str(source.get("url", "")))
        return f"youtube:{video_id}" if video_id else "youtube"
    url = str(source.get("url", "")).strip()
    host = urllib.parse.urlparse(url).netloc
    return f"{s_type}:{host}" if host else s_type


def download_from_youtube(source: Source, dest_dir: Path, timeout_s: int = 120) -> Path | None:
    video_id = str(source.get("video_id", "")).strip()
    url = str(source.get("url", "")).strip()
    if not video_id and url:
        video_id = parse_youtube_video_id(url)
    if not video_id and not url:
        return None
    if not url:
        url = f"https://www.youtube.com/watch?v={video_id}"

    tmp_path = dest_dir / "full.%(ext)s"
    cmd = [
        "yt-dlp",
        url,
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "5",
        "-o",
        str(tmp_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f" yt-dlp error: {stderr[:120]}", end="", flush=True)
            return None
    except subprocess.TimeoutExpired:
        print(" timeout", end="", flush=True)
        return None
    except FileNotFoundError:
        print(" ERROR: yt-dlp not found. Install: pip install yt-dlp")
        sys.exit(1)

    downloaded = list(dest_dir.glob("full.*"))
    return downloaded[0] if downloaded else None


def _safe_suffix_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = Path(path).name
    suffix = Path(name).suffix.lower()
    if suffix and 1 <= len(suffix) <= 8 and re.fullmatch(r"\.[a-z0-9]+", suffix):
        return suffix
    return ".bin"


def download_from_url(source: Source, dest_dir: Path, timeout_s: int = 120) -> Path | None:
    url = str(source.get("url", "")).strip()
    if not url:
        return None
    suffix = _safe_suffix_from_url(url)
    dest = dest_dir / f"full{suffix}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
    except Exception as exc:
        print(f" download error: {type(exc).__name__}", end="", flush=True)
        return None
    if not dest.exists() or dest.stat().st_size < 1000:
        return None
    return dest


def download_source_audio(source: Source, dest_dir: Path, timeout_s: int = 120) -> Path | None:
    s_type = str(source.get("type", "")).strip().lower()
    if s_type == "youtube":
        return download_from_youtube(source, dest_dir, timeout_s=timeout_s)
    if s_type in {"wikimedia", "archive", "url", "http", "https"}:
        return download_from_url(source, dest_dir, timeout_s=timeout_s)
    # Unknown source type with URL fallback.
    if source.get("url"):
        return download_from_url(source, dest_dir, timeout_s=timeout_s)
    return None


def segment_audio(src: Path, stem: str, audio_dir: Path, segment_length: int = 35) -> list[str]:
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

    min_gap = 5.0
    stride = segment_length + min_gap
    n_segments = min(MAX_SEGMENTS, max(1, int((usable_duration + min_gap) // stride)))

    if n_segments == 1:
        starts = [max(0.0, (usable_duration - segment_length) / 2)]
    elif n_segments * segment_length > usable_duration:
        total_needed = n_segments * segment_length
        center_start = max(0.0, (usable_duration - total_needed) / 2)
        starts = [center_start + i * segment_length for i in range(n_segments)]
    else:
        actual_stride = (usable_duration - segment_length) / (n_segments - 1)
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
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src),
                    "-ss",
                    f"{start_time:.1f}",
                    "-t",
                    f"{seg_duration:.1f}",
                    "-acodec",
                    "libmp3lame",
                    "-ab",
                    "192k",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(dest),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                segments.append(seg_stem)
        except subprocess.TimeoutExpired:
            continue

    return segments


def download_and_segment(
    sources: list[Source],
    stem: str,
    audio_dir: Path,
    segment_length: int = 35,
) -> tuple[list[str], Source | None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for source in sources:
            src_audio = download_source_audio(source, tmp)
            if src_audio is None:
                continue
            duration = get_duration(src_audio)
            if duration is None or duration < 15:
                continue
            segments = segment_audio(src_audio, stem, audio_dir, segment_length)
            if segments:
                return segments, source
    return [], None


def write_tab_file(entries: list[tuple[str, Meters]], tab_path: Path) -> None:
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
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_source_types_arg(s: str) -> set[str] | None:
    if not s.strip():
        return None
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def filter_song_sources(song: dict[str, Any], source_types: set[str] | None) -> list[Source]:
    sources = list(song.get("sources", []))
    if source_types:
        sources = [src for src in sources if str(src.get("type", "")).strip().lower() in source_types]
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WIKIMETER dataset")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Max songs (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--meters", type=str, default="", help="Filter meters: 3,4 or 5,7,9,11")
    parser.add_argument("--segment-length", type=int, default=35)
    parser.add_argument("--no-poly", action="store_true", help="Exclude polyrhythmic songs")
    parser.add_argument("--tab-only", action="store_true", help="Regenerate .tab only")
    parser.add_argument(
        "--source-types",
        type=str,
        default="",
        help="Allowed source types (comma separated), e.g. wikimedia,youtube",
    )
    args = parser.parse_args()

    catalog_path = args.catalog.resolve()
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found: {catalog_path}")
        sys.exit(1)

    catalog = load_catalog(catalog_path)
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    tab_path = output_dir / "data_wikimeter.tab"
    skip_existing = not args.no_skip_existing
    meter_filter = set(parse_meters_arg(args.meters)) if args.meters else None
    source_types = parse_source_types_arg(args.source_types)

    songs = list(catalog)
    if args.no_poly:
        songs = [s for s in songs if len(meters_to_list(s["meters"])) == 1]
    if meter_filter:
        songs = [s for s in songs if any(m in meter_filter for m in meters_to_list(s["meters"]))]
    if source_types:
        songs = [s for s in songs if filter_song_sources(s, source_types)]
    if args.limit > 0:
        songs = songs[: args.limit]

    meter_counts: dict[int, int] = {}
    for song in songs:
        for meter in meters_to_list(song["meters"]):
            meter_counts[meter] = meter_counts.get(meter, 0) + 1
    meter_summary = ", ".join(f"{meter_counts.get(m, 0)}×{m}/x" for m in sorted(meter_counts))
    print(f"WIKIMETER — {len(songs)} songs ({meter_summary})")

    if args.dry_run:
        for i, song in enumerate(songs, 1):
            meters = song["meters"]
            meter_list = meters_to_list(meters)
            meters_str = "+".join(str(m) for m in meter_list)
            soft = any(w != 1.0 for w in meters.values())
            tag = " [soft]" if soft else (" [poly]" if len(meter_list) > 1 else "")
            srcs = filter_song_sources(song, source_types)
            src_preview = ", ".join(source_label(s) for s in srcs[:2])
            print(f"  {i:3d}. [{meters_str}] {song['artist']} — {song['title']}{tag}  {src_preview}")
        return

    if args.tab_only:
        successful: list[tuple[str, Meters]] = []
        for song in songs:
            stem = sanitize_filename(song["artist"], song["title"])
            existing = sorted(f for f in audio_dir.glob(f"{stem}_seg*.mp3") if f.stat().st_size > 1000)
            for f in existing[:MAX_SEGMENTS]:
                successful.append((f.stem, song["meters"]))
        write_tab_file(successful, tab_path)
        print(f"  {len(successful)} segments → {tab_path}")
        return

    audio_dir.mkdir(parents=True, exist_ok=True)
    successful: list[tuple[str, Meters]] = []
    stats = {"ok": 0, "skip": 0, "fail": 0}

    for i, song in enumerate(songs, 1):
        artist, title = song["artist"], song["title"]
        meters = song["meters"]
        stem = sanitize_filename(artist, title)
        meters_str = "+".join(str(m) for m in meters_to_list(meters))

        existing = sorted(f for f in audio_dir.glob(f"{stem}_seg*.mp3") if f.stat().st_size > 1000)
        if skip_existing and existing:
            n = min(len(existing), MAX_SEGMENTS)
            for f in existing[:n]:
                successful.append((f.stem, meters))
            stats["skip"] += 1
            continue

        sources = filter_song_sources(song, source_types)
        if not sources:
            print(f"  [{i:3d}/{len(songs)}] {artist} — {title} ({meters_str}) — NO SOURCE")
            stats["fail"] += 1
            continue

        print(f"  [{i:3d}/{len(songs)}] {artist} — {title} ({meters_str})", end="", flush=True)
        segments, used_source = download_and_segment(
            sources=sources,
            stem=stem,
            audio_dir=audio_dir,
            segment_length=args.segment_length,
        )

        if segments:
            src_txt = source_label(used_source or {})
            print(f" — {len(segments)} segs [{src_txt}]")
            for seg_stem in segments:
                successful.append((seg_stem, meters))
            stats["ok"] += 1
        else:
            tried = ", ".join(source_label(s) for s in sources[:3])
            print(f" — FAILED (tried: {tried})")
            stats["fail"] += 1

    write_tab_file(successful, tab_path)

    print(f"\nDone: {stats['ok']} downloaded, {stats['skip']} skipped, {stats['fail']} failed")
    print(f"Segments: {len(successful)} → {tab_path}")


if __name__ == "__main__":
    main()
