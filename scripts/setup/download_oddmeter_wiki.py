#!/usr/bin/env python3
"""Download odd-meter audio from YouTube for training data augmentation.

Curated list of well-known songs in 5/x and 7/x time signatures, sourced from:
- Wikipedia: "List of musical works in unusual time signatures"
- Known prog rock, jazz, Balkan folk, and classical repertoire

Each song is split into multiple 30-second segments (matching METER2800 clip length),
maximizing the amount of training data from each download. A 5-minute song yields
~10 training clips.

Output is a .tab file compatible with METER2800 parse_label_file().

DISCLAIMER: Audio downloaded by this script is intended solely for non-commercial
academic research purposes (fair use). Only 30-second excerpts are retained.

Requirements:
    pip install yt-dlp
    brew install ffmpeg  (or apt-get install ffmpeg)

Usage:
    uv run python scripts/setup/download_oddmeter_wiki.py --dry-run
    uv run python scripts/setup/download_oddmeter_wiki.py --limit 3
    uv run python scripts/setup/download_oddmeter_wiki.py
    uv run python scripts/setup/download_oddmeter_wiki.py --no-skip-existing
    uv run python scripts/setup/download_oddmeter_wiki.py --segment-length 30
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated song list: (artist, title, meters, search_query_override)
#
# meters: list of ints — [5] for pure 5/4, [5, 7] for mixed 5+7, etc.
# Songs with len(meters) > 1 are "mixed" (polyrhythmic / meter-switching).
# search_query_override lets us specify a more precise YouTube search if needed
# ---------------------------------------------------------------------------

Song = tuple[str, str, list[int], str | None]  # artist, title, meters, query

SONGS: list[Song] = [
    # ===== PURE 5/4 and 5/8 =====
    # Jazz
    ("Dave Brubeck", "Take Five", [5], None),
    ("Paul Desmond", "Take Ten", [5], "Paul Desmond Take Ten"),
    ("Brubeck", "Three to Get Ready", [5], "Dave Brubeck Three to Get Ready"),
    ("Chick Corea", "Spain", [5], "Chick Corea Spain"),
    # Rock / Alternative
    ("Radiohead", "15 Step", [5], None),
    ("Radiohead", "Morning Bell", [5], None),
    ("Radiohead", "Everything in Its Right Place", [5], None),
    ("Radiohead", "Sail to the Moon", [5], None),
    ("Gorillaz", "5/4", [5], "Gorillaz 5/4"),
    ("Jethro Tull", "Living in the Past", [5], None),
    ("Sting", "Seven Days", [5], None),
    ("Nick Drake", "River Man", [5], None),
    ("The Mars Volta", "Inertiatic ESP", [5], None),
    ("Sufjan Stevens", "A Good Man Is Hard to Find", [5], None),
    ("Donovan", "Atlantis", [5], None),
    ("Soundgarden", "My Wave", [5], "Soundgarden My Wave"),
    ("Tame Impala", "Apocalypse Dreams", [5], None),
    # Film / TV
    ("Lalo Schifrin", "Mission Impossible Theme", [5], "Mission Impossible theme original"),
    # Classical
    ("Chopin", "Piano Sonata No 1 Larghetto", [5], "Chopin Piano Sonata No 1 Op 4 Larghetto 5/4"),
    ("Tchaikovsky", "Symphony No 6 Movement 2", [5], "Tchaikovsky Symphony 6 second movement 5/4"),
    # Prog / Metal
    ("Dream Theater", "The Mirror", [5], None),
    ("Animals as Leaders", "CAFO", [5], None),
    ("King Crimson", "Discipline", [5], "King Crimson Discipline"),
    ("Björk", "Army of Me", [5], None),
    ("Béla Fleck", "Sinister Minister", [5], "Bela Fleck Sinister Minister"),
    ("Mike Oldfield", "Tubular Bells Part 1", [5], "Mike Oldfield Tubular Bells opening 5/4"),
    ("Vulfpeck", "Dean Town", [5], None),
    ("Jacob Collier", "In My Room", [5], "Jacob Collier In My Room"),
    # Balkan / World
    ("Traditional", "Eleno Mome", [5], "Eleno Mome Bulgarian folk"),
    ("Traditional", "Paidushko Horo", [5], "Paidushko Horo Bulgarian folk"),

    # ===== PURE 7/4 and 7/8 =====
    # Rock / Alternative
    ("Pink Floyd", "Money", [7], None),
    ("Peter Gabriel", "Solsbury Hill", [7], None),
    ("Soundgarden", "Outshined", [7], "Soundgarden Outshined"),
    ("The Beatles", "All You Need Is Love", [7], "Beatles All You Need Is Love"),
    ("Radiohead", "2 + 2 = 5", [7], "Radiohead 2+2=5"),
    ("Rush", "Tom Sawyer", [7], "Rush Tom Sawyer"),
    ("Gentle Giant", "The Runaway", [7], "Gentle Giant The Runaway"),
    ("Alice in Chains", "Them Bones", [7], "Alice in Chains Them Bones"),
    ("King Crimson", "Frame by Frame", [7], "King Crimson Frame by Frame"),
    ("Opeth", "The Drapery Falls", [7], None),
    ("Gentle Giant", "Knots", [7], "Gentle Giant Knots"),
    ("Robert Fripp", "Exposure", [7], "Robert Fripp Exposure"),
    # Jazz
    ("Dave Holland", "Conference of the Birds", [7], "Dave Holland Conference of the Birds"),
    ("John McLaughlin", "Meeting of the Spirits", [7], "Mahavishnu Orchestra Meeting of the Spirits"),
    # Balkan / World
    ("Traditional", "Rachenitsa", [7], "Rachenitsa Bulgarian folk 7/8"),
    ("Traditional", "Makedonsko Devojche", [7], "Makedonsko Devojche folk 7/8"),
    ("Traditional", "Chetvorno Horo", [7], "Chetvorno Horo Bulgarian 7/8"),
    ("Traditional", "Lesnoto", [7], "Lesnoto Macedonian 7/8"),
    ("Traditional", "Ivailo", [7], "Ivailo Bulgarian folk 7/8"),
    ("Traditional", "Pravo Horo", [7], "Pravo Horo Bulgarian"),
    ("Goran Bregović", "Mesečina", [7], "Goran Bregovic Mesecina"),
    ("Fanfare Ciocărlia", "Born to Be Wild", [7], "Fanfare Ciocarlia Born to Be Wild"),
    # Classical
    ("Bernstein", "America from West Side Story", [7], "Bernstein America West Side Story"),
    # Film / TV
    ("Hans Zimmer", "Mombasa", [7], "Hans Zimmer Mombasa Inception"),
    ("Bear McCreary", "BSG Main Theme", [7], "Bear McCreary Battlestar Galactica theme"),
    # Pop / Other
    ("Aimee Mann", "Momentum", [7], None),
    ("Joni Mitchell", "The Silky Veils of Ardor", [7], None),
    ("Broken Social Scene", "7/4 Shoreline", [7], "Broken Social Scene 7/4 Shoreline"),
    ("Iron Maiden", "The Loneliness of the Long Distance Runner", [7], None),
    ("Led Zeppelin", "The Ocean", [7], "Led Zeppelin The Ocean"),
    ("Jeff Beck", "Led Boots", [7], None),
    ("Porcupine Tree", "The Sound of Muzak", [7], None),

    # ===== PURE 9/8 (aksak / additive rhythm: 2+2+2+3 or 3+3+3) =====
    ("Dave Brubeck", "Blue Rondo à la Turk", [9], "Dave Brubeck Blue Rondo a la Turk"),
    ("Traditional", "Daichovo Horo", [9], "Daichovo Horo Bulgarian 9/8"),
    ("Traditional", "Zeimbekiko", [9], "Zeimbekiko Greek dance 9/8"),
    ("Traditional", "Karsilama", [9], "Karsilama Turkish 9/8"),
    ("Traditional", "Arap", [9], "Arap Turkish 9/8 dance"),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 4", [9], "Bartok Mikrokosmos 151 Bulgarian Rhythm 4"),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 5", [9], "Bartok Mikrokosmos 152 Bulgarian Rhythm 5"),
    ("Toto", "Mushanga", [9], "Toto Mushanga"),
    ("Muse", "Butterflies and Hurricanes", [9], "Muse Butterflies and Hurricanes"),
    ("Mahler", "Symphony No 9 Rondo Burleske", [9], "Mahler Symphony 9 Rondo Burleske"),
    ("Stravinsky", "The Rite of Spring Sacrificial Dance", [9], "Stravinsky Rite of Spring Sacrificial Dance"),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 1", [9], "Bartok Bulgarian Rhythm No 1 Mikrokosmos"),

    # ===== PURE 11/8 (additive: 2+2+3+2+2 or similar) =====
    ("Traditional", "Gankino Horo", [11], "Gankino Horo Bulgarian 11/8"),
    ("Traditional", "Kopanitsa", [11], "Kopanitsa Bulgarian folk 11/8"),
    ("Traditional", "Ispayche", [11], "Ispayche Bulgarian 11/8"),
    ("Primus", "Eleven", [11], "Primus Eleven"),
    ("Grateful Dead", "The Eleven", [11], "Grateful Dead The Eleven"),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 6", [11], "Bartok Mikrokosmos 153 Bulgarian Rhythm 6"),
    ("Aksak Maboul", "Saure Gurke", [11], "Aksak Maboul Saure Gurke"),
    ("Frank Zappa", "Outside Now", [11], "Frank Zappa Outside Now"),

    # ===== POLYRHYTHMIC (simultaneous meters throughout the song) =====
    # Unlike polyrhythmic (which changes over time), these songs have
    # multiple meters playing AT THE SAME TIME. Every segment has the
    # same polyrhythmic character, so segmentation is safe.
    # Excluded by default (--include-poly to add).

    # 3-over-4 / 4-over-3 (hemiola, most common polyrhythm)
    ("Traditional", "Kpanlogo", [3, 4], "Kpanlogo Ghanaian drumming"),
    ("Traditional", "Agbekor", [3, 4], "Agbekor Ewe drumming Ghana"),
    ("Traditional", "Gahu", [3, 4], "Gahu drumming Ghana"),
    ("Traditional", "Rumba Guaguancó", [3, 4], "Rumba Guaguanco Cuban"),
    ("Traditional", "Bembe", [3, 4], "Bembe Cuban drumming 6/8 over 4/4"),
    ("Traditional", "Afoxé", [3, 4], "Afoxe Brazilian rhythm"),
    ("Fela Kuti", "Zombie", [3, 4], "Fela Kuti Zombie afrobeat"),
    ("Fela Kuti", "Water No Get Enemy", [3, 4], "Fela Kuti Water No Get Enemy"),
    ("Tony Allen", "Asiko", [3, 4], "Tony Allen Asiko afrobeat"),
    ("Babatunde Olatunji", "Jin-Go-Lo-Ba", [3, 4], "Babatunde Olatunji Jingo"),
    ("Talking Heads", "I Zimbra", [3, 4], "Talking Heads I Zimbra"),
    ("Vampire Weekend", "Cape Cod Kwassa Kwassa", [3, 4], None),
    # 3-over-2 / 2-over-3 (sesquialtera)
    ("Chopin", "Fantaisie-Impromptu", [3, 4], "Chopin Fantaisie Impromptu"),
    ("Debussy", "Clair de Lune", [3, 4], "Debussy Clair de Lune"),
    ("Brahms", "Piano Concerto No 1 Rondo", [3, 4], "Brahms Piano Concerto 1 Rondo"),
    # 5-over-4 (quintuplet feel)
    ("Meshuggah", "Bleed", [5, 4], "Meshuggah Bleed"),
    ("Meshuggah", "Rational Gaze", [5, 4], None),
    ("Meshuggah", "New Millennium Cyanide Christ", [7, 4], None),
    # Gamelan / Indonesian (multiple interlocking meters)
    ("Traditional", "Gamelan Gong Kebyar", [3, 4], "Gamelan Gong Kebyar Bali"),
    ("Traditional", "Gamelan Jegog", [3, 4], "Gamelan Jegog bamboo Bali"),
    # West African 12/8 + 4/4 (standard bell + drum polyrhythm)
    ("Traditional", "Djembe Dununba", [3, 4], "Dununba djembe rhythm West Africa"),
    ("Traditional", "Sinte", [3, 4], "Sinte djembe rhythm Mande"),
    ("Traditional", "Kuku", [3, 4], "Kuku djembe rhythm Guinea"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "oddmeter-wiki"
METER_LABELS = {3: "three", 4: "four", 5: "five", 7: "seven", 9: "nine", 11: "eleven"}
MAX_SEGMENTS = 25  # Cap per song: 25 × 30s = 12.5 min max
MAX_VIDEO_DURATION = 600  # Skip YouTube videos longer than 10 min


def sanitize_filename(artist: str, title: str) -> str:
    """Create a filesystem-safe filename from artist + title."""
    name = f"{artist}_{title}".lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80]


def get_duration(path: Path) -> float | None:
    """Get audio duration in seconds using ffprobe."""
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


def download_full_audio(query: str, dest_dir: Path, timeout: int = 120) -> Path | None:
    """Download full audio from YouTube via yt-dlp. Returns path or None."""
    tmp_path = dest_dir / "full.%(ext)s"

    cmd = [
        "yt-dlp",
        "--default-search", "ytsearch1",
        query,
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(tmp_path),
        "--no-playlist",
        "--max-downloads", "1",
        "--match-filter", f"duration < {MAX_VIDEO_DURATION}",
        "--quiet",
        "--no-warnings",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # yt-dlp returns 101 for --max-downloads limit reached — that's OK
        if result.returncode not in (0, 101):
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
    src: Path, stem: str, audio_dir: Path, segment_length: int = 30,
) -> list[str]:
    """Split audio into segments of segment_length seconds.

    Skips the first and last 10s (likely intro/outro silence or fade).
    Returns list of segment filename stems.

    A minimum segment length of 15s is enforced — shorter tails are dropped.
    """
    duration = get_duration(src)
    if duration is None or duration < 15:
        return []

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Skip first/last 10s for songs > 40s (avoid silence/fades)
    margin = 10.0 if duration > 40 else 0.0
    usable_start = margin
    usable_end = duration - margin
    usable_duration = usable_end - usable_start

    if usable_duration < 15:
        # Song too short after margins — use full duration, single segment
        usable_start = 0
        usable_duration = duration

    segments: list[str] = []
    seg_idx = 0

    offset = 0.0
    while offset + 15 <= usable_duration and seg_idx < MAX_SEGMENTS:
        seg_duration = min(segment_length, usable_duration - offset)
        start_time = usable_start + offset

        seg_stem = f"{stem}_seg{seg_idx:02d}"

        dest = audio_dir / f"{seg_stem}.mp3"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ss", f"{start_time:.1f}",
            "-t", f"{seg_duration:.1f}",
            "-acodec", "libmp3lame",
            "-ab", "192k",
            "-ar", "44100",
            "-ac", "1",
            str(dest),
        ]

        try:
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                segments.append(seg_stem)
        except subprocess.TimeoutExpired:
            pass

        offset += segment_length
        seg_idx += 1

    return segments


def download_and_segment(
    query: str, stem: str, audio_dir: Path,
    segment_length: int = 30, timeout: int = 120,
) -> list[str]:
    """Download audio from YouTube and split into segments.

    Returns list of segment filename stems (without .mp3 extension).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src = download_full_audio(query, Path(tmpdir), timeout)
        if src is None:
            print(" no file downloaded")
            return []

        duration = get_duration(src)
        if duration is None or duration < 15:
            print(f" too short ({duration}s)")
            return []

        return segment_audio(src, stem, audio_dir, segment_length)


def write_tab_file(
    entries: list[tuple[str, list[int]]], tab_path: Path,
) -> None:
    """Write a .tab file with multi-label support.

    entries: list of (filename_stem, meters_list)
    meter column: comma-separated for multi-label, e.g. "5,7"
    """
    tab_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tab_path, "w", encoding="utf-8") as f:
        f.write("filename\tlabel\tmeter\talt_meter\n")
        for stem, meters in entries:
            primary = meters[0]
            label = METER_LABELS.get(primary, str(primary))
            meter_str = ",".join(str(m) for m in meters)
            alt = primary * 2
            f.write(f'"/WIKI/{stem}.mp3"\t"{label}"\t{meter_str}\t{alt}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download odd-meter songs from YouTube for training data augmentation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of songs to download (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-download even if file already exists",
    )
    parser.add_argument(
        "--meter", type=int, default=0,
        help="Only download songs with this primary meter (0 = all)",
    )
    parser.add_argument(
        "--segment-length", type=int, default=30,
        help="Segment length in seconds (default: 30, matching METER2800)",
    )
    parser.add_argument(
        "--include-poly", action="store_true",
        help="Include polyrhythmic songs with multi-label (e.g. 3+4 hemiola)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    tab_path = output_dir / "data_oddmeter_wiki.tab"
    skip_existing = not args.no_skip_existing

    # Filter songs
    songs = SONGS
    if not args.include_poly:
        songs = [s for s in songs if len(s[2]) == 1]  # pure = single meter
    if args.meter:
        songs = [s for s in songs if args.meter in s[2]]
    if args.limit > 0:
        songs = songs[:args.limit]

    n_pure = sum(1 for s in songs if len(s[2]) == 1)
    n_mixed = sum(1 for s in songs if len(s[2]) > 1)
    meter_counts: dict[int, int] = {}
    for s in songs:
        for m in s[2]:
            meter_counts[m] = meter_counts.get(m, 0) + 1

    print(f"{'=' * 70}")
    print(f"  ODDMETER-WIKI Dataset Downloader")
    print(f"{'=' * 70}")
    print(f"  Output:    {output_dir}")
    meter_summary = ", ".join(f"{meter_counts.get(m, 0)} × {m}/x" for m in sorted(meter_counts))
    print(f"  Songs:     {len(songs)} total ({meter_summary})")
    if n_mixed:
        print(f"             {n_pure} pure + {n_mixed} polyrhythmic")
    print(f"  Segment:   {args.segment_length}s (songs split into multiple clips)")
    print(f"  Skip existing: {skip_existing}")
    print()

    if args.dry_run:
        print("  DRY RUN — no files will be downloaded\n")
        for i, (artist, title, meters, query) in enumerate(songs, 1):
            stem = sanitize_filename(artist, title)
            search = query or f"{artist} {title}"
            meters_str = "+".join(str(m) for m in meters)
            tag = " [poly]" if len(meters) > 1 else ""
            print(f"  {i:3d}. [{meters_str}] {artist} — {title}{tag}")
            print(f"       → {stem}.mp3")
            print(f"       search: {search}")
        print(f"\n  Total: {len(songs)} songs")
        return

    # Download
    audio_dir.mkdir(parents=True, exist_ok=True)
    successful: list[tuple[str, list[int]]] = []  # (segment_stem, meters)
    songs_downloaded = 0
    songs_skipped = 0
    songs_failed = 0

    for i, (artist, title, meters, query_override) in enumerate(songs, 1):
        stem = sanitize_filename(artist, title)
        search_query = query_override or f"{artist} {title}"
        meters_str = "+".join(str(m) for m in meters)

        status = f"  [{i:3d}/{len(songs)}] {artist} — {title} ({meters_str})"

        # Check if first segment already exists (means song was already processed)
        first_seg = audio_dir / f"{stem}.mp3"
        if skip_existing and first_seg.exists() and first_seg.stat().st_size > 1000:
            # Count all existing segments for this song
            existing = [f for f in audio_dir.glob(f"{stem}*.mp3")
                        if f.stem == stem or f.stem.startswith(f"{stem}_seg")]
            for f in sorted(existing):
                successful.append((f.stem, meters))
            print(f"{status} — skipped ({len(existing)} segments exist)")
            songs_skipped += 1
            continue

        print(f"{status}", end="", flush=True)
        segments = download_and_segment(
            search_query, stem, audio_dir,
            segment_length=args.segment_length,
        )

        if segments:
            print(f" — OK ({len(segments)} segments)")
            for seg_stem in segments:
                successful.append((seg_stem, meters))
            songs_downloaded += 1
        else:
            print(f" — FAILED")
            songs_failed += 1

    # Write .tab file
    write_tab_file(successful, tab_path)

    # Summary
    seg_counts: dict[int, int] = {}
    for _, meters in successful:
        for m in meters:
            seg_counts[m] = seg_counts.get(m, 0) + 1
    seg_summary = ", ".join(f"{seg_counts.get(m, 0)} × {m}/x" for m in sorted(seg_counts))

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Songs downloaded: {songs_downloaded}")
    print(f"  Songs skipped:    {songs_skipped}")
    print(f"  Songs failed:     {songs_failed}")
    print(f"  Total segments:   {len(successful)} ({seg_summary})")
    print(f"  Tab file:         {tab_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
