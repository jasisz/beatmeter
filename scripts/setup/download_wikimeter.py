#!/usr/bin/env python3
"""Download curated audio from YouTube for meter classification training.

WIKIMETER dataset: songs with well-known, Wikipedia-verified time signatures
across all meter classes (3/4, 4/4, 5/x, 7/x, 9/x, 11/x, polyrhythmic).

Each song is split into multiple 30-second segments (matching METER2800 clip length).

Output is a .tab file compatible with METER2800 parse_label_file().

DISCLAIMER: Audio downloaded by this script is intended solely for non-commercial
academic research purposes (fair use). Only 30-second excerpts are retained.

Requirements:
    pip install yt-dlp
    brew install ffmpeg  (or apt-get install ffmpeg)

Usage:
    uv run python scripts/setup/download_wikimeter.py --dry-run
    uv run python scripts/setup/download_wikimeter.py --limit 3
    uv run python scripts/setup/download_wikimeter.py
    uv run python scripts/setup/download_wikimeter.py --meters 3,4
    uv run python scripts/setup/download_wikimeter.py --meters 5,7,9,11
    uv run python scripts/setup/download_wikimeter.py --include-poly
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated song list: (artist, title, meters, search_query_override, expected_duration_s)
#
# meters: list of ints — [5] for pure 5/4, [3, 4] for polyrhythmic, etc.
# search_query_override: more precise YouTube search if needed
# expected_duration_s: approximate duration in seconds (None = unknown).
#   When set, yt-dlp results outside 0.5x–2.0x range are rejected
#   (catches albums/compilations that slip through).
# ---------------------------------------------------------------------------

Song = tuple[str, str, list[int], str | None, int | None]

SONGS: list[Song] = [
    # =====================================================================
    #  3/4 — Waltzes, folk, classical
    # =====================================================================
    # Classical waltzes (unambiguous 3/4)
    ("Strauss II", "The Blue Danube", [3], "Johann Strauss Blue Danube waltz", 600),
    ("Strauss II", "Tales from the Vienna Woods", [3], "Strauss Tales Vienna Woods waltz", 720),
    ("Strauss II", "Emperor Waltz", [3], "Strauss Emperor Waltz Kaiserwalzer", 660),
    ("Chopin", "Waltz in C-sharp minor Op 64 No 2", [3], "Chopin Waltz Op 64 No 2", 210),
    ("Chopin", "Minute Waltz", [3], "Chopin Minute Waltz Op 64 No 1", 120),
    ("Chopin", "Grande Valse Brillante", [3], "Chopin Grande Valse Brillante Op 18", 300),
    ("Chopin", "Waltz in A minor", [3], "Chopin Waltz A minor B 150", 180),
    ("Tchaikovsky", "Waltz of the Flowers", [3], "Tchaikovsky Waltz of the Flowers Nutcracker", 420),
    ("Tchaikovsky", "Sleeping Beauty Waltz", [3], "Tchaikovsky Sleeping Beauty Waltz", 360),
    ("Tchaikovsky", "Swan Lake Waltz", [3], "Tchaikovsky Swan Lake Waltz", 390),
    ("Shostakovich", "Waltz No 2", [3], "Shostakovich Waltz No 2", 210),
    ("Khachaturian", "Masquerade Waltz", [3], "Khachaturian Masquerade Waltz", 270),
    ("Ravel", "La Valse", [3], "Ravel La Valse orchestral", 780),
    ("Brahms", "Waltz in A-flat major Op 39 No 15", [3], "Brahms Waltz Op 39 No 15", 120),
    # Pop / Rock in 3/4
    ("The Beatles", "Norwegian Wood", [3], None, 125),
    ("Leonard Cohen", "Take This Waltz", [3], None, 420),
    ("Norah Jones", "Come Away With Me", [3], None, 198),
    ("Jeff Buckley", "Lilac Wine", [3], None, 280),
    ("Radiohead", "Codex", [3], None, 280),
    ("Elliott Smith", "Waltz No 2", [3], "Elliott Smith Waltz No 2 XO", 270),
    ("Damien Rice", "The Blower's Daughter", [3], "Damien Rice Blowers Daughter", 260),
    ("Mazzy Star", "Fade Into You", [3], None, 290),
    ("R.E.M.", "Everybody Hurts", [3], None, 318),
    ("Counting Crows", "A Long December", [3], None, 330),
    # Folk / Traditional 3/4
    ("Traditional", "Greensleeves", [3], "Greensleeves traditional", 240),
    ("Traditional", "Scarborough Fair", [3], "Scarborough Fair traditional", 210),
    ("Traditional", "Danny Boy", [3], "Danny Boy traditional Irish", 240),
    ("Traditional", "Amazing Grace", [3], "Amazing Grace traditional", 300),
    ("Traditional", "Edelweiss", [3], "Edelweiss Sound of Music", 150),

    # =====================================================================
    #  4/4 — Rock, pop, electronic (unambiguous straight time)
    # =====================================================================
    # Classic Rock
    ("Queen", "We Will Rock You", [4], None, 122),
    ("Queen", "Another One Bites the Dust", [4], None, 215),
    ("AC/DC", "Back in Black", [4], None, 255),
    ("Deep Purple", "Smoke on the Water", [4], None, 340),
    ("Led Zeppelin", "Whole Lotta Love", [4], None, 333),
    ("The Rolling Stones", "Satisfaction", [4], "Rolling Stones Satisfaction", 224),
    ("Nirvana", "Smells Like Teen Spirit", [4], None, 301),
    ("Metallica", "Enter Sandman", [4], None, 331),
    ("Guns N Roses", "Sweet Child O Mine", [4], "Guns N Roses Sweet Child O Mine", 356),
    ("The White Stripes", "Seven Nation Army", [4], None, 232),
    # Pop
    ("Michael Jackson", "Billie Jean", [4], None, 294),
    ("Michael Jackson", "Beat It", [4], None, 258),
    ("Stevie Wonder", "Superstition", [4], None, 245),
    ("Bee Gees", "Stayin Alive", [4], "Bee Gees Stayin Alive", 285),
    ("ABBA", "Dancing Queen", [4], None, 231),
    ("Daft Punk", "Get Lucky", [4], None, 369),
    ("Daft Punk", "Around the World", [4], None, 427),
    ("Bruno Mars", "Uptown Funk", [4], None, 270),
    ("Pharrell Williams", "Happy", [4], None, 233),
    ("Marvin Gaye", "Aint No Mountain High Enough", [4], "Marvin Gaye Aint No Mountain High Enough", 150),
    # Electronic
    ("Kraftwerk", "The Model", [4], None, 220),
    ("Kraftwerk", "Autobahn", [4], "Kraftwerk Autobahn", 660),
    ("New Order", "Blue Monday", [4], None, 442),
    ("Depeche Mode", "Personal Jesus", [4], None, 295),
    ("The Prodigy", "Firestarter", [4], None, 280),
    # Hip-hop / Funk (straight 4/4)
    ("James Brown", "I Got You", [4], "James Brown I Got You I Feel Good", 167),
    ("Parliament", "Give Up the Funk", [4], "Parliament Give Up the Funk", 348),
    ("Grandmaster Flash", "The Message", [4], None, 445),
    ("A Tribe Called Quest", "Can I Kick It", [4], None, 260),

    # =====================================================================
    #  5/4 and 5/8
    # =====================================================================
    # Jazz
    ("Dave Brubeck", "Take Five", [5], None, 325),
    ("Paul Desmond", "Take Ten", [5], "Paul Desmond Take Ten", 340),
    ("Brubeck", "Three to Get Ready", [5], "Dave Brubeck Three to Get Ready", 340),
    ("Chick Corea", "Spain", [5], "Chick Corea Spain", 600),
    # Rock / Alternative
    ("Radiohead", "15 Step", [5], None, 237),
    ("Radiohead", "Morning Bell", [5], None, 270),
    ("Radiohead", "Everything in Its Right Place", [5], None, 250),
    ("Radiohead", "Sail to the Moon", [5], None, 290),
    ("Gorillaz", "5/4", [5], "Gorillaz 5/4", 241),
    ("Jethro Tull", "Living in the Past", [5], None, 205),
    ("Sting", "Seven Days", [5], None, 290),
    ("Nick Drake", "River Man", [5], None, 275),
    ("The Mars Volta", "Inertiatic ESP", [5], None, 230),
    ("Sufjan Stevens", "A Good Man Is Hard to Find", [5], None, 300),
    ("Donovan", "Atlantis", [5], None, 295),
    ("Soundgarden", "My Wave", [5], "Soundgarden My Wave", 312),
    ("Tame Impala", "Apocalypse Dreams", [5], None, 390),
    # Film / TV
    ("Lalo Schifrin", "Mission Impossible Theme", [5], "Mission Impossible theme original", 180),
    # Classical
    ("Chopin", "Piano Sonata No 1 Larghetto", [5], "Chopin Piano Sonata No 1 Op 4 Larghetto 5/4", 480),
    ("Tchaikovsky", "Symphony No 6 Movement 2", [5], "Tchaikovsky Symphony 6 second movement 5/4", 510),
    # Prog / Metal
    ("Dream Theater", "The Mirror", [5], None, 660),
    ("Animals as Leaders", "CAFO", [5], None, 370),
    ("King Crimson", "Discipline", [5], "King Crimson Discipline", 305),
    ("Björk", "Army of Me", [5], None, 224),
    ("Béla Fleck", "Sinister Minister", [5], "Bela Fleck Sinister Minister", 360),
    ("Mike Oldfield", "Tubular Bells Part 1", [5], "Mike Oldfield Tubular Bells opening 5/4", 600),
    ("Vulfpeck", "Dean Town", [5], None, 210),
    ("Jacob Collier", "In My Room", [5], "Jacob Collier In My Room", 300),
    # Balkan / World
    ("Traditional", "Eleno Mome", [5], "Eleno Mome Bulgarian folk", 240),
    ("Traditional", "Paidushko Horo", [5], "Paidushko Horo Bulgarian folk", 240),

    # =====================================================================
    #  7/4 and 7/8
    # =====================================================================
    # Rock / Alternative
    ("Pink Floyd", "Money", [7], None, 382),
    ("Peter Gabriel", "Solsbury Hill", [7], None, 260),
    ("Soundgarden", "Outshined", [7], "Soundgarden Outshined", 312),
    ("The Beatles", "All You Need Is Love", [7], "Beatles All You Need Is Love", 237),
    ("Radiohead", "2 + 2 = 5", [7], "Radiohead 2+2=5", 202),
    ("Rush", "Tom Sawyer", [7], "Rush Tom Sawyer", 276),
    ("Gentle Giant", "The Runaway", [7], "Gentle Giant The Runaway", 300),
    ("Alice in Chains", "Them Bones", [7], "Alice in Chains Them Bones", 147),
    ("King Crimson", "Frame by Frame", [7], "King Crimson Frame by Frame", 310),
    ("Opeth", "The Drapery Falls", [7], None, 630),
    ("Gentle Giant", "Knots", [7], "Gentle Giant Knots", 300),
    ("Robert Fripp", "Exposure", [7], "Robert Fripp Exposure", 260),
    # Jazz
    ("Dave Holland", "Conference of the Birds", [7], "Dave Holland Conference of the Birds", 480),
    ("John McLaughlin", "Meeting of the Spirits", [7], "Mahavishnu Orchestra Meeting of the Spirits", 420),
    # Balkan / World
    ("Traditional", "Rachenitsa", [7], "Rachenitsa Bulgarian folk 7/8", 240),
    ("Traditional", "Makedonsko Devojche", [7], "Makedonsko Devojche folk 7/8", 240),
    ("Traditional", "Chetvorno Horo", [7], "Chetvorno Horo Bulgarian 7/8", 240),
    ("Traditional", "Lesnoto", [7], "Lesnoto Macedonian 7/8", 240),
    ("Traditional", "Ivailo", [7], "Ivailo Bulgarian folk 7/8", 240),
    ("Traditional", "Pravo Horo", [7], "Pravo Horo Bulgarian", 240),
    ("Goran Bregović", "Mesečina", [7], "Goran Bregovic Mesecina", 240),
    ("Fanfare Ciocărlia", "Born to Be Wild", [7], "Fanfare Ciocarlia Born to Be Wild", 270),
    # Classical
    ("Bernstein", "America from West Side Story", [7], "Bernstein America West Side Story", 300),
    # Film / TV
    ("Hans Zimmer", "Mombasa", [7], "Hans Zimmer Mombasa Inception", 295),
    ("Bear McCreary", "BSG Main Theme", [7], "Bear McCreary Battlestar Galactica theme", 240),
    # Pop / Other
    ("Aimee Mann", "Momentum", [7], None, 260),
    ("Joni Mitchell", "The Silky Veils of Ardor", [7], None, 240),
    ("Broken Social Scene", "7/4 Shoreline", [7], "Broken Social Scene 7/4 Shoreline", 260),
    ("Iron Maiden", "The Loneliness of the Long Distance Runner", [7], None, 390),
    ("Led Zeppelin", "The Ocean", [7], "Led Zeppelin The Ocean", 266),
    ("Jeff Beck", "Led Boots", [7], None, 340),
    ("Porcupine Tree", "The Sound of Muzak", [7], None, 290),

    # =====================================================================
    #  9/8 (aksak / additive: 2+2+2+3 or 3+3+3)
    # =====================================================================
    ("Dave Brubeck", "Blue Rondo à la Turk", [9], "Dave Brubeck Blue Rondo a la Turk", 402),
    ("Traditional", "Daichovo Horo", [9], "Daichovo Horo Bulgarian 9/8", 240),
    ("Traditional", "Zeimbekiko", [9], "Zeimbekiko Greek dance 9/8", 300),
    ("Traditional", "Karsilama", [9], "Karsilama Turkish 9/8", 240),
    ("Traditional", "Arap", [9], "Arap Turkish 9/8 dance", 240),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 4", [9], "Bartok Mikrokosmos 151 Bulgarian Rhythm 4", 90),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 5", [9], "Bartok Mikrokosmos 152 Bulgarian Rhythm 5", 90),
    ("Toto", "Mushanga", [9], "Toto Mushanga", 360),
    ("Muse", "Butterflies and Hurricanes", [9], "Muse Butterflies and Hurricanes", 330),
    ("Mahler", "Symphony No 9 Rondo Burleske", [9], "Mahler Symphony 9 Rondo Burleske", 780),
    ("Stravinsky", "The Rite of Spring Sacrificial Dance", [9], "Stravinsky Rite of Spring Sacrificial Dance", 300),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 1", [9], "Bartok Bulgarian Rhythm No 1 Mikrokosmos", 90),

    # =====================================================================
    #  11/8 (additive: 2+2+3+2+2 or similar)
    # =====================================================================
    ("Traditional", "Gankino Horo", [11], "Gankino Horo Bulgarian 11/8", 240),
    ("Traditional", "Kopanitsa", [11], "Kopanitsa Bulgarian folk 11/8", 240),
    ("Traditional", "Ispayche", [11], "Ispayche Bulgarian 11/8", 240),
    ("Primus", "Eleven", [11], "Primus Eleven", 330),
    ("Grateful Dead", "The Eleven", [11], "Grateful Dead The Eleven", 480),
    ("Bartók", "Six Dances in Bulgarian Rhythm No 6", [11], "Bartok Mikrokosmos 153 Bulgarian Rhythm 6", 120),
    ("Aksak Maboul", "Saure Gurke", [11], "Aksak Maboul Saure Gurke", 300),
    ("Frank Zappa", "Outside Now", [11], "Frank Zappa Outside Now", 340),

    # =====================================================================
    #  POLYRHYTHMIC (simultaneous meters)
    #  Excluded by default (--include-poly to add).
    # =====================================================================
    # 3-over-4 / hemiola
    ("Traditional", "Kpanlogo", [3, 4], "Kpanlogo Ghanaian drumming", 300),
    ("Traditional", "Agbekor", [3, 4], "Agbekor Ewe drumming Ghana", 300),
    ("Traditional", "Gahu", [3, 4], "Gahu drumming Ghana", 300),
    ("Traditional", "Rumba Guaguancó", [3, 4], "Rumba Guaguanco Cuban", 300),
    ("Traditional", "Bembe", [3, 4], "Bembe Cuban drumming 6/8 over 4/4", 300),
    ("Traditional", "Afoxé", [3, 4], "Afoxe Brazilian rhythm", 300),
    ("Fela Kuti", "Zombie", [3, 4], "Fela Kuti Zombie afrobeat", 745),
    ("Fela Kuti", "Water No Get Enemy", [3, 4], "Fela Kuti Water No Get Enemy", 600),
    ("Tony Allen", "Asiko", [3, 4], "Tony Allen Asiko afrobeat", 360),
    ("Babatunde Olatunji", "Jin-Go-Lo-Ba", [3, 4], "Babatunde Olatunji Jingo", 300),
    ("Talking Heads", "I Zimbra", [3, 4], "Talking Heads I Zimbra", 195),
    ("Vampire Weekend", "Cape Cod Kwassa Kwassa", [3, 4], None, 230),
    # Sesquialtera (3-over-2)
    ("Chopin", "Fantaisie-Impromptu", [3, 4], "Chopin Fantaisie Impromptu", 300),
    ("Debussy", "Clair de Lune", [3, 4], "Debussy Clair de Lune", 330),
    ("Brahms", "Piano Concerto No 1 Rondo", [3, 4], "Brahms Piano Concerto 1 Rondo", 600),
    # 5-over-4
    ("Meshuggah", "Bleed", [5, 4], "Meshuggah Bleed", 445),
    ("Meshuggah", "Rational Gaze", [5, 4], None, 330),
    ("Meshuggah", "New Millennium Cyanide Christ", [7, 4], None, 360),
    # Gamelan
    ("Traditional", "Gamelan Gong Kebyar", [3, 4], "Gamelan Gong Kebyar Bali", 420),
    ("Traditional", "Gamelan Jegog", [3, 4], "Gamelan Jegog bamboo Bali", 360),
    # West African 12/8 + 4/4
    ("Traditional", "Djembe Dununba", [3, 4], "Dununba djembe rhythm West Africa", 300),
    ("Traditional", "Sinte", [3, 4], "Sinte djembe rhythm Mande", 300),
    ("Traditional", "Kuku", [3, 4], "Kuku djembe rhythm Guinea", 300),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "wikimeter"
METER_LABELS = {3: "three", 4: "four", 5: "five", 7: "seven", 9: "nine", 11: "eleven"}
MAX_SEGMENTS = 5  # Cap per song: 5 × 30s = 2.5 min max (fair use)
DURATION_TOLERANCE = 2.0  # Accept videos between 0.5x and 2.0x expected duration
MAX_VIDEO_DURATION_FALLBACK = 900  # Fallback for songs without expected_duration (15 min)


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


def resolve_video_id(
    query: str, expected_duration: int | None = None, timeout: int = 30,
) -> str | None:
    """Resolve a search query to a YouTube video ID without downloading."""
    if expected_duration:
        min_dur = int(expected_duration / DURATION_TOLERANCE)
        max_dur = int(expected_duration * DURATION_TOLERANCE)
        duration_filter = f"duration > {min_dur} & duration < {max_dur}"
    else:
        duration_filter = f"duration < {MAX_VIDEO_DURATION_FALLBACK}"

    cmd = [
        "yt-dlp",
        "--default-search", "ytsearch1",
        query,
        "--print", "id",
        "--no-download",
        "--no-playlist",
        "--max-downloads", "1",
        "--match-filter", duration_filter,
        "--quiet",
        "--no-warnings",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        vid_id = result.stdout.strip().split("\n")[0].strip()
        return vid_id if vid_id else None
    except Exception:
        return None


def download_full_audio(
    query: str, dest_dir: Path, timeout: int = 120,
    expected_duration: int | None = None,
    video_id: str | None = None,
) -> tuple[Path | None, str | None]:
    """Download full audio from YouTube via yt-dlp.

    If video_id is provided, downloads that specific video (reproducible).
    Otherwise, searches YouTube with the query and captures the video ID.

    Returns (path, video_id) or (None, None).
    """
    # Resolve video ID first if not known
    if not video_id:
        video_id = resolve_video_id(query, expected_duration)
        if not video_id:
            return None, None

    tmp_path = dest_dir / "full.%(ext)s"
    source = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        source,
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(tmp_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f" yt-dlp error: {stderr[:100]}")
            return None, None
    except subprocess.TimeoutExpired:
        print(" timeout")
        return None, None
    except FileNotFoundError:
        print(" ERROR: yt-dlp not found. Install: pip install yt-dlp")
        sys.exit(1)

    downloaded = list(dest_dir.glob("full.*"))
    if downloaded:
        return downloaded[0], video_id
    return None, None


def segment_audio(
    src: Path, stem: str, audio_dir: Path, segment_length: int = 30,
) -> list[str]:
    """Split audio into non-contiguous segments evenly spaced across the track.

    Segments are spread across the usable portion (excluding 10s margins)
    so they represent independent excerpts, not one continuous chunk.
    Short songs naturally yield fewer segments.
    """
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

    # How many segments fit, capped by MAX_SEGMENTS
    n_segments = min(MAX_SEGMENTS, max(1, int(usable_duration // segment_length)))

    # Evenly space segment start positions across usable duration
    if n_segments == 1:
        # Single segment from the middle
        starts = [(usable_duration - segment_length) / 2] if usable_duration >= segment_length else [0.0]
    else:
        stride = usable_duration / n_segments
        starts = [i * stride for i in range(n_segments)]

    segments: list[str] = []

    for seg_idx, offset in enumerate(starts):
        seg_duration = min(segment_length, usable_duration - offset)
        if seg_duration < 15:
            continue
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

    return segments


def download_and_segment(
    query: str, stem: str, audio_dir: Path,
    segment_length: int = 30, timeout: int = 120,
    expected_duration: int | None = None,
    video_id: str | None = None,
) -> tuple[list[str], str | None]:
    """Download audio from YouTube and split into segments.

    Returns (list of segment filename stems, video_id).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        src, vid_id = download_full_audio(
            query, Path(tmpdir), timeout, expected_duration, video_id,
        )
        if src is None:
            print(" no file downloaded")
            return [], None

        duration = get_duration(src)
        if duration is None or duration < 15:
            print(f" too short ({duration}s)")
            return [], None

        return segment_audio(src, stem, audio_dir, segment_length), vid_id


def write_tab_file(
    entries: list[tuple[str, list[int]]], tab_path: Path,
) -> None:
    """Write a .tab file with multi-label support."""
    tab_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tab_path, "w", encoding="utf-8") as f:
        f.write("filename\tlabel\tmeter\talt_meter\n")
        for stem, meters in entries:
            primary = meters[0]
            label = METER_LABELS.get(primary, str(primary))
            meter_str = ",".join(str(m) for m in meters)
            alt = primary * 2
            f.write(f'"/WIKIMETER/{stem}.mp3"\t"{label}"\t{meter_str}\t{alt}\n')


def parse_meters_arg(s: str) -> list[int]:
    """Parse comma-separated meter list, e.g. '3,4' -> [3, 4]."""
    return [int(x.strip()) for x in s.split(",")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download curated songs from YouTube for meter classification training"
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
        "--meters", type=str, default="",
        help="Only download songs with these meters, comma-separated (e.g. '3,4' or '5,7,9,11')",
    )
    parser.add_argument(
        "--segment-length", type=int, default=30,
        help="Segment length in seconds (default: 30)",
    )
    parser.add_argument(
        "--include-poly", action="store_true",
        help="Include polyrhythmic songs with multi-label",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    tab_path = output_dir / "data_wikimeter.tab"
    skip_existing = not args.no_skip_existing
    meter_filter = set(parse_meters_arg(args.meters)) if args.meters else None

    # Filter songs
    songs: list[Song] = list(SONGS)
    if not args.include_poly:
        songs = [s for s in songs if len(s[2]) == 1]
    if meter_filter:
        songs = [s for s in songs if any(m in meter_filter for m in s[2])]
    if args.limit > 0:
        songs = songs[:args.limit]

    meter_counts: dict[int, int] = {}
    for s in songs:
        for m in s[2]:
            meter_counts[m] = meter_counts.get(m, 0) + 1

    n_pure = sum(1 for s in songs if len(s[2]) == 1)
    n_mixed = sum(1 for s in songs if len(s[2]) > 1)
    n_with_duration = sum(1 for s in songs if s[4] is not None)

    print(f"{'=' * 70}")
    print(f"  WIKIMETER Dataset Downloader")
    print(f"{'=' * 70}")
    print(f"  Output:    {output_dir}")
    meter_summary = ", ".join(f"{meter_counts.get(m, 0)}×{m}/x" for m in sorted(meter_counts))
    print(f"  Songs:     {len(songs)} ({meter_summary})")
    if n_mixed:
        print(f"             {n_pure} pure + {n_mixed} polyrhythmic")
    print(f"  Duration filter: {n_with_duration}/{len(songs)} songs with expected duration")
    print(f"  Segment:   {args.segment_length}s")
    print(f"  Skip existing: {skip_existing}")
    print()

    if args.dry_run:
        print("  DRY RUN — no files will be downloaded\n")
        for i, (artist, title, meters, query, exp_dur) in enumerate(songs, 1):
            stem = sanitize_filename(artist, title)
            search = query or f"{artist} {title}"
            meters_str = "+".join(str(m) for m in meters)
            tag = " [poly]" if len(meters) > 1 else ""
            dur_str = f" ~{exp_dur}s" if exp_dur else ""
            print(f"  {i:3d}. [{meters_str}] {artist} — {title}{tag}{dur_str}")
            print(f"       → {stem}.mp3  |  search: {search}")
        print(f"\n  Total: {len(songs)} songs")
        return

    # Load known video IDs for reproducibility
    video_ids_path = output_dir / "video_ids.json"
    video_ids: dict[str, dict] = {}
    if video_ids_path.exists():
        with open(video_ids_path) as f:
            video_ids = json.load(f)
        print(f"  Loaded {len(video_ids)} known video IDs from {video_ids_path.name}\n")

    # Download
    audio_dir.mkdir(parents=True, exist_ok=True)
    successful: list[tuple[str, list[int]]] = []
    songs_downloaded = 0
    songs_skipped = 0
    songs_failed = 0

    for i, (artist, title, meters, query_override, exp_dur) in enumerate(songs, 1):
        stem = sanitize_filename(artist, title)
        search_query = query_override or f"{artist} {title}"
        meters_str = "+".join(str(m) for m in meters)

        status = f"  [{i:3d}/{len(songs)}] {artist} — {title} ({meters_str})"

        # Check existing segments
        existing = sorted(f for f in audio_dir.glob(f"{stem}_seg*.mp3")
                          if f.stat().st_size > 1000)
        if skip_existing and existing:
            n = min(len(existing), MAX_SEGMENTS)
            for f in existing[:n]:
                successful.append((f.stem, meters))
            print(f"{status} — skipped ({n} segs)")
            songs_skipped += 1
            continue

        # Use known video ID if available (exact reproducibility)
        known_vid = video_ids.get(stem, {}).get("video_id")
        if known_vid:
            print(f"{status} [ID: {known_vid}]", end="", flush=True)
        else:
            print(f"{status}", end="", flush=True)

        segments, vid_id = download_and_segment(
            search_query, stem, audio_dir,
            segment_length=args.segment_length,
            expected_duration=exp_dur,
            video_id=known_vid,
        )

        if segments:
            print(f" — OK ({len(segments)} segs)")
            for seg_stem in segments:
                successful.append((seg_stem, meters))
            songs_downloaded += 1

            # Save video ID for reproducibility
            if vid_id:
                video_ids[stem] = {
                    "video_id": vid_id,
                    "artist": artist,
                    "title": title,
                    "meters": meters,
                    "query": search_query,
                }
                with open(video_ids_path, "w") as f:
                    json.dump(video_ids, f, indent=2, ensure_ascii=False)
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
    seg_summary = ", ".join(f"{seg_counts.get(m, 0)}×{m}/x" for m in sorted(seg_counts))

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Songs downloaded: {songs_downloaded}")
    print(f"  Songs skipped:    {songs_skipped}")
    print(f"  Songs failed:     {songs_failed}")
    print(f"  Total segments:   {len(successful)} ({seg_summary})")
    print(f"  Tab file:         {tab_path}")
    print(f"  Video IDs:        {video_ids_path} ({len(video_ids)} entries)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
