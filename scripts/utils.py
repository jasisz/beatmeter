"""Shared utilities for scripts/ — audio file handling, hashing, downloads."""

import hashlib
import time
import urllib.request
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".oga", ".opus", ".aiff", ".aif"}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, max_retries: int = 3,
                  user_agent: str = "RhythmAnalyzerBenchmark/1.0 (research)") -> bool:
    """Download a file from a URL with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as e:
            if "429" in str(e):
                wait = 5 * (2 ** attempt)
                print(f" rate-limited, waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f" ERROR: {e}", end="")
                return False
    return False


def resolve_audio_path(raw_fname: str, data_dir: Path) -> Path | None:
    """Resolve a METER2800 .tab filename to an actual audio file on disk.

    The .tab files reference paths like "/MAG/00553.wav" but actual files
    are flattened into data_dir/audio/ as "00553.mp3" (or with collision
    prefix like "OWN_0001.mp3").
    """
    raw_fname = raw_fname.strip('"').strip("'").strip()
    if not raw_fname:
        return None

    p = Path(raw_fname)
    src_dir = p.parent.name  # e.g. "MAG", "FMA", "GTZAN", "OWN"
    stem = p.stem
    audio_dir = data_dir / "audio"

    for ext in (".mp3", ".wav", ".ogg", ".flac", p.suffix):
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        if src_dir:
            candidate = audio_dir / f"{src_dir}_{stem}{ext}"
            if candidate.exists():
                return candidate
            candidate = audio_dir / f"{src_dir}_._{stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def parse_label_file(
    label_path: Path, data_dir: Path, valid_meters: set[int] | None = None,
) -> list[tuple[Path, int]]:
    """Parse a CSV/TSV label file from METER2800.

    Auto-detects delimiter (tab vs comma).
    Returns list of (audio_path, meter) tuples where audio_path is resolved.
    If valid_meters is given, only entries with those meter values are included.
    """
    import csv
    import warnings

    filename_cols = ["filename", "file", "audio_file", "audio_path", "audio", "path"]
    label_cols = ["meter", "time_signature", "ts", "time_sig", "signature", "label"]

    entries: list[tuple[Path, int]] = []
    missing = 0

    with open(label_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "\t" if "\t" in first_line else ","

    with open(label_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            warnings.warn(f"Empty label file: {label_path}")
            return entries

        header_map = {
            h.strip().lower().strip('"').strip("'"): h
            for h in reader.fieldnames
        }

        fname_key = None
        for candidate in filename_cols:
            if candidate in header_map:
                fname_key = header_map[candidate]
                break
        if fname_key is None:
            warnings.warn(
                f"No filename column found in {label_path}. "
                f"Headers: {list(reader.fieldnames)}"
            )
            return entries

        label_key = None
        for candidate in label_cols:
            if candidate in header_map:
                label_key = header_map[candidate]
                break
        if label_key is None:
            warnings.warn(
                f"No label column found in {label_path}. "
                f"Headers: {list(reader.fieldnames)}"
            )
            return entries

        for row in reader:
            raw_fname = row.get(fname_key, "").strip().strip('"').strip("'")
            raw_label = row.get(label_key, "").strip().strip('"').strip("'")
            if not raw_fname or not raw_label:
                continue

            meter_str = raw_label.split("/")[0].strip()
            try:
                meter = int(meter_str)
            except ValueError:
                continue

            if valid_meters is not None and meter not in valid_meters:
                continue

            audio_path = resolve_audio_path(raw_fname, data_dir)
            if audio_path is None:
                missing += 1
                continue

            entries.append((audio_path, meter))

    if missing > 0:
        print(f"    ({missing} entries skipped — audio file not found)")

    return entries
