"""Shared utilities for scripts/ — audio file handling, hashing, downloads."""

import csv
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".oga", ".opus", ".aiff", ".aif"}

_corrections_cache: dict[str, dict] | None = None

CORRECTIONS_PATH = PROJECT_ROOT / "scripts" / "setup" / "meter2800_corrections.json"


def load_meter2800_corrections() -> dict[str, dict]:
    """Load METER2800 label corrections and blacklist from meter2800_corrections.json.

    Returns dict keyed by filename stem:
      {"action": "blacklist"|"relabel", "new_meter": N, "reason": "..."}

    Cached after first load.
    """
    global _corrections_cache
    if _corrections_cache is None:
        corrections: dict[str, dict] = {}
        if CORRECTIONS_PATH.exists():
            data = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
            for key, val in data.items():
                if not key.startswith("_"):  # skip _comment, _actions meta keys
                    corrections[key] = val
        _corrections_cache = corrections
    return _corrections_cache


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


def split_by_stem(
    filename_stem: str,
    train_pct: int = 80,
    val_pct: int = 10,
) -> str:
    """Deterministic hash-based split from a filename stem.

    Strips _segNN suffixes so all segments of the same song land in
    the same split (prevents data leakage).

    Returns "train", "val", or "test".
    """
    import re

    song_stem = re.sub(r"_seg\d+$", "", filename_stem)
    h = int(hashlib.sha256(song_stem.encode()).hexdigest(), 16) % 100
    if h < train_pct:
        return "train"
    if h < train_pct + val_pct:
        return "val"
    return "test"


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

    # Check blacklist / corrections
    corrections = load_meter2800_corrections()
    entry = corrections.get(stem)
    if entry and entry.get("action") == "blacklist":
        return None

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


def load_meter2800_entries(
    data_dir: Path,
    split: str,
    valid_meters: set[int] | None = None,
) -> list[tuple[Path, int]]:
    """Canonical loader for METER2800 entries. Use this everywhere.

    Applies corrections from meter2800_corrections.json:
      - blacklisted stems are excluded (already handled by resolve_audio_path)
      - relabeled stems get their meter overridden

    split='tuning' combines train+val (2100 files).
    valid_meters: if given, only entries with those meter values are included
    (filter applied AFTER relabeling).
    """
    if split == "tuning":
        entries = []
        for sub in ("train", "val"):
            entries.extend(load_meter2800_entries(data_dir, sub, valid_meters))
        return entries

    tab_files = [f"data_{split}_4_classes.tab"]
    corrections = load_meter2800_corrections()
    entries: list[tuple[Path, int]] = []

    for tab_name in tab_files:
        label_path = data_dir / tab_name
        if not label_path.exists():
            continue
        with open(label_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fname = row["filename"].strip('"')
                meter = int(row["meter"])
                audio_path = resolve_audio_path(fname, data_dir)
                if audio_path is None:
                    continue
                # Apply relabel corrections
                stem = Path(fname).stem
                corr = corrections.get(stem)
                if corr and corr.get("action") == "relabel":
                    meter = int(corr["new_meter"])
                if valid_meters is not None and meter not in valid_meters:
                    continue
                entries.append((audio_path, meter))

    return entries


# ---------------------------------------------------------------------------
# WIKIMETER loader (for eval.py)
# ---------------------------------------------------------------------------

_LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5,
    "seven": 7, "nine": 9, "eleven": 11,
}


def _parse_meters_str(meter_str: str) -> dict[int, float]:
    """Parse WIKIMETER meter column: '3:0.7,4:0.8' or '4'."""
    meters: dict[int, float] = {}
    for part in meter_str.split(","):
        part = part.strip()
        if ":" in part:
            m_str, w_str = part.split(":", 1)
            try:
                meters[int(m_str)] = float(w_str)
            except ValueError:
                continue
        else:
            try:
                meters[int(part)] = 1.0
            except ValueError:
                continue
    return meters


def load_wikimeter_entries(
    data_dir: Path,
    split: str,
    valid_meters: set[int] | None = None,
) -> list[tuple[Path, int, dict[int, float] | None]]:
    """Load WIKIMETER entries as 3-tuples: (audio_path, primary_meter, meters_dict).

    meters_dict contains all GT meters with weights (e.g., {3: 0.6, 4: 1.0}).
    For single-label entries, meters_dict is None.

    split: 'train', 'val', 'test', or 'tuning' (= train+val), or 'all'.
    """
    if split == "tuning":
        entries: list[tuple[Path, int, dict[int, float] | None]] = []
        for sub in ("train", "val"):
            entries.extend(load_wikimeter_entries(data_dir, sub, valid_meters))
        return entries
    if split == "all":
        entries = []
        for sub in ("train", "val", "test"):
            entries.extend(load_wikimeter_entries(data_dir, sub, valid_meters))
        return entries

    tab_path = data_dir / "data_wikimeter.tab"
    if not tab_path.exists():
        return []

    audio_dir = data_dir / "audio"
    entries = []
    with open(tab_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["label"].strip('"')
            primary_meter = _LABEL_TO_METER.get(label)
            if primary_meter is None:
                continue

            meter_str = row["meter"].strip('"')
            meters_dict = _parse_meters_str(meter_str)
            if not meters_dict:
                meters_dict = {primary_meter: 1.0}

            fname = Path(row["filename"].strip('"')).name
            audio_path = audio_dir / fname
            if not audio_path.exists():
                continue

            # Filter by split (hash-based, same as train_arbiter)
            file_split = split_by_stem(audio_path.stem)
            if file_split != split:
                continue

            if valid_meters is not None and primary_meter not in valid_meters:
                continue

            multi = meters_dict if len(meters_dict) > 1 else None
            entries.append((audio_path, primary_meter, multi))

    return entries
