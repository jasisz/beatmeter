#!/usr/bin/env python3
"""Download and extract the METER2800 dataset from Harvard Dataverse.

METER2800: 2800 audio clips labeled with time signature (3, 4, 5, 7).
Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0CLXBQ

Usage:
    uv run python scripts/download_meter2800.py
    uv run python scripts/download_meter2800.py --output-dir /path/to/output
    uv run python scripts/download_meter2800.py --skip-download  # only extract
"""

import argparse
import hashlib
import json
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DOI = "doi:10.7910/DVN/0CLXBQ"
API_BASE = "https://dataverse.harvard.edu/api"
DATASET_URL = f"{API_BASE}/datasets/:persistentId/?persistentId={DOI}"

# Files we expect in the dataset
TAR_FILES = {"FMA.tar.gz", "MAG.tar.gz", "OWN.tar.gz"}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "meter2800"

from scripts.utils import AUDIO_EXTENSIONS, md5_file


def create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context for HTTPS requests."""
    ctx = ssl.create_default_context()
    return ctx


def api_request(url: str, max_retries: int = 3) -> bytes:
    """Make an API request with retry and exponential backoff."""
    ssl_ctx = create_ssl_context()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "RhythmAnalyzer/1.0 (research)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2 * (2 ** attempt)
                print(f"  HTTP {e.code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 2 * (2 ** attempt)
                print(f"  Connection error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def get_dataset_files() -> list[dict]:
    """Fetch the list of files in the METER2800 dataset from Dataverse API."""
    print(f"Fetching dataset metadata from Harvard Dataverse...")
    print(f"  DOI: {DOI}")
    print(f"  URL: {DATASET_URL}")

    data = api_request(DATASET_URL)
    metadata = json.loads(data)

    if metadata.get("status") != "OK":
        raise RuntimeError(f"API returned status: {metadata.get('status')}")

    files = metadata["data"]["latestVersion"]["files"]
    print(f"  Found {len(files)} files in dataset\n")

    result = []
    for f in files:
        df = f["dataFile"]
        result.append({
            "id": df["id"],
            "filename": df["filename"],
            "size": df.get("filesize", 0),
            "md5": df.get("md5", ""),
            "content_type": df.get("contentType", ""),
        })

    return result


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes == 0:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(file_id: int, filename: str, dest: Path, expected_size: int,
                  max_retries: int = 3) -> bool:
    """Download a file from Dataverse with progress display and retry."""
    url = f"{API_BASE}/access/datafile/{file_id}"
    ssl_ctx = create_ssl_context()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "RhythmAnalyzer/1.0 (research)",
            })
            with urllib.request.urlopen(req, timeout=300, context=ssl_ctx) as resp:
                total = int(resp.headers.get("Content-Length", expected_size))
                downloaded = 0
                chunk_size = 64 * 1024  # 64KB chunks

                # Ensure parent directory exists
                dest.parent.mkdir(parents=True, exist_ok=True)

                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress bar
                        if total > 0:
                            pct = downloaded / total * 100
                            bar_len = 40
                            filled = int(bar_len * downloaded / total)
                            bar = "#" * filled + "-" * (bar_len - filled)
                            sys.stdout.write(
                                f"\r  [{bar}] {pct:5.1f}%  "
                                f"{format_size(downloaded)} / {format_size(total)}  "
                            )
                            sys.stdout.flush()

                print()  # Newline after progress bar
                return True

        except urllib.error.HTTPError as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"\n  HTTP {e.code} downloading {filename}, retrying in {wait}s...")
                time.sleep(wait)
                # Remove partial file
                if dest.exists():
                    dest.unlink()
            else:
                print(f"\n  FAILED: HTTP {e.code} downloading {filename}")
                if dest.exists():
                    dest.unlink()
                return False

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"\n  Error downloading {filename}: {e}, retrying in {wait}s...")
                time.sleep(wait)
                if dest.exists():
                    dest.unlink()
            else:
                print(f"\n  FAILED: {e}")
                if dest.exists():
                    dest.unlink()
                return False

    return False




def extract_tar(tar_path: Path, audio_dir: Path) -> int:
    """Extract audio files from a tar.gz archive, flattening directory structure.

    Returns the number of extracted audio files.
    """
    print(f"  Extracting {tar_path.name}...")
    audio_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        total = len(members)

        for i, member in enumerate(members):
            if not member.isfile():
                continue

            # Get just the filename (flatten directory structure)
            name = Path(member.name).name
            ext = Path(name).suffix.lower()

            if ext not in AUDIO_EXTENSIONS:
                continue

            # Handle filename collisions by prefixing with subdirectory name
            dest = audio_dir / name
            if dest.exists():
                # Use parent directory name as prefix to avoid collision
                parent = Path(member.name).parent.name
                name = f"{parent}_{name}"
                dest = audio_dir / name

            # Extract to a temporary name, then move
            member_copy = tarfile.TarInfo(name=name)
            member_copy.size = member.size
            member_copy.mode = member.mode

            with tar.extractfile(member) as src:
                if src is None:
                    continue
                with open(dest, "wb") as dst:
                    dst.write(src.read())

            count += 1

            # Progress every 100 files
            if count % 100 == 0:
                sys.stdout.write(f"\r    Extracted {count} audio files...")
                sys.stdout.flush()

    if count > 0:
        print(f"\r    Extracted {count} audio files from {tar_path.name}")
    else:
        print(f"    No audio files found in {tar_path.name}")

    return count


def validate_dataset(output_dir: Path):
    """Validate the extracted dataset and print statistics."""
    audio_dir = output_dir / "audio"
    csv_files = list(output_dir.glob("*.csv"))

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION")
    print(f"{'=' * 70}")

    # Count audio files
    if audio_dir.exists():
        audio_files = [
            f for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        print(f"  Audio files: {len(audio_files)}")

        # Count by extension
        ext_counts: dict[str, int] = {}
        for f in audio_files:
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        for ext, count in sorted(ext_counts.items()):
            print(f"    {ext:8s}: {count}")
    else:
        print(f"  Audio directory not found: {audio_dir}")

    # Parse CSV labels if available
    if csv_files:
        print(f"\n  CSV files found:")
        for csv_path in csv_files:
            print(f"    {csv_path.name}")
            try:
                _print_csv_stats(csv_path)
            except Exception as e:
                print(f"      Error reading CSV: {e}")
    else:
        print(f"\n  No CSV label files found")

    print(f"{'=' * 70}")


def _print_csv_stats(csv_path: Path):
    """Parse a CSV file and print label distribution."""
    with open(csv_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"      Empty or header-only CSV")
        return

    header = lines[0].strip().split(",")
    print(f"      Columns: {header}")
    print(f"      Rows: {len(lines) - 1}")

    # Try to find a meter/time_signature column
    meter_col = None
    for i, col in enumerate(header):
        col_lower = col.strip().lower().replace('"', '').replace("'", "")
        if col_lower in ("meter", "time_signature", "timesig", "ts", "label",
                         "class", "time signature", "beats_per_bar",
                         "beats_per_measure"):
            meter_col = i
            break

    if meter_col is not None:
        # Count classes
        class_counts: dict[str, int] = {}
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) > meter_col:
                label = parts[meter_col].strip().replace('"', '').replace("'", "")
                class_counts[label] = class_counts.get(label, 0) + 1

        print(f"      Label distribution (column '{header[meter_col].strip()}'):")
        for label, count in sorted(class_counts.items()):
            print(f"        {label:10s}: {count}")
    else:
        print(f"      No meter/time_signature column detected")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the METER2800 dataset from Harvard Dataverse"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, only extract existing tar.gz files",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    downloads_dir = output_dir / "downloads"

    print(f"{'=' * 70}")
    print(f"  METER2800 Dataset Downloader")
    print(f"{'=' * 70}")
    print(f"  Output:    {output_dir}")
    print(f"  Audio:     {audio_dir}")
    print(f"  Downloads: {downloads_dir}")
    print()

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        # Step 1: Get file list from Dataverse
        try:
            files = get_dataset_files()
        except Exception as e:
            print(f"  ERROR: Failed to fetch dataset metadata: {e}")
            sys.exit(1)

        # Categorize files
        tar_files = [f for f in files if f["filename"] in TAR_FILES]
        csv_files = [f for f in files if f["filename"].lower().endswith(".csv")]
        other_files = [
            f for f in files
            if f not in tar_files and f not in csv_files
        ]

        print(f"  Tar archives to download: {len(tar_files)}")
        for f in tar_files:
            print(f"    {f['filename']:20s} ({format_size(f['size'])})")

        print(f"  CSV files to download:    {len(csv_files)}")
        for f in csv_files:
            print(f"    {f['filename']:20s} ({format_size(f['size'])})")

        if other_files:
            print(f"  Other files:              {len(other_files)}")
            for f in other_files:
                print(f"    {f['filename']:20s} ({format_size(f['size'])})")

        print()

        # Step 2: Download tar.gz files
        download_success = True
        for f in tar_files:
            dest = downloads_dir / f["filename"]

            # Check if already downloaded and valid
            if dest.exists():
                if f["md5"]:
                    existing_md5 = md5_file(dest)
                    if existing_md5 == f["md5"]:
                        print(f"  {f['filename']}: already downloaded, MD5 verified")
                        continue
                    else:
                        print(f"  {f['filename']}: MD5 mismatch, re-downloading")
                elif dest.stat().st_size == f["size"] and f["size"] > 0:
                    print(f"  {f['filename']}: already downloaded, size matches")
                    continue

            print(f"  Downloading {f['filename']} ({format_size(f['size'])})...")
            if not download_file(f["id"], f["filename"], dest, f["size"]):
                print(f"  FAILED to download {f['filename']}")
                download_success = False
                continue

            # Verify MD5 if available
            if f["md5"]:
                actual_md5 = md5_file(dest)
                if actual_md5 == f["md5"]:
                    print(f"    MD5 verified: {actual_md5}")
                else:
                    print(f"    WARNING: MD5 mismatch!")
                    print(f"      Expected: {f['md5']}")
                    print(f"      Actual:   {actual_md5}")

        # Step 3: Download CSV files
        for f in csv_files:
            dest = output_dir / f["filename"]

            if dest.exists():
                print(f"  {f['filename']}: already exists, skipping")
                continue

            print(f"  Downloading {f['filename']} ({format_size(f['size'])})...")
            if not download_file(f["id"], f["filename"], dest, f["size"]):
                print(f"  FAILED to download {f['filename']}")

        # Step 4: Download any other files (e.g., README, metadata)
        for f in other_files:
            # Decide where to put it: CSV-like go to output_dir, archives to downloads
            if f["filename"].lower().endswith((".tar.gz", ".zip", ".gz")):
                dest = downloads_dir / f["filename"]
            else:
                dest = output_dir / f["filename"]

            if dest.exists():
                print(f"  {f['filename']}: already exists, skipping")
                continue

            print(f"  Downloading {f['filename']} ({format_size(f['size'])})...")
            download_file(f["id"], f["filename"], dest, f["size"])

        if not download_success:
            print("\n  WARNING: Some downloads failed. Use --skip-download after "
                  "manually placing files in the downloads directory.")

    # Step 5: Extract tar.gz files
    print(f"\n{'=' * 70}")
    print(f"  EXTRACTION")
    print(f"{'=' * 70}")

    total_extracted = 0
    tar_paths = sorted(downloads_dir.glob("*.tar.gz"))

    if not tar_paths:
        print(f"  No tar.gz files found in {downloads_dir}")
        if args.skip_download:
            print(f"  Download the files first (run without --skip-download)")
        sys.exit(1)

    for tar_path in tar_paths:
        if tar_path.name not in TAR_FILES:
            print(f"  Skipping unexpected archive: {tar_path.name}")
            continue
        count = extract_tar(tar_path, audio_dir)
        total_extracted += count

    print(f"\n  Total audio files extracted: {total_extracted}")

    # Step 6: Validate
    validate_dataset(output_dir)


if __name__ == "__main__":
    main()
