#!/usr/bin/env python3
"""Download new fixture candidates from Wikimedia Commons, compute SHA256 hashes,
and generate entries for download_fixtures.py and benchmark.py.

Uses URLs from wikimedia_candidates.json (fetched via API, so URLs are correct).

Usage:
    python scripts/download_new_fixtures.py              # download all
    python scripts/download_new_fixtures.py --dry-run    # show what would be downloaded
    python scripts/download_new_fixtures.py --category mazurka
    python scripts/download_new_fixtures.py --resume     # retry failed downloads
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
CANDIDATES_FILE = Path(__file__).resolve().parent / "wikimedia_candidates.json"
RESULTS_FILE = Path(__file__).resolve().parent / "new_fixture_entries.json"

# Selection rules per category: how many to pick, what meters to assign
# (category, max_count, default_meters, default_bpm_range, filename_prefix)
CATEGORY_CONFIG = {
    "waltz":    {"max": 10, "meters": [(3, 4)], "bpm": (80, 220), "prefix": "waltz"},
    "mazurka":  {"max": 15, "meters": [(3, 4)], "bpm": (80, 220), "prefix": "mazurka"},
    "march":    {"max": 11, "meters": [(4, 4), (2, 4)], "bpm": (80, 150), "prefix": "march"},
    "blues":    {"max": 13, "meters": [(4, 4), (2, 4)], "bpm": (40, 240), "prefix": "blues"},
    "jazz":     {"max": 12, "meters": [(4, 4), (2, 4)], "bpm": (60, 220), "prefix": "jazz"},
    "ragtime":  {"max": 10, "meters": [(4, 4), (2, 4)], "bpm": (60, 200), "prefix": "ragtime"},
    "tango":    {"max": 10, "meters": [(4, 4), (2, 4)], "bpm": (40, 160), "prefix": "tango"},
    "folk":     {"max": 10, "meters": [(4, 4), (2, 4)], "bpm": (60, 220), "prefix": "folk"},
    "drums":    {"max": 8, "meters": [(4, 4), (2, 4)], "bpm": (60, 220), "prefix": "drums"},
    "middle_eastern": {"max": 3, "meters": [(4, 4), (2, 4)], "bpm": (60, 200), "prefix": "mideast"},
    "polka":   {"max": 10, "meters": [(2, 4), (4, 4)], "bpm": (80, 220), "prefix": "polka"},
    "reggae":  {"max": 10, "meters": [(4, 4), (2, 4)], "bpm": (60, 160), "prefix": "reggae"},
    "samba":   {"max": 10, "meters": [(2, 4), (4, 4)], "bpm": (60, 200), "prefix": "samba"},
    "swing":   {"max": 10, "meters": [(4, 4), (2, 4)], "bpm": (80, 220), "prefix": "swing"},
    "jig":     {"max": 10, "meters": [(6, 8), (6, 4), (3, 4)], "bpm": (60, 200), "prefix": "jig"},
    "barcarolle": {"max": 5, "meters": [(6, 8), (6, 4), (3, 4)], "bpm": (30, 100), "prefix": "barcarolle"},
    "classical": {"max": 10, "meters": [(3, 4)], "bpm": (40, 160), "prefix": "classical"},
}

# Override meters for specific titles (musicological knowledge)
METER_OVERRIDES = {
    # Blues shuffles are often 12/8
    "AcousticShuffle": [(4, 4), (12, 8), (2, 4)],
    "AxesShuffle": [(4, 4), (12, 8), (2, 4)],
    "FastShuffle": [(4, 4), (12, 8), (2, 4)],
    "FastBoogie": [(4, 4), (12, 8), (2, 4)],
    "MedBoogie": [(4, 4), (12, 8), (2, 4)],
    "Blues Slow": [(4, 4), (12, 8), (2, 4)],
    # Folk songs with 3/4 or 6/8
    "House of the Rising Sun": [(3, 4), (6, 8), (6, 4)],
    "Blow the Man Down": [(3, 4), (6, 8)],
    "Early One Morning": [(4, 4), (3, 4)],
    "Greensleeves": [(3, 4), (6, 8), (6, 4)],
}


def sanitize_filename(title: str, prefix: str, index: int) -> str:
    """Create a clean filename from a Wikimedia title."""
    # Get the file extension
    ext = Path(title).suffix.lower()
    if ext not in (".ogg", ".oga", ".wav", ".flac", ".mp3", ".opus"):
        ext = ".ogg"

    # Clean the title for use as filename
    name = Path(title).stem
    # Remove quotes, parens, and common prefixes
    name = re.sub(r'^["\']+|["\']+$', '', name)
    name = re.sub(r'\([^)]*\)', '', name)  # Remove parentheticals
    name = name.strip(' -_,.')
    # Keep only alphanumeric, spaces, hyphens
    name = re.sub(r'[^a-zA-Z0-9\s\-]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_')
    name = name[:40]  # Truncate long names

    if not name:
        name = f"track_{index:02d}"

    return f"{prefix}_{name.lower()}{ext}"


def title_fingerprint(title: str) -> str:
    """Extract a rough 'song identity' to avoid duplicate recordings."""
    t = title.lower()
    # Remove performer info, dates, recording details
    t = re.sub(r'\([^)]*\)', '', t)
    t = re.sub(r'\d{4}', '', t)  # years
    t = re.sub(r'by .*', '', t)
    t = re.sub(r'performed by .*', '', t)
    t = re.sub(r'sung by .*', '', t)
    t = re.sub(r'[^a-z\s]', '', t)
    return ' '.join(t.split())[:30]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "RhythmAnalyzerBenchmark/1.0 (jasisz@gmail.com; research)"
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as e:
            if "429" in str(e):
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f" rate-limited, waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f" ERROR: {e}", end="")
                return False
    return False


def get_meters_for_title(title: str, category: str) -> list[tuple[int, int]] | None:
    """Check if we have a meter override for this title."""
    for key, meters in METER_OVERRIDES.items():
        if key.lower() in title.lower():
            return meters
    return None


def main():
    parser = argparse.ArgumentParser(description="Download new fixture candidates")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--category", type=str, default=None, help="Download only this category")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded files")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between downloads (seconds)")
    args = parser.parse_args()

    if not CANDIDATES_FILE.exists():
        print(f"Candidates file not found: {CANDIDATES_FILE}")
        print("Run search_wikimedia.py first.")
        sys.exit(1)

    candidates = json.loads(CANDIDATES_FILE.read_text())
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing fixtures to avoid duplicates
    existing_urls = set()
    manifest_path = Path(__file__).resolve().parent / "fixtures_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            existing_urls.add(entry["url"])

    # Select candidates per category
    selected: list[dict] = []  # list of {filename, category, meters, bpm, url, title}

    categories = CATEGORY_CONFIG
    if args.category:
        if args.category not in categories:
            print(f"Unknown category: {args.category}")
            sys.exit(1)
        categories = {args.category: categories[args.category]}

    seen_fingerprints: set[str] = set()
    seen_filenames: set[str] = set()

    for cat, config in sorted(categories.items()):
        cat_candidates = candidates.get(cat, [])
        count = 0
        for i, cand in enumerate(cat_candidates):
            if count >= config["max"]:
                break
            url = cand["url"]
            if url in existing_urls:
                continue  # Already have this file

            title = cand["title"]

            # Deduplicate same song in different recordings
            fp = title_fingerprint(title)
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            filename = sanitize_filename(title, config["prefix"], i)
            # Handle filename collisions
            if filename in seen_filenames:
                base, ext = filename.rsplit('.', 1)
                filename = f"{base}_{count}.{ext}"
            seen_filenames.add(filename)

            meters = get_meters_for_title(title, cat) or config["meters"]

            selected.append({
                "filename": filename,
                "category": cat,
                "meters": meters,
                "bpm": config["bpm"],
                "url": url,
                "title": title,
            })
            existing_urls.add(url)  # Prevent within-run duplicates
            count += 1

    # Summary
    from collections import Counter
    cat_counts = Counter(s["category"] for s in selected)
    print(f"{'='*70}")
    print(f"  NEW FIXTURES TO DOWNLOAD: {len(selected)}")
    print(f"{'='*70}")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:18s}: {count}")

    if args.dry_run:
        print("\n  Files to download:")
        for s in selected:
            print(f"  {s['filename']:45s} <- {s['title'][:60]}")
        print("\n  --dry-run: not downloading")
        return

    # Download files
    downloaded = 0
    skipped = 0
    failed = 0
    results = []  # successful downloads

    for i, s in enumerate(selected, 1):
        dest = FIXTURES_DIR / s["filename"]

        if args.resume and dest.exists():
            sha = sha256_file(dest)
            print(f"  [{i}/{len(selected)}] EXISTS: {s['filename']}")
            results.append({**s, "sha256": sha})
            skipped += 1
            continue

        sys.stdout.write(f"  [{i}/{len(selected)}] {s['filename']}...")
        sys.stdout.flush()

        if download_file(s["url"], dest):
            sha = sha256_file(dest)
            print(f" OK ({sha[:12]}...)")
            results.append({**s, "sha256": sha})
            downloaded += 1
        else:
            print(f" FAILED")
            failed += 1

        # Rate limit
        time.sleep(args.delay)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"{'='*70}")

    # Save results for next step
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {RESULTS_FILE}")

    # Generate code snippets
    if results:
        print(f"\n# ===== Add to FIXTURES in scripts/download_fixtures.py =====\n")
        for r in results:
            print(f'    ("{r["filename"]}", "{r["sha256"]}", "{r["url"]}"),')

        print(f"\n# ===== Add to FIXTURE_CATALOGUE in tests/benchmark.py =====\n")
        current_cat = None
        for r in results:
            if r["category"] != current_cat:
                current_cat = r["category"]
                print(f"\n    # {current_cat.replace('_', ' ').title()}")
            meters_str = ", ".join(f"({m[0]}, {m[1]})" for m in r["meters"])
            bpm = r["bpm"]
            pad = " " * max(1, 45 - len(f'"{r["filename"]}":'))
            print(f'    "{r["filename"]}":{pad}("{r["category"]}", [{meters_str}], ({bpm[0]}, {bpm[1]})),')


if __name__ == "__main__":
    main()
