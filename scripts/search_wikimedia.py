#!/usr/bin/env python3
"""Search Wikimedia Commons for audio files in genre categories.

Uses the MediaWiki API to find audio candidates for benchmark fixtures.
Filters by file type, duration (10-180s), and outputs a JSON catalogue.

Usage:
    python scripts/search_wikimedia.py                  # search all categories
    python scripts/search_wikimedia.py --category waltz # search one category
    python scripts/search_wikimedia.py --limit 20       # max per category
    python scripts/search_wikimedia.py --output candidates.json
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://commons.wikimedia.org/w/api.php"

# Mapping: our category name -> list of Wikimedia Commons categories to search
WIKIMEDIA_CATEGORIES: dict[str, list[str]] = {
    "waltz": [
        "Audio files of waltzes",
        "Audio files of waltzes by Frédéric Chopin",
        "Audio files of 16 Waltzes (Brahms)",
    ],
    "march": [
        "Audio files of marches",
        "Military marches",
    ],
    "blues": [
        "Audio files of blues",
        "Delta blues",
        "Blues music from Incompetech",
    ],
    "jazz": [
        "Audio files of jazz music",
        "Jazz songs",
    ],
    "mazurka": [
        "Audio files of mazurkas by Frédéric Chopin",
        "Audio files of mazurkas",
    ],
    "tango": [
        "Audio files of tango music",
    ],
    "polka": [
        "Audio files of polka music",
        "Polka music from Incompetech",
        "Polka music from Free Music Archive",
    ],
    "jig": [
        "Gigue",
        "Music of Ireland",
        "Audio files of music of Ireland",
    ],
    "tarantella": [
        "Tarantella",
    ],
    "barcarolle": [
        "Barcarolle (Chopin)",
        "Sarabande",
        "Audio files of Keyboard suite in D minor (HWV 437), 4. Sarabande",
    ],
    "classical": [
        "Minuets",
        "Sarabande",
    ],
    "samba": [
        "Audio files of samba music",
        "Bossa nova",
    ],
    "reggae": [
        "Reggae music from Incompetech",
    ],
    "swing": [
        "Big band/swing music from Free Music Archive",
        "Jazz music from Incompetech",
    ],
    "folk": [
        "Audio files of folk music",
    ],
    "ragtime": [
        "Audio files of ragtime music",
        "Ragtime compositions",
    ],
    "drums": [
        "Drum patterns",
        "Audio files of drum beats",
        "Audio files of drum patterns",
    ],
    "middle_eastern": [
        "Arabic music",
        "Audio files of Arabic music",
    ],
}

from scripts.utils import AUDIO_EXTENSIONS


def api_request(params: dict) -> dict:
    """Make a request to the MediaWiki API."""
    params["format"] = "json"
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "RhythmAnalyzerBenchmark/1.0 (jasisz@gmail.com)"
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP error {e.code}: {e.reason}", file=sys.stderr)
        return {}
    except urllib.error.URLError as e:
        print(f"  URL error: {e.reason}", file=sys.stderr)
        return {}


def get_category_members(category: str, limit: int = 100) -> list[str]:
    """Get file names from a Wikimedia Commons category."""
    files = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": min(limit, 500),
    }

    while len(files) < limit:
        data = api_request(params)
        if not data or "query" not in data:
            break
        members = data["query"].get("categorymembers", [])
        for m in members:
            title = m["title"]
            # Filter audio files only
            ext = Path(title).suffix.lower()
            if ext in AUDIO_EXTENSIONS:
                files.append(title)

        # Handle continuation
        if "continue" in data and len(files) < limit:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(0.5)  # Be polite to the API
        else:
            break

    return files[:limit]


def get_file_info(titles: list[str]) -> list[dict]:
    """Get file info (URL, description, duration) for a batch of files."""
    if not titles:
        return []

    results = []
    # API accepts max 50 titles per request
    batch_size = 50
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo|categories",
            "iiprop": "url|size|mime|mediatype|extmetadata",
            "iiextmetadatafilter": "ObjectName|ImageDescription|Artist",
        }
        data = api_request(params)
        if not data or "query" not in data:
            continue

        pages = data["query"].get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            title = page.get("title", "")
            imageinfo = page.get("imageinfo", [{}])
            if not imageinfo:
                continue
            info = imageinfo[0]

            # Extract description from extmetadata
            extmeta = info.get("extmetadata", {})
            description = ""
            if "ImageDescription" in extmeta:
                description = extmeta["ImageDescription"].get("value", "")
            artist = ""
            if "Artist" in extmeta:
                artist = extmeta["Artist"].get("value", "")

            # Extract categories
            cats = [c["title"].replace("Category:", "")
                    for c in page.get("categories", [])]

            url = info.get("url", "")
            mime = info.get("mime", "")
            size = info.get("size", 0)

            # Only include audio files
            if not mime.startswith("audio/") and info.get("mediatype") != "AUDIO":
                continue

            results.append({
                "title": title.replace("File:", ""),
                "url": url,
                "mime": mime,
                "size_bytes": size,
                "description": description[:300],  # Truncate long descriptions
                "artist": artist[:200],
                "categories": cats[:10],
            })

        time.sleep(0.5)  # Rate limit

    return results


def search_category(category_name: str, wm_categories: list[str],
                    limit: int) -> list[dict]:
    """Search all Wikimedia categories for a given benchmark category."""
    all_titles = []
    seen = set()

    for wm_cat in wm_categories:
        print(f"  Searching: Category:{wm_cat}")
        titles = get_category_members(wm_cat, limit=limit * 2)
        for t in titles:
            if t not in seen:
                seen.add(t)
                all_titles.append(t)
        time.sleep(0.3)

    # Also try subcategories (one level deep)
    for wm_cat in wm_categories:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{wm_cat}",
            "cmtype": "subcat",
            "cmlimit": 20,
        }
        data = api_request(params)
        if data and "query" in data:
            for sub in data["query"].get("categorymembers", []):
                sub_title = sub["title"].replace("Category:", "")
                # Only search audio subcategories
                if any(kw in sub_title.lower() for kw in
                       ["audio", "ogg", "recording", "song", "music", "file"]):
                    print(f"    Subcategory: {sub_title}")
                    titles = get_category_members(sub_title, limit=limit)
                    for t in titles:
                        if t not in seen:
                            seen.add(t)
                            all_titles.append(t)
                    time.sleep(0.3)

    # Get file info for all found files
    print(f"  Found {len(all_titles)} audio files, fetching info...")
    candidates = get_file_info(all_titles[:limit * 3])

    # Sort by size (prefer medium-sized files, not too short or too long)
    # Rough heuristic: 100KB-5MB is likely 10-180s
    candidates = [c for c in candidates
                  if 50_000 < c["size_bytes"] < 20_000_000]

    return candidates[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Search Wikimedia Commons for benchmark audio candidates")
    parser.add_argument("--category", type=str, default=None,
                        help="Search only this category")
    parser.add_argument("--limit", type=int, default=30,
                        help="Max candidates per category (default: 30)")
    parser.add_argument("--output", type=str,
                        default="scripts/wikimedia_candidates.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    categories = WIKIMEDIA_CATEGORIES
    if args.category:
        if args.category not in categories:
            print(f"Unknown category: {args.category}")
            print(f"Available: {', '.join(sorted(categories.keys()))}")
            sys.exit(1)
        categories = {args.category: categories[args.category]}

    all_candidates: dict[str, list[dict]] = {}
    total = 0

    for cat_name, wm_cats in sorted(categories.items()):
        print(f"\n{'='*60}")
        print(f"Category: {cat_name}")
        print(f"{'='*60}")
        candidates = search_category(cat_name, wm_cats, limit=args.limit)
        all_candidates[cat_name] = candidates
        total += len(candidates)
        print(f"  -> {len(candidates)} candidates")

    # Merge with existing output (don't overwrite other categories)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(all_candidates)
    output_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} candidates across {len(all_candidates)} categories")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")

    # Summary per category
    for cat, cands in sorted(all_candidates.items()):
        print(f"  {cat:18s}: {len(cands):3d} candidates")


if __name__ == "__main__":
    main()
