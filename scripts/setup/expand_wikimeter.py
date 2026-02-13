#!/usr/bin/env python3
"""Expand WIKIMETER catalog from seed files via YouTube search.

Pipeline: seed CSV/JSON/JSONL → yt-dlp search → candidate scoring → review queue.

Usage:
    # Smoke test (3 seeds, dry-run)
    python expand_wikimeter.py --seed seeds.csv --limit 3 --dry-run

    # Full run with auto-approve threshold
    python expand_wikimeter.py --seed seeds.csv --auto-approve 1.2

    # Pipe directly into merge
    python expand_wikimeter.py --seed seeds.csv --auto-approve 1.2 --merge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from wikimeter_tools import (
    DEFAULT_CATALOG,
    PROJECT_ROOT,
    candidate_score,
    load_catalog,
    load_seed_file,
    meters_to_tag,
    now_iso,
    parse_meters,
    save_json,
    song_key,
    song_source_keys,
    song_video_ids,
    source_from_youtube,
    source_key,
    ytsearch,
)

DEFAULT_BLACKLIST = PROJECT_ROOT / "data" / "wikimeter" / "blacklist.json"
DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "wikimeter" / "review_queue.json"


def load_blacklist_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    import json

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    out: set[tuple[str, str]] = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        artist = str(entry.get("artist", "")).strip()
        title = str(entry.get("title", "")).strip()
        meters_raw = entry.get("meters")
        if not artist or not title or meters_raw is None:
            continue
        if isinstance(meters_raw, str):
            meters = parse_meters(meters_raw)
        elif isinstance(meters_raw, dict):
            meters = {int(k): float(v) for k, v in meters_raw.items()}
        elif isinstance(meters_raw, list):
            meters = {int(x): 1.0 for x in meters_raw}
        else:
            continue
        out.add((song_key(artist, title), meters_to_tag(meters)))
    return out


def build_catalog_index(catalog: list[dict[str, Any]]) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, song in enumerate(catalog):
        idx[song_key(song["artist"], song["title"])] = i
        for vid in song_video_ids(song):
            idx[f"yt:{vid}"] = i
        for sk in song_source_keys(song):
            idx[f"src:{sk}"] = i
    return idx


def parse_seed(raw: dict[str, Any]) -> dict[str, Any]:
    artist = str(raw.get("artist", "")).strip()
    title = str(raw.get("title", "")).strip()
    meters_raw = raw.get("meters", raw.get("meter", ""))
    if isinstance(meters_raw, str):
        meters = parse_meters(meters_raw)
    elif isinstance(meters_raw, dict):
        meters = {int(k): float(v) for k, v in meters_raw.items()}
    elif isinstance(meters_raw, (int, float)):
        meters = {int(meters_raw): 1.0}
    elif isinstance(meters_raw, list):
        meters = {int(x): 1.0 for x in meters_raw}
    else:
        raise ValueError(f"invalid meters: {meters_raw!r}")
    query = str(raw.get("query", "")).strip()
    if not query:
        query = f"{artist} {title}"
    if not artist or not title:
        raise ValueError(f"missing artist/title in seed: {raw}")
    return {
        "artist": artist,
        "title": title,
        "meters": meters,
        "query": query,
    }


def search_and_score(seed: dict[str, Any], search_limit: int = 3) -> dict[str, Any] | None:
    """Search YouTube for seed and return best candidate, or None."""
    try:
        results = ytsearch(seed["query"], limit=search_limit)
    except RuntimeError as exc:
        print(f"    search error: {exc}")
        return None

    if not results:
        return None

    best = None
    best_score = -1.0
    for r in results:
        sc = candidate_score(
            artist=seed["artist"],
            title=seed["title"],
            meters=seed["meters"],
            search_rank=r["search_rank"],
            video_title=r["video_title"],
            uploader=r["uploader"],
            duration_s=r["duration_s"],
        )
        if sc > best_score:
            best_score = sc
            best = r

    if best is None:
        return None

    return {
        "video_id": best["video_id"],
        "video_url": best["video_url"],
        "video_title": best["video_title"],
        "uploader": best["uploader"],
        "duration_s": best["duration_s"],
        "score": best_score,
        "source": best["source"],
    }


def run_pipeline(args: argparse.Namespace) -> None:
    seed_path = args.seed.resolve()
    if not seed_path.exists():
        print(f"ERROR: seed file not found: {seed_path}")
        sys.exit(1)

    catalog = load_catalog(args.catalog.resolve())
    catalog_idx = build_catalog_index(catalog)
    blacklist = load_blacklist_keys(args.blacklist.resolve())

    raw_seeds = load_seed_file(seed_path)
    if args.limit > 0:
        raw_seeds = raw_seeds[: args.limit]

    seeds: list[dict[str, Any]] = []
    parse_errors = 0
    for raw in raw_seeds:
        try:
            seeds.append(parse_seed(raw))
        except Exception as exc:
            parse_errors += 1
            print(f"  skip malformed seed: {exc}")

    print(f"Seeds loaded: {len(seeds)} (parse errors: {parse_errors})")

    # Dedup against catalog and blacklist
    deduped: list[dict[str, Any]] = []
    skipped_catalog = 0
    skipped_blacklist = 0
    for seed in seeds:
        skey = song_key(seed["artist"], seed["title"])
        mtag = meters_to_tag(seed["meters"])

        if skey in catalog_idx:
            skipped_catalog += 1
            continue
        if (skey, mtag) in blacklist:
            skipped_blacklist += 1
            continue
        deduped.append(seed)

    print(f"After dedup: {len(deduped)} (catalog: -{skipped_catalog}, blacklist: -{skipped_blacklist})")

    if not deduped:
        print("Nothing to search.")
        return

    if args.dry_run:
        for i, s in enumerate(deduped, 1):
            mtag = meters_to_tag(s["meters"])
            print(f"  {i:3d}. [{mtag}] {s['artist']} — {s['title']}  q={s['query']!r}")
        print(f"\nDry run: {len(deduped)} seeds would be searched.")
        return

    # Search and score
    review_items: list[dict[str, Any]] = []
    stats = {"found": 0, "auto_approved": 0, "pending": 0, "no_match": 0}

    for i, seed in enumerate(deduped, 1):
        mtag = meters_to_tag(seed["meters"])
        print(f"  [{i:3d}/{len(deduped)}] [{mtag}] {seed['artist']} — {seed['title']}", end="", flush=True)

        candidate = search_and_score(seed, search_limit=args.search_limit)

        if candidate is None:
            print(" — NO MATCH")
            stats["no_match"] += 1
            continue

        # Check if this video is already in catalog
        vid = candidate["video_id"]
        if f"yt:{vid}" in catalog_idx or f"src:{source_key(source_from_youtube(vid))}" in catalog_idx:
            print(f" — video already in catalog ({vid})")
            stats["no_match"] += 1
            continue

        status = "approved" if candidate["score"] >= args.auto_approve else "pending"
        if status == "approved":
            stats["auto_approved"] += 1
        else:
            stats["pending"] += 1
        stats["found"] += 1

        review_items.append(
            {
                "id": f"expand-{i:04d}",
                "status": status,
                "seed": {
                    "artist": seed["artist"],
                    "title": seed["title"],
                    "meters": {str(k): v for k, v in seed["meters"].items()},
                },
                "candidate": candidate,
                "created_at": now_iso(),
            }
        )

        score_str = f"{candidate['score']:.2f}"
        status_mark = "+" if status == "approved" else "?"
        print(f" — [{status_mark}] score={score_str} vid={vid} ({candidate['video_title'][:50]})")

        # Rate limit: be nice to YouTube
        if i < len(deduped):
            time.sleep(1.0)

    # Save review queue
    output_path = args.output.resolve()
    save_json(output_path, review_items)

    print(f"\nResults: {stats['found']} found, {stats['auto_approved']} auto-approved, "
          f"{stats['pending']} pending review, {stats['no_match']} no match")
    print(f"Review queue: {output_path}")

    if args.merge and stats["auto_approved"] > 0:
        print("\nRunning merge...")
        import subprocess

        merge_script = Path(__file__).resolve().parent / "merge_wikimeter_reviewed.py"
        cmd = [
            sys.executable,
            str(merge_script),
            "--review-queue",
            str(output_path),
            "--catalog",
            str(args.catalog.resolve()),
            "--blacklist",
            str(args.blacklist.resolve()),
            "--status",
            "approved",
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("ERROR: merge failed")
            sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand WIKIMETER from seed files via YouTube search")
    parser.add_argument("--seed", type=Path, required=True, help="Seed file (CSV/JSON/JSONL)")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REVIEW_QUEUE, help="Review queue output path")
    parser.add_argument("--limit", type=int, default=0, help="Max seeds to process (0=all)")
    parser.add_argument("--search-limit", type=int, default=3, help="YouTube results per seed")
    parser.add_argument("--auto-approve", type=float, default=1.2, help="Auto-approve score threshold")
    parser.add_argument("--dry-run", action="store_true", help="Show seeds without searching")
    parser.add_argument("--merge", action="store_true", help="Auto-merge approved entries after search")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
