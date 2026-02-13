#!/usr/bin/env python3
"""Collect YouTube candidates for WIKIMETER catalog expansion.

Input seed records can come from:
- --seed-file (.json, .jsonl, .csv)
- --seed "Artist|Title|Meters"

Seed schema:
{
  "artist": "Bela Fleck",
  "title": "Sinister Minister",
  "meters": "5"                  # or "3:0.8,4:0.7" or {"5": 1.0}
  "query": "optional explicit yt search query"
}

Output is JSON with ranked candidates, intended for manual review.
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
    meters_for_save,
    normalize_meters,
    parse_meters,
    save_json,
    song_video_ids,
    song_key,
    ytsearch,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "wikimeter" / "curation" / "candidates.json"
BAD_TITLE_HINTS = (
    "live",
    "cover",
    "karaoke",
    "remix",
    "slowed",
    "nightcore",
    "8d",
    "reverb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect WIKIMETER candidate songs from YouTube")
    parser.add_argument(
        "--seed-file",
        type=Path,
        action="append",
        default=[],
        help="Seed file (.json/.jsonl/.csv), can be passed multiple times",
    )
    parser.add_argument(
        "--seed",
        type=str,
        action="append",
        default=[],
        help='Inline seed: "Artist|Title|Meters", e.g. "Tool|Schism|7"',
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-query", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--sleep", type=float, default=0.3, help="Pause between yt-dlp queries")
    parser.add_argument("--max-seeds", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--allow-existing-song",
        action="store_true",
        help="Do not skip seeds already present in wikimeter.json",
    )
    parser.add_argument(
        "--allow-existing-video",
        action="store_true",
        help="Keep candidates even if video_id already exists in catalog",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write output file")
    return parser.parse_args()


def parse_inline_seed(seed: str) -> dict[str, Any]:
    parts = [x.strip() for x in seed.split("|")]
    if len(parts) != 3:
        raise ValueError(f"invalid --seed value: {seed!r} (expected Artist|Title|Meters)")
    artist, title, meters_raw = parts
    return {"artist": artist, "title": title, "meters": meters_raw}


def parse_seed_record(raw: dict[str, Any], seed_id: int) -> dict[str, Any]:
    artist = str(raw.get("artist", "")).strip()
    title = str(raw.get("title", "")).strip()
    query = str(raw.get("query", "")).strip()

    meters_value = raw.get("meters", raw.get("meter"))
    if meters_value is None:
        raise ValueError("missing meters")
    if isinstance(meters_value, dict):
        meters = normalize_meters(meters_value)
    elif isinstance(meters_value, list):
        meters = {int(x): 1.0 for x in meters_value}
    elif isinstance(meters_value, (int, float)):
        meters = {int(meters_value): 1.0}
    else:
        meters = parse_meters(str(meters_value))

    if not query:
        if not artist or not title:
            raise ValueError("missing artist/title and query")
        query = f"{artist} {title} official audio"

    if not artist or not title:
        # Keep a fallback for pure query-only seeds.
        fallback = query.strip()
        artist = artist or "(query)"
        title = title or fallback[:120]

    return {
        "seed_id": f"seed_{seed_id:04d}",
        "artist": artist,
        "title": title,
        "meters": meters,
        "query": query,
    }


def extract_flags(video_title: str, duration_s: int, score: float) -> list[str]:
    flags: list[str] = []
    title_l = video_title.lower()
    if any(h in title_l for h in BAD_TITLE_HINTS):
        flags.append("suspicious_title")
    if duration_s > 0 and duration_s < 50:
        flags.append("very_short")
    if duration_s > 900:
        flags.append("very_long")
    if score < 0.35:
        flags.append("low_score")
    return flags


def load_all_seeds(seed_files: list[Path], inline_seeds: list[str]) -> list[dict[str, Any]]:
    raw_seeds: list[dict[str, Any]] = []
    for seed_file in seed_files:
        if not seed_file.exists():
            raise FileNotFoundError(f"seed file not found: {seed_file}")
        raw_seeds.extend(load_seed_file(seed_file))
    for inline in inline_seeds:
        raw_seeds.append(parse_inline_seed(inline))

    seeds: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_seeds, start=1):
        try:
            seeds.append(parse_seed_record(raw, idx))
        except Exception as exc:
            print(f"  skip bad seed #{idx}: {exc}")
    return seeds


def main() -> None:
    args = parse_args()
    if not args.seed_file and not args.seed:
        print("ERROR: provide at least one --seed-file or --seed entry.")
        sys.exit(1)

    catalog_path = args.catalog.resolve()
    if not catalog_path.exists():
        print(f"ERROR: catalog not found: {catalog_path}")
        sys.exit(1)

    output_path = args.output.resolve()
    songs = load_catalog(catalog_path)
    existing_video_ids: set[str] = set()
    for song in songs:
        existing_video_ids.update(song_video_ids(song))
    existing_song_keys = {song_key(s["artist"], s["title"]) for s in songs}

    seeds = load_all_seeds(args.seed_file, args.seed)
    if args.max_seeds > 0:
        seeds = seeds[: args.max_seeds]
    if not seeds:
        print("No valid seeds to process.")
        sys.exit(1)

    print(f"Catalog songs: {len(songs)}")
    print(f"Seeds to process: {len(seeds)}")

    items: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    skipped_existing_song = 0
    skipped_existing_video = 0
    errors = 0

    for idx, seed in enumerate(seeds, start=1):
        skey = song_key(seed["artist"], seed["title"])
        if skey in existing_song_keys and not args.allow_existing_song:
            skipped_existing_song += 1
            print(f"[{idx}/{len(seeds)}] skip existing song: {seed['artist']} — {seed['title']}")
            continue

        print(f"[{idx}/{len(seeds)}] search: {seed['artist']} — {seed['title']} ({seed['query']})")
        try:
            results = ytsearch(seed["query"], limit=args.limit_per_query)
        except Exception as exc:
            errors += 1
            print(f"  search error: {exc}")
            continue

        kept = 0
        for cand in results:
            video_id = cand["video_id"]
            if video_id in seen_videos:
                continue
            if video_id in existing_video_ids and not args.allow_existing_video:
                skipped_existing_video += 1
                continue

            score = candidate_score(
                artist=seed["artist"],
                title=seed["title"],
                meters=seed["meters"],
                search_rank=int(cand["search_rank"]),
                video_title=str(cand["video_title"]),
                uploader=str(cand["uploader"]),
                duration_s=int(cand["duration_s"]),
            )
            if score < args.min_score:
                continue

            flags = extract_flags(str(cand["video_title"]), int(cand["duration_s"]), score)
            items.append(
                {
                    "seed_id": seed["seed_id"],
                    "artist": seed["artist"],
                    "title": seed["title"],
                    "meters": meters_for_save(seed["meters"]),
                    "query": seed["query"],
                    "video_id": video_id,
                    "video_url": cand["video_url"],
                    "source": dict(cand.get("source") or {}),
                    "video_title": cand["video_title"],
                    "uploader": cand["uploader"],
                    "duration_s": cand["duration_s"],
                    "search_rank": cand["search_rank"],
                    "score": score,
                    "flags": flags,
                }
            )
            seen_videos.add(video_id)
            kept += 1

        print(f"  kept {kept}/{len(results)}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    items.sort(key=lambda x: (x["score"], -x["search_rank"]), reverse=True)
    payload = {
        "meta": {
            "catalog": str(catalog_path),
            "total_seeds": len(seeds),
            "total_candidates": len(items),
            "skipped_existing_song": skipped_existing_song,
            "skipped_existing_video": skipped_existing_video,
            "search_errors": errors,
        },
        "items": items,
    }

    print("")
    print("Summary")
    print(f"  candidates: {len(items)}")
    print(f"  skipped existing songs: {skipped_existing_song}")
    print(f"  skipped existing videos: {skipped_existing_video}")
    print(f"  search errors: {errors}")

    if args.dry_run:
        print("Dry run: output not written.")
        return

    save_json(output_path, payload)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
