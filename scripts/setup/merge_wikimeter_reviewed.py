#!/usr/bin/env python3
"""Merge reviewed queue entries into scripts/setup/wikimeter.json."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from wikimeter_tools import (
    DEFAULT_CATALOG,
    PROJECT_ROOT,
    load_catalog,
    load_json,
    now_iso,
    parse_meters,
    save_catalog,
    song_key,
)

DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "wikimeter" / "curation" / "review_queue.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed candidates into wikimeter.json")
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--status",
        type=str,
        default="approved",
        help="merge entries with this status (case-insensitive), default: approved",
    )
    parser.add_argument("--replace-existing-song", action="store_true")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_review_items(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [dict(x) for x in payload["items"]]
    raise ValueError("review queue file must be list[] or {'items': list[]}")


def is_status_match(item_status: str, expected_status: str) -> bool:
    return item_status.strip().lower() == expected_status.strip().lower()


def parse_review_item(item: dict[str, Any]) -> dict[str, Any]:
    seed = item.get("seed") or {}
    cand = item.get("candidate") or {}
    artist = str(seed.get("artist", item.get("artist", ""))).strip()
    title = str(seed.get("title", item.get("title", ""))).strip()
    meters_raw = seed.get("meters", item.get("meters"))
    if meters_raw is None:
        raise ValueError("missing meters")
    if isinstance(meters_raw, dict):
        meters = {int(k): float(v) for k, v in meters_raw.items()}
    elif isinstance(meters_raw, str):
        meters = parse_meters(meters_raw)
    elif isinstance(meters_raw, list):
        meters = {int(x): 1.0 for x in meters_raw}
    else:
        raise ValueError("invalid meters format")

    video_id = str(cand.get("video_id", item.get("video_id", ""))).strip()
    if not artist or not title:
        raise ValueError("missing artist/title")
    if not video_id:
        raise ValueError("missing video_id")

    return {
        "artist": artist,
        "title": title,
        "meters": meters,
        "video_id": video_id,
    }


def sort_catalog(songs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        songs,
        key=lambda s: (
            min(s["meters"].keys()) if s["meters"] else 999,
            song_key(s["artist"], s["title"]),
        ),
    )


def main() -> None:
    args = parse_args()
    review_path = args.review_queue.resolve()
    catalog_path = args.catalog.resolve()
    do_backup = args.backup and not args.no_backup

    if not review_path.exists():
        print(f"ERROR: review queue file not found: {review_path}")
        sys.exit(1)
    if not catalog_path.exists():
        print(f"ERROR: catalog not found: {catalog_path}")
        sys.exit(1)

    catalog = load_catalog(catalog_path)
    review_items = load_review_items(review_path)

    accepted: list[dict[str, Any]] = []
    parse_errors = 0
    for item in review_items:
        status = str(item.get("status", "")).strip()
        if not is_status_match(status, args.status):
            continue
        try:
            accepted.append(parse_review_item(item))
        except Exception as exc:
            parse_errors += 1
            item_id = item.get("id", "?")
            print(f"  skip malformed review item {item_id}: {exc}")

    if not accepted:
        print("No accepted items to merge.")
        return

    by_song: dict[str, int] = {}
    by_video: dict[str, int] = {}
    for idx, song in enumerate(catalog):
        by_song[song_key(song["artist"], song["title"])] = idx
        by_video[song["video_id"]] = idx

    merged = 0
    replaced = 0
    skipped_existing_song = 0
    skipped_existing_video = 0
    duplicate_in_batch = 0
    seen_batch: set[str] = set()

    for item in accepted:
        skey = song_key(item["artist"], item["title"])
        vid = item["video_id"]
        if vid in seen_batch:
            duplicate_in_batch += 1
            continue
        seen_batch.add(vid)

        if vid in by_video:
            skipped_existing_video += 1
            continue

        if skey in by_song:
            if not args.replace_existing_song:
                skipped_existing_song += 1
                continue
            idx = by_song[skey]
            old_vid = catalog[idx]["video_id"]
            if old_vid in by_video:
                del by_video[old_vid]
            catalog[idx] = item
            by_song[skey] = idx
            by_video[vid] = idx
            replaced += 1
            continue

        catalog.append(item)
        idx = len(catalog) - 1
        by_song[skey] = idx
        by_video[vid] = idx
        merged += 1

    catalog = sort_catalog(catalog)

    print("Merge summary")
    print(f"  accepted in queue: {len(accepted)}")
    print(f"  merged new songs: {merged}")
    print(f"  replaced existing songs: {replaced}")
    print(f"  skipped existing songs: {skipped_existing_song}")
    print(f"  skipped existing videos: {skipped_existing_video}")
    print(f"  duplicates in accepted batch: {duplicate_in_batch}")
    if parse_errors:
        print(f"  malformed accepted entries: {parse_errors}")
    print(f"  final catalog size: {len(catalog)}")

    if args.dry_run:
        print("Dry run: catalog not written.")
        return

    if do_backup:
        ts = now_iso().replace(":", "").replace("-", "")
        backup_path = catalog_path.with_suffix(f".bak.{ts}.json")
        shutil.copy2(catalog_path, backup_path)
        print(f"Backup written: {backup_path}")

    save_catalog(catalog_path, catalog)
    print(f"Catalog updated: {catalog_path}")


if __name__ == "__main__":
    main()
