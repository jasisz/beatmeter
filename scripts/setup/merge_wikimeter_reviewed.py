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
    meters_to_tag,
    merge_sources,
    now_iso,
    parse_meters,
    parse_youtube_video_id,
    save_catalog,
    song_source_keys,
    song_video_ids,
    song_key,
    source_from_youtube,
    source_key,
    source_to_candidate,
    normalize_sources,
)

DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "wikimeter" / "review_queue.json"
DEFAULT_BLACKLIST = PROJECT_ROOT / "data" / "wikimeter" / "blacklist.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed candidates into wikimeter.json")
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
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

    sources = normalize_sources(
        raw_sources=(cand.get("sources") if isinstance(cand.get("sources"), list) else []),
        fallback_video_id="",
        fallback_url="",
    )
    if isinstance(cand.get("source"), dict):
        sources = merge_sources(sources, [source_to_candidate(cand["source"])])
    if isinstance(item.get("sources"), list):
        sources = merge_sources(sources, list(item.get("sources") or []))

    video_id = str(cand.get("video_id", item.get("video_id", ""))).strip()
    video_url = str(cand.get("video_url", item.get("video_url", ""))).strip()
    if not video_id and video_url:
        video_id = parse_youtube_video_id(video_url)
    if video_id:
        sources = merge_sources(sources, [source_from_youtube(video_id=video_id, url=video_url or None)])
    elif video_url and not sources:
        sources = normalize_sources([{"type": "url", "url": video_url}])

    if not artist or not title:
        raise ValueError("missing artist/title")
    if not sources:
        raise ValueError("missing source")

    return {
        "artist": artist,
        "title": title,
        "meters": meters,
        "sources": sources,
    }


def parse_blacklist_item(entry: dict[str, Any]) -> tuple[str, str]:
    artist = str(entry.get("artist", "")).strip()
    title = str(entry.get("title", "")).strip()
    meters_raw = entry.get("meters")
    if not artist or not title or meters_raw is None:
        raise ValueError("missing artist/title/meters")

    if isinstance(meters_raw, dict):
        meters = {int(k): float(v) for k, v in meters_raw.items()}
    elif isinstance(meters_raw, str):
        meters = parse_meters(meters_raw)
    elif isinstance(meters_raw, list):
        meters = {int(x): 1.0 for x in meters_raw}
    else:
        raise ValueError("invalid meters format")

    return (song_key(artist, title), meters_to_tag(meters))


def load_blacklist(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        print(f"WARNING: blacklist file not found: {path} (continuing without blacklist gate)")
        return set()

    payload = load_json(path)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        entries = payload["items"]
    else:
        raise ValueError("blacklist file must be list[] or {'items': list[]}")

    out: set[tuple[str, str]] = set()
    malformed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        try:
            out.add(parse_blacklist_item(entry))
        except Exception:
            malformed += 1

    print(f"Blacklist entries loaded: {len(out)}")
    if malformed:
        print(f"  malformed blacklist entries ignored: {malformed}")
    return out


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
    blacklist_path = args.blacklist.resolve()
    do_backup = args.backup and not args.no_backup

    if not review_path.exists():
        print(f"ERROR: review queue file not found: {review_path}")
        sys.exit(1)
    if not catalog_path.exists():
        print(f"ERROR: catalog not found: {catalog_path}")
        sys.exit(1)

    catalog = load_catalog(catalog_path)
    review_items = load_review_items(review_path)
    blacklist = load_blacklist(blacklist_path)

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
    by_source: dict[str, int] = {}
    for idx, song in enumerate(catalog):
        by_song[song_key(song["artist"], song["title"])] = idx
        for video_id in song_video_ids(song):
            by_video[video_id] = idx
        for src_key in song_source_keys(song):
            by_source[src_key] = idx

    merged = 0
    replaced = 0
    appended_sources = 0
    skipped_existing_song = 0
    skipped_existing_video = 0
    skipped_existing_source = 0
    skipped_blacklist = 0
    duplicate_in_batch = 0
    seen_batch_sources: set[str] = set()

    for item in accepted:
        skey = song_key(item["artist"], item["title"])
        meter_tag = meters_to_tag(item["meters"])
        item_source_keys = {source_key(src) for src in item["sources"]}
        item_video_ids = song_video_ids(item)

        if (skey, meter_tag) in blacklist:
            skipped_blacklist += 1
            continue

        if any(src_key in seen_batch_sources for src_key in item_source_keys):
            duplicate_in_batch += 1
            continue

        if skey in by_song:
            idx = by_song[skey]
            current = catalog[idx]
            current_sources = list(current.get("sources", []))
            merged_song_sources = merge_sources(current_sources, item["sources"])
            added = max(0, len(merged_song_sources) - len(current_sources))

            if args.replace_existing_song:
                catalog[idx] = {
                    "artist": item["artist"],
                    "title": item["title"],
                    "meters": item["meters"],
                    "sources": merged_song_sources,
                }
                replaced += 1
            elif added > 0:
                catalog[idx]["sources"] = merged_song_sources
                appended_sources += added
            else:
                skipped_existing_song += 1

            by_song[skey] = idx
            for video_id in song_video_ids(catalog[idx]):
                by_video[video_id] = idx
            for src_key in song_source_keys(catalog[idx]):
                by_source[src_key] = idx
            seen_batch_sources.update(item_source_keys)
            continue

        if any(video_id in by_video for video_id in item_video_ids):
            skipped_existing_video += 1
            continue
        if any(src_key in by_source for src_key in item_source_keys):
            skipped_existing_source += 1
            continue

        catalog.append(item)
        idx = len(catalog) - 1
        by_song[skey] = idx
        for video_id in item_video_ids:
            by_video[video_id] = idx
        for src_key in item_source_keys:
            by_source[src_key] = idx
        seen_batch_sources.update(item_source_keys)
        merged += 1

    catalog = sort_catalog(catalog)

    print("Merge summary")
    print(f"  accepted in queue: {len(accepted)}")
    print(f"  merged new songs: {merged}")
    print(f"  replaced existing songs: {replaced}")
    print(f"  appended sources to existing songs: {appended_sources}")
    print(f"  skipped existing songs: {skipped_existing_song}")
    print(f"  skipped existing videos: {skipped_existing_video}")
    print(f"  skipped existing sources: {skipped_existing_source}")
    print(f"  skipped by blacklist: {skipped_blacklist}")
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
