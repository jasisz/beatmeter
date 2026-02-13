#!/usr/bin/env python3
"""Build human-review queue from collected WIKIMETER candidates."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from wikimeter_tools import DEFAULT_CATALOG, PROJECT_ROOT, load_catalog, load_json, now_iso, parse_meters, save_json, song_key

DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "wikimeter" / "curation" / "candidates.json"
DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "wikimeter" / "curation" / "review_queue.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review queue from collected candidates")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--top-per-song", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--include-existing-song", action="store_true")
    parser.add_argument("--include-existing-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_candidate_items(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [dict(x) for x in payload["items"]]
    raise ValueError("candidate file must be either list[] or {'items': list[]}")


def normalize_candidate(item: dict[str, Any]) -> dict[str, Any]:
    meters_raw = item.get("meters")
    if isinstance(meters_raw, dict):
        meters = {int(k): float(v) for k, v in meters_raw.items()}
    elif isinstance(meters_raw, str):
        meters = parse_meters(meters_raw)
    else:
        raise ValueError("candidate missing meters")

    return {
        "artist": str(item["artist"]).strip(),
        "title": str(item["title"]).strip(),
        "meters": meters,
        "query": str(item.get("query", "")).strip(),
        "video_id": str(item["video_id"]).strip(),
        "video_url": str(item.get("video_url", "")).strip(),
        "video_title": str(item.get("video_title", "")).strip(),
        "uploader": str(item.get("uploader", "")).strip(),
        "duration_s": int(item.get("duration_s") or 0),
        "search_rank": int(item.get("search_rank") or 999),
        "score": float(item.get("score") or 0.0),
        "flags": list(item.get("flags") or []),
    }


def meter_tag(meters: dict[int, float]) -> str:
    out: list[str] = []
    for meter, weight in sorted(meters.items()):
        if abs(weight - 1.0) < 1e-9:
            out.append(str(meter))
        else:
            out.append(f"{meter}:{weight:g}")
    return ",".join(out)


def main() -> None:
    args = parse_args()
    candidates_path = args.candidates.resolve()
    catalog_path = args.catalog.resolve()
    output_path = args.output.resolve()

    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        sys.exit(1)
    if not catalog_path.exists():
        print(f"ERROR: catalog not found: {catalog_path}")
        sys.exit(1)

    catalog = load_catalog(catalog_path)
    existing_song_keys = {song_key(s["artist"], s["title"]) for s in catalog}
    existing_video_ids = {s["video_id"] for s in catalog}

    raw_items = load_candidate_items(candidates_path)
    print(f"Loaded candidates: {len(raw_items)}")

    normalized: list[dict[str, Any]] = []
    bad_items = 0
    for item in raw_items:
        try:
            normalized.append(normalize_candidate(item))
        except Exception:
            bad_items += 1
    if bad_items:
        print(f"Skipped malformed candidates: {bad_items}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in normalized:
        grouped[song_key(item["artist"], item["title"])].append(item)

    queue: list[dict[str, Any]] = []
    skipped_existing_song = 0
    skipped_existing_video = 0
    skipped_low_score = 0
    skipped_duplicate_video = 0
    queue_video_ids: set[str] = set()

    for skey, group in grouped.items():
        group_sorted = sorted(
            group,
            key=lambda x: (x["score"], -x["search_rank"]),
            reverse=True,
        )
        taken = 0
        for item in group_sorted:
            if item["score"] < args.min_score:
                skipped_low_score += 1
                continue

            if skey in existing_song_keys and not args.include_existing_song:
                skipped_existing_song += 1
                break

            if item["video_id"] in existing_video_ids and not args.include_existing_video:
                skipped_existing_video += 1
                continue

            if item["video_id"] in queue_video_ids:
                skipped_duplicate_video += 1
                continue

            review_flags = list(item["flags"])
            if skey in existing_song_keys:
                review_flags.append("existing_song")
            if item["video_id"] in existing_video_ids:
                review_flags.append("existing_video")
            if item["search_rank"] > 1:
                review_flags.append("not_top_result")

            queue.append(
                {
                    "id": f"review_{len(queue) + 1:05d}",
                    "status": "pending",
                    "review_note": "",
                    "seed": {
                        "artist": item["artist"],
                        "title": item["title"],
                        "meters": meter_tag(item["meters"]),
                        "query": item["query"],
                    },
                    "candidate": {
                        "video_id": item["video_id"],
                        "video_url": item["video_url"],
                        "video_title": item["video_title"],
                        "uploader": item["uploader"],
                        "duration_s": item["duration_s"],
                        "search_rank": item["search_rank"],
                    },
                    "score": item["score"],
                    "flags": sorted(set(review_flags)),
                    "created_at": now_iso(),
                }
            )
            queue_video_ids.add(item["video_id"])
            taken += 1
            if taken >= args.top_per_song:
                break

    queue.sort(key=lambda x: x["score"], reverse=True)
    payload = {
        "meta": {
            "generated_at": now_iso(),
            "catalog": str(catalog_path),
            "candidates_file": str(candidates_path),
            "total_candidates": len(raw_items),
            "queue_size": len(queue),
            "top_per_song": args.top_per_song,
            "min_score": args.min_score,
            "skipped_existing_song": skipped_existing_song,
            "skipped_existing_video": skipped_existing_video,
            "skipped_low_score": skipped_low_score,
            "skipped_duplicate_video": skipped_duplicate_video,
        },
        "items": queue,
    }

    print("Summary")
    print(f"  review queue size: {len(queue)}")
    print(f"  skipped existing song: {skipped_existing_song}")
    print(f"  skipped existing video: {skipped_existing_video}")
    print(f"  skipped low score: {skipped_low_score}")
    print(f"  skipped duplicate video: {skipped_duplicate_video}")

    if args.dry_run:
        print("Dry run: output not written.")
        return

    save_json(output_path, payload)
    print(f"Saved: {output_path}")
    print("")
    print("How to review")
    print("  1) Open JSON and set status=approved/rejected per item.")
    print("  2) Optionally add review_note.")
    print("  3) Run merge_wikimeter_reviewed.py to update catalog.")


if __name__ == "__main__":
    main()
