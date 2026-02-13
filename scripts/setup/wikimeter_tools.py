#!/usr/bin/env python3
"""Shared helpers for WIKIMETER curation scripts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "scripts" / "setup" / "wikimeter.json"


def normalize_meters(meters: dict[Any, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in meters.items():
        meter = int(key)
        out[meter] = float(value)
    return out


def meters_for_save(meters: dict[int, float]) -> dict[str, float]:
    return {str(k): float(v) for k, v in sorted(meters.items())}


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    songs: list[dict[str, Any]] = []
    for entry in raw:
        songs.append(
            {
                "artist": str(entry["artist"]).strip(),
                "title": str(entry["title"]).strip(),
                "meters": normalize_meters(entry["meters"]),
                "video_id": str(entry["video_id"]).strip(),
            }
        )
    return songs


def save_catalog(path: Path, songs: list[dict[str, Any]]) -> None:
    serializable: list[dict[str, Any]] = []
    for s in songs:
        serializable.append(
            {
                "artist": s["artist"],
                "title": s["title"],
                "meters": meters_for_save(s["meters"]),
                "video_id": s["video_id"],
            }
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def song_key(artist: str, title: str) -> str:
    return f"{normalize_text(artist)}::{normalize_text(title)}"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_meters(s: str) -> dict[int, float]:
    """Parse meter string like '5', '7:1.0', '3:0.8,4:0.7'."""
    meters: dict[int, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            meter_s, weight_s = part.split(":", 1)
            meters[int(meter_s.strip())] = float(weight_s.strip())
        else:
            meters[int(part)] = 1.0
    if not meters:
        raise ValueError(f"invalid meters: {s!r}")
    return meters


def meters_to_tag(meters: dict[int, float]) -> str:
    out: list[str] = []
    for meter, weight in sorted(meters.items()):
        if abs(weight - 1.0) < 1e-9:
            out.append(str(meter))
        else:
            out.append(f"{meter}:{weight:g}")
    return ",".join(out)


def load_seed_file(path: Path) -> list[dict[str, Any]]:
    """Load seed records from .json / .jsonl / .csv."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = load_json(path)
        if not isinstance(raw, list):
            raise ValueError("seed .json must be a list")
        return [dict(x) for x in raw]
    if suffix == ".jsonl":
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    if suffix == ".csv":
        out = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out.append(dict(row))
        return out
    raise ValueError(f"unsupported seed file: {path}")


def ytsearch(query: str, limit: int = 5, timeout_s: int = 45) -> list[dict[str, Any]]:
    """Run yt-dlp search and return normalized entry list."""
    cmd = [
        "yt-dlp",
        f"ytsearch{limit}:{query}",
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        "--quiet",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp not found. Install with `pip install yt-dlp`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"yt-dlp search timeout for query: {query}") from exc

    if proc.returncode != 0:
        err = proc.stderr.strip() or "unknown yt-dlp error"
        raise RuntimeError(f"yt-dlp search failed for query {query!r}: {err}")

    payload = json.loads(proc.stdout)
    raw_entries = payload.get("entries", []) or []
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries, start=1):
        video_id = str(entry.get("id", "")).strip()
        if not video_id:
            continue
        out.append(
            {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_title": str(entry.get("title", "")).strip(),
                "uploader": str(entry.get("uploader", "") or entry.get("channel", "")).strip(),
                "duration_s": int(entry.get("duration") or 0),
                "search_rank": idx,
            }
        )
    return out


def candidate_score(
    artist: str,
    title: str,
    meters: dict[int, float],
    search_rank: int,
    video_title: str,
    uploader: str,
    duration_s: int,
) -> float:
    """Heuristic ranking score for candidate review ordering."""
    a = normalize_text(artist)
    t = normalize_text(title)
    vt = normalize_text(video_title)
    up = normalize_text(uploader)
    meter_tokens = [f"{m}/" for m in meters]

    score = 1.0 / max(search_rank, 1)

    if a and a in vt:
        score += 0.4
    title_words = [w for w in t.split(" ") if len(w) >= 3]
    if title_words:
        overlap = sum(1 for w in title_words if w in vt)
        score += min(0.5, 0.1 * overlap)
    if any(token in vt for token in meter_tokens):
        score += 0.15

    trusted_uploader_hints = ("official", "topic", "records", "vevo")
    if any(h in up for h in trusted_uploader_hints):
        score += 0.1

    bad_hints = (
        "live",
        "cover",
        "karaoke",
        "remix",
        "slowed",
        "nightcore",
        "8d",
        "reverb",
    )
    if any(h in vt for h in bad_hints):
        score -= 0.2

    if duration_s <= 0:
        score -= 0.2
    elif duration_s < 50:
        score -= 0.15
    elif duration_s > 900:
        score -= 0.15
    else:
        score += 0.05

    return round(score, 4)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
