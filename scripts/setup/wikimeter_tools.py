#!/usr/bin/env python3
"""Shared helpers for WIKIMETER curation scripts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "scripts" / "setup" / "wikimeter.json"
Source = dict[str, Any]


def _clean_url(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def normalize_meters(meters: dict[Any, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in meters.items():
        meter = int(key)
        out[meter] = float(value)
    return out


def meters_for_save(meters: dict[int, float]) -> dict[str, float]:
    return {str(k): float(v) for k, v in sorted(meters.items())}


def parse_youtube_video_id(url: str) -> str:
    url = str(url).strip()
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0]
    if "youtube.com" in host:
        query = urllib.parse.parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0]
        path_parts = [p for p in parsed.path.split("/") if p]
        if "shorts" in path_parts:
            idx = path_parts.index("shorts")
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]
        if "embed" in path_parts:
            idx = path_parts.index("embed")
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]
    return ""


def source_from_youtube(video_id: str, url: str | None = None, priority: int = 100) -> Source:
    video_id = str(video_id).strip()
    if not video_id and url:
        video_id = parse_youtube_video_id(url)
    if not video_id:
        raise ValueError("youtube source missing video_id")
    url_s = _clean_url(url)
    return {
        "type": "youtube",
        "video_id": video_id,
        "url": url_s or f"https://www.youtube.com/watch?v={video_id}",
        "priority": int(priority),
    }


def normalize_source(raw: Source, index: int = 0) -> Source:
    source_type = str(raw.get("type", "")).strip().lower()
    if not source_type:
        source_type = "youtube" if raw.get("video_id") else "url"
    priority = int(raw.get("priority", 100 + index))

    if source_type == "youtube":
        src = source_from_youtube(
            video_id=str(raw.get("video_id", "")).strip(),
            url=_clean_url(raw.get("url", "")) or None,
            priority=priority,
        )
    else:
        url = _clean_url(raw.get("url", ""))
        if not url:
            raise ValueError(f"{source_type} source missing url")
        src = {
            "type": source_type,
            "url": url,
            "priority": priority,
        }

    for key in ("license", "note", "provider"):
        value = str(raw.get(key, "")).strip()
        if value:
            src[key] = value
    return src


def source_key(source: Source) -> str:
    source_type = str(source.get("type", "")).strip().lower()
    if source_type == "youtube":
        video_id = str(source.get("video_id", "")).strip() or parse_youtube_video_id(str(source.get("url", "")))
        return f"youtube:{video_id}"
    return f"{source_type}:{str(source.get('url', '')).strip().lower()}"


def normalize_sources(
    raw_sources: Any,
    fallback_video_id: str = "",
    fallback_url: str = "",
) -> list[Source]:
    seq = raw_sources if isinstance(raw_sources, list) else []
    out: list[Source] = []
    seen: set[str] = set()

    for idx, raw in enumerate(seq):
        if not isinstance(raw, dict):
            continue
        src = normalize_source(raw, index=idx)
        key = source_key(src)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)

    if not out and (fallback_video_id or fallback_url):
        src = source_from_youtube(
            video_id=fallback_video_id,
            url=fallback_url or None,
            priority=100,
        )
        out.append(src)

    out.sort(key=lambda s: (int(s.get("priority", 9999)), source_key(s)))
    return out


def sources_for_save(sources: list[Source]) -> list[Source]:
    out: list[Source] = []
    for src in sorted(sources, key=lambda s: (int(s.get("priority", 9999)), source_key(s))):
        normalized = normalize_source(src)
        item: Source = {
            "type": normalized["type"],
            "priority": int(normalized["priority"]),
        }
        if normalized["type"] == "youtube":
            item["video_id"] = normalized["video_id"]
        item["url"] = normalized["url"]
        for key in ("license", "note", "provider"):
            if key in normalized:
                item[key] = normalized[key]
        out.append(item)
    return out


def merge_sources(existing: list[Source], incoming: list[Source]) -> list[Source]:
    merged: dict[str, Source] = {}
    for src in normalize_sources(existing):
        merged[source_key(src)] = dict(src)
    for src in normalize_sources(incoming):
        key = source_key(src)
        if key in merged:
            base = merged[key]
            # Keep lower priority number (higher importance) and fill missing metadata.
            base["priority"] = min(int(base.get("priority", 9999)), int(src.get("priority", 9999)))
            for meta_key in ("license", "note", "provider"):
                if meta_key not in base and meta_key in src:
                    base[meta_key] = src[meta_key]
        else:
            merged[key] = dict(src)
    return sorted(merged.values(), key=lambda s: (int(s.get("priority", 9999)), source_key(s)))


def source_to_candidate(source: Source) -> Source:
    src = normalize_source(source)
    out: Source = {
        "type": src["type"],
        "url": src["url"],
    }
    if src["type"] == "youtube":
        out["video_id"] = src["video_id"]
    if "provider" in src:
        out["provider"] = src["provider"]
    if "license" in src:
        out["license"] = src["license"]
    return out


def youtube_video_ids_from_sources(sources: list[Source]) -> set[str]:
    out: set[str] = set()
    for src in normalize_sources(sources):
        if src.get("type") != "youtube":
            continue
        video_id = str(src.get("video_id", "")).strip() or parse_youtube_video_id(str(src.get("url", "")))
        if video_id:
            out.add(video_id)
    return out


def song_video_ids(song: dict[str, Any]) -> set[str]:
    return youtube_video_ids_from_sources(list(song.get("sources", [])))


def song_source_keys(song: dict[str, Any]) -> set[str]:
    return {source_key(src) for src in normalize_sources(song.get("sources", []))}


def preferred_source(song: dict[str, Any], source_types: list[str] | None = None) -> Source | None:
    allowed = {s.strip().lower() for s in source_types} if source_types else None
    for src in normalize_sources(song.get("sources", [])):
        if allowed and str(src.get("type", "")).lower() not in allowed:
            continue
        return src
    return None


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
                "sources": normalize_sources(
                    entry.get("sources"),
                    fallback_video_id=str(entry.get("video_id", "")).strip(),
                    fallback_url=str(entry.get("video_url", "")).strip(),
                ),
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
                "sources": sources_for_save(list(s.get("sources", []))),
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
                "source": source_to_candidate(
                    source_from_youtube(
                        video_id=video_id,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        priority=100 + idx,
                    )
                ),
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
