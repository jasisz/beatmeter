#!/usr/bin/env python3
"""Verify WIKIMETER review queue items against external sources.

This script enriches review items with evidence collected from:
- YouTube metadata (title + description via yt-dlp)
- Wikipedia search + page extracts

It does not silently mutate the main catalog. It writes an annotated review
queue JSON and can optionally apply status suggestions.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wikimeter_tools import PROJECT_ROOT, load_json, now_iso, parse_meters, save_json

DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "wikimeter" / "curation" / "review_queue.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "wikimeter" / "curation" / "review_queue_verified.json"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
DUCKDUCKGO_HTML = "https://duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

METER_TOKEN_RE = re.compile(r"\b(3|4|5|6|7|8|9|10|11|12)\s*/\s*(2|4|8|16)\b")
TIME_SIGNATURE_PHRASE_RE = re.compile(r"time signature|meter|metre", re.IGNORECASE)

RARE_METER_SET = {9, 11}


@dataclass
class Evidence:
    source: str
    source_url: str
    match_meters: list[int]
    has_time_signature_phrase: bool
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_url": self.source_url,
            "match_meters": self.match_meters,
            "has_time_signature_phrase": self.has_time_signature_phrase,
            "snippet": self.snippet,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify WIKIMETER review queue with source evidence")
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--status-filter",
        type=str,
        default="pending",
        help="Only process items with this status (default: pending, empty means all)",
    )
    parser.add_argument("--max-items", type=int, default=0, help="0 means all")
    parser.add_argument("--wiki-hits", type=int, default=3, help="Wikipedia hits to inspect per item")
    parser.add_argument("--web-hits", type=int, default=3, help="Non-Wikipedia web pages to inspect per item")
    parser.add_argument("--max-page-chars", type=int, default=15000, help="How much page text to parse")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--sleep", type=float, default=0.25, help="Pause between items")
    parser.add_argument(
        "--apply-suggestions",
        action="store_true",
        help="Write suggested status back to item.status in output JSON",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_review_items(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return {"generated_at": now_iso(), "source": str(path)}, [dict(x) for x in payload]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        meta = dict(payload.get("meta") or {})
        return meta, [dict(x) for x in payload["items"]]
    raise ValueError("review queue must be list[] or {'items': list[]}")


def request_json(url: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    raw = request_text(full_url, timeout_s=timeout_s)
    return json.loads(raw)


def request_text(url: str, timeout_s: float) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def wikipedia_search(query: str, limit: int, timeout_s: float) -> list[str]:
    payload = request_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": "1",
        },
        timeout_s=timeout_s,
    )
    items = payload.get("query", {}).get("search", []) or []
    out: list[str] = []
    for item in items:
        title = str(item.get("title", "")).strip()
        if title:
            out.append(title)
    return out


def wikipedia_extract(title: str, timeout_s: float) -> str:
    payload = request_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": "1",
            "exintro": "0",
            "format": "json",
            "utf8": "1",
        },
        timeout_s=timeout_s,
    )
    pages = payload.get("query", {}).get("pages", {}) or {}
    for page in pages.values():
        extract = str(page.get("extract", "")).strip()
        if extract:
            return extract
    return ""


def duckduckgo_search(query: str, limit: int, timeout_s: float) -> list[str]:
    params = {"q": query}
    url = f"{DUCKDUCKGO_HTML}?{urllib.parse.urlencode(params)}"
    page = request_text(url, timeout_s=timeout_s)
    hrefs = re.findall(r'class="result__a"[^>]+href="([^"]+)"', page)
    out: list[str] = []
    seen: set[str] = set()

    for raw_href in hrefs:
        href = html.unescape(raw_href)
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            target = qs.get("uddg", [""])[0]
            if target:
                href = urllib.parse.unquote(target)

        parsed = urllib.parse.urlparse(href)
        if parsed.scheme not in ("http", "https"):
            continue
        if not parsed.netloc:
            continue
        canonical = href.strip()
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
        if len(out) >= limit:
            break
    return out


def html_to_text(page: str) -> str:
    page = re.sub(r"(?is)<script.*?>.*?</script>", " ", page)
    page = re.sub(r"(?is)<style.*?>.*?</style>", " ", page)
    page = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", page)
    page = re.sub(r"(?is)<[^>]+>", " ", page)
    page = html.unescape(page)
    page = re.sub(r"\s+", " ", page).strip()
    return page


def yt_metadata(video_id: str, timeout_s: float) -> dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        url,
        "--dump-single-json",
        "--skip-download",
        "--quiet",
        "--no-warnings",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return {"error": "yt-dlp not found"}
    except subprocess.TimeoutExpired:
        return {"error": "yt-dlp timeout"}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or "yt-dlp failed"}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "invalid yt-dlp json"}
    return {
        "title": str(payload.get("title", "")).strip(),
        "description": str(payload.get("description", "")).strip(),
        "uploader": str(payload.get("uploader", "") or payload.get("channel", "")).strip(),
        "webpage_url": str(payload.get("webpage_url", url)).strip(),
    }


def _snippet(text: str, start: int, end: int, radius: int = 72) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    part = text[lo:hi].replace("\n", " ")
    part = re.sub(r"\s+", " ", part).strip()
    if lo > 0:
        part = "..." + part
    if hi < len(text):
        part = part + "..."
    return part


def extract_meter_evidence(text: str, source: str, source_url: str) -> list[Evidence]:
    if not text:
        return []
    out: list[Evidence] = []
    has_phrase = bool(TIME_SIGNATURE_PHRASE_RE.search(text))
    for match in METER_TOKEN_RE.finditer(text):
        meter = int(match.group(1))
        snip = _snippet(text, match.start(), match.end())
        out.append(
            Evidence(
                source=source,
                source_url=source_url,
                match_meters=[meter],
                has_time_signature_phrase=has_phrase,
                snippet=snip,
            )
        )
    return out


def collapse_evidence(evidence: list[Evidence], max_items: int = 6) -> list[Evidence]:
    """Deduplicate by (source, meter, snippet-prefix)."""
    out: list[Evidence] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for ev in evidence:
        key = (ev.source, tuple(ev.match_meters), ev.snippet[:90])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
        if len(out) >= max_items:
            break
    return out


def parse_target_meters(item: dict[str, Any]) -> dict[int, float]:
    seed = item.get("seed") or {}
    raw = seed.get("meters", item.get("meters"))
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {int(k): float(v) for k, v in raw.items()}
    if isinstance(raw, str):
        return parse_meters(raw)
    if isinstance(raw, list):
        return {int(x): 1.0 for x in raw}
    return {}


def evidence_summary(
    target_meters: set[int],
    evidence: list[Evidence],
) -> tuple[float, bool, str, str, str]:
    score = 0.0
    conflict = False
    overlap_hits = 0
    off_target_hits = 0

    for ev in evidence:
        matched = set(ev.match_meters)
        overlap = bool(matched & target_meters)
        if overlap:
            overlap_hits += 1
            if ev.source.startswith("wikipedia"):
                score += 1.4
            elif ev.source == "youtube_description":
                score += 0.7
            elif ev.source == "youtube_title":
                score += 0.25
            else:
                score += 0.4
            if ev.has_time_signature_phrase:
                score += 0.3
        else:
            off_target_hits += 1
            if ev.has_time_signature_phrase:
                score -= 0.35
                conflict = True
            else:
                score -= 0.1

    score = round(score, 3)
    if score >= 2.2:
        confidence = "high"
    elif score >= 1.0:
        confidence = "medium"
    else:
        confidence = "low"

    if not evidence:
        suggested = "pending"
        reason = "No meter evidence found in configured sources."
    elif len(target_meters) > 1:
        suggested = "pending"
        reason = "Poly meter candidate requires manual review."
    elif target_meters & RARE_METER_SET:
        suggested = "pending"
        reason = "Rare meter class requires manual review."
    elif confidence == "high" and not conflict:
        suggested = "approved"
        reason = f"Strong evidence match ({overlap_hits} matching hits)."
    elif conflict:
        suggested = "rejected"
        reason = f"Conflicting meter evidence ({off_target_hits} off-target hits)."
    else:
        suggested = "pending"
        reason = "Evidence is weak or ambiguous."

    return score, conflict, confidence, suggested, f"{reason} target={sorted(target_meters)}"


def verify_one_item(
    item: dict[str, Any],
    wiki_hits: int,
    web_hits: int,
    max_page_chars: int,
    timeout_s: float,
) -> tuple[list[Evidence], list[str]]:
    errors: list[str] = []
    evidence: list[Evidence] = []
    seed = item.get("seed") or {}
    cand = item.get("candidate") or {}
    artist = str(seed.get("artist", item.get("artist", ""))).strip()
    title = str(seed.get("title", item.get("title", ""))).strip()
    video_id = str(cand.get("video_id", item.get("video_id", ""))).strip()

    if video_id:
        meta = yt_metadata(video_id, timeout_s=timeout_s)
        if "error" in meta:
            errors.append(f"youtube: {meta['error']}")
        else:
            evidence.extend(
                extract_meter_evidence(
                    str(meta.get("title", "")),
                    source="youtube_title",
                    source_url=str(meta.get("webpage_url", "")),
                )
            )
            evidence.extend(
                extract_meter_evidence(
                    str(meta.get("description", ""))[:6000],
                    source="youtube_description",
                    source_url=str(meta.get("webpage_url", "")),
                )
            )
    else:
        errors.append("missing video_id")

    queries = []
    if artist and title:
        queries.append(f"{artist} {title} time signature")
        queries.append(f"{artist} {title} meter")
        queries.append(f"{artist} {title} sheet music time signature")
        queries.append(f"{artist} {title} rhythm analysis")
    elif title:
        queries.append(f"{title} time signature")

    wiki_titles: list[str] = []
    for query in queries:
        try:
            wiki_titles.extend(wikipedia_search(query=query, limit=wiki_hits, timeout_s=timeout_s))
        except Exception as exc:
            errors.append(f"wikipedia search failed: {exc}")
            break

    unique_titles: list[str] = []
    seen_titles: set[str] = set()
    for title_item in wiki_titles:
        key = title_item.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique_titles.append(title_item)
        if len(unique_titles) >= wiki_hits:
            break

    for page_title in unique_titles:
        try:
            extract = wikipedia_extract(page_title, timeout_s=timeout_s)
        except Exception as exc:
            errors.append(f"wikipedia extract failed: {exc}")
            continue
        if not extract:
            continue
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"
        ev = extract_meter_evidence(extract, source=f"wikipedia:{page_title}", source_url=page_url)
        evidence.extend(ev)

    web_urls: list[str] = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            found = duckduckgo_search(query=query, limit=max(1, web_hits), timeout_s=timeout_s)
        except Exception as exc:
            errors.append(f"web search failed: {exc}")
            break
        for url in found:
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc.lower()
            if "wikipedia.org" in host:
                continue
            if "youtube.com" in host or "youtu.be" in host:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            web_urls.append(url)
            if len(web_urls) >= web_hits:
                break
        if len(web_urls) >= web_hits:
            break

    for url in web_urls:
        try:
            raw_page = request_text(url, timeout_s=timeout_s)
        except Exception as exc:
            errors.append(f"web fetch failed ({url}): {exc}")
            continue
        text = html_to_text(raw_page)[:max_page_chars]
        if not text:
            continue
        host = urllib.parse.urlparse(url).netloc.lower()
        source = f"web:{host}"
        evidence.extend(extract_meter_evidence(text=text, source=source, source_url=url))

    return collapse_evidence(evidence), errors


def main() -> None:
    args = parse_args()
    review_path = args.review_queue.resolve()
    output_path = args.output.resolve()

    if not review_path.exists():
        raise FileNotFoundError(f"review queue not found: {review_path}")

    meta, items = load_review_items(review_path)
    selected: list[dict[str, Any]] = []
    status_filter = args.status_filter.strip().lower()
    for item in items:
        status = str(item.get("status", "")).strip().lower()
        if status_filter and status != status_filter:
            continue
        selected.append(item)
    if args.max_items > 0:
        selected = selected[: args.max_items]

    print(f"Review items total: {len(items)}")
    print(f"Items selected for verification: {len(selected)}")

    updates = 0
    approved = 0
    rejected = 0
    pending = 0
    with_errors = 0

    for idx, item in enumerate(selected, start=1):
        item_id = str(item.get("id", f"item_{idx}"))
        target = parse_target_meters(item)
        target_set = set(target.keys())
        print(f"[{idx}/{len(selected)}] verify {item_id} target={sorted(target_set)}")

        evidence, errors = verify_one_item(
            item=item,
            wiki_hits=max(1, args.wiki_hits),
            web_hits=max(1, args.web_hits),
            max_page_chars=max(1000, args.max_page_chars),
            timeout_s=max(2.0, args.timeout),
        )
        score, conflict, confidence, suggested_status, reason = evidence_summary(target_set, evidence)

        verification = {
            "checked_at": now_iso(),
            "target_meters": sorted(target_set),
            "confidence": confidence,
            "suggested_status": suggested_status,
            "score": score,
            "reason": reason,
            "errors": errors,
            "evidence": [ev.to_dict() for ev in evidence],
        }
        item["verification"] = verification
        updates += 1

        if errors:
            with_errors += 1
        if suggested_status == "approved":
            approved += 1
        elif suggested_status == "rejected":
            rejected += 1
        else:
            pending += 1

        if args.apply_suggestions:
            old_status = str(item.get("status", "pending"))
            item["status"] = suggested_status
            note = str(item.get("review_note", "")).strip()
            auto_note = f"[auto-verify] {old_status} -> {suggested_status}; confidence={confidence}; score={score}"
            item["review_note"] = f"{note} | {auto_note}" if note else auto_note

        if args.sleep > 0:
            time.sleep(args.sleep)

    out_payload = {
        "meta": {
            **meta,
            "verified_at": now_iso(),
            "verification_config": {
                "status_filter": args.status_filter,
                "max_items": args.max_items,
                "wiki_hits": args.wiki_hits,
                "web_hits": args.web_hits,
                "max_page_chars": args.max_page_chars,
                "timeout": args.timeout,
                "apply_suggestions": args.apply_suggestions,
            },
            "verification_summary": {
                "processed_items": len(selected),
                "updated_items": updates,
                "suggested_approved": approved,
                "suggested_pending": pending,
                "suggested_rejected": rejected,
                "items_with_errors": with_errors,
            },
        },
        "items": items,
    }

    print("Summary")
    print(f"  updated items: {updates}")
    print(f"  suggested approved: {approved}")
    print(f"  suggested pending: {pending}")
    print(f"  suggested rejected: {rejected}")
    print(f"  items with source errors: {with_errors}")

    if args.dry_run:
        print("Dry run: output not written.")
        return

    save_json(output_path, out_payload)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
