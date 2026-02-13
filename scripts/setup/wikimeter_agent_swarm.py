#!/usr/bin/env python3
"""Multi-agent review orchestrator for WIKIMETER queue.

Agents:
- scout: gathers evidence from sources
- skeptic: tries to find conflicting meter evidence
- arbiter: combines scout + skeptic into consensus status
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
from pathlib import Path
from typing import Any

import verify_wikimeter_sources as verify

DEFAULT_REVIEW_QUEUE = verify.DEFAULT_REVIEW_QUEUE
DEFAULT_OUTPUT = verify.PROJECT_ROOT / "data" / "wikimeter" / "curation" / "review_queue_swarm.json"
CANON_METERS = [3, 4, 5, 7, 9, 11]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scout/skeptic/arbiter agent swarm for WIKIMETER review")
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status-filter", type=str, default="pending")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--scout-wiki-hits", type=int, default=3)
    parser.add_argument("--scout-web-hits", type=int, default=3)
    parser.add_argument("--skeptic-web-hits", type=int, default=4)
    parser.add_argument("--max-page-chars", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument(
        "--apply-consensus",
        action="store_true",
        help="Apply arbiter consensus into item.status in output",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def item_identity(item: dict[str, Any]) -> tuple[str, str, str]:
    seed = item.get("seed") or {}
    cand = item.get("candidate") or {}
    artist = str(seed.get("artist", item.get("artist", ""))).strip()
    title = str(seed.get("title", item.get("title", ""))).strip()
    video_id = str(cand.get("video_id", item.get("video_id", ""))).strip()
    return artist, title, video_id


def run_skeptic_agent(
    item: dict[str, Any],
    target_meters: set[int],
    web_hits: int,
    max_page_chars: int,
    timeout_s: float,
) -> tuple[list[verify.Evidence], bool, str, list[str]]:
    """Search for contradicting evidence. Returns (evidence, conflict, reason, errors)."""
    errors: list[str] = []
    evidence: list[verify.Evidence] = []
    artist, title, _ = item_identity(item)

    alt_meters = [m for m in CANON_METERS if m not in target_meters]
    if not artist and not title:
        return [], False, "No artist/title fields for skeptic search.", []

    queries: list[str] = []
    base = f"{artist} {title}".strip()
    for meter in alt_meters[:4]:
        queries.append(f"{base} {meter}/4 time signature")
    queries.append(f"{base} time signature")

    found_urls: list[str] = []
    seen: set[str] = set()
    for q in queries:
        try:
            urls = verify.duckduckgo_search(q, limit=max(1, web_hits), timeout_s=timeout_s)
        except Exception as exc:
            errors.append(f"skeptic search failed: {exc}")
            break
        for url in urls:
            host = urllib.parse.urlparse(url).netloc.lower()
            if "wikipedia.org" in host:
                continue
            if "youtube.com" in host or "youtu.be" in host:
                continue
            if url in seen:
                continue
            seen.add(url)
            found_urls.append(url)
            if len(found_urls) >= web_hits:
                break
        if len(found_urls) >= web_hits:
            break

    for url in found_urls:
        try:
            page = verify.request_text(url, timeout_s=timeout_s)
            text = verify.html_to_text(page)[:max_page_chars]
        except Exception as exc:
            errors.append(f"skeptic fetch failed ({url}): {exc}")
            continue
        if not text:
            continue
        host = urllib.parse.urlparse(url).netloc.lower()
        evidence.extend(verify.extract_meter_evidence(text, source=f"skeptic_web:{host}", source_url=url))

    evidence = verify.collapse_evidence(evidence, max_items=6)
    support_hits = 0
    conflict_hits = 0
    for ev in evidence:
        matched = set(ev.match_meters)
        if matched & target_meters:
            if ev.has_time_signature_phrase:
                support_hits += 1
        elif ev.has_time_signature_phrase:
            conflict_hits += 1

    conflict = conflict_hits >= 2 and conflict_hits >= support_hits
    if conflict:
        reason = f"Skeptic found conflicting mentions ({conflict_hits}) >= support ({support_hits})."
    elif evidence:
        reason = f"Skeptic saw no strong conflict (support={support_hits}, off-target={conflict_hits})."
    else:
        reason = "Skeptic found no usable extra evidence."
    return evidence, conflict, reason, errors


def run_arbiter(
    target_meters: set[int],
    scout_confidence: str,
    scout_suggested: str,
    scout_conflict: bool,
    skeptic_conflict: bool,
) -> tuple[str, str]:
    if not target_meters:
        return "pending", "Missing target meter labels."
    if len(target_meters) > 1:
        return "pending", "Poly meter requires manual decision."
    if target_meters & verify.RARE_METER_SET:
        return "pending", "Rare class requires manual decision."
    if skeptic_conflict and scout_suggested == "approved":
        return "pending", "Scout positive but skeptic detected conflict."
    if skeptic_conflict and scout_confidence == "low":
        return "rejected", "Low-confidence scout and skeptic conflict."
    if scout_conflict and scout_confidence == "low":
        return "rejected", "Scout evidence itself is conflicting and weak."
    if scout_suggested == "approved" and scout_confidence == "high" and not skeptic_conflict:
        return "approved", "High-confidence scout evidence without skeptic conflict."
    return "pending", "Insufficient consensus for auto-approval."


def main() -> None:
    args = parse_args()
    review_path = args.review_queue.resolve()
    output_path = args.output.resolve()
    if not review_path.exists():
        raise FileNotFoundError(f"review queue not found: {review_path}")

    meta, items = verify.load_review_items(review_path)
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
    print(f"Items selected for swarm: {len(selected)}")

    consensus_counts = {"approved": 0, "pending": 0, "rejected": 0}
    with_errors = 0

    for idx, item in enumerate(selected, start=1):
        item_id = str(item.get("id", f"item_{idx}"))
        target = verify.parse_target_meters(item)
        target_set = set(target.keys())
        print(f"[{idx}/{len(selected)}] swarm {item_id} target={sorted(target_set)}")

        scout_evidence, scout_errors = verify.verify_one_item(
            item=item,
            wiki_hits=max(1, args.scout_wiki_hits),
            web_hits=max(1, args.scout_web_hits),
            max_page_chars=max(1000, args.max_page_chars),
            timeout_s=max(2.0, args.timeout),
        )
        scout_score, scout_conflict, scout_confidence, scout_suggested, scout_reason = verify.evidence_summary(
            target_set,
            scout_evidence,
        )

        skeptic_evidence, skeptic_conflict, skeptic_reason, skeptic_errors = run_skeptic_agent(
            item=item,
            target_meters=target_set,
            web_hits=max(1, args.skeptic_web_hits),
            max_page_chars=max(1000, args.max_page_chars),
            timeout_s=max(2.0, args.timeout),
        )

        consensus_status, consensus_reason = run_arbiter(
            target_meters=target_set,
            scout_confidence=scout_confidence,
            scout_suggested=scout_suggested,
            scout_conflict=scout_conflict,
            skeptic_conflict=skeptic_conflict,
        )
        consensus_counts[consensus_status] += 1

        all_errors = scout_errors + skeptic_errors
        if all_errors:
            with_errors += 1

        item["agent_reviews"] = {
            "checked_at": verify.now_iso(),
            "target_meters": sorted(target_set),
            "scout": {
                "confidence": scout_confidence,
                "suggested_status": scout_suggested,
                "score": scout_score,
                "conflict": scout_conflict,
                "reason": scout_reason,
                "errors": scout_errors,
                "evidence": [ev.to_dict() for ev in scout_evidence],
            },
            "skeptic": {
                "conflict": skeptic_conflict,
                "reason": skeptic_reason,
                "errors": skeptic_errors,
                "evidence": [ev.to_dict() for ev in skeptic_evidence],
            },
            "arbiter": {
                "consensus_status": consensus_status,
                "reason": consensus_reason,
            },
        }

        if args.apply_consensus:
            old_status = str(item.get("status", "pending"))
            item["status"] = consensus_status
            note = str(item.get("review_note", "")).strip()
            auto_note = f"[swarm] {old_status} -> {consensus_status}; {consensus_reason}"
            item["review_note"] = f"{note} | {auto_note}" if note else auto_note

        if args.sleep > 0:
            time.sleep(args.sleep)

    out_payload = {
        "meta": {
            **meta,
            "swarm_checked_at": verify.now_iso(),
            "swarm_config": {
                "status_filter": args.status_filter,
                "max_items": args.max_items,
                "scout_wiki_hits": args.scout_wiki_hits,
                "scout_web_hits": args.scout_web_hits,
                "skeptic_web_hits": args.skeptic_web_hits,
                "max_page_chars": args.max_page_chars,
                "timeout": args.timeout,
                "apply_consensus": args.apply_consensus,
            },
            "swarm_summary": {
                "processed_items": len(selected),
                "consensus_approved": consensus_counts["approved"],
                "consensus_pending": consensus_counts["pending"],
                "consensus_rejected": consensus_counts["rejected"],
                "items_with_errors": with_errors,
            },
        },
        "items": items,
    }

    print("Summary")
    print(f"  processed: {len(selected)}")
    print(f"  consensus approved: {consensus_counts['approved']}")
    print(f"  consensus pending: {consensus_counts['pending']}")
    print(f"  consensus rejected: {consensus_counts['rejected']}")
    print(f"  items with source errors: {with_errors}")

    if args.dry_run:
        print("Dry run: output not written.")
        return

    verify.save_json(output_path, out_payload)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
