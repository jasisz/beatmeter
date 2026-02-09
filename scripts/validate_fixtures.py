#!/usr/bin/env python3
"""Validate fixture audio files: load, analyze, compare with expected meter.

For each fixture file:
- Verifies the file loads correctly with librosa
- Checks duration is within 10-180s
- Runs the analysis engine
- Compares detected meter with expected (from FIXTURE_CATALOGUE)
- Outputs a validation report (JSON + terminal summary)

Usage:
    python scripts/validate_fixtures.py                    # validate all
    python scripts/validate_fixtures.py --category waltz   # one category
    python scripts/validate_fixtures.py --output report.json
    python scripts/validate_fixtures.py --mismatches-only  # show only problems
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Suppress noisy logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
for _name in ("beatmeter", "numba", "madmom", "BeatNet", "PySoundFile"):
    logging.getLogger(_name).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.audio.loader import load_audio
from beatmeter.audio.preprocessing import preprocess
from beatmeter.config import settings


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


@dataclass
class ValidationResult:
    filename: str
    category: str
    loads_ok: bool
    duration: float
    duration_ok: bool
    detected_meter: str
    detected_bpm: float
    expected_meters: list[str]
    expected_bpm_range: list[float]
    meter_match: str  # MATCH, MISMATCH, UNKNOWN, ERROR
    tempo_match: str  # MATCH, MISMATCH, UNKNOWN, ERROR
    elapsed: float
    notes: str = ""


def load_catalogue() -> dict:
    """Import FIXTURE_CATALOGUE from benchmark.py."""
    benchmark_path = Path(__file__).resolve().parent.parent / "tests" / "benchmark.py"
    # Use exec to load just the catalogue dict
    namespace = {}
    with open(benchmark_path) as f:
        content = f.read()

    # Extract the FIXTURE_CATALOGUE block
    import re
    match = re.search(
        r"^FIXTURE_CATALOGUE.*?^}",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if match:
        exec(match.group(), namespace)
        return namespace.get("FIXTURE_CATALOGUE", {})

    # Fallback: import directly
    sys.path.insert(0, str(benchmark_path.parent))
    from benchmark import FIXTURE_CATALOGUE
    return FIXTURE_CATALOGUE


def validate_file(engine: AnalysisEngine, filepath: Path,
                  catalogue: dict, max_duration: float = 30.0) -> ValidationResult:
    """Validate a single fixture file."""
    fname = filepath.name
    cat_entry = catalogue.get(fname)

    category = cat_entry[0] if cat_entry else "unknown"
    expected_meters = cat_entry[1] if cat_entry else []
    expected_bpm_range = cat_entry[2] if cat_entry else (0, 999)

    expected_meters_str = [f"{m[0]}/{m[1]}" for m in expected_meters]

    # Step 1: Try to load
    t0 = time.monotonic()
    try:
        audio, sr = load_audio(str(filepath), sr=settings.sample_rate)
        audio = preprocess(audio, sr)
        loads_ok = True
    except Exception as e:
        return ValidationResult(
            filename=fname, category=category,
            loads_ok=False, duration=0, duration_ok=False,
            detected_meter="N/A", detected_bpm=0,
            expected_meters=expected_meters_str,
            expected_bpm_range=list(expected_bpm_range),
            meter_match="ERROR", tempo_match="ERROR",
            elapsed=time.monotonic() - t0,
            notes=f"Load error: {e}",
        )

    # Step 2: Check duration
    duration = len(audio) / sr
    duration_ok = 10.0 <= duration <= 180.0

    # Step 3: Run engine (clip to max_duration for speed)
    max_samples = int(max_duration * sr)
    analysis_audio = audio[:max_samples] if len(audio) > max_samples else audio

    try:
        result = engine.analyze_audio(analysis_audio, sr)
    except Exception as e:
        return ValidationResult(
            filename=fname, category=category,
            loads_ok=True, duration=round(duration, 1), duration_ok=duration_ok,
            detected_meter="N/A", detected_bpm=0,
            expected_meters=expected_meters_str,
            expected_bpm_range=list(expected_bpm_range),
            meter_match="ERROR", tempo_match="ERROR",
            elapsed=time.monotonic() - t0,
            notes=f"Analysis error: {e}",
        )

    elapsed = time.monotonic() - t0

    # Step 4: Compare meter
    detected_meter = "N/A"
    meter_match = "UNKNOWN"
    if result.meter_hypotheses:
        top = result.meter_hypotheses[0]
        detected_meter = f"{top.numerator}/{top.denominator}"
        if expected_meters:
            match_found = any(
                top.numerator == m[0] and top.denominator == m[1]
                for m in expected_meters
            )
            meter_match = "MATCH" if match_found else "MISMATCH"
        else:
            meter_match = "UNKNOWN"

    # Step 5: Compare tempo
    detected_bpm = result.tempo.bpm
    if expected_bpm_range[0] > 0:
        tempo_match = ("MATCH" if expected_bpm_range[0] <= detected_bpm <= expected_bpm_range[1]
                       else "MISMATCH")
    else:
        tempo_match = "UNKNOWN"

    notes = ""
    if not duration_ok:
        notes = f"Duration {duration:.1f}s outside 10-180s range"

    return ValidationResult(
        filename=fname, category=category,
        loads_ok=True, duration=round(duration, 1), duration_ok=duration_ok,
        detected_meter=detected_meter, detected_bpm=round(detected_bpm, 1),
        expected_meters=expected_meters_str,
        expected_bpm_range=list(expected_bpm_range),
        meter_match=meter_match, tempo_match=tempo_match,
        elapsed=round(elapsed, 2), notes=notes,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate fixture audio files")
    parser.add_argument("--category", type=str, default=None,
                        help="Validate only this category")
    parser.add_argument("--output", type=str,
                        default="scripts/validation_report.json",
                        help="Output JSON file path")
    parser.add_argument("--mismatches-only", action="store_true",
                        help="Show only mismatches and errors")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Max seconds to analyze per file (default: 30)")
    args = parser.parse_args()

    catalogue = load_catalogue()

    if not FIXTURES_DIR.exists():
        print(f"Fixtures directory not found: {FIXTURES_DIR}")
        sys.exit(1)

    # Collect files to validate
    files = sorted(
        f for f in FIXTURES_DIR.iterdir()
        if f.suffix in (".ogg", ".oga", ".wav", ".mp3", ".flac", ".opus")
    )

    if args.category:
        # Filter to files in this category
        cat_files = {fname for fname, (cat, _, _) in catalogue.items()
                     if cat == args.category}
        files = [f for f in files if f.name in cat_files]

    print(f"{'='*80}")
    print(f"  FIXTURE VALIDATION - {len(files)} files")
    print(f"{'='*80}")

    engine = AnalysisEngine()
    results: list[ValidationResult] = []

    for i, filepath in enumerate(files, 1):
        sys.stdout.write(f"\r  Validating {i}/{len(files)}: {filepath.name[:50]:50s}")
        sys.stdout.flush()
        result = validate_file(engine, filepath, catalogue, args.max_duration)
        results.append(result)

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Print results table
    print(f"\n  {'status':10s} {'meter':8s} {'tempo':8s} {'dur':>5s} {'detected':10s} "
          f"{'expected':20s} {'filename'}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*5} {'-'*10} {'-'*20} {'-'*40}")

    for r in results:
        if args.mismatches_only and r.meter_match == "MATCH" and r.tempo_match == "MATCH":
            continue

        status = "OK" if r.meter_match == "MATCH" and r.tempo_match == "MATCH" else r.meter_match
        meter_icon = "OK" if r.meter_match == "MATCH" else r.meter_match
        tempo_icon = "OK" if r.tempo_match == "MATCH" else r.tempo_match

        print(f"  {status:10s} {meter_icon:8s} {tempo_icon:8s} {r.duration:5.1f} "
              f"{r.detected_meter:10s} {','.join(r.expected_meters):20s} {r.filename}")
        if r.notes:
            print(f"    NOTE: {r.notes}")

    # Summary
    total = len(results)
    meter_match = sum(1 for r in results if r.meter_match == "MATCH")
    meter_mismatch = sum(1 for r in results if r.meter_match == "MISMATCH")
    meter_error = sum(1 for r in results if r.meter_match == "ERROR")
    meter_unknown = sum(1 for r in results if r.meter_match == "UNKNOWN")
    tempo_match = sum(1 for r in results if r.tempo_match == "MATCH")
    duration_ok = sum(1 for r in results if r.duration_ok)
    loads_ok = sum(1 for r in results if r.loads_ok)

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total files:       {total}")
    print(f"  Loads OK:          {loads_ok}/{total}")
    print(f"  Duration 10-180s:  {duration_ok}/{total}")
    print(f"  Meter MATCH:       {meter_match}/{total}")
    print(f"  Meter MISMATCH:    {meter_mismatch}/{total} (review ground truth)")
    print(f"  Meter ERROR:       {meter_error}/{total}")
    print(f"  Meter UNKNOWN:     {meter_unknown}/{total}")
    print(f"  Tempo MATCH:       {tempo_match}/{total}")

    # Per-category summary
    by_cat: dict[str, list[ValidationResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    print(f"\n  {'category':18s} {'total':>5s} {'match':>6s} {'mismatch':>8s}")
    print(f"  {'-'*18} {'-'*5} {'-'*6} {'-'*8}")
    for cat in sorted(by_cat.keys()):
        group = by_cat[cat]
        n = len(group)
        m = sum(1 for r in group if r.meter_match == "MATCH")
        mm = sum(1 for r in group if r.meter_match == "MISMATCH")
        print(f"  {cat:18s} {n:5d} {m:6d} {mm:8d}")

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total": total,
        "meter_match": meter_match,
        "meter_mismatch": meter_mismatch,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Report saved to: {output_path}")

    # List mismatches for manual review
    mismatches = [r for r in results if r.meter_match == "MISMATCH"]
    if mismatches:
        print(f"\n{'='*80}")
        print(f"  FILES NEEDING MANUAL REVIEW ({len(mismatches)})")
        print(f"{'='*80}")
        for r in mismatches:
            print(f"  {r.filename}: detected={r.detected_meter}, "
                  f"expected={','.join(r.expected_meters)}, "
                  f"bpm={r.detected_bpm}")


if __name__ == "__main__":
    main()
