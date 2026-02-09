#!/usr/bin/env python3
"""Evaluate our meter detection engine on the METER2800 dataset.

Uses subprocess isolation per file to avoid BeatNet/madmom deadlocks.

Usage:
    uv run python scripts/eval_meter2800.py                    # test split, first 100
    uv run python scripts/eval_meter2800.py --limit 0          # full test split (700)
    uv run python scripts/eval_meter2800.py --split val        # validation split
    uv run python scripts/eval_meter2800.py --limit 50 --verbose
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import resolve_audio_path as _resolve_audio_path

# Worker script that runs in a subprocess
WORKER_SCRIPT = """
import sys, json, warnings
warnings.filterwarnings("ignore")
from beatmeter.analysis.engine import AnalysisEngine
engine = AnalysisEngine()
try:
    result = engine.analyze_file(sys.argv[1])
    if result and result.meter_hypotheses:
        best = result.meter_hypotheses[0]
        print(json.dumps({"meter": best.numerator}))
    else:
        print(json.dumps({"meter": None}))
except Exception as e:
    print(json.dumps({"meter": None, "error": str(e)}))
"""


def load_entries(data_dir: Path, split: str) -> list[tuple[Path, int]]:
    """Load entries for a split from .tab label files."""
    for ext in (".tab", ".csv", ".tsv"):
        label_path = data_dir / f"data_{split}_4_classes{ext}"
        if label_path.exists():
            entries = []
            with open(label_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    fname = row["filename"].strip('"')
                    meter = int(row["meter"])
                    audio_path = _resolve_audio_path(fname, data_dir)
                    if audio_path:
                        entries.append((audio_path, meter))
            return entries
    return []


def analyze_one(audio_path: Path, timeout: int = 120) -> int | None:
    """Analyze a single file in a subprocess. Returns predicted meter numerator."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, str(audio_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            return data.get("meter")
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate engine on METER2800")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=100, help="Max files (0=all)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=int, default=120, help="Per-file timeout (s)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    entries = load_entries(data_dir, args.split)
    if not entries:
        print(f"ERROR: No entries found for split '{args.split}' in {data_dir}")
        sys.exit(1)

    if args.limit > 0:
        entries = entries[: args.limit]

    print(f"METER2800 {args.split} split: {len(entries)} files", flush=True)
    print(f"Class distribution: {dict(Counter(m for _, m in entries))}", flush=True)
    print(flush=True)

    correct = 0
    total = 0
    total_by_class: Counter = Counter()
    correct_by_class: Counter = Counter()
    t0 = time.time()

    for i, (audio_path, expected_meter) in enumerate(entries):
        predicted = analyze_one(audio_path, timeout=args.timeout)

        total += 1
        total_by_class[expected_meter] += 1

        ok = predicted == expected_meter
        if ok:
            correct += 1
            correct_by_class[expected_meter] += 1

        if args.verbose:
            status = "OK  " if ok else "FAIL"
            pred_str = f"{predicted}/x" if predicted else "None"
            print(
                f"  {status} {pred_str:5s} exp={expected_meter}/x  {audio_path.name}",
                flush=True,
            )

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta = (len(entries) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(entries)}] {correct}/{total} ({correct/total*100:.0f}%) "
                f" {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"  METER2800 {args.split} — {len(entries)} files — {elapsed:.0f}s", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Overall: {correct}/{total} ({correct / total * 100:.1f}%)", flush=True)
    print(flush=True)
    for m in [3, 4, 5, 7]:
        t = total_by_class[m]
        c = correct_by_class[m]
        if t > 0:
            print(f"  Meter {m}: {c}/{t} ({c / t * 100:.1f}%)", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
