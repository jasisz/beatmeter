#!/usr/bin/env python3
"""Metrics dashboard — shows history of eval runs.

Usage:
    uv run python scripts/dashboard.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


def _pct(c: int, t: int) -> str:
    if t == 0:
        return "  – "
    return f"{c / t * 100:.1f}%"


def _load_runs() -> list[dict]:
    """Load all run snapshots, sorted by timestamp (newest last)."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, KeyError):
            pass
    return runs


def main():
    runs = _load_runs()

    print()
    print("=" * 70)
    print("  BeatMeter — Run History")
    print("=" * 70)

    if not runs:
        print("\n  No saved runs. Use: uv run python scripts/eval.py --save")
        print()
        print("=" * 70)
        print()
        return

    # History table
    print()
    print(f"  {'#':>3}  {'Date':12}  {'Split':18}  {'Score':>10}  {'3/x':>7}  {'4/x':>7}  {'5/x':>7}  {'7/x':>7}")
    print(f"  {'─' * 3}  {'─' * 12}  {'─' * 18}  {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}")

    for i, run in enumerate(runs, 1):
        ts = run.get("timestamp", "?")[:16].replace("T", " ")
        split = run.get("split", "?")
        total = run.get("total", 0)
        correct = run.get("correct", 0)
        score = f"{correct}/{total}" if total else "–"
        pct = _pct(correct, total)

        pc = run.get("per_class", {})
        cols = []
        for m in ["3", "4", "5", "7"]:
            d = pc.get(m, {})
            c, t = d.get("correct", 0), d.get("total", 0)
            cols.append(_pct(c, t) if t > 0 else "  – ")

        print(f"  {i:>3}  {ts:12}  {split:18}  {score:>6} {pct:>4}  {cols[0]:>7}  {cols[1]:>7}  {cols[2]:>7}  {cols[3]:>7}")

    # Latest run detail
    latest = runs[-1]
    print()
    print(f"  {'━' * 66}")
    c, t = latest["correct"], latest["total"]
    print(f"  Latest: {c}/{t} ({_pct(c, t)})  —  {latest.get('split', '?')}  [{latest.get('elapsed_s', '?')}s]")
    print(f"  {'━' * 66}")

    pc = latest.get("per_class", {})
    for m in ["3", "4", "5", "7"]:
        d = pc.get(m, {})
        mc, mt = d.get("correct", 0), d.get("total", 0)
        if mt > 0:
            print(f"    {m}/x   {mc:>4d}/{mt:<4d}  {_pct(mc, mt):>6s}")

    # Failures from latest run
    files = latest.get("files", [])
    failures = [f for f in files if not f.get("ok", False)]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in sorted(failures, key=lambda x: x.get("fname", "")):
            pred = f.get("predicted")
            pred_str = f"{pred}/x" if pred else "None"
            print(f"    {f['fname']:40s}  exp={f['expected']}/x  got={pred_str}")

    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
