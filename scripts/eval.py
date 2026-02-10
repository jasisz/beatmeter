#!/usr/bin/env python3
"""Unified evaluation script for meter detection.

Supports both in-process (--workers 0) and multiprocess (--workers N) modes.
Multiprocess gives true parallelism on cold cache; in-process is simpler
and sufficient on warm cache.

Usage:
    uv run python scripts/eval.py --limit 3            # smoke test
    uv run python scripts/eval.py --quick               # stratified 100 (~20 min)
    uv run python scripts/eval.py --split test --limit 0 # hold-out 700
    uv run python scripts/eval.py --workers 4            # parallel cold cache
    uv run python scripts/eval.py --save                 # save baseline
    uv run python scripts/eval.py --verbose              # per-file details
    uv run python scripts/eval.py --meter 5              # only meter 5
"""

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import resolve_audio_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


def load_meter2800_entries(
    data_dir: Path, split: str
) -> list[tuple[Path, int]]:
    """Load entries for a METER2800 split from .tab label files.

    split='tuning' combines train+val (2100 files).
    """
    if split == "tuning":
        entries = []
        for sub in ("train", "val"):
            entries.extend(load_meter2800_entries(data_dir, sub))
        return entries

    for ext in (".tab", ".csv", ".tsv"):
        label_path = data_dir / f"data_{split}_4_classes{ext}"
        if label_path.exists():
            entries = []
            with open(label_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    fname = row["filename"].strip('"')
                    meter = int(row["meter"])
                    audio_path = resolve_audio_path(fname, data_dir)
                    if audio_path:
                        entries.append((audio_path, meter))
            return entries
    return []


DATASETS = {
    "meter2800": {
        "data_dir": Path("data/meter2800"),
        "loader": load_meter2800_entries,
        "splits": ["train", "val", "test", "tuning"],
        "classes": [3, 4, 5, 7],
    },
}

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    entries: list[tuple[Path, int]], n: int = 100, seed: int = 42
) -> list[tuple[Path, int]]:
    """Stratified sample: proportional to class size, minimum 1 per class."""
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for e in entries:
        by_class.setdefault(e[1], []).append(e)

    total = len(entries)
    rng = random.Random(seed)
    sampled: list[tuple[Path, int]] = []

    remaining = n
    allocations: dict[int, int] = {}
    for meter, items in sorted(by_class.items()):
        alloc = max(1, round(len(items) / total * n))
        allocations[meter] = min(alloc, len(items))
        remaining -= allocations[meter]

    if remaining > 0:
        for meter in sorted(by_class, key=lambda m: len(by_class[m]), reverse=True):
            add = min(remaining, len(by_class[meter]) - allocations[meter])
            if add > 0:
                allocations[meter] += add
                remaining -= add
            if remaining <= 0:
                break

    for meter, items in sorted(by_class.items()):
        sampled.extend(rng.sample(items, allocations.get(meter, 0)))

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Run snapshots
# ---------------------------------------------------------------------------


def load_latest_run() -> dict | None:
    """Load the most recent saved run snapshot."""
    if not RUNS_DIR.exists():
        return None
    runs = sorted(RUNS_DIR.glob("*.json"))
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text())
    except (json.JSONDecodeError, KeyError):
        return None


def detect_regressions(
    current_by_class: dict[int, tuple[int, int]],
    previous: dict | None,
) -> list[str]:
    """Compare current per-class results to previous and list regressions."""
    if previous is None:
        return []

    prev_by_class = previous.get("per_class", {})
    regressions = []
    for meter_str, prev_data in prev_by_class.items():
        meter = int(meter_str)
        prev_correct = prev_data["correct"]
        prev_total = prev_data["total"]
        if meter in current_by_class:
            cur_correct, cur_total = current_by_class[meter]
            if cur_correct < prev_correct and cur_total >= prev_total:
                regressions.append(
                    f"  REGRESSION meter {meter}: was {prev_correct}/{prev_total}, "
                    f"now {cur_correct}/{cur_total}"
                )
    return regressions


def save_run(
    file_results: list[dict],
    correct: int,
    total: int,
    correct_by_class: Counter,
    total_by_class: Counter,
    dataset: str,
    split: str,
    classes: list[int],
    elapsed: float,
):
    """Save a full run snapshot to runs/ directory."""
    per_class = {}
    for m in classes:
        t = total_by_class[m]
        c = correct_by_class[m]
        if t > 0:
            per_class[str(m)] = {"correct": c, "total": t}

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        "elapsed_s": round(elapsed, 1),
        "per_class": per_class,
        "files": file_results,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUNS_DIR / f"{ts}.json"
    path.write_text(json.dumps(snapshot, indent=2))
    print(f"\n  Saved to {path}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _pct(c: int, t: int) -> str:
    if t == 0:
        return "  -  "
    return f"{c / t * 100:.1f}%"


def print_summary(
    correct: int,
    total: int,
    correct_by_class: Counter,
    total_by_class: Counter,
    classes: list[int],
    elapsed: float,
    dataset: str,
    split_label: str,
    n_entries: int,
):
    """Print formatted summary table."""
    w = 50
    print(flush=True)
    print(f"  {'━' * w}", flush=True)
    overall_pct = _pct(correct, total) if total > 0 else "–"
    print(
        f"  {dataset.upper()} {split_label}:  {correct}/{total} ({overall_pct})"
        f"   [{elapsed:.0f}s]",
        flush=True,
    )
    print(f"  {'━' * w}", flush=True)

    for m in classes:
        t = total_by_class[m]
        c = correct_by_class[m]
        if t > 0:
            print(f"    {m}/x   {c:>4d}/{t:<4d}  {_pct(c, t):>6s}", flush=True)

    # Binary 3+4 accuracy
    binary_total = total_by_class[3] + total_by_class[4]
    binary_correct = correct_by_class[3] + correct_by_class[4]
    if binary_total > 0:
        print(f"    {'─' * 26}", flush=True)
        print(
            f"    3+4  {binary_correct:>4d}/{binary_total:<4d}  {_pct(binary_correct, binary_total):>6s}",
            flush=True,
        )

    print(f"  {'━' * w}", flush=True)


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------

_worker_engine = None


def _worker_init():
    """Initialize engine in each worker process."""
    global _worker_engine
    warnings.filterwarnings("ignore")
    import torch  # noqa: F401
    from beatmeter.analysis.cache import AnalysisCache
    from beatmeter.analysis.engine import AnalysisEngine
    _worker_engine = AnalysisEngine(cache=AnalysisCache())


def _worker_process_file(args: tuple[str, int]) -> dict:
    """Process a single file in a worker process."""
    import torch
    audio_path_str, expected_meter = args
    fname = Path(audio_path_str).name
    try:
        with torch.inference_mode():
            result = _worker_engine.analyze_file(audio_path_str, skip_sections=True)
        if result and result.meter_hypotheses:
            best = result.meter_hypotheses[0]
            predicted = best.numerator
            bpm = result.tempo.bpm if result.tempo else None
        else:
            predicted = None
            bpm = None
    except Exception:
        predicted = None
        bpm = None

    return {
        "fname": fname,
        "expected": expected_meter,
        "predicted": predicted,
        "bpm": bpm,
        "ok": predicted == expected_meter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for meter detection"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="meter2800",
        choices=list(DATASETS.keys()),
        help="Dataset to evaluate on (default: meter2800)",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override dataset data directory")
    parser.add_argument(
        "--split",
        default="tuning",
        help="Data split (default: tuning). Use 'test' for hold-out.",
    )
    parser.add_argument("--limit", type=int, default=100,
                        help="Max files (0=all, default=100)")
    parser.add_argument("--quick", action="store_true",
                        help="Stratified 100 from tuning split")
    parser.add_argument("--meter", type=int, default=0,
                        help="Filter by meter class (3, 4, 5, 7)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (0=in-process, default=0)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", action="store_true",
                        help="Save run snapshot for history & regression tracking")
    args = parser.parse_args()

    ds_config = DATASETS[args.dataset]
    data_dir = (args.data_dir or ds_config["data_dir"]).resolve()
    classes = ds_config["classes"]
    loader = ds_config["loader"]

    # Validate split
    valid_splits = ds_config["splits"]
    if args.split not in valid_splits:
        print(f"ERROR: Invalid split '{args.split}' for {args.dataset}. "
              f"Valid: {valid_splits}")
        sys.exit(1)

    # --quick and --save imply tuning split, 100 stratified
    if args.save and not args.quick and args.limit == 100 and args.split == "tuning":
        args.quick = True
    if args.quick:
        args.split = "tuning"

    entries = loader(data_dir, args.split)
    if not entries:
        print(f"ERROR: No entries found for split '{args.split}' in {data_dir}")
        sys.exit(1)

    # Filter by meter class
    if args.meter > 0:
        entries = [(p, m) for p, m in entries if m == args.meter]
        if not entries:
            print(f"ERROR: No files with meter={args.meter}")
            sys.exit(1)

    # Apply sampling/limit
    if args.quick:
        entries = stratified_sample(entries, n=100)
    elif args.limit > 0:
        entries = entries[: args.limit]

    split_label = args.split
    if args.quick:
        split_label = "tuning (quick)"

    class_dist = Counter(m for _, m in entries)
    dist_str = ", ".join(f"{m}/x: {class_dist[m]}" for m in sorted(class_dist))

    n_workers = args.workers
    mode_str = f"{n_workers} workers" if n_workers > 0 else "in-process"

    print(flush=True)
    print(f"  {'─' * 56}", flush=True)
    print(f"  {args.dataset.upper()} {split_label}  |  {len(entries)} files  |  {mode_str}", flush=True)
    print(f"  {dist_str}", flush=True)
    print(f"  {'─' * 56}", flush=True)
    print(flush=True)

    correct = 0
    total = 0
    total_by_class: Counter = Counter()
    correct_by_class: Counter = Counter()
    file_results: list[dict] = []
    t0 = time.time()

    if n_workers > 0:
        # --- Multiprocess mode ---
        print(f"  Starting {n_workers} workers...", flush=True)
        work_items = [(str(p), m) for p, m in entries]

        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_worker_init) as pool:
            pbar = tqdm(
                pool.imap_unordered(_worker_process_file, work_items),
                total=len(entries), desc="Eval", unit="file",
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] acc={postfix}",
            )
            for result in pbar:
                total += 1
                total_by_class[result["expected"]] += 1
                if result["ok"]:
                    correct += 1
                    correct_by_class[result["expected"]] += 1
                file_results.append(result)

                if args.verbose:
                    status = "OK  " if result["ok"] else "FAIL"
                    pred_str = f"{result['predicted']}/x" if result["predicted"] else "None"
                    tqdm.write(f"  {status} {pred_str:5s} exp={result['expected']}/x  {result['fname']}")

                pct = f"{correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "0/0"
                pbar.set_postfix_str(pct)

    else:
        # --- In-process mode ---
        print("  Loading engine...", flush=True)
        import torch
        from beatmeter.analysis.cache import AnalysisCache
        from beatmeter.analysis.engine import AnalysisEngine

        cache = AnalysisCache()
        engine = AnalysisEngine(cache=cache)
        print("  Engine ready.\n", flush=True)

        pbar = tqdm(entries, desc="Eval", unit="file",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] acc={postfix}")
        for audio_path, expected_meter in pbar:
            fname = audio_path.name
            try:
                with torch.inference_mode():
                    result = engine.analyze_file(str(audio_path), skip_sections=True)
                if result and result.meter_hypotheses:
                    best = result.meter_hypotheses[0]
                    predicted = best.numerator
                    bpm = result.tempo.bpm if result.tempo else None
                else:
                    predicted = None
                    bpm = None
            except Exception as e:
                if args.verbose:
                    tqdm.write(f"  ERR  {fname}: {e}")
                predicted = None
                bpm = None

            ok = predicted == expected_meter
            total += 1
            total_by_class[expected_meter] += 1
            if ok:
                correct += 1
                correct_by_class[expected_meter] += 1

            file_results.append({
                "fname": fname,
                "expected": expected_meter,
                "predicted": predicted,
                "bpm": bpm,
                "ok": ok,
            })

            if args.verbose:
                status = "OK  " if ok else "FAIL"
                pred_str = f"{predicted}/x" if predicted else "None"
                tqdm.write(f"  {status} {pred_str:5s} exp={expected_meter}/x  {fname}")

            pct = f"{correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "0/0"
            pbar.set_postfix_str(pct)

    # Summary
    elapsed = time.time() - t0

    print_summary(
        correct, total, correct_by_class, total_by_class,
        classes, elapsed, args.dataset, split_label, len(entries),
    )

    # Regression detection vs last saved run
    previous = load_latest_run()
    current_by_class = {
        m: (correct_by_class[m], total_by_class[m]) for m in classes if total_by_class[m] > 0
    }
    regressions = detect_regressions(current_by_class, previous)
    if regressions:
        print(f"\n  REGRESSIONS DETECTED ({len(regressions)}):")
        for msg in regressions:
            print(msg)
    elif previous:
        prev_correct = previous.get("correct", 0)
        prev_total = previous.get("total", 0)
        print(
            f"\n  vs last saved: {prev_correct}/{prev_total}"
            f" ({_pct(prev_correct, prev_total)})"
        )

    # Save snapshot
    if args.save:
        save_run(
            file_results, correct, total, correct_by_class, total_by_class,
            args.dataset, args.split, classes, elapsed,
        )

    # Exit code: non-zero if regressions
    if regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
