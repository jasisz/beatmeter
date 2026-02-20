#!/usr/bin/env python3
"""Unified evaluation script for meter detection.

Primary: WIKIMETER test (330 segments, 6 classes, balanced accuracy).
Secondary: METER2800 test (700 files, 4 classes, external benchmark).

Usage:
    uv run python scripts/eval.py                        # wikimeter test (primary)
    uv run python scripts/eval.py meter2800              # meter2800 test
    uv run python scripts/eval.py --limit 3              # smoke test (3 files)
    uv run python scripts/eval.py --quick                # stratified 100
    uv run python scripts/eval.py --save                 # run BOTH, save baseline
    uv run python scripts/eval.py --verbose              # per-file details
    uv run python scripts/eval.py --meter 5              # only meter 5
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import load_meter2800_entries, load_wikimeter_entries, resolve_audio_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


DATASETS = {
    "wikimeter": {
        "data_dir": Path("data/wikimeter"),
        "loader": load_wikimeter_entries,
        "splits": ["train", "val", "test", "tuning", "all"],
        "classes": [3, 4, 5, 7, 9, 11],
        "default_split": "test",
    },
    "meter2800": {
        "data_dir": Path("data/meter2800"),
        "loader": load_meter2800_entries,
        "splits": ["train", "val", "test", "tuning"],
        "classes": [3, 4, 5, 7],
        "default_split": "test",
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
# Run snapshots (multi-dataset)
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


def _get_dataset_baseline(previous: dict | None, dataset: str) -> dict | None:
    """Extract baseline for a specific dataset from a saved run.

    Supports both old format (single dataset) and new format (multi-dataset).
    """
    if previous is None:
        return None

    # New format: datasets.wikimeter, datasets.meter2800
    if "datasets" in previous:
        return previous["datasets"].get(dataset)

    # Old format: single dataset in root
    if previous.get("dataset") == dataset:
        return previous

    return None


def detect_regressions(
    current_by_class: dict[int, tuple[int, int]],
    baseline: dict | None,
) -> list[str]:
    """Compare current per-class results to baseline and list regressions."""
    if baseline is None:
        return []

    prev_by_class = baseline.get("per_class", {})
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


def _build_dataset_snapshot(
    correct: int,
    total: int,
    correct_by_class: Counter,
    total_by_class: Counter,
    classes: list[int],
    elapsed: float,
    dataset: str,
    split: str,
    file_results: list[dict],
) -> dict:
    """Build a snapshot dict for one dataset evaluation."""
    per_class = {}
    for m in classes:
        t = total_by_class[m]
        c = correct_by_class[m]
        if t > 0:
            per_class[str(m)] = {"correct": c, "total": t}

    # Balanced accuracy
    per_class_accs = []
    for m in classes:
        t = total_by_class[m]
        c = correct_by_class[m]
        if t > 0:
            per_class_accs.append(c / t)
    balanced_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0.0

    return {
        "dataset": dataset,
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        "balanced_accuracy": round(balanced_acc * 100, 1),
        "elapsed_s": round(elapsed, 1),
        "per_class": per_class,
        "files": file_results,
    }


def save_combined_run(snapshots: dict[str, dict]):
    """Save a combined multi-dataset run snapshot."""
    from beatmeter.experiment import get_git_info, checkpoint_sha256, log_experiment, make_experiment_record

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUNS_DIR / f"{ts}.json"

    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "command": " ".join(sys.argv),
        "checkpoints": {},
        "datasets": snapshots,
    }

    # Save copies of all trained checkpoints alongside the snapshot
    import shutil
    import torch
    checkpoints = {
        "meter_net": PROJECT_ROOT / "data" / "meter_net.pt",
    }
    for name, src in checkpoints.items():
        if src.exists():
            combined["checkpoints"][name] = {
                "path": str(src),
                "sha256": checkpoint_sha256(src),
            }
            # Read experiment info from checkpoint if available
            try:
                ckpt = torch.load(src, map_location="cpu", weights_only=False)
                if "experiment" in ckpt:
                    combined["checkpoints"][name]["trained"] = ckpt["experiment"]["timestamp"]
                    combined["checkpoints"][name]["git_at_train"] = ckpt["experiment"]["git"]
            except Exception:
                pass

    path.write_text(json.dumps(combined, indent=2))
    print(f"\n  Saved to {path}")

    for name, src in checkpoints.items():
        if src.exists():
            dst = RUNS_DIR / f"{ts}_{name}.pt"
            shutil.copy2(src, dst)
            print(f"  {name} checkpoint saved to {dst}")

    # Log to experiments.jsonl
    m2800 = snapshots.get("meter2800", {})
    wiki = snapshots.get("wikimeter", {})
    log_experiment(make_experiment_record(
        type="eval", model="pipeline",
        results={
            "meter2800_acc": m2800.get("accuracy"),
            "meter2800_correct": m2800.get("correct"),
            "meter2800_total": m2800.get("total"),
            "wikimeter_acc": wiki.get("accuracy"),
            "wikimeter_correct": wiki.get("correct"),
            "wikimeter_total": wiki.get("total"),
        },
        extra={"snapshot_path": str(path)},
    ))


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

    # Balanced accuracy
    per_class_accs = []
    for m in classes:
        t = total_by_class[m]
        c = correct_by_class[m]
        if t > 0:
            per_class_accs.append(c / t)
    balanced_acc = sum(per_class_accs) / len(per_class_accs) if per_class_accs else 0.0

    print(
        f"  {dataset.upper()} {split_label}:  {correct}/{total} ({overall_pct})"
        f"   bal={balanced_acc:.1%}"
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


def _worker_process_file(args: tuple[str, int, str, str]) -> dict:
    """Process a single file in a worker process."""
    import torch
    audio_path_str, expected_meter, meters_json, valid_classes_json = args
    fname = Path(audio_path_str).name

    meters_dict = json.loads(meters_json) if meters_json else None
    valid_classes = set(json.loads(valid_classes_json)) if valid_classes_json else None

    try:
        with torch.inference_mode():
            result = _worker_engine.analyze_file(audio_path_str, skip_sections=True)
        if result and result.meter_hypotheses:
            predicted = result.meter_hypotheses[0].numerator
            if valid_classes and predicted not in valid_classes:
                if predicted == 9 and 3 in valid_classes:
                    predicted = 3
                else:
                    for hyp in result.meter_hypotheses[1:]:
                        if hyp.numerator in valid_classes:
                            predicted = hyp.numerator
                            break
            bpm = result.tempo.bpm if result.tempo else None
        else:
            predicted = None
            bpm = None
    except Exception:
        predicted = None
        bpm = None

    if meters_dict and len(meters_dict) > 1 and predicted is not None:
        ok = str(predicted) in meters_dict
    else:
        ok = predicted == expected_meter

    return {
        "fname": fname,
        "expected": expected_meter,
        "predicted": predicted,
        "bpm": bpm,
        "ok": ok,
        "multilabel": meters_dict if meters_dict and len(meters_dict) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Core eval function
# ---------------------------------------------------------------------------


def run_eval(
    dataset: str,
    split: str,
    entries: list[tuple[Path, int, dict[int, float] | None]],
    classes: list[int],
    n_workers: int = 0,
    verbose: bool = False,
    model_label: str = "",
) -> tuple[int, int, Counter, Counter, list[dict], float]:
    """Run evaluation on a list of entries. Returns (correct, total, correct_by_class, total_by_class, file_results, elapsed)."""

    class_dist = Counter(m for _, m, _ in entries)
    dist_str = ", ".join(f"{m}/x: {class_dist[m]}" for m in sorted(class_dist))

    mode_str = f"{n_workers} workers" if n_workers > 0 else "in-process"
    print(flush=True)
    print(f"  {'─' * 56}", flush=True)
    print(f"  {dataset.upper()} {split}  |  {len(entries)} files  |  {mode_str}{model_label}", flush=True)
    print(f"  {dist_str}", flush=True)
    print(f"  {'─' * 56}", flush=True)
    print(flush=True)

    correct = 0
    total = 0
    total_by_class: Counter = Counter()
    correct_by_class: Counter = Counter()
    file_results: list[dict] = []
    t0 = time.time()

    n_multilabel = sum(1 for _, _, md in entries if md and len(md) > 1)
    n_multilabel_ok = 0

    if n_workers > 0:
        print(f"  Starting {n_workers} workers...", flush=True)
        valid_classes_json = json.dumps(classes)
        work_items = [
            (str(p), m, json.dumps({str(k): v for k, v in md.items()}) if md else "",
             valid_classes_json)
            for p, m, md in entries
        ]

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
                    if result.get("multilabel"):
                        n_multilabel_ok += 1
                file_results.append(result)

                if verbose:
                    status = "OK  " if result["ok"] else "FAIL"
                    pred_str = f"{result['predicted']}/x" if result["predicted"] else "None"
                    ml_tag = " [multi]" if result.get("multilabel") else ""
                    tqdm.write(f"  {status} {pred_str:5s} exp={result['expected']}/x  {result['fname']}{ml_tag}")

                pct = f"{correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "0/0"
                pbar.set_postfix_str(pct)

    else:
        print("  Loading engine...", flush=True)
        import torch
        from beatmeter.analysis.cache import AnalysisCache
        from beatmeter.analysis.engine import AnalysisEngine

        cache = AnalysisCache()
        engine = AnalysisEngine(cache=cache)
        print("  Engine ready.\n", flush=True)

        pbar = tqdm(entries, desc="Eval", unit="file",
                    bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] acc={postfix}")
        for audio_path, expected_meter, meters_dict in pbar:
            fname = audio_path.name
            try:
                with torch.inference_mode():
                    result = engine.analyze_file(str(audio_path), skip_sections=True)
                if result and result.meter_hypotheses:
                    valid_set = set(classes)
                    predicted = result.meter_hypotheses[0].numerator
                    if predicted not in valid_set:
                        if predicted == 9 and 3 in valid_set:
                            predicted = 3
                        else:
                            for hyp in result.meter_hypotheses[1:]:
                                if hyp.numerator in valid_set:
                                    predicted = hyp.numerator
                                    break
                    bpm = result.tempo.bpm if result.tempo else None
                else:
                    predicted = None
                    bpm = None
            except Exception as e:
                if verbose:
                    tqdm.write(f"  ERR  {fname}: {e}")
                predicted = None
                bpm = None

            if meters_dict and len(meters_dict) > 1 and predicted is not None:
                ok = predicted in meters_dict
            else:
                ok = predicted == expected_meter

            is_multilabel = meters_dict is not None and len(meters_dict) > 1

            total += 1
            total_by_class[expected_meter] += 1
            if ok:
                correct += 1
                correct_by_class[expected_meter] += 1
                if is_multilabel:
                    n_multilabel_ok += 1

            file_results.append({
                "fname": fname,
                "expected": expected_meter,
                "predicted": predicted,
                "bpm": bpm,
                "ok": ok,
                "multilabel": meters_dict if is_multilabel else None,
            })

            if verbose:
                status = "OK  " if ok else "FAIL"
                pred_str = f"{predicted}/x" if predicted else "None"
                ml_tag = " [multi]" if is_multilabel else ""
                tqdm.write(f"  {status} {pred_str:5s} exp={expected_meter}/x  {fname}{ml_tag}")

            pct = f"{correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "0/0"
            pbar.set_postfix_str(pct)

    elapsed = time.time() - t0

    print_summary(
        correct, total, correct_by_class, total_by_class,
        classes, elapsed, dataset, split, len(entries),
    )

    if n_multilabel > 0:
        print(f"\n  Multi-label (polyrhythmic): {n_multilabel_ok}/{n_multilabel} matched"
              f" ({_pct(n_multilabel_ok, n_multilabel)})", flush=True)

    return correct, total, correct_by_class, total_by_class, file_results, elapsed


# ---------------------------------------------------------------------------
# Entry loading helper
# ---------------------------------------------------------------------------


def _load_entries(
    dataset: str,
    split: str,
    data_dir: Path | None = None,
    meter: int = 0,
    quick: bool = False,
    limit: int = 0,
) -> tuple[list[tuple[Path, int, dict[int, float] | None]], str]:
    """Load and filter entries for a dataset. Returns (entries, split_label)."""
    ds_config = DATASETS[dataset]
    resolved_dir = (data_dir or ds_config["data_dir"]).resolve()
    loader = ds_config["loader"]

    raw_entries = loader(resolved_dir, split)
    if not raw_entries:
        print(f"ERROR: No entries found for split '{split}' in {resolved_dir}")
        sys.exit(1)

    entries: list[tuple[Path, int, dict[int, float] | None]] = []
    for e in raw_entries:
        if len(e) == 2:
            entries.append((e[0], e[1], None))
        else:
            entries.append(e)

    if meter > 0:
        entries = [(p, m, md) for p, m, md in entries if m == meter]

    split_label = split
    if quick:
        two_tuples = [(p, m) for p, m, _ in entries]
        sampled_two = stratified_sample(two_tuples, n=100)
        sampled_set = {(str(p), m) for p, m in sampled_two}
        entries = [(p, m, md) for p, m, md in entries if (str(p), m) in sampled_set]
        split_label = f"{split} (quick)"
    elif limit > 0:
        entries = entries[:limit]

    return entries, split_label


# ---------------------------------------------------------------------------
# Promote flow
# ---------------------------------------------------------------------------


def _run_promote(args):
    """Evaluate a checkpoint on both datasets and optionally promote to meter_net.pt."""
    ckpt_path = Path(args.promote)
    prod_path = PROJECT_ROOT / "data" / "meter_net.pt"
    prev_path = PROJECT_ROOT / "data" / "meter_net.prev.pt"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Use candidate checkpoint for evaluation
    os.environ["METER_NET_CHECKPOINT"] = str(ckpt_path.resolve())
    model_label = f" [{ckpt_path.name}]"

    print(f"  === PROMOTE MODE: evaluating {ckpt_path} ===\n")

    snapshots: dict[str, dict] = {}
    for ds_name in ["wikimeter", "meter2800"]:
        ds_cfg = DATASETS[ds_name]
        ds_split = ds_cfg["default_split"]
        ds_classes = ds_cfg["classes"]

        entries, split_label = _load_entries(ds_name, ds_split)
        correct, total, correct_by_class, total_by_class, file_results, elapsed = run_eval(
            ds_name, split_label, entries, ds_classes, args.workers, args.verbose, model_label,
        )
        snapshots[ds_name] = _build_dataset_snapshot(
            correct, total, correct_by_class, total_by_class,
            ds_classes, elapsed, ds_name, ds_split, file_results,
        )

    # Compare with baseline
    previous = load_latest_run()
    has_baseline = previous is not None

    any_warning = False
    for ds_name in ["wikimeter", "meter2800"]:
        snap = snapshots[ds_name]
        baseline = _get_dataset_baseline(previous, ds_name)
        if baseline is None:
            continue

        ds_classes = DATASETS[ds_name]["classes"]
        cur_correct = snap["correct"]
        cur_total = snap["total"]
        prev_correct = baseline.get("correct", 0)
        prev_total = baseline.get("total", 0)

        print(f"\n  {ds_name.upper()} comparison:")
        print(f"    Previous: {prev_correct}/{prev_total} ({_pct(prev_correct, prev_total)})")
        print(f"    New:      {cur_correct}/{cur_total} ({_pct(cur_correct, cur_total)})")

        if cur_correct < prev_correct and cur_total >= prev_total:
            print(f"    WARNING: overall regression ({prev_correct} -> {cur_correct})")
            any_warning = True

        # Per-class diff
        prev_by_class = baseline.get("per_class", {})
        for m_str, prev_data in sorted(prev_by_class.items(), key=lambda x: int(x[0])):
            m = int(m_str)
            pc = prev_data["correct"]
            pt = prev_data["total"]
            nc = snap["per_class"].get(m_str, {}).get("correct", 0)
            nt = snap["per_class"].get(m_str, {}).get("total", 0)
            diff = nc - pc
            if diff != 0:
                sign = "+" if diff > 0 else ""
                marker = "  WARNING" if diff < -3 else ""
                if diff < -3:
                    any_warning = True
                print(f"    {m}/x: {pc}/{pt} -> {nc}/{nt} ({sign}{diff}){marker}")

    # Decide whether to promote
    if has_baseline and any_warning:
        print("\n  Warnings detected (see above).")

    if not has_baseline:
        print("\n  No baseline found — promoting without comparison.")

    if not args.force:
        try:
            answer = input("\n  Promote? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer != "y":
            print("  Aborted.")
            sys.exit(1)

    # Backup + promote
    if prod_path.exists():
        shutil.copy2(prod_path, prev_path)
    shutil.copy2(ckpt_path, prod_path)

    save_combined_run(snapshots)

    print(f"\n  Promoted {ckpt_path.name} -> meter_net.pt")
    if prev_path.exists():
        print(f"  Previous saved to {prev_path}")


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
        default="wikimeter",
        choices=list(DATASETS.keys()),
        help="Dataset to evaluate on (default: wikimeter)",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override dataset data directory")
    parser.add_argument(
        "--split",
        default=None,
        help="Data split (default: test). Use 'tuning' for dev.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max files (0=all, default=0)")
    parser.add_argument("--quick", action="store_true",
                        help="Stratified 100 from tuning split")
    parser.add_argument("--meter", type=int, default=0,
                        help="Filter by meter class (3, 4, 5, 7, 9, 11)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes (0=in-process, default=0)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", action="store_true",
                        help="Run BOTH datasets and save combined baseline")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Custom MeterNet checkpoint path")
    parser.add_argument("--promote", type=str, default=None, metavar="CHECKPOINT",
                        help="Evaluate checkpoint on both datasets and promote to "
                             "meter_net.pt if approved")
    parser.add_argument("--force", action="store_true",
                        help="Skip interactive confirmation for --promote")
    args = parser.parse_args()

    # --- Promote mode ---
    if args.promote:
        _run_promote(args)
        return

    # Set env vars for meter model selection
    if args.checkpoint:
        os.environ["METER_NET_CHECKPOINT"] = args.checkpoint

    model_label = ""
    if args.checkpoint:
        model_label = f" [{Path(args.checkpoint).name}]"

    ds_config = DATASETS[args.dataset]
    split = args.split or ds_config["default_split"]

    if args.quick:
        split = "tuning"

    # Validate split
    if split not in ds_config["splits"]:
        print(f"ERROR: Invalid split '{split}' for {args.dataset}. "
              f"Valid: {ds_config['splits']}")
        sys.exit(1)

    # --- Normal (single dataset) eval ---
    if not args.save:
        entries, split_label = _load_entries(
            args.dataset, split, args.data_dir, args.meter, args.quick, args.limit,
        )
        classes = ds_config["classes"]

        correct, total, correct_by_class, total_by_class, file_results, elapsed = run_eval(
            args.dataset, split_label, entries, classes, args.workers, args.verbose, model_label,
        )

        # Regression detection
        previous = load_latest_run()
        baseline = _get_dataset_baseline(previous, args.dataset)
        current_by_class = {
            m: (correct_by_class[m], total_by_class[m]) for m in classes if total_by_class[m] > 0
        }
        regressions = detect_regressions(current_by_class, baseline)
        if regressions:
            print(f"\n  REGRESSIONS DETECTED ({len(regressions)}):")
            for msg in regressions:
                print(msg)
        elif baseline:
            prev_correct = baseline.get("correct", 0)
            prev_total = baseline.get("total", 0)
            prev_bal = baseline.get("balanced_accuracy", 0)
            print(
                f"\n  vs last saved: {prev_correct}/{prev_total}"
                f" ({_pct(prev_correct, prev_total)}, bal={prev_bal}%)"
            )

        if regressions:
            sys.exit(1)
        return

    # --- Save mode: run BOTH datasets ---
    print("  === SAVE MODE: evaluating both datasets ===\n")
    snapshots: dict[str, dict] = {}
    all_regressions: list[str] = []
    previous = load_latest_run()

    for ds_name in ["wikimeter", "meter2800"]:
        ds_cfg = DATASETS[ds_name]
        ds_split = ds_cfg["default_split"]
        ds_classes = ds_cfg["classes"]

        entries, split_label = _load_entries(ds_name, ds_split, meter=args.meter)

        correct, total, correct_by_class, total_by_class, file_results, elapsed = run_eval(
            ds_name, split_label, entries, ds_classes, args.workers, args.verbose, model_label,
        )

        snapshots[ds_name] = _build_dataset_snapshot(
            correct, total, correct_by_class, total_by_class,
            ds_classes, elapsed, ds_name, ds_split, file_results,
        )

        # Regression check per dataset
        baseline = _get_dataset_baseline(previous, ds_name)
        current_by_class = {
            m: (correct_by_class[m], total_by_class[m]) for m in ds_classes if total_by_class[m] > 0
        }
        regs = detect_regressions(current_by_class, baseline)
        if regs:
            all_regressions.extend([f"  [{ds_name.upper()}] {r.strip()}" for r in regs])
        elif baseline:
            prev_c = baseline.get("correct", 0)
            prev_t = baseline.get("total", 0)
            prev_bal = baseline.get("balanced_accuracy", 0)
            print(f"\n  vs last saved ({ds_name}): {prev_c}/{prev_t}"
                  f" ({_pct(prev_c, prev_t)}, bal={prev_bal}%)")

    save_combined_run(snapshots)

    if all_regressions:
        print(f"\n  REGRESSIONS DETECTED ({len(all_regressions)}):")
        for msg in all_regressions:
            print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
