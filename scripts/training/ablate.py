#!/usr/bin/env python3
"""MeterNet feature ablation study.

Single mode (default): leave-one-out, zeroing one feature group at a time.
Multi mode (--multi): removes multiple low-impact features at once.

Usage:
    uv run python scripts/training/ablate.py              # single
    uv run python scripts/training/ablate.py --multi       # combo
    uv run python scripts/training/ablate.py --summary     # single results
    uv run python scripts/training/ablate.py --multi --summary  # combo results
    uv run python scripts/training/ablate.py --dry-run     # preview commands
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "meter_net_ablation"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "train.py"

SINGLE_RESULTS_FILE = PROJECT_ROOT / "data" / "meter_net_ablation_results.json"
MULTI_RESULTS_FILE = PROJECT_ROOT / "data" / "meter_net_multi_ablation_results.json"

# Best hyperparameters from grid search
FIXED_ARGS = [
    "--meter2800", "data/meter2800",
    "--epochs", "200",
    "--workers", "4",
    "--hidden", "1024",
    "--dropout-scale", "1.5",
    "--lr", "0.0003",
    "--batch-size", "64",
]

# Single: leave-one-out targets
SINGLE_TARGETS = [
    "full",  # baseline (no ablation)
    "beat_beatnet",
    "beat_beat_this",
    "beat_madmom",
    "sig_beatnet",
    "sig_beat_this",
    "sig_autocorr",
    "sig_bar_tracking",
    "sig_hcdf",
    "tempo",
]

# Multi: cumulative combo ablations
MULTI_COMBOS = {
    "full": [],
    "drop_zero": ["beat_madmom", "sig_autocorr", "tempo"],
    "drop_weak": ["beat_madmom", "sig_autocorr", "tempo", "sig_beat_this"],
    "drop_aggressive": ["beat_madmom", "sig_autocorr", "tempo", "sig_beat_this",
                         "sig_bar_tracking", "sig_hcdf"],
}


def build_cmd(ablate_targets: list[str]) -> list[str]:
    cmd = ["uv", "run", "python", str(TRAIN_SCRIPT)] + FIXED_ARGS
    if ablate_targets:
        cmd += ["--ablate", ",".join(ablate_targets)]
    return cmd


def read_val_acc(path: Path) -> float | None:
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("best_val_acc")
    except Exception:
        return None


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------

def print_single_summary(results: dict) -> None:
    if not results:
        print("No results yet.")
        return

    baseline = results.get("full", {}).get("val_acc")
    print(f"\nMeterNet Leave-One-Out Ablation")
    print(f"{'Target':<20s} {'Val acc':>8s} {'Delta':>8s} {'Dims':>6s}")
    print("-" * 50)

    for target in SINGLE_TARGETS:
        info = results.get(target)
        if not info:
            print(f"  {target:<20s} {'TODO':>8s}")
            continue
        val = info.get("val_acc")
        if val is None:
            print(f"  {target:<20s} {'ERROR':>8s}")
            continue
        acc_str = f"{val:.1%}"
        if baseline is not None and target != "full":
            delta_str = f"{val - baseline:+.1%}"
        else:
            delta_str = "—"
        dims = info.get("dims_zeroed", "?")
        marker = " <- BASE" if target == "full" else ""
        print(f"  {target:<20s} {acc_str:>8s} {delta_str:>8s} {dims:>6s}{marker}")
    print()


def run_single(args) -> None:
    results = load_results(SINGLE_RESULTS_FILE)

    if args.reset and SINGLE_RESULTS_FILE.exists():
        SINGLE_RESULTS_FILE.unlink()
        results = {}
        print("Results cleared.")

    if args.summary:
        print_single_summary(results)
        return

    todo = [t for t in SINGLE_TARGETS if t not in results]
    print(f"Ablation study: {len(SINGLE_TARGETS)} targets, {len(results)} done, {len(todo)} remaining\n")

    if args.dry_run:
        for t in SINGLE_TARGETS:
            status = "DONE" if t in results else "TODO"
            ablate = [t] if t != "full" else []
            save_path = CHECKPOINT_DIR / f"{t}.pt"
            cmd = build_cmd(ablate) + ["--save", str(save_path)]
            print(f"[{status}] {t}")
            print(f"       {' '.join(cmd)}")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from beatmeter.analysis.signals.meter_net_features import ABLATION_TARGETS
        dims_map = {k: str(v[1] - v[0]) for k, v in ABLATION_TARGETS.items()}
    except ImportError:
        dims_map = {}

    for i, target in enumerate(SINGLE_TARGETS, 1):
        if target in results:
            val = results[target].get("val_acc")
            acc_str = f"{val:.1%}" if val else "ERROR"
            print(f"[{i}/{len(SINGLE_TARGETS)}] SKIP {target} (val={acc_str})")
            continue

        ablate = [target] if target != "full" else []
        save_path = CHECKPOINT_DIR / f"{target}.pt"
        cmd = build_cmd(ablate) + ["--save", str(save_path)]
        print(f"\n[{i}/{len(SINGLE_TARGETS)}] {target}")
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60, flush=True)

        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)

        val_acc = None
        if save_path.exists() and proc.returncode == 0:
            val_acc = read_val_acc(save_path)

        results[target] = {
            "val_acc": val_acc,
            "dims_zeroed": dims_map.get(target, "0") if target != "full" else "0",
            "returncode": proc.returncode,
            "checkpoint": str(save_path) if save_path.exists() else None,
            "timestamp": datetime.now().isoformat(),
        }
        save_results(SINGLE_RESULTS_FILE, results)

        acc_str = f"{val_acc:.1%}" if val_acc else "N/A"
        print(f"  Done: val_acc={acc_str}", flush=True)

    print("\n" + "=" * 60)
    print_single_summary(results)

    # Print promote command for the full baseline
    full_info = results.get("full", {})
    full_ckpt = full_info.get("checkpoint")
    if full_ckpt and Path(full_ckpt).exists():
        print(f"\nFull baseline: val={full_info.get('val_acc', 0):.1%}")
        print(f"To promote:\n  uv run python scripts/eval.py --promote {full_ckpt} --workers 4")

    # Cleanup: remove non-baseline checkpoints
    for target in SINGLE_TARGETS:
        if target == "full":
            continue
        info = results.get(target, {})
        ckpt = info.get("checkpoint")
        if ckpt and Path(ckpt).exists():
            Path(ckpt).unlink()
            print(f"  Cleaned: {Path(ckpt).name}")


# ---------------------------------------------------------------------------
# Multi mode
# ---------------------------------------------------------------------------

def print_multi_summary(results: dict) -> None:
    if not results:
        print("No results yet.")
        return

    baseline = results.get("full", {}).get("val_acc")
    print(f"\nMulti-Ablation Summary")
    print(f"{'Name':<20s} {'Removed':>8s} {'Val acc':>8s} {'Delta':>8s}")
    print("-" * 50)

    for name, targets in MULTI_COMBOS.items():
        info = results.get(name, {})
        val = info.get("val_acc")
        if val is None:
            print(f"  {name:<20s} {'?':>8s}")
            continue
        n_removed = len(targets)
        delta = f"{val - baseline:+.1%}" if baseline and name != "full" else "—"
        print(f"  {name:<20s} {n_removed:>8d} {val:>7.1%} {delta:>8s}")


def run_multi(args) -> None:
    results = load_results(MULTI_RESULTS_FILE)

    if args.reset and MULTI_RESULTS_FILE.exists():
        MULTI_RESULTS_FILE.unlink()
        results = {}
        print("Results cleared.")

    if args.summary:
        print_multi_summary(results)
        return

    # Reuse baseline from single ablation if available
    if "full" not in results:
        single = load_results(SINGLE_RESULTS_FILE)
        if "full" in single:
            results["full"] = single["full"]
            save_results(MULTI_RESULTS_FILE, results)
            print(f"Baseline from single ablation: {single['full'].get('val_acc', 0):.1%}")

    todo = [n for n in MULTI_COMBOS if n not in results]
    print(f"Multi-ablation: {len(MULTI_COMBOS)} combos, {len(results)} done, {len(todo)} remaining\n")

    if args.dry_run:
        for name, targets in MULTI_COMBOS.items():
            status = "DONE" if name in results else "TODO"
            save_path = CHECKPOINT_DIR / f"multi_{name}.pt"
            cmd = build_cmd(targets) + ["--save", str(save_path)]
            print(f"[{status}] {name}: {', '.join(targets) if targets else '(baseline)'}")
            print(f"       {' '.join(cmd)}")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for name, targets in MULTI_COMBOS.items():
        if name in results:
            val = results[name].get("val_acc")
            print(f"SKIP {name} (val={val:.1%})" if val else f"SKIP {name} (ERROR)")
            continue

        save_path = CHECKPOINT_DIR / f"multi_{name}.pt"
        cmd = build_cmd(targets) + ["--save", str(save_path)]
        print(f"\n{'='*60}")
        print(f"[{name}] Ablating: {', '.join(targets) if targets else 'nothing (baseline)'}")
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60, flush=True)

        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)

        val_acc = None
        if save_path.exists() and proc.returncode == 0:
            val_acc = read_val_acc(save_path)

        results[name] = {
            "targets": targets,
            "val_acc": val_acc,
            "returncode": proc.returncode,
            "checkpoint": str(save_path) if save_path.exists() else None,
            "timestamp": datetime.now().isoformat(),
        }
        save_results(MULTI_RESULTS_FILE, results)
        print(f"  Done: val_acc={val_acc:.1%}" if val_acc else "  Done: ERROR")

    print(f"\n{'='*60}")
    print_multi_summary(results)

    best_name = max(results.keys(), key=lambda k: results[k].get("val_acc") or 0)
    best_info = results[best_name]
    best_ckpt = best_info.get("checkpoint")
    if best_ckpt and Path(best_ckpt).exists():
        print(f"\nBest: {best_name} (val={best_info.get('val_acc', 0):.1%})")
        print(f"To promote:\n  uv run python scripts/eval.py --promote {best_ckpt} --workers 4")

    # Cleanup: remove non-winner checkpoints
    for name in results:
        info = results[name]
        ckpt = info.get("checkpoint")
        if ckpt and ckpt != best_ckpt and Path(ckpt).exists():
            Path(ckpt).unlink()
            print(f"  Cleaned: {Path(ckpt).name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MeterNet feature ablation study")
    parser.add_argument("--multi", action="store_true", help="Combo ablation mode")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without running")
    parser.add_argument("--summary", action="store_true", help="Print results table")
    parser.add_argument("--reset", action="store_true", help="Clear previous results")
    args = parser.parse_args()

    if args.multi:
        run_multi(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
