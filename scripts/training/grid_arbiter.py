#!/usr/bin/env python3
"""Grid search over arbiter training hyperparameters.

Runs train_arbiter.py with all combinations of parameters, saves each
checkpoint and results. Resumes safely — already completed runs are skipped.

Usage:
    uv run python scripts/training/grid_arbiter.py
    uv run python scripts/training/grid_arbiter.py --dry-run   # show combos only
    uv run python scripts/training/grid_arbiter.py --summary   # show results table
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "data" / "arbiter_grid_results.json"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "arbiter_grid"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "train_arbiter.py"
SOURCE_CKPT = PROJECT_ROOT / "data" / "meter_arbiter.pt"

# ── Grid definition ──────────────────────────────────────────────────────────
# Edit these to define the search space.

GRID = {
    "sharpen": [
        "",
        "autocorr:1.5",
        "autocorr:2.0",
        "onset_mlp:1.5",
        "beat_this:1.5",
    ],
    "boost_rare": [1.0, 1.5, 2.0],
}

FIXED_ARGS = [
    "--train",
    "--extra-data",
    "--seeds", "10",
    "--epochs", "200",
]

# Set to True to re-run --extract before each training run.
# Only needed if the arbiter dataset might be stale (e.g. after warm_cache).
# Once the dataset is fresh, set to False to skip extraction and save time.
EXTRACT_FIRST = False
EXTRACT_ARGS = [
    "--extract",
    "--split", "tuning",
    "--extra-data",
    "--workers", "4",
]

# ─────────────────────────────────────────────────────────────────────────────


def combo_key(params: dict) -> str:
    """Stable string key for a parameter combination."""
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def combo_filename(params: dict) -> str:
    """Safe filename for a parameter combination."""
    parts = []
    for k, v in sorted(params.items()):
        safe_v = str(v).replace(":", "-").replace(".", "p") if v != "" else "none"
        parts.append(f"{k}_{safe_v}")
    return "_".join(parts)


def build_cmd(params: dict) -> list[str]:
    cmd = ["uv", "run", "python", str(TRAIN_SCRIPT)] + FIXED_ARGS
    if params.get("sharpen"):
        cmd += ["--sharpen", params["sharpen"]]
    if params.get("boost_rare", 1.0) != 1.0:
        cmd += ["--boost-rare", str(params["boost_rare"])]
    return cmd


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def read_val_acc_from_checkpoint(path: Path) -> float | None:
    """Read best_val_acc stored inside the checkpoint."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("best_val_acc")
    except Exception as e:
        print(f"  Warning: could not read val_acc from checkpoint: {e}")
        return None



def print_summary(results: dict) -> None:
    if not results:
        print("No results yet.")
        return
    rows = sorted(results.items(), key=lambda x: x[1].get("val_acc") or 0, reverse=True)
    print(f"\n{'#':<4} {'Val acc':<10} Parameters")
    print("-" * 72)
    for i, (key, info) in enumerate(rows, 1):
        val = info.get("val_acc")
        acc_str = f"{val:.1%}" if val is not None else "ERROR"
        marker = " ← BEST" if i == 1 else ""
        rc = info.get("returncode", "?")
        rc_str = "" if rc == 0 else f" [rc={rc}]"
        print(f"  {i:<2}  {acc_str:<10} {key}{rc_str}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show combinations, don't run")
    parser.add_argument("--summary", action="store_true", help="Show results table and exit")
    parser.add_argument("--reset", action="store_true", help="Clear previous results and re-run all combinations")
    args = parser.parse_args()

    if args.reset and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Results cleared.")

    results = load_results()

    if args.summary:
        print_summary(results)
        return

    # Build all combinations
    keys = list(GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]

    todo = [c for c in combos if combo_key(c) not in results]
    print(f"Grid search: {len(combos)} combinations total")
    print(f"Already done: {len(results)}, remaining: {len(todo)}")
    print()

    if args.dry_run:
        for c in combos:
            key = combo_key(c)
            status = "DONE" if key in results else "TODO"
            print(f"[{status}] {key}")
            print(f"       {' '.join(build_cmd(c))}")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_FIRST:
        print("Running extraction phase first...")
        extract_cmd = ["uv", "run", "python", str(TRAIN_SCRIPT)] + EXTRACT_ARGS
        print(f"  CMD: {' '.join(extract_cmd)}", flush=True)
        proc = subprocess.run(extract_cmd, cwd=PROJECT_ROOT)
        if proc.returncode != 0:
            print(f"Extraction failed (rc={proc.returncode}), aborting.")
            sys.exit(1)
        print("Extraction done.\n", flush=True)

    for i, params in enumerate(combos, 1):
        key = combo_key(params)

        if key in results:
            val = results[key].get("val_acc")
            acc_str = f"{val:.1%}" if val is not None else "ERROR"
            print(f"[{i}/{len(combos)}] SKIP (val={acc_str}): {key}")
            continue

        cmd = build_cmd(params)
        print(f"\n[{i}/{len(combos)}] START: {key}")
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60, flush=True)

        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)

        # Read val_acc from checkpoint (train_arbiter saves it there)
        val_acc = None
        ckpt_copy = None
        if SOURCE_CKPT.exists() and proc.returncode == 0:
            val_acc = read_val_acc_from_checkpoint(SOURCE_CKPT)
            fname = combo_filename(params) + ".pt"
            ckpt_copy = str(CHECKPOINT_DIR / fname)
            shutil.copy2(SOURCE_CKPT, ckpt_copy)
            print(f"  Checkpoint: {ckpt_copy}")

        # Save immediately so we don't lose this run on crash
        results[key] = {
            "params": params,
            "val_acc": val_acc,
            "returncode": proc.returncode,
            "checkpoint": ckpt_copy,
            "timestamp": datetime.now().isoformat(),
        }
        save_results(results)

        acc_str = f"{val_acc:.1%}" if val_acc is not None else "N/A"
        print(f"  Done. val_acc={acc_str}, returncode={proc.returncode}", flush=True)

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print_summary(results)

    # Copy best checkpoint back to meter_arbiter.pt
    best_key, best_info = max(results.items(), key=lambda x: x[1].get("val_acc") or 0)
    if best_info.get("checkpoint") and Path(best_info["checkpoint"]).exists():
        shutil.copy2(best_info["checkpoint"], SOURCE_CKPT)
        print(f"Best checkpoint ({best_key}, val={best_info['val_acc']:.1%}) copied to meter_arbiter.pt")


if __name__ == "__main__":
    main()
