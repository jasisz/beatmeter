#!/usr/bin/env python3
"""Grid search over MeterNet training hyperparameters.

Runs train.py with all combinations of parameters, saves each
checkpoint and results. Resumes safely — already completed runs are skipped.

Usage:
    uv run python scripts/training/grid.py
    uv run python scripts/training/grid.py --dry-run   # show combos only
    uv run python scripts/training/grid.py --summary   # show results table
"""

import argparse
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "data" / "meter_net_grid_results.json"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "meter_net_grid"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "train.py"

# -- Grid definition --

GRID = {
    "lr": [3e-4],
    "hidden": [256, 384, 512, 640, 1024],
    "dropout_scale": [1.5],
    "n_blocks": [1, 2],
    "no_beat_features": [False],
}

FIXED_ARGS = [
    "--meter2800", "data/meter2800",
    "--epochs", "200",
    "--workers", "4",
]


def combo_key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def combo_filename(params: dict) -> str:
    parts = []
    for k, v in sorted(params.items()):
        safe_v = str(v).replace(".", "p").replace("-", "m")
        parts.append(f"{k}_{safe_v}")
    return "_".join(parts)


def build_cmd(params: dict) -> list[str]:
    cmd = ["uv", "run", "python", str(TRAIN_SCRIPT)] + FIXED_ARGS
    cmd += ["--lr", str(params["lr"])]
    cmd += ["--hidden", str(params["hidden"])]
    cmd += ["--dropout-scale", str(params["dropout_scale"])]
    cmd += ["--n-blocks", str(params.get("n_blocks", 1))]
    cmd += ["--batch-size", "64"]
    if params.get("no_beat_features"):
        cmd += ["--no-beat-features"]
    return cmd


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def read_val_acc_from_checkpoint(path: Path) -> float | None:
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
    print("-" * 80)
    for i, (key, info) in enumerate(rows, 1):
        val = info.get("val_acc")
        acc_str = f"{val:.1%}" if val is not None else "ERROR"
        marker = " <- BEST" if i == 1 else ""
        rc = info.get("returncode", "?")
        rc_str = "" if rc == 0 else f" [rc={rc}]"
        print(f"  {i:<2}  {acc_str:<10} {key}{rc_str}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Results cleared.")

    results = load_results()

    if args.summary:
        print_summary(results)
        return

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
            save_path = CHECKPOINT_DIR / f"{combo_filename(c)}.pt"
            cmd = build_cmd(c) + ["--save", str(save_path)]
            print(f"[{status}] {key}")
            print(f"       {' '.join(cmd)}")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(combos, 1):
        key = combo_key(params)

        if key in results:
            val = results[key].get("val_acc")
            acc_str = f"{val:.1%}" if val is not None else "ERROR"
            print(f"[{i}/{len(combos)}] SKIP (val={acc_str}): {key}")
            continue

        save_path = CHECKPOINT_DIR / f"{combo_filename(params)}.pt"
        cmd = build_cmd(params) + ["--save", str(save_path)]
        print(f"\n[{i}/{len(combos)}] START: {key}")
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60, flush=True)

        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)

        val_acc = None
        if save_path.exists() and proc.returncode == 0:
            val_acc = read_val_acc_from_checkpoint(save_path)
            print(f"  Checkpoint: {save_path}")

        results[key] = {
            "params": params,
            "val_acc": val_acc,
            "returncode": proc.returncode,
            "checkpoint": str(save_path) if save_path.exists() else None,
            "timestamp": datetime.now().isoformat(),
        }
        save_results(results)

        acc_str = f"{val_acc:.1%}" if val_acc is not None else "N/A"
        print(f"  Done. val_acc={acc_str}, returncode={proc.returncode}", flush=True)

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print_summary(results)

    real_results = {k: v for k, v in results.items() if not k.startswith("_")}
    best_key, best_info = max(real_results.items(), key=lambda x: x[1].get("val_acc") or 0)

    # Mark winner explicitly in results
    from beatmeter.experiment import get_git_info, log_experiment, make_experiment_record
    results["_winner"] = best_key
    results["_winner_val_acc"] = best_info["val_acc"]
    results["_timestamp"] = datetime.now().isoformat()
    results["_git"] = get_git_info()
    save_results(results)

    log_experiment(make_experiment_record(
        type="grid_search", model="meter_net",
        params=best_info["params"],
        results={"val_acc": best_info["val_acc"],
                 "n_combos": len(real_results)},
        checkpoint=best_info.get("checkpoint"),
    ))

    # Print promote command for the winner
    best_ckpt = best_info.get("checkpoint")
    if best_ckpt and Path(best_ckpt).exists():
        print(f"\nBest: {best_key} (val={best_info['val_acc']:.1%})")
        print(f"To promote:\n  uv run python scripts/eval.py --promote {best_ckpt} --workers 4")

    # Cleanup: remove non-winner checkpoints
    for key, info in real_results.items():
        ckpt = info.get("checkpoint")
        if ckpt and ckpt != best_info.get("checkpoint") and Path(ckpt).exists():
            Path(ckpt).unlink()
            print(f"  Cleaned: {Path(ckpt).name}")


if __name__ == "__main__":
    main()
