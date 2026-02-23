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
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "data" / "meter_net_grid_results.json"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "meter_net_grid"
LOG_DIR = CHECKPOINT_DIR / "logs"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "training" / "train.py"

# -- Grid definition --

GRID = {
    "lr": [3e-4],
    "hidden": [512],
    "dropout_scale": [1.5],
    "n_blocks": [2],
    "mert_layer": [3],
    "kfold": [5],
}
# Grid: audio+MERT only (2985d), 5-fold CV

FIXED_ARGS = [
    "--meter2800", "data/meter2800",
    "--epochs", "200",
    "--workers", "4",
]


def combo_key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def combo_filename(params: dict) -> str:
    safe_lr = str(params["lr"]).replace(".", "p").replace("-", "m")
    safe_drop = str(params["dropout_scale"]).replace(".", "p").replace("-", "m")
    parts = [
        f"lr_{safe_lr}",
        f"h_{params['hidden']}",
        f"d_{safe_drop}",
        f"b_{params.get('n_blocks', 1)}",
    ]
    if params.get("no_mert"):
        parts.append("no_mert")
    if params.get("mert_layer") is not None:
        parts.append(f"ml_{params['mert_layer']}")
    am = params.get("aug_mode", "cutmix")
    if am != "cutmix":
        parts.append(f"aug_{am}")
    kf = params.get("kfold", 0)
    if kf > 0:
        parts.append(f"kf_{kf}")
    if params.get("seed") is not None:
        parts.append(f"s_{params['seed']}")
    return "_".join(parts)


def build_cmd(params: dict) -> list[str]:
    cmd = ["uv", "run", "python", str(TRAIN_SCRIPT)] + FIXED_ARGS
    cmd += ["--lr", str(params["lr"])]
    cmd += ["--hidden", str(params["hidden"])]
    cmd += ["--dropout-scale", str(params["dropout_scale"])]
    cmd += ["--n-blocks", str(params.get("n_blocks", 1))]
    cmd += ["--batch-size", "64"]
    if params.get("no_mert"):
        cmd += ["--no-mert"]
    if params.get("mert_layer") is not None:
        cmd += ["--mert-layer", str(params["mert_layer"])]
    am = params.get("aug_mode", "cutmix")
    if am != "cutmix":
        cmd += ["--aug-mode", am]
    kf = params.get("kfold", 0)
    if kf > 0:
        cmd += ["--kfold", str(kf)]
    if params.get("seed") is not None:
        cmd += ["--seed", str(params["seed"])]
    return cmd


def combo_label(params: dict) -> str:
    parts = [
        f"lr={params['lr']}",
        f"h={params['hidden']}",
        f"d={params['dropout_scale']}",
        f"b={params.get('n_blocks', 1)}",
    ]
    if params.get("no_mert"):
        parts.append("no_mert=1")
    if params.get("mert_layer") is not None:
        parts.append(f"layer={params['mert_layer']}")
    am = params.get("aug_mode", "cutmix")
    if am != "cutmix":
        parts.append(f"aug={am}")
    kf = params.get("kfold", 0)
    if kf > 0:
        parts.append(f"kfold={kf}")
    if params.get("seed") is not None:
        parts.append(f"seed={params['seed']}")
    return ", ".join(parts)


def cmd_value(cmd: list[str], flag: str, default=None):
    try:
        idx = cmd.index(flag)
    except ValueError:
        return default
    if idx + 1 < len(cmd):
        return cmd[idx + 1]
    return default


def parse_prune_spec(spec: str) -> dict | None:
    """Parse prune spec.

    Supported:
    - "3" -> gap 3pp, start from best run epoch (recommended)
    - "3,winner" -> same as above
    - "3,40" -> gap 3pp, start at epoch 40
    """
    s = (spec or "").strip().lower()
    if not s or s in {"0", "off", "false", "none"}:
        return None

    if "," in s:
        gap_s, after_s = s.split(",", 1)
    else:
        gap_s, after_s = s, "winner"

    try:
        gap_pp = float(gap_s.strip())
    except ValueError as e:
        raise ValueError("Invalid --prune gap. Use numeric pp value, e.g. '3' or '3,40'.") from e

    if gap_pp <= 0:
        raise ValueError("--prune gap_pp must be > 0.")

    after_s = after_s.strip()
    if after_s in {"winner", "best", "auto"}:
        return {
            "gap_frac": gap_pp / 100.0,
            "mode": "winner_epoch",
            "after_epoch": None,
        }

    try:
        after_epoch = int(float(after_s))
    except ValueError as e:
        raise ValueError(
            "Invalid --prune format. Use '3' (winner epoch) or '3,<epoch>' e.g. '3,40'."
        ) from e

    if after_epoch < 1:
        raise ValueError("--prune after_epoch must be >= 1.")

    return {
        "gap_frac": gap_pp / 100.0,
        "mode": "fixed_epoch",
        "after_epoch": after_epoch,
    }


def short_log_ref(path: Path, verbose: bool = False) -> str:
    if verbose:
        return str(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def run_train_logged(
    cmd: list[str],
    log_path: Path,
    dot_every_epochs: int,
    epochs_total: int | None,
    env_overrides: dict[str, str] | None = None,
    prune_cfg: dict | None = None,
    prune_baseline_val: float | None = None,
    prune_after_epoch: int | None = None,
    run_prefix: str = "",
    kfold_prune_min: int = 3,
) -> tuple[int, int | None, int | None, bool, str | None]:
    """Run train command, write combined output to log_path, print progress dots.

    Parses FOLD_DONE markers to display K-fold progress and optionally prune
    weak configs after kfold_prune_min folds.
    """
    epoch_re = re.compile(r"\bep\s+(\d+)/(\d+)\b")
    epoch_val_re = re.compile(r"\bep\s+(\d+)/(\d+).*?\bval_bal\s+([0-9.]+)%")
    early_stop_re = re.compile(r"Early stopping at epoch\s+(\d+)")
    best_epoch_re = re.compile(r"Best val balanced acc:\s+[0-9.]+%\s+at epoch\s+(\d+)")
    fold_done_re = re.compile(
        r"FOLD_DONE\s+fold=(\d+)/(\d+)\s+val=([0-9.]+)\s+test=(\d+)/(\d+)\s+ep=(\d+)"
    )

    last_epoch = None
    parsed_total = epochs_total
    early_stop_epoch = None
    best_epoch = None
    last_dot_epoch = -1
    fold_val_accs: list[float] = []

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    pruned = False
    prune_reason = None

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# started={datetime.now().isoformat()}\n")
        f.write(f"# cmd={' '.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)

            # K-fold progress: parse FOLD_DONE markers
            fold_match = fold_done_re.search(line)
            if fold_match:
                fold_n = int(fold_match.group(1))
                fold_total = int(fold_match.group(2))
                fold_val = float(fold_match.group(3))
                test_correct = int(fold_match.group(4))
                test_total = int(fold_match.group(5))
                fold_ep = int(fold_match.group(6))
                fold_val_accs.append(fold_val)
                test_pct = test_correct / max(test_total, 1)
                mean_val_so_far = sum(fold_val_accs) / len(fold_val_accs)
                print(
                    f"{run_prefix} fold {fold_n}/{fold_total} "
                    f"val={fold_val:.1%} test={test_correct}/{test_total} ({test_pct:.1%}) "
                    f"ep={fold_ep} [mean_val={mean_val_so_far:.1%}]",
                    flush=True,
                )
                # K-fold prune: after enough folds, check running mean
                if (
                    prune_cfg is not None
                    and prune_baseline_val is not None
                    and len(fold_val_accs) >= kfold_prune_min
                    and fold_n < fold_total  # don't prune on last fold
                ):
                    gap_frac = prune_cfg["gap_frac"]
                    threshold = prune_baseline_val - gap_frac
                    if mean_val_so_far < threshold:
                        pruned = True
                        prune_reason = (
                            f"fold {fold_n}/{fold_total}: "
                            f"mean_val={mean_val_so_far:.1%} < {threshold:.1%} "
                            f"(best={prune_baseline_val:.1%}, gap={gap_frac*100:.1f}pp)"
                        )
                        f.write(f"\n# PRUNED {prune_reason}\n")
                        print(
                            f"{run_prefix} PRUNED {prune_reason}",
                            flush=True,
                        )
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        break
                continue

            match = epoch_re.search(line)
            if match:
                ep = int(match.group(1))
                tot = int(match.group(2))
                last_epoch = ep
                parsed_total = tot
                if (
                    dot_every_epochs > 0
                    and ep != last_dot_epoch
                    and ep % dot_every_epochs == 0
                ):
                    print(".", end="", flush=True)
                    last_dot_epoch = ep

            early = early_stop_re.search(line)
            if early:
                early_stop_epoch = int(early.group(1))

            best = best_epoch_re.search(line)
            if best:
                best_epoch = int(best.group(1))

            ep_val = epoch_val_re.search(line)
            if (
                ep_val
                and prune_cfg is not None
                and prune_baseline_val is not None
            ):
                ep = int(ep_val.group(1))
                val_bal = float(ep_val.group(3)) / 100.0
                gap_frac = prune_cfg["gap_frac"]
                effective_after = prune_after_epoch
                if effective_after is None:
                    continue
                threshold = prune_baseline_val - gap_frac
                if ep >= effective_after and val_bal < threshold:
                    pruned = True
                    prune_reason = (
                        f"ep{ep}: val_bal={val_bal:.1%} < {threshold:.1%} "
                        f"(best={prune_baseline_val:.1%}, gap={gap_frac*100:.1f}pp, "
                        f"start_ep={effective_after})"
                    )
                    f.write(f"\n# PRUNED {prune_reason}\n")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

        rc = proc.wait()

    epochs_done = early_stop_epoch or best_epoch or last_epoch
    return rc, epochs_done, parsed_total, pruned, prune_reason


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def best_completed_state(results: dict) -> tuple[float | None, int | None]:
    best_val = None
    best_epoch = None
    for _, info in results.items():
        if not isinstance(info, dict):
            continue
        if info.get("returncode") != 0:
            continue
        if not info.get("checkpoint"):
            continue
        val = info.get("val_acc")
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_epoch = info.get("best_epoch")
    return best_val, best_epoch


def resolve_prune_after_epoch(
    prune_cfg: dict | None,
    winner_epoch: int | None,
    prune_max_start: int,
) -> int | None:
    if prune_cfg is None:
        return None
    if prune_cfg["mode"] == "fixed_epoch":
        return prune_cfg["after_epoch"]

    # winner_epoch mode: start from winner epoch but cap it to avoid very late pruning.
    if winner_epoch is None:
        return prune_max_start
    return min(winner_epoch, prune_max_start)


def save_results(results: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def read_metrics_from_checkpoint(path: Path) -> tuple[float | None, int | None]:
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("best_val_acc"), ckpt.get("best_epoch")
    except Exception as e:
        print(f"  Warning: could not read metrics from checkpoint: {e}")
        return None, None


def print_summary(results: dict, top: int = 10) -> None:
    if not results:
        print("No results yet.")
        return
    rows = sorted(
        ((k, v) for k, v in results.items() if not k.startswith("_")),
        key=lambda x: x[1].get("val_acc") or 0, reverse=True,
    )
    if not rows:
        print("No results yet.")
        return
    top = max(top, 1)
    shown = rows[:top]
    print(f"\nTop {len(shown)}/{len(rows)} runs by val_acc")
    print(f"{'#':<4} {'Val acc':<10} Parameters")
    print("-" * 80)
    for i, (key, info) in enumerate(shown, 1):
        val = info.get("val_acc")
        acc_str = f"{val:.1%}" if val is not None else "ERROR"
        marker = " <- BEST" if i == 1 else ""
        rc = info.get("returncode", "?")
        flags = []
        if rc != 0:
            flags.append(f"rc={rc}")
        if info.get("pruned"):
            flags.append("pruned")
        flag_str = f" [{' '.join(flags)}]" if flags else ""
        print(f"  {i:<2}  {acc_str:<10} {key}{flag_str}{marker}")
    print()

    # Grouped summary: aggregate by non-seed params, show mean±std
    if "seed" not in GRID:
        return
    groups: dict[str, list[float]] = {}
    for key, info in results.items():
        if key.startswith("_"):
            continue
        val = info.get("val_acc")
        if val is None:
            continue
        # Strip seed from key to get group key
        parts = [p for p in key.split(",") if not p.startswith("seed=")]
        group_key = ",".join(parts)
        groups.setdefault(group_key, []).append(val)

    if not groups:
        return

    ranked = sorted(groups.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    print("Grouped by config (mean ± std across seeds)")
    print(f"{'#':<4} {'Mean':<8} {'Std':<8} {'N':<4} Parameters")
    print("-" * 80)
    for i, (gkey, vals) in enumerate(ranked, 1):
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
        marker = " <- BEST" if i == 1 else ""
        print(f"  {i:<2}  {mean_v:.1%}    {std_v:.1%}    {len(vals):<4} {gkey}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N rows in summary output")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra details (commands, cleanup list)")
    parser.add_argument("--stream-train", action="store_true",
                        help="Stream full train.py output (no log capture)")
    parser.add_argument("--epoch-tick", type=int, default=10,
                        help="Print one dot every N epochs in RUN line (0=off, capture mode only)")
    parser.add_argument(
        "--prune",
        type=str,
        default="",
        help="Prune weak runs: '3' (from winner epoch) or '3,40' (fixed epoch). Disabled by default.",
    )
    parser.add_argument(
        "--prune-max-start",
        type=int,
        default=60,
        help="In winner-epoch prune mode, start no later than this epoch (default: 60).",
    )
    parser.add_argument("--prune-min-folds", type=int, default=3,
                        help="K-fold prune: minimum folds before pruning (default: 3)")
    parser.add_argument("--parallel-runs", type=int, default=1,
                        help="Run up to N training jobs in parallel (CPU mode)")
    parser.add_argument("--threads-per-run", type=int, default=0,
                        help="Set OMP/BLAS threads per run in parallel mode (0=auto)")
    parser.add_argument("--keep-top", type=int, default=2,
                        help="Keep top N checkpoints after grid search (default: 2)")
    parser.add_argument("--final-top", type=int, default=3,
                        help="Number of top candidates for multi-seed finale (default: 3)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds for finale phase (default: 5)")
    parser.add_argument("--no-finale", action="store_true",
                        help="Skip the multi-seed finale phase")
    args = parser.parse_args()

    if args.parallel_runs < 1:
        parser.error("--parallel-runs must be >= 1")
    if args.threads_per_run < 0:
        parser.error("--threads-per-run must be >= 0")
    if args.prune_max_start < 1:
        parser.error("--prune-max-start must be >= 1")
    if args.stream_train and args.parallel_runs > 1:
        parser.error("--stream-train cannot be used with --parallel-runs > 1")
    try:
        prune_cfg = parse_prune_spec(args.prune)
    except ValueError as e:
        parser.error(str(e))
    if args.stream_train and prune_cfg is not None:
        parser.error("--prune requires log-capture mode (disable --stream-train)")

    if args.reset and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Results cleared.")

    results = load_results()

    if args.summary:
        print_summary(results, top=args.top)
        return

    keys = list(GRID.keys())
    raw_combos = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]

    # Deduplicate combos
    seen: set[str] = set()
    combos: list[dict] = []
    for c in raw_combos:
        k = combo_key(c)
        if k not in seen:
            seen.add(k)
            combos.append(c)

    real_results = {k: v for k, v in results.items() if not k.startswith("_")}
    todo = [c for c in combos if combo_key(c) not in real_results]
    print(f"Grid search: {len(combos)} combinations total")
    print(f"Already done: {len(real_results)}, remaining: {len(todo)}")
    if prune_cfg is not None:
        gap_pp = prune_cfg["gap_frac"] * 100.0
        if prune_cfg["mode"] == "winner_epoch":
            print(
                f"Prune: enabled (drop if val_bal < best-{gap_pp:.1f}pp from winner epoch, "
                f"capped at {args.prune_max_start})"
            )
        else:
            print(
                f"Prune: enabled (drop if val_bal < best-{gap_pp:.1f}pp from epoch "
                f"{prune_cfg['after_epoch']}+)"
            )
    print()

    if args.dry_run:
        for c in combos:
            key = combo_key(c)
            status = "DONE" if key in real_results else "TODO"
            save_path = CHECKPOINT_DIR / f"{combo_filename(c)}.pt"
            cmd = build_cmd(c) + ["--save", str(save_path)]
            print(f"[{status}] {combo_label(c)}")
            if args.verbose:
                print(f"       {' '.join(cmd)}")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.stream_train:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.parallel_runs > 1:
        cpu_n = os.cpu_count() or args.parallel_runs
        auto_threads = max(1, cpu_n // args.parallel_runs)
        threads_per_run = args.threads_per_run or auto_threads
        env_overrides = {
            "OMP_NUM_THREADS": str(threads_per_run),
            "MKL_NUM_THREADS": str(threads_per_run),
            "OPENBLAS_NUM_THREADS": str(threads_per_run),
            "NUMEXPR_NUM_THREADS": str(threads_per_run),
            "VECLIB_MAXIMUM_THREADS": str(threads_per_run),
            "BLIS_NUM_THREADS": str(threads_per_run),
        }
        print(
            f"Parallel mode: {args.parallel_runs} runs in-flight, "
            f"threads/run={threads_per_run}"
        )
        if args.epoch_tick > 0:
            print("  Note: epoch dots are disabled in parallel mode.")
        best_lock = threading.Lock()
        best_val0, best_epoch0 = best_completed_state(results)
        best_shared = {"val": best_val0, "epoch": best_epoch0}

        pending: list[tuple[int, dict]] = []
        for i, params in enumerate(combos, 1):
            key = combo_key(params)
            label = combo_label(params)
            if key in results:
                val = results[key].get("val_acc")
                acc_str = f"{val:.1%}" if val is not None else "ERROR"
                print(f"[{i}/{len(combos)}] SKIP {label} | val={acc_str}")
                continue

            pending.append((i, params))

        def _run_one(item_i: int, item_params: dict) -> dict:
            key = combo_key(item_params)
            label = combo_label(item_params)
            save_path = CHECKPOINT_DIR / f"{combo_filename(item_params)}.pt"
            cmd = build_cmd(item_params) + ["--save", str(save_path)]
            log_path = LOG_DIR / f"{combo_filename(item_params)}.log"
            epochs_cfg = int(cmd_value(cmd, "--epochs", 0) or 0)
            epoch_plan = f"{epochs_cfg}" if epochs_cfg > 0 else "?"
            print(f"[{item_i}/{len(combos)}] RUN  {label} | epochs={epoch_plan}", flush=True)

            t0 = time.time()
            kfold_active = item_params.get("kfold", 0) > 1
            with best_lock:
                prune_baseline_val = best_shared["val"]
                # Disable epoch-level prune for K-fold (only fold-level prune)
                prune_after_epoch = None if kfold_active else resolve_prune_after_epoch(
                    prune_cfg=prune_cfg,
                    winner_epoch=best_shared["epoch"],
                    prune_max_start=args.prune_max_start,
                )
            run_pfx = f"[{item_i}/{len(combos)}]"
            rc, epochs_done, epochs_total, pruned, prune_reason = run_train_logged(
                cmd=cmd,
                log_path=log_path,
                dot_every_epochs=0,
                epochs_total=(epochs_cfg or None),
                env_overrides=env_overrides,
                prune_cfg=prune_cfg,
                prune_baseline_val=prune_baseline_val,
                prune_after_epoch=prune_after_epoch,
                run_prefix=run_pfx,
                kfold_prune_min=args.prune_min_folds,
            )
            elapsed = time.time() - t0

            val_acc = None
            best_epoch = None
            if save_path.exists() and rc == 0:
                val_acc, best_epoch = read_metrics_from_checkpoint(save_path)

            return {
                "i": item_i,
                "key": key,
                "params": item_params,
                "rc": rc,
                "val_acc": val_acc,
                "elapsed": elapsed,
                "epochs_done": epochs_done,
                "epochs_total": epochs_total,
                "pruned": pruned,
                "prune_reason": prune_reason,
                "best_epoch": best_epoch,
                "checkpoint": str(save_path) if save_path.exists() else None,
                "log": str(log_path),
            }

        with ThreadPoolExecutor(max_workers=args.parallel_runs) as pool:
            futures = [pool.submit(_run_one, i, params) for i, params in pending]
            for future in as_completed(futures):
                out = future.result()
                key = out["key"]
                run_prefix = f"[{out['i']}/{len(combos)}]"

                results[key] = {
                    "params": out["params"],
                    "val_acc": out["val_acc"],
                    "returncode": out["rc"],
                    "checkpoint": out["checkpoint"],
                    "timestamp": datetime.now().isoformat(),
                    "log": out["log"],
                    "pruned": out["pruned"],
                    "prune_reason": out["prune_reason"],
                    "best_epoch": out["best_epoch"],
                }
                save_results(results)
                with best_lock:
                    best_shared["val"], best_shared["epoch"] = best_completed_state(results)

                best_val, _ = best_completed_state(results)
                best_str = f"{best_val:.1%}" if best_val is not None else "N/A"
                acc_str = f"{out['val_acc']:.1%}" if out["val_acc"] is not None else "N/A"
                status = "PRUNE" if out["pruned"] else ("DONE" if out["rc"] == 0 else "FAIL")

                epochs_str = ""
                if out["epochs_total"]:
                    done_str = "?" if out["epochs_done"] is None else str(out["epochs_done"])
                    epochs_str = f" | ep={done_str}/{out['epochs_total']}"

                log_info = f" | log={short_log_ref(Path(out['log']), verbose=args.verbose)}"
                prune_info = ""
                if out["pruned"] and out["prune_reason"]:
                    prune_info = f" | {out['prune_reason']}"
                print(
                    f"{run_prefix} {status} val={acc_str} | rc={out['rc']} | "
                    f"{out['elapsed']:.1f}s{epochs_str} | best={best_str}{log_info}{prune_info}",
                    flush=True,
                )
    else:
        for i, params in enumerate(combos, 1):
            key = combo_key(params)
            label = combo_label(params)

            if key in results:
                val = results[key].get("val_acc")
                acc_str = f"{val:.1%}" if val is not None else "ERROR"
                print(f"[{i}/{len(combos)}] SKIP {label} | val={acc_str}")
                continue

            save_path = CHECKPOINT_DIR / f"{combo_filename(params)}.pt"
            cmd = build_cmd(params) + ["--save", str(save_path)]
            log_path = LOG_DIR / f"{combo_filename(params)}.log"
            run_prefix = f"[{i}/{len(combos)}]"
            epochs_cfg = int(cmd_value(cmd, "--epochs", 0) or 0)
            epoch_plan = f"{epochs_cfg}" if epochs_cfg > 0 else "?"

            if args.stream_train or args.epoch_tick <= 0:
                print(f"{run_prefix} RUN  {label} | epochs={epoch_plan}", flush=True)
            else:
                print(f"{run_prefix} RUN  {label} | epochs={epoch_plan} ", end="", flush=True)
            if args.verbose:
                print(f"  CMD: {' '.join(cmd)}")

            t0 = time.time()
            if args.stream_train:
                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
                epochs_done = None
                epochs_total = epochs_cfg or None
                rc = proc.returncode
                pruned = False
                prune_reason = None
            else:
                kfold_active = params.get("kfold", 0) > 1
                prune_baseline_val, prune_winner_epoch = best_completed_state(results)
                # Disable epoch-level prune for K-fold (only fold-level prune)
                prune_after_epoch = None if kfold_active else resolve_prune_after_epoch(
                    prune_cfg=prune_cfg,
                    winner_epoch=prune_winner_epoch,
                    prune_max_start=args.prune_max_start,
                )
                rc, epochs_done, epochs_total, pruned, prune_reason = run_train_logged(
                    cmd=cmd,
                    log_path=log_path,
                    dot_every_epochs=max(args.epoch_tick, 0),
                    epochs_total=(epochs_cfg or None),
                    prune_cfg=prune_cfg,
                    prune_baseline_val=prune_baseline_val,
                    prune_after_epoch=prune_after_epoch,
                    run_prefix=run_prefix,
                    kfold_prune_min=args.prune_min_folds,
                )
            elapsed = time.time() - t0

            if not args.stream_train and args.epoch_tick > 0:
                print("", flush=True)

            val_acc = None
            best_epoch = None
            if save_path.exists() and rc == 0:
                val_acc, best_epoch = read_metrics_from_checkpoint(save_path)

            results[key] = {
                "params": params,
                "val_acc": val_acc,
                "returncode": rc,
                "checkpoint": str(save_path) if save_path.exists() else None,
                "timestamp": datetime.now().isoformat(),
                "log": None if args.stream_train else str(log_path),
                "pruned": pruned,
                "prune_reason": prune_reason,
                "best_epoch": best_epoch,
            }
            save_results(results)

            best_val, _ = best_completed_state(results)
            best_str = f"{best_val:.1%}" if best_val is not None else "N/A"
            acc_str = f"{val_acc:.1%}" if val_acc is not None else "N/A"
            status = "PRUNE" if pruned else ("DONE" if rc == 0 else "FAIL")

            epochs_str = ""
            if epochs_total:
                done_str = "?" if epochs_done is None else str(epochs_done)
                epochs_str = f" | ep={done_str}/{epochs_total}"

            log_info = ""
            if not args.stream_train:
                log_info = f" | log={short_log_ref(log_path, verbose=args.verbose)}"
            prune_info = ""
            if pruned and prune_reason:
                prune_info = f" | {prune_reason}"
            print(
                f"{run_prefix} {status} val={acc_str} | rc={rc} | {elapsed:.1f}s{epochs_str} | "
                f"best={best_str}{log_info}{prune_info}",
                flush=True,
            )

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print_summary(results, top=args.top)

    real_results = {k: v for k, v in results.items() if not k.startswith("_")}
    completed = {
        k: v for k, v in real_results.items()
        if v.get("returncode") == 0 and v.get("checkpoint") and v.get("val_acc") is not None
    }
    if not completed:
        print("No successful completed checkpoints.")
        return

    # -----------------------------------------------------------------------
    # Phase 2: Multi-seed finale
    # -----------------------------------------------------------------------
    finale_top = min(args.final_top, len(completed))
    n_seeds = args.seeds

    seeds_in_grid = "seed" in GRID and len(GRID["seed"]) > 1
    kfold_in_grid = "kfold" in GRID and any(k > 1 for k in GRID["kfold"])
    if seeds_in_grid:
        print("Seeds are part of the grid — skipping Phase 2 finale (already multi-seed).")
    if kfold_in_grid:
        print("K-fold in grid — skipping Phase 2 finale (K-fold gives variance estimate).")

    if not seeds_in_grid and not kfold_in_grid and not args.no_finale and finale_top > 0 and n_seeds > 1:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: FINALE — top {finale_top} × {n_seeds} seeds")
        print("=" * 60)

        phase1_ranked = sorted(
            completed.items(),
            key=lambda x: x[1].get("val_acc") or 0,
            reverse=True,
        )
        finalists = phase1_ranked[:finale_top]

        # Default seed used in phase 1
        default_seed = 42
        extra_seeds = [s for s in range(1, n_seeds + 1) if s != default_seed]
        # Include default_seed result from phase 1 + run extra seeds
        extra_seeds = extra_seeds[: n_seeds - 1]  # we already have 1 from phase 1

        finale_results: dict[str, dict] = {}  # key -> {params, seeds: {seed: val_acc}}
        finale_dir = CHECKPOINT_DIR / "finale"
        finale_dir.mkdir(parents=True, exist_ok=True)
        finale_log_dir = finale_dir / "logs"
        finale_log_dir.mkdir(parents=True, exist_ok=True)

        for rank, (key, info) in enumerate(finalists, 1):
            params = info["params"]
            label = combo_label(params)
            phase1_val = info["val_acc"]
            print(f"\n  Finalist #{rank}: {label} (phase1 val={phase1_val:.1%})")

            seed_vals = {default_seed: phase1_val}  # reuse phase 1 result

            for seed in extra_seeds:
                fname = f"{combo_filename(params)}_s{seed}"
                save_path = finale_dir / f"{fname}.pt"
                log_path = finale_log_dir / f"{fname}.log"

                if save_path.exists():
                    sv, _ = read_metrics_from_checkpoint(save_path)
                    if sv is not None:
                        seed_vals[seed] = sv
                        print(f"    seed={seed}: SKIP (cached val={sv:.1%})")
                        continue

                cmd = build_cmd(params) + ["--seed", str(seed), "--save", str(save_path)]
                print(f"    seed={seed}: ", end="", flush=True)

                t0 = time.time()
                rc, epochs_done, _, _, _ = run_train_logged(
                    cmd=cmd,
                    log_path=log_path,
                    dot_every_epochs=max(args.epoch_tick, 0),
                    epochs_total=None,
                )
                elapsed = time.time() - t0
                print("", flush=True)  # newline after dots

                if save_path.exists() and rc == 0:
                    sv, _ = read_metrics_from_checkpoint(save_path)
                    if sv is not None:
                        seed_vals[seed] = sv
                        print(f"    seed={seed}: val={sv:.1%} ({elapsed:.0f}s)")
                    else:
                        print(f"    seed={seed}: FAIL (no val_acc in checkpoint)")
                else:
                    print(f"    seed={seed}: FAIL (rc={rc}, {elapsed:.0f}s)")

            vals = list(seed_vals.values())
            mean_val = sum(vals) / len(vals)
            std_val = (sum((v - mean_val) ** 2 for v in vals) / len(vals)) ** 0.5
            finale_results[key] = {
                "params": params,
                "seeds": seed_vals,
                "mean_val": mean_val,
                "std_val": std_val,
                "n_seeds": len(vals),
            }
            print(f"    => mean={mean_val:.1%} ± {std_val:.1%} (n={len(vals)})")

        # Rank by mean val_balanced
        finale_ranked = sorted(
            finale_results.items(),
            key=lambda x: x[1]["mean_val"],
            reverse=True,
        )

        print(f"\n{'=' * 60}")
        print("FINALE RESULTS")
        print(f"{'#':<4} {'Mean±Std':<16} {'n':>3}  Parameters")
        print("-" * 80)
        for i, (key, finfo) in enumerate(finale_ranked, 1):
            m = finfo["mean_val"]
            s = finfo["std_val"]
            n = finfo["n_seeds"]
            marker = " <- WINNER" if i == 1 else ""
            print(f"  {i:<2}  {m:.1%} ± {s:.1%}  {n:>3}  {combo_label(finfo['params'])}{marker}")
        print()

        # Winner from finale
        best_key, best_finale = finale_ranked[0]
        best_info = completed[best_key]

        # Find best seed checkpoint for the winner
        best_seed = max(best_finale["seeds"], key=best_finale["seeds"].get)
        if best_seed == default_seed:
            winner_ckpt = best_info.get("checkpoint")
        else:
            fname = f"{combo_filename(best_info['params'])}_s{best_seed}"
            winner_ckpt = str(finale_dir / f"{fname}.pt")

        # Save finale results
        results["_finale"] = {
            "rankings": [
                {
                    "key": key,
                    "mean_val": finfo["mean_val"],
                    "std_val": finfo["std_val"],
                    "n_seeds": finfo["n_seeds"],
                    "seeds": finfo["seeds"],
                }
                for key, finfo in finale_ranked
            ],
            "winner_key": best_key,
            "winner_mean_val": best_finale["mean_val"],
            "winner_std_val": best_finale["std_val"],
            "winner_best_seed": best_seed,
            "winner_checkpoint": winner_ckpt,
        }

        # Cleanup finale: keep only winner's best seed checkpoint
        for fname_pt in finale_dir.glob("*.pt"):
            if str(fname_pt) != winner_ckpt:
                fname_pt.unlink()

    else:
        best_key, best_info = max(completed.items(), key=lambda x: x[1].get("val_acc") or 0)
        winner_ckpt = best_info.get("checkpoint")

    # -----------------------------------------------------------------------
    # Wrap up
    # -----------------------------------------------------------------------
    from beatmeter.experiment import get_git_info, log_experiment, make_experiment_record
    results["_winner"] = best_key
    results["_winner_val_acc"] = best_info.get("val_acc")
    results["_timestamp"] = datetime.now().isoformat()
    results["_git"] = get_git_info()
    save_results(results)

    log_experiment(make_experiment_record(
        type="grid_search", model="meter_net",
        params=best_info["params"],
        results={"val_acc": best_info.get("val_acc"),
                 "n_combos": len(completed)},
        checkpoint=best_info.get("checkpoint"),
    ))

    # Print promote command
    if winner_ckpt and Path(winner_ckpt).exists():
        val = best_info.get("val_acc")
        val_str = f"{val:.1%}" if val is not None else "N/A"
        finale_str = ""
        if "_finale" in results:
            fm = results["_finale"]["winner_mean_val"]
            fs = results["_finale"]["winner_std_val"]
            finale_str = f", finale mean={fm:.1%}±{fs:.1%}"
        print(f"\nWinner: {best_key} (phase1 val={val_str}{finale_str})")
        print(f"To promote:\n  uv run python scripts/eval.py --promote {winner_ckpt} --workers 4")

    # Cleanup phase 1: keep top N checkpoints, remove the rest
    keep_n = max(args.keep_top, 1)
    ranked = sorted(
        completed.items(),
        key=lambda x: x[1].get("val_acc") or 0,
        reverse=True,
    )
    keep_ckpts = {info.get("checkpoint") for _, info in ranked[:keep_n]}
    if winner_ckpt:
        keep_ckpts.add(winner_ckpt)
    cleaned = []
    for key, info in completed.items():
        ckpt = info.get("checkpoint")
        if ckpt and ckpt not in keep_ckpts and Path(ckpt).exists():
            Path(ckpt).unlink()
            cleaned.append(Path(ckpt).name)
    kept_names = [Path(c).name for c in keep_ckpts if c and Path(c).exists()]
    if kept_names:
        print(f"\nKept top {len(kept_names)} checkpoints:")
        for name in kept_names:
            print(f"  {name}")
    if cleaned:
        print(f"Cleaned {len(cleaned)} lower-ranked checkpoints.")
        if args.verbose:
            for name in cleaned:
                print(f"  Cleaned: {name}")


if __name__ == "__main__":
    main()
