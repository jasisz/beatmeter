#!/usr/bin/env python3
"""Check orthogonality of onset MLP vs engine on METER2800.

Compares onset MLP predictions against engine predictions from a saved eval run.
Reports agreement rate, complementarity ratio, gains/losses, and gate check.

Usage:
    # Using latest saved engine run:
    uv run python scripts/training/check_onset_orthogonality.py

    # Using specific engine run:
    uv run python scripts/training/check_onset_orthogonality.py --run data/runs/20260215_120000.json

    # With specific MLP checkpoint:
    uv run python scripts/training/check_onset_orthogonality.py --checkpoint data/meter_onset_mlp.pt

    # Filter by meter:
    uv run python scripts/training/check_onset_orthogonality.py --meter 7 --verbose

Prerequisites:
    # Save engine predictions first (takes ~2.5h):
    uv run python scripts/eval.py --split test --limit 0 --workers 4 --save
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.training.train_onset_mlp import (
    FEATURE_VERSION,
    OnsetMLP,
    _cache_key,
    _standardize,
    extract_features,
)
from scripts.utils import load_meter2800_entries, resolve_audio_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


# ---------------------------------------------------------------------------
# Onset MLP predictions
# ---------------------------------------------------------------------------


def get_onset_predictions(
    entries: list[tuple[Path, int]],
    checkpoint_path: Path,
    device: str = "cpu",
) -> dict[str, int]:
    """Get onset MLP predictions for all entries. Returns {filename: predicted_meter}."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_meters = ckpt["class_meters"]
    meter_to_idx = ckpt["meter_to_idx"]
    idx_to_meter = {v: k for k, v in meter_to_idx.items()}
    input_dim = ckpt["input_dim"]
    n_classes = ckpt["n_classes"]
    feat_mean = ckpt.get("feat_mean")
    feat_std = ckpt.get("feat_std")

    model = OnsetMLP(input_dim, n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"  Classes: {class_meters}")
    print(f"  Input dim: {input_dim}, Feature version: {ckpt.get('feature_version', '?')}")
    if ckpt.get("best_val_acc"):
        print(f"  Val accuracy: {ckpt['best_val_acc']:.1%}")

    # Extract features with caching
    cache_dir = PROJECT_ROOT / f"data/onset_features_cache_{FEATURE_VERSION}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    skipped = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(entries, desc="Onset MLP inference")
    except ImportError:
        iterator = entries

    for audio_path, _meter in iterator:
        key = _cache_key(audio_path)
        cache_path = cache_dir / f"{key}.npy"

        if cache_path.exists():
            feat = np.load(cache_path)
        else:
            feat = extract_features(audio_path)
            if feat is not None:
                np.save(cache_path, feat)

        if feat is None:
            skipped += 1
            continue

        # Standardize
        feat = feat.astype(np.float32)
        if feat_mean is not None and feat_std is not None:
            feat = (feat - feat_mean) / np.where(feat_std < 1e-8, 1.0, feat_std)

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = logits.argmax(dim=1).item()

        pred_meter = idx_to_meter.get(pred_idx)
        if pred_meter is not None:
            predictions[audio_path.name] = pred_meter

    if skipped:
        print(f"  ({skipped} files skipped — extraction failed)")

    return predictions


# ---------------------------------------------------------------------------
# Engine predictions from run snapshot
# ---------------------------------------------------------------------------


def load_engine_predictions_from_run(run_path: Path) -> dict[str, int | None]:
    """Load engine predictions from a saved eval.py run."""
    data = json.loads(run_path.read_text())
    preds = {}
    for entry in data.get("files", []):
        preds[entry["fname"]] = entry.get("predicted")
    return preds


def find_latest_run() -> Path | None:
    """Find the most recent run snapshot."""
    if not RUNS_DIR.exists():
        return None
    runs = sorted(RUNS_DIR.glob("*.json"))
    if not runs:
        return None
    return runs[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Check onset MLP orthogonality vs engine on METER2800"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/meter_onset_mlp.pt"))
    parser.add_argument("--run", type=Path, default=None,
                        help="Engine run snapshot JSON (default: latest)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--meter", type=int, default=0,
                        help="Filter to specific meter class")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()

    # Load entries
    print(f"Loading METER2800 {args.split} split...")
    entries = load_meter2800_entries(data_dir, args.split)
    if not entries:
        print("ERROR: No entries found")
        sys.exit(1)

    if args.meter > 0:
        entries = [(p, m) for p, m in entries if m == args.meter]
    print(f"  {len(entries)} files")

    # Load onset MLP
    print(f"\nLoading onset MLP from {checkpoint_path}...")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    onset_preds = get_onset_predictions(entries, checkpoint_path, args.device)
    print(f"  Got predictions for {len(onset_preds)}/{len(entries)} files")

    # Load engine predictions
    run_path = args.run or find_latest_run()
    if not run_path or not run_path.exists():
        print("\nERROR: No engine run snapshot found.")
        print("  Run first:  uv run python scripts/eval.py --split test --limit 0 --workers 4 --save")
        sys.exit(1)

    print(f"\nLoading engine predictions from {run_path.name}...")
    engine_preds = load_engine_predictions_from_run(run_path)
    print(f"  Got predictions for {len(engine_preds)} files")

    # Compare
    both_correct = 0
    both_wrong = 0
    engine_only = 0  # engine right, MLP wrong → losses
    mlp_only = 0     # MLP right, engine wrong → gains

    gains = []
    losses = []
    skipped = 0

    for audio_path, expected in entries:
        fname = audio_path.name
        if fname not in onset_preds or fname not in engine_preds:
            skipped += 1
            continue

        mlp_pred = onset_preds[fname]
        engine_pred = engine_preds[fname]

        if engine_pred is None:
            skipped += 1
            continue

        mlp_ok = (mlp_pred == expected)
        engine_ok = (engine_pred == expected)

        if engine_ok and mlp_ok:
            both_correct += 1
        elif not engine_ok and not mlp_ok:
            both_wrong += 1
        elif engine_ok and not mlp_ok:
            engine_only += 1
            losses.append((fname, expected, engine_pred, mlp_pred))
        else:
            mlp_only += 1
            gains.append((fname, expected, engine_pred, mlp_pred))

        if args.verbose:
            marker = ""
            if mlp_ok and not engine_ok:
                marker = " ** GAIN"
            elif engine_ok and not mlp_ok:
                marker = " ** LOSS"
            if marker or not (mlp_ok and engine_ok):
                print(f"  {fname}: engine={engine_pred} mlp={mlp_pred} "
                      f"expected={expected}{marker}")

    total = both_correct + both_wrong + engine_only + mlp_only

    # Report
    print(f"\n{'=' * 60}")
    print(f"Onset MLP Orthogonality Report — METER2800 {args.split} ({total} files)")
    print(f"{'=' * 60}")

    if total == 0:
        print("No files evaluated!")
        return

    agreement = (both_correct + both_wrong) / total
    mlp_acc = (both_correct + mlp_only) / total
    engine_acc = (both_correct + engine_only) / total

    print(f"\nAccuracy:")
    print(f"  Engine:    {engine_acc:.1%} ({both_correct + engine_only}/{total})")
    print(f"  Onset MLP: {mlp_acc:.1%} ({both_correct + mlp_only}/{total})")

    print(f"\nAgreement matrix:")
    print(f"  Both correct:  {both_correct:4d}  ({both_correct/total:.1%})")
    print(f"  Both wrong:    {both_wrong:4d}  ({both_wrong/total:.1%})")
    print(f"  Engine only:   {engine_only:4d}  ({engine_only/total:.1%}) [LOSSES if MLP added]")
    print(f"  MLP only:      {mlp_only:4d}  ({mlp_only/total:.1%}) [GAINS if MLP added]")

    print(f"\nKey metrics:")
    print(f"  Agreement rate: {agreement:.1%}  (target: 0.65-0.80, NOT >0.85)")
    if engine_only > 0:
        ratio = mlp_only / engine_only
        print(f"  Complementarity ratio: {ratio:.2f}  (gains/losses, target: >1.5)")
    else:
        print(f"  Complementarity ratio: inf  (no losses)")

    if skipped > 0:
        print(f"  Skipped: {skipped} (missing predictions)")

    # Per-class breakdown
    print(f"\nPer-class accuracy:")
    report_meters = sorted({m for _, m in entries})
    for m in report_meters:
        m_entries = [
            (p, e) for p, e in entries
            if e == m and p.name in onset_preds and p.name in engine_preds
        ]
        if not m_entries:
            continue
        m_total = len(m_entries)
        m_mlp_ok = sum(1 for p, e in m_entries if onset_preds.get(p.name) == e)
        m_eng_ok = sum(1 for p, e in m_entries if engine_preds.get(p.name) == e)
        m_gains = sum(
            1 for p, e in m_entries
            if onset_preds.get(p.name) == e and engine_preds.get(p.name) != e
        )
        m_losses = sum(
            1 for p, e in m_entries
            if engine_preds.get(p.name) == e and onset_preds.get(p.name) != e
        )
        print(f"  {m}/x:  engine={m_eng_ok}/{m_total} ({m_eng_ok/m_total:.0%})  "
              f"mlp={m_mlp_ok}/{m_total} ({m_mlp_ok/m_total:.0%})  "
              f"gains=+{m_gains} losses=-{m_losses}")

    # Gate check
    print(f"\n{'=' * 60}")
    print("GATE CHECK:")
    gate_pass = True

    if agreement > 0.85:
        print(f"  FAIL: agreement {agreement:.1%} > 85% (not orthogonal enough)")
        gate_pass = False
    elif agreement < 0.50:
        print(f"  WARN: agreement {agreement:.1%} < 50% (MLP may be unreliable)")

    if engine_only > 0 and mlp_only / engine_only < 1.5:
        print(f"  FAIL: complementarity {mlp_only/engine_only:.2f} < 1.5 (not enough gains)")
        gate_pass = False

    if gate_pass:
        print(f"  PASS: Onset MLP signal is orthogonal and complementary!")
    print(f"{'=' * 60}")

    # Detail on gains/losses
    if gains:
        print(f"\nGAINS ({len(gains)} files where MLP correct, engine wrong):")
        for fname, expected, eng, mlp in sorted(gains, key=lambda x: x[1]):
            print(f"  {fname}: engine={eng} mlp={mlp} expected={expected}")

    if losses:
        print(f"\nLOSSES ({len(losses)} files where engine correct, MLP wrong):")
        for fname, expected, eng, mlp in sorted(losses, key=lambda x: x[1])[:50]:
            print(f"  {fname}: engine={eng} mlp={mlp} expected={expected}")
        if len(losses) > 50:
            print(f"  ... and {len(losses) - 50} more")


if __name__ == "__main__":
    main()
