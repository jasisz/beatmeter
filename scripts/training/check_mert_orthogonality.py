#!/usr/bin/env python3
"""Check orthogonality of MERT classifier vs engine on METER2800.

Compares MERT predictions (from pre-extracted embeddings + trained classifier)
against engine predictions (from eval.py run snapshot or live eval).

Reports agreement rate, complementarity ratio, gains/losses, and gate check.

Usage:
    # Using saved eval run (fast, no recomputation):
    uv run python scripts/training/check_mert_orthogonality.py --run data/runs/latest.json

    # Live engine eval + MERT from embeddings:
    uv run python scripts/training/check_mert_orthogonality.py

    # With specific MERT checkpoint:
    uv run python scripts/training/check_mert_orthogonality.py --checkpoint data/meter_mert_multilayer.pt

    # Filter by meter:
    uv run python scripts/training/check_mert_orthogonality.py --meter 3 --verbose
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import resolve_audio_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_METERS = [3, 4, 5, 7]
METER_TO_IDX = {m: i for i, m in enumerate(CLASS_METERS)}
IDX_TO_METER = {i: m for i, m in enumerate(CLASS_METERS)}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# MERT prediction from embeddings
# ---------------------------------------------------------------------------


def load_mert_classifier(checkpoint_path: Path, device: torch.device):
    """Load MERT classifier from checkpoint. Returns (model, metadata)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    multi_layer = ckpt.get("multi_layer", False)
    class_map = ckpt.get("class_map", IDX_TO_METER)

    if multi_layer:
        from scripts.training.train_meter_mert import MultiLayerMLP
        model = MultiLayerMLP(
            num_layers=ckpt["num_layers"],
            pooled_dim=ckpt["pooled_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt.get("dropout", 0.3),
            layer_drop=ckpt.get("layer_drop", 0.0),
        )
    else:
        from scripts.training.train_meter_mert import MeterMLP
        model = MeterMLP(
            input_dim=ckpt.get("input_dim", 1536),
            hidden_dim=ckpt.get("hidden_dim", 256),
            num_classes=ckpt.get("num_classes", 4),
            dropout=ckpt.get("dropout", 0.3),
        )

    model.load_state_dict(ckpt["classifier_state_dict"])
    model.to(device)
    model.eval()

    return model, {
        "multi_layer": multi_layer,
        "class_map": class_map,
        "layer_idx": ckpt.get("layer_idx"),
        "num_layers": ckpt.get("num_layers"),
        "test_accuracy": ckpt.get("test_accuracy"),
        "model_name": ckpt.get("model_name", "unknown"),
    }


@torch.no_grad()
def get_mert_predictions(
    entries: list[tuple[Path, int]],
    embeddings_dir: Path,
    model: nn.Module,
    meta: dict,
    device: torch.device,
) -> dict[str, int]:
    """Get MERT predictions for all entries. Returns {filename: predicted_meter}."""
    predictions = {}
    multi_layer = meta["multi_layer"]
    class_map = meta["class_map"]
    layer_idx = meta.get("layer_idx")
    num_layers = meta.get("num_layers")

    for audio_path, _meter in entries:
        emb_path = embeddings_dir / f"{audio_path.stem}.npy"
        if not emb_path.exists():
            continue

        try:
            emb = np.load(emb_path)

            if multi_layer:
                x = torch.from_numpy(emb[:num_layers]).float().unsqueeze(0).to(device)
            else:
                x = torch.from_numpy(emb[layer_idx]).float().unsqueeze(0).to(device)

            logits = model(x)
            pred_idx = logits.argmax(dim=1).item()
            pred_meter = class_map.get(pred_idx, class_map.get(str(pred_idx)))
            if isinstance(pred_meter, str):
                pred_meter = int(pred_meter)
            predictions[audio_path.name] = pred_meter
        except Exception:
            continue

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


def find_latest_run(runs_dir: Path) -> Path | None:
    """Find the most recent run snapshot."""
    if not runs_dir.exists():
        return None
    runs = sorted(runs_dir.glob("*.json"))
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Check MERT orthogonality vs engine on METER2800"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--embeddings-dir", type=Path,
                        default=Path("data/mert_embeddings/meter2800"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/meter_mert_classifier.pt"),
                        help="MERT classifier checkpoint")
    parser.add_argument("--run", type=Path, default=None,
                        help="Engine run snapshot JSON (default: latest from data/runs/)")
    parser.add_argument("--split", default="test",
                        help="METER2800 split to evaluate (default: test)")
    parser.add_argument("--meter", type=int, default=0,
                        help="Filter to specific meter class")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_dir = args.data_dir.resolve()
    embeddings_dir = args.embeddings_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()

    # Load METER2800 entries
    print(f"Loading METER2800 {args.split} split...")
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from eval import load_meter2800_entries
    entries = load_meter2800_entries(data_dir, args.split)
    if not entries:
        print("ERROR: No entries found")
        sys.exit(1)

    if args.meter > 0:
        entries = [(p, m) for p, m in entries if m == args.meter]
    print(f"  {len(entries)} files")

    # Load MERT classifier
    print(f"\nLoading MERT classifier from {checkpoint_path}...")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    model, meta = load_mert_classifier(checkpoint_path, device)
    print(f"  Model: {meta['model_name']}, multi_layer={meta['multi_layer']}")
    if meta.get("test_accuracy"):
        print(f"  Standalone test accuracy: {meta['test_accuracy']:.1%}")

    # Get MERT predictions
    print(f"\nGetting MERT predictions from embeddings ({embeddings_dir})...")
    mert_preds = get_mert_predictions(entries, embeddings_dir, model, meta, device)
    print(f"  Got predictions for {len(mert_preds)}/{len(entries)} files")

    # Get engine predictions
    run_path = args.run
    if run_path is None:
        run_path = find_latest_run(PROJECT_ROOT / "data" / "runs")

    engine_preds: dict[str, int | None] = {}
    if run_path and run_path.exists():
        print(f"\nLoading engine predictions from {run_path.name}...")
        engine_preds = load_engine_predictions_from_run(run_path)
        print(f"  Got predictions for {len(engine_preds)} files")
    else:
        print("\nWARNING: No engine run snapshot found.")
        print("  Run: uv run python scripts/eval.py --split test --limit 0 --save")
        print("  Using ground truth as 'engine' to show MERT standalone accuracy.")
        # Fall back: treat ground truth as engine (just shows MERT accuracy)
        for audio_path, meter in entries:
            engine_preds[audio_path.name] = meter

    # Compare
    both_correct = 0
    both_wrong = 0
    engine_only = 0
    mert_only = 0
    skipped = 0

    gains = []
    losses = []

    for audio_path, expected in entries:
        fname = audio_path.name
        if fname not in mert_preds or fname not in engine_preds:
            skipped += 1
            continue

        mert_pred = mert_preds[fname]
        engine_pred = engine_preds[fname]

        if engine_pred is None:
            skipped += 1
            continue

        mert_ok = (mert_pred == expected)
        engine_ok = (engine_pred == expected)

        if engine_ok and mert_ok:
            both_correct += 1
        elif not engine_ok and not mert_ok:
            both_wrong += 1
        elif engine_ok and not mert_ok:
            engine_only += 1
            losses.append((fname, expected, engine_pred, mert_pred))
        else:
            mert_only += 1
            gains.append((fname, expected, engine_pred, mert_pred))

        if args.verbose:
            marker = ""
            if mert_ok and not engine_ok:
                marker = " ** GAIN"
            elif engine_ok and not mert_ok:
                marker = " ** LOSS"
            if marker or not (mert_ok and engine_ok):
                print(f"  {fname}: engine={engine_pred} mert={mert_pred} "
                      f"expected={expected}{marker}")

    total = both_correct + both_wrong + engine_only + mert_only

    # Report
    print(f"\n{'=' * 60}")
    print(f"MERT Orthogonality Report — METER2800 {args.split} ({total} files)")
    print(f"{'=' * 60}")

    if total == 0:
        print("No files evaluated!")
        return

    agreement = (both_correct + both_wrong) / total
    mert_acc = (both_correct + mert_only) / total
    engine_acc = (both_correct + engine_only) / total

    print(f"\nAccuracy:")
    print(f"  Engine: {engine_acc:.1%} ({both_correct + engine_only}/{total})")
    print(f"  MERT:   {mert_acc:.1%} ({both_correct + mert_only}/{total})")

    print(f"\nAgreement matrix:")
    print(f"  Both correct:  {both_correct:4d}  ({both_correct/total:.1%})")
    print(f"  Both wrong:    {both_wrong:4d}  ({both_wrong/total:.1%})")
    print(f"  Engine only:   {engine_only:4d}  ({engine_only/total:.1%}) [LOSSES if MERT added]")
    print(f"  MERT only:     {mert_only:4d}  ({mert_only/total:.1%}) [GAINS if MERT added]")

    print(f"\nKey metrics:")
    print(f"  Agreement rate: {agreement:.1%}  (target: 0.65-0.80, NOT >0.85)")
    if engine_only > 0:
        ratio = mert_only / engine_only
        print(f"  Complementarity ratio: {ratio:.2f}  (gains/losses, target: >1.5)")
    else:
        print(f"  Complementarity ratio: inf  (no losses)")

    if skipped > 0:
        print(f"  Skipped: {skipped} (missing embeddings or engine predictions)")

    # Per-class breakdown
    print(f"\nPer-class accuracy:")
    for m in CLASS_METERS:
        m_entries = [(p, e) for p, e in entries if e == m and p.name in mert_preds and p.name in engine_preds]
        if not m_entries:
            continue
        m_total = len(m_entries)
        m_mert_ok = sum(1 for p, e in m_entries if mert_preds.get(p.name) == e)
        m_eng_ok = sum(1 for p, e in m_entries if engine_preds.get(p.name) == e)
        print(f"  {m}/x:  engine={m_eng_ok}/{m_total} ({m_eng_ok/m_total:.0%})  "
              f"mert={m_mert_ok}/{m_total} ({m_mert_ok/m_total:.0%})")

    # Gate check
    print(f"\n{'=' * 60}")
    print("GATE CHECK:")
    gate_pass = True

    if agreement > 0.85:
        print(f"  FAIL: agreement {agreement:.1%} > 85% (not orthogonal enough)")
        gate_pass = False
    elif agreement < 0.50:
        print(f"  WARN: agreement {agreement:.1%} < 50% (MERT may be unreliable)")

    if engine_only > 0 and mert_only / engine_only < 1.5:
        print(f"  FAIL: complementarity {mert_only/engine_only:.2f} < 1.5 (not enough gains)")
        gate_pass = False

    if gate_pass:
        print(f"  PASS: MERT signal is orthogonal and complementary!")
    print(f"{'=' * 60}")

    # Detail on gains/losses
    if gains:
        print(f"\nGAINS ({len(gains)} files where MERT correct, engine wrong):")
        for fname, expected, eng, mert in gains[:30]:
            print(f"  {fname}: engine={eng} MERT={mert} expected={expected}")

    if losses:
        print(f"\nLOSSES ({len(losses)} files where engine correct, MERT wrong):")
        for fname, expected, eng, mert in losses[:30]:
            print(f"  {fname}: engine={eng} MERT={mert} expected={expected}")


if __name__ == "__main__":
    main()
