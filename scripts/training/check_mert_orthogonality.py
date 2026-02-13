#!/usr/bin/env python3
"""Check orthogonality of MERT classifier vs engine on METER2800.

Compares MERT predictions (from legacy embeddings or finetuned audio inference)
against engine predictions (from eval.py run snapshot or live eval).

Reports agreement rate, complementarity ratio, gains/losses, and gate check.

Usage:
    # Using saved eval run (fast, no recomputation):
    uv run python scripts/training/check_mert_orthogonality.py --run data/runs/20260213_120000.json

    # Live engine eval + notebook-aligned finetuned checkpoint:
    uv run python scripts/training/check_mert_orthogonality.py

    # With specific checkpoint (legacy or finetuned):
    uv run python scripts/training/check_mert_orthogonality.py --checkpoint data/meter_mert_finetuned.pt

    # Filter by meter:
    uv run python scripts/training/check_mert_orthogonality.py --meter 3 --verbose
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.training.finetune_mert import (
    MERTClassificationHead,
    MAX_DURATION_S,
    MERT_SR,
    mert_forward_pool,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUN_DIRS = [
    PROJECT_ROOT / "data" / "runs",
    PROJECT_ROOT / ".cache" / "meter2800" / "runs",  # legacy layout
    PROJECT_ROOT / ".cache" / "runs",                 # older docs/reference layout
]


# ---------------------------------------------------------------------------
# MERT prediction from embeddings
# ---------------------------------------------------------------------------


def _normalize_class_map(raw_map: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, v in raw_map.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def load_mert_classifier(checkpoint_path: Path, device: torch.device):
    """Load MERT model/checkpoint metadata.

    Supports two checkpoint formats:
    - legacy embedding classifier (`classifier_state_dict`)
    - notebook LoRA fine-tuning checkpoint (`head_state_dict`)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Legacy embedding classifier (train_meter_mert.py)
    if "classifier_state_dict" in ckpt:
        multi_layer = ckpt.get("multi_layer", False)
        class_map = _normalize_class_map(ckpt.get("class_map", {0: 3, 1: 4, 2: 5, 3: 7}))

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
            "mode": "legacy_embedding",
            "multi_layer": multi_layer,
            "class_map": class_map,
            "layer_idx": ckpt.get("layer_idx"),
            "num_layers": ckpt.get("num_layers"),
            "test_accuracy": ckpt.get("test_accuracy"),
            "model_name": ckpt.get("model_name", "unknown"),
        }

    # Notebook-aligned LoRA fine-tuning checkpoint (finetune_mert.py)
    if "head_state_dict" in ckpt:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        model_name = ckpt.get("model_name", "m-a-p/MERT-v1-330M")
        num_layers = int(ckpt.get("num_layers", 24))
        hidden_dim = int(ckpt.get("hidden_dim", 1024))
        pooled_dim = int(ckpt.get("pooled_dim", hidden_dim * 2))
        head_dim = int(ckpt.get("head_dim", 256))
        num_classes = int(ckpt.get("num_classes", 6))
        dropout = float(ckpt.get("dropout", 0.4))
        class_map = _normalize_class_map(ckpt.get("class_map", {0: 3, 1: 4, 2: 5, 3: 7, 4: 9, 5: 11}))

        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        mert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)

        lora_state = ckpt.get("lora_state_dict")
        if lora_state:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise RuntimeError("peft is required to evaluate LoRA checkpoints") from e
            lora_config = LoraConfig(
                r=int(ckpt.get("lora_rank", 16)),
                lora_alpha=int(ckpt.get("lora_alpha", 32)),
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            mert_model = get_peft_model(mert_model, lora_config)

        mert_model = mert_model.to(device)
        mert_model.eval()

        if lora_state:
            for name, param_data in lora_state.items():
                parts = name.split(".")
                obj = mert_model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                getattr(obj, parts[-1]).data.copy_(param_data.to(device))

        head = MERTClassificationHead(
            num_layers=num_layers,
            pooled_dim=pooled_dim,
            num_classes=num_classes,
            head_dim=head_dim,
            dropout=dropout,
        ).to(device)
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()

        model = {
            "mert_model": mert_model,
            "processor": processor,
            "head": head,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        }
        return model, {
            "mode": "finetuned_audio",
            "class_map": class_map,
            "test_accuracy": ckpt.get("test_accuracy"),
            "model_name": model_name,
            "num_layers": num_layers,
        }

    raise RuntimeError("Unsupported checkpoint format")


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
    multi_layer = meta.get("multi_layer", False)
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
            pred_meter = class_map.get(pred_idx)
            if pred_meter is not None:
                predictions[audio_path.name] = pred_meter
        except Exception:
            continue

    return predictions


def _load_audio_for_mert(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=MERT_SR, mono=True)
    max_samples = MAX_DURATION_S * MERT_SR
    if len(audio) > max_samples:
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]
    if len(audio) < MERT_SR:
        audio = np.pad(audio, (0, MERT_SR - len(audio)))
    return audio.astype(np.float32)


@torch.no_grad()
def get_finetuned_predictions(
    entries: list[tuple[Path, int]],
    model: dict,
    meta: dict,
    device: torch.device,
) -> dict[str, int]:
    """Get predictions from notebook-style LoRA checkpoint by audio inference."""
    predictions: dict[str, int] = {}
    class_map = meta["class_map"]
    mert_model = model["mert_model"]
    processor = model["processor"]
    head = model["head"]
    num_layers = model["num_layers"]
    hidden_dim = model["hidden_dim"]

    for audio_path, _meter in tqdm(entries, desc="MERT finetuned inference"):
        try:
            audio = _load_audio_for_mert(audio_path)
            pooled = mert_forward_pool(
                [audio],
                mert_model,
                processor,
                device,
                num_layers,
                hidden_dim,
            )
            logits = head(pooled)
            pred_idx = int(logits.argmax(dim=1).item())
            pred_meter = class_map.get(pred_idx)
            if pred_meter is not None:
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


def find_latest_run(run_dirs: list[Path]) -> Path | None:
    """Find the most recent run snapshot across known run directories."""
    candidates: list[Path] = []
    for run_dir in run_dirs:
        if run_dir.exists():
            candidates.extend(run_dir.glob("*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
                        default=Path("data/meter_mert_finetuned.pt"),
                        help="MERT checkpoint (finetuned audio or legacy embedding classifier)")
    parser.add_argument("--run", type=Path, default=None,
                        help="Engine run snapshot JSON (default: latest from known run dirs)")
    parser.add_argument("--split", default="test",
                        help="METER2800 split to evaluate (default: test)")
    parser.add_argument("--meter", type=int, default=0,
                        help="Filter to specific meter class")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
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
    print(f"  Model: {meta['model_name']}, mode={meta['mode']}")
    if meta.get("test_accuracy"):
        print(f"  Standalone test accuracy: {meta['test_accuracy']:.1%}")

    # Get MERT predictions
    if meta["mode"] == "legacy_embedding":
        print(f"\nGetting MERT predictions from embeddings ({embeddings_dir})...")
        mert_preds = get_mert_predictions(entries, embeddings_dir, model, meta, device)
    else:
        print("\nGetting MERT predictions via finetuned audio inference...")
        mert_preds = get_finetuned_predictions(entries, model, meta, device)
    print(f"  Got predictions for {len(mert_preds)}/{len(entries)} files")

    # Get engine predictions
    run_path = args.run
    if run_path is None:
        run_path = find_latest_run(DEFAULT_RUN_DIRS)

    engine_preds: dict[str, int | None] = {}
    if run_path and run_path.exists():
        print(f"\nLoading engine predictions from {run_path.name}...")
        engine_preds = load_engine_predictions_from_run(run_path)
        print(f"  Got predictions for {len(engine_preds)} files")
    else:
        print("\nWARNING: No engine run snapshot found.")
        print("  Run: uv run python scripts/eval.py --split test --limit 0 --save")
        print("  Searched:")
        for run_dir in DEFAULT_RUN_DIRS:
            print(f"    - {run_dir}")
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
    report_meters = sorted({m for _, m in entries})
    for m in report_meters:
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
