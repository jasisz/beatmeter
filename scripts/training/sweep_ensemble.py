#!/usr/bin/env python3
"""Sweep ensemble weights (MLP vs FTT) on METER2800.

Tests global and per-class weight combinations to find optimal mixing.

Usage:
    uv run --group training python scripts/training/sweep_ensemble.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from beatmeter.analysis.meter import _build_meter_net, _build_ftt, _build_hybrid


def load_checkpoint_and_model(ckpt_path: str):
    """Load checkpoint and build corresponding model."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model_type = ckpt.get("model_type", "mlp")
    input_dim = ckpt["input_dim"]
    n_classes = ckpt["n_classes"]
    hidden = ckpt.get("hidden_dim", 640)
    ds = ckpt.get("dropout_scale", 1.0)
    n_blocks = ckpt.get("n_blocks", 1)
    state = ckpt["model_state"]

    if model_type == "hybrid":
        model = _build_hybrid(input_dim, n_classes, hidden, ds,
                              ckpt.get("mlp_blocks", n_blocks),
                              ckpt.get("ftt_d_model", 128),
                              ckpt.get("ftt_n_layers", 2), state)
    elif model_type == "ftt":
        model = _build_ftt(input_dim, n_classes, hidden, n_blocks, state)
    else:
        model = _build_meter_net(input_dim, n_classes, hidden, ds, n_blocks)
        fixed = {}
        for k, v in state.items():
            if k.startswith("residual.block."):
                fixed["residual.0." + k[len("residual."):]] = v
            else:
                fixed[k] = v
        state = fixed

    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def get_probs(model, X_raw, feat_mean, feat_std):
    """Standardize features and get sigmoid probabilities."""
    std = feat_std.copy()
    std[std < 1e-8] = 1.0
    X = ((X_raw - feat_mean) / std).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def accuracy(probs, true_labels):
    return int((probs.argmax(1) == true_labels).sum())


def per_class_report(probs, true_labels, idx_to_meter, n_classes):
    preds = probs.argmax(1)
    parts = []
    for ci in range(n_classes):
        mask = true_labels == ci
        if mask.sum() == 0:
            continue
        correct = int((preds[mask] == ci).sum())
        total = int(mask.sum())
        parts.append(f"{idx_to_meter[ci]}/x:{correct}/{total}")
    return "  ".join(parts)


def main():
    # Import train.py infrastructure for feature loading
    import scripts.training.train as T
    from beatmeter.analysis.cache import NumpyLMDB

    # Set up MERT globals
    T._MERT_LAYER_ACTIVE = 3
    T._MERT_LOOKUP = T._build_mert_lookup(
        Path("data/mert_embeddings/meter2800"),
        Path("data/mert_embeddings/wikimeter"),
    )
    if T._MERT_LOOKUP:
        T._N_MERT_ACTIVE = T._detect_mert_dim(T._MERT_LOOKUP, T._MERT_LAYER_ACTIVE)
        T._AUDIO_FEATURES_ACTIVE = T.TOTAL_FEATURES_V6
        T._TOTAL_FEATURES_ACTIVE = T._AUDIO_FEATURES_ACTIVE + T._N_MERT_ACTIVE

    valid_meters = set(T.CLASS_METERS_6)
    idx_to_meter = {i: m for i, m in enumerate(T.CLASS_METERS_6)}
    n_classes = len(T.CLASS_METERS_6)

    # Load METER2800 val + test
    val_entries = T.load_meter2800_split(Path("data/meter2800"), "val", valid_meters)
    test_entries = T.load_meter2800_split(Path("data/meter2800"), "test", valid_meters)

    feat_db = NumpyLMDB("data/features.lmdb")
    print("Loading features...")
    X_val, y_val, _ = T.extract_all_features(val_entries, feat_db, "val", workers=4)
    X_test, y_test, _ = T.extract_all_features(test_entries, feat_db, "test", workers=4)
    true_val = y_val.argmax(axis=1)
    true_test = y_test.argmax(axis=1)
    print(f"  val={len(X_val)}, test={len(X_test)}")

    # Load models
    print("\nLoading models...")
    mlp_model, mlp_ckpt = load_checkpoint_and_model("data/meter_net.pt")
    ftt_model, ftt_ckpt = load_checkpoint_and_model("data/meter_net_ftt.pt")
    print(f"  MLP: {mlp_ckpt.get('model_type', 'mlp')}, dim={mlp_ckpt['input_dim']}")
    print(f"  FTT: {ftt_ckpt.get('model_type', 'ftt')}, dim={ftt_ckpt['input_dim']}")

    # Get probs
    val_p_mlp = get_probs(mlp_model, X_val, mlp_ckpt["feat_mean"], mlp_ckpt["feat_std"])
    val_p_ftt = get_probs(ftt_model, X_val, ftt_ckpt["feat_mean"], ftt_ckpt["feat_std"])
    test_p_mlp = get_probs(mlp_model, X_test, mlp_ckpt["feat_mean"], mlp_ckpt["feat_std"])
    test_p_ftt = get_probs(ftt_model, X_test, ftt_ckpt["feat_mean"], ftt_ckpt["feat_std"])

    N_val = len(true_val)
    N_test = len(true_test)

    # ── 1. Global weight sweep ──
    print(f"\n{'='*55}")
    print("GLOBAL WEIGHT SWEEP: probs = w·MLP + (1-w)·FTT")
    print(f"{'='*55}")
    print(f"{'w_mlp':>6}  {'val':>10}  {'test':>10}")
    print("-" * 35)

    best_w, best_val = 0.5, 0
    for wi in range(0, 21):
        w = wi / 20.0
        va = accuracy(w * val_p_mlp + (1 - w) * val_p_ftt, true_val)
        ta = accuracy(w * test_p_mlp + (1 - w) * test_p_ftt, true_test)
        marker = ""
        if va > best_val:
            best_val = va
            best_w = w
            marker = " *BEST"
        print(f"  {w:.2f}   {va:>4}/{N_val}    {ta:>4}/{N_test}{marker}")

    # Report best global
    best_test_probs = best_w * test_p_mlp + (1 - best_w) * test_p_ftt
    best_test_acc = accuracy(best_test_probs, true_test)
    print(f"\nBest global: w_mlp={best_w:.2f} → test={best_test_acc}/{N_test}")
    print(f"  {per_class_report(best_test_probs, true_test, idx_to_meter, n_classes)}")

    # ── 2. Per-class weight sweep ──
    print(f"\n{'='*55}")
    print("PER-CLASS WEIGHT SWEEP")
    print(f"{'='*55}")

    pc_w = np.full(n_classes, 0.5)
    for ci in range(n_classes):
        mask = true_val == ci
        if mask.sum() == 0:
            continue
        best_ci_w, best_ci_v = 0.5, 0
        for wi in range(0, 21):
            w = wi / 20.0
            mixed = w * val_p_mlp[mask] + (1 - w) * val_p_ftt[mask]
            c = int((mixed.argmax(1) == ci).sum())
            if c > best_ci_v or (c == best_ci_v and abs(w - 0.5) < abs(best_ci_w - 0.5)):
                best_ci_v = c
                best_ci_w = w
        pc_w[ci] = best_ci_w
        m = idx_to_meter[ci]
        print(f"  {m}/x: w_mlp={best_ci_w:.2f}  val={best_ci_v}/{mask.sum()}")

    # Apply per-class weights
    test_pc = np.zeros_like(test_p_mlp)
    for ci in range(n_classes):
        test_pc[:, ci] = pc_w[ci] * test_p_mlp[:, ci] + (1 - pc_w[ci]) * test_p_ftt[:, ci]

    pc_acc = accuracy(test_pc, true_test)
    print(f"\nPer-class → test={pc_acc}/{N_test}")
    print(f"  {per_class_report(test_pc, true_test, idx_to_meter, n_classes)}")

    # ── 3. Comparison ──
    print(f"\n{'='*55}")
    print("COMPARISON")
    print(f"{'='*55}")
    eq_acc = accuracy(0.5 * test_p_mlp + 0.5 * test_p_ftt, true_test)
    mlp_acc = accuracy(test_p_mlp, true_test)
    ftt_acc = accuracy(test_p_ftt, true_test)
    print(f"  MLP solo:             {mlp_acc}/{N_test}")
    print(f"  FTT solo:             {ftt_acc}/{N_test}")
    print(f"  Equal (0.50/0.50):    {eq_acc}/{N_test}")
    print(f"  Best global (w={best_w:.2f}): {best_test_acc}/{N_test}")
    print(f"  Per-class weights:    {pc_acc}/{N_test}")

    # ── 4. Bonus: try adding hybrid as 3rd model ──
    hybrid_path = Path("data/checkpoints/20260224_113612.pt")
    if hybrid_path.exists():
        print(f"\n{'='*55}")
        print("BONUS: 3-WAY ENSEMBLE (MLP + FTT + Hybrid)")
        print(f"{'='*55}")

        hyb_model, hyb_ckpt = load_checkpoint_and_model(str(hybrid_path))
        test_p_hyb = get_probs(hyb_model, X_test, hyb_ckpt["feat_mean"], hyb_ckpt["feat_std"])
        val_p_hyb = get_probs(hyb_model, X_val, hyb_ckpt["feat_mean"], hyb_ckpt["feat_std"])

        # Sweep w_mlp, w_ftt (w_hyb = 1 - w_mlp - w_ftt)
        best_3w, best_3val, best_3test = (0.33, 0.33), 0, 0
        for wm in range(0, 21, 2):
            for wf in range(0, 21 - wm, 2):
                wm_f = wm / 20.0
                wf_f = wf / 20.0
                wh_f = 1.0 - wm_f - wf_f
                if wh_f < -0.01:
                    continue
                va = accuracy(wm_f * val_p_mlp + wf_f * val_p_ftt + wh_f * val_p_hyb, true_val)
                if va > best_3val:
                    best_3val = va
                    best_3w = (wm_f, wf_f)
                    ta = accuracy(wm_f * test_p_mlp + wf_f * test_p_ftt + wh_f * test_p_hyb, true_test)
                    best_3test = ta

        wm, wf = best_3w
        wh = 1.0 - wm - wf
        print(f"  Best: MLP={wm:.2f} FTT={wf:.2f} Hybrid={wh:.2f}")
        test_3 = wm * test_p_mlp + wf * test_p_ftt + wh * test_p_hyb
        acc3 = accuracy(test_3, true_test)
        print(f"  Test: {acc3}/{N_test}")
        print(f"  {per_class_report(test_3, true_test, idx_to_meter, n_classes)}")


if __name__ == "__main__":
    main()
