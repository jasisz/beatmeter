#!/usr/bin/env python3
"""Visualize MeterNet activations on real audio files.

Generates a PNG showing how the network processes different meter types:
- Row 1: Autocorrelation features (16×64 heatmap)
- Row 2: Signal scores (5×12 heatmap)
- Row 3: Hidden activations as fingerprint heatmaps
- Row 4: Residual activations as fingerprint heatmaps
- Row 5: Activation histograms
- Row 6: Output class probabilities

Usage:
    uv run python scripts/visualize_meter_net.py
    uv run python scripts/visualize_meter_net.py --average 20
    uv run python scripts/visualize_meter_net.py --meters 3 4 5 7 9 11
    uv run python scripts/visualize_meter_net.py --output my_plot.png
"""

import argparse
import sys
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from beatmeter.analysis.cache import AnalysisCache
from beatmeter.analysis.meter import _build_meter_net
from beatmeter.analysis.signals.meter_net_features import (
    extract_beat_features,
    extract_signal_scores,
    extract_tempo_features,
    TOTAL_FEATURES,
)
from beatmeter.analysis.signals.onset_mlp_features import (
    SR, MAX_DURATION_S, extract_features_v5,
)


METER_LABELS = {3: "3/x", 4: "4/x", 5: "5/x", 7: "7/x", 9: "9/x", 11: "11/x"}
CLASS_METERS = [3, 4, 5, 7, 9, 11]


def load_model():
    """Load MeterNet from checkpoint."""
    ckpt_path = PROJECT_ROOT / "data" / "meter_net.pt"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _build_meter_net(
        ckpt["input_dim"],
        ckpt["n_classes"],
        hidden=ckpt.get("hidden_dim", 640),
        dropout_scale=ckpt.get("dropout_scale", 1.0),
        n_blocks=ckpt.get("n_blocks", 1),
    )
    # Backward compat: old checkpoints have "residual.block.*" not "residual.0.block.*"
    state = ckpt["model_state"]
    fixed_state = {}
    for k, v in state.items():
        if k.startswith("residual.block."):
            fixed_state["residual.0." + k[len("residual."):]] = v
        else:
            fixed_state[k] = v
    model.load_state_dict(fixed_state)
    model.eval()

    info = {
        "hidden": ckpt.get("hidden_dim", 640),
        "n_blocks": ckpt.get("n_blocks", 1),
        "dropout_scale": ckpt.get("dropout_scale", 1.0),
        "feat_mean": ckpt.get("feat_mean"),
        "feat_std": ckpt.get("feat_std"),
    }
    return model, info


# ---------------------------------------------------------------------------
# Sample picking — METER2800 + WIKIMETER
# ---------------------------------------------------------------------------

def _collect_meter2800(meters_wanted: list[int], max_per_class: int) -> dict[int, list[Path]]:
    """Collect audio paths from METER2800 test split."""
    tab = PROJECT_ROOT / "data" / "meter2800" / "data_test_4_classes.tab"
    audio_dir = PROJECT_ROOT / "data" / "meter2800" / "audio"
    result: dict[int, list[Path]] = {m: [] for m in meters_wanted}

    if not tab.exists():
        return result

    for line in tab.read_text().splitlines()[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        fn = parts[0].strip('"').split("/")[-1].replace(".wav", "")
        meter = int(parts[2])
        if meter not in meters_wanted or len(result[meter]) >= max_per_class:
            continue
        path = audio_dir / f"{fn}.mp3"
        if not path.exists():
            path = audio_dir / f"{fn}.wav"
        if path.exists():
            result[meter].append(path)

    return result


_LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5, "seven": 7, "nine": 9, "eleven": 11,
}


def _collect_wikimeter(meters_wanted: list[int], max_per_class: int) -> dict[int, list[Path]]:
    """Collect audio paths from WIKIMETER."""
    tab = PROJECT_ROOT / "data" / "wikimeter" / "data_wikimeter.tab"
    audio_dir = PROJECT_ROOT / "data" / "wikimeter" / "audio"
    result: dict[int, list[Path]] = {m: [] for m in meters_wanted}

    if not tab.exists():
        return result

    for line in tab.read_text().splitlines()[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        fn = parts[0].strip('"').split("/")[-1].replace(".wav", "").replace(".mp3", "")
        label = parts[1].strip('"')
        meter = _LABEL_TO_METER.get(label)
        if meter is None or meter not in meters_wanted:
            continue
        if len(result[meter]) >= max_per_class:
            continue
        path = audio_dir / f"{fn}.mp3"
        if not path.exists():
            path = audio_dir / f"{fn}.wav"
        if path.exists():
            result[meter].append(path)

    return result


def collect_samples(
    meters_wanted: list[int], max_per_class: int,
) -> dict[int, list[Path]]:
    """Collect audio file paths from METER2800 + WIKIMETER."""
    m2800 = _collect_meter2800(meters_wanted, max_per_class)
    wiki = _collect_wikimeter(meters_wanted, max_per_class)

    result: dict[int, list[Path]] = {}
    for m in meters_wanted:
        combined = m2800.get(m, []) + wiki.get(m, [])
        result[m] = combined[:max_per_class]

    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(path: Path, cache: AnalysisCache) -> np.ndarray:
    """Extract full 1467-dim MeterNet feature vector."""
    y, sr = librosa.load(str(path), sr=SR, duration=MAX_DURATION_S)
    audio_feat = extract_features_v5(y, sr)

    audio_hash = cache.audio_hash(str(path))
    beat_feat = extract_beat_features(cache, audio_hash)
    signal_feat = extract_signal_scores(cache, audio_hash)
    tempo_feat = extract_tempo_features(cache, audio_hash)

    full = np.concatenate([audio_feat, beat_feat, signal_feat, tempo_feat])
    assert full.shape[0] == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {full.shape[0]}"
    return full


def run_with_hooks(
    model, feat_z: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run model and capture intermediate activations."""
    activations = {}

    def make_hook(name):
        def hook(mod, inp, out):
            activations[name] = out.detach().cpu().numpy()[0]
        return hook

    hooks = [
        model.input_proj.register_forward_hook(make_hook("proj")),
        model.residual.register_forward_hook(make_hook("residual")),
    ]

    x = torch.tensor(feat_z, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    for h in hooks:
        h.remove()

    return probs, activations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    all_data: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    meters: list[int],
    labels: dict[int, str],
    model_info: dict,
    output_path: Path,
):
    """Generate the visualization."""
    active_meters = [m for m in meters if m in all_data]
    n = len(active_meters)
    if n == 0:
        print("No data to plot.")
        return
    hidden = model_info["hidden"]

    # Compute grid size for fingerprint heatmaps
    side = int(np.ceil(np.sqrt(hidden)))

    fig = plt.figure(figsize=(4.5 * n, 22))
    title = (
        f"MeterNet Activations — hidden={model_info['hidden']}, "
        f"blocks={model_info['n_blocks']}, "
        f"dropout={model_info['dropout_scale']}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(6, n, hspace=0.5, wspace=0.3,
                           height_ratios=[1.0, 0.6, 1.2, 1.2, 0.7, 0.9])

    # Collect all activations for shared color scale
    all_proj = [all_data[m][2]["proj"] for m in active_meters]
    all_res = [all_data[m][2]["residual"] for m in active_meters]
    vmax_proj = max(np.abs(a).max() for a in all_proj)
    vmax_res = max(np.abs(a).max() for a in all_res)

    for col, meter in enumerate(active_meters):
        feat_z, probs, acts = all_data[meter]
        label = labels.get(meter, METER_LABELS.get(meter, str(meter)))

        # --- Row 0: Autocorrelation matrix (most important feature group) ---
        ax = fig.add_subplot(gs[0, col])
        autocorr = feat_z[:1024].reshape(16, 64)
        ax.imshow(autocorr, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.set_xlabel("lag", fontsize=7)
        ax.set_yticks(range(0, 16, 4))
        ax.set_yticklabels([f"t{i}" for i in range(4)], fontsize=7)
        ax.set_title(label, fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Autocorrelation\n(16×64)", fontsize=10)

        # --- Row 1: Signal scores heatmap (5 signals × 12 meters) ---
        ax = fig.add_subplot(gs[1, col])
        sig_data = feat_z[1403:1463].reshape(5, 12)
        sig_names = ["beatnet", "beat_this", "autocorr", "bar_trk", "hcdf"]
        m_names = ["2/4", "3/4", "3/8", "4/4", "5/4", "5/8",
                   "6/8", "7/4", "7/8", "9/8", "11/8", "12/8"]
        ax.imshow(sig_data, aspect="auto", cmap="YlOrRd",
                  vmin=0, vmax=max(sig_data.max(), 0.1), interpolation="nearest")
        ax.set_yticks(range(5))
        ax.set_yticklabels(sig_names, fontsize=6)
        ax.set_xticks(range(12))
        ax.set_xticklabels(m_names, fontsize=5, rotation=45, ha="right")
        if col == 0:
            ax.set_ylabel("Signal scores\n(5×12)", fontsize=10)

        # --- Row 2: Projection activations as fingerprint ---
        ax = fig.add_subplot(gs[2, col])
        act_proj = acts["proj"]
        padded = np.zeros(side * side)
        padded[:len(act_proj)] = act_proj
        img = padded.reshape(side, side)
        ax.imshow(img, cmap="RdYlGn", vmin=-vmax_proj, vmax=vmax_proj,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        n_active = (act_proj > 0).sum()
        ax.text(0.02, 0.98, f"{n_active}/{len(act_proj)} active",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
        if col == 0:
            ax.set_ylabel(f"Projection\n({hidden}d)", fontsize=10)

        # --- Row 3: Residual activations as fingerprint ---
        ax = fig.add_subplot(gs[3, col])
        act_res = acts["residual"]
        padded = np.zeros(side * side)
        padded[:len(act_res)] = act_res
        img = padded.reshape(side, side)
        ax.imshow(img, cmap="RdYlGn", vmin=-vmax_res, vmax=vmax_res,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        n_active = (act_res > 0).sum()
        ax.text(0.02, 0.98, f"{n_active}/{len(act_res)} active",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
        if col == 0:
            ax.set_ylabel(f"Residual\n({hidden}d)", fontsize=10)

        # --- Row 4: Activation distribution (histogram overlay) ---
        ax = fig.add_subplot(gs[4, col])
        ax.hist(act_proj, bins=50, alpha=0.5, color="#4CAF50", label="proj",
                density=True, edgecolor="none")
        ax.hist(act_res, bins=50, alpha=0.5, color="#FF9800", label="res",
                density=True, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlim(-vmax_res * 0.5, vmax_res * 0.5)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("Distribution", fontsize=10)

        # --- Row 5: Prediction ---
        ax = fig.add_subplot(gs[5, col])
        class_labels = [METER_LABELS[m] for m in CLASS_METERS[:len(probs)]]
        bar_colors = []
        pred_idx = np.argmax(probs)
        for i, m in enumerate(CLASS_METERS[:len(probs)]):
            if m == meter:
                bar_colors.append("#F44336")  # ground truth
            elif i == pred_idx:
                bar_colors.append("#FF9800")  # predicted (wrong)
            else:
                bar_colors.append("#90CAF9")
        ax.barh(class_labels, probs, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 1)

        correct = CLASS_METERS[pred_idx] == meter
        verdict = "+" if correct else "-"
        color = "green" if correct else "red"
        ax.set_xlabel(
            f"{verdict} pred={class_labels[pred_idx]} p={probs[pred_idx]:.2f}",
            fontsize=10, color=color, fontweight="bold",
        )
        if col == 0:
            ax.set_ylabel("Prediction", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.text(0.02, 0.005,
             "Autocorr: blue=neg, red=pos (z-scored) | "
             "Signals: yellow=low, red=high | "
             "Fingerprints: red=neg, green=pos (shared scale) | "
             "Prediction: red=truth, orange=pred",
             fontsize=7, color="gray")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize MeterNet activations")
    parser.add_argument("--meters", type=int, nargs="+", default=[3, 4, 5, 7, 9, 11],
                        help="Meter classes to show (default: 3 4 5 7 9 11)")
    parser.add_argument("--average", type=int, default=0,
                        help="Average N samples per class (0 = single sample)")
    parser.add_argument("--output", type=str, default="data/meter_net_activations.png")
    args = parser.parse_args()

    print("Loading model...")
    model, info = load_model()
    print(f"  hidden={info['hidden']}, blocks={info['n_blocks']}, "
          f"dropout={info['dropout_scale']}")

    n_per_class = max(args.average, 1)
    averaging = args.average > 0

    print(f"Collecting samples ({n_per_class} per class)...")
    samples = collect_samples(args.meters, n_per_class)
    for m in args.meters:
        paths = samples.get(m, [])
        print(f"  {METER_LABELS.get(m, str(m))}: {len(paths)} files")

    print("Extracting features...")
    cache = AnalysisCache()
    all_data = {}
    labels = {}

    for meter in args.meters:
        paths = samples.get(meter, [])
        if not paths:
            print(f"  {METER_LABELS.get(meter, str(meter))}: skipped (no files)")
            continue

        feats_z_list = []
        probs_list = []
        acts_list: list[dict[str, np.ndarray]] = []

        for i, path in enumerate(paths):
            try:
                feat = extract_features(path, cache)
            except Exception as e:
                print(f"  {path.name}: SKIP ({e})")
                continue

            if info["feat_mean"] is not None:
                feat_z = (feat - info["feat_mean"]) / np.where(
                    info["feat_std"] < 1e-8, 1.0, info["feat_std"]
                )
            else:
                feat_z = feat.copy()

            probs, acts = run_with_hooks(model, feat_z)
            feats_z_list.append(feat_z)
            probs_list.append(probs)
            acts_list.append(acts)

        if not feats_z_list:
            continue

        n_ok = len(feats_z_list)
        # Average features and activations
        avg_feat = np.mean(feats_z_list, axis=0)
        avg_probs = np.mean(probs_list, axis=0)
        avg_acts = {}
        for key in acts_list[0]:
            avg_acts[key] = np.mean([a[key] for a in acts_list], axis=0)

        all_data[meter] = (avg_feat, avg_probs, avg_acts)

        pred = CLASS_METERS[np.argmax(avg_probs)]
        ok = "+" if pred == meter else "-"
        if averaging:
            labels[meter] = f"{METER_LABELS[meter]}\n(avg {n_ok} files)"
        else:
            labels[meter] = f"{METER_LABELS[meter]}\n{paths[0].stem[:25]}"

        print(f"  {METER_LABELS[meter]} ({n_ok} files): pred={METER_LABELS.get(pred, pred)} {ok}")

    print("Plotting...")
    output = Path(args.output)
    plot(all_data, args.meters, labels, info, output)


if __name__ == "__main__":
    main()
