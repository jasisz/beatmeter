#!/usr/bin/env python3
"""Visualize MeterNet activations and feature redundancy.

Modes:
  activations (default) — per-meter-class activation fingerprints
  redundancy            — correlation heatmap between feature sub-groups
  attention             — FT-Transformer attention patterns per meter class

Usage:
    uv run python scripts/visualize.py
    uv run python scripts/visualize.py --average 20
    uv run python scripts/visualize.py --mode redundancy --samples 200
    uv run python scripts/visualize.py --mode attention --checkpoint data/meter_net_grid/m_ftt_feat_both_h_256_d_1p5_b_2_kf_5.pt
    uv run python scripts/visualize.py --output my_plot.png
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

from beatmeter.analysis.cache import AnalysisCache, NumpyLMDB
from beatmeter.analysis.meter import _build_meter_net
from beatmeter.analysis.signals.meter_net_features import (
    TOTAL_FEATURES,
    N_AUDIO_FEATURES,
    N_MERT_FEATURES,
    feature_groups,
)
from beatmeter.analysis.signals.onset_mlp_features import (
    SR, MAX_DURATION_S, extract_features_v6,
    N_AUTOCORR_V5, N_TEMPOGRAM_BINS, N_MFCC, N_CONTRAST_DIMS,
    N_ONSET_STATS,
    N_BEAT_POSITION_FEATURES_V6, N_AUTOCORR_RATIO_FEATURES_V6,
    N_TEMPOGRAM_SALIENCE, BAR_BEAT_LENGTHS_V6,
    N_BEAT_POSITION_BINS,
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
        "input_dim": ckpt["input_dim"],
        "n_audio": ckpt.get("n_audio_features", N_AUDIO_FEATURES),
        "n_mert": ckpt.get("n_mert_features", 0),
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
# Feature extraction (audio + MERT only)
# ---------------------------------------------------------------------------

_MERT_LOOKUP: dict[str, Path] | None = None


def _build_mert_lookup() -> dict[str, Path]:
    """Build stem -> .npy path lookup from MERT embedding directories."""
    lookup: dict[str, Path] = {}
    for d in [
        PROJECT_ROOT / "data" / "mert_embeddings" / "meter2800",
        PROJECT_ROOT / "data" / "mert_embeddings" / "wikimeter",
    ]:
        if not d.exists():
            continue
        for npy in d.glob("*.npy"):
            lookup[npy.stem] = npy
    return lookup


def _load_mert_embedding(path: Path, layer: int = 3) -> np.ndarray:
    """Load pre-extracted MERT embedding for a file."""
    global _MERT_LOOKUP
    if _MERT_LOOKUP is None:
        _MERT_LOOKUP = _build_mert_lookup()

    npy_path = _MERT_LOOKUP.get(path.stem)
    if npy_path is None:
        return np.zeros(N_MERT_FEATURES, dtype=np.float32)
    try:
        emb = np.load(npy_path)  # shape (12, 1536)
        return emb[layer].astype(np.float32)
    except Exception:
        return np.zeros(N_MERT_FEATURES, dtype=np.float32)


def extract_features(path: Path) -> np.ndarray:
    """Extract full MeterNet feature vector (audio 1449d + MERT 1536d = 2985d)."""
    y, sr = librosa.load(str(path), sr=SR, duration=MAX_DURATION_S)

    audio_feat = extract_features_v6(y, sr)
    mert_feat = _load_mert_embedding(path)

    full = np.concatenate([audio_feat, mert_feat])
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
    """Generate the activation visualization."""
    active_meters = [m for m in meters if m in all_data]
    n = len(active_meters)
    if n == 0:
        print("No data to plot.")
        return
    hidden = model_info["hidden"]
    fg = feature_groups()

    # Compute grid size for fingerprint heatmaps
    side = int(np.ceil(np.sqrt(hidden)))

    fig = plt.figure(figsize=(4.5 * n, 26))
    title = (
        f"MeterNet v7-slim Activations — hidden={model_info['hidden']}, "
        f"blocks={model_info['n_blocks']}, "
        f"dropout={model_info['dropout_scale']}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(7, n, hspace=0.5, wspace=0.3,
                           height_ratios=[1.0, 0.7, 1.0, 1.2, 1.2, 0.7, 0.9])

    # Collect all activations for shared color scale
    all_proj = [all_data[m][2]["proj"] for m in active_meters]
    all_res = [all_data[m][2]["residual"] for m in active_meters]
    vmax_proj = max(np.abs(a).max() for a in all_proj)
    vmax_res = max(np.abs(a).max() for a in all_res)

    for col, meter in enumerate(active_meters):
        feat_z, probs, acts = all_data[meter]
        label = labels.get(meter, METER_LABELS.get(meter, str(meter)))

        # --- Row 0: Audio features — autocorrelation matrix (16 channels x 64 lags) ---
        ax = fig.add_subplot(gs[0, col])
        autocorr = feat_z[:1024].reshape(16, 64)
        ax.imshow(autocorr, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.set_xlabel("lag", fontsize=7)
        ax.set_yticks(range(0, 16, 4))
        ax.set_yticklabels([f"t{i}" for i in range(4)], fontsize=7)
        ax.set_title(label, fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Autocorrelation\n(16x64)", fontsize=10)

        # --- Row 1: Beat-position histograms (7 bar lengths x 32 bins) ---
        ax = fig.add_subplot(gs[1, col])
        bph_start = N_AUTOCORR_V5 + N_TEMPOGRAM_BINS + N_MFCC * 2 + N_CONTRAST_DIMS + N_ONSET_STATS
        bph_end = bph_start + N_BEAT_POSITION_FEATURES_V6
        bph_data = feat_z[bph_start:bph_end].reshape(len(BAR_BEAT_LENGTHS_V6), N_BEAT_POSITION_BINS)
        ax.imshow(bph_data, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.set_yticks(range(len(BAR_BEAT_LENGTHS_V6)))
        ax.set_yticklabels([f"{b}-beat" for b in BAR_BEAT_LENGTHS_V6], fontsize=7)
        ax.set_xlabel("beat position bin", fontsize=7)
        ax.set_xticks(range(0, N_BEAT_POSITION_BINS, 4))
        ax.set_xticklabels(range(0, N_BEAT_POSITION_BINS, 4), fontsize=6)
        if col == 0:
            ax.set_ylabel(f"Beat-pos hist\n(7x{N_BEAT_POSITION_BINS})", fontsize=10)

        # --- Row 2: MERT embedding heatmap (mean 24x32 + max 24x32 stacked) ---
        ax = fig.add_subplot(gs[2, col])
        mert_start = N_AUDIO_FEATURES
        mert_raw = feat_z[mert_start:mert_start + N_MERT_FEATURES]
        mert_mean = mert_raw[:_MERT_HALF].reshape(24, 32)
        mert_max = mert_raw[_MERT_HALF:].reshape(24, 32)
        mert_combined = np.vstack([mert_mean, mert_max])
        ax.imshow(mert_combined, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.axhline(23.5, color="black", linewidth=1.5, linestyle="--")
        ax.set_yticks([12, 36])
        ax.set_yticklabels(["mean", "max"], fontsize=7)
        ax.set_xlabel("hidden dim", fontsize=7)
        if col == 0:
            ax.set_ylabel("MERT L3\n(768+768d)", fontsize=10)

        # --- Row 3: Projection activations as fingerprint ---
        ax = fig.add_subplot(gs[3, col])
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

        # --- Row 4: Residual activations as fingerprint ---
        ax = fig.add_subplot(gs[4, col])
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

        # --- Row 5: Activation distribution (histogram overlay) ---
        ax = fig.add_subplot(gs[5, col])
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

        # --- Row 6: Prediction ---
        ax = fig.add_subplot(gs[6, col])
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
             "Fingerprints: red=neg, green=pos (shared scale) | "
             "Prediction: red=truth, orange=pred",
             fontsize=7, color="gray")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Sub-groups within the 2985-dim feature vector (for redundancy analysis)
# ---------------------------------------------------------------------------

_BPH_START = N_AUTOCORR_V5 + N_TEMPOGRAM_BINS + N_MFCC * 2 + N_CONTRAST_DIMS + N_ONSET_STATS

# Audio sub-groups (within first 1449 dims)
AUDIO_SUBGROUPS = {
    "autocorr\n(1024d)": (0, N_AUTOCORR_V5),
    "tempogram\n(64d)": (N_AUTOCORR_V5, N_AUTOCORR_V5 + N_TEMPOGRAM_BINS),
    "MFCC\n(26d)": (N_AUTOCORR_V5 + N_TEMPOGRAM_BINS,
                     N_AUTOCORR_V5 + N_TEMPOGRAM_BINS + N_MFCC * 2),
    "spectral\n(14d)": (N_AUTOCORR_V5 + N_TEMPOGRAM_BINS + N_MFCC * 2,
                         N_AUTOCORR_V5 + N_TEMPOGRAM_BINS + N_MFCC * 2 + N_CONTRAST_DIMS),
    "onset_stats\n(4d)": (N_AUTOCORR_V5 + N_TEMPOGRAM_BINS + N_MFCC * 2 + N_CONTRAST_DIMS,
                           _BPH_START),
    "beat_pos_hist\n(224d)": (_BPH_START, _BPH_START + N_BEAT_POSITION_FEATURES_V6),
    "autocorr_ratios\n(84d)": (N_AUDIO_FEATURES - N_TEMPOGRAM_SALIENCE - N_AUTOCORR_RATIO_FEATURES_V6,
                                N_AUDIO_FEATURES - N_TEMPOGRAM_SALIENCE),
    "tg_salience\n(9d)": (N_AUDIO_FEATURES - N_TEMPOGRAM_SALIENCE, N_AUDIO_FEATURES),
}

# MERT embedding (after audio features): 1536d = mean(768d) + max(768d)
_MERT_HALF = N_MERT_FEATURES // 2  # 768
_MERT_CHUNK = _MERT_HALF // 3      # 256
MERT_SUBGROUPS = {
    f"MERT_mean_A\n({_MERT_CHUNK}d)": (N_AUDIO_FEATURES, N_AUDIO_FEATURES + _MERT_CHUNK),
    f"MERT_mean_B\n({_MERT_CHUNK}d)": (N_AUDIO_FEATURES + _MERT_CHUNK, N_AUDIO_FEATURES + 2 * _MERT_CHUNK),
    f"MERT_mean_C\n({_MERT_CHUNK}d)": (N_AUDIO_FEATURES + 2 * _MERT_CHUNK, N_AUDIO_FEATURES + _MERT_HALF),
    f"MERT_max_A\n({_MERT_CHUNK}d)": (N_AUDIO_FEATURES + _MERT_HALF, N_AUDIO_FEATURES + _MERT_HALF + _MERT_CHUNK),
    f"MERT_max_B\n({_MERT_CHUNK}d)": (N_AUDIO_FEATURES + _MERT_HALF + _MERT_CHUNK, N_AUDIO_FEATURES + _MERT_HALF + 2 * _MERT_CHUNK),
    f"MERT_max_C\n({_MERT_CHUNK}d)": (N_AUDIO_FEATURES + _MERT_HALF + 2 * _MERT_CHUNK, N_AUDIO_FEATURES + N_MERT_FEATURES),
}

ALL_SUBGROUPS = {**AUDIO_SUBGROUPS, **MERT_SUBGROUPS}


def _collect_raw_features(
    n_samples: int, meters: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw (un-normalized) feature vectors and labels."""
    n_per = max(1, n_samples // len(meters))
    samples = collect_samples(meters, n_per)

    feats = []
    labels = []
    for meter in meters:
        for path in samples.get(meter, []):
            try:
                feat = extract_features(path)
                feats.append(feat)
                labels.append(meter)
            except Exception as e:
                print(f"  {path.name}: SKIP ({e})")
    return np.array(feats), np.array(labels)


def _group_mean(X: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean of feature columns in [start, end) for each sample."""
    return X[:, start:end].mean(axis=1)


def plot_redundancy(
    X: np.ndarray, labels: np.ndarray, output_path: Path,
):
    """Plot inter-group correlation heatmap + PCA effective rank."""
    from sklearn.decomposition import PCA

    names = list(ALL_SUBGROUPS.keys())
    n_groups = len(names)

    # 1. Mean-aggregate each group, compute correlation
    G = np.column_stack([_group_mean(X, *ALL_SUBGROUPS[n]) for n in names])
    corr = np.corrcoef(G.T)

    # 2. PCA effective rank per group (how many PCs explain 95% variance)
    eff_ranks = []
    dims = []
    for name in names:
        s, e = ALL_SUBGROUPS[name]
        d = e - s
        dims.append(d)
        sub = X[:, s:e]
        # Skip constant features
        std = sub.std(axis=0)
        sub = sub[:, std > 1e-8]
        if sub.shape[1] < 2:
            eff_ranks.append(sub.shape[1])
            continue
        pca = PCA().fit(sub)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        rank95 = int(np.searchsorted(cumvar, 0.95)) + 1
        eff_ranks.append(rank95)

    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(28, 10),
                              gridspec_kw={"width_ratios": [1.2, 0.4, 0.4]})
    fig.suptitle(f"MeterNet v7-slim Feature Redundancy ({X.shape[0]} samples, {TOTAL_FEATURES}d)",
                 fontsize=14, fontweight="bold")

    # Heatmap
    ax = axes[0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(names, fontsize=7)
    # Add correlation values
    for i in range(n_groups):
        for j in range(n_groups):
            v = corr[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color=color)
    ax.set_title("Mean-aggregated inter-group correlation", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Divider between audio and MERT groups
    n_audio = len(AUDIO_SUBGROUPS)
    ax.axhline(n_audio - 0.5, color="black", linewidth=2)
    ax.axvline(n_audio - 0.5, color="black", linewidth=2)

    # PCA effective rank bar chart
    ax = axes[1]
    colors = ["#4CAF50"] * len(AUDIO_SUBGROUPS) + ["#FF9800"] * len(MERT_SUBGROUPS)
    short_names = [n.split("\n")[0] for n in names]
    bars = ax.barh(range(n_groups), eff_ranks, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("PCA rank (95% var)")
    ax.set_title("Effective dimensionality", fontsize=11)
    ax.invert_yaxis()
    # Label bars with rank/total
    for i, (r, d) in enumerate(zip(eff_ranks, dims)):
        ax.text(r + 0.3, i, f"{r}/{d}", va="center", fontsize=7)

    # Compression ratio
    ax = axes[2]
    ratios = [d / max(r, 1) for r, d in zip(eff_ranks, dims)]
    ax.barh(range(n_groups), ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Compression ratio (dims / PCA rank)")
    ax.set_title("Redundancy ratio (higher = more redundant)", fontsize=11)
    ax.invert_yaxis()
    for i, r in enumerate(ratios):
        ax.text(r + 0.1, i, f"{r:.1f}x", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# FT-Transformer attention visualization
# ---------------------------------------------------------------------------

# Token names for uniform 12-token split of 2985d = 1449 audio + 1536 MERT
# First ~6 tokens are audio, last ~6 tokens are MERT (each ~249d)
UNIFORM_TOKEN_NAMES_12 = [
    "autocorr\nA", "autocorr\nB", "autocorr\nC", "autocorr\nD",
    "tg+mfcc\n+spec", "bph+\nratios",
    "MERT\n1", "MERT\n2", "MERT\n3", "MERT\n4", "MERT\n5", "MERT\n6",
]

# Semantic token names (8 audio + 6 MERT = 14 tokens)
SEMANTIC_TOKEN_NAMES = [
    "acorr\nt1", "acorr\nt2", "acorr\nt3", "acorr\nt4",
    "tg+mfcc\n+spec", "beat\nhist", "acorr\nratios", "tg\nsal",
    "MERT\n1", "MERT\n2", "MERT\n3", "MERT\n4", "MERT\n5", "MERT\n6",
]


def load_ftt_model(checkpoint_path: Path):
    """Load FTTransformer from checkpoint."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.training.train import FTTransformer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") != "ftt":
        print(f"Error: checkpoint is '{ckpt.get('model_type')}', expected 'ftt'")
        sys.exit(1)

    # Detect n_tokens and token sizes from state_dict
    state = ckpt["model_state"]
    n_tokens = sum(1 for k in state if k.startswith("tokenizers.") and k.endswith(".weight"))
    token_sizes = [state[f"tokenizers.{i}.weight"].shape[1] for i in range(n_tokens)]

    d_model = ckpt["hidden_dim"]
    n_heads = state["encoder.layers.0.self_attn.in_proj_weight"].shape[0] // (3 * d_model) * d_model
    # n_heads: in_proj_weight is (3*d_model, d_model), so n_heads = d_model / head_dim
    # Actually detect from shape: in_proj is (3*d_model, d_model) — n_heads encoded in the layer
    # Let's just try common values
    for nh in [1, 2, 4, 8, 16]:
        if d_model % nh == 0:
            n_heads = nh  # will be overwritten, take largest divisor <= 16
    # Better: count from the QKV projection
    n_heads = d_model // (d_model // max(1, min(d_model, 4)))  # heuristic
    # Simplest: try 4 (our default), fallback
    n_heads = 4

    n_layers = ckpt.get("n_blocks", 2)

    model = FTTransformer(
        input_dim=ckpt["input_dim"],
        n_classes=ckpt["n_classes"],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_tokens=n_tokens,
        dropout=0.0,  # inference
        token_sizes=token_sizes,
    )
    model.load_state_dict(state)
    model.eval()

    info = {
        "hidden": d_model,
        "n_blocks": n_layers,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "token_sizes": token_sizes,
        "dropout_scale": ckpt.get("dropout_scale", 1.0),
        "feat_mean": ckpt.get("feat_mean"),
        "feat_std": ckpt.get("feat_std"),
        "input_dim": ckpt["input_dim"],
        "n_audio": ckpt.get("n_audio_features", N_AUDIO_FEATURES),
        "n_mert": ckpt.get("n_mert_features", 0),
        "kfold_mean": ckpt.get("kfold_mean"),
        "kfold_std": ckpt.get("kfold_std"),
    }
    return model, info


def _get_token_names(n_tokens: int, token_sizes: list[int]) -> list[str]:
    """Get human-readable token names based on token count and sizes."""
    if n_tokens == len(SEMANTIC_TOKEN_NAMES) and token_sizes == [256, 256, 256, 256, 108, 224, 84, 9, 256, 256, 256, 256, 256, 256]:
        return SEMANTIC_TOKEN_NAMES
    if n_tokens == 12:
        return UNIFORM_TOKEN_NAMES_12
    # Generic fallback
    return [f"T{i}" for i in range(n_tokens)]


def run_ftt_with_attention(
    model, feat_z: np.ndarray, n_heads: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Run FTT and extract per-layer attention weights.

    Returns:
        probs: (n_classes,) sigmoid probabilities
        attn_weights: list of (n_heads, seq_len, seq_len) per layer
        cls_embedding: (d_model,) the CLS output vector
    """
    attn_weights = []

    def make_attn_hook(layer_idx):
        def hook(mod, args, kwargs, output):
            # nn.TransformerEncoderLayer doesn't directly expose attention.
            # We need to hook into the MultiheadAttention module instead.
            pass
        return hook

    # Hook into self_attn modules directly
    hooks = []
    for i, layer in enumerate(model.encoder.layers):
        def make_hook(idx):
            def hook(mod, args, kwargs, output):
                # MultiheadAttention forward returns (attn_output, attn_weights)
                # when need_weights=True
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    attn_weights.append(output[1].detach().cpu().numpy()[0])
            return hook
        hooks.append(layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True))

    # Temporarily enable attention weight output
    orig_flags = []
    for layer in model.encoder.layers:
        # PyTorch TransformerEncoderLayer calls self_attn with need_weights=False by default
        # We need to monkey-patch to get weights out
        orig_flags.append(getattr(layer, '_sa_need_weights', False))

    # Patch: override the forward to pass need_weights=True
    orig_forwards = []
    for layer in model.encoder.layers:
        orig_forward = layer.self_attn.forward
        orig_forwards.append(orig_forward)

        def patched_forward(orig_fn):
            def wrapper(*args, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False  # get per-head weights
                return orig_fn(*args, **kwargs)
            return wrapper
        layer.self_attn.forward = patched_forward(orig_forward)

    x = torch.tensor(feat_z, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Get CLS embedding (run forward again to capture intermediate)
    with torch.no_grad():
        B = 1
        tokens = []
        offset = 0
        for i, gs in enumerate(model.group_sizes):
            chunk = x[:, offset:offset + gs]
            tokens.append(model.tokenizers[i](chunk))
            offset += gs
        tokens = torch.stack(tokens, dim=1)
        cls = model.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = model.encoder(tokens)
        cls_embedding = encoded[:, 0, :].cpu().numpy()[0]

    # Restore original forwards
    for layer, orig_fw in zip(model.encoder.layers, orig_forwards):
        layer.self_attn.forward = orig_fw

    for h in hooks:
        h.remove()

    return probs, attn_weights, cls_embedding


def plot_attention(
    all_data: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]],
    meters: list[int],
    labels: dict[int, str],
    model_info: dict,
    output_path: Path,
):
    """Generate FTT attention visualization.

    all_data values: (feat_z, probs, attn_weights_per_layer, cls_embedding)
    """
    active_meters = [m for m in meters if m in all_data]
    n = len(active_meters)
    if n == 0:
        print("No data to plot.")
        return

    n_tokens = model_info["n_tokens"]
    n_heads = model_info["n_heads"]
    n_layers = model_info["n_blocks"]
    token_names = _get_token_names(n_tokens, model_info["token_sizes"])
    seq_names = ["CLS"] + token_names  # CLS + feature tokens

    # Rows:
    # 0: CLS attention per token (layer 0, averaged over heads) — bar chart
    # 1: CLS attention per token (layer 1, averaged over heads) — bar chart
    # 2: Full attention heatmap layer 0 (head average)
    # 3: Full attention heatmap layer 1 (head average)
    # 4: Per-head CLS attention (last layer) — stacked bars
    # 5: CLS embedding fingerprint
    # 6: Prediction bar chart
    n_rows = 2 + n_layers + 1 + 1 + 1  # = 7 for 2 layers
    height_ratios = [0.6] * n_layers + [1.2] * n_layers + [0.8, 1.0, 0.9]

    fig = plt.figure(figsize=(4.5 * n, 3.5 * n_rows))
    val_str = ""
    if model_info.get("kfold_mean"):
        val_str = f", val={model_info['kfold_mean']:.1%}±{model_info['kfold_std']:.1%}"
    title = (
        f"FT-Transformer Attention — d={model_info['hidden']}, "
        f"heads={n_heads}, layers={n_layers}, "
        f"tokens={n_tokens}{val_str}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(n_rows, n, hspace=0.55, wspace=0.35, height_ratios=height_ratios)

    # Color palette for heads
    head_colors = plt.cm.Set2(np.linspace(0, 1, n_heads))

    for col, meter in enumerate(active_meters):
        feat_z, probs, attn_layers, cls_emb = all_data[meter]
        label = labels.get(meter, METER_LABELS.get(meter, str(meter)))

        row = 0

        # --- Rows 0-1: CLS attention bar charts per layer ---
        for layer_idx in range(n_layers):
            ax = fig.add_subplot(gs[row, col])
            if layer_idx < len(attn_layers):
                # attn shape: (n_heads, seq_len, seq_len)
                attn = attn_layers[layer_idx]
                # CLS attention = row 0 (CLS queries all tokens)
                cls_attn = attn[:, 0, 1:]  # (n_heads, n_tokens) — skip CLS self-attn
                avg_cls_attn = cls_attn.mean(axis=0)  # (n_tokens,)

                # Color: audio tokens green, MERT tokens orange
                n_audio_tokens = sum(1 for s in model_info["token_sizes"]
                                      if s > 0 and sum(model_info["token_sizes"][:model_info["token_sizes"].index(s)+1]) <= model_info["n_audio"])
                # Simpler: count tokens in audio portion
                cumsum = np.cumsum(model_info["token_sizes"])
                n_audio_tokens = int(np.searchsorted(cumsum, model_info["n_audio"], side="right"))
                colors = ["#4CAF50"] * n_audio_tokens + ["#FF9800"] * (n_tokens - n_audio_tokens)

                bars = ax.bar(range(n_tokens), avg_cls_attn, color=colors,
                              edgecolor="black", linewidth=0.5, alpha=0.85)
                ax.set_xticks(range(n_tokens))
                ax.set_xticklabels(token_names, fontsize=6, rotation=0)
                ax.set_ylim(0, max(0.2, avg_cls_attn.max() * 1.3))
                ax.tick_params(labelsize=6)

                # Mark top-2 tokens
                top2 = np.argsort(avg_cls_attn)[-2:]
                for tidx in top2:
                    ax.text(tidx, avg_cls_attn[tidx] + 0.002,
                            f"{avg_cls_attn[tidx]:.3f}", ha="center", fontsize=5,
                            fontweight="bold")

            if col == 0:
                ax.set_ylabel(f"CLS attn\nL{layer_idx}", fontsize=10)
            if row == 0:
                ax.set_title(label, fontsize=13, fontweight="bold")
            row += 1

        # --- Rows 2-3: Full attention heatmaps ---
        for layer_idx in range(n_layers):
            ax = fig.add_subplot(gs[row, col])
            if layer_idx < len(attn_layers):
                attn = attn_layers[layer_idx]
                avg_attn = attn.mean(axis=0)  # (seq_len, seq_len)

                im = ax.imshow(avg_attn, cmap="YlOrRd", vmin=0,
                               vmax=max(0.3, avg_attn.max()),
                               aspect="equal", interpolation="nearest")
                ax.set_xticks(range(len(seq_names)))
                ax.set_xticklabels(seq_names, fontsize=5, rotation=45, ha="right")
                ax.set_yticks(range(len(seq_names)))
                ax.set_yticklabels(seq_names, fontsize=5)

                # Annotate values
                for i in range(avg_attn.shape[0]):
                    for j in range(avg_attn.shape[1]):
                        v = avg_attn[i, j]
                        if v > 0.05:
                            color = "white" if v > avg_attn.max() * 0.6 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=3.5, color=color)

            if col == 0:
                ax.set_ylabel(f"Full attn\nL{layer_idx}", fontsize=10)
            row += 1

        # --- Row 4: Per-head CLS attention (last layer) ---
        ax = fig.add_subplot(gs[row, col])
        if attn_layers:
            last_attn = attn_layers[-1]  # (n_heads, seq_len, seq_len)
            x_pos = np.arange(n_tokens)
            bar_width = 0.8 / n_heads
            for h in range(n_heads):
                head_cls = last_attn[h, 0, 1:]  # CLS → feature tokens
                offset_h = (h - n_heads / 2 + 0.5) * bar_width
                ax.bar(x_pos + offset_h, head_cls, bar_width,
                       color=head_colors[h], edgecolor="none",
                       alpha=0.8, label=f"H{h}")
            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(token_names, fontsize=6)
            ax.legend(fontsize=6, ncol=n_heads, loc="upper right",
                      framealpha=0.7, handlelength=0.8)
            ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel(f"Per-head\nCLS attn", fontsize=10)
        row += 1

        # --- Row 5: CLS embedding fingerprint ---
        ax = fig.add_subplot(gs[row, col])
        d_model = model_info["hidden"]
        side = int(np.ceil(np.sqrt(d_model)))
        padded = np.zeros(side * side)
        padded[:len(cls_emb)] = cls_emb
        img = padded.reshape(side, side)
        vmax = max(np.abs(cls_emb).max(), 0.01)
        ax.imshow(img, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        n_active = int((cls_emb > 0).sum())
        ax.text(0.02, 0.98, f"{n_active}/{d_model} active",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
        if col == 0:
            ax.set_ylabel(f"CLS embed\n({d_model}d)", fontsize=10)
        row += 1

        # --- Row 6: Prediction ---
        ax = fig.add_subplot(gs[row, col])
        class_labels = [METER_LABELS[m] for m in CLASS_METERS[:len(probs)]]
        bar_colors = []
        pred_idx = np.argmax(probs)
        for i, m in enumerate(CLASS_METERS[:len(probs)]):
            if m == meter:
                bar_colors.append("#F44336")
            elif i == pred_idx:
                bar_colors.append("#FF9800")
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
             "CLS attn: green=audio, orange=MERT | "
             "Heatmap: brighter=higher attention | "
             "Prediction: red=truth, orange=pred",
             fontsize=7, color="gray")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize MeterNet")
    parser.add_argument("--mode", choices=["activations", "redundancy", "attention"],
                        default="activations",
                        help="Visualization mode")
    parser.add_argument("--meters", type=int, nargs="+", default=[3, 4, 5, 7, 9, 11],
                        help="Meter classes (default: 3 4 5 7 9 11)")
    parser.add_argument("--average", type=int, default=0,
                        help="Average N samples per class (activations mode)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Total samples for redundancy mode")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Model checkpoint path (required for attention mode)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "attention":
        output = Path(args.output or "data/ftt_attention.png")
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Try to find a FTT checkpoint in grid dir
            grid_dir = PROJECT_ROOT / "data" / "meter_net_grid"
            ftt_ckpts = sorted(grid_dir.glob("*ftt*")) if grid_dir.exists() else []
            if ftt_ckpts:
                ckpt_path = ftt_ckpts[-1]
                print(f"Auto-detected FTT checkpoint: {ckpt_path}")
            else:
                print("Error: no --checkpoint provided and no FTT checkpoint found in data/meter_net_grid/")
                sys.exit(1)

        print(f"Loading FTT model from {ckpt_path}...")
        model, info = load_ftt_model(ckpt_path)
        print(f"  d_model={info['hidden']}, heads={info['n_heads']}, "
              f"layers={info['n_blocks']}, tokens={info['n_tokens']}")
        print(f"  token_sizes={info['token_sizes']}")

        n_per_class = max(args.average, 1)
        averaging = args.average > 0

        print(f"Collecting samples ({n_per_class} per class)...")
        samples = collect_samples(args.meters, n_per_class)
        for m in args.meters:
            paths = samples.get(m, [])
            print(f"  {METER_LABELS.get(m, str(m))}: {len(paths)} files")

        print("Extracting features & attention...")
        all_data = {}
        act_labels = {}

        for meter in args.meters:
            paths = samples.get(meter, [])
            if not paths:
                continue

            feats_z_list = []
            probs_list = []
            attn_list: list[list[np.ndarray]] = []
            cls_list = []

            for path in paths:
                try:
                    feat = extract_features(path)
                except Exception as e:
                    print(f"  {path.name}: SKIP ({e})")
                    continue

                model_dim = info["input_dim"]
                if info["feat_mean"] is not None:
                    model_feat = feat[:model_dim]
                    model_z = (model_feat - info["feat_mean"]) / np.where(
                        info["feat_std"] < 1e-8, 1.0, info["feat_std"]
                    )
                    feat_z = np.zeros_like(feat)
                    feat_z[:model_dim] = model_z
                else:
                    model_z = feat[:model_dim].copy()
                    feat_z = feat.copy()

                probs, attn_weights, cls_emb = run_ftt_with_attention(
                    model, model_z, info["n_heads"],
                )
                feats_z_list.append(feat_z)
                probs_list.append(probs)
                attn_list.append(attn_weights)
                cls_list.append(cls_emb)

            if not feats_z_list:
                continue

            n_ok = len(feats_z_list)
            avg_feat = np.mean(feats_z_list, axis=0)
            avg_probs = np.mean(probs_list, axis=0)
            avg_cls = np.mean(cls_list, axis=0)

            # Average attention weights per layer
            n_attn_layers = len(attn_list[0])
            avg_attn = []
            for li in range(n_attn_layers):
                layer_attns = [a[li] for a in attn_list]
                avg_attn.append(np.mean(layer_attns, axis=0))

            all_data[meter] = (avg_feat, avg_probs, avg_attn, avg_cls)

            pred = CLASS_METERS[np.argmax(avg_probs)]
            ok = "+" if pred == meter else "-"
            if averaging:
                act_labels[meter] = f"{METER_LABELS[meter]}\n(avg {n_ok} files)"
            else:
                act_labels[meter] = f"{METER_LABELS[meter]}\n{paths[0].stem[:25]}"
            print(f"  {METER_LABELS[meter]} ({n_ok} files): pred={METER_LABELS.get(pred, pred)} {ok}")

        print("Plotting attention...")
        plot_attention(all_data, args.meters, act_labels, info, output)
        return

    if args.mode == "redundancy":
        output = Path(args.output or "data/meter_net_redundancy.png")
        print(f"Collecting {args.samples} samples for redundancy analysis...")
        X, y = _collect_raw_features(args.samples, args.meters)
        print(f"  Collected {X.shape[0]} samples, {X.shape[1]} dims")
        print("Computing correlations and PCA...")
        plot_redundancy(X, y, output)
        return

    # --- Activations mode ---
    output = Path(args.output or "data/meter_net_activations.png")

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
    all_data = {}
    act_labels = {}

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
                feat = extract_features(path)
            except Exception as e:
                print(f"  {path.name}: SKIP ({e})")
                continue

            # Normalize only the dims the model knows about
            model_dim = info["input_dim"]
            if info["feat_mean"] is not None:
                model_feat = feat[:model_dim]
                model_z = (model_feat - info["feat_mean"]) / np.where(
                    info["feat_std"] < 1e-8, 1.0, info["feat_std"]
                )
                # For visualization: z-score the full vector (extra dims raw)
                feat_z = np.zeros_like(feat)
                feat_z[:model_dim] = model_z
                if len(feat) > model_dim:
                    feat_z[model_dim:] = feat[model_dim:]
            else:
                model_z = feat[:model_dim].copy()
                feat_z = feat.copy()

            probs, acts = run_with_hooks(model, model_z)
            feats_z_list.append(feat_z)
            probs_list.append(probs)
            acts_list.append(acts)

        if not feats_z_list:
            continue

        n_ok = len(feats_z_list)
        avg_feat = np.mean(feats_z_list, axis=0)
        avg_probs = np.mean(probs_list, axis=0)
        avg_acts = {}
        for key in acts_list[0]:
            avg_acts[key] = np.mean([a[key] for a in acts_list], axis=0)

        all_data[meter] = (avg_feat, avg_probs, avg_acts)

        pred = CLASS_METERS[np.argmax(avg_probs)]
        ok = "+" if pred == meter else "-"
        if averaging:
            act_labels[meter] = f"{METER_LABELS[meter]}\n(avg {n_ok} files)"
        else:
            act_labels[meter] = f"{METER_LABELS[meter]}\n{paths[0].stem[:25]}"

        print(f"  {METER_LABELS[meter]} ({n_ok} files): pred={METER_LABELS.get(pred, pred)} {ok}")

    print("Plotting...")
    plot(all_data, args.meters, act_labels, info, output)


if __name__ == "__main__":
    main()
