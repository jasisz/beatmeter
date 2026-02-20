#!/usr/bin/env python3
"""Visualize MeterNet activations and feature redundancy.

Modes:
  activations (default) — per-meter-class activation fingerprints
  redundancy            — correlation heatmap between feature sub-groups

Usage:
    uv run python scripts/visualize.py
    uv run python scripts/visualize.py --average 20
    uv run python scripts/visualize.py --mode redundancy --samples 200
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

from beatmeter.analysis.cache import AnalysisCache
from beatmeter.analysis.meter import _build_meter_net
from beatmeter.analysis.signals.meter_net_features import (
    extract_beat_features,
    extract_signal_scores,
    extract_tempo_features,
    TOTAL_FEATURES,
    N_AUDIO_FEATURES,
    N_SSM_FEATURES,
    N_BEAT_FEATURES,
    N_SIGNAL_FEATURES,
    N_TEMPO_FEATURES,
    SIGNAL_NAMES,
    METER_KEYS,
)
from beatmeter.analysis.signals.onset_mlp_features import (
    SR, MAX_DURATION_S, extract_features_v6,
    N_AUTOCORR_V5, N_TEMPOGRAM_BINS, N_MFCC, N_CONTRAST_DIMS,
    N_ONSET_STATS,
    N_BEAT_POSITION_FEATURES_V6, N_AUTOCORR_RATIO_FEATURES_V6,
    N_TEMPOGRAM_SALIENCE, BAR_BEAT_LENGTHS_V6,
    N_BEAT_POSITION_BINS,
)
from beatmeter.analysis.signals.meter_net_features import (
    feature_groups,
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
    """Extract full MeterNet feature vector (v6 audio + SSM)."""
    from beatmeter.analysis.signals.ssm_features import extract_ssm_features_cached

    y, sr = librosa.load(str(path), sr=SR, duration=MAX_DURATION_S)
    audio_hash = cache.audio_hash(str(path))

    audio_feat = extract_features_v6(y, sr)
    ssm_feat = extract_ssm_features_cached(y, sr, cache, audio_hash)

    beat_feat = extract_beat_features(cache, audio_hash)
    signal_feat = extract_signal_scores(cache, audio_hash)
    tempo_feat = extract_tempo_features(cache, audio_hash)

    full = np.concatenate([
        audio_feat, ssm_feat,
        beat_feat, signal_feat, tempo_feat,
    ])
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
    n_audio = model_info.get("n_audio", N_AUDIO_FEATURES)
    fg = feature_groups(n_audio)

    # Compute grid size for fingerprint heatmaps
    side = int(np.ceil(np.sqrt(hidden)))

    fig = plt.figure(figsize=(4.5 * n, 28))
    title = (
        f"MeterNet v6 Activations — hidden={model_info['hidden']}, "
        f"blocks={model_info['n_blocks']}, "
        f"dropout={model_info['dropout_scale']}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(8, n, hspace=0.5, wspace=0.3,
                           height_ratios=[1.0, 0.7, 0.4, 0.6, 1.2, 1.2, 0.7, 0.9])

    # Collect all activations for shared color scale
    all_proj = [all_data[m][2]["proj"] for m in active_meters]
    all_res = [all_data[m][2]["residual"] for m in active_meters]
    vmax_proj = max(np.abs(a).max() for a in all_proj)
    vmax_res = max(np.abs(a).max() for a in all_res)

    for col, meter in enumerate(active_meters):
        feat_z, probs, acts = all_data[meter]
        label = labels.get(meter, METER_LABELS.get(meter, str(meter)))

        # --- Row 0: Audio features top block ---
        ax = fig.add_subplot(gs[0, col])
        # v6: autocorrelation matrix (16 channels x 64 lags)
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

        # --- Row 2: Beat-sync chroma SSM (3 trackers × 11 lags) ---
        ax = fig.add_subplot(gs[2, col])
        ssm_start = fg["ssm"][0]
        ssm_end = fg["ssm"][1]
        # 3 trackers × 25 dims; show raw similarity profiles (first 11 per tracker)
        ssm_raw = feat_z[ssm_start:ssm_end]
        ssm_profiles = np.zeros((3, 11))
        for ti in range(3):
            ssm_profiles[ti] = ssm_raw[ti * 25: ti * 25 + 11]
        ax.imshow(ssm_profiles, aspect="auto", cmap="YlOrRd",
                  vmin=0, vmax=max(ssm_profiles.max(), 0.1), interpolation="nearest")
        ax.set_yticks(range(3))
        ax.set_yticklabels(["beatnet", "beat_this", "madmom"], fontsize=6)
        ax.set_xticks(range(11))
        ax.set_xticklabels([str(i) for i in range(2, 13)], fontsize=6)
        ax.set_xlabel("beat lag", fontsize=7)
        if col == 0:
            ax.set_ylabel("Chroma SSM\n(3×11)", fontsize=10)

        # --- Row 3: Signal scores heatmap (5 signals × 12 meters) ---
        ax = fig.add_subplot(gs[3, col])
        sig_start = fg["signal"][0]
        sig_end = fg["signal"][1]
        sig_data = feat_z[sig_start:sig_end].reshape(5, 12)
        sig_names = ["beatnet", "beat_this", "autocorr", "bar_trk", "hcdf"]
        m_names = ["2/4", "3/4", "4/4", "5/4", "5/8", "6/8",
                   "7/4", "7/8", "9/8", "10/8", "11/8", "12/8"]
        ax.imshow(sig_data, aspect="auto", cmap="YlOrRd",
                  vmin=0, vmax=max(sig_data.max(), 0.1), interpolation="nearest")
        ax.set_yticks(range(5))
        ax.set_yticklabels(sig_names, fontsize=6)
        ax.set_xticks(range(12))
        ax.set_xticklabels(m_names, fontsize=5, rotation=45, ha="right")
        if col == 0:
            ax.set_ylabel("Signal scores\n(5×12)", fontsize=10)

        # --- Row 4: Projection activations as fingerprint ---
        ax = fig.add_subplot(gs[4, col])
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

        # --- Row 5: Residual activations as fingerprint ---
        ax = fig.add_subplot(gs[5, col])
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

        # --- Row 6: Activation distribution (histogram overlay) ---
        ax = fig.add_subplot(gs[6, col])
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

        # --- Row 7: Prediction ---
        ax = fig.add_subplot(gs[7, col])
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

# ---------------------------------------------------------------------------
# Sub-groups within the 1555-dim feature vector (for redundancy analysis)
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

# New feature groups (between audio and beat)
_SSM_START = N_AUDIO_FEATURES
_BEAT_START = _SSM_START + N_SSM_FEATURES

# Non-audio groups
EXTRA_SUBGROUPS = {
    "ssm\n(75d)": (_SSM_START, _SSM_START + N_SSM_FEATURES),
    "beat_beatnet\n(12d)": (_BEAT_START, _BEAT_START + 12),
    "beat_beat_this\n(12d)": (_BEAT_START + 12, _BEAT_START + 24),
    "beat_madmom\n(12d)": (_BEAT_START + 24, _BEAT_START + 36),
    "beat_agree\n(6d)": (_BEAT_START + 36, _BEAT_START + 42),
    "sig_beatnet\n(12d)": (_BEAT_START + N_BEAT_FEATURES,
                            _BEAT_START + N_BEAT_FEATURES + 12),
    "sig_beat_this\n(12d)": (_BEAT_START + N_BEAT_FEATURES + 12,
                              _BEAT_START + N_BEAT_FEATURES + 24),
    "sig_autocorr\n(12d)": (_BEAT_START + N_BEAT_FEATURES + 24,
                              _BEAT_START + N_BEAT_FEATURES + 36),
    "sig_bar_trk\n(12d)": (_BEAT_START + N_BEAT_FEATURES + 36,
                             _BEAT_START + N_BEAT_FEATURES + 48),
    "sig_hcdf\n(12d)": (_BEAT_START + N_BEAT_FEATURES + 48,
                          _BEAT_START + N_BEAT_FEATURES + 60),
    "tempo\n(4d)": (_BEAT_START + N_BEAT_FEATURES + N_SIGNAL_FEATURES,
                     TOTAL_FEATURES),
}

ALL_SUBGROUPS = {**AUDIO_SUBGROUPS, **EXTRA_SUBGROUPS}


def _collect_raw_features(
    n_samples: int, meters: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw (un-normalized) feature vectors and labels."""
    n_per = max(1, n_samples // len(meters))
    samples = collect_samples(meters, n_per)
    cache = AnalysisCache()

    feats = []
    labels = []
    for meter in meters:
        for path in samples.get(meter, []):
            try:
                feat = extract_features(path, cache)
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
    fig.suptitle(f"MeterNet Feature Redundancy ({X.shape[0]} samples)", fontsize=14, fontweight="bold")

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

    # Divider between audio and non-audio groups
    n_audio = len(AUDIO_SUBGROUPS)
    ax.axhline(n_audio - 0.5, color="black", linewidth=2)
    ax.axvline(n_audio - 0.5, color="black", linewidth=2)

    # PCA effective rank bar chart
    ax = axes[1]
    colors = ["#4CAF50"] * len(AUDIO_SUBGROUPS) + ["#FF9800"] * len(EXTRA_SUBGROUPS)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize MeterNet")
    parser.add_argument("--mode", choices=["activations", "redundancy"], default="activations",
                        help="Visualization mode")
    parser.add_argument("--meters", type=int, nargs="+", default=[3, 4, 5, 7, 9, 11],
                        help="Meter classes (default: 3 4 5 7 9 11)")
    parser.add_argument("--average", type=int, default=0,
                        help="Average N samples per class (activations mode)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Total samples for redundancy mode")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

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
    cache = AnalysisCache()
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
                feat = extract_features(path, cache)
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
