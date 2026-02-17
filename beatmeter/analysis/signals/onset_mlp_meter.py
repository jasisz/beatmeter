"""Signal 9: Onset MLP meter classification from multi-tempo autocorrelation features.

Provides an orthogonal meter signal that classifies time signature from onset,
RMS, spectral flux, and chroma autocorrelation features at multiple tempos,
combined with tempogram, MFCC, spectral contrast, and onset rate statistics.

v5: 1361-dim features (4 tempo candidates + beat-position histograms +
autocorrelation ratios + tempogram meter salience). Residual MLP architecture.
Backward compatible: loads v4 (876-dim, Sequential) or v5 (1361-dim, Residual).

The model is trained on METER2800 + WIKIMETER (6 classes: 3, 4, 5, 7, 9, 11)
and loaded lazily on first use. If the model file is not found, the signal
returns an empty dict (graceful degradation — zero weight in ensemble).
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — populated lazily by _load_model()
# ---------------------------------------------------------------------------
_model = None
_device = None
_idx_to_meter: dict[int, int] | None = None
_feat_mean: np.ndarray | None = None
_feat_std: np.ndarray | None = None
_input_dim: int | None = None
_n_classes: int | None = None
_arch_version: str | None = None  # "v4" or "v5"
_lock = threading.Lock()

# Search paths for the model checkpoint (checked in priority order)
MODEL_PATHS = [
    Path("data/meter_onset_mlp.pt"),  # local development
    Path.home() / ".beatmeter" / "meter_onset_mlp.pt",  # user install
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model() -> bool:
    """Load the Onset MLP meter classifier from disk (singleton, thread-safe).

    Returns True if the model is ready, False if no checkpoint was found.
    """
    global _model, _device, _idx_to_meter, _feat_mean, _feat_std
    global _input_dim, _n_classes, _arch_version

    if _model is not None:
        return True

    with _lock:
        if _model is not None:
            return True
        return _load_model_locked()


def _load_model_locked() -> bool:
    """Inner model loading (called under lock)."""
    global _model, _device, _idx_to_meter, _feat_mean, _feat_std
    global _input_dim, _n_classes, _arch_version

    try:
        import torch
        import torch.nn as nn

        # Find checkpoint file
        model_path: Path | None = None
        for candidate in MODEL_PATHS:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            logger.debug(
                "Onset MLP model not found at any of: %s",
                [str(p) for p in MODEL_PATHS],
            )
            return False

        # Pick device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        input_dim = checkpoint["input_dim"]
        n_classes = checkpoint["n_classes"]
        meter_to_idx = checkpoint["meter_to_idx"]
        feat_mean = checkpoint.get("feat_mean")
        feat_std = checkpoint.get("feat_std")
        arch_version = checkpoint.get("arch_version", "v4")

        if arch_version == "v5":
            model = _build_v5_model(input_dim, n_classes)
            model.load_state_dict(checkpoint["model_state"])
        else:
            # v4 fallback: nn.Sequential(512→256→128)
            model = _build_v4_model(input_dim, n_classes)
            # Strip "net." prefix from state dict keys
            state_dict = checkpoint["model_state"]
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith("net."):
                    stripped[k[4:]] = v
                else:
                    stripped[k] = v
            model.load_state_dict(stripped)

        model.to(device)
        model.eval()

        # Store singletons
        _model = model
        _device = device
        _idx_to_meter = {v: k for k, v in meter_to_idx.items()}
        _feat_mean = feat_mean
        _feat_std = feat_std
        _input_dim = input_dim
        _n_classes = n_classes
        _arch_version = arch_version

        logger.info(
            "Loaded Onset MLP model from %s (arch=%s, device=%s, dim=%d, classes=%d)",
            model_path, arch_version, device, input_dim, n_classes,
        )
        return True

    except Exception as e:
        logger.warning("Failed to load Onset MLP model: %s", e)
        return False


def _build_v5_model(input_dim: int, n_classes: int):
    """Build v5 Residual MLP (must match OnsetMLPv5 in train_onset_mlp.py)."""
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float = 0.25):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.block(x)

    class OnsetMLPv5(nn.Module):
        def __init__(self, input_dim: int, n_classes: int):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, 640),
                nn.BatchNorm1d(640),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.residual = ResidualBlock(640, dropout=0.25)
            self.head = nn.Sequential(
                nn.Linear(640, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.residual(x)
            return self.head(x)

    return OnsetMLPv5(input_dim, n_classes)


def _build_v4_model(input_dim: int, n_classes: int):
    """Build v4 Sequential MLP (backward compatibility)."""
    import torch.nn as nn

    hidden = 512
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden // 2),
        nn.BatchNorm1d(hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(hidden // 2, hidden // 4),
        nn.BatchNorm1d(hidden // 4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden // 4, n_classes),
    )


# ---------------------------------------------------------------------------
# Feature extraction — delegates to shared module
# ---------------------------------------------------------------------------

def _extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extract features from audio array, version-aware.

    Uses v5 features for v5 models, v4 for v4 models.
    """
    from beatmeter.analysis.signals.onset_mlp_features import (
        SR, MAX_DURATION_S, extract_features_v4, extract_features_v5,
    )
    import librosa

    if len(audio) < sr * 2:
        return None

    # Trim to MAX_DURATION_S from center
    max_samples = sr * MAX_DURATION_S
    if len(audio) > max_samples:
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]

    # Resample if needed
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR

    if _arch_version == "v5":
        return extract_features_v5(audio, sr)
    return extract_features_v4(audio, sr)


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def signal_onset_mlp_meter(audio: np.ndarray, sr: int) -> dict[tuple[int, int], float]:
    """Signal 9: Onset MLP meter classification.

    Returns a dict mapping (numerator, denominator) tuples to scores in
    [0, 1]. The highest-scoring hypothesis is normalised to 1.0. Returns
    an empty dict when the model is unavailable or inference fails.
    """
    try:
        import torch

        if not _load_model():
            return {}

        # Feature extraction
        t0 = time.perf_counter()
        feat = _extract_features_from_audio(audio, sr)
        if feat is None:
            return {}

        # Standardize using training stats
        feat = feat.astype(np.float32)
        if _feat_mean is not None and _feat_std is not None:
            feat = (feat - _feat_mean) / np.where(_feat_std < 1e-8, 1.0, _feat_std)

        # Inference
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(_device)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Onset MLP meter inference: %.1f ms (arch=%s)", elapsed_ms, _arch_version)

        # Map class probabilities to meter hypotheses
        scores: dict[tuple[int, int], float] = {}

        for cls_idx, prob in enumerate(probs):
            beats_in_bar = _idx_to_meter.get(cls_idx)
            if beats_in_bar is None:
                continue
            prob = float(prob)

            if beats_in_bar == 3:
                scores[(3, 4)] = scores.get((3, 4), 0.0) + prob
                scores[(6, 8)] = scores.get((6, 8), 0.0) + prob * 0.4
            elif beats_in_bar == 4:
                scores[(4, 4)] = scores.get((4, 4), 0.0) + prob
                scores[(2, 4)] = scores.get((2, 4), 0.0) + prob * 0.3
                scores[(12, 8)] = scores.get((12, 8), 0.0) + prob * 0.2
            elif beats_in_bar == 5:
                scores[(5, 4)] = scores.get((5, 4), 0.0) + prob
                scores[(5, 8)] = scores.get((5, 8), 0.0) + prob * 0.3
            elif beats_in_bar == 7:
                scores[(7, 4)] = scores.get((7, 4), 0.0) + prob
                scores[(7, 8)] = scores.get((7, 8), 0.0) + prob * 0.3
            elif beats_in_bar == 9:
                scores[(9, 8)] = scores.get((9, 8), 0.0) + prob
            elif beats_in_bar == 11:
                scores[(11, 8)] = scores.get((11, 8), 0.0) + prob

        # Normalise so highest score = 1.0
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        logger.debug("Onset MLP meter signal: %s", scores)
        return scores

    except Exception as e:
        logger.warning("Onset MLP meter signal failed: %s", e)
        return {}
