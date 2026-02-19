"""Signal 9: Onset MLP meter classification from multi-tempo autocorrelation features.

Provides an orthogonal meter signal that classifies time signature from onset,
RMS, spectral flux, and chroma autocorrelation features at multiple tempos,
combined with tempogram, MFCC, spectral contrast, and onset rate statistics.

v5: 1361-dim features (4 tempo candidates + beat-position histograms +
autocorrelation ratios + tempogram meter salience). Residual MLP architecture.
Hidden dim and dropout scale read from checkpoint (default: 640, 1.0).

The model is trained on WIKIMETER + METER2800 (6 classes: 3, 4, 5, 7, 9, 11)
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
    global _input_dim, _n_classes

    if _model is not None:
        return True

    with _lock:
        if _model is not None:
            return True
        return _load_model_locked()


def _load_model_locked() -> bool:
    """Inner model loading (called under lock)."""
    global _model, _device, _idx_to_meter, _feat_mean, _feat_std
    global _input_dim, _n_classes

    try:
        import torch
        import torch.nn as nn

        # Find checkpoint file (env var overrides defaults)
        import os
        custom_ckpt = os.environ.get("ONSET_MLP_CHECKPOINT")
        model_path: Path | None = None
        if custom_ckpt:
            model_path = Path(custom_ckpt) if Path(custom_ckpt).exists() else None
        else:
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

        hidden_dim = checkpoint.get("hidden_dim", 640)
        ds = checkpoint.get("dropout_scale", 1.0)
        model = _build_v5_model(input_dim, n_classes, hidden=hidden_dim, dropout_scale=ds)
        model.load_state_dict(checkpoint["model_state"])

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

        logger.info(
            "Loaded Onset MLP model from %s (hidden=%d, device=%s, dim=%d, classes=%d)",
            model_path, hidden_dim, device, input_dim, n_classes,
        )
        return True

    except Exception as e:
        logger.warning("Failed to load Onset MLP model: %s", e)
        return False


def _build_v5_model(input_dim: int, n_classes: int, hidden: int = 640, dropout_scale: float = 1.0):
    """Build v5 Residual MLP (must match OnsetMLPv5 in train_onset_mlp.py)."""
    import torch.nn as nn

    ds = dropout_scale
    h2 = max(int(hidden * 0.4), 64)   # 640→256, 320→128, 1024→409
    h4 = max(int(hidden * 0.2), 32)   # 640→128, 320→64,  1024→204

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
                nn.Linear(input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(min(0.3 * ds, 0.5)),
            )
            self.residual = ResidualBlock(hidden, dropout=min(0.25 * ds, 0.5))
            self.head = nn.Sequential(
                nn.Linear(hidden, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Dropout(min(0.2 * ds, 0.5)),
                nn.Linear(h2, h4),
                nn.BatchNorm1d(h4),
                nn.ReLU(),
                nn.Dropout(min(0.15 * ds, 0.5)),
                nn.Linear(h4, n_classes),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.residual(x)
            return self.head(x)

    return OnsetMLPv5(input_dim, n_classes)


# ---------------------------------------------------------------------------
# Feature extraction — delegates to shared module
# ---------------------------------------------------------------------------

def _extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extract features from audio array, version-aware.

    Uses v5 features for v5 models, v4 for v4 models.
    """
    from beatmeter.analysis.signals.onset_mlp_features import (
        SR, MAX_DURATION_S, extract_features_v5,
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

    return extract_features_v5(audio, sr)


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
        logger.debug("Onset MLP meter inference: %.1f ms", elapsed_ms)

        # Map class probabilities to meter hypotheses
        scores: dict[tuple[int, int], float] = {}

        # Clean 1:1 mapping: class → single canonical meter
        _CLASS_TO_METER = {3: (3, 4), 4: (4, 4), 5: (5, 4), 7: (7, 4), 9: (9, 8), 11: (11, 8)}

        for cls_idx, prob in enumerate(probs):
            beats_in_bar = _idx_to_meter.get(cls_idx)
            if beats_in_bar is None:
                continue
            meter = _CLASS_TO_METER.get(beats_in_bar)
            if meter is not None:
                scores[meter] = float(prob)

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
