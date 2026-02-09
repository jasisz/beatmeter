"""Signal 8: ResNet18 meter classification from MFCC spectrogram.

Provides an orthogonal meter signal that works directly on audio spectrograms,
bypassing beat tracking entirely. Useful when beat trackers have low alignment.

The model is trained on METER2800 (4 classes: meter 3, 4, 5, 7) and loaded
lazily on first use. If the model file is not found, the signal returns an
empty dict (graceful degradation — zero weight in ensemble).
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — populated lazily by _load_resnet_model()
# ---------------------------------------------------------------------------
_resnet_model = None
_resnet_device = None
_resnet_class_map: dict[int, int] | None = None
_resnet_mfcc_params: dict | None = None
_resnet_lock = threading.Lock()

# Search paths for the model checkpoint (checked in priority order)
MODEL_PATHS = [
    Path("data/meter_resnet18.pt"),  # local development
    Path.home() / ".beatmeter" / "meter_resnet18.pt",  # user install
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_resnet_model() -> bool:
    """Load the ResNet18 meter classifier from disk (singleton, thread-safe).

    Returns True if the model is ready, False if no checkpoint was found.
    Catches all exceptions so the pipeline never crashes.
    """
    global _resnet_model, _resnet_device, _resnet_class_map, _resnet_mfcc_params

    # Double-check locking: fast path without lock
    if _resnet_model is not None:
        return True

    with _resnet_lock:
        # Re-check after acquiring lock
        if _resnet_model is not None:
            return True

        return _load_resnet_model_locked()


def _load_resnet_model_locked() -> bool:
    """Inner model loading (called under lock)."""
    global _resnet_model, _resnet_device, _resnet_class_map, _resnet_mfcc_params

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        # Find checkpoint file
        model_path: Path | None = None
        for candidate in MODEL_PATHS:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            logger.debug(
                "ResNet18 meter model not found at any of: %s",
                [str(p) for p in MODEL_PATHS],
            )
            return False

        # Pick device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Reconstruct architecture
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Store in singletons
        _resnet_model = model
        _resnet_device = device
        _resnet_class_map = checkpoint.get("class_map", {0: 3, 1: 4, 2: 5, 3: 7})
        _resnet_mfcc_params = checkpoint.get(
            "mfcc_params",
            {"n_mfcc": 13, "n_mels": 128, "hop_length": 512},
        )

        logger.info("Loaded ResNet18 meter model from %s (device=%s)", model_path, device)
        return True

    except Exception as e:
        logger.warning("Failed to load ResNet18 meter model: %s", e)
        return False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extract MFCC spectrogram, normalise and resize to (1, 224, 224).

    Returns None if the audio is too short (< 1 s) or extraction fails.
    """
    try:
        import librosa
        import torch
        import torch.nn.functional as F

        duration = len(audio) / sr
        if duration < 1.0:
            logger.debug("ResNet MFCC: audio too short (%.2f s)", duration)
            return None

        # Take center 30 seconds (or full clip if shorter)
        max_samples = sr * 30
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            audio = audio[start : start + max_samples]

        # MFCC params (from checkpoint or defaults)
        params = _resnet_mfcc_params or {"n_mfcc": 13, "n_mels": 128, "hop_length": 512}
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=params["n_mfcc"],
            n_mels=params["n_mels"],
            hop_length=params["hop_length"],
        )  # shape: (n_mfcc, time_frames)

        # Normalise to zero mean, unit variance
        mean = np.mean(mfcc)
        std = np.std(mfcc)
        if std < 1e-6:
            logger.debug("ResNet MFCC: constant signal, skipping")
            return None
        mfcc = (mfcc - mean) / std

        # Resize to 224x224 via bilinear interpolation
        tensor = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        return tensor.squeeze(0).numpy()  # (1, 224, 224)

    except Exception as e:
        logger.warning("ResNet MFCC extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def signal_resnet_meter(audio: np.ndarray, sr: int) -> dict[tuple[int, int], float]:
    """Signal 8: ResNet18 meter classification from MFCC spectrogram.

    Returns a dict mapping ``(numerator, denominator)`` tuples to scores in
    [0, 1].  The highest-scoring hypothesis is normalised to 1.0.  Returns
    an empty dict when the model is unavailable or inference fails.
    """
    try:
        import torch

        # Lazy model init
        if not _load_resnet_model():
            return {}

        # Feature extraction
        mfcc = _extract_mfcc(audio, sr)
        if mfcc is None:
            return {}

        # Inference
        t0 = time.perf_counter()

        input_tensor = torch.from_numpy(mfcc).float().unsqueeze(0)  # (1, 1, 224, 224)
        input_tensor = input_tensor.to(_resnet_device)

        with torch.no_grad():
            logits = _resnet_model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("ResNet meter inference: %.1f ms", elapsed_ms)

        # Map class probabilities to meter hypotheses
        class_map = _resnet_class_map or {0: 3, 1: 4, 2: 5, 3: 7}
        scores: dict[tuple[int, int], float] = {}

        for cls_idx, prob in enumerate(probs):
            beats_in_bar = class_map.get(cls_idx)
            if beats_in_bar is None:
                continue
            prob = float(prob)

            if beats_in_bar == 3:
                # class 3 → (3,4) primary + (6,8) secondary
                scores[(3, 4)] = scores.get((3, 4), 0.0) + prob
                scores[(6, 8)] = scores.get((6, 8), 0.0) + prob * 0.4
            elif beats_in_bar == 4:
                # class 4 → (4,4) primary + (2,4) + (12,8) secondary
                scores[(4, 4)] = scores.get((4, 4), 0.0) + prob
                scores[(2, 4)] = scores.get((2, 4), 0.0) + prob * 0.3
                scores[(12, 8)] = scores.get((12, 8), 0.0) + prob * 0.2
            elif beats_in_bar == 5:
                scores[(5, 4)] = scores.get((5, 4), 0.0) + prob
            elif beats_in_bar == 7:
                scores[(7, 4)] = scores.get((7, 4), 0.0) + prob

        # Normalise so highest score = 1.0
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        logger.debug("ResNet meter signal: %s", scores)
        return scores

    except Exception as e:
        logger.warning("ResNet meter signal failed: %s", e)
        return {}
