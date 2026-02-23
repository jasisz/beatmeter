"""Feature extraction for MeterNet — unified meter classification model.

MeterNet combines:
- Audio features (1449 dims) from onset_mlp_features.py v6
- MERT-v1-95M embedding (1536 dims) from layer 3

Total: 2985 dimensions.

Two extraction paths:
1. Cache-based: for training (reads from .cache/ or .npy files)
2. Live: for inference (runtime extraction via mert.py)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group constants
# ---------------------------------------------------------------------------

# v6 audio features
N_AUDIO_FEATURES = 1449

# MERT-v1-95M embedding (single layer, 1536 dims)
N_MERT_FEATURES = 1536
MERT_LAYER = 3  # best layer from prior experiments

TOTAL_FEATURES = N_AUDIO_FEATURES + N_MERT_FEATURES  # 2985

FEATURE_VERSION = "mn_v7_slim"

# Feature group offsets
FEATURE_GROUPS = {
    "audio": (0, N_AUDIO_FEATURES),
    "mert": (N_AUDIO_FEATURES, N_AUDIO_FEATURES + N_MERT_FEATURES),
}

# Canonical group ordering (determines concatenation order in feature vectors)
ALL_GROUP_NAMES = ["audio", "mert"]

# Per-group dimensions
GROUP_DIMS = {
    "audio": N_AUDIO_FEATURES,
    "mert": N_MERT_FEATURES,
}


def feature_groups(n_audio: int | None = None, n_mert: int = 0) -> dict[str, tuple[int, int]]:
    """Return feature group offsets for a given audio feature size.

    Args:
        n_audio: Audio feature dimensionality (None = default N_AUDIO_FEATURES).
        n_mert: MERT embedding dimensionality (0 = use default N_MERT_FEATURES).
    """
    a = n_audio if n_audio is not None else N_AUDIO_FEATURES
    m = n_mert if n_mert > 0 else N_MERT_FEATURES
    return {
        "audio": (0, a),
        "mert": (a, a + m),
    }
