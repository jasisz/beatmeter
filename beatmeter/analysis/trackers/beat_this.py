"""Beat This! beat and downbeat tracking (ISMIR 2024, CPJKU)."""

import logging
import os
import tempfile
import threading

import numpy as np
import soundfile as sf

from beatmeter.analysis.models import Beat

logger = logging.getLogger(__name__)

_beat_this_model = None
_beat_this_lock = threading.Lock()


def _select_device() -> str:
    """Pick best available device: MPS > CUDA > CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_beat_this_model():
    """Get or create singleton Beat This! model."""
    global _beat_this_model
    if _beat_this_model is None:
        from beat_this.inference import File2Beats
        device = _select_device()
        _beat_this_model = File2Beats(device=device, dbn=False)
        logger.info(f"Beat This! using device: {device}")
    return _beat_this_model


def track_beats_beat_this(audio: np.ndarray, sr: int = 22050, tmp_path: str | None = None) -> list[Beat]:
    """Track beats using Beat This! (ISMIR 2024, CPJKU).

    Conv-Transformer architecture - state-of-the-art beat/downbeat tracker.
    Returns list of Beat objects with downbeat markers.
    """
    try:
        owns_tmp = tmp_path is None
        if owns_tmp:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                tmp_path = tmp.name

        try:
            with _beat_this_lock:
                file2beats = _get_beat_this_model()
                beat_times, downbeat_times = file2beats(tmp_path)
        finally:
            if owns_tmp:
                os.unlink(tmp_path)

        if beat_times is None or len(beat_times) == 0:
            logger.warning("Beat This! returned no results")
            return []

        # Build Beat objects; mark beats that coincide with downbeats
        downbeat_set = set()
        for dt in downbeat_times:
            downbeat_set.add(round(float(dt), 4))

        beats = []
        for bt in beat_times:
            t = float(bt)
            # Check if this beat is close to any downbeat (within 30ms)
            is_db = any(abs(t - dbt) < 0.03 for dbt in downbeat_times)
            beats.append(Beat(time=t, is_downbeat=is_db, strength=1.0))
        return beats

    except Exception as e:
        logger.warning(f"Beat This! failed: {e}")
        return []
