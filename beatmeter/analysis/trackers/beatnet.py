"""BeatNet beat and downbeat tracking (CRNN + particle filtering)."""

import logging
import os
import tempfile
import threading

import numpy as np
import soundfile as sf

from beatmeter.analysis.models import Beat

logger = logging.getLogger(__name__)

_beatnet_model = None
_beatnet_lock = threading.Lock()


def _get_beatnet_model():
    """Get or create singleton BeatNet model."""
    global _beatnet_model
    if _beatnet_model is None:
        from BeatNet.BeatNet import BeatNet as BeatNetModel
        _beatnet_model = BeatNetModel(
            1,  # mode: offline
            inference_model="PF",  # particle filtering
            plot=[],  # no plots
            thread=False,
        )
    return _beatnet_model


def track_beats_beatnet(audio: np.ndarray, sr: int = 22050, tmp_path: str | None = None) -> list[Beat]:
    """Track beats using BeatNet (CRNN + particle filtering).

    BeatNet provides beats, downbeats, and inferred meter.
    Returns list of Beat objects with downbeat markers.
    """
    try:
        owns_tmp = tmp_path is None
        if owns_tmp:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                tmp_path = tmp.name

        try:
            with _beatnet_lock:
                estimator = _get_beatnet_model()
                output = estimator.process(tmp_path)
        finally:
            if owns_tmp:
                os.unlink(tmp_path)

        if output is None or len(output) == 0:
            logger.warning("BeatNet returned no results")
            return []

        # BeatNet output: Nx2 array [[time, beat_number], ...]
        # beat_number == 1 indicates downbeat
        beats = []
        for row in output:
            time_s = float(row[0])
            beat_num = int(row[1])
            beats.append(Beat(
                time=time_s,
                is_downbeat=(beat_num == 1),
                strength=1.0,
            ))
        return beats

    except Exception as e:
        logger.warning(f"BeatNet failed: {e}")
        return []
