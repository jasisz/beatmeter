"""madmom RNN-based beat and downbeat tracking."""

import logging
import os
import tempfile
import threading

import numpy as np
import soundfile as sf

from beatmeter.analysis.models import Beat

logger = logging.getLogger(__name__)

_madmom_rnn_processor = None
_madmom_lock = threading.Lock()


def _get_madmom_rnn_processor():
    """Get or create singleton madmom RNNDownBeatProcessor."""
    global _madmom_rnn_processor
    if _madmom_rnn_processor is None:
        from madmom.features.downbeats import RNNDownBeatProcessor
        _madmom_rnn_processor = RNNDownBeatProcessor()
    return _madmom_rnn_processor


# Activation cache also needs lock for thread-safe access
_madmom_activation_cache_lock = threading.Lock()

# Cache madmom RNN activations to avoid recomputing for each beats_per_bar.
# Keyed by id(audio) — valid only within a single analyze_file call.
_madmom_activation_cache: dict[int, np.ndarray] = {}


def clear_activation_cache() -> None:
    """Clear the madmom activation cache. Call after each file."""
    with _madmom_activation_cache_lock:
        _madmom_activation_cache.clear()


def _get_madmom_activations(audio: np.ndarray, sr: int, tmp_path: str | None = None) -> np.ndarray | None:
    """Compute madmom RNN activations (cached per audio hash)."""
    cache_key = id(audio)  # unique per array object; avoids cross-file contamination
    with _madmom_activation_cache_lock:
        if cache_key in _madmom_activation_cache:
            return _madmom_activation_cache[cache_key]

    owns_tmp = tmp_path is None
    if owns_tmp:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name

    try:
        with _madmom_lock:
            proc = _get_madmom_rnn_processor()
            activations = proc(tmp_path)
        with _madmom_activation_cache_lock:
            _madmom_activation_cache[cache_key] = activations
        return activations
    finally:
        if owns_tmp:
            os.unlink(tmp_path)


def track_beats_madmom(
    audio: np.ndarray,
    sr: int = 22050,
    beats_per_bar: int = 4,
    tmp_path: str | None = None,
) -> list[Beat]:
    """Track beats using madmom with specified beats_per_bar.

    Uses RNNDownBeatProcessor + DBNDownBeatTrackingProcessor.
    """
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor

        activations = _get_madmom_activations(audio, sr, tmp_path=tmp_path)
        if activations is None:
            return []

        dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=[beats_per_bar],
            fps=100,
        )
        result = dbn(activations)

        if result is None or len(result) == 0:
            return []

        # result: Nx2 array [[time, beat_position], ...]
        # beat_position == 1 indicates downbeat
        beats = []
        for row in result:
            time_s = float(row[0])
            beat_pos = int(row[1])
            beats.append(Beat(
                time=time_s,
                is_downbeat=(beat_pos == 1),
                strength=1.0,
            ))
        return beats

    except Exception as e:
        logger.warning(f"madmom (beats_per_bar={beats_per_bar}) failed: {e}")
        return []
