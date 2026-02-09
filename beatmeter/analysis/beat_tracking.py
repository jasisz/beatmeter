"""Beat and downbeat tracking using BeatNet (primary) and madmom (secondary)."""

import logging
import tempfile
import os
import threading

import numpy as np
import soundfile as sf

from beatmeter.analysis.models import Beat

logger = logging.getLogger(__name__)

# Singleton model instances (lazy-initialized, persist across calls)
# Protected by locks because models are not thread-safe.
_beatnet_model = None
_beatnet_lock = threading.Lock()
_beat_this_model = None
_beat_this_lock = threading.Lock()
_madmom_rnn_processor = None
_madmom_lock = threading.Lock()


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


def _get_beat_this_model():
    """Get or create singleton Beat This! model."""
    global _beat_this_model
    if _beat_this_model is None:
        from beat_this.inference import File2Beats
        _beat_this_model = File2Beats(device="cpu", dbn=False)
    return _beat_this_model


def _get_madmom_rnn_processor():
    """Get or create singleton madmom RNNDownBeatProcessor."""
    global _madmom_rnn_processor
    if _madmom_rnn_processor is None:
        from madmom.features.downbeats import RNNDownBeatProcessor
        _madmom_rnn_processor = RNNDownBeatProcessor()
    return _madmom_rnn_processor


# Activation cache also needs lock for thread-safe access
_madmom_activation_cache_lock = threading.Lock()


# Cache madmom RNN activations to avoid recomputing for each beats_per_bar
_madmom_activation_cache: dict[int, np.ndarray] = {}


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


def track_beats_librosa(audio: np.ndarray, sr: int = 22050) -> list[Beat]:
    """Fallback beat tracking using librosa.

    No downbeat detection - all beats marked as non-downbeat.
    """
    import librosa

    _tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    beats = []
    for t in beat_times:
        beats.append(Beat(time=float(t), is_downbeat=False, strength=1.0))
    return beats
