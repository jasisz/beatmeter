"""Signal 7: madmom DBNBarTrackingProcessor meter inference."""

import logging
import os
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Quality gates
BAR_TRACKING_SPARSE_THRESHOLD = 0.15
NOISE_FLOOR_RATIO = 0.10

# Module-level singleton for RNNBarProcessor (loads 12 models, ~2-3s startup)
_bar_processor = None


def _get_bar_processor():
    global _bar_processor
    if _bar_processor is None:
        from madmom.features.downbeats import RNNBarProcessor
        _bar_processor = RNNBarProcessor()
    return _bar_processor


def signal_bar_tracking(
    audio: np.ndarray,
    sr: int,
    beat_times_array: np.ndarray,
    meters_to_test: list[int] | None = None,
    tmp_path: str | None = None,
) -> dict[tuple[int, int], float]:
    """Signal 7: madmom DBNBarTrackingProcessor meter inference.

    Uses beat-synchronous harmonic+percussive features via GRU-RNN
    to estimate downbeat probability at each beat, then Viterbi decoding
    per candidate meter to determine most likely beats-per-bar.
    """
    if meters_to_test is None:
        meters_to_test = [3, 4, 5, 7]

    scores: dict[tuple[int, int], float] = {}
    if len(beat_times_array) < 6:
        return scores

    # Quality gate: skip on sparse/synthetic audio where RNNBarProcessor
    # features (harmonic+percussive) are meaningless.
    rms = float(np.sqrt(np.mean(audio[:min(len(audio), sr * 5)] ** 2)))
    non_silent = float(np.mean(np.abs(audio) > rms * NOISE_FLOOR_RATIO))
    if non_silent < BAR_TRACKING_SPARSE_THRESHOLD:
        logger.debug(f"Bar tracking: sparse audio (non_silent={non_silent:.2f}), skipping")
        return scores

    owns_tmp = False
    try:
        from madmom.features.downbeats import DBNBarTrackingProcessor

        # RNNBarProcessor needs a file path — reuse shared temp WAV if available
        owns_tmp = tmp_path is None
        if owns_tmp:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, sr)

        bar_proc = _get_bar_processor()
        act = bar_proc((tmp_path, beat_times_array))

        # Remove last row (often NaN) and take downbeat activation column
        activations = act[:-1, 1] if act.shape[0] > 1 else act[:, 1]

        if len(activations) < 4:
            return scores

        # Run Viterbi for each meter and collect normalized log-probs
        log_probs = {}
        for bpb in meters_to_test:
            tracker = DBNBarTrackingProcessor(beats_per_bar=[bpb])
            _, log_prob = tracker.hmm.viterbi(activations)
            log_probs[bpb] = log_prob / len(activations)

        # Convert to probabilities via softmax
        values = np.array(list(log_probs.values()))
        exp_values = np.exp(values - np.max(values))
        probs = exp_values / np.sum(exp_values)

        for bpb, prob in zip(meters_to_test, probs):
            if prob > 0.05:
                scores[(bpb, 4)] = float(prob)

        # Normalize to [0, 1]
        if scores:
            max_s = max(scores.values())
            if max_s > 0:
                scores = {k: v / max_s for k, v in scores.items()}

        logger.debug(f"Bar tracking signal: {scores}")
    except Exception as e:
        logger.warning(f"Bar tracking signal failed: {e}")
    finally:
        if owns_tmp and tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return scores
