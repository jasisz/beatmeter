"""Onset detection using librosa."""

import numpy as np
import librosa

from beatmeter.analysis.models import OnsetStrength


def detect_onsets(audio: np.ndarray, sr: int = 22050) -> list[OnsetStrength]:
    """Detect onsets in audio using librosa.

    Returns list of OnsetStrength with time and strength.
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)

    # Peak picking for onset detection
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, onset_envelope=onset_env, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Get strength at each onset
    onsets = []
    max_strength = onset_env.max() if onset_env.max() > 0 else 1.0
    for t in onset_times:
        # Find nearest frame
        frame_idx = np.argmin(np.abs(times - t))
        strength = float(onset_env[frame_idx] / max_strength)
        onsets.append(OnsetStrength(time=float(t), strength=strength))

    return onsets


def onset_strength_envelope(audio: np.ndarray, sr: int = 22050) -> tuple[np.ndarray, np.ndarray]:
    """Return full onset strength envelope.

    Returns (times, strengths) arrays.
    """
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    # Normalize
    max_val = onset_env.max()
    if max_val > 0:
        onset_env = onset_env / max_val
    return times, onset_env
