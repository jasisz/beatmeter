"""Audio preprocessing utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to the range [-1, 1].

    If the audio is silent (all zeros), it is returned unchanged.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak


def high_pass_filter(
    audio: np.ndarray,
    sr: int,
    cutoff: float = 60.0,
) -> np.ndarray:
    """Apply a Butterworth high-pass filter.

    Parameters
    ----------
    audio:
        Input audio signal.
    sr:
        Sample rate in Hz.
    cutoff:
        High-pass cutoff frequency in Hz. Defaults to 60 Hz.
    """
    sos = butter(N=4, Wn=cutoff, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio)


def preprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply the full preprocessing pipeline (normalize then high-pass filter)."""
    audio = normalize(audio)
    audio = high_pass_filter(audio, sr)
    return audio
