"""Audio file loading utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import librosa
import numpy as np


def load_audio(
    file_path_or_buffer: Union[str, Path, BytesIO],
    sr: int = 22050,
) -> tuple[np.ndarray, int]:
    """Load an audio file or buffer and convert to mono.

    Parameters
    ----------
    file_path_or_buffer:
        Path to an audio file or a BytesIO buffer containing audio data.
    sr:
        Target sample rate. Defaults to 22050 Hz.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple of (audio_array, sample_rate).
    """
    audio, sample_rate = librosa.load(file_path_or_buffer, sr=sr, mono=True)
    return audio, sample_rate
