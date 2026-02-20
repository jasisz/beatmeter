"""HCDF (Harmonic Change Detection Function) meter signal.

Pure DSP signal — no learned parameters, no overfitting risk.
Measures the rate of harmonic change in chromagram features.
Triple meters show harmonic changes every 3 beats, duple every 2 or 4.

Included as one of the input signals for MeterNet.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def signal_hcdf_meter(
    audio: np.ndarray,
    sr: int,
    beat_interval: float | None = None,
) -> dict[tuple[int, int], float]:
    """Compute HCDF-based meter scores.

    Extracts chromagram, computes cosine-distance HCDF, then autocorrelates
    the HCDF curve at candidate bar-level lags (beat_interval * N beats).

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    beat_interval : float or None
        Estimated beat interval in seconds. If None, uses tempo estimation.

    Returns
    -------
    dict mapping (numerator, denominator) -> score in [0, 1]
    """
    try:
        import librosa

        duration = len(audio) / sr
        if duration < 4.0:
            return {}

        # Chromagram (CQT-based, more robust than STFT for harmony)
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
        # shape: (12, n_frames)

        if chroma.shape[1] < 10:
            return {}

        # HCDF: cosine distance between consecutive chroma frames
        # Normalize chroma columns to unit vectors
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms < 1e-8] = 1.0
        chroma_norm = chroma / norms

        # Cosine distance = 1 - dot product of consecutive frames
        dots = np.sum(chroma_norm[:, :-1] * chroma_norm[:, 1:], axis=0)
        hcdf = 1.0 - np.clip(dots, -1.0, 1.0)
        # hcdf shape: (n_frames - 1,)

        if len(hcdf) < 20:
            return {}

        # Estimate beat interval if not provided
        if beat_interval is None or beat_interval <= 0:
            tempo = librosa.beat.tempo(y=audio, sr=sr)
            if len(tempo) > 0 and tempo[0] > 0:
                beat_interval = 60.0 / tempo[0]
            else:
                return {}

        # Frame rate
        frame_rate = sr / hop_length  # frames per second

        # Autocorrelation of HCDF at candidate bar-level lags
        hcdf_centered = hcdf - np.mean(hcdf)
        hcdf_norm = np.sqrt(np.sum(hcdf_centered**2))
        if hcdf_norm < 1e-8:
            return {}
        hcdf_centered = hcdf_centered / hcdf_norm

        candidates = [
            ((2, 4), 2),
            ((3, 4), 3),
            ((4, 4), 4),
            ((5, 4), 5),
            ((5, 8), 5),
            ((6, 8), 6),
            ((7, 4), 7),
            ((7, 8), 7),
            ((9, 8), 9),
            ((11, 8), 11),
            ((12, 8), 12),
        ]

        scores: dict[tuple[int, int], float] = {}

        for meter, n_beats in candidates:
            # For /8 meters, beat_interval represents the eighth note
            if meter[1] == 8:
                bar_duration = beat_interval * n_beats * 0.5
            else:
                bar_duration = beat_interval * n_beats

            lag_frames = int(round(bar_duration * frame_rate))

            if lag_frames < 2 or lag_frames >= len(hcdf_centered):
                continue

            # Autocorrelation at this lag
            n = len(hcdf_centered) - lag_frames
            if n < 10:
                continue
            acorr = np.sum(hcdf_centered[:n] * hcdf_centered[lag_frames:lag_frames + n])
            # Also check half-lag (should be lower for odd meters, similar for even)
            scores[meter] = max(float(acorr), 0.0)

        # Normalize to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    except Exception as e:
        logger.warning("HCDF meter signal failed: %s", e)
        return {}
