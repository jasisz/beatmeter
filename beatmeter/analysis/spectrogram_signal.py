"""Signal 8: Multi-band beat-energy accent pattern analysis.

Computes mel-spectrogram energy at each beat in multiple frequency bands,
then scores meter hypotheses by how well the per-band accent patterns
show periodic structure.

This is complementary to Signal 4 (RMS accent) and Signal 5 (RMS
periodicity) because it captures *timbral* differences between beats.
E.g., in 4/4 rock the low band (kick) and mid band (snare) have
different accent patterns; in 3/4 waltz the bass note and chord
accompaniment create distinct band-wise patterns.

Inspired by "Music time signature detection using ResNet18" (EURASIP
2024) which showed MFCC/mel features carry strong meter information.
"""

import logging

import numpy as np

from beatmeter.analysis.models import Beat

logger = logging.getLogger(__name__)


def signal_spectrogram_classifier(
    beats: list[Beat],
    audio: np.ndarray,
    sr: int,
) -> dict[tuple[int, int], float]:
    """Multi-band beat-energy accent pattern analysis for meter detection.

    Args:
        beats: List of detected beat positions.
        audio: Raw audio signal (mono, float).
        sr: Sample rate.

    Returns:
        Dict mapping (numerator, 4) -> score in [0, 1].
    """
    import librosa

    scores: dict[tuple[int, int], float] = {}

    if len(beats) < 8 or len(audio) < sr * 2:
        return scores

    # Compute mel spectrogram in a few broad bands
    hop_length = 512
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=40, hop_length=hop_length, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Split into 4 frequency bands
    band_slices = [
        slice(0, 10),    # sub-bass + bass (~0-500 Hz)
        slice(10, 20),   # low-mid (~500-2000 Hz)
        slice(20, 30),   # high-mid (~2000-4000 Hz)
        slice(30, 40),   # high (~4000-8000 Hz)
    ]

    frames_per_sec = sr / hop_length
    beat_times = np.array([b.time for b in beats])
    n_beats = len(beat_times)
    n_bands = len(band_slices)
    band_energies = np.zeros((n_bands, n_beats))

    for bi, bslice in enumerate(band_slices):
        band = mel_db[bslice, :]
        for i, bt in enumerate(beat_times):
            frame = int(bt * frames_per_sec)
            start = max(0, frame - 2)
            end = min(band.shape[1], frame + 3)
            if end > start:
                band_energies[bi, i] = float(np.mean(band[:, start:end]))

    # For each candidate meter, compute accent-pattern contrast across bands
    for beats_per_bar in [2, 3, 4, 5, 6, 7]:
        n_complete_bars = n_beats // beats_per_bar
        if n_complete_bars < 3:
            continue

        band_scores = []
        for bi in range(n_bands):
            be = band_energies[bi, :n_complete_bars * beats_per_bar]
            if np.std(be) < 1e-6:
                continue

            bars = be.reshape(n_complete_bars, beats_per_bar)
            avg_pattern = np.mean(bars, axis=0)
            pattern_mean = np.mean(avg_pattern)
            if pattern_mean == 0:
                continue

            # Pattern contrast: CV of average accent pattern
            pattern_cv = float(np.std(avg_pattern) / abs(pattern_mean))

            # Consistency: correlate each bar with the average pattern
            if np.std(avg_pattern) > 1e-6:
                corrs = []
                for bar_i in range(n_complete_bars):
                    bar = bars[bar_i]
                    if np.std(bar) > 1e-6:
                        corr = float(np.corrcoef(avg_pattern, bar)[0, 1])
                        corrs.append(corr)
                consistency = float(np.mean(corrs)) if corrs else 0.0
            else:
                consistency = 0.0

            if consistency > 0 and pattern_cv > 0.01:
                band_scores.append(pattern_cv * max(0.0, consistency))

        if band_scores:
            band_scores.sort(reverse=True)
            top_k = min(2, len(band_scores))
            score = float(np.mean(band_scores[:top_k]))
            if score > 0.01:
                scores[(beats_per_bar, 4)] = score

    # Normalize
    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    logger.debug(f"Spectrogram classifier signal: {scores}")
    return scores
