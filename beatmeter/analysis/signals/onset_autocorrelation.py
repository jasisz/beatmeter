"""Signal 3: Onset autocorrelation to find bar-level periodicity."""

import numpy as np


def signal_onset_autocorrelation(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    beat_interval: float | None = None,
    sr: int = 22050,
) -> dict[tuple[int, int], float]:
    """Signal 3: Onset autocorrelation to find bar-level periodicity.

    Uses the detected tempo to check autocorrelation at expected bar lags.
    """
    scores: dict[tuple[int, int], float] = {}

    if len(onset_times) < 10 or beat_interval is None or beat_interval <= 0:
        return scores

    duration = float(onset_times[-1])
    if duration < 3:
        return scores

    # Regular-sampled onset strength signal (50 Hz)
    analysis_sr = 50
    n_samples = int(duration * analysis_sr)
    if n_samples < 50:
        return scores

    signal = np.zeros(n_samples)
    for t, s in zip(onset_times, onset_strengths):
        idx = int(t * analysis_sr)
        if 0 <= idx < n_samples:
            signal[idx] = max(signal[idx], s)

    signal = signal - np.mean(signal)
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return scores

    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]

    for beats_per_bar in [2, 3, 4, 5, 6, 7, 9, 11, 12]:
        # Quarter note based
        bar_dur = beat_interval * beats_per_bar
        lag = int(bar_dur * analysis_sr)

        if 2 < lag < len(autocorr) - 2:
            window = max(1, int(lag * 0.05))
            start = max(0, lag - window)
            end = min(len(autocorr), lag + window + 1)
            peak = float(np.max(autocorr[start:end]))
            if peak > 0.02:
                scores[(beats_per_bar, 4)] = peak

        # Eighth note based
        if beats_per_bar in (3, 5, 6, 7, 9, 11, 12):
            bar_dur_8 = (beat_interval / 2) * beats_per_bar
            lag_8 = int(bar_dur_8 * analysis_sr)
            if 2 < lag_8 < len(autocorr) - 2:
                window = max(1, int(lag_8 * 0.05))
                start = max(0, lag_8 - window)
                end = min(len(autocorr), lag_8 + window + 1)
                peak = float(np.max(autocorr[start:end]))
                if peak > 0.02:
                    scores[(beats_per_bar, 8)] = peak

    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores
