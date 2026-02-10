"""Signal 5: Beat-level accent autocorrelation (periodicity)."""

import numpy as np


def signal_beat_strength_periodicity(
    beat_energies: np.ndarray,
    normalize: bool = True,
) -> dict[tuple[int, int], float]:
    """Signal 5: Beat-level accent autocorrelation.

    Given energy (RMS) at each beat position, autocorrelates to find
    how many beats form one bar (accent period).

    For 3/4 with accented downbeats, the energy sequence is
    [H, L, L, H, L, L, ...] and the autocorrelation peaks at lag=3.
    """
    scores: dict[tuple[int, int], float] = {}

    if len(beat_energies) < 8:
        return scores

    be = beat_energies.copy()
    be -= np.mean(be)
    norm = np.sum(be ** 2)
    if norm < 1e-10:
        return scores

    # Autocorrelation at beat-count lags
    n = len(be)
    autocorr = np.correlate(be, be, mode='full')
    autocorr = autocorr[n - 1:]  # positive lags only
    autocorr = autocorr / autocorr[0]

    raw_peaks: dict[int, float] = {}
    for beats_per_bar in [2, 3, 4, 5, 6, 7, 9, 11, 12]:
        lag = beats_per_bar
        if lag >= n:
            continue
        peak = float(autocorr[lag])
        if peak > 0.02:
            raw_peaks[beats_per_bar] = peak

    if not raw_peaks:
        return scores

    # Handle the "multiples problem": if both lag=N and lag=2N have peaks,
    # the shorter period N is the fundamental bar length. Boost N, penalize 2N.
    fundamentals_found = set()
    for bpb in sorted(raw_peaks.keys()):
        if bpb in fundamentals_found:
            continue
        # Check if this is a multiple of an already-found fundamental
        is_multiple = False
        for fund in fundamentals_found:
            if bpb % fund == 0:
                is_multiple = True
                break
        if not is_multiple and raw_peaks[bpb] > 0.05:
            # This is a potential fundamental period
            fundamentals_found.add(bpb)
            raw_peaks[bpb] *= 1.4
            # Penalize its multiples
            for mult in range(2, 5):
                multiple = bpb * mult
                if multiple in raw_peaks:
                    raw_peaks[multiple] *= 0.5

    for bpb, peak in raw_peaks.items():
        if peak > 0.02:
            scores[(bpb, 4)] = peak

    if normalize and scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores
