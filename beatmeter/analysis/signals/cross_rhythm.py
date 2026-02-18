"""Signal 10: Cross-rhythm detection.

Detects secondary meter in polyrhythmic music by analyzing onset patterns
within bars of the primary meter. For example, in Fela Kuti's music with
4/4 as primary meter, this signal detects the cross-rhythm 3-grouping
(onset peaks at 1/3 and 2/3 of each bar).

Algorithm:
1. Segment audio into bars using beat_times + primary_meter
2. Normalize onset positions within each bar to [0, 1)
3. Build weighted histogram (n_bins = 24 = LCM(3,4,6,8))
4. For each candidate cross-meter, check if histogram peaks align
   with that meter's expected positions
5. Score = ratio of cross-meter position strength vs median
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Cross-meter candidates for each primary meter numerator
_CROSS_CANDIDATES = {
    3: [4],     # triple -> check for duple cross-rhythm
    4: [3],     # duple -> check for triple cross-rhythm
    5: [3, 4],
    7: [3, 4],
}

# Expected normalized positions within a bar for each cross-meter grouping
# e.g., 3-grouping in a 4-beat bar: onsets at 0/3, 1/3, 2/3 of bar
_CROSS_POSITIONS = {
    3: [0.0, 1 / 3, 2 / 3],
    4: [0.0, 0.25, 0.5, 0.75],
}

N_BINS = 24  # LCM(3, 4, 6, 8) — allows clean alignment for all groupings
POSITION_TOLERANCE = 1.5 / N_BINS  # ~0.0625, half a bin width


def signal_cross_rhythm(
    beat_times: np.ndarray,
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    primary_meter: int = 4,
    n_bins: int = N_BINS,
) -> dict[tuple[int, int], float]:
    """Detect cross-rhythms by analyzing onset histogram within bars.

    Args:
        beat_times: Array of beat positions (seconds) from best tracker.
        onset_times: Array of onset positions (seconds).
        onset_strengths: Array of onset strengths (energy).
        primary_meter: Detected primary meter numerator (e.g., 4 for 4/4).
        n_bins: Number of histogram bins per bar.

    Returns:
        Dict mapping (numerator, denominator) -> score for detected cross-rhythms.
        Empty dict if no cross-rhythm detected.
    """
    scores: dict[tuple[int, int], float] = {}

    if len(beat_times) < primary_meter * 2 or len(onset_times) < 8:
        return scores

    candidates = _CROSS_CANDIDATES.get(primary_meter, [])
    if not candidates:
        return scores

    # Build bars: every primary_meter beats = 1 bar
    bar_starts = beat_times[::primary_meter]
    if len(bar_starts) < 2:
        return scores

    # Compute bar durations for normalization
    bar_durs = np.diff(bar_starts)
    if len(bar_durs) == 0:
        return scores

    # Collect normalized onset positions within bars, weighted by strength
    positions: list[float] = []
    weights: list[float] = []

    for i in range(len(bar_starts) - 1):
        bar_start = bar_starts[i]
        bar_end = bar_starts[i + 1]
        bar_dur = bar_end - bar_start
        if bar_dur <= 0:
            continue

        # Find onsets in this bar
        mask = (onset_times >= bar_start) & (onset_times < bar_end)
        bar_onsets = onset_times[mask]
        bar_strengths = onset_strengths[mask]

        for t, s in zip(bar_onsets, bar_strengths):
            pos = (t - bar_start) / bar_dur  # normalize to [0, 1)
            positions.append(pos)
            weights.append(s)

    if len(positions) < 8:
        return scores

    positions_arr = np.array(positions)
    weights_arr = np.array(weights)

    # Normalize weights
    w_max = weights_arr.max()
    if w_max > 0:
        weights_arr = weights_arr / w_max

    # Build weighted histogram
    hist = np.zeros(n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for pos, w in zip(positions_arr, weights_arr):
        bin_idx = min(int(pos * n_bins), n_bins - 1)
        hist[bin_idx] += w

    # Smooth with small kernel to reduce noise
    kernel = np.array([0.25, 0.5, 0.25])
    hist_smooth = np.convolve(hist, kernel, mode="same")

    median_val = np.median(hist_smooth)
    if median_val <= 0:
        median_val = np.mean(hist_smooth) or 1.0

    # Score each candidate cross-meter
    for cross_meter in candidates:
        expected_positions = _CROSS_POSITIONS.get(cross_meter, [])
        if not expected_positions:
            continue

        # For each expected position, find the histogram value
        position_scores = []
        for exp_pos in expected_positions:
            bin_idx = min(int(exp_pos * n_bins), n_bins - 1)
            # Check neighboring bins too (tolerance)
            window = []
            for offset in [-1, 0, 1]:
                idx = (bin_idx + offset) % n_bins
                window.append(hist_smooth[idx])
            position_scores.append(max(window))

        # Score: geometric mean of position strengths / median
        # Geometric mean ensures ALL positions must be strong
        pos_ratios = [max(s / median_val, 0.01) for s in position_scores]
        geo_mean = np.exp(np.mean(np.log(pos_ratios)))

        # Normalize: score of 1.0 means cross-positions are exactly at median
        # score > 2.0 means strong cross-rhythm
        # Subtract 1 and clamp to [0, 1] for normalized signal score
        raw_score = max(0.0, (geo_mean - 1.0) / 3.0)  # 4.0 ratio -> 1.0 score
        score = min(1.0, raw_score)

        if score > 0.05:
            # Use denominator 4 for cross_meter (standard time signatures)
            den = 4 if cross_meter <= 7 else 8
            scores[(cross_meter, den)] = round(score, 4)
            logger.debug(
                f"Cross-rhythm: {cross_meter}/{den} in {primary_meter}/x "
                f"(geo_mean={geo_mean:.2f}, score={score:.3f})"
            )

    return scores
