"""Sub-beat division detection (simple /4 vs compound /8)."""

import logging

import numpy as np

from beatmeter.analysis.models import Beat

logger = logging.getLogger(__name__)

# Compound meter detection thresholds
COMPOUND_ONSET_MIN = 1.5
COMPOUND_ONSET_MAX = 3.5
COMPOUND_EVENNESS_CV_MAX = 0.4
COMPOUND_TRIPLET_POS_MIN = 0.4


def signal_sub_beat_division(
    all_beats: list[Beat],
    onset_event_times: np.ndarray,
    sr: int = 22050,
) -> int | None:
    """Detect whether the music is simple (/4) or compound (/8) based on sub-beat onset density.

    Counts onsets between consecutive beats. If the median onset count is ~3,
    the music has a compound (triplet) feel -> /8 denominator. If ~2 or fewer,
    it is simple -> /4 denominator.

    Returns:
        8 for compound meter, 4 for simple meter, None if inconclusive.
    """
    if len(all_beats) < 4 or len(onset_event_times) < 8:
        return None

    beat_times = np.array([b.time for b in all_beats])
    onset_counts = []

    for i in range(len(beat_times) - 1):
        t_start = beat_times[i]
        t_end = beat_times[i + 1]
        interval = t_end - t_start

        # Skip very short or very long intervals (likely errors)
        if interval < 0.15 or interval > 2.0:
            continue

        # Count onsets strictly between beats (exclude beat positions themselves)
        margin = 0.03  # 30ms margin to exclude the beat onset itself
        mask = (onset_event_times > t_start + margin) & (onset_event_times < t_end - margin)
        n_onsets = int(np.sum(mask))
        onset_counts.append(n_onsets)

    if len(onset_counts) < 4:
        return None

    median_count = float(np.median(onset_counts))
    logger.debug(f"Sub-beat division: median onsets between beats = {median_count:.1f}")

    # If median is ~2 (i.e., 2 onsets between beats = 3 sub-divisions = triplet feel)
    # this indicates compound meter (/8)
    if COMPOUND_ONSET_MIN <= median_count <= COMPOUND_ONSET_MAX:
        # Evenness check: true compound meter has evenly-spaced sub-beats.
        margin = 0.03
        interval_cvs = []
        for i in range(len(beat_times) - 1):
            t_start, t_end = beat_times[i], beat_times[i + 1]
            mask = (onset_event_times > t_start + margin) & (onset_event_times < t_end - margin)
            sub_onsets = onset_event_times[mask]
            if len(sub_onsets) >= 2:
                intervals = np.diff(sub_onsets)
                mean_interval = float(np.mean(intervals))
                if mean_interval > 0:
                    cv = float(np.std(intervals) / mean_interval)
                    interval_cvs.append(cv)

        if interval_cvs:
            median_cv = float(np.median(interval_cvs))
            logger.debug(f"Sub-beat evenness: median CV = {median_cv:.3f}")
            if median_cv > COMPOUND_EVENNESS_CV_MAX:
                logger.debug(f"Sub-beat intervals uneven (CV > {COMPOUND_EVENNESS_CV_MAX}): not compound")
                return 4

        # Positional consistency: true compound meter has sub-onsets at ~1/3
        # and ~2/3 of the beat interval.
        margin = 0.03
        triplet_beats = 0
        checked_beats = 0
        for i in range(len(beat_times) - 1):
            t_start, t_end = beat_times[i], beat_times[i + 1]
            interval = t_end - t_start
            if interval < 0.15 or interval > 2.0:
                continue
            sub = onset_event_times[(onset_event_times > t_start + margin) & (onset_event_times < t_end - margin)]
            if len(sub) == 2:
                checked_beats += 1
                pos1 = (sub[0] - t_start) / interval
                pos2 = (sub[1] - t_start) / interval
                # True triplets: onsets at ~0.33 and ~0.67 (tolerance 0.15)
                if abs(pos1 - 1/3) < 0.15 and abs(pos2 - 2/3) < 0.15:
                    triplet_beats += 1

        if checked_beats >= 4:
            triplet_frac = triplet_beats / checked_beats
            logger.debug(f"Triplet position check: {triplet_beats}/{checked_beats} = {triplet_frac:.2f}")
            if triplet_frac < COMPOUND_TRIPLET_POS_MIN:
                logger.debug("Sub-onsets not at triplet positions: not compound")
                return 4

        return 8
    # If median is ~1 or 0 (1 onset between = 2 sub-divisions = simple feel)
    elif median_count < COMPOUND_ONSET_MIN:
        return 4
    # Very dense onsets (>3.5) - could be fast ornaments, inconclusive
    return None
