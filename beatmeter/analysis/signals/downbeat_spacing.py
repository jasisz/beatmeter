"""Signal 1: Downbeat spacing analysis from any tracker with downbeat info."""

from collections import Counter

from beatmeter.analysis.models import Beat


def signal_downbeat_spacing(beats: list[Beat]) -> dict[tuple[int, int], float]:
    """Signal 1: Analyze downbeat spacing from any tracker with downbeat info."""
    scores: dict[tuple[int, int], float] = {}

    downbeat_indices = [i for i, b in enumerate(beats) if b.is_downbeat]
    if len(downbeat_indices) < 2:
        return scores

    spacings = []
    for i in range(len(downbeat_indices) - 1):
        n_beats = downbeat_indices[i + 1] - downbeat_indices[i]
        if n_beats > 0:
            spacings.append(n_beats)

    if not spacings:
        return scores

    counter = Counter(spacings)
    total = len(spacings)

    for spacing, count in counter.items():
        ratio = count / total
        if ratio < 0.15:
            continue
        if spacing in range(2, 13):
            scores[(spacing, 4)] = ratio
            # Octave ambiguity: if BeatNet says 2, also consider 4
            if spacing * 2 <= 12:
                double = (spacing * 2, 4)
                scores[double] = scores.get(double, 0) + ratio * 0.4

    return scores
