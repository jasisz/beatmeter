"""Signal 2: madmom RNN downbeat activation scoring."""

import numpy as np

from beatmeter.analysis.models import Beat


def signal_madmom_activation(
    madmom_results: dict[int, list[Beat]],
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Signal 2: Score madmom meters by checking if their downbeats land on accents.

    For each madmom run, check whether the downbeat positions correspond
    to actually louder onsets. The correct meter should have downbeats
    on the strongest onsets.
    """
    scores: dict[tuple[int, int], float] = {}

    if not madmom_results or len(onset_times) < 4:
        return scores

    for bpb, beats in madmom_results.items():
        downbeats = [b for b in beats if b.is_downbeat]
        non_downbeats = [b for b in beats if not b.is_downbeat]

        if len(downbeats) < 2 or len(non_downbeats) < 2:
            continue

        # Get onset strength at downbeat positions (max in +-50ms window)
        db_strengths = []
        for db in downbeats:
            window_mask = np.abs(onset_times - db.time) < 0.05
            if np.any(window_mask):
                db_strengths.append(float(np.max(onset_strengths[window_mask])))

        # Get onset strength at non-downbeat positions (max in +-50ms window)
        ndb_strengths = []
        for ndb in non_downbeats:
            window_mask = np.abs(onset_times - ndb.time) < 0.05
            if np.any(window_mask):
                ndb_strengths.append(float(np.max(onset_strengths[window_mask])))

        if not db_strengths or not ndb_strengths:
            continue

        # Score: ratio of downbeat strength to non-downbeat strength
        mean_db = float(np.mean(db_strengths))
        mean_ndb = float(np.mean(ndb_strengths))

        if mean_ndb > 0:
            ratio = mean_db / mean_ndb
            # Ratio > 1 means downbeats are louder (good!)
            score = min(1.0, max(0.0, (ratio - 0.8) * 2.5))
        else:
            score = 0.5

        scores[(bpb, 4)] = score

    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores
