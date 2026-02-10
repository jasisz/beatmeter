"""Signal 4: Accent pattern analysis + beat energy computation."""

from collections import Counter

import numpy as np

from beatmeter.analysis.models import Beat


def compute_beat_energies(
    beats: list[Beat],
    audio: np.ndarray,
    sr: int,
    window_ms: float = 30.0,
) -> np.ndarray:
    """Compute RMS energy in a window around each beat position.

    Uses raw audio amplitude rather than spectral flux, giving more
    reliable accent detection (especially for percussive sounds).
    """
    if not beats:
        return np.array([])
    beat_times = np.array([b.time for b in beats])
    window_samples = int(window_ms / 1000.0 * sr)
    energies = np.zeros(len(beat_times))
    for i, bt in enumerate(beat_times):
        center = int(bt * sr)
        start = max(0, center - window_samples)
        end = min(len(audio), center + window_samples)
        if end > start:
            energies[i] = float(np.sqrt(np.mean(audio[start:end] ** 2)))
    return energies


def signal_accent_pattern(
    beats: list[Beat],
    beat_energies: np.ndarray,
    normalize: bool = True,
) -> dict[tuple[int, int], float]:
    """Signal 4: Accent pattern analysis.

    Groups beats into hypothetical bars and checks if beat 1 is consistently
    the strongest beat (indicating it's a real downbeat).
    Uses RMS energy at each beat position for reliable accent detection.
    """
    scores: dict[tuple[int, int], float] = {}

    if len(beats) < 6 or len(beat_energies) < 6:
        return scores

    for beats_per_bar in [2, 3, 4, 5, 6, 7, 9, 11, 12]:
        n_complete_bars = len(beat_energies) // beats_per_bar
        if n_complete_bars < 3:
            continue

        trimmed = beat_energies[:n_complete_bars * beats_per_bar]
        bars = trimmed.reshape(n_complete_bars, beats_per_bar)

        # Average accent pattern across bars
        avg_pattern = np.mean(bars, axis=0)
        pattern_mean = np.mean(avg_pattern)
        if pattern_mean <= 0:
            continue

        # Key metric: how much does the pattern vary within a bar?
        # The correct meter should show clear accent structure.
        # Wrong meters will have flat patterns (all beats equally strong).
        pattern_cv = float(np.std(avg_pattern) / pattern_mean) if pattern_mean > 0 else 0

        # Also check: is the strongest beat consistently in the same position?
        max_positions = [int(np.argmax(bar)) for bar in bars]
        mode_count = max(Counter(max_positions).values())
        consistency = mode_count / n_complete_bars

        # Combined score: pattern contrast * consistency
        score = pattern_cv * consistency
        if score > 0.01:
            scores[(beats_per_bar, 4)] = score

    if normalize and scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores
