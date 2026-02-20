"""Beat-synchronous chroma Self-Similarity Matrix features for MeterNet (75 dims).

Measures how often the harmonic (chroma) pattern repeats at beat-aligned lags.
If chroma repeats every 3 beats → evidence for 3/4, every 4 → 4/4, etc.

Per tracker (25 dims):
  - Diagonal similarity at lags 2-12: 11 raw values
  - Normalized (each / max): 11 values
  - Peak lag (argmax / 12): 1 value
  - Peak height: 1 value
  - Entropy of lag profile: 1 value

3 trackers (BeatNet, Beat This!, Madmom) x 25 = 75 dimensions.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_LAG = 2
MAX_LAG = 12
N_LAGS = MAX_LAG - MIN_LAG + 1  # 11
N_SSM_PER_TRACKER = 2 * N_LAGS + 3  # 11 raw + 11 norm + 3 stats = 25
N_TRACKERS = 3
N_SSM_FEATURES = N_TRACKERS * N_SSM_PER_TRACKER  # 75

_SR = 22050
_HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _compute_chroma(y: np.ndarray, sr: int = _SR) -> np.ndarray:
    """Compute chroma features from audio. Returns (12, n_frames)."""
    import librosa

    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP_LENGTH)


def _beat_sync_chroma(
    chroma: np.ndarray,
    beat_times: np.ndarray,
    sr: int = _SR,
) -> np.ndarray | None:
    """Average chroma between consecutive beats.

    Args:
        chroma: (12, n_frames) chroma matrix.
        beat_times: Beat onset times in seconds.
        sr: Sample rate.

    Returns:
        (12, n_beats-1) beat-synchronous chroma, or None if too few beats.
    """
    import librosa

    if len(beat_times) < 4:
        return None

    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=_HOP_LENGTH)
    n_frames = chroma.shape[1]

    segments = []
    for i in range(len(beat_frames) - 1):
        start = max(0, beat_frames[i])
        end = min(n_frames, beat_frames[i + 1])
        if end <= start:
            segments.append(np.zeros(12))
        else:
            segments.append(chroma[:, start:end].mean(axis=1))

    if len(segments) < MIN_LAG + 1:
        return None

    result = np.column_stack(segments)  # (12, n_segments)
    # L2-normalize each beat column for cosine similarity
    norms = np.linalg.norm(result, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    result = result / norms

    return result


def _diagonal_similarity_profile(
    beat_chroma: np.ndarray,
    min_lag: int = MIN_LAG,
    max_lag: int = MAX_LAG,
) -> np.ndarray:
    """Mean cosine similarity along diagonals at each lag.

    Args:
        beat_chroma: (12, n_beats) L2-normalized beat-synchronous chroma.
        min_lag: Minimum lag in beats.
        max_lag: Maximum lag in beats.

    Returns:
        (max_lag - min_lag + 1,) similarity profile.
    """
    n_beats = beat_chroma.shape[1]
    n_lags = max_lag - min_lag + 1
    profile = np.zeros(n_lags)

    for i, lag in enumerate(range(min_lag, max_lag + 1)):
        if lag >= n_beats:
            continue
        # Cosine similarity between beat_chroma[:, j] and beat_chroma[:, j+lag]
        # Since columns are L2-normalized, dot product = cosine similarity
        sims = np.sum(
            beat_chroma[:, :n_beats - lag] * beat_chroma[:, lag:],
            axis=0,
        )
        if len(sims) > 0:
            profile[i] = float(np.mean(sims))

    return profile


def _extract_ssm_for_beats(
    chroma: np.ndarray,
    beat_times: np.ndarray,
    sr: int = _SR,
) -> np.ndarray:
    """Extract 25-dim SSM features for one set of beat times.

    Layout: 11 raw + 11 normalized + peak_lag + peak_height + entropy.
    Returns zeros if beats are insufficient.
    """
    feat = np.zeros(N_SSM_PER_TRACKER)

    beat_chroma = _beat_sync_chroma(chroma, beat_times, sr)
    if beat_chroma is None:
        return feat

    profile = _diagonal_similarity_profile(beat_chroma)

    # Raw similarity values (11 dims)
    feat[:N_LAGS] = profile

    # Normalized (each / max) (11 dims)
    pmax = profile.max()
    if pmax > 1e-8:
        feat[N_LAGS:2 * N_LAGS] = profile / pmax
    else:
        feat[N_LAGS:2 * N_LAGS] = 0.0

    # Peak lag (normalized to [0, 1])
    peak_idx = int(np.argmax(profile))
    feat[2 * N_LAGS] = (peak_idx + MIN_LAG) / MAX_LAG

    # Peak height
    feat[2 * N_LAGS + 1] = pmax

    # Entropy of lag profile
    abs_profile = np.abs(profile)
    total = abs_profile.sum()
    if total > 1e-8:
        probs = abs_profile / total
        probs = probs[probs > 0]
        feat[2 * N_LAGS + 2] = float(-np.sum(probs * np.log2(probs)))
    else:
        feat[2 * N_LAGS + 2] = 0.0

    return feat


# ---------------------------------------------------------------------------
# Cache-based extraction (for training)
# ---------------------------------------------------------------------------


def extract_ssm_features_cached(
    y: np.ndarray,
    sr: int,
    cache,
    audio_hash: str,
    preloaded=None,
) -> np.ndarray:
    """Extract SSM features using cached beat data (75 dims).

    Args:
        y: Audio signal (mono).
        sr: Sample rate.
        cache: AnalysisCache instance.
        audio_hash: Audio file hash for cache lookups.
        preloaded: Optional dict from cache.load_all_for_audio() for batch reads.

    Returns:
        Feature vector of shape (75,).
    """
    tracker_names = ["beatnet", "beat_this", "madmom"]

    try:
        chroma = _compute_chroma(y, sr)
    except Exception as e:
        logger.warning("Chroma computation failed: %s", e)
        return np.zeros(N_SSM_FEATURES)

    parts = []
    for tracker_name in tracker_names:
        beat_times = _get_beat_times_from_cache(
            cache, audio_hash, tracker_name, preloaded=preloaded,
        )
        if beat_times is not None and len(beat_times) >= 4:
            parts.append(_extract_ssm_for_beats(chroma, beat_times, sr))
        else:
            parts.append(np.zeros(N_SSM_PER_TRACKER))

    feat = np.concatenate(parts)
    assert feat.shape[0] == N_SSM_FEATURES, (
        f"SSM dim mismatch: {feat.shape[0]} != {N_SSM_FEATURES}"
    )
    return feat


def _get_beat_times_from_cache(
    cache, audio_hash: str, tracker_name: str, preloaded=None,
) -> np.ndarray | None:
    """Load beat times from cache for a given tracker."""

    def _load_beats(tracker):
        if preloaded:
            return preloaded["beats"].get(tracker)
        return cache.load_beats(audio_hash, tracker)

    if tracker_name == "madmom":
        # Pick best BPB variant (most beats)
        best_data = None
        best_count = 0
        for bpb in [3, 4, 5, 7]:
            data = _load_beats(f"madmom_bpb{bpb}")
            if data and len(data) > best_count:
                best_count = len(data)
                best_data = data
        if best_data:
            return np.array([b["time"] for b in best_data])
        return None
    else:
        data = _load_beats(tracker_name)
        if data and len(data) > 0:
            return np.array([b["time"] for b in data])
        return None


# ---------------------------------------------------------------------------
# Live extraction (for inference in meter.py)
# ---------------------------------------------------------------------------


def extract_ssm_features_live(
    y: np.ndarray,
    sr: int,
    beatnet_beats,
    beat_this_beats,
    madmom_results: dict,
) -> np.ndarray:
    """Extract SSM features from live beat data (75 dims).

    Args:
        y: Audio signal (mono).
        sr: Sample rate.
        beatnet_beats: list of Beat objects from BeatNet.
        beat_this_beats: list of Beat objects from Beat This!, or None.
        madmom_results: dict mapping bpb -> list of Beat objects.

    Returns:
        Feature vector of shape (75,).
    """
    try:
        chroma = _compute_chroma(y, sr)
    except Exception as e:
        logger.warning("Chroma computation failed: %s", e)
        return np.zeros(N_SSM_FEATURES)

    parts = []

    # BeatNet
    if beatnet_beats and len(beatnet_beats) >= 4:
        bt = np.array([b.time for b in beatnet_beats])
        parts.append(_extract_ssm_for_beats(chroma, bt, sr))
    else:
        parts.append(np.zeros(N_SSM_PER_TRACKER))

    # Beat This!
    if beat_this_beats and len(beat_this_beats) >= 4:
        bt = np.array([b.time for b in beat_this_beats])
        parts.append(_extract_ssm_for_beats(chroma, bt, sr))
    else:
        parts.append(np.zeros(N_SSM_PER_TRACKER))

    # Madmom (pick variant with most beats)
    best_beats = None
    best_count = 0
    if madmom_results:
        for bpb, beats in madmom_results.items():
            if len(beats) > best_count:
                best_count = len(beats)
                best_beats = beats
    if best_beats and len(best_beats) >= 4:
        bt = np.array([b.time for b in best_beats])
        parts.append(_extract_ssm_for_beats(chroma, bt, sr))
    else:
        parts.append(np.zeros(N_SSM_PER_TRACKER))

    feat = np.concatenate(parts)
    assert feat.shape[0] == N_SSM_FEATURES, (
        f"SSM dim mismatch: {feat.shape[0]} != {N_SSM_FEATURES}"
    )
    return feat
