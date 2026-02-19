"""Feature extraction for MeterNet — unified meter classification model.

MeterNet combines:
- Audio features (1361 dims) from onset_mlp_features.py
- Beat tracker features (42 dims) from cached beat data
- Signal scores (60 dims) from cached signal data
- Tempo features (4 dims) from cached tempo data

Total: 1467 dimensions.

Two extraction paths:
1. Cache-based: for training (reads from .cache/)
2. Live: for inference (from engine data, no cache needed)
"""

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group constants
# ---------------------------------------------------------------------------

N_AUDIO_FEATURES = 1361  # from onset_mlp_features.py v5

# Beat features: 3 trackers × 12 dims + 6 cross-tracker agreement
N_TRACKERS = 3  # beatnet, beat_this, madmom (best BPB)
N_BEAT_FEATURES_PER_TRACKER = 12
N_DOWNBEAT_HIST_BINS = 6  # bar lengths [2, 3, 4, 5, 6, 7]
N_AGREEMENT_METERS = 6  # [3, 4, 5, 7, 9, 11]
N_BEAT_FEATURES = N_TRACKERS * N_BEAT_FEATURES_PER_TRACKER + N_AGREEMENT_METERS  # 42

# Signal scores: 5 signals × 12 meter keys
N_SIGNAL_NAMES = 5
N_METER_KEYS = 12
N_SIGNAL_FEATURES = N_SIGNAL_NAMES * N_METER_KEYS  # 60

N_TEMPO_FEATURES = 4

TOTAL_FEATURES = (
    N_AUDIO_FEATURES + N_BEAT_FEATURES + N_SIGNAL_FEATURES + N_TEMPO_FEATURES
)  # 1467

FEATURE_VERSION = "mn_v1"

# Tracker names (order matters for feature layout)
TRACKER_NAMES = ["beatnet", "beat_this", "madmom"]

# Downbeat spacing histogram bins (bar lengths in beats)
DOWNBEAT_HIST_BINS = [2, 3, 4, 5, 6, 7]

# Cross-tracker agreement meters
AGREEMENT_METERS = [3, 4, 5, 7, 9, 11]

# Signal names (minus onset_mlp — MeterNet replaces it)
SIGNAL_NAMES = ["beatnet", "beat_this", "autocorr", "bar_tracking", "hcdf"]
SIGNAL_CACHE_MAP = {
    "beatnet": "beatnet_spacing",
    "beat_this": "beat_this_spacing",
    "autocorr": "onset_autocorr",
    "bar_tracking": "bar_tracking",
    "hcdf": "hcdf_meter",
}

# Meter keys (same order as arbiter)
METER_KEYS = [
    "2_4", "3_4", "4_4", "5_4", "5_8", "6_8",
    "7_4", "7_8", "9_8", "10_8", "11_8", "12_8",
]

# Feature group offsets
FEATURE_GROUPS = {
    "audio": (0, N_AUDIO_FEATURES),
    "beat": (N_AUDIO_FEATURES, N_AUDIO_FEATURES + N_BEAT_FEATURES),
    "signal": (
        N_AUDIO_FEATURES + N_BEAT_FEATURES,
        N_AUDIO_FEATURES + N_BEAT_FEATURES + N_SIGNAL_FEATURES,
    ),
    "tempo": (
        N_AUDIO_FEATURES + N_BEAT_FEATURES + N_SIGNAL_FEATURES,
        TOTAL_FEATURES,
    ),
}


# ---------------------------------------------------------------------------
# Beat feature extraction helpers
# ---------------------------------------------------------------------------


def _onset_alignment_score(
    beat_times: np.ndarray,
    onset_event_times: np.ndarray,
    max_dist: float = 0.07,
) -> float:
    """Score how well beats align with onset events (F-measure)."""
    if len(beat_times) == 0 or len(onset_event_times) == 0:
        return 0.0

    total_fwd = 0.0
    for b in beat_times:
        min_dist = float(np.min(np.abs(onset_event_times - b)))
        total_fwd += min(min_dist, max_dist)
    precision = 1.0 - total_fwd / len(beat_times) / max_dist

    total_rev = 0.0
    for ot in onset_event_times:
        min_dist = float(np.min(np.abs(beat_times - ot)))
        total_rev += min(min_dist, max_dist)
    recall = 1.0 - total_rev / len(onset_event_times) / max_dist

    if precision + recall > 0:
        return 2.0 * precision * recall / (precision + recall)
    return 0.0


def _compute_tracker_features(
    beat_times: np.ndarray,
    is_downbeat: np.ndarray | None,
    onset_event_times: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Compute 12-dim features for a single tracker.

    Layout:
      [0] IBI median (normalized /3.0)
      [1] IBI std (normalized /1.0)
      [2] IBI CV (capped at 2.0, normalized)
      [3] beat count per second (normalized /10.0)
      [4] onset alignment score [0..1]
      [5:11] downbeat spacing histogram [2,3,4,5,6,7] (normalized)
      [11] tempo from IBI (BPM / 300)
    """
    feat = np.zeros(N_BEAT_FEATURES_PER_TRACKER)

    if len(beat_times) < 3:
        return feat

    # IBI statistics
    ibis = np.diff(beat_times)
    valid_ibis = ibis[(ibis > 0.1) & (ibis < 3.0)]
    if len(valid_ibis) >= 2:
        ibi_median = float(np.median(valid_ibis))
        ibi_std = float(np.std(valid_ibis))
        ibi_cv = ibi_std / max(ibi_median, 1e-8)
        feat[0] = ibi_median / 3.0
        feat[1] = ibi_std / 1.0
        feat[2] = min(ibi_cv, 2.0) / 2.0

    # Beat count per second
    if duration > 0:
        feat[3] = len(beat_times) / duration / 10.0

    # Onset alignment
    if len(onset_event_times) > 0:
        feat[4] = _onset_alignment_score(beat_times, onset_event_times)

    # Downbeat spacing histogram
    if is_downbeat is not None and np.any(is_downbeat):
        db_indices = np.where(is_downbeat)[0]
        if len(db_indices) >= 2:
            spacings = np.diff(db_indices)
            hist = np.zeros(N_DOWNBEAT_HIST_BINS)
            for spacing in spacings:
                for j, n in enumerate(DOWNBEAT_HIST_BINS):
                    if spacing == n:
                        hist[j] += 1
                        break
            total = hist.sum()
            if total > 0:
                hist /= total
            feat[5:11] = hist

    # Tempo from IBI
    if len(valid_ibis) >= 2:
        tempo_bpm = 60.0 / float(np.median(valid_ibis))
        feat[11] = min(tempo_bpm, 300.0) / 300.0

    return feat


def _get_dominant_meter(is_downbeat: np.ndarray | None) -> int | None:
    """Get dominant meter from downbeat pattern (mode of downbeat spacings)."""
    if is_downbeat is None or not np.any(is_downbeat):
        return None
    db_indices = np.where(is_downbeat)[0]
    if len(db_indices) < 2:
        return None
    spacings = np.diff(db_indices)
    if len(spacings) == 0:
        return None
    counts = Counter(int(s) for s in spacings)
    mode = counts.most_common(1)[0][0]
    return mode if mode in AGREEMENT_METERS else None


def _compute_cross_tracker_agreement(
    tracker_dominant_meters: list[int | None],
) -> np.ndarray:
    """Compute 6-dim cross-tracker meter agreement."""
    agreement = np.zeros(N_AGREEMENT_METERS)
    n_active = sum(1 for m in tracker_dominant_meters if m is not None)
    if n_active < 2:
        return agreement

    for i, meter in enumerate(AGREEMENT_METERS):
        count = sum(1 for m in tracker_dominant_meters if m == meter)
        agreement[i] = count / n_active

    return agreement


# ---------------------------------------------------------------------------
# Cache-based extraction (for training)
# ---------------------------------------------------------------------------


def extract_beat_features(cache, audio_hash: str) -> np.ndarray:
    """Extract beat features from analysis cache (42 dims).

    Reads cached beats for 3 trackers + onsets, computes IBI stats,
    alignment, downbeat histograms, and cross-tracker agreement.
    """
    feat = np.zeros(N_BEAT_FEATURES)

    # Load onset events for alignment
    onset_data = cache.load_onsets(audio_hash)
    onset_event_times = (
        np.array(onset_data["onset_events"]) if onset_data else np.array([])
    )

    # Estimate duration from onset times
    duration = 30.0
    if onset_data and "onset_times" in onset_data:
        ot = onset_data["onset_times"]
        if ot:
            duration = max(ot[-1], 1.0)

    tracker_dominant: list[int | None] = []

    for t_idx, tracker_name in enumerate(TRACKER_NAMES):
        offset = t_idx * N_BEAT_FEATURES_PER_TRACKER

        if tracker_name == "madmom":
            # Try all BPB variants, pick best alignment
            best_data = None
            best_alignment = -1.0
            for bpb in [3, 4, 5, 7]:
                data = cache.load_beats(audio_hash, f"madmom_bpb{bpb}")
                if data and len(data) > 0:
                    bt = np.array([b["time"] for b in data])
                    if len(onset_event_times) > 0 and len(bt) >= 3:
                        align = _onset_alignment_score(bt, onset_event_times)
                        if align > best_alignment:
                            best_alignment = align
                            best_data = data
                    elif best_data is None:
                        best_data = data

            if best_data:
                bt = np.array([b["time"] for b in best_data])
                is_db = np.array([b.get("is_downbeat", False) for b in best_data])
                feat[offset : offset + N_BEAT_FEATURES_PER_TRACKER] = (
                    _compute_tracker_features(bt, is_db, onset_event_times, duration)
                )
                tracker_dominant.append(_get_dominant_meter(is_db))
            else:
                tracker_dominant.append(None)
        else:
            data = cache.load_beats(audio_hash, tracker_name)
            if data and len(data) > 0:
                bt = np.array([b["time"] for b in data])
                is_db = np.array([b.get("is_downbeat", False) for b in data])
                feat[offset : offset + N_BEAT_FEATURES_PER_TRACKER] = (
                    _compute_tracker_features(bt, is_db, onset_event_times, duration)
                )
                tracker_dominant.append(_get_dominant_meter(is_db))
            else:
                tracker_dominant.append(None)

    # Cross-tracker agreement
    ag_offset = N_TRACKERS * N_BEAT_FEATURES_PER_TRACKER
    feat[ag_offset : ag_offset + N_AGREEMENT_METERS] = (
        _compute_cross_tracker_agreement(tracker_dominant)
    )

    return feat


def extract_signal_scores(cache, audio_hash: str) -> np.ndarray:
    """Extract signal score features from cache (60 dims).

    5 signals x 12 meter keys = 60 features.
    """
    feat = np.zeros(N_SIGNAL_FEATURES)

    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        cache_name = SIGNAL_CACHE_MAP[sig_name]
        cached = cache.load_signal(audio_hash, cache_name)
        if cached:
            for m_idx, meter_key in enumerate(METER_KEYS):
                feat[s_idx * N_METER_KEYS + m_idx] = cached.get(meter_key, 0.0)

    return feat


def extract_tempo_features(cache, audio_hash: str) -> np.ndarray:
    """Extract tempo features from cache (4 dims).

    [0] tempo_librosa BPM (normalized /300)
    [1] tempo_tempogram BPM (normalized /300)
    [2] tempo ratio (lib / tg)
    [3] tempo agreement (1 - |lib - tg| / max)
    """
    feat = np.zeros(N_TEMPO_FEATURES)

    tempo_lib = 0.0
    tempo_tg = 0.0

    td = cache.load_signal(audio_hash, "tempo_librosa")
    if td:
        tempo_lib = td.get("bpm", 0.0)

    td = cache.load_signal(audio_hash, "tempo_tempogram")
    if td:
        tempo_tg = td.get("bpm", 0.0)

    return _compute_tempo_features(tempo_lib, tempo_tg)


def _compute_tempo_features(tempo_lib: float, tempo_tg: float) -> np.ndarray:
    """Compute 4-dim tempo feature vector from two BPM estimates."""
    feat = np.zeros(N_TEMPO_FEATURES)
    feat[0] = min(tempo_lib, 300.0) / 300.0
    feat[1] = min(tempo_tg, 300.0) / 300.0

    if tempo_lib > 0 and tempo_tg > 0:
        feat[2] = tempo_lib / tempo_tg
        feat[3] = 1.0 - abs(tempo_lib - tempo_tg) / max(tempo_lib, tempo_tg)
    else:
        feat[2] = 1.0
        feat[3] = 0.0 if (tempo_lib == 0 and tempo_tg == 0) else 0.5

    return feat


# ---------------------------------------------------------------------------
# Live extraction (for inference in meter.py)
# ---------------------------------------------------------------------------


def extract_beat_features_live(
    beatnet_beats,
    beat_this_beats,
    madmom_results: dict,
    onset_event_times: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Extract beat features from live engine data (42 dims).

    Args:
        beatnet_beats: list of Beat objects from BeatNet.
        beat_this_beats: list of Beat objects from Beat This, or None.
        madmom_results: dict mapping bpb -> list of Beat objects.
        onset_event_times: onset event times array.
        duration: audio duration in seconds.
    """
    feat = np.zeros(N_BEAT_FEATURES)
    tracker_dominant: list[int | None] = []

    # BeatNet
    if beatnet_beats and len(beatnet_beats) >= 3:
        bt = np.array([b.time for b in beatnet_beats])
        is_db = np.array([b.is_downbeat for b in beatnet_beats])
        feat[0:N_BEAT_FEATURES_PER_TRACKER] = _compute_tracker_features(
            bt, is_db, onset_event_times, duration,
        )
        tracker_dominant.append(_get_dominant_meter(is_db))
    else:
        tracker_dominant.append(None)

    # Beat This
    offset_bt = N_BEAT_FEATURES_PER_TRACKER
    if beat_this_beats and len(beat_this_beats) >= 3:
        bt = np.array([b.time for b in beat_this_beats])
        is_db = np.array([b.is_downbeat for b in beat_this_beats])
        feat[offset_bt : offset_bt + N_BEAT_FEATURES_PER_TRACKER] = (
            _compute_tracker_features(bt, is_db, onset_event_times, duration)
        )
        tracker_dominant.append(_get_dominant_meter(is_db))
    else:
        tracker_dominant.append(None)

    # Madmom (best alignment)
    offset_mm = 2 * N_BEAT_FEATURES_PER_TRACKER
    best_beats = None
    best_alignment = -1.0
    if madmom_results:
        for bpb, beats in madmom_results.items():
            if len(beats) >= 3:
                bt = np.array([b.time for b in beats])
                if len(onset_event_times) > 0:
                    align = _onset_alignment_score(bt, onset_event_times)
                    if align > best_alignment:
                        best_alignment = align
                        best_beats = beats
                elif best_beats is None:
                    best_beats = beats

    if best_beats:
        bt = np.array([b.time for b in best_beats])
        is_db = np.array([b.is_downbeat for b in best_beats])
        feat[offset_mm : offset_mm + N_BEAT_FEATURES_PER_TRACKER] = (
            _compute_tracker_features(bt, is_db, onset_event_times, duration)
        )
        tracker_dominant.append(_get_dominant_meter(is_db))
    else:
        tracker_dominant.append(None)

    # Cross-tracker agreement
    ag_offset = N_TRACKERS * N_BEAT_FEATURES_PER_TRACKER
    feat[ag_offset : ag_offset + N_AGREEMENT_METERS] = (
        _compute_cross_tracker_agreement(tracker_dominant)
    )

    return feat


def extract_signal_scores_live(
    signal_results: dict[str, dict[tuple[int, int], float]],
) -> np.ndarray:
    """Extract signal score features from signal_results dict (60 dims).

    signal_results maps signal_name -> {(num, den): score}.
    Only uses the 5 signals MeterNet expects (excludes onset_mlp).
    """
    feat = np.zeros(N_SIGNAL_FEATURES)

    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        if sig_name in signal_results:
            scores = signal_results[sig_name]
            for m_idx, meter_key in enumerate(METER_KEYS):
                num, den = meter_key.split("_")
                tup = (int(num), int(den))
                feat[s_idx * N_METER_KEYS + m_idx] = scores.get(tup, 0.0)

    return feat


def extract_tempo_features_live(
    tempo_librosa_bpm: float, tempo_tempogram_bpm: float,
) -> np.ndarray:
    """Extract tempo features from BPM values (4 dims)."""
    return _compute_tempo_features(tempo_librosa_bpm, tempo_tempogram_bpm)
