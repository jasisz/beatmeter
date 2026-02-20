"""Feature extraction for MeterNet — unified meter classification model.

MeterNet combines:
- Audio features (1449 dims) from onset_mlp_features.py v6
- Beat-synchronous chroma SSM (75 dims) from ssm_features.py
- Beat tracker features (42 dims) from cached beat data
- Signal scores (60 dims) from cached signal data
- Tempo features (4 dims) from cached tempo data

Total: 1630 dimensions.

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

# v6 audio features
N_AUDIO_FEATURES_V6 = 1449

N_AUDIO_FEATURES = N_AUDIO_FEATURES_V6

# Beat-synchronous chroma SSM: 3 trackers × 25 dims
from beatmeter.analysis.signals.ssm_features import N_SSM_FEATURES  # 75

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
    N_AUDIO_FEATURES + N_SSM_FEATURES
    + N_BEAT_FEATURES + N_SIGNAL_FEATURES + N_TEMPO_FEATURES
)  # 1630

FEATURE_VERSION = "mn_v6"  # v6 = v4 + SSM (75)

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

# Meter keys (canonical order for signal score features)
METER_KEYS = [
    "2_4", "3_4", "4_4", "5_4", "5_8", "6_8",
    "7_4", "7_8", "9_8", "10_8", "11_8", "12_8",
]

# Feature group offsets (v6 default — use feature_groups() for version-aware)
_AFTER_AUDIO = N_AUDIO_FEATURES
_AFTER_SSM = _AFTER_AUDIO + N_SSM_FEATURES
_AFTER_BEAT = _AFTER_SSM + N_BEAT_FEATURES
_AFTER_SIGNAL = _AFTER_BEAT + N_SIGNAL_FEATURES
_AFTER_TEMPO = _AFTER_SIGNAL + N_TEMPO_FEATURES

FEATURE_GROUPS = {
    "audio": (0, _AFTER_AUDIO),
    "ssm": (_AFTER_AUDIO, _AFTER_SSM),
    "beat": (_AFTER_SSM, _AFTER_BEAT),
    "signal": (_AFTER_BEAT, _AFTER_SIGNAL),
    "tempo": (_AFTER_SIGNAL, _AFTER_TEMPO),
}


def feature_groups(n_audio: int | None = None) -> dict[str, tuple[int, int]]:
    """Return feature group offsets for a given audio feature size."""
    if n_audio is None:
        return FEATURE_GROUPS
    a = n_audio
    ssm = a + N_SSM_FEATURES
    b = ssm + N_BEAT_FEATURES
    s = b + N_SIGNAL_FEATURES
    t = s + N_TEMPO_FEATURES
    return {
        "audio": (0, a),
        "ssm": (a, ssm),
        "beat": (ssm, b),
        "signal": (b, s),
        "tempo": (s, t),
    }

# Granular ablation targets (individual features within groups)
_BT = N_BEAT_FEATURES_PER_TRACKER  # 12
_MK = N_METER_KEYS  # 12
ABLATION_TARGETS = {
    # SSM feature group
    "ssm": (_AFTER_AUDIO, _AFTER_SSM),
    # Beat trackers (12 dims each)
    "beat_beatnet": (_AFTER_SSM, _AFTER_SSM + _BT),
    "beat_beat_this": (_AFTER_SSM + _BT, _AFTER_SSM + 2 * _BT),
    "beat_madmom": (_AFTER_SSM + 2 * _BT, _AFTER_SSM + 3 * _BT),
    # Signal scores (12 dims each)
    "sig_beatnet": (_AFTER_BEAT, _AFTER_BEAT + _MK),
    "sig_beat_this": (_AFTER_BEAT + _MK, _AFTER_BEAT + 2 * _MK),
    "sig_autocorr": (_AFTER_BEAT + 2 * _MK, _AFTER_BEAT + 3 * _MK),
    "sig_bar_tracking": (_AFTER_BEAT + 3 * _MK, _AFTER_BEAT + 4 * _MK),
    "sig_hcdf": (_AFTER_BEAT + 4 * _MK, _AFTER_BEAT + 5 * _MK),
    # Tempo (4 dims)
    "tempo": (_AFTER_SIGNAL, _AFTER_TEMPO),
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


def extract_beat_features(cache, audio_hash: str, preloaded=None) -> np.ndarray:
    """Extract beat features from analysis cache (42 dims).

    Reads cached beats for 3 trackers + onsets, computes IBI stats,
    alignment, downbeat histograms, and cross-tracker agreement.

    Args:
        preloaded: Optional dict from cache.load_all_for_audio() for batch reads.
    """
    feat = np.zeros(N_BEAT_FEATURES)

    # Load onset events for alignment
    onset_data = (
        preloaded.get("onsets") if preloaded
        else cache.load_onsets(audio_hash)
    )
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

    def _load_beats(tracker):
        if preloaded:
            return preloaded["beats"].get(tracker)
        return cache.load_beats(audio_hash, tracker)

    for t_idx, tracker_name in enumerate(TRACKER_NAMES):
        offset = t_idx * N_BEAT_FEATURES_PER_TRACKER

        if tracker_name == "madmom":
            # Try all BPB variants, pick best alignment
            best_data = None
            best_alignment = -1.0
            for bpb in [3, 4, 5, 7]:
                data = _load_beats(f"madmom_bpb{bpb}")
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
            data = _load_beats(tracker_name)
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


def extract_signal_scores(cache, audio_hash: str, preloaded=None) -> np.ndarray:
    """Extract signal score features from cache (60 dims).

    5 signals x 12 meter keys = 60 features.

    Args:
        preloaded: Optional dict from cache.load_all_for_audio() for batch reads.
    """
    feat = np.zeros(N_SIGNAL_FEATURES)

    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        cache_name = SIGNAL_CACHE_MAP[sig_name]
        cached = (
            preloaded["signals"].get(cache_name) if preloaded
            else cache.load_signal(audio_hash, cache_name)
        )
        if cached:
            for m_idx, meter_key in enumerate(METER_KEYS):
                feat[s_idx * N_METER_KEYS + m_idx] = cached.get(meter_key, 0.0)

    return feat


def extract_tempo_features(cache, audio_hash: str, preloaded=None) -> np.ndarray:
    """Extract tempo features from cache (4 dims).

    [0] tempo_librosa BPM (normalized /300)
    [1] tempo_tempogram BPM (normalized /300)
    [2] tempo ratio (lib / tg)
    [3] tempo agreement (1 - |lib - tg| / max)

    Args:
        preloaded: Optional dict from cache.load_all_for_audio() for batch reads.
    """
    feat = np.zeros(N_TEMPO_FEATURES)

    def _load_signal(name):
        if preloaded:
            return preloaded["signals"].get(name)
        return cache.load_signal(audio_hash, name)

    tempo_lib = 0.0
    tempo_tg = 0.0

    td = _load_signal("tempo_librosa")
    if td:
        tempo_lib = td.get("bpm", 0.0)

    td = _load_signal("tempo_tempogram")
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
# Tracker tempo extraction
# ---------------------------------------------------------------------------


def _ibi_to_bpm(beat_data: list[dict]) -> float:
    """Compute BPM from beat data IBI median. Returns 0.0 on failure."""
    if not beat_data or len(beat_data) < 3:
        return 0.0
    times = [b["time"] for b in beat_data]
    ibis = np.diff(times)
    valid = [x for x in ibis if 0.1 < x < 3.0]
    if len(valid) < 2:
        return 0.0
    return 60.0 / float(np.median(valid))


def _get_per_tracker_bpms(cache, audio_hash: str) -> dict[str, float]:
    """Get BPM from each cached tracker. Returns {name: bpm} (0.0 if unavailable)."""
    result = {}
    for tracker_name in TRACKER_NAMES:
        if tracker_name == "madmom":
            best_bpm = 0.0
            for bpb in [3, 4, 5, 7]:
                data = cache.load_beats(audio_hash, f"madmom_bpb{bpb}")
                bpm = _ibi_to_bpm(data) if data else 0.0
                if bpm > 0:
                    best_bpm = max(best_bpm, bpm)
            result[tracker_name] = best_bpm
        else:
            data = cache.load_beats(audio_hash, tracker_name)
            result[tracker_name] = _ibi_to_bpm(data) if data else 0.0
    return result


def _get_tempogram_bpm(cache, audio_hash: str) -> float:
    """Get tempogram BPM from cache. Returns 0.0 if unavailable."""
    td = cache.load_signal(audio_hash, "tempo_librosa")
    if td:
        bpm = td.get("bpm", 0.0)
        if bpm > 0:
            return bpm
    td = cache.load_signal(audio_hash, "tempo_tempogram")
    if td:
        bpm = td.get("bpm", 0.0)
        if bpm > 0:
            return bpm
    return 0.0


def _build_4_tempos(t_consensus: float, t_tempogram: float) -> list[float]:
    """Build 4 tempo candidates for v6.1 autocorrelation features.

    Layout (mirrors v5/v6 estimate_tempos_v5 but from beat trackers):
      [0] t_consensus        — median of tracker IBIs (most reliable)
      [1] t_tempogram        — independent librosa estimate
      [2] t_consensus / 2    — octave down
      [3] t_consensus × 1.5  — compound (6/8, 9/8, 12/8)
    """
    if t_tempogram <= 0:
        t_tempogram = t_consensus

    return [float(np.clip(t, 30.0, 300.0)) for t in [
        t_consensus,
        t_tempogram,
        t_consensus / 2,
        t_consensus * 1.5,
    ]]


def extract_tracker_tempos(cache, audio_hash: str) -> list[float]:
    """Extract 4 tempo candidates from cached beat trackers.

    Returns list of 4 floats for v6.1 autocorrelation features.
    See _build_4_tempos for layout.
    """
    per_tracker = _get_per_tracker_bpms(cache, audio_hash)
    valid_bpms = [v for v in per_tracker.values() if v > 0]
    t_consensus = float(np.median(valid_bpms)) if valid_bpms else 120.0
    t_tempogram = _get_tempogram_bpm(cache, audio_hash)

    return _build_4_tempos(t_consensus, t_tempogram)


def extract_tracker_tempos_live(
    beatnet_beats, beat_this_beats, madmom_results: dict,
    tempo_librosa_bpm: float = 0.0,
) -> list[float]:
    """Extract 4 tempo candidates from live beat data (for inference).

    Returns list of 4 floats for v6.1 autocorrelation features.
    See _build_4_tempos for layout.
    """
    def _beats_to_bpm(beats) -> float:
        if not beats or len(beats) < 3:
            return 0.0
        times = [b.time for b in beats]
        ibis = np.diff(times)
        valid = [x for x in ibis if 0.1 < x < 3.0]
        if len(valid) < 2:
            return 0.0
        return 60.0 / float(np.median(valid))

    t_beatnet = _beats_to_bpm(beatnet_beats)
    t_beat_this = _beats_to_bpm(beat_this_beats)

    t_madmom = 0.0
    if madmom_results:
        best_bpm = 0.0
        for bpb, beats in madmom_results.items():
            bpm = _beats_to_bpm(beats)
            if bpm > 0:
                best_bpm = max(best_bpm, bpm)
        t_madmom = best_bpm

    valid_bpms = [t for t in [t_beatnet, t_beat_this, t_madmom] if t > 0]
    t_consensus = float(np.median(valid_bpms)) if valid_bpms else 120.0

    return _build_4_tempos(t_consensus, tempo_librosa_bpm)


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
