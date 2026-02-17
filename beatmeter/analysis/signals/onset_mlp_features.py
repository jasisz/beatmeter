"""Shared feature extraction for Onset MLP meter classification.

v4 features (876 dims):
    768 (autocorrelation: 3 tempos × 4 signals × 64 lags)
    + 64 (tempogram profile) + 26 (MFCC) + 14 (spectral contrast) + 4 (onset stats)

v5 features (1361 dims) = v4 + 485 new dims:
    +256 (4th tempo candidate × 4 signals × 64 lags)
    +160 (beat-position histograms: 5 bar lengths × 32 bins)
    +60  (autocorrelation ratios: 4 signals × 15 ratios)
    +9   (tempogram meter salience: 9 candidate meters)

Used by both scripts/training/train_onset_mlp.py and
beatmeter/analysis/signals/onset_mlp_meter.py.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR = 22050
HOP_LENGTH = 512
MAX_DURATION_S = 30
N_BEAT_FEATURES = 64
BEAT_RANGE = (0.5, 16.0)
WINDOW_PCT = 0.05
N_SIGNALS = 4
N_TEMPOGRAM_BINS = 64
N_MFCC = 13
N_CONTRAST_BANDS = 6
N_CONTRAST_DIMS = (N_CONTRAST_BANDS + 1) * 2  # 14
N_ONSET_STATS = 4

# v4 dims
N_TEMPO_CANDIDATES_V4 = 3
N_AUTOCORR_V4 = N_BEAT_FEATURES * N_TEMPO_CANDIDATES_V4 * N_SIGNALS  # 768
N_EXTRA_V4 = N_TEMPOGRAM_BINS + N_MFCC * 2 + N_CONTRAST_DIMS + N_ONSET_STATS  # 108
TOTAL_FEATURES_V4 = N_AUTOCORR_V4 + N_EXTRA_V4  # 876

# v5 new feature dims
N_TEMPO_CANDIDATES_V5 = 4
N_AUTOCORR_V5 = N_BEAT_FEATURES * N_TEMPO_CANDIDATES_V5 * N_SIGNALS  # 1024
N_BEAT_POSITION_BARS = 5  # bar lengths: 2, 3, 4, 5, 7
N_BEAT_POSITION_BINS = 32
N_BEAT_POSITION_FEATURES = N_BEAT_POSITION_BARS * N_BEAT_POSITION_BINS  # 160
N_AUTOCORR_RATIO_PER_SIGNAL = 15  # 6 absolute + 9 discriminative pairs
N_AUTOCORR_RATIO_FEATURES = N_SIGNALS * N_AUTOCORR_RATIO_PER_SIGNAL  # 60
N_TEMPOGRAM_SALIENCE = 9  # meters: 2, 3, 4, 5, 6, 7, 9, 11, 12
N_NEW_V5 = (N_AUTOCORR_V5 - N_AUTOCORR_V4) + N_BEAT_POSITION_FEATURES + N_AUTOCORR_RATIO_FEATURES + N_TEMPOGRAM_SALIENCE  # 485
TOTAL_FEATURES_V5 = TOTAL_FEATURES_V4 + N_NEW_V5  # 1361

FEATURE_VERSION_V4 = "v4"
FEATURE_VERSION_V5 = "v5"

# Bar lengths for beat-position histograms (in beats)
BAR_BEAT_LENGTHS = [2, 3, 4, 5, 7]

# Autocorrelation ratio config
RATIO_LAGS = [2, 3, 4, 5, 6, 7]  # absolute lags
RATIO_PAIRS = [
    (3, 4), (5, 4), (7, 4),  # vs 4
    (3, 2), (5, 3), (7, 3),  # vs smaller
    (5, 7), (9, 4), (11, 4),  # odd vs even
]

# Tempogram meter salience candidate meters
SALIENCE_METERS = [2, 3, 4, 5, 6, 7, 9, 11, 12]


# ---------------------------------------------------------------------------
# Core signal extraction (shared between v4 and v5)
# ---------------------------------------------------------------------------

def normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation of a 1-D signal."""
    if len(signal) < 10:
        return np.zeros(1)
    signal = signal - signal.mean()
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return np.zeros(len(signal))
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    return autocorr


def sample_autocorr_at_tempo(
    autocorr: np.ndarray, tempo_bpm: float, sr: int = SR
) -> np.ndarray:
    """Sample autocorrelation at beat-relative lags for a given tempo."""
    beat_period_frames = (60.0 / tempo_bpm) * (sr / HOP_LENGTH)
    beat_multiples = np.linspace(BEAT_RANGE[0], BEAT_RANGE[1], N_BEAT_FEATURES)
    features = np.zeros(N_BEAT_FEATURES)
    for i, k in enumerate(beat_multiples):
        lag = int(k * beat_period_frames)
        if 0 < lag < len(autocorr):
            window = max(1, int(lag * WINDOW_PCT))
            start = max(0, lag - window)
            end = min(len(autocorr), lag + window + 1)
            features[i] = float(np.max(autocorr[start:end]))
    return features


def compute_autocorrelations(y: np.ndarray, sr: int = SR, onset_env: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """Compute 4 autocorrelation signals: onset, RMS, spectral flux, chroma.

    Returns (list of 4 autocorrelation arrays, onset_env). Any autocorrelation
    may be short (len < 10) if the audio is too brief. onset_env is returned
    so callers can reuse it without recomputing.
    """
    import librosa

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    ac_onset = normalized_autocorrelation(onset_env)

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    ac_rms = normalized_autocorrelation(rms)

    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    ac_flux = normalized_autocorrelation(flux)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    chroma_energy = np.sum(chroma, axis=0)
    ac_chroma = normalized_autocorrelation(chroma_energy)

    return [ac_onset, ac_rms, ac_flux, ac_chroma], onset_env


def estimate_tempos_v4(y: np.ndarray, sr: int = SR) -> list[float]:
    """v4: 3 tempo candidates — T, T/2, T×2."""
    import librosa
    tempo = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)
    t = float(tempo[0]) if len(tempo) > 0 else 120.0
    if t < 30 or t > 300:
        t = 120.0
    return [t, max(30.0, t / 2), min(300.0, t * 2)]


def _pad_to_4_tempos(t1: float, y: np.ndarray | None = None, sr: int = SR) -> list[float]:
    """Ensure exactly 4 tempo candidates from a primary tempo."""
    import librosa
    if y is not None:
        t_lib = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)
        t2 = float(t_lib[0]) if len(t_lib) > 0 else t1
        if t2 < 30 or t2 > 300:
            t2 = t1
    else:
        t2 = t1
    return [
        t1,
        t2,
        max(30.0, t1 / 2),
        float(np.clip(t1 * 3 / 2, 30.0, 300.0)),
    ]


def estimate_tempos_v5(y: np.ndarray, sr: int = SR, avg_tg: np.ndarray | None = None, onset_env: np.ndarray | None = None) -> list[float]:
    """v5: 4 tempo candidates — top-2 tempogram peaks + T/2 of primary + T×3/2 (compound).

    Uses scipy.signal.find_peaks on the averaged tempogram to find the
    two strongest tempo peaks. Falls back to librosa.feature.tempo if
    tempogram analysis fails.

    If avg_tg is provided, skips tempogram computation.
    """
    import librosa
    from scipy.signal import find_peaks

    if avg_tg is None:
        if onset_env is None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        avg_tg = tg.mean(axis=1)

    # Tempogram lags → BPM conversion
    # lag index i corresponds to period of i frames = i * HOP_LENGTH / sr seconds
    # BPM = 60 / (period in seconds)
    # We want peaks in meaningful BPM range (30–300)
    min_lag = max(1, int((60.0 / 300.0) * (sr / HOP_LENGTH)))  # 300 BPM
    max_lag = min(len(avg_tg) - 1, int((60.0 / 30.0) * (sr / HOP_LENGTH)))  # 30 BPM

    if max_lag <= min_lag:
        return _pad_to_4_tempos(120.0, y, sr)

    # Find peaks in the valid range
    tg_slice = avg_tg[min_lag:max_lag + 1]
    peaks, properties = find_peaks(tg_slice, height=0, distance=3)

    if len(peaks) < 1:
        return _pad_to_4_tempos(120.0, y, sr)

    # Sort by height (descending), take top 2
    heights = properties["peak_heights"]
    sorted_idx = np.argsort(heights)[::-1]
    top_peaks = peaks[sorted_idx[:2]]

    # Convert lag indices back to BPM
    tempos = []
    for p in top_peaks:
        lag = p + min_lag
        if lag > 0:
            bpm = 60.0 * (sr / HOP_LENGTH) / lag
            tempos.append(np.clip(bpm, 30.0, 300.0))

    if not tempos:
        return _pad_to_4_tempos(120.0, y, sr)

    t1 = tempos[0]

    # Second tempo: either second tempogram peak or librosa fallback
    if len(tempos) >= 2:
        t2 = tempos[1]
    else:
        t_lib = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)
        t2 = float(t_lib[0]) if len(t_lib) > 0 else 120.0
        if t2 < 30 or t2 > 300:
            t2 = 120.0

    return [
        t1,
        t2,
        max(30.0, t1 / 2),                    # sub-harmonic
        float(np.clip(t1 * 3 / 2, 30.0, 300.0)),  # compound (6/8)
    ]


# ---------------------------------------------------------------------------
# Tempo-independent feature extraction (shared v4/v5)
# ---------------------------------------------------------------------------

def tempogram_profile(y: np.ndarray, sr: int = SR, avg_tg: np.ndarray | None = None, onset_env: np.ndarray | None = None) -> np.ndarray:
    """Averaged tempogram profile at log-spaced BPMs (64 dims)."""
    if avg_tg is None:
        import librosa
        if onset_env is None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        avg_tg = tg.mean(axis=1)
    bpm_bins = np.logspace(np.log10(30), np.log10(300), N_TEMPOGRAM_BINS)
    profile = np.zeros(N_TEMPOGRAM_BINS)
    for i, bpm in enumerate(bpm_bins):
        lag = int((60.0 / bpm) * (sr / HOP_LENGTH))
        if 0 < lag < len(avg_tg):
            w = max(1, lag // 20)
            start = max(0, lag - w)
            end = min(len(avg_tg), lag + w + 1)
            profile[i] = float(np.max(avg_tg[start:end]))
    pmax = profile.max()
    if pmax > 1e-10:
        profile /= pmax
    return profile


def mfcc_statistics(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Mean and std of MFCC coefficients (26 dims)."""
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def spectral_contrast_statistics(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Mean and std of spectral contrast (14 dims)."""
    import librosa
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_bands=N_CONTRAST_BANDS,
    )
    return np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])


def onset_rate_statistics(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Onset timing statistics (4 dims)."""
    import librosa
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH)
    if len(onset_frames) < 3:
        return np.zeros(N_ONSET_STATS)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    intervals = np.diff(onset_times)
    if len(intervals) == 0:
        return np.zeros(N_ONSET_STATS)
    duration = len(y) / sr
    return np.array([
        float(np.mean(intervals)),
        float(np.std(intervals)),
        float(np.median(intervals)),
        float(len(onset_frames) / max(duration, 1.0)),
    ])


# ---------------------------------------------------------------------------
# v5-only feature groups
# ---------------------------------------------------------------------------

def beat_position_histograms(
    y: np.ndarray, sr: int = SR, primary_tempo: float | None = None,
    onset_env: np.ndarray | None = None,
) -> np.ndarray:
    """Beat-position histograms for 5 bar lengths (160 dims).

    For each hypothetical bar length (2, 3, 4, 5, 7 beats), fold onset
    times modulo bar_duration and build a histogram weighted by onset
    strength. This captures accent patterns within a bar.

    Args:
        y: Audio signal.
        sr: Sample rate.
        primary_tempo: Primary tempo in BPM. If None, estimated from audio.
        onset_env: Pre-computed onset envelope. If None, computed from y.

    Returns:
        160-dim feature vector (5 bar lengths × 32 bins).
    """
    import librosa

    if primary_tempo is None:
        tempo = librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)
        primary_tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        if primary_tempo < 30 or primary_tempo > 300:
            primary_tempo = 120.0

    beat_duration = 60.0 / primary_tempo  # seconds per beat

    # Get onset times and strengths
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP_LENGTH, onset_envelope=onset_env,
    )
    if len(onset_frames) < 3:
        return np.zeros(N_BEAT_POSITION_BARS * N_BEAT_POSITION_BINS)

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    # Onset strengths at detected frames
    onset_strengths = onset_env[onset_frames]

    features = []
    for n_beats in BAR_BEAT_LENGTHS:
        bar_duration = beat_duration * n_beats
        if bar_duration < 0.1:
            features.append(np.zeros(N_BEAT_POSITION_BINS))
            continue

        # Fold onset times modulo bar_duration → position in [0, 1)
        positions = (onset_times % bar_duration) / bar_duration

        # Weighted histogram
        hist, _ = np.histogram(
            positions,
            bins=N_BEAT_POSITION_BINS,
            range=(0.0, 1.0),
            weights=onset_strengths,
        )
        # Normalize
        hmax = hist.max()
        if hmax > 1e-10:
            hist = hist / hmax
        features.append(hist.astype(np.float64))

    return np.concatenate(features)


def autocorrelation_ratios(autocorrs: list[np.ndarray], sr: int = SR, primary_tempo: float = 120.0) -> np.ndarray:
    """Autocorrelation ratios for meter discrimination (60 dims).

    For each of the 4 autocorrelation signals, compute:
    - 6 absolute peak values at lag = 2, 3, 4, 5, 6, 7 beats
    - 9 discriminative ratios: (3 vs 4), (5 vs 4), (7 vs 4), etc.

    This explicitly encodes "is peak@5 > peak@4?" which the model
    otherwise has to learn from raw autocorrelation values.

    Returns:
        60-dim feature vector (4 signals × 15 ratios).
    """
    beat_period_frames = (60.0 / primary_tempo) * (sr / HOP_LENGTH)
    features = []

    for ac in autocorrs:
        # 6 absolute values at integer-beat lags
        abs_vals = {}
        for n in RATIO_LAGS:
            lag = int(n * beat_period_frames)
            if 0 < lag < len(ac):
                window = max(1, int(lag * WINDOW_PCT))
                start = max(0, lag - window)
                end = min(len(ac), lag + window + 1)
                abs_vals[n] = float(np.max(ac[start:end]))
            else:
                abs_vals[n] = 0.0

        signal_feats = [abs_vals[n] for n in RATIO_LAGS]  # 6 values

        # 9 discriminative ratio pairs
        for a, b in RATIO_PAIRS:
            va = abs_vals.get(a, 0.0)
            vb = abs_vals.get(b, 0.0)
            # Ratio normalized to [-1, 1]: (a - b) / (a + b + eps)
            signal_feats.append((va - vb) / (va + vb + 1e-8))

        features.append(np.array(signal_feats))

    return np.concatenate(features)


def tempogram_meter_salience(y: np.ndarray, sr: int = SR, avg_tg: np.ndarray | None = None, onset_env: np.ndarray | None = None) -> np.ndarray:
    """Tempogram meter salience for 9 candidate meters (9 dims).

    For each candidate meter (2, 3, 4, 5, 6, 7, 9, 11, 12), find the
    peak strength in the tempogram at the corresponding bar-level BPM.
    This is tempo-independent: it doesn't require a tempo estimate.

    Returns:
        9-dim feature vector.
    """
    if avg_tg is None:
        import librosa
        if onset_env is None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        avg_tg = tg.mean(axis=1)

    # For each candidate meter N, we look for the strongest peak that
    # could correspond to a bar of N beats. We scan a range of tempos
    # and for each, compute bar_bpm = tempo / N. Take max across tempos.
    features = np.zeros(N_TEMPOGRAM_SALIENCE)

    # Scan tempo range 60-200 BPM in steps of 2
    tempo_range = np.arange(60, 201, 2)

    for i, meter in enumerate(SALIENCE_METERS):
        best = 0.0
        for tempo in tempo_range:
            bar_bpm = tempo / meter  # bars per minute
            bar_period_s = 60.0 / bar_bpm  # seconds per bar
            lag = int(bar_period_s * sr / HOP_LENGTH)
            if 0 < lag < len(avg_tg):
                w = max(1, lag // 20)
                start = max(0, lag - w)
                end = min(len(avg_tg), lag + w + 1)
                val = float(np.max(avg_tg[start:end]))
                if val > best:
                    best = val
        features[i] = best

    # Normalize
    fmax = features.max()
    if fmax > 1e-10:
        features /= fmax

    return features


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def extract_features_v4(y: np.ndarray, sr: int = SR) -> np.ndarray | None:
    """Extract v4 features (876 dims) from pre-loaded audio array.

    Returns None on failure.
    """
    try:
        import librosa

        if len(y) < sr * 2:
            return None

        # Compute shared signals once
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        autocorrs, onset_env = compute_autocorrelations(y, sr, onset_env=onset_env)
        if any(len(ac) < 10 for ac in autocorrs):
            return None

        tempos = estimate_tempos_v4(y, sr)

        parts = []
        for t in tempos:
            for ac in autocorrs:
                parts.append(sample_autocorr_at_tempo(ac, t, sr))

        parts.append(tempogram_profile(y, sr, onset_env=onset_env))
        parts.append(mfcc_statistics(y, sr))
        parts.append(spectral_contrast_statistics(y, sr))
        parts.append(onset_rate_statistics(y, sr))

        feat = np.concatenate(parts)
        assert feat.shape[0] == TOTAL_FEATURES_V4, f"v4 dim mismatch: {feat.shape[0]} != {TOTAL_FEATURES_V4}"
        return feat

    except Exception as e:
        logger.warning("v4 feature extraction failed: %s", e)
        return None


def extract_features_v5(y: np.ndarray, sr: int = SR) -> np.ndarray | None:
    """Extract v5 features (1361 dims) from pre-loaded audio array.

    v5 = v4 base + 4th tempo candidate + beat-position histograms
    + autocorrelation ratios + tempogram meter salience.

    Computes onset_env and tempogram only once (saves ~50% time vs naive).

    Returns None on failure.
    """
    try:
        import librosa

        if len(y) < sr * 2:
            return None

        # === Compute shared signals ONCE ===
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        avg_tg = tg.mean(axis=1)

        autocorrs, onset_env = compute_autocorrelations(y, sr, onset_env=onset_env)
        if any(len(ac) < 10 for ac in autocorrs):
            return None

        tempos_v5 = estimate_tempos_v5(y, sr, avg_tg=avg_tg)
        primary_tempo = tempos_v5[0]

        # Part 1: Autocorrelation features at 4 tempos (1024 dims)
        parts = []
        for t in tempos_v5:
            for ac in autocorrs:
                parts.append(sample_autocorr_at_tempo(ac, t, sr))

        # Part 2: v4 tempo-independent features (108 dims)
        parts.append(tempogram_profile(y, sr, avg_tg=avg_tg))
        parts.append(mfcc_statistics(y, sr))
        parts.append(spectral_contrast_statistics(y, sr))
        parts.append(onset_rate_statistics(y, sr))

        # Part 3: v5 new features (229 dims)
        parts.append(beat_position_histograms(y, sr, primary_tempo, onset_env=onset_env))  # 160
        parts.append(autocorrelation_ratios(autocorrs, sr, primary_tempo))  # 60
        parts.append(tempogram_meter_salience(y, sr, avg_tg=avg_tg))  # 9

        feat = np.concatenate(parts)
        assert feat.shape[0] == TOTAL_FEATURES_V5, f"v5 dim mismatch: {feat.shape[0]} != {TOTAL_FEATURES_V5}"
        return feat

    except Exception as e:
        logger.warning("v5 feature extraction failed: %s", e)
        return None


def extract_features_from_path(audio_path, version: str = "v5") -> np.ndarray | None:
    """Extract features from an audio file path.

    Handles loading, duration limiting, and resampling.
    """
    import librosa
    from pathlib import Path

    try:
        y, sr = librosa.load(str(audio_path), sr=SR, duration=MAX_DURATION_S, mono=True)
    except Exception:
        return None

    if version == "v4":
        return extract_features_v4(y, sr)
    return extract_features_v5(y, sr)
