"""Meter (time signature) hypothesis generation.

This is the heart of the project. Generates multiple meter hypotheses
with confidence scores by combining evidence from:
1. BeatNet downbeat spacing (primary - direct meter observation)
2. madmom RNN downbeat activation scoring (raw neural net output)
3. Onset autocorrelation (finds bar-level periodicity using detected tempo)
4. Accent pattern analysis (checks if onset strengths match hypothetical accent pattern)

Key design: we use the *detected tempo* (beat interval) to constrain
all meter hypotheses. Each signal scores "how well does beat_interval*N
match the observed bar-level periodicity?"
"""

import logging
import os
import tempfile
from collections import Counter

import numpy as np
import soundfile as sf

from beatmeter.analysis.models import Beat, MeterHypothesis

logger = logging.getLogger(__name__)

# Meter descriptions for musicians
METER_DESCRIPTIONS = {
    (2, 4): "Marsz, polka",
    (3, 4): "Walc (np. Blue Danube, Walc Chopina)",
    (3, 8): "Szybki walc, gigue",
    (4, 4): "Standardowy rock/pop (np. Billie Jean, Hey Jude)",
    (5, 4): "Take Five (Dave Brubeck) - grupowanie 3+2 lub 2+3",
    (5, 8): "Mission Impossible - grupowanie 3+2 lub 2+3",
    (6, 4): "Wolne 6 (rzadkie)",
    (6, 8): "Tarantella, Nothing Else Matters - dwie grupy po 3",
    (7, 4): "Rzadkie - np. Money (Pink Floyd)",
    (7, 8): "Money (Pink Floyd) - grupowanie 2+2+3 lub 3+2+2",
    (9, 8): "Blue Rondo à la Turk (Dave Brubeck) - 2+2+2+3",
    (10, 8): "Rzadkie - np. niektóre utwory Tool",
    (11, 8): "Bardzo rzadkie - np. I Hang on to a Dream (The Nice)",
    (12, 8): "Blues shuffle, Everybody Wants To Rule The World - cztery grupy po 3",
}

GROUPINGS = {
    (5, 4): ["3+2", "2+3"],
    (5, 8): ["3+2", "2+3"],
    (7, 4): ["2+2+3", "3+2+2", "2+3+2"],
    (7, 8): ["2+2+3", "3+2+2", "2+3+2"],
    (9, 8): ["2+2+2+3", "3+3+3", "3+2+2+2"],
    (10, 8): ["3+3+2+2", "2+3+3+2", "3+2+3+2"],
    (11, 8): ["3+3+3+2", "2+2+3+2+2", "3+2+3+3"],
}

# ---------------------------------------------------------------------------
# Tunable constants — all magic numbers extracted for easy adjustment.
# ---------------------------------------------------------------------------

# Trust calibration: NN signals are weighted by alignment quality.
# Trust = 0 when alignment < TRUST_LOWER, ramps to 1 at TRUST_UPPER.
TRUST_LOWER = 0.4
TRUST_RANGE = 0.4  # TRUST_UPPER - TRUST_LOWER (0.8 - 0.4)

# Signal weights (pre-normalization)
W_BEATNET = 0.13
W_BEAT_THIS = 0.16       # Beat This! gets slightly higher weight (SOTA)
W_MADMOM = 0.10
W_AUTOCORR = 0.13
W_ACCENT = 0.18
W_PERIODICITY = 0.20
W_PERIODICITY_CAPPED = 0.16  # used when all NNs untrusted
W_BAR_TRACKING = 0.12
W_RESNET = 0.0           # disabled pending better model

# Consensus bonus: meters supported by multiple signals
CONSENSUS_SUPPORT_THRESHOLD = 0.3   # min signal score to count as "supporting"
CONSENSUS_3_BONUS = 1.15            # 3 supporting signals
CONSENSUS_4_BONUS = 1.25            # 4+ supporting signals

# NN 3/4 penalty: when neural nets favor even meters
NN_PENALTY_STRONG = 0.55            # trusted NNs, no bar tracking conflict
NN_PENALTY_WEAK = 0.65              # untrusted NNs, no bar tracking conflict
NN_PENALTY_MILD = 0.80              # any NNs, but bar tracking supports 3/4

# Compound meter detection (signal_sub_beat_division)
COMPOUND_ONSET_MIN = 1.5
COMPOUND_ONSET_MAX = 3.5
COMPOUND_EVENNESS_CV_MAX = 0.4      # max CV for "evenly spaced" sub-beats
COMPOUND_TRIPLET_POS_MIN = 0.4      # min fraction of beats at triplet positions
COMPOUND_TRANSFER_RATIO = 0.5       # fraction of /4 score transferred to /8
COMPOUND_BOOST = 1.3                # boost for existing /8 scores

# 2/4 sub-period suppression
SUBPERIOD_4_4_THRESHOLD = 0.4       # min ratio of 4/4 to 2/4 score
SUBPERIOD_4_4_BOOST = 1.3
SUBPERIOD_3_4_THRESHOLD = 0.7       # min ratio of 3/4 to 2/4 score
SUBPERIOD_3_4_BOOST = 1.2

# Quality gates
BAR_TRACKING_SPARSE_THRESHOLD = 0.15  # min non-silent fraction for bar tracking
FLAT_DYNAMICS_CV = 0.05               # max CV to consider dynamics "flat"
NOISE_FLOOR_RATIO = 0.10              # min energy fraction to count beat as active
ACTIVE_BEATS_MIN_FRAC = 0.7           # min fraction of active beats per tracker

# Noise filter: remove hypotheses scoring less than this fraction of the best
NOISE_FILTER_RATIO = 0.10

# Rarity penalties for uncommon meters
RARITY_PENALTY = {5: 0.65, 7: 0.55, 8: 0.3, 9: 0.3, 10: 0.3, 11: 0.3, 12: 0.3}

# Prior: common meters get a slight boost (kept mild to avoid overwhelming signals)
METER_PRIOR = {
    (4, 4): 1.13,
    (3, 4): 1.05,
    (6, 8): 1.05,
    (2, 4): 1.05,
    (12, 8): 1.02,
}


def _get_description(num: int, den: int) -> str:
    return METER_DESCRIPTIONS.get((num, den), f"{num}/{den}")


def _get_grouping(num: int, den: int) -> str | None:
    groupings = GROUPINGS.get((num, den))
    return groupings[0] if groupings else None


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

        # Get onset strength at downbeat positions (max in ±50ms window)
        db_strengths = []
        for db in downbeats:
            window_mask = np.abs(onset_times - db.time) < 0.05
            if np.any(window_mask):
                db_strengths.append(float(np.max(onset_strengths[window_mask])))

        # Get onset strength at non-downbeat positions (max in ±50ms window)
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


def signal_onset_autocorrelation(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    beat_interval: float | None = None,
    sr: int = 22050,
) -> dict[tuple[int, int], float]:
    """Signal 3: Onset autocorrelation to find bar-level periodicity.

    Uses the detected tempo to check autocorrelation at expected bar lags.
    """
    scores: dict[tuple[int, int], float] = {}

    if len(onset_times) < 10 or beat_interval is None or beat_interval <= 0:
        return scores

    duration = float(onset_times[-1])
    if duration < 3:
        return scores

    # Regular-sampled onset strength signal (50 Hz)
    analysis_sr = 50
    n_samples = int(duration * analysis_sr)
    if n_samples < 50:
        return scores

    signal = np.zeros(n_samples)
    for t, s in zip(onset_times, onset_strengths):
        idx = int(t * analysis_sr)
        if 0 <= idx < n_samples:
            signal[idx] = max(signal[idx], s)

    signal = signal - np.mean(signal)
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return scores

    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]

    for beats_per_bar in [2, 3, 4, 5, 6, 7, 9, 11, 12]:
        # Quarter note based
        bar_dur = beat_interval * beats_per_bar
        lag = int(bar_dur * analysis_sr)

        if 2 < lag < len(autocorr) - 2:
            window = max(1, int(lag * 0.05))
            start = max(0, lag - window)
            end = min(len(autocorr), lag + window + 1)
            peak = float(np.max(autocorr[start:end]))
            if peak > 0.02:
                scores[(beats_per_bar, 4)] = peak

        # Eighth note based
        if beats_per_bar in (3, 5, 6, 7, 9, 11, 12):
            bar_dur_8 = (beat_interval / 2) * beats_per_bar
            lag_8 = int(bar_dur_8 * analysis_sr)
            if 2 < lag_8 < len(autocorr) - 2:
                window = max(1, int(lag_8 * 0.05))
                start = max(0, lag_8 - window)
                end = min(len(autocorr), lag_8 + window + 1)
                peak = float(np.max(autocorr[start:end]))
                if peak > 0.02:
                    scores[(beats_per_bar, 8)] = peak

    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores


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
        # Measure CV (std/mean) of inter-onset intervals within each beat.
        # Ornaments and accompaniment arpeggios produce unevenly spaced onsets.
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
            # True triplets: CV < 0.3 (evenly spaced)
            # Ornaments: CV > 0.5 (unevenly spaced)
            if median_cv > COMPOUND_EVENNESS_CV_MAX:
                logger.debug(f"Sub-beat intervals uneven (CV > {COMPOUND_EVENNESS_CV_MAX}): not compound")
                return 4

        # Positional consistency: true compound meter has sub-onsets at ~1/3
        # and ~2/3 of the beat interval. Accompaniment patterns (waltz "oom-pah",
        # polka chords) create onsets at different fractional positions.
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


# Module-level singleton for RNNBarProcessor (loads 12 models, ~2-3s startup)
_bar_processor = None


def _get_bar_processor():
    global _bar_processor
    if _bar_processor is None:
        from madmom.features.downbeats import RNNBarProcessor
        _bar_processor = RNNBarProcessor()
    return _bar_processor


def signal_bar_tracking(
    audio: np.ndarray,
    sr: int,
    beat_times_array: np.ndarray,
    meters_to_test: list[int] | None = None,
) -> dict[tuple[int, int], float]:
    """Signal 7: madmom DBNBarTrackingProcessor meter inference.

    Uses beat-synchronous harmonic+percussive features via GRU-RNN
    to estimate downbeat probability at each beat, then Viterbi decoding
    per candidate meter to determine most likely beats-per-bar.
    """
    if meters_to_test is None:
        meters_to_test = [3, 4, 5, 7]

    scores: dict[tuple[int, int], float] = {}
    if len(beat_times_array) < 6:
        return scores

    # Quality gate: skip on sparse/synthetic audio where RNNBarProcessor
    # features (harmonic+percussive) are meaningless.
    # Check that audio has enough non-silent content.
    rms = float(np.sqrt(np.mean(audio[:min(len(audio), sr * 5)] ** 2)))
    non_silent = float(np.mean(np.abs(audio) > rms * NOISE_FLOOR_RATIO))
    if non_silent < BAR_TRACKING_SPARSE_THRESHOLD:
        logger.debug(f"Bar tracking: sparse audio (non_silent={non_silent:.2f}), skipping")
        return scores

    try:
        from madmom.features.downbeats import DBNBarTrackingProcessor

        # RNNBarProcessor needs a file path — write temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)

        bar_proc = _get_bar_processor()
        act = bar_proc((tmp_path, beat_times_array))

        # Clean up temp file
        os.unlink(tmp_path)

        # Remove last row (often NaN) and take downbeat activation column
        activations = act[:-1, 1] if act.shape[0] > 1 else act[:, 1]

        if len(activations) < 4:
            return scores

        # Run Viterbi for each meter and collect normalized log-probs
        log_probs = {}
        for bpb in meters_to_test:
            tracker = DBNBarTrackingProcessor(beats_per_bar=[bpb])
            _, log_prob = tracker.hmm.viterbi(activations)
            log_probs[bpb] = log_prob / len(activations)

        # Convert to probabilities via softmax
        values = np.array(list(log_probs.values()))
        exp_values = np.exp(values - np.max(values))
        probs = exp_values / np.sum(exp_values)

        for bpb, prob in zip(meters_to_test, probs):
            if prob > 0.05:
                scores[(bpb, 4)] = float(prob)

        # Normalize to [0, 1]
        if scores:
            max_s = max(scores.values())
            if max_s > 0:
                scores = {k: v / max_s for k, v in scores.items()}

        logger.debug(f"Bar tracking signal: {scores}")
    except Exception as e:
        logger.warning(f"Bar tracking signal failed: {e}")

    return scores


def generate_hypotheses(
    beatnet_beats: list[Beat],
    madmom_results: dict[int, list[Beat]],
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    all_beats: list[Beat],
    sr: int = 22050,
    max_hypotheses: int = 5,
    tempo_bpm: float | None = None,
    beatnet_alignment: float = 1.0,
    madmom_alignment: float = 1.0,
    audio: np.ndarray | None = None,
    librosa_beats: list[Beat] | None = None,
    beat_this_beats: list[Beat] | None = None,
    beat_this_alignment: float = 0.0,
    onset_event_times: np.ndarray | None = None,
    skip_bar_tracking: bool = False,
    skip_resnet: bool = False,
) -> list[MeterHypothesis]:
    """Generate meter hypotheses from all signals.

    Merges and normalizes scores, returns top N with confidence.
    """
    # Beat interval from tempo
    beat_interval = None
    if tempo_bpm and tempo_bpm > 0:
        beat_interval = 60.0 / tempo_bpm
    elif len(all_beats) >= 3:
        times = np.array([b.time for b in all_beats])
        ibis = np.diff(times)
        if len(ibis) > 0:
            beat_interval = float(np.median(ibis))

    beat_times = np.array([b.time for b in all_beats]) if all_beats else np.array([])

    # Adaptive weights based on available data AND tracker reliability.
    # Neural net signals (BeatNet, madmom) are weighted by how well their
    # beats align with actual audio events. If alignment is poor (e.g.
    # hallucinated beats), these signals get zero weight.
    has_beatnet = len(beatnet_beats) > 0 and any(b.is_downbeat for b in beatnet_beats)
    has_beat_this = beat_this_beats is not None and len(beat_this_beats) > 0 and any(b.is_downbeat for b in beat_this_beats)
    has_madmom = len(madmom_results) > 0

    # Trust factor: 0 when alignment < TRUST_LOWER, ramps to 1 at TRUST_UPPER
    beatnet_trust = max(0.0, min(1.0, (beatnet_alignment - TRUST_LOWER) / TRUST_RANGE)) if has_beatnet else 0.0
    beat_this_trust = max(0.0, min(1.0, (beat_this_alignment - TRUST_LOWER) / TRUST_RANGE)) if has_beat_this else 0.0
    madmom_trust = max(0.0, min(1.0, (madmom_alignment - TRUST_LOWER) / TRUST_RANGE)) if has_madmom else 0.0

    # Signal 8: ResNet18 — check availability once
    _resnet_mod = None
    resnet_available = False
    if not skip_resnet and audio is not None:
        try:
            import beatmeter.analysis.resnet_signal as _resnet_mod
            resnet_available = True
        except ImportError:
            pass

    # Base weights with trust scaling
    w_beatnet = W_BEATNET * beatnet_trust
    w_beat_this = W_BEAT_THIS * beat_this_trust
    w_madmom = W_MADMOM * madmom_trust
    w_autocorr = W_AUTOCORR
    w_accent = W_ACCENT
    w_bar_tracking = W_BAR_TRACKING
    w_resnet = W_RESNET

    # Periodicity cap: when all NN trackers have zero trust, periodicity
    # naturally gets ~44% of total weight (0.24 / 0.55), which lets it force
    # 3/4 on duple music (marches, blues). Cap it to prevent dominance.
    total_nn_trust = beatnet_trust + beat_this_trust + madmom_trust
    if total_nn_trust < 0.01:
        w_periodicity = W_PERIODICITY_CAPPED
        logger.debug(f"Periodicity cap: all NNs untrusted, reducing {W_PERIODICITY}→{W_PERIODICITY_CAPPED}")
    else:
        w_periodicity = W_PERIODICITY

    # Normalize to sum to 1.0
    total_w = (w_beatnet + w_beat_this + w_madmom + w_autocorr
               + w_accent + w_periodicity + w_bar_tracking + w_resnet)
    if total_w > 0:
        weights = {
            "beatnet": w_beatnet / total_w,
            "beat_this": w_beat_this / total_w,
            "madmom": w_madmom / total_w,
            "autocorr": w_autocorr / total_w,
            "accent": w_accent / total_w,
            "periodicity": w_periodicity / total_w,
            "bar_tracking": w_bar_tracking / total_w,
            "resnet": w_resnet / total_w,
        }
    else:
        weights = {"beatnet": 0, "beat_this": 0, "madmom": 0, "autocorr": 0.2, "accent": 0.25, "periodicity": 0.35, "bar_tracking": 0.1, "resnet": 0.1}

    logger.debug(f"Meter weights: {weights}")

    # Collect per-signal score dicts for product-of-experts combination.
    # Each signal produces {meter: score} normalized to [0, 1].
    signal_results: dict[str, dict[tuple[int, int], float]] = {}

    # Signal 1a: BeatNet downbeat spacing
    if weights["beatnet"] > 0.01:
        s1 = signal_downbeat_spacing(beatnet_beats)
        if s1:
            signal_results["beatnet"] = s1
        logger.debug(f"BeatNet signal: {s1}")

    # Signal 1b: Beat This! downbeat spacing
    if weights["beat_this"] > 0.01 and beat_this_beats:
        s1b = signal_downbeat_spacing(beat_this_beats)
        if s1b:
            signal_results["beat_this"] = s1b
        logger.debug(f"Beat This! signal: {s1b}")

    # Signal 2: madmom activation scoring
    if weights["madmom"] > 0.01:
        s2 = signal_madmom_activation(madmom_results, onset_times, onset_strengths)
        if s2:
            signal_results["madmom"] = s2
        logger.debug(f"madmom signal: {s2}")

    # Signal 3: onset autocorrelation
    if weights["autocorr"] > 0 and len(onset_times) > 0:
        s3 = signal_onset_autocorrelation(onset_times, onset_strengths, beat_interval, sr)
        if s3:
            signal_results["autocorr"] = s3
        logger.debug(f"Autocorr signal: {s3}")

    # Signal 7: Bar tracking (DBNBarTrackingProcessor)
    if not skip_bar_tracking and weights["bar_tracking"] > 0.01 and audio is not None and len(beat_times) >= 6:
        # Use Beat This! beats if available (best tracker), else primary
        if beat_this_beats and len(beat_this_beats) >= 6:
            bar_beat_times = np.array([b.time for b in beat_this_beats])
        else:
            bar_beat_times = beat_times
        s7 = signal_bar_tracking(audio, sr, bar_beat_times)
        if s7:
            signal_results["bar_tracking"] = s7

    # Signal 8: ResNet18 MFCC classifier (orthogonal to beat tracking)
    if resnet_available and weights["resnet"] > 0.01 and audio is not None and _resnet_mod is not None:
        s8 = _resnet_mod.signal_resnet_meter(audio, sr)
        if s8:
            signal_results["resnet"] = s8
            logger.debug(f"ResNet signal: {s8}")

    # Signals 4 & 5: Multi-tracker accent analysis.
    # Run accent signals on ALL tracker candidates (not just primary) and take
    # the best RAW score per meter. This prevents a poorly-chosen primary from
    # producing wrong accent patterns. The tracker with the strongest accent
    # signal for each meter naturally dominates.
    accent_trackers: list[tuple[str, list[Beat]]] = []
    if all_beats and len(all_beats) >= 8:
        accent_trackers.append(('primary', all_beats))
    if beatnet_beats and len(beatnet_beats) >= 8:
        accent_trackers.append(('beatnet', beatnet_beats))
    if beat_this_beats and len(beat_this_beats) >= 8:
        accent_trackers.append(('beat_this', beat_this_beats))
    if librosa_beats and len(librosa_beats) >= 8:
        accent_trackers.append(('librosa', librosa_beats))
    for bpb, beats in madmom_results.items():
        if len(beats) >= 8:
            accent_trackers.append((f'madmom_{bpb}', beats))

    if audio is not None and accent_trackers:
        merged_accent: dict[tuple[int, int], float] = {}
        merged_periodicity: dict[tuple[int, int], float] = {}

        # Quality gate: if beat energy CV is very low (flat dynamics), signals
        # 4+5 pick up noise at arbitrary lags and produce spurious odd meters.
        # Compute CV across all tracker candidates; if ALL are flat, suppress.
        _all_cvs: list[float] = []
        for _name, _beats in accent_trackers:
            _energies = compute_beat_energies(_beats, audio, sr)
            if len(_energies) > 2:
                _mean = float(np.mean(_energies))
                if _mean > 0:
                    _all_cvs.append(float(np.std(_energies)) / _mean)
        _max_cv = max(_all_cvs) if _all_cvs else 0.0
        if _max_cv < FLAT_DYNAMICS_CV:
            logger.debug(f"Flat dynamics (max CV={_max_cv:.4f}): suppressing accent/periodicity signals")
            weights["accent"] = 0.0
            weights["periodicity"] = 0.0
            # Redistribute weight to other signals
            remaining_w = weights["beatnet"] + weights["beat_this"] + weights["madmom"] + weights["autocorr"] + weights["bar_tracking"] + weights["resnet"]
            if remaining_w > 0:
                scale = 1.0 / remaining_w
                weights["beatnet"] *= scale
                weights["beat_this"] *= scale
                weights["madmom"] *= scale
                weights["autocorr"] *= scale
                weights["bar_tracking"] *= scale
                weights["resnet"] *= scale

        for name, beats in accent_trackers:
            # Skip trackers whose tempo deviates >25% from consensus.
            # e.g. madmom_3 at 78.9 BPM on a 107 BPM track creates artificial
            # accent alignments that produce wrong 3/4.
            # Allow octave relationships (half/double speed).
            if tempo_bpm and tempo_bpm > 0 and len(beats) >= 3:
                _bt = np.array([b.time for b in beats])
                _ibis = np.diff(_bt)
                _valid_ibis = _ibis[(_ibis > 0.1) & (_ibis < 3.0)]
                if len(_valid_ibis) >= 2:
                    tracker_bpm = 60.0 / float(np.median(_valid_ibis))
                    # Normalize to the same octave as consensus tempo
                    ratio = tracker_bpm / tempo_bpm
                    while ratio > 1.4:
                        ratio /= 2.0
                    while ratio < 0.6:
                        ratio *= 2.0
                    if abs(ratio - 1.0) > 0.25:
                        logger.debug(f"Skipping {name}: BPM {tracker_bpm:.1f} deviates "
                                     f"from consensus {tempo_bpm:.1f} (ratio={ratio:.2f})")
                        continue

            energies = compute_beat_energies(beats, audio, sr)

            # Skip trackers where many beats land in silence (wrong tempo).
            # If a tracker is at the wrong BPM, some of its beats fall between
            # real audio events, giving near-zero energy. This creates artificial
            # accent patterns (e.g., madmom at 120 BPM on 160 BPM audio shows
            # spurious 3/4 because every 3rd madmom beat aligns with a click).
            max_e = float(np.max(energies)) if len(energies) > 0 else 0
            if max_e > 0:
                active_frac = float(np.sum(energies > max_e * NOISE_FLOOR_RATIO)) / len(energies)
                if active_frac < ACTIVE_BEATS_MIN_FRAC:
                    logger.debug(f"Skipping {name}: only {active_frac:.0%} beats have energy")
                    continue

            s4 = signal_accent_pattern(beats, energies, normalize=False)
            s5 = signal_beat_strength_periodicity(energies, normalize=False)
            for meter, score in s4.items():
                if score > merged_accent.get(meter, 0):
                    merged_accent[meter] = score
            for meter, score in s5.items():
                if score > merged_periodicity.get(meter, 0):
                    merged_periodicity[meter] = score
            logger.debug(f"Accent ({name}): s4={s4}, s5={s5}")

        # Normalize merged scores
        for scores_dict in [merged_accent, merged_periodicity]:
            if scores_dict:
                max_s = max(scores_dict.values())
                if max_s > 0:
                    for k in scores_dict:
                        scores_dict[k] /= max_s

        if weights["accent"] > 0 and merged_accent:
            signal_results["accent"] = merged_accent
            logger.debug(f"Merged accent signal: {merged_accent}")

        if weights["periodicity"] > 0 and merged_periodicity:
            signal_results["periodicity"] = merged_periodicity
            logger.debug(f"Merged periodicity signal: {merged_periodicity}")
    else:
        # Fallback: single-tracker analysis with onset envelope
        beat_energies = np.zeros(len(all_beats))
        for i, b in enumerate(all_beats):
            window_mask = np.abs(onset_times - b.time) < 0.05
            if np.any(window_mask):
                beat_energies[i] = float(np.max(onset_strengths[window_mask]))
            else:
                beat_energies[i] = float(np.mean(onset_strengths)) if len(onset_strengths) > 0 else 0.0

        if weights["accent"] > 0:
            s4 = signal_accent_pattern(all_beats, beat_energies)
            if s4:
                signal_results["accent"] = s4

        if weights["periodicity"] > 0:
            s5 = signal_beat_strength_periodicity(beat_energies)
            if s5:
                signal_results["periodicity"] = s5

    # -----------------------------------------------------------------------
    # Weighted additive combination with consensus bonus.
    # Each signal contributes weight * score. Meters supported by multiple
    # signals get a mild bonus to reward cross-signal agreement.
    # -----------------------------------------------------------------------
    all_candidate_meters: set[tuple[int, int]] = set()
    for sig_scores in signal_results.values():
        all_candidate_meters.update(sig_scores.keys())

    all_scores: dict[tuple[int, int], float] = {}
    for meter in all_candidate_meters:
        score = 0.0
        n_supporting = 0
        for sig_name, sig_scores in signal_results.items():
            w = weights.get(sig_name, 0)
            if w < 0.001:
                continue
            if meter in sig_scores:
                score += w * sig_scores[meter]
                if sig_scores[meter] > CONSENSUS_SUPPORT_THRESHOLD:
                    n_supporting += 1
        # Consensus bonus: meters supported by multiple signals get a boost
        if n_supporting >= 4:
            score *= CONSENSUS_4_BONUS
        elif n_supporting >= 3:
            score *= CONSENSUS_3_BONUS
        all_scores[meter] = score

    if not all_scores:
        return [MeterHypothesis(
            numerator=4, denominator=4, confidence=0.3,
            description=_get_description(4, 4),
        )]

    # Apply priors
    for meter, prior in METER_PRIOR.items():
        if meter in all_scores:
            all_scores[meter] *= prior

    # Suppress rare meters: odd/high meters are uncommon in most music.
    # Without this, noise in accent signals produces spurious 5/4, 7/4, etc.
    # 5/4 and 7/4 get mild penalties; 8+ get strong penalties.
    for num, penalty in RARITY_PENALTY.items():
        key = (num, 4)
        if key in all_scores:
            all_scores[key] *= penalty

    # 2/4 sub-period suppression: 2/4 is a sub-period of nearly any accent
    # structure. If 4/4 is also present and reasonably close, boost it.
    # Also boost 3/4 if it's already close to 2/4 (waltz detection).
    if (2, 4) in all_scores:
        s2 = all_scores[(2, 4)]
        if (4, 4) in all_scores and all_scores[(4, 4)] > s2 * SUBPERIOD_4_4_THRESHOLD:
            all_scores[(4, 4)] *= SUBPERIOD_4_4_BOOST
        if (3, 4) in all_scores and all_scores[(3, 4)] > s2 * SUBPERIOD_3_4_THRESHOLD:
            all_scores[(3, 4)] *= SUBPERIOD_3_4_BOOST

    # 6/8 vs 3/4 disambiguation
    _disambiguate_compound(all_scores, all_beats, beat_interval)

    # Sub-beat division analysis: detect /4 vs /8 denominator
    sub_beat_den = 4  # default: simple meter
    if onset_event_times is not None and len(onset_event_times) > 0:
        sub_beat_den = signal_sub_beat_division(all_beats, onset_event_times, sr)
        if sub_beat_den == 8:
            logger.debug("Sub-beat analysis: compound (/8) detected")
            # Boost /8 variants and convert /4 scores to /8 equivalents
            # 3/4 -> 6/8 (same bar duration, different subdivision)
            # 4/4 -> 12/8 (same bar duration, different subdivision)
            # 2/4 -> 6/8 (same bar duration, different subdivision)
            _compound_map = {
                (3, 4): (6, 8),
                (4, 4): (12, 8),
                (2, 4): (6, 8),
            }
            for simple, compound in _compound_map.items():
                if simple in all_scores:
                    # Transfer part of the simple meter score to compound
                    transfer = all_scores[simple] * COMPOUND_TRANSFER_RATIO
                    all_scores[compound] = all_scores.get(compound, 0) + transfer
            # Boost existing /8 scores
            for meter in list(all_scores.keys()):
                if meter[1] == 8:
                    all_scores[meter] *= COMPOUND_BOOST

    # 3/4 vs 4/4 disambiguation using neural net tracker evidence.
    # When neural net trackers report even-beat spacing (2 or 4) and NEITHER
    # reports 3, penalize 3/4. Works in two modes:
    #
    # 1) TRUSTED NNs (in signal_results): strong penalty (0.55)
    # 2) UNTRUSTED NNs: weak penalty (0.65) using raw downbeat spacing,
    #    only when compound was NOT detected (to protect 6/8).
    #
    # Gate: if Signal 7 (bar tracking) strongly supports 3/4, reduce or
    # skip penalty — bar tracking is independent of trust mechanism and
    # directly estimates meter via Viterbi decoding.
    bar_tracking_supports_triple = False
    if "bar_tracking" in signal_results:
        bt_score_3 = signal_results["bar_tracking"].get((3, 4), 0)
        if bt_score_3 > 0.5:
            bar_tracking_supports_triple = True
            logger.debug(f"Bar tracking supports 3/4 (score={bt_score_3:.2f})")

    if (3, 4) in all_scores and ((4, 4) in all_scores or (2, 4) in all_scores):
        nn_trackers_favor_even = 0
        nn_trackers_total = 0
        # Check trusted signals
        for sig_name in ["beatnet", "beat_this"]:
            if sig_name in signal_results:
                sig = signal_results[sig_name]
                nn_trackers_total += 1
                has_even = any(sig.get(m, 0) > 0.5 for m in [(2, 4), (4, 4)])
                has_triple = sig.get((3, 4), 0) > 0.3
                if has_even and not has_triple:
                    nn_trackers_favor_even += 1
        if nn_trackers_favor_even >= 1 and nn_trackers_favor_even == nn_trackers_total:
            if bar_tracking_supports_triple:
                # Conflict: trusted NNs say even but bar tracking says 3/4.
                # Apply mild penalty instead of strong — bar tracking is reliable.
                all_scores[(3, 4)] *= NN_PENALTY_MILD
                logger.debug(f"3/4 vs even: trusted NNs favor even but bar tracking says 3/4, mild penalty {NN_PENALTY_MILD}")
            else:
                all_scores[(3, 4)] *= NN_PENALTY_STRONG
                logger.debug(f"3/4 vs even: {nn_trackers_favor_even}/{nn_trackers_total} trusted NNs favor even, strong penalty")
        elif nn_trackers_total == 0 and sub_beat_den != 8:
            # No trusted NNs available — try untrusted raw downbeat spacing
            # as a weak signal. Only when compound was NOT detected.
            raw_nn_favor_even = 0
            raw_nn_total = 0
            for raw_beats, raw_name in [(beatnet_beats, "beatnet"), (beat_this_beats, "beat_this")]:
                if raw_beats and len(raw_beats) > 4 and any(b.is_downbeat for b in raw_beats):
                    raw_sig = signal_downbeat_spacing(raw_beats)
                    raw_nn_total += 1
                    even_score = max(raw_sig.get((2, 4), 0), raw_sig.get((4, 4), 0))
                    triple_score = raw_sig.get((3, 4), 0)
                    if even_score > 0.4 and triple_score < 0.3:
                        raw_nn_favor_even += 1
            if raw_nn_favor_even >= 1 and raw_nn_favor_even == raw_nn_total:
                if bar_tracking_supports_triple:
                    # Conflict: untrusted NNs say even but bar tracking says 3/4.
                    # Apply reduced penalty — bar tracking is somewhat reliable but
                    # can be fooled by syncopated music (ragtime, tango).
                    all_scores[(3, 4)] *= NN_PENALTY_MILD
                    logger.debug(f"3/4 vs even: untrusted NNs favor even but bar tracking says 3/4, reduced penalty {NN_PENALTY_MILD}")
                else:
                    all_scores[(3, 4)] *= NN_PENALTY_WEAK
                    logger.debug(f"3/4 vs even: {raw_nn_favor_even}/{raw_nn_total} untrusted NNs favor even, weak penalty")
            elif raw_nn_total == 0 and bar_tracking_supports_triple:
                # No NN data at all + bar tracking says 3/4: skip penalty.
                logger.debug("3/4 vs even: no NN data, bar tracking supports 3/4, skipping penalty")

    # Filter noise: remove hypotheses scoring less than 10% of the best
    max_raw = max(all_scores.values())
    if max_raw > 0:
        all_scores = {k: v for k, v in all_scores.items() if v >= max_raw * NOISE_FILTER_RATIO}

    # Normalize to probabilities
    total = sum(all_scores.values())
    if total > 0:
        all_scores = {k: v / total for k, v in all_scores.items()}

    # Sort, take top N, re-normalize
    sorted_meters = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_meters[:max_hypotheses]

    top_total = sum(s for _, s in top)
    if top_total > 0:
        top = [(m, s / top_total) for m, s in top]

    hypotheses = []
    meter_set = {m for m, _ in top}
    for (num, den), confidence in top:
        hint = _get_disambiguation_hint(num, den, meter_set, {m: s for m, s in top})
        hypotheses.append(MeterHypothesis(
            numerator=num,
            denominator=den,
            confidence=round(confidence, 3),
            grouping=_get_grouping(num, den),
            description=_get_description(num, den),
            disambiguation_hint=hint,
        ))

    return hypotheses


def _disambiguate_compound(
    scores: dict[tuple[int, int], float],
    beats: list[Beat],
    beat_interval: float | None,
):
    """Handle 6/8 vs 3/4 ambiguity."""
    if (3, 4) not in scores and (6, 8) not in scores:
        return
    if beat_interval and beat_interval > 0:
        if beat_interval > 0.6:
            if (6, 8) in scores:
                scores[(6, 8)] *= 1.2
        else:
            if (3, 4) in scores:
                scores[(3, 4)] *= 1.1


# Disambiguation hint keys for ambiguous meter pairs.
# The actual translated text lives in the frontend i18n module.
_DISAMBIGUATION_PAIRS = {
    ((6, 8), (3, 4)): "6_8_vs_3_4",
    ((3, 4), (6, 8)): "3_4_vs_6_8",
    ((4, 4), (2, 4)): "4_4_vs_2_4",
    ((2, 4), (4, 4)): "2_4_vs_4_4",
    ((12, 8), (4, 4)): "12_8_vs_4_4",
}


def _get_disambiguation_hint(
    num: int,
    den: int,
    all_meters: set[tuple[int, int]],
    scores: dict[tuple[int, int], float],
) -> str | None:
    """Return disambiguation hint key when ambiguous meters are close in confidence."""
    meter = (num, den)

    for (m1, m2), hint_key in _DISAMBIGUATION_PAIRS.items():
        if meter == m1 and m2 in all_meters:
            s1 = scores.get(m1, 0)
            s2 = scores.get(m2, 0)
            # Only show hint when scores are close (within 2x)
            if s2 > 0 and s1 / s2 < 2.0:
                return hint_key

    return None
