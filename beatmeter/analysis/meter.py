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

Signal functions live in beatmeter.analysis.signals/ — this module handles
scoring, weights, priors, and hypothesis generation.
"""

import logging
import math

import numpy as np

from beatmeter.analysis.models import Beat, MeterHypothesis
from beatmeter.analysis.signals import (
    signal_downbeat_spacing,
    signal_madmom_activation,
    signal_onset_autocorrelation,
    signal_accent_pattern,
    compute_beat_energies,
    signal_beat_strength_periodicity,
    signal_bar_tracking,
    signal_sub_beat_division,
)
import beatmeter.analysis.signals.resnet_meter as _resnet_mod
import beatmeter.analysis.signals.mert_meter as _mert_mod
import beatmeter.analysis.signals.onset_mlp_meter as _onset_mlp_mod

logger = logging.getLogger(__name__)

# Module-level capture for arbiter training.
# After generate_hypotheses() runs, this holds the last signal details.
last_signal_details: dict | None = None

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
W_MERT = 0.0             # disabled until orthogonality verified
W_ONSET_MLP = 0.12       # onset MLP signal (gate PASS: complementarity 2.68)

# Consensus bonus: meters supported by multiple signals
CONSENSUS_SUPPORT_THRESHOLD = 0.3   # min signal score to count as "supporting"
CONSENSUS_3_BONUS = 1.15            # 3 supporting signals
CONSENSUS_4_BONUS = 1.25            # 4+ supporting signals

# NN 3/4 penalty: when neural nets favor even meters
NN_PENALTY_STRONG = 0.55            # trusted NNs, no bar tracking conflict
NN_PENALTY_WEAK = 0.65              # untrusted NNs, no bar tracking conflict
NN_PENALTY_MILD = 0.80              # any NNs, but bar tracking supports 3/4

# Compound meter transfer/boost (disabled: 0 benefit, 9 false 6/8 regressions)
COMPOUND_TRANSFER_RATIO = 0.0
COMPOUND_BOOST = 1.0

# 2/4 sub-period suppression
SUBPERIOD_4_4_THRESHOLD = 0.4       # min ratio of 4/4 to 2/4 score
SUBPERIOD_4_4_BOOST = 1.3
SUBPERIOD_3_4_THRESHOLD = 0.7       # min ratio of 3/4 to 2/4 score
SUBPERIOD_3_4_BOOST = 1.2

# Quality gates
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


def _compute_weights(
    beatnet_beats: list[Beat],
    madmom_results: dict[int, list[Beat]],
    beat_this_beats: list[Beat] | None,
    beatnet_alignment: float,
    beat_this_alignment: float,
    madmom_alignment: float,
) -> tuple[dict[str, float], float, float, float]:
    """Compute normalized signal weights based on tracker trust.

    Returns (weights_dict, beatnet_trust, beat_this_trust, madmom_trust).
    """
    has_beatnet = len(beatnet_beats) > 0 and any(b.is_downbeat for b in beatnet_beats)
    has_beat_this = beat_this_beats is not None and len(beat_this_beats) > 0 and any(b.is_downbeat for b in beat_this_beats)
    has_madmom = len(madmom_results) > 0

    beatnet_trust = max(0.0, min(1.0, (beatnet_alignment - TRUST_LOWER) / TRUST_RANGE)) if has_beatnet else 0.0
    beat_this_trust = max(0.0, min(1.0, (beat_this_alignment - TRUST_LOWER) / TRUST_RANGE)) if has_beat_this else 0.0
    madmom_trust = max(0.0, min(1.0, (madmom_alignment - TRUST_LOWER) / TRUST_RANGE)) if has_madmom else 0.0

    w_beatnet = W_BEATNET * beatnet_trust
    w_beat_this = W_BEAT_THIS * beat_this_trust
    w_madmom = W_MADMOM * madmom_trust
    w_autocorr = W_AUTOCORR
    w_accent = W_ACCENT
    w_bar_tracking = W_BAR_TRACKING
    w_resnet = W_RESNET
    w_mert = W_MERT
    w_onset_mlp = W_ONSET_MLP

    total_nn_trust = beatnet_trust + beat_this_trust + madmom_trust
    if total_nn_trust < 0.01:
        w_periodicity = W_PERIODICITY_CAPPED
        logger.debug(f"Periodicity cap: all NNs untrusted, reducing {W_PERIODICITY}→{W_PERIODICITY_CAPPED}")
    else:
        w_periodicity = W_PERIODICITY

    total_w = (w_beatnet + w_beat_this + w_madmom + w_autocorr
               + w_accent + w_periodicity + w_bar_tracking + w_resnet + w_mert
               + w_onset_mlp)
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
            "mert": w_mert / total_w,
            "onset_mlp": w_onset_mlp / total_w,
        }
    else:
        weights = {"beatnet": 0, "beat_this": 0, "madmom": 0, "autocorr": 0.2, "accent": 0.25, "periodicity": 0.35, "bar_tracking": 0.1, "resnet": 0.1, "mert": 0, "onset_mlp": 0}

    logger.debug(f"Meter weights: {weights}")
    return weights, beatnet_trust, beat_this_trust, madmom_trust


def _collect_signal_scores(
    weights: dict[str, float],
    beatnet_beats: list[Beat],
    madmom_results: dict[int, list[Beat]],
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    beat_interval: float | None,
    beat_times: np.ndarray,
    sr: int,
    audio: np.ndarray | None,
    beat_this_beats: list[Beat] | None,
    skip_bar_tracking: bool,
    skip_resnet: bool,
    skip_mert: bool = False,
    skip_onset_mlp: bool = False,
    cache=None,
    audio_hash: str | None = None,
    tmp_path: str | None = None,
) -> dict[str, dict[tuple[int, int], float]]:
    """Collect scores from signals 1a, 1b, 2, 3, 7, 8a, 8b, 9."""
    from beatmeter.analysis.cache import str_to_meter_key

    signal_results: dict[str, dict[tuple[int, int], float]] = {}

    def _try_cache(sig_name: str) -> dict[tuple[int, int], float] | None:
        if cache and audio_hash:
            raw = cache.load_signal(audio_hash, sig_name)
            if raw is not None:
                return {str_to_meter_key(k): v for k, v in raw.items()}
        return None

    def _save_cache(sig_name: str, scores: dict[tuple[int, int], float]) -> None:
        if cache and audio_hash and scores:
            cache.save_signal(audio_hash, sig_name, scores)

    # Signal 1a: BeatNet downbeat spacing
    if weights["beatnet"] > 0.01:
        s1 = _try_cache("beatnet_spacing")
        if s1 is None:
            s1 = signal_downbeat_spacing(beatnet_beats)
            _save_cache("beatnet_spacing", s1)
        if s1:
            signal_results["beatnet"] = s1
        logger.debug(f"BeatNet signal: {s1}")

    # Signal 1b: Beat This! downbeat spacing
    if weights["beat_this"] > 0.01 and beat_this_beats:
        s1b = _try_cache("beat_this_spacing")
        if s1b is None:
            s1b = signal_downbeat_spacing(beat_this_beats)
            _save_cache("beat_this_spacing", s1b)
        if s1b:
            signal_results["beat_this"] = s1b
        logger.debug(f"Beat This! signal: {s1b}")

    # Signal 2: madmom activation scoring
    if weights["madmom"] > 0.01:
        s2 = _try_cache("madmom_activation")
        if s2 is None:
            s2 = signal_madmom_activation(madmom_results, onset_times, onset_strengths)
            _save_cache("madmom_activation", s2)
        if s2:
            signal_results["madmom"] = s2
        logger.debug(f"madmom signal: {s2}")

    # Signal 3: onset autocorrelation
    if weights["autocorr"] > 0 and len(onset_times) > 0:
        s3 = _try_cache("onset_autocorr")
        if s3 is None:
            s3 = signal_onset_autocorrelation(onset_times, onset_strengths, beat_interval, sr)
            _save_cache("onset_autocorr", s3)
        if s3:
            signal_results["autocorr"] = s3
        logger.debug(f"Autocorr signal: {s3}")

    # Signal 7: Bar tracking (DBNBarTrackingProcessor)
    if not skip_bar_tracking and weights["bar_tracking"] > 0.01 and audio is not None and len(beat_times) >= 6:
        s7 = _try_cache("bar_tracking")
        if s7 is None:
            if beat_this_beats and len(beat_this_beats) >= 6:
                bar_beat_times = np.array([b.time for b in beat_this_beats])
            else:
                bar_beat_times = beat_times
            s7 = signal_bar_tracking(audio, sr, bar_beat_times, tmp_path=tmp_path)
            _save_cache("bar_tracking", s7)
        if s7:
            signal_results["bar_tracking"] = s7

    # Signal 8a: ResNet18 MFCC classifier
    if not skip_resnet and weights["resnet"] > 0.01 and audio is not None:
        s8 = _try_cache("resnet_meter")
        if s8 is None:
            s8 = _resnet_mod.signal_resnet_meter(audio, sr)
            _save_cache("resnet_meter", s8)
        if s8:
            signal_results["resnet"] = s8
            logger.debug(f"ResNet signal: {s8}")

    # Signal 8b: MERT meter classifier
    if not skip_mert and weights.get("mert", 0) > 0.01 and audio is not None:
        s8b = _try_cache("mert_meter")
        if s8b is None:
            s8b = _mert_mod.signal_mert_meter(audio, sr)
            _save_cache("mert_meter", s8b)
        if s8b:
            signal_results["mert"] = s8b
            logger.debug(f"MERT signal: {s8b}")

    # Signal 9: Onset MLP meter classifier
    if not skip_onset_mlp and weights.get("onset_mlp", 0) > 0.01 and audio is not None:
        s9 = _try_cache("onset_mlp_meter")
        if s9 is None:
            s9 = _onset_mlp_mod.signal_onset_mlp_meter(audio, sr)
            _save_cache("onset_mlp_meter", s9)
        if s9:
            signal_results["onset_mlp"] = s9
            logger.debug(f"Onset MLP signal: {s9}")

    return signal_results


def _collect_accent_scores(
    weights: dict[str, float],
    signal_results: dict[str, dict[tuple[int, int], float]],
    all_beats: list[Beat],
    beatnet_beats: list[Beat],
    beat_this_beats: list[Beat] | None,
    librosa_beats: list[Beat] | None,
    madmom_results: dict[int, list[Beat]],
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    audio: np.ndarray | None,
    sr: int,
    tempo_bpm: float | None,
    cache=None,
    audio_hash: str | None = None,
) -> None:
    """Collect accent (signal 4) and periodicity (signal 5) scores.

    Mutates weights and signal_results in place.
    """
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
        # Try loading cached accent/periodicity signals
        from beatmeter.analysis.cache import str_to_meter_key as _s2mk
        if cache and audio_hash:
            cached_s4 = cache.load_signal(audio_hash, "accent_pattern")
            cached_s5 = cache.load_signal(audio_hash, "beat_periodicity")
            if cached_s4 is not None and cached_s5 is not None:
                if cached_s4:
                    signal_results["accent"] = {_s2mk(k): v for k, v in cached_s4.items()}
                if cached_s5:
                    signal_results["periodicity"] = {_s2mk(k): v for k, v in cached_s5.items()}
                return

        merged_accent: dict[tuple[int, int], float] = {}
        merged_periodicity: dict[tuple[int, int], float] = {}

        # Quality gate: if beat energy CV is very low (flat dynamics), signals
        # 4+5 pick up noise at arbitrary lags and produce spurious odd meters.
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
            remaining_w = weights["beatnet"] + weights["beat_this"] + weights["madmom"] + weights["autocorr"] + weights["bar_tracking"] + weights["resnet"] + weights.get("mert", 0) + weights.get("onset_mlp", 0)
            if remaining_w > 0:
                scale = 1.0 / remaining_w
                weights["beatnet"] *= scale
                weights["beat_this"] *= scale
                weights["madmom"] *= scale
                weights["autocorr"] *= scale
                weights["bar_tracking"] *= scale
                weights["resnet"] *= scale
                if "mert" in weights:
                    weights["mert"] *= scale
                if "onset_mlp" in weights:
                    weights["onset_mlp"] *= scale

        for name, beats in accent_trackers:
            # Skip trackers whose tempo deviates >25% from consensus.
            if tempo_bpm and tempo_bpm > 0 and len(beats) >= 3:
                _bt = np.array([b.time for b in beats])
                _ibis = np.diff(_bt)
                _valid_ibis = _ibis[(_ibis > 0.1) & (_ibis < 3.0)]
                if len(_valid_ibis) >= 2:
                    tracker_bpm = 60.0 / float(np.median(_valid_ibis))
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

        # Save to cache
        if cache and audio_hash:
            cache.save_signal(audio_hash, "accent_pattern", merged_accent)
            cache.save_signal(audio_hash, "beat_periodicity", merged_periodicity)

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


def _apply_score_adjustments(
    all_scores: dict[tuple[int, int], float],
    signal_results: dict[str, dict[tuple[int, int], float]],
    all_beats: list[Beat],
    beat_interval: float | None,
    onset_event_times: np.ndarray | None,
    sr: int,
    beatnet_beats: list[Beat],
    beat_this_beats: list[Beat] | None,
) -> int:
    """Apply priors, rarity penalties, compound meter, and NN 3/4 penalty.

    Mutates all_scores in place. Returns detected sub-beat denominator (4 or 8).
    """
    # Apply priors
    for meter, prior in METER_PRIOR.items():
        if meter in all_scores:
            all_scores[meter] *= prior

    # Suppress rare meters
    for num, penalty in RARITY_PENALTY.items():
        key = (num, 4)
        if key in all_scores:
            all_scores[key] *= penalty

    # 2/4 sub-period suppression
    if (2, 4) in all_scores:
        s2 = all_scores[(2, 4)]
        if (4, 4) in all_scores and all_scores[(4, 4)] > s2 * SUBPERIOD_4_4_THRESHOLD:
            all_scores[(4, 4)] *= SUBPERIOD_4_4_BOOST
        if (3, 4) in all_scores and all_scores[(3, 4)] > s2 * SUBPERIOD_3_4_THRESHOLD:
            all_scores[(3, 4)] *= SUBPERIOD_3_4_BOOST

    # 6/8 vs 3/4 disambiguation
    _disambiguate_compound(all_scores, all_beats, beat_interval)

    # Sub-beat division analysis: detect /4 vs /8 denominator
    sub_beat_den = 4
    if onset_event_times is not None and len(onset_event_times) > 0:
        sub_beat_den = signal_sub_beat_division(all_beats, onset_event_times, sr)
        if sub_beat_den == 8:
            logger.debug("Sub-beat analysis: compound (/8) detected")
            _compound_map = {
                (3, 4): (6, 8),
                (4, 4): (12, 8),
                (2, 4): (6, 8),
            }
            for simple, compound in _compound_map.items():
                if simple in all_scores:
                    transfer = all_scores[simple] * COMPOUND_TRANSFER_RATIO
                    all_scores[compound] = all_scores.get(compound, 0) + transfer
            for meter in list(all_scores.keys()):
                if meter[1] == 8:
                    all_scores[meter] *= COMPOUND_BOOST

    # 3/4 vs 4/4 disambiguation using neural net tracker evidence.
    bar_tracking_supports_triple = False
    if "bar_tracking" in signal_results:
        bt_score_3 = signal_results["bar_tracking"].get((3, 4), 0)
        if bt_score_3 > 0.5:
            bar_tracking_supports_triple = True
            logger.debug(f"Bar tracking supports 3/4 (score={bt_score_3:.2f})")

    if (3, 4) in all_scores and ((4, 4) in all_scores or (2, 4) in all_scores):
        nn_trackers_favor_even = 0
        nn_trackers_total = 0
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
                all_scores[(3, 4)] *= NN_PENALTY_MILD
                logger.debug(f"3/4 vs even: trusted NNs favor even but bar tracking says 3/4, mild penalty {NN_PENALTY_MILD}")
            else:
                all_scores[(3, 4)] *= NN_PENALTY_STRONG
                logger.debug(f"3/4 vs even: {nn_trackers_favor_even}/{nn_trackers_total} trusted NNs favor even, strong penalty")
        elif nn_trackers_total == 0 and sub_beat_den != 8:
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
                    all_scores[(3, 4)] *= NN_PENALTY_MILD
                    logger.debug(f"3/4 vs even: untrusted NNs favor even but bar tracking says 3/4, reduced penalty {NN_PENALTY_MILD}")
                else:
                    all_scores[(3, 4)] *= NN_PENALTY_WEAK
                    logger.debug(f"3/4 vs even: {raw_nn_favor_even}/{raw_nn_total} untrusted NNs favor even, weak penalty")
            elif raw_nn_total == 0 and bar_tracking_supports_triple:
                logger.debug("3/4 vs even: no NN data, bar tracking supports 3/4, skipping penalty")

    return sub_beat_den


def _compute_ambiguity(scores: dict[tuple[int, int], float]) -> float:
    """Compute normalized entropy of meter score distribution.

    Returns 0.0 when one meter dominates (engine is certain),
    1.0 when all meters are equally likely (maximum ambiguity).
    """
    total = sum(scores.values())
    if total <= 0:
        return 1.0
    probs = [v / total for v in scores.values() if v > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs))
    return round(entropy / max_entropy, 3)


def _format_hypotheses(
    all_scores: dict[tuple[int, int], float],
    max_hypotheses: int,
) -> tuple[list[MeterHypothesis], float]:
    """Filter, normalize, sort and format final hypotheses.

    Returns (hypotheses, meter_ambiguity).
    """
    # Compute ambiguity before filtering (on full distribution)
    ambiguity = _compute_ambiguity(all_scores)

    # Filter noise
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

    return hypotheses, ambiguity


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
    skip_mert: bool = False,
    skip_onset_mlp: bool = False,
    cache=None,
    audio_hash: str | None = None,
    tmp_path: str | None = None,
    return_signal_details: bool = False,
) -> tuple[list[MeterHypothesis], float] | tuple[list[MeterHypothesis], float, dict]:
    """Generate meter hypotheses from all signals.

    Merges and normalizes scores, returns (top N with confidence, meter_ambiguity).
    When return_signal_details=True, also returns a dict with signal_results,
    weights, and trust values for arbiter training.
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

    # Step 1: Compute signal weights based on tracker trust
    weights, beatnet_trust, beat_this_trust, madmom_trust = _compute_weights(
        beatnet_beats, madmom_results, beat_this_beats,
        beatnet_alignment, beat_this_alignment, madmom_alignment,
    )

    # Step 2: Collect scores from signals 1a, 1b, 2, 3, 7, 8, 8b, 9
    signal_results = _collect_signal_scores(
        weights, beatnet_beats, madmom_results,
        onset_times, onset_strengths, beat_interval, beat_times,
        sr, audio, beat_this_beats, skip_bar_tracking, skip_resnet,
        skip_mert, skip_onset_mlp,
        cache=cache, audio_hash=audio_hash, tmp_path=tmp_path,
    )

    # Step 3: Collect accent and periodicity scores (signals 4 & 5)
    _collect_accent_scores(
        weights, signal_results,
        all_beats, beatnet_beats, beat_this_beats, librosa_beats,
        madmom_results, onset_times, onset_strengths, audio, sr, tempo_bpm,
        cache=cache, audio_hash=audio_hash,
    )

    # Step 4: Weighted additive combination with consensus bonus
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
        if n_supporting >= 4:
            score *= CONSENSUS_4_BONUS
        elif n_supporting >= 3:
            score *= CONSENSUS_3_BONUS
        all_scores[meter] = score

    if not all_scores:
        fallback = ([MeterHypothesis(
            numerator=4, denominator=4, confidence=0.3,
            description=_get_description(4, 4),
        )], 1.0)
        if return_signal_details:
            return fallback[0], fallback[1], {
                "signal_results": signal_results,
                "weights": weights,
                "trust": {
                    "beatnet": beatnet_trust,
                    "beat_this": beat_this_trust,
                    "madmom": madmom_trust,
                },
                "tempo_bpm": tempo_bpm,
            }
        return fallback

    # Step 5: Apply adjustments (priors, rarity, compound, NN penalties)
    _apply_score_adjustments(
        all_scores, signal_results, all_beats, beat_interval,
        onset_event_times, sr, beatnet_beats, beat_this_beats,
    )

    # Capture signal details for arbiter training (always, lightweight)
    global last_signal_details
    last_signal_details = {
        "signal_results": {
            sig: {f"{k[0]}_{k[1]}": v for k, v in scores.items()}
            for sig, scores in signal_results.items()
        },
        "weights": weights,
        "trust": {
            "beatnet": beatnet_trust,
            "beat_this": beat_this_trust,
            "madmom": madmom_trust,
        },
        "tempo_bpm": tempo_bpm,
    }

    # Step 6: Filter, normalize, and format hypotheses
    hypotheses, ambiguity = _format_hypotheses(all_scores, max_hypotheses)

    if return_signal_details:
        return hypotheses, ambiguity, last_signal_details

    return hypotheses, ambiguity


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
