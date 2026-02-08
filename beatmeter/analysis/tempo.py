"""Multi-method tempo estimation with consensus."""

import numpy as np
import librosa

from beatmeter.analysis.models import Beat, BpmCandidate, TempoResult, TempoCurvePoint


def estimate_from_ibi(beats: list[Beat], min_bpm: float = 40, max_bpm: float = 300) -> BpmCandidate | None:
    """Estimate tempo from inter-beat intervals."""
    if len(beats) < 3:
        return None

    times = np.array([b.time for b in beats])
    ibis = np.diff(times)

    # Filter out outlier intervals
    valid = ibis[(ibis > 60.0 / max_bpm) & (ibis < 60.0 / min_bpm)]
    if len(valid) < 2:
        return None

    median_ibi = float(np.median(valid))
    bpm = 60.0 / median_ibi

    # Confidence based on consistency of IBIs
    std = float(np.std(valid))
    cv = std / median_ibi if median_ibi > 0 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - cv * 2))

    return BpmCandidate(bpm=round(bpm, 1), confidence=confidence, method="inter_beat")


def estimate_from_librosa(audio: np.ndarray, sr: int = 22050) -> BpmCandidate | None:
    """Estimate tempo using librosa's beat tracker."""
    try:
        tempo = librosa.feature.rhythm.tempo(y=audio, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        return BpmCandidate(bpm=round(bpm, 1), confidence=0.7, method="librosa")
    except Exception:
        return None


def estimate_from_tempogram(audio: np.ndarray, sr: int = 22050) -> BpmCandidate | None:
    """Estimate tempo from tempogram peaks."""
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

        # Average tempogram across time
        avg_tempogram = np.mean(tempogram, axis=1)

        # BPM axis
        bpm_axis = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)

        # Find peak in valid range (40-300 BPM)
        valid_mask = (bpm_axis >= 40) & (bpm_axis <= 300)
        if not np.any(valid_mask):
            return None

        valid_tempos = bpm_axis[valid_mask]
        valid_strengths = avg_tempogram[valid_mask]

        peak_idx = np.argmax(valid_strengths)
        bpm = float(valid_tempos[peak_idx])
        peak_strength = float(valid_strengths[peak_idx])
        max_strength = float(valid_strengths.max()) if valid_strengths.max() > 0 else 1.0
        confidence = peak_strength / max_strength * 0.8  # tempogram is less reliable

        return BpmCandidate(bpm=round(bpm, 1), confidence=confidence, method="tempogram")
    except Exception:
        return None


def consensus_tempo(
    candidates: list[BpmCandidate],
    min_bpm: float = 40,
    max_bpm: float = 300,
) -> TempoResult:
    """Build consensus from multiple BPM estimates."""
    if not candidates:
        return TempoResult(bpm=120.0, confidence=0.0, candidates=[])

    # Weight by confidence
    total_weight = sum(c.confidence for c in candidates)
    if total_weight == 0:
        bpm = candidates[0].bpm
        return TempoResult(bpm=bpm, confidence=0.1, candidates=candidates)

    # Check if candidates agree (within 5% tolerance)
    bpms = [c.bpm for c in candidates]
    reference = max(candidates, key=lambda c: c.confidence).bpm

    # Handle tempo octave ambiguity: 60 vs 120 vs 240 BPM
    normalized = []
    for c in candidates:
        b = c.bpm
        while b < reference * 0.7 and b * 2 <= max_bpm:
            b *= 2
        while b > reference * 1.4 and b / 2 >= min_bpm:
            b /= 2
        normalized.append(b)

    # Weighted average of normalized tempos
    weighted_bpm = sum(b * c.confidence for b, c in zip(normalized, candidates)) / total_weight

    # Half-speed correction: if consensus lands below 80 BPM but a candidate
    # exists near the doubled value, prefer the doubled value. This fixes
    # the common half-speed problem where tempogram and IBI both pick
    # a sub-harmonic (e.g., 55 instead of 110).
    if weighted_bpm < 80 and weighted_bpm * 2 <= max_bpm:
        doubled = weighted_bpm * 2
        for c in candidates:
            if abs(c.bpm - doubled) / doubled < 0.15:
                weighted_bpm = doubled
                break

    # Double-speed correction: if consensus lands above 200 BPM, the primary
    # tracker may be tracking at double speed (e.g. tango at 252 instead of 126).
    # Prefer the halved value if a candidate exists near it or if halved is in
    # the comfortable range (80-180 BPM).
    if weighted_bpm > 200 and weighted_bpm / 2 >= min_bpm:
        halved = weighted_bpm / 2
        has_support = any(
            abs(c.bpm - halved) / halved < 0.15 for c in candidates
        )
        if has_support or (80 <= halved <= 180):
            weighted_bpm = halved

    # Confidence: high if candidates agree, low if they disagree
    deviations = [abs(b - weighted_bpm) / weighted_bpm for b in normalized]
    avg_deviation = sum(d * c.confidence for d, c in zip(deviations, candidates)) / total_weight
    confidence = max(0.0, min(1.0, 1.0 - avg_deviation * 5))

    return TempoResult(
        bpm=round(weighted_bpm, 1),
        confidence=round(confidence, 2),
        candidates=candidates,
    )


def classify_tempo_variability(cv: float) -> str:
    """Classify tempo variability by coefficient of variation.

    Returns: "steady", "slightly_variable", "variable", or "rubato".
    """
    if cv < 0.03:
        return "steady"
    elif cv < 0.07:
        return "slightly_variable"
    elif cv < 0.15:
        return "variable"
    else:
        return "rubato"


def compute_tempo_curve(
    beats: list[Beat],
    window_beats: int = 8,
) -> tuple[list[TempoCurvePoint], bool, tuple[float, float] | None, str]:
    """Compute tempo over time from beat positions.

    Returns (tempo_curve, is_variable, bpm_range, tempo_category).
    """
    if len(beats) < 4:
        return [], False, None, "steady"

    times = np.array([b.time for b in beats])
    ibis = np.diff(times)

    curve = []
    bpms = []
    for i in range(len(ibis)):
        # Use a window of beats for smoothing
        start = max(0, i - window_beats // 2)
        end = min(len(ibis), i + window_beats // 2 + 1)
        window_ibis = ibis[start:end]
        valid = window_ibis[window_ibis > 0]
        if len(valid) > 0:
            local_bpm = 60.0 / float(np.median(valid))
            if 30 <= local_bpm <= 400:
                curve.append(TempoCurvePoint(
                    time=float(times[i]),
                    bpm=round(local_bpm, 1),
                ))
                bpms.append(local_bpm)

    if not bpms:
        return [], False, None, "steady"

    bpm_array = np.array(bpms)
    bpm_std = float(np.std(bpm_array))
    bpm_mean = float(np.mean(bpm_array))

    cv = (bpm_std / bpm_mean) if bpm_mean > 0 else 0.0
    is_variable = cv > 0.05
    bpm_range = (round(float(bpm_array.min()), 1), round(float(bpm_array.max()), 1))
    tempo_category = classify_tempo_variability(cv)

    return curve, is_variable, bpm_range, tempo_category
