"""Analysis orchestrator - combines all analysis modules."""

import logging

import numpy as np

from beatmeter.analysis.models import AnalysisResult, Beat, BpmCandidate, MeterHypothesis, Section, TempoResult
from beatmeter.analysis.onset import detect_onsets, onset_strength_envelope
from beatmeter.analysis.trackers import track_beats_librosa
from beatmeter.analysis.tempo import (
    estimate_from_ibi,
    estimate_from_librosa,
    estimate_from_tempogram,
    consensus_tempo,
    compute_tempo_curve,
    classify_tempo_variability,
)
from beatmeter.analysis.meter import generate_hypotheses
from beatmeter.audio.loader import load_audio
from beatmeter.audio.preprocessing import preprocess
from beatmeter.config import settings

logger = logging.getLogger(__name__)


def _beats_to_dicts(beats: list[Beat]) -> list[dict]:
    return [{"time": b.time, "is_downbeat": b.is_downbeat, "strength": b.strength}
            for b in beats]


def _dicts_to_beats(data: list[dict]) -> list[Beat]:
    return [Beat(time=d["time"], is_downbeat=d["is_downbeat"], strength=d["strength"])
            for d in data]


class AnalysisEngine:
    """Orchestrates the full analysis pipeline."""

    def __init__(self, cache=None):
        self.cache = cache  # AnalysisCache | None
        self._audio_hash: str | None = None

    def analyze_file(self, file_path: str, skip_sections: bool = False) -> AnalysisResult:
        """Analyze an audio file."""
        self._file_path = file_path
        if self.cache:
            from beatmeter.analysis.cache import AnalysisCache
            self._audio_hash = AnalysisCache.audio_hash(file_path)

        # Fast path: skip expensive audio decoding if cache is fully warm
        audio, sr = None, settings.sample_rate
        if self.cache and self._audio_hash and self._is_cache_warm(self._audio_hash):
            logger.debug("Fast path: skipping audio load for %s", file_path)
        else:
            audio, sr = load_audio(file_path, sr=settings.sample_rate)
            audio = preprocess(audio, sr)

        return self.analyze_audio(audio, sr, skip_sections=skip_sections)

    def _is_cache_warm(self, ah: str) -> bool:
        """Check if all critical cache entries exist for fast path."""
        c = self.cache
        if c.load_onsets(ah) is None:
            return False
        if c.load_beats(ah, "librosa") is None:
            return False
        if c.load_array(ah, "meter_net_audio") is None:
            return False
        if c.load_array(ah, "meter_net_mert") is None:
            return False
        return True

    def analyze_audio(self, audio: np.ndarray, sr: int = 22050, audio_hash: str | None = None, skip_sections: bool = False) -> AnalysisResult:
        """Analyze pre-loaded audio data."""
        if audio_hash is not None:
            self._audio_hash = audio_hash
        duration = len(audio) / sr if audio is not None else 0.0
        logger.info(f"Analyzing {duration:.1f}s of audio at {sr}Hz")

        ah = self._audio_hash  # may be None if called directly

        # Step 1: Onset detection
        logger.info("Step 1: Onset detection")
        cached_onsets = None
        if self.cache and ah:
            cached_onsets = self.cache.load_onsets(ah)

        if cached_onsets is not None:
            onset_times = np.array(cached_onsets["onset_times"])
            onset_strengths = np.array(cached_onsets["onset_strengths"])
            onset_event_times = np.array(cached_onsets["onset_events"])
            logger.info("  Onsets loaded from cache")
        elif audio is not None:
            onsets = detect_onsets(audio, sr)
            onset_times, onset_strengths = onset_strength_envelope(audio, sr)
            onset_event_times = np.array([o.time for o in onsets])
            if self.cache and ah:
                self.cache.save_onsets(ah, {
                    "onset_times": onset_times.tolist(),
                    "onset_strengths": onset_strengths.tolist(),
                    "onset_events": onset_event_times.tolist(),
                })
        else:
            onset_times = np.array([])
            onset_strengths = np.array([])
            onset_event_times = np.array([])

        # Step 2: Beat tracking (librosa only)
        logger.info("Step 2: Beat tracking (librosa)")
        librosa_beats = self._run_librosa_beats(audio, sr, ah)
        logger.info(f"  librosa: {len(librosa_beats)} beats")

        primary_beats = librosa_beats

        # Estimate duration from beats if audio was not loaded
        if duration == 0.0 and primary_beats:
            duration = round(primary_beats[-1].time + 1.0, 2)

        # Step 3: Tempo estimation
        logger.info("Step 3: Tempo estimation")
        candidates = []

        ibi_est = estimate_from_ibi(primary_beats, settings.min_bpm, settings.max_bpm)
        if ibi_est:
            candidates.append(ibi_est)

        # Librosa tempo (cached)
        librosa_est = None
        if self.cache and ah:
            cached = self.cache.load_signal(ah, "tempo_librosa")
            if cached is not None:
                librosa_est = BpmCandidate(bpm=cached["bpm"], confidence=cached["confidence"], method=cached["method"])
        if librosa_est is None and audio is not None:
            librosa_est = estimate_from_librosa(audio, sr)
            if librosa_est and self.cache and ah:
                self.cache.save_signal(ah, "tempo_librosa", {"bpm": librosa_est.bpm, "confidence": librosa_est.confidence, "method": librosa_est.method})
        if librosa_est:
            candidates.append(librosa_est)

        # Tempogram tempo (cached)
        tempogram_est = None
        if self.cache and ah:
            cached = self.cache.load_signal(ah, "tempo_tempogram")
            if cached is not None:
                tempogram_est = BpmCandidate(bpm=cached["bpm"], confidence=cached["confidence"], method=cached["method"])
        if tempogram_est is None and audio is not None:
            tempogram_est = estimate_from_tempogram(audio, sr)
            if tempogram_est and self.cache and ah:
                self.cache.save_signal(ah, "tempo_tempogram", {"bpm": tempogram_est.bpm, "confidence": tempogram_est.confidence, "method": tempogram_est.method})
        if tempogram_est:
            candidates.append(tempogram_est)

        tempo = consensus_tempo(candidates, settings.min_bpm, settings.max_bpm)
        logger.info(f"  Tempo: {tempo.bpm} BPM (confidence: {tempo.confidence})")

        # Step 4: Tempo curve (variable tempo detection)
        logger.info("Step 4: Tempo curve")
        tempo_curve, is_variable, bpm_range, tempo_category = compute_tempo_curve(primary_beats)
        tempo.is_variable = is_variable
        tempo.bpm_range = bpm_range
        tempo.tempo_category = tempo_category
        if is_variable and bpm_range:
            logger.info(f"  Variable tempo: {bpm_range[0]}-{bpm_range[1]} BPM ({tempo_category})")

        # Step 5: Meter hypothesis generation (MeterNet)
        logger.info("Step 5: Meter hypothesis generation")
        meter_hypotheses, meter_ambiguity = generate_hypotheses(
            audio=audio,
            sr=sr,
            cache=self.cache,
            audio_hash=ah,
            audio_file_path=getattr(self, '_file_path', None),
        )
        for h in meter_hypotheses:
            logger.info(f"  {h.label}: {h.confidence:.1%} - {h.description}")

        # Step 6: Section detection
        if skip_sections:
            sections = []
        else:
            logger.info("Step 6: Section detection")
            sections = self._detect_sections(primary_beats, audio, sr, tempo.bpm)
            for s in sections:
                if s.meter:
                    logger.info(f"  Section {s.start:.1f}-{s.end:.1f}s: {s.meter.label}")

        return AnalysisResult(
            tempo=tempo,
            meter_hypotheses=meter_hypotheses,
            beats=primary_beats,
            tempo_curve=tempo_curve,
            sections=sections,
            duration=duration,
            meter_ambiguity=meter_ambiguity,
        )

    def _run_librosa_beats(
        self, audio: np.ndarray | None, sr: int, ah: str | None,
    ) -> list[Beat]:
        """Run librosa beat tracking with caching."""
        if self.cache and ah:
            raw = self.cache.load_beats(ah, "librosa")
            if raw is not None:
                return _dicts_to_beats(raw)

        if audio is None:
            return []

        beats = track_beats_librosa(audio, sr)
        if self.cache and ah:
            self.cache.save_beats(ah, "librosa", _beats_to_dicts(beats))
        return beats

    def _detect_sections(
        self,
        beats: list[Beat],
        audio: np.ndarray | None,
        sr: int,
        global_bpm: float,
    ) -> list[Section]:
        """Detect sections and analyze meter per section.

        Uses tempo curve change points to find section boundaries.
        Falls back to ~20s segments if no clear boundaries found.
        """
        if audio is None:
            return []
        duration = len(audio) / sr
        if duration < 15:
            return [Section(start=0.0, end=round(duration, 2))]

        # Find section boundaries from tempo changes
        boundaries = self._find_tempo_boundaries(beats, duration)

        # Analyze each section
        sections = []
        for i in range(len(boundaries) - 1):
            sec_start = boundaries[i]
            sec_end = boundaries[i + 1]

            # Get beats in this section
            sec_beats = [b for b in beats if sec_start <= b.time < sec_end]

            # Per-section tempo
            sec_tempo = None
            if len(sec_beats) >= 4:
                times = np.array([b.time for b in sec_beats])
                ibis = np.diff(times)
                valid = ibis[(ibis > 0.15) & (ibis < 2.0)]
                if len(valid) >= 2:
                    sec_bpm = round(60.0 / float(np.median(valid)), 1)
                    cv = float(np.std(valid)) / float(np.median(valid)) if np.median(valid) > 0 else 0
                    sec_tempo = TempoResult(
                        bpm=sec_bpm,
                        confidence=round(max(0.0, 1.0 - cv * 2), 2),
                        tempo_category=classify_tempo_variability(cv),
                    )

            sections.append(Section(
                start=round(sec_start, 2),
                end=round(sec_end, 2),
                tempo=sec_tempo,
            ))

        return sections

    def _find_tempo_boundaries(
        self,
        beats: list[Beat],
        duration: float,
    ) -> list[float]:
        """Find section boundaries based on tempo changes."""
        boundaries = [0.0]

        if len(beats) < 8:
            boundaries.append(duration)
            return boundaries

        times = np.array([b.time for b in beats])
        ibis = np.diff(times)

        # Compute local BPM with small window
        window = 4
        local_bpms = []
        local_times = []
        for i in range(len(ibis)):
            start = max(0, i - window // 2)
            end = min(len(ibis), i + window // 2 + 1)
            w = ibis[start:end]
            valid = w[(w > 0.1) & (w < 3.0)]
            if len(valid) > 0:
                local_bpms.append(60.0 / float(np.median(valid)))
                local_times.append(float(times[i]))

        if len(local_bpms) < 4:
            boundaries.append(duration)
            return boundaries

        bpm_arr = np.array(local_bpms)
        time_arr = np.array(local_times)

        # Look for significant tempo jumps (>15% change in local BPM)
        for i in range(1, len(bpm_arr)):
            if bpm_arr[i - 1] > 0:
                change = abs(bpm_arr[i] - bpm_arr[i - 1]) / bpm_arr[i - 1]
                if change > 0.15:
                    t = time_arr[i]
                    if t - boundaries[-1] > 8.0:
                        boundaries.append(round(t, 2))

        # If no tempo changes found, split into ~20s segments
        if len(boundaries) == 1 and duration > 25:
            segment_dur = 20.0
            t = segment_dur
            while t < duration - 10:
                dists = np.abs(times - t)
                nearest_beat_idx = np.argmin(dists)
                snap_time = float(times[nearest_beat_idx])
                if snap_time - boundaries[-1] > 10.0:
                    boundaries.append(round(snap_time, 2))
                t += segment_dur

        boundaries.append(round(duration, 2))
        return boundaries
