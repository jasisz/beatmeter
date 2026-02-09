"""Analysis orchestrator - combines all analysis modules."""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from beatmeter.analysis.models import AnalysisResult, Beat, BpmCandidate, MeterHypothesis, Section, TempoResult
from beatmeter.analysis.onset import detect_onsets, onset_strength_envelope
from beatmeter.analysis.beat_tracking import (
    track_beats_beatnet,
    track_beats_beat_this,
    track_beats_madmom,
    track_beats_librosa,
)
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


class AnalysisEngine:
    """Orchestrates the full analysis pipeline."""

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze an audio file."""
        audio, sr = load_audio(file_path, sr=settings.sample_rate)
        audio = preprocess(audio, sr)
        return self.analyze_audio(audio, sr)

    def analyze_audio(self, audio: np.ndarray, sr: int = 22050) -> AnalysisResult:
        """Analyze pre-loaded audio data."""
        duration = len(audio) / sr
        logger.info(f"Analyzing {duration:.1f}s of audio at {sr}Hz")

        # Step 1: Onset detection
        logger.info("Step 1: Onset detection")
        onsets = detect_onsets(audio, sr)
        onset_times, onset_strengths = onset_strength_envelope(audio, sr)
        onset_event_times = np.array([o.time for o in onsets])

        # Step 2: Beat tracking (run all methods in parallel)
        logger.info("Step 2: Beat tracking")

        # Write shared temp WAV once for all trackers that need file paths
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            shared_tmp_path = tmp.name

        try:
            def _run_all_madmom(audio, sr, tmp_path):
                results = {}
                for bpb in [3, 4, 5, 7]:
                    beats = track_beats_madmom(audio, sr, beats_per_bar=bpb, tmp_path=tmp_path)
                    if beats:
                        results[bpb] = beats
                return results

            with ThreadPoolExecutor(max_workers=4) as pool:
                f_beatnet = pool.submit(track_beats_beatnet, audio, sr, shared_tmp_path)
                f_beat_this = pool.submit(track_beats_beat_this, audio, sr, shared_tmp_path)
                f_madmom = pool.submit(_run_all_madmom, audio, sr, shared_tmp_path)
                f_librosa = pool.submit(track_beats_librosa, audio, sr)

            beatnet_beats = f_beatnet.result()
            beat_this_beats = f_beat_this.result()
            madmom_results: dict[int, list[Beat]] = f_madmom.result()
            librosa_beats = f_librosa.result()
        finally:
            os.unlink(shared_tmp_path)

        logger.info(f"  BeatNet: {len(beatnet_beats)} beats, "
                     f"{sum(1 for b in beatnet_beats if b.is_downbeat)} downbeats")
        logger.info(f"  Beat This!: {len(beat_this_beats)} beats, "
                     f"{sum(1 for b in beat_this_beats if b.is_downbeat)} downbeats")
        for bpb, madmom_beats in madmom_results.items():
            logger.info(f"  madmom ({bpb}/4): {len(madmom_beats)} beats, "
                         f"{sum(1 for b in madmom_beats if b.is_downbeat)} downbeats")

        # Validate madmom: if all variants return ~same beat count with IBI ~0.5s
        # (120 BPM), madmom's DBN has fallen back to its default prior and the
        # results are meaningless. Discard them entirely.
        if len(madmom_results) >= 2:
            _counts = [len(b) for b in madmom_results.values()]
            _counts_similar = (max(_counts) - min(_counts)) <= 2
            _all_120 = True
            for _bpb, _mb in madmom_results.items():
                if len(_mb) >= 3:
                    _t = np.array([b.time for b in _mb])
                    _ibi = float(np.median(np.diff(_t)))
                    if abs(_ibi - 0.5) > 0.03:  # not close to 120 BPM
                        _all_120 = False
                        break
                else:
                    _all_120 = False
                    break
            if _counts_similar and _all_120:
                logger.info("  madmom returned ~120 BPM for all variants (default prior); discarding")
                madmom_results = {}

        logger.info(f"  librosa: {len(librosa_beats)} beats")

        # Step 2.5: Select best primary beats by onset alignment
        # Beat trackers can hallucinate beats at wrong positions.
        # We validate by checking which tracker's beats land on actual audio events.
        logger.info("Step 2.5: Beat selection by onset alignment")
        tracker_candidates = []
        if beatnet_beats:
            tracker_candidates.append(('BeatNet', beatnet_beats))
        if beat_this_beats:
            tracker_candidates.append(('Beat This!', beat_this_beats))
        for bpb, beats in madmom_results.items():
            tracker_candidates.append((f'madmom {bpb}/4', beats))
        if librosa_beats:
            tracker_candidates.append(('librosa', librosa_beats))

        primary_beats = librosa_beats  # default fallback
        best_name = 'librosa'
        best_alignment = 0.0
        beatnet_alignment = 0.0
        beat_this_alignment = 0.0
        madmom_best_alignment = 0.0

        for name, beats in tracker_candidates:
            if not beats:
                continue
            score = self._onset_alignment_score(beats, onset_event_times)
            logger.info(f"  {name}: alignment={score:.3f} ({len(beats)} beats)")

            # Track per-tracker alignment for meter signal weighting
            if name == 'BeatNet':
                beatnet_alignment = score
            elif name == 'Beat This!':
                beat_this_alignment = score
            elif name.startswith('madmom'):
                madmom_best_alignment = max(madmom_best_alignment, score)

            if score > best_alignment:
                best_alignment = score
                primary_beats = beats
                best_name = name

        logger.info(f"  Primary: {best_name} (alignment={best_alignment:.3f}, "
                     f"{len(primary_beats)} beats)")

        # Step 3: Tempo estimation
        logger.info("Step 3: Tempo estimation")
        candidates = []

        ibi_est = estimate_from_ibi(primary_beats, settings.min_bpm, settings.max_bpm)
        if ibi_est:
            # Scale IBI confidence by beat-onset alignment quality.
            # Poorly aligned trackers produce unreliable IBI estimates.
            scaled_conf = round(ibi_est.confidence * max(best_alignment, 0.1), 2)
            ibi_est = BpmCandidate(
                bpm=ibi_est.bpm, confidence=scaled_conf, method=ibi_est.method,
            )
            candidates.append(ibi_est)

        librosa_est = estimate_from_librosa(audio, sr)
        if librosa_est:
            candidates.append(librosa_est)

        tempogram_est = estimate_from_tempogram(audio, sr)
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

        # Step 5: Meter hypothesis generation
        # Pass tempo_bpm so meter detection can use it to constrain search
        logger.info("Step 5: Meter hypothesis generation")
        meter_hypotheses = generate_hypotheses(
            beatnet_beats=beatnet_beats,
            madmom_results=madmom_results,
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            all_beats=primary_beats,
            sr=sr,
            max_hypotheses=settings.max_meter_hypotheses,
            tempo_bpm=tempo.bpm,
            beatnet_alignment=beatnet_alignment,
            madmom_alignment=madmom_best_alignment,
            audio=audio,
            librosa_beats=librosa_beats,
            beat_this_beats=beat_this_beats,
            beat_this_alignment=beat_this_alignment,
            onset_event_times=onset_event_times,
        )
        for h in meter_hypotheses:
            logger.info(f"  {h.label}: {h.confidence:.1%} - {h.description}")

        # Step 6: Section detection with per-section meter analysis
        logger.info("Step 6: Section detection")
        sections = self._detect_sections(
            primary_beats, beatnet_beats, beat_this_beats, madmom_results, librosa_beats,
            onset_times, onset_strengths, onset_event_times, audio, sr, tempo.bpm,
        )
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
        )

    @staticmethod
    def _onset_alignment_score(
        beats: list[Beat],
        onset_event_times: np.ndarray,
        max_dist: float = 0.07,
    ) -> float:
        """Score how well beat positions align with detected onset events.

        Uses F1-like score combining:
        - Precision: do beats land on actual onsets?
        - Recall: do onsets have a nearby beat?

        This prevents selecting a tracker that found only half the beats
        (high precision, low recall) over one that found nearly all.
        """
        if not beats or len(onset_event_times) == 0:
            return 0.0

        beat_times = np.array([b.time for b in beats])

        # Forward: how well beats align with onsets (precision)
        total_fwd = 0.0
        for b in beats:
            min_dist = float(np.min(np.abs(onset_event_times - b.time)))
            total_fwd += min(min_dist, max_dist)
        precision = 1.0 - total_fwd / len(beats) / max_dist

        # Reverse: how well onsets are covered by beats (recall)
        total_rev = 0.0
        for ot in onset_event_times:
            min_dist = float(np.min(np.abs(beat_times - ot)))
            total_rev += min(min_dist, max_dist)
        recall = 1.0 - total_rev / len(onset_event_times) / max_dist

        # F1 score: harmonic mean of precision and recall
        if precision + recall > 0:
            return 2.0 * precision * recall / (precision + recall)
        return 0.0

    def _detect_sections(
        self,
        beats: list[Beat],
        beatnet_beats: list[Beat],
        beat_this_beats: list[Beat],
        madmom_results: dict[int, list[Beat]],
        librosa_beats: list[Beat],
        onset_times: np.ndarray,
        onset_strengths: np.ndarray,
        onset_event_times: np.ndarray,
        audio: np.ndarray,
        sr: int,
        global_bpm: float,
    ) -> list[Section]:
        """Detect sections and analyze meter per section.

        Uses tempo curve change points to find section boundaries.
        Falls back to ~20s segments if no clear boundaries found.
        """
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
            sec_beatnet = [b for b in beatnet_beats if sec_start <= b.time < sec_end]
            sec_beat_this = [b for b in beat_this_beats if sec_start <= b.time < sec_end]
            sec_librosa = [b for b in librosa_beats if sec_start <= b.time < sec_end]

            # Get madmom beats in this section
            sec_madmom: dict[int, list[Beat]] = {}
            for bpb, mb in madmom_results.items():
                sec_mb = [b for b in mb if sec_start <= b.time < sec_end]
                if sec_mb:
                    sec_madmom[bpb] = sec_mb

            # Get onsets in this section
            onset_mask = (onset_times >= sec_start) & (onset_times < sec_end)
            sec_onset_times = onset_times[onset_mask]
            sec_onset_strengths = onset_strengths[onset_mask]

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

            # Per-section meter hypothesis (top 1)
            sec_meter = None
            if len(sec_beats) >= 4 and len(sec_onset_times) >= 4:
                sec_bpm_val = sec_tempo.bpm if sec_tempo else global_bpm
                hyps = generate_hypotheses(
                    beatnet_beats=sec_beatnet,
                    madmom_results=sec_madmom,
                    onset_times=sec_onset_times,
                    onset_strengths=sec_onset_strengths,
                    all_beats=sec_beats,
                    sr=sr,
                    max_hypotheses=1,
                    tempo_bpm=sec_bpm_val,
                    audio=audio,
                    librosa_beats=sec_librosa,
                    beat_this_beats=sec_beat_this,
                    onset_event_times=onset_event_times,
                    skip_bar_tracking=True,
                )
                if hyps:
                    sec_meter = hyps[0]

            sections.append(Section(
                start=round(sec_start, 2),
                end=round(sec_end, 2),
                meter=sec_meter,
                tempo=sec_tempo,
            ))

        return sections

    def _find_tempo_boundaries(
        self,
        beats: list[Beat],
        duration: float,
    ) -> list[float]:
        """Find section boundaries based on tempo changes.

        Returns list of boundary times (always starts with 0 and ends with duration).
        """
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
                    # Don't add boundaries too close together (<8s)
                    if t - boundaries[-1] > 8.0:
                        boundaries.append(round(t, 2))

        # If no tempo changes found, split into ~20s segments
        if len(boundaries) == 1 and duration > 25:
            segment_dur = 20.0
            t = segment_dur
            while t < duration - 10:
                # Snap to nearest beat
                dists = np.abs(times - t)
                nearest_beat_idx = np.argmin(dists)
                snap_time = float(times[nearest_beat_idx])
                if snap_time - boundaries[-1] > 10.0:
                    boundaries.append(round(snap_time, 2))
                t += segment_dur

        boundaries.append(round(duration, 2))
        return boundaries
