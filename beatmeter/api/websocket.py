"""WebSocket endpoint for live audio analysis."""

import asyncio
import json
import struct
import time
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.analysis.onset import detect_onsets
from beatmeter.audio.stream import StreamBuffer
from beatmeter.api.schemas import (
    AnalysisResponse,
    TempoResponse,
    MeterHypothesisResponse,
    BeatResponse,
    TempoCurvePointResponse,
    SectionResponse,
)
from beatmeter.config import settings

router = APIRouter()

# Convergence parameters
_CONVERGENCE_HISTORY = 3
_CONVERGENCE_BPM_TOLERANCE = 0.03  # 3% variance


def result_to_response(result, converged: bool = False, source_type: str = "recorded") -> dict:
    """Convert AnalysisResult to dict for JSON serialization."""
    return AnalysisResponse(
        tempo=TempoResponse(
            bpm=result.tempo.bpm,
            confidence=result.tempo.confidence,
            is_variable=result.tempo.is_variable,
            bpm_range=list(result.tempo.bpm_range) if result.tempo.bpm_range else None,
            tempo_category=result.tempo.tempo_category,
        ),
        meter_hypotheses=[
            MeterHypothesisResponse(
                numerator=h.numerator,
                denominator=h.denominator,
                confidence=h.confidence,
                grouping=h.grouping,
                description=h.description,
                disambiguation_hint=h.disambiguation_hint,
            )
            for h in result.meter_hypotheses
        ],
        beats=[
            BeatResponse(time=b.time, is_downbeat=b.is_downbeat, strength=b.strength)
            for b in result.beats
        ],
        tempo_curve=[
            TempoCurvePointResponse(time=p.time, bpm=p.bpm)
            for p in result.tempo_curve
        ],
        sections=[
            SectionResponse(
                start=s.start,
                end=s.end,
                meter=MeterHypothesisResponse(
                    numerator=s.meter.numerator,
                    denominator=s.meter.denominator,
                    confidence=s.meter.confidence,
                    grouping=s.meter.grouping,
                    description=s.meter.description,
                ) if s.meter else None,
                tempo=TempoResponse(
                    bpm=s.tempo.bpm,
                    confidence=s.tempo.confidence,
                    is_variable=s.tempo.is_variable,
                    bpm_range=list(s.tempo.bpm_range) if s.tempo.bpm_range else None,
                ) if s.tempo else None,
            )
            for s in result.sections
        ],
        duration=result.duration,
        meter_ambiguity=result.meter_ambiguity,
        converged=converged,
        source_type=source_type,
    ).model_dump()


def _check_convergence(history: deque) -> bool:
    """Check if the last N results have converged.

    Converged means: BPM variance < 3% AND same top meter for all entries.
    """
    if len(history) < _CONVERGENCE_HISTORY:
        return False

    bpms = [h["bpm"] for h in history]
    meters = [h["meter"] for h in history]

    # Check meter consistency
    if len(set(meters)) != 1:
        return False

    # Check BPM variance (relative to mean)
    mean_bpm = sum(bpms) / len(bpms)
    if mean_bpm == 0:
        return False
    max_deviation = max(abs(b - mean_bpm) / mean_bpm for b in bpms)
    return max_deviation < _CONVERGENCE_BPM_TOLERANCE


@router.websocket("/ws/live")
async def live_analysis(websocket: WebSocket):
    """Live audio analysis via WebSocket.

    Protocol:
    - Client sends binary Float32 PCM chunks (22050 Hz, mono)
    - Server sends JSON messages:
      - {"type": "warmup_progress", "seconds": N, "total": 8, "onset_count": K}
      - {"type": "onset", "time": T, "strength": S}
      - {"type": "analysis", "data": {...}}
    """
    await websocket.accept()

    stream_buffer = StreamBuffer(
        sr=settings.sample_rate,
        max_duration=settings.stream_buffer_seconds,
    )
    engine = AnalysisEngine()
    last_analysis_time = 0.0
    start_time = time.monotonic()
    onset_count = 0
    last_onset_check_duration = 0.0
    convergence_history: deque = deque(maxlen=_CONVERGENCE_HISTORY)

    try:
        while True:
            # Receive binary audio data
            data = await websocket.receive_bytes()

            # Decode Float32 PCM
            n_samples = len(data) // 4
            if n_samples == 0:
                continue
            chunk = np.frombuffer(data, dtype=np.float32)
            stream_buffer.append(chunk)

            elapsed = time.monotonic() - start_time
            buffer_duration = stream_buffer.duration

            # Phase 1: Warmup (0-8s)
            if buffer_duration < settings.warmup_seconds:
                # Periodically count onsets in buffered audio (every ~1s)
                if buffer_duration - last_onset_check_duration >= 1.0:
                    audio = stream_buffer.get_audio()
                    loop = asyncio.get_event_loop()
                    onsets = await loop.run_in_executor(
                        None, detect_onsets, audio, settings.sample_rate,
                    )
                    onset_count = len(onsets)
                    last_onset_check_duration = buffer_duration

                await websocket.send_json({
                    "type": "warmup_progress",
                    "seconds": round(buffer_duration, 1),
                    "total": settings.warmup_seconds,
                    "onset_count": onset_count,
                })
                continue

            # Phase 2 & 3: Analysis
            # Adaptive interval: faster during first 10s of analysis, then normal
            now = time.monotonic()
            analysis_elapsed = buffer_duration - settings.warmup_seconds
            if analysis_elapsed < 10.0:
                interval = settings.warmup_reanalysis_interval
            else:
                interval = settings.reanalysis_interval

            if (last_analysis_time == 0.0 or
                    now - last_analysis_time >= interval):
                last_analysis_time = now

                # Get audio window for analysis
                window_seconds = min(
                    buffer_duration,
                    settings.sliding_window_seconds,
                )
                audio = stream_buffer.get_audio(last_n_seconds=window_seconds)

                # Warm-up beat trimming: discard first 2 seconds from
                # short buffers to avoid irregular initial beats
                trim_seconds = 2.0
                trim_samples = int(trim_seconds * settings.sample_rate)
                if buffer_duration < 15.0 and len(audio) > trim_samples * 2:
                    audio = audio[trim_samples:]

                # Run analysis in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    engine.analyze_audio,
                    audio,
                    settings.sample_rate,
                )

                # Track convergence
                top_meter = ""
                if result.meter_hypotheses:
                    h = result.meter_hypotheses[0]
                    top_meter = f"{h.numerator}/{h.denominator}"
                convergence_history.append({
                    "bpm": result.tempo.bpm,
                    "meter": top_meter,
                })
                converged = _check_convergence(convergence_history)

                response = result_to_response(result, converged=converged, source_type="live")
                await websocket.send_json({
                    "type": "analysis",
                    "data": response,
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
