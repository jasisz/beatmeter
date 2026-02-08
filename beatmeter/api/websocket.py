"""WebSocket endpoint for live audio analysis."""

import asyncio
import json
import struct
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from beatmeter.analysis.engine import AnalysisEngine
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


def result_to_response(result) -> dict:
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
    ).model_dump()


@router.websocket("/ws/live")
async def live_analysis(websocket: WebSocket):
    """Live audio analysis via WebSocket.

    Protocol:
    - Client sends binary Float32 PCM chunks (22050 Hz, mono)
    - Server sends JSON messages:
      - {"type": "warmup_progress", "seconds": N, "total": 8}
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
                await websocket.send_json({
                    "type": "warmup_progress",
                    "seconds": round(buffer_duration, 1),
                    "total": settings.warmup_seconds,
                })
                continue

            # Phase 2 & 3: Analysis
            now = time.monotonic()
            if (last_analysis_time == 0.0 or
                    now - last_analysis_time >= settings.reanalysis_interval):
                last_analysis_time = now

                # Get audio window for analysis
                window_seconds = min(
                    buffer_duration,
                    settings.sliding_window_seconds,
                )
                audio = stream_buffer.get_audio(last_n_seconds=window_seconds)

                # Run analysis in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    engine.analyze_audio,
                    audio,
                    settings.sample_rate,
                )

                response = result_to_response(result)
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
