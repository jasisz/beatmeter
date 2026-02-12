"""File upload endpoint for audio analysis."""

import io
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException

from beatmeter.api.schemas import (
    AnalysisResponse,
    TempoResponse,
    MeterHypothesisResponse,
    BeatResponse,
    TempoCurvePointResponse,
    SectionResponse,
)
from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(file: UploadFile = File(...)):
    """Analyze an uploaded audio file for tempo and meter."""
    # Validate file
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    # Read file content
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {settings.max_upload_mb} MB)")

    # Write to temp file (librosa needs file path for some formats)
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1].lower()

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        engine = AnalysisEngine()
        result = engine.analyze_file(tmp_path)

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
        )
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
