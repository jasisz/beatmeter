"""Pydantic response models for API."""

from pydantic import BaseModel


class BeatResponse(BaseModel):
    time: float
    is_downbeat: bool
    strength: float


class TempoResponse(BaseModel):
    bpm: float
    confidence: float
    is_variable: bool = False
    bpm_range: list[float] | None = None
    tempo_category: str = "steady"


class MeterHypothesisResponse(BaseModel):
    numerator: int
    denominator: int
    confidence: float
    grouping: str | None = None
    description: str = ""
    disambiguation_hint: str | None = None


class TempoCurvePointResponse(BaseModel):
    time: float
    bpm: float


class SectionResponse(BaseModel):
    start: float
    end: float
    meter: MeterHypothesisResponse | None = None
    tempo: TempoResponse | None = None


class AnalysisResponse(BaseModel):
    tempo: TempoResponse
    meter_hypotheses: list[MeterHypothesisResponse]
    beats: list[BeatResponse]
    tempo_curve: list[TempoCurvePointResponse] = []
    sections: list[SectionResponse] = []
    duration: float = 0.0
    converged: bool = False
    source_type: str = "recorded"  # "recorded" or "live"


# WebSocket message types

class OnsetMessage(BaseModel):
    type: str = "onset"
    time: float
    strength: float


class WarmupProgressMessage(BaseModel):
    type: str = "warmup_progress"
    seconds: float
    total: float
    onset_count: int = 0


class AnalysisMessage(BaseModel):
    type: str = "analysis"
    data: AnalysisResponse
