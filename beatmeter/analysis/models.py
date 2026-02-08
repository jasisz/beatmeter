"""Core data models for rhythm analysis."""

from dataclasses import dataclass, field


@dataclass
class Beat:
    """A single detected beat."""
    time: float  # seconds
    is_downbeat: bool = False
    strength: float = 1.0  # 0.0-1.0


@dataclass
class OnsetStrength:
    """Onset strength at a point in time."""
    time: float
    strength: float


@dataclass
class BpmCandidate:
    """A BPM estimate from a single method."""
    bpm: float
    confidence: float  # 0.0-1.0
    method: str  # e.g. "inter_beat", "librosa", "tempogram"


@dataclass
class TempoResult:
    """Consensus tempo estimation."""
    bpm: float
    confidence: float
    is_variable: bool = False
    bpm_range: tuple[float, float] | None = None
    candidates: list[BpmCandidate] = field(default_factory=list)
    # "steady" | "slightly_variable" | "variable" | "rubato"
    tempo_category: str = "steady"


@dataclass
class MeterHypothesis:
    """A hypothesized time signature with confidence."""
    numerator: int  # e.g. 7
    denominator: int  # e.g. 8
    confidence: float  # 0.0-1.0
    grouping: str | None = None  # e.g. "2+2+3" for 7/8
    description: str = ""
    disambiguation_hint: str | None = None  # e.g. for 6/8 vs 3/4

    @property
    def label(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class TempoCurvePoint:
    """A point on the tempo-over-time curve."""
    time: float
    bpm: float


@dataclass
class Section:
    """A section of audio with consistent meter/tempo."""
    start: float
    end: float
    meter: MeterHypothesis | None = None
    tempo: TempoResult | None = None


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    tempo: TempoResult
    meter_hypotheses: list[MeterHypothesis]
    beats: list[Beat]
    tempo_curve: list[TempoCurvePoint] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    duration: float = 0.0
