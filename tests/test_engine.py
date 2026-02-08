"""Integration tests for the analysis engine."""

import io
import numpy as np
import soundfile as sf
import pytest

from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.analysis.models import AnalysisResult
from tests.conftest import generate_click_track


def test_analyze_audio_returns_result():
    """Engine should return a valid AnalysisResult for a simple click track."""
    audio = generate_click_track(bpm=120, beats_per_bar=4, duration_seconds=8)
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr=22050)

    assert isinstance(result, AnalysisResult)
    assert result.tempo is not None
    assert result.tempo.bpm > 0
    assert len(result.meter_hypotheses) > 0
    assert len(result.beats) > 0
    assert result.duration > 0


def test_tempo_estimation_accuracy():
    """Tempo estimation should be within 10% of the true BPM."""
    audio = generate_click_track(bpm=120, beats_per_bar=4, duration_seconds=10)
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr=22050)

    # Allow octave errors (60 or 240 are also acceptable)
    bpm = result.tempo.bpm
    acceptable = any(
        abs(bpm - target) / target < 0.1
        for target in [60, 120, 240]
    )
    assert acceptable, f"BPM {bpm} not within 10% of 60, 120, or 240"


def test_analyze_file(tmp_path):
    """Engine should be able to analyze a WAV file from disk."""
    audio = generate_click_track(bpm=100, beats_per_bar=3, duration_seconds=8)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 22050)

    engine = AnalysisEngine()
    result = engine.analyze_file(str(wav_path))

    assert isinstance(result, AnalysisResult)
    assert result.tempo.bpm > 0
    assert len(result.meter_hypotheses) > 0


def test_beats_are_sorted():
    """Beats should be in chronological order."""
    audio = generate_click_track(bpm=120, beats_per_bar=4, duration_seconds=8)
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr=22050)

    times = [b.time for b in result.beats]
    assert times == sorted(times)


def test_meter_hypotheses_confidence_sums_to_1():
    """Meter hypothesis confidences should approximately sum to 1."""
    audio = generate_click_track(bpm=120, beats_per_bar=4, duration_seconds=10)
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr=22050)

    total_confidence = sum(h.confidence for h in result.meter_hypotheses)
    assert 0.9 <= total_confidence <= 1.1, f"Total confidence: {total_confidence}"


def test_api_analyze_endpoint(client, tmp_path):
    """POST /api/analyze should return valid JSON."""
    audio = generate_click_track(bpm=120, beats_per_bar=4, duration_seconds=5)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 22050)

    with open(wav_path, "rb") as f:
        response = client.post("/api/analyze", files={"file": ("test.wav", f, "audio/wav")})

    assert response.status_code == 200
    data = response.json()
    assert "tempo" in data
    assert "meter_hypotheses" in data
    assert "beats" in data
    assert data["tempo"]["bpm"] > 0
    assert len(data["meter_hypotheses"]) > 0


def test_health_endpoint(client):
    """GET /api/health should return ok."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
