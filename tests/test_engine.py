"""Integration tests for the analysis engine."""

import numpy as np
import soundfile as sf
import pytest

from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.analysis.models import AnalysisResult
from beatmeter.analysis.cache import AnalysisCache
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


def test_catch_all_blocks_path_traversal(client, monkeypatch, tmp_path):
    """Catch-all route should not serve files outside frontend root."""
    import beatmeter.main as main_module

    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text("<html>INDEX</html>")
    (tmp_path / "secret.txt").write_text("TOP_SECRET")

    monkeypatch.setattr(main_module, "FRONTEND_DIR", frontend_dir)
    monkeypatch.setattr(main_module, "FRONTEND_ROOT", frontend_dir.resolve())

    response = client.get("/..%2Fsecret.txt")

    assert response.status_code == 200
    assert "TOP_SECRET" not in response.text
    assert "INDEX" in response.text


def test_api_analyze_rejects_oversized_file(client, monkeypatch):
    """Upload endpoint should reject files larger than configured limit."""
    from beatmeter.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 1)
    payload = b"x" * (1024 * 1024 + 1)

    response = client.post(
        "/api/analyze",
        files={"file": ("big.wav", payload, "audio/wav")},
    )

    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]


def test_api_analyze_tempfile_failure_returns_generic_error(client, monkeypatch):
    """Upload endpoint should not leak internal exception details."""
    import beatmeter.api.upload as upload_module

    def _raise_tempfile_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(upload_module.tempfile, "NamedTemporaryFile", _raise_tempfile_error)

    response = client.post(
        "/api/analyze",
        files={"file": ("test.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Analysis failed"


def test_analyze_audio_cleans_tmp_wav_on_exception(monkeypatch, tmp_path):
    """Shared tracker temp WAV should be removed even if later stages fail."""
    import beatmeter.analysis.engine as engine_module

    tmp_wav = tmp_path / "shared.wav"
    tmp_wav.write_bytes(b"tmp")

    def _fake_run_beat_tracking(self, audio, sr, ah):
        return [], [], {}, [], str(tmp_wav)

    def _raise_meter_error(*args, **kwargs):
        raise RuntimeError("forced meter failure")

    monkeypatch.setattr(AnalysisEngine, "_run_beat_tracking", _fake_run_beat_tracking)
    monkeypatch.setattr(engine_module, "generate_hypotheses", _raise_meter_error)

    engine = AnalysisEngine()
    audio = np.zeros(22050, dtype=np.float32)

    with pytest.raises(RuntimeError, match="forced meter failure"):
        engine.analyze_audio(audio, sr=22050)

    assert not tmp_wav.exists()


def test_audio_hash_differs_for_same_name_different_content(tmp_path):
    """Cache key should include content, not only the filename stem."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    file_a = dir_a / "song.wav"
    file_b = dir_b / "song.wav"
    file_a.write_bytes(b"A" * 1024)
    file_b.write_bytes(b"B" * 1024)

    hash_a = AnalysisCache.audio_hash(str(file_a))
    hash_b = AnalysisCache.audio_hash(str(file_b))

    assert hash_a != hash_b
