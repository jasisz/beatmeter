"""Shared test fixtures for rhythm analyzer tests."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from beatmeter.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def generate_click_track(
    bpm: float,
    beats_per_bar: int,
    duration_seconds: float = 10.0,
    sr: int = 22050,
    accent_ratio: float = 2.0,
) -> np.ndarray:
    """Generate a synthetic click track with accented downbeats.

    Returns mono audio at the given sample rate.
    """
    n_samples = int(duration_seconds * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    beat_interval = 60.0 / bpm  # seconds per beat
    click_duration = 0.02  # 20ms click
    click_samples = int(click_duration * sr)

    # Create click sound (short sine burst with envelope)
    t_click = np.arange(click_samples) / sr
    click = np.sin(2 * np.pi * 1000 * t_click) * np.exp(-t_click * 100)

    beat = 0
    time = 0.0
    while time < duration_seconds:
        sample_pos = int(time * sr)
        is_downbeat = (beat % beats_per_bar) == 0
        amplitude = accent_ratio if is_downbeat else 1.0

        end = min(sample_pos + click_samples, n_samples)
        length = end - sample_pos
        if length > 0:
            audio[sample_pos:end] += click[:length] * amplitude

        time += beat_interval
        beat += 1

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


@pytest.fixture
def click_4_4():
    """Click track in 4/4 at 120 BPM."""
    return generate_click_track(bpm=120, beats_per_bar=4, duration_seconds=10)


@pytest.fixture
def click_3_4():
    """Click track in 3/4 at 100 BPM."""
    return generate_click_track(bpm=100, beats_per_bar=3, duration_seconds=10)


@pytest.fixture
def click_7_8():
    """Click track in 7/8 at 140 BPM (as eighth notes)."""
    return generate_click_track(bpm=140, beats_per_bar=7, duration_seconds=10)


@pytest.fixture
def click_5_4():
    """Click track in 5/4 at 110 BPM."""
    return generate_click_track(bpm=110, beats_per_bar=5, duration_seconds=10)
