"""Tests for meter hypothesis generation."""

import numpy as np

from beatmeter.analysis.models import Beat
from beatmeter.analysis.meter import generate_hypotheses
from beatmeter.analysis.signals.meter_net_features import (
    TOTAL_FEATURES,
    N_AUDIO_FEATURES,
    N_MERT_FEATURES,
    feature_groups,
)


def _make_beats(beats_per_bar: int, n_bars: int = 8, ibi: float = 0.5) -> list[Beat]:
    """Create synthetic beats with downbeats every beats_per_bar beats."""
    beats = []
    for bar in range(n_bars):
        for beat in range(beats_per_bar):
            t = (bar * beats_per_bar + beat) * ibi
            beats.append(Beat(
                time=t,
                is_downbeat=(beat == 0),
                strength=1.0 if beat == 0 else 0.6,
            ))
    return beats


def test_generate_hypotheses_returns_list():
    """generate_hypotheses should return valid hypotheses."""
    hypotheses, ambiguity = generate_hypotheses()

    assert len(hypotheses) > 0
    assert hypotheses[0].numerator > 0
    assert hypotheses[0].denominator > 0
    assert 0 <= hypotheses[0].confidence <= 1.0
    assert 0.0 <= ambiguity <= 1.0


def test_generate_hypotheses_have_descriptions():
    hypotheses, _ = generate_hypotheses()

    for h in hypotheses:
        assert h.description != ""


# ---------------------------------------------------------------------------
# MeterNet feature layout tests
# ---------------------------------------------------------------------------


def test_meter_net_total_features():
    """Total MeterNet features should be 2985 (audio + MERT)."""
    expected = N_AUDIO_FEATURES + N_MERT_FEATURES
    assert TOTAL_FEATURES == expected, f"Expected {expected}, got {TOTAL_FEATURES}"
    assert TOTAL_FEATURES == 2985, f"Expected 2985, got {TOTAL_FEATURES}"


def test_feature_groups_layout():
    """Feature groups should cover the full feature vector without gaps."""
    fg = feature_groups()
    prev_end = 0
    for name in ["audio", "mert"]:
        start, end = fg[name]
        assert start == prev_end, f"Gap before {name}: expected {prev_end}, got {start}"
        prev_end = end
    assert prev_end == TOTAL_FEATURES, f"Feature groups end at {prev_end}, expected {TOTAL_FEATURES}"
