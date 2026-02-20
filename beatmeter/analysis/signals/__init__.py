"""Meter signal subpackage — each signal in its own module."""

from beatmeter.analysis.signals.downbeat_spacing import signal_downbeat_spacing
from beatmeter.analysis.signals.onset_autocorrelation import signal_onset_autocorrelation
from beatmeter.analysis.signals.bar_tracking import signal_bar_tracking

__all__ = [
    "signal_downbeat_spacing",
    "signal_onset_autocorrelation",
    "signal_bar_tracking",
]
