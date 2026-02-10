"""Meter signal subpackage — each signal in its own module."""

from beatmeter.analysis.signals.downbeat_spacing import signal_downbeat_spacing
from beatmeter.analysis.signals.madmom_activation import signal_madmom_activation
from beatmeter.analysis.signals.onset_autocorrelation import signal_onset_autocorrelation
from beatmeter.analysis.signals.accent_pattern import signal_accent_pattern, compute_beat_energies
from beatmeter.analysis.signals.beat_periodicity import signal_beat_strength_periodicity
from beatmeter.analysis.signals.bar_tracking import signal_bar_tracking
from beatmeter.analysis.signals.sub_beat_division import signal_sub_beat_division

__all__ = [
    "signal_downbeat_spacing",
    "signal_madmom_activation",
    "signal_onset_autocorrelation",
    "signal_accent_pattern",
    "compute_beat_energies",
    "signal_beat_strength_periodicity",
    "signal_bar_tracking",
    "signal_sub_beat_division",
]
