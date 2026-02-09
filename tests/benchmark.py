#!/usr/bin/env python3
"""Unified benchmark runner for the rhythm analyzer.

Runs synthetic click tests + real audio tests, produces a summary table,
detects regressions against previous results, and saves JSON for tracking.

Run:  uv run python tests/benchmark.py
      uv run python tests/benchmark.py --save          # persist results
      uv run python tests/benchmark.py --verbose       # per-signal diagnostics
      uv run python tests/benchmark.py --category drums  # filter by category
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import numpy as np

# Suppress noisy library logging before any imports touch them
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
for _name in ("beatmeter", "numba", "madmom", "BeatNet", "PySoundFile"):
    logging.getLogger(_name).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.audio.loader import load_audio
from beatmeter.audio.preprocessing import preprocess
from beatmeter.config import settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
RESULTS_FILE = TESTS_DIR / "benchmark_results.json"
CACHE_DIR = FIXTURES_DIR / ".cache"

SR = settings.sample_rate

# ---------------------------------------------------------------------------
# Beat tracker result caching
# ---------------------------------------------------------------------------

_cache_enabled = True  # toggled by --no-cache
_engine_cache_enabled = True  # toggled by --no-cache


def _audio_hash(audio: np.ndarray) -> str:
    """Fast content hash for an audio array (for tracker-level caching)."""
    return hashlib.sha256(audio.tobytes()[:200_000]).hexdigest()[:16]


def _audio_hash_full(audio: np.ndarray) -> str:
    """Full content hash for an audio array (for engine-level caching)."""
    return hashlib.sha256(audio.tobytes()).hexdigest()[:20]


def _code_hash() -> str:
    """Hash of analysis source files for engine result cache invalidation."""
    analysis_dir = Path(__file__).resolve().parent.parent / "beatmeter" / "analysis"
    source_files = sorted(analysis_dir.glob("*.py"))
    h = hashlib.sha256()
    for f in source_files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# Computed once per run
_cached_code_hash: str | None = None


def _get_code_hash() -> str:
    global _cached_code_hash
    if _cached_code_hash is None:
        _cached_code_hash = _code_hash()
    return _cached_code_hash


def _engine_cache_path(full_audio_hash: str, code_hash: str) -> Path:
    """Return cache file path for a full engine result."""
    return CACHE_DIR / f"engine_{full_audio_hash}_{code_hash}.json"


def _load_engine_cache(full_audio_hash: str) -> dict | None:
    """Load cached engine result if available and code unchanged."""
    if not _engine_cache_enabled:
        return None
    path = _engine_cache_path(full_audio_hash, _get_code_hash())
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, KeyError):
        return None


def _save_engine_cache(full_audio_hash: str, result_dict: dict):
    """Save engine result to cache."""
    if not _engine_cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _engine_cache_path(full_audio_hash, _get_code_hash())
    path.write_text(json.dumps(result_dict))


def _beats_to_json(beats) -> list[dict]:
    """Serialize Beat objects to JSON-serializable dicts."""
    return [{"time": b.time, "is_downbeat": b.is_downbeat, "strength": b.strength}
            for b in beats]


def _beats_from_json(data: list[dict]):
    """Deserialize Beat objects from dicts."""
    from beatmeter.analysis.models import Beat
    return [Beat(time=d["time"], is_downbeat=d["is_downbeat"], strength=d["strength"])
            for d in data]


def _cache_path(tracker_name: str, audio_hash: str, suffix: str = "") -> Path:
    """Return cache file path for a tracker + audio hash."""
    fname = f"{tracker_name}_{audio_hash}{suffix}.json"
    return CACHE_DIR / fname


def _load_cached(tracker_name: str, audio_hash: str, suffix: str = ""):
    """Load cached beats if available. Returns list[Beat] or None."""
    if not _cache_enabled:
        return None
    path = _cache_path(tracker_name, audio_hash, suffix)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return _beats_from_json(data)
    except (json.JSONDecodeError, KeyError):
        return None


def _save_cache(tracker_name: str, audio_hash: str, beats, suffix: str = ""):
    """Save beat results to cache."""
    if not _cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(tracker_name, audio_hash, suffix)
    path.write_text(json.dumps(_beats_to_json(beats)))


def _save_bar_tracking_cache(audio_hash: str, beat_times_hash: str, scores: dict):
    """Save bar tracking scores to cache."""
    if not _cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path("bar_tracking", audio_hash, suffix=f"_{beat_times_hash}")
    path.write_text(json.dumps({str(k): v for k, v in scores.items()}))


def _load_bar_tracking_cache(audio_hash: str, beat_times_hash: str) -> dict | None:
    """Load cached bar tracking scores if available."""
    if not _cache_enabled:
        return None
    path = _cache_path("bar_tracking", audio_hash, suffix=f"_{beat_times_hash}")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        # Convert string keys back to tuple keys
        return {eval(k): v for k, v in data.items()}
    except (json.JSONDecodeError, SyntaxError):
        return None


def _save_resnet_cache(audio_hash: str, scores: dict):
    """Save ResNet meter scores to cache."""
    if not _cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path("resnet", audio_hash)
    path.write_text(json.dumps({str(k): v for k, v in scores.items()}))


def _load_resnet_cache(audio_hash: str) -> dict | None:
    """Load cached ResNet meter scores if available."""
    if not _cache_enabled:
        return None
    path = _cache_path("resnet", audio_hash)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return {eval(k): v for k, v in data.items()}
    except (json.JSONDecodeError, SyntaxError):
        return None


def install_cache_wrappers():
    """Monkey-patch beat tracking functions with caching wrappers."""
    import beatmeter.analysis.beat_tracking as bt
    import beatmeter.analysis.engine as eng

    _orig_beatnet = bt.track_beats_beatnet
    _orig_madmom = bt.track_beats_madmom
    _orig_librosa = bt.track_beats_librosa
    _orig_beat_this = bt.track_beats_beat_this

    @wraps(_orig_beatnet)
    def cached_beatnet(audio, sr=22050, tmp_path=None):
        h = _audio_hash(audio)
        cached = _load_cached("beatnet", h)
        if cached is not None:
            return cached
        result = _orig_beatnet(audio, sr, tmp_path=tmp_path)
        _save_cache("beatnet", h, result)
        return result

    @wraps(_orig_madmom)
    def cached_madmom(audio, sr=22050, beats_per_bar=4, tmp_path=None):
        h = _audio_hash(audio)
        cached = _load_cached("madmom", h, suffix=f"_bpb{beats_per_bar}")
        if cached is not None:
            return cached
        result = _orig_madmom(audio, sr, beats_per_bar=beats_per_bar, tmp_path=tmp_path)
        _save_cache("madmom", h, result, suffix=f"_bpb{beats_per_bar}")
        return result

    @wraps(_orig_librosa)
    def cached_librosa(audio, sr=22050):
        h = _audio_hash(audio)
        cached = _load_cached("librosa", h)
        if cached is not None:
            return cached
        result = _orig_librosa(audio, sr)
        _save_cache("librosa", h, result)
        return result

    @wraps(_orig_beat_this)
    def cached_beat_this(audio, sr=22050, tmp_path=None):
        h = _audio_hash(audio)
        cached = _load_cached("beat_this", h)
        if cached is not None:
            return cached
        result = _orig_beat_this(audio, sr, tmp_path=tmp_path)
        _save_cache("beat_this", h, result)
        return result

    # Patch both the module and the engine's imports
    bt.track_beats_beatnet = cached_beatnet
    bt.track_beats_madmom = cached_madmom
    bt.track_beats_librosa = cached_librosa
    bt.track_beats_beat_this = cached_beat_this
    eng.track_beats_beatnet = cached_beatnet
    eng.track_beats_madmom = cached_madmom
    eng.track_beats_librosa = cached_librosa
    eng.track_beats_beat_this = cached_beat_this

    # Cache bar tracking (Signal 7) results
    import beatmeter.analysis.meter as meter_mod
    _orig_bar_tracking = meter_mod.signal_bar_tracking

    @wraps(_orig_bar_tracking)
    def cached_bar_tracking(audio, sr, beat_times_array, meters_to_test=None):
        h = _audio_hash(audio)
        bt_hash = hashlib.sha256(beat_times_array.tobytes()).hexdigest()[:12]
        cached = _load_bar_tracking_cache(h, bt_hash)
        if cached is not None:
            return cached
        result = _orig_bar_tracking(audio, sr, beat_times_array, meters_to_test)
        _save_bar_tracking_cache(h, bt_hash, result)
        return result

    meter_mod.signal_bar_tracking = cached_bar_tracking

    # Cache ResNet (Signal 8) results
    try:
        import beatmeter.analysis.resnet_signal as resnet_mod
        _orig_resnet = resnet_mod.signal_resnet_meter

        @wraps(_orig_resnet)
        def cached_resnet(audio, sr):
            h = _audio_hash(audio)
            cached = _load_resnet_cache(h)
            if cached is not None:
                return cached
            result = _orig_resnet(audio, sr)
            _save_resnet_cache(h, result)
            return result

        resnet_mod.signal_resnet_meter = cached_resnet
    except ImportError:
        pass  # ResNet signal module not available

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    category: str
    expected_meter: list[tuple[int, int]]  # list of acceptable meters
    expected_bpm_range: tuple[float, float]
    # Either a path to a file OR a numpy audio array
    filepath: str | None = None
    audio: np.ndarray | None = field(default=None, repr=False)
    max_duration: float = 30.0


@dataclass
class TestResult:
    name: str
    category: str
    meter_correct: bool
    tempo_correct: bool
    detected_meter: str
    detected_bpm: float
    expected_meters: list[str]
    expected_bpm_range: list[float]
    elapsed_seconds: float = 0.0
    beatnet_raw_meter: str = "N/A"
    beatnet_raw_correct: bool = False


# ---------------------------------------------------------------------------
# BeatNet raw meter extraction
# ---------------------------------------------------------------------------


def get_beatnet_raw_meter(beats) -> tuple[int, int] | None:
    """Extract meter directly from BeatNet downbeat spacing."""
    db_indices = [i for i, b in enumerate(beats) if b.is_downbeat]
    if len(db_indices) < 2:
        return None
    spacings = [db_indices[i + 1] - db_indices[i] for i in range(len(db_indices) - 1)]
    if not spacings:
        return None
    most_common = Counter(spacings).most_common(1)[0][0]
    return (most_common, 4)


# ---------------------------------------------------------------------------
# Synthetic audio generation
# ---------------------------------------------------------------------------


def generate_click(bpm, beats_per_bar, duration=15.0, accent_ratio=3.0, sr=SR):
    """Generate a click track with accented downbeats."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    beat_interval = 60.0 / bpm
    click_dur = 0.015
    click_samples = int(click_dur * sr)
    t_click = np.arange(click_samples) / sr

    click_accent = np.sin(2 * np.pi * 1200 * t_click) * np.exp(-t_click * 120)
    click_normal = np.sin(2 * np.pi * 800 * t_click) * np.exp(-t_click * 120)

    beat = 0
    t = 0.0
    while t < duration:
        pos = int(t * sr)
        is_downbeat = (beat % beats_per_bar) == 0
        click = click_accent if is_downbeat else click_normal
        amplitude = accent_ratio if is_downbeat else 1.0
        end = min(pos + click_samples, n_samples)
        length = end - pos
        if length > 0:
            audio[pos:end] += click[:length] * amplitude
        t += beat_interval
        beat += 1

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def generate_compound_click(bpm, groups, duration=15.0, sr=SR):
    """Generate a compound meter click (e.g. 7/8 = 2+2+3)."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    unit_interval = 60.0 / bpm
    click_dur = 0.015
    click_samples = int(click_dur * sr)
    t_click = np.arange(click_samples) / sr

    click_accent = np.sin(2 * np.pi * 1200 * t_click) * np.exp(-t_click * 120) * 3.0
    click_group = np.sin(2 * np.pi * 1000 * t_click) * np.exp(-t_click * 120) * 2.0
    click_normal = np.sin(2 * np.pi * 800 * t_click) * np.exp(-t_click * 120) * 1.0

    t = 0.0
    while t < duration:
        beat_in_bar = 0
        for gi, group_size in enumerate(groups):
            for beat_in_group in range(group_size):
                pos = int(t * sr)
                if pos >= n_samples:
                    break
                if beat_in_bar == 0:
                    click = click_accent
                elif beat_in_group == 0:
                    click = click_group
                else:
                    click = click_normal
                end = min(pos + click_samples, n_samples)
                length = end - pos
                if length > 0:
                    audio[pos:end] += click[:length]
                t += unit_interval
                beat_in_bar += 1
            if int(t * sr) >= n_samples:
                break

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


# ---------------------------------------------------------------------------
# Edge-case synthetic generators
# ---------------------------------------------------------------------------


def generate_short_click(bpm, beats_per_bar, duration=4.0, sr=SR):
    """Very short clip (minimum viable input)."""
    return generate_click(bpm, beats_per_bar, duration=duration, sr=sr)


def generate_quiet_click(bpm, beats_per_bar, duration=15.0, amplitude=0.05, sr=SR):
    """Very quiet audio (low amplitude)."""
    audio = generate_click(bpm, beats_per_bar, duration=duration, sr=sr)
    audio *= amplitude
    return audio


def generate_tempo_change(bpm_start, bpm_end, beats_per_bar, duration=20.0, sr=SR):
    """Audio with linear tempo ramp."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    click_dur = 0.015
    click_samples = int(click_dur * sr)
    t_click = np.arange(click_samples) / sr

    click_accent = np.sin(2 * np.pi * 1200 * t_click) * np.exp(-t_click * 120)
    click_normal = np.sin(2 * np.pi * 800 * t_click) * np.exp(-t_click * 120)

    beat = 0
    t = 0.0
    while t < duration:
        # Linearly interpolate BPM
        progress = t / duration
        current_bpm = bpm_start + (bpm_end - bpm_start) * progress
        beat_interval = 60.0 / current_bpm

        pos = int(t * sr)
        is_downbeat = (beat % beats_per_bar) == 0
        click = click_accent if is_downbeat else click_normal
        amplitude = 3.0 if is_downbeat else 1.0
        end = min(pos + click_samples, n_samples)
        length = end - pos
        if length > 0:
            audio[pos:end] += click[:length] * amplitude
        t += beat_interval
        beat += 1

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def generate_silence_gaps(bpm, beats_per_bar, duration=15.0, sr=SR):
    """Click track with 2-second silence gaps every 5 seconds."""
    audio = generate_click(bpm, beats_per_bar, duration=duration, sr=sr)
    gap_dur = int(2.0 * sr)
    interval = int(5.0 * sr)
    pos = interval
    while pos + gap_dur < len(audio):
        audio[pos:pos + gap_dur] = 0.0
        pos += interval + gap_dur
    return audio


# ---------------------------------------------------------------------------
# Fixture-file metadata catalogue
# ---------------------------------------------------------------------------

# Maps filename -> (category, acceptable_meters, bpm_range).
# Files discovered on disk but not in the catalogue are assigned "unknown".

FIXTURE_CATALOGUE: dict[str, tuple[str, list[tuple[int, int]], tuple[float, float]]] = {
    # Drums 4/4
    "rock_beat.ogg":       ("drums",        [(4, 4), (2, 4)], (80, 180)),
    "drum_beat.ogg":       ("drums",        [(4, 4), (2, 4)], (80, 180)),
    "19_16_drum_beat.ogg": ("drums",        [(4, 4), (2, 4)], (80, 200)),
    "jazz_ride.ogg":       ("drums",        [(4, 4), (2, 4)], (80, 200)),
    "shuffle.ogg":         ("drums",        [(4, 4), (2, 4)], (80, 200)),
    "blast_beat.ogg":      ("drums",        [(4, 4), (2, 4)], (150, 300)),
    "dubstep_drums.ogg":   ("drums",        [(4, 4), (2, 4)], (60, 160)),
    "reggae_one_drop.ogg": ("drums",        [(4, 4), (2, 4)], (60, 160)),
    "bossa_nova.ogg":      ("drums",        [(4, 4), (2, 4)], (70, 170)),
    "djembe.ogg":          ("drums",        [(4, 4), (2, 4), (3, 4), (6, 4)], (60, 200)),
    "drum_cadence_a.ogg":  ("drums",        [(4, 4), (2, 4)], (80, 180)),
    "drum_cadence_b.ogg":  ("drums",        [(4, 4), (2, 4)], (80, 180)),

    # Middle Eastern
    "ayub.ogg":            ("middle_eastern", [(2, 4), (4, 4)], (80, 200)),
    "baladi.ogg":          ("middle_eastern", [(4, 4), (2, 4)], (60, 160)),
    "maksum.ogg":          ("middle_eastern", [(4, 4), (2, 4)], (60, 160)),
    "malfuf.ogg":          ("middle_eastern", [(2, 4), (4, 4)], (80, 200)),
    "saidi.ogg":           ("middle_eastern", [(4, 4), (2, 4)], (60, 160)),

    # Waltzes 3/4
    "greensleeves.ogg":         ("waltz", [(3, 4), (6, 4), (6, 8)], (60, 170)),
    "midnight_waltz.ogg":       ("waltz", [(3, 4)], (80, 200)),
    "joplin_waltz.ogg":         ("waltz", [(3, 4)], (80, 200)),
    "blowing_bubbles_waltz.ogg": ("waltz", [(3, 4)], (80, 200)),
    "waltz_stefan.ogg":         ("waltz", [(3, 4)], (80, 200)),

    # Classical (3/4 family: waltzes, minuets, mazurkas, sarabandes)
    "chopin_waltz.ogg":         ("classical", [(3, 4)], (100, 200)),
    "blue_danube.ogg":          ("classical", [(3, 4)], (100, 220)),
    "bach_minuet.ogg":          ("classical", [(3, 4)], (80, 160)),
    "minuet_beethoven.ogg":     ("classical", [(3, 4)], (80, 160)),
    "minuet_paderewski.ogg":    ("classical", [(3, 4)], (60, 160)),
    "mazurka_chopin_op7.ogg":   ("classical", [(3, 4)], (100, 200)),
    "sarabande_bach.ogg":       ("classical", [(3, 4)], (40, 160)),
    "sarabande_handel.oga":     ("classical", [(3, 4)], (40, 160)),

    # Barcarolles / Sicilianas (6/8)
    "bach_siciliana.ogg":       ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),
    "barcarolle_chopin.ogg":    ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),
    "barcarolle_offenbach.ogg": ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),

    # Marches 4/4 or 2/4
    "erika_march.ogg":          ("march", [(4, 4), (2, 4)], (100, 140)),
    "march_grandioso.ogg":      ("march", [(4, 4), (2, 4)], (100, 140)),
    "march_military.ogg":       ("march", [(4, 4), (2, 4)], (100, 140)),
    "march_suffrage.ogg":       ("march", [(4, 4), (2, 4)], (100, 140)),

    # Polkas 2/4
    "polka_kathi.ogg":          ("polka", [(2, 4), (4, 4)], (80, 180)),
    "polka_pixel_peeker.ogg":   ("polka", [(2, 4), (4, 4)], (100, 200)),
    "polka_smetana.ogg":        ("polka", [(2, 4), (4, 4)], (100, 180)),
    "polka_tritsch_tratsch.ogg": ("polka", [(2, 4), (4, 4)], (100, 200)),

    # Jigs 6/8 / 9/8
    "jig_doethion.ogg":         ("jig", [(6, 8), (6, 4), (3, 4)], (80, 160)),
    "jigs_gwerinos.ogg":        ("jig", [(6, 8), (6, 4), (3, 4)], (80, 160)),
    "roxys_birthday_jigs.ogg":  ("jig", [(6, 8), (6, 4), (3, 4)], (80, 160)),
    "slip_jigs.ogg":            ("jig", [(9, 8), (9, 4), (3, 4), (6, 4), (6, 8)], (80, 160)),
    "irish_reel_mountain_road.ogg": ("jig", [(4, 4), (2, 4)], (100, 180)),

    # Tangos 4/4
    "tango_albeniz.ogg":        ("tango", [(4, 4), (2, 4)], (50, 120)),
    "tango_argentino.ogg":      ("tango", [(4, 4), (2, 4)], (50, 140)),

    # Tarantellas 6/8
    "bushwick_tarantella.oga":      ("tarantella", [(6, 8), (6, 4), (3, 4)], (80, 180)),
    "celebre_tarantella.ogg":       ("tarantella", [(6, 8), (6, 4), (3, 4)], (80, 200)),
    "tarantella_choir.ogg":         ("tarantella", [(6, 8), (6, 4), (3, 4)], (80, 200)),
    "tarantella_napoletana.ogg":    ("tarantella", [(6, 8), (6, 4), (3, 4)], (100, 200)),
    "tarantella_welsh_tenors.ogg":  ("tarantella", [(6, 8), (6, 4), (3, 4)], (80, 200)),

    # Blues 4/4
    "lost_train_blues.ogg":     ("blues", [(4, 4), (2, 4)], (60, 160)),
    "blues_guitar.ogg":         ("blues", [(4, 4), (12, 8), (2, 4)], (60, 160)),

    # Blues (new)
    "blues_12barblues002.ogg": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_12barbluestutorial.ogg": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_2009-05-29medboogie.ogg": ("blues", [(4, 4), (12, 8), (2, 4)], (40, 240)),
    "blues_2009-05-30fastboogie.ogg": ("blues", [(4, 4), (12, 8), (2, 4)], (40, 240)),
    "blues_2009-05-30fastshuffle.ogg": ("blues", [(4, 4), (12, 8), (2, 4)], (40, 240)),
    "blues_31st_street_blues_-_hendersons_club_alab.mp3": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_acousticshuffle.ogg": ("blues", [(4, 4), (12, 8), (2, 4)], (40, 240)),
    "blues_aleshavlicek-hodinky.ogg": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_axesshuffle.ogg": ("blues", [(4, 4), (12, 8), (2, 4)], (40, 240)),
    "blues_ballade_of_july_mikees_blues_express_mic.ogg": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_bluesjam1.ogg": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_dreaming_blues_blues_piano_roll_played_b.oga": ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_taint_nobodys_busness_if_i_do.ogg": ("blues", [(4, 4), (2, 4)], (40, 240)),

    # Drums (new)
    "drums_4_cavacha_variations.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_dembow_perreo_classic_dembow_and_rich_de.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_emu_orbit_9090_v2_-_dance_beat_patterns.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_inverted_ride_pattern.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_karatchi_ejemplo.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_mazhar_demo.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_no_overheads.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),
    "drums_overheads.ogg": ("drums", [(4, 4), (2, 4)], (60, 220)),

    # Folk
    "folk_ada_jones_und_len_spencer_return_of_the_.mp3": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_aiken_drum.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_anonimo_-_the_house_of_the_rising_sun.ogg": ("folk", [(3, 4), (6, 8), (6, 4)], (60, 220)),
    "folk_arabicqanunsample.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_arkansas_traveler.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_bachn_ringing_sark_folk_festival_2011.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_barnyards_of_delgaty.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_bevagna_festa.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_blow_the_man_down.ogg": ("folk", [(3, 4), (6, 8)], (60, 220)),
    "folk_bonnie_dundee.ogg": ("folk", [(4, 4), (2, 4)], (60, 220)),

    # Jazz
    "jazz_afghanistan_-_fox_trot-_princes_dance_or.ogg": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_afghanistan_by_harry_donnelly_and_willia.mp3": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_afghanistan_fox-trot_by_charles_a_prince.mp3": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_afghanistan_performed_by_tuxedo_syncopat.ogg": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_afghanistan_sung_by_the_premier-american.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_bluin_the_black_keys_by_arthur_schutt.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_brejeiro_by_ernesto_nazareth.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_clementine_by_jean_goldkette_his_orchest.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_crooked_notes_-_jean_paques.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_crooked_notes_by_jean_paques.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_cupids_garden_intermezzo_-_william_h_rei.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_daffy-dill_by_edith_althoff.opus": ("jazz", [(4, 4), (2, 4)], (60, 220)),

    # March (new)
    "march_01.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_01_on_brave_old_army_team.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_07_old_grads_march.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_2nd_regiment_connecticut_national_guard_.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_a_warrior_bold_-_concert_band_-_united_s.mp3": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_adjutants_call_cmtc_march_-_concert_band.mp3": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_adjutants_call_to_honor_with_dignity_-_c.mp3": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_anchor_star_by_john_philip_sousa_perform.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_ataque_de_uchumayo_-_banda_federal_de_ar.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_avenida_de_las_camelias.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_badonviller_-_us_marine_band.ogg": ("march", [(4, 4), (2, 4)], (80, 150)),

    # Mazurka
    "mazurka_chopin_-_mazurka_no_10_in_b-flat_major_o.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_12_in_a-flat_major_o.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_13_in_a_minor_op_17_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_15_in_c_major_op_24_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_17_in_b-flat_minor_o.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_18_in_c_minor_op_30_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_20_in_d-flat_major_o.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_21_in_c-sharp_minor_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_27_in_e_minor_op_41_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_28_in_b_major_op_41_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_30_in_g_major_op_50_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_38_in_f-sharp_minor_.flac": ("mazurka", [(3, 4)], (80, 220)),

    # Middle Eastern (new)
    "mideast_majida_el_roumi_-_song_sample.ogg": ("middle_eastern", [(4, 4), (2, 4)], (60, 200)),
    "mideast_mohamed_el_fares_mala_ghounouzahabi_1.ogg": ("middle_eastern", [(4, 4), (2, 4)], (60, 200)),
    "mideast_ya-habeby-yamuhammad.ogg": ("middle_eastern", [(4, 4), (2, 4)], (60, 200)),

    # Ragtime
    "ragtime_10th_interval_rag_by_harry_ruby.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_7_come_11_-_just_rag_by_paul_j_deitsch.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_bag_of_rags_by_william_mckanlass.opus": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_black_bawl_by_harry_c_thompson.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_certain_party_rag_by_tom_kelley.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_cotton_patch_by_charles_tyler.opus": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_little_bit_of_rag_by_paul_c_pratt.opus": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_night_on_the_levee_by_theodore_haverme.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_ragtime_nightmare_by_tom_turpin.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_tennesse_jubilee_by_thomas_e_broady.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),

    # Tango (new)
    "tango_bandonen.ogg": ("tango", [(4, 4), (2, 4)], (40, 160)),
    "tango_canaro-membrives-carasucia.ogg": ("tango", [(4, 4), (2, 4)], (40, 160)),
    "tango_carlos_gardel-_pobre_mi_madre.ogg": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlos_gardel-ay_ay_ay.ogg": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlos_gardel-desdichas.ogg": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlos_gardel_-_congojas.flac": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlos_gardel_-_flor_de_fango.ogg": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlos_gardel_-loca.ogg": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlosgardel-minochetriste.ogg": ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_cello_encores_john_michel-mats_lidstrom_.ogg": ("tango", [(4, 4), (2, 4)], (40, 160)),

    # Waltz (new)
    "waltz_bethena_by_scott_joplin.opus": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_grande_valse_brillante_in_e_fla.ogg": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_in_d-flat_op-64_no-1.wav": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_in_e_minor_b_56.mp3": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_no_11_in_g-flat_major_op_.flac": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_no_15_in_e_major_b_44.flac": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_no_16_in_a-flat_major_b_2.flac": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_no_17_in_e-flat_major_b_4.flac": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_no_18_in_e-flat_b_133.flac": ("waltz", [(3, 4)], (80, 220)),
    "waltz_chopin_-_waltz_no_19_in_a_minor_b_150.flac": ("waltz", [(3, 4)], (80, 220)),

    # Barcarolle (round 2)
    "barcarolle_bach_-_partita_for_solo_flute_-_modern_f.ogg": ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),
    "barcarolle_bach_-_partita_for_solo_flute_-_traverso.ogg": ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),
    "barcarolle_handel_-_suite_vol_2_no_4_in_d_minor_hwv.oga": ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),
    "barcarolle_lloyd-suite_4_-_sarabande.ogg":          ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),
    "barcarolle_octatonic_bars_from_sarabande_from_engli.wav": ("barcarolle", [(6, 8), (6, 4), (3, 4)], (30, 200)),

    # Blues (round 2)
    "blues_believe.ogg":                                 ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_blues-cerneho-domu.ogg":                      ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_blues_no1_by_michael_huber.ogg":              ("blues", [(4, 4), (2, 4)], (40, 240)),
    "blues_blues_slow_by_michael_huber.ogg":             ("blues", [(4, 4), (12, 8), (2, 4)], (40, 240)),

    # Classical (round 2)
    "classical_sarabande_from_harpsichord_suite_hwv_437.opus": ("classical", [(3, 4)], (40, 160)),
    "classical_sarabande_cortada.mp3":                   ("classical", [(3, 4)], (40, 160)),
    "classical_satie_sarabande_3_chord_sequence.ogg":    ("classical", [(3, 4)], (40, 160)),
    "classical_soundtrack_organ_-_handels_keyboard_suit.ogg": ("classical", [(3, 4)], (40, 160)),
    "classical_vosssarabande.ogg":                       ("classical", [(3, 4)], (40, 160)),

    # Drums (round 2)
    "drums_rock_beat_ride_cymbal.ogg":                   ("drums", [(4, 4), (2, 4)], (60, 220)),

    # Folk (round 2)
    "folk_02_-_breezy_may_acoustic.ogg":                 ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_03_-_bluebell_acoustic.ogg":                   ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_05_-_cinus_laurent_-_fte_au_chteau.ogg":       ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_06_-_cinus_laurent_-_cline_valse.oga":         ("folk", [(3, 4)], (60, 220)),
    "folk_azerbaijan_folk_dance_naz_eleme.ogg":          ("folk", [(6, 8)], (60, 220)),
    "folk_azerbaijan_folk_dance_uzundara.ogg":           ("folk", [(6, 8)], (60, 220)),
    "folk_breakdown.ogg":                                ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_brian_borus_march.ogg":                        ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_canzone-bambini-boca.ogg":                     ("folk", [(4, 4), (2, 4)], (60, 220)),
    "folk_chasing_dawn.mp3":                             ("folk", [(4, 4), (2, 4)], (60, 220)),

    # Jazz (round 2)
    "jazz_afghanistan_performed_by_the_lopez_and_h.ogg": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_diving_darlings_by_al_j_markgraf.oga":         ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_dont_mind_the_rain_robert_english_parlop.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_gay_birds_by_edward_claypoole.oga":            ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_georgie_porgie_by_billy_mayerl_and_geral.mp3": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_georgie_porgie_sung_by_rudy_valle_with_c.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),
    "jazz_how_ya_gonna_keep_em_down_on_the_farm_by.oga": ("jazz", [(4, 4), (2, 4)], (60, 220)),

    # Jig (round 2)
    "jig_1-07_the_irish_ballad.mp3":                     ("jig", [(4, 4), (2, 4)], (60, 200)),
    "jig_bwv816_gigue-french_suites_no5_bach_play.ogg":  ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_bach-french-suite-3-gigue.ogg":                 ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_bach-french-suite-6-gigue-bwv_817.ogg":         ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_bach_-_cello_suite_no_1_in_g_major_bwv_1.ogg":  ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_bach_cello_suite_bwv_1008_gigue.ogg":           ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_believe_me_if_all_those_endearing_young_.mp3":  ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_bodhran-improvisation_hinnerk-ruemenapf.ogg":   ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),
    "jig_brian_borus_march_-_us_marine_band.ogg":        ("jig", [(4, 4), (2, 4)], (60, 200)),
    "jig_broken_hands_improvisation.ogg":                ("jig", [(6, 8), (6, 4), (3, 4)], (60, 200)),

    # March (round 2)
    "march_1-01.ogg":                                    ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_basic_outdoor_parade_sequence_-_ceremoni.mp3": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_belfords_carnival.ogg":                       ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_bombasto_-_concert_band_-_united_states_.mp3": ("march", [(4, 4), (2, 4)], (80, 150)),
    "march_british_grenadiers.ogg":                      ("march", [(4, 4), (2, 4)], (80, 150)),

    # Mazurka (round 2)
    "mazurka_adam_tarnowski_-_fik_mik_mazur_2-gi_z_ko.wav": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka-op-50-no-3.ogg":           ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_16_in_a-flat_major_o.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_26_in_c-sharp_minor_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_33_in_b_major_op_56_.flac": ("mazurka", [(3, 4)], (80, 220)),
    "mazurka_chopin_-_mazurka_no_34_in_c_major_op_56_.flac": ("mazurka", [(3, 4)], (80, 220)),

    # Polka (round 2)
    "polka_anonimo_skkijrven_polkka.ogg":                ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_carlo_curti_-_la_tipica.ogg":                 ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_chiquinha_gonzaga_-_sultana_1908.ogg":        ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_cruzes_minha_prima_agenor_bens_joaquim_c.ogg": ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_double_polka.mp3":                            ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_dvorak_polka.mp3":                            ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_four_beers_polka.mp3":                        ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_glee_club_polka.mp3":                         ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_ievan_polkka_short_parody.ogg":               ("polka", [(2, 4), (4, 4)], (80, 220)),
    "polka_jennylind.ogg":                               ("polka", [(2, 4), (4, 4)], (80, 220)),

    # Ragtime (round 2)
    "ragtime_a_cotton_patch_by_charles_a_tyler.oga":     ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_a_tennessee_jubilee_by_thomas_e_broady.oga": ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_aint_i_lucky_by_bess_rudisill.oga":         ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_american_boy_by_julia_adams_turley.opus":   ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_angel_food_rag_by_albert_f_marzian.oga":    ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_anoma_by_ford_dabney.opus":                 ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_arcadia_by_luella_lockwood_moore.oga":      ("ragtime", [(4, 4), (2, 4)], (60, 200)),
    "ragtime_automobile_by_rose_de_haven.opus":          ("ragtime", [(4, 4), (2, 4)], (60, 200)),

    # Reggae (round 2)
    "reggae_b-roll.mp3":                                 ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_beach_party.mp3":                            ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_dub_eastern.mp3":                            ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_dub_feral.mp3":                              ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_easy_jam.mp3":                               ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_firmament.mp3":                              ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_gonna_start.mp3":                            ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_gonna_start_v2.mp3":                         ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_maccary_bay.mp3":                            ("reggae", [(4, 4), (2, 4)], (60, 160)),
    "reggae_mandeville.mp3":                             ("reggae", [(4, 4), (2, 4)], (60, 160)),

    # Samba (round 2)
    "samba_344211_giomilko_c-major-9-bossa-nova-gui.wav": ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_609562_migfus20_background-music.ogg":        ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_agora_cinza-mario_reis-bide_e_maral1933.ogg": ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_agradeco_e_dou_respeito.ogg":                 ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_axe_bahia.ogg":                               ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_benzinho.ogg":                                ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_cachorro_vira-lata_-_carmen_miranda.opus":    ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_carmen_miranda_e_mrio_reis_-_alo_alo.ogg":    ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_extrait_dune_samba_funk_joue_par_une_bat.ogg": ("samba", [(2, 4), (4, 4)], (60, 200)),
    "samba_ginga-ginga.ogg":                             ("samba", [(2, 4), (4, 4)], (60, 200)),

    # Swing (round 2)
    "swing_acid_trumpet.mp3":                            ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_airport_lounge.mp3":                          ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_apero_hour.mp3":                              ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_as_i_figure.mp3":                             ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_awesome_call.mp3":                            ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_backbay_lounge.mp3":                          ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_backed_vibes.mp3":                            ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_bass_soli.mp3":                               ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_bass_vibes.mp3":                              ("swing", [(4, 4), (2, 4)], (80, 220)),
    "swing_bass_walker.mp3":                             ("swing", [(4, 4), (2, 4)], (80, 220)),

    # Tango (round 2)
    "tango_carlos_gardel_-_pobre_amigo.flac":            ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_carlos_gardel_-_pobre_madrecita.flac":        ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_ciudad_perdida.ogg":                          ("tango", [(4, 4), (2, 4)], (40, 160)),
    "tango_el_irresistible_by_lorenzo_logatti.opus":     ("tango", [(4, 4), (2, 4)], (40, 160)),
    "tango_frank_ferera_helen_louise_-_hawaiian_por.ogg": ("tango", [(4, 4), (2, 4)], (40, 160)),
    "tango_gardelrazzano-amamemucho.ogg":                ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_gardel-razzano-el_carretero.ogg":             ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_gardel_-la_maanita.ogg":                      ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_gardel_-_me_dejaste.ogg":                     ("tango", [(4, 4), (2, 4), (3, 4)], (40, 160)),
    "tango_gobbi_alfredo_y_flora_-_el_criollo_falsi.ogg": ("tango", [(4, 4), (2, 4)], (40, 160)),

    # Waltz (round 2)
    "waltz_chopin_minute_waltz.ogg":                     ("waltz", [(3, 4)], (80, 220)),
    "waltz_een_wals_gespeeld_door_een_paillard_spee.ogg": ("waltz", [(3, 4)], (80, 220)),
    "waltz_falena_grupo_chiquinha_gonzaga_1913.ogg":     ("waltz", [(3, 4)], (80, 220)),
    "waltz_from_ravel_la_valse_01.wav":                  ("waltz", [(3, 4)], (80, 220)),
    "waltz_homocord-4-3035-tc-466.ogg":                  ("waltz", [(3, 4)], (80, 220)),
    "waltz_im_drifting_back_to_dreamland_vernon_dal.opus": ("waltz", [(3, 4)], (80, 220)),
    "waltz_internationl_childrens_day_gifts_-_1_sta.ogg": ("waltz", [(3, 4)], (80, 220)),
}


# ---------------------------------------------------------------------------
# Test-case builders
# ---------------------------------------------------------------------------


def build_synthetic_tests() -> list[TestCase]:
    """Synthetic click-track tests with known ground truth."""
    tests = []

    # Standard meters
    for bpm, bpb, label in [
        (120, 4, "4/4 @ 120"),
        (90,  4, "4/4 @ 90"),
        (160, 4, "4/4 @ 160"),
        (100, 3, "3/4 @ 100"),
        (140, 3, "3/4 @ 140 (fast waltz)"),
        (120, 2, "2/4 @ 120 (march)"),
        (80,  6, "6/4 @ 80"),
    ]:
        tests.append(TestCase(
            name=f"Synth: {label}",
            category="synthetic",
            expected_meter=[(bpb, 4)],
            expected_bpm_range=(bpm * 0.90, bpm * 1.10),
            audio=generate_click(bpm, bpb),
        ))

    # Odd meters
    for bpm, bpb in [(110, 5), (100, 7)]:
        tests.append(TestCase(
            name=f"Synth: {bpb}/4 @ {bpm}",
            category="synthetic",
            expected_meter=[(bpb, 4)],
            expected_bpm_range=(bpm * 0.50, bpm * 1.10),  # wide range; tempo halving is common
            audio=generate_click(bpm, bpb),
        ))

    # Compound meters
    tests.append(TestCase(
        name="Synth: 7/8 (2+2+3) @ 200",
        category="synthetic",
        expected_meter=[(7, 8), (7, 4)],
        expected_bpm_range=(90, 220),
        audio=generate_compound_click(200, [2, 2, 3]),
    ))
    tests.append(TestCase(
        name="Synth: 5/8 (3+2) @ 220",
        category="synthetic",
        expected_meter=[(5, 8), (5, 4)],
        expected_bpm_range=(100, 240),
        audio=generate_compound_click(220, [3, 2]),
    ))

    # 6/8 compound (two groups of 3)
    tests.append(TestCase(
        name="Synth: 6/8 (3+3) @ 180",
        category="synthetic",
        expected_meter=[(6, 8), (6, 4), (3, 4)],
        expected_bpm_range=(80, 200),
        audio=generate_compound_click(180, [3, 3]),
    ))

    # 12/8 compound (four groups of 3)
    tests.append(TestCase(
        name="Synth: 12/8 (3+3+3+3) @ 150",
        category="synthetic",
        expected_meter=[(12, 8), (4, 4)],
        expected_bpm_range=(40, 160),
        audio=generate_compound_click(150, [3, 3, 3, 3]),
    ))

    # 9/8 compound (three groups of 3)
    tests.append(TestCase(
        name="Synth: 9/8 (3+3+3) @ 170",
        category="synthetic",
        expected_meter=[(9, 8), (9, 4), (3, 4)],
        expected_bpm_range=(50, 180),
        audio=generate_compound_click(170, [3, 3, 3]),
    ))

    # 3/8 fast
    tests.append(TestCase(
        name="Synth: 3/8 fast @ 240",
        category="synthetic",
        expected_meter=[(3, 8), (3, 4), (6, 8)],
        expected_bpm_range=(100, 260),
        audio=generate_click(240, 3),
    ))

    # Very slow 4/4
    tests.append(TestCase(
        name="Synth: 4/4 slow @ 55",
        category="synthetic",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(45, 65),
        audio=generate_click(55, 4, duration=25.0),
    ))

    # Very fast 4/4
    tests.append(TestCase(
        name="Synth: 4/4 fast @ 200",
        category="synthetic",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(180, 220),
        audio=generate_click(200, 4),
    ))

    return tests


def build_edge_case_tests() -> list[TestCase]:
    """Edge-case tests: short clips, quiet audio, tempo changes, silence gaps."""
    tests = []

    # Short clips (minimum viable input)
    tests.append(TestCase(
        name="Edge: 4/4 short clip (4s)",
        category="edge_case",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(108, 132),
        audio=generate_short_click(120, 4, duration=4.0),
        max_duration=5.0,
    ))
    tests.append(TestCase(
        name="Edge: 3/4 short clip (5s)",
        category="edge_case",
        expected_meter=[(3, 4)],
        expected_bpm_range=(90, 110),
        audio=generate_short_click(100, 3, duration=5.0),
        max_duration=6.0,
    ))

    # Very quiet audio
    tests.append(TestCase(
        name="Edge: quiet 4/4 (amp=0.05)",
        category="edge_case",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(108, 132),
        audio=generate_quiet_click(120, 4, amplitude=0.05),
    ))

    # Tempo changes
    tests.append(TestCase(
        name="Edge: accel 100->140 BPM (4/4)",
        category="edge_case",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(90, 150),
        audio=generate_tempo_change(100, 140, 4),
    ))
    tests.append(TestCase(
        name="Edge: decel 140->90 BPM (3/4)",
        category="edge_case",
        expected_meter=[(3, 4)],
        expected_bpm_range=(80, 150),
        audio=generate_tempo_change(140, 90, 3),
    ))

    # Silence gaps
    tests.append(TestCase(
        name="Edge: 4/4 with silence gaps",
        category="edge_case",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(108, 132),
        audio=generate_silence_gaps(120, 4),
    ))

    # Very slow tempo
    tests.append(TestCase(
        name="Edge: very slow 3/4 @ 50 BPM",
        category="edge_case",
        expected_meter=[(3, 4)],
        expected_bpm_range=(40, 60),
        audio=generate_click(50, 3, duration=25.0),
    ))

    # Very fast tempo
    tests.append(TestCase(
        name="Edge: very fast 4/4 @ 220 BPM",
        category="edge_case",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(200, 240),
        audio=generate_click(220, 4),
    ))

    # Strong rubato (tempo change wide range)
    tests.append(TestCase(
        name="Edge: strong rubato 80->160 BPM (3/4)",
        category="edge_case",
        expected_meter=[(3, 4)],
        expected_bpm_range=(70, 170),
        audio=generate_tempo_change(80, 160, 3),
    ))

    # Very short 6/8
    tests.append(TestCase(
        name="Edge: 6/8 short clip (5s)",
        category="edge_case",
        expected_meter=[(6, 8), (6, 4), (3, 4)],
        expected_bpm_range=(80, 200),
        audio=generate_compound_click(160, [3, 3], duration=5.0),
        max_duration=6.0,
    ))

    # Quiet 3/4
    tests.append(TestCase(
        name="Edge: quiet 3/4 (amp=0.03)",
        category="edge_case",
        expected_meter=[(3, 4)],
        expected_bpm_range=(90, 110),
        audio=generate_quiet_click(100, 3, amplitude=0.03),
    ))

    # Silence gaps in 3/4
    tests.append(TestCase(
        name="Edge: 3/4 with silence gaps",
        category="edge_case",
        expected_meter=[(3, 4)],
        expected_bpm_range=(90, 110),
        audio=generate_silence_gaps(100, 3),
    ))

    # Weak accent ratio (subtle downbeats)
    tests.append(TestCase(
        name="Edge: 4/4 weak accent (ratio=1.5)",
        category="edge_case",
        expected_meter=[(4, 4), (2, 4)],
        expected_bpm_range=(108, 132),
        audio=generate_click(120, 4, accent_ratio=1.5),
    ))

    # 2/4 short
    tests.append(TestCase(
        name="Edge: 2/4 short clip (4s)",
        category="edge_case",
        expected_meter=[(2, 4), (4, 4)],
        expected_bpm_range=(108, 132),
        audio=generate_short_click(120, 2, duration=4.0),
        max_duration=5.0,
    ))

    return tests


def build_real_audio_tests() -> list[TestCase]:
    """Auto-discover fixture files and build test cases from catalogue."""
    tests = []
    if not FIXTURES_DIR.exists():
        return tests

    for filepath in sorted(FIXTURES_DIR.iterdir()):
        if filepath.suffix not in (".ogg", ".oga", ".wav", ".mp3", ".flac", ".opus"):
            continue
        fname = filepath.name
        if fname in FIXTURE_CATALOGUE:
            cat, meters, bpm_range = FIXTURE_CATALOGUE[fname]
            tests.append(TestCase(
                name=f"Audio: {fname}",
                category=cat,
                expected_meter=meters,
                expected_bpm_range=bpm_range,
                filepath=str(filepath),
            ))
        else:
            # Unknown file -- include it but mark as unknown so it still runs
            tests.append(TestCase(
                name=f"Audio: {fname} (uncatalogued)",
                category="unknown",
                expected_meter=[(4, 4), (3, 4), (2, 4)],  # very permissive
                expected_bpm_range=(40, 300),
                filepath=str(filepath),
            ))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(engine: AnalysisEngine, tc: TestCase) -> TestResult:
    """Run a single test case and return the result."""
    t0 = time.monotonic()

    if tc.audio is not None:
        audio = tc.audio
        sr = SR
    else:
        audio, sr = load_audio(tc.filepath, sr=settings.sample_rate)
        audio = preprocess(audio, sr)

    # Clip duration
    max_samples = int(tc.max_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    a_hash = _audio_hash_full(audio)

    # Try engine-level cache first (covers ALL signals, not just trackers)
    cached = _load_engine_cache(a_hash)
    if cached is not None:
        elapsed = time.monotonic() - t0
        # Evaluate against current test case expectations (GT may have changed)
        detected_meter = cached["detected_meter"]
        detected_bpm = cached["detected_bpm"]
        beatnet_raw_meter_str = cached["beatnet_raw_meter"]

        meter_correct = False
        if detected_meter != "N/A":
            parts = detected_meter.split("/")
            num, den = int(parts[0]), int(parts[1])
            for m in tc.expected_meter:
                if num == m[0] and den == m[1]:
                    meter_correct = True

        tempo_correct = tc.expected_bpm_range[0] <= detected_bpm <= tc.expected_bpm_range[1]

        beatnet_raw_correct = False
        if beatnet_raw_meter_str != "N/A":
            parts = beatnet_raw_meter_str.split("/")
            bn_num, bn_den = int(parts[0]), int(parts[1])
            for m in tc.expected_meter:
                if bn_num == m[0] and bn_den == m[1]:
                    beatnet_raw_correct = True

        return TestResult(
            name=tc.name,
            category=tc.category,
            meter_correct=meter_correct,
            tempo_correct=tempo_correct,
            detected_meter=detected_meter,
            detected_bpm=detected_bpm,
            expected_meters=[f"{m[0]}/{m[1]}" for m in tc.expected_meter],
            expected_bpm_range=[tc.expected_bpm_range[0], tc.expected_bpm_range[1]],
            elapsed_seconds=round(elapsed, 2),
            beatnet_raw_meter=beatnet_raw_meter_str,
            beatnet_raw_correct=beatnet_raw_correct,
        )

    # Get BeatNet beats for raw meter comparison.
    # We call the (possibly cached) function directly so we can inspect
    # downbeat positions independently of the engine's meter logic.
    from beatmeter.analysis.beat_tracking import track_beats_beatnet
    beatnet_beats = track_beats_beatnet(audio, sr)

    result = engine.analyze_audio(audio, sr)
    elapsed = time.monotonic() - t0

    # Evaluate meter
    meter_correct = False
    detected_meter = "N/A"
    if result.meter_hypotheses:
        top = result.meter_hypotheses[0]
        detected_meter = f"{top.numerator}/{top.denominator}"
        for m in tc.expected_meter:
            if top.numerator == m[0] and top.denominator == m[1]:
                meter_correct = True

    # Evaluate tempo
    detected_bpm = result.tempo.bpm
    tempo_correct = tc.expected_bpm_range[0] <= detected_bpm <= tc.expected_bpm_range[1]

    # BeatNet raw meter
    beatnet_raw_meter_str = "N/A"
    beatnet_raw_correct = False
    raw_meter = get_beatnet_raw_meter(beatnet_beats)
    if raw_meter:
        beatnet_raw_meter_str = f"{raw_meter[0]}/{raw_meter[1]}"
        for m in tc.expected_meter:
            if raw_meter[0] == m[0] and raw_meter[1] == m[1]:
                beatnet_raw_correct = True

    # Save to engine cache for future runs
    _save_engine_cache(a_hash, {
        "detected_meter": detected_meter,
        "detected_bpm": round(detected_bpm, 1),
        "beatnet_raw_meter": beatnet_raw_meter_str,
    })

    return TestResult(
        name=tc.name,
        category=tc.category,
        meter_correct=meter_correct,
        tempo_correct=tempo_correct,
        detected_meter=detected_meter,
        detected_bpm=round(detected_bpm, 1),
        expected_meters=[f"{m[0]}/{m[1]}" for m in tc.expected_meter],
        expected_bpm_range=[tc.expected_bpm_range[0], tc.expected_bpm_range[1]],
        elapsed_seconds=round(elapsed, 2),
        beatnet_raw_meter=beatnet_raw_meter_str,
        beatnet_raw_correct=beatnet_raw_correct,
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def load_previous_results() -> dict | None:
    """Load the most recent benchmark results from JSON."""
    if not RESULTS_FILE.exists():
        return None
    try:
        data = json.loads(RESULTS_FILE.read_text())
        # Could be a list of runs; take the last one
        if isinstance(data, list):
            return data[-1] if data else None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def detect_regressions(
    current: list[TestResult], previous: dict | None
) -> list[str]:
    """Compare current results to previous and list regressions."""
    if previous is None:
        return []

    prev_by_name = {}
    for r in previous.get("results", []):
        prev_by_name[r["name"]] = r

    regressions = []
    for cur in current:
        prev = prev_by_name.get(cur.name)
        if prev is None:
            continue
        if prev.get("meter_correct") and not cur.meter_correct:
            regressions.append(
                f"  REGRESSION meter: {cur.name} was correct, now {cur.detected_meter}"
            )
        if prev.get("tempo_correct") and not cur.tempo_correct:
            regressions.append(
                f"  REGRESSION tempo: {cur.name} was correct, "
                f"now {cur.detected_bpm} BPM"
            )
    return regressions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(results: list[TestResult], title: str):
    """Print a formatted result table."""
    if not results:
        return
    print(f"\n{'=' * 115}")
    print(f"  {title}")
    print(f"{'=' * 115}")
    print(f"  {'M':4s} {'T':4s} {'time':>5s} | {'engine':8s} {'BNraw':8s} {'bpm':>6s} | "
          f"{'expected':20s} | {'name'}")
    print(f"  {'-'*4} {'-'*4} {'-'*5} | {'-'*8} {'-'*8} {'-'*6} | {'-'*20} | {'-'*35}")

    for r in results:
        m_flag = "OK" if r.meter_correct else "FAIL"
        t_flag = "OK" if r.tempo_correct else "FAIL"
        bn_flag = "*" if r.beatnet_raw_correct else " "
        exp_str = ",".join(r.expected_meters)
        print(f"  {m_flag:4s} {t_flag:4s} {r.elapsed_seconds:5.1f}s | "
              f"{r.detected_meter:8s} {r.beatnet_raw_meter:7s}{bn_flag} {r.detected_bpm:6.1f} | "
              f"{exp_str:20s} | {r.name}")


def print_category_summary(results: list[TestResult]):
    """Print per-category breakdown with engine vs BeatNet raw meter comparison."""
    categories: dict[str, list[TestResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print(f"\n{'=' * 115}")
    print("  PER-CATEGORY BREAKDOWN (engine meter vs BeatNet raw meter)")
    print(f"{'=' * 115}")
    print(f"  {'category':18s} {'tests':>5s} {'engine_m':>12s} {'BN_raw_m':>12s} {'tempo':>12s} {'avg_time':>9s}")
    print(f"  {'-'*18} {'-'*5} {'-'*12} {'-'*12} {'-'*12} {'-'*9}")

    for cat in sorted(categories.keys()):
        group = categories[cat]
        total = len(group)
        m_pass = sum(1 for r in group if r.meter_correct)
        bn_pass = sum(1 for r in group if r.beatnet_raw_correct)
        t_pass = sum(1 for r in group if r.tempo_correct)
        avg_time = sum(r.elapsed_seconds for r in group) / total if total else 0
        m_pct = f"{m_pass}/{total} ({m_pass/total*100:.0f}%)" if total else "N/A"
        bn_pct = f"{bn_pass}/{total} ({bn_pass/total*100:.0f}%)" if total else "N/A"
        t_pct = f"{t_pass}/{total} ({t_pass/total*100:.0f}%)" if total else "N/A"
        print(f"  {cat:18s} {total:5d} {m_pct:>12s} {bn_pct:>12s} {t_pct:>12s} {avg_time:8.1f}s")


def print_overall_summary(results: list[TestResult]):
    """Print the overall summary line."""
    total = len(results)
    m_pass = sum(1 for r in results if r.meter_correct)
    bn_pass = sum(1 for r in results if r.beatnet_raw_correct)
    t_pass = sum(1 for r in results if r.tempo_correct)
    both = sum(1 for r in results if r.meter_correct and r.tempo_correct)
    total_time = sum(r.elapsed_seconds for r in results)

    print(f"\n{'=' * 115}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 115}")
    print(f"  Total tests:           {total}")
    print(f"  Engine meter correct:  {m_pass}/{total} ({m_pass/total*100:.0f}%)")
    print(f"  BeatNet raw meter:     {bn_pass}/{total} ({bn_pass/total*100:.0f}%)")
    print(f"  Tempo correct:         {t_pass}/{total} ({t_pass/total*100:.0f}%)")
    print(f"  Both correct:          {both}/{total} ({both/total*100:.0f}%)")
    print(f"  Total time:            {total_time:.1f}s")


def save_results(results: list[TestResult], filepath: Path):
    """Append results to the JSON history file."""
    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "meter_pass": sum(1 for r in results if r.meter_correct),
        "tempo_pass": sum(1 for r in results if r.tempo_correct),
        "results": [asdict(r) for r in results],
    }

    history = []
    if filepath.exists():
        try:
            history = json.loads(filepath.read_text())
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            history = []

    history.append(run_record)
    filepath.write_text(json.dumps(history, indent=2))
    print(f"\n  Results saved to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global _cache_enabled, _engine_cache_enabled

    parser = argparse.ArgumentParser(description="Rhythm Analyzer Benchmark")
    parser.add_argument("--save", action="store_true",
                        help="Save results to benchmark_results.json")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose per-signal output")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only tests in this category")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Skip synthetic tests")
    parser.add_argument("--no-real", action="store_true",
                        help="Skip real audio tests")
    parser.add_argument("--no-edge", action="store_true",
                        help="Skip edge case tests")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable beat tracker result caching (force fresh inference)")
    args = parser.parse_args()

    if args.no_cache:
        _cache_enabled = False
        _engine_cache_enabled = False

    # Install caching wrappers on beat tracking functions
    install_cache_wrappers()

    if args.verbose:
        logging.getLogger("beatmeter").setLevel(logging.INFO)

    cache_status = "ON" if _cache_enabled else "OFF (--no-cache)"
    n_cached = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
    code_h = _get_code_hash()
    n_engine_cached = len(list(CACHE_DIR.glob(f"engine_*_{code_h}.json"))) if CACHE_DIR.exists() and _engine_cache_enabled else 0

    print("=" * 115)
    print("  RHYTHM ANALYZER - UNIFIED BENCHMARK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cache: {cache_status} ({n_cached} cached files, {n_engine_cached} engine results for current code)")
    print("=" * 115)

    # Build test suite
    all_tests: list[TestCase] = []
    if not args.no_synthetic:
        all_tests.extend(build_synthetic_tests())
    if not args.no_edge:
        all_tests.extend(build_edge_case_tests())
    if not args.no_real:
        all_tests.extend(build_real_audio_tests())

    # Filter by category if requested
    if args.category:
        all_tests = [t for t in all_tests if t.category == args.category]

    print(f"\n  {len(all_tests)} test cases loaded")
    categories = set(t.category for t in all_tests)
    print(f"  Categories: {', '.join(sorted(categories))}")

    # Discover fixture files not in catalogue
    if FIXTURES_DIR.exists():
        all_files = set(
            f.name for f in FIXTURES_DIR.iterdir()
            if f.suffix in (".ogg", ".oga", ".wav", ".mp3", ".flac", ".opus")
        )
        catalogued = set(FIXTURE_CATALOGUE.keys())
        uncatalogued = all_files - catalogued
        if uncatalogued:
            print(f"  Uncatalogued fixture files ({len(uncatalogued)}): "
                  f"{', '.join(sorted(uncatalogued))}")

    # Run
    engine = AnalysisEngine()
    results: list[TestResult] = []

    for i, tc in enumerate(all_tests, 1):
        sys.stdout.write(f"\r  Running {i}/{len(all_tests)}: {tc.name[:50]:50s}")
        sys.stdout.flush()
        try:
            result = run_test(engine, tc)
            results.append(result)
        except Exception as e:
            print(f"\n  EXCEPTION on {tc.name}: {e}")
            results.append(TestResult(
                name=tc.name,
                category=tc.category,
                meter_correct=False,
                tempo_correct=False,
                detected_meter="ERR",
                detected_bpm=0.0,
                expected_meters=[f"{m[0]}/{m[1]}" for m in tc.expected_meter],
                expected_bpm_range=[tc.expected_bpm_range[0], tc.expected_bpm_range[1]],
            ))

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Print results grouped by category
    categories_seen: dict[str, list[TestResult]] = {}
    for r in results:
        categories_seen.setdefault(r.category, []).append(r)

    for cat in sorted(categories_seen.keys()):
        print_table(categories_seen[cat], f"{cat.upper()} ({len(categories_seen[cat])} tests)")

    # Failures summary
    failures = [r for r in results if not r.meter_correct or not r.tempo_correct]
    if failures:
        print(f"\n{'=' * 95}")
        print(f"  FAILURES ({len(failures)})")
        print(f"{'=' * 95}")
        for r in failures:
            issues = []
            if not r.meter_correct:
                issues.append(f"meter={r.detected_meter} expected={','.join(r.expected_meters)}")
            if not r.tempo_correct:
                issues.append(
                    f"bpm={r.detected_bpm} expected={r.expected_bpm_range[0]}-{r.expected_bpm_range[1]}"
                )
            print(f"  {r.name}: {'; '.join(issues)}")

    print_category_summary(results)
    print_overall_summary(results)

    # Regression detection
    previous = load_previous_results()
    regressions = detect_regressions(results, previous)
    if regressions:
        print(f"\n{'=' * 95}")
        print(f"  REGRESSIONS DETECTED ({len(regressions)})")
        print(f"{'=' * 95}")
        for msg in regressions:
            print(msg)
    elif previous:
        prev_meter = previous.get("meter_pass", 0)
        prev_total = previous.get("total", 0)
        cur_meter = sum(1 for r in results if r.meter_correct)
        print(f"\n  No regressions vs previous run "
              f"(was {prev_meter}/{prev_total}, now {cur_meter}/{len(results)})")

    # Save
    if args.save:
        save_results(results, RESULTS_FILE)

    # Exit code: non-zero if regressions
    if regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
