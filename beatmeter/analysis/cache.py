"""Per-tracker and per-signal analysis cache with file-hash invalidation.

Cache structure:
    .cache/
    ├── beats/{tracker_name}/{tracker_hash}/
    │   └── {audio_hash}.json
    ├── onsets/{onset_hash}/
    │   └── {audio_hash}.json
    ├── signals/{signal_name}/{deps_hash}/
    │   └── {audio_hash}.json

Each tracker is hashed independently — changing BeatNet code does not
invalidate the madmom cache. Each signal has explicit upstream dependencies;
its cache is invalidated when any dependency file changes.

Version is encoded in the directory path (hash of source files).
No version parameter needed — file changes = automatic invalidation.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum number of hash subdirectories to keep per level.
# Older ones (by mtime) are pruned at init time.
_MAX_HASH_DIRS = 2

# ---------------------------------------------------------------------------
# Per-signal dependency map: signal_name -> list of source files that affect it.
# When ANY of these files change, the signal's cache is invalidated.
# ---------------------------------------------------------------------------

SIGNAL_DEPS: dict[str, list[str]] = {
    "beatnet_spacing": [
        "beatmeter/analysis/signals/downbeat_spacing.py",
        "beatmeter/analysis/trackers/beatnet.py",
    ],
    "beat_this_spacing": [
        "beatmeter/analysis/signals/downbeat_spacing.py",
        "beatmeter/analysis/trackers/beat_this.py",
    ],
    "madmom_activation": [
        "beatmeter/analysis/signals/madmom_activation.py",
        "beatmeter/analysis/trackers/madmom_tracker.py",
        "beatmeter/analysis/onset.py",
    ],
    "onset_autocorr": [
        "beatmeter/analysis/signals/onset_autocorrelation.py",
        "beatmeter/analysis/onset.py",
    ],
    "accent_pattern": [
        "beatmeter/analysis/signals/accent_pattern.py",
        "beatmeter/analysis/trackers/beatnet.py",
        "beatmeter/analysis/trackers/beat_this.py",
        "beatmeter/analysis/trackers/madmom_tracker.py",
        "beatmeter/analysis/trackers/librosa_tracker.py",
    ],
    "beat_periodicity": [
        "beatmeter/analysis/signals/beat_periodicity.py",
        "beatmeter/analysis/trackers/beatnet.py",
        "beatmeter/analysis/trackers/beat_this.py",
        "beatmeter/analysis/trackers/madmom_tracker.py",
        "beatmeter/analysis/trackers/librosa_tracker.py",
    ],
    "bar_tracking": [
        "beatmeter/analysis/signals/bar_tracking.py",
        "beatmeter/analysis/trackers/beatnet.py",
        "beatmeter/analysis/trackers/beat_this.py",
        "beatmeter/analysis/trackers/madmom_tracker.py",
        "beatmeter/analysis/trackers/librosa_tracker.py",
    ],
    "resnet_meter": [
        "beatmeter/analysis/signals/resnet_meter.py",
    ],
    "mert_meter": [
        "beatmeter/analysis/signals/mert_meter.py",
    ],
    "tempo_librosa": [
        "beatmeter/analysis/tempo.py",
    ],
    "tempo_tempogram": [
        "beatmeter/analysis/tempo.py",
    ],
}

# Per-tracker source file for beat cache invalidation.
TRACKER_FILES: dict[str, str] = {
    "beatnet": "beatmeter/analysis/trackers/beatnet.py",
    "beat_this": "beatmeter/analysis/trackers/beat_this.py",
    "madmom_bpb3": "beatmeter/analysis/trackers/madmom_tracker.py",
    "madmom_bpb4": "beatmeter/analysis/trackers/madmom_tracker.py",
    "madmom_bpb5": "beatmeter/analysis/trackers/madmom_tracker.py",
    "madmom_bpb7": "beatmeter/analysis/trackers/madmom_tracker.py",
    "librosa": "beatmeter/analysis/trackers/librosa_tracker.py",
}


class AnalysisCache:
    """Unified per-signal cache for the analysis pipeline."""

    def __init__(self, cache_dir: Path | str = ".cache"):
        self.cache_dir = Path(cache_dir)

        # Per-tracker hashes
        self._tracker_hashes: dict[str, str] = {}
        for tracker_name, src_file in TRACKER_FILES.items():
            self._tracker_hashes[tracker_name] = self._file_hash(src_file)

        # Onset hash
        self._onset_hash = self._file_hash("beatmeter/analysis/onset.py")

        # Per-signal dependency hashes
        self._signal_hashes: dict[str, str] = {}
        for sig_name, deps in SIGNAL_DEPS.items():
            self._signal_hashes[sig_name] = self._combined_hash(*deps)

        self._cleanup_old_dirs()

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def audio_hash(file_path: str) -> str:
        """Derive a cache key from the file path.

        Uses the filename stem (no extension) which is unique within the
        METER2800 dataset and stable across runs.  Falls back to a short
        content hash only when the stem is ambiguous (< 3 chars).
        """
        from pathlib import Path
        stem = Path(file_path).stem
        if len(stem) >= 3:
            return stem
        # Very short stem — hash a bit of content for safety
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read(200_000)).hexdigest()[:16]

    @staticmethod
    def audio_hash_from_array(audio) -> str:
        """SHA-256 of first 200 KB of audio bytes -> 16 hex chars."""
        return hashlib.sha256(audio.tobytes()[:200_000]).hexdigest()[:16]

    @staticmethod
    def _file_hash(rel_path: str) -> str:
        """SHA-256 of a source file -> 12 hex chars."""
        p = _project_root() / rel_path
        if not p.exists():
            return "missing"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]

    @staticmethod
    def _combined_hash(*rel_paths: str) -> str:
        """SHA-256 of concatenated source files -> 12 hex chars."""
        h = hashlib.sha256()
        for rp in sorted(rel_paths):
            p = _project_root() / rp
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()[:12]

    # ------------------------------------------------------------------
    # Beats: .cache/beats/{tracker}/{tracker_hash}/{audio_hash}.json
    # ------------------------------------------------------------------

    def _beats_dir(self, tracker: str) -> Path:
        h = self._tracker_hashes.get(tracker, "unknown")
        return self.cache_dir / "beats" / tracker / h

    def load_beats(self, audio_hash: str, tracker: str) -> list[dict] | None:
        path = self._beats_dir(tracker) / f"{audio_hash}.json"
        return _read_json(path)

    def save_beats(self, audio_hash: str, tracker: str, beats: list[dict]) -> None:
        d = self._beats_dir(tracker)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{audio_hash}.json"
        path.write_text(json.dumps(beats))

    # ------------------------------------------------------------------
    # Onsets: .cache/onsets/{onset_hash}/{audio_hash}.json
    # ------------------------------------------------------------------

    def _onsets_dir(self) -> Path:
        return self.cache_dir / "onsets" / self._onset_hash

    def load_onsets(self, audio_hash: str) -> dict | None:
        path = self._onsets_dir() / f"{audio_hash}.json"
        return _read_json(path)

    def save_onsets(self, audio_hash: str, data: dict) -> None:
        d = self._onsets_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{audio_hash}.json"
        path.write_text(json.dumps(data))

    # ------------------------------------------------------------------
    # Signals: .cache/signals/{signal_name}/{deps_hash}/{audio_hash}.json
    # ------------------------------------------------------------------

    def _signal_dir(self, signal_name: str) -> Path:
        h = self._signal_hashes.get(signal_name, "unknown")
        return self.cache_dir / "signals" / signal_name / h

    def load_signal(self, audio_hash: str, signal_name: str) -> dict | None:
        path = self._signal_dir(signal_name) / f"{audio_hash}.json"
        return _read_json(path)

    def save_signal(self, audio_hash: str, signal_name: str, scores: dict) -> None:
        d = self._signal_dir(signal_name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{audio_hash}.json"
        serializable = {_meter_key_to_str(k): v for k, v in scores.items()}
        path.write_text(json.dumps(serializable))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, audio_hashes: list[str]) -> dict:
        """Check how many files are cached per level."""
        total = len(audio_hashes)
        beats_hit = 0
        onsets_hit = 0
        signals_hit = 0

        onsets_dir = self._onsets_dir()

        for ah in audio_hashes:
            # Check if at least one beat tracker file exists
            beat_found = False
            for tracker in TRACKER_FILES:
                d = self._beats_dir(tracker)
                if d.exists() and (d / f"{ah}.json").exists():
                    beat_found = True
                    break
            if beat_found:
                beats_hit += 1

            if onsets_dir.exists() and (onsets_dir / f"{ah}.json").exists():
                onsets_hit += 1

            # Check if at least one signal file exists
            sig_found = False
            for sig_name in SIGNAL_DEPS:
                d = self._signal_dir(sig_name)
                if d.exists() and (d / f"{ah}.json").exists():
                    sig_found = True
                    break
            if sig_found:
                signals_hit += 1

        return {
            "beats": f"{beats_hit}/{total}",
            "onsets": f"{onsets_hit}/{total}",
            "signals": f"{signals_hit}/{total}",
        }

    # ------------------------------------------------------------------
    # Auto-cleanup: keep only _MAX_HASH_DIRS newest per hash level
    # ------------------------------------------------------------------

    def _cleanup_old_dirs(self) -> None:
        """Remove old hash subdirectories, keeping only the newest ones."""
        # Onsets: .cache/onsets/{hash}/
        self._cleanup_level(self.cache_dir / "onsets")

        # Beats: .cache/beats/{tracker}/{hash}/
        beats_root = self.cache_dir / "beats"
        if beats_root.exists():
            for tracker_dir in beats_root.iterdir():
                if tracker_dir.is_dir():
                    self._cleanup_level(tracker_dir)

        # Signals: .cache/signals/{signal_name}/{hash}/
        signals_root = self.cache_dir / "signals"
        if signals_root.exists():
            for sig_dir in signals_root.iterdir():
                if sig_dir.is_dir():
                    self._cleanup_level(sig_dir)

    @staticmethod
    def _cleanup_level(level_dir: Path) -> None:
        if not level_dir.exists():
            return
        subdirs = [d for d in level_dir.iterdir() if d.is_dir()]
        if len(subdirs) <= _MAX_HASH_DIRS:
            return
        subdirs.sort(key=lambda d: d.stat().st_mtime)
        for d in subdirs[:-_MAX_HASH_DIRS]:
            logger.info(f"Cache cleanup: removing old {d}")
            _rmtree(d)


# ======================================================================
# Module-level helpers
# ======================================================================

_project_root_cache: Path | None = None


def _project_root() -> Path:
    global _project_root_cache
    if _project_root_cache is None:
        _project_root_cache = Path(__file__).resolve().parent.parent.parent
    return _project_root_cache


def _read_json(path: Path):
    """Read and parse a JSON file. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _meter_key_to_str(key) -> str:
    """Convert (num, den) tuple to 'num_den' string."""
    if isinstance(key, tuple):
        return f"{key[0]}_{key[1]}"
    return str(key)


def str_to_meter_key(s: str) -> tuple[int, int]:
    """Convert 'num_den' string back to (num, den) tuple."""
    a, b = s.split("_")
    return (int(a), int(b))


def _rmtree(path: Path) -> None:
    """Remove a directory tree (shutil.rmtree equivalent)."""
    import shutil
    shutil.rmtree(path, ignore_errors=True)
