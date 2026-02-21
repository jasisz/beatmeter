"""Per-tracker and per-signal analysis cache backed by LMDB.

Cache structure:
    .cache/analysis.lmdb/
    ├── data.mdb
    └── lock.mdb

Key format:
    beats:{tracker}:{tracker_hash}:{audio_hash}       → JSON string
    onsets:{onset_hash}:{audio_hash}                   → JSON string
    signals:{signal_name}:{deps_hash}:{audio_hash}     → JSON string
    features:{group}:{deps_hash}:{audio_hash}           → raw float32 bytes

Feature groups (meter_net_audio, meter_net_ssm) are keyed on source code
only — NOT on the model checkpoint. This means expensive audio feature
extraction (~5s) survives model retraining; only the cheap forward pass
(~10ms) is re-run. See SIGNAL_DEPS for per-group dependency files.

Hash in the key provides automatic invalidation — changing source code
produces a new hash, so old entries simply aren't read. Stale entries
are cleaned up at startup via prefix scan.

Migration: on first run after switching from the old JSON-file cache,
data is lazily migrated (LMDB miss → check old JSON path → insert +
delete JSON). One full pipeline run = complete migration.
"""

import functools
import hashlib
import json
import logging
from pathlib import Path

import lmdb
import numpy as np

logger = logging.getLogger(__name__)

# LMDB map size: 4 GB virtual address space (file grows on demand).
_MAP_SIZE = 4 * 1024 * 1024 * 1024

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
    "onset_autocorr": [
        "beatmeter/analysis/signals/onset_autocorrelation.py",
        "beatmeter/analysis/onset.py",
    ],
    "bar_tracking": [
        "beatmeter/analysis/signals/bar_tracking.py",
        "beatmeter/analysis/trackers/beatnet.py",
        "beatmeter/analysis/trackers/beat_this.py",
        "beatmeter/analysis/trackers/madmom_tracker.py",
        "beatmeter/analysis/trackers/librosa_tracker.py",
    ],
    "hcdf_meter": [
        "beatmeter/analysis/signals/hcdf_meter.py",
    ],
    # Feature cache groups (NO checkpoint dependency → survive model retraining)
    "meter_net_audio": [
        "beatmeter/analysis/signals/onset_mlp_features.py",
    ],
    "meter_net_ssm": [
        "beatmeter/analysis/signals/ssm_features.py",
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
    """Unified per-signal cache for the analysis pipeline (LMDB backend)."""

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

        # Open LMDB environment
        lmdb_path = self.cache_dir / "analysis.lmdb"
        lmdb_path.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(lmdb_path),
            map_size=_MAP_SIZE,
            max_dbs=0,
            readahead=False,
        )

        self._cleanup_stale_entries()

    # ------------------------------------------------------------------
    # Key builders
    # ------------------------------------------------------------------

    def _beats_key(self, tracker: str, audio_hash: str) -> bytes:
        h = self._tracker_hashes.get(tracker, "unknown")
        return f"beats:{tracker}:{h}:{audio_hash}".encode()

    def _onsets_key(self, audio_hash: str) -> bytes:
        return f"onsets:{self._onset_hash}:{audio_hash}".encode()

    def _signal_key(self, signal_name: str, audio_hash: str) -> bytes:
        h = self._signal_hashes.get(signal_name, "unknown")
        return f"signals:{signal_name}:{h}:{audio_hash}".encode()

    def _features_key(self, group: str, audio_hash: str) -> bytes:
        h = self._signal_hashes.get(group, "unknown")
        return f"features:{group}:{h}:{audio_hash}".encode()

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def audio_hash(file_path: str) -> str:
        """Derive a stable cache key from filename and file size.

        Uses stem + file size — fast (single stat, no content read) and
        practically unique for our datasets.
        """
        p = Path(file_path)
        h = hashlib.sha256()
        h.update(p.stem.encode("utf-8", errors="ignore"))
        try:
            h.update(str(p.stat().st_size).encode("ascii"))
        except OSError:
            pass
        return h.hexdigest()[:16]

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
    # Beats: beats:{tracker}:{hash}:{audio_hash}
    # ------------------------------------------------------------------

    def load_beats(self, audio_hash: str, tracker: str) -> list[dict] | None:
        key = self._beats_key(tracker, audio_hash)
        with self._env.begin() as txn:
            data = txn.get(key)
        if data is not None:
            return json.loads(data)
        # Lazy migration from old JSON cache
        return self._migrate_beats(audio_hash, tracker, key)

    def _migrate_beats(self, audio_hash: str, tracker: str, key: bytes) -> list[dict] | None:
        h = self._tracker_hashes.get(tracker, "unknown")
        old_path = self.cache_dir / "beats" / tracker / h / f"{audio_hash}.json"
        data = _read_json(old_path)
        if data is not None:
            with self._env.begin(write=True) as txn:
                txn.put(key, json.dumps(data).encode())
            old_path.unlink(missing_ok=True)
        return data

    def save_beats(self, audio_hash: str, tracker: str, beats: list[dict]) -> None:
        key = self._beats_key(tracker, audio_hash)
        with self._env.begin(write=True) as txn:
            txn.put(key, json.dumps(beats).encode())

    # ------------------------------------------------------------------
    # Onsets: onsets:{hash}:{audio_hash}
    # ------------------------------------------------------------------

    def load_onsets(self, audio_hash: str) -> dict | None:
        key = self._onsets_key(audio_hash)
        with self._env.begin() as txn:
            data = txn.get(key)
        if data is not None:
            return json.loads(data)
        return self._migrate_onsets(audio_hash, key)

    def _migrate_onsets(self, audio_hash: str, key: bytes) -> dict | None:
        old_path = self.cache_dir / "onsets" / self._onset_hash / f"{audio_hash}.json"
        data = _read_json(old_path)
        if data is not None:
            with self._env.begin(write=True) as txn:
                txn.put(key, json.dumps(data).encode())
            old_path.unlink(missing_ok=True)
        return data

    def save_onsets(self, audio_hash: str, data: dict) -> None:
        key = self._onsets_key(audio_hash)
        with self._env.begin(write=True) as txn:
            txn.put(key, json.dumps(data).encode())

    # ------------------------------------------------------------------
    # Signals: signals:{name}:{hash}:{audio_hash}
    # ------------------------------------------------------------------

    def load_signal(self, audio_hash: str, signal_name: str) -> dict | None:
        key = self._signal_key(signal_name, audio_hash)
        with self._env.begin() as txn:
            data = txn.get(key)
        if data is not None:
            return json.loads(data)
        return self._migrate_signal(audio_hash, signal_name, key)

    def _migrate_signal(self, audio_hash: str, signal_name: str, key: bytes) -> dict | None:
        h = self._signal_hashes.get(signal_name, "unknown")
        old_path = self.cache_dir / "signals" / signal_name / h / f"{audio_hash}.json"
        data = _read_json(old_path)
        if data is not None:
            with self._env.begin(write=True) as txn:
                txn.put(key, json.dumps(data).encode())
            old_path.unlink(missing_ok=True)
        return data

    def save_signal(self, audio_hash: str, signal_name: str, scores: dict) -> None:
        key = self._signal_key(signal_name, audio_hash)
        serializable = {_meter_key_to_str(k): v for k, v in scores.items()}
        with self._env.begin(write=True) as txn:
            txn.put(key, json.dumps(serializable).encode())

    # ------------------------------------------------------------------
    # Feature arrays: features:{group}:{hash}:{audio_hash}
    # ------------------------------------------------------------------

    def load_array(self, audio_hash: str, group: str) -> np.ndarray | None:
        """Load a cached float32 feature array. Returns None on miss."""
        key = self._features_key(group, audio_hash)
        with self._env.begin() as txn:
            data = txn.get(key)
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).copy()

    def save_array(self, audio_hash: str, group: str, arr: np.ndarray) -> None:
        """Save a numpy feature array as raw float32 bytes."""
        key = self._features_key(group, audio_hash)
        with self._env.begin(write=True) as txn:
            txn.put(key, arr.astype(np.float32).tobytes())

    # ------------------------------------------------------------------
    # Batch reads
    # ------------------------------------------------------------------

    def load_all_for_audio(self, audio_hash: str) -> dict:
        """Load all cached data for an audio file in a single LMDB transaction.

        Returns dict with keys 'beats', 'onsets', 'signals' containing
        all available cached data. Much faster than individual load_* calls
        when multiple data types are needed (1 txn vs 20+ separate reads).
        """
        result: dict = {"beats": {}, "onsets": None, "signals": {}, "features": {}}
        with self._env.begin() as txn:
            for tracker in TRACKER_FILES:
                data = txn.get(self._beats_key(tracker, audio_hash))
                if data is not None:
                    result["beats"][tracker] = json.loads(data)
            data = txn.get(self._onsets_key(audio_hash))
            if data is not None:
                result["onsets"] = json.loads(data)
            for sig_name in SIGNAL_DEPS:
                data = txn.get(self._signal_key(sig_name, audio_hash))
                if data is not None:
                    result["signals"][sig_name] = json.loads(data)
            for group in ["meter_net_audio", "meter_net_ssm"]:
                data = txn.get(self._features_key(group, audio_hash))
                if data is not None:
                    result["features"][group] = np.frombuffer(data, dtype=np.float32).copy()
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, audio_hashes: list[str]) -> dict:
        """Check how many files are cached per level."""
        total = len(audio_hashes)
        beats_hit = 0
        onsets_hit = 0
        signals_hit = 0

        with self._env.begin() as txn:
            for ah in audio_hashes:
                # Check if at least one beat tracker is cached
                for tracker in TRACKER_FILES:
                    if txn.get(self._beats_key(tracker, ah)) is not None:
                        beats_hit += 1
                        break

                # Check onsets
                if txn.get(self._onsets_key(ah)) is not None:
                    onsets_hit += 1

                # Check if at least one signal is cached
                for sig_name in SIGNAL_DEPS:
                    if txn.get(self._signal_key(sig_name, ah)) is not None:
                        signals_hit += 1
                        break

        return {
            "beats": f"{beats_hit}/{total}",
            "onsets": f"{onsets_hit}/{total}",
            "signals": f"{signals_hit}/{total}",
        }

    # ------------------------------------------------------------------
    # Stale entry cleanup
    # ------------------------------------------------------------------

    def _cleanup_stale_entries(self) -> None:
        """Remove LMDB entries whose hashes don't match current source code."""
        stat = self._env.stat()
        if stat["entries"] == 0:
            return

        # Build set of valid prefixes (everything before the audio_hash)
        valid_prefixes: set[str] = set()
        for tracker, h in self._tracker_hashes.items():
            valid_prefixes.add(f"beats:{tracker}:{h}:")
        valid_prefixes.add(f"onsets:{self._onset_hash}:")
        for sig, h in self._signal_hashes.items():
            valid_prefixes.add(f"signals:{sig}:{h}:")
            valid_prefixes.add(f"features:{sig}:{h}:")

        with self._env.begin(write=True) as txn:
            cursor = txn.cursor()
            stale: list[bytes] = []
            for key_bytes, _ in cursor:
                k = key_bytes.decode()
                # Extract prefix = everything up to last ':'
                last_colon = k.rfind(":")
                if last_colon < 0:
                    stale.append(key_bytes)
                    continue
                prefix = k[: last_colon + 1]
                if prefix not in valid_prefixes:
                    stale.append(key_bytes)
            for key_bytes in stale:
                txn.delete(key_bytes)

        if stale:
            logger.info("LMDB cleanup: removed %d stale entries", len(stale))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env:
            self._env.close()
            self._env = None


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


class NumpyLMDB:
    """LMDB-backed cache for numpy feature arrays.

    Stores arrays as raw float32 bytes — faster than np.save/np.load
    (no .npy header parsing). Shape must be known by the caller.

    Default location: data/features.lmdb
    """

    def __init__(self, path: str | Path = "data/features.lmdb", map_size: int = _MAP_SIZE):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(str(path), map_size=map_size, readahead=False)

    def load(self, key: str) -> np.ndarray | None:
        """Load a float32 array by key. Returns None on miss."""
        with self._env.begin() as txn:
            data = txn.get(key.encode())
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).copy()

    def save(self, key: str, arr: np.ndarray) -> None:
        """Save a numpy array as raw float32 bytes."""
        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), arr.astype(np.float32).tobytes())

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env:
            self._env.close()
            self._env = None
