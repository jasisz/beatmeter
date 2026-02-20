#!/usr/bin/env python3
"""Warm the analysis cache for training and inference.

Phases run one at a time to limit memory (load model → process all → free):
  1. beatnet      GPU beat tracker
  2. beat_this    GPU beat tracker (MPS, 1 worker)
  3. madmom       CPU beat tracker (bpb 3,4,5,7)
  4. librosa      CPU beat tracker
  5. onsets       onset detection (needed for signals)
  6. signals      meter signals from cached beats/onsets
  7. tempo        librosa + tempogram tempo estimation
  8. hcdf         harmonic change detection function
  9. ssm          beat-synchronous chroma SSM (75d)

Usage:
    uv run python scripts/setup/warm.py --workers 4
    uv run python scripts/setup/warm.py --phase ssm --workers 4
    uv run python scripts/setup/warm.py --limit 5  # smoke test
"""

import argparse
import csv
import gc
import multiprocessing as mp
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.utils import resolve_audio_path

SR = 22050

ALL_PHASES = [
    "beatnet", "beat_this", "madmom", "librosa",
    "onsets", "signals", "tempo", "hcdf",
    "ssm",
]
GPU_PHASES = {"beat_this"}

# What MeterNet needs from the signal cache
NEEDED_SIGNALS = [
    "beatnet_spacing", "beat_this_spacing", "onset_autocorr",
    "bar_tracking", "hcdf_meter",
]

LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5,
    "seven": 7, "nine": 9, "eleven": 11,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_entries(data_dir: Path, split: str) -> list[tuple[Path, int]]:
    """Load METER2800 entries."""
    if split == "tuning":
        tab_files = ["data_train_4_classes.tab", "data_val_4_classes.tab"]
    elif split == "test":
        tab_files = ["data_test_4_classes.tab"]
    elif split == "all":
        tab_files = [
            "data_train_4_classes.tab",
            "data_val_4_classes.tab",
            "data_test_4_classes.tab",
        ]
    else:
        tab_files = [f"data_{split}_4_classes.tab"]

    entries = []
    for tab in tab_files:
        label_path = data_dir / tab
        if not label_path.exists():
            continue
        with open(label_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fname = row["filename"].strip('"')
                meter = int(row["meter"])
                audio_path = resolve_audio_path(fname, data_dir)
                if audio_path:
                    entries.append((audio_path, meter))
    return entries


def load_wikimeter_entries(wikimeter_dir: Path) -> list[tuple[Path, int]]:
    """Load WIKIMETER entries."""
    tab_path = wikimeter_dir / "data_wikimeter.tab"
    if not tab_path.exists():
        print(f"  WIKIMETER not found at {tab_path}", flush=True)
        return []
    audio_dir = wikimeter_dir / "audio"
    entries = []
    with open(tab_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["label"].strip('"')
            meter = LABEL_TO_METER.get(label)
            if meter is None:
                continue
            fname = row["filename"].strip('"')
            audio_path = audio_dir / Path(fname).name
            if audio_path.exists():
                entries.append((audio_path, meter))
    return entries


def _load_audio(path, sr: int = SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _progress(computed: int, need: int, t_start: float):
    if computed > 0 and computed % 50 == 0:
        rate = computed / (time.time() - t_start)
        eta = (need - computed) / max(rate, 0.01)
        print(f"    {computed}/{need} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_w_cache = None
_w_phase = None
_w_feat_db = None


def _pool_init(phase: str):
    global _w_cache, _w_phase, _w_feat_db
    warnings.filterwarnings("ignore")
    from beatmeter.analysis.cache import AnalysisCache
    _w_cache = AnalysisCache()
    _w_phase = phase
    if phase == "ssm":
        from beatmeter.analysis.cache import NumpyLMDB
        _w_feat_db = NumpyLMDB("data/features.lmdb")


# ---------------------------------------------------------------------------
# Phase workers
# ---------------------------------------------------------------------------

def _worker_tracker(args: tuple[str, str]) -> bool:
    """Beat tracker worker (beatnet, beat_this, madmom, librosa)."""
    import soundfile as sf
    from beatmeter.analysis.engine import _beats_to_dicts

    path_str, ah = args
    phase = _w_phase

    if phase == "madmom":
        from beatmeter.analysis.trackers.madmom_tracker import track_beats_madmom
        bpbs = [3, 4, 5, 7]
        if all(_w_cache.load_beats(ah, f"madmom_bpb{bpb}") is not None for bpb in bpbs):
            return False
        y = _load_audio(path_str)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, SR)
            tmp_path = tmp.name
        try:
            for bpb in bpbs:
                key = f"madmom_bpb{bpb}"
                if _w_cache.load_beats(ah, key) is not None:
                    continue
                try:
                    beats = track_beats_madmom(y, SR, beats_per_bar=bpb, tmp_path=tmp_path)
                    _w_cache.save_beats(ah, key, _beats_to_dicts(beats))
                except Exception as e:
                    print(f"    ERR {Path(path_str).name} bpb={bpb}: {e}", flush=True)
                    _w_cache.save_beats(ah, key, [])
        finally:
            os.unlink(tmp_path)
        return True

    # beatnet, beat_this, librosa
    if _w_cache.load_beats(ah, phase) is not None:
        return False

    if phase == "beatnet":
        from beatmeter.analysis.trackers.beatnet import track_beats_beatnet as fn
    elif phase == "beat_this":
        from beatmeter.analysis.trackers.beat_this import track_beats_beat_this as fn
    elif phase == "librosa":
        from beatmeter.analysis.trackers.librosa_tracker import track_beats_librosa as fn
    else:
        return False

    y = _load_audio(path_str)
    if phase in ("beatnet", "beat_this"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, SR)
            tmp_path = tmp.name
        try:
            beats = fn(y, SR, tmp_path=tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        beats = fn(y, SR)

    _w_cache.save_beats(ah, phase, _beats_to_dicts(beats))
    return True


def _worker_onsets(args: tuple[str, str]) -> bool:
    path_str, ah = args
    if _w_cache.load_onsets(ah) is not None:
        return False
    from beatmeter.analysis.onset import detect_onsets, onset_strength_envelope
    y = _load_audio(path_str)
    onsets = detect_onsets(y, SR)
    onset_times, onset_strengths = onset_strength_envelope(y, SR)
    onset_event_times = np.array([o.time for o in onsets])
    _w_cache.save_onsets(ah, {
        "onset_times": onset_times.tolist(),
        "onset_strengths": onset_strengths.tolist(),
        "onset_events": onset_event_times.tolist(),
    })
    return True


def _worker_signals(args: tuple[str, str]) -> bool:
    """Compute meter signals from cached beats/onsets."""
    path_str, ah = args

    # Check which signals are missing
    needed = [s for s in NEEDED_SIGNALS if s != "hcdf_meter"]  # hcdf is separate phase
    if all(_w_cache.load_signal(ah, s) is not None for s in needed):
        return False

    import soundfile as sf
    from beatmeter.analysis.engine import _dicts_to_beats
    import beatmeter.analysis.meter as meter_mod

    def _load_beats(tracker: str):
        data = _w_cache.load_beats(ah, tracker)
        return _dicts_to_beats(data) if data else []

    beatnet_beats = _load_beats("beatnet")
    beat_this_beats = _load_beats("beat_this") or None
    madmom_results = {}
    for bpb in [3, 4, 5, 7]:
        beats = _load_beats(f"madmom_bpb{bpb}")
        if beats:
            madmom_results[bpb] = beats

    onset_data = _w_cache.load_onsets(ah)
    if onset_data is None:
        return False
    onset_times = np.array(onset_data["onset_times"])
    onset_strengths = np.array(onset_data["onset_strengths"])

    # Best beats + tempo from median IBI
    all_tracker_beats = [beatnet_beats]
    if beat_this_beats:
        all_tracker_beats.append(beat_this_beats)
    for beats in madmom_results.values():
        all_tracker_beats.append(beats)
    all_beats = max((b for b in all_tracker_beats if b), key=len, default=[])
    beat_times = np.array([b.time for b in all_beats]) if all_beats else np.array([])

    beat_interval = None
    if len(beat_times) >= 3:
        ibis = np.diff(beat_times)
        valid = ibis[(ibis > 0.1) & (ibis < 3.0)]
        if len(valid) >= 2:
            tempo_bpm = 60.0 / float(np.median(valid))
            beat_interval = 60.0 / tempo_bpm

    y = _load_audio(path_str)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, SR)
        tmp_path = tmp.name

    try:
        meter_mod._collect_signal_scores(
            beatnet_beats=beatnet_beats,
            madmom_results=madmom_results,
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            beat_interval=beat_interval,
            beat_times=beat_times,
            sr=SR,
            audio=y,
            beat_this_beats=beat_this_beats,
            skip_bar_tracking=False,
            cache=_w_cache,
            audio_hash=ah,
            tmp_path=tmp_path,
        )
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    finally:
        os.unlink(tmp_path)

    # Save empty dict for any signals that failed (e.g. bar_tracking dimension error)
    # so the cache knows "tried, can't compute" and doesn't retry forever
    needed = [s for s in NEEDED_SIGNALS if s != "hcdf_meter"]
    for sig in needed:
        if _w_cache.load_signal(ah, sig) is None:
            _w_cache.save_signal(ah, sig, {})

    return True


def _worker_tempo(args: tuple[str, str]) -> bool:
    path_str, ah = args
    has_lib = _w_cache.load_signal(ah, "tempo_librosa") is not None
    has_tg = _w_cache.load_signal(ah, "tempo_tempogram") is not None
    if has_lib and has_tg:
        return False

    from beatmeter.analysis.tempo import estimate_from_librosa, estimate_from_tempogram
    y = _load_audio(path_str)

    if not has_lib:
        try:
            est = estimate_from_librosa(y, SR)
            if est:
                _w_cache.save_signal(ah, "tempo_librosa", {
                    "bpm": est.bpm, "confidence": est.confidence, "method": est.method,
                })
            else:
                _w_cache.save_signal(ah, "tempo_librosa", {"bpm": 0.0})
        except Exception as e:
            print(f"    ERR tempo_librosa {Path(path_str).name}: {e}", flush=True)
            _w_cache.save_signal(ah, "tempo_librosa", {"bpm": 0.0})

    if not has_tg:
        try:
            est = estimate_from_tempogram(y, SR)
            if est:
                _w_cache.save_signal(ah, "tempo_tempogram", {
                    "bpm": est.bpm, "confidence": est.confidence, "method": est.method,
                })
            else:
                _w_cache.save_signal(ah, "tempo_tempogram", {"bpm": 0.0})
        except Exception as e:
            print(f"    ERR tempo_tempogram {Path(path_str).name}: {e}", flush=True)
            _w_cache.save_signal(ah, "tempo_tempogram", {"bpm": 0.0})

    return True


def _worker_hcdf(args: tuple[str, str]) -> bool:
    path_str, ah = args
    if _w_cache.load_signal(ah, "hcdf_meter") is not None:
        return False
    y = _load_audio(path_str)
    try:
        from beatmeter.analysis.signals.hcdf_meter import signal_hcdf_meter
        scores = signal_hcdf_meter(y, SR)
        _w_cache.save_signal(ah, "hcdf_meter", scores)
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    return True


def _worker_ssm(args: tuple[str, str]) -> bool:
    """Extract and cache beat-synchronous chroma SSM features (75d)."""
    import hashlib
    path_str, ah = args

    from beatmeter.analysis.signals.onset_mlp_features import FEATURE_VERSION_V6
    st = Path(path_str).stat()
    raw = f"{Path(path_str).resolve()}::{st.st_size}::{st.st_mtime_ns}::ssm_v1"
    key = hashlib.sha1(raw.encode()).hexdigest()
    lmdb_key = f"ssm:{key}"

    if _w_feat_db.load(lmdb_key) is not None:
        return False

    from beatmeter.analysis.signals.ssm_features import extract_ssm_features_cached
    y = _load_audio(path_str)
    feat = extract_ssm_features_cached(y, SR, _w_cache, ah)
    _w_feat_db.save(lmdb_key, feat)
    return True


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

PHASE_CONFIG = {
    "beatnet":   (_worker_tracker, "beatnet"),
    "beat_this": (_worker_tracker, "beat_this"),
    "madmom":    (_worker_tracker, "madmom"),
    "librosa":   (_worker_tracker, "librosa"),
    "onsets":    (_worker_onsets, "onsets"),
    "signals":   (_worker_signals, "signals"),
    "tempo":     (_worker_tempo, "tempo"),
    "hcdf":      (_worker_hcdf, "hcdf"),
    "ssm":       (_worker_ssm, "ssm"),
}


def _count_cached(phase: str, audio_hashes, cache, feat_db=None) -> int:
    """Count how many files are already cached for a phase."""
    import hashlib

    count = 0
    for path_str, ah in audio_hashes:
        if phase == "madmom":
            if all(cache.load_beats(ah, f"madmom_bpb{bpb}") is not None for bpb in [3, 4, 5, 7]):
                count += 1
        elif phase in ("beatnet", "beat_this", "librosa"):
            if cache.load_beats(ah, phase) is not None:
                count += 1
        elif phase == "onsets":
            if cache.load_onsets(ah) is not None:
                count += 1
        elif phase == "signals":
            needed = [s for s in NEEDED_SIGNALS if s != "hcdf_meter"]
            if all(cache.load_signal(ah, s) is not None for s in needed):
                count += 1
        elif phase == "tempo":
            if (cache.load_signal(ah, "tempo_librosa") is not None
                    and cache.load_signal(ah, "tempo_tempogram") is not None):
                count += 1
        elif phase == "hcdf":
            if cache.load_signal(ah, "hcdf_meter") is not None:
                count += 1
        elif phase == "ssm" and feat_db is not None:
            st = Path(path_str).stat()
            raw = f"{Path(path_str).resolve()}::{st.st_size}::{st.st_mtime_ns}::ssm_v1"
            key = hashlib.sha1(raw.encode()).hexdigest()
            if feat_db.load(f"ssm:{key}") is not None:
                count += 1
    return count


def run_phase(phase: str, entries, cache, workers: int, feat_db=None):
    """Run one phase with multiprocessing."""
    pool_fn, init_phase = PHASE_CONFIG[phase]

    audio_hashes = [(str(path), cache.audio_hash(str(path))) for path, _ in entries]
    already = _count_cached(phase, audio_hashes, cache, feat_db=feat_db)
    need = len(entries) - already

    if need == 0:
        print(f"  all {len(entries)} cached", flush=True)
        return

    print(f"  {already} cached, {need} to compute ({workers}w)", flush=True)

    t_start = time.time()
    computed = 0

    if workers <= 1:
        _pool_init(init_phase)
        for item in audio_hashes:
            if pool_fn(item):
                computed += 1
                _progress(computed, need, t_start)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_pool_init, initargs=(init_phase,), maxtasksperchild=50) as pool:
            for did_work in pool.imap_unordered(pool_fn, audio_hashes):
                if did_work:
                    computed += 1
                    _progress(computed, need, t_start)

    elapsed = time.time() - t_start
    print(f"  done: {computed} files in {elapsed:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Warm analysis cache for MeterNet training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--wikimeter-dir", type=Path, default=Path("data/wikimeter"))
    parser.add_argument("--split", default="all")
    parser.add_argument("--phase", action="append", default=None,
                        help="Phase(s) to run (repeat for multiple). Default: all.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("-w", "--workers", type=int, default=1)
    args = parser.parse_args()

    # Load entries
    data_dir = args.data_dir.resolve()
    entries = load_entries(data_dir, args.split)
    wiki_entries = load_wikimeter_entries(args.wikimeter_dir.resolve())
    entries.extend(wiki_entries)
    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"Loaded {len(entries)} files ({args.split} + WIKIMETER)", flush=True)

    from beatmeter.analysis.cache import AnalysisCache, NumpyLMDB
    cache = AnalysisCache()
    feat_db = NumpyLMDB("data/features.lmdb")

    phases = args.phase if args.phase else ALL_PHASES

    for phase in phases:
        if phase not in PHASE_CONFIG:
            print(f"\nUnknown phase: {phase} (available: {', '.join(ALL_PHASES)})")
            continue
        w = 1 if phase in GPU_PHASES else args.workers
        print(f"\n=== {phase} ===", flush=True)
        run_phase(phase, entries, cache, w, feat_db=feat_db)
        gc.collect()
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass

    # Summary: check MeterNet readiness
    print("\n=== MeterNet cache summary ===")
    audio_hashes = [(str(path), cache.audio_hash(str(path))) for path, _ in entries]
    for phase in ALL_PHASES:
        n = _count_cached(phase, audio_hashes, cache, feat_db=feat_db)
        pct = n / len(entries) * 100
        status = "OK" if pct > 95 else "LOW"
        print(f"  {phase:12s} {n:5d}/{len(entries)} ({pct:5.1f}%) {status}")


if __name__ == "__main__":
    main()
