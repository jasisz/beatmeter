#!/usr/bin/env python3
"""Warm the analysis cache by running one tracker/signal at a time.

Instead of loading ALL models per file (3GB+ RAM, GPU contention),
this script processes files one tracker/signal at a time:

  Phase 1: BeatNet     → all files → cache → free
  Phase 2: Beat This!  → all files → cache → free
  Phase 3: madmom      → all files → cache → free
  Phase 4: librosa     → all files → cache → free
  Phase 5: onsets      → all files → cache
  Phase 6: onset_mlp   → all files → cache → free
  Phase 7: resnet      → all files → cache → free
  Phase 8: hcdf        → all files → cache

Supports --workers N for CPU-bound phases (beatnet, madmom, librosa, onsets,
onset_mlp). Beat This uses MPS/GPU so defaults to 1 worker.

After warming, arbiter extraction reads from cache = instant.

Usage:
    uv run python scripts/setup/warm_cache.py --workers 4          # warm all, 4 workers
    uv run python scripts/setup/warm_cache.py --phase beatnet -w 4 # one phase
    uv run python scripts/setup/warm_cache.py --limit 5            # smoke test
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ALL_PHASES = ["beatnet", "beat_this", "madmom", "librosa", "onsets", "onset_mlp", "resnet", "hcdf"]
GPU_PHASES = {"beat_this"}  # only 1 worker for GPU phases


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


def _load_audio(path, sr: int = 22050) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def _progress(computed: int, need: int, t_start: float):
    if computed > 0 and computed % 50 == 0:
        rate = computed / (time.time() - t_start)
        eta = (need - computed) / max(rate, 0.01)
        print(f"    {computed}/{need} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)


def _get_audio_hashes(entries, cache):
    return [(path, cache.audio_hash(str(path))) for path, _ in entries]


# ---- Worker functions for multiprocessing ----

_w_cache = None
_w_phase = None


def _pool_init(phase: str):
    """Initialize worker: create cache and import the right tracker."""
    global _w_cache, _w_phase
    warnings.filterwarnings("ignore")
    from beatmeter.analysis.cache import AnalysisCache
    _w_cache = AnalysisCache()
    _w_phase = phase


def _pool_process_tracker(args: tuple[str, str]) -> bool:
    """Process one file for a beat tracker in a worker."""
    import soundfile as sf
    from beatmeter.analysis.engine import _beats_to_dicts

    path_str, ah = args
    phase = _w_phase
    path = Path(path_str)

    if phase == "madmom":
        from beatmeter.analysis.trackers.madmom_tracker import track_beats_madmom
        bpbs = [3, 4, 5, 7]
        if all(_w_cache.load_beats(ah, f"madmom_bpb{bpb}") is not None for bpb in bpbs):
            return False
        y = _load_audio(path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, 22050)
            tmp_path = tmp.name
        try:
            for bpb in bpbs:
                key = f"madmom_bpb{bpb}"
                if _w_cache.load_beats(ah, key) is not None:
                    continue
                try:
                    beats = track_beats_madmom(y, 22050, beats_per_bar=bpb, tmp_path=tmp_path)
                    _w_cache.save_beats(ah, key, _beats_to_dicts(beats))
                except Exception as e:
                    print(f"    ERR {path.name} bpb={bpb}: {e}", flush=True)
                    _w_cache.save_beats(ah, key, [])
        finally:
            os.unlink(tmp_path)
        return True

    # Non-madmom tracker
    if _w_cache.load_beats(ah, phase) is not None:
        return False

    if phase == "beatnet":
        from beatmeter.analysis.trackers.beatnet import track_beats_beatnet as fn
    elif phase == "beat_this":
        from beatmeter.analysis.trackers.beat_this import track_beats_beat_this as fn
    elif phase == "librosa":
        from beatmeter.analysis.trackers.librosa_tracker import track_beats_librosa as fn

    y = _load_audio(path)
    if phase in ("beatnet", "beat_this"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, 22050)
            tmp_path = tmp.name
        try:
            beats = fn(y, 22050, tmp_path=tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        beats = fn(y, 22050)

    _w_cache.save_beats(ah, phase, _beats_to_dicts(beats))
    return True


def _pool_process_onsets(args: tuple[str, str]) -> bool:
    """Process one file for onset detection in a worker."""
    path_str, ah = args
    if _w_cache.load_onsets(ah) is not None:
        return False
    from beatmeter.analysis.onset import detect_onsets, onset_strength_envelope
    y = _load_audio(Path(path_str))
    onsets = detect_onsets(y, 22050)
    onset_times, onset_strengths = onset_strength_envelope(y, 22050)
    onset_event_times = np.array([o.time for o in onsets])
    _w_cache.save_onsets(ah, {
        "onset_times": onset_times.tolist(),
        "onset_strengths": onset_strengths.tolist(),
        "onset_events": onset_event_times.tolist(),
    })
    return True


def _pool_process_onset_mlp(args: tuple[str, str]) -> bool:
    """Process one file for onset_mlp signal in a worker."""
    path_str, ah = args
    sig_name = "onset_mlp_meter"
    if _w_cache.load_signal(ah, sig_name) is not None:
        return False
    from beatmeter.analysis.signals.onset_mlp_meter import signal_onset_mlp_meter
    y = _load_audio(Path(path_str))
    try:
        scores = signal_onset_mlp_meter(y, 22050)
        _w_cache.save_signal(ah, sig_name, scores)
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    return True


def _pool_process_resnet(args: tuple[str, str]) -> bool:
    """Process one file for ResNet meter signal in a worker."""
    path_str, ah = args
    sig_name = "resnet_meter"
    if _w_cache.load_signal(ah, sig_name) is not None:
        return False
    from beatmeter.analysis.signals.resnet_meter import signal_resnet_meter
    y = _load_audio(Path(path_str))
    try:
        scores = signal_resnet_meter(y, 22050)
        # Convert tuple keys to string keys for JSON serialization
        scores_str = {f"{k[0]}_{k[1]}": v for k, v in scores.items()}
        _w_cache.save_signal(ah, sig_name, scores_str)
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    return True


def _pool_process_hcdf(args: tuple[str, str]) -> bool:
    """Process one file for HCDF meter signal in a worker."""
    path_str, ah = args
    sig_name = "hcdf_meter"
    if _w_cache.load_signal(ah, sig_name) is not None:
        return False
    from beatmeter.analysis.signals.hcdf_meter import signal_hcdf_meter
    y = _load_audio(Path(path_str))
    try:
        scores = signal_hcdf_meter(y, 22050)
        # Convert tuple keys to string keys for JSON serialization
        scores_str = {f"{k[0]}_{k[1]}": v for k, v in scores.items()}
        _w_cache.save_signal(ah, sig_name, scores_str)
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    return True


# ---- Phase runners ----

def _run_parallel(phase, entries, cache, workers, pool_fn, init_phase=None):
    """Run a phase with optional multiprocessing."""
    from beatmeter.analysis.cache import AnalysisCache

    audio_hashes = _get_audio_hashes(entries, cache)

    # Count cached
    if phase == "madmom":
        bpbs = [3, 4, 5, 7]
        already = sum(
            1 for _, ah in audio_hashes
            if all(cache.load_beats(ah, f"madmom_bpb{bpb}") is not None for bpb in bpbs)
        )
    elif phase in ("beatnet", "beat_this", "librosa"):
        already = sum(1 for _, ah in audio_hashes if cache.load_beats(ah, phase) is not None)
    elif phase == "onsets":
        already = sum(1 for _, ah in audio_hashes if cache.load_onsets(ah) is not None)
    elif phase == "onset_mlp":
        already = sum(1 for _, ah in audio_hashes if cache.load_signal(ah, "onset_mlp_meter") is not None)
    elif phase == "resnet":
        already = sum(1 for _, ah in audio_hashes if cache.load_signal(ah, "resnet_meter") is not None)
    elif phase == "hcdf":
        already = sum(1 for _, ah in audio_hashes if cache.load_signal(ah, "hcdf_meter") is not None)
    else:
        already = 0

    need = len(entries) - already
    if need == 0:
        print(f"  all {len(entries)} cached, skipping", flush=True)
        return

    print(f"  {already} cached, {need} to compute ({workers} workers)", flush=True)

    work_items = [(str(path), ah) for path, ah in audio_hashes]
    t_start = time.time()
    computed = 0

    if workers <= 1:
        _pool_init(init_phase or phase)
        for item in work_items:
            if pool_fn(item):
                computed += 1
                _progress(computed, need, t_start)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_pool_init, initargs=(init_phase or phase,), maxtasksperchild=50) as pool:
            for did_work in pool.imap_unordered(pool_fn, work_items):
                if did_work:
                    computed += 1
                    _progress(computed, need, t_start)

    elapsed = time.time() - t_start
    print(f"    Done: {computed} files in {elapsed:.0f}s", flush=True)


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Warm analysis cache per-tracker/signal")
    parser.add_argument("--data-dir", type=Path, default=Path("data/meter2800"))
    parser.add_argument("--split", default="all", help="all / tuning / test")
    parser.add_argument("--phase", default=None, help="Single phase to warm")
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Worker processes")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    entries = load_entries(data_dir, args.split)
    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"Loaded {len(entries)} files ({args.split} split)", flush=True)

    from beatmeter.analysis.cache import AnalysisCache
    cache = AnalysisCache()

    phases = [args.phase] if args.phase else ALL_PHASES

    phase_config = {
        "beatnet":   (_pool_process_tracker, "beatnet"),
        "beat_this": (_pool_process_tracker, "beat_this"),
        "madmom":    (_pool_process_tracker, "madmom"),
        "librosa":   (_pool_process_tracker, "librosa"),
        "onsets":    (_pool_process_onsets, "onsets"),
        "onset_mlp": (_pool_process_onset_mlp, "onset_mlp"),
        "resnet":    (_pool_process_resnet, "resnet"),
        "hcdf":      (_pool_process_hcdf, "hcdf"),
    }

    for phase in phases:
        if phase not in phase_config:
            print(f"\nUnknown phase: {phase}", flush=True)
            continue
        pool_fn, init_phase = phase_config[phase]
        w = 1 if phase in GPU_PHASES else args.workers
        print(f"\n=== {phase} ===", flush=True)
        _run_parallel(phase, entries, cache, w, pool_fn, init_phase)
        gc.collect()
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
