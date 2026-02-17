#!/usr/bin/env python3
"""Warm the analysis cache by running one tracker/signal at a time.

Instead of loading ALL models per file (3GB+ RAM, GPU contention),
this script processes files one tracker/signal at a time:

  Phase 1: BeatNet     → all files → cache → free
  Phase 2: Beat This!  → all files → cache → free
  Phase 3: madmom      → all files → cache → free
  Phase 4: librosa     → all files → cache → free
  Phase 5: onsets      → all files → cache
  Phase 6: signals     → all files → cache (7 main meter signals from cached beats/onsets)
  Phase 7: tempo       → all files → cache (librosa + tempogram)
  Phase 8: onset_mlp   → all files → cache → free
  Phase 9: resnet      → all files → cache → free
  Phase 10: hcdf       → all files → cache

Supports --workers N for CPU-bound phases (beatnet, madmom, librosa, onsets,ok, to co robimy
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

ALL_PHASES = ["beatnet", "beat_this", "madmom", "librosa", "onsets", "signals", "tempo", "onset_mlp", "resnet", "hcdf"]
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


LABEL_TO_METER = {
    "three": 3, "four": 4, "five": 5,
    "seven": 7, "nine": 9, "eleven": 11,
}


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
            basename = Path(fname).name
            audio_path = audio_dir / basename
            if audio_path.exists():
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


def _pool_process_tempo(args: tuple[str, str]) -> bool:
    """Compute tempo_librosa and tempo_tempogram for one file."""
    path_str, ah = args
    has_librosa = _w_cache.load_signal(ah, "tempo_librosa") is not None
    has_tempogram = _w_cache.load_signal(ah, "tempo_tempogram") is not None
    if has_librosa and has_tempogram:
        return False

    from beatmeter.analysis.tempo import estimate_from_librosa, estimate_from_tempogram
    y = _load_audio(Path(path_str))

    if not has_librosa:
        try:
            est = estimate_from_librosa(y, 22050)
            if est:
                _w_cache.save_signal(ah, "tempo_librosa", {"bpm": est.bpm, "confidence": est.confidence, "method": est.method})
        except Exception as e:
            print(f"    ERR tempo_librosa {Path(path_str).name}: {e}", flush=True)

    if not has_tempogram:
        try:
            est = estimate_from_tempogram(y, 22050)
            if est:
                _w_cache.save_signal(ah, "tempo_tempogram", {"bpm": est.bpm, "confidence": est.confidence, "method": est.method})
        except Exception as e:
            print(f"    ERR tempo_tempogram {Path(path_str).name}: {e}", flush=True)

    return True


def _pool_process_audio_signal(args: tuple[str, str]) -> bool:
    """Process one file for any audio-only signal (onset_mlp, resnet, hcdf)."""
    path_str, ah = args
    phase = _w_phase
    sig_name = {"onset_mlp": "onset_mlp_meter", "resnet": "resnet_meter", "hcdf": "hcdf_meter"}[phase]
    if _w_cache.load_signal(ah, sig_name) is not None:
        return False
    y = _load_audio(Path(path_str))
    try:
        if phase == "onset_mlp":
            from beatmeter.analysis.signals.onset_mlp_meter import signal_onset_mlp_meter
            scores = signal_onset_mlp_meter(y, 22050)
        elif phase == "resnet":
            from beatmeter.analysis.signals.resnet_meter import signal_resnet_meter
            scores = signal_resnet_meter(y, 22050)
        else:
            from beatmeter.analysis.signals.hcdf_meter import signal_hcdf_meter
            scores = signal_hcdf_meter(y, 22050)
        _w_cache.save_signal(ah, sig_name, scores)
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    return True


# Signal names that meter._collect_signal_scores caches
_METER_SIGNAL_NAMES = [
    "beatnet_spacing", "beat_this_spacing", "madmom_activation",
    "onset_autocorr", "accent_pattern", "beat_periodicity", "bar_tracking",
]


def _pool_process_meter_signals(args: tuple[str, str]) -> bool:
    """Compute all 7 main meter signals using meter.py (reuses its cache logic)."""
    path_str, ah = args

    # Skip if all 7 already cached
    if all(_w_cache.load_signal(ah, s) is not None for s in _METER_SIGNAL_NAMES):
        return False

    import soundfile as sf
    from beatmeter.analysis.engine import _dicts_to_beats
    import beatmeter.analysis.meter as meter_mod

    # Load cached beats
    def _load_beats(tracker: str):
        data = _w_cache.load_beats(ah, tracker)
        return _dicts_to_beats(data) if data else []

    beatnet_beats = _load_beats("beatnet")
    beat_this_beats = _load_beats("beat_this") or None
    librosa_beats = _load_beats("librosa") or None
    madmom_results = {}
    for bpb in [3, 4, 5, 7]:
        beats = _load_beats(f"madmom_bpb{bpb}")
        if beats:
            madmom_results[bpb] = beats

    # Load cached onsets
    onset_data = _w_cache.load_onsets(ah)
    if onset_data is None:
        return False
    onset_times = np.array(onset_data["onset_times"])
    onset_strengths = np.array(onset_data["onset_strengths"])

    # Pick best beats for primary + compute tempo
    all_tracker_beats = [beatnet_beats, beat_this_beats, librosa_beats]
    for beats in madmom_results.values():
        all_tracker_beats.append(beats)
    all_beats = max((b for b in all_tracker_beats if b), key=len, default=[])
    beat_times = np.array([b.time for b in all_beats]) if all_beats else np.array([])

    # Tempo from median IBI
    tempo_bpm = None
    if len(beat_times) >= 3:
        ibis = np.diff(beat_times)
        valid = ibis[(ibis > 0.1) & (ibis < 3.0)]
        if len(valid) >= 2:
            tempo_bpm = 60.0 / float(np.median(valid))
    beat_interval = 60.0 / tempo_bpm if tempo_bpm and tempo_bpm > 0 else None

    # Load audio (needed for accent, periodicity, bar_tracking)
    y = _load_audio(Path(path_str))

    # Write tmp wav for bar_tracking (madmom needs it)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, 22050)
        tmp_path = tmp.name

    try:
        # All weights = 1.0 to force all signals to compute
        weights = {k: 1.0 for k in [
            "beatnet", "beat_this", "madmom", "autocorr", "accent",
            "periodicity", "bar_tracking", "resnet", "mert", "onset_mlp",
        ]}

        # Signals 1a, 1b, 2, 3, 7 (with cache read/write built in)
        signal_results = meter_mod._collect_signal_scores(
            weights=weights,
            beatnet_beats=beatnet_beats,
            madmom_results=madmom_results,
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            beat_interval=beat_interval,
            beat_times=beat_times,
            sr=22050,
            audio=y,
            beat_this_beats=beat_this_beats,
            skip_bar_tracking=False,
            skip_resnet=True,  # already warmed separately
            skip_mert=True,
            skip_onset_mlp=True,  # already warmed separately
            cache=_w_cache,
            audio_hash=ah,
            tmp_path=tmp_path,
        )

        # Signals 4, 5 (accent + periodicity, with cache read/write built in)
        meter_mod._collect_accent_scores(
            weights=weights,
            signal_results=signal_results,
            all_beats=all_beats,
            beatnet_beats=beatnet_beats,
            beat_this_beats=beat_this_beats,
            librosa_beats=librosa_beats,
            madmom_results=madmom_results,
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            audio=y,
            sr=22050,
            tempo_bpm=tempo_bpm,
            cache=_w_cache,
            audio_hash=ah,
        )
    except Exception as e:
        print(f"    ERR {Path(path_str).name}: {e}", flush=True)
    finally:
        os.unlink(tmp_path)

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
    elif phase == "signals":
        already = sum(
            1 for _, ah in audio_hashes
            if all(cache.load_signal(ah, s) is not None for s in _METER_SIGNAL_NAMES)
        )
    elif phase == "tempo":
        already = sum(
            1 for _, ah in audio_hashes
            if cache.load_signal(ah, "tempo_librosa") is not None
            and cache.load_signal(ah, "tempo_tempogram") is not None
        )
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
    parser.add_argument("--wikimeter-dir", type=Path, default=Path("data/wikimeter"))
    parser.add_argument("--split", default="all", help="all / tuning / test")
    parser.add_argument("--extra-data", action="store_true", help="Include WIKIMETER")
    parser.add_argument("--phase", default=None, help="Single phase to warm")
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Worker processes")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    entries = load_entries(data_dir, args.split)
    if args.extra_data:
        wiki_entries = load_wikimeter_entries(args.wikimeter_dir.resolve())
        entries.extend(wiki_entries)
    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"Loaded {len(entries)} files ({args.split} split{' + WIKIMETER' if args.extra_data else ''})", flush=True)

    from beatmeter.analysis.cache import AnalysisCache
    cache = AnalysisCache()

    phases = [args.phase] if args.phase else ALL_PHASES

    phase_config = {
        "beatnet":   (_pool_process_tracker, "beatnet"),
        "beat_this": (_pool_process_tracker, "beat_this"),
        "madmom":    (_pool_process_tracker, "madmom"),
        "librosa":   (_pool_process_tracker, "librosa"),
        "onsets":    (_pool_process_onsets, "onsets"),
        "signals":   (_pool_process_meter_signals, "signals"),
        "tempo":     (_pool_process_tempo, "tempo"),
        "onset_mlp": (_pool_process_audio_signal, "onset_mlp"),
        "resnet":    (_pool_process_audio_signal, "resnet"),
        "hcdf":      (_pool_process_audio_signal, "hcdf"),
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
