#!/usr/bin/env python3
"""Diagnostic script: synthesize known rhythms and see what the engine detects.

Run: uv run python tests/diagnose_quality.py
"""
import sys
import os
import logging
import numpy as np

# Suppress noisy library logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logging.getLogger("beatmeter").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("madmom").setLevel(logging.ERROR)
logging.getLogger("BeatNet").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.analysis.meter import (
    signal_downbeat_spacing,
    signal_madmom_activation,
    signal_onset_autocorrelation,
    signal_accent_pattern,
    signal_beat_strength_periodicity,
    compute_beat_energies,
)
from beatmeter.analysis.onset import detect_onsets, onset_strength_envelope
from beatmeter.analysis.beat_tracking import (
    track_beats_beatnet,
    track_beats_madmom,
    track_beats_librosa,
)
from beatmeter.analysis.tempo import estimate_from_ibi, estimate_from_librosa, estimate_from_tempogram

SR = 22050


def generate_click(bpm, beats_per_bar, duration=15.0, accent_ratio=3.0, sr=SR):
    """Generate a click track with accented downbeats."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    beat_interval = 60.0 / bpm
    click_dur = 0.015
    click_samples = int(click_dur * sr)
    t_click = np.arange(click_samples) / sr

    # Downbeat: higher pitch, louder
    click_accent = np.sin(2 * np.pi * 1200 * t_click) * np.exp(-t_click * 120)
    click_normal = np.sin(2 * np.pi * 800 * t_click) * np.exp(-t_click * 120)

    beat = 0
    time = 0.0
    while time < duration:
        pos = int(time * sr)
        is_downbeat = (beat % beats_per_bar) == 0
        click = click_accent if is_downbeat else click_normal
        amplitude = accent_ratio if is_downbeat else 1.0

        end = min(pos + click_samples, n_samples)
        length = end - pos
        if length > 0:
            audio[pos:end] += click[:length] * amplitude

        time += beat_interval
        beat += 1

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def generate_compound_click(bpm, groups, duration=15.0, sr=SR):
    """Generate a click track with compound meter (e.g., 7/8 = 2+2+3).

    bpm = tempo of the smallest unit (eighth note).
    groups = list of group sizes, e.g., [2, 2, 3] for 7/8.
    """
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    unit_interval = 60.0 / bpm  # interval per eighth note
    click_dur = 0.015
    click_samples = int(click_dur * sr)
    t_click = np.arange(click_samples) / sr

    click_accent = np.sin(2 * np.pi * 1200 * t_click) * np.exp(-t_click * 120) * 3.0
    click_group = np.sin(2 * np.pi * 1000 * t_click) * np.exp(-t_click * 120) * 2.0
    click_normal = np.sin(2 * np.pi * 800 * t_click) * np.exp(-t_click * 120) * 1.0

    beats_per_bar = sum(groups)
    time = 0.0
    while time < duration:
        beat_in_bar = 0
        for gi, group_size in enumerate(groups):
            for beat_in_group in range(group_size):
                pos = int(time * sr)
                if pos >= n_samples:
                    break
                if beat_in_bar == 0:
                    click = click_accent  # downbeat
                elif beat_in_group == 0:
                    click = click_group   # group accent
                else:
                    click = click_normal  # regular

                end = min(pos + click_samples, n_samples)
                length = end - pos
                if length > 0:
                    audio[pos:end] += click[:length]

                time += unit_interval
                beat_in_bar += 1
            if int(time * sr) >= n_samples:
                break

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def diagnose_single(name, audio, expected_meter, expected_bpm, sr=SR):
    """Run full diagnosis on one test case."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"  Expected: {expected_meter[0]}/{expected_meter[1]} at ~{expected_bpm} BPM")
    print(f"{'='*70}")

    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s, {len(audio)} samples")

    # --- Onset detection ---
    onset_times, onset_strengths = onset_strength_envelope(audio, sr)
    onsets = detect_onsets(audio, sr)
    print(f"\n  Onsets: {len(onsets)} detected, envelope has {len(onset_times)} points")

    # --- Beat tracking ---
    print(f"\n  --- Beat Tracking ---")

    beatnet_beats = track_beats_beatnet(audio, sr)
    print(f"  BeatNet: {len(beatnet_beats)} beats, "
          f"{sum(1 for b in beatnet_beats if b.is_downbeat)} downbeats")
    if beatnet_beats:
        # Show downbeat spacings
        db_indices = [i for i, b in enumerate(beatnet_beats) if b.is_downbeat]
        if len(db_indices) >= 2:
            spacings = [db_indices[i+1] - db_indices[i] for i in range(len(db_indices)-1)]
            from collections import Counter
            print(f"    BeatNet downbeat spacings: {Counter(spacings)}")

    madmom_results = {}
    for bpb in [3, 4, 5, 7]:
        mb = track_beats_madmom(audio, sr, beats_per_bar=bpb)
        if mb:
            madmom_results[bpb] = mb
            n_db = sum(1 for b in mb if b.is_downbeat)
            print(f"  madmom ({bpb}/4): {len(mb)} beats, {n_db} downbeats")

    librosa_beats = track_beats_librosa(audio, sr)
    print(f"  librosa: {len(librosa_beats)} beats")

    # Choose primary beats (same logic as engine)
    if 4 in madmom_results:
        primary_beats = madmom_results[4]
        primary_source = "madmom 4/4"
    elif madmom_results:
        primary_beats = next(iter(madmom_results.values()))
        primary_source = f"madmom (first)"
    elif beatnet_beats:
        primary_beats = beatnet_beats
        primary_source = "BeatNet"
    else:
        primary_beats = librosa_beats
        primary_source = "librosa"
    print(f"  Primary beats: {primary_source} ({len(primary_beats)} beats)")

    # --- Tempo ---
    print(f"\n  --- Tempo ---")
    ibi_est = estimate_from_ibi(primary_beats, 40, 300)
    lib_est = estimate_from_librosa(audio, sr)
    tgram_est = estimate_from_tempogram(audio, sr)
    print(f"  IBI estimate: {ibi_est.bpm if ibi_est else 'N/A'} BPM "
          f"(conf={ibi_est.confidence:.2f})" if ibi_est else "  IBI: N/A")
    print(f"  librosa estimate: {lib_est.bpm if lib_est else 'N/A'} BPM" if lib_est else "  librosa: N/A")
    print(f"  tempogram estimate: {tgram_est.bpm if tgram_est else 'N/A'} BPM" if tgram_est else "  tempogram: N/A")

    # Compute beat interval for meter signals
    if primary_beats and len(primary_beats) >= 3:
        times = np.array([b.time for b in primary_beats])
        ibis = np.diff(times)
        beat_interval = float(np.median(ibis[(ibis > 0.15) & (ibis < 2.0)])) if len(ibis) > 0 else None
    else:
        beat_interval = None
    print(f"  Median beat interval: {beat_interval:.4f}s" if beat_interval else "  Beat interval: N/A")
    if beat_interval:
        print(f"  => implied BPM: {60.0/beat_interval:.1f}")

    # --- Meter signals ---
    print(f"\n  --- Meter Signals ---")

    s1 = signal_downbeat_spacing(beatnet_beats)
    print(f"  Signal 1 (BeatNet downbeat): {_fmt_scores(s1)}")

    s2 = signal_madmom_activation(madmom_results, onset_times, onset_strengths)
    print(f"  Signal 2 (madmom accent): {_fmt_scores(s2)}")

    s3 = signal_onset_autocorrelation(onset_times, onset_strengths, beat_interval, sr)
    print(f"  Signal 3 (autocorrelation): {_fmt_scores(s3)}")

    # Compute beat energies from raw audio for Signals 4 & 5
    primary_energies = compute_beat_energies(primary_beats, audio, sr)
    s4 = signal_accent_pattern(primary_beats, primary_energies)
    print(f"  Signal 4 (accent pattern): {_fmt_scores(s4)}")

    # Use the engine's actual primary (best-aligned with onsets)
    onset_event_times = np.array([o.time for o in onsets])
    candidates_for_alignment = []
    if beatnet_beats:
        candidates_for_alignment.append(('BeatNet', beatnet_beats))
    for bpb, beats in madmom_results.items():
        candidates_for_alignment.append((f'madmom_{bpb}', beats))
    if librosa_beats:
        candidates_for_alignment.append(('librosa', librosa_beats))

    engine_primary = librosa_beats
    engine_primary_name = 'librosa'
    best_align = 0.0
    for name, beats in candidates_for_alignment:
        if not beats:
            continue
        total_d = 0.0
        for b in beats:
            min_d = float(np.min(np.abs(onset_event_times - b.time)))
            total_d += min(min_d, 0.07)
        align = 1.0 - (total_d / len(beats)) / 0.07
        if align > best_align:
            best_align = align
            engine_primary = beats
            engine_primary_name = name
    print(f"\n  Engine primary: {engine_primary_name} (alignment={best_align:.3f}, {len(engine_primary)} beats)")

    engine_energies = compute_beat_energies(engine_primary, audio, sr)
    s5 = signal_beat_strength_periodicity(engine_energies)
    print(f"  Signal 5 (periodicity, engine primary): {_fmt_scores(s5)}")
    s4e = signal_accent_pattern(engine_primary, engine_energies)
    print(f"  Signal 4 (accent, engine primary): {_fmt_scores(s4e)}")

    # --- Full engine ---
    print(f"\n  --- Full Engine Result ---")
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr)

    print(f"  Tempo: {result.tempo.bpm} BPM (confidence={result.tempo.confidence})")
    bpm_ok = abs(result.tempo.bpm - expected_bpm) / expected_bpm < 0.10
    print(f"  Tempo correct (±10%): {'YES' if bpm_ok else 'NO'}")

    meter_correct = False
    for i, h in enumerate(result.meter_hypotheses):
        marker = ""
        if h.numerator == expected_meter[0] and h.denominator == expected_meter[1]:
            marker = " <-- EXPECTED"
            if i == 0:
                meter_correct = True
        print(f"    #{i+1}: {h.numerator}/{h.denominator} conf={h.confidence:.3f} "
              f"({h.description}){marker}")

    print(f"\n  METER CORRECT (top-1): {'YES' if meter_correct else 'NO'}")
    print(f"  TEMPO CORRECT (±10%): {'YES' if bpm_ok else 'NO'}")

    return meter_correct, bpm_ok


def _fmt_scores(scores):
    if not scores:
        return "(empty)"
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{n}/{d}={v:.3f}" for (n, d), v in sorted_s[:6]]
    return ", ".join(parts)


def main():
    print("=" * 70)
    print("RHYTHM ANALYZER - COMPREHENSIVE QUALITY DIAGNOSTIC")
    print("=" * 70)

    tests = []

    # --- Simple meters ---
    tests.append(("4/4 @ 120 BPM", generate_click(120, 4), (4, 4), 120))
    tests.append(("4/4 @ 90 BPM", generate_click(90, 4), (4, 4), 90))
    tests.append(("4/4 @ 160 BPM", generate_click(160, 4), (4, 4), 160))
    tests.append(("3/4 @ 100 BPM", generate_click(100, 3), (3, 4), 100))
    tests.append(("3/4 @ 140 BPM (fast waltz)", generate_click(140, 3), (3, 4), 140))
    tests.append(("2/4 @ 120 BPM (march)", generate_click(120, 2), (2, 4), 120))
    tests.append(("6/8 @ 80 BPM", generate_click(80, 6, accent_ratio=3.0), (6, 4), 80))

    # --- Odd meters ---
    tests.append(("5/4 @ 110 BPM", generate_click(110, 5), (5, 4), 110))
    tests.append(("7/4 @ 100 BPM", generate_click(100, 7), (7, 4), 100))

    # --- Compound meters with grouping ---
    tests.append(("7/8 (2+2+3) @ 200 BPM", generate_compound_click(200, [2, 2, 3]), (7, 8), 200))
    tests.append(("5/8 (3+2) @ 220 BPM", generate_compound_click(220, [3, 2]), (5, 8), 220))

    results = []
    for name, audio, expected_meter, expected_bpm in tests:
        try:
            meter_ok, bpm_ok = diagnose_single(name, audio, expected_meter, expected_bpm)
            results.append((name, meter_ok, bpm_ok))
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, False))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    meter_pass = sum(1 for _, m, _ in results if m)
    bpm_pass = sum(1 for _, _, b in results if b)
    total = len(results)

    for name, meter_ok, bpm_ok in results:
        status = f"{'M:OK' if meter_ok else 'M:FAIL'} {'T:OK' if bpm_ok else 'T:FAIL'}"
        print(f"  [{status}] {name}")

    print(f"\n  Meter: {meter_pass}/{total} correct ({meter_pass/total*100:.0f}%)")
    print(f"  Tempo: {bpm_pass}/{total} correct ({bpm_pass/total*100:.0f}%)")


if __name__ == "__main__":
    main()
