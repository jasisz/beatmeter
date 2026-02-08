#!/usr/bin/env python3
"""Diagnose meter detection failures on real audio files.

Runs each failure case through individual meter signals and reports
detailed scores to identify the root cause of wrong answers.

Run: uv run python tests/diagnose_failures.py
"""
import sys
import os
import logging
import numpy as np
from collections import Counter

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
from beatmeter.audio.loader import load_audio
from beatmeter.audio.preprocessing import preprocess

SR = 22050
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Failure cases: (filename, expected_meter, got_meter, description)
FAILURE_CASES = [
    ("shuffle.ogg", (4, 4), (7, 4), "Shuffle rhythm"),
    ("reggae_one_drop.ogg", (4, 4), (3, 4), "Reggae One-Drop"),
    ("baladi.ogg", (4, 4), (3, 4), "Baladi (Middle-Eastern)"),
    ("drum_cadence_b.ogg", (4, 4), (9, 4), "Drum Cadence B"),
    ("blowing_bubbles_waltz.ogg", (3, 4), (5, 4), "I'm Forever Blowing Bubbles waltz"),
    ("chopin_waltz.ogg", (3, 4), (5, 4), "Chopin Waltz A minor"),
    ("blue_danube.ogg", (3, 4), (11, 4), "Blue Danube Waltz"),
]


def _fmt_scores(scores, top_n=8):
    """Format score dict for display."""
    if not scores:
        return "(empty)"
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{n}/{d}={v:.3f}" for (n, d), v in sorted_s[:top_n]]
    return ", ".join(parts)


def _onset_alignment_score(beats, onset_event_times, max_dist=0.07):
    """Compute F1-like onset alignment score (same logic as engine)."""
    if not beats or len(onset_event_times) == 0:
        return 0.0
    beat_times = np.array([b.time for b in beats])

    # Precision: beats -> onsets
    total_fwd = 0.0
    for b in beats:
        min_dist = float(np.min(np.abs(onset_event_times - b.time)))
        total_fwd += min(min_dist, max_dist)
    precision = 1.0 - total_fwd / len(beats) / max_dist

    # Recall: onsets -> beats
    total_rev = 0.0
    for ot in onset_event_times:
        min_dist = float(np.min(np.abs(beat_times - ot)))
        total_rev += min(min_dist, max_dist)
    recall = 1.0 - total_rev / len(onset_event_times) / max_dist

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return f1, precision, recall


def diagnose_file(filename, expected_meter, got_meter, description):
    """Run full diagnosis on a single failure case."""
    filepath = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  SKIP: {filepath} not found")
        return None

    print(f"\n{'='*80}")
    print(f"FAILURE: {description}")
    print(f"  File: {filename}")
    print(f"  Expected: {expected_meter[0]}/{expected_meter[1]}")
    print(f"  Engine returned: {got_meter[0]}/{got_meter[1]}")
    print(f"{'='*80}")

    # Load and preprocess
    audio, sr = load_audio(filepath, sr=SR)
    audio = preprocess(audio, sr)
    duration = len(audio) / sr
    print(f"\n  Duration: {duration:.1f}s, {len(audio)} samples")

    # --- Onset detection ---
    onset_times, onset_strengths = onset_strength_envelope(audio, sr)
    onsets = detect_onsets(audio, sr)
    onset_event_times = np.array([o.time for o in onsets])
    print(f"  Onsets: {len(onsets)} events, envelope has {len(onset_times)} points")

    # --- Beat tracking (all trackers) ---
    print(f"\n  --- Beat Tracking ---")

    beatnet_beats = track_beats_beatnet(audio, sr)
    n_bn_db = sum(1 for b in beatnet_beats if b.is_downbeat)
    print(f"  BeatNet: {len(beatnet_beats)} beats, {n_bn_db} downbeats")
    if beatnet_beats:
        db_indices = [i for i, b in enumerate(beatnet_beats) if b.is_downbeat]
        if len(db_indices) >= 2:
            spacings = [db_indices[i+1] - db_indices[i] for i in range(len(db_indices)-1)]
            print(f"    Downbeat spacings: {Counter(spacings)}")
        # Show BeatNet IBI
        bn_times = np.array([b.time for b in beatnet_beats])
        bn_ibis = np.diff(bn_times)
        if len(bn_ibis) > 0:
            med_ibi = np.median(bn_ibis)
            print(f"    Median IBI: {med_ibi:.4f}s => {60/med_ibi:.1f} BPM")

    madmom_results = {}
    for bpb in [3, 4, 5, 7]:
        mb = track_beats_madmom(audio, sr, beats_per_bar=bpb)
        if mb:
            madmom_results[bpb] = mb
            n_db = sum(1 for b in mb if b.is_downbeat)
            mm_times = np.array([b.time for b in mb])
            mm_ibis = np.diff(mm_times)
            med_ibi = float(np.median(mm_ibis)) if len(mm_ibis) > 0 else 0
            print(f"  madmom ({bpb}/4): {len(mb)} beats, {n_db} downbeats, "
                  f"IBI={med_ibi:.4f}s => {60/med_ibi:.1f} BPM" if med_ibi > 0 else
                  f"  madmom ({bpb}/4): {len(mb)} beats, {n_db} downbeats")

    librosa_beats = track_beats_librosa(audio, sr)
    if librosa_beats:
        lib_times = np.array([b.time for b in librosa_beats])
        lib_ibis = np.diff(lib_times)
        med_ibi = float(np.median(lib_ibis)) if len(lib_ibis) > 0 else 0
        print(f"  librosa: {len(librosa_beats)} beats, "
              f"IBI={med_ibi:.4f}s => {60/med_ibi:.1f} BPM" if med_ibi > 0 else
              f"  librosa: {len(librosa_beats)} beats")

    # --- Onset alignment (primary selection) ---
    print(f"\n  --- Onset Alignment (Primary Selection) ---")
    tracker_candidates = []
    if beatnet_beats:
        tracker_candidates.append(('BeatNet', beatnet_beats))
    for bpb, beats in madmom_results.items():
        tracker_candidates.append((f'madmom_{bpb}', beats))
    if librosa_beats:
        tracker_candidates.append(('librosa', librosa_beats))

    primary_beats = librosa_beats
    primary_name = 'librosa'
    best_alignment = 0.0
    beatnet_alignment = 0.0
    madmom_best_alignment = 0.0

    for name, beats in tracker_candidates:
        if not beats:
            continue
        result = _onset_alignment_score(beats, onset_event_times)
        f1, prec, rec = result
        print(f"  {name}: F1={f1:.3f} (precision={prec:.3f}, recall={rec:.3f}), {len(beats)} beats")

        if name == 'BeatNet':
            beatnet_alignment = f1
        elif name.startswith('madmom'):
            madmom_best_alignment = max(madmom_best_alignment, f1)

        if f1 > best_alignment:
            best_alignment = f1
            primary_beats = beats
            primary_name = name

    print(f"  => SELECTED PRIMARY: {primary_name} (F1={best_alignment:.3f})")

    # --- Tempo ---
    print(f"\n  --- Tempo ---")
    if primary_beats and len(primary_beats) >= 3:
        times = np.array([b.time for b in primary_beats])
        ibis = np.diff(times)
        valid_ibis = ibis[(ibis > 0.15) & (ibis < 2.0)]
        beat_interval = float(np.median(valid_ibis)) if len(valid_ibis) > 0 else None
    else:
        beat_interval = None

    if beat_interval:
        print(f"  Beat interval (primary): {beat_interval:.4f}s => {60/beat_interval:.1f} BPM")
    else:
        print(f"  Beat interval: N/A")

    ibi_est = estimate_from_ibi(primary_beats, 40, 300)
    lib_est = estimate_from_librosa(audio, sr)
    tgram_est = estimate_from_tempogram(audio, sr)
    print(f"  IBI: {ibi_est.bpm:.1f} BPM (conf={ibi_est.confidence:.2f})" if ibi_est else "  IBI: N/A")
    print(f"  librosa: {lib_est.bpm:.1f} BPM (conf={lib_est.confidence:.2f})" if lib_est else "  librosa: N/A")
    print(f"  tempogram: {tgram_est.bpm:.1f} BPM (conf={tgram_est.confidence:.2f})" if tgram_est else "  tempogram: N/A")

    # --- Individual meter signals ---
    print(f"\n  --- Meter Signal Scores ---")

    s1 = signal_downbeat_spacing(beatnet_beats)
    print(f"  Signal 1 (BeatNet downbeat):   {_fmt_scores(s1)}")

    s2 = signal_madmom_activation(madmom_results, onset_times, onset_strengths)
    print(f"  Signal 2 (madmom activation):  {_fmt_scores(s2)}")

    s3 = signal_onset_autocorrelation(onset_times, onset_strengths, beat_interval, sr)
    print(f"  Signal 3 (onset autocorr):     {_fmt_scores(s3)}")

    # Signal 4 & 5 with primary beats
    primary_energies = compute_beat_energies(primary_beats, audio, sr)
    s4 = signal_accent_pattern(primary_beats, primary_energies)
    print(f"  Signal 4 (accent, primary):    {_fmt_scores(s4)}")

    s5 = signal_beat_strength_periodicity(primary_energies)
    print(f"  Signal 5 (periodicity, prim):  {_fmt_scores(s5)}")

    # Signals 4 & 5 for ALL trackers (multi-tracker analysis like engine does)
    print(f"\n  --- Multi-Tracker Accent Analysis (as engine does) ---")
    accent_trackers = []
    if primary_beats and len(primary_beats) >= 8:
        accent_trackers.append(('primary', primary_beats))
    if beatnet_beats and len(beatnet_beats) >= 8:
        accent_trackers.append(('beatnet', beatnet_beats))
    if librosa_beats and len(librosa_beats) >= 8:
        accent_trackers.append(('librosa', librosa_beats))
    for bpb, beats in madmom_results.items():
        if len(beats) >= 8:
            accent_trackers.append((f'madmom_{bpb}', beats))

    merged_accent = {}
    merged_periodicity = {}

    for tname, tbeats in accent_trackers:
        energies = compute_beat_energies(tbeats, audio, sr)
        max_e = float(np.max(energies)) if len(energies) > 0 else 0
        if max_e > 0:
            active_frac = float(np.sum(energies > max_e * 0.1)) / len(energies)
        else:
            active_frac = 0.0

        skipped = ""
        if active_frac < 0.7:
            skipped = " ** SKIPPED (active_frac too low) **"

        s4_raw = signal_accent_pattern(tbeats, energies, normalize=False)
        s5_raw = signal_beat_strength_periodicity(energies, normalize=False)

        # Show top scores for this tracker
        s4_top = sorted(s4_raw.items(), key=lambda x: x[1], reverse=True)[:3]
        s5_top = sorted(s5_raw.items(), key=lambda x: x[1], reverse=True)[:3]
        s4_str = ", ".join(f"{n}/{d}={v:.3f}" for (n, d), v in s4_top) if s4_top else "(empty)"
        s5_str = ", ".join(f"{n}/{d}={v:.3f}" for (n, d), v in s5_top) if s5_top else "(empty)"

        print(f"  {tname}: active={active_frac:.2f}{skipped}")
        print(f"    accent(raw):      {s4_str}")
        print(f"    periodicity(raw): {s5_str}")

        if active_frac >= 0.7:
            for meter, score in s4_raw.items():
                if score > merged_accent.get(meter, 0):
                    merged_accent[meter] = score
            for meter, score in s5_raw.items():
                if score > merged_periodicity.get(meter, 0):
                    merged_periodicity[meter] = score

    # Normalize merged
    for scores_dict in [merged_accent, merged_periodicity]:
        if scores_dict:
            max_s = max(scores_dict.values())
            if max_s > 0:
                for k in scores_dict:
                    scores_dict[k] /= max_s

    print(f"\n  Merged accent (normalized):      {_fmt_scores(merged_accent)}")
    print(f"  Merged periodicity (normalized): {_fmt_scores(merged_periodicity)}")

    # --- Beat energy pattern ---
    print(f"\n  --- Beat Energy Pattern (Primary: {primary_name}) ---")
    if len(primary_energies) >= 8:
        # Show first 24 beat energies
        n_show = min(24, len(primary_energies))
        max_e = np.max(primary_energies) if np.max(primary_energies) > 0 else 1
        normalized_e = primary_energies[:n_show] / max_e
        pattern_str = " ".join(f"{e:.2f}" for e in normalized_e)
        print(f"  First {n_show} beats (normalized RMS): {pattern_str}")

        # For expected meter, show average accent pattern
        exp_bpb = expected_meter[0]
        n_bars = len(primary_energies) // exp_bpb
        if n_bars >= 2:
            trimmed = primary_energies[:n_bars * exp_bpb]
            bars = trimmed.reshape(n_bars, exp_bpb)
            avg = np.mean(bars, axis=0) / max_e
            pattern_str = " ".join(f"{e:.3f}" for e in avg)
            print(f"  Avg pattern (expected {exp_bpb}/4): [{pattern_str}]")

            # Check if beat 1 is consistently strongest
            max_positions = [int(np.argmax(bar)) for bar in bars]
            mode = Counter(max_positions).most_common()
            print(f"    Strongest-beat positions: {mode}")
            cv = float(np.std(avg) / np.mean(avg)) if np.mean(avg) > 0 else 0
            print(f"    Pattern CV (contrast): {cv:.3f}")

        # Also show for the wrong meter
        wrong_bpb = got_meter[0]
        n_bars_w = len(primary_energies) // wrong_bpb
        if n_bars_w >= 2 and wrong_bpb != exp_bpb:
            trimmed_w = primary_energies[:n_bars_w * wrong_bpb]
            bars_w = trimmed_w.reshape(n_bars_w, wrong_bpb)
            avg_w = np.mean(bars_w, axis=0) / max_e
            pattern_str_w = " ".join(f"{e:.3f}" for e in avg_w)
            print(f"  Avg pattern (wrong {wrong_bpb}/4):    [{pattern_str_w}]")
            cv_w = float(np.std(avg_w) / np.mean(avg_w)) if np.mean(avg_w) > 0 else 0
            print(f"    Pattern CV (contrast): {cv_w:.3f}")
    else:
        print(f"  Too few beats ({len(primary_energies)}) to analyze energy pattern")

    # --- Full engine result ---
    print(f"\n  --- Full Engine Result ---")
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr)
    print(f"  Tempo: {result.tempo.bpm} BPM (confidence={result.tempo.confidence})")

    for i, h in enumerate(result.meter_hypotheses):
        marker = ""
        if h.numerator == expected_meter[0] and h.denominator == expected_meter[1]:
            marker = " <-- EXPECTED"
        if i == 0:
            marker += " <-- TOP"
        print(f"    #{i+1}: {h.numerator}/{h.denominator} conf={h.confidence:.3f} ({h.description}){marker}")

    top = result.meter_hypotheses[0] if result.meter_hypotheses else None
    correct = top and top.numerator == expected_meter[0] and top.denominator == expected_meter[1]
    print(f"\n  RESULT: {'CORRECT' if correct else 'WRONG'}")

    # --- Weight analysis ---
    print(f"\n  --- Weight Analysis ---")
    beatnet_trust = max(0.0, min(1.0, (beatnet_alignment - 0.4) / 0.4)) if beatnet_beats else 0.0
    madmom_trust = max(0.0, min(1.0, (madmom_best_alignment - 0.4) / 0.4)) if madmom_results else 0.0

    w_beatnet = 0.15 * beatnet_trust
    w_madmom = 0.15 * madmom_trust
    w_autocorr = 0.15
    w_accent = 0.20
    w_periodicity = 0.35
    total_w = w_beatnet + w_madmom + w_autocorr + w_accent + w_periodicity

    print(f"  BeatNet trust={beatnet_trust:.2f} => weight={w_beatnet/total_w:.3f}")
    print(f"  madmom  trust={madmom_trust:.2f} => weight={w_madmom/total_w:.3f}")
    print(f"  autocorr => weight={w_autocorr/total_w:.3f}")
    print(f"  accent   => weight={w_accent/total_w:.3f}")
    print(f"  period.  => weight={w_periodicity/total_w:.3f}")

    # Compute weighted scores to see what the engine computes
    print(f"\n  --- Reconstructed Final Scores ---")
    final_scores = {}
    weights = {
        "beatnet": w_beatnet / total_w,
        "madmom": w_madmom / total_w,
        "autocorr": w_autocorr / total_w,
        "accent": w_accent / total_w,
        "periodicity": w_periodicity / total_w,
    }

    for meter, score in s1.items():
        final_scores[meter] = final_scores.get(meter, 0) + score * weights["beatnet"]
    for meter, score in s2.items():
        final_scores[meter] = final_scores.get(meter, 0) + score * weights["madmom"]
    for meter, score in s3.items():
        final_scores[meter] = final_scores.get(meter, 0) + score * weights["autocorr"]
    for meter, score in merged_accent.items():
        final_scores[meter] = final_scores.get(meter, 0) + score * weights["accent"]
    for meter, score in merged_periodicity.items():
        final_scores[meter] = final_scores.get(meter, 0) + score * weights["periodicity"]

    # Apply priors
    METER_PRIOR = {(4, 4): 1.10, (3, 4): 1.08, (6, 8): 1.05, (2, 4): 1.02, (12, 8): 1.02}
    for meter, prior in METER_PRIOR.items():
        if meter in final_scores:
            final_scores[meter] *= prior

    sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top scores (pre-normalization):")
    for (n, d), v in sorted_final[:8]:
        marker = ""
        if (n, d) == expected_meter:
            marker = " <-- EXPECTED"
        if (n, d) == got_meter:
            marker += " <-- ENGINE TOP"
        print(f"    {n}/{d} = {v:.4f}{marker}")

    return correct


def main():
    print("=" * 80)
    print("METER DETECTION FAILURE DIAGNOSIS")
    print("=" * 80)
    print(f"Analyzing {len(FAILURE_CASES)} failure cases...")

    results = []
    for filename, expected, got, desc in FAILURE_CASES:
        try:
            correct = diagnose_file(filename, expected, got, desc)
            results.append((desc, correct))
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((desc, False))

    # Summary
    print(f"\n\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    for desc, correct in results:
        status = "FIXED" if correct else "STILL WRONG"
        print(f"  [{status}] {desc}")

    fixed = sum(1 for _, c in results if c)
    print(f"\n  Fixed: {fixed}/{len(results)}")


if __name__ == "__main__":
    main()
