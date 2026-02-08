#!/usr/bin/env python3
"""Test the engine on real audio files (Wikimedia Commons, public domain).

Run: uv run python tests/test_real_audio.py

All files from Wikimedia Commons with known time signatures.
"""
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logging.getLogger("beatmeter").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("madmom").setLevel(logging.ERROR)
logging.getLogger("BeatNet").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from beatmeter.analysis.engine import AnalysisEngine
from beatmeter.analysis.beat_tracking import track_beats_beatnet
from beatmeter.audio.loader import load_audio
from beatmeter.audio.preprocessing import preprocess
from beatmeter.config import settings

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def get_beatnet_raw_meter(beats) -> tuple[int, int] | None:
    """Extract meter from raw BeatNet output (downbeat spacing)."""
    if not beats:
        return None
    db_indices = [i for i, b in enumerate(beats) if b.is_downbeat]
    if len(db_indices) < 2:
        return None
    spacings = [db_indices[i + 1] - db_indices[i] for i in range(len(db_indices) - 1)]
    from collections import Counter
    most_common = Counter(spacings).most_common(1)[0][0]
    return (most_common, 4)


def run_test(name, filepath, expected_meter, expected_bpm_range, max_duration=30.0):
    """Run analysis on a real audio file."""
    if not os.path.exists(filepath):
        print(f"  SKIP: {name} - file not found")
        return None, None, None

    if isinstance(expected_meter, list):
        meters_str = " or ".join(f"{m[0]}/{m[1]}" for m in expected_meter)
    else:
        meters_str = f"{expected_meter[0]}/{expected_meter[1]}"

    # Load and preprocess
    audio, sr = load_audio(filepath, sr=settings.sample_rate)
    audio = preprocess(audio, sr)

    # Limit duration for speed
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    duration = len(audio) / sr

    # Raw BeatNet
    beatnet_beats = track_beats_beatnet(audio, sr)
    raw_meter = get_beatnet_raw_meter(beatnet_beats)

    # Full engine
    engine = AnalysisEngine()
    result = engine.analyze_audio(audio, sr)

    bpm_ok = expected_bpm_range[0] <= result.tempo.bpm <= expected_bpm_range[1]

    # Check engine meter
    meter_ok = False
    acceptable = expected_meter if isinstance(expected_meter, list) else [expected_meter]
    top_meter = None
    if result.meter_hypotheses:
        top = result.meter_hypotheses[0]
        top_meter = f"{top.numerator}/{top.denominator}"
        for m in acceptable:
            if top.numerator == m[0] and top.denominator == m[1]:
                meter_ok = True

    # Check raw BeatNet meter
    beatnet_meter_ok = False
    if raw_meter:
        for m in acceptable:
            if raw_meter[0] == m[0] and raw_meter[1] == m[1]:
                beatnet_meter_ok = True

    bn_str = f"{raw_meter[0]}/{raw_meter[1]}" if raw_meter else "N/A"

    print(f"  {'OK' if meter_ok else 'FAIL':4s} {'OK' if beatnet_meter_ok else 'FAIL':4s} "
          f"{'OK' if bpm_ok else 'FAIL':4s} | "
          f"eng={top_meter or 'N/A':5s} bn={bn_str:5s} bpm={result.tempo.bpm:5.0f} | "
          f"expect {meters_str:10s} {expected_bpm_range[0]}-{expected_bpm_range[1]:3.0f} | "
          f"{name}")

    return meter_ok, beatnet_meter_ok, bpm_ok


def main():
    print("=" * 100)
    print("REAL AUDIO TEST - ENGINE vs RAW BEATNET")
    print("=" * 100)

    tests = [
        # === DRUM PATTERNS (4/4) ===
        ("Rock Beat (4/4)",
         os.path.join(FIXTURES, "rock_beat.ogg"),
         [(4, 4), (2, 4)], (80, 180)),

        ("Drum Beat (4/4)",
         os.path.join(FIXTURES, "drum_beat.ogg"),
         [(4, 4), (2, 4)], (80, 180)),

        ("Jazz Ride (4/4 swing)",
         os.path.join(FIXTURES, "jazz_ride.ogg"),
         [(4, 4), (2, 4)], (80, 200)),

        ("Shuffle (4/4)",
         os.path.join(FIXTURES, "shuffle.ogg"),
         [(4, 4), (2, 4)], (80, 200)),

        ("Blast Beat (4/4, fast metal)",
         os.path.join(FIXTURES, "blast_beat.ogg"),
         [(4, 4), (2, 4)], (150, 300)),

        ("Dubstep Drums (4/4, half-time feel)",
         os.path.join(FIXTURES, "dubstep_drums.ogg"),
         [(4, 4), (2, 4)], (60, 160)),

        ("Reggae One-Drop (4/4)",
         os.path.join(FIXTURES, "reggae_one_drop.ogg"),
         [(4, 4), (2, 4)], (60, 160)),

        ("Bossa Nova (4/4)",
         os.path.join(FIXTURES, "bossa_nova.ogg"),
         [(4, 4), (2, 4)], (70, 170)),

        # === MIDDLE EASTERN RHYTHMS ===
        # Ayub: 2/4 (dum-tek pattern, fast)
        ("Ayub (2/4, Middle Eastern)",
         os.path.join(FIXTURES, "ayub.ogg"),
         [(2, 4), (4, 4)], (80, 200)),

        # Baladi: 4/4 (dum-dum-tek-dum-tek)
        ("Baladi (4/4, Middle Eastern)",
         os.path.join(FIXTURES, "baladi.ogg"),
         [(4, 4), (2, 4)], (60, 160)),

        # Maksum: 4/4 (dum-tek--tek-dum--tek)
        ("Maksum (4/4, Middle Eastern)",
         os.path.join(FIXTURES, "maksum.ogg"),
         [(4, 4), (2, 4)], (60, 160)),

        # Malfuf: 2/4 (dum-tek-tek)
        ("Malfuf (2/4, Middle Eastern)",
         os.path.join(FIXTURES, "malfuf.ogg"),
         [(2, 4), (4, 4)], (80, 200)),

        # Saidi: 4/4 (dum-tek-dum-dum-tek)
        ("Saidi (4/4, Middle Eastern)",
         os.path.join(FIXTURES, "saidi.ogg"),
         [(4, 4), (2, 4)], (60, 160)),

        # === DJEMBE ===
        ("Djembe Accompaniment",
         os.path.join(FIXTURES, "djembe.ogg"),
         [(4, 4), (2, 4), (3, 4), (6, 4)], (60, 200)),  # unknown meter

        # === DRUM CADENCES (marching, 4/4 or 2/4) ===
        ("Drum Cadence A (march)",
         os.path.join(FIXTURES, "drum_cadence_a.ogg"),
         [(4, 4), (2, 4)], (80, 180)),

        ("Drum Cadence B (march)",
         os.path.join(FIXTURES, "drum_cadence_b.ogg"),
         [(4, 4), (2, 4)], (80, 180)),

        # === WALTZES (3/4) ===
        ("Greensleeves (3/4 or 6/8)",
         os.path.join(FIXTURES, "greensleeves.ogg"),
         [(3, 4), (6, 4), (6, 8)], (60, 170)),

        ("Midnight Waltz (3/4)",
         os.path.join(FIXTURES, "midnight_waltz.ogg"),
         (3, 4), (80, 200)),

        ("Joplin Harmony Club Waltz (3/4)",
         os.path.join(FIXTURES, "joplin_waltz.ogg"),
         (3, 4), (80, 200)),

        ("I'm Forever Blowing Bubbles (3/4 waltz)",
         os.path.join(FIXTURES, "blowing_bubbles_waltz.ogg"),
         (3, 4), (80, 200)),

        # === CLASSICAL (hard mode) ===
        ("Chopin Waltz A minor (3/4, rubato)",
         os.path.join(FIXTURES, "chopin_waltz.ogg"),
         (3, 4), (100, 200)),

        ("Blue Danube Waltz (3/4, orchestral)",
         os.path.join(FIXTURES, "blue_danube.ogg"),
         (3, 4), (100, 220)),
    ]

    print(f"  {'ENG':4s} {'BN':4s} {'BPM':4s} | {'eng':5s}  {'bn':5s}  {'bpm':>5s} | "
          f"{'expected':10s} {'range':>7s} | name")
    print(f"  {'-'*4} {'-'*4} {'-'*4} | {'-'*5}  {'-'*5}  {'-'*5} | {'-'*10} {'-'*7} | {'-'*30}")

    results = []
    for name, filepath, expected_meter, expected_bpm_range in tests:
        try:
            engine_ok, beatnet_ok, bpm_ok = run_test(
                name, filepath, expected_meter, expected_bpm_range)
            if engine_ok is not None:
                results.append((name, engine_ok, beatnet_ok, bpm_ok))
        except Exception as e:
            print(f"  EXCEPTION on {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, False, False))

    # Summary
    total = len(results)
    if total == 0:
        print("\nNo tests ran (no fixture files found?)")
        return

    engine_pass = sum(1 for _, e, _, _ in results if e)
    beatnet_pass = sum(1 for _, _, b, _ in results if b)
    bpm_pass = sum(1 for _, _, _, b in results if b)

    print(f"\n{'=' * 100}")
    print(f"SUMMARY ({total} tests)")
    print(f"  Engine meter:  {engine_pass}/{total} ({engine_pass / total * 100:.0f}%)")
    print(f"  BeatNet raw:   {beatnet_pass}/{total} ({beatnet_pass / total * 100:.0f}%)")
    print(f"  Tempo:         {bpm_pass}/{total} ({bpm_pass / total * 100:.0f}%)")


if __name__ == "__main__":
    main()
