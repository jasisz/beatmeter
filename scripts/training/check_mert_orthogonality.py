#!/usr/bin/env python3
"""Check orthogonality of MERT signal vs current engine on benchmark fixtures.

Runs MERT signal on all benchmark fixtures, compares predictions with the
current engine's meter output, and reports agreement rate, complementarity,
and potential gains/losses.

Usage:
    uv run python scripts/check_mert_orthogonality.py
    uv run python scripts/check_mert_orthogonality.py --category waltz
    uv run python scripts/check_mert_orthogonality.py --verbose
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Suppress noisy logging
logging.basicConfig(level=logging.WARNING)
for _name in ("beatmeter", "numba", "madmom", "BeatNet", "PySoundFile"):
    logging.getLogger(_name).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from beatmeter.audio.loader import load_audio
from beatmeter.audio.preprocessing import preprocess
from beatmeter.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mert_prediction(audio: np.ndarray, sr: int) -> tuple[int, int] | None:
    """Run MERT signal and return the top-scoring meter."""
    try:
        import beatmeter.analysis.signals.mert_meter as _mert_mod
        scores = _mert_mod.signal_mert_meter(audio, sr)
        if not scores:
            return None
        return max(scores.items(), key=lambda x: x[1])[0]
    except Exception as e:
        print(f"  MERT error: {e}")
        return None


def get_engine_prediction(audio: np.ndarray, sr: int) -> tuple[int, int] | None:
    """Run the full engine and return its top meter prediction."""
    try:
        from beatmeter.analysis.engine import AnalysisEngine
        engine = AnalysisEngine()
        result = engine.analyze_audio(audio, sr)
        if result.meter_hypotheses:
            top = result.meter_hypotheses[0]
            return (top.numerator, top.denominator)
        return None
    except Exception as e:
        print(f"  Engine error: {e}")
        return None


def load_fixture_catalogue():
    """Import the FIXTURE_CATALOGUE from benchmark.py."""
    benchmark_path = Path(__file__).resolve().parent.parent / "tests"
    sys.path.insert(0, str(benchmark_path))
    from benchmark import FIXTURE_CATALOGUE, FIXTURES_DIR
    return FIXTURE_CATALOGUE, FIXTURES_DIR


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check MERT orthogonality vs engine")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to specific category")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached engine results from benchmark cache")
    args = parser.parse_args()

    catalogue, fixtures_dir = load_fixture_catalogue()
    sr = settings.sample_rate

    # Collect test cases from catalogue
    test_cases = []
    for filename, (category, expected_meters, bpm_range) in catalogue.items():
        if args.category and category != args.category:
            continue
        filepath = fixtures_dir / filename
        if filepath.exists():
            test_cases.append((filename, category, expected_meters, filepath))

    print(f"Running orthogonality check on {len(test_cases)} files...")

    # Optionally install cache wrappers
    if args.use_cache:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
        from benchmark import BenchmarkCache, CACHE_DIR
        cache = BenchmarkCache(CACHE_DIR)
        cache.install_wrappers()

    # Results
    both_correct = 0
    both_wrong = 0
    engine_only = 0  # engine correct, MERT wrong
    mert_only = 0    # MERT correct, engine wrong
    mert_fails = 0
    engine_fails = 0

    gains = []   # files where engine FAIL + MERT OK
    losses = []  # files where engine OK + MERT FAIL

    t_start = time.time()

    for i, (filename, category, expected_meters, filepath) in enumerate(test_cases):
        print(f"  Processing [{i+1}/{len(test_cases)}] {filename}...", flush=True)
        try:
            audio, _ = load_audio(str(filepath))
            audio = preprocess(audio, sr)
        except Exception as e:
            print(f"  [{i+1}] {filename}: load failed: {e}")
            continue

        # Get predictions
        mert_pred = get_mert_prediction(audio, sr)
        engine_pred = get_engine_prediction(audio, sr)

        if mert_pred is None:
            mert_fails += 1
            if args.verbose:
                print(f"  [{i+1}/{len(test_cases)}] {filename}: MERT failed")
            continue
        if engine_pred is None:
            engine_fails += 1
            if args.verbose:
                print(f"  [{i+1}/{len(test_cases)}] {filename}: Engine failed")
            continue

        mert_correct = mert_pred in expected_meters
        engine_correct = engine_pred in expected_meters

        if engine_correct and mert_correct:
            both_correct += 1
        elif not engine_correct and not mert_correct:
            both_wrong += 1
        elif engine_correct and not mert_correct:
            engine_only += 1
            losses.append((filename, category, expected_meters, engine_pred, mert_pred))
        else:  # mert correct, engine wrong
            mert_only += 1
            gains.append((filename, category, expected_meters, engine_pred, mert_pred))

        if args.verbose:
            marker = ""
            if mert_correct and not engine_correct:
                marker = " ** GAIN"
            elif engine_correct and not mert_correct:
                marker = " ** LOSS"
            print(f"  [{i+1}/{len(test_cases)}] {filename}: "
                  f"engine={engine_pred} mert={mert_pred} "
                  f"expected={expected_meters}{marker}")

        elapsed = time.time() - t_start
        if (i + 1) % 10 == 0:
            rate = (i + 1) / elapsed
            eta = (len(test_cases) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(test_cases)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    total_evaluated = both_correct + both_wrong + engine_only + mert_only
    elapsed = time.time() - t_start

    # Report
    print(f"\n{'=' * 60}")
    print(f"MERT Orthogonality Report ({total_evaluated} files, {elapsed:.0f}s)")
    print(f"{'=' * 60}")

    if total_evaluated == 0:
        print("No files evaluated!")
        return

    agreement = (both_correct + both_wrong) / total_evaluated
    mert_acc = (both_correct + mert_only) / total_evaluated
    engine_acc = (both_correct + engine_only) / total_evaluated

    print(f"\nAccuracy:")
    print(f"  Engine: {engine_acc:.1%} ({both_correct + engine_only}/{total_evaluated})")
    print(f"  MERT:   {mert_acc:.1%} ({both_correct + mert_only}/{total_evaluated})")

    print(f"\nAgreement matrix:")
    print(f"  Both correct:  {both_correct:4d}  ({both_correct/total_evaluated:.1%})")
    print(f"  Both wrong:    {both_wrong:4d}  ({both_wrong/total_evaluated:.1%})")
    print(f"  Engine only:   {engine_only:4d}  ({engine_only/total_evaluated:.1%}) [LOSSES if MERT added]")
    print(f"  MERT only:     {mert_only:4d}  ({mert_only/total_evaluated:.1%}) [GAINS if MERT added]")

    print(f"\nKey metrics:")
    print(f"  Agreement rate: {agreement:.1%}  (target: 0.65-0.80, NOT >0.85)")
    print(f"  Complementarity ratio: ", end="")
    if engine_only > 0:
        ratio = mert_only / engine_only
        print(f"{ratio:.2f}  (gains/losses, target: >1.5)")
    else:
        print(f"inf  (no losses)")

    print(f"\n  MERT failures: {mert_fails}")
    print(f"  Engine failures: {engine_fails}")

    # Gate check
    print(f"\n{'=' * 60}")
    print("GATE CHECK:")
    gate_pass = True
    if agreement > 0.85:
        print(f"  FAIL: agreement {agreement:.1%} > 85% (not orthogonal enough)")
        gate_pass = False
    elif agreement < 0.50:
        print(f"  WARN: agreement {agreement:.1%} < 50% (MERT may be unreliable)")

    if engine_only > 0 and mert_only / engine_only < 1.5:
        print(f"  FAIL: complementarity {mert_only/engine_only:.2f} < 1.5 (not enough gains)")
        gate_pass = False

    if gate_pass:
        print(f"  PASS: MERT signal is orthogonal and complementary!")
    print(f"{'=' * 60}")

    # Detail on gains/losses
    if gains:
        print(f"\nPotential GAINS ({len(gains)} files where MERT correct, engine wrong):")
        for fname, cat, expected, eng, mert in gains[:20]:
            print(f"  {fname} ({cat}): engine={eng} MERT={mert} expected={expected}")

    if losses:
        print(f"\nPotential LOSSES ({len(losses)} files where engine correct, MERT wrong):")
        for fname, cat, expected, eng, mert in losses[:20]:
            print(f"  {fname} ({cat}): engine={eng} MERT={mert} expected={expected}")


if __name__ == "__main__":
    main()
