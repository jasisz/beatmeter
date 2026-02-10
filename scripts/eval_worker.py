#!/usr/bin/env python3
"""Persistent eval worker — loads engine once, processes files from stdin.

Used by eval.py's WorkerPool. Not meant to be run directly.
Reads file paths from stdin (one per line), writes JSON results to stdout.
"""

import gc
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

from beatmeter.analysis.cache import AnalysisCache
from beatmeter.analysis.engine import AnalysisEngine

cache = AnalysisCache()
engine = AnalysisEngine(cache=cache)

print("READY", flush=True)

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        with torch.inference_mode():
            result = engine.analyze_file(path)
        if result and result.meter_hypotheses:
            best = result.meter_hypotheses[0]
            bpm = result.tempo.bpm if result.tempo else None
            print(json.dumps({"meter": best.numerator, "bpm": bpm}), flush=True)
        else:
            print(json.dumps({"meter": None, "bpm": None}), flush=True)
    except Exception as e:
        print(json.dumps({"meter": None, "bpm": None, "error": str(e)}), flush=True)
    finally:
        gc.collect()
