#!/bin/bash
# Sweep ensemble weights through the real eval pipeline
echo "=== ENSEMBLE WEIGHT SWEEP (via eval.py) ==="
echo "w = weight of primary model (meter_net.pt)"
echo ""

for w in 0.00 0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80 0.90 1.00; do
    result=$(METER_NET_ENSEMBLE=1 METER_NET_ENSEMBLE_W=$w uv run python scripts/eval.py meter2800 --split test --workers 4 2>/dev/null | grep "METER2800 test:")
    echo "w=$w  $result"
done
