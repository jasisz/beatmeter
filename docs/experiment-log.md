# Meter Detection — Experiment Log

Chronological log of all experiments to improve meter detection accuracy.
Each entry records the change, result, regressions, and decision.

---

## Round 3 (2026-02-08): Parameter Tuning

### Baseline: 53/72 (74%)

### R3-1: Product-of-Experts (PoE)
- **Change**: Replace additive combination with weighted geometric mean
- **Result**: 50/72 (71%) — WORSE by 3
- **Why**: A single signal giving 0 to a meter kills it entirely.
- **Decision**: REVERTED.

### R3-2: Weight rebalance + consensus bonus + NN 3/4 penalty
- **Changes**: Periodicity 0.30→0.24, BeatNet 0.12→0.15, Beat This! 0.15→0.18, priors, consensus 3+→1.15x 4+→1.25x, NN 3/4 penalty 0.65x
- **Result**: 54/72 (75%) — +1
- **Decision**: KEPT.

### R3-3: Trust threshold 0.25
- **Result**: 54/72 (75%) — same total, 6 regressions
- **Root cause**: NNs report 2/4 for 6/8 content → overwhelms compound detection
- **Key finding**: Trust threshold and compound detection are COUPLED
- **Decision**: REVERTED.

### R3-4: Trust threshold 0.30
- **Result**: 52/72 (72%) — WORSE. Same coupling.
- **Decision**: REVERTED.

### R3-5..7: Compound /8 threshold tuning (1.7, 2.2, reduced transfer)
- All neutral or worse. Threshold changes can't fix compound + trust coupling.
- **Decision**: All REVERTED.

### R3 Final: 54/72 (75%) with Exp 2 parameters.

---

## Round 4 (2026-02-08): Signal 7 + Compound Fixes

### Baseline: 54/72 (75%)

### R4-1: Signal 7 — DBNBarTrackingProcessor (w=0.12)
- madmom GRU-RNN + Viterbi bar tracking
- Quality gate: skip on sparse/synthetic audio (non_silent < 0.3)
- Weight taken from NN signals (not periodicity/accent)
- **Result**: +1 net

### R4-2: Evenness check for compound /8
- CV > 0.4 → not compound (blocks ornamental sub-beats)
- **Result**: +1 net

### R4 Final: 55/72 (76%). New baseline saved.

---

## Round 5 (2026-02-09): Compound Detection + Weight Tuning

### Baseline: 55/72 (76%) meter, 66/72 (92%) tempo

| Category | Baseline |
|----------|----------|
| barcarolle | 2/3 (67%) |
| blues | 0/2 (0%) |
| classical | 5/8 (62%) |
| drums | 11/12 (92%) |
| edge_case | 5/6 (83%) |
| jig | 4/5 (80%) |
| march | 2/4 (50%) |
| middle_eastern | 5/5 (100%) |
| polka | 3/4 (75%) |
| synthetic | 10/11 (91%) |
| tango | 1/2 (50%) |
| tarantella | 4/5 (80%) |
| waltz | 3/5 (60%) |

### R5-1: Signal 8 — Spectrogram multi-band accent (w=0.07)
- **Result**: 55/72 (76%) — SAME, 5 regressions
- Weight redistribution gains march+polka+tango but loses odd/compound meters.
- **Root cause**: Spectrogram signal correlated with Signals 4+5. Not orthogonal.
- **Decision**: REVERTED.

### R5-2: Signal 8 — Conservative weights (w=0.05)
- **Result**: 54/72 (75%) — WORSE, 4 regressions
- **Decision**: REVERTED. Signal 8 (spectrogram accent) abandoned.

### R5-3: Minimum NN trust floor (0.10)
- **Result**: 52/72 (72%) — MUCH WORSE, 6 regressions
- NNs report 2/4 for compound content. Cannot give them weight.
- **Decision**: REVERTED.

### R5-4: NN penalty via nn_downbeat_results + compound guard
- **Result**: 54/72 (75%) — 1 regression (bach_siciliana)
- NN penalty reduces 3/4 before compound transfer → 6/8 loses source
- **Decision**: REVERTED.

### R5-5: NN penalty AFTER compound detection
- **Result**: 54/72 (75%) — 1 regression (bach_siciliana)
- bach_siciliana has no compound evidence, penalty still reduces 3/4
- **Decision**: REVERTED.

### R5-6: Periodicity weight cap when NNs untrusted ✅
- When all NNs have zero trust, periodicity gets ~44% weight → cap to 0.17 (~31%)
- **Result at 0.17**: 56/72 (78%) — +1 (tango), zero regressions
- Also tested 0.18 (neutral), 0.15 (breaks jig/tarantella)
- **Decision**: KEPT. `w_periodicity = 0.17` when `total_nn_trust < 0.01`.

### R5-7: Accent weight boost when NNs untrusted
- **Result**: 53/72 (74%) — WORSE, 3 regressions
- Accent also breaks compound meters.
- **Decision**: REVERTED.

### R5-8: Tighter compound detection (median range + CV threshold)
- Tested median 1.7-2.8 and CV 0.35/0.30 — all neutral
- False compound files have genuinely evenly-spaced sub-beats (CV < 0.30)
- **Decision**: REVERTED.

### R5-9: Triplet position consistency check ✅
- Require sub-onsets at ~1/3 and ~2/3 of beat interval (tolerance 0.15)
- True compound (jig, tarantella) → onsets at triplet positions → pass
- False compound (waltz, polka, classical) → accompaniment positions → blocked
- **Result**: 59/72 (82%) — +3, zero regressions
- Fixes: sarabande_bach, polka_tritsch_tratsch, waltz_stefan
- **Decision**: KEPT.

### R5 Final: 59/72 (82%). New baseline saved.

| Category | R4 Baseline | R5 Final | Change |
|----------|-------------|----------|--------|
| barcarolle | 2/3 (67%) | 2/3 (67%) | — |
| blues | 0/2 (0%) | 0/2 (0%) | — |
| classical | 5/8 (62%) | **6/8 (75%)** | **+1** |
| drums | 11/12 (92%) | 11/12 (92%) | — |
| edge_case | 5/6 (83%) | 5/6 (83%) | — |
| jig | 4/5 (80%) | 4/5 (80%) | — |
| march | 2/4 (50%) | 2/4 (50%) | — |
| middle_eastern | 5/5 (100%) | 5/5 (100%) | — |
| polka | 3/4 (75%) | **4/4 (100%)** | **+1** |
| synthetic | 10/11 (91%) | 10/11 (91%) | — |
| tango | 1/2 (50%) | **2/2 (100%)** | **+1** |
| tarantella | 4/5 (80%) | 4/5 (80%) | — |
| waltz | 3/5 (60%) | **4/5 (80%)** | **+1** |

---

## Key Learnings

1. **Trust threshold and compound detection are coupled** — lowering trust helps marches but breaks 6/8.
2. **NNs don't understand compound meters** — they report 2/4 or 4/4 for 6/8 content. Never give them weight for compound decisions.
3. **Periodicity dominates when NNs absent** — capping it at 0.17 (31%) prevents false 3/4.
4. **CV-based evenness isn't enough** — false compound files have genuinely even sub-beats. Positional check (triplet at 1/3+2/3) is more discriminative.
5. **New signals must be orthogonal** — spectrogram accent is too correlated with RMS accent. Need CNN/metrogram.
6. **Reducing periodicity below 0.20 breaks odd meter detection** (5/4, 7/8).
7. **Parameter tuning ceiling ~82%** — structural changes (CNN classifier, better beat tracking) needed for 85%+.

## Remaining Failure Patterns

- **Pattern A** (false 3/4 on duple): 4 files — march, blues, drum. Periodicity forces 3/4.
- **Pattern B** (false compound /8): down from 6 to 2 files after triplet check.
- **Pattern C** (low NN trust + wrong tracker): 4 files — classical, waltz with extreme tracker failure.
