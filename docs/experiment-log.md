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

## Round 6 (2026-02-09): BeatNet Fix + Re-tuning

### Discovery: BeatNet was broken (pyaudio missing)
- BeatNet import failed silently; all Round 3-5 results used stale cache from an older run.
- `uv add pyaudio` fixed the import. Cache cleared, fresh baseline established.

### Baseline: 58/72 (81%) meter, 66/72 (92%) tempo (with live BeatNet)

| Category | R5 (cached BN) | R6 Baseline (live BN) | Change |
|----------|----------------|----------------------|--------|
| blues | 0/2 (0%) | **1/2 (50%)** | **+1** |
| synth | 9/11 | **10/11 (91%)** | **+1** |
| march | 3/4 (75%) | 2/4 (50%) | -1 |
| waltz | 4/5 (80%) | 4/5 (80%) | — |
| tango | 2/2 (100%) | 1/2 (50%) | -1 |
| polka | 4/4 (100%) | 3/4 (75%) | -1 |

### R6-1: Trust ramp 0.2→0.6 (was 0.4→0.8)
- **Result**: 56/72 (78%), 5 regressions
- NNs override compound meters (jig, tarantella) and 3/4 classical.
- **Decision**: REVERTED.

### R6-2: Trust ramp 0.3→0.7
- **Result**: 53/72 (74%), 5 regressions
- Even worse. Trust threshold and compound detection remain coupled.
- **Decision**: REVERTED.

### R6-3: Cap bar_tracking 0.12→0.08 when NNs untrusted
- **Result**: 57/72 (79%), 1 regression (blues_guitar)
- Bar tracking doesn't always bias 3/4 — for blues_guitar it correctly said 4/4.
  Capping it removed that correct signal.
- **Decision**: REVERTED.

### R6-4: Cap accent+bar_tracking when NNs untrusted
- **Result**: 57/72 (79%), 1 regression (blues_guitar)
- Same result as R6-3. Accent cap had no additional effect.
- **Decision**: REVERTED.

### R6-5: Untrusted NN weak vote for 3/4 penalty ✅
- When all NNs are untrusted AND compound not detected, compute raw downbeat
  spacing from BeatNet/Beat This! beats. If ALL available NNs favor even meter
  (even > 0.4, triple < 0.3), apply mild 3/4 penalty (×0.65).
- Tested penalties: 0.80, 0.75, 0.70, 0.65 (all gave 59/72), 0.55 (regressed bach_siciliana)
- **Result**: 59/72 (82%), zero regressions
- Fix: tango_albeniz now correct (4/4).
- **Decision**: KEPT at 0.65.

### R6-6: Relaxed condition (any NN even, no NN triple)
- **Result**: 57/72 (79%), 5 regressions (3 tarantellas, sarabande, edge)
- Beat This! says even for tarantellas → penalty fires on genuine 3/4.
- **Root cause**: NNs say 2/4 for 6/8 content. Cannot relax without compound guard.
- **Decision**: REVERTED.

### R6 Final: 59/72 (82%). New baseline saved.

| Category | R6 Baseline | R6 Final | Change |
|----------|-------------|----------|--------|
| tango | 1/2 (50%) | **2/2 (100%)** | **+1** |
| (all others unchanged) |

---

## Round 7 (2026-02-09): Signal 8 Attempts

### Baseline: 59/72 (82%) meter, 66/72 (92%) tempo

### R7-1: Signal 8 — Tempogram ratio
- **Standalone test**: Computed tempogram at 2-beat and 3-beat periods, took ratio.
- All files (duple and triple) gave ratios ~1.20. No discrimination.
- **Decision**: ABANDONED without integration. Dead end.

### R7-2: Signal 8 — Chromagram HCDF periodicity (w=0.07)
- **Standalone test**: HCDF autocorrelation correctly separated duple from triple in 8/10 test files.
- **Integrated**: w_hcdf=0.07, w_bar_tracking 0.12→0.11 to accommodate.
- **Result**: 56/72 (78%) — WORSE, 6 regressions (synth 5/4, synth 7/8, decel 3/4, bach_siciliana, blues_guitar, jig_doethion)
- **Root cause**: HCDF disrupts odd meter detection (5/4, 7/8) and compound meters. Weight redistribution from bar_tracking hurts. Signal may be correlated with periodicity (both use autocorrelation of audio features).
- **Decision**: REVERTED.

### R7 Final: 59/72 (82%). No change from baseline.

---

## Key Learnings

1. **Trust threshold and compound detection are coupled** — lowering trust helps marches but breaks 6/8.
2. **NNs don't understand compound meters** — they report 2/4 or 4/4 for 6/8 content. Never give them weight for compound decisions.
3. **Periodicity dominates when NNs absent** — capping it at 0.17 (31%) prevents false 3/4.
4. **CV-based evenness isn't enough** — false compound files have genuinely even sub-beats. Positional check (triplet at 1/3+2/3) is more discriminative.
5. **New signals must be orthogonal** — spectrogram accent is too correlated with RMS accent. Need CNN/metrogram.
6. **Reducing periodicity below 0.20 breaks odd meter detection** (5/4, 7/8).
7. **Parameter tuning ceiling ~82%** — structural changes (CNN classifier, better beat tracking) needed for 85%+.
8. **BeatNet pyaudio dependency** — BeatNet import fails silently without pyaudio. Always verify trackers actually run.
9. **Untrusted NNs carry useful duple/triple info** — raw downbeat spacing from untrusted NNs can provide weak 3/4 penalty, but ONLY with strict consensus (all NNs agree) and compound guard.
10. **Capping non-NN signals is fragile** — bar_tracking and accent don't always bias 3/4. Capping them loses correct signal for some files.
11. **Always test signals standalone first** — tempogram ratio dead end caught in seconds; HCDF integration without standalone test would've been harder to debug.
12. **HCDF autocorrelation not orthogonal enough** — discriminates duple/triple standalone but when integrated, disrupts odd meters (5/4, 7/8). Both HCDF and periodicity use autocorrelation → correlated.

## Remaining Failure Patterns (13 meter failures)

- **Pattern A** (false 3/4 on duple): 6 files — 19_16_drum, erika_march, irish_reel, lost_train_blues, march_grandioso, polka_smetana. Periodicity+accent+bar_tracking force 3/4 when NNs untrusted or ambiguous.
- **Pattern B** (false 4/4 on 3/4): 2 files — blowing_bubbles_waltz, sarabande_bach.
- **Pattern C** (compound miss): 2 files — barcarolle_offenbach, tarantella_napoletana.
- **Pattern D** (other): mazurka→7/4, synth 5/8→4/4, edge 4/4 short→3/4.
