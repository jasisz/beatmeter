# Publication Gaps — What's Missing

Status: **Work in Progress**
Last updated: 2026-02-24

## Current Results

| Benchmark | Score | CI (95%) | Previous SOTA |
|-----------|-------|----------|---------------|
| METER2800 test (700 files, 4 classes) | **639/700 (91.3%)** | — | 86.9% (ResNet18) |
| Ballroom (698 files, 2 classes) | **657/698 (94.1%)** | [92.3%, 95.8%] | 84.6% (SVM, Schuller 2008) |
| WIKIMETER test (298 segments, 6 classes) | 195/298 (65.4%) | — | — (no prior work) |

Method: MLP+FTT ensemble (w=0.50), zero-shot on Ballroom (no training on that data).

## What We Already Have

- [x] Large-scale benchmark (METER2800, 2800 files, 4 classes) — SOTA
- [x] Cross-dataset eval (Ballroom, 698 files, binary 3/4 vs 4/4) — SOTA
- [x] Third benchmark (WIKIMETER, 683 songs / 2937 segments, 6 classes)
- [x] Clear SOTA on METER2800 (+4.4 pp over ResNet18)
- [x] Clear SOTA on Ballroom (+9.5 pp over SVM, zero-shot)
- [x] Bootstrap confidence intervals (--ci flag)
- [x] Confusion matrices (--confusion flag)
- [x] Ablation study (audio-only vs MERT-only vs both)
- [x] Feature importance analysis (MERT activations, redundancy plots)
- [x] Architecture comparison (MLP vs FTT vs Hybrid vs Ensemble)
- [x] K-fold cross-validation (5-fold, stratified)
- [x] Negative results documented (hybrid worse than ensemble)
- [x] Training reproducibility (fixed seeds, deterministic splits)
- [x] Label correction methodology (manual audit of METER2800 errors)

## What's Still Missing (Priority Order)

### 1. Comparison with More Baselines — MEDIUM PRIORITY
**Why:** Strengthens claims.

- Librosa beat_track + heuristic meter
- madmom downbeat tracker + meter rule
- Random / majority-class baseline
- Would add 2-3 rows to comparison table

### 2. Runtime / Efficiency Analysis — LOW PRIORITY
**Why:** Practical deployment consideration.

- Inference time per file (with/without cache)
- Model size comparison (ResNet18 vs our MLP+FTT)
- Feature extraction breakdown

### 3. Per-Genre / Per-Source Analysis — LOW PRIORITY
**Why:** Interesting insight, not required.

- METER2800 has source tags (FMA_, MAG_, GTZAN_, OWN_)
- Ballroom has genre info (which dance styles we get wrong)
- Would reveal if model is biased toward certain production styles

## Completed Gaps

- ~~Cross-Dataset Generalization~~ → Ballroom eval: 94.1% (SOTA)
- ~~Statistical Significance Testing~~ → Bootstrap CI implemented
- ~~Confusion Matrix~~ → --confusion flag in eval.py

## What's NOT Missing (but might feel like it)

These were present in comparable papers (e.g., the ResNet18/METER2800 paper)
and we match or exceed them:

- **Detailed architecture description**: We have it (RESEARCH.md Section 2.1)
- **Hyperparameter search**: We have it (grid search, 40+ configs)
- **Training details**: We have them (learning rate, scheduler, augmentation)
- **Feature engineering justification**: We have it (ablation)
- **Reproducibility info**: Seeds, splits, checkpoint hashes logged

## Minimum Viable Paper

All three "must-haves" are done:
1. ~~Cross-dataset eval~~ → Ballroom 94.1%
2. ~~Confusion matrix~~ → eval.py --confusion
3. ~~Bootstrap confidence interval~~ → eval.py --ci

Remaining items (baselines, runtime, per-genre) are nice-to-have, not blockers.
The core contribution (DSP+MERT ensemble beating CNN-only by 4.4 pp on METER2800,
+9.5 pp on Ballroom zero-shot) is solid.
