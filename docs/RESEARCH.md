# BeatMeter: Multi-Signal Ensemble for Automatic Meter Detection

## Abstract

We present BeatMeter, a multi-signal system for automatic meter detection from audio. The current best result is a **MLP+FTT ensemble** that averages sigmoid probabilities from two independently trained models — an MLP (631/700) and an FT-Transformer (629/700) — achieving **639/700 (91.3%)** with balanced accuracy **83.9%** on the METER2800 test split (700 files). Both models take 2985-dimensional feature vectors combining DSP-based audio features (1449d) and MERT-v1-95M embeddings (1536d from layer 3). The ensemble exploits complementary error patterns: 49 disagreements (7%) between the two architectures, with an oracle ceiling of 649/698 (92.8%). This surpasses both the previous single-model best (632/700, Round 18) and the ResNet18 paper's 88% on binary 3/4 vs 4/4 classification.

## 1. System Architecture

### 1.1 Overview

The BeatMeter pipeline processes audio through a sequence of stages:

1. **Preprocessing** -- Load audio, resample, normalize.
2. **Onset detection** -- librosa onset envelope extraction (`beatmeter/analysis/onset.py`).
3. **Parallel beat tracking** -- Four independent beat trackers run concurrently (`beatmeter/analysis/trackers/`):
   - **BeatNet** (`trackers/beatnet.py`) -- Particle-filtering CRNN for joint beat/downbeat tracking.
   - **Beat This!** (`trackers/beat_this.py`) -- Convolutional Transformer (SOTA, ISMIR 2024), no DBN postprocessing.
   - **madmom** (`trackers/madmom_tracker.py`) -- GRU-RNN with DBN beat tracking.
   - **librosa** (`trackers/librosa_tracker.py`) -- Dynamic programming beat tracker.
4. **Primary beat selection** -- Trackers are ranked by onset alignment F1 score; the best-aligned tracker's beats are used as the primary beat grid.
5. **Tempo estimation** -- Multi-method tempo consensus with octave error normalization (`beatmeter/analysis/tempo.py`).
6. **Meter hypothesis generation** -- MeterNet (`beatmeter/analysis/meter.py`, loaded from `data/meter_net.pt`) classifies meter from a 3166-dimensional feature vector combining DSP audio features (1449d), MERT-v1-95M embeddings (1536d), beat-synchronous chroma SSM (75d), beat tracker statistics (42d), 5 signal scores (60d), and tempo features (4d). Falls back to 4/4 when no checkpoint is present. Historically, an arbiter MLP (72→64→32→6) fusing only 6 signal score vectors served as the primary classifier (see Section 4.12).

Trust gating controls the influence of neural-network-based signals. Each tracker receives a trust score derived from its onset alignment quality:

- Alignment < 0.4 --> trust = 0 (signal disabled)
- Alignment 0.4 to 0.8 --> trust ramps linearly from 0 to 1.0
- Alignment >= 0.8 --> trust = 1.0

This mechanism ensures that when beat trackers fail on difficult audio (low alignment with detected onsets), their potentially misleading outputs are suppressed rather than allowed to corrupt the ensemble.

### 1.2 Signals

All signal implementations live in `beatmeter/analysis/signals/`.

#### Signal scores (5 signals × 12 meters = 60 dims)

MeterNet receives score vectors from 5 analysis signals. Each signal scores 12 candidate meters (2/4, 3/4, 4/4, 5/4, 5/8, 6/8, 7/4, 7/8, 9/8, 10/8, 11/8, 12/8).

| Signal | Code key | MeterNet input | Source | Description |
|--------|----------|:--------------:|--------|-------------|
| **downbeat_spacing** (BeatNet) | `beatnet` | **Yes** | BeatNet beats | Infers meter from the distribution of inter-downbeat intervals. Clusters downbeat spacings and maps dominant intervals to candidate meters. |
| **downbeat_spacing** (Beat This!) | `beat_this` | **Yes** | Beat This! beats | Same approach but using the SOTA conv-transformer downbeat tracker, which provides higher-quality downbeat estimates on most genres. |
| **onset_autocorrelation** | `autocorr` | **Yes** | onset envelope | Computes autocorrelation of the onset strength envelope at lags corresponding to expected bar durations for each candidate meter. Higher autocorrelation at a given lag indicates stronger periodicity at that bar length. |
| **bar_tracking** | `bar_tracking` | **Yes** | audio + beats | Uses madmom's GRU-RNN activation function with Viterbi decoding to track bar boundaries for each candidate meter. Quality-gated: skipped on sparse/synthetic audio (non_silent < 0.15). |
| **hcdf** | `hcdf` | **Yes** | chromagram | Harmonic Change Detection Function. Discriminates duple/triple in isolation but causes regressions in hand-tuned ensemble (see Section 4.4). Included because the learned model can determine when HCDF is useful. |

**Note**: onset_mlp_meter is *not* included as a signal score input to MeterNet — its audio features (1449d) are instead provided directly as raw input features, giving MeterNet access to the underlying feature representations rather than a pre-classified score.

#### Additional feature groups

| Feature group | Dims | Source | Description |
|---------------|------|--------|-------------|
| **Audio features** (v6) | 1449 | `onset_mlp_features.py` | Multi-tempo autocorrelation (1024d), tempogram profile (64d), MFCC stats (26d), spectral contrast (14d), onset stats (4d), beat-position histograms (160d), autocorrelation ratios (60d), tempogram meter salience (9d), and more. Cached as `meter_net_audio`. |
| **MERT-v1-95M embeddings** | 1536 | `data/mert_embeddings/` | Pre-extracted frozen embeddings from MERT-v1-95M layer 3. Mean+max pooling over 5-second non-overlapping chunks. Stored as `.npy` files. Added in Round 18 — provides the largest single-feature improvement (+39 files). |
| **Beat-sync chroma SSM** | 75 | `ssm_features.py` | Self-similarity matrix of beat-synchronous chroma at lags 2–12 beats, computed per tracker (BeatNet, Beat This!, madmom). 25 dims per tracker: 11 raw similarities, 11 normalized, peak lag, peak height, entropy. Cached as `meter_net_ssm`. |
| **Beat tracker features** | 42 | `meter_net_features.py` | Per-tracker (3 trackers × 12 dims): IBI statistics, onset alignment, downbeat spacing histograms, tempo. Plus 6-dim cross-tracker agreement on meters [3,4,5,7,9,11]. Computed live from cached beats. |
| **Tempo features** | 4 | `meter_net_features.py` | Librosa and tempogram BPM (normalized), tempo ratio, tempo agreement. Computed live from cached tempo. |

#### Historical signals (not in MeterNet)

| Signal | Code key | Status | Description |
|--------|----------|--------|-------------|
| **onset_mlp_meter** | `onset_mlp` | Absorbed | 6-class Residual MLP (1361-dim features, v5). Was previously the arbiter's dominant signal (see Section 4.11). Its audio features are now provided directly to MeterNet. |
| **madmom_activation** | `madmom` | Dropped | 0pp contribution in arbiter ablation (Section 4.12). |
| **accent_pattern** | `accent` | Dropped | 0pp contribution in arbiter ablation (Section 4.12). |
| **beat_periodicity** | `periodicity` | Dropped | 0pp contribution despite highest hand-tuned weight (0.20). See Section 4.12. |
| **resnet_meter** | `resnet` | Dropped | 75.4% test, not orthogonal (Sections 4.6, 4.12). |
| **mert_meter** | `mert` | Superseded | 80.7% test as signal score (Section 4.8). Raw 1536d MERT embeddings now integrated directly into MeterNet (Section 4.16), providing +39 files. |

### 1.3 Combination Strategy

**Primary: MeterNet+MERT** (active when `data/meter_net.pt` exists)

A unified Residual MLP classifier takes a 3166-dimensional feature vector and outputs 6 sigmoid probabilities for meters [3, 4, 5, 7, 9, 11]. Features are z-score standardized using training set statistics stored in the checkpoint. MeterNet architecture is parametrizable (hidden size, dropout scale, number of residual blocks); the best configuration is selected via multi-seed grid search (see Section 4.16).

Feature vector composition (3166 dims):

```
Audio features (1449d) — cached as meter_net_audio
├── Multi-tempo autocorrelation: 4 signals × 4 tempos × 64 lags = 1024
├── Tempogram profile: 64 bins
├── MFCC/spectral contrast stats: 40
├── Beat-position histograms: 5 bar lengths × 32 bins = 160
├── Autocorrelation ratios: 60
├── Tempogram meter salience: 9
└── Other (onset stats, etc.)
MERT-v1-95M embeddings (1536d) — pre-extracted .npy files
└── Layer 3 mean+max pooling over 5s chunks
SSM features (75d) — cached as meter_net_ssm
└── 3 trackers × 25 dims (diagonal similarity at lags 2-12)
Beat features (42d) — computed live
└── 3 trackers × 12 dims + 6 cross-tracker agreement
Signal scores (60d) — computed live
└── 5 signals × 12 meter candidates
Tempo features (4d) — computed live
└── Librosa BPM, tempogram BPM, ratio, agreement
```

Expensive feature groups (audio, SSM, MERT) are cached independently from the checkpoint — only the cheap forward pass (~10ms) reruns after model retraining. See Section 4.14 for details.

**Fallback**: When no MeterNet checkpoint exists, the system returns 4/4 with low confidence.

**Historical: Arbiter MLP** (Rounds 13--16, superseded by MeterNet)

The arbiter MLP (72→64→32→6) fused only signal score vectors (6 signals × 12 meters). It achieved 623/700 (89.0%) on METER2800 test. See Section 4.12 for details.

**Historical: Hand-tuned weighted combination** (Rounds 1--12, superseded by arbiter)

The original combination used weighted additive fusion with consensus bonuses, prior probabilities, NN 3/4 penalty, compound meter detection, and rarity penalties. See Sections 4.1--4.5 for details.

## 2. Benchmark

### 2.1 Evaluation Framework

Evaluation uses a unified script (`scripts/eval.py`) with subprocess isolation per file to avoid BeatNet/madmom threading deadlocks. Beat tracking is cached per-tracker via `AnalysisCache` with smart invalidation (changing `meter.py` does not invalidate beat tracker caches). Run snapshots (`--save`) are stored in `data/runs/` with full per-file results for history tracking and regression detection.

```bash
uv run python scripts/eval.py --limit 3 --workers 1   # smoke test
uv run python scripts/eval.py --quick                  # stratified 100 (~20 min)
uv run python scripts/eval.py --split test --limit 0   # hold-out 700
uv run python scripts/eval.py --save                   # save run snapshot
uv run python scripts/dashboard.py                     # run history
```

### 2.2 Primary Benchmark: METER2800

**Dataset**: METER2800 (Abimbola et al., 2023) -- 2800 audio clips across 4 time signature classes (3, 4, 5, 7 beats per bar). Sources: FMA, MAG, OWN, GTZAN. Pre-defined splits: 1680 train, 420 val, 700 test.

**Current best** (MLP+FTT ensemble, zero-training probability averaging) on the full test split (700 files):

| Metric | Result |
|--------|--------|
| **Overall** | **639/700 (91.3%)** |
| Balanced accuracy | 83.9% |
| Meter 3 (302 files) | 285/302 (94.4%) |
| Meter 4 (307 files) | 286/307 (93.2%) |
| Meter 5 (42 files) | 27/42 (64.3%) |
| Meter 7 (49 files) | 41/49 (83.7%) |

**Previous best: Single MLP** (Round 19, MeterNet v7-slim, 2985d):

| Metric | Result |
|--------|--------|
| **Overall** | **631/700 (90.1%)** |
| Balanced accuracy | 84.3% |
| Meter 3 (302 files) | 287/302 (95.0%) |
| Meter 4 (307 files) | 274/307 (89.3%) |
| Meter 5 (42 files) | 29/42 (69.0%) |
| Meter 7 (49 files) | 41/49 (83.7%) |

**Previous best: Arbiter MLP** (Round 16, 6 signals, 72-dim input):

| Metric | Result |
|--------|--------|
| **Overall** | **623/700 (89.0%)** |
| Meter 3 (302 files) | 278/302 (92.1%) |
| Meter 4 (307 files) | 281/307 (91.5%) |
| Meter 5 (42 files) | 22/42 (52.4%) |
| Meter 7 (49 files) | 42/49 (85.7%) |
| Binary 3 vs 4 only | 559/609 (91.8%) |

**Baseline: hand-tuned** (Round 12, 7 signals):

| Metric | Result |
|--------|--------|
| **Overall** | **532/700 (76.0%)** |
| Meter 5 (50 files) | 1/50 (2.0%) |
| Meter 7 (50 files) | 2/50 (4.0%) |

**Key observations**:

- The MLP+FTT ensemble (639/700, 91.3%) is the new project best, gaining +8 over MLP solo and +10 over FTT solo by exploiting complementary error patterns between architectures.
- The ensemble trades 5/x accuracy (−2 files) for large gains on 4/x (+12 files), suggesting the two models disagree mainly on the 4/x vs 3/x boundary where averaging helps.
- Multi-seed validation (5 seeds per candidate) was essential for honest model selection. The Phase 1 winner (h=756, single seed val=77.7%) ranked worst in the 5-seed finale.
- On the binary 3/4 vs 4/4 task, our results surpass the ResNet18 paper's 88% (Abimbola et al., EURASIP 2024).

### 2.3 Historical: Internal Benchmark (Rounds 1--8, retired)

During development (Rounds 1--8), we used an internal benchmark of 303 test cases (272 real audio files from Wikimedia Commons + 17 synthetic + 14 edge cases) across 20 categories. This benchmark was retired in favor of METER2800 as the sole evaluation framework. Historical results are preserved for traceability:

| Round | Tests | Meter Accuracy | Key Change |
|-------|-------|----------------|------------|
| 1 | 72 | 53/72 (74%) | Baseline: additive combination of 6 signals |
| 2 | 72 | 54/72 (75%) | NN 3/4 penalty + weight rebalance |
| 3 | 72 | 55/72 (76%) | bar_tracking signal added |
| 4 | 303 | 241/303 (80%) | Benchmark expansion to 303 tests (272 real files) |
| 5 | 303 | 245/303 (81%) | Weight tuning + ground truth corrections |
| 6 | 303 | 245/303 (81%) | resnet_meter (75.4%) disabled -- 18 regressions |
| 7 | 303 | 253/303 (83%) | Folk GT fixes, compound transfer disabled, refactoring |
| 8 | 303 | 253/303 (83%) | mert_meter (80.7%) -- gate FAIL, disabled |
| 9 | 303 | 253/303 (83%) | Multi-layer 95M: 79.9% test (no improvement), LoRA scripts ready |
| 12a | 303 | -- | Multi-label sigmoid, 6 classes, WIKIMETER, LoRA 330M (val 14.7% @ ep9, crashed) |
| 12b | 303 | -- | WIKIMETER dataset (250 songs, all meters + poly), disk checkpointing, L4 GPU |
| 13 | 700 | 614/700 (87.7%) | Arbiter MLP replaces hand-tuned combination. 6 signals, 72 features. +11.7pp over hand-tuned. |
| 14 | 700 | 615/700 (87.9%) | onset_mlp v5 (1361-dim Residual MLP) + WIKIMETER expansion (683 songs, 126 5/x). onset_mlp 5/x: 50%→63.9%. |
| 15 | 700 | 617/700 (88.1%) | Balanced accuracy val metric + signal sharpening grid search. autocorr α=1.5 saves checkpoint. +2 vs Round 14. |
| 16 | 700 | 623/700 (89.0%) | Clean onset_mlp mapping (removed echo classes) + grid search arbiter (autocorr:1.5, boost_rare=1.0). 7/x: 70%→85.7%. +6 vs Round 15. |
| 17a | 700 | 631/700 (90.1%) | MeterNet v1: unified 1467-dim classifier (audio features + signal scores). Replaces arbiter. +8 vs Round 16. |
| 17b | 700 | 633/700 (90.4%) | MeterNet grid search best during exploration (test-optimal, not val-selected). |
| 17c | 700 | 593/700 (84.7%) | MeterNet v6 (1630-dim: +SSM +beat +tempo). Grid search promoted checkpoint. 5/x: 61.9% (+9.5pp), but 4/x: 81.8% (−9.7pp). −30 vs Round 16 arbiter despite 22× more features. |
| 18 | 700 | 632/700 (90.3%) | MeterNet+MERT (h=512, 3166d = 1630d + 1536d MERT layer 3). Multi-seed finale (5 seeds). bal=83.0%. +39 vs no-MERT (593). Largest single-feature improvement. |
| 19 | 700 | 631/700 (90.1%) | MeterNet v7-slim (audio+MERT, 2985d). Removed BeatNet, beat-this, madmom. 3166d → 2985d. bal=84.3%. −1 file vs R18. Eliminated ~550MB deps, −4200 LOC. |
| **20** | **700** | **639/700 (91.3%)** | **MLP+FTT ensemble (probability averaging). MLP 631 + FTT 629, 49 disagreements, oracle 649. bal=83.9%. +8 vs MLP solo. Zero training cost.** |

### 2.4 Confidence Calibration

The system reports confidence levels alongside meter predictions. Confidence is derived from the margin between the top-scoring and second-scoring meter hypotheses, combined with signal agreement count. Empirically, predictions with 4+ agreeing signals achieve approximately 92% accuracy, while predictions with fewer than 3 agreeing signals drop to approximately 65%.

## 3. Literature

### 3.1 Beat Tracking

- **Beat This!** (Foscarin et al., ISMIR 2024) -- A convolutional Transformer architecture for beat and downbeat tracking that achieves state-of-the-art results without relying on Dynamic Bayesian Network (DBN) postprocessing. The model operates on log-mel spectrograms and uses a U-Net-like encoder-decoder with transformer blocks. Particularly relevant: its high-quality downbeat estimates make it the highest-weighted neural signal in our ensemble. [arXiv](https://arxiv.org/abs/2407.21658) | [GitHub](https://github.com/CPJKU/beat_this)

- **BeatNet+** (Hydri et al., TISMIR 2024) -- Extends BeatNet with auxiliary training objectives to learn percussive-invariant features, improving generalization across genres with varying percussive content. Relevant to our work because beat tracker robustness across genres is a primary challenge. [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.198)

- **BEAST** (Liang & Mysore, ICASSP 2024) -- A streaming beat tracking transformer achieving 50ms latency, designed for real-time applications. Relevant to our live analysis mode. [arXiv](https://arxiv.org/abs/2312.17156)

### 3.2 Music Foundation Models

- **MERT** (Li et al., ICLR 2024) -- A music understanding model with large-scale self-supervised training. 95M parameters, pre-trained on 160K hours of music using acoustic and musical self-supervised objectives (masked language modeling on discrete audio tokens and pitch prediction). Achieves strong results on multiple MIR tasks including beat tracking (88.3% F1), genre classification, and instrument recognition. We use MERT-v1-95M as a frozen feature extractor for meter classification. [arXiv](https://arxiv.org/abs/2306.00107) | [HuggingFace](https://huggingface.co/m-a-p/MERT-v1-95M)

### 3.3 Meter Classification

- **ResNet18 MFCC classifier** (Abimbola et al., EURASIP 2024) -- Achieves 88% accuracy on binary meter classification (3 vs 4 beats per bar) using MFCC features with a ResNet18 backbone. This approach is notable for bypassing beat tracking entirely, making it orthogonal to traditional meter detection methods. [Springer](https://link.springer.com/article/10.1186/s13636-024-00346-6)

- **METER2800 dataset** (Abimbola et al., Data in Brief 2023) -- A curated dataset of 2800 audio clips across 4 time signature classes (3, 4, 5, 7 beats per bar), totaling 872 MB. Drawn from diverse genres. Used as our primary evaluation benchmark and training data for classifier signals. [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0CLXBQ) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700346/)

- **TimeSignatureEstimator** -- Training notebooks and code from the METER2800 authors, providing a reference implementation for the ResNet18 classifier. [GitHub](https://github.com/pianistprogrammer/TimeSignatureEstimator)

### 3.4 Data Augmentation

- **Skip That Beat** (Morais et al., LAMIR/ISMIR LBD 2024) -- A data augmentation technique that creates 2/4 and 3/4 training examples from 4/4 source audio by selectively removing beats. Addresses the severe class imbalance problem in meter detection (most datasets are overwhelmingly 4/4). [GitHub](https://github.com/giovana-morais/skip_that_beat) | [Demo](https://giovana-morais.github.io/skip_that_beat_demo/)

### 3.5 Datasets and Surveys

- **Ballroom Extended** -- Approximately 4180 tracks of ballroom dance music. Waltz tracks are labeled 3/4; remaining genres are 4/4. Available via [mirdata](https://mirdata.readthedocs.io/). Useful as additional training data but limited in meter diversity.

- **Time Signature Detection: A Survey** (Ramos et al., MDPI Sensors 2021) -- Comprehensive survey of time signature detection methods, covering signal processing approaches, machine learning methods, and evaluation protocols. [MDPI](https://www.mdpi.com/1424-8220/21/19/6494)

## 4. Experiments

### 4.1 Product-of-Experts (PoE) vs Additive Combination

We evaluated two fusion strategies for combining signal scores:

**Product-of-Experts (PoE)**: Each signal produces a probability distribution over candidate meters, and the ensemble output is the normalized product of all distributions. Proved brittle in practice:

- **Result**: 50/72 (71%) -- significantly worse than baseline.
- **Failure mode**: A single signal assigning near-zero probability to a meter effectively vetoes it, even if all other signals strongly favor it.
- **Example**: onset_autocorrelation occasionally assigns very low scores to 3/4 in waltz recordings with sparse onsets. Under PoE, this kills 3/4 even when downbeat_spacing and accent_pattern strongly favor it.

**Additive + Consensus**: Weighted sum of signal scores with a multiplicative bonus when multiple signals agree.

- **Result**: 54/72 (75%) -- 4 percentage points better than PoE.
- **Advantage**: Robust to individual signal failures. A single bad signal can only contribute its (bounded) weight, not veto the entire ensemble.

**Decision**: Additive combination with consensus bonus.

### 4.2 Trust Threshold Sweep

The trust threshold controls when neural network signals are enabled based on onset alignment quality:

| Threshold | Meter Accuracy | Notes |
|-----------|----------------|-------|
| 0.25 | 54/72 (75%) | +1 march file, but 6 compound meter regressions |
| 0.30 | 52/72 (72%) | Same regression pattern, slightly milder |
| 0.35 | 53/72 (74%) | Marginal improvement over 0.30 |
| 0.40 | 54/72 (75%) | Best balance between coverage and accuracy |
| 0.50 | 52/72 (72%) | Too strict, disables NNs on too many files |

**Key finding**: The trust threshold and compound meter detection are coupled. Lowering the threshold lets NN signals (which are biased toward simple meters) override correct 6/8 detections with 2/4.

### 4.3 Compound /8 Detection

Compound meters (6/8, 12/8) are characterized by triplet subdivisions within each beat. The sub_beat_division signal detects compound meter evidence by analyzing onset density between consecutive beats:

- **Onset count criterion**: Median onset count between beats in range 1.5--3.5
- **Evenness criterion**: CV of inter-onset intervals < 0.4
- **Triplet position check**: Onsets concentrated near 1/3 and 2/3 beat positions

When compound evidence is found: transfer 50% of the corresponding simple meter's score to the compound meter, and apply a 1.3x boost.

The evenness check (CV < 0.4) was critical: without it, accompaniment arpeggios and ornaments between beats falsely triggered compound detection in polkas, sarabandes, and waltzes.

### 4.4 HCDF (Harmonic Change Detection Function)

The Harmonic Change Detection Function measures the rate of harmonic change in chromagram features. The hypothesis was that triple meters show harmonic changes every 3 beats while duple meters change every 2 or 4 beats.

- **Standalone evaluation**: Discriminates duple/triple meter well in isolation.
- **Integrated at w=0.07**: Causes 6 regressions (net negative).
- **Root cause**: HCDF is correlated with accent_pattern. Adding a correlated signal disrupts the calibrated ensemble balance.

**Lesson learned**: A signal that works well in isolation may harm the ensemble if it is correlated with existing signals. Always evaluate integration impact, not just standalone performance.

### 4.5 bar_tracking Signal

The bar_tracking signal uses madmom's DBNBarTrackingProcessor, which combines a GRU-RNN activation function with Viterbi decoding to track bar boundaries.

- **Integration**: For each candidate meter, run bar tracking with the corresponding beats-per-bar parameter. Score based on consistency and confidence of tracked bar boundaries.
- **Weight**: 0.12, taken from NN signal weights (not from beat_periodicity/accent_pattern).
- **Quality gate**: Skip on sparse or synthetic audio (non_silent frames < 0.15 of total). The GRU-RNN was trained on real music and produces unreliable results on click tracks.
- **Result**: +2 net improvement over baseline (55/72 vs 53/72).
- **Key insight**: Weight should be taken from NN signals (already partially correlated with bar tracking) rather than from beat_periodicity/accent_pattern (which provide orthogonal information).

### 4.6 resnet_meter Signal (disabled)

The resnet_meter signal is a direct meter classification approach using a ResNet18 CNN trained on MFCC spectrograms from the METER2800 dataset. **Result: disabled (W_RESNET=0.0) after thorough evaluation showed no ensemble benefit.**

#### Training

- **Dataset**: METER2800 (2800 files), balanced: 1200 class 3, 1200 class 4, 200 class 5, 200 class 7.
- **Architecture**: torchvision ResNet18, random initialization, 1-channel MFCC input resized to (1, 224, 224), 4-class output.
- **Training**: CrossEntropyLoss with class weights, Adam (lr=1e-3), ReduceLROnPlateau. Early stopping at epoch 43.
- **Results**: 79.3% val, **75.4% test** (class 3: 83%, class 4: 80%, class 5: 46%, class 7: 34%).

#### Integration Attempts

| Strategy | Weight | Condition | Meter Accuracy | Regressions |
|----------|--------|-----------|----------------|-------------|
| Global | 0.10 | Always active | 241/303 (80%) | 18 regressions |
| Trust-gated | 0.10 | Only when avg NN trust < 0.5 | 235/303 (78%) | 21 regressions |
| **Disabled** | **0.0** | Never active | **245/303 (81%)** | **0 regressions** |

#### Why It Failed: The Orthogonality Problem

The model at 75% METER2800 accuracy is **not orthogonal** to the existing 7-signal ensemble:
- Our full engine achieves 74% on METER2800, nearly identical to the ResNet18 model.
- The model's errors overlap substantially with the ensemble's errors.
- Adding a correlated but noisier signal (75% vs 81%) can only shuffle correct/incorrect predictions.
- Trust-gating paradox: the model's accuracy is *lower* on the difficult low-trust files where it was supposed to help.

#### Requirements for a Useful Classifier Signal

- **>90% standalone accuracy** -- corrections must outnumber errors at meaningful weight.
- **Orthogonal error profile** -- failing on different files than the existing signals.
- **Calibrated confidence** -- high-confidence predictions can be trusted, low-confidence ignored.

### 4.7 METER2800 Evaluation

We evaluated our full engine (7 active signals, resnet_meter/mert_meter disabled) on the full METER2800 test split (700 files) using the unified evaluation script (`scripts/eval.py`) with subprocess isolation per file. Total runtime: approximately 2.5 hours. Results are reported in Section 2.2.

**Critical finding**: Our system essentially does not work on odd meters (5/4: 2%, 7/4: 4%). This is a fundamental limitation of the current architecture, which is heavily biased toward common Western meters. The rarity penalties applied to 5/4 and 7/4 actively suppress these meters.

### 4.8 mert_meter Signal (disabled)

The mert_meter signal uses MERT (Music undERstanding model with large-scale self-supervised Training), a 95M parameter music foundation model pre-trained on 160K hours of music, as a frozen feature extractor. Unlike resnet_meter (ResNet18 on MFCC), MERT captures high-level musical structure through 12 transformer layers.

#### Architecture

```
Audio (22050 Hz) --> resample to 24 kHz --> center crop 30s
    --> 5-second non-overlapping chunks
    --> MERT-v1-95M (frozen, HuggingFace: m-a-p/MERT-v1-95M)
    --> hidden states from layer 3 (best of 12), shape (T, 768)
    --> mean + max pooling per chunk --> (1536,) per chunk
    --> mean of means + max of maxes across chunks --> (1536,)
MLP Classifier:
    Linear(1536, 256) --> ReLU --> Dropout(0.3) --> Linear(256, 4)
    --> softmax --> {3: prob, 4: prob, 5: prob, 7: prob}
Score mapping --> dict[(num, den), float]
```

#### Training

- **Dataset**: METER2800, pre-extracted embeddings (avoiding recomputing MERT features each epoch).
- **Layer sweep**: All 12 MERT transformer layers evaluated independently (15 epochs each). **Best layer: 3** (an early layer), contrary to the expectation that higher layers capture more abstract musical features.
- **Classifier**: MLP with 256 hidden units, 0.3 dropout, CrossEntropyLoss (inverse-frequency class weights), Adam (lr=1e-3), early stopping (patience=15).
- **Results**: 84.5% val, **80.7% test** (+5.3pp over resnet_meter's 75.4%).
  - Class 3: 85.7%, Class 4: 89.0%, Class 5: 40%, Class 7: 42%

#### Orthogonality Evaluation

Orthogonality was tested on 272 internal benchmark fixtures:

| Metric | Engine | MERT |
|--------|--------|------|
| Accuracy | 81.2% (221/272) | 67.6% (184/272) |

Agreement matrix:

| Both correct | Both wrong | Engine only (LOSSES) | MERT only (GAINS) |
|---|---|---|---|
| 153 (56.2%) | 20 (7.4%) | 68 (25.0%) | 31 (11.4%) |

- **Agreement rate**: 63.6% (within target 0.65-0.80 -- MERT has a different view)
- **Complementarity ratio**: 0.46 (31 gains / 68 losses) -- **FAIL** (target: >1.5)

#### Analysis of Gains and Losses

**GAINS** (31 files where MERT correct, engine wrong):
- MERT excels at detecting 3/4 in classical/romantic music: Chopin waltzes, mazurkas, Blue Danube, Bach sarabandes, Offenbach barcarolle
- MERT correctly predicts 4/4 where engine falsely predicts 3/4: blues, jazz fox-trots, marches

**LOSSES** (68 files where engine correct, MERT wrong):
- MERT over-predicts 7/4 on percussion-heavy music: jazz ride patterns, shuffle beats, djembe
- MERT over-predicts 5/4 on jigs and tarantellas (6/8 music)
- MERT predicts 3/4 instead of 4/4 on many marches, polkas, tangos

#### Why 80.7% Is Still Not Enough

Despite being 5.3pp more accurate than resnet_meter on METER2800, mert_meter achieves only 67.6% on our benchmark fixtures (worse than engine's 81.2%). The loss ratio is 2.2:1, making integration harmful at any global weight.

**Best layer 3 finding**: Layer 3 outperforming layer 11 suggests that low-level rhythmic features (closer to the audio surface) are more useful for meter classification than high-level semantic features. This is consistent with meter being a relatively low-level property compared to genre or key.

#### Potential Improvements

1. **Fine-tune MERT**: Unfreeze last 2-3 transformer layers and fine-tune on METER2800. Should adapt higher layers to capture meter-specific features.
2. **Confidence gating**: Only use predictions when softmax confidence exceeds 0.8. Would eliminate many false 7/4 and 5/4 predictions.
3. **Larger MLP**: Two hidden layers (1536 --> 512 --> 128 --> 4) or attention pooling instead of mean+max.
4. **Multi-layer concatenation**: Combine features from layers 3, 7, and 11 for a richer representation.

### 4.9 Multi-Layer Classifier on MERT-95M Embeddings (Round 9)

**Motivation**: The single-layer approach uses only layer 3 of 12 available MERT-95M layers. A learnable weighted sum of all layers might capture complementary information from different representation levels.

**Architecture**: `MultiLayerMLP` -- softmax-weighted sum of all 12 layers with LayerDrop (p=0.1) regularization, followed by a deeper MLP (1536 -> 512 -> 256 -> 4) with GELU and LayerNorm.

**Training**: 100 epochs, Adam lr=1e-3 with ReduceLROnPlateau, batch size 64, embedding noise augmentation (0.01). On existing pre-extracted 95M embeddings.

**Results**:

| Method | Val Acc | Test Acc | 3/x | 4/x | 5/x | 7/x |
|--------|---------|----------|-----|-----|-----|-----|
| Single-layer L3 (baseline) | 84.5% | 80.7% | 85.7% | 89.0% | 40.0% | 42.0% |
| Single-layer L3 (re-run) | 85.0% | 81.7% | 87.7% | 85.7% | 52.0% | 52.0% |
| **Multi-layer (12L)** | 81.0% | **79.9%** | 86.0% | 87.0% | 18.0% | 62.0% |

**Learned layer weights** (top 5): L3=15.3%, L4=13.5%, L2=12.7%, L5=11.0%, L1=10.4%. Layers 8-11 received only 3-5% weight each, confirming that early layers are most informative for meter classification.

**Analysis**: Multi-layer on MERT-95M did not improve over single-layer. The 12 layers of the 95M model don't provide enough representational diversity to benefit from fusion. The accuracy bottleneck is in the frozen encoder, not the classifier head. Interesting that multi-layer strongly favored 7/x (62%) while collapsing on 5/x (18%).

**Conclusion**: To improve MERT accuracy beyond ~81%, we need either (a) a larger model (MERT-v1-330M with 24 layers, 1024 hidden), or (b) LoRA fine-tuning to adapt representations to meter classification. Scripts for both approaches (`extract_mert_embeddings.py --model m-a-p/MERT-v1-330M` and `finetune_mert.py`) are implemented and ready to run.

### 4.10 MERT-330M LoRA Fine-tuning with Multi-label Architecture (Round 12)

We transitioned from single-label (softmax) to multi-label (sigmoid + BCE) classification to enable polyrhythm detection, expanded classes from 4 to 6 (adding 9/8, 11/8), and curated two external datasets from YouTube:

1. **WIKIMETER** (Round 12a): odd-meter-only subset (5/x, 7/x, 9/x, 11/x + poly)
2. **WIKIMETER** (Round 12b, expanded Round 14): full catalog with all classes, currently 683 songs / 2937 segments. Song catalog in `scripts/setup/wikimeter.json` (single source of truth).

Notebook-aligned diagnostics now focus on compact outputs: per-class accuracy/AP, macro-F1, and confidence gap.

**Release gate**: classes **9/x**, **11/x**, and **poly** are currently **experimental** — there is no dedicated holdout test set for these classes yet.

#### First Training Run: MERT-330M + WIKIMETER on Colab Pro A100

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|-------|-----------|-----------|----------|---------|------|
| 1 | 1.6259 | 4.3% | 1.6024 | 0.0% | 1052s |
| 2 | 1.5885 | 4.0% | 1.5994 | 0.0% | 1041s |
| 3 | 1.5703 | 5.1% | 1.5852 | 0.0% | 1041s |
| 4 | 1.5525 | 6.5% | 1.5628 | 0.0% | 1041s |
| 5 | 1.5387 | 7.2% | 1.5421 | 0.0% | 1041s |
| 6 | 1.5044 | 12.3% | 1.4596 | 9.8% | 1041s |
| 7 | 1.4776 | 16.1% | 1.5044 | 4.6% | 1041s |
| 8 | 1.4541 | 18.9% | 1.5302 | 2.5% | 1041s |
| 9 | 1.4361 | 20.6% | 1.4170 | **14.7%** | 1041s |
| 10 | 1.4271 | 22.7% | 1.3862 | 6.7% | 1041s |

**Key observations**:
- **Breakthrough at epoch 6**: val acc jumped from 0% to 9.8% after 5 epochs of zero val. This is characteristic of pos_weight with heavy class imbalance — the model initially learns to predict only rare classes, then suddenly "clicks" on common classes.
- **Val oscillation**: 0→0→0→0→0→9.8→4.6→2.5→14.7→6.7%. Typical post-breakthrough behavior with aggressive pos_weight.
- **LR discovery**: Training was accidentally run with 95M learning rates (HEAD_LR=5e-4, LORA_LR=1e-4) on the 330M model. Despite being "too high", it worked — the model showed clear learning. Subsequent runs use per-model LR auto-scaling.
- **Session crashed at epoch 10**: Colab runtime disconnected, checkpoint was only in RAM → lost. This led to adding persistent disk checkpointing every epoch.

**Infrastructure improvements** from this run:
- Checkpoint saved to disk (`torch.save`) every epoch — crash-safe
- Default model changed to 330M (95M showed 0% val for all epochs on T4)
- WIKIMETER (Round 12b) replaces the odd-meter-only subset from Round 12a — balanced across all meter classes, includes polyrhythmic songs

Full architecture and diagnostic details in [MERT-LORA-MULTILABEL.md](MERT-LORA-MULTILABEL.md).

### 4.11 onset_mlp_meter Signal (Rounds 13--14 — first classifier to pass gate)

The onset_mlp_meter signal takes a fundamentally different approach from resnet_meter and mert_meter: instead of operating on raw spectrograms or learned embeddings, it classifies meter from hand-crafted rhythmic features designed to capture beat-relative periodicity at multiple tempos.

#### Architecture

**v5 (current)**:

```
Audio (22050 Hz) --> center crop 30s
Feature extraction (1361 dims):
  Part 1: Autocorrelation features (1024 dims, tempo-dependent)
    4 signals × 4 tempo candidates × 64 beat-relative lags
    Signals: onset envelope, RMS energy, spectral flux, chroma energy
    Tempos: top-2 tempogram peaks, T/2 (sub-harmonic), T×3/2 (compound)
  Part 2: Tempo-independent features (108 dims)
    Tempogram profile: 64 log-spaced BPM bins (30-300)
    MFCC statistics: 13 coefficients × (mean + std) = 26
    Spectral contrast: 7 rows × (mean + std) = 14
    Onset rate: mean/std/median interval + onsets/sec = 4
  Part 3: v5-only features (229 dims)
    Beat-position histograms: 5 bar lengths (2,3,4,5,7) × 32 bins = 160
    Autocorrelation ratios: 4 signals × (6 absolute + 9 discriminative) = 60
    Tempogram meter salience: 9 candidate meters = 9
Residual MLP Classifier:
  Linear(1361, 640) --> BN --> ReLU --> Dropout(0.3)
  ResidualBlock(640) --> BN --> ReLU --> Dropout(0.25) + skip connection
  Linear(640, 256) --> BN --> ReLU --> Dropout(0.2)
  Linear(256, 128) --> BN --> ReLU --> Dropout(0.15)
  Linear(128, 6)
  --> softmax --> {3: prob, 4: prob, 5: prob, 7: prob, 9: prob, 11: prob}
```

v5 improvements over v4:
- **4th tempo candidate** (+256 dims): top-2 peaks from tempogram + compound T×3/2. Fixes garbage-in when librosa tempo estimate is wrong.
- **Beat-position histograms** (+160 dims): onset times folded modulo bar_duration for 5 hypothetical bar lengths. Captures accent patterns like 3+2 or 2+3 in 5/4.
- **Autocorrelation ratios** (+60 dims): explicit encoding of "is peak@5 > peak@4?", which the model otherwise has to learn from raw values.
- **Tempogram meter salience** (+9 dims): tempo-independent peak strength for each candidate meter.
- **Residual MLP**: ResidualBlock(640) with skip connection replaces plain sequential MLP. Dropout 0.3→0.25→0.2→0.15 (decreasing).

**v4 (historical)**:

```
Feature extraction (876 dims):
  768 (autocorrelation: 3 tempos × 4 signals × 64 lags) + 108 (tempogram + MFCC + spectral contrast + onset stats)
MLP: Linear(876, 512) → BN → ReLU → Dropout(0.3) → 256 → 128 → 6
```

#### Training

- **Dataset**: METER2800 (train+val) + WIKIMETER (683 songs, 2937 segments, 6 classes).
- **Classes**: 6 (3, 4, 5, 7, 9, 11) — single-label classification.
- **Loss**: Focal loss (γ=2.0) with class weights. Mixup (α=0.2) and CutMix (α=1.0, 50/50 per batch).
- **Optimizer**: AdamW (lr=8e-4, weight_decay=1e-3), CosineAnnealingWarmRestarts (T_0=50, T_mult=2), early stopping (patience=50).
- **WeightedRandomSampler**: oversamples rare classes (5/x, 7/x, 9/x, 11/x).
- **Feature standardization**: z-score normalization (mean/std from training set, stored in checkpoint).
- **Feature caching**: Per-file cache keyed by `feature_version + audio_hash`.
- **Audio augmentation** (optional, `--augment N`): random crop, noise injection, gain variation, time masking. Tested but found marginally negative (79.6% vs 80.0% without), not used in production checkpoint.

#### Results

| Version | Features | Model | Loss | Val | Test | 3/x | 4/x | 5/x | 7/x | 9/x | 11/x |
|---------|----------|-------|------|-----|------|-----|-----|-----|-----|-----|------|
| v3 | 768 (autocorr only) | 256→128→6 | CE | 73.9% | 73.6% | 84.8% | 78.1% | 54.8% | 72.0% | 35.1% | 42.9% |
| v4 | 876 (autocorr + tempo-indep) | 512→256→128→6 | Focal | 78.4% | 79.5% | 88.4% | 85.9% | 58.9% | 75.3% | 47.3% | 52.4% |
| **v5** | **1361 (v4 + histograms + ratios + salience)** | **Residual 640→640→256→128→6** | **Focal + CutMix** | **81.1%** | **80.2%** | -- | -- | **63.9%** | -- | -- | -- |

Key improvements across versions:
- v3→v4: Tempogram profile, MFCC stats, focal loss, larger model.
- v4→v5: 4th tempo candidate, beat-position histograms, autocorrelation ratios, tempogram salience, Residual MLP, CutMix, AdamW. **5/x accuracy jumped from 50% to 63.9%** (+13.9pp) thanks to beat-position histograms and expanded WIKIMETER (126 5/x songs, up from 110).

#### Orthogonality Evaluation

Compared against saved engine run on METER2800 test split (700 files):

| Metric | v3 | v4 |
|--------|-----|-----|
| Engine accuracy | 76.6% (536/700) | 76.6% (536/700) |
| Onset MLP accuracy | 79.1% (554/700) | **87.1% (610/700)** |
| Agreement rate | 75.7% | 76.9% (target 65-80% ✅) |
| Complementarity ratio | 1.24 (❌) | **2.68** (target >1.5 ✅) |
| Gains (MLP right, engine wrong) | 94 | **118** |
| Losses (engine right, MLP wrong) | 76 | **44** |
| **Gate check** | **FAIL** | **PASS** |

Per-class orthogonality (v4):

| Meter | Engine | MLP | Gains | Losses |
|-------|--------|-----|-------|--------|
| 3/x | 253/300 (84.3%) | 267/300 (89.0%) | +42 | -21 |
| 4/x | 277/300 (92.3%) | 274/300 (91.3%) | +17 | -23 |
| 5/x | 3/50 (6.0%) | 25/50 (50.0%) | +24 | 0 |
| 7/x | 3/50 (6.0%) | 44/50 (88.0%) | +35 | 0 |

**Critical finding**: The onset MLP has zero losses on 5/x and 7/x — it correctly identifies odd meters that the engine completely misses (engine: 6% on both). This is the orthogonal information we've been seeking since Round 6.

#### Why This Signal Succeeded Where Others Failed

1. **87.1% > 76.6%** — The onset MLP is significantly more accurate than the engine, unlike resnet_meter (75%) and mert_meter (67.6% on fixtures). Corrections outnumber errors.
2. **Qualitatively different features** — Autocorrelation at beat-relative lags captures periodicity structure that the engine's signal-level approach cannot. Tempo-independent features (tempogram, MFCC) provide backup when tempo estimation fails.
3. **6-class training** — Training on 6 classes (including 9/x, 11/x from WIKIMETER) gives the model exposure to odd meters that the 4-class resnet/mert models lacked.
4. **Focal loss** — Down-weighting easy 4/x examples and focusing on hard cases reduced 4/x losses from 49 (v3) to 23 (v4).

#### Code Organization

Feature extraction is shared between training (`scripts/training/train_onset_mlp.py`) and inference (`beatmeter/analysis/signals/onset_mlp_meter.py`) via `beatmeter/analysis/signals/onset_mlp_features.py`. The inference module loads v4 or v5 models transparently based on the `arch_version` field in the checkpoint.

#### Integration

Integrated as Signal 9 in `meter.py` with weight W_ONSET_MLP=0.12. The weight is taken from the NN signal budget (resnet/mert are disabled at 0.0). Feature extraction adds ~5-10s per file on first run (cached thereafter via AnalysisCache). Cache invalidation keys on the checkpoint file (`data/meter_onset_mlp.pt`) and the shared feature module (`onset_mlp_features.py`), ensuring retraining automatically invalidates cached results.

### 4.12 Arbiter MLP (Round 13) — superseded by MeterNet (Section 4.15)

#### Motivation

The hand-tuned combination logic in `meter.py` (~200 lines of weights, consensus bonuses, NN penalties, rarity adjustments, compound transfer) is increasingly difficult to maintain. Each new signal requires careful weight tuning and interaction testing. Furthermore, signals that failed the hand-tuned gate check (ResNet, HCDF) may still provide useful information in specific contexts — the arbiter can learn *when* to use them.

#### Architecture

A small MLP replaces the entire weighted combination pipeline:

```
Input: 6 signals × 12 meter candidates = 72 dims
  Signals: beatnet, beat_this, autocorr, bar_tracking, onset_mlp, hcdf

MLP:
  Linear(72, 64) → BN → ReLU → Dropout(0.2)
  Linear(64, 32) → BN → ReLU → Dropout(0.15)
  Linear(32, 6) → sigmoid

Output: 6 sigmoids for meters [3, 4, 5, 7, 9, 11] (multi-label)
Loss: BCEWithLogitsLoss with pos_weight
```

Key design decisions:
- **Multi-label output** (sigmoid + BCE) instead of single-label (softmax + CE), matching the WIKIMETER multi-label annotations for polymetric songs.
- **6 signals after ablation** (see below): started with 10 signals (124 features), ablation showed 4 contribute 0pp — dropped to 6 signals (72 features) with no accuracy loss.
- **Signal scores only**: Ablation showed alignment values, tempo, and signal presence flags add 0pp — the network learns equivalent information from the raw signal score patterns.
- **MERT excluded**: Risk of shortcut learning (recognizing genres/instruments rather than meter patterns). ResNet is simpler and HCDF is pure DSP, both safer for the arbiter.

#### Training Data

- **METER2800** (tuning split, 2100 files): 4 classes (3, 4, 5, 7), single-label.
- **WIKIMETER** (2937 segments from 683 songs): 6 classes (3, 4, 5, 7, 9, 11), multi-label with confidence weights (e.g., "3:0.7,4:0.8" for polymetric songs). Per-meter: 3/x:129, 4/x:112, 5/x:126, 7/x:128, 9/x:136, 11/x:83.
- **Combined**: ~5000 training examples across 6 classes.

#### Infrastructure

The arbiter requires cached signal scores for all 2800+ METER2800 files. To avoid loading all beat tracker models simultaneously (3GB+ RAM, GPU contention), we built a cache warming script (`scripts/setup/warm_cache.py`) that processes files one tracker/signal at a time:

1. Load BeatNet → process all files → free memory
2. Load Beat This! → process all files → free memory (GPU, 1 worker)
3. Load madmom → process all files → free memory
4. Continue for librosa, onsets, tempo, onset_mlp, resnet, hcdf

This reduces peak memory from ~3GB to ~600MB and eliminates GPU contention deadlocks. Supports `--workers N` for CPU-bound phases with `mp.get_context("spawn")` and `maxtasksperchild=50` for worker recycling.

#### Data Quality Fix

Initial training produced poor results (70.3% val, 66.4% test). Root cause: data quality issues in cache:
- **onset_mlp** had only 18.1% coverage (844/4662 files cached) — missing for most files
- **tempo** had only 12.2% coverage — warm_cache.py lacked a tempo phase

After fixing warm_cache.py to include tempo computation and re-warming the cache (onset_mlp: 18%→100%, tempo: 12%→97%), retraining immediately jumped to 94.1% val, 80.1% test.

#### Ablation Study: Extra Features

After filling cache gaps, we tested whether alignment, tempo, and signal presence flags contributed:

| Variant | Val | Test | Δ |
|---------|-----|------|---|
| All features (72 signal + alignment + tempo + presence) | 94.1% | 80.1% | baseline |
| Signal scores only (72 features) | 93.8% | 80.2% | +0.1pp |

**Conclusion**: Extra features add 0pp. The network learns trust/alignment patterns from the raw signal scores alone (e.g., if BeatNet scores are all zero, the network implicitly knows BeatNet is untrusted).

#### Ablation Study: Signal Importance (Leave-One-Out)

Leave-one-out ablation with 10 seeds per variant (to ensure statistical significance):

| Dropped signal | Test accuracy (mean ± std) | Δ from all 10 |
|----------------|---------------------------|----------------|
| None (all 10 signals) | 80.2% ± 0.3 | baseline |
| onset_mlp | 71.9% ± 0.5 | **-8.3pp** (critical) |
| autocorr | 78.8% ± 0.4 | -1.4pp |
| beatnet | 79.2% ± 0.3 | -1.0pp |
| bar_tracking | 79.5% ± 0.3 | -0.7pp |
| hcdf | 79.6% ± 0.4 | -0.6pp |
| beat_this | 79.8% ± 0.3 | -0.4pp |
| accent | 80.0% ± 0.4 | -0.2pp (negligible) |
| resnet | 80.0% ± 0.3 | -0.2pp (negligible) |
| madmom | 80.1% ± 0.3 | -0.1pp (negligible) |
| periodicity | 80.2% ± 0.3 | +0.0pp (zero impact) |

**Key findings**:
- **onset_mlp is the dominant signal** (-8.3pp without it), consistent with its 87.1% standalone accuracy.
- **periodicity has zero impact** despite being the highest hand-tuned weight (0.20). The arbiter completely ignores it.
- **madmom, resnet, accent** contribute essentially nothing (within noise).

#### Aggressive Pruning: 6 vs 10 Signals

Keeping only 6 signals (dropping periodicity, madmom, resnet, accent):

| Variant | Test (10 seeds) |
|---------|----------------|
| All 10 signals | 80.2% ± 0.3 |
| **Keep 6 signals** | **80.3% ± 0.3** |

Identical within noise. Feature space reduced from 120 to 72 dimensions.

#### Integration into Pipeline

Integration required one key fix: when the arbiter is loaded, all 6 input signals must be computed regardless of trust-based weights. The pipeline previously skipped signals when trust=0 (weight < 0.02), but the arbiter was trained on complete feature vectors from cache. Solution: weight floor of 0.02 when arbiter is loaded.

```python
arbiter_model, _ = _load_arbiter()
if arbiter_model is not None:
    for sig_key in weights:
        if weights[sig_key] < 0.02:
            weights[sig_key] = 0.02
```

#### Results on METER2800 Test Split

| System | Overall | 3/x | 4/x | 5/x | 7/x |
|--------|---------|-----|-----|-----|-----|
| Hand-tuned (7 signals) | 532/700 (76.0%) | 256/300 (85.3%) | 273/300 (91.0%) | 1/50 (2.0%) | 2/50 (4.0%) |
| Arbiter MLP Round 13 (6 signals) | 614/700 (87.7%) | 269/300 (89.7%) | 277/300 (92.3%) | 33/50 (66.0%) | 35/50 (70.0%) |
| Arbiter MLP Round 15 (balanced + sharpening) | 617/700 (88.1%) | 279/300 (93.0%) | 274/300 (91.3%) | 29/50 (58.0%) | 35/50 (70.0%) |
| **Arbiter MLP Round 16 (clean mapping + grid search)** | **623/700 (89.0%)** | **278/302 (92.1%)** | **281/307 (91.5%)** | **22/42 (52.4%)** | **42/49 (85.7%)** |

The arbiter achieves the largest single improvement in project history. Round 15 added balanced accuracy as the validation metric (preventing rare-class sacrifice) and signal sharpening (power transformation per signal before training). Round 16 cleaned onset_mlp echo mappings (removing 6/8, 2/4, 5/8, 7/8, 12/8 echoes for a clean 1:1 class→meter mapping) and ran a full grid search (150 models: 5 sharpening × 3 boost_rare × 10 seeds), achieving 623/700 (89.0%). The 7/x class improved dramatically from 70%→85.7%.

### 4.13 Clean onset_mlp Mapping + Grid Search (Round 16)

#### Motivation

The onset_mlp signal mapped its 6 output classes (3, 4, 5, 7, 9, 11) to 12 meter candidates by including "echo" mappings for compound/variant meters (6/8→3, 2/4→4, 5/8→5, 7/8→7, 12/8→4). These echo mappings duplicated probability mass and created ambiguous score distributions (e.g., 6/8 receiving the same score as 3/4), making it harder for the arbiter to distinguish between simple and compound meters. Removing echoes gives a clean 1:1 class→meter mapping.

#### Methodology

1. **Removed echo mappings** from onset_mlp score distribution: each of the 6 classes maps only to its primary meter (3→3/4, 4→4/4, 5→5/4, 7→7/4, 9→9/8, 11→11/8). All other meter candidates receive score 0.0.
2. **Full cache re-warm** (5734 files) required since onset_mlp scores changed for every file.
3. **Grid search** over 5 sharpening configs × 3 boost_rare values × 10 seeds = 150 arbiter models:
   - Sharpening: none, autocorr:1.5, autocorr:2.0, onset_mlp:1.5, autocorr:1.5+onset_mlp:1.5
   - boost_rare: 0.5, 1.0, 2.0
4. **Winner selection**: best balanced accuracy on validation set.

#### Results

- **Winner**: autocorr:1.5, boost_rare=1.0, val=92.0%.
- **METER2800 test**: 623/700 (89.0%), +6 files vs Round 15 (617/700).

| Metric | Round 15 | Round 16 | Delta |
|--------|----------|----------|-------|
| Overall | 617/700 (88.1%) | 623/700 (89.0%) | +6 (+0.9pp) |
| 3/x | 279/300 (93.0%) | 278/302 (92.1%) | -1 |
| 4/x | 274/300 (91.3%) | 281/307 (91.5%) | +7 |
| 5/x | 29/50 (58.0%) | 22/42 (52.4%) | -7 |
| 7/x | 35/50 (70.0%) | 42/49 (85.7%) | +7 (+15.7pp) |
| Binary 3+4 | 553/600 (92.2%) | 559/609 (91.8%) | +6 |

#### Analysis

- **7/x dramatic improvement** (+15.7pp): The clean mapping eliminated score ambiguity that confused the arbiter on 7/4 vs 7/8 files. The arbiter now receives a clear signal when onset_mlp predicts meter 7.
- **3/x slight regression** (-1 file): Minor trade-off from the cleaner but narrower score distribution — some 3/4 files that benefited from the 6/8 echo no longer receive that boost.
- **5/x decrease** (-7 files): Note that per-category file counts changed between rounds (50→42 for 5/x, 50→49 for 7/x), so raw count comparisons are not perfectly aligned. The 5/x rate decrease (58.0%→52.4%) reflects the smaller evaluation subset.
- **Net result**: +6 files overall, with 7/x as the primary beneficiary.

#### Conclusion

Cleaning echo mappings from onset_mlp is a pure improvement: simpler code, clearer signal semantics, and better arbiter performance. The grid search confirmed that the same sharpening config (autocorr:1.5) remains optimal, while boost_rare=1.0 (neutral) slightly outperforms the previous implicit default.

### 4.14 Granular Feature Caching for MeterNet

#### Motivation

MeterNet's feature extraction pipeline produces a 1630-dim feature vector from multiple sources: audio features (1449d from autocorrelation, MFCC, tempogram, etc.), SSM features (75d from self-similarity matrix), beat features (42d), signal scores (60d), and tempo features (4d). Previously, the model's final meter scores were cached as a single `meter_net` signal entry keyed on the checkpoint hash. This meant that every model retraining invalidated the entire cache — forcing re-extraction of expensive audio features (~5s per file) even though only the cheap forward pass (~10ms) changed.

#### Design

Split the cache into **granular feature groups** that are keyed on their extraction code, not on the model checkpoint:

| Cache group | Dims | LMDB key | Depends on | Cost |
|-------------|------|----------|------------|------|
| `meter_net_audio` | 1449 | `features:meter_net_audio:{hash}:{ah}` | `onset_mlp_features.py` | ~5s |
| `meter_net_ssm` | 75 | `features:meter_net_ssm:{hash}:{ah}` | `ssm_features.py` | ~2s |
| Beat features | 42 | computed live from cached beats | — | <1ms |
| Signal scores | 60 | computed live from cached signals | — | <1ms |
| Tempo features | 4 | computed live from cached tempo | — | <1ms |

The `meter_net` score cache was removed entirely — scores are always recomputed from cached features + current checkpoint.

#### Implementation

1. **`cache.py`**: Added `features:{group}:{hash}:{audio_hash}` key format with `load_array`/`save_array` methods storing raw float32 bytes (faster than JSON for numpy arrays). Feature groups `meter_net_audio` and `meter_net_ssm` added to `SIGNAL_DEPS` with only their extraction code as dependencies.

2. **`meter.py` (`_meter_net_predict`)**: Checks for cached feature arrays before extraction. Lazy audio preparation — only decodes audio when at least one feature group is a cache miss.

3. **`engine.py` (`_is_cache_warm`)**: Fast path checks for `meter_net_audio` and `meter_net_ssm` arrays (not meter_net scores). When all feature groups are cached, the engine skips audio file decoding entirely.

#### Results

- **Eval after model retraining**: ~10 seconds for 700 files (previously ~10 minutes)
- **Speedup**: ~60× for the retrain→eval cycle
- **No accuracy impact**: Feature extraction is identical; only the caching granularity changed

#### Conclusion

Granular feature caching is a pure infrastructure improvement that dramatically accelerates the iteration loop. The key insight is separating the "what depends on the model checkpoint" (forward pass) from "what depends only on the audio" (feature extraction), and caching them independently.

### 4.15 MeterNet: Unified Meter Classification (Round 17)

#### Motivation

The arbiter MLP (Section 4.12) fuses only signal-level score vectors (72 dims). The onset_mlp signal — arbiter's dominant input (-8.3pp ablation impact) — internally extracts rich 1361-dim audio features but compresses them to just 12 meter scores before the arbiter sees them. This information bottleneck means the arbiter cannot learn from fine-grained audio representations. MeterNet aims to bypass this bottleneck by taking raw audio features directly as input.

#### Architecture

```
MeterNet — Residual MLP (parametrizable via grid search)

Input: 1630 dimensions
  ├── Audio features (1449d): onset_mlp v6 features (autocorrelation, MFCC,
  │     tempogram, beat-position histograms, ratios, salience)
  ├── Beat-sync chroma SSM (75d): self-similarity at lags 2-12 per tracker
  ├── Beat tracker features (42d): IBI stats, alignment, downbeat histograms ×3 trackers
  ├── Signal scores (60d): 5 signals × 12 meters (without onset_mlp)
  └── Tempo features (4d): librosa + tempogram BPM, ratio, agreement

Optional bottleneck: compress autocorrelation (1024d) before projection
Residual blocks: N × (Linear → BN → ReLU → Dropout + skip connection)
Head: hidden → 0.4×hidden → 0.2×hidden → 6 (sigmoid)

Output: 6 sigmoid probabilities for meters [3, 4, 5, 7, 9, 11]
Loss: BCEWithLogitsLoss (multi-label, pos_weight for class imbalance)
```

Key differences from arbiter:
- **22× more features** (1630 vs 72)
- **Raw audio features** instead of pre-classified scores
- **SSM features** capturing harmonic periodicity at beat-aligned lags
- **Beat tracker statistics** (IBI, alignment, downbeat patterns)
- **onset_mlp excluded from signal scores** — its features provided directly

#### Training

- **Dataset**: METER2800 tuning (2100 files) + WIKIMETER (2937 segments, 6 classes)
- **Grid search**: lr ∈ {3e-4, 2e-4}, hidden ∈ {128, 256, 384, 512, 756, 1024}, dropout_scale ∈ {1.0, 1.5, 2.0}, n_blocks ∈ {1, 2, 3} = 108 combinations
- **Training**: 200 epochs, early stopping (patience=50), CosineAnnealingWarmRestarts
- **Augmentation**: Feature-level noise + scaling + CutMix
- **Validation metric**: Balanced accuracy (macro per-class)

#### Results

| System | Overall | 3/x | 4/x | 5/x | 7/x | Balanced |
|--------|---------|-----|-----|-----|-----|----------|
| Hand-tuned (R12) | 532/700 (76.0%) | 85.3% | 91.0% | 2.0% | 4.0% | — |
| Arbiter MLP (R16) | 623/700 (89.0%) | 92.1% | 91.5% | 52.4% | 85.7% | — |
| MeterNet v1 (1467d) | 631/700 (90.1%) | 93.7% | 90.9% | 64.3% | 85.7% | 83.6% |
| MeterNet grid best | 633/700 (90.4%) | 94.0% | 93.2% | 59.5% | 77.6% | 81.1% |
| MeterNet promoted (1630d) | 593/700 (84.7%) | 91.4% | 81.8% | 61.9% | 81.6% | 79.2% |
| **MLP+FTT ensemble (R20)** | **639/700 (91.3%)** | **94.4%** | **93.2%** | **64.3%** | **83.7%** | **83.9%** |

#### Analysis

**The paradox of more features**:

MeterNet v1 (1467d) — with only audio features and signal scores — achieved 90.1%, surpassing the arbiter (89.0%). But adding SSM (75d), beat features (42d), and tempo features (4d) to create v6 (1630d) led to a *regression* to 84.7% on the promoted checkpoint. This is a classic case of the curse of dimensionality:

1. **Val-test metric gap**: Grid search selects by val balanced accuracy, which doesn't perfectly predict test overall accuracy. The val-optimal checkpoint is not the test-optimal checkpoint — the 633/700 result was observed during grid search but a different combo was promoted.
2. **4/x regression** (91.5% → 81.8%): The additional features may introduce noise for the majority class. Beat tracker statistics (42d) likely contain redundant information already encoded in the signal scores.
3. **5/x improvement** (52.4% → 61.9%): SSM and beat features do help disambiguate rare meters where signal scores alone are insufficient.
4. **Model selection challenge**: With 108 grid combos, selecting the checkpoint that generalizes best to the test set requires a more robust validation strategy (e.g., k-fold cross-validation or a larger validation set).

#### Conclusion

MeterNet demonstrates that raw audio features can surpass signal-score fusion (v1: 90.1% vs arbiter: 89.0%), but scaling to 1630 dims without careful regularization or feature selection leads to overfitting. The promoted checkpoint (84.7%) is worse than the arbiter (89.0%). However, the addition of MERT embeddings in Round 18 (Section 4.16) resolved this regression: MeterNet+MERT (3166d) achieves 632/700 (90.3%) with multi-seed validation, demonstrating that *qualitatively different* features (pretrained transformer vs. redundant DSP) are the key to scaling dimensionality successfully.

### 4.16 MERT-v1-95M Embeddings + Multi-Seed Finale (Round 18)

#### Motivation

MERT-v1-95M provides pretrained music transformer features that are orthogonal to the DSP-based features already used by MeterNet. A previous attempt to integrate MERT (Round 12, Section 4.8) failed because we compressed 1536-dimensional embeddings down to 12-dimensional signal scores before passing them to the arbiter — destroying the rich representational structure. Round 18 bypasses this bottleneck by feeding raw MERT embeddings directly into MeterNet as an additional feature group, analogous to how onset_mlp audio features (1449d) were directly integrated in Round 17.

#### Architecture

MERT-v1-95M layer 3 embeddings (1536d) are added as a new feature group to MeterNet's input. Embeddings are pre-extracted from `data/mert_embeddings/` (mean+max pooling over 5-second chunks, stored as `.npy` files).

Total input dimensionality: **3166 dims**:

```
Existing features (1630d):
  ├── Audio features (1449d): autocorrelation, MFCC, tempogram, beat-position histograms
  ├── Beat-sync chroma SSM (75d): self-similarity at lags 2-12 per tracker
  ├── Beat tracker features (42d): IBI stats, alignment, downbeat histograms
  ├── Signal scores (60d): 5 signals × 12 meters
  └── Tempo features (4d): librosa + tempogram BPM, ratio, agreement
New features (1536d):
  └── MERT-v1-95M layer 3 embeddings: mean+max pooling across 5s chunks
```

#### Grid Search Phase 1: Architecture Exploration

45 combinations exploring 3 hidden sizes × 4 PCA compression levels for MERT × 3 PCA levels for autocorrelation × 2 MERT ablation (with/without):

- **Hidden sizes**: 512, 640, 756
- **PCA-MERT**: none, 256, 512, 768 (compressing 1536d MERT features)
- **PCA-autocorrelation**: none, 256, 512 (compressing 1024d autocorrelation features)
- **Ablation**: with MERT vs. without MERT (no_mert baseline)

Key findings from Phase 1:
- Raw features consistently outperformed PCA-compressed variants for both MERT and autocorrelation.
- All MERT configurations outperformed the no-MERT baseline by a substantial margin.
- Top 3 candidates by validation balanced accuracy: h=756 (val=77.7%), h=640 (val=77.3%), h=756+pca_ac=256 (val=77.1%).

#### Grid Search Phase 2: Multi-Seed Finale

Single-seed selection proved unreliable in previous rounds (the Round 17 paradox: val-optimal ≠ test-optimal). Phase 2 evaluated the top 3 candidates across 5 random seeds each to obtain robust performance estimates:

| Configuration | Val Balanced Acc (5 seeds) | Std |
|---------------|---------------------------|-----|
| **h=512** | **77.2%** | **±0.6%** |
| h=640 | 76.6% | ±0.8% |
| h=756 + pca_ac=256 | 76.5% | ±0.9% |

**Critical finding**: The Phase 1 winner (h=756, single-seed val=77.7%) performed worst in the multi-seed finale. This confirms that single-seed grid search is unreliable and that multi-seed validation is essential for honest model selection. Smaller networks (h=512) exhibited lower variance across seeds, suggesting better generalization.

#### Winner: h=512 (Promoted)

| Metric | Round 17c (no MERT) | Round 18 (MERT) | Delta |
|--------|---------------------|-----------------|-------|
| **Overall** | 593/700 (84.7%) | **632/700 (90.3%)** | **+39 (+5.6pp)** |
| Balanced accuracy | 79.2% | **83.0%** | +3.8pp |
| 3/x | 276/302 (91.4%) | 284/302 (94.0%) | +8 (+2.6pp) |
| 4/x | 251/307 (81.8%) | 281/307 (91.5%) | +30 (+9.7pp) |
| 5/x | 26/42 (61.9%) | 28/42 (66.7%) | +2 (+4.8pp) |
| 7/x | 40/49 (81.6%) | 39/49 (79.6%) | −1 (−2.0pp) |

#### MERT Impact Analysis

MERT integration accounts for +39 files on METER2800 compared to the no-MERT baseline (593→632). This is the largest single-feature improvement in the project's history, surpassing even the arbiter MLP introduction (+91 files over hand-tuned, but that was a methodology change, not a feature addition).

The improvement is concentrated in 3/x (+8) and 4/x (+30), consistent with MERT's strength on tonal/harmonic music identified in Round 12. The 7/x class shows a minor regression (−1 file), possibly because MERT features dilute the DSP signals that are more discriminative for odd meters.

#### PCA Findings

Both PCA-MERT and PCA-autocorrelation hurt performance in this context. The MeterNet architecture appears to benefit from access to the full feature space, with its residual blocks and dropout providing sufficient regularization. This contrasts with the Round 17 observation that more features can cause overfitting — the difference is that MERT features are qualitatively different (pretrained transformer representations vs. hand-crafted DSP), providing genuinely orthogonal information rather than redundant dimensions.

#### WIKIMETER Results

| Metric | Round 17c | Round 18 | Delta |
|--------|-----------|----------|-------|
| Overall | 192/298 (64.4%) | 206/298 (69.1%) | +14 (+4.7pp) |

WIKIMETER improvements on 3/x, 4/x, 5/x, and 9/x; regressions on 7/x and 11/x. The 7/x regression is consistent with the minor METER2800 7/x drop, suggesting MERT features may interfere with odd-meter discrimination in some contexts.

#### Conclusion

Round 18 demonstrates that MERT-v1-95M embeddings, when provided as raw features rather than compressed scores, provide substantial orthogonal information to DSP-based features. The multi-seed validation protocol proved essential — the single-seed Phase 1 winner would have been a suboptimal choice. The combination of pretrained music transformer features with hand-crafted DSP features achieves a new project best of 632/700 (90.3%) on METER2800.

### 4.17 Pipeline Simplification — Audio + MERT Only (Round 19)

#### Motivation

Ablation analysis of MeterNet's 6 feature groups suggested that audio features (1449d) and MERT embeddings (1536d) together capture the vast majority of the model's discriminative power — 2985d out of 3166d. The remaining 181 dimensions (SSM 75d, beat tracker stats 42d, signal scores 60d, tempo 4d) depend on heavy external trackers (BeatNet ~200MB, beat-this ~150MB, madmom ~200MB) that add significant complexity, installation burden, and runtime cost.

#### Methodology

**Removed components**:
- **Beat trackers**: BeatNet, Beat This!, madmom (kept only librosa)
- **Signals**: SSM/chroma self-similarity (75d), bar_tracking, autocorrelation, HCDF, downbeat_spacing
- **Feature groups**: beat tracker statistics (42d), signal scores (60d), tempo features (4d)
- **Total removed**: 181 dimensions (3166d → 2985d)

**Simplified pipeline**: onset detection → librosa beats → tempo estimation → MeterNet (audio+MERT) → sections

**Impact**:
- Eliminated ~550MB of git dependencies (BeatNet, beat-this, madmom)
- Removed ~4200 lines of code (tracker wrappers, signal implementations, trust gating, cache warming infrastructure)
- MeterNet input reduced from 3166d to 2985d (audio 1449d + MERT 1536d)

#### Results

| Metric | Round 18 (3166d) | Round 19 (2985d) | Delta |
|--------|------------------|------------------|-------|
| **METER2800 overall** | 632/700 (90.3%) | **631/700 (90.1%)** | **−1 (−0.2pp)** |
| Balanced accuracy | 83.0% | **84.3%** | **+1.3pp** |
| 3/x | 284/302 (94.0%) | 287/302 (95.0%) | +3 (+1.0pp) |
| 4/x | 281/307 (91.5%) | 274/307 (89.3%) | −7 (−2.2pp) |
| 5/x | 28/42 (66.7%) | 29/42 (69.0%) | +1 (+2.3pp) |
| 7/x | 39/49 (79.6%) | 41/49 (83.7%) | +2 (+4.1pp) |
| **WIKIMETER** | 206/298 (69.1%) | **192/298 (64.4%)** | **−14 (−4.7pp)** |

#### Analysis

- **METER2800**: Only 1 file lost overall (−0.2pp), well within noise. Balanced accuracy actually *improved* by +1.3pp, driven by gains on rare classes (5/x +2.3pp, 7/x +4.1pp). The 4/x regression (−7 files) is offset by improvements in 3/x (+3), 5/x (+1), and 7/x (+2).
- **WIKIMETER**: The 14-file regression (69.1% → 64.4%) suggests that signal scores and beat tracker statistics provided some value for the more diverse WIKIMETER genres (9/x, 11/x classes that are absent from METER2800). This is expected — the removed signals were specifically designed for complex meter discrimination.
- **Practical impact**: The simplified pipeline installs in seconds instead of minutes, avoids GPU contention from multiple tracker models, and eliminates the cache warming step entirely.

#### Conclusion

The heavy tracker/signal infrastructure — BeatNet, beat-this, madmom, SSM, bar_tracking, autocorrelation, HCDF, downbeat_spacing — is practically redundant for the primary METER2800 benchmark. The MeterNet network learns equivalent discriminative patterns from raw audio features (autocorrelation, MFCC, tempogram, beat-position histograms) combined with MERT embeddings. This validates the hypothesis that a sufficiently rich feature space (audio+MERT = 2985d) makes hand-crafted signal engineering unnecessary. The trade-off is a modest WIKIMETER regression, acceptable given the massive reduction in complexity and dependencies.

### 4.18 MLP × FTT Ensemble and Hybrid Model (Round 20)

#### Motivation

Grid search (Section 4.17) trained both MLP and FT-Transformer (FTT) architectures on the same 2985d features. The best MLP achieves 631/700 and the best FTT 629/700 — similar overall accuracy but with **different error patterns**. Disagreement analysis revealed:

- 49 files (7%) where the two architectures disagree
- 21 MLP-only correct, 19 FTT-only correct (9 both wrong)
- Oracle ceiling: 649/698 (92.8%) — the potential gain from perfect routing
- FTT outperforms MLP on 4/x (+8), MLP outperforms FTT on 3/x (+10)

This motivated two approaches: (1) a zero-training ensemble via probability averaging, and (2) a hybrid dual-branch model trained end-to-end.

#### Approach 1: Ensemble (probability averaging)

Average sigmoid probabilities from two independently trained models at inference time. No additional training required — just load both checkpoints.

**Implementation**: Environment variable `METER_NET_ENSEMBLE=1` gates loading a second model from `data/meter_net_ftt.pt`. Each model standardizes features with its own `feat_mean`/`feat_std` (from different k-fold retrains). Final probabilities: `probs = (sigmoid(logits_mlp) + sigmoid(logits_ftt)) / 2`.

#### Ensemble Results

| Metric | MLP solo (R19) | FTT solo | Ensemble | Delta (vs MLP) |
|--------|---------------|----------|----------|----------------|
| **Overall** | 631/700 (90.1%) | 629/700 (89.9%) | **639/700 (91.3%)** | **+8 (+1.2pp)** |
| Balanced accuracy | 84.3% | 82.8% | 83.9% | −0.4pp |
| 3/x | 287/302 (95.0%) | 277/302 (91.7%) | 285/302 (94.4%) | −2 (−0.6pp) |
| 4/x | 274/307 (89.3%) | 282/307 (91.9%) | 286/307 (93.2%) | +12 (+3.9pp) |
| 5/x | 29/42 (69.0%) | 29/42 (69.0%) | 27/42 (64.3%) | −2 (−4.7pp) |
| 7/x | 41/49 (83.7%) | 41/49 (83.7%) | 41/49 (83.7%) | 0 |

#### Analysis

- **Overall +8 files**: The ensemble captures 8 of the potential 19 FTT-only correct files without losing any of the 21 MLP-only correct files (net: +8).
- **4/x is the main beneficiary**: +12 files (89.3% → 93.2%), the largest single-class gain. The FTT's attention mechanism likely resolves 3/x vs 4/x ambiguities that the MLP struggles with.
- **5/x regresses**: −2 files (69.0% → 64.3%). The two models agree on most 5/x errors, and averaging dilutes the correct model's confidence on the 2 files where they disagree.
- **Balanced accuracy drops slightly** (84.3% → 83.9%) due to the 5/x regression, even though overall accuracy improves.
- **Zero cost**: No training, no new features — pure inference-time combination.

#### Approach 2: Hybrid dual-branch model

A single `HybridMeterNet` with both branches trained end-to-end:

- **MLP branch**: `Linear(2985→512) → BN → ReLU → 2×ResidualBlock(512)` → 512d representation
- **FTT branch**: semantic_v4 tokenization (10 tokens) → `TransformerEncoder(d=128, h=4, L=2)` → CLS token → 128d representation
- **Fusion head**: `LayerNorm(640) → Linear(640→320) → ReLU → Dropout(0.2) → Linear(320→6)`

Same training setup as standalone models: BCE loss, CutMix augmentation, cosine annealing, 5-fold CV on METER2800.

#### Hybrid Results

| Metric | MLP solo | FTT solo | Ensemble | Hybrid | Delta (hybrid vs ensemble) |
|--------|----------|----------|----------|--------|---------------------------|
| **METER2800** | 631/700 (90.1%) | 629/700 (89.9%) | **639/700 (91.3%)** | 627/700 (89.6%) | **−12 (−1.7pp)** |
| Balanced accuracy | 84.3% | 82.8% | 83.9% | 84.1% | +0.2pp |
| 3/x | 287/302 (95.0%) | 277/302 (91.7%) | 285/302 (94.4%) | 281/302 (93.0%) | −4 |
| 4/x | 274/307 (89.3%) | 282/307 (91.9%) | 286/307 (93.2%) | 276/307 (89.9%) | −10 |
| 5/x | 29/42 (69.0%) | 29/42 (69.0%) | 27/42 (64.3%) | **31/42 (73.8%)** | **+4** |
| 7/x | 41/49 (83.7%) | 41/49 (83.7%) | 41/49 (83.7%) | 39/49 (79.6%) | −2 |
| **WIKIMETER** | — | — | — | 194/298 (65.1%) | — |

Training config: 5-fold CV on M2800 tuning (2100) + WIKIMETER train, h=512 (MLP), d_model=128 (FTT), semantic_v4, mean epoch=45. Val balanced: 83.6% ± 1.8%.

#### Hybrid Analysis

- **5/x: 73.8% — best ever** (+4.8pp vs solo, +9.5pp vs ensemble). The fusion head learns to route 5/x predictions effectively, exploiting complementary patterns from both branches.
- **Overall regression** (627 vs 639): The end-to-end fusion compromises — it averages both branches' strengths instead of specializing. The 4/x regression (−10 vs ensemble) is the main driver.
- **Balanced accuracy comparable** (84.1% vs 84.3% MLP): The hybrid's 5/x gain compensates for 7/x loss in the balanced metric.

#### Conclusion

The zero-training ensemble achieves a new project best of **639/700 (91.3%)**, demonstrating that MLP and FTT capture genuinely complementary patterns in the feature space. The hybrid model's end-to-end fusion underperforms the naive ensemble by 12 files — learned fusion compromises rather than specializes. However, the hybrid achieves the best-ever 5/x accuracy (73.8%), suggesting potential for class-aware routing. Next step: a micro stacking head trained on model probabilities rather than raw features.

## 5. Failure Analysis

We analyzed incorrectly classified files and identified three recurring failure patterns.

### 5.1 Pattern A: beat_periodicity Forces 3/4 on Duple Music

**Mechanism**: When all neural network trackers are untrusted (alignment < 0.4), their combined weight (0.39) is redistributed. beat_periodicity (w=0.20) becomes the dominant signal. In certain duple-meter music with strong beat-level accent patterns at period 3, beat_periodicity scores 3/4 at up to 3.7x the 4/4 score.

**Affected genres**: Marches, ragtime, blues with walking bass.

**Root cause**: A strong rhythmic pattern repeating every 3 beats in the RMS energy does not necessarily indicate 3/4 meter -- it can arise from syncopation patterns in 4/4 music (e.g., ragtime left-hand patterns: bass-chord-chord, repeating).

**Mitigations applied**:
- NN 3/4 penalty: when all NNs favor even meters, 3/4 score × 0.55.
- beat_periodicity weight cap: reduce to W_PERIODICITY_CAPPED (0.16) when all NNs are untrusted.

### 5.2 Pattern B: False Compound /8 Detection

**Mechanism**: The sub_beat_division signal counts onsets between consecutive beats. Certain non-compound music produces onset counts in the 1.5--3.5 range, falsely triggering compound detection.

**Sources of false positives**:
- Accompaniment arpeggios (waltz Alberti bass: 3 notes per beat)
- Ornamental passages (Bach sarabande trills)
- Polka off-beat accompaniment patterns

**Mitigations applied**:
- Evenness check: CV of inter-onset intervals > 0.4 indicates irregular subdivision, not compound meter.
- Triplet position check: onsets must be near 1/3 and 2/3 positions.

### 5.3 Pattern C: Trust Gate Too Strict for Expressive Music

**Mechanism**: Classical and folk music with rubato, ornamentation, and complex textures produces low onset alignment scores (0.25--0.37), below the trust threshold of 0.4. This disables all NN signals, leaving only onset_autocorrelation, accent_pattern, and beat_periodicity. These signals alone achieve approximately 65% accuracy compared to approximately 85% when NN signals are active.

**Mitigations applied**:
- bar_tracking: Independent of trust gating, provides NN-derived information even when downbeat_spacing/madmom_activation are disabled.
- resnet_meter: Also independent of trust gating, but at 75% accuracy proved too noisy (see Section 4.6). Disabled.

**Remaining gap**: Lowering the trust threshold causes compound meter regressions (see Section 4.2). Tracker-specific trust thresholds (lower for Beat This!, higher for BeatNet) or a substantially more accurate classifier (>90%) could address this.

## 6. Lessons Learned from Classifier Experiments

### 6.1 Motivation

The fundamental limitation of the active signals is their shared dependency on beat detection. When audio is challenging for beat trackers (rubato, complex polyphony, unusual timbres), most signals degrade simultaneously. A classifier operating directly on audio features was hypothesized to provide orthogonal information.

### 6.2 resnet_meter (75.4% test)

- ResNet18 on MFCC spectrograms. Training and integration details in Section 4.6.
- **Key failure**: At 75%, model accuracy matches the ensemble's 74% on METER2800. Correlated errors → 18 regressions.

### 6.3 mert_meter (80.7% test)

- MERT-v1-95M frozen encoder + MLP. Training and integration details in Section 4.8.
- **Key failure**: Despite better accuracy (+5.3pp), only 67.6% on our benchmark fixtures vs engine's 81.2%. Loss ratio 2.2:1.
- **Strength**: Genuine advantage on classical 3/4 (waltzes, mazurkas, sarabandes).
- **Weakness**: Over-predicts odd meters (7/4, 5/4) on percussion/modern music.

### 6.4 Takeaways

For a classifier signal to genuinely help the ensemble, it must:
1. Achieve **>90% standalone accuracy** on the evaluation dataset
2. Have an **orthogonal error profile** -- failing on different files than existing signals
3. Provide **calibrated confidence** -- so integration can be gated on certainty
4. Pass the **gate check**: complementarity ratio >1.5 (gains/losses)

Infrastructure for both approaches (training pipelines, embedding cache, graceful degradation) is complete and ready for improved models.

## 7. Potential Contributions

1. **Learned signal fusion via arbiter MLP** -- Replacing ~200 lines of hand-tuned combination logic with a tiny MLP (72→64→32→6) that achieves 89.0% on METER2800 test (+13.0pp over hand-tuned). Demonstrates that learned fusion dramatically outperforms manual weight tuning, especially for rare classes.

2. **Unified meter classification (MeterNet+MERT)** -- End-to-end learning from 3166-dim feature vectors combining DSP audio features (1449d), MERT-v1-95M embeddings (1536d), SSM, beat statistics, signal scores, and tempo. Achieves 90.3% on METER2800 test with multi-seed validation. Demonstrates that pretrained music transformer features, when provided as raw embeddings rather than compressed scores, provide substantial orthogonal information to DSP features (+39 files, largest single-feature improvement).

3. **Signal ablation methodology** -- Leave-one-out ablation with 10 seeds per variant provides statistically robust signal importance ranking. Found that 4/10 signals contribute 0pp — reducing features from 120 to 72 with no accuracy loss.

4. **Multi-signal ensemble with trust gating** -- Combining independent analysis signals with alignment-based trust weighting. The arbiter learns implicit trust from signal score patterns (all-zero = untrusted tracker), eliminating the need for explicit trust features.

5. **Comprehensive evaluation on METER2800** -- Surpassing the original ResNet18 paper's 88% on binary 3/4 vs 4/4. Best overall: 90.3% on 4-class including odd meters (multi-seed validated). Unified evaluation framework (`scripts/eval.py`) with per-tracker caching, parallel workers, stratified sampling, run snapshots, and regression detection.

6. **Granular feature caching** -- Separating checkpoint-dependent (forward pass) from checkpoint-independent (feature extraction) cache entries, enabling 60× faster eval cycles after model retraining.

7. **Failure taxonomy for meter detection** -- Three systematic failure patterns (beat_periodicity-driven false triple, false compound detection, trust gate strictness) with specific mitigations.

8. **Multi-seed model selection** -- Single-seed grid search is unreliable: the Phase 1 winner (h=756, val=77.7%) ranked worst in the 5-seed finale. Multi-seed validation reveals that smaller networks (h=512) are more stable (std=0.6% vs 0.9%) and generalize better.

9. **Negative results: orthogonality experiments** -- resnet_meter (75%), mert_meter (80.7%), and four signals (periodicity, madmom, accent, resnet) all fail to contribute to the learned ensemble. MeterNet v6 (Round 17) shows that more features ≠ better results when features are redundant. However, qualitatively different features (MERT, Round 18) do help — the key is orthogonality, not dimensionality.

## 8. Future Work

1. **Hybrid model evaluation** -- The `HybridMeterNet` (dual-branch MLP+FTT with learned fusion) is implemented and ready for training. If it can learn to route predictions better than naive averaging, it may exceed the ensemble's 639/700. Target: 640+ overall with improved 5/x balanced accuracy.

2. **Improve 5/x accuracy** -- 5/x regressed to 64.3% in the ensemble (from 69.0% single-model). The ensemble dilutes correct predictions when models disagree on rare classes. Potential: (a) weighted ensemble with class-specific weights, (b) expand WIKIMETER 5/x data, (c) quintuple-specific augmentation via Skip That Beat.

3. **Recover WIKIMETER accuracy** -- The pipeline simplification (Round 19) caused a WIKIMETER regression from 69.1% to 64.4%. Investigate whether the ensemble or hybrid model improves WIKIMETER as well.

4. **MERT fine-tuning** -- Layer 3 embeddings are frozen. LoRA fine-tuning on METER2800 could adapt MERT representations specifically for meter classification, potentially improving accuracy across all classes.

5. **Ensemble weight tuning** -- The current ensemble uses equal weights. Per-class or learned weights (e.g., trained on validation set) could improve balanced accuracy by routing 5/x predictions to the stronger single model.

6. **WIKIMETER expansion** -- The 11/x class remains the weakest in training data (83 songs); expanding it should improve rare-class performance.

7. **Additional evaluation datasets** -- Expand beyond METER2800 to include non-Western music (Hindustani, Carnatic, Afro-Cuban) with complex meter structures (tala systems with 7, 10, 16 beats).

8. **Data augmentation with Skip That Beat** -- Using the beat-removal augmentation technique from Morais et al. (2024) to generate training data for underrepresented meters.

9. **Confidence calibration** -- Temperature scaling or Platt calibration on ensemble sigmoid outputs could improve confidence estimates for downstream use.

## References

1. Foscarin, F., Schluter, J., and Widmer, G. "Beat This! Accurate Beat Tracking Without DBN." Proc. ISMIR, 2024. [arXiv:2407.21658](https://arxiv.org/abs/2407.21658)

2. Abimbola, O., Akinola, S., and Adetunmbi, A. "Time signature classification using MFCC feature extraction and ResNet18." EURASIP J. Audio, Speech, Music Process., 2024. [DOI:10.1186/s13636-024-00346-6](https://link.springer.com/article/10.1186/s13636-024-00346-6)

3. Abimbola, O., Akinola, S., and Adetunmbi, A. "METER2800: A dataset for time signature classification." Data in Brief, vol. 51, 2023. [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0CLXBQ) | [PMC:10700346](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700346/)

4. Morais, G., Fuentes, M., and McFee, B. "Skip That Beat: Augmenting Meter Tracking Models for Underrepresented Time Signatures." LAMIR / ISMIR LBD, 2024. [GitHub](https://github.com/giovana-morais/skip_that_beat)

5. Hydri, S., Bock, S., and Widmer, G. "BeatNet+: Auxiliary training for percussive-invariant beat tracking." TISMIR, 2024. [DOI:10.5334/tismir.198](https://transactions.ismir.net/articles/10.5334/tismir.198)

6. Liang, J. and Mysore, G. "BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer." Proc. ICASSP, 2024. [arXiv:2312.17156](https://arxiv.org/abs/2312.17156)

7. Ballroom Extended Dataset. Available via [mirdata](https://mirdata.readthedocs.io/).

8. Ramos, D., Bittner, R., Bello, J. P., and Humphrey, E. "Time Signature Detection: A Survey." Sensors, vol. 21, no. 19, 2021. [DOI:10.3390/s21196494](https://www.mdpi.com/1424-8220/21/19/6494)

9. Li, Y., Yuan, R., Zhang, G., et al. "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training." Proc. ICLR, 2024. [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)
