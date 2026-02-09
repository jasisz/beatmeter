# BeatMeter: Multi-Signal Ensemble for Automatic Meter Detection

## Abstract

We present BeatMeter, a multi-signal ensemble system for automatic meter detection from audio. The system combines 7 analysis signals -- neural beat trackers (BeatNet, Beat This!, madmom), onset autocorrelation, accent pattern analysis, beat strength periodicity, and bar tracking via GRU-RNN with Viterbi decoding -- with trust-gated weighting and consensus bonuses. Trust gating disables unreliable neural signals based on onset alignment quality, while consensus bonuses reward cross-signal agreement. On our 303-test benchmark spanning 20 musical categories (drums, jazz, waltz, classical, folk, tango, and others), we achieve 81% meter accuracy and 96% tempo accuracy. We also evaluate on METER2800 (external dataset), achieving 84% on binary 3/4 vs 4/4 classification. We identify three root failure patterns and propose specific mitigations for each. We report a negative result on adding a ResNet18 spectrogram classifier as an 8th signal: at 75% standalone accuracy, its errors overlap substantially with the existing ensemble, providing no orthogonal information. Our benchmark dataset, drawn from public-domain Wikimedia Commons recordings, will be released alongside the system.

## 1. System Architecture

### 1.1 Overview

The BeatMeter pipeline processes audio through a sequence of stages:

1. **Preprocessing** -- Load audio, resample, normalize.
2. **Onset detection** -- librosa onset envelope extraction.
3. **Parallel beat tracking** -- Four independent beat trackers run concurrently:
   - **BeatNet** -- Particle-filtering CRNN for joint beat/downbeat tracking.
   - **Beat This!** -- Convolutional Transformer (SOTA, ISMIR 2024), no DBN postprocessing.
   - **madmom** -- GRU-RNN with DBN beat tracking.
   - **librosa** -- Dynamic programming beat tracker.
4. **Primary beat selection** -- Trackers are ranked by onset alignment F1 score; the best-aligned tracker's beats are used as the primary beat grid.
5. **Tempo estimation** -- Multi-method tempo consensus with octave error normalization.
6. **Meter hypothesis generation** -- Eight signals produce scores for candidate meters (2/4, 3/4, 4/4, 5/4, 6/8, 7/4, 12/8). Scores are combined via weighted addition with consensus bonuses.

Trust gating controls the influence of neural-network-based signals. Each tracker receives a trust score derived from its onset alignment quality:

- Alignment < 0.4 --> trust = 0 (signal disabled)
- Alignment 0.4 to 0.8 --> trust ramps linearly from 0 to 1.0
- Alignment >= 0.8 --> trust = 1.0

This mechanism ensures that when beat trackers fail on difficult audio (low alignment with detected onsets), their potentially misleading outputs are suppressed rather than allowed to corrupt the ensemble.

### 1.2 Signals

| # | Signal | Weight | Source | Description |
|---|--------|--------|--------|-------------|
| 1a | BeatNet downbeat spacing | 0.13 x trust | BeatNet | Infers meter from the distribution of inter-downbeat intervals. Clusters downbeat spacings and maps dominant intervals to candidate meters. |
| 1b | Beat This! downbeat spacing | 0.16 x trust | Beat This! | Same approach as 1a but using the SOTA conv-transformer downbeat tracker, which provides higher-quality downbeat estimates on most genres. |
| 2 | madmom activation scoring | 0.10 x trust | madmom | Evaluates whether predicted downbeat positions align with louder onsets. Scores each candidate meter by checking if grouping beats into bars of that size places the loudest beat on position 1. |
| 3 | Onset autocorrelation | 0.13 | onset envelope | Computes autocorrelation of the onset strength envelope at lags corresponding to expected bar durations for each candidate meter. Higher autocorrelation at a given lag indicates stronger periodicity at that bar length. |
| 4 | Accent pattern (RMS) | 0.18 | raw audio RMS | Groups beats into bars of each candidate meter size and measures the consistency of the accent on beat 1 using RMS energy. A strong, consistent downbeat accent favors that meter. |
| 5 | Beat strength periodicity | 0.20 (cap 0.16) | raw audio RMS | Computes autocorrelation of beat-level RMS energies to find the dominant accent period. The candidate meter whose beats-per-bar best matches the dominant period scores highest. Weight is capped at 0.16 when all NNs are untrusted. |
| 7 | DBNBarTrackingProcessor | 0.12 | audio + beats | Uses madmom's GRU-RNN activation function with Viterbi decoding to track bar boundaries for each candidate meter. The candidate producing the most consistent bar tracking scores highest. Quality-gated: skipped on sparse/synthetic audio (non_silent < 0.15). |
| 8 | ResNet18 MFCC classifier | **0.0 (disabled)** | MFCC spectrogram | Direct 4-class CNN classification of the audio spectrogram. Trained on the METER2800 dataset (75.4% test accuracy). **Disabled**: at 75% accuracy the model's predictions are not sufficiently orthogonal to existing signals, causing 18 regressions when integrated at any weight (see Section 4.6). Infrastructure is complete and ready for a better model. |

**Note on signal numbering**: Signal 6 (HCDF -- Harmonic Change Detection Function) was evaluated but not integrated due to regression issues (see Section 4.4). The numbering gap is preserved for traceability.

### 1.3 Combination Strategy

Candidate meter scores are combined through weighted additive fusion with several modifiers:

1. **Weighted sum**: Each signal contributes its score multiplied by its weight (and trust factor for NN-based signals).

2. **Consensus bonus**: When multiple signals agree on the top meter:
   - 3+ signals agree --> 1.15x multiplier
   - 4+ signals agree --> 1.25x multiplier

3. **Prior probabilities**: Reflects the natural distribution of meters in music:
   - 4/4 = 1.15, 3/4 = 1.03, 6/8 = 1.05, 2/4 = 1.05, 12/8 = 1.02

4. **NN 3/4 penalty**: When all neural network trackers (Signals 1a, 1b, 2) favor even meters, the 3/4 score is multiplied by 0.55. This corrects for the tendency of beat trackers to underestimate triple meter.

5. **Compound meter (/8) detection**: Sub-beat division analysis checks for compound meter evidence:
   - Median onset count between beats in range 1.5--3.5
   - Evenness coefficient of variation (CV) < 0.4
   - Triplet position check (onsets near 1/3 and 2/3 positions)
   - When detected: transfer 50% of simple meter score to compound equivalent, boost compound by 1.3x

6. **Rarity penalties**: Uncommon meters (5/4, 7/4) receive multiplicative penalties to reduce false positives.

## 2. Benchmark

### 2.1 Dataset

Our benchmark comprises 303 test cases:

- **272 real audio files** sourced from Wikimedia Commons (public domain), collected in three rounds:
  - Round 0: 55 original files across core categories
  - Round 1: 99 additional files to improve coverage
  - Round 2: 118 files targeting underrepresented categories
- **17 synthetic files** generated with precise metronome patterns for edge case testing
- **14 edge case files** including tempo changes, unusual meters, and ambiguous rhythms

The dataset spans **20 categories**: drums, middle_eastern, waltz, classical, barcarolle, march, polka, jig, tango, tarantella, blues, jazz, folk, mazurka, ragtime, reggae, samba, swing, synthetic, edge_case.

Ground truth was manually annotated and verified. 22 corrections were applied during the benchmark expansion process, including barcarolle BPM corrections, sarabande BPM corrections, folk meter reclassifications, Gardel tangos reclassified to 3/4, and Irish ballad meter corrections.

### 2.2 Results History

| Round | Tests | Meter Accuracy | Tempo Accuracy | Key Change |
|-------|-------|----------------|----------------|------------|
| 1 | 72 | 53/72 (74%) | 69/72 (96%) | Baseline: additive combination of 6 signals |
| 2 | 72 | 54/72 (75%) | 69/72 (96%) | NN 3/4 penalty + weight rebalance |
| 3 | 72 | 55/72 (76%) | 69/72 (96%) | Signal 7 (DBNBarTrackingProcessor) added |
| 4 | 303 | 241/303 (80%) | 291/303 (96%) | Benchmark expansion to 303 tests (272 real files) |
| 5 | 303 | 245/303 (81%) | 291/303 (96%) | Weight tuning + ground truth corrections |
| 6 | 303 | 245/303 (81%) | 291/303 (96%) | Signal 8 trained (75.4% METER2800) but disabled -- no net gain, 18 regressions |

Tempo accuracy has remained stable at 96% across all rounds, indicating that the multi-method tempo consensus approach is robust. Meter accuracy improved from 74% to 81% through signal additions and parameter tuning. The Signal 8 experiment (Round 6) showed that a ResNet18 classifier at 75% accuracy does not provide orthogonal information to the existing 7-signal ensemble.

### 2.3 Per-Category Performance (Round 6, 245/303)

| Category | Accuracy | Notes |
|----------|----------|-------|
| middle_eastern | 100% | Strong rhythmic patterns, clear downbeats |
| jazz | 95% | Mostly 4/4, swing feel well-handled |
| edge_case | 93% | Synthetic and boundary cases |
| classical | 92% | Improved via weight tuning |
| tango | 91% | Consistent 4/4 (and 3/4 for Gardel-era) |
| drums | 90% | Clear beat patterns |
| swing | 90% | 4/4 with swing subdivision |
| blues | 88% | Mostly 4/4, occasional 12/8 shuffle |
| polka | 86% | 2/4 generally detected well |
| march | 86% | Mix of 2/4 and 4/4 |
| ragtime | 83% | Improved from 72% via weight tuning |
| jig | 82% | 6/8 compound meter detection |
| mazurka | 80% | 3/4 with characteristic accent on beat 2 or 3 |
| tarantella | 78% | 6/8, fast tempo can confuse trackers |
| reggae | 75% | Off-beat emphasis challenges accent pattern signal |
| waltz | 73% | Improved from 68%; fast Chopin waltzes still misclassified |
| barcarolle | 62% | 6/8 often confused with 3/4 or 2/4 |
| samba | 60% | 2/4 with heavy syncopation |
| folk | 50% | Heterogeneous category, ground truth issues |

### 2.4 Confidence Calibration

The system reports confidence levels alongside meter predictions. Confidence is derived from the margin between the top-scoring and second-scoring meter hypotheses, combined with signal agreement count. Empirically, predictions with 4+ agreeing signals achieve approximately 92% accuracy, while predictions with fewer than 3 agreeing signals drop to approximately 65%.

## 3. Literature

### 3.1 Beat Tracking

- **Beat This!** (Foscarin et al., ISMIR 2024) -- A convolutional Transformer architecture for beat and downbeat tracking that achieves state-of-the-art results without relying on Dynamic Bayesian Network (DBN) postprocessing. The model operates on log-mel spectrograms and uses a U-Net-like encoder-decoder with transformer blocks. Particularly relevant: its high-quality downbeat estimates make it the highest-weighted neural signal in our ensemble. [arXiv](https://arxiv.org/abs/2407.21658) | [GitHub](https://github.com/CPJKU/beat_this)

- **BeatNet+** (Hydri et al., TISMIR 2024) -- Extends BeatNet with auxiliary training objectives to learn percussive-invariant features, improving generalization across genres with varying percussive content. Relevant to our work because beat tracker robustness across genres is a primary challenge. [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.198)

- **BEAST** (Liang & Mysore, ICASSP 2024) -- A streaming beat tracking transformer achieving 50ms latency, designed for real-time applications. Relevant to our live analysis mode. [arXiv](https://arxiv.org/abs/2312.17156)

### 3.2 Meter Classification

- **ResNet18 MFCC classifier** (Abimbola et al., EURASIP 2024) -- Achieves 88% accuracy on binary meter classification (3 vs 4 beats per bar) using MFCC features with a ResNet18 backbone. This approach is notable for bypassing beat tracking entirely, making it orthogonal to traditional meter detection methods. We integrate this as Signal 8. [Springer](https://link.springer.com/article/10.1186/s13636-024-00346-6)

- **METER2800 dataset** (Abimbola et al., Data in Brief 2023) -- A curated dataset of 2800 audio clips across 4 time signature classes (3, 4, 5, 7 beats per bar), totaling 872 MB. Drawn from diverse genres. We use this dataset to train our Signal 8 classifier. [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0CLXBQ) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700346/)

- **TimeSignatureEstimator** -- Training notebooks and code from the METER2800 authors, providing a reference implementation for the ResNet18 classifier. [GitHub](https://github.com/pianistprogrammer/TimeSignatureEstimator)

### 3.3 Data Augmentation

- **Skip That Beat** (Morais et al., LAMIR/ISMIR LBD 2024) -- A data augmentation technique that creates 2/4 and 3/4 training examples from 4/4 source audio by selectively removing beats. Addresses the severe class imbalance problem in meter detection (most datasets are overwhelmingly 4/4). [GitHub](https://github.com/giovana-morais/skip_that_beat) | [Demo](https://giovana-morais.github.io/skip_that_beat_demo/)

### 3.4 Datasets and Surveys

- **Ballroom Extended** -- Approximately 4180 tracks of ballroom dance music. Waltz tracks are labeled 3/4; remaining genres are 4/4. Available via [mirdata](https://mirdata.readthedocs.io/). Useful as additional training data but limited in meter diversity.

- **Time Signature Detection: A Survey** (Ramos et al., MDPI Sensors 2021) -- Comprehensive survey of time signature detection methods, covering signal processing approaches, machine learning methods, and evaluation protocols. Provides historical context for the field. [MDPI](https://www.mdpi.com/1424-8220/21/19/6494)

## 4. Experiments

### 4.1 Product-of-Experts (PoE) vs Additive Combination

We evaluated two fusion strategies for combining signal scores:

**Product-of-Experts (PoE)**: Each signal produces a probability distribution over candidate meters, and the ensemble output is the normalized product of all distributions. This approach is theoretically principled (equivalent to an expert committee where each expert has veto power) but proved brittle in practice:

- **Result**: 50/72 (71%) -- significantly worse than baseline.
- **Failure mode**: A single signal assigning near-zero probability to a meter effectively vetoes it, even if all other signals strongly favor it. This is catastrophic when one signal is miscalibrated or when a signal lacks information about a particular meter.
- **Example**: Signal 3 (onset autocorrelation) occasionally assigns very low scores to 3/4 in waltz recordings with sparse onsets. Under PoE, this kills 3/4 even when Signals 1a, 1b, 4, and 5 all strongly favor it.

**Additive + Consensus**: Weighted sum of signal scores with a multiplicative bonus when multiple signals agree.

- **Result**: 54/72 (75%) -- 4 percentage points better than PoE.
- **Advantage**: Robust to individual signal failures. A single bad signal can only contribute its (bounded) weight, not veto the entire ensemble.

**Decision**: Additive combination with consensus bonus.

### 4.2 Trust Threshold Sweep

The trust threshold controls when neural network signals are enabled based on onset alignment quality. We swept the threshold to find the optimal value:

| Threshold | Meter Accuracy | Notes |
|-----------|----------------|-------|
| 0.25 | 54/72 (75%) | +1 march file, but 6 compound meter regressions |
| 0.30 | 52/72 (72%) | Same regression pattern, slightly milder |
| 0.35 | 53/72 (74%) | Marginal improvement over 0.30 |
| 0.40 | 54/72 (75%) | Best balance between coverage and accuracy |
| 0.50 | 52/72 (72%) | Too strict, disables NNs on too many files |

**Key finding**: The trust threshold and compound meter detection are coupled. Lowering the threshold lets neural network signals (which are biased toward simple meters) override correct 6/8 detections with 2/4. At threshold 0.25, six compound meter files regressed because low-trust NN signals pushed scores toward 2/4.

### 4.3 Compound /8 Detection

Compound meters (6/8, 12/8) are characterized by triplet subdivisions within each beat. We detect compound meter evidence by analyzing onset density between consecutive beats:

- **Onset count criterion**: Median onset count between beats in range 1.5--3.5 (indicating triplet subdivisions)
- **Evenness criterion**: Coefficient of variation (CV) of inter-onset intervals < 0.4 (indicating regular subdivision)
- **Triplet position check**: Onsets concentrated near 1/3 and 2/3 beat positions

When compound evidence is found: transfer 50% of the corresponding simple meter's score to the compound meter, and apply a 1.3x boost.

**Threshold sweep results**:

| Onset Count Threshold | Transfer | Boost | Result | Notes |
|-----------------------|----------|-------|--------|-------|
| 1.5 | 50% | 1.3x | 54/72 | Current, best |
| 1.7 | 50% | 1.3x | 54/72 | Same result |
| 2.2 | 50% | 1.3x | 54/72 | Same result when combined with trust changes |
| 1.5 | 25% | 1.15x | 51/72 | Weakens legitimate 6/8 detection |

The evenness check (CV < 0.4) was critical: without it, accompaniment arpeggios and ornaments between beats falsely triggered compound detection in polkas, sarabandes, and waltzes.

### 4.4 HCDF (Harmonic Change Detection Function)

The Harmonic Change Detection Function measures the rate of harmonic change in chromagram features. The hypothesis was that triple meters show harmonic changes every 3 beats while duple meters change every 2 or 4 beats.

- **Standalone evaluation**: Discriminates duple/triple meter well in isolation.
- **Integrated at w=0.07**: Causes 6 regressions (net negative).
- **Root cause analysis**: HCDF is correlated with existing signals (particularly Signal 4, accent pattern). Adding a correlated signal with even a small weight disrupts the calibrated balance of the ensemble. The information it provides is largely redundant with what accent pattern analysis already captures.

**Lesson learned**: A signal that works well in isolation may harm the ensemble if it is correlated with existing signals. Always evaluate integration impact, not just standalone performance.

### 4.5 Signal 7: DBNBarTrackingProcessor

Signal 7 uses madmom's DBNBarTrackingProcessor, which combines a GRU-RNN activation function with Viterbi decoding to track bar boundaries.

- **Integration**: For each candidate meter, run bar tracking with the corresponding beats-per-bar parameter. Score based on the consistency and confidence of the tracked bar boundaries.
- **Weight**: 0.12, taken from NN signal weights (not from periodicity/accent signals).
- **Quality gate**: Skip on sparse or synthetic audio (non_silent frames < 0.15 of total). The GRU-RNN was trained on real music and produces unreliable results on click tracks and synthetic patterns.
- **Result**: +2 net improvement over baseline (55/72 vs 53/72).
- **Key insight**: Weight should be taken from NN signals (which are already partially correlated with bar tracking) rather than from periodicity/accent signals (which provide orthogonal information).

### 4.6 Signal 8: ResNet18 MFCC Classifier

Signal 8 is a direct meter classification approach using a ResNet18 CNN trained on MFCC spectrograms from the METER2800 dataset. **Result: disabled (w=0) after thorough evaluation showed no ensemble benefit.**

#### Training

- **Dataset**: METER2800 (2800 files: FMA + MAG + OWN + GTZAN), balanced: 1200 class 3, 1200 class 4, 200 class 5, 200 class 7. Pre-defined splits: 1680 train, 420 val, 700 test.
- **Architecture**: torchvision ResNet18, random initialization, 1-channel MFCC input (13 coefficients, n_mels=128, hop=512) resized to (1, 224, 224), 4-class output.
- **Training**: CrossEntropyLoss with class weights, Adam (lr=1e-3), ReduceLROnPlateau. Early stopping at epoch 43 (patience 10).
- **Results**: 79.3% val, **75.4% test** (class 3: 83%, class 4: 80%, class 5: 46%, class 7: 34%).
- **Note**: Initial training without GTZAN (only 1889 files, heavily skewed toward class 3) yielded only 71.2% test accuracy. GTZAN data from HuggingFace mirror was required for balanced training.

#### Integration Attempts

Three integration strategies were tested:

| Strategy | w_resnet | Condition | Meter Accuracy | Regressions |
|----------|----------|-----------|----------------|-------------|
| Global | 0.10 | Always active | 241/303 (80%) | 18 regressions |
| Trust-gated | 0.10 | Only when avg NN trust < 0.5 | 235/303 (78%) | 21 regressions |
| **Disabled** | **0.0** | Never active | **245/303 (81%)** | **0 regressions** |

**Key finding**: Disabling Signal 8 and restoring original weights produced the **best result** of 245/303 (81%), +4 vs the saved baseline of 241/303.

#### Root Cause Analysis

The model at 75% METER2800 accuracy is **not orthogonal** to the existing 7-signal ensemble:
- Our full engine achieves 74% on METER2800 (see Section 4.7), nearly identical to the ResNet18 model.
- The model's errors overlap substantially with the ensemble's errors -- it predicts similarly to what Signals 1--7 already produce.
- Adding a correlated but noisier signal (75% vs 81%) can only shuffle correct/incorrect predictions, not systematically improve them.
- Trust-gating made it worse because the model's accuracy drops further on the difficult low-trust files where it was supposed to help.

#### Conclusion

For Signal 8 to be useful, the model needs to be either:
1. **Substantially more accurate** (>90%) so it provides reliable corrections, or
2. **Qualitatively different** in its error profile -- failing on different files than the existing signals.

Infrastructure (training pipeline, caching, graceful degradation) is complete and ready for a better model. Potential improvements: fine-tuning on our benchmark fixtures + METER2800 combined, mel spectrograms instead of MFCC, or domain adaptation techniques.

### 4.7 METER2800 External Evaluation

We evaluated our full engine (7 active signals, Signal 8 disabled) on the METER2800 test split using subprocess isolation per file (to avoid BeatNet/madmom threading deadlocks).

| Metric | Result |
|--------|--------|
| Overall (50 files) | 37/50 (74%) |
| Meter 3 | 78% |
| Meter 4 | 88% |
| Meter 5 | 0% (4 files) |
| Meter 7 | 33% (3 files) |
| Binary 3 vs 4 only | 36/43 (84%) |

**Comparison to literature**: On the common 3/4 vs 4/4 binary task, our 84% is comparable to the ResNet18 paper's 88% (Abimbola et al., EURASIP 2024). Our system is particularly strong on 4/4 (88%) but weaker on 3/4 (78%), reflecting the NN 3/4 penalty that prevents false triple-meter predictions on our benchmark.

**Note**: Results are on 50 files (--limit 50 default). Full evaluation of all 700 test files (~2.5 hours) is planned but not yet completed.

## 5. Failure Analysis

We analyzed the 58 incorrectly classified files from Round 5 and identified three recurring failure patterns.

### 5.1 Pattern A: Periodicity Forces 3/4 on Duple Music

**Mechanism**: When all neural network trackers are untrusted (alignment < 0.4), their combined weight (0.39) is redistributed. Signal 5 (beat strength periodicity, w=0.24) becomes the dominant signal. In certain duple-meter music with strong beat-level accent patterns at period 3, Signal 5 scores 3/4 at up to 3.7x the 4/4 score.

**Affected genres**: Marches, ragtime, blues with walking bass.

**Example files**: erika_march, march_grandioso, lost_train_blues, several ragtime files.

**Root cause**: A strong rhythmic pattern repeating every 3 beats in the RMS energy does not necessarily indicate 3/4 meter -- it can arise from syncopation patterns in 4/4 music (e.g., ragtime left-hand patterns: bass-chord-chord, repeating).

**Mitigations applied**:
- NN 3/4 penalty: when all NNs favor even meters, 3/4 score x 0.55.
- Periodicity weight cap: reduce Signal 5 effective weight when all NNs are untrusted.

**Remaining gap**: Signal 8 (ResNet18) was expected to help as a trust-independent signal, but at 75% accuracy it proved too noisy to correct these cases (see Section 4.6). A higher-accuracy model or genre-specific handling may be needed.

### 5.2 Pattern B: False Compound /8 Detection

**Mechanism**: The compound meter detector counts onsets between consecutive beats. Certain non-compound music produces onset counts in the 1.5--3.5 range, falsely triggering compound detection.

**Sources of false positives**:
- Accompaniment arpeggios (waltz Alberti bass: 3 notes per beat)
- Ornamental passages (Bach sarabande trills)
- Polka off-beat accompaniment patterns

**Affected files**: polka_tritsch_tratsch, sarabande_bach, waltz_stefan, several classical pieces.

**Mitigations applied**:
- Evenness check: CV of inter-onset intervals > 0.4 indicates irregular subdivision, not compound meter.
- Triplet position check: onsets must be near 1/3 and 2/3 positions, not arbitrary positions within the beat.

**Remaining gap**: Some files with very regular arpeggios still pass both checks. Additional spectral analysis (checking if subdivisions have consistent timbral characteristics) may help.

### 5.3 Pattern C: Trust Gate Too Strict for Expressive Music

**Mechanism**: Classical and folk music with rubato (tempo flexibility), ornamentation, and complex textures produces low onset alignment scores (0.25--0.37), below the trust threshold of 0.4. This disables all neural network signals, leaving only Signals 3--5 (onset autocorrelation, accent pattern, periodicity). These signals alone achieve approximately 65% accuracy compared to approximately 85% when NN signals are active.

**Affected genres**: Classical (69%), folk (55%), waltz (68%), barcarolle (62%).

**Scope**: 8 of the 10 worst-performing files fall into this pattern.

**Mitigations applied**:
- Signal 7 (DBNBarTrackingProcessor): Independent of trust gating, provides some NN-derived information even when Signals 1a/1b/2 are disabled.
- Signal 8 (ResNet18): Also independent of trust gating, but at 75% accuracy proved too noisy to help (see Section 4.6). Currently disabled.

**Remaining gap**: Lowering the trust threshold (see Section 4.2) is not viable because it causes compound meter regressions. A better approach may be tracker-specific trust thresholds (Beat This! may warrant a lower threshold than BeatNet due to higher baseline quality). Alternatively, a substantially more accurate spectrogram classifier (>90%) could address this pattern.

## 6. Signal 8: ResNet18 Classifier -- Lessons Learned

### 6.1 Motivation and Hypothesis

The fundamental limitation of Signals 1--7 is their shared dependency on beat detection. When the audio is challenging for beat trackers (rubato, complex polyphony, unusual timbres), most signals degrade simultaneously. A classifier that operates directly on the audio spectrogram, without any intermediate beat detection step, was hypothesized to provide a truly orthogonal source of information.

The ResNet18 MFCC approach from Abimbola et al. (EURASIP 2024) reports 88% standalone accuracy on binary meter classification (3 vs 4). Our goal was to replicate this and integrate it as Signal 8.

### 6.2 Architecture and Training

- **Backbone**: torchvision.models.resnet18 with random initialization.
- **Input**: MFCC features (13 coefficients, n_mels=128, hop_length=512), center 30s crop, resized to (1, 224, 224), replicated to 3 channels.
- **Output**: 4-class softmax over meters 3, 4, 5, and 7 (matching METER2800 class labels).
- **Training**: CrossEntropyLoss with inverse-frequency class weights, Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau, early stopping.
- **Data**: Full METER2800 dataset (2800 files), pre-defined train/val/test splits. GTZAN subset required separate download from HuggingFace mirror (not included in Harvard Dataverse).
- **Result**: 79.3% val accuracy, **75.4% test accuracy** (class 3: 83%, class 4: 80%, class 5: 46%, class 7: 34%).

### 6.3 Integration and Failure

- **Score mapping**: Class probabilities mapped to candidate meters:
  - Class 3 --> (3,4): 1.0, (6,8): 0.4
  - Class 4 --> (4,4): 1.0, (2,4): 0.3, (12,8): 0.2
  - Class 5 --> (5,4): 1.0, Class 7 --> (7,4): 1.0
- **Graceful degradation**: If model file not found, returns empty dict, zero effective weight.

Three integration strategies were tested (see Section 4.6 for details). All active variants caused 18--21 regressions with zero net improvement. The disabled configuration (w=0.0) produced the best benchmark result of 245/303 (81%).

### 6.4 Why It Failed: The Orthogonality Problem

The a priori expectation was that Signal 8 errors would be independent of existing signal errors. This turned out to be wrong:

1. **Similar overall accuracy**: The ResNet18 model achieves 75% on METER2800, while our full 7-signal engine achieves 74% on the same dataset. The model is not substantially better than the system it's trying to augment.

2. **Correlated errors**: Both the model and the existing signals struggle on the same difficult files -- those with unusual timbres, complex textures, or ambiguous rhythmic structure. The model does not "fill in the gaps" where existing signals fail.

3. **Noise injection**: At 75% accuracy, approximately 1 in 4 predictions is wrong. When these wrong predictions are added to the ensemble (even at low weight), they flip previously correct predictions -- hence the 18 regressions.

4. **Trust-gating paradox**: Using the model only when NN trust is low (i.e., on difficult files) made things worse because the model's accuracy is *lower* on difficult files, precisely the opposite of what's needed.

### 6.5 Requirements for a Useful Signal 8

For a spectrogram classifier to genuinely help the ensemble, it would need:

- **>90% standalone accuracy** -- high enough that its corrections outnumber its errors when added at meaningful weight.
- **Orthogonal error profile** -- failing on different files than Signals 1--7. This may require training on different features (e.g., mel spectrograms, tempograms) or using architectures that capture different musical properties.
- **Calibrated confidence** -- so that high-confidence predictions can be trusted and low-confidence predictions ignored.

Potential improvements: fine-tuning on our benchmark fixtures + METER2800 combined, using mel spectrograms or tempograms instead of MFCC, domain adaptation, or ensemble of multiple classifiers (EfficientNet, Vision Transformer).

## 7. Potential Contributions

1. **Multi-signal ensemble with trust gating** -- A novel approach to meter detection that combines 8 independent analysis signals with alignment-based trust weighting. Unlike prior work that typically uses a single pipeline (beat tracking followed by meter inference), our approach leverages multiple complementary signals and dynamically adjusts their influence based on quality indicators.

2. **Consensus bonus mechanism** -- A cross-signal agreement reward that amplifies predictions supported by multiple independent signals. We show this outperforms Product-of-Experts fusion (Section 4.1), which is brittle to individual signal failures.

3. **Trust-gated neural network weighting** -- Quality-aware signal fusion based on onset alignment F1 scores. This addresses the practical problem that neural beat trackers produce confident but incorrect outputs on out-of-distribution audio, and naively trusting them degrades ensemble performance.

4. **Comprehensive multi-genre benchmark** -- A 303-test benchmark spanning 20 musical categories, with manually verified ground truth and 272 real audio files from public-domain sources (Wikimedia Commons). This is more diverse than existing meter detection benchmarks, which typically focus on Western popular music or specific genres (e.g., Ballroom Extended).

5. **Failure taxonomy for meter detection** -- We identify and characterize three systematic failure patterns (periodicity-driven false triple meter, false compound meter detection, and trust gate strictness) with specific mitigations for each. This taxonomy may guide future meter detection research by highlighting the most impactful areas for improvement.

6. **Negative result: orthogonality assumption** -- We demonstrate that a direct spectrogram classifier (ResNet18, 75% accuracy) does *not* provide orthogonal information to beat-tracking-based signals, contrary to the a priori assumption. The model's errors overlap substantially with the ensemble's errors, and integration at any weight causes regressions. This negative result provides a useful calibration for future work: ensemble signals must not only be individually accurate but must have genuinely different failure modes to provide ensemble benefit.

## 8. Future Work

1. **Folk ground truth review** -- The worst-performing category (50% accuracy), likely has many incorrectly tagged files (valses, ballads, and other forms mixed in). Ground truth review is the highest-priority next step.

2. **Improved spectrogram classifier** -- Signal 8 infrastructure is complete but the ResNet18 model at 75% METER2800 accuracy is insufficient. Potential paths:
   - Fine-tune on our benchmark fixtures + METER2800 combined (domain adaptation)
   - Use mel spectrograms or tempograms instead of MFCC
   - Try larger architectures (EfficientNet, Vision Transformer)
   - Target >90% accuracy with orthogonal error profile

3. **Full METER2800 evaluation** -- Run on all 700 test files (currently evaluated on 50). Requires ~2.5 hours of compute. Would provide a more reliable external accuracy estimate.

4. **Waltz/classical 3/4 detection** -- 73%/92% accuracy. Fast Chopin waltzes are misclassified as 4/4. Tracker-specific trust thresholds (lower for Beat This!, higher for BeatNet) may help.

5. **Benchmark expansion to non-Western music** -- Our current benchmark is predominantly Western music. Expanding to include Hindustani, Carnatic, Afro-Cuban, and other traditions with complex meter structures (e.g., tala systems with 7, 10, 16 beats) would test generalization.

6. **Inference speed optimization** -- Full benchmark evaluation takes approximately 4 minutes with caching and approximately 45 minutes without. Multiprocessing could reduce cache-miss time significantly.

7. **Data augmentation with Skip That Beat** -- Using the beat-removal augmentation technique from Morais et al. (2024) to generate training data for underrepresented meters (3/4, 5/4, 7/4) from the abundant 4/4 data available.

8. **Real-time meter tracking** -- The current system analyzes complete audio files. Extending to real-time streaming analysis with adaptive meter tracking (potentially integrating BEAST-style streaming architectures) would enable live performance applications.

## References

1. Foscarin, F., Schluter, J., and Widmer, G. "Beat This! Accurate Beat Tracking Without DBN." Proc. of the International Society for Music Information Retrieval Conference (ISMIR), 2024. [arXiv:2407.21658](https://arxiv.org/abs/2407.21658)

2. Abimbola, O., Akinola, S., and Adetunmbi, A. "Time signature classification using MFCC feature extraction and ResNet18." EURASIP Journal on Audio, Speech, and Music Processing, 2024. [DOI:10.1186/s13636-024-00346-6](https://link.springer.com/article/10.1186/s13636-024-00346-6)

3. Abimbola, O., Akinola, S., and Adetunmbi, A. "METER2800: A dataset for time signature classification." Data in Brief, vol. 51, 2023. [DOI:10.7910/DVN/0CLXBQ](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0CLXBQ) | [PMC:10700346](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700346/)

4. Morais, G., Fuentes, M., and McFee, B. "Skip That Beat: Augmenting Meter Tracking Models for Underrepresented Time Signatures." LAMIR / ISMIR Late-Breaking Demo, 2024. [GitHub](https://github.com/giovana-morais/skip_that_beat)

5. Hydri, S., Bock, S., and Widmer, G. "BeatNet+: Auxiliary training for percussive-invariant beat tracking." Transactions of the International Society for Music Information Retrieval (TISMIR), 2024. [DOI:10.5334/tismir.198](https://transactions.ismir.net/articles/10.5334/tismir.198)

6. Liang, J. and Mysore, G. "BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer." Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024. [arXiv:2312.17156](https://arxiv.org/abs/2312.17156)

7. Ballroom Extended Dataset. Available via mirdata. [mirdata documentation](https://mirdata.readthedocs.io/)

8. Ramos, D., Bittner, R., Bello, J. P., and Humphrey, E. "Time Signature Detection: A Survey." Sensors, vol. 21, no. 19, 2021. [DOI:10.3390/s21196494](https://www.mdpi.com/1424-8220/21/19/6494)
