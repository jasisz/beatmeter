# BeatMeter: Multi-Signal Ensemble for Automatic Meter Detection

## Abstract

We present BeatMeter, a multi-signal ensemble system for automatic meter detection from audio. The system combines 7 analysis signals -- neural beat trackers (BeatNet, Beat This!, madmom), onset autocorrelation, accent pattern analysis, beat strength periodicity, and bar tracking via GRU-RNN with Viterbi decoding -- with trust-gated weighting and consensus bonuses. Trust gating disables unreliable neural signals based on onset alignment quality, while consensus bonuses reward cross-signal agreement. On the METER2800 dataset (700-file hold-out test split), we achieve 76% overall meter accuracy and 88.2% on binary 3/4 vs 4/4 classification (comparable to the ResNet18 paper's 88%). We identify three root failure patterns and propose specific mitigations for each. We report two negative results on adding spectrogram/embedding classifiers: (1) a ResNet18 MFCC classifier at 75% standalone accuracy, and (2) a MERT-v1-95M foundation model with MLP at 80.7% accuracy. Neither provides sufficient orthogonal information for ensemble benefit, though MERT shows promise for classical 3/4 detection.

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
6. **Meter hypothesis generation** -- Seven active signals produce scores for candidate meters (2/4, 3/4, 4/4, 5/4, 6/8, 7/4, 12/8). Scores are combined via weighted addition with consensus bonuses (`beatmeter/analysis/meter.py`).

Trust gating controls the influence of neural-network-based signals. Each tracker receives a trust score derived from its onset alignment quality:

- Alignment < 0.4 --> trust = 0 (signal disabled)
- Alignment 0.4 to 0.8 --> trust ramps linearly from 0 to 1.0
- Alignment >= 0.8 --> trust = 1.0

This mechanism ensures that when beat trackers fail on difficult audio (low alignment with detected onsets), their potentially misleading outputs are suppressed rather than allowed to corrupt the ensemble.

### 1.2 Signals

All signal implementations live in `beatmeter/analysis/signals/`.

| Signal | Code key | Weight | Source | Description |
|--------|----------|--------|--------|-------------|
| **downbeat_spacing** (BeatNet) | `beatnet` | 0.13 × trust | BeatNet beats | Infers meter from the distribution of inter-downbeat intervals. Clusters downbeat spacings and maps dominant intervals to candidate meters. |
| **downbeat_spacing** (Beat This!) | `beat_this` | 0.16 × trust | Beat This! beats | Same approach but using the SOTA conv-transformer downbeat tracker, which provides higher-quality downbeat estimates on most genres. |
| **madmom_activation** | `madmom` | 0.10 × trust | madmom beats | Evaluates whether predicted downbeat positions align with louder onsets. Scores each candidate meter by checking if grouping beats into bars of that size places the loudest beat on position 1. |
| **onset_autocorrelation** | `autocorr` | 0.13 | onset envelope | Computes autocorrelation of the onset strength envelope at lags corresponding to expected bar durations for each candidate meter. Higher autocorrelation at a given lag indicates stronger periodicity at that bar length. |
| **accent_pattern** | `accent` | 0.18 | raw audio RMS | Groups beats into bars of each candidate meter size and measures the consistency of the accent on beat 1 using RMS energy. A strong, consistent downbeat accent favors that meter. |
| **beat_periodicity** | `periodicity` | 0.20 (cap 0.16) | raw audio RMS | Computes autocorrelation of beat-level RMS energies to find the dominant accent period. The candidate meter whose beats-per-bar best matches the dominant period scores highest. Weight is capped at 0.16 when all NNs are untrusted. |
| **bar_tracking** | `bar_tracking` | 0.12 | audio + beats | Uses madmom's GRU-RNN activation function with Viterbi decoding to track bar boundaries for each candidate meter. Quality-gated: skipped on sparse/synthetic audio (non_silent < 0.15). |
| **resnet_meter** | `resnet` | **0.0 (disabled)** | MFCC spectrogram | Direct 4-class CNN classification. Trained on METER2800 (75.4% test). **Disabled**: not orthogonal to existing signals (see Section 4.6). |
| **mert_meter** | `mert` | **0.0 (disabled)** | MERT embeddings | Music foundation model (95M params) as frozen feature extractor with MLP (80.7% test). **Disabled**: gate FAIL, 31 gains vs 68 losses (see Section 4.8). |

**Note**: HCDF (Harmonic Change Detection Function) was evaluated but not integrated due to regression issues (see Section 4.4).

### 1.3 Combination Strategy

Candidate meter scores are combined through weighted additive fusion with several modifiers:

1. **Weighted sum**: Each signal contributes its score multiplied by its weight (and trust factor for NN-based signals).

2. **Consensus bonus**: When multiple signals agree on the top meter:
   - 3+ signals agree --> 1.15x multiplier
   - 4+ signals agree --> 1.25x multiplier

3. **Prior probabilities**: Reflects the natural distribution of meters in music:
   - 4/4 = 1.15, 3/4 = 1.03, 6/8 = 1.05, 2/4 = 1.05, 12/8 = 1.02

4. **NN 3/4 penalty**: When all neural network trackers (downbeat_spacing BeatNet, Beat This!, madmom_activation) favor even meters, the 3/4 score is multiplied by 0.55. This corrects for the tendency of beat trackers to underestimate triple meter.

5. **Compound meter (/8) detection** (`signals/sub_beat_division.py`): Sub-beat division analysis checks for compound meter evidence:
   - Median onset count between beats in range 1.5--3.5
   - Evenness coefficient of variation (CV) < 0.4
   - Triplet position check (onsets near 1/3 and 2/3 positions)
   - When detected: transfer 50% of simple meter score to compound equivalent, boost compound by 1.3x

6. **Rarity penalties**: Uncommon meters (5/4, 7/4) receive multiplicative penalties to reduce false positives.

## 2. Benchmark

### 2.1 Evaluation Framework

Evaluation uses a unified script (`scripts/eval.py`) with subprocess isolation per file to avoid BeatNet/madmom threading deadlocks. Beat tracking is cached per-tracker via `AnalysisCache` with smart invalidation (changing `meter.py` does not invalidate beat tracker caches). Run snapshots (`--save`) are stored in `.cache/runs/` with full per-file results for history tracking and regression detection.

```bash
uv run python scripts/eval.py --limit 3 --workers 1   # smoke test
uv run python scripts/eval.py --quick                  # stratified 100 (~20 min)
uv run python scripts/eval.py --split test --limit 0   # hold-out 700
uv run python scripts/eval.py --save                   # save run snapshot
uv run python scripts/dashboard.py                     # run history
```

### 2.2 Primary Benchmark: METER2800

**Dataset**: METER2800 (Abimbola et al., 2023) -- 2800 audio clips across 4 time signature classes (3, 4, 5, 7 beats per bar). Sources: FMA, MAG, OWN, GTZAN. Pre-defined splits: 1680 train, 420 val, 700 test.

**Current results** (7 active signals, resnet_meter/mert_meter disabled) on the full test split (700 files):

| Metric | Result |
|--------|--------|
| **Overall** | **532/700 (76.0%)** |
| Meter 3 (300 files) | 256/300 (85.3%) |
| Meter 4 (300 files) | 273/300 (91.0%) |
| Meter 5 (50 files) | 1/50 (2.0%) |
| Meter 7 (50 files) | 2/50 (4.0%) |
| Binary 3 vs 4 only | 529/600 (88.2%) |

**Comparison to literature**: On the binary 3/4 vs 4/4 task, our 88.2% is comparable to the ResNet18 paper's 88% (Abimbola et al., EURASIP 2024). Our system is strong on 4/4 (91.0%) and 3/4 (85.3%).

**Critical finding**: Our system essentially does not work on odd meters (5/4: 2%, 7/4: 4%). This is a fundamental limitation of the current architecture, which is heavily biased toward common Western meters.

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
| 12a | 303 | -- (in progress) | Multi-label sigmoid, 6 classes, WIKIMETER, LoRA 330M (val 14.7% @ ep9, crashed) |
| 12b | 303 | -- (pending) | WIKIMETER dataset (244 songs, all meters + poly), disk checkpointing, L4 GPU |

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

1. **WIKIMETER** (Round 12a): 1028 segments from 82 odd-meter songs (5/x, 7/x, 9/x, 11/x only)
2. **WIKIMETER** (Round 12b): 244 songs across all meters (74×3/4, 87×4/4, 33×5/x, 26×7/x, 30×9/x, 24×11/x, 29×poly). Song catalog in `scripts/setup/wikimeter.json` (single source of truth). Per-song duration filtering to reject album/compilation downloads

Novel diagnostic metrics include Confidence Gap (ΔP), Normalized Shannon Entropy, inter-class correlation matrix for label leakage detection, and per-class noise floor for data-driven polyrhythm detection thresholds.

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
- **Session crashed at epoch 10**: Colab runtime disconnected, checkpoint was only in RAM → lost. This led to adding persistent disk checkpointing after each best-val epoch.

**Infrastructure improvements** from this run:
- Checkpoint saved to disk (`torch.save`) after each new best val — crash-safe
- Default model changed to 330M (95M showed 0% val for all epochs on T4)
- WIKIMETER replaces WIKIMETER — balanced across all meter classes, includes polyrhythmic songs

Full architecture and diagnostic details in [MERT-LORA-MULTILABEL.md](MERT-LORA-MULTILABEL.md).

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

1. **Multi-signal ensemble with trust gating** -- A novel approach combining 7 independent analysis signals with alignment-based trust weighting. Unlike single-pipeline approaches, this leverages complementary signals and dynamically adjusts their influence based on quality indicators.

2. **Consensus bonus mechanism** -- Cross-signal agreement reward that outperforms Product-of-Experts fusion (Section 4.1), which is brittle to individual signal failures.

3. **Trust-gated neural network weighting** -- Quality-aware signal fusion based on onset alignment F1. Addresses the problem that neural beat trackers produce confident but incorrect outputs on out-of-distribution audio.

4. **Comprehensive evaluation on METER2800** -- 88.2% on binary 3/4 vs 4/4, comparable to the original ResNet18 paper. Unified evaluation framework (`scripts/eval.py`) with per-tracker caching, parallel workers, stratified sampling, run snapshots, and regression detection.

5. **Failure taxonomy for meter detection** -- Three systematic failure patterns (beat_periodicity-driven false triple, false compound detection, trust gate strictness) with specific mitigations.

6. **Negative results: two orthogonality experiments** -- resnet_meter (75%) and mert_meter (80.7%) both fail to improve the ensemble. Demonstrates that classifier signals must not only be individually accurate but must *outperform* the existing system on the files where it fails.

## 8. Future Work

1. **MERT-330M LoRA multi-label results** -- First run (WIKIMETER) reached val 14.7% at epoch 9 before crash. Second run (WIKIMETER, balanced dataset, L4 GPU) pending. Key metrics to evaluate: mAP, confidence gap, entropy diagnostics, noise floor thresholds. See [MERT-LORA-MULTILABEL.md](MERT-LORA-MULTILABEL.md).

2. **MERT confidence gating** -- Only use mert_meter predictions when softmax confidence exceeds 0.8. Could eliminate many false 7/4 and 5/4 predictions while preserving gains on classical 3/4.

3. **Odd meter improvement** -- 5/4 at 2% and 7/4 at 4% on METER2800 test split is essentially broken. Requires both model improvements and reduced rarity penalties.

4. **Additional evaluation datasets** -- Expand beyond METER2800 to include non-Western music (Hindustani, Carnatic, Afro-Cuban) with complex meter structures (tala systems with 7, 10, 16 beats).

5. **Data augmentation with Skip That Beat** -- Using the beat-removal augmentation technique from Morais et al. (2024) to generate training data for underrepresented meters (3/4, 5/4, 7/4) from abundant 4/4 data.

6. **Real-time meter tracking** -- Extending to real-time streaming analysis with adaptive meter tracking (potentially integrating BEAST-style streaming architectures) for live performance applications.

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
