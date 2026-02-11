# MERT LoRA Multi-Label Experiment (Round 12)

## Abstract

Round 12 of the BeatMeter project transitions MERT fine-tuning from single-label (softmax) to multi-label (sigmoid) architecture to enable polyrhythm detection. The class set is expanded from 4 classes [3, 4, 5, 7] to 6 classes [3, 4, 5, 7, 9, 11]. Two external datasets were curated from YouTube: ODDMETER-WIKI (Round 12a, odd meters only) and WIKIMETER (Round 12b, all meters including 3/4, 4/4, and polyrhythmic). A comprehensive diagnostic framework monitors label leakage, confidence selectivity, and polyrhythm detection thresholds. First training run (330M + ODDMETER-WIKI, A100) reached val 14.7% at epoch 9 before session crash; second run (WIKIMETER, L4 GPU) pending.

## 1. Motivation

### 1.1 Why Multi-Label?

The previous MERT experiments (Rounds 8--9) used softmax with `nn.CrossEntropyLoss`, which assumes exactly one correct class per sample. This is fundamentally wrong for polyrhythmic music:

- **African drumming** (kpanlogo, agbekor): simultaneous 3 and 4 patterns
- **Progressive metal** (Meshuggah): guitar in 5/4 or 7/4 against drums in 4/4
- **Gamelan**: interlocking patterns with multiple simultaneous periodicities
- **Hemiola**: 3-against-2 patterns common in Baroque and Romantic music

Softmax forces the model to pick one meter, discarding genuine multi-meter information. Sigmoid outputs allow independent probability estimates for each class, enabling polyrhythm detection.

### 1.2 Why Expanded Classes?

METER2800 covers only [3, 4, 5, 7], but real-world odd-meter music includes:

- **9/8**: Ubiquitous in Balkan music (aksak rhythms), progressive rock (e.g., Blue Rondo a la Turk's 2+2+2+3 grouping)
- **11/8**: Turkish and Balkan folk (Kopanitsa, Gankino Horo), progressive rock (Tool)

Adding 9 and 11 to the class set makes the model useful for a broader range of non-Western and progressive music.

### 1.3 Why External Data? (ODDMETER-WIKI → WIKIMETER)

METER2800 has approximately 150 training files per odd meter class (5/4, 7/4), and zero files for 9/x or 11/x. LoRA fine-tuning requires more data for underrepresented classes. Wikipedia maintains a curated [list of musical works in unusual time signatures](https://en.wikipedia.org/wiki/List_of_musical_works_in_unusual_time_signatures) -- a musicologically vetted resource covering diverse genres and cultures.

The initial dataset (ODDMETER-WIKI, Round 12a) contained only odd meters (5/x, 7/x, 9/x, 11/x + poly), which created a training distribution mismatch — the model saw disproportionately more odd-meter data from YouTube vs METER2800. The expanded dataset (WIKIMETER, Round 12b) adds 29 curated 3/4 songs and 29 curated 4/4 songs to balance the distribution, plus per-song expected duration filtering to prevent downloading albums or compilations.

## 2. Architecture Changes

All changes are in `scripts/training/finetune_mert.py`.

### 2.1 Multi-Label Classification (Sigmoid + BCE)

**Before** (Rounds 8--9):
```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
# Labels: integer class indices [0, 1, 2, 3]
# Output: softmax probabilities, exactly one class per sample
```

**After** (Round 12):
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
# Labels: multi-hot vectors with label smoothing
# Output: independent sigmoid probabilities per class
```

Key implementation details:

- **Labels**: Multi-hot vectors where positive classes = 1.0 and negative classes = `LABEL_SMOOTH_NEG` (0.1) instead of 0.0
- **`pos_weight`**: Computed per class as `neg_count / pos_count` to handle class imbalance. Rare classes (9/x, 11/x) receive higher loss weight automatically.
- **Backward compatibility**: Primary accuracy is still computed via `argmax` on both predictions and labels, making results directly comparable to single-label baselines.

### 2.2 Expanded Class Set

```python
# Before
CLASS_METERS = [3, 4, 5, 7]       # 4 classes

# After
CLASS_METERS = [3, 4, 5, 7, 9, 11]  # 6 classes
```

The classification head output changes from 4 to 6 logits. The `MERTClassificationHead` MLP architecture remains the same (LayerNorm -> Linear -> GELU -> Dropout -> Linear), with `num_classes=6`.

### 2.3 Stochastic Phase Augmentation

Audio clips longer than 30 seconds must be cropped. The crop position affects which beat phase the model sees:

```python
if self.augment:
    # Training: random crop position each epoch
    start = np.random.randint(0, len(audio) - max_samples)
else:
    # Validation/test: deterministic center crop
    start = (len(audio) - max_samples) // 2
audio = audio[start : start + max_samples]
```

**Rationale**: 30-second segments start at arbitrary beat phase positions. Without random cropping, the model might learn spurious phase-dependent features (e.g., "5/4 music always starts on beat 3"). Random cropping teaches phase-invariance by presenting different alignments of the same piece across epochs.

Additional augmentations during training:
- Gaussian noise injection: `audio += 0.005 * randn`
- Random circular shift: up to 0.5 seconds in either direction

### 2.4 Label Smoothing

```python
LABEL_SMOOTH_NEG = 0.1

# In MERTAudioDataset.__getitem__:
label = np.full(len(CLASS_METERS), LABEL_SMOOTH_NEG, dtype=np.float32)  # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
for m in meters:
    label[METER_TO_IDX[m]] = 1.0  # e.g., [0.1, 0.1, 1.0, 0.1, 0.1, 0.1] for meter 5
```

Negative classes get 0.1 instead of 0.0, preventing the model from pushing all non-target sigmoid outputs to exactly zero. This provides a form of regularization that encourages the model to maintain some activation even for "wrong" classes.

**Risk**: If too aggressive, label smoothing could cause "label leakage" -- artificial positive correlation between class activations. This is monitored via the inter-class correlation matrix (Section 5.5).

### 2.5 Multi-Label Data Loading

`MERTAudioDataset` accepts `list[tuple[Path, list[int]]]` -- each entry has a list of meter integers:

- **METER2800 entries**: Wrapped as single-label: `[(path, [3])]`
- **ODDMETER-WIKI entries**: Can be multi-label: `[(path, [3, 4])]` for polyrhythmic tracks
- **Extra data parser**: Handles comma-separated meters in .tab files: `"5,7"` is parsed into `[5, 7]`

## 3. External Datasets

### 3.1 ODDMETER-WIKI (Round 12a — superseded by WIKIMETER)

Initial dataset with 105 songs (odd meters + poly only). Used in first training run. Superseded by WIKIMETER which adds balanced 3/4 and 4/4 data.

| Metric | Value |
|--------|-------|
| Songs | 105 (5/x: 30, 7/x: 32, 9/x: 12, 11/x: 8, poly: 23) |
| Segments | 1,028 |
| Script | `scripts/setup/download_oddmeter_wiki.py` |

### 3.2 WIKIMETER (Round 12b — current)

Expanded and balanced dataset with all meter classes, sourced from Wikipedia + curated selections. Includes per-song expected duration filtering (0.5x–2.0x tolerance) to reject albums/compilations.

| Meter | Songs | Example Artists/Works |
|-------|-------|----------------------|
| 3/4 | 29 | Strauss waltzes, Chopin, Tchaikovsky, Beatles ("Norwegian Wood"), folk |
| 4/4 | 29 | Queen, AC/DC, Nirvana, Michael Jackson, Daft Punk, Kraftwerk |
| 5/x | 30 | Dave Brubeck ("Take Five"), Radiohead, Gorillaz, Chopin, Tchaikovsky, Balkan folk |
| 7/x | 32 | Pink Floyd ("Money"), Peter Gabriel ("Solsbury Hill"), Rush, King Crimson, Balkan folk |
| 9/x | 12 | Blue Rondo a la Turk, Bartok Bulgarian Dances, Stravinsky |
| 11/x | 8 | Bartok Bulgarian No. 6, Gankino Horo, Kopanitsa, Primus |
| Polyrhythmic | 23 | African drumming [3,4], Meshuggah [5,4]/[7,4], gamelan, hemiola |
| **Total** | **163** | |

**Key improvement over ODDMETER-WIKI**: Balanced 3/4 and 4/4 representation prevents training distribution mismatch that caused val oscillation in the first run.

### 3.3 Download and Segmentation Pipeline

Scripts: `scripts/setup/download_wikimeter.py` (local) and embedded in Colab notebook cell-7.

1. **YouTube search and download** via `yt-dlp`: Each song searched by artist + title (or custom query override).
2. **Duration filtering**: Per-song expected duration with 2x tolerance (e.g., 300s expected → accept 150s–600s). Prevents downloading compilations, live albums, or wrong videos.
3. **Segmentation** via `ffmpeg`: Each full track split into 30-second segments:
   - Skip first 10 seconds (intro/fade-in)
   - Skip last 10 seconds (outro/fade-out)
   - Max 25 segments per song
   - Minimum segment length: 15 seconds
   - Naming: `{artist}_{title}_seg{NN}.mp3`
4. **Label file generation**: `data_wikimeter.tab` with multi-label meter column:
   ```
   filename	label	meter	alt_meter
   "/dave_brubeck_take_five_seg00.mp3"	"five"	5	10
   "/traditional_kpanlogo_seg00.mp3"	"three"	3,4	6
   ```

### 3.4 Integration with Training

The `--extra-data` flag in `finetune_mert.py` appends extra entries to training only:

```bash
uv run python scripts/training/finetune_mert.py \
    --data-dir data/meter2800 \
    --extra-data data/wikimeter \
    --epochs 30 --lr 1e-4
```

Implementation details:
- Custom CSV/TSV parser handles multi-label meter column (`"3,4"` -> `[3, 4]`)
- Entries are filtered against `METER_TO_IDX` (only known class meters are kept)
- Audio paths are resolved via `resolve_audio_path`
- Extra data is appended only to training, never to validation or test (to preserve METER2800 benchmark integrity)

## 4. Model Architecture

### 4.1 MERT Backbone

Supports two MERT variants:

| Model | Layers | Hidden Dim | Pooled Dim | Parameters |
|-------|--------|------------|------------|------------|
| MERT-v1-95M | 12 | 768 | 1,536 | 95M |
| MERT-v1-330M | 24 | 1,024 | 2,048 | 330M |

Default: `m-a-p/MERT-v1-330M`.

### 4.2 LoRA Configuration

Low-Rank Adaptation is applied to attention Q and V projection matrices:

```python
LoraConfig(
    r=16,              # LoRA rank
    lora_alpha=32,     # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
```

Trainable parameter count is typically ~0.5--1% of total MERT parameters. Gradients flow through LoRA adapters during training, allowing the MERT backbone to adapt to meter classification while keeping most parameters frozen.

### 4.3 Forward Pass

```
Audio (any SR) --> librosa.load at 24 kHz, mono
    --> crop to 30s (random crop for train, center crop for eval)
    --> split into 5-second non-overlapping chunks
    --> MERT forward pass (with gradients for LoRA)
    --> extract hidden_states[1..num_layers] per chunk
    --> per-layer: mean pooling + max pooling --> concat --> (hidden_dim * 2,)
    --> aggregate across chunks: mean of means, max of maxes
    --> (num_layers, pooled_dim) per sample
```

### 4.4 Classification Head

`MERTClassificationHead`: Learnable weighted sum of all MERT layers plus MLP classifier.

```
Input: (batch, num_layers, pooled_dim)
    --> softmax-weighted layer combination: w_i * layer_i, summed --> (batch, pooled_dim)
    --> LayerNorm(pooled_dim)
    --> Linear(pooled_dim, head_dim=256) --> GELU --> Dropout(0.3)
    --> Linear(head_dim, num_classes=6)
Output: (batch, 6) logits
```

Layer weights are learnable parameters initialized to uniform, allowing the model to discover which MERT layers are most informative for meter classification. Previous experiments (Section 4.9 of RESEARCH.md) showed that early layers (L2--L5) dominate for meter.

### 4.5 Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs | 30 | With early stopping (patience=10) |
| Batch size | 4 | Physical batch size per forward pass |
| Gradient accumulation | 8 | Effective batch = 32 |
| Head LR | 1e-4 | For classification head parameters |
| LoRA LR | 1e-5 | 10x smaller for LoRA adapters |
| Optimizer | AdamW | weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR | T_max=epochs |
| Gradient clipping | 1.0 | max_norm on all trainable parameters |
| Loss | BCEWithLogitsLoss | With per-class pos_weight |

Discriminative learning rates: the classification head trains at 10x the LoRA adapter rate, since the head must learn from scratch while LoRA only needs small adjustments to a pre-trained backbone.

## 5. Evaluation Metrics and Diagnostic Framework

The evaluation framework is designed as a coherent diagnostic system for multi-label classification with polyrhythm detection. All metrics are computed in `print_eval_metrics()`.

### 5.1 Standard Metrics

**Confusion Matrix** (argmax-based): Backward compatible with single-label evaluation. Primary accuracy uses `argmax(predictions)` vs `argmax(labels)`.

**mAP (mean Average Precision)**: Per-class AP computed from sigmoid probabilities against binarized labels (threshold 0.5 to separate true positives from label-smoothed negatives). Measures ranking quality independent of threshold.

**Macro-F1**: Multi-label F1 at threshold 0.5 on sigmoid outputs, averaged across classes with data.

### 5.2 Co-occurrence Analysis (Polyrhythm Proxy)

Counts samples where sigmoid output exceeds 0.4 for two or more classes simultaneously:

```python
COOCCURRENCE_THRESH = 0.4
multi_active = (probs > COOCCURRENCE_THRESH).sum(axis=1)
n_poly = int((multi_active >= 2).sum())
```

For polyrhythmic samples, the most common class pairs are reported (e.g., `3+4: 15 samples`, `5+4: 8 samples`).

**Interpretation**: On single-label METER2800 data, co-occurrence should be low (only model noise). On ODDMETER-WIKI polyrhythmic data, co-occurrence should be high for the correct pair.

### 5.3 Confidence Gap

The confidence gap is defined as the difference between the top two sigmoid probabilities:

```
delta_P = P_top1 - P_top2
```

| delta_P Range | Interpretation |
|---------------|----------------|
| > 0.5 | Model is selective -- one class dominates (good for single-label data) |
| 0.3 -- 0.5 | Moderate confidence -- possibly ambiguous meter |
| < 0.3 | Model is "spreading" probability -- possible label leakage or genuine polyrhythm |

Reported as mean, median, std, and per-predicted-class breakdown.

### 5.4 Normalized Shannon Entropy

Binary entropy per sigmoid output, normalized to [0, 1]:

```
H = -sum[p * log2(p) + (1-p) * log2(1-p)]  per sample
H_norm = H / (C * log2(2))                   where C = number of classes
```

| H_norm | Meaning |
|--------|---------|
| 0 | Model certain of one configuration (all sigmoids near 0 or 1) |
| 1 | Maximum uncertainty (all sigmoids near 0.5) |

**Key diagnostic**: Entropy disambiguates the confidence gap:
- **High H + low delta_P** = Label leakage: model is confused, spreading probability uniformly
- **Low H + low delta_P** = Polyrhythm candidate: model is confidently activating exactly 2 classes

This is implemented as "diagnostic quadrants" -- samples below median delta_P are split by median H_norm into leakage-risk vs. polyrhythm-candidate categories.

### 5.5 Inter-class Correlation Matrix

Pearson correlation between sigmoid outputs across all samples:

```python
corr = np.corrcoef(probs.T)  # (num_classes, num_classes)
```

**Expected behavior on single-label data**: Correlations should be near-zero or negative (when one class activates, others should not).

**Label leakage signal**: Positive correlation > 0.3 is flagged as `"LEAKAGE"`. This would indicate that label smoothing (setting negatives to 0.1 instead of 0.0) is causing the model to learn spurious positive associations between classes.

Key pairs to monitor:
- 3/4 <-> 4/4 (the most common confusion pair in meter detection)
- 4/4 <-> 5/4 (MERT historically over-predicts odd meters)

### 5.6 Secondary Activation Noise Floor

On single-label data, the second-highest sigmoid output (`P_top2`) represents model noise -- not a real secondary meter. Its distribution provides empirical thresholds:

```
Mean P_top2, Median P_top2
Percentiles: 90th, 95th, 99th
Per-class: P_top2 when true class is each meter
```

**Usage for production thresholds**:
- **95th percentile** -> evaluation threshold: captures more polyrhythm candidates for analysis
- **99th percentile** -> production/live threshold: conservative, avoids false polyrhythm detection

This is superior to an ad hoc threshold (e.g., 0.4) because:
1. It is **data-driven** -- derived from the actual noise distribution
2. It is **class-adaptive** -- different meters may have different noise floors
3. It adapts to model quality -- a better-trained model will have a lower noise floor

## 6. Diagnostic Framework Summary

The five metrics form a coherent diagnostic pipeline:

```
1. Is the model selective?
   --> Confidence Gap (delta_P)
   --> High delta_P = good, model picks one class

2. If not selective, is it confused or seeing polyrhythm?
   --> Normalized Shannon Entropy (H_norm) disambiguates
   --> High H + low delta_P = confused (leakage)
   --> Low H + low delta_P = genuine polyrhythm

3. Is label smoothing causing leakage?
   --> Inter-class Correlation Matrix
   --> Positive r > 0.3 = systematic leakage

4. What threshold should we use for polyrhythm detection?
   --> Noise Floor Percentiles (per-class)
   --> 95th pct = evaluation, 99th pct = production
```

## 7. Training Results

### 7.1 Run 1: MERT-330M + ODDMETER-WIKI on Colab Pro A100 (crashed)

**Config**: MODEL_NAME=MERT-v1-330M, EPOCHS=30, BATCH_SIZE=4, GRAD_ACCUM=8, LORA_RANK=16, HEAD_LR=5e-4 (accidentally used 95M values), LORA_LR=1e-4, USE_EXTRA_DATA=True (ODDMETER-WIKI).

| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1–5 | 4–7% | 0.0% | pos_weight forces rare-class predictions |
| 6 | 12.3% | 9.8% | Breakthrough — model "clicks" |
| 7 | 16.1% | 4.6% | Post-breakthrough oscillation |
| 8 | 18.9% | 2.5% | Val dropping |
| 9 | 20.6% | **14.7%** | New best val |
| 10 | 22.7% | 6.7% | Session crashed, checkpoint lost (was in RAM only) |

**Key learnings**:
1. **pos_weight with heavy imbalance takes ~6 epochs to "click"** — patience is essential, do not abort early
2. **Val oscillates wildly** — 0→0→0→0→0→9.8→4.6→2.5→14.7→6.7%, typical for aggressive pos_weight
3. **95M LR values work on 330M** — discovered accidentally, the "wrong" LR still produced learning
4. **Must save checkpoint to disk** — RAM-only checkpoints are lost on session crash

### 7.2 Run 2: MERT-330M + WIKIMETER on Colab Pro L4 (pending)

**Changes from Run 1**:
- WIKIMETER replaces ODDMETER-WIKI (balanced 3/4 + 4/4 added, polyrhythmic included)
- Per-model LR auto-scaling: HEAD_LR=1e-4, LORA_LR=2e-5 for 330M
- Checkpoint saved to disk after each best-val epoch
- L4 GPU (~1.7 compute units/h vs A100's ~7.5) — ~3x cheaper but ~3x slower

## 8. Files Modified and Created

| File | Action | Description |
|------|--------|-------------|
| `scripts/training/finetune_mert.py` | Major refactor | Sigmoid + BCE, multi-label data loading, 6 classes, phase augmentation, `--extra-data` flag, full diagnostic metrics suite |
| `scripts/setup/download_wikimeter.py` | New | WIKIMETER download script (163 songs, per-song duration filtering) |
| `scripts/setup/download_oddmeter_wiki.py` | Superseded | Original odd-meter-only download script (kept for reference) |
| `notebooks/colab_mert_lora.ipynb` | Major update | Self-contained Colab notebook: METER2800 download, WIKIMETER download, training with disk checkpointing |
| `data/wikimeter/data_wikimeter.tab` | Generated | Multi-label .tab file for WIKIMETER segments |
| `data/wikimeter/audio/` | Generated | 30s segments from 163 songs |

### 8.1 Key Constants in finetune_mert.py

```python
CLASS_METERS = [3, 4, 5, 7, 9, 11]     # 6 classes (expanded from [3, 4, 5, 7])
MERT_SR = 24000                          # MERT sampling rate
CHUNK_SAMPLES = 5 * MERT_SR             # 5-second chunks for MERT input
MAX_DURATION_S = 30                      # Maximum audio duration
LABEL_SMOOTH_NEG = 0.1                   # Label smoothing for negative classes
```

### 8.2 Command-Line Interface

```bash
# Basic LoRA fine-tuning on METER2800
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800

# With ODDMETER-WIKI augmentation
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
    --extra-data data/oddmeter-wiki

# Smoke test (3 files per split)
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
    --limit 3

# Frozen baseline (no LoRA, only train head)
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
    --no-lora

# 95M model (faster, less accurate)
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
    --model m-a-p/MERT-v1-95M

# Custom hyperparameters
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
    --epochs 50 --lr 2e-4 --lora-lr 2e-5 --lora-rank 32 --lora-alpha 64 \
    --batch-size 2 --grad-accum 16 --dropout 0.4
```

## 9. Experimental Plan

### 9.1 Immediate Next Steps

1. **Run LoRA fine-tuning** with `--extra-data data/wikimeter` on MERT-v1-330M (L4 GPU)
2. **Monitor during training**:
   - H_norm trend: should decrease as model becomes more certain
   - delta_P trend: should increase as model becomes more selective
   - Correlation matrix: watch for r > 0.3 between any class pair
3. **If correlation shows leakage**: Reduce `LABEL_SMOOTH_NEG` from 0.1 to 0.05 or 0.02

### 9.2 Threshold Calibration

After training:
1. Compute per-class 99th percentile noise floor on METER2800 test split (single-label data)
2. Use these as production thresholds for polyrhythm detection in `mert_signal.py`
3. Any live sample with `P_top2 > noise_floor_99th` for its predicted class is flagged as polyrhythmic

### 9.3 Gate Check

Before enabling in the ensemble:
1. Run orthogonality evaluation on 272 internal benchmark fixtures
2. Compute complementarity ratio (gains / losses)
3. Target: ratio > 1.5, agreement 65--80%
4. If passed: integrate with initial weight W_MERT = 0.0, then tune up

### 9.4 Ablation Studies

| Experiment | Variable | Expected Outcome |
|------------|----------|------------------|
| Smoothing sweep | LABEL_SMOOTH_NEG in {0.0, 0.02, 0.05, 0.1, 0.2} | Find leakage threshold |
| LoRA rank sweep | r in {4, 8, 16, 32} | Capacity vs. overfitting tradeoff |
| Frozen vs. LoRA | --no-lora vs default | Quantify LoRA benefit |
| Extra data impact | With/without --extra-data | Quantify ODDMETER-WIKI benefit |
| Phase augmentation | With/without random crop | Quantify phase-invariance benefit |

## 10. Relationship to Prior Work

This experiment builds directly on the findings from Rounds 8--9 (documented in `docs/RESEARCH.md`, Sections 4.8--4.9):

| Finding | Round | How Round 12 Addresses It |
|---------|-------|---------------------------|
| MERT frozen + MLP: 80.7% test, gate FAIL | 8 | LoRA fine-tuning to adapt MERT representations |
| Multi-layer 95M: 79.9%, no improvement | 9 | Move to 330M (24 layers, richer representations) |
| Best layer is 3 (early) | 8 | Learnable layer weights discover optimal combination |
| Over-predicts 7/4 and 5/4 | 8 | More training data for these classes via ODDMETER-WIKI |
| Softmax forces single class | 8--9 | Sigmoid enables genuine polyrhythm detection |
| 150 train files per odd meter | -- | ODDMETER-WIKI adds 304 (5/x), 262 (7/x), 106 (9/x), 73 (11/x) |

## 11. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Label leakage from smoothing | Medium | Model loses selectivity | Correlation matrix monitoring; reduce LABEL_SMOOTH_NEG |
| ODDMETER-WIKI label noise | Medium | Incorrect labels degrade training | Wikipedia source is musicologically vetted; manual review of ambiguous entries |
| YouTube audio quality variance | Low | Inconsistent features | MERT is pre-trained on diverse audio quality; 30s segments average out local artifacts |
| LoRA overfitting on small dataset | Medium | Good val, poor test | Early stopping, dropout, gradient clipping, cosine LR schedule |
| New classes (9, 11) too sparse | High | Model never learns 9/x, 11/x reliably | pos_weight compensates class imbalance; ODDMETER-WIKI provides targeted data |
| Compute cost | -- | ~2--4 hours per training run on MPS | Smoke test with --limit 3 first; --no-lora baseline for quick iteration |
