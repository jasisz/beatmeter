# MERT LoRA Multi-Label Experiment (Round 12)

## Abstract

Round 12 of the BeatMeter project transitions MERT fine-tuning from single-label (softmax) to multi-label (sigmoid) architecture to enable polyrhythm detection. The class set is expanded from 4 classes [3, 4, 5, 7] to 6 classes [3, 4, 5, 7, 9, 11]. Two external datasets were curated from YouTube: WIKIMETER (Round 12a, odd meters only) and WIKIMETER (Round 12b, all meters including 3/4, 4/4, and polyrhythmic). The current notebook-aligned evaluation reports per-class accuracy/AP, macro-F1, and confidence gap. First training run (330M + WIKIMETER, A100) reached val 14.7% at epoch 9 before session crash; second run (WIKIMETER, L4 GPU) pending.

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

### 1.3 Why External Data? (WIKIMETER)

METER2800 has approximately 150 training files per odd meter class (5/4, 7/4), and zero files for 9/x or 11/x. LoRA fine-tuning requires more data for underrepresented classes. Wikipedia maintains a curated [list of musical works in unusual time signatures](https://en.wikipedia.org/wiki/List_of_musical_works_in_unusual_time_signatures) -- a musicologically vetted resource covering diverse genres and cultures.

The initial dataset (WIKIMETER, Round 12a) contained only odd meters (5/x, 7/x, 9/x, 11/x + poly), which created a training distribution mismatch — the model saw disproportionately more odd-meter data from YouTube vs METER2800. The expanded dataset (WIKIMETER, Round 12b) adds curated 3/4 and 4/4 songs to balance the distribution.

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
- Gaussian noise injection: `audio += 0.01 * randn` (default, configurable via `--noise-std`)
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

**Risk**: If too aggressive, label smoothing can reduce class selectivity. In the current workflow this is monitored indirectly via per-class AP, Macro-F1, and confidence gap trends.

### 2.5 Multi-Label Data Loading

`MERTAudioDataset` accepts `list[tuple[Path, list[int]]]` -- each entry has a list of meter integers:

- **METER2800 entries**: Wrapped as single-label: `[(path, [3])]`
- **WIKIMETER entries**: Can be multi-label: `[(path, [3, 4])]` for polyrhythmic tracks
- **Extra data parser**: Handles comma-separated meters in .tab files: `"5,7"` is parsed into `[5, 7]`

## 3. External Datasets

### 3.1 WIKIMETER

Curated dataset with all meter classes, sourced from Wikipedia-verified time signatures. Song catalog lives in `scripts/setup/wikimeter.json` (single source of truth, committed to repo).

| Meter | Songs | Example Artists/Works |
|-------|-------|----------------------|
| 3/4 | 49 | Strauss waltzes, Chopin, Tchaikovsky, Beatles ("Norwegian Wood"), folk |
| 4/4 | 61 | Queen, AC/DC, Nirvana, Michael Jackson, Daft Punk, Kraftwerk |
| 5/x | 32 | Dave Brubeck ("Take Five"), Radiohead, Holst ("Mars"), Muse, Halloween theme |
| 7/x | 29 | Pink Floyd ("Money"), Peter Gabriel ("Solsbury Hill"), King Crimson, Balkan folk |
| 9/x | 25 | Blue Rondo a la Turk, Greek zeimbekiko, Irish slip jigs, Turkish karsilama |
| 11/x | 29 | Gankino Horo, Kopanitsa variants, Brubeck ("Eleven Four"), Primus, Tool |
| Polyrhythmic | 25 | African drumming [3,4], Meshuggah [5,4]/[7,4], gamelan, hemiola |
| **Total** | **250** | |

**History**: Round 12a used an odd-meter-only subset (5/7/9/11 + poly). This caused training distribution mismatch. Round 12b added balanced 3/4 and 4/4 data. Current catalog includes all meters (250 songs).

### 3.2 Download and Segmentation Pipeline

Scripts: `scripts/setup/download_wikimeter.py` (reads `wikimeter.json`) and embedded in Colab notebook cell-7.

1. **YouTube search and download** via `yt-dlp`: Each song searched by artist + title (or custom query override).
2. **Segmentation** via `ffmpeg`: Each full track split into short excerpts:
   - Skip first 10 seconds (intro/fade-in)
   - Skip last 10 seconds (outro/fade-out)
   - Max 5 segments per song
   - Default segment length: 35 seconds
   - Minimum segment length: 15 seconds
   - Naming: `{artist}_{title}_seg{NN}.mp3`
3. **Label file generation**: `data_wikimeter.tab` with multi-label meter column:
   ```
   filename	label	meter	alt_meter
   "/dave_brubeck_take_five_seg00.mp3"	"five"	5	10
   "/traditional_kpanlogo_seg00.mp3"	"three"	3,4	6
   ```

### 3.4 Integration with Training

The `--extra-data` flag in `finetune_mert.py` appends extra entries and performs a per-song stratified split into train/val:

```bash
uv run python scripts/training/finetune_mert.py \
    --data-dir data/meter2800 \
    --extra-data data/wikimeter
```

Implementation details:
- Custom CSV/TSV parser handles multi-label meter column (`"3,4"` -> `[3, 4]`)
- Entries are filtered against `METER_TO_IDX` (only known class meters are kept)
- Audio paths are resolved via `resolve_audio_path`
- Extra song groups are split into train/val by `--extra-val-ratio` (default 0.1), preventing segment leakage between splits
- METER2800 test split remains untouched (benchmark integrity preserved)

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
    --> Linear(pooled_dim, head_dim=256) --> GELU --> Dropout(0.4)
    --> Linear(head_dim, num_classes=6)
Output: (batch, 6) logits
```

Layer weights are learnable parameters initialized to uniform, allowing the model to discover which MERT layers are most informative for meter classification. Previous experiments (Section 4.9 of RESEARCH.md) showed that early layers (L2--L5) dominate for meter.

### 4.5 Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs | 80 | Early stopping enabled (`patience=15`) |
| Batch size | 4 (330M), 8 (95M) | Physical batch size per forward pass |
| Gradient accumulation | 8 (330M), 4 (95M) | Effective batch = 32 |
| Head LR | 5e-4 (330M), 1e-3 (95M) | Auto-selected per model unless overridden |
| LoRA LR | 5e-5 (330M/95M) | Separate LR for LoRA adapters |
| Optimizer | AdamW | weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau | mode=max, factor=0.5, patience=5, min_lr=1e-6 |
| Gradient clipping | 1.0 | max_norm on all trainable parameters |
| Loss | BCEWithLogitsLoss | With per-class pos_weight |

Notebook behavior is mirrored in `scripts/training/finetune_mert.py`:

- Automatic LR defaults per model (`--lr`, `--lora-lr`) with manual override support
- Per-song train/val split for `--extra-data` to avoid segment leakage
- Checkpoint saved every epoch (`head_state_dict`, optional `lora_state_dict`, optimizer/scheduler state)
- Auto-resume from `--checkpoint` (or explicit `--resume`)
- Test evaluation loads the latest saved checkpoint and writes back `test_accuracy`

## 5. Evaluation (Notebook-Aligned)

Current `print_eval_metrics()` output is intentionally compact and follows the notebook:

1. Per-class primary accuracy (argmax) and AP (Average Precision)
2. Overall primary accuracy and mAP summary
3. Macro-F1 (thresholded multi-label output, classes with positives only)
4. Confidence gap (`top1_prob - top2_prob`, mean + median)

This replaced earlier extended diagnostics (entropy/correlation/noise-floor reporting) to keep CLI output consistent with the notebook.

## 6. Run Status

### 6.1 Run 1 (A100, crashed)

- Configuration: 330M, LoRA rank 16 / alpha 32, BCE + `pos_weight`, WIKIMETER augmentation
- Best observed validation accuracy: **14.7% at epoch 9**
- Session crashed before completion

### 6.2 Current baseline for ongoing runs

- 330M defaults: batch 4, grad_accum 8, head LR 5e-4, LoRA LR 5e-5
- Early stopping: patience 15
- Scheduler: ReduceLROnPlateau
- Epoch checkpoints: enabled by default (crash-safe)

## 7. Files and Artifacts

| File | Status |
|------|--------|
| `notebooks/colab_mert_lora.ipynb` | Current reference workflow |
| `scripts/training/finetune_mert.py` | Synced to notebook training loop + metrics + checkpoint flow |
| `scripts/training/check_mert_orthogonality.py` | Supports both legacy embedding checkpoints and finetuned notebook checkpoints |
| `scripts/setup/wikimeter.json` | Catalog source of truth (250 songs) |
| `scripts/setup/download_wikimeter.py` | Segmentation pipeline (max 5 segments, 35s default) |

## 8. Quick Commands

```bash
# Notebook-aligned default training (330M)
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800

# Add WIKIMETER data
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
  --extra-data data/wikimeter

# Resume explicitly
uv run python scripts/training/finetune_mert.py --data-dir data/meter2800 \
  --resume data/meter_mert_finetuned.pt

# Orthogonality check with finetuned checkpoint
uv run python scripts/training/check_mert_orthogonality.py \
  --checkpoint data/meter_mert_finetuned.pt
```

## 9. Open Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Label noise in WIKIMETER | Slower convergence / noisy validation | Keep per-song split, inspect high-loss files |
| Sparse 9/x and 11/x effective coverage | Weak class-wise generalization | Continue targeted data expansion and monitor AP per class |
| Overfitting on augmented data | Good val, poor test transfer | Early stopping + dropout + checkpointed validation tracking |
