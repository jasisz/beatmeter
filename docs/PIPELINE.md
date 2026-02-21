# Training & Evaluation Pipeline

## Full pipeline (from scratch)

### 1. Warm engine cache
Fills signal caches (beatnet, beat_this, autocorr, bar_tracking, hcdf) and
MeterNet feature caches (`meter_net_audio`, `meter_net_ssm`) for all training files.
```bash
uv run python scripts/setup/warm.py --split tuning --extra-data -w 4
```

### 2. Extract MeterNet dataset
Runs the engine on every file and dumps features to `data/meter_net_dataset/`.
Must be re-run after warm_cache or any signal/feature code change.
```bash
uv run python scripts/training/train.py --extract --split tuning --extra-data --workers 4
```

### 3. Grid search (hyperparameter selection)
Tries combinations of hyperparameters, saves each checkpoint.
Resumes automatically if interrupted. Uses val_acc (NOT test set) to pick winner.
```bash
uv run python scripts/training/grid.py
```
Useful flags:
```bash
--dry-run   # show what would run, don't execute
--reset     # clear previous results and re-run all
--summary   # print results table and exit
```

### 4. Eval on test split
Run once after grid search picks the best checkpoint (already copied to `meter_net.pt`).
```bash
uv run python scripts/eval.py --split test --limit 0
```
If result is better than baseline, save it:
```bash
uv run python scripts/eval.py --split test --limit 0 --save
```

---

## Quick iteration (warm cache already done)

```bash
# re-extract if signal/feature code changed, otherwise skip
uv run python scripts/training/train.py --extract --split tuning --extra-data --workers 4

# grid search
uv run python scripts/training/grid.py

# smoke test first
uv run python scripts/eval.py --limit 3 --workers 1

# full eval
uv run python scripts/eval.py --split test --limit 0
```

## Retraining onset_mlp only

```bash
uv run python scripts/training/train_onset_mlp.py
# then re-run from step 2 (extract) — onset_mlp cache will auto-invalidate
```

## Cache architecture

The pipeline uses a multi-level LMDB cache (`.cache/analysis.lmdb`) to avoid
redundant computation. Each key embeds a hash of its source code dependencies,
providing automatic invalidation when code changes.

### Cache levels

| Key prefix | Format | What's cached | Depends on |
|------------|--------|---------------|------------|
| `beats:{tracker}:{hash}:{ah}` | JSON | Beat positions per tracker | tracker source file |
| `onsets:{hash}:{ah}` | JSON | Onset times/strengths | `onset.py` |
| `signals:{name}:{hash}:{ah}` | JSON | Per-signal meter scores | signal source file(s) |
| `features:{group}:{hash}:{ah}` | float32 bytes | MeterNet feature vectors | feature extraction code only |

### Granular feature caching (MeterNet)

MeterNet's input features are split into groups cached **independently from
the model checkpoint**:

| Group | Dims | Source | Cost |
|-------|------|--------|------|
| `meter_net_audio` | 1449 | `onset_mlp_features.py` | ~5s (autocorrelation, MFCC, tempogram) |
| `meter_net_ssm` | 75 | `ssm_features.py` | ~2s (self-similarity matrix features) |
| Beat features | 42 | live from cached beats | <1ms |
| Signal scores | 60 | live from cached signals | <1ms |
| Tempo features | 4 | live from cached tempo | <1ms |

Because the expensive audio/SSM features are keyed on their extraction code
(not on the checkpoint), retraining the model only requires re-running the
cheap forward pass (~10ms per file). This means a full 700-file eval after
model retraining takes ~10 seconds instead of ~10 minutes.

### Fast path

When all cache levels are warm (`_is_cache_warm` in `engine.py`), the engine
skips audio file decoding entirely — beats, onsets, and MeterNet features are
all loaded from LMDB. The fast path checks:
- All beat trackers cached (beatnet, beat_this, librosa, madmom ×4)
- Onsets cached
- `meter_net_audio` and `meter_net_ssm` feature arrays cached

MeterNet scores are NOT checked for the fast path — they depend on the
checkpoint and are expected to be recomputed after retraining.

## Notes

- **Always smoke-test** (`--limit 3`) before a full eval after any code change
- **Never use test split for hyperparameter selection** — val_acc from training is enough
- **Extract must happen after warm_cache** — otherwise MeterNet dataset is stale
- Grid search saves all checkpoints to `data/meter_net_grid/` and copies the best to `data/meter_net.pt`
