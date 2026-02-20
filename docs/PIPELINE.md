# Training & Evaluation Pipeline

## Full pipeline (from scratch)

### 1. Warm engine cache
Fills signal caches (beatnet, beat_this, autocorr, onset_mlp, ...) for all training files.
```bash
uv run python scripts/setup/warm.py --split tuning --extra-data -w 4
```

### 2. Extract arbiter dataset
Runs the engine on every file and dumps signal scores to `data/arbiter_dataset/`.
Must be re-run after warm_cache or any signal code change.
```bash
uv run python scripts/training/train_arbiter.py --extract --split tuning --extra-data --workers 4
```

### 3. Grid search (hyperparameter selection)
Tries all combinations of `sharpen` × `boost_rare`, saves each checkpoint.
Resumes automatically if interrupted. Uses val_acc (NOT test set) to pick winner.
```bash
uv run python scripts/training/grid_arbiter.py
```
Useful flags:
```bash
--dry-run   # show what would run, don't execute
--reset     # clear previous results and re-run all
--summary   # print results table and exit
```

### 4. Eval on test split
Run once after grid search picks the best checkpoint (already copied to `meter_arbiter.pt`).
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
# re-extract if signal code changed, otherwise skip
uv run python scripts/training/train_arbiter.py --extract --split tuning --extra-data --workers 4

# grid search
uv run python scripts/training/grid_arbiter.py

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

## Notes

- **Always smoke-test** (`--limit 3`) before a full eval after any code change
- **Never use test split for hyperparameter selection** — val_acc from training is enough
- **Extract must happen after warm_cache** — otherwise arbiter_dataset is stale
- Grid search saves all checkpoints to `data/arbiter_grid/` and copies the best to `data/meter_arbiter.pt`
