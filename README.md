# Beatmeter

Audio tempo and time signature analyzer. Detects BPM, meter (time signature), and rhythm patterns from audio files or live microphone input.

## Features

- **Multi-algorithm beat tracking** — combines BeatNet, Beat This! (ISMIR 2024), madmom, and librosa for robust beat detection
- **Meter detection** — identifies time signatures (4/4, 3/4, 6/8, 7/8, etc.) using 7 independent signals with weighted voting
- **Variable tempo detection** — tracks tempo changes over time, classifies stability (steady / rubato)
- **Live analysis** — real-time rhythm analysis via microphone (WebSocket)
- **Web UI** — drag-and-drop file upload, waveform visualization, metronome, i18n (PL/EN)

## Installation

Requires Python 3.12+.

```bash
# Clone the repository
git clone https://github.com/jasisz/beatmeter.git
cd beatmeter

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Start the server

```bash
uv run beatmeter
# or
uv run python -m beatmeter.main
```

Open http://localhost:8000 in your browser.

### API

```bash
# Upload a file for analysis
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@song.mp3"

# Health check
curl http://localhost:8000/api/health
```

### Configuration

Environment variables (prefix `BEATMETER_`):

| Variable | Default | Description |
|---|---|---|
| `BEATMETER_HOST` | `0.0.0.0` | Server bind address |
| `BEATMETER_PORT` | `8000` | Server port |
| `BEATMETER_SAMPLE_RATE` | `22050` | Audio sample rate |
| `BEATMETER_MAX_UPLOAD_MB` | `50` | Max upload size |

## Architecture

### Beat Tracking

Four beat trackers run in parallel (`beatmeter/analysis/trackers/`):

| Tracker | Source |
|---|---|
| BeatNet | Neural network beat/downbeat tracking |
| Beat This! | ISMIR 2024 transformer model |
| madmom | DBNBeatTrackingProcessor |
| librosa | Onset-based beat tracking |

### Meter Detection

Seven active signals vote on meter hypotheses (`beatmeter/analysis/signals/`):

| Signal | Weight | Description |
|---|---|---|
| `downbeat_spacing` | 0.13–0.16 | Downbeat interval patterns from BeatNet / Beat This! |
| `madmom_activation` | 0.10 | Beat activation scoring from madmom |
| `onset_autocorrelation` | 0.13 | Periodicity in onset envelope |
| `accent_pattern` | 0.18 | RMS energy accent grouping |
| `beat_periodicity` | 0.20 | Beat strength periodicity analysis |
| `bar_tracking` | 0.12 | DBNBarTrackingProcessor bar-level inference |
| `sub_beat_division` | — | Compound meter detection (6/8, 12/8) |

Trust weighting adjusts neural network signals based on beat tracker agreement. Consensus bonuses reward convergence across multiple signals.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run unit tests
uv run pytest tests/test_meter.py -v
```

### Evaluation (METER2800)

The primary benchmark uses the [METER2800](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WQSKRO) dataset (2800 files, 4 meter classes: 3/4, 4/4, 5/4, 7/4).

```bash
# Download METER2800 dataset
uv run python scripts/setup/download_meter2800.py

# Smoke test (3 files)
uv run python scripts/eval.py --limit 3 --workers 1

# Stratified quick run (~100 files, ~20 min)
uv run python scripts/eval.py --quick

# Full test split (700 files, hold-out)
uv run python scripts/eval.py --split test --limit 0

# Save baseline for regression detection
uv run python scripts/eval.py --save

# Dashboard
uv run python scripts/dashboard.py
```

Current accuracy on the METER2800 test split: **76% overall** (88% on binary 3/4 vs 4/4).

### WIKIMETER Catalog Workflow

Use this when you want to expand `scripts/setup/wikimeter.json` in a controlled way.
Catalog entries support multi-source `sources[]`.

```bash
# 1) Prepare a reviewed queue JSON (manual curation)
#    default path: data/wikimeter/review_queue.json

# 2) Merge approved entries into catalog (gate step)
uv run python scripts/setup/merge_wikimeter_reviewed.py \
  --review-queue data/wikimeter/review_queue.json

# 3) Download/refresh dataset from catalog
uv run python scripts/setup/download_wikimeter.py
```

Review queue entries should include:
- `seed.artist`
- `seed.title`
- `seed.meters` (example: `5` or `3:0.8,4:0.7`)
- `status` (`approved` to merge)
- source info in `candidate` (`source`, `sources`, `video_id`/`video_url` legacy)

Blacklist gate:
- `data/wikimeter/blacklist.json` blocks known bad song+meter pairs during merge.

## Project Structure

```
beatmeter/
├── analysis/
│   ├── engine.py              # Orchestrator (onset → beat → tempo → meter)
│   ├── meter.py               # Hypothesis generation + weighted voting
│   ├── tempo.py               # Multi-method tempo estimation
│   ├── onset.py               # Onset detection
│   ├── models.py              # Data models (Beat, MeterHypothesis, etc.)
│   ├── cache.py               # Analysis cache
│   ├── signals/               # Meter signal implementations
│   └── trackers/              # Beat tracker wrappers
├── api/                       # FastAPI endpoints
├── audio/                     # Audio I/O utilities
└── config.py                  # Settings
frontend/
├── index.html
├── js/                        # Visualizer, i18n, audio capture
└── css/
scripts/
├── eval.py                    # Unified evaluation (METER2800)
├── dashboard.py               # Metrics dashboard
├── utils.py                   # Shared utilities
├── setup/                     # Dataset download scripts
└── training/                  # Model training scripts
tests/
└── test_meter.py              # Unit tests for meter detection
docs/
└── RESEARCH.md                # Research documentation
```

## License

MIT
