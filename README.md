# Beatmeter

Audio tempo and time signature analyzer. Detects BPM, meter (time signature), and rhythm patterns from audio files or live microphone input.

## Features

- **Multi-algorithm beat tracking** — combines BeatNet, Beat This! (ISMIR 2024), madmom, and librosa for robust beat detection
- **Meter detection** — identifies time signatures (4/4, 3/4, 6/8, 7/8, etc.) with confidence scores and disambiguation hints
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

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests (synthetic — no audio fixtures needed)
uv run pytest tests/test_engine.py tests/test_meter.py -v

# Run full benchmark (requires fixture files)
python scripts/download_new_fixtures.py
uv run python tests/benchmark.py
```

## License

MIT
