"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings with env var overrides."""

    # Audio
    sample_rate: int = 22050
    mono: bool = True

    # Live streaming
    stream_buffer_seconds: float = 60.0
    warmup_seconds: float = 8.0
    reanalysis_interval: float = 2.0
    sliding_window_seconds: float = 30.0
    chunk_duration_ms: int = 100  # ms per WebSocket chunk

    # Analysis
    min_bpm: float = 40.0
    max_bpm: float = 300.0
    max_meter_hypotheses: int = 5
    onset_threshold: float = 0.3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 50

    model_config = {"env_prefix": "BEATMETER_"}


settings = Settings()
