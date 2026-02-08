"""Ring-buffer for live audio streaming."""

from __future__ import annotations

import numpy as np

_DEFAULT_SR = 22050
_MAX_DURATION_SECONDS = 60


class StreamBuffer:
    """Fixed-capacity ring buffer that stores up to 60 seconds of audio.

    Parameters
    ----------
    sr:
        Sample rate in Hz. Defaults to 22050.
    max_duration:
        Maximum buffer duration in seconds. Defaults to 60.
    """

    def __init__(self, sr: int = _DEFAULT_SR, max_duration: float = _MAX_DURATION_SECONDS) -> None:
        self._sr = sr
        self._max_samples = int(sr * max_duration)
        self._buffer = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos = 0
        self._length = 0  # how many valid samples are in the buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, chunk: np.ndarray) -> None:
        """Append an audio chunk to the buffer.

        If the chunk is larger than the buffer capacity, only the last
        ``max_samples`` samples are kept.
        """
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        n = len(chunk)

        if n == 0:
            return

        if n >= self._max_samples:
            # Only keep the tail that fits.
            chunk = chunk[-self._max_samples:]
            n = self._max_samples
            self._buffer[:] = chunk
            self._write_pos = 0
            self._length = self._max_samples
            return

        end = self._write_pos + n
        if end <= self._max_samples:
            self._buffer[self._write_pos:end] = chunk
        else:
            first = self._max_samples - self._write_pos
            self._buffer[self._write_pos:] = chunk[:first]
            self._buffer[:n - first] = chunk[first:]

        self._write_pos = end % self._max_samples
        self._length = min(self._length + n, self._max_samples)

    def get_audio(self, last_n_seconds: float | None = None) -> np.ndarray:
        """Return buffered audio samples.

        Parameters
        ----------
        last_n_seconds:
            If provided, return only the most recent *N* seconds of audio.
            If ``None``, return all buffered audio.
        """
        if self._length == 0:
            return np.zeros(0, dtype=np.float32)

        if last_n_seconds is not None:
            n_samples = min(int(self._sr * last_n_seconds), self._length)
        else:
            n_samples = self._length

        start = (self._write_pos - n_samples) % self._max_samples

        if start + n_samples <= self._max_samples:
            return self._buffer[start:start + n_samples].copy()

        first = self._max_samples - start
        return np.concatenate([
            self._buffer[start:],
            self._buffer[:n_samples - first],
        ])

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        return self._length / self._sr

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer[:] = 0
        self._write_pos = 0
        self._length = 0
