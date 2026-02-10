"""librosa fallback beat tracking."""

import numpy as np

from beatmeter.analysis.models import Beat


def track_beats_librosa(audio: np.ndarray, sr: int = 22050) -> list[Beat]:
    """Fallback beat tracking using librosa.

    No downbeat detection - all beats marked as non-downbeat.
    """
    import librosa

    _tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    beats = []
    for t in beat_times:
        beats.append(Beat(time=float(t), is_downbeat=False, strength=1.0))
    return beats
