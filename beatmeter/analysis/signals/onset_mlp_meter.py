"""Signal 9: Onset MLP meter classification from multi-tempo autocorrelation features.

Provides an orthogonal meter signal that classifies time signature from onset,
RMS, spectral flux, and chroma autocorrelation features at multiple tempos,
combined with tempogram, MFCC, spectral contrast, and onset rate statistics.

The model is trained on METER2800 + WIKIMETER (6 classes: 3, 4, 5, 7, 9, 11)
and loaded lazily on first use. If the model file is not found, the signal
returns an empty dict (graceful degradation — zero weight in ensemble).
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — populated lazily by _load_model()
# ---------------------------------------------------------------------------
_model = None
_device = None
_idx_to_meter: dict[int, int] | None = None
_feat_mean: np.ndarray | None = None
_feat_std: np.ndarray | None = None
_input_dim: int | None = None
_n_classes: int | None = None
_lock = threading.Lock()

# Search paths for the model checkpoint (checked in priority order)
MODEL_PATHS = [
    Path("data/meter_onset_mlp.pt"),  # local development
    Path.home() / ".beatmeter" / "meter_onset_mlp.pt",  # user install
]

# ---------------------------------------------------------------------------
# Feature extraction constants (must match train_onset_mlp.py v4)
# ---------------------------------------------------------------------------
SR = 22050
HOP_LENGTH = 512
MAX_DURATION_S = 30
N_BEAT_FEATURES = 64
BEAT_RANGE = (0.5, 16.0)
WINDOW_PCT = 0.05
N_TEMPO_CANDIDATES = 3
N_SIGNALS = 4
N_TEMPOGRAM_BINS = 64
N_MFCC = 13
N_CONTRAST_BANDS = 6


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model() -> bool:
    """Load the Onset MLP meter classifier from disk (singleton, thread-safe).

    Returns True if the model is ready, False if no checkpoint was found.
    """
    global _model, _device, _idx_to_meter, _feat_mean, _feat_std
    global _input_dim, _n_classes

    if _model is not None:
        return True

    with _lock:
        if _model is not None:
            return True
        return _load_model_locked()


def _load_model_locked() -> bool:
    """Inner model loading (called under lock)."""
    global _model, _device, _idx_to_meter, _feat_mean, _feat_std
    global _input_dim, _n_classes

    try:
        import torch
        import torch.nn as nn

        # Find checkpoint file
        model_path: Path | None = None
        for candidate in MODEL_PATHS:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            logger.debug(
                "Onset MLP model not found at any of: %s",
                [str(p) for p in MODEL_PATHS],
            )
            return False

        # Pick device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        input_dim = checkpoint["input_dim"]
        n_classes = checkpoint["n_classes"]
        meter_to_idx = checkpoint["meter_to_idx"]
        feat_mean = checkpoint.get("feat_mean")
        feat_std = checkpoint.get("feat_std")

        # Reconstruct model (must match OnsetMLP from train_onset_mlp.py)
        hidden = 512
        model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 4, n_classes),
        )

        # The checkpoint stores state for OnsetMLP.net (nn.Sequential)
        # Map keys from "net.0.weight" → "0.weight" etc.
        state_dict = checkpoint["model_state"]
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                stripped[k[4:]] = v
            else:
                stripped[k] = v
        model.load_state_dict(stripped)
        model.to(device)
        model.eval()

        # Store singletons
        _model = model
        _device = device
        _idx_to_meter = {v: k for k, v in meter_to_idx.items()}
        _feat_mean = feat_mean
        _feat_std = feat_std
        _input_dim = input_dim
        _n_classes = n_classes

        logger.info(
            "Loaded Onset MLP model from %s (device=%s, dim=%d, classes=%d)",
            model_path, device, input_dim, n_classes,
        )
        return True

    except Exception as e:
        logger.warning("Failed to load Onset MLP model: %s", e)
        return False


# ---------------------------------------------------------------------------
# Feature extraction (mirrors train_onset_mlp.py v4 exactly)
# ---------------------------------------------------------------------------

def _normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    if len(signal) < 10:
        return np.zeros(1)
    signal = signal - signal.mean()
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return np.zeros(len(signal))
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    return autocorr


def _sample_autocorr_at_tempo(
    autocorr: np.ndarray, tempo_bpm: float, sr: int = SR
) -> np.ndarray:
    beat_period_frames = (60.0 / tempo_bpm) * (sr / HOP_LENGTH)
    beat_multiples = np.linspace(BEAT_RANGE[0], BEAT_RANGE[1], N_BEAT_FEATURES)
    features = np.zeros(N_BEAT_FEATURES)
    for i, k in enumerate(beat_multiples):
        lag = int(k * beat_period_frames)
        if 0 < lag < len(autocorr):
            window = max(1, int(lag * WINDOW_PCT))
            start = max(0, lag - window)
            end = min(len(autocorr), lag + window + 1)
            features[i] = float(np.max(autocorr[start:end]))
    return features


def _extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extract v4 features from audio array (876 dims).

    Same pipeline as train_onset_mlp.extract_features but takes array instead of path.
    """
    try:
        import librosa

        if len(audio) < sr * 2:
            return None

        # Trim to MAX_DURATION_S from center
        max_samples = sr * MAX_DURATION_S
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            audio = audio[start : start + max_samples]

        # Resample if needed
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            sr = SR

        # Part 1: autocorrelation features (768 dims)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH)
        ac_onset = _normalized_autocorrelation(onset_env)

        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
        ac_rms = _normalized_autocorrelation(rms)

        S = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        ac_flux = _normalized_autocorrelation(flux)

        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH)
        chroma_energy = np.sum(chroma, axis=0)
        ac_chroma = _normalized_autocorrelation(chroma_energy)

        autocorrs = [ac_onset, ac_rms, ac_flux, ac_chroma]
        if any(len(ac) < 10 for ac in autocorrs):
            return None

        # Estimate tempos: T, T/2, T×2
        tempo = librosa.feature.tempo(y=audio, sr=sr, hop_length=HOP_LENGTH)
        t = float(tempo[0]) if len(tempo) > 0 else 120.0
        if t < 30 or t > 300:
            t = 120.0
        tempos = [t, max(30.0, t / 2), min(300.0, t * 2)]

        parts = []
        for tp in tempos:
            for ac in autocorrs:
                parts.append(_sample_autocorr_at_tempo(ac, tp, sr))

        # Part 2: tempo-independent features (108 dims)
        # Tempogram profile (64 dims)
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        avg_tg = tg.mean(axis=1)
        bpm_bins = np.logspace(np.log10(30), np.log10(300), N_TEMPOGRAM_BINS)
        profile = np.zeros(N_TEMPOGRAM_BINS)
        for i, bpm in enumerate(bpm_bins):
            lag = int((60.0 / bpm) * (sr / HOP_LENGTH))
            if 0 < lag < len(avg_tg):
                w = max(1, lag // 20)
                s = max(0, lag - w)
                e = min(len(avg_tg), lag + w + 1)
                profile[i] = float(np.max(avg_tg[s:e]))
        pmax = profile.max()
        if pmax > 1e-10:
            profile /= pmax
        parts.append(profile)

        # MFCC stats (26 dims)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        parts.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))

        # Spectral contrast stats (14 dims)
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, hop_length=HOP_LENGTH, n_bands=N_CONTRAST_BANDS,
        )
        parts.append(np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)]))

        # Onset rate stats (4 dims)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=HOP_LENGTH)
        if len(onset_frames) < 3:
            parts.append(np.zeros(4))
        else:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
            intervals = np.diff(onset_times)
            if len(intervals) == 0:
                parts.append(np.zeros(4))
            else:
                duration = len(audio) / sr
                parts.append(np.array([
                    float(np.mean(intervals)),
                    float(np.std(intervals)),
                    float(np.median(intervals)),
                    float(len(onset_frames) / max(duration, 1.0)),
                ]))

        return np.concatenate(parts)

    except Exception as e:
        logger.warning("Onset MLP feature extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def signal_onset_mlp_meter(audio: np.ndarray, sr: int) -> dict[tuple[int, int], float]:
    """Signal 9: Onset MLP meter classification.

    Returns a dict mapping (numerator, denominator) tuples to scores in
    [0, 1]. The highest-scoring hypothesis is normalised to 1.0. Returns
    an empty dict when the model is unavailable or inference fails.
    """
    try:
        import torch

        if not _load_model():
            return {}

        # Feature extraction
        t0 = time.perf_counter()
        feat = _extract_features_from_audio(audio, sr)
        if feat is None:
            return {}

        # Standardize using training stats
        feat = feat.astype(np.float32)
        if _feat_mean is not None and _feat_std is not None:
            feat = (feat - _feat_mean) / np.where(_feat_std < 1e-8, 1.0, _feat_std)

        # Inference
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(_device)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Onset MLP meter inference: %.1f ms", elapsed_ms)

        # Map class probabilities to meter hypotheses
        scores: dict[tuple[int, int], float] = {}

        for cls_idx, prob in enumerate(probs):
            beats_in_bar = _idx_to_meter.get(cls_idx)
            if beats_in_bar is None:
                continue
            prob = float(prob)

            if beats_in_bar == 3:
                scores[(3, 4)] = scores.get((3, 4), 0.0) + prob
                scores[(6, 8)] = scores.get((6, 8), 0.0) + prob * 0.4
            elif beats_in_bar == 4:
                scores[(4, 4)] = scores.get((4, 4), 0.0) + prob
                scores[(2, 4)] = scores.get((2, 4), 0.0) + prob * 0.3
                scores[(12, 8)] = scores.get((12, 8), 0.0) + prob * 0.2
            elif beats_in_bar == 5:
                scores[(5, 4)] = scores.get((5, 4), 0.0) + prob
                scores[(5, 8)] = scores.get((5, 8), 0.0) + prob * 0.3
            elif beats_in_bar == 7:
                scores[(7, 4)] = scores.get((7, 4), 0.0) + prob
                scores[(7, 8)] = scores.get((7, 8), 0.0) + prob * 0.3
            elif beats_in_bar == 9:
                scores[(9, 8)] = scores.get((9, 8), 0.0) + prob
            elif beats_in_bar == 11:
                scores[(11, 8)] = scores.get((11, 8), 0.0) + prob

        # Normalise so highest score = 1.0
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        logger.debug("Onset MLP meter signal: %s", scores)
        return scores

    except Exception as e:
        logger.warning("Onset MLP meter signal failed: %s", e)
        return {}
