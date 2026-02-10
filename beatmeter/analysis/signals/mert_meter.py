"""Signal 8b: MERT-v1-95M meter classification.

Uses a pre-trained music foundation model (MERT) to extract embeddings,
then classifies meter using a lightweight MLP trained on METER2800.
The MERT model captures high-level musical structure that may be
complementary to hand-crafted signals.

The MERT model and MLP classifier are loaded lazily on first use.
If models are unavailable, the signal returns an empty dict (graceful
degradation — zero weight in ensemble).
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — populated lazily
# ---------------------------------------------------------------------------
_mert_model = None
_mert_processor = None
_mert_classifier = None
_mert_device = None
_mert_layer_idx: int | None = None
_mert_class_map: dict[int, int] | None = None
_mert_lock = threading.Lock()

MODEL_NAME = "m-a-p/MERT-v1-95M"
MERT_SR = 24000
CHUNK_SAMPLES = 5 * MERT_SR  # 5-second chunks

# Search paths for the MLP classifier checkpoint
CLASSIFIER_PATHS = [
    Path("data/meter_mert_classifier.pt"),
    Path.home() / ".beatmeter" / "meter_mert_classifier.pt",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_mert_models() -> bool:
    """Load MERT + MLP classifier (singleton, thread-safe).

    Returns True if both models are ready, False otherwise.
    """
    global _mert_model, _mert_processor, _mert_classifier, _mert_device
    global _mert_layer_idx, _mert_class_map

    # Double-check locking: fast path without lock
    if _mert_model is not None and _mert_classifier is not None:
        return True

    with _mert_lock:
        if _mert_model is not None and _mert_classifier is not None:
            return True
        return _load_mert_models_locked()


def _load_mert_models_locked() -> bool:
    """Inner model loading (called under lock)."""
    global _mert_model, _mert_processor, _mert_classifier, _mert_device
    global _mert_layer_idx, _mert_class_map

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        # Find MLP classifier checkpoint
        classifier_path: Path | None = None
        for candidate in CLASSIFIER_PATHS:
            if candidate.exists():
                classifier_path = candidate
                break

        if classifier_path is None:
            logger.debug(
                "MERT classifier not found at: %s",
                [str(p) for p in CLASSIFIER_PATHS],
            )
            return False

        # Pick device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load MLP checkpoint first (fast, to fail early if corrupt)
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
        layer_idx = checkpoint.get("layer_idx", 11)
        input_dim = checkpoint.get("input_dim", 1536)
        hidden_dim = checkpoint.get("hidden_dim", 256)
        num_classes = checkpoint.get("num_classes", 4)
        dropout = checkpoint.get("dropout", 0.3)
        class_map = checkpoint.get("class_map", {0: 3, 1: 4, 2: 5, 3: 7})

        # Build and load MLP (matches MeterMLP from train_meter_mert.py)
        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
            def forward(self, x):
                return self.net(x)

        classifier = _MLP()
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        classifier.to(device)
        classifier.eval()

        # Load MERT (from HuggingFace cache)
        logger.info("Loading MERT model %s ...", MODEL_NAME)
        t0 = time.perf_counter()
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, output_hidden_states=True
        )
        model.to(device)
        model.eval()
        elapsed = time.perf_counter() - t0
        logger.info("MERT loaded in %.1fs (device=%s, layer=%d)", elapsed, device, layer_idx)

        # Store singletons
        _mert_model = model
        _mert_processor = processor
        _mert_classifier = classifier
        _mert_device = device
        _mert_layer_idx = layer_idx
        _mert_class_map = class_map

        return True

    except Exception as e:
        logger.warning("Failed to load MERT models: %s", e)
        return False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_mert_embedding(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extract pooled embedding from one MERT layer.

    Returns (1536,) vector or None on failure.
    """
    try:
        import torch
        import torchaudio

        # Resample to 24 kHz
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        if sr != MERT_SR:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=MERT_SR)
        audio_24k = waveform.squeeze(0).numpy()

        # Center crop to 30s
        max_samples = 30 * MERT_SR
        if len(audio_24k) > max_samples:
            start = (len(audio_24k) - max_samples) // 2
            audio_24k = audio_24k[start : start + max_samples]

        if len(audio_24k) < MERT_SR:
            logger.debug("MERT: audio too short (%.2fs)", len(audio_24k) / MERT_SR)
            return None

        # Split into 5s chunks
        chunks = []
        for start in range(0, len(audio_24k), CHUNK_SAMPLES):
            chunk = audio_24k[start : start + CHUNK_SAMPLES]
            if len(chunk) < MERT_SR:
                continue
            chunks.append(chunk)
        if not chunks:
            chunks = [audio_24k]

        layer_idx = _mert_layer_idx if _mert_layer_idx is not None else 11
        chunk_means = []
        chunk_maxes = []

        with torch.no_grad():
            for chunk in chunks:
                inputs = _mert_processor(
                    chunk, sampling_rate=MERT_SR, return_tensors="pt"
                )
                inputs = {k: v.to(_mert_device) for k, v in inputs.items()}
                outputs = _mert_model(**inputs)

                # hidden_states[0] = conv features, [1..12] = transformer layers
                hs = outputs.hidden_states[layer_idx + 1].squeeze(0)  # (T, 768)
                chunk_means.append(hs.mean(dim=0).cpu().numpy())
                chunk_maxes.append(hs.max(dim=0).values.cpu().numpy())

        # Aggregate: mean of means, max of maxes
        mean_vec = np.mean(chunk_means, axis=0)   # (768,)
        max_vec = np.max(chunk_maxes, axis=0)      # (768,)
        return np.concatenate([mean_vec, max_vec]).astype(np.float32)  # (1536,)

    except Exception as e:
        logger.warning("MERT embedding extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def signal_mert_meter(audio: np.ndarray, sr: int) -> dict[tuple[int, int], float]:
    """Signal 8b: MERT meter classification.

    Returns a dict mapping (numerator, denominator) tuples to scores in
    [0, 1]. The highest-scoring hypothesis is normalised to 1.0. Returns
    an empty dict when models are unavailable or inference fails.
    """
    try:
        import torch

        if not _load_mert_models():
            return {}

        # Extract embedding
        t0 = time.perf_counter()
        embedding = _extract_mert_embedding(audio, sr)
        if embedding is None:
            return {}

        # Classify
        input_tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(_mert_device)
        with torch.no_grad():
            logits = _mert_classifier(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("MERT meter inference: %.1f ms", elapsed_ms)

        # Map class probabilities to meter hypotheses
        class_map = _mert_class_map or {0: 3, 1: 4, 2: 5, 3: 7}
        scores: dict[tuple[int, int], float] = {}

        for cls_idx, prob in enumerate(probs):
            beats_in_bar = class_map.get(cls_idx)
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
            elif beats_in_bar == 7:
                scores[(7, 4)] = scores.get((7, 4), 0.0) + prob

        # Normalise so highest score = 1.0
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        logger.debug("MERT meter signal: %s", scores)
        return scores

    except Exception as e:
        logger.warning("MERT meter signal failed: %s", e)
        return {}
