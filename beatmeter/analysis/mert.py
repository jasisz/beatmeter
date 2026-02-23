"""Runtime MERT-v1-95M embedding extraction (lazy-loaded singleton).

Extracts 1536-dim embeddings from layer 3 of MERT-v1-95M.
Used by MeterNet when pre-extracted .npy embeddings are not available in cache.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_mert_model = None
_mert_processor = None

MERT_SR = 24000
MERT_LAYER = 3
MERT_DIM = 1536
MAX_DURATION_S = 30


def _load_mert():
    """Load MERT model and processor (once)."""
    global _mert_model, _mert_processor
    if _mert_model is not None:
        return _mert_model, _mert_processor

    import torch
    from transformers import Wav2Vec2FeatureExtractor, AutoModel

    model_name = "m-a-p/MERT-v1-95M"
    logger.info("Loading MERT model: %s", model_name)
    _mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    _mert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    _mert_model.eval()
    logger.info("MERT model loaded")
    return _mert_model, _mert_processor


def extract_mert_embedding(
    audio: np.ndarray, sr: int, layer: int = MERT_LAYER,
) -> np.ndarray:
    """Extract MERT embedding from audio.

    Args:
        audio: Mono audio signal (float32).
        sr: Sample rate of the input audio.
        layer: Hidden layer to extract (default: 3).

    Returns:
        1536-dim float32 array (mean-pooled over time).
    """
    try:
        import torch
        import librosa

        model, processor = _load_mert()

        # Resample to MERT's expected 24kHz
        if sr != MERT_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=MERT_SR)

        # Trim to max duration
        max_samples = MERT_SR * MAX_DURATION_S
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            audio = audio[start:start + max_samples]

        # Process
        inputs = processor(audio, sampling_rate=MERT_SR, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract specified layer, mean-pool over time
        hidden = outputs.hidden_states[layer]  # (1, T, 768) or (1, T, 1536)
        embedding = hidden.squeeze(0).mean(dim=0).numpy().astype(np.float32)

        if embedding.shape[0] != MERT_DIM:
            logger.warning(
                "MERT embedding dim mismatch: %d != %d", embedding.shape[0], MERT_DIM,
            )
            return np.zeros(MERT_DIM, dtype=np.float32)

        return embedding

    except Exception as e:
        logger.warning("MERT extraction failed: %s", e)
        return np.zeros(MERT_DIM, dtype=np.float32)
