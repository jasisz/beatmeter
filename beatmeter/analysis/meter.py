"""Meter (time signature) hypothesis generation.

Score combination chain: MeterNet (audio+MERT) → 4/4 fallback.
"""

import logging
import math
import time

import numpy as np

from beatmeter.analysis.models import MeterHypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MeterNet inference (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_meter_net_model = None
_meter_net_meta = None  # feat_mean, feat_std, class_meters, etc.

# Ensemble: optional second model (FTT) for probability averaging
_ensemble_model = None
_ensemble_meta = None


def _build_meter_net(input_dim, n_classes, hidden, dropout_scale, n_blocks):
    """Build MeterNet model (must match MeterNet in scripts/training/train.py)."""
    import torch
    import torch.nn as nn

    ds = dropout_scale
    h2 = max(int(hidden * 0.4), 64)
    h4 = max(int(hidden * 0.2), 32)

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.25):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            )
        def forward(self, x):
            return x + self.block(x)

    class MeterNetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Dropout(min(0.3 * ds, 0.5)),
            )
            self.residual = nn.Sequential(
                *[ResidualBlock(hidden, dropout=min(0.25 * ds, 0.5)) for _ in range(n_blocks)]
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, h2), nn.BatchNorm1d(h2), nn.ReLU(),
                nn.Dropout(min(0.2 * ds, 0.5)),
                nn.Linear(h2, h4), nn.BatchNorm1d(h4), nn.ReLU(),
                nn.Dropout(min(0.15 * ds, 0.5)),
                nn.Linear(h4, n_classes),
            )

        def forward(self, x):
            return self.head(self.residual(self.input_proj(x)))

    return MeterNetModel()


def _build_ftt(input_dim, n_classes, d_model, n_layers, state_dict):
    """Build FTTransformer model, inferring structure from state_dict."""
    import torch
    import torch.nn as nn

    # Detect n_tokens and token sizes from state_dict
    n_tokens = sum(1 for k in state_dict if k.startswith("tokenizers.") and k.endswith(".weight"))
    token_sizes = [state_dict[f"tokenizers.{i}.weight"].shape[1] for i in range(n_tokens)]
    n_heads = 4  # default; works for all our checkpoints

    class FTTransformerInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.group_sizes = list(token_sizes)
            self.n_tokens = len(self.group_sizes)
            self.tokenizers = nn.ModuleList([
                nn.Linear(gs, d_model) for gs in self.group_sizes
            ])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=0.0,
                activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            B = x.size(0)
            tokens = []
            offset = 0
            for i, gs in enumerate(self.group_sizes):
                chunk = x[:, offset:offset + gs]
                tokens.append(self.tokenizers[i](chunk))
                offset += gs
            tokens = torch.stack(tokens, dim=1)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = self.encoder(tokens)
            return self.head(tokens[:, 0, :])

    return FTTransformerInference()



def _build_hybrid(input_dim, n_classes, mlp_hidden, mlp_dropout_scale, mlp_blocks,
                   ftt_d_model, ftt_n_layers, state_dict):
    """Build HybridMeterNet for inference, inferring FTT structure from state_dict."""
    import torch
    import torch.nn as nn

    ds = mlp_dropout_scale

    # Detect FTT token sizes from state_dict
    n_tokens = sum(1 for k in state_dict if k.startswith("ftt_tokenizers.") and k.endswith(".weight"))
    token_sizes = [state_dict[f"ftt_tokenizers.{i}.weight"].shape[1] for i in range(n_tokens)]
    n_heads = 4  # default

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.25):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            )
        def forward(self, x):
            return x + self.block(x)

    class HybridMeterNetInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.ftt_group_sizes = list(token_sizes)

            # MLP branch
            self.mlp_proj = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden), nn.ReLU(),
                nn.Dropout(min(0.3 * ds, 0.5)),
            )
            self.mlp_residual = nn.Sequential(
                *[ResidualBlock(mlp_hidden, dropout=min(0.25 * ds, 0.5))
                  for _ in range(mlp_blocks)]
            )

            # FTT branch
            self.ftt_tokenizers = nn.ModuleList([
                nn.Linear(gs, ftt_d_model) for gs in self.ftt_group_sizes
            ])
            self.ftt_cls_token = nn.Parameter(torch.randn(1, 1, ftt_d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=ftt_d_model, nhead=n_heads,
                dim_feedforward=ftt_d_model * 4, dropout=0.0,
                activation="gelu", batch_first=True,
            )
            self.ftt_encoder = nn.TransformerEncoder(encoder_layer, num_layers=ftt_n_layers)

            # Fusion head
            fusion_dim = mlp_hidden + ftt_d_model
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_dim // 2, n_classes),
            )

        def forward(self, x):
            B = x.size(0)
            mlp_repr = self.mlp_residual(self.mlp_proj(x))
            tokens = []
            offset = 0
            for i, gs in enumerate(self.ftt_group_sizes):
                chunk = x[:, offset:offset + gs]
                tokens.append(self.ftt_tokenizers[i](chunk))
                offset += gs
            tokens = torch.stack(tokens, dim=1)
            cls = self.ftt_cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = self.ftt_encoder(tokens)
            ftt_repr = tokens[:, 0, :]
            combined = torch.cat([mlp_repr, ftt_repr], dim=1)
            return self.fusion(combined)

    return HybridMeterNetInference()


def _load_ensemble_model():
    """Load secondary FTT model for ensemble (from data/meter_net_ftt.pt)."""
    global _ensemble_model, _ensemble_meta
    if _ensemble_model is not None:
        return _ensemble_model, _ensemble_meta

    import os
    import pathlib

    if os.environ.get("METER_NET_ENSEMBLE") != "1":
        return None, None

    ckpt_path = pathlib.Path("data/meter_net_ftt.pt")
    if not ckpt_path.exists():
        logger.warning("Ensemble enabled but data/meter_net_ftt.pt not found")
        return None, None

    try:
        import torch

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        input_dim = ckpt["input_dim"]
        n_classes = ckpt["n_classes"]
        hidden_dim = ckpt.get("hidden_dim", 128)
        n_blocks = ckpt.get("n_blocks", 2)
        model_type = ckpt.get("model_type", "ftt")
        state = ckpt["model_state"]

        if model_type == "ftt":
            model = _build_ftt(input_dim, n_classes, hidden_dim, n_blocks, state)
        elif model_type == "hybrid":
            model = _build_hybrid(input_dim, n_classes, hidden_dim,
                                  ckpt.get("dropout_scale", 1.5), ckpt.get("mlp_blocks", 2),
                                  ckpt.get("ftt_d_model", 128), ckpt.get("ftt_n_layers", 2),
                                  state)
        else:
            model = _build_meter_net(input_dim, n_classes, hidden_dim,
                                     ckpt.get("dropout_scale", 1.0), n_blocks)
        model.load_state_dict(state)
        model.eval()

        _ensemble_model = model
        _ensemble_meta = {
            "feat_mean": ckpt["feat_mean"],
            "feat_std": ckpt["feat_std"],
            "class_meters": ckpt["class_meters"],
            "meter_to_idx": ckpt["meter_to_idx"],
            "feature_version": ckpt.get("feature_version", "mn_v3"),
            "n_mert_features": ckpt.get("n_mert_features", 0),
            "mert_layer": ckpt.get("mert_layer", 3),
        }
        logger.info("Ensemble FTT loaded: dim=%d, hidden=%d, classes=%d",
                     input_dim, hidden_dim, n_classes)
        return _ensemble_model, _ensemble_meta

    except Exception as e:
        logger.warning("Failed to load ensemble model: %s", e)
        return None, None


def _load_meter_net():
    """Load MeterNet checkpoint once. Returns (model, meta) or (None, None)."""
    global _meter_net_model, _meter_net_meta
    if _meter_net_model is not None:
        return _meter_net_model, _meter_net_meta

    import os
    import pathlib

    custom_ckpt = os.environ.get("METER_NET_CHECKPOINT")
    if custom_ckpt:
        ckpt_path = pathlib.Path(custom_ckpt)
    else:
        ckpt_path = pathlib.Path("data/meter_net.pt")
    if not ckpt_path.exists():
        return None, None

    try:
        import torch

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        input_dim = ckpt["input_dim"]
        n_classes = ckpt["n_classes"]
        hidden_dim = ckpt.get("hidden_dim", 640)
        ds = ckpt.get("dropout_scale", 1.0)
        n_blocks = ckpt.get("n_blocks", 1)
        model_type = ckpt.get("model_type", "mlp")

        state = ckpt["model_state"]

        if model_type == "hybrid":
            ftt_d_model = ckpt.get("ftt_d_model", 128)
            ftt_n_layers = ckpt.get("ftt_n_layers", 2)
            mlp_blocks = ckpt.get("mlp_blocks", n_blocks)
            model = _build_hybrid(input_dim, n_classes, hidden_dim, ds, mlp_blocks,
                                   ftt_d_model, ftt_n_layers, state)
            model.load_state_dict(state)
        elif model_type == "ftt":
            model = _build_ftt(input_dim, n_classes, hidden_dim, n_blocks, state)
            model.load_state_dict(state)
        else:
            model = _build_meter_net(input_dim, n_classes, hidden_dim, ds, n_blocks)
            # Backward compat: old checkpoints (pre-n_blocks) have "residual.block.*"
            # instead of "residual.0.block.*".
            fixed_state = {}
            for k, v in state.items():
                if k.startswith("residual.block."):
                    fixed_state["residual.0." + k[len("residual."):]] = v
                else:
                    fixed_state[k] = v
            model.load_state_dict(fixed_state)

        model.eval()

        _meter_net_model = model
        _meter_net_meta = {
            "feat_mean": ckpt["feat_mean"],
            "feat_std": ckpt["feat_std"],
            "class_meters": ckpt["class_meters"],
            "meter_to_idx": ckpt["meter_to_idx"],
            "feature_version": ckpt.get("feature_version", "mn_v3"),
            "n_mert_features": ckpt.get("n_mert_features", 0),
            "mert_layer": ckpt.get("mert_layer", 3),
        }
        type_label = {"ftt": "FTT", "hybrid": "Hybrid", "mlp": "MeterNet"}.get(model_type, "MeterNet")
        logger.info(
            "%s loaded: dim=%d, hidden=%d, blocks=%d, classes=%d",
            type_label, input_dim, hidden_dim, n_blocks, n_classes,
        )
        return _meter_net_model, _meter_net_meta

    except Exception as e:
        logger.warning("Failed to load MeterNet: %s", e)
        return None, None


def _prepare_audio(audio: np.ndarray, sr: int, target_sr: int, max_duration_s: int) -> np.ndarray:
    """Prepare audio for feature extraction (copy, trim to max_duration_s, resample)."""
    audio_copy = audio.copy()
    max_samples = sr * max_duration_s
    if len(audio_copy) > max_samples:
        start = (len(audio_copy) - max_samples) // 2
        audio_copy = audio_copy[start:start + max_samples]
    if sr != target_sr:
        import librosa
        audio_copy = librosa.resample(audio_copy, orig_sr=sr, target_sr=target_sr)
    return audio_copy


def _meter_net_predict(
    audio: np.ndarray | None,
    sr: int,
    cache=None,
    audio_hash: str | None = None,
) -> dict[tuple[int, int], float] | None:
    """Run MeterNet inference. Returns meter scores or None."""
    model, meta = _load_meter_net()
    if model is None:
        return None

    try:
        import torch
        from beatmeter.analysis.signals.onset_mlp_features import (
            extract_features_v6, SR, MAX_DURATION_S,
        )
        from beatmeter.analysis.signals.meter_net_features import N_MERT_FEATURES

        t0 = time.perf_counter()

        # 1. Audio features (1449d) — cached independently from checkpoint
        audio_feat = None
        audio_feat_cached = False
        if cache and audio_hash:
            audio_feat = cache.load_array(audio_hash, "meter_net_audio")
            if audio_feat is not None:
                audio_feat_cached = True

        if audio_feat is None:
            if audio is None:
                return None
            audio_copy = _prepare_audio(audio, sr, SR, MAX_DURATION_S)
            audio_feat = extract_features_v6(audio_copy, SR)
            if audio_feat is None:
                return None
            if cache and audio_hash:
                cache.save_array(audio_hash, "meter_net_audio", audio_feat)

        # 2. MERT embedding (1536d) — from cache or runtime extraction
        n_mert = meta.get("n_mert_features", 0)
        mert_feat = None
        mert_cached = False

        if n_mert > 0:
            if cache and audio_hash:
                mert_feat = cache.load_array(audio_hash, "meter_net_mert")
                if mert_feat is not None:
                    mert_cached = True

            if mert_feat is None:
                if audio is not None:
                    from beatmeter.analysis.mert import extract_mert_embedding
                    mert_layer = meta.get("mert_layer", 3)
                    mert_feat = extract_mert_embedding(audio, sr, layer=mert_layer)
                    if cache and audio_hash:
                        cache.save_array(audio_hash, "meter_net_mert", mert_feat)
                else:
                    mert_feat = np.zeros(N_MERT_FEATURES, dtype=np.float32)

        # 3. Concatenate: audio + MERT
        parts = [audio_feat]
        if mert_feat is not None:
            parts.append(mert_feat)
        full_feat = np.concatenate(parts)

        expected_dim = len(meta["feat_mean"])
        if full_feat.shape[0] != expected_dim:
            logger.warning("MeterNet feature dim mismatch: %d != %d", full_feat.shape[0], expected_dim)
            return None

        # Standardize (keep raw copy for ensemble)
        full_feat = full_feat.astype(np.float32)
        full_feat_raw = full_feat.copy()
        feat_mean = meta["feat_mean"]
        feat_std = meta["feat_std"]
        full_feat = (full_feat - feat_mean) / np.where(feat_std < 1e-8, 1.0, feat_std)

        # 4. Forward pass (with optional ensemble)
        x = torch.tensor(full_feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        # Ensemble: average with second model if available
        ens_model, ens_meta = _load_ensemble_model()
        if ens_model is not None:
            ens_mean = ens_meta["feat_mean"]
            ens_std = ens_meta["feat_std"]
            ens_feat = full_feat_raw.astype(np.float32)
            ens_feat = (ens_feat - ens_mean) / np.where(ens_std < 1e-8, 1.0, ens_std)
            x_ens = torch.tensor(ens_feat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits_ens = ens_model(x_ens)
                probs_ens = torch.sigmoid(logits_ens).squeeze(0).numpy()
            import os
            w = float(os.environ.get("METER_NET_ENSEMBLE_W", "0.5"))
            probs = w * probs + (1 - w) * probs_ens

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("MeterNet: %.1f ms (audio=%s, mert=%s%s)",
                     elapsed_ms,
                     "cached" if audio_feat_cached else "live",
                     "cached" if mert_cached else "live",
                     ", ensemble" if ens_model is not None else "")

        # Convert class probabilities to meter scores
        class_meters = meta["class_meters"]
        scores: dict[tuple[int, int], float] = {}
        for cls_idx, meter_num in enumerate(class_meters):
            prob = float(probs[cls_idx])
            if prob < 0.01:
                continue
            # Default denominator: 4 for common meters, 8 for larger
            den = 4 if meter_num <= 7 else 8
            scores[(meter_num, den)] = prob

        return scores

    except Exception as e:
        logger.warning("MeterNet prediction failed: %s", e)
        return None


# Meter descriptions for musicians
METER_DESCRIPTIONS = {
    (2, 4): "Marsz, polka",
    (3, 4): "Walc (np. Blue Danube, Walc Chopina)",
    (3, 8): "Szybki walc, gigue",
    (4, 4): "Standardowy rock/pop (np. Billie Jean, Hey Jude)",
    (5, 4): "Take Five (Dave Brubeck) - grupowanie 3+2 lub 2+3",
    (5, 8): "Mission Impossible - grupowanie 3+2 lub 2+3",
    (6, 4): "Wolne 6 (rzadkie)",
    (6, 8): "Tarantella, Nothing Else Matters - dwie grupy po 3",
    (7, 4): "Rzadkie - np. Money (Pink Floyd)",
    (7, 8): "Money (Pink Floyd) - grupowanie 2+2+3 lub 3+2+2",
    (9, 8): "Blue Rondo à la Turk (Dave Brubeck) - 2+2+2+3",
    (10, 8): "Rzadkie - np. niektóre utwory Tool",
    (11, 8): "Bardzo rzadkie - np. I Hang on to a Dream (The Nice)",
    (12, 8): "Blues shuffle, Everybody Wants To Rule The World - cztery grupy po 3",
}

GROUPINGS = {
    (5, 4): ["3+2", "2+3"],
    (5, 8): ["3+2", "2+3"],
    (7, 4): ["2+2+3", "3+2+2", "2+3+2"],
    (7, 8): ["2+2+3", "3+2+2", "2+3+2"],
    (9, 8): ["2+2+2+3", "3+3+3", "3+2+2+2"],
    (10, 8): ["3+3+2+2", "2+3+3+2", "3+2+3+2"],
    (11, 8): ["3+3+3+2", "2+2+3+2+2", "3+2+3+3"],
}

# Noise filter: remove hypotheses scoring less than this fraction of the best
NOISE_FILTER_RATIO = 0.10


def _get_description(num: int, den: int) -> str:
    return METER_DESCRIPTIONS.get((num, den), f"{num}/{den}")


def _get_grouping(num: int, den: int) -> str | None:
    groupings = GROUPINGS.get((num, den))
    return groupings[0] if groupings else None


# ---------------------------------------------------------------------------
# Hypothesis formatting
# ---------------------------------------------------------------------------

def _compute_ambiguity(scores: dict[tuple[int, int], float]) -> float:
    """Compute normalized entropy of meter score distribution."""
    total = sum(scores.values())
    if total <= 0:
        return 1.0
    probs = [v / total for v in scores.values() if v > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs))
    return round(entropy / max_entropy, 3)


def _format_hypotheses(
    all_scores: dict[tuple[int, int], float],
    max_hypotheses: int,
) -> tuple[list[MeterHypothesis], float]:
    """Filter, normalize, sort and format final hypotheses."""
    ambiguity = _compute_ambiguity(all_scores)

    # Filter noise
    max_raw = max(all_scores.values())
    if max_raw > 0:
        all_scores = {k: v for k, v in all_scores.items() if v >= max_raw * NOISE_FILTER_RATIO}

    # Normalize to probabilities
    total = sum(all_scores.values())
    if total > 0:
        all_scores = {k: v / total for k, v in all_scores.items()}

    # Sort, take top N, re-normalize
    sorted_meters = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_meters[:max_hypotheses]

    top_total = sum(s for _, s in top)
    if top_total > 0:
        top = [(m, s / top_total) for m, s in top]

    hypotheses = []
    meter_set = {m for m, _ in top}
    for (num, den), confidence in top:
        hint = _get_disambiguation_hint(num, den, meter_set, {m: s for m, s in top})
        hypotheses.append(MeterHypothesis(
            numerator=num,
            denominator=den,
            confidence=round(confidence, 3),
            grouping=_get_grouping(num, den),
            description=_get_description(num, den),
            disambiguation_hint=hint,
        ))

    return hypotheses, ambiguity


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_hypotheses(
    audio: np.ndarray | None = None,
    sr: int = 22050,
    max_hypotheses: int = 5,
    cache=None,
    audio_hash: str | None = None,
    audio_file_path: str | None = None,
) -> tuple[list[MeterHypothesis], float]:
    """Generate meter hypotheses. MeterNet → 4/4 fallback."""
    meter_net_scores = _meter_net_predict(
        audio, sr,
        cache=cache, audio_hash=audio_hash,
    )

    if meter_net_scores:
        all_scores = meter_net_scores
        logger.debug("Using MeterNet for meter scoring")
    else:
        all_scores = {}

    if not all_scores:
        return [MeterHypothesis(
            numerator=4, denominator=4, confidence=0.3,
            description=_get_description(4, 4),
        )], 1.0

    # Format hypotheses
    return _format_hypotheses(all_scores, max_hypotheses)


# Disambiguation hint keys for ambiguous meter pairs.
_DISAMBIGUATION_PAIRS = {
    ((6, 8), (3, 4)): "6_8_vs_3_4",
    ((3, 4), (6, 8)): "3_4_vs_6_8",
    ((4, 4), (2, 4)): "4_4_vs_2_4",
    ((2, 4), (4, 4)): "2_4_vs_4_4",
    ((12, 8), (4, 4)): "12_8_vs_4_4",
}


def _get_disambiguation_hint(
    num: int,
    den: int,
    all_meters: set[tuple[int, int]],
    scores: dict[tuple[int, int], float],
) -> str | None:
    """Return disambiguation hint key when ambiguous meters are close in confidence."""
    meter = (num, den)
    for (m1, m2), hint_key in _DISAMBIGUATION_PAIRS.items():
        if meter == m1 and m2 in all_meters:
            s1 = scores.get(m1, 0)
            s2 = scores.get(m2, 0)
            if s2 > 0 and s1 / s2 < 2.0:
                return hint_key
    return None
