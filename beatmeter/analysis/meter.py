"""Meter (time signature) hypothesis generation.

Score combination chain: MeterNet → 4/4 fallback.
Signal functions live in beatmeter.analysis.signals/.
"""

import logging
import math
import time

import numpy as np

from beatmeter.analysis.models import Beat, MeterHypothesis
from beatmeter.analysis.signals import (
    signal_downbeat_spacing,
    signal_onset_autocorrelation,
    signal_bar_tracking,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MeterNet inference (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_meter_net_model = None
_meter_net_meta = None  # feat_mean, feat_std, class_meters, etc.


def _build_meter_net(input_dim, n_classes, hidden, dropout_scale, n_blocks,
                     bottleneck=0, ac_split=1024):
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
            self.bottleneck_dim = bottleneck
            self.ac_split = ac_split

            if bottleneck > 0:
                self.ac_compress = nn.Sequential(
                    nn.Linear(ac_split, bottleneck), nn.ReLU(),
                )
                proj_input = bottleneck + (input_dim - ac_split)
            else:
                self.ac_compress = None
                proj_input = input_dim

            self.input_proj = nn.Sequential(
                nn.Linear(proj_input, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
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
            if self.ac_compress is not None:
                ac = x[:, :self.ac_split]
                rest = x[:, self.ac_split:]
                x = torch.cat([self.ac_compress(ac), rest], dim=1)
            return self.head(self.residual(self.input_proj(x)))

    return MeterNetModel()



def _load_meter_net():
    """Load MeterNet checkpoint once. Returns (model, meta) or (None, None).

    Respects env var:
      METER_NET_CHECKPOINT → custom checkpoint path (default: data/meter_net.pt)
    """
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
        import torch.nn as nn

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        input_dim = ckpt["input_dim"]
        n_classes = ckpt["n_classes"]
        hidden_dim = ckpt.get("hidden_dim", 640)
        ds = ckpt.get("dropout_scale", 1.0)
        n_blocks = ckpt.get("n_blocks", 1)

        bottleneck = ckpt.get("bottleneck", 0)
        model = _build_meter_net(input_dim, n_classes, hidden_dim, ds, n_blocks,
                                 bottleneck=bottleneck)

        # Backward compat: old checkpoints (pre-n_blocks) have "residual.block.*"
        # instead of "residual.0.block.*".
        state = ckpt["model_state"]
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
        }
        logger.info(
            "MeterNet loaded: dim=%d, hidden=%d, blocks=%d, classes=%d",
            input_dim, hidden_dim, n_blocks, n_classes,
        )
        return _meter_net_model, _meter_net_meta

    except Exception as e:
        logger.warning("Failed to load MeterNet: %s", e)
        return None, None


def _meter_net_predict(
    audio: np.ndarray,
    sr: int,
    signal_results: dict[str, dict[tuple[int, int], float]],
    beatnet_beats: list[Beat],
    beat_this_beats: list[Beat] | None,
    madmom_results: dict[int, list[Beat]],
    onset_event_times: np.ndarray | None,
    tempo_bpm: float | None,
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
        from beatmeter.analysis.signals.meter_net_features import (
            extract_beat_features_live,
            extract_signal_scores_live,
            extract_tempo_features_live,
        )
        from beatmeter.analysis.signals.ssm_features import extract_ssm_features_live
        import librosa

        t0 = time.perf_counter()

        # Prepare audio
        audio_copy = audio.copy()
        max_samples = sr * MAX_DURATION_S
        if len(audio_copy) > max_samples:
            start = (len(audio_copy) - max_samples) // 2
            audio_copy = audio_copy[start:start + max_samples]
        if sr != SR:
            audio_copy = librosa.resample(audio_copy, orig_sr=sr, target_sr=SR)

        # Tempo info from cache/live
        tempo_lib = 0.0
        tempo_tg = 0.0
        if cache and audio_hash:
            td = cache.load_signal(audio_hash, "tempo_librosa")
            if td:
                tempo_lib = td.get("bpm", 0.0)
            td = cache.load_signal(audio_hash, "tempo_tempogram")
            if td:
                tempo_tg = td.get("bpm", 0.0)
        if tempo_lib == 0 and tempo_tg == 0 and tempo_bpm:
            tempo_lib = tempo_bpm
            tempo_tg = tempo_bpm

        # Audio features (v6: 1449d, tempos from internal tempogram)
        audio_feat = extract_features_v6(audio_copy, SR)
        if audio_feat is None:
            return None

        # Beat-synchronous chroma SSM (75 dims)
        ssm_feat = extract_ssm_features_live(
            audio_copy, SR, beatnet_beats, beat_this_beats, madmom_results,
        )

        # Beat features (42 dims)
        duration = len(audio) / sr
        oet = onset_event_times if onset_event_times is not None else np.array([])
        beat_feat = extract_beat_features_live(
            beatnet_beats, beat_this_beats, madmom_results, oet, duration,
        )

        # Signal scores (60 dims) — 5 signals, excludes onset_mlp
        signal_feat = extract_signal_scores_live(signal_results)

        # Tempo features (4 dims)
        tempo_feat = extract_tempo_features_live(tempo_lib, tempo_tg)

        # Concatenate
        full_feat = np.concatenate([
            audio_feat, ssm_feat,
            beat_feat, signal_feat, tempo_feat,
        ])
        expected_dim = len(meta["feat_mean"])
        if full_feat.shape[0] != expected_dim:
            logger.warning("MeterNet feature dim mismatch: %d != %d", full_feat.shape[0], expected_dim)
            return None

        # Standardize
        feat_mean = meta["feat_mean"]
        feat_std = meta["feat_std"]
        full_feat = full_feat.astype(np.float32)
        full_feat = (full_feat - feat_mean) / np.where(feat_std < 1e-8, 1.0, feat_std)

        # Inference
        x = torch.tensor(full_feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("MeterNet inference: %.1f ms", elapsed_ms)

        # Convert class probabilities to meter scores
        class_meters = meta["class_meters"]
        scores: dict[tuple[int, int], float] = {}
        for cls_idx, meter_num in enumerate(class_meters):
            prob = float(probs[cls_idx])
            if prob < 0.01:
                continue
            # Find best denominator from signal_results
            best_den = None
            best_sig_score = -1.0
            for sig_scores in signal_results.values():
                for (num, den), sc in sig_scores.items():
                    if num == meter_num and sc > best_sig_score:
                        best_sig_score = sc
                        best_den = den
            if best_den is None:
                best_den = 4 if meter_num <= 7 else 8
            scores[(meter_num, best_den)] = prob

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
# Signal collection — compute signals needed by MeterNet
# ---------------------------------------------------------------------------


def _collect_signal_scores(
    beatnet_beats: list[Beat],
    madmom_results: dict[int, list[Beat]],
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    beat_interval: float | None,
    beat_times: np.ndarray,
    sr: int,
    audio: np.ndarray | None,
    beat_this_beats: list[Beat] | None,
    skip_bar_tracking: bool,
    cache=None,
    audio_hash: str | None = None,
    tmp_path: str | None = None,
) -> dict[str, dict[tuple[int, int], float]]:
    """Collect scores from all active signals."""
    from beatmeter.analysis.cache import str_to_meter_key

    signal_results: dict[str, dict[tuple[int, int], float]] = {}

    def _try_cache(sig_name: str) -> dict[tuple[int, int], float] | None:
        if cache and audio_hash:
            raw = cache.load_signal(audio_hash, sig_name)
            if raw is not None:
                return {str_to_meter_key(k): v for k, v in raw.items()}
        return None

    def _save_cache(sig_name: str, scores: dict[tuple[int, int], float]) -> None:
        if cache and audio_hash and scores:
            cache.save_signal(audio_hash, sig_name, scores)

    # Signal 1a: BeatNet downbeat spacing
    s1 = _try_cache("beatnet_spacing")
    if s1 is None:
        s1 = signal_downbeat_spacing(beatnet_beats)
        _save_cache("beatnet_spacing", s1)
    if s1:
        signal_results["beatnet"] = s1

    # Signal 1b: Beat This! downbeat spacing
    if beat_this_beats:
        s1b = _try_cache("beat_this_spacing")
        if s1b is None:
            s1b = signal_downbeat_spacing(beat_this_beats)
            _save_cache("beat_this_spacing", s1b)
        if s1b:
            signal_results["beat_this"] = s1b

    # Signal 3: onset autocorrelation
    if len(onset_times) > 0:
        s3 = _try_cache("onset_autocorr")
        if s3 is None:
            s3 = signal_onset_autocorrelation(onset_times, onset_strengths, beat_interval, sr)
            _save_cache("onset_autocorr", s3)
        if s3:
            signal_results["autocorr"] = s3

    # Signal 7: Bar tracking (DBNBarTrackingProcessor)
    if not skip_bar_tracking and audio is not None and len(beat_times) >= 6:
        s7 = _try_cache("bar_tracking")
        if s7 is None:
            if beat_this_beats and len(beat_this_beats) >= 6:
                bar_beat_times = np.array([b.time for b in beat_this_beats])
            else:
                bar_beat_times = beat_times
            s7 = signal_bar_tracking(audio, sr, bar_beat_times, tmp_path=tmp_path)
            _save_cache("bar_tracking", s7)
        if s7:
            signal_results["bar_tracking"] = s7

    # Signal 6: HCDF meter
    if cache and audio_hash:
        hcdf = _try_cache("hcdf_meter")
        if hcdf:
            signal_results["hcdf"] = hcdf

    return signal_results


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
    beatnet_beats: list[Beat],
    madmom_results: dict[int, list[Beat]],
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    all_beats: list[Beat],
    sr: int = 22050,
    max_hypotheses: int = 5,
    tempo_bpm: float | None = None,
    beatnet_alignment: float = 1.0,
    madmom_alignment: float = 1.0,
    audio: np.ndarray | None = None,
    librosa_beats: list[Beat] | None = None,
    beat_this_beats: list[Beat] | None = None,
    beat_this_alignment: float = 0.0,
    onset_event_times: np.ndarray | None = None,
    skip_bar_tracking: bool = False,
    cache=None,
    audio_hash: str | None = None,
    tmp_path: str | None = None,
    audio_file_path: str | None = None,
) -> tuple[list[MeterHypothesis], float]:
    """Generate meter hypotheses. MeterNet → 4/4 fallback."""
    # Beat interval from tempo
    beat_interval = None
    if tempo_bpm and tempo_bpm > 0:
        beat_interval = 60.0 / tempo_bpm
    elif len(all_beats) >= 3:
        times = np.array([b.time for b in all_beats])
        ibis = np.diff(times)
        if len(ibis) > 0:
            beat_interval = float(np.median(ibis))

    beat_times = np.array([b.time for b in all_beats]) if all_beats else np.array([])

    # Collect signal scores (needed by MeterNet for 60-dim signal features)
    signal_results = _collect_signal_scores(
        beatnet_beats, madmom_results,
        onset_times, onset_strengths, beat_interval, beat_times,
        sr, audio, beat_this_beats, skip_bar_tracking,
        cache=cache, audio_hash=audio_hash, tmp_path=tmp_path,
    )

    # Score combination: MeterNet → 4/4 fallback
    meter_net_scores = None
    meter_net_model, _ = _load_meter_net()
    if meter_net_model is not None and audio is not None:
        # Try cache first
        if cache and audio_hash:
            from beatmeter.analysis.cache import str_to_meter_key
            cached_mn = cache.load_signal(audio_hash, "meter_net")
            if cached_mn is not None:
                meter_net_scores = {str_to_meter_key(k): v for k, v in cached_mn.items()}

        if meter_net_scores is None:
            meter_net_scores = _meter_net_predict(
                audio, sr, signal_results,
                beatnet_beats, beat_this_beats, madmom_results,
                onset_event_times, tempo_bpm,
                cache=cache, audio_hash=audio_hash,
            )
            # Cache the result for next run
            if meter_net_scores and cache and audio_hash:
                cache.save_signal(audio_hash, "meter_net", meter_net_scores)

    if meter_net_scores is not None:
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
