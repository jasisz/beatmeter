#!/usr/bin/env python3
"""Generate HTML review page for potential GT label errors in METER2800 test set."""

import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import load_meter2800_entries, resolve_audio_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "meter2800"
DATASET_DIR = PROJECT_ROOT / "data" / "arbiter_dataset"
OUT_HTML = PROJECT_ROOT / "data" / "gt_suspects.html"
OUT_ZIP = PROJECT_ROOT / "data" / "gt_suspects.zip"

SIGNAL_NAMES = ["beatnet", "beat_this", "autocorr", "bar_tracking", "onset_mlp", "hcdf"]


def load_test_labels():
    """Load ground truth for test split with corrections applied."""
    labels = {}
    fname_to_path = {}
    for audio_path, meter in load_meter2800_entries(DATA_DIR, "test"):
        labels[audio_path.name] = meter
        fname_to_path[audio_path.name] = audio_path
    return labels, fname_to_path


def get_signal_top_meter(sig_scores):
    if not sig_scores:
        return None, 0.0
    by_num = defaultdict(float)
    for key, score in sig_scores.items():
        num = int(key.split("_")[0])
        by_num[num] = max(by_num[num], score)
    if not by_num:
        return None, 0.0
    best_num = max(by_num, key=by_num.get)
    return best_num, by_num[best_num]


WIKIMETER_DIR = PROJECT_ROOT / "data" / "wikimeter"

# Map WIKIMETER meter label → numerator used in METER2800 classes
WIKIMETER_METER_MAP = {3: 3, 4: 4, 5: 5, 7: 7, 9: 9, 11: 11}


def _load_wikimeter_multilabel() -> dict[str, dict[int, float]]:
    """Load multi-label meters from wikimeter.json, keyed by segment stem prefix.

    Returns dict: song_stem -> {meter_num: weight}, e.g.
      "fela_kuti_water_no_get_enemy" -> {3: 0.6, 4: 1.0}
    """
    catalog_path = PROJECT_ROOT / "scripts" / "setup" / "wikimeter.json"
    if not catalog_path.exists():
        return {}

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    result: dict[str, dict[int, float]] = {}
    for song in catalog:
        meters = song.get("meters", {})
        if not isinstance(meters, dict) or not meters:
            continue
        # Reproduce sanitize_filename from download_wikimeter.py
        name = f"{song['artist']}_{song['title']}".lower()
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")[:80]
        result[name] = {int(k): float(v) for k, v in meters.items()}
    return result


def _segment_stem_to_song_stem(segment_fname: str) -> str | None:
    """Extract song stem from a segment filename like 'foo_bar_seg03.mp3'."""
    stem = Path(segment_fname).stem
    m = re.match(r"^(.+)_seg\d+$", stem)
    return m.group(1) if m else None


def load_wikimeter_labels():
    """Load WIKIMETER labels from tuning dataset entries with _seg in name."""
    labels = {}
    fname_to_path = {}
    # Read from tuning.json (WIKIMETER entries are mixed in when --extra-data)
    tuning_path = DATASET_DIR / "tuning.json"
    if not tuning_path.exists():
        return labels, fname_to_path
    with open(tuning_path) as f:
        dataset = json.load(f)
    for entry in dataset:
        fname = entry["fname"]
        if "_seg" not in fname:
            continue
        meter = entry.get("expected") or entry.get("meter")
        if meter is None:
            continue
        audio_path = WIKIMETER_DIR / "audio" / fname
        if audio_path.exists():
            labels[fname] = meter
            fname_to_path[fname] = audio_path
    return labels, fname_to_path


def _load_arbiter():
    """Load arbiter model and checkpoint data."""
    import torch
    import torch.nn as nn
    ckpt = torch.load(PROJECT_ROOT / "data" / "meter_arbiter.pt", weights_only=False, map_location="cpu")

    n_feat = ckpt["total_features"]
    n_cls = ckpt["n_classes"]
    model = nn.Sequential(
        nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(32, n_cls),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def build_suspects_from(labels, fname_to_path, by_fname, model, ckpt, dataset_name, wm_multilabel=None):
    """Find suspect files for a given set of labels."""
    import torch
    class_meters = ckpt["class_meters"]
    feat_mean = ckpt["feat_mean"]
    feat_std = ckpt["feat_std"]
    sharpening = ckpt.get("sharpening", {})
    signal_names = ckpt["signal_names"]
    meter_keys = ckpt["meter_keys"]
    n_signals = len(signal_names)
    n_meters = len(meter_keys)

    suspects = []
    for fname, gt_meter in labels.items():
        entry = by_fname.get(fname)
        if entry is None:
            continue

        if gt_meter not in class_meters:
            continue

        sig_results = entry["signal_results"]
        x = np.zeros(n_signals * n_meters, dtype=np.float32)
        for s_idx, sig_name in enumerate(signal_names):
            if sig_name in sig_results:
                alpha = sharpening.get(sig_name, 1.0)
                raw = [sig_results[sig_name].get(mk, 0.0) for mk in meter_keys]
                if alpha != 1.0:
                    arr = np.array(raw)
                    if arr.max() > 0:
                        arr = arr ** alpha
                        arr /= arr.max()
                    for m_idx, val in enumerate(arr):
                        x[s_idx * n_meters + m_idx] = val
                else:
                    for m_idx, val in enumerate(raw):
                        x[s_idx * n_meters + m_idx] = val

        x_std = (x - feat_mean) / feat_std
        with torch.no_grad():
            logits = model(torch.tensor(x_std).unsqueeze(0))
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        pred_idx = int(probs.argmax())
        pred_meter = class_meters[pred_idx]
        pred_conf = float(probs[pred_idx])
        gt_class_idx = class_meters.index(gt_meter)
        gt_prob = float(probs[gt_class_idx])

        sig_votes = {}
        sig_full = {}  # full per-numerator scores for multi-label display
        for sig_name in SIGNAL_NAMES:
            if sig_name in sig_results:
                top_m, top_s = get_signal_top_meter(sig_results[sig_name])
                sig_votes[sig_name] = (top_m, top_s)
                # Aggregate by numerator: keep max score per numerator
                by_num = defaultdict(float)
                for key, score in sig_results[sig_name].items():
                    num = int(key.split("_")[0])
                    by_num[num] = max(by_num[num], score)
                sig_full[sig_name] = dict(sorted(by_num.items(), key=lambda x: -x[1])[:3])
            else:
                sig_votes[sig_name] = (None, 0.0)
                sig_full[sig_name] = {}

        gt_agreement = sum(1 for m, _ in sig_votes.values() if m == gt_meter)

        # All class probabilities
        all_probs = {class_meters[i]: float(probs[i]) for i in range(len(class_meters))}

        # GT multi-label lookup from wikimeter.json
        gt_meters = None
        if wm_multilabel:
            song_stem = _segment_stem_to_song_stem(fname)
            if song_stem and song_stem in wm_multilabel:
                gt_meters = wm_multilabel[song_stem]

        is_polymetric = gt_meters is not None and len(gt_meters) > 1

        if pred_meter != gt_meter and gt_agreement <= 1 and pred_conf > 0.6:
            suspects.append({
                "fname": fname,
                "fpath": fname_to_path[fname],
                "gt": gt_meter,
                "pred": pred_meter,
                "pred_conf": pred_conf,
                "gt_prob": gt_prob,
                "gt_agreement": gt_agreement,
                "sig_votes": sig_votes,
                "sig_full": sig_full,
                "dataset": dataset_name,
                "all_probs": all_probs,
                "polymetric": is_polymetric,
                "gt_meters": gt_meters,
            })

    suspects.sort(key=lambda e: -e["pred_conf"])
    return suspects


def build_suspects():
    model, ckpt = _load_arbiter()

    # Merge all arbiter dataset JSONs
    by_fname = {}
    for json_name in ["test.json", "tuning.json"]:
        path = DATASET_DIR / json_name
        if path.exists():
            for entry in json.load(open(path)):
                by_fname[entry["fname"]] = entry

    # Load WIKIMETER multi-label ground truth from catalog
    wm_multilabel = _load_wikimeter_multilabel()

    # METER2800 test
    m2800_labels, m2800_paths = load_test_labels()
    m2800_suspects = build_suspects_from(m2800_labels, m2800_paths, by_fname, model, ckpt, "meter2800")

    # WIKIMETER
    wm_labels, wm_paths = load_wikimeter_labels()
    wm_suspects = build_suspects_from(wm_labels, wm_paths, by_fname, model, ckpt, "wikimeter", wm_multilabel=wm_multilabel)

    print(f"  METER2800: {len(m2800_suspects)} suspects / {len(m2800_labels)} files "
          f"({len(m2800_suspects)/max(len(m2800_labels),1)*100:.1f}%)")
    print(f"  WIKIMETER: {len(wm_suspects)} suspects / {len(wm_labels)} files "
          f"({len(wm_suspects)/max(len(wm_labels),1)*100:.1f}%)")

    all_suspects = m2800_suspects + wm_suspects
    all_suspects.sort(key=lambda e: -e["pred_conf"])
    return all_suspects


def render_html(suspects, portable=False):
    """Render review HTML. portable=True uses relative audio/ paths for zip."""
    meter_colors = {3: "#e8f4f8", 4: "#f8f4e8", 5: "#f8e8e8", 7: "#e8f8e8"}

    rows = []
    for i, e in enumerate(suspects, 1):
        fpath = e["fpath"]
        audio_src = f"audio/{fpath.name}" if portable else fpath.as_uri()

        gt_meter_nums = set(e.get("gt_meters", {}).keys()) if e.get("gt_meters") else {e["gt"]}

        votes_html = ""
        for sig_name in SIGNAL_NAMES:
            full = e.get("sig_full", {}).get(sig_name, {})
            if not full:
                votes_html += f'<div class="sig-block"><span class="sig-name">{sig_name}</span> <span style="color:#aaa">∅</span></div>'
                continue
            # Show all meters above threshold + always include GT meters
            # Skip 2/x — not a real class, just sub-harmonic of 4/4
            show_meters = {}
            for m_num, sc in full.items():
                if m_num == 2:
                    continue
                if sc >= 0.15 or m_num in gt_meter_nums:
                    show_meters[m_num] = sc
            # Sort by score descending
            sorted_meters = sorted(show_meters.items(), key=lambda x: -x[1])
            rows_inner = ""
            for m_num, sc in sorted_meters:
                bar_w = int(sc * 60)
                is_gt = m_num in gt_meter_nums
                is_pred = m_num == e["pred"]
                color = "#c0392b" if is_gt else "#27ae60" if is_pred else "#999"
                weight = "bold" if is_gt else "normal"
                rows_inner += (
                    f'<div style="display:flex;align-items:center;gap:3px;margin:1px 0">'
                    f'<span style="width:28px;text-align:right;font-weight:{weight};color:{color};font-size:11px;white-space:nowrap">{m_num}/x</span>'
                    f'<span style="width:{bar_w}px;height:7px;background:{color};border-radius:2px;display:inline-block"></span>'
                    f'<span style="color:#888;font-size:10px">{sc:.2f}</span>'
                    f'</div>'
                )
            votes_html += f'<div class="sig-block"><span class="sig-name">{sig_name}</span>{rows_inner}</div>'

        bg = meter_colors.get(e["gt"], "#fff")
        conf_pct = int(e["pred_conf"] * 100)

        rows.append(f"""
        <tr style="background:{bg}">
            <td style="font-weight:bold;color:#555">{i}</td>
            <td style="font-family:monospace;font-size:12px;max-width:220px;word-break:break-all">
                {fpath.name}<br>
                <span style="font-size:10px;padding:1px 4px;border-radius:3px;background:{'#d1ecf1' if e.get('dataset')=='meter2800' else '#d4edda'};color:#333">{e.get('dataset','?')}</span>
            </td>
            <td style="text-align:center">
                <span style="font-size:20px;font-weight:bold;color:#c0392b">{e['gt']}/x</span>
                {f'<br><span style="font-size:10px;padding:2px 6px;border-radius:3px;background:#fff3cd;color:#856404">{" + ".join(f"{m}/x ({w:.1f})" for m, w in sorted(e["gt_meters"].items()))}</span>' if e.get('gt_meters') and len(e.get('gt_meters', {})) > 1 else ''}
            </td>
            <td style="text-align:center">
                <span style="font-size:20px;font-weight:bold;color:#27ae60">{e['pred']}/x</span>
                <br><span style="font-size:11px;color:#888">{conf_pct}% conf</span>
            </td>
            <td style="font-size:11px;font-family:monospace;white-space:nowrap">{''.join(
                f'<div style="margin:1px 0"><span style="display:inline-block;width:30px;text-align:right;font-weight:{"bold" if m == e["gt"] else "bold" if m == e["pred"] else "normal"};color:{"#c0392b" if m == e["gt"] else "#27ae60" if m == e["pred"] else "#999"}">{m}/x</span> '
                f'<span style="display:inline-block;width:{int(p*120)}px;height:10px;background:{"#c0392b" if m == e["gt"] else "#27ae60" if m == e["pred"] else "#ccc"};border-radius:2px"></span> '
                f'<span style="color:#666">{p:.0%}</span></div>'
                for m, p in sorted(e.get("all_probs", {}).items()) if p > 0.05
            )}</td>
            <td style="font-size:12px">{votes_html}</td>
            <td>
                <audio controls style="width:220px">
                    <source src="{audio_src}" type="audio/mpeg">
                    <source src="{audio_src}" type="audio/wav">
                </audio>
            </td>
            <td>
                <select id="verdict_{i}" style="font-size:14px;padding:4px">
                    <option value="">— wybierz —</option>
                    <option value="gt_wrong">GT błędne → {e['pred']}/x</option>
                    <option value="gt_correct">GT poprawne → {e['gt']}/x</option>
                    <option value="ambiguous">Niejednoznaczne</option>
                </select>
            </td>
        </tr>""")

    rows_html = "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>GT Suspects Review — METER2800 ({len(suspects)} plików)</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #f9f9f9; }}
  h1 {{ color: #333; }}
  .summary {{ background: #fff3cd; border: 1px solid #ffc107; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #eee; vertical-align: middle; }}
  tr:hover td {{ filter: brightness(0.96); }}
  .vote {{ display: inline-block; margin: 2px; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-family: monospace; }}
  .vote-gt {{ background: #d4edda; color: #155724; }}
  .vote-pred {{ background: #f8d7da; color: #721c24; }}
  .vote-none {{ background: #e2e3e5; color: #383d41; }}
  .sig-block {{ display: inline-block; vertical-align: top; margin: 2px 6px 2px 0; padding: 3px 6px; background: #f8f9fa; border-radius: 4px; border: 1px solid #e9ecef; font-family: monospace; min-width: 90px; }}
  .sig-name {{ font-size: 10px; color: #666; display: block; margin-bottom: 1px; font-weight: bold; }}
  .stats {{ margin-top: 20px; padding: 12px; background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  button {{ margin: 10px 0; padding: 8px 16px; background: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #1e8449; }}
</style>
</head>
<body>
<h1>GT Suspects — METER2800 Test ({len(suspects)} plików)</h1>

<div class="summary">
  <strong>Kryteria:</strong> ≤1 sygnał zgadza się z GT, pewność arbitra &gt;0.6.<br>
  <strong>Dla każdego pliku:</strong> posłuchaj i wybierz werdykt — GT błędne czy poprawne.<br>
  <strong>Kolor wiersza:</strong> odpowiada etykiecie GT (czerwony=5/x, zielony=7/x, niebieski=3/x, żółty=4/x).<br>
  <strong>GT multi-label</strong> = plik ma wiele metrów w ground truth (z wikimeter.json) — np. 3/x + 4/x dla Fela Kuti.
</div>

<button onclick="exportVerdicts()">Eksportuj werdykty (JSON)</button>

<table>
<thead>
  <tr>
    <th>#</th>
    <th>Plik</th>
    <th>GT</th>
    <th>Arbiter predykcja</th>
    <th>All probs</th>
    <th>Sygnały</th>
    <th>Audio</th>
    <th>Werdykt</th>
  </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<div class="stats" id="stats"></div>

<script>
const data = {json.dumps([
    {
        "id": i,
        "fname": e["fname"],
        "fpath": str(e["fpath"]),
        "gt": e["gt"],
        "pred": e["pred"],
        "pred_conf": round(e["pred_conf"], 3),
    }
    for i, e in enumerate(suspects, 1)
], indent=2)};

function exportVerdicts() {{
    const verdicts = data.map(d => ({{
        ...d,
        verdict: document.getElementById('verdict_' + d.id)?.value || ''
    }}));
    const wrong = verdicts.filter(v => v.verdict === 'gt_wrong');
    const correct = verdicts.filter(v => v.verdict === 'gt_correct');
    const ambig = verdicts.filter(v => v.verdict === 'ambiguous');
    const undecided = verdicts.filter(v => !v.verdict);

    document.getElementById('stats').innerHTML = `
        <strong>Postęp:</strong> ${{verdicts.length - undecided.length}}/${{verdicts.length}} ocenionych<br>
        GT błędne: ${{wrong.length}} | GT poprawne: ${{correct.length}} | Niejednoznaczne: ${{ambig.length}} | Niezdecydowane: ${{undecided.length}}<br>
        <br><strong>Pliki z błędnym GT:</strong><br>
        ${{wrong.map(v => v.fpath).join('<br>')}}
    `;

    const blob = new Blob([JSON.stringify(verdicts, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gt_verdicts.json';
    a.click();
}}

// Live stats update
document.querySelectorAll('select').forEach(sel => {{
    sel.addEventListener('change', () => {{
        const verdicts = data.map(d => document.getElementById('verdict_' + d.id)?.value || '');
        const done = verdicts.filter(v => v).length;
        const wrong = verdicts.filter(v => v === 'gt_wrong').length;
        document.getElementById('stats').innerHTML =
            `<strong>Postęp:</strong> ${{done}}/${{data.length}} | GT błędne dotychczas: ${{wrong}}`;
    }});
}});
</script>
</body>
</html>"""

    return html


if __name__ == "__main__":
    print("Loading data and computing predictions...")
    suspects = build_suspects()
    print(f"Found {len(suspects)} suspect files.")

    # Local HTML (file:// paths)
    local_html = render_html(suspects, portable=False)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(local_html)
    print(f"Local HTML: {OUT_HTML}")

    # Portable zip (HTML + audio files)
    portable_html = render_html(suspects, portable=True)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("review.html", portable_html)
        for e in suspects:
            fpath = e["fpath"]
            zf.write(fpath, f"audio/{fpath.name}")
    print(f"Portable zip: {OUT_ZIP} ({len(suspects)} audio files)")
    print(f"Open locally: file://{OUT_HTML}")
