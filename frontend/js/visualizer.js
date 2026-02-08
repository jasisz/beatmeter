/**
 * Visualization module: WaveSurfer waveform, Chart.js charts, circular beat grid,
 * BPM display, sections timeline, disambiguation hints, and friendly meter descriptions.
 * All text goes through the i18n module.
 */

import { t, meterDesc, hintText, tempoCategoryLabel, tempoCategoryDesc } from './i18n.js';

let wavesurfer = null;
let meterChart = null;
let tempoChart = null;

/**
 * Initialise WaveSurfer on the given container element.
 */
export function initWaveSurfer(container) {
  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }
  wavesurfer = WaveSurfer.create({
    container,
    waveColor: '#4a5568',
    progressColor: '#e94560',
    cursorColor: '#e94560',
    height: 100,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    responsive: true,
    backend: 'WebAudio',
  });
  return wavesurfer;
}

/**
 * Load an audio file into WaveSurfer and add beat markers.
 */
export function loadWaveform(file, beats) {
  if (!wavesurfer) return;
  const url = URL.createObjectURL(file);
  wavesurfer.load(url);

  wavesurfer.on('ready', () => {
    wavesurfer.clearMarkers?.();
    if (beats && beats.length > 0) {
      addBeatMarkers(beats);
    }
  });
}

function addBeatMarkers(beats) {
  if (!wavesurfer) return;
  const wrapper = wavesurfer.getWrapper();
  const duration = wavesurfer.getDuration();
  if (!wrapper || !duration) return;

  wrapper.querySelectorAll('.beat-marker').forEach(el => el.remove());

  for (const beat of beats) {
    const pct = (beat.time / duration) * 100;
    const marker = document.createElement('div');
    marker.className = 'beat-marker';
    marker.style.cssText = `
      position: absolute;
      left: ${pct}%;
      top: 0;
      width: 2px;
      height: 100%;
      background: ${beat.is_downbeat ? '#e94560' : '#2ecc71'};
      opacity: ${0.4 + beat.strength * 0.6};
      pointer-events: none;
      z-index: 3;
    `;
    wrapper.appendChild(marker);
  }
}

/**
 * Render the large BPM display with tempo category badge.
 */
export function renderBpmDisplay(container, tempo) {
  const conf = Math.round(tempo.confidence * 100);
  const category = tempo.tempo_category || 'steady';
  const categoryLabel = tempoCategoryLabel(category);
  const categoryDescText = tempoCategoryDesc(category);

  const confText = t('bpm.confidence');
  const confStr = typeof confText === 'function' ? confText(conf) : `${conf}%`;

  let rangeHtml = '';
  if (tempo.is_variable && tempo.bpm_range) {
    rangeHtml = `<div class="bpm-confidence">${tempo.bpm_range[0].toFixed(0)}&ndash;${tempo.bpm_range[1].toFixed(0)} BPM</div>`;
  }

  container.innerHTML = `
    <div class="bpm-display">
      <div class="bpm-value">${tempo.bpm.toFixed(1)}</div>
      <div class="bpm-label">BPM</div>
      <div class="bpm-confidence">${confStr}</div>
      ${rangeHtml}
      <div class="tempo-badge ${category}" title="${categoryDescText}">${categoryLabel}</div>
    </div>
  `;
}

/**
 * Render meter info with a friendly description.
 */
export function renderMeterInfo(container, hypothesis) {
  const sig = `${hypothesis.numerator}/${hypothesis.denominator}`;
  const desc = meterDesc(sig);
  const conf = Math.round(hypothesis.confidence * 100);

  const confText = t('meter.confidence');
  const confStr = typeof confText === 'function' ? confText(conf) : `${conf}%`;

  let groupingHtml = '';
  if (hypothesis.grouping) {
    const gText = t('meter.grouping');
    const gStr = typeof gText === 'function' ? gText(hypothesis.grouping) : hypothesis.grouping;
    groupingHtml = `<div class="meter-grouping">${gStr}</div>`;
  }

  container.innerHTML = `
    <div class="meter-info">
      <div class="meter-signature">${sig}</div>
      <div class="meter-description">${desc}</div>
      ${groupingHtml}
      <div class="bpm-confidence">${confStr}</div>
    </div>
  `;
}

/**
 * Render disambiguation hint if available.
 */
export function renderDisambiguationHint(container, hypotheses) {
  if (!container) return;
  const top = hypotheses[0];
  if (top && top.disambiguation_hint) {
    const text = hintText(top.disambiguation_hint);
    if (text) {
      container.textContent = text;
      container.classList.add('visible');
      return;
    }
  }
  container.classList.remove('visible');
  container.textContent = '';
}

/**
 * Render a horizontal bar chart of meter hypotheses.
 */
export function renderMeterChart(canvas, hypotheses) {
  if (meterChart) {
    meterChart.destroy();
  }
  const labels = hypotheses.map(h => {
    const sig = `${h.numerator}/${h.denominator}`;
    const desc = meterDesc(sig);
    return desc && desc !== sig ? `${sig} - ${desc}` : sig;
  });
  const data = hypotheses.map(h => Math.round(h.confidence * 100));
  const colors = hypotheses.map((_, i) =>
    i === 0 ? '#e94560' : 'rgba(233,69,96,0.4)'
  );

  meterChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderRadius: 4,
        barThickness: 28,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const fn = t('chart.confidence_label');
              return typeof fn === 'function' ? fn(ctx.raw) : `${ctx.raw}%`;
            },
          },
        },
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: { color: '#aab', callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          ticks: {
            color: '#eee',
            font: { size: 12 },
          },
          grid: { display: false },
        },
      },
    },
  });
}

/**
 * Render a tempo curve (BPM over time).
 */
export function renderTempoCurve(canvas, tempoCurve) {
  if (!tempoCurve || tempoCurve.length === 0) return;
  if (tempoChart) {
    tempoChart.destroy();
  }

  tempoChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: tempoCurve.map(p => p.time.toFixed(1) + 's'),
      datasets: [{
        label: 'BPM',
        data: tempoCurve.map(p => p.bpm),
        borderColor: '#e94560',
        backgroundColor: 'rgba(233,69,96,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { color: '#aab', maxTicksLimit: 10 },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          ticks: { color: '#aab', callback: v => v + ' BPM' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
    },
  });
}

/**
 * Draw a circular beat grid (clock-like accent pattern visualization).
 */
export function renderBeatGrid(canvas, numerator, grouping) {
  const ctx = canvas.getContext('2d');
  const size = Math.min(canvas.parentElement.clientWidth, 240);
  canvas.width = size;
  canvas.height = size;

  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 24;

  ctx.clearRect(0, 0, size, size);

  const accents = new Set([0]);
  if (grouping) {
    const groups = grouping.split('+').map(Number);
    let pos = 0;
    for (const g of groups) {
      accents.add(pos);
      pos += g;
    }
  }

  for (let i = 0; i < numerator; i++) {
    const angle = (i / numerator) * Math.PI * 2 - Math.PI / 2;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    const isAccent = accents.has(i);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, isAccent ? 12 : 8, 0, Math.PI * 2);
    ctx.fillStyle = isAccent ? '#e94560' : '#0f3460';
    ctx.fill();
    ctx.strokeStyle = isAccent ? '#e94560' : '#2a2a4a';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#eee';
    ctx.font = `${isAccent ? 'bold ' : ''}11px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i + 1, x, y);
  }
}

/**
 * Render sections timeline and detail.
 */
export function renderSections(timelineContainer, detailContainer, sections, duration) {
  if (!timelineContainer || !sections || sections.length === 0) return;

  timelineContainer.innerHTML = '';
  for (const sec of sections) {
    const segment = document.createElement('div');
    segment.className = 'section-segment';
    const secDur = sec.end - sec.start;
    segment.style.flex = `${secDur}`;

    const label = sec.meter
      ? `${sec.meter.numerator}/${sec.meter.denominator}`
      : '?';
    segment.textContent = label;
    segment.title = `${formatTime(sec.start)} - ${formatTime(sec.end)}`;
    timelineContainer.appendChild(segment);
  }

  if (!detailContainer) return;
  detailContainer.innerHTML = '';
  for (const sec of sections) {
    const item = document.createElement('div');
    item.className = 'section-detail-item';

    const time = document.createElement('span');
    time.className = 'section-time';
    time.textContent = `${formatTime(sec.start)} - ${formatTime(sec.end)}`;
    item.appendChild(time);

    if (sec.meter) {
      const meterBadge = document.createElement('span');
      meterBadge.className = 'section-meter-badge';
      meterBadge.textContent = `${sec.meter.numerator}/${sec.meter.denominator}`;
      item.appendChild(meterBadge);
    }

    if (sec.tempo) {
      const tempoBadge = document.createElement('span');
      tempoBadge.className = 'section-tempo-badge';
      tempoBadge.textContent = `${sec.tempo.bpm} BPM`;
      item.appendChild(tempoBadge);
    }

    detailContainer.appendChild(item);
  }
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * Render all visualizations from an analysis result.
 */
export function renderResults(result, containers, audioFile) {
  const { bpmContainer, meterContainer, meterChartCanvas,
          tempoCurveCanvas, beatGridCanvas, waveformContainer,
          disambiguationContainer, sectionsTimeline, sectionsDetail } = containers;

  if (bpmContainer && result.tempo) {
    renderBpmDisplay(bpmContainer, result.tempo);
  }

  if (meterContainer && result.meter_hypotheses && result.meter_hypotheses.length > 0) {
    renderMeterInfo(meterContainer, result.meter_hypotheses[0]);
  }

  if (disambiguationContainer && result.meter_hypotheses) {
    renderDisambiguationHint(disambiguationContainer, result.meter_hypotheses);
  }

  if (meterChartCanvas && result.meter_hypotheses && result.meter_hypotheses.length > 0) {
    renderMeterChart(meterChartCanvas, result.meter_hypotheses);
  }

  if (tempoCurveCanvas && result.tempo_curve) {
    renderTempoCurve(tempoCurveCanvas, result.tempo_curve);
  }

  if (beatGridCanvas && result.meter_hypotheses && result.meter_hypotheses.length > 0) {
    const top = result.meter_hypotheses[0];
    renderBeatGrid(beatGridCanvas, top.numerator, top.grouping);
  }

  if (sectionsTimeline && result.sections && result.sections.length > 0) {
    renderSections(sectionsTimeline, sectionsDetail, result.sections, result.duration);
  }

  if (waveformContainer && audioFile) {
    if (!wavesurfer) {
      initWaveSurfer(waveformContainer);
    }
    loadWaveform(audioFile, result.beats);
  }
}

export function getDescription(numerator, denominator) {
  return meterDesc(`${numerator}/${denominator}`);
}

export function destroy() {
  if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
  if (meterChart) { meterChart.destroy(); meterChart = null; }
  if (tempoChart) { tempoChart.destroy(); tempoChart = null; }
}
