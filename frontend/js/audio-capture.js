/**
 * Live audio capture module: mic -> AudioWorklet -> WebSocket -> display results.
 */

import { renderResults, renderBpmDisplay, renderMeterInfo } from './visualizer.js';
import { setAnalysisResult, onReRender } from './app.js';
import { t } from './i18n.js';

let audioCtx = null;
let micStream = null;
let workletNode = null;
let ws = null;
let isRecording = false;

// Minimum input guard state
let totalOnsetCount = 0;
let totalDuration = 0;
const MIN_ONSETS = 8;
const MIN_DURATION = 6; // seconds
let inputGuardMet = false;

let startBtn;
let stopBtn;
let indicator;
let warmupBar;
let warmupFill;
let warmupText;
let errorMsg;
let resultsArea;
let liveOnsets;
let hearItBtn;
let guardMessage;

export function init() {
  startBtn = document.getElementById('live-start');
  stopBtn = document.getElementById('live-stop');
  indicator = document.getElementById('recording-indicator');
  warmupBar = document.getElementById('live-warmup-bar');
  warmupFill = document.getElementById('live-warmup-fill');
  warmupText = document.getElementById('live-warmup-text');
  errorMsg = document.getElementById('live-error');
  resultsArea = document.getElementById('live-results');
  liveOnsets = document.getElementById('live-onsets');
  hearItBtn = document.getElementById('live-hear-it');

  guardMessage = document.getElementById('live-guard-message');

  startBtn.addEventListener('click', startCapture);
  stopBtn.addEventListener('click', stopCapture);
  stopBtn.disabled = true;

  // "Hear It" button: switch to metronome tab and apply detected result
  if (hearItBtn) {
    hearItBtn.addEventListener('click', () => {
      // Switch to metronome tab
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      const metTab = document.querySelector('[data-tab="tab-metronome"]');
      if (metTab) metTab.classList.add('active');
      document.getElementById('tab-metronome').classList.add('active');
      // Click "Apply Detected" in metronome
      const applyBtn = document.getElementById('met-apply');
      if (applyBtn && !applyBtn.disabled) applyBtn.click();
    });
  }

  // Re-render on language change
  let lastResult = null;
  onReRender(() => {
    if (lastResult && resultsArea.classList.contains('visible')) {
      showResults(lastResult);
    }
  });
  // Expose setter for lastResult
  _setLastResult = (r) => { lastResult = r; };
}

let _setLastResult = () => {};

async function startCapture() {
  hideError();
  hideResults();
  // Reset input guard state
  totalOnsetCount = 0;
  totalDuration = 0;
  inputGuardMet = false;
  if (guardMessage) {
    guardMessage.textContent = '';
    guardMessage.classList.remove('visible');
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showError(t('live.error_mic'));
    return;
  }

  try {
    audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(micStream);

    await audioCtx.audioWorklet.addModule('./js/worklet/audio-processor.js');
    workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');

    source.connect(workletNode);
    // Worklet does not need to be connected to destination (no playback)

    // Open WebSocket
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/api/ws/live`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      isRecording = true;
      updateUI();
      warmupFill.style.width = '0%';
      warmupText.textContent = t('live.warmup');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch { /* ignore non-JSON */ }
    };

    ws.onerror = () => {
      showError(t('live.error_ws'));
      stopCapture();
    };

    ws.onclose = () => {
      if (isRecording) {
        stopCapture();
      }
    };

    // Forward audio chunks to WebSocket
    workletNode.port.onmessage = (event) => {
      if (event.data.type === 'chunk' && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data.data.buffer);
      }
    };

  } catch (err) {
    showError(t('live.error_capture') + err.message);
    stopCapture();
  }
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case 'warmup_progress': {
      const pct = Math.min(100, (msg.seconds / msg.total) * 100);
      warmupFill.style.width = pct + '%';
      totalOnsetCount = msg.onset_count || 0;
      totalDuration = msg.seconds || 0;

      const fn = t('live.warmup_progress');
      let text = typeof fn === 'function' ? fn(msg.seconds, msg.total) : `${msg.seconds}s / ${msg.total}s`;
      if (msg.onset_count > 0) {
        const onsetFn = t('live.onset_count');
        const onsetText = typeof onsetFn === 'function' ? onsetFn(msg.onset_count) : `Detected ${msg.onset_count} onsets...`;
        text += ' \u2014 ' + onsetText;
      }
      warmupText.textContent = text;

      // No-onset detection: 6+ seconds but 0 onsets
      if (guardMessage && totalDuration >= MIN_DURATION && totalOnsetCount === 0) {
        guardMessage.textContent = t('live.no_onsets');
        guardMessage.classList.add('visible');
      } else if (guardMessage && totalOnsetCount > 0) {
        guardMessage.classList.remove('visible');
      }
      break;
    }
    case 'onset': {
      // Track total onsets for minimum input guard
      totalOnsetCount++;
      // Flash a visual indicator - dot size varies by onset strength (6-14px)
      if (liveOnsets) {
        const dot = document.createElement('span');
        dot.className = 'onset-dot';
        const size = Math.round(6 + msg.strength * 8);
        dot.style.cssText = `
          display: inline-block;
          width: ${size}px; height: ${size}px;
          border-radius: 50%;
          background: ${msg.strength > 0.6 ? '#e94560' : '#2ecc71'};
          margin: 2px;
          opacity: ${0.4 + msg.strength * 0.6};
          vertical-align: middle;
        `;
        liveOnsets.appendChild(dot);
        // Keep only the last 60 dots
        while (liveOnsets.children.length > 60) {
          liveOnsets.removeChild(liveOnsets.firstChild);
        }
      }
      break;
    }
    case 'analysis': {
      const result = msg.data;
      // Mark as live source for frontend display logic
      result.source_type = 'live';

      // Minimum input guard: need 8+ onsets and 6+ seconds
      if (!inputGuardMet) {
        if (totalOnsetCount < MIN_ONSETS || totalDuration < MIN_DURATION) {
          // Show "keep tapping" message instead of results
          if (guardMessage) {
            guardMessage.textContent = t('live.keep_tapping');
            guardMessage.classList.add('visible');
          }
          warmupFill.style.width = '100%';
          warmupText.textContent = t('live.getting_groove');
          break;
        }
        inputGuardMet = true;
        if (guardMessage) {
          guardMessage.classList.remove('visible');
        }
      }

      setAnalysisResult(result);
      warmupFill.style.width = '100%';
      warmupText.textContent = t('live.ready');
      _setLastResult(result);
      showResults(result);
      updateConvergenceBadge(result.converged);
      break;
    }
  }
}

function showResults(result) {
  resultsArea.classList.add('visible');
  renderResults(result, {
    bpmContainer: document.getElementById('live-bpm'),
    meterContainer: document.getElementById('live-meter'),
    meterChartCanvas: document.getElementById('live-meter-chart'),
    tempoCurveCanvas: null,
    beatGridCanvas: document.getElementById('live-beat-grid'),
    waveformContainer: null,
    disambiguationContainer: document.getElementById('live-disambiguation'),
    sectionsTimeline: null,
    sectionsDetail: null,
  }, null);
}

function updateConvergenceBadge(converged) {
  let badge = document.getElementById('live-convergence-badge');
  if (!badge) {
    badge = document.createElement('span');
    badge.id = 'live-convergence-badge';
    // Insert badge next to the warmup text area
    warmupText.parentNode.insertBefore(badge, warmupText.nextSibling);
  }
  if (converged) {
    badge.textContent = t('live.converged');
    badge.className = 'convergence-badge converged';
  } else {
    badge.textContent = t('live.analyzing');
    badge.className = 'convergence-badge analyzing';
  }
}

function stopCapture() {
  isRecording = false;

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(tr => tr.stop());
    micStream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }

  updateUI();
}

function updateUI() {
  startBtn.disabled = isRecording;
  stopBtn.disabled = !isRecording;
  if (isRecording) {
    indicator.classList.add('active');
  } else {
    indicator.classList.remove('active');
  }
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add('visible');
}

function hideError() {
  errorMsg.classList.remove('visible');
}

function hideResults() {
  resultsArea.classList.remove('visible');
}
