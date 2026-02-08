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

let startBtn;
let stopBtn;
let indicator;
let warmupBar;
let warmupFill;
let warmupText;
let errorMsg;
let resultsArea;
let liveOnsets;

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

  startBtn.addEventListener('click', startCapture);
  stopBtn.addEventListener('click', stopCapture);
  stopBtn.disabled = true;

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
      const fn = t('live.warmup_progress');
      warmupText.textContent = typeof fn === 'function' ? fn(msg.seconds, msg.total) : `${msg.seconds}s / ${msg.total}s`;
      break;
    }
    case 'onset': {
      // Flash a visual indicator
      if (liveOnsets) {
        const dot = document.createElement('span');
        dot.className = 'onset-dot';
        dot.style.cssText = `
          display: inline-block;
          width: 8px; height: 8px;
          border-radius: 50%;
          background: ${msg.strength > 0.6 ? '#e94560' : '#2ecc71'};
          margin: 2px;
          opacity: ${0.4 + msg.strength * 0.6};
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
      setAnalysisResult(result);
      warmupFill.style.width = '100%';
      warmupText.textContent = t('live.ready');
      _setLastResult(result);
      showResults(result);
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
