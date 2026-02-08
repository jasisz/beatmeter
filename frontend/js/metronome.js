/**
 * Web Audio API metronome with precise lookahead scheduling,
 * configurable BPM/meter, visual beat indicator, tap tempo, and volume.
 */

import { getAnalysisResult } from './app.js';

let audioCtx = null;
let gainNode = null;
let isPlaying = false;
let currentBeat = 0;
let timerId = null;
let nextNoteTime = 0;

// Scheduling parameters
const LOOKAHEAD = 25;       // ms between scheduler calls
const SCHEDULE_AHEAD = 0.1; // seconds to schedule ahead

// State
let bpm = 120;
let numerator = 4;
let denominator = 4;

// Tap tempo
const tapTimes = [];

// DOM
let bpmSlider, bpmInput, meterSelect, volumeSlider;
let startBtn, stopBtn, tapBtn, applyBtn;
let beatContainer;

export function init() {
  bpmSlider = document.getElementById('met-bpm-slider');
  bpmInput = document.getElementById('met-bpm-input');
  meterSelect = document.getElementById('met-meter');
  volumeSlider = document.getElementById('met-volume');
  startBtn = document.getElementById('met-start');
  stopBtn = document.getElementById('met-stop');
  tapBtn = document.getElementById('met-tap');
  applyBtn = document.getElementById('met-apply');
  beatContainer = document.getElementById('met-beats');

  // Initial values
  bpmSlider.value = bpm;
  bpmInput.value = bpm;

  // BPM slider/input sync
  bpmSlider.addEventListener('input', () => {
    bpm = Number(bpmSlider.value);
    bpmInput.value = bpm;
  });
  bpmInput.addEventListener('change', () => {
    bpm = clampBpm(Number(bpmInput.value));
    bpmInput.value = bpm;
    bpmSlider.value = bpm;
  });

  // Meter
  meterSelect.addEventListener('change', () => {
    parseMeter(meterSelect.value);
    buildBeatDots();
  });
  parseMeter(meterSelect.value);

  // Volume
  volumeSlider.addEventListener('input', () => {
    if (gainNode) gainNode.gain.value = Number(volumeSlider.value);
  });

  // Controls
  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
  stopBtn.disabled = true;

  // Tap tempo
  tapBtn.addEventListener('click', handleTap);

  // Apply detected
  applyBtn.addEventListener('click', applyDetected);

  buildBeatDots();
}

function parseMeter(value) {
  const [n, d] = value.split('/').map(Number);
  numerator = n || 4;
  denominator = d || 4;
}

function clampBpm(v) {
  return Math.max(40, Math.min(300, Math.round(v)));
}

function buildBeatDots() {
  beatContainer.innerHTML = '';
  for (let i = 0; i < numerator; i++) {
    const dot = document.createElement('div');
    dot.className = 'beat-dot' + (i === 0 ? ' accent-beat' : '');
    dot.textContent = i + 1;
    beatContainer.appendChild(dot);
  }
}

function start() {
  if (isPlaying) return;

  audioCtx = new AudioContext();
  gainNode = audioCtx.createGain();
  gainNode.gain.value = Number(volumeSlider.value);
  gainNode.connect(audioCtx.destination);

  isPlaying = true;
  currentBeat = 0;
  nextNoteTime = audioCtx.currentTime;

  startBtn.disabled = true;
  stopBtn.disabled = false;

  scheduler();
}

function stop() {
  isPlaying = false;
  if (timerId) {
    clearTimeout(timerId);
    timerId = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
    gainNode = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;

  // Reset visual
  const dots = beatContainer.querySelectorAll('.beat-dot');
  dots.forEach(d => d.classList.remove('active'));
}

function scheduler() {
  while (nextNoteTime < audioCtx.currentTime + SCHEDULE_AHEAD) {
    scheduleNote(currentBeat, nextNoteTime);
    scheduleVisual(currentBeat, nextNoteTime);
    advanceBeat();
  }
  timerId = setTimeout(scheduler, LOOKAHEAD);
}

function scheduleNote(beat, time) {
  const osc = audioCtx.createOscillator();
  const env = audioCtx.createGain();
  osc.connect(env);
  env.connect(gainNode);

  const isAccent = beat === 0;
  osc.frequency.value = isAccent ? 880 : 440;
  osc.type = 'sine';

  env.gain.setValueAtTime(isAccent ? 1.0 : 0.6, time);
  env.gain.exponentialRampToValueAtTime(0.001, time + 0.08);

  osc.start(time);
  osc.stop(time + 0.08);
}

function scheduleVisual(beat, time) {
  const delay = Math.max(0, (time - audioCtx.currentTime) * 1000);
  setTimeout(() => {
    const dots = beatContainer.querySelectorAll('.beat-dot');
    dots.forEach(d => d.classList.remove('active'));
    if (dots[beat]) {
      dots[beat].classList.add('active');
    }
  }, delay);
}

function advanceBeat() {
  // Seconds per beat depends on denominator:
  // quarter-note-based: 60/bpm
  // eighth-note-based (x/8): 30/bpm
  const secondsPerBeat = denominator === 8 ? 30 / bpm : 60 / bpm;
  nextNoteTime += secondsPerBeat;
  currentBeat = (currentBeat + 1) % numerator;
}

function handleTap() {
  const now = performance.now();
  tapTimes.push(now);

  // Only keep recent taps (last 8)
  while (tapTimes.length > 8) tapTimes.shift();
  // Need at least 2 taps
  if (tapTimes.length < 2) return;

  // Discard taps older than 3 seconds from the last tap
  const cutoff = now - 3000;
  while (tapTimes.length > 1 && tapTimes[0] < cutoff) tapTimes.shift();
  if (tapTimes.length < 2) return;

  // Compute average interval
  let totalInterval = 0;
  for (let i = 1; i < tapTimes.length; i++) {
    totalInterval += tapTimes[i] - tapTimes[i - 1];
  }
  const avgMs = totalInterval / (tapTimes.length - 1);
  const tappedBpm = clampBpm(Math.round(60000 / avgMs));

  bpm = tappedBpm;
  bpmSlider.value = bpm;
  bpmInput.value = bpm;
}

function applyDetected() {
  const result = getAnalysisResult();
  if (!result) return;

  if (result.tempo && result.tempo.bpm) {
    bpm = clampBpm(Math.round(result.tempo.bpm));
    bpmSlider.value = bpm;
    bpmInput.value = bpm;
  }

  if (result.meter_hypotheses && result.meter_hypotheses.length > 0) {
    const top = result.meter_hypotheses[0];
    const sig = `${top.numerator}/${top.denominator}`;
    // Try to find matching option
    const option = [...meterSelect.options].find(o => o.value === sig);
    if (option) {
      meterSelect.value = sig;
    } else {
      // Add custom option
      const opt = document.createElement('option');
      opt.value = sig;
      opt.textContent = sig;
      meterSelect.appendChild(opt);
      meterSelect.value = sig;
    }
    parseMeter(sig);
    buildBeatDots();
  }

  // Restart if playing to apply new settings
  if (isPlaying) {
    stop();
    start();
  }
}
