/**
 * File upload module: drag-and-drop + file input, POST to /api/analyze.
 * Includes loading step animation and user meter override.
 */

import { renderResults, renderMeterInfo, renderBeatGrid, renderMeterChart } from './visualizer.js';
import { setAnalysisResult, onReRender } from './app.js';
import { t, meterDesc } from './i18n.js';

let dropZone;
let fileInput;
let loadingOverlay;
let errorMsg;
let resultsArea;
let overrideSelect;
let uploadedFile = null;
let currentResult = null;

const GROUPINGS = {
  '5/4': '3+2', '5/8': '3+2',
  '7/4': '2+2+3', '7/8': '2+2+3',
  '9/8': '2+2+2+3',
  '10/8': '3+3+2+2',
  '11/8': '3+3+3+2',
};

export function init() {
  dropZone = document.getElementById('drop-zone');
  fileInput = document.getElementById('file-input');
  loadingOverlay = document.getElementById('upload-loading');
  errorMsg = document.getElementById('upload-error');
  resultsArea = document.getElementById('upload-results');
  overrideSelect = document.getElementById('upload-meter-override');

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener('change', handleFileSelect);

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  });

  if (overrideSelect) {
    overrideSelect.addEventListener('change', handleOverride);
  }

  // Re-render on language change
  onReRender(() => {
    if (currentResult && resultsArea.classList.contains('visible')) {
      showResults(currentResult);
    }
  });
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files.length > 0) {
    processFile(files[0]);
  }
}

function handleOverride() {
  if (!currentResult || !overrideSelect.value) return;

  const [num, den] = overrideSelect.value.split('/').map(Number);
  const sig = `${num}/${den}`;
  const desc = meterDesc(sig);
  const grouping = GROUPINGS[sig] || null;

  const overrideHypothesis = {
    numerator: num,
    denominator: den,
    confidence: 1.0,
    grouping,
    description: desc,
    disambiguation_hint: null,
  };

  const meterContainer = document.getElementById('upload-meter');
  if (meterContainer) {
    renderMeterInfo(meterContainer, overrideHypothesis);
  }

  const beatGridCanvas = document.getElementById('upload-beat-grid');
  if (beatGridCanvas) {
    renderBeatGrid(beatGridCanvas, num, grouping);
  }

  const meterChartCanvas = document.getElementById('upload-meter-chart');
  if (meterChartCanvas) {
    const combined = [overrideHypothesis, ...currentResult.meter_hypotheses.filter(
      h => !(h.numerator === num && h.denominator === den)
    )];
    renderMeterChart(meterChartCanvas, combined);
  }

  const disambig = document.getElementById('upload-disambiguation');
  if (disambig) {
    disambig.classList.remove('visible');
  }

  const overrideResult = {
    ...currentResult,
    meter_hypotheses: [overrideHypothesis, ...currentResult.meter_hypotheses],
  };
  setAnalysisResult(overrideResult);
}

async function processFile(file) {
  if (!file.type.startsWith('audio/')) {
    showError(t('upload.error_not_audio'));
    return;
  }

  if (file.size > 50 * 1024 * 1024) {
    showError(t('upload.error_too_large'));
    return;
  }

  uploadedFile = file;
  currentResult = null;
  hideError();
  showLoading();
  hideResults();

  if (overrideSelect) overrideSelect.value = '';

  dropZone.querySelector('p').textContent = file.name;

  try {
    animateLoadingSteps();

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `${t('upload.error_server')} (${response.status})`);
    }

    const result = await response.json();
    currentResult = result;
    setAnalysisResult(result);
    finishLoadingSteps();
    showResults(result);
  } catch (err) {
    showError(err.message || t('upload.error_generic'));
    dropZone.querySelector('p').textContent = t('upload.dropzone');
  } finally {
    hideLoading();
    fileInput.value = '';
  }
}

function showResults(result) {
  resultsArea.classList.add('visible');
  renderResults(result, {
    bpmContainer: document.getElementById('upload-bpm'),
    meterContainer: document.getElementById('upload-meter'),
    meterChartCanvas: document.getElementById('upload-meter-chart'),
    tempoCurveCanvas: document.getElementById('upload-tempo-curve'),
    beatGridCanvas: document.getElementById('upload-beat-grid'),
    waveformContainer: document.getElementById('upload-waveform'),
    disambiguationContainer: document.getElementById('upload-disambiguation'),
    sectionsTimeline: document.getElementById('upload-sections-timeline'),
    sectionsDetail: document.getElementById('upload-sections-detail'),
  }, uploadedFile);

  const sectionsCard = document.getElementById('upload-sections-card');
  if (sectionsCard && result.sections && result.sections.length <= 1) {
    sectionsCard.style.display = 'none';
  } else if (sectionsCard) {
    sectionsCard.style.display = '';
  }
}

let loadingStepTimer = null;

function animateLoadingSteps() {
  const steps = document.querySelectorAll('.loading-steps .step');
  steps.forEach(s => { s.classList.remove('active', 'done'); });

  let current = 0;
  if (steps.length > 0) steps[0].classList.add('active');

  loadingStepTimer = setInterval(() => {
    if (current < steps.length) {
      steps[current].classList.remove('active');
      steps[current].classList.add('done');
    }
    current++;
    if (current < steps.length) {
      steps[current].classList.add('active');
    } else {
      clearInterval(loadingStepTimer);
      loadingStepTimer = null;
    }
  }, 1500);
}

function finishLoadingSteps() {
  if (loadingStepTimer) {
    clearInterval(loadingStepTimer);
    loadingStepTimer = null;
  }
  const steps = document.querySelectorAll('.loading-steps .step');
  steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
}

function showLoading() {
  loadingOverlay.classList.add('active');
}

function hideLoading() {
  loadingOverlay.classList.remove('active');
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
