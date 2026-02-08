/**
 * Main application controller: tab switching, shared state, module initialization, i18n.
 */

import { init as initFileUpload } from './file-upload.js';
import { init as initAudioCapture } from './audio-capture.js';
import { init as initMetronome } from './metronome.js';
import { initLangSwitcher, applyStaticTranslations, onLangChange } from './i18n.js';

// Shared analysis result (set by upload or live modules, read by metronome)
let currentAnalysisResult = null;

// Re-render callbacks for language change
let reRenderCallbacks = [];

export function setAnalysisResult(result) {
  currentAnalysisResult = result;
  // Enable the "Apply detected" button in metronome
  const applyBtn = document.getElementById('met-apply');
  if (applyBtn) {
    applyBtn.disabled = !result;
  }
}

export function getAnalysisResult() {
  return currentAnalysisResult;
}

/**
 * Register a callback to re-render dynamic content on language change.
 */
export function onReRender(cb) {
  reRenderCallbacks.push(cb);
}

function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(target).classList.add('active');
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  // Init i18n first
  initLangSwitcher();
  applyStaticTranslations();

  // On language change, re-apply static translations and re-render dynamic content
  onLangChange(() => {
    applyStaticTranslations();
    reRenderCallbacks.forEach(cb => cb());
  });

  initTabs();
  initFileUpload();
  initAudioCapture();
  initMetronome();
});
