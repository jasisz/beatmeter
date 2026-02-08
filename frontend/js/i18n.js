/**
 * Internationalization module.
 * Supports PL and EN. Language preference stored in localStorage.
 */

const translations = {
  // ===== Header =====
  'header.subtitle': {
    pl: 'Wykryj tempo, metrum i wzorce rytmiczne w swojej muzyce',
    en: 'Detect tempo, time signature, and rhythm patterns in your music',
  },

  // ===== Tabs =====
  'tab.upload': { pl: 'Upload', en: 'Upload' },
  'tab.live': { pl: 'Live', en: 'Live' },
  'tab.metronome': { pl: 'Metronom', en: 'Metronome' },

  // ===== Upload =====
  'upload.dropzone': {
    pl: 'Upusc plik audio tutaj lub kliknij, zeby wybrac',
    en: 'Drop an audio file here or click to browse',
  },
  'upload.hint': {
    pl: 'MP3, WAV, FLAC, OGG \u2014 do 50 MB',
    en: 'MP3, WAV, FLAC, OGG \u2014 up to 50 MB',
  },
  'upload.analyzing': { pl: 'Analizuje rytm...', en: 'Analyzing rhythm...' },
  'upload.step1': { pl: 'Wykrywanie uderzen', en: 'Onset detection' },
  'upload.step2': { pl: 'Sledzenie beatow', en: 'Beat tracking' },
  'upload.step3': { pl: 'Estymacja tempa', en: 'Tempo estimation' },
  'upload.step4': { pl: 'Analiza metrum', en: 'Meter analysis' },
  'upload.override_label': {
    pl: 'Wiesz lepiej? Ustaw metrum recznie:',
    en: 'Know better? Set meter manually:',
  },
  'upload.override_auto': {
    pl: '-- automatyczne --',
    en: '-- automatic --',
  },
  'upload.error_not_audio': {
    pl: 'Wybierz plik audio (MP3, WAV, FLAC, OGG...).',
    en: 'Please select an audio file (MP3, WAV, FLAC, OGG...).',
  },
  'upload.error_server': {
    pl: 'Blad serwera',
    en: 'Server error',
  },
  'upload.error_generic': {
    pl: 'Analiza nie powiodla sie. Sprobuj ponownie.',
    en: 'Analysis failed. Please try again.',
  },

  // ===== Section headings =====
  'heading.waveform': {
    pl: 'Przebieg audio i beaty',
    en: 'Waveform & Beats',
  },
  'heading.meter_hypotheses': {
    pl: 'Wykryte metrum',
    en: 'Detected Meter',
  },
  'heading.beat_grid': {
    pl: 'Siatka beatow',
    en: 'Beat Grid',
  },
  'heading.tempo_curve': {
    pl: 'Tempo w czasie',
    en: 'Tempo Over Time',
  },
  'heading.sections': {
    pl: 'Sekcje utworu',
    en: 'Song Sections',
  },

  // ===== Live =====
  'live.title': { pl: 'Analiza na zywo', en: 'Live Analysis' },
  'live.description': {
    pl: 'Uzyj mikrofonu, zeby analizowac rytm w czasie rzeczywistym. Stukaj w stol, klaskaj lub graj na instrumencie.',
    en: 'Use your microphone to analyze rhythm in real time. Tap on a table, clap, or play an instrument.',
  },
  'live.start': { pl: 'Zacznij sluchac', en: 'Start Listening' },
  'live.stop': { pl: 'Stop', en: 'Stop' },
  'live.warmup': { pl: 'Slucham... rozgrzewka', en: 'Listening... warming up' },
  'live.warmup_progress': {
    pl: (s, total) => `Rozgrzewka... ${s}s / ${total}s`,
    en: (s, total) => `Warming up... ${s}s / ${total}s`,
  },
  'live.ready': { pl: 'Analiza gotowa', en: 'Analysis ready' },
  'live.onset_count': {
    pl: (n) => `Wykryto ${n} uderzen...`,
    en: (n) => `Detected ${n} onsets...`,
  },
  'live.converged': { pl: 'Stabilne', en: 'Stable' },
  'live.analyzing': { pl: 'Analizuje...', en: 'Analyzing...' },
  'live.error_mic': {
    pl: 'Nie mozna uzyskac dostepu do mikrofonu. Zezwol na dostep i sprobuj ponownie.',
    en: 'Could not access microphone. Please allow mic access and try again.',
  },
  'live.error_ws': {
    pl: 'Blad polaczenia WebSocket. Czy serwer jest uruchomiony?',
    en: 'WebSocket connection error. Is the server running?',
  },
  'live.error_capture': {
    pl: 'Nie mozna rozpoczac przechwytywania audio: ',
    en: 'Could not start audio capture: ',
  },

  // ===== Metronome =====
  'met.title': { pl: 'Metronom', en: 'Metronome' },
  'met.tempo': { pl: 'Tempo (BPM)', en: 'Tempo (BPM)' },
  'met.meter': { pl: 'Metrum', en: 'Meter' },
  'met.volume': { pl: 'Glosnosc', en: 'Volume' },
  'met.start': { pl: 'Start', en: 'Start' },
  'met.stop': { pl: 'Stop', en: 'Stop' },
  'met.tap': { pl: 'Tap Tempo', en: 'Tap Tempo' },
  'met.apply': { pl: 'Zagraj to metrum', en: 'Play This Meter' },

  // ===== BPM display =====
  'bpm.confidence': {
    pl: (pct) => `Pewnosc: ${pct}%`,
    en: (pct) => `Confidence: ${pct}%`,
  },
  'meter.confidence': {
    pl: (pct) => `Pewnosc: ${pct}%`,
    en: (pct) => `Confidence: ${pct}%`,
  },
  'meter.grouping': {
    pl: (g) => `Grupowanie: ${g}`,
    en: (g) => `Grouping: ${g}`,
  },
  'chart.confidence_label': {
    pl: (pct) => `${pct}% pewnosci`,
    en: (pct) => `${pct}% confidence`,
  },

  // Confidence word badges
  'confidence.confident': { pl: 'Pewne', en: 'Confident' },
  'confidence.likely': { pl: 'Prawdopodobne', en: 'Likely' },
  'confidence.uncertain': { pl: 'Niepewne', en: 'Uncertain' },
  'confidence.not_sure': { pl: 'Nie jestem pewien', en: 'Not sure' },

  // "Hear It" button
  'live.hear_it': { pl: 'Posluchaj', en: 'Hear It' },

  // Minimum input guard
  'live.keep_tapping': {
    pl: 'Stukaj dalej! Potrzebuje jeszcze kilku taktow...',
    en: 'Keep tapping! I need a few more bars...',
  },

  // No onsets detected
  'live.no_onsets': {
    pl: 'Nie slysze zadnych uderzen. Upewnij sie, ze mikrofon dziala i stukaj mocniej.',
    en: "I don't hear any taps yet. Make sure your mic is working and tap firmly.",
  },

  // Low confidence fallback (<40%)
  'live.low_confidence_fallback': {
    pl: 'Jeszcze nie jestem pewien. Sprobuj stukac dluzej z wyrazniejszym akcentem na raz.',
    en: "I'm not sure yet. Try tapping a bit longer with clearer accents on beat 1.",
  },

  // "Could also be" for medium confidence
  'live.could_also_be': {
    pl: (sig) => `moze tez byc ${sig}`,
    en: (sig) => `could also be ${sig}`,
  },

  // Warmup groove label
  'live.getting_groove': {
    pl: 'Lapie rytm...',
    en: 'Getting your groove...',
  },

  // ===== Tempo categories =====
  'tempo.steady': { pl: 'Stale tempo', en: 'Steady tempo' },
  'tempo.slightly_variable': { pl: 'Lekko zmienne', en: 'Slightly variable' },
  'tempo.variable': { pl: 'Zmienne tempo', en: 'Variable tempo' },
  'tempo.rubato': { pl: 'Rubato / freestyle', en: 'Rubato / freestyle' },
  'tempo.steady_desc': {
    pl: 'Tempo jest stabilne i regularne.',
    en: 'Tempo is stable and regular.',
  },
  'tempo.slightly_variable_desc': {
    pl: 'Tempo jest w wiekszosci stale z drobnymi wahaniami.',
    en: 'Tempo is mostly steady with minor fluctuations.',
  },
  'tempo.variable_desc': {
    pl: 'Tempo zmienia sie wyraznie w trakcie utworu. Moze to byc celowy efekt artystyczny.',
    en: 'Tempo changes noticeably throughout the piece. This may be an intentional artistic effect.',
  },
  'tempo.rubato_desc': {
    pl: 'Tempo jest bardzo swobodne - typowe dla rubato, improwizacji lub freestyle.',
    en: 'Tempo is very free-flowing - typical of rubato, improvisation, or freestyle.',
  },

  // ===== Meter descriptions (lead with structure, examples secondary) =====
  'meter.2/4': { pl: 'Dwa uderzenia na takt \u2014 jak marsz', en: 'Two beats per bar \u2014 like a march' },
  'meter.3/4': { pl: 'Trzy uderzenia na takt \u2014 walc (np. Blue Danube)', en: 'Three beats per bar \u2014 waltz feel (e.g. Blue Danube)' },
  'meter.3/8': { pl: 'Trzy szybkie uderzenia na takt \u2014 gigue', en: 'Three fast beats per bar \u2014 gigue' },
  'meter.4/4': { pl: 'Cztery uderzenia na takt \u2014 najpopularniejsze metrum', en: 'Four beats per bar \u2014 the most common meter' },
  'meter.5/4': { pl: 'Piec uderzen na takt \u2014 grupowanie 3+2 lub 2+3 (np. Take Five)', en: 'Five beats per bar \u2014 grouped 3+2 or 2+3 (e.g. Take Five)' },
  'meter.5/8': { pl: 'Piec szybkich uderzen na takt \u2014 grupowanie 3+2 lub 2+3', en: 'Five fast beats per bar \u2014 grouped 3+2 or 2+3' },
  'meter.6/4': { pl: 'Szesc uderzen na takt \u2014 wolne, w dwoch grupach po 3', en: 'Six beats per bar \u2014 slow, in two groups of three' },
  'meter.6/8': { pl: 'Szesc uderzen na takt w dwoch grupach po trzy \u2014 kolyszacy feel (np. Nothing Else Matters)', en: 'Six beats per bar in two groups of three \u2014 a swaying feel (e.g. Nothing Else Matters)' },
  'meter.7/4': { pl: 'Siedem uderzen na takt \u2014 asymetryczne metrum (np. Money, Pink Floyd)', en: 'Seven beats per bar \u2014 asymmetric meter (e.g. Money by Pink Floyd)' },
  'meter.7/8': { pl: 'Siedem szybkich uderzen na takt \u2014 czesto 2+2+3 (np. Money, Pink Floyd)', en: 'Seven fast beats per bar \u2014 often grouped 2+2+3 (e.g. Money by Pink Floyd)' },
  'meter.9/8': { pl: 'Dziewiec uderzen na takt \u2014 czesto 2+2+2+3 (np. Blue Rondo a la Turk)', en: 'Nine beats per bar \u2014 often 2+2+2+3 (e.g. Blue Rondo a la Turk)' },
  'meter.10/8': { pl: 'Dziesiec uderzen na takt \u2014 rzadkie, asymetryczne metrum', en: 'Ten beats per bar \u2014 rare, asymmetric meter' },
  'meter.11/8': { pl: 'Jedenascie uderzen na takt \u2014 rzadkie, asymetryczne metrum', en: 'Eleven beats per bar \u2014 rare, asymmetric meter' },
  'meter.12/8': { pl: 'Dwanascie uderzen na takt w czterech grupach po 3 \u2014 blues shuffle', en: 'Twelve beats per bar in four groups of three \u2014 blues shuffle' },

  // ===== Disambiguation hints =====
  'hint.6_8_vs_3_4': {
    pl: '6/8 i 3/4 maja taka sama liczbe impulsow, ale inny podzial wewnetrzny. 6/8 = dwie grupy po 3 (raz-dwa-trzy-raz-dwa-trzy), 3/4 = trzy rowne uderzenia (raz-dwa-trzy). Sprobuj zaklaskac - jesli "naturalnie" dzielisz na dwie grupy, to 6/8.',
    en: '6/8 and 3/4 have the same number of pulses but different internal divisions. 6/8 = two groups of 3 (ONE-two-three-ONE-two-three), 3/4 = three equal beats (ONE-two-three). Try clapping along - if you naturally divide into two groups, it\'s 6/8.',
  },
  'hint.3_4_vs_6_8': {
    pl: '3/4 i 6/8 sa czesto mylone. 3/4 to trzy rowne beaty (walc: raz-dwa-trzy). 6/8 to dwie grupy po trzy osmki. Jesli czujesz "kolysanie", to 6/8.',
    en: '3/4 and 6/8 are often confused. 3/4 has three equal beats (waltz: ONE-two-three). 6/8 has two groups of three eighth notes. If you feel a "swaying" motion, it\'s 6/8.',
  },
  'hint.4_4_vs_2_4': {
    pl: '2/4 to w zasadzie 4/4 ze skroconym taktem. Jesli muzyka ma wyrazny akcent co 4 uderzenia, to raczej 4/4.',
    en: '2/4 is essentially 4/4 with a shorter bar. If the music has a clear accent every 4 beats, it\'s probably 4/4.',
  },
  'hint.2_4_vs_4_4': {
    pl: '2/4 czesto brzmi jak szybkie 4/4. Jesli czujesz 2 silne uderzenia w cyklu (jak marsz), to 2/4.',
    en: '2/4 often sounds like fast 4/4. If you feel 2 strong beats per cycle (like a march), it\'s 2/4.',
  },
  'hint.12_8_vs_4_4': {
    pl: '12/8 to "swingowe" 4/4 - kazdy beat jest podzielony na trojki. Typowe dla bluesa i slow rock. Jesli beat "kolysza sie", to 12/8.',
    en: '12/8 is "swung" 4/4 - each beat is divided into triplets. Typical of blues and slow rock. If the beats "swing", it\'s 12/8.',
  },

  // ===== Language switcher =====
  'lang.pl': { pl: 'PL', en: 'PL' },
  'lang.en': { pl: 'EN', en: 'EN' },
};

let currentLang = localStorage.getItem('beatmeter-lang') || 'pl';
let onChangeCallbacks = [];

/**
 * Get translation for a key. Returns the string for current language.
 * If the value is a function, returns the function (for parameterized strings).
 */
export function t(key) {
  const entry = translations[key];
  if (!entry) return key;
  return entry[currentLang] || entry['en'] || key;
}

/**
 * Get meter description by signature string (e.g. "4/4").
 */
export function meterDesc(sig) {
  return t(`meter.${sig}`) || sig;
}

/**
 * Get disambiguation hint by key.
 */
export function hintText(hintKey) {
  if (!hintKey) return null;
  return t(`hint.${hintKey}`);
}

/**
 * Get tempo category label.
 */
export function tempoCategoryLabel(category) {
  return t(`tempo.${category}`) || category;
}

/**
 * Get tempo category description.
 */
export function tempoCategoryDesc(category) {
  return t(`tempo.${category}_desc`) || '';
}

export function getLang() {
  return currentLang;
}

export function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('beatmeter-lang', lang);
  document.documentElement.lang = lang;
  onChangeCallbacks.forEach(cb => cb(lang));
}

/**
 * Register a callback for language changes.
 */
export function onLangChange(cb) {
  onChangeCallbacks.push(cb);
}

/**
 * Apply translations to all elements with data-i18n attribute.
 */
export function applyStaticTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const text = t(key);
    if (typeof text === 'string') {
      el.textContent = text;
    }
  });
  // Also handle placeholders
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    const text = t(key);
    if (typeof text === 'string') {
      el.placeholder = text;
    }
  });
}

/**
 * Initialize language switcher UI.
 */
export function initLangSwitcher() {
  const switcher = document.getElementById('lang-switcher');
  if (!switcher) return;

  const btns = switcher.querySelectorAll('.lang-btn');
  btns.forEach(btn => {
    const lang = btn.dataset.lang;
    if (lang === currentLang) btn.classList.add('active');

    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setLang(lang);
      applyStaticTranslations();
    });
  });

  document.documentElement.lang = currentLang;
}
