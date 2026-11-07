'use strict';
const $ = (id) => document.getElementById(id);
const labels = ['negative', 'neutral', 'positive'];
const names = {minilm_v2:'MiniLM v2 · fine-tuned', linearsvc_v2:'LinearSVC v2 · reconstructed', qwen3_4b:'Qwen3 4B · local zero-shot'};
const examples = {negative:'Das Unternehmen meldet einen Gewinneinbruch und kündigt weitere Entlassungen an.', neutral:'The company will publish its annual report on Tuesday.', positive:'The company reported record earnings and raised its outlook.'};
let busy = false;
function modelQuery() { return $('model-choice').value ? '?model=' + encodeURIComponent($('model-choice').value) : ''; }
$('model-choice').addEventListener('change', checkHealth);
for (const radio of document.querySelectorAll('[name=source]')) {
  radio.addEventListener('change', () => {
    const isText = document.querySelector('[name=source]:checked').value === 'text';
    $('text-input').hidden = !isText; $('url-input').hidden = isText;
    $('text').disabled = !isText; $('text').required = isText;
    $('url').disabled = isText; $('url').required = !isText;
    $('error').hidden = true;
  });
}
for (const button of document.querySelectorAll('[data-example]')) {
  button.addEventListener('click', () => { $('text').value = examples[button.dataset.example]; $('text').focus(); });
}
async function checkHealth() {
  try {
    const response = await fetch('/health' + modelQuery(), {cache:'no-store'});
    if (!response.ok) throw new Error('Health request failed');
    const health = await response.json();
    $('health').textContent = health.model_loaded ? '● Local model ready' : health.model_available ? '● Ready to load' : '● Model unavailable';
    $('health').dataset.state = health.model_available ? 'ok' : 'error';
    $('loaded-model').textContent = health.model_loaded ? `Loaded: ${health.model_version}` : health.model_available ? 'Local artifact available. It will load when you classify.' : 'Weights are unavailable. Start the documented local model setup; requests will return 503.';
  } catch (_) { $('health').textContent = '● Service unreachable'; $('health').dataset.state = 'error'; }
}
$('predict-form').addEventListener('submit', (event) => { event.preventDefault(); runPrediction('/predict'); });
$('highlight').addEventListener('click', () => { if ($('predict-form').reportValidity()) runPrediction('/analyze'); });
async function runPrediction(endpoint) {
  if (busy) return;
  const source = document.querySelector('[name=source]:checked').value;
  const value = $(source).value.trim();
  if (!value) { $('error').textContent = 'Enter text or one public article URL.'; $('error').hidden = false; return; }
  busy = true; $('submit').disabled = true; $('highlight').disabled = true;
  $('model-choice').disabled = true;
  $('submit').textContent = endpoint === '/analyze' ? 'Analysing sentences…' : source === 'url' ? 'Fetching and classifying…' : 'Classifying…';
  $('sentence-section').hidden = true;
  $('error').hidden = true; $('prediction').hidden = true; $('result-title').textContent = 'Working…'; delete $('result-title').dataset.label;
  $('result-help').textContent = source === 'url' ? 'Checking access and fetching the article.' : 'Running inference on this machine.';
  const started = performance.now();
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(endpoint + modelQuery(), {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({[source]:value}), signal:controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : (data.detail || []).map((d) => d.msg).join('; ');
      throw new Error(`HTTP ${response.status}: ${detail || 'Request failed'}`);
    }
    if (!labels.includes(data.label) || !labels.every((label) => Number.isFinite(data.scores?.[label]))) throw new Error('The service returned an invalid prediction.');
    $('result-title').textContent = data.label[0].toUpperCase() + data.label.slice(1); $('result-title').dataset.label = data.label;
    $('result-help').textContent = 'Sentiment of the reported outcome.';
    $('score-bars').replaceChildren();
    for (const label of labels) {
      const row = document.createElement('div'); row.className = 'score-row';
      const heading = document.createElement('div'); heading.className = 'score-label';
      const name = document.createElement('span'); name.textContent = label;
      const value = document.createElement('span'); value.textContent = (100 * data.scores[label]).toFixed(1) + '%';
      heading.append(name, value);
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.dataset.label = label;
      fill.style.width = Math.max(0, Math.min(100, 100 * data.scores[label])) + '%';
      track.append(fill); row.append(heading, track); $('score-bars').append(row);
    }
    $('model-version').textContent = data.model_version;
    $('request-time').textContent = `${(performance.now() - started).toFixed(0)} ms including HTTP${source === 'url' ? ' and fetching' : ''}`;
    $('excerpt').textContent = data.scraped_excerpt || ''; $('excerpt').hidden = !data.scraped_excerpt;
    $('prediction').hidden = false;
    if (endpoint === '/analyze') renderHighlights(data);
  } catch (error) {
    $('error').textContent = error.name === 'AbortError' ? 'The request exceeded 90 seconds. The service may still be finishing it; check health before retrying.' : error.message;
    $('error').hidden = false; $('result-title').textContent = 'No result'; $('result-help').textContent = 'Resolve the request error and try again.';
  } finally { clearTimeout(timeout); busy = false; $('submit').disabled = false; $('highlight').disabled = false; $('model-choice').disabled = false; $('submit').textContent = 'Classify article'; await checkHealth(); }
}
function renderHighlights(data) {
  const container = $('highlighted-text'); container.replaceChildren();
  $('sentence-scores').textContent = 'Select a coloured sentence to inspect its scores.';
  // Python offsets count Unicode code points; Array.from also handles emoji correctly.
  const characters = Array.from(data.text); let cursor = 0;
  for (const sentence of data.sentences) {
    container.append(document.createTextNode(characters.slice(cursor, sentence.start).join('')));
    const mark = document.createElement('mark'); mark.dataset.label = sentence.label; mark.tabIndex = 0;
    const description = `${sentence.label}: ` + labels.map((label) => `${label} ${(100 * sentence.scores[label]).toFixed(1)}%`).join(', ');
    mark.textContent = characters.slice(sentence.start, sentence.end).join('');
    mark.title = description; mark.setAttribute('aria-label', `${mark.textContent} — ${description}`);
    for (const event of ['focus', 'mouseenter', 'click']) mark.addEventListener(event, () => { $('sentence-scores').textContent = description; });
    container.append(mark); cursor = sentence.end;
  }
  container.append(document.createTextNode(characters.slice(cursor).join('')));
  $('highlight-limit').hidden = !data.highlighting_limited;
  $('highlight-limit').textContent = data.highlighting_limited ? `Highlighted the first ${data.sentences.length} of ${data.total_sentences} sentences. Remaining text is uncoloured and has not been classified separately.` : '';
  $('sentence-section').hidden = false;
}
async function comparison() {
  try {
    const response = await fetch('/comparison', {cache:'no-store'});
    if (!response.ok) throw new Error('Recorded comparison is unavailable.');
    const data = await response.json();
    $('selected-model').textContent = `Selected by the recorded benchmark: ${names[data.selected] || data.selected}.`;
    $('comparison-rows').replaceChildren();
    for (const record of data.results) {
      const row = document.createElement('tr'); row.dataset.selected = String(record.model === data.selected);
      for (const value of [names[record.model] || record.model, record.macro_f1_en.toFixed(4), record.macro_f1_de.toFixed(4), record.inference_ms_per_text.toFixed(3), (record.model_size_bytes / 1048576).toFixed(2)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      $('comparison-rows').append(row);
    }
  } catch (error) { $('selected-model').textContent = error.message; $('comparison-chart').hidden = true; }
}
async function loadModels() {
  try {
    const response = await fetch('/models'); if (!response.ok) throw new Error('Model list unavailable');
    const data = await response.json();
    for (const model of data.models) {
      const option = document.createElement('option'); option.value = model.id;
      option.textContent = model.name + (model.is_default ? ' · default' : '') + (!model.available ? ' · unavailable' : '');
      option.disabled = !model.available; $('model-choice').append(option);
    }
    const requested = new URLSearchParams(window.location.search).get('model');
    if (requested && data.models.some((model) => model.id === requested && model.available)) {
      $('model-choice').value = requested; await checkHealth();
    }
  } catch (_) { /* The default predictor still works when model discovery fails. */ }
}
async function newsComparison() {
  try {
    const response = await fetch('/comparison/news-v3', {cache:'no-store'});
    if (!response.ok) throw new Error('AI news pilot results unavailable.');
    const data = await response.json();
    $('news-status').textContent = `${data.test_rows} shared test texts · AI-labelled · Codex-reviewed. The serving default remains Qwen.`;
    $('news-rows').replaceChildren();
    for (const record of data.results) {
      const row = document.createElement('tr');
      row.dataset.selected = String(record.model === data.pilot_leader);
      for (const value of [record.name, record.balanced_language_macro_f1.toFixed(4), record.macro_f1_de.toFixed(4), record.macro_f1_en.toFixed(4), record.inference_ms_per_text.toFixed(3), (record.model_size_bytes / 1048576).toFixed(2)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      $('news-rows').append(row);
    }
  } catch (error) { $('news-status').textContent = error.message; $('news-chart').hidden = true; }
}
async function expandedComparison() {
  try {
    const response = await fetch('/comparison/news-v4', {cache:'no-store'});
    if (!response.ok) throw new Error('Expanded data results unavailable.');
    const data = await response.json();
    const winner = data.results.find((row) => row.model === data.descriptive_leader);
    $('expanded-status').textContent = `${data.test_rows} shared test texts · best experimental choice: ${winner?.name || data.descriptive_leader}. Qwen remains the benchmark default.`;
    $('expanded-rows').replaceChildren();
    for (const record of data.results) {
      const row = document.createElement('tr'); row.dataset.selected = String(record.model === data.descriptive_leader);
      for (const value of [record.name, record.balanced_language_macro_f1.toFixed(4), record.macro_f1_de.toFixed(4), record.macro_f1_en.toFixed(4), (100 * record.recall_positive).toFixed(1) + '%', record.inference_ms_per_text.toFixed(3), (record.model_size_bytes / 1048576).toFixed(2)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      $('expanded-rows').append(row);
    }
  } catch (error) { $('expanded-status').textContent = error.message; $('expanded-chart').hidden = true; }
}
checkHealth(); loadModels(); comparison(); newsComparison(); expandedComparison();
