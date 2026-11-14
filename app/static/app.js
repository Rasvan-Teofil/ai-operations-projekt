'use strict';
const $ = (id) => document.getElementById(id);
const labels = ['negative', 'neutral', 'positive'];
const names = {minilm_v2:'MiniLM v2 · fine-tuned', linearsvc_v2:'LinearSVC v2 · reconstructed', qwen3_4b:'Qwen3 4B · local zero-shot'};
const examples = {negative:'Das Unternehmen meldet einen Gewinneinbruch und kündigt weitere Entlassungen an.', neutral:'The company will publish its annual report on Tuesday.', positive:'The company reported record earnings and raised its outlook.', japanese:'強い地震で多数の住宅が倒壊し、住民が負傷した。'};
let busy = false;
let modelRecords = [];
const articleErrors = {
  upstream_forbidden: 'The publisher refused automated access. Paste the article text to continue.',
  robots_denied: 'This site does not allow this article to be fetched automatically. Paste the text to continue.',
  robots_unavailable: 'The site’s access rules could not be checked. Try later or paste the article text.',
  upstream_timeout: 'The publisher took too long to respond. Try later or paste the text.',
  dns_failed: 'The article host could not be reached. Check the address or paste the text.',
  redirect_blocked: 'This address redirects to another page. Enter the final public article URL or paste the text.',
  access_page: 'The site returned a sign-in, access-check or error page. Paste the article text to continue.',
  extraction_failed: 'No readable article body was found in this layout. Paste the text to continue.',
  article_context_exceeded: 'This article exceeds the selected model’s full-text token limit. No text was classified or silently cut.',
  article_too_long: 'The article exceeds 20,000 characters. No text was classified or silently cut.',
  unsupported_content: 'The address did not return an HTML article. Paste the article text.',
  response_too_large: 'The publisher response is too large to fetch. Paste the article text.',
  transport_failed: 'The connection to the publisher failed. Try later or paste the text.',
  upstream_http: 'The publisher returned an error. Try later or paste the article text.'
};
function modelQuery() { return $('model-choice').value ? '?model=' + encodeURIComponent($('model-choice').value) : ''; }
$('model-choice').addEventListener('change', checkHealth);
$('paste-instead').addEventListener('click', () => {
  const radio = document.querySelector('[name=source][value=text]');
  radio.checked = true; radio.dispatchEvent(new Event('change')); $('text').focus();
});
for (const radio of document.querySelectorAll('[name=source]')) {
  radio.addEventListener('change', () => {
    const isText = document.querySelector('[name=source]:checked').value === 'text';
    $('text-input').hidden = !isText; $('url-input').hidden = isText;
    $('text').disabled = !isText; $('text').required = isText;
    $('url').disabled = isText; $('url').required = !isText;
    $('error').hidden = true;
    $('paste-instead').hidden = true;
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
    const selected = modelRecords.find((row) => row.id === ($('model-choice').value || health.model_id));
    $('context-policy').textContent = selected?.context_policy ? selected.context_policy + ' ' + (selected.evaluated_scope || '') : 'Full-article quality is being evaluated; current measured results use short texts.';
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
  $('error').hidden = true; $('paste-instead').hidden = true; $('prediction').hidden = true; $('result-title').textContent = 'Working…'; delete $('result-title').dataset.label;
  $('result-help').textContent = source === 'url' ? 'Checking access and fetching the article.' : 'Running inference on this machine.';
  const started = performance.now();
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(endpoint + modelQuery(), {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({[source]:value}), signal:controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : (data.detail || []).map((d) => d.msg).join('; ');
      const code = response.headers.get('X-Article-Error');
      throw new Error(articleErrors[code] || `HTTP ${response.status}: ${detail || 'Request failed'}`);
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
    $('paste-instead').hidden = source !== 'url';
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
    modelRecords = data.models;
    const qualityChoice = data.article_research_choice || 'article_qwen_lora_v1';
    const fastChoice = data.article_fast_alternative || 'article_minilm_v1';
    const suggested = document.createElement('optgroup'); suggested.label = 'Full-article study · DE / EN / JA';
    const archive = document.createElement('optgroup'); archive.label = 'More methods and historical data versions';
    for (const model of data.models.slice().sort((a, b) => {
      const order = [qualityChoice, fastChoice];
      const rank = (id) => order.includes(id) ? order.indexOf(id) : 2;
      return rank(a.id) - rank(b.id);
    })) {
      const option = document.createElement('option'); option.value = model.id;
      const role = model.id === qualityChoice ? ' · selected before Test' : model.id === fastChoice ? ' · fast alternative' : '';
      option.textContent = model.name + role + (model.is_default ? ' · API default' : '') + (!model.available ? ' · unavailable' : '');
      option.disabled = !model.available;
      (model.comparison_group === 'full-article-v1' ? suggested : archive).append(option);
    }
    $('model-choice').append(suggested, archive);
    const requested = new URLSearchParams(window.location.search).get('model');
    if (requested && data.models.some((model) => model.id === requested && model.available)) {
      $('model-choice').value = requested; await checkHealth();
    }
    else await checkHealth();
  } catch (_) { /* The default predictor still works when model discovery fails. */ }
}
async function newsComparison() {
  try {
    const response = await fetch('/comparison/news-v3', {cache:'no-store'});
    if (!response.ok) throw new Error('AI news pilot results unavailable.');
    const data = await response.json();
    $('news-status').textContent = `${data.test_rows} historical shared test texts · Project-reviewed references.`;
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
    $('expanded-status').textContent = `${data.test_rows} historical shared v4 test texts · choice on that Test: ${winner?.name || data.descriptive_leader}.`;
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
async function llmComparison() {
  try {
    const response = await fetch('/comparison/news-v5', {cache:'no-store'});
    if (!response.ok) throw new Error('Matched-task LLM results unavailable.');
    const data = await response.json();
    const chosen = data.results.find((row) => row.model === data.descriptive_leader);
    $('llm-status').textContent = `${data.test_rows} new shared test texts · Project-reviewed references · descriptive choice: ${chosen?.name || data.descriptive_leader}.`;
    $('llm-try').href = '/?model=' + encodeURIComponent(data.descriptive_leader);
    $('llm-rows').replaceChildren();
    for (const record of data.results) {
      const row = document.createElement('tr'); row.dataset.selected = String(record.model === data.descriptive_leader);
      for (const value of [record.name, record.balanced_language_macro_f1.toFixed(4), record.macro_f1_de.toFixed(4), record.macro_f1_en.toFixed(4), (100 * record.recall_positive).toFixed(1) + '%', record.inference_ms_per_text.toFixed(3), (record.model_size_bytes / 1048576).toFixed(2)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      $('llm-rows').append(row);
    }
    const prompt = data.paired_changes.find((row) => row.from === 'qwen3_4b' && row.to === 'qwen_news_v5');
    const tuned = data.paired_changes.find((row) => row.from === 'minilm_bf16_news_v4' && row.to === 'qwen_lora_v5');
    const signed = (value) => (value >= 0 ? '+' : '') + value.toFixed(4);
    $('llm-interpretation').textContent = `On this same Test: adapting the whole task prompt changes mean F1 by ${signed(prompt.mean_language_f1_delta)}; own Qwen LoRA versus unchanged MiniLM BF16 differs by ${signed(tuned.mean_language_f1_delta)}. The fixed rule chooses ${chosen?.name || data.descriptive_leader}; within 0.005 of best, latency then full size decide. Paired uncertainty intervals and all conditions remain in the report.`;
    const lora = data.results.find((row) => row.model === 'qwen_lora_v5');
    $('llm-cost').textContent = `LoRA updates 2.95 million parameters with a frozen 4B base; full-Train fitting took ${lora.training_seconds.toFixed(1)} seconds on this tower. Full size includes the base, profile and adapter, even though serving shares one base allocation. Six-example context cost is included in latency. These are normalized model scores, not calibrated confidence.`;
  } catch (error) { $('llm-status').textContent = error.message; $('llm-chart').hidden = true; $('llm-development-chart').hidden = true; }
}
async function fullArticleDevelopment() {
  try {
    const response = await fetch('/comparison/full-article-development', {cache:'no-store'});
    if (!response.ok) throw new Error('Full-article development results are unavailable.');
    const data = await response.json();
    const zero = data.out_of_fold.find((row) => row.model === 'qwen-zero');
    const lora = data.out_of_fold.find((row) => row.model === 'qwen-lora');
    $('full-article-development-status').textContent = `Training-only OOF agreement: Qwen news rubric ${zero.primary_score.toFixed(4)}; own Qwen LoRA ${lora.primary_score.toFixed(4)} mean DE/EN macro-F1. The small difference is descriptive; it is not a final-test result or evidence of 95% accuracy.`;
  } catch (error) {
    $('full-article-development-status').textContent = error.message;
    $('full-article-development-chart').hidden = true;
  }
}
async function fullArticleComparison() {
  try {
    const response = await fetch('/comparison/full-article', {cache:'no-store'});
    if (!response.ok) throw new Error('Full-article final comparison is unavailable.');
    const data = await response.json();
    const display = {frozen128:'Frozen MiniLM + head', minilm128:'Own MiniLM fine-tuning', 'qwen-zero':'Qwen news rubric', 'qwen-lora':'Own Qwen LoRA'};
    $('final-status').textContent = `${data.test_rows} full articles · 240 Train · development-selected: ${display[data.selected_on_development]}. Selection stayed fixed after Test.`;
    $('final-rows').replaceChildren();
    for (const record of data.models) {
      const row = document.createElement('tr'); row.dataset.selected = String(record.model === data.selected_on_development);
      const ci = record.confidence_intervals_95.primary_score;
      const values = [display[record.model], record.primary_score.toFixed(4), `[${ci[0].toFixed(4)}, ${ci[1].toFixed(4)}]`, record.macro_f1_de.toFixed(4), record.macro_f1_en.toFixed(4), record.macro_f1_ja.toFixed(4), record.inference_ms_per_text.toFixed(2), (record.full_model_size_bytes/1048576).toFixed(2)];
      for (const value of values) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      $('final-rows').append(row);
    }
    const selected = data.models.find((row) => row.model === data.selected_on_development);
    $('final-interpretation').textContent = `The preselected LoRA scores ${selected.primary_score.toFixed(4)} equal-weight DE/EN macro-F1 against the fixed references, subsequently checked by the project owner. The review does not change the recorded scores or establish 95% accuracy. Warm latency includes tokenization and complete-article inference, with model loading and URL fetching excluded. Model size includes Qwen base weights and adapter.`;
  } catch (error) { $('final-status').textContent = error.message; $('final-chart').hidden = true; }
}
checkHealth(); loadModels(); fullArticleComparison(); comparison(); newsComparison(); expandedComparison(); llmComparison(); fullArticleDevelopment();
