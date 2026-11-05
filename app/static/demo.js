const form = document.querySelector("#predict-form");
const urlInput = document.querySelector("#article-url");
const textInput = document.querySelector("#article-text");
const statusEl = document.querySelector("#form-status");
const resultEl = document.querySelector("#result");
const healthEl = document.querySelector("#model-status");
const LABEL_ORDER = ["negative", "neutral", "positive"];

function setStatus(message, kind) {
  statusEl.textContent = message || "";
  statusEl.dataset.kind = kind || "";
}

function detailMessage(body) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join(" ");
  }
  return "The request was rejected.";
}

function renderScores(scores) {
  const list = document.querySelector("#scores");
  list.replaceChildren();
  for (const name of LABEL_ORDER) {
    const value = Number(scores[name] ?? 0);
    const row = document.createElement("div");
    row.className = "score";
    const label = document.createElement("span");
    label.className = "score-label";
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "score-track";
    const bar = document.createElement("span");
    bar.className = `score-bar score-bar-${name}`;
    const width = Math.max(0, Math.min(100, value * 100));
    bar.style.width = `${width}%`;
    track.append(bar);
    const number = document.createElement("span");
    number.className = "score-value";
    number.textContent = value.toFixed(3);
    row.append(label, track, number);
    list.append(row);
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const body = await response.json();
    if (body.model_loaded) {
      const version = body.model_version ? ` (${body.model_version})` : "";
      healthEl.textContent = `Loaded model: ${body.model_name || "ready"}${version}`;
      return;
    }
    healthEl.textContent = "No model is loaded. Run python -m src.train, then restart the API.";
  } catch (_error) {
    healthEl.textContent = "Could not read /health.";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = urlInput.value.trim();
  const text = textInput.value.trim();
  if (!url && !text) {
    setStatus("Enter a URL or paste the article text.", "error");
    return;
  }
  if (url && text) {
    setStatus("Send exactly one of URL or text, not both.", "error");
    return;
  }
  const button = form.querySelector("button");
  button.disabled = true;
  setStatus("Classifying…", "pending");
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(url ? { url } : { text }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      resultEl.hidden = true;
      setStatus(detailMessage(body), "error");
      return;
    }
    resultEl.hidden = false;
    const label = document.querySelector("#label");
    label.textContent = body.label || "";
    label.className = `label label-${body.label || ""}`;
    document.querySelector("#result-model").textContent = body.model_version || "unknown";
    const excerptBlock = document.querySelector("#excerpt-block");
    const excerpt = document.querySelector("#scraped-excerpt");
    const preview = typeof body.scraped_excerpt === "string" ? body.scraped_excerpt.trim() : "";
    if (preview) {
      excerpt.textContent = preview;
      excerptBlock.hidden = false;
    } else {
      excerpt.textContent = "";
      excerptBlock.hidden = true;
    }
    renderScores(body.scores || {});
    document.querySelector("#raw-json").textContent = JSON.stringify(body, null, 2);
    setStatus("", "");
  } catch (_error) {
    resultEl.hidden = true;
    setStatus("The API could not be reached.", "error");
  } finally {
    button.disabled = false;
  }
});

loadHealth();
