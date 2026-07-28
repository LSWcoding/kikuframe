const form = document.querySelector("#extract-form");
const saveButton = document.querySelector("#save-button");
const extractButton = document.querySelector("#extract-button");
const jobPanel = document.querySelector("#job-panel");
const jobStatus = document.querySelector("#job-status");
const jobUrl = document.querySelector("#job-url");
const jobResult = document.querySelector("#job-result");
const resultLinks = document.querySelector("#result-links");
const historyBody = document.querySelector("#history-body");
const errorDialog = document.querySelector("#error-dialog");
const errorTitle = document.querySelector("#error-title");
const errorMessage = document.querySelector("#error-message");
const apiKeyInput = document.querySelector("#api-key");
const keyHint = document.querySelector("#key-hint");
const learningApiKeyInput = document.querySelector("#learning-api-key");
const learningKeyHint = document.querySelector("#learning-key-hint");
const toast = document.querySelector("#toast");
const playerPanel = document.querySelector("#player-panel");
const playerTitle = document.querySelector("#player-title");
const playerSummary = document.querySelector("#player-summary");
const playerAudio = document.querySelector("#subtitle-audio");
const sentenceList = document.querySelector("#sentence-list");
const currentTime = document.querySelector("#current-time");
const previousSentence = document.querySelector("#previous-sentence");
const nextSentence = document.querySelector("#next-sentence");
const loopSentence = document.querySelector("#loop-sentence");
const startBoundarySelectionButton = document.querySelector("#start-boundary-selection");
const analysisDialog = document.querySelector("#analysis-dialog");
const analysisSentence = document.querySelector("#analysis-sentence");
const analysisContent = document.querySelector("#analysis-content");
const analysisCacheStatus = document.querySelector("#analysis-cache-status");
const reanalyzeSentence = document.querySelector("#reanalyze-sentence");
const boundaryToolbar = document.querySelector("#boundary-toolbar");
const boundarySelectionCount = document.querySelector("#boundary-selection-count");
const openBoundaryEditorButton = document.querySelector("#open-boundary-editor");
const cancelBoundarySelectionButton = document.querySelector("#cancel-boundary-selection");
const boundaryDialog = document.querySelector("#boundary-dialog");
const boundaryEditorText = document.querySelector("#boundary-editor-text");
const boundaryPreview = document.querySelector("#boundary-preview");
const saveBoundaryEditButton = document.querySelector("#save-boundary-edit");

let pollingTimer = null;
let playerSentences = [];
let activeSentenceIndex = -1;
let activeAnalysisUrl = "";
let sentenceLoopEnabled = false;
let analysisSentenceIndex = -1;
let activeResegmentUrl = "";
let boundarySelectionAnchor = -1;
let boundarySelectionEnd = -1;
let boundaryLongPressTimer = null;
let suppressSentenceClick = false;

function stopPolling() {
  window.clearTimeout(pollingTimer);
  pollingTimer = null;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* empty response */ }
  if (!response.ok) throw new Error(payload.error || `请求失败（HTTP ${response.status}）`);
  return payload;
}

function formValues() {
  const data = new FormData(form);
  return Object.fromEntries([...data.entries()].map(([key, value]) => [key, String(value).trim()]));
}

function fillForm(config) {
  for (const [key, value] of Object.entries(config)) {
    if (key.endsWith("_API_KEY_CONFIGURED")) continue;
    const input = form.elements.namedItem(key);
    if (input) input.value = value || "";
  }
  const configured = Boolean(config.SUBMD_OCR_API_KEY_CONFIGURED);
  keyHint.textContent = configured ? "已配置；留空将保留现有 Key" : "尚未配置";
  apiKeyInput.placeholder = configured ? "••••••••（留空保持不变）" : "输入后保存在本机";
  apiKeyInput.required = !configured;
  const learningConfigured = Boolean(config.SUBMD_LEARNING_API_KEY_CONFIGURED);
  learningKeyHint.textContent = learningConfigured
    ? "已单独配置；留空将保留现有 Key"
    : "未单独配置；将复用 OCR API Key";
  learningApiKeyInput.placeholder = learningConfigured
    ? "••••••••（留空保持不变）"
    : "留空时复用 OCR API Key";
}

function setBusy(busy) {
  extractButton.disabled = busy;
  extractButton.classList.toggle("loading", busy);
  saveButton.disabled = busy;
}

function notify(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2200);
}

function showError(message, title = "字幕处理失败") {
  errorTitle.textContent = title;
  errorMessage.textContent = message;
  if (typeof errorDialog.showModal === "function") errorDialog.showModal();
  else window.alert(`字幕提取失败\n\n${message}`);
}

async function loadConfig() {
  try { fillForm(await api("/api/config")); }
  catch (error) { showError(error.message); }
}

function formatTime(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

function formatAudioTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00";
  const whole = Math.floor(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remaining = whole % 60;
  const clock = `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
  return hours ? `${String(hours).padStart(2, "0")}:${clock}` : clock;
}

function statusLabel(status) {
  return ({
    running: "提取中", succeeded: "成功", partial: "整理失败", failed: "失败", interrupted: "已中断",
  })[status] || status;
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

async function loadHistory() {
  try {
    const { items } = await api("/api/history");
    if (!items.length) {
      historyBody.innerHTML = '<tr><td colspan="4" class="empty-state">还没有提取记录</td></tr>';
      return;
    }
    historyBody.innerHTML = items.map((item) => {
      const files = Array.isArray(item.results) && item.results.length
        ? item.results
        : (item.download_url ? [{
            label: item.result_name || "下载 Markdown", download_url: item.download_url,
          }] : []);
      const downloads = files.length
        ? files.map((file) => `<a class="download-link" href="${file.download_url}">${escapeHtml(file.label || file.name)}</a>`).join("")
        : "—";
      const player = item.player_url
        ? `<button class="player-link" type="button" data-player-url="${escapeHtml(item.player_url)}">打开字幕播放器</button>`
        : "";
      const result = `${downloads}${player}`;
      const title = item.error ? `${item.message}：${item.error}` : item.message;
      return `<tr>
        <td>${formatTime(item.started_at)}</td>
        <td class="video-cell" title="${escapeHtml(item.source_url)}">${escapeHtml(item.source_url)}</td>
        <td><span class="status-pill status-${escapeHtml(item.status)}" title="${escapeHtml(title)}">${statusLabel(item.status)}</span></td>
        <td>${result}</td>
      </tr>`;
    }).join("");
  } catch (error) { notify(`历史加载失败：${error.message}`); }
}

function renderJob(job) {
  jobPanel.classList.remove("hidden");
  jobPanel.classList.toggle("done", job.status === "succeeded");
  jobPanel.classList.toggle(
    "failed", ["partial", "failed", "interrupted"].includes(job.status),
  );
  jobStatus.textContent = job.message || "正在处理…";
  jobUrl.textContent = job.source_url || "";
  const files = Array.isArray(job.results) && job.results.length
    ? job.results
    : (job.download_url ? [{
        label: job.result_name || "下载 Markdown", download_url: job.download_url,
      }] : []);
  if (files.length) {
    jobResult.classList.remove("hidden");
    const downloads = files.map((file) =>
      `<a href="${file.download_url}">${escapeHtml(file.label || file.name)}</a>`
    ).join("");
    const player = job.player_url
      ? `<button class="player-link" type="button" data-player-url="${escapeHtml(job.player_url)}">打开字幕播放器</button>`
      : "";
    resultLinks.innerHTML = `${downloads}${player}`;
  } else {
    jobResult.classList.add("hidden");
    resultLinks.innerHTML = "";
  }
}

async function pollJob(jobId) {
  stopPolling();
  try {
    const job = await api(`/api/jobs/${jobId}`);
    renderJob(job);
    if (job.status === "running") {
      pollingTimer = window.setTimeout(() => pollJob(jobId), 1400);
      return;
    }
    stopPolling();
    setBusy(false);
    await loadHistory();
    if (["partial", "failed", "interrupted"].includes(job.status)) {
      const title = job.status === "partial" ? "整理版生成失败" : "字幕提取失败";
      showError(job.error || job.message || "未知错误", title);
    } else {
      notify("字幕提取完成");
      if (job.player_url) openPlayer(job.player_url);
    }
  } catch (error) {
    stopPolling();
    setBusy(false);
    showError(error.message);
  }
}

async function openPlayer(url) {
  try {
    const data = await api(url);
    playerSentences = Array.isArray(data.sentences) ? data.sentences : [];
    activeSentenceIndex = -1;
    activeAnalysisUrl = data.analysis_url || `${url.replace(/\/$/, "")}/analysis`;
    activeResegmentUrl = data.resegment_url || `${url.replace(/\/$/, "")}/resegment`;
    clearBoundarySelection();
    setSentenceLoop(false);
    playerTitle.textContent = data.title || "字幕音频播放器";
    playerSummary.textContent = `${playerSentences.length} 句话 · 点击句子播放，点击“分析”学习`;
    playerAudio.src = data.audio_url;
    renderPlayerSentences();
    previousSentence.disabled = !playerSentences.length;
    nextSentence.disabled = !playerSentences.length;
    currentTime.textContent = "00:00";
    playerPanel.classList.remove("hidden");
    playerPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message, "播放器打开失败");
  }
}

function renderPlayerSentences() {
  sentenceList.innerHTML = playerSentences.map((sentence, index) => `
      <li class="sentence-card">
        <button class="sentence-row" type="button" data-sentence-index="${index}"
          data-start-ms="${Number(sentence.start_ms) || 0}">
          <span class="sentence-time">${formatAudioTime((Number(sentence.start_ms) || 0) / 1000)}</span>
          <span class="sentence-text">${escapeHtml(sentence.text)}</span>
        </button>
        <div class="sentence-actions">
          <button class="analyze-button" type="button" data-analysis-index="${index}">分析</button>
        </div>
      </li>`).join("");
  if (activeSentenceIndex >= playerSentences.length) activeSentenceIndex = -1;
  updateBoundarySelection();
}

function selectedBoundaryIndexes() {
  if (boundarySelectionAnchor < 0 || boundarySelectionEnd < 0) return [];
  const start = Math.min(boundarySelectionAnchor, boundarySelectionEnd);
  const end = Math.max(boundarySelectionAnchor, boundarySelectionEnd);
  return Array.from({ length: end - start + 1 }, (_, offset) => start + offset);
}

function updateBoundarySelection() {
  const selected = selectedBoundaryIndexes();
  sentenceList.querySelectorAll(".sentence-card").forEach((card, index) => {
    card.classList.toggle("boundary-selected", selected.includes(index));
  });
  boundaryToolbar.classList.toggle("hidden", !selected.length);
  boundarySelectionCount.textContent = selected.length
    ? `已连续选择 ${selected.length} 句（${selected[0] + 1}–${selected.at(-1) + 1}）`
    : "已选择 0 句";
}

function clearBoundarySelection() {
  boundarySelectionAnchor = -1;
  boundarySelectionEnd = -1;
  window.clearTimeout(boundaryLongPressTimer);
  boundaryLongPressTimer = null;
  updateBoundarySelection();
}

function splitBoundaryPreview(text) {
  const pieces = [];
  let pending = "";
  for (const character of String(text || "").replace(/\r\n/g, "\n")) {
    if (character === "\n") {
      if (pending.trim()) pieces.push(pending.replace(/\s+/g, "").trim());
      pending = "";
      continue;
    }
    if (/\s/u.test(character)) continue;
    pending += character;
    if (["。", "．", "."].includes(character)) {
      pieces.push(pending);
      pending = "";
    }
  }
  if (pending.trim()) pieces.push(pending.trim());
  return pieces.filter(Boolean);
}

function renderBoundaryPreview() {
  const pieces = splitBoundaryPreview(boundaryEditorText.value);
  boundaryPreview.innerHTML = pieces.length
    ? pieces.map((piece) => `<li>${escapeHtml(piece)}</li>`).join("")
    : '<li class="muted">请至少保留一句字幕</li>';
}

function openBoundaryEditor() {
  const selected = selectedBoundaryIndexes();
  if (!selected.length) return;
  boundaryEditorText.value = selected.map((index) => playerSentences[index].text).join("\n");
  renderBoundaryPreview();
  boundaryDialog.showModal();
  boundaryEditorText.focus();
}

async function saveBoundaryEdit() {
  const selected = selectedBoundaryIndexes();
  if (!selected.length || !activeResegmentUrl) return;
  saveBoundaryEditButton.disabled = true;
  saveBoundaryEditButton.textContent = "保存中…";
  try {
    const data = await api(activeResegmentUrl, {
      method: "POST",
      body: JSON.stringify({
        sentence_ids: selected.map((index) => playerSentences[index].sentence_id),
        edited_text: boundaryEditorText.value,
      }),
    });
    playerSentences = Array.isArray(data.sentences) ? data.sentences : [];
    activeAnalysisUrl = data.analysis_url || activeAnalysisUrl;
    activeResegmentUrl = data.resegment_url || activeResegmentUrl;
    activeSentenceIndex = -1;
    boundaryDialog.close();
    clearBoundarySelection();
    renderPlayerSentences();
    playerSummary.textContent = `${playerSentences.length} 句话 · 手动断句已保存并重新对齐音频`;
    notify("断句修改已保存，音频时间已重新对齐");
    await loadHistory();
  } catch (error) {
    showError(error.message, "断句修改失败");
  } finally {
    saveBoundaryEditButton.disabled = false;
    saveBoundaryEditButton.textContent = "保存并重新对齐音频";
  }
}

function setSentenceLoop(enabled) {
  sentenceLoopEnabled = Boolean(enabled);
  loopSentence.classList.toggle("active", sentenceLoopEnabled);
  loopSentence.setAttribute("aria-pressed", String(sentenceLoopEnabled));
  loopSentence.textContent = sentenceLoopEnabled ? "单句循环：开" : "单句循环：关";
}

function playSentence(index) {
  if (index < 0 || index >= playerSentences.length) return;
  const sentence = playerSentences[index];
  playerAudio.currentTime = (Number(sentence.start_ms) || 0) / 1000;
  setActiveSentence(index, true);
  playerAudio.play().catch(() => { /* browser may require another user gesture */ });
}

function setActiveSentence(index, scroll = false) {
  if (index === activeSentenceIndex) return;
  sentenceList.querySelector(".sentence-row.active")?.classList.remove("active");
  activeSentenceIndex = index;
  const row = sentenceList.querySelector(`[data-sentence-index="${index}"]`);
  if (row) {
    row.classList.add("active");
    if (scroll) row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  previousSentence.disabled = index <= 0;
  nextSentence.disabled = index < 0 || index >= playerSentences.length - 1;
}

function syncSentenceToAudio() {
  const timeMs = playerAudio.currentTime * 1000;
  currentTime.textContent = formatAudioTime(playerAudio.currentTime);
  if (sentenceLoopEnabled && activeSentenceIndex >= 0) {
    const active = playerSentences[activeSentenceIndex];
    const startMs = Number(active.start_ms) || 0;
    const endMs = Number(active.end_ms) || startMs;
    if (endMs > startMs && timeMs >= endMs) {
      playerAudio.currentTime = startMs / 1000;
      currentTime.textContent = formatAudioTime(startMs / 1000);
      playerAudio.play().catch(() => { /* browser may require another user gesture */ });
      return;
    }
  }
  let found = -1;
  for (let index = 0; index < playerSentences.length; index += 1) {
    const sentence = playerSentences[index];
    if (timeMs >= Number(sentence.start_ms) && timeMs < Number(sentence.end_ms)) {
      found = index;
      break;
    }
    if (timeMs >= Number(sentence.start_ms)) found = index;
  }
  if (found >= 0) setActiveSentence(found, true);
}

function appendAnalysisSection(label, lines) {
  const section = document.createElement("section");
  section.className = "analysis-section";
  const heading = document.createElement("h3");
  heading.textContent = label;
  section.appendChild(heading);
  if (!lines.length) {
    const empty = document.createElement("p");
    empty.className = "analysis-item muted";
    empty.textContent = "无";
    section.appendChild(empty);
  } else {
    for (const line of lines) {
      const item = document.createElement("p");
      item.className = "analysis-item";
      item.textContent = line;
      section.appendChild(item);
    }
  }
  analysisContent.appendChild(section);
}

function renderLearningAnalysis(data) {
  analysisContent.replaceChildren();
  const translation = document.createElement("p");
  translation.className = "analysis-translation";
  const translationLabel = document.createElement("strong");
  translationLabel.textContent = "整句翻译：";
  translation.append(translationLabel, document.createTextNode(String(data.translation || "—")));
  analysisContent.appendChild(translation);
  const vocabulary = Array.isArray(data.vocabulary) ? data.vocabulary.map((item) => {
    const reading = item.reading ? `（${item.reading}）` : "";
    return `${item.expression || "—"}${reading}：${item.meaning || "—"}`;
  }) : [];
  appendAnalysisSection("词汇以及搭配：", vocabulary);
  const grammar = Array.isArray(data.grammar) ? data.grammar.map((item) =>
    `${item.pattern || "—"}：${item.explanation || "—"}`
  ) : [];
  appendAnalysisSection("文法：", grammar);
  analysisContent.scrollTop = 0;
}

async function openSentenceAnalysis(index, button, force = false) {
  if (index < 0 || index >= playerSentences.length || !activeAnalysisUrl) return;
  const sentence = playerSentences[index];
  analysisSentenceIndex = index;
  analysisSentence.textContent = sentence.text || "";
  analysisCacheStatus.textContent = "";
  reanalyzeSentence.classList.add("hidden");
  reanalyzeSentence.disabled = true;
  const loadingMessage = force
    ? "正在重新分析，成功后将覆盖上次结果…"
    : "正在加载已保存结果或调用语言学习模型…";
  analysisContent.innerHTML = `<div class="analysis-loading"><span></span><p>${loadingMessage}</p></div>`;
  if (!analysisDialog.open) analysisDialog.showModal();
  if (button) {
    button.disabled = true;
    button.textContent = "分析中…";
  }
  try {
    const data = await api(activeAnalysisUrl, {
      method: "POST",
      body: JSON.stringify({ sentence_id: sentence.sentence_id, force }),
    });
    renderLearningAnalysis(data);
    analysisCacheStatus.textContent = data.cached ? "已加载上次结果" : "已保存新结果";
  } catch (error) {
    analysisContent.replaceChildren();
    const message = document.createElement("p");
    message.className = "analysis-error";
    const retained = force ? "\n旧分析缓存未被删除，可以关闭弹窗后重新加载。" : "";
    message.textContent = `分析失败：${error.message}${retained}`;
    analysisContent.appendChild(message);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "分析";
    }
    reanalyzeSentence.disabled = false;
    reanalyzeSentence.classList.remove("hidden");
  }
}

saveButton.addEventListener("click", async () => {
  try {
    fillForm(await api("/api/config", { method: "POST", body: JSON.stringify(formValues()) }));
    notify("配置已保存到 .env");
  } catch (error) { showError(error.message); }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  stopPolling();
  setBusy(true);
  jobResult.classList.add("hidden");
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(formValues()) });
    renderJob(job);
    pollJob(job.job_id);
  } catch (error) {
    setBusy(false);
    showError(error.message);
  }
});

document.querySelector("#toggle-key").addEventListener("click", (event) => {
  const reveal = apiKeyInput.type === "password";
  apiKeyInput.type = reveal ? "text" : "password";
  event.currentTarget.textContent = reveal ? "隐藏" : "显示";
});
document.querySelector("#toggle-learning-key").addEventListener("click", (event) => {
  const reveal = learningApiKeyInput.type === "password";
  learningApiKeyInput.type = reveal ? "text" : "password";
  event.currentTarget.textContent = reveal ? "隐藏" : "显示";
});
document.querySelector("#close-error").addEventListener("click", () => errorDialog.close());
document.querySelector("#close-analysis").addEventListener("click", () => analysisDialog.close());
document.querySelector("#close-boundary-dialog").addEventListener("click", () => boundaryDialog.close());
document.querySelector("#cancel-boundary-edit").addEventListener("click", () => boundaryDialog.close());
cancelBoundarySelectionButton.addEventListener("click", clearBoundarySelection);
openBoundaryEditorButton.addEventListener("click", openBoundaryEditor);
boundaryEditorText.addEventListener("input", renderBoundaryPreview);
saveBoundaryEditButton.addEventListener("click", saveBoundaryEdit);
reanalyzeSentence.addEventListener("click", () => {
  const button = sentenceList.querySelector(
    `[data-analysis-index="${analysisSentenceIndex}"]`,
  );
  openSentenceAnalysis(analysisSentenceIndex, button, true);
});
document.querySelector("#refresh-history").addEventListener("click", loadHistory);
document.querySelector("#close-player").addEventListener("click", () => {
  playerAudio.pause();
  playerPanel.classList.add("hidden");
});
sentenceList.addEventListener("click", (event) => {
  const analysisButton = event.target.closest("[data-analysis-index]");
  if (analysisButton) {
    openSentenceAnalysis(Number(analysisButton.dataset.analysisIndex), analysisButton);
    return;
  }
  const row = event.target.closest("[data-sentence-index]");
  if (row) {
    const index = Number(row.dataset.sentenceIndex);
    if (suppressSentenceClick) {
      suppressSentenceClick = false;
      return;
    }
    if (boundarySelectionAnchor >= 0) {
      boundarySelectionEnd = index;
      updateBoundarySelection();
      return;
    }
    playSentence(index);
  }
});
sentenceList.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || event.target.closest("[data-analysis-index]")) return;
  const row = event.target.closest("[data-sentence-index]");
  if (!row) return;
  window.clearTimeout(boundaryLongPressTimer);
  boundaryLongPressTimer = window.setTimeout(() => {
    const index = Number(row.dataset.sentenceIndex);
    boundarySelectionAnchor = index;
    boundarySelectionEnd = index;
    suppressSentenceClick = true;
    updateBoundarySelection();
    if (navigator.vibrate) navigator.vibrate(30);
  }, 560);
});
for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) {
  sentenceList.addEventListener(eventName, () => {
    window.clearTimeout(boundaryLongPressTimer);
    boundaryLongPressTimer = null;
  });
}
sentenceList.addEventListener("contextmenu", (event) => {
  if (boundarySelectionAnchor >= 0) event.preventDefault();
});
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-player-url]");
  if (button) openPlayer(button.dataset.playerUrl);
});
playerAudio.addEventListener("timeupdate", syncSentenceToAudio);
playerAudio.addEventListener("seeked", syncSentenceToAudio);
playerAudio.addEventListener("ended", syncSentenceToAudio);
previousSentence.addEventListener("click", () => playSentence(activeSentenceIndex - 1));
nextSentence.addEventListener("click", () => playSentence(activeSentenceIndex + 1));
loopSentence.addEventListener("click", () => setSentenceLoop(!sentenceLoopEnabled));
startBoundarySelectionButton.addEventListener("click", () => {
  const index = activeSentenceIndex >= 0 ? activeSentenceIndex : 0;
  if (index >= playerSentences.length) return;
  boundarySelectionAnchor = index;
  boundarySelectionEnd = index;
  updateBoundarySelection();
});

loadConfig();
loadHistory();
