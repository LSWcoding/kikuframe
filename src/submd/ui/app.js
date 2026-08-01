const form = document.querySelector("#extract-form");
const saveButton = document.querySelector("#save-button");
const extractButton = document.querySelector("#extract-button");
const jobPanel = document.querySelector("#job-panel");
const jobStatus = document.querySelector("#job-status");
const jobUrl = document.querySelector("#job-url");
const jobResult = document.querySelector("#job-result");
const resultLinks = document.querySelector("#result-links");
const cancelJobButton = document.querySelector("#cancel-job");
const historyBody = document.querySelector("#history-body");
const clearHistoryButton = document.querySelector("#clear-history");
const errorDialog = document.querySelector("#error-dialog");
const errorTitle = document.querySelector("#error-title");
const errorMessage = document.querySelector("#error-message");
const sharedValueInputs = [...document.querySelectorAll("[data-shared-type]")];
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
const libraryDialog = document.querySelector("#library-dialog");
const libraryContent = document.querySelector("#library-content");
const librarySummary = document.querySelector("#library-summary");
const exportLibrary = document.querySelector("#export-library");

let pollingTimer = null;
let activeJobId = "";
let playerSentences = [];
let activeSentenceIndex = -1;
let activeAnalysisUrl = "";
let activeLibraryUrl = "";
let activeAnalysisData = null;
let sentenceLoopEnabled = false;
let analysisSentenceIndex = -1;
let activeResegmentUrl = "";
let activeSentenceDeleteUrl = "";
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
  for (const hint of document.querySelectorAll("[data-key-hint]")) {
    const fieldName = hint.dataset.keyHint;
    const configured = Boolean(config[`${fieldName}_CONFIGURED`]);
    const input = form.elements.namedItem(fieldName);
    hint.textContent = configured
      ? "已单独配置；留空保留此 Key"
      : "未单独配置；留空采用第一个已填写的 Key";
    if (input) {
      input.placeholder = configured
        ? "••••••••（留空保持不变）"
        : "留空时采用第一个 Key";
    }
  }
}

function sharedConcreteValues(type) {
  return [...new Set(sharedValueInputs
    .filter((input) => input.dataset.sharedType === type)
    .map((input) => input.value.trim())
    .filter(Boolean))];
}

function suggestionPanel(input) {
  const field = input.closest(".field");
  let panel = field.querySelector(".shared-suggestions");
  if (!panel) {
    panel = document.createElement("div");
    panel.className = "shared-suggestions hidden";
    panel.addEventListener("mousedown", (event) => event.preventDefault());
    panel.addEventListener("click", (event) => {
      const button = event.target.closest("[data-shared-value]");
      if (!button) return;
      input.value = button.dataset.sharedValue;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.focus();
    });
    field.appendChild(panel);
  }
  return panel;
}

function renderSharedSuggestions(input) {
  const panel = suggestionPanel(input);
  const values = sharedConcreteValues(input.dataset.sharedType)
    .filter((value) => value !== input.value.trim());
  panel.replaceChildren();
  if (!values.length) {
    panel.classList.add("hidden");
    return;
  }
  const label = document.createElement("span");
  label.textContent = input.value.trim() ? "其他已填写值" : `留空默认使用第一个值 · 点击填入`;
  panel.appendChild(label);
  for (const value of values) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.sharedValue = value;
    button.textContent = value;
    button.title = `点击填入：${value}`;
    panel.appendChild(button);
  }
  panel.classList.remove("hidden");
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
    running: "提取中", cancelling: "正在中止", cancelled: "已中止",
    succeeded: "成功", partial: "整理失败", failed: "失败", interrupted: "已中断",
  })[status] || status;
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function learningKindLabel(kind) {
  return ({ word: "单词", collocation: "搭配", grammar: "文法" })[kind] || "词条";
}

function renderStudyLibrary(data) {
  const items = Array.isArray(data.items) ? data.items : [];
  const totalEncounters = Number(data.encounter_count) || 0;
  librarySummary.textContent = `${items.length} 个词条 · 累计遇见 ${totalEncounters} 次`;
  exportLibrary.href = data.export_url || "/api/library/export";
  exportLibrary.classList.toggle("disabled", items.length === 0);
  exportLibrary.setAttribute("aria-disabled", items.length ? "false" : "true");
  if (!items.length) {
    libraryContent.innerHTML = '<p class="library-empty">还没有收藏单词、搭配或文法。</p>';
    return;
  }
  libraryContent.innerHTML = `<div class="library-list">${items.map((item) => {
    const reading = item.reading ? `（${escapeHtml(item.reading)}）` : "";
    const meanings = Array.isArray(item.meanings) && item.meanings.length
      ? item.meanings.map(escapeHtml).join("；")
      : "—";
    const count = Number(item.encounter_count) || 0;
    const display = item.display || item.lemma;
    return `<article class="library-entry" data-library-entry="${Number(item.entry_id)}">
      <div class="library-entry-main">
        <span class="library-kind">${learningKindLabel(item.kind)}</span>
        <p class="library-expression"><strong>${escapeHtml(display)}${reading}</strong><span>：${meanings}</span></p>
        <button class="library-count" type="button" data-library-count="${Number(item.entry_id)}"
          aria-expanded="false" aria-label="查看 ${count} 次遇见记录">${count}</button>
        <button class="library-delete" type="button" data-library-delete="${Number(item.entry_id)}"
          aria-label="删除 ${escapeHtml(display)}">删除</button>
      </div>
      <div class="library-occurrences hidden" data-library-occurrences="${Number(item.entry_id)}"></div>
    </article>`;
  }).join("")}</div>`;
}

async function openStudyLibrary() {
  librarySummary.textContent = "正在读取本地学习记录…";
  libraryContent.innerHTML = '<div class="library-loading"><span></span><p>正在加载单词库…</p></div>';
  if (!libraryDialog.open) libraryDialog.showModal();
  try {
    renderStudyLibrary(await api("/api/library"));
  } catch (error) {
    libraryContent.innerHTML = `<p class="analysis-error">${escapeHtml(error.message)}</p>`;
  }
}

async function toggleLibraryOccurrences(button) {
  const entryId = Number(button.dataset.libraryCount);
  const panel = libraryContent.querySelector(`[data-library-occurrences="${entryId}"]`);
  if (!panel || !entryId) return;
  const isOpen = button.getAttribute("aria-expanded") === "true";
  if (isOpen) {
    button.setAttribute("aria-expanded", "false");
    panel.classList.add("hidden");
    return;
  }
  button.setAttribute("aria-expanded", "true");
  panel.classList.remove("hidden");
  if (panel.dataset.loaded === "true") return;
  panel.innerHTML = '<p class="library-detail-loading">正在读取遇见记录…</p>';
  try {
    const entry = await api(`/api/library/${entryId}`);
    const encounters = Array.isArray(entry.encounters) ? entry.encounters : [];
    panel.innerHTML = encounters.length
      ? `<ol>${encounters.map((encounter) => {
          const title = encounter.article_title || encounter.source_url || "未知视频";
          const titleMarkup = encounter.source_url
            ? `<a href="${escapeHtml(encounter.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(title)}</a>`
            : `<strong>${escapeHtml(title)}</strong>`;
          return `<li><div>${titleMarkup}<span>${escapeHtml(encounter.meaning || "")}</span></div><p>${escapeHtml(encounter.sentence)}</p></li>`;
        }).join("")}</ol>`
      : '<p class="library-detail-loading">没有找到遇见记录。</p>';
    panel.dataset.loaded = "true";
  } catch (error) {
    panel.innerHTML = `<p class="analysis-error">${escapeHtml(error.message)}</p>`;
  }
}

async function deleteLibraryEntry(button) {
  const entryId = Number(button.dataset.libraryDelete);
  if (!entryId) return;
  button.disabled = true;
  try {
    await api(`/api/library/${entryId}`, { method: "DELETE" });
    renderStudyLibrary(await api("/api/library"));
    notify("词库条目已删除");
  } catch (error) {
    button.disabled = false;
    showError(error.message, "删除词库条目失败");
  }
}

async function loadHistory() {
  try {
    const { items } = await api("/api/history");
    if (!items.length) {
      historyBody.innerHTML = '<tr><td colspan="5" class="empty-state">还没有提取记录</td></tr>';
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
      const active = ["running", "cancelling"].includes(item.status);
      const viewStatus = active
        ? `<button class="history-status" type="button" data-history-status="${escapeHtml(item.job_id)}">查看状态</button>`
        : "";
      const deleteLabel = active ? "中止并删除" : "删除";
      return `<tr${active ? ` class="history-active" data-history-status="${escapeHtml(item.job_id)}"` : ""}>
        <td>${formatTime(item.started_at)}</td>
        <td class="video-cell" title="${escapeHtml(item.source_url)}">${escapeHtml(item.video_title || item.result_name || item.source_url)}</td>
        <td><span class="status-pill status-${escapeHtml(item.status)}" title="${escapeHtml(title)}">${statusLabel(item.status)}</span></td>
        <td>${result}</td>
        <td><div class="history-row-actions">${viewStatus}<button class="history-delete" type="button" data-history-delete="${escapeHtml(item.job_id)}"${active ? ' data-force-stop="true"' : ""}>${deleteLabel}</button></div></td>
      </tr>`;
    }).join("");
  } catch (error) { notify(`历史加载失败：${error.message}`); }
}

async function viewHistoryJob(jobId) {
  if (!jobId) return;
  activeJobId = jobId;
  setBusy(true);
  await pollJob(jobId);
  jobPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function deleteHistoryEntry(button) {
  const jobId = button.dataset.historyDelete;
  if (!jobId) return;
  if (button.dataset.forceStop === "true" && !window.confirm(
    "确定立即停止并删除这个任务吗？\n\n后台提取进程会被终止，本次 URL 已产生的媒体、帧、字幕、检查点和输出都会删除。",
  )) return;
  button.disabled = true;
  try {
    await api(`/api/history/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    playerAudio.pause();
    playerPanel.classList.add("hidden");
    await loadHistory();
    notify("提取记录和关联文件已删除；单词库保持不变");
  } catch (error) {
    button.disabled = false;
    showError(error.message, "删除提取记录失败");
  }
}

async function clearHistory() {
  clearHistoryButton.disabled = true;
  try {
    await api("/api/history", { method: "DELETE" });
    playerAudio.pause();
    playerPanel.classList.add("hidden");
    jobPanel.classList.add("hidden");
    await loadHistory();
    notify("提取历史已全部清空；单词库及来源句已保留");
  } catch (error) {
    showError(error.message, "清空提取历史失败");
  } finally {
    clearHistoryButton.disabled = false;
  }
}

function renderJob(job) {
  jobPanel.classList.remove("hidden");
  jobPanel.classList.toggle("done", job.status === "succeeded");
  jobPanel.classList.toggle(
    "failed", ["partial", "failed", "interrupted", "cancelled"].includes(job.status),
  );
  jobStatus.textContent = job.message || "正在处理…";
  jobUrl.textContent = job.source_url || "";
  const canCancel = ["running", "cancelling"].includes(job.status);
  cancelJobButton.classList.toggle("hidden", !canCancel);
  cancelJobButton.disabled = job.status === "cancelling";
  cancelJobButton.textContent = job.status === "cancelling"
    ? "正在中止并清理…"
    : "中止并删除本次数据";
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
    if (["running", "cancelling"].includes(job.status)) {
      pollingTimer = window.setTimeout(() => pollJob(jobId), 1400);
      return;
    }
    stopPolling();
    setBusy(false);
    await loadHistory();
    if (job.status === "cancelled") {
      activeJobId = "";
      notify("处理已中止，本次 URL 的所有已处理数据已删除");
      return;
    }
    if (["partial", "failed", "interrupted"].includes(job.status)) {
      const title = job.status === "partial" ? "整理版生成失败" : "字幕提取失败";
      showError(job.error || job.message || "未知错误", title);
    } else {
      notify("字幕提取完成");
      if (job.player_url) openPlayer(job.player_url);
    }
    activeJobId = "";
  } catch (error) {
    stopPolling();
    setBusy(false);
    showError(error.message);
  }
}

async function cancelActiveJob() {
  if (!activeJobId) return;
  const confirmed = window.confirm(
    "确定中止当前 URL 的处理吗？\n\n本次 URL 已下载的媒体、帧、字幕、检查点、输出文件和历史记录都会删除，个人单词库保持不变。",
  );
  if (!confirmed) return;
  stopPolling();
  cancelJobButton.disabled = true;
  cancelJobButton.textContent = "正在中止并清理…";
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(activeJobId)}`, {
      method: "DELETE",
    });
    renderJob(job);
    if (job.status === "cancelling") {
      pollingTimer = window.setTimeout(() => pollJob(activeJobId), 700);
    } else {
      setBusy(false);
      activeJobId = "";
      await loadHistory();
      notify("处理已中止，本次 URL 的所有已处理数据已删除");
    }
  } catch (error) {
    cancelJobButton.disabled = false;
    cancelJobButton.textContent = "中止并删除本次数据";
    showError(error.message, "中止处理失败");
    if (activeJobId) pollingTimer = window.setTimeout(() => pollJob(activeJobId), 1400);
  }
}

async function openPlayer(url) {
  try {
    const data = await api(url);
    playerSentences = Array.isArray(data.sentences) ? data.sentences : [];
    activeSentenceIndex = -1;
    activeAnalysisUrl = data.analysis_url || `${url.replace(/\/$/, "")}/analysis`;
    activeLibraryUrl = data.library_url || `${url.replace(/\/$/, "")}/library`;
    activeResegmentUrl = data.resegment_url || `${url.replace(/\/$/, "")}/resegment`;
    activeSentenceDeleteUrl = data.sentence_delete_url || `${url.replace(/\/$/, "")}/sentences`;
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
          <button class="sentence-delete" type="button" data-sentence-delete="${index}"
            aria-label="删除这句字幕">删除</button>
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
    activeLibraryUrl = data.library_url || activeLibraryUrl;
    activeResegmentUrl = data.resegment_url || activeResegmentUrl;
    activeSentenceDeleteUrl = data.sentence_delete_url || activeSentenceDeleteUrl;
    activeSentenceIndex = -1;
    boundaryDialog.close();
    clearBoundarySelection();
    renderPlayerSentences();
    playerSummary.textContent = `${playerSentences.length} 句话 · 字幕与断句已保存并重新对齐音频`;
    notify("字幕与断句修改已保存，音频时间已重新对齐");
    await loadHistory();
  } catch (error) {
    showError(error.message, "字幕修改失败");
  } finally {
    saveBoundaryEditButton.disabled = false;
    saveBoundaryEditButton.textContent = "保存并重新对齐音频";
  }
}

async function deletePlayerSentence(index, button) {
  const sentence = playerSentences[index];
  if (!sentence || !activeSentenceDeleteUrl) return;
  button.disabled = true;
  button.textContent = "删除中…";
  try {
    const data = await api(
      `${activeSentenceDeleteUrl}/${encodeURIComponent(sentence.sentence_id)}`,
      { method: "DELETE" },
    );
    playerAudio.pause();
    playerSentences = Array.isArray(data.sentences) ? data.sentences : [];
    activeAnalysisUrl = data.analysis_url || activeAnalysisUrl;
    activeLibraryUrl = data.library_url || activeLibraryUrl;
    activeResegmentUrl = data.resegment_url || activeResegmentUrl;
    activeSentenceDeleteUrl = data.sentence_delete_url || activeSentenceDeleteUrl;
    activeSentenceIndex = -1;
    clearBoundarySelection();
    renderPlayerSentences();
    playerSummary.textContent = `${playerSentences.length} 句话 · 已删除不存在的字幕句，单词库不受影响`;
    notify("该句字幕已删除；已有单词库内容仍保留");
    await loadHistory();
  } catch (error) {
    button.disabled = false;
    button.textContent = "删除";
    showError(error.message, "删除单句失败");
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

function libraryStatusText(state) {
  const count = Number(state?.encounter_count) || 0;
  const meaningCount = Number(state?.meaning_count) || 0;
  const senses = meaningCount > 1 ? ` · ${meaningCount} 个语境释义` : "";
  if (state?.context_saved) return `本句已记录 · 累计 ${count} 次${senses}`;
  if (state?.exists && !state?.meaning_saved) {
    return `已有原型 · 发现新语境释义 · 点击记录${senses}`;
  }
  if (state?.exists) return `词库已有 · 累计 ${count} 次 · 点击记录本句${senses}`;
  return "点击存入个人词库";
}

function appendAnalysisSection(label, items, itemType) {
  const section = document.createElement("section");
  section.className = "analysis-section";
  const heading = document.createElement("h3");
  heading.textContent = label;
  section.appendChild(heading);
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "analysis-item muted";
    empty.textContent = "无";
    section.appendChild(empty);
  } else {
    const grid = document.createElement("div");
    grid.className = "analysis-block-grid";
    items.forEach((item, index) => {
      const state = item.library || {};
      const hasKanji = itemType === "vocabulary"
        && /[\u3400-\u9fff]/u.test(`${item.expression || ""}${item.lemma || ""}`);
      const readingMissing = hasKanji && !String(item.reading || "").trim();
      const block = document.createElement("button");
      block.type = "button";
      block.className = `analysis-learning-block${state.context_saved ? " saved" : ""}${readingMissing ? " reading-missing" : ""}`;
      block.dataset.libraryType = itemType;
      block.dataset.libraryIndex = String(index);
      block.disabled = Boolean(state.context_saved || readingMissing);

      const top = document.createElement("span");
      top.className = "analysis-block-top";
      const expression = document.createElement("strong");
      const surface = itemType === "vocabulary" ? item.expression : item.pattern;
      const reading = readingMissing
        ? "（读音待补充）"
        : (itemType === "vocabulary" && item.reading ? `（${item.reading}）` : "");
      expression.textContent = `${surface || "—"}${reading}`;
      const badge = document.createElement("span");
      badge.className = "analysis-kind-badge";
      badge.textContent = itemType === "grammar"
        ? "文法"
        : (item.kind === "collocation" ? "搭配" : "单词");
      top.append(expression, badge);

      const lemma = document.createElement("span");
      lemma.className = "analysis-block-lemma";
      lemma.textContent = `原型：${item.lemma || surface || "—"}`;
      const meaning = document.createElement("span");
      meaning.className = "analysis-block-meaning";
      meaning.textContent = itemType === "grammar"
        ? (item.explanation || "—")
        : (item.meaning || "—");
      const status = document.createElement("span");
      status.className = "analysis-block-status";
      status.textContent = readingMissing
        ? "模型仍未返回读音；请点击“重新分析”后再收藏"
        : libraryStatusText(state);
      block.append(top, lemma, meaning, status);
      grid.appendChild(block);
    });
    section.appendChild(grid);
  }
  analysisContent.appendChild(section);
}

async function saveLearningBlock(block) {
  if (!activeAnalysisData || !activeLibraryUrl || block.disabled) return;
  const itemType = block.dataset.libraryType;
  const index = Number(block.dataset.libraryIndex);
  const source = itemType === "grammar"
    ? activeAnalysisData.grammar
    : activeAnalysisData.vocabulary;
  const item = Array.isArray(source) ? source[index] : null;
  if (!item) return;
  const submitted = itemType === "grammar"
    ? { pattern: item.pattern, lemma: item.lemma, explanation: item.explanation }
    : {
        kind: item.kind,
        expression: item.expression,
        lemma: item.lemma,
        reading: item.reading || "",
        meaning: item.meaning,
      };
  const status = block.querySelector(".analysis-block-status");
  block.disabled = true;
  if (status) status.textContent = "正在保存…";
  try {
    const data = await api(activeLibraryUrl, {
      method: "POST",
      body: JSON.stringify({
        sentence_id: activeAnalysisData.sentence_id,
        item_type: itemType,
        item: submitted,
      }),
    });
    item.library = data.library || {};
    block.classList.toggle("saved", Boolean(item.library.context_saved));
    block.disabled = Boolean(item.library.context_saved);
    if (status) status.textContent = libraryStatusText(item.library);
    if (item.library.added_entry) {
      notify(`已把「${data.lemma}」加入个人词库`);
    } else if (item.library.added_meaning) {
      notify(`已为「${data.lemma}」添加新的语境释义`);
    } else if (item.library.added_encounter) {
      notify(`已累计「${data.lemma}」的学习次数`);
    } else {
      notify("本句中的这个条目已经记录过");
    }
  } catch (error) {
    block.disabled = false;
    if (status) status.textContent = libraryStatusText(item.library || {});
    showError(error.message, "保存到个人词库失败");
  }
}

function renderLearningAnalysis(data) {
  activeAnalysisData = data;
  analysisContent.replaceChildren();
  const translation = document.createElement("p");
  translation.className = "analysis-translation";
  const translationLabel = document.createElement("strong");
  translationLabel.textContent = "整句翻译：";
  translation.append(translationLabel, document.createTextNode(String(data.translation || "—")));
  analysisContent.appendChild(translation);
  const hint = document.createElement("p");
  hint.className = "analysis-library-hint";
  hint.textContent = "点击下面的单词、搭配或文法块即可存入个人词库；同一条目在新句子中再次收藏会累计次数。";
  analysisContent.appendChild(hint);
  const vocabulary = Array.isArray(data.vocabulary) ? data.vocabulary : [];
  appendAnalysisSection("词汇以及搭配：", vocabulary, "vocabulary");
  const grammar = Array.isArray(data.grammar) ? data.grammar : [];
  appendAnalysisSection("文法：", grammar, "grammar");
  analysisContent.scrollTop = 0;
}

async function openSentenceAnalysis(index, button, force = false) {
  if (index < 0 || index >= playerSentences.length || !activeAnalysisUrl) return;
  const sentence = playerSentences[index];
  analysisSentenceIndex = index;
  activeAnalysisData = null;
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
    activeJobId = job.job_id;
    renderJob(job);
    pollJob(job.job_id);
  } catch (error) {
    setBusy(false);
    showError(error.message);
  }
});
cancelJobButton.addEventListener("click", cancelActiveJob);

for (const button of document.querySelectorAll("[data-key-toggle]")) {
  button.addEventListener("click", () => {
    const input = document.getElementById(button.dataset.keyToggle);
    const reveal = input.type === "password";
    input.type = reveal ? "text" : "password";
    button.textContent = reveal ? "隐藏" : "显示";
  });
}
for (const input of sharedValueInputs) {
  input.addEventListener("focus", () => renderSharedSuggestions(input));
  input.addEventListener("input", () => renderSharedSuggestions(input));
  input.addEventListener("blur", () => {
    window.setTimeout(() => suggestionPanel(input).classList.add("hidden"), 120);
  });
}
document.querySelector("#close-error").addEventListener("click", () => errorDialog.close());
document.querySelector("#open-library").addEventListener("click", openStudyLibrary);
document.querySelector("#close-library").addEventListener("click", () => libraryDialog.close());
libraryContent.addEventListener("click", (event) => {
  const deleteButton = event.target.closest("[data-library-delete]");
  if (deleteButton) {
    deleteLibraryEntry(deleteButton);
    return;
  }
  const button = event.target.closest("[data-library-count]");
  if (button) toggleLibraryOccurrences(button);
});
exportLibrary.addEventListener("click", (event) => {
  if (exportLibrary.getAttribute("aria-disabled") === "true") event.preventDefault();
});
document.querySelector("#close-analysis").addEventListener("click", () => analysisDialog.close());
analysisContent.addEventListener("click", (event) => {
  const block = event.target.closest("[data-library-type][data-library-index]");
  if (block) saveLearningBlock(block);
});
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
clearHistoryButton.addEventListener("click", clearHistory);
historyBody.addEventListener("click", (event) => {
  const deleteButton = event.target.closest("[data-history-delete]");
  if (deleteButton) {
    deleteHistoryEntry(deleteButton);
    return;
  }
  const statusTarget = event.target.closest("[data-history-status]");
  if (statusTarget) {
    viewHistoryJob(statusTarget.dataset.historyStatus);
  }
});
document.querySelector("#close-player").addEventListener("click", () => {
  playerAudio.pause();
  playerPanel.classList.add("hidden");
});
sentenceList.addEventListener("click", (event) => {
  const sentenceDeleteButton = event.target.closest("[data-sentence-delete]");
  if (sentenceDeleteButton) {
    deletePlayerSentence(Number(sentenceDeleteButton.dataset.sentenceDelete), sentenceDeleteButton);
    return;
  }
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
  if (event.button !== 0 || event.target.closest("[data-analysis-index], [data-sentence-delete]")) return;
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
