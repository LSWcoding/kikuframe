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
const toast = document.querySelector("#toast");

let pollingTimer = null;

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
    if (key === "SUBMD_OCR_API_KEY_CONFIGURED") continue;
    const input = form.elements.namedItem(key);
    if (input) input.value = value || "";
  }
  const configured = Boolean(config.SUBMD_OCR_API_KEY_CONFIGURED);
  keyHint.textContent = configured ? "已配置；留空将保留现有 Key" : "尚未配置";
  apiKeyInput.placeholder = configured ? "••••••••（留空保持不变）" : "输入后保存在本机";
  apiKeyInput.required = !configured;
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
      const result = files.length
        ? files.map((file) => `<a class="download-link" href="${file.download_url}">${escapeHtml(file.label || file.name)}</a>`).join("")
        : "—";
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
    resultLinks.innerHTML = files.map((file) =>
      `<a href="${file.download_url}">${escapeHtml(file.label || file.name)}</a>`
    ).join("");
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
    }
  } catch (error) {
    stopPolling();
    setBusy(false);
    showError(error.message);
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
document.querySelector("#close-error").addEventListener("click", () => errorDialog.close());
document.querySelector("#refresh-history").addEventListener("click", loadHistory);

loadConfig();
loadHistory();
