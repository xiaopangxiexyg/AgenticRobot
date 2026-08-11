"use strict";

const state = {
  episodes: [],
  detail: null,
  draft: null,
  savedDraftSnapshot: null,
  selectedEpisode: null,
  selectedCamera: null,
  selectedOccurrenceId: null,
  abnormalStart: null,
  recoverableFailureStart: null,
  dragging: null,
  busy: false,
  workspaceConfigured: false,
  workspace: null,
  serverOnline: true,
  episodeRefreshInFlight: false,
  episodeRevision: null,
  episodeRenderLimit: 200,
  episodeScanInProgress: false,
  lastEpisodeRefreshError: null,
  workspaceGeneration: 0,
  privilegedBaseline: null,
  timelineMap: null,
  numericSampleTimestamps: null,
  pendingSeekTimestamp: null,
  pendingSeekFrame: 0,
  loadedMediaKey: null,
  loadingMediaKey: null,
  videoLoading: false,
  videoPrefetch: null,
  videoPrefetchTimer: 0,
  datasetBrowserPath: "",
  datasetBrowserPurpose: "launcher",
  datasetBrowserLoading: false,
  deletionHistory: [],
  sealedReeditBaseline: null,
  sealedReeditSourceId: null,
  episodePanelCollapsed: false,
  activeOperationId: null,
  operationProgressKind: null,
};

let pendingDraftTransition = null;
let activeVideoLoadCleanup = null;
let episodeSummaryObserver = null;
let episodeSummaryFlushTimer = 0;
const pendingEpisodeSummaryIds = new Set();
const inFlightEpisodeSummaryIds = new Set();
const episodeSummaryRefreshedAt = new Map();

const EPISODE_REFRESH_INTERVAL_MS = 4000;
const EPISODE_RENDER_BATCH_SIZE = 200;
const EPISODE_SUMMARY_BATCH_SIZE = 32;
const EPISODE_SUMMARY_REFRESH_MS = 15000;
const EPISODE_PANEL_STORAGE_KEY =
  "dexteleop.annotation.episode_panel_collapsed";
const DISPLAY_GAP_THRESHOLD_SEC = 1.0;
const DISPLAY_GAP_DURATION_SEC = 1.0;
const STEP_COLOR_FAMILIES = [
  {hue: 176, saturation: 63, coarseLightness: 60},
  {hue: 43, saturation: 84, coarseLightness: 65},
  {hue: 257, saturation: 88, coarseLightness: 72},
  {hue: 14, saturation: 91, coarseLightness: 68},
  {hue: 116, saturation: 56, coarseLightness: 64},
  {hue: 324, saturation: 75, coarseLightness: 69},
];

const $ = (id) => document.getElementById(id);
const video = $("episode-video");

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function cancelPendingSeek() {
  if (state.pendingSeekFrame) {
    window.cancelAnimationFrame(state.pendingSeekFrame);
  }
  state.pendingSeekFrame = 0;
  state.pendingSeekTimestamp = null;
}

function clearVideoLoadListeners() {
  if (!activeVideoLoadCleanup) return;
  const cleanup = activeVideoLoadCleanup;
  activeVideoLoadCleanup = null;
  cleanup();
}

function flushPendingSeek() {
  const timestamp = state.pendingSeekTimestamp;
  cancelPendingSeek();
  if (timestamp !== null) seekTimeline(timestamp);
}

function scheduleSeek(timestamp) {
  const value = Number(timestamp);
  if (!Number.isFinite(value)) return;
  state.pendingSeekTimestamp = value;
  if (state.pendingSeekFrame) return;
  state.pendingSeekFrame = window.requestAnimationFrame(() => {
    state.pendingSeekFrame = 0;
    const pending = state.pendingSeekTimestamp;
    state.pendingSeekTimestamp = null;
    if (pending !== null) seekTimeline(pending);
  });
}

function showToast(message, kind = "info", timeout = 3200) {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast ${kind}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.className = "toast hidden";
  }, timeout);
}

function setServerOnline(online) {
  const normalized = Boolean(online);
  const changed = state.serverOnline !== normalized;
  state.serverOnline = normalized;
  document.body.classList.toggle("server-offline", !normalized);
  $("connection-status").classList.toggle("hidden", normalized);
  if (changed && normalized) {
    showToast("标注服务连接已恢复，编辑功能已解锁。", "info", 4200);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {cache: "no-store", ...options});
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`服务返回了非 JSON 响应 (${response.status})`);
  }
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `请求失败 (${response.status})`);
  }
  return payload;
}

function renderOperationProgress({kind, message, percent = null}) {
  const root = $("operation-progress");
  const fill = $("operation-progress-fill");
  const numeric = Number(percent);
  const determinate = Number.isFinite(numeric);
  state.operationProgressKind = kind;
  $("operation-progress-label").textContent = message;
  $("operation-progress-percent").textContent = determinate
    ? `${Math.round(clamp(numeric, 0, 100))}%`
    : "处理中";
  fill.classList.toggle("indeterminate", !determinate);
  fill.style.width = determinate
    ? `${clamp(numeric, 0, 100)}%`
    : "38%";
  root.classList.remove("hidden");
}

function hideOperationProgress(expectedKind = null) {
  if (
    expectedKind
    && state.operationProgressKind !== expectedKind
  ) return;
  $("operation-progress").classList.add("hidden");
  state.operationProgressKind = null;
}

function newOperationId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `operation-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function pollOperationProgress(operationId) {
  while (state.activeOperationId === operationId) {
    try {
      const payload = await requestJson(
        `/api/operation-progress?id=${encodeURIComponent(operationId)}`,
      );
      const progress = payload.progress;
      if (progress && state.activeOperationId === operationId) {
        renderOperationProgress({
          kind: progress.kind,
          message: progress.message,
          percent: progress.percent,
        });
        if (!progress.active) return;
      }
    } catch (_error) {
      // The main request remains authoritative. A transient polling failure
      // must not turn a successful annotation operation into a UI error.
    }
    await new Promise((resolve) => window.setTimeout(resolve, 120));
  }
}

async function requestJsonWithProgress(
  url,
  options,
  {kind, message, completeMessage},
) {
  const operationId = newOperationId();
  state.activeOperationId = operationId;
  renderOperationProgress({kind, message, percent: 0});
  void pollOperationProgress(operationId);
  try {
    const headers = {
      ...(options?.headers || {}),
      "X-Annotation-Operation": operationId,
    };
    const payload = await requestJson(url, {...(options || {}), headers});
    if (state.activeOperationId === operationId) {
      renderOperationProgress({
        kind,
        message: completeMessage,
        percent: 100,
      });
    }
    return payload;
  } finally {
    if (state.activeOperationId === operationId) {
      state.activeOperationId = null;
      window.setTimeout(() => hideOperationProgress(kind), 450);
    }
  }
}

function renderCatalogScanProgress(payload) {
  if (state.activeOperationId) return;
  if (!payload.refreshing) {
    hideOperationProgress("CATALOG_SCAN");
    return;
  }
  const completed = Number(payload.refresh_progress?.completed);
  const total = Number(payload.refresh_progress?.total);
  const determinate = Number.isFinite(total) && total > 0;
  renderOperationProgress({
    kind: "CATALOG_SCAN",
    message: determinate
      ? `正在扫描可标注 Episode（${completed}/${total}）`
      : "正在读取 Episode 目录",
    percent: determinate ? completed / total * 100 : null,
  });
}

function rememberedValues(key) {
  try {
    const values = JSON.parse(window.localStorage.getItem(key) || "[]");
    return Array.isArray(values)
      ? values.filter((value) => typeof value === "string" && value.trim())
      : [];
  } catch (_error) {
    return [];
  }
}

function rememberValue(key, value) {
  const values = [
    String(value).trim(),
    ...rememberedValues(key),
  ].filter((item, index, all) => item && all.indexOf(item) === index);
  try {
    window.localStorage.setItem(key, JSON.stringify(values.slice(0, 8)));
  } catch (_error) {
    // Workspace selection must not fail when browser storage is unavailable.
  }
}

function storedBoolean(key, fallback = false) {
  try {
    const value = window.localStorage.getItem(key);
    if (value === null) return fallback;
    return value === "1" || value === "true";
  } catch (_error) {
    return fallback;
  }
}

function storeBoolean(key, value) {
  try {
    window.localStorage.setItem(key, value ? "1" : "0");
  } catch (_error) {
    // Layout preferences are optional and must never block annotation.
  }
}

function canonicalDraftValue(value) {
  if (Array.isArray(value)) return value.map(canonicalDraftValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalDraftValue(value[key])]),
    );
  }
  return value;
}

function draftsEqual(left, right) {
  return JSON.stringify(canonicalDraftValue(left))
    === JSON.stringify(canonicalDraftValue(right));
}

function setSavedDraftSnapshot(draft) {
  state.savedDraftSnapshot = draft ? structuredClone(draft) : null;
}

function hasUnsavedDraftChanges() {
  if (!state.detail || !state.draft) return false;
  if (!state.savedDraftSnapshot) return true;
  return !draftsEqual(state.draft, state.savedDraftSnapshot);
}

function settleDraftTransition(decision) {
  if (!pendingDraftTransition) return;
  const pending = pendingDraftTransition;
  pendingDraftTransition = null;
  $("draft-warning-modal").classList.add("hidden");
  pending.resolve(decision);
}

function askDraftTransition(targetLabel) {
  if (!hasUnsavedDraftChanges()) return Promise.resolve("proceed");
  if (pendingDraftTransition) return Promise.resolve("cancel");
  return new Promise((resolve) => {
    pendingDraftTransition = {resolve};
    $("draft-warning-target").textContent = targetLabel;
    $("draft-warning-modal").classList.remove("hidden");
    $("draft-warning-save").focus();
  });
}

async function ensureAnnotationsSealedBeforeTransition(targetLabel) {
  const decision = await askDraftTransition(targetLabel);
  if (decision === "proceed" || decision === "discard") return true;
  if (decision !== "seal") return false;
  try {
    const report = await publishCurrentAnnotations();
    showToast(
      report.draft_cache_warning
        ? `标注已封存，但工作缓存更新失败：${report.draft_cache_warning}`
        : "当前修改已校验并封存，继续切换。",
      report.draft_cache_warning ? "error" : "info",
      report.draft_cache_warning ? 9000 : 4200,
    );
    return true;
  } catch (error) {
    showToast(`封存失败，已取消切换：${error.message}`, "error", 8000);
    return false;
  }
}

function selectedEpisodeListIndex() {
  return state.episodes.findIndex(
    (value) => value.episode_uuid === state.selectedEpisode,
  );
}

function updateEpisodeNavigation() {
  const count = state.episodes.length;
  const index = selectedEpisodeListIndex();
  const hasSelection = index >= 0;
  const positionText = hasSelection ? `${index + 1} / ${count}` : `0 / ${count}`;
  $("episode-position").textContent = positionText;
  $("collapsed-episode-position").textContent = positionText;
  $("collapsed-episode-number").textContent = hasSelection
    ? `#${state.episodes[index].episode_index}`
    : "—";
  $("collapsed-episode-number").title = hasSelection
    ? state.episodes[index].episode_uuid
    : "尚未选择 Episode";
  const selected = hasSelection ? state.episodes[index] : null;
  const status = $("collapsed-episode-status");
  const labels = [];
  if (selected?.whole_episode_abnormal) labels.push("整条异常");
  if (selected?.review_state === "PENDING") labels.push("待复核");
  else if (selected?.review_state === "COMPLETED") labels.push("已复核");
  status.textContent = labels.join(" / ") || "—";
  status.className = [
    "collapsed-episode-status",
    selected?.whole_episode_abnormal ? "abnormal" : "",
    selected?.review_state === "PENDING" ? "review" : "",
    selected?.review_state === "COMPLETED" ? "review-complete" : "",
  ].filter(Boolean).join(" ");
  status.title = labels.join("；") || "当前 Episode 无整条异常或复核状态";
  for (const id of ["previous-episode", "previous-episode-collapsed"]) {
    $(id).disabled = !hasSelection || index <= 0;
  }
  for (const id of ["next-episode", "next-episode-collapsed"]) {
    $(id).disabled = count === 0 || (hasSelection && index >= count - 1);
  }
}

function setEpisodePanelCollapsed(collapsed, {persist = true} = {}) {
  state.episodePanelCollapsed = Boolean(collapsed);
  $("annotation-workspace").classList.toggle(
    "episode-panel-is-collapsed",
    state.episodePanelCollapsed,
  );
  $("episode-panel-expanded").classList.toggle(
    "hidden",
    state.episodePanelCollapsed,
  );
  $("episode-panel-collapsed").classList.toggle(
    "hidden",
    !state.episodePanelCollapsed,
  );
  $("episode-panel").setAttribute(
    "aria-label",
    state.episodePanelCollapsed ? "Episode 快速切换" : "Episode 选择列表",
  );
  if (persist) {
    storeBoolean(EPISODE_PANEL_STORAGE_KEY, state.episodePanelCollapsed);
  }
  updateEpisodeNavigation();
}

async function navigateEpisode(offset) {
  if (state.busy || !state.episodes.length) return;
  const index = selectedEpisodeListIndex();
  if (index < 0) {
    const initialIndex = offset < 0 ? state.episodes.length - 1 : 0;
    await selectEpisode(state.episodes[initialIndex].episode_uuid);
    return;
  }
  const targetIndex = clamp(index + offset, 0, state.episodes.length - 1);
  if (targetIndex === index) return;
  await selectEpisode(state.episodes[targetIndex].episode_uuid);
}

function renderRememberedOptions() {
  for (const [listId, storageKey] of [
    ["recent-dataset-roots", "dexteleop.annotation.dataset_roots"],
    ["recent-annotator-ids", "dexteleop.annotation.annotator_ids"],
  ]) {
    const root = $(listId);
    root.replaceChildren();
    for (const value of rememberedValues(storageKey)) {
      const option = document.createElement("option");
      option.value = value;
      root.appendChild(option);
    }
  }
}

function showConfiguredWorkspace(workspace) {
  state.workspaceConfigured = true;
  state.workspace = workspace;
  $("workspace-launcher").classList.add("hidden");
  $("annotation-workspace").classList.remove("hidden");
  $("dataset-root").textContent = workspace.dataset_root;
  $("dataset-root").title =
    `${workspace.dataset_root} · annotator=${workspace.annotator_id}`;
  $("annotation-mode").textContent = workspace.second_review
    ? "二次复核"
    : "普通标注";
  $("annotation-mode").title = workspace.second_review
    ? "本次封存将标记为 SECOND_REVIEW"
    : "本次封存将标记为 STANDARD";
  $("change-dataset-root").disabled = false;
  $("refresh-episodes").disabled = false;
  startEpisodePolling();
}

async function initializeWorkspace() {
  state.episodePanelCollapsed = storedBoolean(
    EPISODE_PANEL_STORAGE_KEY,
    false,
  );
  setEpisodePanelCollapsed(state.episodePanelCollapsed, {persist: false});
  renderRememberedOptions();
  try {
    const payload = await requestJson("/api/workspace");
    const workspace = payload.workspace;
    $("launcher-dataset-root").value = workspace.dataset_root || "";
    $("launcher-annotator-id").value = workspace.annotator_id || "";
    $("launcher-second-review").checked = Boolean(workspace.second_review);
    if (workspace.configured) {
      showConfiguredWorkspace(workspace);
      await loadEpisodes();
    }
  } catch (error) {
    $("launcher-status").textContent = error.message;
    $("launcher-status").classList.add("error");
  }
}

async function openWorkspace() {
  if (state.busy) return;
  const datasetRoot = $("launcher-dataset-root").value.trim();
  const annotatorId = $("launcher-annotator-id").value.trim();
  const secondReview = $("launcher-second-review").checked;
  state.busy = true;
  $("open-workspace").disabled = true;
  try {
    const payload = await requestJson("/api/workspace", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        dataset_root: datasetRoot,
        annotator_id: annotatorId,
        second_review: secondReview,
      }),
    });
    rememberValue("dexteleop.annotation.dataset_roots", datasetRoot);
    rememberValue("dexteleop.annotation.annotator_ids", annotatorId);
    showConfiguredWorkspace(payload.workspace);
    await loadEpisodes();
    $("launcher-status").classList.remove("error");
    showToast(
      payload.workspace.episode_scan_in_progress
        ? "工作区已打开，正在后台扫描可标注 Episode。"
        : `工作区已打开，可标注 ${payload.workspace.eligible_episode_count} 个 Episode。`,
      "info",
      5000,
    );
  } catch (error) {
    $("launcher-status").textContent = error.message;
    $("launcher-status").classList.add("error");
  } finally {
    state.busy = false;
    $("open-workspace").disabled = false;
  }
}

function setDatasetBrowserStatus(message, kind = "") {
  const status = $("dataset-browser-status");
  status.textContent = message;
  status.className = `dataset-browser-status${kind ? ` ${kind}` : ""}`;
}

function renderDatasetBrowserDirectories(directories) {
  const root = $("dataset-browser-list");
  root.replaceChildren();
  if (!directories.length) {
    root.appendChild(
      element(
        "div",
        "dataset-browser-empty",
        "当前目录没有可进入的子目录。可以直接选择当前目录，或在上方输入路径。",
      ),
    );
    return;
  }
  for (const directory of directories) {
    const button = element("button", "dataset-browser-entry");
    button.type = "button";
    button.title = directory.path;
    const label = element("span");
    label.appendChild(element("strong", "", directory.name));
    label.appendChild(element("small", "", "目录"));
    button.appendChild(label);
    button.appendChild(element("small", "", "进入 ›"));
    button.addEventListener("click", () => browseDatasetPath(directory.path));
    root.appendChild(button);
  }
}

async function browseDatasetPath(path = "") {
  if (state.datasetBrowserLoading) return;
  const requested = String(path || $("dataset-browser-path").value || "").trim();
  state.datasetBrowserLoading = true;
  state.datasetBrowserPath = "";
  $("dataset-browser-go").disabled = true;
  $("dataset-browser-parent").disabled = true;
  $("dataset-browser-select").disabled = true;
  setDatasetBrowserStatus("正在读取目录…");
  try {
    const payload = await requestJson(
      `/api/dataset-browser?path=${encodeURIComponent(requested)}`,
    );
    setServerOnline(true);
    state.datasetBrowserPath = payload.current;
    $("dataset-browser-path").value = payload.current;
    $("dataset-browser-current").textContent = payload.current;
    $("dataset-browser-current").title = payload.current;
    const parent = payload.parent || "";
    $("dataset-browser-parent").dataset.path = parent;
    $("dataset-browser-parent").disabled = !parent;
    renderDatasetBrowserDirectories(payload.directories || []);
    setDatasetBrowserStatus(
      `${(payload.directories || []).length} 个子目录 · 选择当前目录不会修改数据集。`,
    );
  } catch (error) {
    setDatasetBrowserStatus(error.message, "error");
  } finally {
    state.datasetBrowserLoading = false;
    $("dataset-browser-go").disabled = false;
    $("dataset-browser-select").disabled = !state.datasetBrowserPath;
  }
}

function closeDatasetBrowser() {
  $("dataset-browser-modal").classList.add("hidden");
  state.datasetBrowserLoading = false;
}

function openDatasetBrowser(purpose = "launcher") {
  if (state.busy || !state.serverOnline) return;
  state.datasetBrowserPurpose = purpose;
  state.datasetBrowserPath = "";
  const initial = purpose === "workspace"
    ? state.workspace?.dataset_root || ""
    : $("launcher-dataset-root").value.trim();
  $("dataset-browser-modal").classList.remove("hidden");
  $("dataset-browser-title").textContent = purpose === "workspace"
    ? "更换数据集目录"
    : "选择数据集目录";
  $("dataset-browser-path").value = initial;
  $("dataset-browser-select").disabled = true;
  void browseDatasetPath(initial);
}

function resetAnnotationStateForWorkspace() {
  state.workspaceGeneration += 1;
  cancelPendingSeek();
  cancelVideoPrefetch();
  clearVideoLoadListeners();
  video.pause();
  video.removeAttribute("src");
  video.load();
  state.episodes = [];
  state.episodeRevision = null;
  state.episodeRenderLimit = EPISODE_RENDER_BATCH_SIZE;
  state.episodeScanInProgress = false;
  state.lastEpisodeRefreshError = null;
  pendingEpisodeSummaryIds.clear();
  inFlightEpisodeSummaryIds.clear();
  episodeSummaryRefreshedAt.clear();
  if (episodeSummaryObserver) episodeSummaryObserver.disconnect();
  state.detail = null;
  state.draft = null;
  setSavedDraftSnapshot(null);
  state.selectedEpisode = null;
  state.selectedCamera = null;
  state.selectedOccurrenceId = null;
  state.abnormalStart = null;
  state.recoverableFailureStart = null;
  state.timelineMap = null;
  state.numericSampleTimestamps = null;
  state.loadedMediaKey = null;
  state.loadingMediaKey = null;
  state.videoLoading = false;
  renderEpisodeList();
  renderEditor();
  updateEpisodeNavigation();
}

async function switchWorkspace(datasetRoot) {
  if (state.busy || !state.workspace) return;
  const annotatorId = state.workspace.annotator_id;
  state.busy = true;
  $("change-dataset-root").disabled = true;
  try {
    if (!await ensureAnnotationsSealedBeforeTransition("切换数据集")) return;
    const payload = await requestJson("/api/switch-workspace", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        dataset_root: datasetRoot,
        annotator_id: annotatorId,
        second_review: Boolean(state.workspace.second_review),
      }),
    });
    rememberValue("dexteleop.annotation.dataset_roots", datasetRoot);
    resetAnnotationStateForWorkspace();
    showConfiguredWorkspace(payload.workspace);
    await loadEpisodes();
    showToast(
      payload.workspace.episode_scan_in_progress
        ? "已切换数据集，正在后台扫描可标注 Episode。"
        : `已切换数据集，可标注 ${payload.workspace.eligible_episode_count} 个 Episode。`,
      "info",
      5000,
    );
  } catch (error) {
    showToast(`切换数据集失败：${error.message}`, "error", 8000);
  } finally {
    state.busy = false;
    $("change-dataset-root").disabled = false;
  }
}

function selectDatasetBrowserPath() {
  const selected = state.datasetBrowserPath;
  if (!selected) return;
  if (state.datasetBrowserPurpose === "launcher") {
    $("launcher-dataset-root").value = selected;
    $("launcher-status").textContent =
      "已选择数据集目录；请填写标注员 ID 后打开工作区。";
    $("launcher-status").classList.remove("error");
    closeDatasetBrowser();
    return;
  }
  if (selected === state.workspace?.dataset_root) {
    closeDatasetBrowser();
    return;
  }
  closeDatasetBrowser();
  void switchWorkspace(selected);
}

function episodeDuration() {
  if (!state.detail) return 1;
  return Math.max(1e-6, state.detail.end_timestamp - state.detail.start_timestamp);
}

function sampleTimestamps() {
  if (state.numericSampleTimestamps !== null) {
    return state.numericSampleTimestamps;
  }
  const values = state.detail?.sample_timestamps;
  state.numericSampleTimestamps = Array.isArray(values) && values.length
    ? values.map(Number)
    : [state.detail.start_timestamp, state.detail.end_timestamp];
  return state.numericSampleTimestamps;
}

function buildTimelineMap() {
  const real = sampleTimestamps();
  const positiveDeltas = [];
  for (let index = 1; index < real.length; index += 1) {
    const delta = real[index] - real[index - 1];
    if (delta > 0 && delta <= DISPLAY_GAP_THRESHOLD_SEC) positiveDeltas.push(delta);
  }
  positiveDeltas.sort((a, b) => a - b);
  const nominal = positiveDeltas.length
    ? positiveDeltas[Math.floor(positiveDeltas.length / 2)]
    : 1 / 15;
  const display = [0];
  const compressedGaps = [];
  for (let index = 1; index < real.length; index += 1) {
    const realDelta = Math.max(0, real[index] - real[index - 1]);
    const compressed = realDelta > DISPLAY_GAP_THRESHOLD_SEC;
    const displayDelta = compressed
      ? DISPLAY_GAP_DURATION_SEC
      : Math.max(realDelta, nominal);
    const displayStart = display[index - 1];
    display.push(displayStart + displayDelta);
    if (compressed) {
      compressedGaps.push({
        realStart: real[index - 1],
        realEnd: real[index],
        realDuration: realDelta,
        displayStart,
        displayEnd: displayStart + displayDelta,
      });
    }
  }
  state.timelineMap = {
    real,
    display,
    compressedGaps,
    displayDuration: Math.max(display[display.length - 1] || nominal, nominal),
  };
}

function segmentAtOrBefore(values, target) {
  let lower = 0;
  let upper = values.length - 1;
  while (lower < upper) {
    const middle = Math.ceil((lower + upper) / 2);
    if (values[middle] <= target) lower = middle;
    else upper = middle - 1;
  }
  return Math.min(lower, values.length - 2);
}

function interpolateSegment(source, target, value) {
  if (source.length < 2 || target.length < 2) return target[0] ?? value;
  if (value <= source[0]) return target[0];
  if (value >= source[source.length - 1]) return target[target.length - 1];
  const index = segmentAtOrBefore(source, value);
  const span = source[index + 1] - source[index];
  const fraction = span > 0 ? (value - source[index]) / span : 0;
  return target[index] + fraction * (target[index + 1] - target[index]);
}

function timelineRatio(timestamp) {
  if (!state.timelineMap) buildTimelineMap();
  const offset = interpolateSegment(
    state.timelineMap.real,
    state.timelineMap.display,
    Number(timestamp),
  );
  return clamp(offset / state.timelineMap.displayDuration, 0, 1);
}

function timelineAtRatio(ratio) {
  if (!state.timelineMap) buildTimelineMap();
  return interpolateSegment(
    state.timelineMap.display,
    state.timelineMap.real,
    clamp(ratio, 0, 1) * state.timelineMap.displayDuration,
  );
}

function firstSampleIndexAtOrAfter(timestamp) {
  const timestamps = sampleTimestamps();
  let lower = 0;
  let upper = timestamps.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (timestamps[middle] < timestamp) lower = middle + 1;
    else upper = middle;
  }
  return clamp(lower, 0, Math.max(0, timestamps.length - 1));
}

function nearestSampleIndex(timestamp) {
  const timestamps = sampleTimestamps();
  if (timestamps.length <= 1) return 0;
  const right = firstSampleIndexAtOrAfter(timestamp);
  if (right <= 0) return 0;
  if (right >= timestamps.length) return timestamps.length - 1;
  const left = right - 1;
  return Math.abs(timestamps[right] - timestamp)
    < Math.abs(timestamp - timestamps[left])
    ? right
    : left;
}

function lastSampleIndexAtOrBefore(timestamp) {
  const timestamps = sampleTimestamps();
  let lower = 0;
  let upper = timestamps.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (timestamps[middle] <= timestamp) lower = middle + 1;
    else upper = middle;
  }
  return clamp(lower - 1, 0, Math.max(0, timestamps.length - 1));
}

function timestampAtVideoRatio(ratio) {
  const timestamps = sampleTimestamps();
  if (timestamps.length < 2) return timestamps[0] ?? state.detail.start_timestamp;
  const coordinate = clamp(ratio, 0, 1) * (timestamps.length - 1);
  const index = Math.min(Math.floor(coordinate), timestamps.length - 2);
  const fraction = coordinate - index;
  return timestamps[index] + fraction * (timestamps[index + 1] - timestamps[index]);
}

function videoRatioAtTimestamp(timestamp) {
  const timestamps = sampleTimestamps();
  if (timestamps.length < 2) return 0;
  const value = Number(timestamp);
  if (value <= timestamps[0]) return 0;
  if (value >= timestamps[timestamps.length - 1]) return 1;
  const index = segmentAtOrBefore(timestamps, value);
  const span = timestamps[index + 1] - timestamps[index];
  const fraction = span > 0 ? (value - timestamps[index]) / span : 0;
  return clamp((index + fraction) / (timestamps.length - 1), 0, 1);
}

function currentTimelineTime() {
  if (!state.detail) return 0;
  if (state.pendingSeekTimestamp !== null) {
    return state.pendingSeekTimestamp;
  }
  const duration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : 1;
  return timestampAtVideoRatio(video.currentTime / duration);
}

function seekTimeline(timestamp) {
  if (!state.detail) return;
  if (
    video.readyState < HTMLMediaElement.HAVE_METADATA
    || !Number.isFinite(video.duration)
    || video.duration <= 0
  ) {
    return;
  }
  const ratio = videoRatioAtTimestamp(timestamp);
  try {
    video.currentTime = ratio * video.duration;
  } catch (error) {
    showToast(`视频定位失败：${error.message || "媒体尚未就绪"}`, "error", 7000);
    return;
  }
  $("video-scrubber").value = String(Math.round(ratio * 1000));
  updatePlayhead(timestamp);
}

function sortedOccurrences() {
  return [...(state.draft?.occurrences || [])].sort((a, b) =>
    a.start_at - b.start_at
    || a.end_at - b.end_at
    || String(a.annotation_id).localeCompare(String(b.annotation_id))
  );
}

function attemptsById() {
  return new Map((state.detail?.attempts || []).map((value) => [value.attempt_id, value]));
}

function effectiveProgressBoundary(transition) {
  const override = (state.draft?.progress_boundary_overrides || []).find(
    (value) => value.source_advance_fact_id === transition.advance_fact_id,
  );
  return Number(override?.corrected_effective_at ?? transition.effective_at);
}

function effectiveAttemptBounds() {
  const bounds = new Map(
    (state.detail?.attempts || []).map((attempt) => [
      attempt.attempt_id,
      {
        start_at: Number(attempt.start_at),
        end_at: attempt.end_at === null ? null : Number(attempt.end_at),
      },
    ]),
  );
  for (const transition of state.detail?.transitions || []) {
    const timestamp = effectiveProgressBoundary(transition);
    const from = bounds.get(transition.from_attempt_id);
    const to = bounds.get(transition.to_attempt_id);
    if (from) from.end_at = timestamp;
    if (to) to.start_at = timestamp;
  }
  return bounds;
}

function coarseBoundaryLimits(transition) {
  const bounds = effectiveAttemptBounds();
  const from = bounds.get(transition.from_attempt_id);
  const to = bounds.get(transition.to_attempt_id);
  if (
    !from
    || !to
    || !Number.isFinite(from.start_at)
    || !Number.isFinite(to.end_at)
  ) {
    return null;
  }
  const minimum = Math.max(
    0.02,
    episodeDuration() / Math.max(10000, state.detail.frame_count * 20),
  );
  const lower = from.start_at + minimum;
  const upper = to.end_at - minimum;
  return lower < upper ? {lower, upper} : null;
}

function applyCoarseBoundaryCorrection(transition, proposed) {
  const limits = coarseBoundaryLimits(transition);
  if (!limits) return null;
  const corrected = clamp(Number(proposed), limits.lower, limits.upper);
  state.draft.progress_boundary_overrides =
    (state.draft.progress_boundary_overrides || []).filter(
      (value) => value.source_advance_fact_id !== transition.advance_fact_id,
    );
  if (Math.abs(corrected - Number(transition.effective_at)) > 1e-7) {
    state.draft.progress_boundary_overrides.push({
      annotation_id: `boundary:${transition.advance_fact_id}`,
      source_advance_fact_id: transition.advance_fact_id,
      corrected_effective_at: corrected,
    });
  }
  const fromValues = sortedOccurrences().filter(
    (value) => value.coarse_attempt_id === transition.from_attempt_id,
  );
  const toValues = sortedOccurrences().filter(
    (value) => value.coarse_attempt_id === transition.to_attempt_id,
  );
  const fromTail = fromValues.at(-1);
  const toHead = toValues[0];
  if (fromTail) fromTail.end_at = corrected;
  if (toHead) toHead.start_at = corrected;
  return corrected;
}

function restoreOriginalCoarseBoundaries() {
  for (const transition of state.detail?.transitions || []) {
    const original = Number(transition.effective_at);
    const fromValues = sortedOccurrences().filter(
      (value) => value.coarse_attempt_id === transition.from_attempt_id,
    );
    const toValues = sortedOccurrences().filter(
      (value) => value.coarse_attempt_id === transition.to_attempt_id,
    );
    if (fromValues.length) fromValues.at(-1).end_at = original;
    if (toValues.length) toValues[0].start_at = original;
  }
  state.draft.progress_boundary_overrides = [];
}

function editableTailOutcome() {
  const attempts = state.detail?.attempts || [];
  const attempt = attempts.at(-1);
  if (!state.draft || !attempt || attempt.failure_fact_id) return null;
  const attemptEnd = Number(attempt.end_at);
  if (!Number.isFinite(attemptEnd)) return null;
  const occurrence = sortedOccurrences()
    .filter((value) => value.coarse_attempt_id === attempt.attempt_id)
    .at(-1);
  if (
    !occurrence
    || occurrence.source_failure_fact_id
    || !["SUCCESS", "FAILURE"].includes(occurrence.outcome)
    || Math.abs(Number(occurrence.end_at) - attemptEnd) > 1e-7
  ) {
    return null;
  }
  return {attempt, occurrence};
}

function failureTriggers() {
  return [...(state.draft?.failure_triggers || [])].sort((a, b) =>
    Number(a.upper_bound_at) - Number(b.upper_bound_at)
    || String(a.source_failure_fact_id).localeCompare(String(b.source_failure_fact_id))
  );
}

function recoverableFailures() {
  return [...(state.draft?.recoverable_failures || [])].sort((a, b) =>
    Number(a.onset_sample_index) - Number(b.onset_sample_index)
    || Number(a.recovered_sample_index) - Number(b.recovered_sample_index)
    || String(a.annotation_id).localeCompare(String(b.annotation_id))
  );
}

function synchronizeFailureTriggerWindows() {
  if (!state.detail || !state.draft) return;
  const occurrences = new Map(
    sortedOccurrences()
      .filter((value) => value.outcome === "FAILURE" && value.source_failure_fact_id)
      .map((value) => [value.source_failure_fact_id, value]),
  );
  const attempts = attemptsById();
  for (const trigger of state.draft.failure_triggers || []) {
    const occurrence = occurrences.get(trigger.source_failure_fact_id);
    const attempt = occurrence ? attempts.get(occurrence.coarse_attempt_id) : null;
    if (!occurrence || !attempt || attempt.failure_reported_at === null) continue;
    const lower = firstSampleIndexAtOrAfter(occurrence.start_at);
    const upper = lastSampleIndexAtOrBefore(attempt.failure_reported_at);
    if (lower > upper) continue;
    const previous = Number(trigger.trigger_sample_index);
    const rebased = Number.isInteger(previous) && previous >= lower && previous <= upper;
    const index = rebased ? previous : upper;
    trigger.coarse_attempt_id = occurrence.coarse_attempt_id;
    trigger.progress_step_id = occurrence.progress_step_id;
    trigger.stage_id = occurrence.stage_id;
    trigger.lower_bound_sample_index = lower;
    trigger.upper_bound_sample_index = upper;
    trigger.lower_bound_at = sampleTimestamps()[lower];
    trigger.upper_bound_at = sampleTimestamps()[upper];
    trigger.trigger_sample_index = index;
    trigger.trigger_at = sampleTimestamps()[index];
    if (!rebased) {
      trigger.source = "CAPTURE_DEFAULT";
      trigger.confidence = 1.0;
    }
  }
}

function stepsById() {
  return new Map((state.detail?.progress_steps || []).map((value) => [value.progress_step_id, value]));
}

function stagesForStep(stepId) {
  return stepsById().get(stepId)?.stages || [];
}

function stageById(stageId) {
  for (const step of state.detail?.progress_steps || []) {
    const found = step.stages.find((value) => value.stage_id === stageId);
    if (found) return found;
  }
  return null;
}

function stepForStage(stageId) {
  return (state.detail?.progress_steps || []).find(
    (step) => step.stages.some((stage) => stage.stage_id === stageId),
  ) || null;
}

function occurrenceOutcomeLabel(outcome) {
  if (outcome === "FAILURE") return "失败";
  if (outcome === "TRUNCATED_BY_ABNORMAL") return "异常截断";
  return "成功";
}

function stagePalette(stageId) {
  const stage = stageById(stageId);
  const step = stepForStage(stageId);
  const family = stepColorFamily(step?.progress_step_id);
  const stages = step?.stages || [];
  const index = Math.max(0, Number(stage?.ordinal || 1) - 1);
  const denominator = Math.max(1, stages.length - 1);
  const tone = clamp(index / denominator, 0, 1);
  const solidLightness = 42 + 32 * tone;
  const backgroundLightness = 28 + 40 * tone;
  const backgroundAlpha = 0.42 - 0.08 * tone;
  return {
    solid:
      `hsl(${family.hue} ${family.saturation}% ${solidLightness.toFixed(1)}%)`,
    background:
      `hsl(${family.hue} ${family.saturation}% ${backgroundLightness.toFixed(1)}% / ${backgroundAlpha.toFixed(2)})`,
    text: `hsl(${family.hue} 48% 94%)`,
  };
}

function stepColorFamily(stepId) {
  const step = stepsById().get(stepId);
  const index = Math.max(0, Number(step?.ordinal || 1) - 1);
  return STEP_COLOR_FAMILIES[index % STEP_COLOR_FAMILIES.length];
}

function stepPalette(stepId) {
  const family = stepColorFamily(stepId);
  return {
    solid:
      `hsl(${family.hue} ${family.saturation}% ${family.coarseLightness}%)`,
    background:
      `hsl(${family.hue} ${family.saturation}% ${family.coarseLightness}% / 0.27)`,
    text: `hsl(${family.hue} 48% 94%)`,
  };
}

function intervalsOverlap(leftStart, leftEnd, rightStart, rightEnd) {
  return leftStart < rightEnd && rightStart < leftEnd;
}

function occurrenceOverlapsAbnormal(occurrence) {
  return (state.draft?.abnormal_intervals || []).some((abnormal) =>
    intervalsOverlap(
      occurrence.start_at,
      occurrence.end_at,
      abnormal.start_at,
      abnormal.end_at,
    )
  );
}

function occurrenceReferenceIssues() {
  const attempts = attemptsById();
  const issues = [];
  for (const value of state.draft?.occurrences || []) {
    const attempt = attempts.get(value.coarse_attempt_id);
    const start = Number(value.start_at);
    const end = Number(value.end_at);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      issues.push(`${value.annotation_id || "未知 occurrence"} 的时间范围无效`);
      continue;
    }
    if (!attempt) {
      issues.push(`${value.annotation_id || "未知 occurrence"} 引用了未知 Attempt`);
      continue;
    }
    if (!stagesForStep(attempt.progress_step_id).some(
      (stage) => stage.stage_id === value.stage_id,
    )) {
      issues.push(`${value.annotation_id || "未知 occurrence"} 的 stage 不属于其 Attempt`);
    }
  }
  return issues;
}

function timelineCoverageGaps() {
  if (!state.detail) return [];
  const episodeStart = Number(state.detail.start_timestamp);
  const episodeEnd = Number(state.detail.end_timestamp);
  const intervals = sortedOccurrences()
    .map((value) => ({
      start: clamp(Number(value.start_at), episodeStart, episodeEnd),
      end: clamp(Number(value.end_at), episodeStart, episodeEnd),
    }))
    .filter((value) => (
      Number.isFinite(value.start)
      && Number.isFinite(value.end)
      && value.end > value.start
    ));
  const merged = [];
  for (const interval of intervals) {
    const previous = merged.at(-1);
    if (previous && interval.start <= previous.end + 1e-9) {
      previous.end = Math.max(previous.end, interval.end);
    } else {
      merged.push({...interval});
    }
  }
  const rawGaps = [];
  let cursor = episodeStart;
  for (const interval of merged) {
    if (interval.start > cursor + 1e-9) {
      rawGaps.push({start: cursor, end: interval.start});
    }
    cursor = Math.max(cursor, interval.end);
  }
  if (cursor < episodeEnd - 1e-9) {
    rawGaps.push({start: cursor, end: episodeEnd});
  }

  const attempts = Array.from(effectiveAttemptBounds().values());
  const abnormal = state.draft?.abnormal_intervals || [];
  const classified = [];
  for (const gap of rawGaps) {
    const boundaries = [gap.start, gap.end];
    for (const interval of [...attempts, ...abnormal]) {
      const start = Number(interval.start_at);
      const end = Number(interval.end_at);
      if (start > gap.start && start < gap.end) boundaries.push(start);
      if (end > gap.start && end < gap.end) boundaries.push(end);
    }
    boundaries.sort((left, right) => left - right);
    for (let index = 1; index < boundaries.length; index += 1) {
      const start = boundaries[index - 1];
      const end = boundaries[index];
      if (end <= start + 1e-9) continue;
      const midpoint = (start + end) / 2;
      const isAbnormal = abnormal.some(
        (value) => Number(value.start_at) <= midpoint && midpoint < Number(value.end_at),
      );
      const insideAttempt = attempts.some(
        (value) => (
          Number(value.start_at) <= midpoint
          && (value.end_at === null || midpoint < Number(value.end_at))
        ),
      );
      classified.push({
        start,
        end,
        kind: isAbnormal ? "abnormal" : insideAttempt ? "unexplained" : "attempt-pause",
      });
    }
  }
  return classified;
}

function normalizeOrderedOccurrences() {
  const values = sortedOccurrences();
  const byAttempt = new Map();
  for (const value of values) {
    if (!byAttempt.has(value.coarse_attempt_id)) byAttempt.set(value.coarse_attempt_id, []);
    byAttempt.get(value.coarse_attempt_id).push(value);
  }

  const retained = [];
  for (const attempt of state.detail.attempts || []) {
    const attemptValues = (byAttempt.get(attempt.attempt_id) || []).sort((a, b) =>
      a.start_at - b.start_at
      || a.end_at - b.end_at
      || String(a.annotation_id).localeCompare(String(b.annotation_id))
    );
    const allowedStageIds = new Set(
      stagesForStep(attempt.progress_step_id).map((stage) => stage.stage_id),
    );
    let occurrenceIndex = 1;
    for (const value of attemptValues) {
      if (!allowedStageIds.has(value.stage_id)) continue;
      value.progress_step_id = attempt.progress_step_id;
      value.occurrence_index_in_attempt = occurrenceIndex++;
      retained.push(value);
    }
  }
  const retainedIds = new Set(retained.map((value) => value.annotation_id));
  const unresolved = values.filter(
    (value) => !retainedIds.has(value.annotation_id),
  );
  state.draft.occurrences = [...retained, ...unresolved].sort((a, b) =>
    Number(a.start_at) - Number(b.start_at)
    || Number(a.end_at) - Number(b.end_at)
    || String(a.annotation_id).localeCompare(String(b.annotation_id))
  );
  if (
    state.selectedOccurrenceId
    && !retained.some(
      (value) => value.annotation_id === state.selectedOccurrenceId,
    )
  ) {
    state.selectedOccurrenceId = null;
  }
}

function failedAttemptEditorModels() {
  const expectedByStep = new Map(
    (state.detail?.progress_steps || []).map((step) => [step.progress_step_id, 0]),
  );
  const entryStage = stageById(state.detail?.entry_stage_id);
  if (entryStage) {
    expectedByStep.set(
      entryStage.progress_step_id,
      Math.max(0, Number(entryStage.ordinal || 1) - 1),
    );
  }
  const grouped = new Map();
  for (const occurrence of sortedOccurrences()) {
    if (!grouped.has(occurrence.coarse_attempt_id)) {
      grouped.set(occurrence.coarse_attempt_id, []);
    }
    grouped.get(occurrence.coarse_attempt_id).push(occurrence);
  }

  const models = [];
  for (const attempt of state.detail?.attempts || []) {
    const stages = stagesForStep(attempt.progress_step_id);
    let expected = expectedByStep.get(attempt.progress_step_id) || 0;
    const entryIndex = clamp(expected, 0, Math.max(0, stages.length - 1));
    const values = grouped.get(attempt.attempt_id) || [];
    const failure = values.find(
      (value) =>
        value.outcome === "FAILURE"
        && value.source_failure_fact_id === attempt.failure_fact_id,
    );
    if (attempt.outcome === "FAILURE" && attempt.failure_fact_id) {
      models.push({
        attempt,
        stages,
        entryIndex,
        failure,
        values,
      });
    }
    for (const occurrence of values) {
      const index = stages.findIndex(
        (stage) => stage.stage_id === occurrence.stage_id,
      );
      if (index < 0) continue;
      if (occurrence.outcome === "SUCCESS") {
        expected = index + 1;
      } else if (occurrence.outcome === "FAILURE") {
        expected = index;
        break;
      }
    }
    expectedByStep.set(attempt.progress_step_id, expected);
  }
  return models;
}

async function reflowFailedAttemptStage(model, failedStageId, select) {
  if (!state.draft || !state.detail || state.busy) return;
  const currentStageId = model.failure?.stage_id || "";
  if (failedStageId === currentStageId) return;
  const attempts = state.detail.attempts || [];
  const attemptIndex = attempts.findIndex(
    (value) => value.attempt_id === model.attempt.attempt_id,
  );
  const affectedAttemptIds = new Set();
  for (const attempt of attempts.slice(attemptIndex)) {
    if (attempt.progress_step_id !== model.attempt.progress_step_id) break;
    affectedAttemptIds.add(attempt.attempt_id);
  }
  const affectedOccurrences = sortedOccurrences().filter(
    (value) => affectedAttemptIds.has(value.coarse_attempt_id),
  );
  if (
    affectedOccurrences.length > 1
    && !window.confirm(
      "更改失败阶段会按 profile 重新生成当前 Attempt 及同一子任务的后续 Retry 阶段块；这些 Attempt 中已手动调整的子阶段边界将被重新均分。原始 Failure、视频和样本不会改变。是否继续？",
    )
  ) {
    select.value = currentStageId;
    return;
  }

  state.busy = true;
  select.disabled = true;
  try {
    const payload = await requestJson(
      `/api/reflow-failed-stage/${encodeURIComponent(state.detail.episode_uuid)}`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          draft: state.draft,
          attempt_id: model.attempt.attempt_id,
          failed_stage_id: failedStageId,
        }),
      },
    );
    state.draft = payload.draft;
    state.deletionHistory = [];
    state.selectedOccurrenceId = null;
    state.abnormalStart = null;
    state.recoverableFailureStart = null;
    state.privilegedBaseline = null;
    renderTimeline();
    renderRecoverableFailureTools();
    renderPrivilegedTools();
    updateEditButtonState();
    const selectedStage = stageById(failedStageId);
    showToast(
      `失败阶段已设为“${selectedStage?.display_name || failedStageId}”，并重排 ${payload.changed_attempt_ids.length} 个 Attempt；请继续拖动白色边界精调。`,
      "info",
      6500,
    );
  } catch (error) {
    select.value = currentStageId;
    showToast(`失败阶段调整失败：${error.message}`, "error", 8500);
  } finally {
    state.busy = false;
    select.disabled = false;
  }
}

function renderFailedAttemptEditors() {
  const root = $("failed-attempt-editors");
  root.replaceChildren();
  for (const model of failedAttemptEditorModels()) {
    if (!model.stages.length) continue;
    const card = element("div", "failed-attempt-editor");
    const summary = element("div", "");
    const entry = model.stages[model.entryIndex];
    summary.appendChild(
      element(
        "strong",
        "",
        `${model.attempt.progress_step_display_name} · 失败 Attempt`,
      ),
    );
    summary.appendChild(
      element(
        "small",
        "",
        `本次从“${entry?.display_name || "未知"}”开始；原始 Failure ${formatTime(Number(model.attempt.failure_reported_at) - state.detail.start_timestamp)}，只读。`,
      ),
    );
    const label = element("label", "", "失败发生于");
    const select = element("select", "");
    for (const stage of model.stages.slice(model.entryIndex)) {
      const option = document.createElement("option");
      option.value = stage.stage_id;
      option.textContent = `${stage.ordinal}. ${stage.display_name}`;
      select.appendChild(option);
    }
    select.value = model.failure?.stage_id || entry?.stage_id || "";
    select.addEventListener("change", () => {
      void reflowFailedAttemptStage(model, select.value, select);
    });
    label.appendChild(select);
    card.appendChild(summary);
    card.appendChild(label);
    root.appendChild(card);
  }
  root.classList.toggle("hidden", !root.childElementCount);
}

function updatePlayhead(timestamp = currentTimelineTime()) {
  if (!state.detail) return;
  const left = `${timelineRatio(timestamp) * 100}%`;
  for (const track of [
    $("coarse-track"),
    $("stage-track"),
    $("failure-trigger-track"),
    $("recoverable-failure-track"),
  ]) {
    let playhead = track.querySelector(".playhead");
    if (!playhead) {
      playhead = element("div", "playhead");
      track.appendChild(playhead);
    }
    playhead.style.left = left;
  }
  $("video-timecode").textContent = formatTime(timestamp - state.detail.start_timestamp);
  const occurrence = sortedOccurrences().find(
    (value) => value.start_at <= timestamp && timestamp <= value.end_at,
  );
  if (!occurrence) {
    $("video-context").textContent = "未标注区间";
    return;
  }
  const stage = stageById(occurrence.stage_id);
  $("video-context").textContent =
    `${stage?.display_name || occurrence.stage_id} · ${occurrenceOutcomeLabel(occurrence.outcome)}`;
}

async function loadEpisodes({silent = false, force = false} = {}) {
  if (state.episodeRefreshInFlight || !state.workspaceConfigured) return;
  state.episodeRefreshInFlight = true;
  const previousIds = new Set(
    state.episodes.map((value) => value.episode_uuid),
  );
  try {
    const query = force ? "?refresh=1" : "?stale_ok=1";
    const payload = await requestJson(`/api/episodes${query}`);
    setServerOnline(true);
    state.episodeScanInProgress = Boolean(payload.refreshing);
    renderCatalogScanProgress(payload);
    if (payload.refresh_error && payload.refresh_error !== state.lastEpisodeRefreshError) {
      state.lastEpisodeRefreshError = payload.refresh_error;
      showToast(`Episode 扫描失败：${payload.refresh_error}`, "error", 9000);
    } else if (!payload.refresh_error) {
      state.lastEpisodeRefreshError = null;
    }
    const nextRevision = Number(payload.revision);
    const revisionKnown = Number.isFinite(nextRevision);
    const episodesChanged = !revisionKnown
      || state.episodeRevision !== nextRevision;
    if (episodesChanged) {
      state.episodes = payload.episodes;
      state.episodeRevision = revisionKnown ? nextRevision : null;
      const selectedIndex = state.episodes.findIndex(
        (value) => value.episode_uuid === state.selectedEpisode,
      );
      state.episodeRenderLimit = Math.max(
        state.episodeRenderLimit,
        EPISODE_RENDER_BATCH_SIZE,
        selectedIndex + 1,
      );
    }
    $("dataset-root").textContent = payload.dataset_root;
    $("episode-count").textContent = String(payload.episodes.length);
    if (episodesChanged) renderEpisodeList();
    else if (state.episodes.slice(0, state.episodeRenderLimit).some((item) => (
      item.episode_uuid !== state.selectedEpisode
      && Date.now() - (episodeSummaryRefreshedAt.get(item.episode_uuid) || 0)
        >= EPISODE_SUMMARY_REFRESH_MS
    ))) renderEpisodeList();
    if (
      state.selectedEpisode
      && !payload.episodes.some((value) => value.episode_uuid === state.selectedEpisode)
    ) {
      state.selectedEpisode = null;
      state.detail = null;
      state.draft = null;
      state.abnormalStart = null;
      state.recoverableFailureStart = null;
      renderEditor();
    }
    updateEpisodeNavigation();
    const added = payload.episodes.filter(
      (value) => !previousIds.has(value.episode_uuid),
    );
    if (silent && previousIds.size > 0 && added.length) {
      showToast(
        `发现 ${added.length} 个新封存 Episode，已加入列表。`,
        "info",
        4200,
      );
    }
  } catch (error) {
    setServerOnline(false);
    if (!silent) showToast(error.message, "error", 6000);
  } finally {
    state.episodeRefreshInFlight = false;
  }
}

function startEpisodePolling() {
  window.clearInterval(startEpisodePolling.timer);
  startEpisodePolling.timer = window.setInterval(() => {
    if (
      state.workspaceConfigured
      && !state.busy
      && document.visibilityState === "visible"
    ) {
      loadEpisodes({silent: true});
    }
  }, EPISODE_REFRESH_INTERVAL_MS);
}

function queueEpisodeSummary(episodeUuid) {
  if (inFlightEpisodeSummaryIds.has(episodeUuid)) return;
  pendingEpisodeSummaryIds.add(episodeUuid);
  if (episodeSummaryFlushTimer) return;
  episodeSummaryFlushTimer = window.setTimeout(
    flushEpisodeSummaryQueue,
    40,
  );
}

async function flushEpisodeSummaryQueue() {
  episodeSummaryFlushTimer = 0;
  const workspaceGeneration = state.workspaceGeneration;
  const episodeUuids = Array.from(pendingEpisodeSummaryIds).slice(
    0,
    EPISODE_SUMMARY_BATCH_SIZE,
  );
  if (!episodeUuids.length || !state.workspaceConfigured) return;
  for (const episodeUuid of episodeUuids) {
    pendingEpisodeSummaryIds.delete(episodeUuid);
    inFlightEpisodeSummaryIds.add(episodeUuid);
  }
  try {
    const payload = await requestJson("/api/episode-summaries", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({episode_uuids: episodeUuids}),
    });
    if (workspaceGeneration !== state.workspaceGeneration) return;
    const byId = new Map(
      payload.summaries.map((value) => [value.episode_uuid, value]),
    );
    const errorsById = new Map(
      payload.errors.map((value) => [value.episode_uuid, value.error]),
    );
    state.episodes = state.episodes.map((item) => (
      byId.has(item.episode_uuid)
        ? {
            ...item,
            ...byId.get(item.episode_uuid),
            annotation_summary_error: null,
          }
        : errorsById.has(item.episode_uuid)
          ? {
              ...item,
              annotation_summary_error: errorsById.get(item.episode_uuid),
            }
          : item
    ));
    const refreshedAt = Date.now();
    for (const episodeUuid of byId.keys()) {
      episodeSummaryRefreshedAt.set(episodeUuid, refreshedAt);
    }
    if (payload.errors.length) {
      showToast(
        `有 ${payload.errors.length} 个 Episode 的标注摘要暂时无法读取。`,
        "error",
        7000,
      );
    }
    renderEpisodeList();
  } catch (error) {
    if (workspaceGeneration === state.workspaceGeneration) {
      showToast(`读取可见 Episode 标注摘要失败：${error.message}`, "error", 7000);
    }
  } finally {
    for (const episodeUuid of episodeUuids) {
      inFlightEpisodeSummaryIds.delete(episodeUuid);
    }
    if (pendingEpisodeSummaryIds.size && !episodeSummaryFlushTimer) {
      episodeSummaryFlushTimer = window.setTimeout(
        flushEpisodeSummaryQueue,
        40,
      );
    }
  }
}

function renderEpisodeList() {
  const root = $("episode-list");
  root.replaceChildren();
  if (episodeSummaryObserver) episodeSummaryObserver.disconnect();
  episodeSummaryObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      episodeSummaryObserver.unobserve(entry.target);
      queueEpisodeSummary(entry.target.dataset.episodeUuid);
    }
  }, {root, rootMargin: "160px"});
  if (!state.episodes.length) {
    root.appendChild(
      element(
        "p",
        "panel-note",
        state.episodeScanInProgress
          ? "正在后台扫描可标注 Episode；页面保持可响应。"
          : "没有满足“Progress/migration 已封存、阶段 profile 完整、外部源已显式授权”的可标注 Episode。",
      ),
    );
    updateEpisodeNavigation();
    return;
  }
  const renderedEpisodes = state.episodes.slice(0, state.episodeRenderLimit);
  for (const item of renderedEpisodes) {
    const button = element(
      "button",
      `episode-item${item.episode_uuid === state.selectedEpisode ? " active" : ""}`,
    );
    button.type = "button";
    button.title = item.episode_uuid;
    button.dataset.episodeUuid = item.episode_uuid;
    button.appendChild(element("strong", "", `#${item.episode_index} · ${formatTime(item.duration_sec)}`));
    const badges = element("div", "episode-badges");
    badges.appendChild(element("span", "badge", `${item.frame_count} 帧`));
    if (item.resolution_kind === "RETROSPECTIVE_MIGRATED") {
      badges.appendChild(element("span", "badge", "历史迁移"));
    }
    if (item.source_batch_id) {
      badges.appendChild(element("span", "badge", item.source_batch_id));
    }
    if (item.annotation_set_count) {
      badges.appendChild(
        element(
          "span",
          "badge sealed",
          `已封存 ${item.annotation_set_count} 版`,
        ),
      );
    }
    if (item.annotation_summary_error) {
      const badge = element("span", "badge summary-error", "状态读取失败");
      badge.title = item.annotation_summary_error;
      badges.appendChild(badge);
    } else if (!item.annotation_summary_loaded) {
      badges.appendChild(element("span", "badge loading", "状态读取中"));
    }
    if (item.whole_episode_abnormal) {
      badges.appendChild(
        element(
          "span",
          "badge abnormal",
          `整条异常${item.whole_episode_abnormal_reason ? ` · ${item.whole_episode_abnormal_reason}` : ""}`,
        ),
      );
    }
    if (item.review_state === "PENDING" || item.needs_second_review) {
      badges.appendChild(element("span", "badge review", "待二次复核"));
    }
    if (
      item.review_state === "COMPLETED"
      || (
        item.review_state == null
        && item.latest_annotation_mode === "SECOND_REVIEW"
        && !item.needs_second_review
      )
    ) {
      badges.appendChild(element("span", "badge review-complete", "二次复核完成"));
    }
    button.appendChild(badges);
    button.addEventListener("click", () => selectEpisode(item.episode_uuid));
    root.appendChild(button);
    const refreshedAt = episodeSummaryRefreshedAt.get(item.episode_uuid) || 0;
    if (
      item.episode_uuid !== state.selectedEpisode
      && (
        !item.annotation_summary_loaded
        || Date.now() - refreshedAt >= EPISODE_SUMMARY_REFRESH_MS
      )
    ) {
      episodeSummaryObserver.observe(button);
    }
  }
  if (renderedEpisodes.length < state.episodes.length) {
    const remaining = state.episodes.length - renderedEpisodes.length;
    const loadMore = element(
      "button",
      "episode-item",
      `加载后续 ${Math.min(EPISODE_RENDER_BATCH_SIZE, remaining)} 条`,
    );
    loadMore.type = "button";
    loadMore.addEventListener("click", () => {
      state.episodeRenderLimit += EPISODE_RENDER_BATCH_SIZE;
      renderEpisodeList();
    });
    root.appendChild(loadMore);
  }
  updateEpisodeNavigation();
}

async function selectEpisode(episodeUuid) {
  if (state.busy || episodeUuid === state.selectedEpisode) return;
  state.busy = true;
  try {
    if (!await ensureAnnotationsSealedBeforeTransition("切换 Episode")) return;
    const payload = await requestJsonWithProgress(
      `/api/episode/${encodeURIComponent(episodeUuid)}`,
      {},
      {
        kind: "OPEN_EPISODE",
        message: "正在打开 Episode",
        completeMessage: "Episode 已打开",
      },
    );
    cancelPendingSeek();
    cancelVideoPrefetch();
    video.pause();
    state.selectedEpisode = episodeUuid;
    state.detail = payload.episode;
    state.draft = structuredClone(payload.episode.draft);
    setSavedDraftSnapshot(
      payload.episode.draft_requires_seal ? null : state.draft,
    );
    state.selectedCamera = payload.episode.default_camera_key;
    state.selectedOccurrenceId = null;
    state.abnormalStart = null;
    state.recoverableFailureStart = null;
    state.privilegedBaseline = null;
    state.timelineMap = null;
    state.numericSampleTimestamps = null;
    state.loadedMediaKey = null;
    state.loadingMediaKey = null;
    state.videoLoading = false;
    state.deletionHistory = structuredClone(
      state.draft.deletion_history || [],
    );
    state.sealedReeditBaseline = null;
    state.sealedReeditSourceId = null;
    const selectedIndex = state.episodes.findIndex(
      (value) => value.episode_uuid === episodeUuid,
    );
    if (selectedIndex >= 0) {
      state.episodes[selectedIndex] = {
        ...state.episodes[selectedIndex],
        annotation_set_count: payload.episode.annotation_set_count,
        needs_second_review: payload.episode.needs_second_review,
        review_state: payload.episode.review_state,
        latest_annotation_mode: payload.episode.latest_annotation_mode,
        whole_episode_abnormal: payload.episode.whole_episode_abnormal,
        whole_episode_abnormal_reason:
          payload.episode.whole_episode_abnormal_reason,
        annotation_summary_loaded: true,
        annotation_summary_error: null,
      };
      state.episodeRenderLimit = Math.max(
        state.episodeRenderLimit,
        selectedIndex + 1,
      );
      episodeSummaryRefreshedAt.set(episodeUuid, Date.now());
    }
    buildTimelineMap();
    renderEpisodeList();
    renderEditor();
    if (payload.episode.review_label_auto_cleared) {
      showToast(
        "已进入二次复核编辑并自动清除“需要二次复核”标签；重新封存后才会解除已发布版本的整条训练排除。",
        "info",
        8000,
      );
    } else if (payload.episode.draft_requires_seal) {
      showToast(
        "检测到尚未封存的兼容草稿；切换 Episode 或数据集前必须封存或明确放弃。",
        "info",
        7000,
      );
    }
  } catch (error) {
    showToast(error.message, "error", 6000);
  } finally {
    state.busy = false;
    updateEpisodeNavigation();
  }
}

function renderEditor() {
  const available = Boolean(state.detail && state.draft);
  $("empty-state").classList.toggle("hidden", available);
  $("editor-content").classList.toggle("hidden", !available);
  if (!available) return;
  $("episode-title").textContent = `Episode #${state.detail.episode_index} · 视频核验`;
  const fullTask = state.detail.full_task_outcome === "NOT_APPLICABLE"
    ? "FULL TASK N/A"
    : `FULL TASK ${state.detail.full_task_outcome || "—"}`;
  $("episode-semantics").textContent = [
    state.detail.resolution_kind,
    state.detail.recording_scope,
    fullTask,
    state.detail.source_batch_id,
  ].filter(Boolean).join(" · ");
  $("privileged-toggle").checked = Boolean(state.draft.privileged_tools_enabled);
  $("privileged-tools").classList.toggle(
    "hidden",
    !state.draft.privileged_tools_enabled,
  );
  renderCameraTabs();
  renderTimeline();
  renderRecoverableFailureTools();
  renderReviewControls();
  renderPrivilegedTools();
  renderSealedReeditControls();
  updateEditButtonState();
}

function episodeStageCount(value) {
  const explicit = Number(value?.stage_count);
  if (Number.isFinite(explicit)) return explicit;
  return (value?.progress_steps || []).reduce(
    (total, step) => total + (step.stages || []).length,
    0,
  );
}

function isSingleStageEpisode(value) {
  return episodeStageCount(value) === 1;
}

function mediaKeyForEpisode(episode, camera) {
  if (!episode || !camera) return null;
  return `${episode.episode_uuid}:${camera}`;
}

function cancelVideoPrefetch() {
  if (state.videoPrefetchTimer) {
    window.clearTimeout(state.videoPrefetchTimer);
    state.videoPrefetchTimer = 0;
  }
  if (state.videoPrefetch) {
    state.videoPrefetch.removeAttribute("src");
    state.videoPrefetch.load();
    state.videoPrefetch = null;
  }
}

function scheduleNextEpisodeWarmup() {
  cancelVideoPrefetch();
  const index = selectedEpisodeListIndex();
  const next = index >= 0 ? state.episodes[index + 1] : null;
  if (!next || isSingleStageEpisode(next)) return;
  const camera = next.default_camera_key || next.camera_keys?.[0];
  if (!camera) return;
  state.videoPrefetchTimer = window.setTimeout(() => {
    state.videoPrefetchTimer = 0;
    const prefetch = document.createElement("video");
    prefetch.preload = "metadata";
    prefetch.muted = true;
    prefetch.src = `/media/${encodeURIComponent(next.episode_uuid)}/${encodeURIComponent(camera)}`;
    prefetch.addEventListener("error", () => {
      // Warmup is opportunistic; the visible player reports the real error.
    }, {once: true});
    prefetch.load();
    state.videoPrefetch = prefetch;
  }, 250);
}

function showVideoLoadPrompt() {
  clearVideoLoadListeners();
  state.loadedMediaKey = null;
  state.loadingMediaKey = null;
  state.videoLoading = false;
  video.pause();
  video.removeAttribute("src");
  video.load();
  $("load-video").classList.remove("hidden");
  $("play-toggle").disabled = true;
  $("video-scrubber").disabled = true;
}

function renderCameraTabs(preserveTimestamp = null, {forceLoad = false} = {}) {
  const root = $("camera-tabs");
  root.replaceChildren();
  for (const key of state.detail.camera_keys) {
    const button = element(
      "button",
      `camera-tab${key === state.selectedCamera ? " active" : ""}`,
      key,
    );
    button.type = "button";
    button.addEventListener("click", () => {
      const timestamp = state.loadedMediaKey
        ? currentTimelineTime()
        : state.detail.start_timestamp;
      state.selectedCamera = key;
      renderCameraTabs(timestamp, {forceLoad: true});
    });
    root.appendChild(button);
  }
  const currentKey = mediaKeyForEpisode(state.detail, state.selectedCamera);
  if (
    state.loadedMediaKey === currentKey
    || (state.videoLoading && state.loadingMediaKey === currentKey)
  ) return;
  if (isSingleStageEpisode(state.detail) && !forceLoad) {
    showVideoLoadPrompt();
  } else {
    loadSelectedVideo(preserveTimestamp ?? state.detail.start_timestamp);
  }
}

function loadSelectedVideo(preserveTimestamp) {
  if (!state.detail) return;
  cancelPendingSeek();
  clearVideoLoadListeners();
  const source = state.detail.media_urls[state.selectedCamera];
  const mediaKey = mediaKeyForEpisode(state.detail, state.selectedCamera);
  if (!source || !mediaKey) return;
  state.loadedMediaKey = mediaKey;
  state.loadingMediaKey = mediaKey;
  state.videoLoading = true;
  $("load-video").classList.add("hidden");
  $("play-toggle").disabled = true;
  $("video-scrubber").disabled = true;
  video.preload = "auto";
  video.dataset.episode = state.detail.episode_uuid;
  const cleanup = () => {
    video.removeEventListener("loadedmetadata", onLoadedMetadata);
    video.removeEventListener("error", onError);
  };
  const finishLoad = () => {
    if (activeVideoLoadCleanup === cleanup) {
      activeVideoLoadCleanup = null;
    }
    cleanup();
  };
  const onLoadedMetadata = () => {
    finishLoad();
    if (state.loadedMediaKey !== mediaKey) return;
    state.videoLoading = false;
    state.loadingMediaKey = null;
    $("play-toggle").disabled = false;
    $("video-scrubber").disabled = false;
    seekTimeline(preserveTimestamp);
    scheduleNextEpisodeWarmup();
  };
  const onError = () => {
    finishLoad();
    if (state.loadedMediaKey !== mediaKey) return;
    state.loadedMediaKey = null;
    state.loadingMediaKey = null;
    state.videoLoading = false;
    $("play-toggle").disabled = true;
    $("video-scrubber").disabled = true;
    const code = video.error?.code;
    const detail = code === 4
      ? "浏览器不支持该视频编码；服务端应自动准备 H.264 兼容缓存。"
      : "请检查视频文件和标注服务日志。";
    showToast(`视频无法加载：${detail}`, "error", 9000);
  };
  activeVideoLoadCleanup = cleanup;
  video.addEventListener("loadedmetadata", onLoadedMetadata);
  video.addEventListener("error", onError);
  video.src = source;
  video.load();
}

function setPosition(node, startAt, endAt) {
  const start = timelineRatio(startAt);
  const end = timelineRatio(endAt);
  node.style.left = `${start * 100}%`;
  node.style.width = `${Math.max(0.001, end - start) * 100}%`;
}

function renderTimeline() {
  synchronizeFailureTriggerWindows();
  const coarse = $("coarse-track");
  const stages = $("stage-track");
  const triggerTrack = $("failure-trigger-track");
  const recoverableTrack = $("recoverable-failure-track");
  coarse.replaceChildren();
  stages.replaceChildren();
  triggerTrack.replaceChildren();
  recoverableTrack.replaceChildren();
  const stepMap = stepsById();
  const end = state.detail.end_timestamp;
  const attemptBounds = effectiveAttemptBounds();

  for (const gap of state.timelineMap?.compressedGaps || []) {
    for (const track of [coarse, stages, triggerTrack, recoverableTrack]) {
      const marker = element("div", "compressed-gap");
      setPosition(marker, gap.realStart, gap.realEnd);
      marker.title =
        `无样本/Hold 间隔：真实 ${formatTime(gap.realDuration)}，标注显示为 1 秒`;
      track.appendChild(marker);
    }
  }

  for (const attempt of state.detail.attempts) {
    const bounds = attemptBounds.get(attempt.attempt_id) || attempt;
    const block = element(
      "div",
      `coarse-block${attempt.outcome === "FAILURE" ? " failure" : ""}`,
    );
    setPosition(block, bounds.start_at, bounds.end_at ?? end);
    const step = stepMap.get(attempt.progress_step_id);
    const palette = stepPalette(attempt.progress_step_id);
    block.style.setProperty("--coarse-color", palette.solid);
    block.style.setProperty("--coarse-bg", palette.background);
    block.style.setProperty("--coarse-text", palette.text);
    block.textContent =
      `${step?.display_name || attempt.progress_step_id} · ${attempt.outcome}`;
    coarse.appendChild(block);
    if (attempt.failure_reported_at !== null) {
      const marker = element("div", "fixed-marker");
      marker.style.left = `${timelineRatio(attempt.failure_reported_at) * 100}%`;
      marker.title = `原始 Failure · ${formatTime(attempt.failure_reported_at - state.detail.start_timestamp)}`;
      coarse.appendChild(marker);
      const stageMarker = marker.cloneNode();
      stages.appendChild(stageMarker);
    }
  }

  if (state.draft.privileged_tools_enabled) {
    for (const transition of state.detail.transitions || []) {
      const handle = element(
        "button",
        "boundary-handle coarse-boundary-handle",
      );
      handle.type = "button";
      const current = effectiveProgressBoundary(transition);
      handle.style.left = `${timelineRatio(current) * 100}%`;
      handle.title =
        `${transition.from_progress_step_id} → ${transition.to_progress_step_id}`
        + " · 拖动修正主阶段边界";
      handle.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        clearDeletionHistory();
        handle.setPointerCapture(event.pointerId);
        state.dragging = {
          kind: "coarse-boundary",
          transition,
          handle,
          pointerId: event.pointerId,
          rect: coarse.getBoundingClientRect(),
          lastTimestamp: null,
        };
      });
      handle.addEventListener("pointermove", (event) => {
        if (
          !state.dragging
          || state.dragging.kind !== "coarse-boundary"
          || state.dragging.pointerId !== event.pointerId
        ) return;
        const ratio = clamp(
          (event.clientX - state.dragging.rect.left) / state.dragging.rect.width,
          0,
          1,
        );
        const corrected = applyCoarseBoundaryCorrection(
          transition,
          timelineAtRatio(ratio),
        );
        if (corrected === null) return;
        handle.style.left = `${timelineRatio(corrected) * 100}%`;
        if (state.dragging.lastTimestamp === corrected) return;
        state.dragging.lastTimestamp = corrected;
        scheduleSeek(corrected);
      });
      const finishCoarseBoundaryDrag = () => {
        if (!state.dragging || state.dragging.kind !== "coarse-boundary") return;
        flushPendingSeek();
        state.dragging = null;
        renderTimeline();
        renderPrivilegedTools();
      };
      handle.addEventListener("pointerup", finishCoarseBoundaryDrag);
      handle.addEventListener("pointercancel", finishCoarseBoundaryDrag);
      handle.addEventListener("lostpointercapture", finishCoarseBoundaryDrag);
      coarse.appendChild(handle);
    }
  }

  const values = sortedOccurrences();
  for (const value of values) {
    const stage = stageById(value.stage_id);
    const selected = value.annotation_id === state.selectedOccurrenceId;
    const overlapsAbnormal = occurrenceOverlapsAbnormal(value);
    const block = element(
      "div",
      [
        "stage-block",
        value.outcome === "FAILURE" ? "failure" : "",
        value.outcome === "TRUNCATED_BY_ABNORMAL" ? "truncated" : "",
        selected ? "selected" : "",
        overlapsAbnormal ? "overlaps-abnormal" : "",
      ].filter(Boolean).join(" "),
    );
    const palette = stagePalette(value.stage_id);
    block.style.setProperty("--stage-color", palette.solid);
    block.style.setProperty("--stage-bg", palette.background);
    block.style.setProperty("--stage-text", palette.text);
    setPosition(block, value.start_at, value.end_at);
    block.appendChild(element("strong", "", stage?.display_name || value.stage_id));
    block.appendChild(
      element(
        "small",
        "",
        `${occurrenceOutcomeLabel(value.outcome)} · ${formatTime(value.end_at - value.start_at)}`,
      ),
    );
    block.title = overlapsAbnormal
      ? "该阶段与异常区间重叠；部分覆盖会在封存时沿异常边界拆分，完全覆盖则需选中后用右上角红色按钮删除。"
      : `${stage?.display_name || value.stage_id} · 点击选中`;
    block.addEventListener("click", () => {
      state.selectedOccurrenceId = value.annotation_id;
      renderTimeline();
      seekTimeline(value.start_at);
    });
    stages.appendChild(block);
  }

  const coverageGaps = timelineCoverageGaps();
  for (const gap of coverageGaps) {
    if (gap.kind === "abnormal") continue;
    const block = element("div", `timeline-gap ${gap.kind}`);
    setPosition(block, gap.start, gap.end);
    block.title = gap.kind === "attempt-pause"
      ? `暂停 / Attempt 间隔 · ${formatTime(gap.end - gap.start)}`
      : `阶段标注未解释的时间缺口 · ${formatTime(gap.end - gap.start)}`;
    stages.appendChild(block);
  }

  for (const abnormal of state.draft.abnormal_intervals || []) {
    const block = element("div", "abnormal-block");
    setPosition(block, abnormal.start_at, abnormal.end_at);
    block.title = `${abnormal.reason_code}${abnormal.note ? ` · ${abnormal.note}` : ""}`;
    stages.appendChild(block);
  }

  const referenceIssues = occurrenceReferenceIssues();
  const unexplainedGaps = coverageGaps.filter(
    (value) => value.kind === "unexplained",
  );
  const warning = $("timeline-warning");
  const warningParts = [];
  if (unexplainedGaps.length) {
    warningParts.push(
      `${unexplainedGaps.length} 处阶段时间缺口不属于异常区间或 Attempt 暂停`,
    );
  }
  if (referenceIssues.length) {
    warningParts.push(`${referenceIssues.length} 条阶段引用无效`);
    warning.title = referenceIssues.join("；");
  } else {
    warning.title = "";
  }
  warning.textContent = warningParts.length
    ? `时间轴需要复核：${warningParts.join("；")}。`
    : "";
  warning.classList.toggle("hidden", !warningParts.length);

  for (const trigger of failureTriggers()) {
    const window = element(
      "div",
      "failure-trigger-window",
      `${trigger.stage_id} · failure_trigger`,
    );
    setPosition(window, trigger.lower_bound_at, trigger.upper_bound_at);
    window.title =
      `允许范围 ${formatTime(trigger.lower_bound_at - state.detail.start_timestamp)}`
      + ` - ${formatTime(trigger.upper_bound_at - state.detail.start_timestamp)}`;
    triggerTrack.appendChild(window);

    const failureMarker = element("div", "failure-trigger-marker");
    failureMarker.style.left = `${timelineRatio(trigger.upper_bound_at) * 100}%`;
    failureMarker.title =
      `原始 Failure · ${formatTime(trigger.upper_bound_at - state.detail.start_timestamp)}`;
    triggerTrack.appendChild(failureMarker);

    const handle = element("button", "failure-trigger-handle");
    handle.type = "button";
    handle.style.left = `${timelineRatio(trigger.trigger_at) * 100}%`;
    handle.title =
      `failure_trigger · ${formatTime(trigger.trigger_at - state.detail.start_timestamp)}；拖动确认失败发生时刻`;
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      clearDeletionHistory();
      handle.setPointerCapture(event.pointerId);
      state.dragging = {
        kind: "failure-trigger",
        trigger,
        handle,
        pointerId: event.pointerId,
        rect: triggerTrack.getBoundingClientRect(),
        lastTimestamp: null,
      };
    });
    handle.addEventListener("pointermove", (event) => {
      if (
        !state.dragging
        || state.dragging.kind !== "failure-trigger"
        || state.dragging.pointerId !== event.pointerId
      ) return;
      const rect = state.dragging.rect;
      const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
      const proposed = timelineAtRatio(ratio);
      const sampleIndex = clamp(
        nearestSampleIndex(proposed),
        trigger.lower_bound_sample_index,
        trigger.upper_bound_sample_index,
      );
      const timestamp = sampleTimestamps()[sampleIndex];
      trigger.trigger_sample_index = sampleIndex;
      trigger.trigger_at = timestamp;
      trigger.source = "POST_CAPTURE_REFINED";
      trigger.confidence = 1.0;
      handle.style.left = `${timelineRatio(timestamp) * 100}%`;
      if (state.dragging.lastTimestamp === timestamp) return;
      state.dragging.lastTimestamp = timestamp;
      scheduleSeek(timestamp);
    });
    const finishTriggerDrag = () => {
      if (!state.dragging || state.dragging.kind !== "failure-trigger") return;
      flushPendingSeek();
      state.dragging = null;
      renderTimeline();
    };
    handle.addEventListener("pointerup", finishTriggerDrag);
    handle.addEventListener("pointercancel", finishTriggerDrag);
    handle.addEventListener("lostpointercapture", finishTriggerDrag);
    triggerTrack.appendChild(handle);
  }

  for (const event of recoverableFailures()) {
    const stage = stageById(event.stage_id);
    const block = element(
      "div",
      "recoverable-failure-block",
      stage?.display_name || event.stage_id,
    );
    setPosition(block, event.onset_at, event.recovered_at);
    block.title =
      `可恢复失误 · ${stage?.display_name || event.stage_id}`
      + ` · ${formatTime(event.onset_at - state.detail.start_timestamp)}`
      + ` – ${formatTime(event.recovered_at - state.detail.start_timestamp)}`
      + `${event.note ? ` · ${event.note}` : ""}`;
    recoverableTrack.appendChild(block);
  }

  for (let index = 1; index < values.length; index += 1) {
    const left = values[index - 1];
    const right = values[index];
    if (
      left.coarse_attempt_id !== right.coarse_attempt_id
      || Math.abs(left.end_at - right.start_at) > 1e-7
    ) continue;
    const handle = element("button", "boundary-handle");
    handle.type = "button";
    handle.style.left = `${timelineRatio(right.start_at) * 100}%`;
    handle.title = "拖动阶段边界";
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      clearDeletionHistory();
      handle.setPointerCapture(event.pointerId);
      state.dragging = {
        left,
        right,
        handle,
        pointerId: event.pointerId,
        rect: stages.getBoundingClientRect(),
        lastTimestamp: null,
      };
    });
    handle.addEventListener("pointermove", (event) => {
      if (!state.dragging || state.dragging.pointerId !== event.pointerId) return;
      const rect = state.dragging.rect;
      const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
      const minimum = Math.max(0.02, episodeDuration() / Math.max(10000, state.detail.frame_count * 20));
      const timestamp = clamp(
        timelineAtRatio(ratio),
        state.dragging.left.start_at + minimum,
        state.dragging.right.end_at - minimum,
      );
      state.dragging.left.end_at = timestamp;
      state.dragging.right.start_at = timestamp;
      state.dragging.handle.style.left = `${timelineRatio(timestamp) * 100}%`;
      if (state.dragging.lastTimestamp === timestamp) return;
      state.dragging.lastTimestamp = timestamp;
      scheduleSeek(timestamp);
    });
    const finishBoundaryDrag = () => {
      if (!state.dragging) return;
      flushPendingSeek();
      state.dragging = null;
      renderTimeline();
    };
    handle.addEventListener("pointerup", finishBoundaryDrag);
    handle.addEventListener("pointercancel", finishBoundaryDrag);
    handle.addEventListener("lostpointercapture", finishBoundaryDrag);
    stages.appendChild(handle);
  }

  const coarsePlayhead = element("div", "playhead");
  const stagePlayhead = element("div", "playhead");
  const triggerPlayhead = element("div", "playhead");
  coarse.appendChild(coarsePlayhead);
  stages.appendChild(stagePlayhead);
  triggerTrack.appendChild(triggerPlayhead);
  renderFailedAttemptEditors();
  renderStageDescriptions();
  updatePlayhead();
  updateEditButtonState();
}

function renderStageDescriptions() {
  const root = $("stage-descriptions");
  root.replaceChildren();
  const used = new Set(sortedOccurrences().map((value) => value.stage_id));
  for (const step of state.detail.progress_steps) {
    for (const stage of step.stages) {
      if (!used.has(stage.stage_id)) continue;
      const selected = sortedOccurrences().some(
        (value) =>
          value.annotation_id === state.selectedOccurrenceId
          && value.stage_id === stage.stage_id,
      );
      const card = element(
        "div",
        `stage-description${selected ? " selected" : ""}`,
      );
      const palette = stagePalette(stage.stage_id);
      card.style.setProperty("--stage-color", palette.solid);
      card.style.setProperty("--stage-bg", palette.background);
      card.style.setProperty("--stage-text", palette.text);
      card.appendChild(element("strong", "", `${step.display_name} / ${stage.display_name}`));
      card.appendChild(element("span", "", stage.description));
      root.appendChild(card);
    }
  }
}

function occurrenceDeletionPosition(value) {
  const values = sortedOccurrences();
  if (!value || !values.length) return null;
  const first = values[0];
  const last = values[values.length - 1];
  if (value === first || value === last) {
    const sameStep = values.filter(
      (candidate) => candidate.progress_step_id === value.progress_step_id,
    );
    if (
      sameStep.length > 1
      && value.outcome === "SUCCESS"
      && !value.source_failure_fact_id
    ) {
      return value === first ? {kind: "episode_start"} : {kind: "episode_end"};
    }
  }

  for (const abnormal of state.draft.abnormal_intervals || []) {
    if (
      intervalsOverlap(
        value.start_at,
        value.end_at,
        abnormal.start_at,
        abnormal.end_at,
      )
    ) {
      return {kind: "overlap", abnormal};
    }
    const before = [...values]
      .filter((candidate) => candidate.end_at <= abnormal.start_at)
      .at(-1);
    const after = values.find(
      (candidate) => candidate.start_at >= abnormal.end_at,
    );
    if (value === before) return {kind: "before_abnormal", abnormal};
    if (value === after) return {kind: "after_abnormal", abnormal};
  }
  return null;
}

function occurrenceCanReceiveMergedTime(candidate, selected) {
  return Boolean(
    candidate
    && candidate.coarse_attempt_id === selected.coarse_attempt_id
    && candidate.progress_step_id === selected.progress_step_id
  );
}

function occurrenceDeletionMergePlan(value) {
  if (
    !value
    || value.outcome === "FAILURE"
    || value.source_failure_fact_id
  ) {
    return null;
  }
  const position = occurrenceDeletionPosition(value);
  if (!position) return null;

  const values = sortedOccurrences();
  const selectedIndex = values.indexOf(value);
  const previous = occurrenceCanReceiveMergedTime(
    values[selectedIndex - 1],
    value,
  ) ? values[selectedIndex - 1] : null;
  const next = occurrenceCanReceiveMergedTime(
    values[selectedIndex + 1],
    value,
  ) ? values[selectedIndex + 1] : null;

  let target = null;
  let direction = null;
  if (position.kind === "episode_start") {
    target = next;
    direction = "next";
  } else if (position.kind === "episode_end") {
    target = previous;
    direction = "previous";
  } else if (position.kind === "before_abnormal") {
    target = previous || next;
    direction = previous ? "previous" : "next";
  } else if (position.kind === "after_abnormal") {
    target = next || previous;
    direction = next ? "next" : "previous";
  } else {
    const abnormal = position.abnormal;
    const visibleBefore = Math.max(
      0,
      Math.min(value.end_at, abnormal.start_at) - value.start_at,
    );
    const visibleAfter = Math.max(
      0,
      value.end_at - Math.max(value.start_at, abnormal.end_at),
    );
    if (visibleAfter > visibleBefore) {
      target = next || previous;
      direction = next ? "next" : "previous";
    } else {
      target = previous || next;
      direction = previous ? "previous" : "next";
    }
  }

  if (!target) return null;
  return {selected: value, target, direction, position};
}

function deleteSelectedOccurrence() {
  const selected = sortedOccurrences().find(
    (value) => value.annotation_id === state.selectedOccurrenceId,
  );
  if (!selected) {
    showToast("请先点选一个阶段时间块。", "warning");
    return;
  }
  const mergePlan = occurrenceDeletionMergePlan(selected);
  if (!mergePlan) {
    showToast(
      "该阶段没有同一 Attempt 内可接收其时间范围的相邻有效阶段；承载原始 Failure 的阶段不能删除。",
      "warning",
      5200,
    );
    return;
  }
  const targetStartBefore = mergePlan.target.start_at;
  const targetEndBefore = mergePlan.target.end_at;
  if (mergePlan.direction === "previous") {
    mergePlan.target.end_at = Math.max(
      mergePlan.target.end_at,
      mergePlan.selected.end_at,
    );
  } else {
    mergePlan.target.start_at = Math.min(
      mergePlan.target.start_at,
      mergePlan.selected.start_at,
    );
  }
  state.deletionHistory.push({
    episodeUuid: state.detail.episode_uuid,
    deletedOccurrence: structuredClone(mergePlan.selected),
    targetAnnotationId: mergePlan.target.annotation_id,
    targetStartBefore,
    targetEndBefore,
    targetStartAfter: mergePlan.target.start_at,
    targetEndAfter: mergePlan.target.end_at,
  });
  syncDeletionHistoryToDraft();
  state.draft.occurrences = state.draft.occurrences.filter(
    (value) => value.annotation_id !== selected.annotation_id,
  );
  state.selectedOccurrenceId = null;
  normalizeOrderedOccurrences();
  renderTimeline();
  updateEditButtonState();
  const targetStage = stageById(mergePlan.target.stage_id);
  showToast(
    `已删除所选阶段，其完整时间范围已并入${mergePlan.direction === "previous" ? "前一" : "后一"}有效阶段“${targetStage?.display_name || mergePlan.target.stage_id}”；原始 Failure、视频和样本均未改变。`,
    "info",
    5200,
  );
}

function clearDeletionHistory() {
  state.deletionHistory = [];
  syncDeletionHistoryToDraft();
  const button = $("undo-delete");
  if (button) button.disabled = true;
}

function syncDeletionHistoryToDraft() {
  if (state.draft) {
    state.draft.deletion_history = structuredClone(state.deletionHistory);
  }
}

function undoLastDeletion() {
  const entry = state.deletionHistory.at(-1);
  if (!entry || entry.episodeUuid !== state.detail?.episode_uuid) {
    showToast("当前没有可回退的阶段删除。", "warning");
    clearDeletionHistory();
    return;
  }
  const target = state.draft.occurrences.find(
    (value) => value.annotation_id === entry.targetAnnotationId,
  );
  if (
    !target
    || Math.abs(target.start_at - entry.targetStartAfter) > 1e-7
    || Math.abs(target.end_at - entry.targetEndAfter) > 1e-7
  ) {
    showToast(
      "删除后的目标阶段已被其他编辑改变，不能猜测性回退；可重新打开 Episode 恢复最近一次封存版本。",
      "warning",
      6500,
    );
    clearDeletionHistory();
    return;
  }
  target.start_at = entry.targetStartBefore;
  target.end_at = entry.targetEndBefore;
  state.draft.occurrences.push(structuredClone(entry.deletedOccurrence));
  state.deletionHistory.pop();
  syncDeletionHistoryToDraft();
  state.selectedOccurrenceId = entry.deletedOccurrence.annotation_id;
  normalizeOrderedOccurrences();
  renderTimeline();
  showToast("已撤销上一次阶段删除，并恢复合并前的完整时间边界。", "info");
}

function toggleTailOutcome() {
  const tail = editableTailOutcome();
  if (!tail) {
    showToast(
      "只能修正最后一个 Attempt 中、到达该 Attempt 结束边界且没有原始 Failure 的末段。",
      "warning",
      6500,
    );
    return;
  }
  clearDeletionHistory();
  tail.occurrence.outcome =
    tail.occurrence.outcome === "FAILURE" ? "SUCCESS" : "FAILURE";
  renderTimeline();
  updateEditButtonState();
  showToast(
    `末段已在当前编辑中改为${tail.occurrence.outcome === "FAILURE" ? "失败" : "成功"}；点击“封存标注”后写入新版本。`,
    "info",
    5000,
  );
}

function updateEditButtonState() {
  const values = sortedOccurrences();
  const tail = editableTailOutcome();
  const selected = values.find(
    (value) => value.annotation_id === state.selectedOccurrenceId,
  );
  const mergePlan = occurrenceDeletionMergePlan(selected);
  const deleteButton = $("delete-selected");
  const outcomeButton = $("toggle-tail-outcome");
  $("undo-delete").disabled = !state.deletionHistory.length;
  outcomeButton.disabled = !tail;
  outcomeButton.textContent =
    tail?.occurrence.outcome === "FAILURE" ? "末段改为成功" : "末段改为失败";
  outcomeButton.title = tail
    ? "修正最后一个 Attempt 的末段结果；原始 Failure 不可改写"
    : "当前末段不是可修正的无原始 Failure 尾段";
  deleteButton.disabled = !mergePlan;
  deleteButton.title = !selected
    ? "请先点选阶段时间块"
    : mergePlan
      ? `删除所选阶段，并将时间范围并入${mergePlan.direction === "previous" ? "前一" : "后一"}有效阶段`
      : "该阶段没有同一 Attempt 内可接收其时间范围的相邻有效阶段";
}

function renderSealedReeditControls() {
  const root = $("sealed-reedit-controls");
  const values = state.detail?.annotation_sets || [];
  root.classList.toggle("hidden", !values.length);
  const select = $("sealed-annotation-set");
  const previous = select.value;
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value.annotation_set_id;
    option.textContent =
      `${new Date(value.created_at * 1000).toLocaleString()} · ${value.annotator_id}`
      + ` · ${value.annotation_mode === "SECOND_REVIEW" ? "二次复核" : "普通标注"}`
      + ` · ${value.annotation_set_id}`;
    select.appendChild(option);
  }
  if (values.some((value) => value.annotation_set_id === previous)) {
    select.value = previous;
  }
  $("cancel-reopen-sealed").classList.toggle(
    "hidden",
    !state.sealedReeditBaseline,
  );
  $("sealed-reedit-status").textContent = state.sealedReeditSourceId
    ? `正在基于 ${state.sealedReeditSourceId} 创建新版本；旧版本保持只读。`
    : "";
}

async function reopenSealedAnnotation() {
  if (!state.detail || state.busy) return;
  const annotationSetId = $("sealed-annotation-set").value;
  if (!annotationSetId) return;
  if (!window.confirm(
    `将所选封存版本 ${annotationSetId} 加载为新的可编辑版本。旧封存版本不会被覆盖；只有再次校验并封存才会发布新版本，继续吗？`,
  )) {
    return;
  }
  state.busy = true;
  try {
    const timestamp = currentTimelineTime();
    const payload = await requestJson(
      `/api/reopen/${encodeURIComponent(state.detail.episode_uuid)}`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({annotation_set_id: annotationSetId}),
      },
    );
    state.sealedReeditBaseline = structuredClone(state.draft);
    state.sealedReeditSourceId =
      payload.source_annotation_set.annotation_set_id;
    state.draft = payload.draft;
    state.deletionHistory = structuredClone(
      state.draft.deletion_history || [],
    );
    state.selectedOccurrenceId = null;
    state.abnormalStart = null;
    state.recoverableFailureStart = null;
    state.privilegedBaseline = null;
    renderEditor();
    seekTimeline(timestamp);
    showToast(
      payload.review_label_auto_cleared
        ? "封存版本已加载为二次复核编辑，并自动清除待复核标签；再次封存后才会解除整条训练排除。"
        : "封存版本已加载为新的页面编辑；只有再次校验并封存才会写入，新封存会生成独立版本。",
      "info",
      7200,
    );
  } catch (error) {
    showToast(`重新编辑失败：${error.message}`, "error", 8000);
  } finally {
    state.busy = false;
  }
}

function cancelReopenSealedAnnotation() {
  if (!state.sealedReeditBaseline) return;
  const timestamp = currentTimelineTime();
  state.draft = structuredClone(state.sealedReeditBaseline);
  state.deletionHistory = structuredClone(
    state.draft.deletion_history || [],
  );
  state.sealedReeditBaseline = null;
  state.sealedReeditSourceId = null;
  state.selectedOccurrenceId = null;
  state.abnormalStart = null;
  state.recoverableFailureStart = null;
  state.privilegedBaseline = null;
  renderEditor();
  seekTimeline(timestamp);
  showToast("已取消本次重新编辑，并恢复进入前的页面状态。", "info");
}

async function reconcileAbnormalDraft() {
  const payload = await requestJson(
    `/api/reconcile-abnormal-intervals/${encodeURIComponent(state.detail.episode_uuid)}`,
    {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({draft: state.draft}),
    },
  );
  state.draft = payload.draft;
  state.deletionHistory = structuredClone(
    state.draft.deletion_history || [],
  );
  state.selectedOccurrenceId = null;
}

function renderRecoverableFailureTools() {
  const root = $("recoverable-failure-list");
  root.replaceChildren();
  for (const event of recoverableFailures()) {
    const stage = stageById(event.stage_id);
    const row = element(
      "div",
      "compact-item",
      `${stage?.display_name || event.stage_id} · `
      + `${formatTime(event.onset_at - state.detail.start_timestamp)} – `
      + `${formatTime(event.recovered_at - state.detail.start_timestamp)}`
      + `${event.note ? ` · ${event.note}` : ""}`,
    );
    const remove = element("button", "secondary compact", "移除");
    remove.type = "button";
    remove.addEventListener("click", () => {
      if (state.busy) return;
      state.draft.recoverable_failures = (
        state.draft.recoverable_failures || []
      ).filter((value) => value.annotation_id !== event.annotation_id);
      renderTimeline();
      renderRecoverableFailureTools();
      updateEditButtonState();
    });
    row.appendChild(remove);
    root.appendChild(row);
  }
  $("recoverable-failure-pending").textContent = (
    state.recoverableFailureStart === null
      ? "尚未取点"
      : `失误发生 ${formatTime(
        state.recoverableFailureStart - state.detail.start_timestamp,
      )}`
  );
}

function renderPrivilegedTools() {
  const enabled = Boolean(state.draft.privileged_tools_enabled);
  $("privileged-tools").classList.toggle("hidden", !enabled);
  const abnormalRoot = $("abnormal-list");
  abnormalRoot.replaceChildren();
  for (const abnormal of state.draft.abnormal_intervals || []) {
    const row = element(
      "div",
      "compact-item",
      `${formatTime(abnormal.start_at - state.detail.start_timestamp)} – ${formatTime(abnormal.end_at - state.detail.start_timestamp)} · ${abnormal.reason_code}`,
    );
    const remove = element("button", "secondary compact", "移除");
    remove.type = "button";
    remove.addEventListener("click", async () => {
      if (state.busy) return;
      const baseline = structuredClone(state.draft);
      state.busy = true;
      remove.disabled = true;
      try {
        state.draft.abnormal_intervals = state.draft.abnormal_intervals.filter(
          (value) => value.annotation_id !== abnormal.annotation_id,
        );
        await reconcileAbnormalDraft();
        renderTimeline();
        renderPrivilegedTools();
        updateEditButtonState();
        showToast("异常区间已移除，相关阶段片段已立即恢复。", "info");
      } catch (error) {
        state.draft = baseline;
        showToast(`移除异常区间失败：${error.message}`, "error", 8000);
        renderTimeline();
        renderPrivilegedTools();
      } finally {
        state.busy = false;
      }
    });
    row.appendChild(remove);
    abnormalRoot.appendChild(row);
  }
  $("abnormal-pending").textContent = state.abnormalStart === null
    ? "尚未取点"
    : `起点 ${formatTime(state.abnormalStart - state.detail.start_timestamp)}`;

  const overrideRoot = $("coarse-overrides");
  overrideRoot.replaceChildren();
  for (const transition of state.detail.transitions || []) {
    const existing = (state.draft.progress_boundary_overrides || []).find(
      (value) => value.source_advance_fact_id === transition.advance_fact_id,
    );
    const row = element("div", "compact-item");
    row.appendChild(
      element(
        "span",
        "",
        `${transition.from_progress_step_id} → ${transition.to_progress_step_id}`,
      ),
    );
    const slider = element("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1000";
    slider.value = String(Math.round(timelineRatio(
      existing?.corrected_effective_at ?? transition.effective_at,
    ) * 1000));
    const valueLabel = element(
      "span",
      "",
      formatTime((existing?.corrected_effective_at ?? transition.effective_at) - state.detail.start_timestamp),
    );
    slider.addEventListener("input", () => {
      const corrected = applyCoarseBoundaryCorrection(
        transition,
        timelineAtRatio(Number(slider.value) / 1000),
      );
      if (corrected === null) return;
      slider.value = String(Math.round(timelineRatio(corrected) * 1000));
      valueLabel.textContent = formatTime(corrected - state.detail.start_timestamp);
      seekTimeline(corrected);
      renderTimeline();
    });
    row.appendChild(slider);
    row.appendChild(valueLabel);
    overrideRoot.appendChild(row);
  }

  const abnormal = state.draft.episode_abnormal;
  $("episode-abnormal-reason").value = abnormal?.reason_code || "BAD_CAPTURE";
  $("episode-abnormal-note").value = abnormal?.note || "";
  $("toggle-episode-abnormal").textContent = abnormal
    ? "取消整条异常"
    : "标记整条异常";
}

function renderReviewControls() {
  const request = state.draft?.episode_review_request || null;
  const card = $("toggle-review-request").closest(".review-request-card");
  card.classList.toggle("pending", Boolean(request));
  $("review-request-note").value = request?.note || "";
  $("toggle-review-request").textContent = request
    ? "取消二次复核标签"
    : "标记需要二次复核";
  $("review-request-status").textContent = request
    ? "当前编辑封存后将整条排除；二次复核模式重新封存后恢复正常。"
    : state.detail?.annotation_mode === "SECOND_REVIEW"
      ? "当前是二次复核编辑；封存后生成 SECOND_REVIEW 版本。"
      : "标记后整条 Episode 默认不进入训练，直到二次复核版本封存。";
}

async function publishCurrentAnnotations() {
  if (!state.draft || !state.detail) {
    throw new Error("当前没有可封存的 Episode 标注。");
  }
  const payload = await requestJsonWithProgress(
    `/api/commit/${encodeURIComponent(state.detail.episode_uuid)}`,
    {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state.draft),
    },
    {
      kind: "COMMIT_ANNOTATION",
      message: "正在校验并封存标注",
      completeMessage: "标注已封存",
    },
  );
  const report = payload.report;
  state.draft = report.draft;
  setSavedDraftSnapshot(state.draft);
  state.sealedReeditBaseline = null;
  state.sealedReeditSourceId = null;
  state.detail.annotation_sets = [
    {
      annotation_set_id: report.annotation_set_id,
      annotator_id: report.annotator_id,
      annotation_mode: report.annotation_mode,
      created_at: report.created_at,
      manifest_hash: report.manifest_hash,
      annotation_count: state.draft.occurrences.length,
      needs_second_review: Boolean(state.draft.episode_review_request),
    },
    ...(state.detail.annotation_sets || []),
  ];
  state.detail.annotation_set_count = state.detail.annotation_sets.length;
  state.detail.needs_second_review = Boolean(
    state.draft.episode_review_request,
  );
  state.detail.review_state = state.detail.needs_second_review
    ? "PENDING"
    : report.annotation_mode === "SECOND_REVIEW"
      ? "COMPLETED"
      : "NONE";
  state.detail.latest_annotation_mode = report.annotation_mode;
  state.detail.whole_episode_abnormal = Boolean(
    state.draft.episode_abnormal,
  );
  state.detail.whole_episode_abnormal_reason =
    state.draft.episode_abnormal?.reason_code || null;
  const episodeIndex = state.episodes.findIndex(
    (value) => value.episode_uuid === state.detail.episode_uuid,
  );
  if (episodeIndex >= 0) {
    state.episodes[episodeIndex] = {
      ...state.episodes[episodeIndex],
      annotation_set_count: state.detail.annotation_set_count,
      needs_second_review: state.detail.needs_second_review,
      review_state: state.detail.review_state,
      latest_annotation_mode: state.detail.latest_annotation_mode,
      whole_episode_abnormal: state.detail.whole_episode_abnormal,
      whole_episode_abnormal_reason:
        state.detail.whole_episode_abnormal_reason,
      annotation_summary_loaded: true,
      annotation_summary_error: null,
    };
    episodeSummaryRefreshedAt.set(state.detail.episode_uuid, Date.now());
  }
  state.detail.draft_persistence_state = "SEALED_SNAPSHOT";
  state.detail.draft_requires_seal = false;
  renderSealedReeditControls();
  $("coverage-preview").textContent =
    `已封存 ${report.annotation_set_id} · coverage=${report.coverage_status} · 排除区间 ${report.exclusion_count}`;
  await loadEpisodes();
  return report;
}

async function commitAnnotations() {
  if (!state.draft || state.busy) return;
  state.busy = true;
  try {
    const report = await publishCurrentAnnotations();
    showToast(
      report.draft_cache_warning
        ? `不可变标注已发布，但工作缓存更新失败：${report.draft_cache_warning}`
        : "标注校验通过，已发布不可变 annotation set。",
      report.draft_cache_warning ? "error" : "info",
      report.draft_cache_warning ? 9000 : 5000,
    );
  } catch (error) {
    showToast(`封存失败：${error.message}`, "error", 9000);
  } finally {
    state.busy = false;
  }
}

$("refresh-episodes").addEventListener(
  "click",
  () => loadEpisodes({force: true}),
);
$("draft-warning-save").addEventListener(
  "click",
  () => settleDraftTransition("seal"),
);
$("draft-warning-discard").addEventListener(
  "click",
  () => settleDraftTransition("discard"),
);
$("draft-warning-cancel").addEventListener(
  "click",
  () => settleDraftTransition("cancel"),
);
$("collapse-episode-panel").addEventListener(
  "click",
  () => setEpisodePanelCollapsed(true),
);
$("expand-episode-panel").addEventListener(
  "click",
  () => setEpisodePanelCollapsed(false),
);
$("previous-episode").addEventListener("click", () => navigateEpisode(-1));
$("next-episode").addEventListener("click", () => navigateEpisode(1));
$("previous-episode-collapsed").addEventListener(
  "click",
  () => navigateEpisode(-1),
);
$("next-episode-collapsed").addEventListener(
  "click",
  () => navigateEpisode(1),
);
$("open-workspace").addEventListener("click", openWorkspace);
$("choose-dataset-root").addEventListener(
  "click",
  () => openDatasetBrowser("launcher"),
);
$("change-dataset-root").addEventListener(
  "click",
  () => openDatasetBrowser("workspace"),
);
$("dataset-browser-parent").addEventListener("click", () => {
  const parent = $("dataset-browser-parent").dataset.path;
  if (parent) void browseDatasetPath(parent);
});
$("dataset-browser-go").addEventListener(
  "click",
  () => void browseDatasetPath(),
);
$("dataset-browser-path").addEventListener("keydown", (event) => {
  if (event.key === "Enter") void browseDatasetPath();
});
$("dataset-browser-select").addEventListener(
  "click",
  selectDatasetBrowserPath,
);
$("dataset-browser-cancel").addEventListener("click", closeDatasetBrowser);
$("dataset-browser-cancel-top").addEventListener("click", closeDatasetBrowser);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (pendingDraftTransition) {
    settleDraftTransition("cancel");
    return;
  }
  closeDatasetBrowser();
});
$("launcher-annotator-id").addEventListener("keydown", (event) => {
  if (event.key === "Enter") openWorkspace();
});
$("launcher-dataset-root").addEventListener("keydown", (event) => {
  if (event.key === "Enter") openWorkspace();
});
$("load-video").addEventListener("click", () => {
  if (state.detail) loadSelectedVideo(state.detail.start_timestamp);
});
$("play-toggle").addEventListener("click", () => {
  if (video.paused) {
    video.play().catch((error) => showToast(error.message, "warning"));
  } else {
    video.pause();
  }
});
video.addEventListener("play", () => { $("play-toggle").textContent = "暂停"; });
video.addEventListener("pause", () => { $("play-toggle").textContent = "播放"; });
video.addEventListener("timeupdate", () => {
  const duration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : episodeDuration();
  $("video-scrubber").value = String(Math.round(clamp(video.currentTime / duration, 0, 1) * 1000));
  updatePlayhead();
});
$("playback-rate").addEventListener("change", (event) => {
  video.playbackRate = Number(event.target.value);
});
$("video-scrubber").addEventListener("input", (event) => {
  scheduleSeek(timestampAtVideoRatio(Number(event.target.value) / 1000));
});
$("video-scrubber").addEventListener("change", flushPendingSeek);
$("delete-selected").addEventListener("click", deleteSelectedOccurrence);
$("undo-delete").addEventListener("click", undoLastDeletion);
$("toggle-tail-outcome").addEventListener("click", toggleTailOutcome);
$("commit-annotations").addEventListener("click", commitAnnotations);
$("reopen-sealed").addEventListener("click", reopenSealedAnnotation);
$("cancel-reopen-sealed").addEventListener(
  "click",
  cancelReopenSealedAnnotation,
);

$("mark-recoverable-start").addEventListener("click", () => {
  if (!state.draft || state.busy) return;
  state.recoverableFailureStart = currentTimelineTime();
  renderRecoverableFailureTools();
  showToast(
    "已记录失误发生点；请定位到恢复正常的样本后添加。",
    "info",
    4200,
  );
});
$("mark-recoverable-end").addEventListener("click", async () => {
  if (!state.draft || !state.detail || state.busy) return;
  const recoveredAt = currentTimelineTime();
  if (
    state.recoverableFailureStart === null
    || recoveredAt <= state.recoverableFailureStart
  ) {
    showToast("恢复正常点必须晚于失误发生点。", "warning");
    return;
  }
  state.busy = true;
  $("mark-recoverable-end").disabled = true;
  try {
    const payload = await requestJson(
      `/api/recoverable-failure/${encodeURIComponent(state.detail.episode_uuid)}`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          draft: state.draft,
          onset_at: state.recoverableFailureStart,
          recovered_at: recoveredAt,
          note: $("recoverable-failure-note").value.trim() || null,
        }),
      },
    );
    state.draft = payload.draft;
    state.recoverableFailureStart = null;
    $("recoverable-failure-note").value = "";
    renderTimeline();
    renderRecoverableFailureTools();
    updateEditButtonState();
    const added = payload.recoverable_failure;
    const stage = added ? stageById(added.stage_id) : null;
    showToast(
      `可恢复失误已绑定到“${stage?.display_name || added?.stage_id || "当前子阶段"}”；只会作为分析附加信息随下次封存保存。`,
      "info",
      6500,
    );
  } catch (error) {
    showToast(
      `无法添加可恢复失误：${error.message}`,
      "error",
      8500,
    );
  } finally {
    state.busy = false;
    $("mark-recoverable-end").disabled = false;
  }
});

$("toggle-review-request").addEventListener("click", () => {
  if (!state.draft) return;
  if (state.draft.episode_review_request) {
    state.draft.episode_review_request = null;
  } else {
    state.draft.episode_review_request = {
      annotation_id: `episode-review:${crypto.randomUUID()}`,
      note: $("review-request-note").value.trim() || null,
    };
  }
  renderReviewControls();
});
$("review-request-note").addEventListener("input", () => {
  if (state.draft?.episode_review_request) {
    state.draft.episode_review_request.note =
      $("review-request-note").value.trim() || null;
  }
});

$("privileged-toggle").addEventListener("change", async (event) => {
  if (state.busy) {
    event.target.checked = Boolean(state.draft?.privileged_tools_enabled);
    return;
  }
  clearDeletionHistory();
  const enabled = Boolean(event.target.checked);
  if (enabled) {
    state.privilegedBaseline = structuredClone(state.draft);
    state.draft.privileged_tools_enabled = true;
  } else {
    const hasSpecial = Boolean(
      state.draft.abnormal_intervals.length
      || state.draft.progress_boundary_overrides.length
      || state.draft.episode_abnormal,
    );
    if (
      hasSpecial
      && !window.confirm(
        "关闭特殊工具会撤销本次启用后产生的异常/边界修正，并恢复启用前阶段状态，是否继续？",
      )
    ) {
      event.target.checked = true;
      return;
    }
    if (state.privilegedBaseline) {
      const revision = state.draft.draft_revision;
      const committedId = state.draft.last_committed_annotation_set_id;
      const committedHash = state.draft.last_committed_manifest_hash;
      state.draft = structuredClone(state.privilegedBaseline);
      state.draft.draft_revision = revision;
      if (committedId) {
        state.draft.last_committed_annotation_set_id = committedId;
      }
      if (committedHash) {
        state.draft.last_committed_manifest_hash = committedHash;
      }
    } else {
      // A draft loaded with privileged tools already active has no in-memory
      // pre-enable snapshot. New abnormal edits are non-destructive, so
      // clear special fields and explicitly restore every coarse boundary in
      // the editable occurrences before leaving privileged mode.
      const baseline = structuredClone(state.draft);
      state.draft.abnormal_intervals = [];
      restoreOriginalCoarseBoundaries();
      state.draft.episode_abnormal = null;
      state.draft.privileged_tools_enabled = false;
      state.busy = true;
      try {
        await reconcileAbnormalDraft();
      } catch (error) {
        state.draft = baseline;
        event.target.checked = true;
        state.busy = false;
        showToast(
          `关闭特殊修正工具失败，页面修改已恢复：${error.message}`,
          "error",
          8500,
        );
        renderTimeline();
        renderPrivilegedTools();
        return;
      }
      state.busy = false;
    }
    state.draft.privileged_tools_enabled = false;
    state.privilegedBaseline = null;
  }
  state.abnormalStart = null;
  state.selectedOccurrenceId = null;
  renderTimeline();
  renderPrivilegedTools();
  updateEditButtonState();
});
$("mark-abnormal-start").addEventListener("click", () => {
  if (!state.draft?.privileged_tools_enabled) {
    showToast("请先启用特殊修正工具。", "warning");
    return;
  }
  state.abnormalStart = currentTimelineTime();
  renderPrivilegedTools();
});
$("mark-abnormal-end").addEventListener("click", () => {
  if (!state.draft?.privileged_tools_enabled) {
    showToast("特殊修正工具已关闭，不能添加异常区间。", "warning");
    state.abnormalStart = null;
    return;
  }
  const end = currentTimelineTime();
  if (state.abnormalStart === null || end <= state.abnormalStart) {
    showToast("异常区间终点必须晚于起点。", "warning");
    return;
  }
  if ((state.draft.abnormal_intervals || []).some((value) =>
    intervalsOverlap(
      state.abnormalStart,
      end,
      value.start_at,
      value.end_at,
    )
  )) {
    showToast("异常区间之间不能重叠。", "warning");
    return;
  }
  state.draft.abnormal_intervals.push({
    annotation_id: `abnormal:${crypto.randomUUID()}`,
    start_at: state.abnormalStart,
    end_at: end,
    reason_code: "BAD_CAPTURE",
    note: null,
  });
  state.abnormalStart = null;
  renderTimeline();
  renderPrivilegedTools();
  showToast(
    "异常区间已添加；部分覆盖的阶段会在封存时沿异常边界拆分，不会丢失两侧有效时间。完全覆盖的成功阶段仍需人工删除。",
    "info",
    6800,
  );
});
$("toggle-episode-abnormal").addEventListener("click", () => {
  if (!state.draft?.privileged_tools_enabled) {
    showToast("请先启用特殊修正工具。", "warning");
    return;
  }
  if (state.draft.episode_abnormal) {
    state.draft.episode_abnormal = null;
  } else {
    state.draft.episode_abnormal = {
      annotation_id: `episode-abnormal:${crypto.randomUUID()}`,
      reason_code: $("episode-abnormal-reason").value,
      note: $("episode-abnormal-note").value.trim() || null,
    };
  }
  renderPrivilegedTools();
});
$("episode-abnormal-reason").addEventListener("change", () => {
  if (state.draft.episode_abnormal) {
    state.draft.episode_abnormal.reason_code = $("episode-abnormal-reason").value;
  }
});
$("episode-abnormal-note").addEventListener("input", () => {
  if (state.draft.episode_abnormal) {
    state.draft.episode_abnormal.note = $("episode-abnormal-note").value.trim() || null;
  }
});

initializeWorkspace();
