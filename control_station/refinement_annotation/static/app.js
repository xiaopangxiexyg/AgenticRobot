"use strict";

const state = {
  episodes: [],
  detail: null,
  draft: null,
  savedDraftSnapshot: null,
  volatilePostSealDraft: false,
  selectedEpisode: null,
  selectedCamera: null,
  selectedOccurrenceId: null,
  selectedDetailOccurrenceId: null,
  abnormalStart: null,
  recoverableFailureStart: null,
  recoverableFailureStartOccurrenceId: null,
  recoverableFailureEditingId: null,
  dragging: null,
  busy: false,
  workspaceConfigured: false,
  workspace: null,
  serverOnline: true,
  episodeRefreshInFlight: false,
  episodePageInFlight: false,
  episodeRevision: null,
  episodeTotalCount: 0,
  episodeHasMore: false,
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
  datasetBrowserDirectories: [],
  datasetBrowserRenderLimit: 250,
  defaultCameraPreference: "auto",
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
const episodeSummaryCommitEpoch = new Map();
let pendingEpisodeCommitRefresh = null;
const episodeRowNodes = new Map();
const episodeRowSignatures = new Map();
const datasetBrowserCache = new Map();
const datasetBrowserRequests = new Map();

const EPISODE_REFRESH_INTERVAL_MS = 4000;
// Keep the initial HTTP response and the browser's retained catalog bounded.
// The sidebar loads additional pages only when the operator asks for them.
const EPISODE_PAGE_SIZE = 200;
const EPISODE_RENDER_BATCH_SIZE = 200;
const EPISODE_SUMMARY_BATCH_SIZE = 32;
const EPISODE_SUMMARY_REFRESH_MS = 60000;
const DATASET_BROWSER_CACHE_MS = 15000;
const DATASET_BROWSER_RENDER_BATCH_SIZE = 250;
const EPISODE_PANEL_STORAGE_KEY =
  "dexteleop.annotation.episode_panel_collapsed";
const DEFAULT_CAMERA_STORAGE_KEY =
  "dexteleop.annotation.default_camera";
const DISPLAY_GAP_THRESHOLD_SEC = 1.0;
// Preserve short pauses at their real visual duration while bounding long
// no-sample spans so one operator hold cannot consume the whole timeline.
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

function compareEpisodeOrder(left, right) {
  const indexDelta = Number(left.episode_index) - Number(right.episode_index);
  return indexDelta || String(left.episode_uuid).localeCompare(
    String(right.episode_uuid),
  );
}

const STATUS_LOADING = {
  loaded: false,
  attention_level: "LOADING",
  attention_required: null,
  primary_label: "状态读取中",
  badges: [{code: "STATUS_LOADING", label: "状态读取中", tone: "LOADING"}],
};

function episodeStatus(item) {
  const status = item?.annotation_status;
  if (
    status
    && status.schema === "dexteleop.refinement_annotation.episode_status.v1"
    && Array.isArray(status.badges)
  ) return status;
  // Badge meaning belongs to the server; missing status is not reconstructed
  // from the flattened episode fields.
  return STATUS_LOADING;
}

function applyCommittedEpisodeSummary(payload, {markCommit = true} = {}) {
  const episodeUuid = state.detail?.episode_uuid;
  if (episodeUuid && markCommit) {
    episodeSummaryCommitEpoch.set(
      episodeUuid,
      (episodeSummaryCommitEpoch.get(episodeUuid) || 0) + 1,
    );
  }
  const summary = payload.episode_summary;
  const valid = summary?.episode_uuid === episodeUuid
    && summary.annotation_summary_loaded === true
    && episodeStatus(summary) !== STATUS_LOADING;
  const error = valid
    ? null
    : String(payload.episode_summary_error || "服务端未返回有效的 Episode 状态摘要。");
  if (valid) Object.assign(state.detail, summary);
  const index = state.episodes.findIndex(
    (item) => item.episode_uuid === episodeUuid,
  );
  if (index >= 0) {
    state.episodes[index] = valid
      ? {...state.episodes[index], ...summary, annotation_summary_error: null}
      : {
          ...state.episodes[index],
          annotation_status: null,
          annotation_summary_loaded: false,
          annotation_summary_error: error,
        };
    if (valid) episodeSummaryRefreshedAt.set(episodeUuid, Date.now());
    else episodeSummaryRefreshedAt.delete(episodeUuid);
  }
  return error;
}

function annotationSummarySignature(item) {
  return JSON.stringify({
    annotation_set_count: item.annotation_set_count,
    annotation_summary_loaded: item.annotation_summary_loaded,
    annotation_summary_error: item.annotation_summary_error || null,
    needs_second_review: item.needs_second_review,
    review_state: item.review_state,
    latest_annotation_mode: item.latest_annotation_mode,
    whole_episode_abnormal: item.whole_episode_abnormal,
    whole_episode_abnormal_reason: item.whole_episode_abnormal_reason,
    failure_stage_count: item.failure_stage_count,
    failure_attempt_count: item.failure_attempt_count,
    detail_annotation_complete: item.detail_annotation_complete,
    detail_annotation_state: item.detail_annotation_state,
    annotation_status: item.annotation_status || null,
    has_draft: item.has_draft,
    draft_requires_seal: item.draft_requires_seal,
  });
}

function badgeClassForStatus(code, tone) {
  const byCode = {
    WHOLE_EPISODE_ABNORMAL: "abnormal",
    SECOND_REVIEW_PENDING: "review",
    SECOND_REVIEW_COMPLETED: "review-complete",
    DETAIL_COMPLETE: "detail-complete",
    DETAIL_PENDING: "detail-pending",
    DETAIL_STALE: "detail-stale",
    DETAIL_ERROR: "summary-error",
    ANNOTATED_FAILURE: "failure",
    FAILURE_ATTEMPT: "failure",
    COARSE_SEALED: "sealed",
    COARSE_PENDING: "pending",
    LOCAL_DRAFT: "draft",
    STATUS_LOADING: "loading",
  };
  return byCode[code] || `tone-${String(tone || "info").toLowerCase()}`;
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

function commitWarnings(report) {
  const warnings = [];
  if (report?.draft_cache_warning) {
    warnings.push(`工作缓存更新失败：${report.draft_cache_warning}`);
  }
  if (report?.status_refresh_warning) {
    warnings.push(`Episode 状态读取失败：${report.status_refresh_warning}`);
  }
  return warnings;
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

function draftWithoutDetailOverlay(draft) {
  if (!draft) return draft;
  const copy = structuredClone(draft);
  delete copy.detail_overlay;
  return copy;
}

function hasUnsavedCoarseDraftChanges() {
  if (!state.detail || !state.draft) return false;
  if (!state.savedDraftSnapshot) return true;
  return !draftsEqual(
    draftWithoutDetailOverlay(state.draft),
    draftWithoutDetailOverlay(state.savedDraftSnapshot),
  );
}

function hasUnsavedDetailDraftChanges() {
  if (!state.detail || !state.draft) return false;
  if (!state.savedDraftSnapshot) return false;
  return !draftsEqual(
    state.draft.detail_overlay || null,
    state.savedDraftSnapshot.detail_overlay || null,
  );
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
    if (state.volatilePostSealDraft) {
      await persistVolatilePostSealDraft();
      showToast("粗标注后的详细工作草稿已持久化，继续切换。", "info", 4200);
      return true;
    }
    const coarseChanged = hasUnsavedCoarseDraftChanges();
    const detailChanged = hasUnsavedDetailDraftChanges();
    const report = coarseChanged
      ? await publishCurrentAnnotations()
      : detailChanged
        ? await publishDetailAnnotations()
        : null;
    const warnings = commitWarnings(report);
    showToast(
      warnings.length
        ? `标注已封存，但${warnings.join("；")}`
        : report?.pending_coarse
          ? "详细草稿已保存；粗标注仍可独立编辑，继续切换。"
          : "当前修改已校验并保存，继续切换。",
      warnings.length ? "error" : "info",
      warnings.length ? 9000 : 4200,
    );
    return !report?.draft_cache_warning;
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
  const count = state.episodeTotalCount || state.episodes.length;
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
  const projectedStatus = selected ? episodeStatus(selected) : null;
  const labels = (projectedStatus?.badges || [])
    .filter((badge) => [
      "WHOLE_EPISODE_ABNORMAL",
      "SECOND_REVIEW_PENDING",
      "SECOND_REVIEW_COMPLETED",
      "DETAIL_COMPLETE",
      "DETAIL_STALE",
      "DETAIL_ERROR",
      "ANNOTATED_FAILURE",
      "FAILURE_ATTEMPT",
    ].includes(String(badge.code)))
    .map((badge) => String(badge.label));
  status.textContent = labels.join(" / ") || "—";
  status.className = [
    "collapsed-episode-status",
    projectedStatus?.attention_level === "CRITICAL" ? "abnormal" : "",
    projectedStatus?.attention_level === "WARNING" ? "review" : "",
    projectedStatus?.badges?.some(
      (badge) => badge.code === "SECOND_REVIEW_COMPLETED",
    ) ? "review-complete" : "",
    projectedStatus?.badges?.some(
      (badge) => badge.code === "DETAIL_COMPLETE",
    ) ? "detail-complete" : "",
    projectedStatus?.badges?.some(
      (badge) => badge.code === "ANNOTATED_FAILURE"
        || badge.code === "FAILURE_ATTEMPT",
    ) && ["失败阶段标注", "含失败 Attempt"].includes(
      projectedStatus?.primary_label,
    ) ? "failure" : "",
  ].filter(Boolean).join(" ");
  status.title = labels.join("；")
    || "当前 Episode 无整条异常、复核、失败或详细标注状态";
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
  let targetIndex = index + offset;
  if (
    targetIndex >= state.episodes.length
    && state.episodeHasMore
  ) {
    await loadEpisodes({append: true});
  }
  targetIndex = clamp(
    targetIndex,
    0,
    state.episodes.length - 1,
  );
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

function storedDefaultCameraPreference() {
  const value = String(
    window.localStorage.getItem(DEFAULT_CAMERA_STORAGE_KEY) || "auto",
  );
  return ["auto", "front", "left_wrist", "right_wrist"].includes(value)
    ? value
    : "auto";
}

function preferredCameraForEpisode(episode) {
  const available = Array.isArray(episode?.camera_keys)
    ? episode.camera_keys
    : [];
  const preferred = state.defaultCameraPreference;
  if (preferred !== "auto" && available.includes(preferred)) return preferred;
  if (available.includes(episode?.default_camera_key)) {
    return episode.default_camera_key;
  }
  return available[0] || null;
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
  state.defaultCameraPreference = storedDefaultCameraPreference();
  $("launcher-default-camera").value = state.defaultCameraPreference;
  try {
    const payload = await requestJson("/api/workspace");
    const workspace = payload.workspace;
    $("launcher-dataset-root").value = workspace.dataset_root || "";
    $("launcher-annotator-id").value = workspace.annotator_id || "";
    $("launcher-second-review").checked = Boolean(workspace.second_review);
    if (workspace.configured) {
      showConfiguredWorkspace(workspace);
      await loadEpisodes();
    } else {
      // Hide the first remote directory round trip behind launcher setup.
      void fetchDatasetBrowser(workspace.dataset_root || "").catch(() => {
        // The visible picker will report an actionable error if opened.
      });
    }
  } catch (error) {
    $("launcher-status").textContent = error.message;
    $("launcher-status").classList.add("error");
  }
}

async function fetchDatasetBrowser(path = "") {
  const requested = String(path || "").trim();
  const cached = datasetBrowserCache.get(requested);
  if (cached && Date.now() - cached.loadedAt < DATASET_BROWSER_CACHE_MS) {
    return cached.payload;
  }
  const inFlight = datasetBrowserRequests.get(requested);
  if (inFlight) return inFlight;
  const pending = requestJson(
    `/api/dataset-browser?path=${encodeURIComponent(requested)}`,
  ).then((payload) => {
    const value = {payload, loadedAt: Date.now()};
    datasetBrowserCache.set(requested, value);
    datasetBrowserCache.set(payload.current, value);
    return payload;
  }).finally(() => {
    datasetBrowserRequests.delete(requested);
  });
  datasetBrowserRequests.set(requested, pending);
  return pending;
}

async function openWorkspace() {
  if (state.busy) return;
  const datasetRoot = $("launcher-dataset-root").value.trim();
  const annotatorId = $("launcher-annotator-id").value.trim();
  const secondReview = $("launcher-second-review").checked;
  state.defaultCameraPreference = $("launcher-default-camera").value;
  window.localStorage.setItem(
    DEFAULT_CAMERA_STORAGE_KEY,
    state.defaultCameraPreference,
  );
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
    updateTimelinePointCaptureState();
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
  const visible = directories.slice(0, state.datasetBrowserRenderLimit);
  const fragment = document.createDocumentFragment();
  for (const directory of visible) {
    const button = element("button", "dataset-browser-entry");
    button.type = "button";
    button.title = directory.path;
    const label = element("span");
    label.appendChild(element("strong", "", directory.name));
    label.appendChild(element("small", "", "目录"));
    button.appendChild(label);
    button.appendChild(element("small", "", "进入 ›"));
    button.addEventListener("click", () => browseDatasetPath(directory.path));
    fragment.appendChild(button);
  }
  root.appendChild(fragment);
  if (visible.length < directories.length) {
    const remaining = directories.length - visible.length;
    const more = element(
      "button",
      "dataset-browser-entry",
      `显示后续 ${Math.min(DATASET_BROWSER_RENDER_BATCH_SIZE, remaining)} 个目录`,
    );
    more.type = "button";
    more.addEventListener("click", () => {
      state.datasetBrowserRenderLimit += DATASET_BROWSER_RENDER_BATCH_SIZE;
      renderDatasetBrowserDirectories(state.datasetBrowserDirectories);
    });
    root.appendChild(more);
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
    const payload = await fetchDatasetBrowser(requested);
    setServerOnline(true);
    state.datasetBrowserPath = payload.current;
    $("dataset-browser-path").value = payload.current;
    $("dataset-browser-current").textContent = payload.current;
    $("dataset-browser-current").title = payload.current;
    const parent = payload.parent || "";
    $("dataset-browser-parent").dataset.path = parent;
    $("dataset-browser-parent").disabled = !parent;
    state.datasetBrowserDirectories = payload.directories || [];
    state.datasetBrowserRenderLimit = DATASET_BROWSER_RENDER_BATCH_SIZE;
    renderDatasetBrowserDirectories(state.datasetBrowserDirectories);
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
  state.episodePageInFlight = false;
  state.episodeRevision = null;
  state.episodeTotalCount = 0;
  state.episodeHasMore = false;
  state.episodeRenderLimit = EPISODE_RENDER_BATCH_SIZE;
  state.episodeScanInProgress = false;
  state.lastEpisodeRefreshError = null;
  pendingEpisodeSummaryIds.clear();
  inFlightEpisodeSummaryIds.clear();
  episodeSummaryRefreshedAt.clear();
  episodeSummaryCommitEpoch.clear();
  if (pendingEpisodeCommitRefresh) {
    pendingEpisodeCommitRefresh.resolve(false);
    pendingEpisodeCommitRefresh = null;
  }
  episodeRowNodes.clear();
  episodeRowSignatures.clear();
  if (episodeSummaryObserver) episodeSummaryObserver.disconnect();
  state.detail = null;
  state.draft = null;
  setSavedDraftSnapshot(null);
  state.selectedEpisode = null;
  state.selectedCamera = null;
  state.selectedOccurrenceId = null;
  state.abnormalStart = null;
  clearRecoverableFailureCapture({clearNote: true});
  state.timelineMap = null;
  state.numericSampleTimestamps = null;
  state.loadedMediaKey = null;
  state.loadingMediaKey = null;
  state.videoLoading = false;
  renderEpisodeList({preserveViewport: false});
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
    updateTimelinePointCaptureState();
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
  const explicitBreaks = new Set(
    (state.detail?.sample_continuity_break_indices || [])
      .map(Number)
      .filter((value) => Number.isInteger(value) && value > 0 && value < real.length),
  );
  const pauseSegments = (state.detail?.recording_pause_segments || [])
    .map((value) => ({
      ...value,
      start_timestamp: Number(value.start_timestamp),
      end_timestamp: Number(value.end_timestamp),
    }))
    .filter((value) => (
      Number.isFinite(value.start_timestamp)
      && Number.isFinite(value.end_timestamp)
      && value.end_timestamp > value.start_timestamp
    ));
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
    const continuityBreak = explicitBreaks.has(index);
    const compressed = continuityBreak || realDelta > DISPLAY_GAP_THRESHOLD_SEC;
    const displayDelta = compressed
      ? Math.max(nominal, Math.min(realDelta, DISPLAY_GAP_DURATION_SEC))
      : Math.max(realDelta, nominal);
    const displayStart = display[index - 1];
    display.push(displayStart + displayDelta);
    if (compressed) {
      const pause = pauseSegments.find((value) => (
        value.start_timestamp < real[index]
        && value.end_timestamp > real[index - 1]
      ));
      compressedGaps.push({
        realStart: real[index - 1],
        realEnd: real[index],
        realDuration: realDelta,
        displayDuration: displayDelta,
        displayStart,
        displayEnd: displayStart + displayDelta,
        continuityBreak,
        pause: pause || null,
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

function compressedGapTitle(gap) {
  const reason = gap.pause
    ? gap.pause.reason || (gap.pause.reasons || []).join(", ") || "未记录原因"
    : "未记录原因";
  const pauseDuration = gap.pause
    ? Number(gap.pause.end_timestamp) - Number(gap.pause.start_timestamp)
    : null;
  const duration = Number.isFinite(pauseDuration) && pauseDuration > 0
    ? `暂停 ${formatTime(pauseDuration)}，相邻样本间隔 ${formatTime(gap.realDuration)}`
    : `相邻样本间隔 ${formatTime(gap.realDuration)}`;
  return gap.continuityBreak
    ? `采集暂停/连续性中断：${reason}；${duration}，时间轴显示 ${formatTime(gap.displayDuration)}`
    : `无样本间隔：${duration}，时间轴显示 ${formatTime(gap.displayDuration)}`;
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

function snapTimelineBoundaryTimestamp(proposed, lower, upper) {
  const timestamps = sampleTimestamps();
  const minimum = Number(lower);
  const maximum = Number(upper);
  const target = Number(proposed);
  if (
    !timestamps.length
    || !Number.isFinite(target)
    || !Number.isFinite(minimum)
    || !Number.isFinite(maximum)
    || minimum > maximum
  ) return null;

  const candidates = [];
  const first = firstSampleIndexAtOrAfter(minimum);
  const last = lastSampleIndexAtOrBefore(maximum);
  if (
    first <= last
    && timestamps[first] >= minimum
    && timestamps[last] <= maximum
  ) {
    const nearest = clamp(nearestSampleIndex(target), first, last);
    candidates.push(timestamps[nearest]);
  }

  // A pause edge is an authoritative wall-clock fact even when recording was
  // already stopped and no sample exists exactly at that instant.
  for (const pause of state.detail?.recording_pause_segments || []) {
    for (const value of [pause.start_timestamp, pause.end_timestamp]) {
      const timestamp = Number(value);
      if (
        Number.isFinite(timestamp)
        && timestamp >= minimum
        && timestamp <= maximum
      ) candidates.push(timestamp);
    }
  }
  candidates.sort((left, right) =>
    Math.abs(left - target) - Math.abs(right - target) || left - right
  );
  return candidates[0] ?? null;
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

function clearRecoverableFailureCapture({clearNote = false} = {}) {
  state.recoverableFailureStart = null;
  state.recoverableFailureStartOccurrenceId = null;
  state.recoverableFailureEditingId = null;
  if (clearNote) {
    const note = $("recoverable-failure-note");
    if (note) note.value = "";
  }
}

function currentTimelineSamplePoint() {
  const timestamps = sampleTimestamps();
  if (!timestamps.length) return null;
  const sampleIndex = nearestSampleIndex(currentTimelineTime());
  return {
    sampleIndex,
    timestamp: timestamps[sampleIndex],
  };
}

function successOccurrenceOwningSample(timestamp) {
  return sortedOccurrences().find((value) =>
    value.outcome === "SUCCESS"
    && value.start_at <= timestamp
    && timestamp < value.end_at
  ) || null;
}

function recoverableFailureConflict(
  onsetSampleIndex,
  recoveredSampleIndex,
  {excludingAnnotationId = null} = {},
) {
  return recoverableFailures().find((event) =>
    event.annotation_id !== excludingAnnotationId
    && onsetSampleIndex <= Number(event.recovered_sample_index)
    && Number(event.onset_sample_index) <= recoveredSampleIndex
  ) || null;
}

function recoverableFailureAmbiguityIds() {
  const result = new Set();
  const byOccurrence = new Map();
  for (const event of recoverableFailures()) {
    const key = event.stage_occurrence_annotation_id;
    if (!byOccurrence.has(key)) byOccurrence.set(key, []);
    byOccurrence.get(key).push(event);
  }
  for (const values of byOccurrence.values()) {
    for (let index = 1; index < values.length; index += 1) {
      const previous = values[index - 1];
      const current = values[index];
      if (
        Number(current.onset_sample_index)
        <= Number(previous.recovered_sample_index)
      ) {
        result.add(previous.annotation_id);
        result.add(current.annotation_id);
      }
    }
  }
  return result;
}

function timelinePointCaptureReady() {
  return Boolean(
    state.detail
    && state.loadedMediaKey
    && !state.videoLoading
    && video.readyState >= HTMLMediaElement.HAVE_METADATA
    && Number.isFinite(video.duration)
    && video.duration > 0
  );
}

function updateTimelinePointCaptureState() {
  const ready = timelinePointCaptureReady();
  const disabled = state.busy || !ready;
  const recoverableStart = $("mark-recoverable-start");
  const recoverableEnd = $("mark-recoverable-end");
  recoverableStart.disabled = disabled;
  recoverableEnd.disabled = disabled || state.recoverableFailureStart === null;
  recoverableStart.title = ready ? "" : "请先加载视频并等待时间轴就绪";
  recoverableEnd.title = !ready
    ? "请先加载视频并等待时间轴就绪"
    : state.recoverableFailureStart === null
      ? "请先记录失误发生样本"
      : "";
  for (const id of ["mark-abnormal-start", "mark-abnormal-end"]) {
    const button = $(id);
    button.disabled = disabled;
    button.title = ready ? "" : "请先加载视频并等待时间轴就绪";
  }
  $("cancel-recoverable-capture").disabled = state.busy;
}

function requireTimelinePointCaptureReady() {
  if (timelinePointCaptureReady()) return true;
  showToast("请先加载视频并等待时间轴就绪后再取点。", "warning");
  return false;
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

function annotationAttempts() {
  const raw = state.detail?.attempts || [];
  const editor = state.detail?.capture_end_step_editor;
  const captureEnd = state.draft?.capture_end_step;
  if (!editor || !captureEnd) return raw;
  const sourceIndex = raw.findIndex(
    (value) => value.attempt_id === captureEnd.source_attempt_id,
  );
  const selectedIndex = (editor.options || []).findIndex(
    (value) => value.progress_step_id === captureEnd.end_progress_step_id,
  );
  const sourceStepIndex = (editor.options || []).findIndex(
    (value) => value.progress_step_id === editor.source_progress_step_id,
  );
  if (sourceIndex < 0 || selectedIndex < 0 || sourceStepIndex < 0) return raw;
  let anchorIndex = sourceIndex;
  let projectedOptions;
  if (selectedIndex <= sourceStepIndex) {
    anchorIndex = -1;
    for (let index = sourceIndex; index >= 0; index -= 1) {
      if (raw[index].progress_step_id === captureEnd.end_progress_step_id) {
        anchorIndex = index;
        break;
      }
    }
    if (anchorIndex < 0) return raw;
    projectedOptions = [editor.options[selectedIndex]];
  } else {
    projectedOptions = editor.options.slice(sourceStepIndex, selectedIndex + 1);
  }
  const anchor = raw[anchorIndex];
  const projected = [];
  for (const option of projectedOptions) {
    const attemptId = option.progress_step_id === anchor.progress_step_id
      ? anchor.attempt_id
      : `${anchor.attempt_id}:capture-end-step:${option.progress_step_id}`;
    const values = sortedOccurrences().filter(
      (value) => value.coarse_attempt_id === attemptId,
    );
    if (!values.length) return raw;
    projected.push({
      ...anchor,
      attempt_id: attemptId,
      progress_step_id: option.progress_step_id,
      progress_step_display_name: option.display_name,
      start_at: Math.min(...values.map((value) => Number(value.start_at))),
      end_at: Math.max(...values.map((value) => Number(value.end_at))),
      outcome: values.at(-1).outcome,
      failure_fact_id: null,
      failure_reported_at: null,
      annotation_projected_from_capture_end: true,
    });
  }
  return [...raw.slice(0, anchorIndex), ...projected];
}

function attemptsById() {
  return new Map(annotationAttempts().map((value) => [value.attempt_id, value]));
}

function effectiveProgressBoundary(transition) {
  const override = (state.draft?.progress_boundary_overrides || []).find(
    (value) => value.source_advance_fact_id === transition.advance_fact_id,
  );
  return Number(override?.corrected_effective_at ?? transition.effective_at);
}

function effectiveAttemptBounds() {
  const bounds = new Map(
    annotationAttempts().map((attempt) => [
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
  const corrected = snapTimelineBoundaryTimestamp(
    proposed,
    limits.lower,
    limits.upper,
  );
  if (corrected === null) return null;
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
  const attempts = annotationAttempts();
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

function detailAnnotationMeta() {
  return state.detail?.detail_annotation || null;
}

function detailProfileOptions() {
  const values = detailAnnotationMeta()?.profile_options;
  return Array.isArray(values) ? values : [];
}

function selectedDetailProfileSelectionId() {
  const draft = state.draft;
  const meta = detailAnnotationMeta();
  if (draft && Object.prototype.hasOwnProperty.call(
    draft,
    "detail_profile_selection_id",
  )) {
    if (typeof draft.detail_profile_selection_id === "string") {
      return draft.detail_profile_selection_id;
    }
  }
  return typeof meta?.selected_profile_selection_id === "string"
    ? meta.selected_profile_selection_id
    : "";
}

function detailOverlayDraft() {
  return state.draft?.detail_overlay || null;
}

function detailStages() {
  return detailAnnotationMeta()?.profile?.stages || [];
}

function detailStageById(stageId) {
  return detailStages().find((value) => value.stage_id === stageId) || null;
}

function detailSortedOccurrences() {
  return [...(detailOverlayDraft()?.occurrences || [])].sort((a, b) =>
    Number(a.start_at) - Number(b.start_at)
    || Number(a.end_at) - Number(b.end_at)
    || String(a.annotation_id).localeCompare(String(b.annotation_id))
  );
}

function detailStagePalette(stageId) {
  const stage = detailStageById(stageId);
  const target = detailAnnotationMeta()?.profile?.target_progress_step_id;
  const family = stepColorFamily(target);
  const stages = detailStages();
  const index = Math.max(0, Number(stage?.ordinal || 1) - 1);
  const denominator = Math.max(1, stages.length - 1);
  const tone = clamp(index / denominator, 0, 1);
  return {
    solid:
      "hsl(" + family.hue + " " + family.saturation + "% "
      + (42 + 32 * tone).toFixed(1) + "%)",
    background:
      "hsl(" + family.hue + " " + family.saturation + "% "
      + (28 + 40 * tone).toFixed(1) + "% / "
      + (0.42 - 0.08 * tone).toFixed(2) + ")",
    text: "hsl(" + family.hue + " 48% 94%)",
  };
}

function detailFailureAttempt(attemptId) {
  return (detailAnnotationMeta()?.failure_attempts || []).find(
    (value) => value.attempt_id === attemptId,
  ) || null;
}

function setDetailFailureStage(attemptId, stageId) {
  const overlay = detailOverlayDraft();
  const metaAttempt = detailFailureAttempt(attemptId);
  if (!overlay || !metaAttempt) return;
  const stages = detailStages();
  const selectedIndex = stageId
    ? stages.findIndex((value) => value.stage_id === stageId)
    : stages.length - 1;
  if (stageId && selectedIndex < 0) {
    showToast("所选详细失败子阶段不属于当前 profile。", "error");
    return;
  }
  const values = detailSortedOccurrences().filter(
    (value) => value.coarse_attempt_id === attemptId,
  );
  if (!values.length) {
    showToast("当前 Failure Attempt 没有可编辑的详细时间范围。", "error");
    return;
  }
  const startAt = Math.min(...values.map((value) => Number(value.start_at)));
  const endAt = Math.max(...values.map((value) => Number(value.end_at)));
  const selectedStages = stages.slice(0, selectedIndex + 1);
  const replacement = selectedStages.map((stage, index) => ({
    annotation_id:
      "detail-edit:" + attemptId + ":" + stage.stage_id + ":" + (index + 1),
    coarse_attempt_id: attemptId,
    progress_step_id: metaAttempt.progress_step_id,
    stage_id: stage.stage_id,
    occurrence_index_in_attempt: index + 1,
    start_at: startAt + (endAt - startAt) * index / selectedStages.length,
    end_at: startAt + (endAt - startAt) * (index + 1) / selectedStages.length,
    outcome: index === selectedStages.length - 1 && stageId
      ? "FAILURE"
      : "SUCCESS",
    confidence: 1.0,
    source_failure_fact_id: index === selectedStages.length - 1 && stageId
      ? metaAttempt.failure_fact_id
      : null,
  }));
  overlay.occurrences = detailSortedOccurrences()
    .filter((value) => value.coarse_attempt_id !== attemptId)
    .concat(replacement)
    .sort((a, b) =>
      Number(a.start_at) - Number(b.start_at)
      || Number(a.end_at) - Number(b.end_at)
      || String(a.annotation_id).localeCompare(String(b.annotation_id))
    );
  overlay.failure_stage_by_attempt[attemptId] = stageId || null;
  overlay.confirmed = false;
  state.selectedDetailOccurrenceId = replacement.at(-1)?.annotation_id || null;
  renderTimeline();
  if (stageId) {
    showToast(
      "已将 " + attemptId + " 的失败归属到“"
      + (detailStageById(stageId)?.display_name || stageId)
      + "”；请确认后封存详细补充。",
      "info",
      5000,
    );
  }
}

function confirmDetailAnnotation() {
  const meta = detailAnnotationMeta();
  const overlay = detailOverlayDraft();
  if (state.busy || !meta?.available || !overlay) return;
  const unresolved = (meta.failure_attempts || []).filter(
    (value) => !overlay.failure_stage_by_attempt?.[value.attempt_id],
  );
  if (unresolved.length) {
    showToast(
      "还有 " + unresolved.length + " 个原始 Failure Attempt 未选择详细失败子阶段。",
      "warning",
      6500,
    );
    return;
  }
  overlay.confirmed = true;
  renderDetailAnnotation();
  updateEditButtonState();
  showToast(
    "详细时间轴已确认；可直接保存详细草稿，或在粗标注版本确定后封存 detail。",
    "info",
    5200,
  );
}

function applyCurrentEpisodePayload(episode) {
  cancelPendingSeek();
  state.detail = episode;
  state.draft = structuredClone(episode.draft);
  state.volatilePostSealDraft = false;
  setSavedDraftSnapshot(
    episode.draft_requires_seal ? null : state.draft,
  );
  state.selectedDetailOccurrenceId = null;
  state.abnormalStart = null;
  clearRecoverableFailureCapture({clearNote: true});
  state.privilegedBaseline = null;
  state.timelineMap = null;
  state.numericSampleTimestamps = null;
  state.deletionHistory = structuredClone(
    state.draft.deletion_history || [],
  );
  const selectedIndex = state.episodes.findIndex(
    (value) => value.episode_uuid === episode.episode_uuid,
  );
  if (selectedIndex >= 0) {
    state.episodes[selectedIndex] = {
      ...state.episodes[selectedIndex],
      annotation_set_count: episode.annotation_set_count,
      needs_second_review: episode.needs_second_review,
      review_state: episode.review_state,
      latest_annotation_mode: episode.latest_annotation_mode,
      whole_episode_abnormal: episode.whole_episode_abnormal,
      whole_episode_abnormal_reason: episode.whole_episode_abnormal_reason,
      detail_annotation_complete:
        episode.detail_annotation_complete === true
        || episode.detail_annotation?.resolution_level === "DETAIL_COMPLETE",
      detail_annotation_state: episode.detail_annotation_state,
      annotation_status: episode.annotation_status,
      annotation_summary_loaded: true,
      annotation_summary_error: null,
    };
    state.episodeRenderLimit = Math.max(
      state.episodeRenderLimit,
      selectedIndex + 1,
    );
    episodeSummaryRefreshedAt.set(episode.episode_uuid, Date.now());
  }
  buildTimelineMap();
  renderEpisodeList({preserveViewport: true});
  renderEditor();
}

async function changeDetailProfileSelection(event) {
  const select = event.target;
  const nextSelectionId = String(select.value || "");
  const previousSelectionId = selectedDetailProfileSelectionId();
  if (!state.detail || !state.draft || state.busy) {
    select.value = previousSelectionId;
    return;
  }
  if (nextSelectionId === previousSelectionId) return;
  const draft = structuredClone(state.draft);
  draft.detail_profile_selection_id = nextSelectionId;
  // A detail draft is bound to the selected projected profile.  Switching the
  // choice therefore starts a fresh optional overlay, while the coarse draft
  // and its capture facts remain untouched.
  draft.detail_overlay = null;
  state.busy = true;
  try {
    await requestJsonWithProgress(
      "/api/draft/" + encodeURIComponent(state.detail.episode_uuid),
      {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(draft),
      },
      {
        kind: "SAVE_DETAIL_PROFILE_SELECTION",
        message: "正在保存详细标注 profile 选择",
        completeMessage: "详细标注 profile 已切换",
      },
    );
    const payload = await requestJson(
      "/api/episode/" + encodeURIComponent(state.detail.episode_uuid),
    );
    applyCurrentEpisodePayload(payload.episode);
    showToast(
      nextSelectionId
        ? "已选择详细标注 profile；粗粒度采集事实和时间轴保持不变。"
        : "已关闭详细补充；当前只使用粗粒度标注。",
      "info",
      5200,
    );
  } catch (error) {
    select.value = previousSelectionId;
    showToast(`切换详细标注 profile 失败：${error.message}`, "error", 8500);
  } finally {
    state.busy = false;
    renderDetailAnnotation();
    updateTimelinePointCaptureState();
  }
}

function renderDetailAnnotation() {
  const panel = $("detail-annotation-panel");
  if (!panel) return;
  const meta = detailAnnotationMeta();
  const overlay = detailOverlayDraft();
  const options = detailProfileOptions();
  const selectedSelectionId = selectedDetailProfileSelectionId();
  const profileStages = Array.isArray(meta?.profile?.stages)
    ? meta.profile.stages
    : [];
  const available = Boolean(meta?.available && overlay);
  const previewAvailable = Boolean(!available && meta?.profile && profileStages.length);
  const hasProfileChoice = options.length > 0;
  const canRenderDetail = available || previewAvailable;
  panel.classList.toggle(
    "hidden",
    !(canRenderDetail || hasProfileChoice),
  );
  if (!(canRenderDetail || hasProfileChoice)) return;
  const profileSelect = $("detail-profile-select");
  if (profileSelect) {
    profileSelect.replaceChildren();
    const coarseOption = document.createElement("option");
    coarseOption.value = "";
    coarseOption.textContent = "仅粗粒度标注（不启用 detail）";
    profileSelect.appendChild(coarseOption);
    for (const optionValue of options) {
      const option = document.createElement("option");
      option.value = optionValue.selection_id;
      const stageNames = (optionValue.stages || [])
        .map((stage) => stage.display_name || stage.stage_id)
        .join(" → ");
      option.textContent = [
        optionValue.target_progress_step_id,
        `${optionValue.profile_id}@${optionValue.profile_version}`,
        stageNames,
      ].filter(Boolean).join(" · ");
      profileSelect.appendChild(option);
    }
    profileSelect.value = options.some(
      (value) => value.selection_id === selectedSelectionId,
    ) ? selectedSelectionId : "";
    profileSelect.disabled = state.busy || !state.detail || !state.draft;
  }
  const selectionNote = $("detail-profile-selection-note");
  if (selectionNote) {
    selectionNote.textContent = hasProfileChoice
      ? selectedSelectionId && meta?.profile
        ? "当前已选择该详细 profile；切换选择会重新生成独立 detail 草稿，粗粒度标注不会被替换。"
        : "当前仅使用粗粒度标注；选择一个 profile 后即可直接补充详细阶段，不要求先封存粗标注。"
      : "当前 Episode 没有可兼容的详细标注 profile；粗粒度标注仍可正常使用。";
  }
  if (!meta?.profile) {
    const status = $("detail-annotation-status");
    if (status) {
      status.textContent = hasProfileChoice
        ? `${options.length} 个详细标注 profile 可选；当前仍只使用粗粒度标注。`
        : (meta?.reason || "没有匹配的详细 profile。");
    }
    $("detail-annotation-actions")?.classList.add("hidden");
    $("confirm-detail-annotation")?.classList.add("hidden");
    $("publish-detail-annotation")?.classList.add("hidden");
    $("detail-stage-track")?.replaceChildren();
    $("detail-failure-editors")?.replaceChildren();
    $("detail-stage-descriptions")?.replaceChildren();
    return;
  }
  const status = $("detail-annotation-status");
  const sealed = meta.resolution_level === "DETAIL_COMPLETE";
  const confirmed = Boolean(overlay?.confirmed);
  const confirmButton = $("confirm-detail-annotation");
  const publishButton = $("publish-detail-annotation");
  $("detail-annotation-actions")?.classList.toggle(
    "hidden",
    !canRenderDetail,
  );
  confirmButton?.classList.toggle("hidden", !available);
  publishButton?.classList.toggle("hidden", !available);
  if (publishButton) {
    publishButton.textContent = meta.base
      ? "封存详细补充"
      : "保存详细草稿";
    publishButton.title = meta.base
      ? "发布绑定当前粗标注版本的不可变 detail overlay"
      : "先保存详细草稿；粗标注封存后再发布 detail overlay";
  }
  status.textContent = previewAvailable
    ? [
        "详细 profile 已匹配；当前仅显示"
        + profileStages.length + "个子阶段预览",
        meta.profile?.target_progress_step_id
          ? "补充子任务：" + meta.profile.target_progress_step_id
          : "",
        meta.reason || "",
      ].filter(Boolean).join(" · ")
    : [
        sealed ? "已发布详细 overlay" : "当前有效分析仍使用粗粒度 fallback",
        confirmed ? "当前详细草稿已确认" : "当前详细草稿未确认",
        !meta.base ? "粗、细标注可独立编辑" : "",
        meta.profile?.target_progress_step_id
          ? "补充子任务：" + meta.profile.target_progress_step_id
          : "",
        meta.reason || "",
      ].filter(Boolean).join(" · ");
  const track = $("detail-stage-track");
  track.replaceChildren();
  const values = detailSortedOccurrences();
  for (const value of values) {
    const stage = detailStageById(value.stage_id);
    const block = element(
      "div",
      [
        "detail-stage-block",
        value.outcome === "FAILURE" ? "failure" : "",
        value.outcome === "TRUNCATED_BY_ABNORMAL" ? "truncated" : "",
        value.annotation_id === state.selectedDetailOccurrenceId
          ? "selected"
          : "",
      ].filter(Boolean).join(" "),
    );
    const palette = detailStagePalette(value.stage_id);
    block.style.setProperty("--stage-color", palette.solid);
    block.style.setProperty("--stage-bg", palette.background);
    block.style.setProperty("--stage-text", palette.text);
    setPosition(block, value.start_at, value.end_at);
    block.appendChild(element("strong", "", stage?.display_name || value.stage_id));
    block.appendChild(
      element(
        "small",
        "",
        occurrenceOutcomeLabel(value.outcome)
        + " · " + formatTime(Number(value.end_at) - Number(value.start_at)),
      ),
    );
    block.title = "详细阶段 · 点击定位视频";
    block.addEventListener("click", () => {
      state.selectedDetailOccurrenceId = value.annotation_id;
      renderDetailAnnotation();
      seekTimeline(value.start_at);
    });
    track.appendChild(block);
  }
  if (previewAvailable) {
    track.appendChild(
      element(
        "div",
        "detail-stage-preview-empty",
        "粗粒度标注封存后，这里会按详细 profile 生成可编辑时间块。",
      ),
    );
  }
  for (const gap of state.timelineMap?.compressedGaps || []) {
    const marker = element(
      "div",
      `compressed-gap${gap.continuityBreak ? " collection-pause" : ""}`,
    );
    setPosition(marker, gap.realStart, gap.realEnd);
    marker.title = compressedGapTitle(gap);
    track.appendChild(marker);
  }
  for (let index = 1; index < values.length; index += 1) {
    const left = values[index - 1];
    const right = values[index];
    if (
      left.coarse_attempt_id !== right.coarse_attempt_id
      || Math.abs(Number(left.end_at) - Number(right.start_at)) > 1e-7
    ) continue;
    const handle = element("button", "boundary-handle detail-boundary-handle");
    handle.type = "button";
    handle.style.left = timelineRatio(right.start_at) * 100 + "%";
    handle.title = "拖动详细阶段边界";
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      state.dragging = {
        kind: "detail-boundary",
        left,
        right,
        handle,
        pointerId: event.pointerId,
        rect: track.getBoundingClientRect(),
        lastTimestamp: null,
      };
    });
    handle.addEventListener("pointermove", (event) => {
      if (
        !state.dragging
        || state.dragging.kind !== "detail-boundary"
        || state.dragging.pointerId !== event.pointerId
      ) return;
      const rect = state.dragging.rect;
      const ratio = clamp(
        (event.clientX - rect.left) / rect.width,
        0,
        1,
      );
      const minimum = Math.max(
        0.02,
        episodeDuration() / Math.max(10000, state.detail.frame_count * 20),
      );
      const timestamp = snapTimelineBoundaryTimestamp(
        timelineAtRatio(ratio),
        state.dragging.left.start_at + minimum,
        state.dragging.right.end_at - minimum,
      );
      if (timestamp === null) return;
      state.dragging.left.end_at = timestamp;
      state.dragging.right.start_at = timestamp;
      handle.style.left = timelineRatio(timestamp) * 100 + "%";
      state.draft.detail_overlay.confirmed = false;
      if (state.dragging.lastTimestamp === timestamp) return;
      state.dragging.lastTimestamp = timestamp;
      scheduleSeek(timestamp);
    });
    const finish = () => {
      if (!state.dragging || state.dragging.kind !== "detail-boundary") return;
      flushPendingSeek();
      state.dragging = null;
      renderTimeline();
    };
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
    handle.addEventListener("lostpointercapture", finish);
    track.appendChild(handle);
  }
  const playhead = element("div", "playhead");
  track.appendChild(playhead);
  const descriptions = $("detail-stage-descriptions");
  descriptions.replaceChildren();
  for (const stage of profileStages) {
    const hasOccurrence = values.some((value) => value.stage_id === stage.stage_id);
    const card = element("div", "stage-description");
    if (!hasOccurrence) card.classList.add("not-yet-materialized");
    const palette = detailStagePalette(stage.stage_id);
    card.style.setProperty("--stage-color", palette.solid);
    card.style.setProperty("--stage-bg", palette.background);
    card.style.setProperty("--stage-text", palette.text);
    card.appendChild(element("strong", "", stage.display_name));
    card.appendChild(element("span", "", stage.description));
    if (!hasOccurrence) {
      card.appendChild(element("small", "", "等待详细时间块生成"));
    }
    descriptions.appendChild(card);
  }
  const failureEditors = $("detail-failure-editors");
  failureEditors.replaceChildren();
  for (const attempt of available ? (meta.failure_attempts || []) : []) {
    const row = element("div", "detail-failure-editor");
    row.appendChild(element("strong", "", "Failure Attempt " + attempt.attempt_id));
    const select = document.createElement("select");
    select.appendChild(element("option", "", "请选择失败发生于哪个详细子阶段"));
    select.options[0].value = "";
    for (const optionValue of attempt.options || []) {
      const option = document.createElement("option");
      option.value = optionValue.stage_id;
      option.textContent = optionValue.display_name;
      select.appendChild(option);
    }
    select.value = overlay.failure_stage_by_attempt?.[attempt.attempt_id] || "";
    select.addEventListener("change", () => {
      setDetailFailureStage(attempt.attempt_id, select.value || null);
    });
    row.appendChild(select);
    failureEditors.appendChild(row);
  }
  if (confirmButton) {
    confirmButton.disabled = !available || state.busy || confirmed;
  }
  if (publishButton) {
    publishButton.disabled = !available || state.busy || !confirmed;
  }
  $("detail-stage-track").querySelectorAll(".playhead").forEach((node) => {
    node.style.left = timelineRatio(currentTimelineTime()) * 100 + "%";
  });
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
  for (const attempt of annotationAttempts()) {
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
  for (const attempt of annotationAttempts()) {
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
  const attempts = annotationAttempts();
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
    clearRecoverableFailureCapture({clearNote: true});
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
    updateTimelinePointCaptureState();
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

async function reflowCaptureEndStep(editor, endProgressStepId, select) {
  if (!state.draft || !state.detail || state.busy || !endProgressStepId) return;
  const current = state.draft.capture_end_step?.end_progress_step_id || "";
  if (endProgressStepId === current) return;
  if (
    sortedOccurrences().length > 0
    && !window.confirm(
      "更改采集结束子任务会按 profile 重新生成最后一段的子阶段，并重新均分这些子阶段的初始边界；原始 MANUAL_STOP、视频、样本和此前已确认的采集 Failure 不会改变。是否继续？",
    )
  ) {
    select.value = current;
    return;
  }
  state.busy = true;
  select.disabled = true;
  try {
    const payload = await requestJson(
      `/api/reflow-capture-end-step/${encodeURIComponent(state.detail.episode_uuid)}`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          draft: state.draft,
          end_progress_step_id: endProgressStepId,
        }),
      },
    );
    state.draft = payload.draft;
    state.deletionHistory = [];
    state.selectedOccurrenceId = null;
    state.abnormalStart = null;
    clearRecoverableFailureCapture({clearNote: true});
    state.privilegedBaseline = null;
    renderEditor();
    const selected = (editor.options || []).find(
      (value) => value.progress_step_id === endProgressStepId,
    );
    showToast(
      `采集结束位置已设为“${selected?.display_name || endProgressStepId}”；该子任务默认正常完成，请拖动白色边界精调其子阶段。`,
      "info",
      6500,
    );
  } catch (error) {
    select.value = current;
    showToast(`采集结束子任务调整失败：${error.message}`, "error", 8500);
  } finally {
    state.busy = false;
    updateTimelinePointCaptureState();
  }
}

function renderCaptureEndStepEditor() {
  const root = $("capture-end-step-editor");
  root.replaceChildren();
  const editor = state.detail?.capture_end_step_editor;
  if (!editor) {
    root.classList.add("hidden");
    return;
  }
  const summary = element("div", "");
  summary.appendChild(element("strong", "", "采集结束位置"));
  summary.appendChild(
    element(
      "small",
      "",
      `原始 ${editor.termination_cause} / ${editor.source_attempt_outcome} 保持只读；这里只补充后标注的子任务结束解释。`,
    ),
  );
  const label = element("label", "", "结束于子任务");
  const select = element("select", "");
  const current = state.draft?.capture_end_step?.end_progress_step_id || "";
  if (!current) {
    const legacy = document.createElement("option");
    legacy.value = "";
    legacy.textContent = "旧标注：尚未声明结束子任务";
    select.appendChild(legacy);
  }
  for (const option of editor.options || []) {
    const node = document.createElement("option");
    node.value = option.progress_step_id;
    node.textContent = `${option.ordinal}. ${option.display_name}`;
    node.disabled = option.selectable === false;
    if (option.disabled_reason) node.title = option.disabled_reason;
    select.appendChild(node);
  }
  select.value = current;
  select.addEventListener("change", () => {
    void reflowCaptureEndStep(editor, select.value, select);
  });
  label.appendChild(select);
  root.append(summary, label);
  root.classList.remove("hidden");
}

function occurrenceAtTimelineTime(timestamp) {
  const values = sortedOccurrences();
  const occurrence = values.find(
    (value) => value.start_at <= timestamp && timestamp < value.end_at,
  );
  if (occurrence) return occurrence;
  const finalOccurrence = values.at(-1);
  return (
    finalOccurrence
    && Math.abs(timestamp - finalOccurrence.end_at) <= 1e-9
  ) ? finalOccurrence : null;
}

function updatePlayhead(timestamp = currentTimelineTime()) {
  if (!state.detail) return;
  const left = `${timelineRatio(timestamp) * 100}%`;
  for (const track of [
    $("coarse-track"),
    $("stage-track"),
    $("failure-trigger-track"),
    $("recoverable-failure-track"),
    $("detail-stage-track"),
  ]) {
    if (!track) continue;
    let playhead = track.querySelector(".playhead");
    if (!playhead) {
      playhead = element("div", "playhead");
      track.appendChild(playhead);
    }
    playhead.style.left = left;
  }
  $("video-timecode").textContent = formatTime(timestamp - state.detail.start_timestamp);
  const occurrence = occurrenceAtTimelineTime(timestamp);
  if (!occurrence) {
    $("video-context").textContent = "未标注区间";
    return;
  }
  const stage = stageById(occurrence.stage_id);
  $("video-context").textContent =
    `${stage?.display_name || occurrence.stage_id} · ${occurrenceOutcomeLabel(occurrence.outcome)}`;
}

async function loadEpisodes({
  silent = false,
  force = false,
  append = false,
  afterCommit = false,
} = {}) {
  if (!state.workspaceConfigured || (append && !state.episodeHasMore)) return false;
  if (state.episodeRefreshInFlight || state.episodePageInFlight) {
    if (!afterCommit) return false;
    // Commit callers await one shared refresh after the current request. Their
    // existing summary restoration then runs after that refresh, not before it.
    if (!pendingEpisodeCommitRefresh) {
      let resolve;
      const promise = new Promise((done) => { resolve = done; });
      pendingEpisodeCommitRefresh = {
        promise, resolve, force: false, generation: state.workspaceGeneration,
      };
    }
    pendingEpisodeCommitRefresh.force ||= force;
    return pendingEpisodeCommitRefresh.promise;
  }
  state.episodeRefreshInFlight = true;
  state.episodePageInFlight = append;
  const previousIds = new Set(
    state.episodes.map((value) => value.episode_uuid),
  );
  const workspaceGeneration = state.workspaceGeneration;
  const requestedCommitEpochs = new Map(episodeSummaryCommitEpoch);
  try {
    const params = new URLSearchParams();
    params.set("stale_ok", "1");
    params.set("offset", String(append ? state.episodes.length : 0));
    params.set("limit", String(EPISODE_PAGE_SIZE));
    if (force) params.set("refresh", "1");
    const query = `?${params.toString()}`;
    const payload = await requestJson(`/api/episodes${query}`);
    if (workspaceGeneration !== state.workspaceGeneration) return false;
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
    const revisionChanged = !revisionKnown
      || state.episodeRevision !== nextRevision;
    const replaceLoadedPage = !append
      && (state.episodeRevision === null || state.episodes.length === 0);
    // A catalog request issued before a commit must not replace its newer
    // authoritative status when that older request finishes later.
    const committedSinceRequest = (episodeUuid) => (
      (episodeSummaryCommitEpoch.get(episodeUuid) || 0)
        !== (requestedCommitEpochs.get(episodeUuid) || 0)
    );
    const retainNewerCommit = (item, existing) => (
      existing && committedSinceRequest(item.episode_uuid)
    ) ? existing : item;
    if (append) {
      const byId = new Map(
        state.episodes.map((value) => [value.episode_uuid, value]),
      );
      for (const item of payload.episodes) {
        byId.set(
          item.episode_uuid,
          retainNewerCommit(item, byId.get(item.episode_uuid)),
        );
      }
      state.episodes = Array.from(byId.values()).sort(
        compareEpisodeOrder,
      );
    } else if (replaceLoadedPage || revisionChanged) {
      // Retain already loaded pages (and the currently selected row) during a
      // background revision update.  Only the first page is replaced; this
      // avoids pulling the entire catalog back over the wire every 4 seconds.
      const retainedItems = state.episodes.filter(
        (value) => value.episode_uuid === state.selectedEpisode
          || committedSinceRequest(value.episode_uuid),
      );
      const byId = new Map(
        (replaceLoadedPage ? retainedItems : state.episodes).map(
          (value) => [value.episode_uuid, value],
        ),
      );
      for (const item of payload.episodes) {
        byId.set(
          item.episode_uuid,
          retainNewerCommit(item, byId.get(item.episode_uuid)),
        );
      }
      state.episodes = Array.from(byId.values()).sort(
        compareEpisodeOrder,
      );
    } else {
      const byId = new Map(
        state.episodes.map((value) => [value.episode_uuid, value]),
      );
      // A lightweight catalog poll intentionally omits expensive annotation
      // summaries. If its revision is unchanged, retain the already hydrated
      // row instead of regressing it to loading and rereading remote files.
      for (const item of payload.episodes) {
        const existing = byId.get(item.episode_uuid);
        byId.set(
          item.episode_uuid,
          existing ? {...item, ...existing} : item,
        );
      }
      state.episodes = Array.from(byId.values()).sort(
        compareEpisodeOrder,
      );
    }
    state.episodeRevision = revisionKnown ? nextRevision : null;
    state.episodeTotalCount = Number.isFinite(Number(payload.total_count))
      ? Number(payload.total_count)
      : state.episodes.length;
    state.episodeHasMore = Boolean(payload.has_more);
    const episodesChanged = append || replaceLoadedPage || revisionChanged;
    if (episodesChanged) {
      // Do not expand the DOM just because a selected Episode lives in a
      // later page.  The operator can explicitly reveal more rows below.
      state.episodeRenderLimit = Math.max(
        EPISODE_RENDER_BATCH_SIZE,
        Math.min(
          state.episodeRenderLimit,
          Math.max(state.episodes.length, EPISODE_RENDER_BATCH_SIZE),
        ),
      );
    }
    $("dataset-root").textContent = payload.dataset_root;
    $("episode-count").textContent = String(state.episodeTotalCount);
    if (episodesChanged) renderEpisodeList({preserveViewport: true});
    else if (state.episodes.slice(0, state.episodeRenderLimit).some((item) => (
      item.episode_uuid !== state.selectedEpisode
      && Date.now() - (episodeSummaryRefreshedAt.get(item.episode_uuid) || 0)
        >= EPISODE_SUMMARY_REFRESH_MS
    ))) renderEpisodeList({preserveViewport: true});
    if (
      state.selectedEpisode
      && !state.episodes.some((value) => value.episode_uuid === state.selectedEpisode)
    ) {
      state.selectedEpisode = null;
      state.detail = null;
      state.draft = null;
      state.abnormalStart = null;
      clearRecoverableFailureCapture({clearNote: true});
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
    return true;
  } catch (error) {
    setServerOnline(false);
    if (!silent) showToast(error.message, "error", 6000);
    return false;
  } finally {
    state.episodeRefreshInFlight = false;
    state.episodePageInFlight = false;
    const queued = pendingEpisodeCommitRefresh;
    pendingEpisodeCommitRefresh = null;
    if (queued) {
      if (queued.generation === state.workspaceGeneration && state.workspaceConfigured) {
        void loadEpisodes({silent: true, force: queued.force}).then(
          queued.resolve, () => queued.resolve(false),
        );
      } else queued.resolve(false);
    }
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
  const requestedCommitEpochs = new Map(
    episodeUuids.map((episodeUuid) => [
      episodeUuid,
      episodeSummaryCommitEpoch.get(episodeUuid) || 0,
    ]),
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
    const isCurrent = (value) => requestedCommitEpochs.get(value.episode_uuid)
      === (episodeSummaryCommitEpoch.get(value.episode_uuid) || 0);
    const byId = new Map(
      payload.summaries.filter(isCurrent).map((value) => [value.episode_uuid, value]),
    );
    const errorsById = new Map(
      payload.errors.filter(isCurrent).map((value) => [value.episode_uuid, value.error]),
    );
    let summariesChanged = false;
    state.episodes = state.episodes.map((item) => {
      const updated = byId.has(item.episode_uuid)
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
          : item;
      if (
        updated !== item
        && annotationSummarySignature(updated)
          !== annotationSummarySignature(item)
      ) summariesChanged = true;
      return updated;
    });
    const refreshedAt = Date.now();
    for (const episodeUuid of byId.keys()) {
      episodeSummaryRefreshedAt.set(episodeUuid, refreshedAt);
    }
    if (errorsById.size) {
      showToast(
        `有 ${errorsById.size} 个 Episode 的标注摘要暂时无法读取。`,
        "error",
        7000,
      );
    }
    if (summariesChanged) renderEpisodeList({preserveViewport: true});
    else updateEpisodeNavigation();
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

function renderEpisodeRow(item) {
  const status = episodeStatus(item);
  const signature = JSON.stringify({
    episode_index: item.episode_index,
    duration_sec: item.duration_sec,
    frame_count: item.frame_count,
    resolution_kind: item.resolution_kind,
    source_batch_id: item.source_batch_id || null,
    selected: item.episode_uuid === state.selectedEpisode,
    summary: annotationSummarySignature(item),
  });
  let button = episodeRowNodes.get(item.episode_uuid);
  if (!button) {
    button = element("button", "episode-item");
    button.type = "button";
    button.addEventListener("click", () => {
      void selectEpisode(button.dataset.episodeUuid);
    });
    episodeRowNodes.set(item.episode_uuid, button);
  }
  if (episodeRowSignatures.get(item.episode_uuid) === signature) return button;

  const hasFailureStatus = (status.badges || []).some(
    (value) => value.code === "ANNOTATED_FAILURE"
      || value.code === "FAILURE_ATTEMPT",
  );
  button.className = [
    "episode-item",
    item.episode_uuid === state.selectedEpisode ? "active" : "",
    status.attention_level === "CRITICAL" ? "attention-critical" : "",
    status.attention_level === "WARNING" ? "attention-warning" : "",
    hasFailureStatus
      && ["失败阶段标注", "含失败 Attempt"].includes(status.primary_label)
      ? "attention-failure"
      : "",
  ].filter(Boolean).join(" ");
  button.title = `${item.episode_uuid} · ${status.primary_label || "状态未知"}`;
  button.dataset.episodeUuid = item.episode_uuid;

  const content = document.createDocumentFragment();
  content.appendChild(
    element(
      "strong",
      "",
      `#${item.episode_index} · ${formatTime(item.duration_sec)}`,
    ),
  );
  if (["CRITICAL", "WARNING"].includes(status.attention_level)) {
    content.appendChild(
      element("small", "episode-attention-line", status.primary_label),
    );
  }
  const badges = element("div", "episode-badges");
  badges.appendChild(element("span", "badge", `${item.frame_count} 帧`));
  if (item.resolution_kind === "RETROSPECTIVE_MIGRATED") {
    badges.appendChild(element("span", "badge", "历史迁移"));
  }
  if (item.source_batch_id) {
    badges.appendChild(element("span", "badge", item.source_batch_id));
  }
  if (item.annotation_summary_error) {
    const badge = element("span", "badge summary-error", "状态读取失败");
    badge.title = item.annotation_summary_error;
    badges.appendChild(badge);
  }
  for (const badgeStatus of status.badges || []) {
    if (
      item.annotation_summary_error
      && badgeStatus.code === "STATUS_LOADING"
    ) continue;
    badges.appendChild(
      element(
        "span",
        `badge ${badgeClassForStatus(badgeStatus.code, badgeStatus.tone)}`,
        String(badgeStatus.label),
      ),
    );
  }
  content.appendChild(badges);
  button.replaceChildren(content);
  episodeRowSignatures.set(item.episode_uuid, signature);
  return button;
}

function renderEpisodeList({preserveViewport = true} = {}) {
  const root = $("episode-list");
  const scrollTop = preserveViewport ? root.scrollTop : 0;
  const focusedEpisodeUuid = preserveViewport
    && document.activeElement?.closest?.("[data-episode-uuid]")
    ?.dataset.episodeUuid;
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
    root.scrollTop = scrollTop;
    return;
  }
  const renderedEpisodes = state.episodes.slice(0, state.episodeRenderLimit);
  for (const item of renderedEpisodes) {
    const button = renderEpisodeRow(item);
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
  const retainedEpisodeIds = new Set(
    state.episodes.map((item) => item.episode_uuid),
  );
  for (const episodeUuid of episodeRowNodes.keys()) {
    if (retainedEpisodeIds.has(episodeUuid)) continue;
    episodeRowNodes.delete(episodeUuid);
    episodeRowSignatures.delete(episodeUuid);
    episodeSummaryRefreshedAt.delete(episodeUuid);
    // Commit ordering outlives DOM rows, including rows scrolled off screen.
    // The workspace reset clears these epochs with the catalog itself.
  }
  if (renderedEpisodes.length < state.episodes.length) {
    const remaining = state.episodes.length - renderedEpisodes.length;
    const loadMore = element(
      "button",
      "episode-item",
      `显示后续 ${Math.min(EPISODE_RENDER_BATCH_SIZE, remaining)} 条`,
    );
    loadMore.type = "button";
    loadMore.addEventListener("click", () => {
      state.episodeRenderLimit += EPISODE_RENDER_BATCH_SIZE;
      renderEpisodeList();
    });
    root.appendChild(loadMore);
  } else if (state.episodeHasMore) {
    const loadMore = element(
      "button",
      "episode-item",
      state.episodePageInFlight
        ? "正在读取后续 Episode…"
        : `读取后续 ${EPISODE_PAGE_SIZE} 条`,
    );
    loadMore.type = "button";
    loadMore.disabled = state.episodePageInFlight;
    loadMore.addEventListener("click", () => {
      void loadEpisodes({append: true});
    });
    root.appendChild(loadMore);
  }
  root.scrollTop = scrollTop;
  if (focusedEpisodeUuid) {
    const focusTarget = Array.from(
      root.querySelectorAll("[data-episode-uuid]"),
    ).find((node) => node.dataset.episodeUuid === focusedEpisodeUuid);
    focusTarget?.focus({preventScroll: true});
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
    state.volatilePostSealDraft = false;
    setSavedDraftSnapshot(
      payload.episode.draft_requires_seal ? null : state.draft,
    );
    state.selectedCamera = preferredCameraForEpisode(payload.episode);
    state.selectedOccurrenceId = null;
    state.selectedDetailOccurrenceId = null;
    state.abnormalStart = null;
    clearRecoverableFailureCapture({clearNote: true});
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
        failure_stage_count: payload.episode.failure_stage_count,
        failure_attempt_count: payload.episode.failure_attempt_count,
        has_draft: payload.episode.has_draft,
        draft_requires_seal: payload.episode.draft_requires_seal,
        detail_annotation_complete:
          payload.episode.detail_annotation_complete === true
          || payload.episode.detail_annotation?.resolution_level
            === "DETAIL_COMPLETE",
        detail_annotation_state: payload.episode.detail_annotation_state,
        annotation_status: payload.episode.annotation_status,
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
    renderEpisodeList({preserveViewport: true});
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
    updateTimelinePointCaptureState();
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
  renderCaptureEndStepEditor();
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
  const camera = preferredCameraForEpisode(next);
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
  updateTimelinePointCaptureState();
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
  updateTimelinePointCaptureState();
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
    updateTimelinePointCaptureState();
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
    updateTimelinePointCaptureState();
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
      const marker = element(
        "div",
        `compressed-gap${gap.continuityBreak ? " collection-pause" : ""}`,
      );
      setPosition(marker, gap.realStart, gap.realEnd);
      marker.title = compressedGapTitle(gap);
      track.appendChild(marker);
    }
  }

  for (const attempt of annotationAttempts()) {
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
      const timestamp = snapTimelineBoundaryTimestamp(
        timelineAtRatio(ratio),
        state.dragging.left.start_at + minimum,
        state.dragging.right.end_at - minimum,
      );
      if (timestamp === null) return;
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
  renderDetailAnnotation();
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
    clearRecoverableFailureCapture({clearNote: true});
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
    updateTimelinePointCaptureState();
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
  clearRecoverableFailureCapture({clearNote: true});
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

function locateRecoverableFailure(event, timestamp = event.onset_at) {
  if (!requireTimelinePointCaptureReady()) return;
  seekTimeline(timestamp);
  showToast(
    `已定位到可恢复失误的样本 ${nearestSampleIndex(timestamp)}。`,
    "info",
    2600,
  );
}

function beginRecoverableFailureReannotation(event) {
  if (state.busy || !requireTimelinePointCaptureReady()) return;
  state.recoverableFailureEditingId = event.annotation_id;
  state.recoverableFailureStart = null;
  state.recoverableFailureStartOccurrenceId = null;
  $("recoverable-failure-note").value = event.note || "";
  seekTimeline(event.onset_at);
  renderRecoverableFailureTools();
  showToast(
    "原区间会保留到新起止点通过校验；请先定位并重新记录失误发生样本。",
    "info",
    6000,
  );
}

function cancelRecoverableFailureCapture() {
  const wasEditing = Boolean(state.recoverableFailureEditingId);
  clearRecoverableFailureCapture({clearNote: true});
  renderRecoverableFailureTools();
  showToast(
    wasEditing ? "已取消重标，原可恢复失误区间保持不变。" : "已取消本次取点。",
    "info",
    3200,
  );
}

function renderRecoverableFailureTools() {
  const root = $("recoverable-failure-list");
  root.replaceChildren();
  const ambiguityIds = recoverableFailureAmbiguityIds();
  for (const event of recoverableFailures()) {
    const stage = stageById(event.stage_id);
    const row = element(
      "div",
      `compact-item${
        event.annotation_id === state.recoverableFailureEditingId
          ? " recoverable-editing"
          : ""
      }${ambiguityIds.has(event.annotation_id) ? " recoverable-ambiguous" : ""}`,
    );
    row.appendChild(element(
      "span",
      "compact-item-label",
      `${stage?.display_name || event.stage_id} · `
      + `样本 ${event.onset_sample_index}–${event.recovered_sample_index} · `
      + `${formatTime(event.onset_at - state.detail.start_timestamp)} – `
      + `${formatTime(event.recovered_at - state.detail.start_timestamp)}`
      + `${event.note ? ` · ${event.note}` : ""}`
      + `${ambiguityIds.has(event.annotation_id) ? " · 需重标" : ""}`,
    ));
    const actions = element("span", "compact-item-actions");
    const locateStart = element("button", "secondary compact", "定位起点");
    locateStart.type = "button";
    locateStart.addEventListener(
      "click",
      () => locateRecoverableFailure(event, event.onset_at),
    );
    const locateEnd = element("button", "secondary compact", "定位恢复");
    locateEnd.type = "button";
    locateEnd.addEventListener(
      "click",
      () => locateRecoverableFailure(event, event.recovered_at),
    );
    const reannotate = element("button", "secondary compact", "重标");
    reannotate.type = "button";
    reannotate.addEventListener(
      "click",
      () => beginRecoverableFailureReannotation(event),
    );
    const remove = element("button", "secondary compact", "移除");
    remove.type = "button";
    remove.addEventListener("click", () => {
      if (state.busy) return;
      if (state.recoverableFailureEditingId === event.annotation_id) {
        clearRecoverableFailureCapture({clearNote: true});
      }
      state.draft.recoverable_failures = (
        state.draft.recoverable_failures || []
      ).filter((value) => value.annotation_id !== event.annotation_id);
      renderTimeline();
      renderRecoverableFailureTools();
      updateEditButtonState();
    });
    actions.append(locateStart, locateEnd, reannotate, remove);
    row.appendChild(actions);
    root.appendChild(row);
  }
  const editing = recoverableFailures().find(
    (event) => event.annotation_id === state.recoverableFailureEditingId,
  );
  if (state.recoverableFailureEditingId && !editing) {
    clearRecoverableFailureCapture({clearNote: true});
  }
  const startOccurrence = sortedOccurrences().find(
    (value) => (
      value.annotation_id === state.recoverableFailureStartOccurrenceId
    ),
  );
  const startStage = startOccurrence
    ? stageById(startOccurrence.stage_id)
    : null;
  const pending = $("recoverable-failure-pending");
  if (state.recoverableFailureStart !== null) {
    pending.textContent = `${editing ? "重标" : "新增"} · `
      + `失误样本 ${nearestSampleIndex(state.recoverableFailureStart)} · `
      + `${startStage?.display_name || startOccurrence?.stage_id || "未知阶段"} · `
      + formatTime(
        state.recoverableFailureStart - state.detail.start_timestamp,
      );
  } else if (editing) {
    pending.textContent = "正在重标；原区间仍保留，请先取新的失误发生样本。";
  } else if (ambiguityIds.size) {
    pending.textContent = "检测到历史区间重叠或共用边界样本；请使用“重标”逐项消除歧义后再封存。";
  } else {
    pending.textContent = "尚未取点";
  }
  $("mark-recoverable-start").textContent = state.recoverableFailureStart === null
    ? editing ? "重标：取失误发生" : "取当前为失误发生"
    : "重新取失误发生";
  $("mark-recoverable-end").textContent = editing
    ? "取恢复正常并替换"
    : "取当前为恢复正常并添加";
  $("cancel-recoverable-capture").classList.toggle(
    "hidden",
    !editing && state.recoverableFailureStart === null,
  );
  updateTimelinePointCaptureState();
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
        updateTimelinePointCaptureState();
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
  const sealedPending = Boolean(
    state.detail?.review_state === "PENDING"
    && state.detail?.annotation_mode !== "SECOND_REVIEW"
  );
  const card = $("toggle-review-request").closest(".review-request-card");
  card.classList.toggle("pending", Boolean(request));
  $("review-request-note").value = request?.note || "";
  const toggle = $("toggle-review-request");
  toggle.textContent = request
    ? "取消二次复核标签"
    : "标记需要二次复核";
  toggle.disabled = sealedPending && Boolean(request);
  toggle.title = toggle.disabled
    ? "该标签来自已封存版本，只能由二次复核封存解除"
    : "";
  $("review-request-status").textContent = sealedPending
    ? request
      ? "该待复核标签来自已封存版本；普通标注不能解除，需进入二次复核模式并封存。"
      : "已封存版本仍要求二次复核；当前草稿缺少标签，普通模式封存将被拒绝。"
    : request
    ? "当前编辑封存后将整条排除；二次复核模式重新封存后恢复正常。"
    : state.detail?.annotation_mode === "SECOND_REVIEW"
      ? "当前是二次复核编辑；封存后生成 SECOND_REVIEW 版本。"
      : "标记后整条 Episode 默认不进入训练，直到二次复核版本封存。";
}

async function publishDetailAnnotations() {
  const meta = detailAnnotationMeta();
  const overlay = detailOverlayDraft();
  if (!state.draft || !state.detail || !meta?.available || !overlay) {
    throw new Error("当前 Episode 没有可封存的详细补充。");
  }
  if (!overlay.confirmed) {
    throw new Error("请先确认详细时间轴，并为每个原始 Failure 选择失败子阶段。");
  }
  if (!meta.base) {
    const payload = await requestJsonWithProgress(
      "/api/draft/" + encodeURIComponent(state.detail.episode_uuid),
      {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(state.draft),
      },
      {
        kind: "SAVE_DETAIL_DRAFT",
        message: "正在保存详细草稿",
        completeMessage: "详细草稿已保存",
      },
    );
    state.draft = payload.draft;
    const pendingMeta = {
      ...meta,
      draft_state: "EDITING_UNSEALED",
      reason: (
        "详细草稿已保存；粗标注可稍后独立封存，"
        + "之后再发布 detail overlay。"
      ),
    };
    state.detail.detail_annotation = pendingMeta;
    state.detail.detail_annotation_complete = false;
    state.detail.detail_annotation_state = "PENDING";
    setSavedDraftSnapshot(state.draft);
    renderTimeline();
    renderEpisodeList({preserveViewport: true});
    await loadEpisodes();
    return {
      pending_coarse: true,
      draft: state.draft,
      detail_annotation: pendingMeta,
      draft_cache_warning: null,
    };
  }
  const payload = await requestJsonWithProgress(
    "/api/commit-detail/" + encodeURIComponent(state.detail.episode_uuid),
    {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state.draft),
    },
    {
      kind: "COMMIT_DETAIL_ANNOTATION",
      message: "正在校验并封存详细补充",
      completeMessage: "详细补充已封存",
    },
  );
  const report = payload.report;
  state.draft = report.draft;
  if (report.detail_annotation) {
    state.detail.detail_annotation = report.detail_annotation;
  }
  state.detail.detail_annotation_complete =
    report.detail_annotation?.resolution_level === "DETAIL_COMPLETE";
  state.detail.detail_annotation_state = state.detail.detail_annotation_complete
    ? "COMPLETE"
    : report.detail_annotation?.available
      ? "PENDING"
      : "NOT_APPLICABLE";
  const statusRefreshWarning = applyCommittedEpisodeSummary(payload);
  setSavedDraftSnapshot(state.draft);
  renderTimeline();
  renderEpisodeList({preserveViewport: true});
  await loadEpisodes({afterCommit: true});
  // A changed catalog revision may replace the hydrated row with a lightweight
  // one. Restore the summary returned by this successful commit afterward.
  applyCommittedEpisodeSummary(payload, {markCommit: false});
  renderEpisodeList({preserveViewport: true});
  return {...report, status_refresh_warning: statusRefreshWarning};
}

async function commitDetailAnnotations() {
  if (!state.draft || state.busy) return;
  state.busy = true;
  try {
    const report = await publishDetailAnnotations();
    const warnings = commitWarnings(report);
    showToast(
      report.pending_coarse
        ? "详细草稿已保存；粗标注仍可独立编辑，稍后封存粗标注后再发布 detail overlay。"
        : warnings.length
        ? `详细补充已封存，但${warnings.join("；")}`
        : "详细补充校验通过，已发布不可变 detail overlay。",
      warnings.length ? "error" : "info",
      warnings.length ? 9000 : 5000,
    );
  } catch (error) {
    showToast("详细补充封存失败：" + error.message, "error", 9000);
  } finally {
    state.busy = false;
    renderDetailAnnotation();
    updateTimelinePointCaptureState();
  }
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
  if (report.detail_annotation) {
    state.detail.detail_annotation = report.detail_annotation;
  }
  state.detail.detail_annotation_complete =
    report.detail_annotation?.resolution_level === "DETAIL_COMPLETE";
  state.detail.detail_annotation_state = state.detail.detail_annotation_complete
    ? "COMPLETE"
    : report.detail_annotation?.available
      ? "PENDING"
      : "NOT_APPLICABLE";
  state.volatilePostSealDraft = Boolean(
    report.draft_cache_warning && state.draft?.detail_overlay,
  );
  if (state.volatilePostSealDraft) {
    const durableBaseline = structuredClone(state.draft);
    durableBaseline.detail_overlay = null;
    setSavedDraftSnapshot(durableBaseline);
  } else {
    setSavedDraftSnapshot(state.draft);
  }
  state.sealedReeditBaseline = null;
  state.sealedReeditSourceId = null;
  renderTimeline();
  renderCaptureEndStepEditor();
  renderRecoverableFailureTools();
  renderReviewControls();
  renderSealedReeditControls();
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
  state.detail.draft_persistence_state = state.volatilePostSealDraft
    ? "VOLATILE_POST_SEAL_DETAIL_DRAFT"
    : "SEALED_SNAPSHOT";
  state.detail.draft_requires_seal = false;
  const statusRefreshWarning = applyCommittedEpisodeSummary(payload);
  renderEpisodeList();
  renderSealedReeditControls();
  $("coverage-preview").textContent =
    `已封存 ${report.annotation_set_id} · coverage=${report.coverage_status} · 排除区间 ${report.exclusion_count}`;
  await loadEpisodes({afterCommit: true});
  applyCommittedEpisodeSummary(payload, {markCommit: false});
  renderEpisodeList({preserveViewport: true});
  return {...report, status_refresh_warning: statusRefreshWarning};
}

async function persistVolatilePostSealDraft() {
  if (!state.volatilePostSealDraft || !state.detail || !state.draft) return null;
  const payload = await requestJsonWithProgress(
    "/api/draft/" + encodeURIComponent(state.detail.episode_uuid),
    {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state.draft),
    },
    {
      kind: "RETRY_POST_SEAL_DETAIL_DRAFT",
      message: "正在重试保存粗标注后的详细工作草稿",
      completeMessage: "详细工作草稿已持久化",
    },
  );
  state.draft = payload.draft;
  state.volatilePostSealDraft = false;
  state.detail.draft_persistence_state = "UNSEALED_DRAFT";
  state.detail.draft_requires_seal = true;
  setSavedDraftSnapshot(state.draft);
  renderTimeline();
  renderSealedReeditControls();
  return payload;
}

async function commitAnnotations() {
  if (!state.draft || state.busy) return;
  state.busy = true;
  try {
    if (state.volatilePostSealDraft) {
      await persistVolatilePostSealDraft();
      showToast(
        "粗标注已封存；详细工作草稿现已单独持久化，未重复发布粗标注。",
        "info",
        6000,
      );
      return;
    }
    const report = await publishCurrentAnnotations();
    const warnings = commitWarnings(report);
    showToast(
      warnings.length
        ? `不可变标注已发布，但${warnings.join("；")}`
        : "标注校验通过，已发布不可变 annotation set。",
      warnings.length ? "error" : "info",
      warnings.length ? 9000 : 5000,
    );
  } catch (error) {
    showToast(`封存失败：${error.message}`, "error", 9000);
  } finally {
    state.busy = false;
    renderDetailAnnotation();
    updateTimelinePointCaptureState();
  }
}

$("refresh-episodes").addEventListener(
  "click",
  () => loadEpisodes({force: true}),
);
window.addEventListener("beforeunload", (event) => {
  if (!hasUnsavedDraftChanges() && !state.volatilePostSealDraft) return;
  event.preventDefault();
  event.returnValue = "";
});
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
$("launcher-default-camera").addEventListener("change", (event) => {
  state.defaultCameraPreference = event.target.value;
  window.localStorage.setItem(
    DEFAULT_CAMERA_STORAGE_KEY,
    state.defaultCameraPreference,
  );
});
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
$("confirm-detail-annotation").addEventListener(
  "click",
  confirmDetailAnnotation,
);
$("detail-profile-select").addEventListener(
  "change",
  changeDetailProfileSelection,
);
$("publish-detail-annotation").addEventListener(
  "click",
  commitDetailAnnotations,
);
$("reopen-sealed").addEventListener("click", reopenSealedAnnotation);
$("cancel-reopen-sealed").addEventListener(
  "click",
  cancelReopenSealedAnnotation,
);

$("mark-recoverable-start").addEventListener("click", () => {
  if (!state.draft || state.busy) return;
  if (!requireTimelinePointCaptureReady()) return;
  const point = currentTimelineSamplePoint();
  const occurrence = point
    ? successOccurrenceOwningSample(point.timestamp)
    : null;
  if (!point || !occurrence) {
    showToast(
      "失误发生样本必须位于一个 SUCCESS 子阶段内；请移动到紫色轨迹实际发生失误的位置。",
      "warning",
      6500,
    );
    return;
  }
  state.recoverableFailureStart = point.timestamp;
  state.recoverableFailureStartOccurrenceId = occurrence.annotation_id;
  seekTimeline(point.timestamp);
  renderRecoverableFailureTools();
  const stage = stageById(occurrence.stage_id);
  showToast(
    `已吸附到样本 ${point.sampleIndex}，属于“${
      stage?.display_name || occurrence.stage_id
    }”；请定位到恢复正常样本。`,
    "info",
    5200,
  );
});
$("mark-recoverable-end").addEventListener("click", async () => {
  if (!state.draft || !state.detail || state.busy) return;
  if (!requireTimelinePointCaptureReady()) return;
  const recoveredPoint = currentTimelineSamplePoint();
  if (!recoveredPoint) return;
  if (state.recoverableFailureStart === null) {
    showToast("请先记录失误发生样本。", "warning");
    return;
  }
  const onsetSampleIndex = nearestSampleIndex(
    state.recoverableFailureStart,
  );
  if (recoveredPoint.sampleIndex <= onsetSampleIndex) {
    showToast("恢复正常样本必须晚于失误发生样本。", "warning");
    return;
  }
  const startOccurrence = successOccurrenceOwningSample(
    state.recoverableFailureStart,
  );
  const recoveredOccurrence = successOccurrenceOwningSample(
    recoveredPoint.timestamp,
  );
  if (
    !startOccurrence
    || startOccurrence.annotation_id
      !== state.recoverableFailureStartOccurrenceId
  ) {
    state.recoverableFailureStart = null;
    state.recoverableFailureStartOccurrenceId = null;
    renderRecoverableFailureTools();
    showToast(
      "取点后阶段边界已经变化，请重新记录失误发生样本。原重标区间仍保持不变。",
      "warning",
      7000,
    );
    return;
  }
  if (
    !recoveredOccurrence
    || recoveredOccurrence.annotation_id !== startOccurrence.annotation_id
  ) {
    showToast(
      "失误发生和恢复正常必须位于同一个 SUCCESS 子阶段；不能跨越白色阶段边界。",
      "warning",
      7000,
    );
    return;
  }
  const conflict = recoverableFailureConflict(
    onsetSampleIndex,
    recoveredPoint.sampleIndex,
    {excludingAnnotationId: state.recoverableFailureEditingId},
  );
  if (conflict) {
    showToast(
      "该区间与已有可恢复失误重叠或共用边界样本；请保留至少一个明确正常样本。",
      "warning",
      7000,
    );
    return;
  }
  const editingId = state.recoverableFailureEditingId;
  const requestDraft = editingId
    ? {
        ...structuredClone(state.draft),
        recoverable_failures: state.draft.recoverable_failures.filter(
          (event) => event.annotation_id !== editingId,
        ),
      }
    : state.draft;
  state.busy = true;
  updateTimelinePointCaptureState();
  try {
    const payload = await requestJson(
      `/api/recoverable-failure/${encodeURIComponent(state.detail.episode_uuid)}`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          draft: requestDraft,
          onset_at: state.recoverableFailureStart,
          recovered_at: recoveredPoint.timestamp,
          note: $("recoverable-failure-note").value.trim() || null,
        }),
      },
    );
    state.draft = payload.draft;
    clearRecoverableFailureCapture({clearNote: true});
    renderTimeline();
    renderRecoverableFailureTools();
    updateEditButtonState();
    const added = payload.recoverable_failure;
    const stage = added ? stageById(added.stage_id) : null;
    showToast(
      `可恢复失误已${editingId ? "安全替换并" : ""}绑定到“${
        stage?.display_name || added?.stage_id || "当前子阶段"
      }”的样本 ${added?.onset_sample_index}–${
        added?.recovered_sample_index
      }；随下次封存保存。`,
      "info",
      6500,
    );
  } catch (error) {
    showToast(
      `无法${editingId ? "重标" : "添加"}可恢复失误：${error.message}`,
      "error",
      8500,
    );
  } finally {
    state.busy = false;
    updateTimelinePointCaptureState();
  }
});
$("cancel-recoverable-capture").addEventListener(
  "click",
  cancelRecoverableFailureCapture,
);

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
        updateTimelinePointCaptureState();
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
      updateTimelinePointCaptureState();
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
  if (!requireTimelinePointCaptureReady()) return;
  state.abnormalStart = currentTimelineTime();
  renderPrivilegedTools();
});
$("mark-abnormal-end").addEventListener("click", () => {
  if (!state.draft?.privileged_tools_enabled) {
    showToast("特殊修正工具已关闭，不能添加异常区间。", "warning");
    state.abnormalStart = null;
    return;
  }
  if (!requireTimelinePointCaptureReady()) return;
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
