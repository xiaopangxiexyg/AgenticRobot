const text = (id, value) => {
  document.getElementById(id).textContent = value == null || value === "" ? "—" : String(value);
};

const diagnosticApi = window.RuleWritingDiagnostics;
const selectionApi = window.RuleWritingSelection;

function diagnosticBadge(label, kind) {
  const badge = document.createElement("span");
  badge.className = `diagnostic-badge diagnostic-badge-${kind}`;
  badge.textContent = label;
  return badge;
}

function renderActiveDiagnostics(state) {
  const diagnostics = diagnosticApi.buildActiveDiagnostics(state);
  const container = document.getElementById("active-diagnostics");
  const empty = document.getElementById("diagnostics-empty");
  empty.hidden = diagnostics.length !== 0;
  text(
    "diagnostics-summary",
    diagnostics.length
      ? `${diagnostics.length} 项活动问题 · 优先处理 ${diagnostics[0].severityLabel}`
      : "0 项活动问题",
  );
  container.replaceChildren(...diagnostics.map((diagnostic) => {
    const card = document.createElement("article");
    card.className = `diagnostic-card severity-${diagnostic.severity}`;

    const badges = document.createElement("div");
    badges.className = "diagnostic-badges";
    badges.append(
      diagnosticBadge(diagnostic.severityLabel, `severity-${diagnostic.severity}`),
      diagnosticBadge(diagnostic.domainLabel, "domain"),
      diagnosticBadge(diagnostic.componentLabel, "component"),
    );

    const heading = document.createElement("h3");
    heading.textContent = `${diagnostic.domainLabel} · ${diagnostic.componentLabel}`;
    const summary = document.createElement("p");
    summary.className = "diagnostic-summary";
    summary.textContent = diagnostic.summary;
    const action = document.createElement("p");
    action.className = "diagnostic-action";
    const actionLabel = document.createElement("strong");
    actionLabel.textContent = `${diagnostic.recovery.label}：`;
    action.append(actionLabel, document.createTextNode(diagnostic.recovery.instruction));
    const evidence = document.createElement("details");
    const evidenceTitle = document.createElement("summary");
    evidenceTitle.textContent = "查看原始证据";
    const evidenceBody = document.createElement("pre");
    evidenceBody.textContent = diagnostic.evidence;
    evidence.append(evidenceTitle, evidenceBody);
    card.append(badges, heading, summary, action, evidence);
    return card;
  }));
}

function renderMessageHistory(messages) {
  const list = document.getElementById("messages");
  list.replaceChildren(...(messages || []).slice(-12).map((message) => {
    const diagnostic = diagnosticApi.classifyDiagnostic({
      tag: message.tag,
      message: message.message,
      source: "message",
    });
    const item = document.createElement("li");
    item.className = `message-entry severity-${diagnostic.severity}`;
    const heading = document.createElement("div");
    heading.className = "message-heading";
    heading.append(
      diagnosticBadge(diagnostic.severityLabel, `severity-${diagnostic.severity}`),
      diagnosticBadge(diagnostic.componentLabel, "component"),
    );
    const tag = document.createElement("strong");
    tag.textContent = message.tag;
    heading.append(tag);
    const body = document.createElement("p");
    body.textContent = message.message;
    item.append(heading, body);
    return item;
  }));
}

const pendingRequests = new Set();

async function postJson(path, payload, key) {
  if (pendingRequests.has(key)) return false;
  pendingRequests.add(key);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) await refresh();
    return response.ok;
  } catch (_error) {
    // A disconnected browser must not create an unhandled rejection or leave
    // a button looking busy forever.  The backend remains authoritative.
    await refresh();
    return false;
  } finally {
    window.clearTimeout(timeout);
    pendingRequests.delete(key);
    await refresh();
  }
}

async function sendEvent(event) {
  await postJson("/event", {event}, `event:${event}`);
}

async function sendHeightCalibration(operation) {
  const payload = {operation};
  if (operation === "capture") {
    payload.lower_tolerance_mm = Number(document.getElementById("height-lower-tolerance").value);
    payload.upper_tolerance_mm = Number(document.getElementById("height-upper-tolerance").value);
  }
  await postJson("/height-calibration", payload, `height-calibration:${operation}`);
}

async function advanceTaskChain() {
  await postJson("/task-chain", {
      event: "advance",
      lower_tolerance_mm: Number(document.getElementById("height-lower-tolerance").value),
      upper_tolerance_mm: Number(document.getElementById("height-upper-tolerance").value),
    }, "task-chain");
}

let optionsLoaded = false;
let trajectoryChoices = [];
let latestState = null;
let configurationDirty = false;
let configurationBatch = [];
let collectionMode = "rule_based_writing";
let tactileCaptureLocked = false;
let writingSpeedEdited = false;
let refreshInFlight = false;
let optionsRefreshInFlight = false;
let lastOptionsRefreshAt = 0;

const configurationInputs = () => [
  "full-replay-source", "pre-pen-down-source", "pre-pen-down-normal-episodes", "pen-down-source", "write-source", "pen-up-source", "trajectory",
  "direction", "writing-size", "writing-speed", "alignment-speed-input",
  "collect-tactile", "tactile-bundle-root", "tactile-min-free", "tactile-compression",
  "writer-workers", "writer-queue-size", "process-queue-size",
  "storage-queue-size", "storage-queue-enqueue-timeout", "camera-frame-max-age", "collection-warning-hz", "write-parquet", "write-video",
  "collection-flush-timeout", "sample-drain-timeout", "terminal-event-timeout",
].map((id) => document.getElementById(id));

const fullReplayMode = () => collectionMode === "full_trajectory_replay";

function applyModeUi() {
  const fullReplay = fullReplayMode();
  document.querySelectorAll(".writing-option").forEach((element) => {
    element.hidden = fullReplay;
  });
  document.getElementById("full-replay-source-field").hidden = !fullReplay;
  document.getElementById("height-calibration-panel").hidden = fullReplay;
  text("collection-mode", fullReplay ? "完整数据集轨迹 Replay" : "规则式书写");
  text("mode-eyebrow", fullReplay
    ? "DEXTELEOP · FULL TRAJECTORY REPLAY"
    : "DEXTELEOP · RULE-BASED WRITING");
  text("mode-heading", fullReplay ? "完整轨迹 Replay 采集" : "规则采集");
  text("configuration-title", fullReplay ? "完整轨迹来源" : "本次书写组合");
}

const addOptions = (select, values) => {
  select.replaceChildren(...(values || []).map((entry) => {
    const option = document.createElement("option");
    option.value = String(entry && entry.value != null ? entry.value : entry);
    option.textContent = String(entry && entry.label != null ? entry.label : option.value);
    return option;
  }));
};

async function loadOptions() {
  if (optionsRefreshInFlight) return;
  optionsRefreshInFlight = true;
  try {
  let response;
  try {
    response = await fetch("/options.json", {cache: "no-store"});
  } catch (_error) {
    optionsRefreshInFlight = false;
    return;
  }
  if (!response.ok) {
    optionsRefreshInFlight = false;
    return;
  }
  let options;
  try {
    options = await response.json();
  } catch (_error) {
    optionsRefreshInFlight = false;
    return;
  }
  collectionMode = options.collection_mode || "rule_based_writing";
  const sources = options.source_candidates_by_phase || {};
  if (fullReplayMode()) {
    if (!(options.full_replay_candidates || []).length) {
      optionsRefreshInFlight = false;
      return;
    }
    addOptions(
      document.getElementById("full-replay-source"),
      options.full_replay_candidates,
    );
  } else {
    if (!(sources.pen_down || []).length || !(sources.write || []).length ||
        !(sources.pen_up || []).length || !(options.trajectory_ids || []).length ||
        !(options.directions || []).length) {
      optionsRefreshInFlight = false;
      return;
    }
    addOptions(
      document.getElementById("pre-pen-down-source"),
      options.pre_pen_down_replay_candidates || [],
    );
    addOptions(document.getElementById("pen-down-source"), sources.pen_down);
    addOptions(document.getElementById("write-source"), sources.write);
    addOptions(document.getElementById("pen-up-source"), sources.pen_up);
    addOptions(document.getElementById("trajectory"), options.trajectory_ids);
    trajectoryChoices = options.trajectory_ids;
    addOptions(document.getElementById("direction"), options.directions);
  }
  applyModeUi();
  addOptions(
    document.getElementById("tactile-compression"),
    (options.select_fields || {}).tactile_compression || [],
  );
  const bounds = options.numeric_bounds || {};
  const sizeBounds = bounds.writing_size_m || [0.005, 0.2];
  const preludeIntervalBounds = bounds.pre_pen_down_replay_normal_episodes || [0, 10000];
  const speedBounds = bounds.initial_alignment_speed_scale || [0.05, 1.0];
  const writingSpeedBounds = bounds.writing_speed_m_s || [0.001, 0.2];
  const size = document.getElementById("writing-size");
  size.min = sizeBounds[0];
  size.max = sizeBounds[1];
  const preludeInterval = document.getElementById("pre-pen-down-normal-episodes");
  preludeInterval.min = preludeIntervalBounds[0];
  preludeInterval.max = preludeIntervalBounds[1];
  const speed = document.getElementById("alignment-speed-input");
  speed.min = speedBounds[0];
  speed.max = speedBounds[1];
  const writingSpeed = document.getElementById("writing-speed");
  writingSpeed.min = writingSpeedBounds[0];
  writingSpeed.max = writingSpeedBounds[1];
  const numericInputs = {
    "tactile-min-free": "tactile_bundle_min_free_gb",
    "writer-workers": "writer_workers",
    "writer-queue-size": "writer_queue_size",
    "process-queue-size": "collection_process_queue_size",
    "storage-queue-size": "storage_queue_size",
    "storage-queue-enqueue-timeout": "storage_queue_enqueue_timeout_sec",
    "camera-frame-max-age": "camera_frame_max_age_sec",
    "collection-warning-hz": "collection_warning_hz",
    "collection-flush-timeout": "collection_flush_timeout_sec",
    "sample-drain-timeout": "collection_sample_drain_timeout_sec",
    "terminal-event-timeout": "collection_terminal_event_put_timeout_sec",
  };
  Object.entries(numericInputs).forEach(([id, key]) => {
    if (!bounds[key]) return;
    const input = document.getElementById(id);
    input.min = bounds[key][0];
    input.max = bounds[key][1];
  });
  const info = options.runtime_info || {};
  const free = info.tactile_bundle_free_gb;
  const collectTactile = document.getElementById("collect-tactile");
  tactileCaptureLocked = Boolean(info.tactile_capture_locked);
  collectTactile.disabled = tactileCaptureLocked;
  collectTactile.title = tactileCaptureLocked
    ? `当前数据集已锁定触觉模式：${info.tactile_capture_mode || "existing"}`
    : "空数据集可选择是否采集触觉";
  text("sample-hz", info.sample_hz == null ? null : `${Number(info.sample_hz).toFixed(1)} Hz`);
  text(
    "position-reference-hz",
    info.position_reference_hz == null ? null : `${Number(info.position_reference_hz).toFixed(1)} Hz`,
  );
  text("arm-control-hz", info.arm_control_hz == null ? null : `${Number(info.arm_control_hz).toFixed(1)} Hz`);
  text(
    "storage-status",
    `正式数据集：${info.dataset_root || "—"}；触觉模式：${info.tactile_capture_mode || "—"}`
      + `；触觉盘当前可用：${free == null ? "未知" : `${Number(free).toFixed(1)} GiB`}`,
  );
  optionsLoaded = true;
  if (latestState) syncConfiguration(latestState);
  optionsRefreshInFlight = false;
  } finally {
    optionsRefreshInFlight = false;
  }
}

function syncConfiguration(state) {
  if (!optionsLoaded || configurationDirty || document.activeElement && configurationInputs().includes(document.activeElement)) return;
  const selected = state.rule_configuration || {};
  document.getElementById("full-replay-source").value = selected.full_replay_candidate_id || "";
  document.getElementById("pre-pen-down-source").value = selected.pre_pen_down_replay_candidate_id || "";
  if (selected.pre_pen_down_replay_normal_episodes != null) document.getElementById("pre-pen-down-normal-episodes").value = selected.pre_pen_down_replay_normal_episodes;
  document.getElementById("pen-down-source").value = selected.pen_down_candidate_id || "";
  document.getElementById("write-source").value = selected.write_candidate_id || "";
  document.getElementById("pen-up-source").value = selected.pen_up_candidate_id || "";
  document.getElementById("trajectory").value = selected.trajectory_id || "";
  document.getElementById("direction").value = selected.direction || "";
  if (selected.writing_size_m != null) document.getElementById("writing-size").value = selected.writing_size_m;
  if (selected.writing_speed_m_s != null) document.getElementById("writing-speed").value = selected.writing_speed_m_s;
  if (selected.initial_alignment_speed_scale != null) document.getElementById("alignment-speed-input").value = selected.initial_alignment_speed_scale;
  document.getElementById("collect-tactile").checked = Boolean(selected.collect_tactile);
  document.getElementById("tactile-bundle-root").value = selected.tactile_bundle_root || "";
  if (selected.tactile_bundle_min_free_gb != null) document.getElementById("tactile-min-free").value = selected.tactile_bundle_min_free_gb;
  if (selected.tactile_compression != null) document.getElementById("tactile-compression").value = selected.tactile_compression;
  if (selected.writer_workers != null) document.getElementById("writer-workers").value = selected.writer_workers;
  if (selected.writer_queue_size != null) document.getElementById("writer-queue-size").value = selected.writer_queue_size;
  if (selected.collection_process_queue_size != null) document.getElementById("process-queue-size").value = selected.collection_process_queue_size;
  if (selected.storage_queue_size != null) document.getElementById("storage-queue-size").value = selected.storage_queue_size;
  if (selected.storage_queue_enqueue_timeout_sec != null) document.getElementById("storage-queue-enqueue-timeout").value = selected.storage_queue_enqueue_timeout_sec;
  if (selected.camera_frame_max_age_sec != null) document.getElementById("camera-frame-max-age").value = selected.camera_frame_max_age_sec;
  if (selected.collection_warning_hz != null) document.getElementById("collection-warning-hz").value = selected.collection_warning_hz;
  if (selected.collection_flush_timeout_sec != null) document.getElementById("collection-flush-timeout").value = selected.collection_flush_timeout_sec;
  if (selected.collection_sample_drain_timeout_sec != null) document.getElementById("sample-drain-timeout").value = selected.collection_sample_drain_timeout_sec;
  if (selected.collection_terminal_event_put_timeout_sec != null) document.getElementById("terminal-event-timeout").value = selected.collection_terminal_event_put_timeout_sec;
  document.getElementById("write-parquet").checked = Boolean(selected.write_parquet);
  document.getElementById("write-video").checked = Boolean(selected.write_video);
}

function currentConfiguration() {
  const shared = {
    initial_alignment_speed_scale: Number(document.getElementById("alignment-speed-input").value),
    collect_tactile: document.getElementById("collect-tactile").checked,
    tactile_bundle_root: document.getElementById("tactile-bundle-root").value.trim(),
    tactile_bundle_min_free_gb: Number(document.getElementById("tactile-min-free").value),
    tactile_compression: document.getElementById("tactile-compression").value,
    writer_workers: Number(document.getElementById("writer-workers").value),
    writer_queue_size: Number(document.getElementById("writer-queue-size").value),
    collection_process_queue_size: Number(document.getElementById("process-queue-size").value),
    storage_queue_size: Number(document.getElementById("storage-queue-size").value),
    storage_queue_enqueue_timeout_sec: Number(document.getElementById("storage-queue-enqueue-timeout").value),
    camera_frame_max_age_sec: Number(document.getElementById("camera-frame-max-age").value),
    collection_warning_hz: Number(document.getElementById("collection-warning-hz").value),
    collection_flush_timeout_sec: Number(document.getElementById("collection-flush-timeout").value),
    collection_sample_drain_timeout_sec: Number(document.getElementById("sample-drain-timeout").value),
    collection_terminal_event_put_timeout_sec: Number(document.getElementById("terminal-event-timeout").value),
    write_parquet: document.getElementById("write-parquet").checked,
    write_video: document.getElementById("write-video").checked,
  };
  if (fullReplayMode()) {
    return {
      full_replay_candidate_id: document.getElementById("full-replay-source").value,
      ...shared,
    };
  }
  return {
    pre_pen_down_replay_candidate_id: document.getElementById("pre-pen-down-source").value,
    pre_pen_down_replay_normal_episodes: Number(document.getElementById("pre-pen-down-normal-episodes").value),
    pen_down_candidate_id: document.getElementById("pen-down-source").value,
    write_candidate_id: document.getElementById("write-source").value,
    pen_up_candidate_id: document.getElementById("pen-up-source").value,
    trajectory_id: document.getElementById("trajectory").value,
    direction: document.getElementById("direction").value,
    writing_size_m: Number(document.getElementById("writing-size").value),
    writing_speed_m_s: Number(document.getElementById("writing-speed").value),
    writing_speed_from_source: !writingSpeedEdited
      && Boolean(latestState && latestState.rule_configuration
        && latestState.rule_configuration.writing_speed_from_source),
    ...shared,
  };
}

const sourceLabel = (value) =>
  value === "__rule_source_pool_random__" ? "随机" : (value || "—");

const batchLabel = (entry, index) =>
  fullReplayMode()
    ? `${index + 1}. 完整 Episode · ${entry.full_replay_candidate_id}`
    : `${index + 1}. ${entry.trajectory_id} · 额外完整轨迹:${entry.pre_pen_down_replay_candidate_id || "关闭"} · 间隔:${entry.pre_pen_down_replay_normal_episodes || 0} · 靠近:${sourceLabel(entry.pen_down_candidate_id)} / 书写:${sourceLabel(entry.write_candidate_id)} / 离开:${sourceLabel(entry.pen_up_candidate_id)}`;

function renderBatch() {
  const validated = latestState && Array.isArray(latestState.rule_configuration_batch)
    ? latestState.rule_configuration_batch
    : [];
  const entries = configurationDirty ? configurationBatch : validated;
  const list = document.getElementById("batch-items");
  list.replaceChildren(...entries.map((entry, index) => {
    const item = document.createElement("li");
    item.textContent = batchLabel(entry, index);
    return item;
  }));
}

function addCurrentToBatch() {
  configurationBatch.push(currentConfiguration());
  configurationDirty = true;
  renderBatch();
}

function clearBatch() {
  configurationBatch = [];
  configurationDirty = true;
  renderBatch();
}

async function applyConfiguration() {
  const current = currentConfiguration();
  const payload = configurationBatch.length
    ? {batch: configurationBatch}
    : current;
  if (await postJson("/configuration", payload, "configuration")) {
    configurationDirty = false;
    writingSpeedEdited = false;
    await refresh();
  }
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
  let response;
  try {
    response = await fetch("/state.json", {cache: "no-store"});
  } catch (_error) {
    return;
  }
  if (!response.ok) {
    refreshInFlight = false;
    return;
  }
  let state;
  try {
    state = await response.json();
  } catch (_error) {
    refreshInFlight = false;
    return;
  }
  latestState = state;
  collectionMode = state.collection_mode || collectionMode;
  applyModeUi();
  text("phase", state.phase);
  text("title", state.title);
  text("detail", state.detail);
  text("step", state.step_id);
  text("kind", state.directive_kind);
  text("sequence", state.sequence_index);
  text("writer", state.recording_ready ? "就绪" : "等待");
  text("topology", state.control_topology);
  text("stage", state.stage_label);
  text("position", state.collection_position);
  const recorded = Number.isFinite(Number(state.recorded_sample_count)) ? state.recorded_sample_count : 0;
  const planned = Number.isFinite(Number(state.planned_sample_count)) ? state.planned_sample_count : 0;
  text("samples", `${recorded} / ${planned}`);
  const batchCount = Math.max(1, Number(state.batch_count) || 1);
  const batchIndex = Math.min(batchCount - 1, Math.max(0, Number(state.batch_index) || 0));
  const batchCompleted = Math.min(batchCount, Math.max(0, Number(state.batch_completed_count) || 0));
  text(
    "batch-progress",
    state.batch_cycle_complete
      ? `本轮 ${batchCount} / ${batchCount} 已完成；下一条 1 / ${batchCount}`
      : `当前 ${batchIndex + 1} / ${batchCount} · 已完成 ${batchCompleted}`,
  );
  const speed = state.initial_alignment_speed_scale;
  text("alignment-speed", speed == null ? null : `${Number(speed).toFixed(2)}×`);
  syncConfiguration(state);
  const lowerHeightInput = document.getElementById("height-lower-tolerance");
  const upperHeightInput = document.getElementById("height-upper-tolerance");
  if (document.activeElement !== lowerHeightInput && state.height_lower_tolerance_m != null) {
    lowerHeightInput.value = (Number(state.height_lower_tolerance_m) * 1000).toFixed(1);
  }
  if (document.activeElement !== upperHeightInput && state.height_upper_tolerance_m != null) {
    upperHeightInput.value = (Number(state.height_upper_tolerance_m) * 1000).toFixed(1);
  }
  const configurationEnabled = optionsLoaded
    && state.can_configure
    && !state.configuration_pending
    && !state.gripper_action_pending
    && !pendingRequests.has("configuration");
  configurationInputs().forEach((input) => {
    input.disabled = !configurationEnabled
      || (input.id === "collect-tactile" && tactileCaptureLocked);
  });
  document.getElementById("apply-configuration").disabled = !configurationEnabled;
  document.getElementById("add-to-batch").disabled = !configurationEnabled;
  document.getElementById("clear-batch").disabled = !configurationEnabled;
  text(
    "configuration-status",
    state.configuration_pending
      ? "正在后台验证所选组合…"
      : configurationDirty
        ? `组合/批次已修改但尚未应用；当前队列 ${configurationBatch.length || 1} 条。`
      : state.configuration_error || (
          optionsLoaded
            ? `当前批次已预验证：${batchCount} 条；每条仍需人工点击开始。`
            : "正在读取可用组合…"
        ),
  );
  renderBatch();
  const gripperSides = state.gripper_sides || [];
  const gripperStatus = state.gripper_action_status || (
    state.gripper_control_available
      ? `夹爪独立控制可用：${gripperSides.join("、")}`
      : gripperSides.length
        ? `夹爪已配置：${gripperSides.join("、")}；当前安全/Session 阶段不允许独立控制。`
        : "夹爪独立控制：当前拓扑未启用。"
  );
  text("gripper-status", gripperStatus);
  text("task-chain-step", state.task_chain_step);
  text("task-chain-title", state.task_chain_title);
  text("task-chain-detail", state.task_chain_detail);
  const chainSelection = state.task_chain_selected_configuration || {};
  const chainIndex = state.task_chain_selected_configuration_index;
  const resolvedSelection = state.rule_source_selection || {};
  const resolvedPhases = resolvedSelection.phases || {};
  const resolvedCandidate = (phase) => {
    const value = resolvedPhases[phase];
    return value && typeof value === "object"
      ? value.candidate_id || "未知"
      : value || "未知";
  };
  const resolvedSources = Object.keys(resolvedPhases).length
    ? ` · 实际来源 靠近:${resolvedCandidate("pen_down")} / 书写:${resolvedCandidate("write")} / 离开:${resolvedCandidate("pen_up")}`
    : "";
  const variant = resolvedSelection.writing_variant_selection || {};
  const writingDetail = fullReplayMode()
    ? chainSelection.full_replay_candidate_id || "已验证组合"
    : selectionApi.describeWritingVariant(variant, chainSelection, trajectoryChoices);
  const runningWithoutChain = state.phase === "running"
    && !fullReplayMode() && Boolean(variant.selected_trajectory_id);
  const cadence = ` · 抓笔周期 ${Number(state.task_chain_episodes_since_grasp_adjustment || 0)}/${Number(state.task_chain_grasp_adjustment_interval || 10)}`;
  let selectionText = `尚未选择本轮轨迹${cadence}。`;
  if (Object.keys(chainSelection).length) {
    selectionText = `本轮组合：池中第 ${Number(chainIndex) + 1} 条 · ${writingDetail}${resolvedSources}${cadence}`;
  } else if (runningWithoutChain) {
    selectionText = `当前${writingDetail}${resolvedSources}${cadence}`;
  }
  text("task-chain-selection", selectionText);
  const chainButton = document.getElementById("task-chain-advance");
  chainButton.disabled = !state.task_chain_can_advance
    || configurationDirty
    || pendingRequests.has("task-chain");
  const chainButtonLabels = {
    idle: "开始任务链（Alt+N）",
    ready_close: "下一步：闭紧夹爪（Alt+N）",
    ready_height: "下一步：标定高度（Alt+N）",
    ready_rule: "下一步：随机执行规则采集（Alt+N）",
    ready_final_release: "下一步：松开夹爪完成（Alt+N）",
    complete: state.task_chain_grasp_adjustment_due
      ? "下一条：调整抓笔（Alt+N）"
      : "下一条：直接规则采集（Alt+N）",
  };
  chainButton.textContent = state.task_chain_owner_recovery_required
    ? "下一步：恢复右夹爪闭合（Alt+N）"
    : chainButtonLabels[state.task_chain_phase] || "当前步骤执行中…";
  document.body.dataset.phase = state.phase;
  document.getElementById("start").disabled = !state.can_start
    || configurationDirty || pendingRequests.has("event:rule_start");
  document.getElementById("start").textContent = state.writer_blocked
    ? "按当前配置重试 Writer"
    : state.batch_cycle_complete
      ? "重新开始本批次"
      : batchCount > 1 && batchCompleted > 0
        ? `开始下一条（${batchIndex + 1}/${batchCount}）`
        : `${fullReplayMode() ? "开始完整 Replay 采集" : "开始规则采集"}（${batchIndex + 1}/${batchCount}）`;
  document.getElementById("stop").disabled = !state.can_stop || pendingRequests.has("event:rule_stop");
  document.getElementById("exclude-last").disabled = !state.can_exclude_last_episode
    || pendingRequests.has("event:exclude_last_episode");
  text("curation-status", ["excluded", "error"].includes(state.episode_event_status)
    && /^(舍弃|已可恢复地排除)/.test(state.episode_event_message || "")
    ? state.episode_event_message : "");
  document.getElementById("retry").disabled = !state.can_retry_devices || pendingRequests.has("event:device_retry");
  document.getElementById("runtime-home-retry").disabled = !state.can_runtime_home || pendingRequests.has("event:runtime_home_retry");
  document.getElementById("release").disabled = !state.can_release_grippers || pendingRequests.has("event:release_grippers");
  document.getElementById("close-grippers").disabled = !state.can_close_grippers || pendingRequests.has("event:close_grippers");
  lowerHeightInput.disabled = !state.can_capture_height || pendingRequests.has("height-calibration:capture");
  upperHeightInput.disabled = !state.can_capture_height || pendingRequests.has("height-calibration:capture");
  document.getElementById("capture-height").disabled = !state.can_capture_height || pendingRequests.has("height-calibration:capture");
  document.getElementById("clear-height").disabled = !state.can_clear_height || pendingRequests.has("height-calibration:clear");
  const calibratedZ = state.height_calibrated_z_m;
  const lowerBound = state.height_lower_bound_z_m;
  const upperBound = state.height_upper_bound_z_m;
  text(
    "height-calibration-status",
    state.height_calibration_pending
      ? "高度标定：正在读取新鲜右臂 TCP 实测快照…"
      : state.height_calibration_error
        ? `高度标定失败：${state.height_calibration_error}`
        : state.height_calibration_enabled && calibratedZ != null
          ? `高度标定已启用：Z=${Number(calibratedZ).toFixed(6)} m；安全带 [${Number(lowerBound).toFixed(6)}, ${Number(upperBound).toFixed(6)}] m。`
          : "高度标定：未启用；沿用来源 replay 的原始高度。",
  );
  const debugButton = document.getElementById("debug-mode");
  debugButton.disabled = !state.can_toggle_debug || pendingRequests.has("event:debug_toggle");
  debugButton.textContent = state.debug_mode_pending
    ? "正在切换 Debug…"
    : state.debug_mode_enabled
      ? "关闭 Debug 诊断"
      : "开启 Debug 诊断";
  text(
    "debug-status",
    state.debug_mode_pending
      ? "Debug 诊断：正在切换…"
      : state.debug_mode_enabled
        ? `Debug 诊断：已开启；异步日志 ${state.debug_log_path || "正在建立"}`
        : state.debug_log_path
          ? `Debug 诊断：已关闭；最近日志 ${state.debug_log_path}`
          : "Debug 诊断：关闭。",
  );
  document.getElementById("quit").disabled = Boolean(state.shutdown);
  renderActiveDiagnostics(state);
  renderMessageHistory(state.messages);
  refreshInFlight = false;
  } finally {
    refreshInFlight = false;
  }
}

document.querySelectorAll("button[data-event]").forEach((button) => {
  button.addEventListener("click", () => sendEvent(button.dataset.event));
});
document.getElementById("apply-configuration").addEventListener("click", applyConfiguration);
document.getElementById("add-to-batch").addEventListener("click", addCurrentToBatch);
document.getElementById("clear-batch").addEventListener("click", clearBatch);
document.getElementById("capture-height").addEventListener("click", () => sendHeightCalibration("capture"));
document.getElementById("clear-height").addEventListener("click", () => sendHeightCalibration("clear"));
document.getElementById("task-chain-advance").addEventListener("click", advanceTaskChain);
document.addEventListener("keydown", (event) => {
  const target = event.target;
  const editing = target instanceof HTMLElement
    && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName);
  if (event.altKey && !event.ctrlKey && !event.metaKey
      && event.key.toLowerCase() === "n" && !editing) {
    const button = document.getElementById("task-chain-advance");
    if (!button.disabled) {
      event.preventDefault();
      advanceTaskChain();
    }
  }
});
configurationInputs().forEach((input) => {
  input.addEventListener("input", () => { configurationDirty = true; });
  input.addEventListener("change", () => { configurationDirty = true; });
});
document.getElementById("writing-speed").addEventListener("input", () => {
  writingSpeedEdited = true;
});
setInterval(refresh, 250);
setInterval(() => {
  const now = Date.now();
  const editing = document.activeElement
    && configurationInputs().includes(document.activeElement);
  if ((!optionsLoaded || now - lastOptionsRefreshAt >= 2000)
      && !configurationDirty && !editing) {
    lastOptionsRefreshAt = now;
    loadOptions();
  }
}, 500);
loadOptions();
refresh();
