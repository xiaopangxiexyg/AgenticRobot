"use strict";

// Optional RealTeleop long-horizon progress panel.
//
// This plugin consumes rebuildable Control Station state and emits only the
// typed event vocabulary admitted by the local UI server. It owns no dataset,
// robot, Hold, writer, or progress-reducer state. Removing this script leaves
// the existing RealTeleop collection GUI and A/B/C shortcuts unchanged.
(function () {
  let panel = null;
  let phaseElement = null;
  let stepElement = null;
  let metaElement = null;
  let messageElement = null;
  let statusElement = null;
  let profileSelect = null;
  let profileApplyButton = null;
  let profileBindingElement = null;
  let armControlModeSelect = null;
  let armControlModeApplyButton = null;
  let armControlModeStatusElement = null;
  let nextSelect = null;
  let nextStepElement = null;
  let nextStepMetaElement = null;
  let nextStepOptions = [];
  let advanceButton = null;
  let undoButton = null;
  let failureButton = null;
  let requestPending = false;
  let pendingSignature = "";
  let pendingTimer = null;
  let renderedDatasetRoot = "";

  function addStyle() {
    if (document.getElementById("realTeleopProgressStyle")) return;
    const style = document.createElement("style");
    style.id = "realTeleopProgressStyle";
    style.textContent = `
      .long-horizon-progress {
        margin-top: 14px;
        padding: 14px;
        border: 1px solid #89a6cc;
        border-radius: 8px;
        background: #f2f7ff;
        box-shadow: 0 0 0 2px rgba(55, 111, 189, 0.08);
      }
      .long-horizon-progress[data-phase="ACTIVE_ATTEMPT"] {
        border-color: #54b88d;
        background: #edf9f4;
      }
      .long-horizon-progress[data-phase="AWAITING_HOLD_RECEIPT"],
      .long-horizon-progress[data-phase="PAUSED_NO_ACTIVE_ATTEMPT"] {
        border-color: #e38c8c;
        background: #fff0f0;
      }
      .long-horizon-progress[data-phase="RESUME_RECEIVED"] {
        border-color: #dfa855;
        background: #fff7e8;
      }
      .long-horizon-progress-title {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: baseline;
        font-weight: 900;
      }
      .long-horizon-progress-phase {
        border: 1px solid rgba(29, 39, 53, 0.18);
        border-radius: 8px;
        padding: 4px 8px;
        font-size: 12px;
      }
      .long-horizon-progress-step {
        margin-top: 10px;
        font-size: 20px;
        font-weight: 900;
      }
      .long-horizon-progress-stages {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
      }
      .long-horizon-progress-stage-card {
        min-width: 0;
        padding: 12px 14px;
        border: 1px solid rgba(29, 39, 53, 0.18);
        border-radius: 8px;
      }
      .long-horizon-progress-stage-current {
        border-color: #54b88d;
        background: #edf9f4;
      }
      .long-horizon-progress-stage-next {
        border-color: #dfa855;
        background: #fff7e8;
      }
      .long-horizon-progress-stage-label {
        color: #526074;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.02em;
      }
      .long-horizon-progress-stage-name {
        margin-top: 6px;
        font-size: 19px;
        font-weight: 900;
        line-height: 1.3;
        overflow-wrap: anywhere;
      }
      .long-horizon-progress-stage-meta {
        margin-top: 7px;
        color: #526074;
        font-size: 12px;
        line-height: 1.45;
        overflow-wrap: anywhere;
      }
      .long-horizon-progress-meta,
      .long-horizon-progress-message,
      .long-horizon-progress-status {
        margin-top: 7px;
        color: #526074;
        font-size: 12px;
        line-height: 1.45;
      }
      .long-horizon-progress-controls {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto;
        gap: 8px;
        margin-top: 12px;
        align-items: center;
      }
      .long-horizon-progress-profile {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 8px;
        margin-top: 12px;
      }
      .long-horizon-progress-profile select {
        min-width: 0;
        border: 1px solid #d7dee8;
        border-radius: 8px;
        padding: 10px;
        background: #fff;
      }
      .long-horizon-progress-controls select {
        min-width: 0;
        border: 1px solid #d7dee8;
        border-radius: 8px;
        padding: 10px;
        background: #fff;
      }
      .long-horizon-progress-actions {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        margin-top: 8px;
      }
      .long-horizon-progress-actions button {
        min-height: 48px;
        font-weight: 850;
      }
      .long-horizon-progress-actions .progress-failure {
        grid-column: 1 / -1;
        background: #ffe1e1;
        border-color: #d96d6d;
        color: #861f1f;
      }
      @media (max-width: 760px) {
        .long-horizon-progress-stages {
          grid-template-columns: 1fr;
        }
        .long-horizon-progress-controls,
        .long-horizon-progress-profile,
        .long-horizon-progress-actions {
          grid-template-columns: 1fr;
        }
        .long-horizon-progress-actions .progress-failure {
          grid-column: auto;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function makeButton(className, label) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    return button;
  }

  function ensurePanel() {
    if (panel) return panel;
    addStyle();
    panel = document.createElement("section");
    panel.id = "longHorizonProgressPanel";
    panel.className = "long-horizon-progress";
    panel.hidden = true;

    const title = document.createElement("div");
    title.className = "long-horizon-progress-title";
    const titleText = document.createElement("span");
    titleText.textContent = "长程任务进度";
    phaseElement = document.createElement("span");
    phaseElement.className = "long-horizon-progress-phase";
    title.append(titleText, phaseElement);

    stepElement = document.createElement("div");
    stepElement.className = "long-horizon-progress-stage-name";
    metaElement = document.createElement("div");
    metaElement.className = "long-horizon-progress-stage-meta mono";
    messageElement = document.createElement("div");
    messageElement.className = "long-horizon-progress-message";

    const stages = document.createElement("div");
    stages.className = "long-horizon-progress-stages";
    const currentStage = document.createElement("div");
    currentStage.className = (
      "long-horizon-progress-stage-card "
      + "long-horizon-progress-stage-current"
    );
    const currentStageLabel = document.createElement("div");
    currentStageLabel.className = "long-horizon-progress-stage-label";
    currentStageLabel.textContent = "当前阶段（本次采集）";
    currentStage.append(currentStageLabel, stepElement, metaElement);

    const nextStage = document.createElement("div");
    nextStage.className = (
      "long-horizon-progress-stage-card "
      + "long-horizon-progress-stage-next"
    );
    const nextStageLabel = document.createElement("div");
    nextStageLabel.className = "long-horizon-progress-stage-label";
    nextStageLabel.textContent = "下一阶段（完成当前后进入）";
    nextStepElement = document.createElement("div");
    nextStepElement.className = "long-horizon-progress-stage-name";
    nextStepMetaElement = document.createElement("div");
    nextStepMetaElement.className = "long-horizon-progress-stage-meta mono";
    nextStage.append(nextStageLabel, nextStepElement, nextStepMetaElement);
    stages.append(currentStage, nextStage);

    const profileControls = document.createElement("div");
    profileControls.className = "long-horizon-progress-profile";
    profileSelect = document.createElement("select");
    profileSelect.id = "progressProfileSelect";
    profileApplyButton = makeButton(
      "secondary progress-profile-select",
      "选择 Profile",
    );
    profileControls.append(profileSelect, profileApplyButton);
    profileBindingElement = document.createElement("div");
    profileBindingElement.className = "long-horizon-progress-meta mono";

    const armControlModeControls = document.createElement("div");
    armControlModeControls.className = "long-horizon-progress-profile";
    armControlModeSelect = document.createElement("select");
    armControlModeSelect.id = "progressArmControlModeSelect";
    armControlModeApplyButton = makeButton(
      "secondary progress-arm-control-mode-select",
      "应用臂控制模式",
    );
    armControlModeControls.append(
      armControlModeSelect,
      armControlModeApplyButton,
    );
    armControlModeStatusElement = document.createElement("div");
    armControlModeStatusElement.className = "long-horizon-progress-meta mono";

    const controls = document.createElement("div");
    controls.className = "long-horizon-progress-controls";
    nextSelect = document.createElement("select");
    nextSelect.id = "progressNextStepSelect";
    const nextSelectLabel = document.createElement("label");
    nextSelectLabel.htmlFor = nextSelect.id;
    nextSelectLabel.textContent = "推进目标";
    advanceButton = makeButton("progress-advance", "进入下一阶段");
    controls.append(nextSelectLabel, nextSelect, advanceButton);

    const actions = document.createElement("div");
    actions.className = "long-horizon-progress-actions";
    undoButton = makeButton("secondary progress-undo", "撤销刚才的阶段推进");
    failureButton = makeButton(
      "danger progress-failure",
      "记录当前阶段失败并 Hold",
    );
    actions.append(undoButton, failureButton);

    statusElement = document.createElement("div");
    statusElement.className = "long-horizon-progress-status mono";
    panel.append(
      title,
      stages,
      messageElement,
      profileControls,
      profileBindingElement,
      armControlModeControls,
      armControlModeStatusElement,
      controls,
      actions,
      statusElement,
    );

    const mount = document.getElementById("longHorizonProgressMount");
    if (mount) {
      mount.appendChild(panel);
    } else {
      const controlPanel = document.querySelector("main .panel:last-child .panel-body");
      if (controlPanel) controlPanel.appendChild(panel);
    }

    advanceButton.addEventListener("click", () => {
      const target = String(nextSelect.value || "").trim();
      if (target) submitEvent(`progress_advance:${target}`);
    });
    nextSelect.addEventListener("change", renderNextStagePreview);
    profileApplyButton.addEventListener("click", () => {
      const profileId = String(profileSelect.value || "").trim();
      if (profileId) submitEvent(`progress_profile_select:${profileId}`);
    });
    armControlModeApplyButton.addEventListener("click", () => {
      const mode = String(armControlModeSelect.value || "").trim();
      if (mode) submitEvent(`progress_arm_control_mode_select:${mode}`);
    });
    undoButton.addEventListener("click", () => submitEvent("progress_undo"));
    // Failure is a time-critical capture fact. The button is already
    // state-gated and request-deduplicated, so a modal confirmation would
    // only delay the effective timestamp and Hold request.
    failureButton.addEventListener(
      "click",
      () => submitEvent("progress_failure"),
    );
    return panel;
  }

  function renderNextStagePreview() {
    if (!nextStepElement || !nextStepMetaElement || !nextSelect) return;
    const selectedId = String(nextSelect.value || "");
    const selected = nextStepOptions.find(
      (item) => String(item.progress_step_id) === selectedId,
    );
    if (!selected) {
      nextStepElement.textContent = "暂无可用下一阶段";
      nextStepMetaElement.textContent = nextStepOptions.length > 0
        ? "请先从下方选择推进目标。"
        : "当前阶段完成后暂无可推进目标。";
      return;
    }
    nextStepElement.textContent = String(
      selected.display_name || selected.progress_step_id || "-",
    );
    nextStepMetaElement.textContent = [
      `id=${String(selected.progress_step_id || "-")}`,
      nextStepOptions.length > 1
        ? `${nextStepOptions.length} 个候选，当前已选`
        : "唯一可推进目标",
      "点击下方按钮后生效",
    ].join(" · ");
  }

  function stateSignature(progress) {
    return JSON.stringify([
      progress.phase || "",
      progress.current_progress_step_id || "",
      progress.current_attempt_id || "",
      progress.profile_id || "",
      Boolean(progress.profile_selection_locked),
      Boolean(progress.can_undo_last_advance),
      Boolean(progress.application_held),
      Boolean(progress.failure_promotes_existing_hold),
      progress.arm_control_mode || "",
      Number(progress.transport_revision || 0),
    ]);
  }

  function setPending(value, signature) {
    requestPending = Boolean(value);
    pendingSignature = requestPending ? String(signature || "") : "";
    if (pendingTimer !== null) {
      window.clearTimeout(pendingTimer);
      pendingTimer = null;
    }
    if (requestPending) {
      pendingTimer = window.setTimeout(() => {
        requestPending = false;
        pendingSignature = "";
        if (statusElement) {
          statusElement.textContent = "请求处理超时；请根据当前状态重试。";
        }
      }, 3000);
    }
  }

  async function submitEvent(eventName) {
    if (requestPending) return;
    const progress = (window.__dexteleopLatestProgressState || {}).progress_ui || {};
    setPending(true, stateSignature(progress));
    if (statusElement) statusElement.textContent = `已提交：${eventName}`;
    try {
      const response = await fetch("/event", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({event: eventName}),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      setPending(false, "");
      if (statusElement) {
        statusElement.textContent = `请求未生效：${String(error)}`;
      }
    }
  }

  function phaseLabel(phase) {
    const labels = {
      IDLE: "等待 Episode",
      ACTIVE_ATTEMPT: "阶段采集中",
      AWAITING_HOLD_RECEIPT: "等待 Hold",
      PAUSED_NO_ACTIVE_ATTEMPT: "失败后暂停",
      RESUME_RECEIVED: "等待 Retry 首帧",
      TERMINATED: "已结束",
      ABSENT: "未启用",
    };
    return labels[String(phase || "")] || String(phase || "-");
  }

  function render(state) {
    window.__dexteleopLatestProgressState = state || {};
    const progress = (state && state.progress_ui) || {};
    const root = ensurePanel();
    const enabled = Boolean(progress.enabled);
    const module = document.getElementById("longHorizonProgressModule");
    if (module) module.hidden = !enabled;
    root.hidden = !enabled;
    if (!enabled) return;

    const signature = stateSignature(progress);
    if (requestPending && signature !== pendingSignature) {
      setPending(false, "");
    }
    const activeControls = new Set(
      Array.isArray(state.active_controls) ? state.active_controls.map(String) : []
    );
    const nextSteps = Array.isArray(progress.next_steps)
      ? progress.next_steps
      : [];
    nextStepOptions = nextSteps;
    const profiles = Array.isArray(progress.available_profiles)
      ? progress.available_profiles
      : [];
    const armControlModes = Array.isArray(progress.arm_control_modes)
      ? progress.arm_control_modes.map(String)
      : [];
    const priorProfileSelection = String(profileSelect.value || "");
    const currentDatasetRoot = String(
      (state && (state.dataset_root || state.dataset_base_root)) || "",
    );
    const datasetChanged = currentDatasetRoot !== renderedDatasetRoot;
    profileSelect.replaceChildren();
    for (const item of profiles) {
      const option = document.createElement("option");
      option.value = String(item.profile_id || "");
      option.textContent = [
        String(item.display_name || item.profile_id || "-"),
        `profile-task=${String(item.task_id || "-")}`,
        `v${String(item.profile_version || "-")}`,
        `${Number(item.step_count || 0)} 阶段`,
      ].join(" · ");
      profileSelect.appendChild(option);
    }
    const selectedProfileId = String(progress.profile_id || "");
    // Keep the operator's local draft while polling one dataset. A dataset
    // switch must instead adopt the target dataset's canonical binding;
    // otherwise a locked target can misleadingly display the previous
    // dataset's profile forever.
    if (
      !datasetChanged
      && !Boolean(progress.profile_selection_locked)
      && profiles.some(
        (item) => String(item.profile_id) === priorProfileSelection,
      )
    ) {
      profileSelect.value = priorProfileSelection;
    } else if (
      profiles.some((item) => String(item.profile_id) === selectedProfileId)
    ) {
      profileSelect.value = selectedProfileId;
    }
    renderedDatasetRoot = currentDatasetRoot;
    const priorSelection = String(nextSelect.value || "");
    nextSelect.replaceChildren();
    for (const item of nextSteps) {
      const option = document.createElement("option");
      option.value = String(item.progress_step_id || "");
      option.textContent = String(item.display_name || item.progress_step_id || "-");
      nextSelect.appendChild(option);
    }
    if (nextSteps.some((item) => String(item.progress_step_id) === priorSelection)) {
      nextSelect.value = priorSelection;
    }
    renderNextStagePreview();

    const phase = String(progress.phase || "IDLE");
    root.dataset.phase = phase;
    phaseElement.textContent = phaseLabel(phase);
    stepElement.textContent = progress.current_progress_step_display_name
      || progress.current_progress_step_id
      || "等待开始";
    const ordinal = progress.current_step_ordinal;
    const count = Number(progress.total_step_count || 0);
    metaElement.textContent = [
      ordinal == null || count <= 0 ? "" : `阶段 ${ordinal} / ${count}`,
      progress.current_progress_step_id
        ? `id=${progress.current_progress_step_id}`
        : "",
      progress.current_attempt_id
        ? `attempt=${progress.current_attempt_id}`
        : "",
      progress.profile_id ? `profile=${progress.profile_id}` : "",
    ].filter(Boolean).join(" · ");
    messageElement.textContent = String(progress.message || "");
    const profileSelectionAllowed = (
      activeControls.has("progress_profile_select")
      && Boolean(progress.profile_selection_enabled)
      && !Boolean(progress.profile_selection_locked)
      && profiles.length > 0
      && !requestPending
    );
    profileSelect.disabled = !profileSelectionAllowed;
    profileApplyButton.disabled = !profileSelectionAllowed;
    profileApplyButton.textContent = progress.configured
      ? "切换 Profile（首个 Episode 前）"
      : "选择 Profile";
    if (Boolean(progress.profile_selection_locked)) {
      profileBindingElement.textContent = (
        String(progress.profile_binding_source || "")
          === "populated_profileless_dataset"
          ? "当前数据集已写入普通 Episode，保持 profile-less；普通数采不受阻塞。启用长程 progress 请切换新的空数据集。"
          : `数据集 Profile 已锁定 · ${String(progress.profile_binding_source || "dataset_binding")}；如需更换，请切换新的数据集。`
      );
    } else if (!Boolean(progress.profile_selection_enabled)) {
      profileBindingElement.textContent = (
        "当前 Episode 正在准备、录制、暂停或保存；回到空闲后即可选择或切换 Profile。"
      );
    } else {
      profileBindingElement.textContent = profiles.length > 0
        ? "当前可自由选择 Profile；Task name/id 与 Profile 独立，首个 progress Episode 才会按 hash 绑定。"
        : "当前没有可选的 LINEAR managed profile。";
    }

    const armModeLabels = {
      strict_profile: "严格 Profile：禁止手动 Fix/单臂踏板，并启用 writer 一致性门",
      profile: "默认 Profile：阶段切换自动跟随，仍允许手动切换",
      ignore_profile: "无视 Profile：只遵循踏板与按钮",
    };
    const priorArmMode = String(armControlModeSelect.value || "");
    armControlModeSelect.replaceChildren();
    for (const mode of armControlModes) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = armModeLabels[mode] || mode;
      armControlModeSelect.appendChild(option);
    }
    const selectedArmMode = String(progress.arm_control_mode || "profile");
    armControlModeSelect.value = armControlModes.includes(priorArmMode)
      ? priorArmMode
      : selectedArmMode;
    const armModeSelectionAllowed = (
      activeControls.has("progress_arm_control_mode_select")
      && Boolean(progress.arm_control_mode_selection_enabled)
      && !requestPending
    );
    armControlModeSelect.disabled = !armModeSelectionAllowed;
    armControlModeApplyButton.disabled = !armModeSelectionAllowed;
    const stepArmControl = progress.current_step_arm_control || {};
    const sideModeLabel = (side, mode) => {
      if (mode === "teleop") return `${side}遥操`;
      if (mode === "fixed_hold") return `${side}固定（机械臂+末端）`;
      return `${side}未声明`;
    };
    armControlModeStatusElement.textContent = [
      armModeLabels[selectedArmMode] || selectedArmMode,
      `当前子任务：${sideModeLabel("左臂", String(stepArmControl.left || ""))} · ${sideModeLabel("右臂", String(stepArmControl.right || ""))}`,
    ].join("；");

    const advanceAllowed = (
      activeControls.has("progress_advance")
      && nextSteps.length > 0
      && !requestPending
    );
    nextSelect.disabled = !advanceAllowed;
    advanceButton.disabled = !advanceAllowed;
    advanceButton.textContent = nextSteps.length === 1
      ? `完成当前阶段并进入：${String(nextSteps[0].display_name || nextSteps[0].progress_step_id)}`
      : "完成当前阶段并进入所选阶段";
    undoButton.disabled = (
      !activeControls.has("progress_undo")
      || !Boolean(progress.can_undo_last_advance)
      || requestPending
    );
    failureButton.disabled = (
      !activeControls.has("progress_failure")
      || requestPending
    );
    failureButton.textContent = Boolean(
      progress.failure_promotes_existing_hold
    )
      ? "将当前 Hold 记为阶段失败"
      : "记录当前阶段失败并 Hold";
    if (!requestPending) {
      statusElement.textContent = (
        Boolean(progress.failure_promotes_existing_hold)
          ? "沿用本次 Hold 的原暂停边界；不会再次暂停或移动失败时间。"
          : "独立语义层：不改变 A/B/C、机械臂控制或原数采按钮。"
      );
    }
  }

  window.__dexteleopRenderProgressUi = render;
})();
