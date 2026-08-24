
(function () {
  if (window.__calibrationWebUiExternalFallbackLoaded) return;
  window.__calibrationWebUiExternalFallbackLoaded = true;
  var fallbackLatestState = {};
  var fallbackActiveControls = {};
  var fallbackTrackerSideConfirmationToken = "";
  var fallbackPlusReviewToken = "";
  var fallbackPlusReviewDisposition = "unknown";
  var fallbackPlusReviewReason = "unknown";
  var fallbackPlusVariantDraft = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function setHealth(message) {
    var el = byId("scriptHealth");
    if (!el) return;
    el.hidden = false;
    el.textContent = message;
  }

  function setText(id, value) {
    var el = byId(id);
    if (el) el.textContent = value == null || value === "" ? "-" : String(value);
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function normalizeTaskTypeFallback(value) {
    var text = String(value || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
    return /^(roundtrip|round_trip|back_and_forth|in_out|inout|往返|来回)$/.test(text) ? "round_trip" : "general";
  }

  function normalizeRoundTripPhaseFallback(value) {
    return String(value || "").trim().toLowerCase() === "out" ? "out" : "in";
  }

  function plusPendingRootLabelFallback(value) {
    var text = String(value || "").trim();
    if (!text) return "";
    var parts = text.split(/[\\/]+/).filter(Boolean);
    return parts.length ? parts[parts.length - 1] : text;
  }

  function updatePlusReviewControlsFallback(state) {
    var pending = Boolean(state && state.plus_curation_pending);
    var reviewReady = pending && Boolean(fallbackActiveControls.plus_curation_save) && String(state.plus_curation_status || "") !== "saving";
    var disposition = byId("plusDispositionSelect");
    var reason = byId("plusReasonSelect");
    if (disposition) {
      disposition.disabled = !reviewReady;
      if (document.activeElement !== disposition) disposition.value = fallbackPlusReviewDisposition || "unknown";
    }
    if (reason) {
      var reasonAllowed = fallbackPlusReviewDisposition === "keep_recovery" || fallbackPlusReviewDisposition === "exclude";
      reason.disabled = !reviewReady || !reasonAllowed;
      if (!reasonAllowed) fallbackPlusReviewReason = "unknown";
      if (document.activeElement !== reason) reason.value = fallbackPlusReviewReason || "unknown";
    }
  }

  function labelStage(value) {
    var labels = {
      "Starting...": "启动中...",
      "RM65 真机遥操准备": "RM65 真机遥操准备",
      "Home Capture": "起点采集",
      "HomeCapture": "起点采集",
      "runtime_home": "runtime home",
      "teleop": "遥操"
    };
    return labels[value] || value || "-";
  }

  function labelPhase(value) {
    var labels = {
      idle: "空闲",
      ready: "准备就绪",
      preparing: "准备倒计时",
      capturing: "正在采集",
      waiting: "等待样本",
      teleop: "遥操中"
    };
    return labels[value] || value || "-";
  }

  function setCaptureClass(phase) {
    var el = byId("captureMode");
    if (el) el.className = "capture-mode " + (phase || "idle");
  }

  function setProgress(percent) {
    var el = byId("captureProgressBar");
    if (!el) return;
    if (!isFinite(percent)) percent = 0;
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    el.style.width = String(percent) + "%";
  }

  function captureSamplesText(state) {
    if (state.capture_left_samples != null || state.capture_right_samples != null) {
      return "左 " + (state.capture_left_samples || 0) + "/" + (state.capture_min_samples || 0) +
        "  右 " + (state.capture_right_samples || 0) + "/" + (state.capture_min_samples || 0);
    }
    if (state.capture_samples != null) {
      return String(state.capture_samples) + "/" + (state.capture_min_samples || 0);
    }
    return "-";
  }

  function captureValidText(state) {
    if (state.capture_left_valid != null || state.capture_right_valid != null) {
      return "左 " + (state.capture_left_valid ? "有效" : "缺失") +
        " / 右 " + (state.capture_right_valid ? "有效" : "缺失");
    }
    if (state.capture_valid != null) return state.capture_valid ? "有效" : "缺失";
    return "-";
  }

  function renderRuntimeHomeDiagnosticFallback(state) {
    var panel = byId("runtimeHomeDiagnostic");
    var title = byId("runtimeHomeDiagnosticTitle");
    var detail = byId("runtimeHomeDiagnosticDetail");
    if (!panel) return;
    var diagnostic = state.runtime_home_diagnostic || {};
    var visible = Boolean(diagnostic.visible);
    panel.hidden = !visible;
    if (!visible) {
      panel.className = "runtime-home-diagnostic";
      if (title) title.textContent = "";
      if (detail) detail.textContent = "";
      return;
    }
    var level = String(diagnostic.level || "capturing");
    if (level !== "capturing" && level !== "error") level = "capturing";
    panel.className = "runtime-home-diagnostic " + level;
    if (title) title.textContent = diagnostic.title || "Runtime Home 状态";
    if (detail) detail.textContent = diagnostic.detail || "-";
  }

  function numberText(value, digits) {
    var numeric = Number(value);
    if (!isFinite(numeric)) return value == null ? "nan" : String(value);
    return numeric.toFixed(digits);
  }

  function vectorText(values, digits) {
    if (!values || !values.length) return "-";
    var parts = [];
    for (var i = 0; i < values.length; i += 1) {
      parts.push(numberText(values[i], digits));
    }
    return parts.join(", ");
  }

  function rawPoseText(pose) {
    if (pose && pose.enabled === false) {
      return {status: "未启用", text: "当前 logical side 未启用"};
    }
    if (!pose) {
      return {status: "无数据", text: "未收到 channel"};
    }
    var status = pose.status_label || (pose.valid ? "有效" : "无效");
    return {
      status: status,
      text: [
        "tracker: " + (pose.tracker_name || "-"),
        "role:    " + (pose.tracker_role || "-"),
        "seq:     " + (pose.seq == null ? "-" : String(pose.seq)),
        "age:     " + (pose.age_sec == null ? "-" : numberText(pose.age_sec, 2) + " s"),
        "xyz(m):  " + vectorText(pose.xyz_m, 4),
        "rpy(deg):" + vectorText(pose.rpy_deg, 1)
      ].join("\n")
    };
  }

  function setRawPoseClass(id, pose) {
    var el = byId(id);
    if (!el) return;
    var level = pose && pose.status_level ? String(pose.status_level) : "ok";
    el.className = "raw-pose-card " + level;
  }

  function renderCaptureMode(state) {
    var phase = state.teleop_running ? "teleop" : (state.capture_phase || "idle");
    if (!state.teleop_running && !state.capture_stage) phase = "idle";
    var detail = "等待流程进入采集阶段";
    var progress = 0;
    var prepTotal = Number(state.capture_prep_total_sec || 0);
    var prepRemaining = Number(state.capture_prep_remaining_sec || 0);
    var duration = Number(state.capture_duration_sec || 0);
    var elapsed = Number(state.capture_elapsed_sec || 0);
    var minSamples = Number(state.capture_min_samples || 0);
    var leftSamples = Number(state.capture_left_samples || 0);
    var rightSamples = Number(state.capture_right_samples || 0);
    var captureStage = String(state.capture_stage || "");
    var sampleLabel = captureStage === "runtime_home" ? "Home 样本" : (captureStage === "force_control_warmup" ? "力控预热" : "标定样本");
    if (phase === "ready") {
      detail = "按“开始采集”后会先进入准备倒计时";
    } else if (phase === "preparing") {
      progress = prepTotal > 0 ? (1 - prepRemaining / prepTotal) * 100 : 100;
      detail = captureStage === "force_control_warmup"
        ? "正在预热 RM65 力控，预计剩余 " + prepRemaining.toFixed(1) + "s"
        : "倒计时 " + prepRemaining.toFixed(1) + "s，当前还未写入" + sampleLabel;
    } else if (phase === "capturing") {
      if (minSamples > 0 && (state.capture_left_samples != null || state.capture_right_samples != null)) {
        progress = Math.min(leftSamples, rightSamples) / minSamples * 100;
        detail = "正在采集 " + sampleLabel + "，左 " + leftSamples + "/" + minSamples + "，右 " + rightSamples + "/" + minSamples;
      } else if (duration > 0) {
        progress = elapsed / duration * 100;
        detail = "正在写入样本，已采集 " + elapsed.toFixed(1) + " / " + duration.toFixed(1) + "s";
      }
    } else if (phase === "waiting") {
      progress = 100;
      detail = "等待 VIVE/Q6/相机样本进入采集窗口";
    } else if (phase === "teleop") {
      progress = 100;
      detail = state.gripper_control_enabled ? "真机遥操中，Q6 夹爪控制已启用" : "真机遥操中";
    }
    setCaptureClass(phase);
    setText("captureModeLabel", labelPhase(phase));
    setText("captureModeDetail", detail);
    setProgress(progress);
  }

  function renderButtons(state) {
    var active = {};
    var controls = state.active_controls || [];
    var footPadMode = String(state.special_arm_control_mode || "manual") === "foot-pad";
    var twoPedal = footPadMode && Number(state.foot_pad_count || 0) === 2;
    var fivePedal = footPadMode && Number(state.foot_pad_count || 0) === 5;
    var twoPedalStyle = twoPedal || fivePedal;
    var i;
    for (i = 0; i < controls.length; i += 1) active[String(controls[i])] = true;
    fallbackActiveControls = active;
    var shortcutC = byId("shortcutCLabel");
    if (shortcutC) {
      if (footPadMode && state.fix_arm_available) {
        shortcutC.textContent = twoPedalStyle
          ? "进入双臂 Hold"
          : "踏板 1/2：固定 / 重采左右臂 Home";
      } else if (String(state.ui_mode || "") === "umi_collection" && state.fix_arm_available) {
        var umiFixSide = state.fix_arm_fixed_side || state.fix_arm_selected_side || "left";
        var umiFixLabel = umiFixSide === "right" ? "右臂" : "左臂";
        shortcutC.textContent = (state.fix_arm_active ? "取消 " : "标注 ") + umiFixLabel + " fixed";
      } else if (state.fix_arm_available && state.teleop_running) {
        var shortcutSide = state.fix_arm_fixed_side || state.fix_arm_selected_side || "left";
        var shortcutLabel = shortcutSide === "right" ? "右臂" : "左臂";
        shortcutC.textContent = state.fix_arm_recalibrating || state.fix_arm_active
          ? "重采 " + shortcutLabel + " Home"
          : "固定 " + shortcutLabel;
      } else if (state.teleop_running) {
        shortcutC.textContent = "无 fix-arm 时不可用";
      } else {
        shortcutC.textContent = "固定臂不可用";
      }
    }
    var shortcutG = byId("shortcutGLabel");
    if (shortcutG) {
      if (String(state.ui_mode || "") === "umi_collection" && footPadMode && state.init_arm_available) {
        shortcutG.textContent = twoPedal
          ? "2 踏板仅切换 fixed；init 标注使用 G/GUI"
          : fivePedal
          ? "5 踏板 3/4/5 用于 progress；init 标注使用 G/GUI"
          : "踏板 3/4：切换左右 init 标注";
      } else if (twoPedal && state.init_arm_available) {
        shortcutG.textContent = "双踏板不执行 Init；Init 仍可使用 G/GUI";
      } else if (footPadMode && state.init_arm_available) {
        shortcutG.textContent = fivePedal
          ? "5 踏板 3/4/5：推进/失败/撤销；Init 使用 G/GUI"
          : "踏板 3/4 或 G/GUI：Init 左/右臂";
      } else if (String(state.ui_mode || "") === "umi_collection" && state.init_arm_available) {
        var umiInitSide = state.init_arm_active_side || state.init_arm_selected_side || "left";
        var umiInitLabel = umiInitSide === "right" ? "右臂" : "左臂";
        shortcutG.textContent = (state.init_arm_active ? "取消 " : "标注 ") + umiInitLabel + " init";
      } else if (state.init_arm_available && state.teleop_running) {
        var initShortcutSide = state.init_arm_active_side || state.init_arm_selected_side || "left";
        var initShortcutLabel = initShortcutSide === "right" ? "右臂" : "左臂";
        shortcutG.textContent = state.init_arm_recalibrating || state.init_arm_active
          ? "重采 " + initShortcutLabel + " Home"
          : initShortcutLabel + "归位";
      } else {
        shortcutG.textContent = "归位不可用";
      }
    }
    var buttons = document.querySelectorAll ? document.querySelectorAll("button[data-event]") : [];
    for (i = 0; i < buttons.length; i += 1) {
      var eventName = buttons[i].getAttribute("data-event");
      buttons[i].disabled = !active[eventName];
      if (
        String(state.ui_mode || "") === "umi_collection" &&
        (eventName === "c" || eventName === "fix_arm_toggle" || eventName === "init_arm_toggle")
      ) {
        var annotationSide = eventName === "init_arm_toggle"
          ? (state.init_arm_active_side || state.init_arm_selected_side || "left")
          : (state.fix_arm_fixed_side || state.fix_arm_selected_side || "left");
        var annotationLabel = annotationSide === "right" ? "右臂" : "左臂";
        var annotationActive = eventName === "init_arm_toggle"
          ? Boolean(state.init_arm_active)
          : Boolean(state.fix_arm_active);
        var annotationName = eventName === "init_arm_toggle" ? "init" : "fixed";
        buttons[i].textContent = (annotationActive ? "取消 " : "标注 ") + annotationLabel + " " + annotationName;
        continue;
      }
      if (eventName === "c") {
        if (state.fix_arm_available && state.teleop_running) {
          var side = state.fix_arm_fixed_side || state.fix_arm_selected_side || "left";
          var label = side === "right" ? "右臂" : "左臂";
          buttons[i].textContent = state.fix_arm_recalibrating
            ? "重采 " + label + " Home..."
            : state.fix_arm_active
              ? "重采 " + label + " Home"
              : "固定 " + label;
        } else if (state.teleop_running) {
          buttons[i].textContent = "无 fix-arm 时不可用";
        } else {
          buttons[i].textContent = "固定臂不可用";
        }
      } else if (eventName === "fix_arm_toggle") {
        var fixSide = state.fix_arm_fixed_side || state.fix_arm_selected_side || "left";
        var fixLabel = fixSide === "right" ? "右臂" : "左臂";
        buttons[i].textContent = state.fix_arm_recalibrating
          ? "重采 " + fixLabel + " Home..."
          : state.fix_arm_active
            ? "重采 " + fixLabel + " Home"
            : "固定 " + fixLabel;
      } else if (eventName === "init_arm_toggle") {
        var initSide = state.init_arm_active_side || state.init_arm_selected_side || "left";
        var initLabel = initSide === "right" ? "右臂" : "左臂";
        buttons[i].textContent = state.init_arm_recalibrating
          ? "重采 " + initLabel + " Home..."
          : state.init_arm_active
            ? "重采 " + initLabel + " Home"
            : "Cartesian 标准位姿：" + initLabel;
      } else if (eventName === "init_arm_move_to_qpos") {
        var moveInitSide = state.init_arm_active_side || state.init_arm_selected_side || "left";
        var moveInitLabel = moveInitSide === "right" ? "右臂" : "左臂";
        buttons[i].hidden = String(state.ui_mode || "") === "umi_collection";
        buttons[i].disabled = !active[eventName] || Boolean(state.init_arm_active_side);
        buttons[i].textContent = state.init_arm_active_side
          ? moveInitLabel + "精确归位中"
          : "精确归位所选臂：" + moveInitLabel;
      } else if (eventName === "init_arm_move_both_to_qpos") {
        var sequenceSide = state.init_arm_sequence_side || state.init_arm_active_side || "left";
        var sequenceLabel = sequenceSide === "right" ? "右臂" : "左臂";
        buttons[i].hidden = String(state.ui_mode || "") === "umi_collection";
        buttons[i].disabled = !active[eventName] || Boolean(state.init_arm_active_side);
        buttons[i].textContent = state.init_arm_sequence_active
          ? "双臂顺序归位中：当前" + sequenceLabel
          : "双臂顺序精确归位：左臂 → 右臂";
      }
    }
    var speedScales = state.init_arm_qpos_speed_scale_by_side || {};
    var speedMin = Number(state.init_arm_qpos_speed_scale_min == null ? 0.1 : state.init_arm_qpos_speed_scale_min);
    var speedMax = Number(state.init_arm_qpos_speed_scale_max == null ? 10.0 : state.init_arm_qpos_speed_scale_max);
    var speedEnabled = Boolean(active.init_arm_qpos_speed_set);
    var speedInputs = {
      left: byId("leftInitArmQposSpeedInput"),
      right: byId("rightInitArmQposSpeedInput")
    };
    for (var speedSide in speedInputs) {
      if (!Object.prototype.hasOwnProperty.call(speedInputs, speedSide)) continue;
      var speedInput = speedInputs[speedSide];
      if (!speedInput) continue;
      speedInput.min = String(speedMin);
      speedInput.max = String(speedMax);
      speedInput.disabled = !speedEnabled;
      var speedValue = Number(speedScales[speedSide] == null ? 5.0 : speedScales[speedSide]);
      if (document.activeElement !== speedInput && isFinite(speedValue)) {
        speedInput.value = speedValue.toFixed(3);
      }
    }
    var speedApply = byId("initArmQposSpeedApplyButton");
    if (speedApply) speedApply.disabled = !speedEnabled;
    var leftScale = Number(speedScales.left == null ? 5.0 : speedScales.left);
    var rightScale = Number(speedScales.right == null ? 5.0 : speedScales.right);
    setText(
      "initArmQposSpeedStatus",
      speedEnabled
        ? "当前：左臂 " + leftScale.toFixed(3) + "× / 右臂 " + rightScale.toFixed(3) + "×；基线 1.0 = rm65_init_slow。"
        : state.init_arm_qpos_active
          ? "归位执行中已锁定速度；如需停止请使用 Init 急停。"
          : "数采流程中已锁定速度；本次归位使用开始前设置。"
    );
    renderFootPadTestFallback(state);
  }

  function renderFootPadTestFallback(state) {
    var available = Boolean(state.foot_pad_available);
    var requested = Boolean(state.foot_pad_requested);
    var active = Boolean(state.foot_pad_test_active);
    var completed = Boolean(state.foot_pad_test_completed);
    var twoPedal = Number(state.foot_pad_count || 0) === 2;
    var fivePedal = Number(state.foot_pad_count || 0) === 5;
    var banner = byId("footPadTestBanner");
    var controls = byId("footPadTestControls");
    if (banner) {
      banner.hidden = !(requested || available);
      banner.className = "foot-pad-test-banner" + (active ? " active" : (!available ? " unavailable" : (completed ? " completed" : "")));
    }
    if (controls) controls.hidden = !available;
    setText("footPadTestTitle", !available ? "踏板未连接 / 未就绪" : (active ? "脚踏测试模式已开启（仅显示，不执行操作）" : (completed ? "本次连接的踏板测试已完成" : (Number(state.foot_pad_count || 0) + " 个踏板已连接"))));
    setText(
      "footPadTestDetail",
      !available
        ? (state.foot_pad_status_message || "未找到可用的 2/4/5 踏板虚拟键盘。")
        : active
      ? (twoPedal
        ? "双踏板请依次踩踏板 1、2；检测到两个按键后自动完成。测试期间只显示按键，不会固定、归位、移动机械臂或写入数采。"
        : fivePedal
        ? "五踏板请依次踩踏板 1、2、3、4、5；检测到五个按键后自动完成。测试期间只显示按键，不会触发 Fix、进度语义或写入数采。"
        : "四踏板请依次踩踏板 1、2、3、4；检测到四个按键后自动完成。测试期间只显示按键，不会固定、归位、移动机械臂或写入数采。")
        : completed
        ? "测试结果已在本次连接期间保留，不会重复要求测试；只有主动点击“重新测试踏板”或设备断开重连后才会重置。"
        : (twoPedal
          ? "F13/踏板1 进入仅左臂遥操（Fix 右臂）；F14/踏板2 进入仅右臂遥操（Fix 左臂）；双踏板不执行 Init。映射测试仅用于按需核对，不会阻塞踏板或 A/B/C 控制。"
          : fivePedal
          ? "五踏板：F13/F14 与双踏板一致，分别进入仅左/右臂遥操；F15 推进阶段；F16 记录当前阶段失败并 Hold；F17 撤销上一次进度推进。未选择 progress profile 时仅 F13/F14 可用。"
          : "F13/踏板1 固定左臂；F14/踏板2 固定右臂；F15/踏板3 左臂精确 Init qpos；F16/踏板4 右臂精确 Init qpos。映射测试仅用于按需核对，不会阻塞其他控制。")
    );
    setText("footPadTestLast", "最近检测：" + (state.foot_pad_test_last_event || "-"));
    setText("footPadTestStatus", state.foot_pad_test_status || "-");
    var start = byId("footPadTestStartButton");
    var stop = byId("footPadTestStopButton");
    if (start) {
      start.className = "warn" + (!active ? " active" : "");
      start.textContent = active ? "测试进行中" : (completed ? "重新测试踏板" : "开始测试踏板");
    }
    if (stop) stop.className = "danger" + (active ? " active" : "");
  }

  function renderEpisodeStateFallback(state) {
    var available = Boolean(state.episode_recording_available);
    var recording = Boolean(state.episode_recording);
    var preparing = String(state.episode_event_status || "") === "preparing";
    var finalizing = String(state.episode_event_status || "") === "finalizing";
    var startButton = byId("episodeStartButton");
    var umiCalibrationSection = byId("umiCalibrationSection");
    var stopButton = byId("episodeStopButton");
    var deleteInput = byId("episodeDeleteIndexInput");
    var deleteButton = byId("episodeDeleteButton");
    var statusEl = byId("episodeStatus");
    var plusSection = byId("plusCurationSection");
    var plusVariant = byId("plusVariantSelect");
    var plusDisposition = byId("plusDispositionSelect");
    var plusReason = byId("plusReasonSelect");
    var plusSave = byId("plusCurationSaveButton");
    var plusStatus = byId("plusCurationStatus");
    var plusEnabled = Boolean(state.plus_curation_enabled);
    var plusPending = Boolean(state.plus_curation_pending);
    var plusPendingToken = plusPending ? String(state.plus_curation_pending_token || "") : "";
    if (plusSection) plusSection.hidden = !plusEnabled;
    if (plusPendingToken !== fallbackPlusReviewToken) {
      fallbackPlusReviewToken = plusPendingToken;
      fallbackPlusReviewDisposition = String(state.plus_curation_disposition || "unknown") || "unknown";
      fallbackPlusReviewReason = String(state.plus_curation_reason_code || "unknown") || "unknown";
      if (plusDisposition) plusDisposition.value = fallbackPlusReviewDisposition;
      if (plusReason) plusReason.value = fallbackPlusReviewReason;
    }
    if (plusEnabled && plusVariant) {
      var variants = Array.isArray(state.plus_curation_variants) ? state.plus_curation_variants : [];
      var options = ['<option value="">未指定</option>'];
      for (var variantIndex = 0; variantIndex < variants.length; variantIndex += 1) {
        var variant = variants[variantIndex] || {};
        options.push('<option value="' + escapeHtml(variant.variant_id || "") + '">' + escapeHtml(variant.label || variant.variant_id || "未指定") + '</option>');
      }
      plusVariant.innerHTML = options.join("");
      if (fallbackPlusVariantDraft === null) fallbackPlusVariantDraft = state.plus_curation_variant_id || "";
      if (document.activeElement !== plusVariant) plusVariant.value = fallbackPlusVariantDraft;
      plusVariant.disabled = recording || preparing || finalizing || !fallbackActiveControls.episode_start;
    }
    if (plusSave) plusSave.disabled = !plusPending || !fallbackActiveControls.plus_curation_save;
    if (plusStatus) {
      var plusRoot = plusPendingRootLabelFallback(state.plus_curation_pending_dataset_root);
      plusStatus.textContent = [
        state.plus_curation_status || "",
        plusPending ? "episode " + state.plus_curation_episode_index : "",
        plusRoot ? "dataset " + plusRoot : "",
        state.plus_curation_message || ""
      ].filter(Boolean).join(" · ") || "Plus 未启用";
    }
    updatePlusReviewControlsFallback(state);
    var calibrationWarning = state.calibration_validity_warning || {};
    var warning = calibrationWarning.active
      ? calibrationWarning
      : (state.episode_frequency_warning || {});
    var warningBanner = byId("episodeWarningBanner");
    var warningText = byId("episodeWarningText");
    var warningTitle = byId("episodeWarningTitle");
    var status = String(state.episode_event_status || "");
    var paused = status === "paused" || status === "resume_pending";
    var statusLabels = {
      unavailable: "数采不可用",
      idle: "未开始数采",
      preparing: "3 秒后开始数采",
      recording: "正在数采",
      paused: "等待有效快照",
      resume_pending: "等待恢复写入",
      finalizing: "停止中 / 后台保存",
      delete_queued: "删除请求已排队",
      saved: "已保存",
      discarded: "已停止 / 无样本",
      error: "保存失败",
      deleted: "已删除并重编号"
    };
    if (!available) {
      status = "unavailable";
    } else if (recording && !paused) {
      status = "recording";
    } else if (!status) {
      status = "idle";
    }
    paused = status === "paused" || status === "resume_pending";
    if (startButton) {
      startButton.textContent = paused ? "等待有效快照" : (recording ? "数采进行中" : (preparing ? "准备中..." : "开始数采"));
      startButton.disabled = !fallbackActiveControls.episode_start || recording || preparing || finalizing;
    }
    if (stopButton) {
      stopButton.disabled = !fallbackActiveControls.episode_stop || (!recording && !preparing);
    }
    var deleteAllowed = Boolean(fallbackActiveControls.episode_delete) && !recording && !preparing && !finalizing;
    if (deleteInput) deleteInput.disabled = !deleteAllowed;
    if (deleteButton) deleteButton.disabled = !deleteAllowed;
    var runtimeConfigured = String(state.ui_mode || "") === "umi_collection";
    if (umiCalibrationSection) umiCalibrationSection.hidden = !runtimeConfigured;
    var datasetConfigurationAvailable = Boolean(fallbackActiveControls.task_select);
    var pathLocked = recording || preparing || finalizing || !datasetConfigurationAvailable;
    var locked = pathLocked || runtimeConfigured;
    var pathFields = ["datasetBaseRootInput", "datasetChooseRootButton"];
    for (var pathIndex = 0; pathIndex < pathFields.length; pathIndex += 1) {
      var pathField = byId(pathFields[pathIndex]);
      if (pathField) pathField.disabled = pathLocked;
    }
    var fields = ["taskTypeSelect", "taskSelect", "taskEnglishNameInput", "taskChineseNameInput"];
    for (var i = 0; i < fields.length; i += 1) {
      var field = byId(fields[i]);
      if (field) field.disabled = locked;
    }
    var roundTripButton = byId("roundTripToggleButton");
    if (roundTripButton) {
      roundTripButton.disabled = locked || normalizeTaskTypeFallback(state.collection_task_type) !== "round_trip" || !fallbackActiveControls.round_trip_toggle;
    }
    var roundTripStatus = byId("roundTripStatus");
    if (roundTripStatus) {
      roundTripStatus.textContent = normalizeTaskTypeFallback(state.collection_task_type) === "round_trip"
        ? "来回类: 当前将采 _" + normalizeRoundTripPhaseFallback(state.collection_round_trip_phase)
        : "一般类任务：每次采集使用同一个 task 标签。";
    }
    if (statusEl) {
      var lines = [
        available ? "episode 录制可用" : "episode 录制不可用",
        paused ? "状态: 等待有效传感器快照" : (recording ? "状态: 录制中" : (preparing ? "状态: 倒计时准备中" : "状态: 未录制"))
      ];
      if (state.episode_dir) lines.push("路径: " + state.episode_dir);
      if (state.dataset_root) lines.push("保存根目录: " + state.dataset_root);
      if (state.episode_frame_count != null) lines.push("帧数: " + state.episode_frame_count);
      statusEl.textContent = lines.join("\n");
    }
    if (warningBanner) warningBanner.hidden = !Boolean(warning.active);
    if (warningBanner) {
      warningBanner.className = "ik-warning-banner" + (calibrationWarning.active
        ? " calibration-warning " + String(calibrationWarning.level || "warning")
        : "");
    }
    if (warningTitle) warningTitle.textContent = warning.title || "DATA WARNING";
    if (warningText) warningText.textContent = warning.message || "采集频率低于目标值";
    var banner = byId("episodeBanner");
    if (banner) {
      var bannerClass = "episode-banner";
      if (/^(preparing|recording|paused|resume_pending|finalizing|saved|discarded|error|unavailable)$/.test(status)) {
        bannerClass += " " + (status === "discarded" ? "finalizing" : status);
      }
      banner.className = bannerClass;
    }
    setText("episodeBannerState", statusLabels[status] || status || "未开始数采");
    var frameText = state.episode_frame_count == null ? "-" : String(state.episode_frame_count) + " 帧";
    var hzText = "";
    var sampleHz = Number(state.episode_sample_hz);
    var targetHz = Number(state.episode_target_hz);
    if (isFinite(sampleHz)) {
      hzText = sampleHz.toFixed(1) + " Hz" + (isFinite(targetHz) ? " / " + targetHz.toFixed(1) + " Hz" : "");
    }
    setText("episodeBannerCounter", hzText ? frameText + " · " + hzText : frameText);
    var detail = state.episode_event_message || (
      paused
        ? "当前 episode 保持打开；恢复遥操且输入稳定后会继续写入。"
        : recording
        ? "正在写入正式数据集；结束前请确认画面、触觉和机械臂状态正常。"
        : available
          ? "点击“开始数采”或按 A 开始；再次按 A 停止。"
          : "当前启动参数禁用了正式数据集和 debug episode 录制。"
    );
    var path = state.episode_dir || state.dataset_root || "";
    setText("episodeBannerDetail", detail + (path ? "\n路径: " + path : ""));
    setText("episodeBannerMeta", state.collection_task_name || state.collection_task_id || "");
  }

  function renderMessages(messages) {
    var el = byId("messages");
    if (!el) return;
    if (!messages || !messages.length) {
      el.innerHTML = '<div class="message">等待更新...</div>';
      return;
    }
    var html = [];
    for (var i = 0; i < messages.length; i += 1) {
      html.push(
        '<div class="message"><span class="tag">[' +
        escapeHtml(messages[i].tag || "LOG") +
        ']</span>' +
        escapeHtml(messages[i].message || "") +
        '</div>'
      );
    }
    el.innerHTML = html.join("");
  }

  function maybePromptTrackerSideConfirmation(state) {
    var check = state.tracker_side_check || {};
    var token = String(check.confirmation_token || "");
    if (!check.confirmation_required || !token || token === fallbackTrackerSideConfirmationToken) {
      return;
    }
    fallbackTrackerSideConfirmationToken = token;
    var message = String(
      check.confirmation_message ||
      check.message ||
      "Tracker 左右位置检查未通过。是否仍要继续本次 runtime Home？"
    );
    postEvent(
      window.confirm(message)
        ? "tracker_side_confirm_continue"
        : "tracker_side_confirm_cancel"
    );
  }

  function renderRawPoses(state) {
    var poses = state.raw_tracker_poses || {};
    var left = rawPoseText(poses.left);
    var right = rawPoseText(poses.right);
    setText("leftRawStatus", left.status);
    setText("rightRawStatus", right.status);
    setText("leftRawPose", left.text);
    setText("rightRawPose", right.text);
    setRawPoseClass("leftRawCard", poses.left);
    setRawPoseClass("rightRawCard", poses.right);
  }

  function poseQualityLevelFallback(quality) {
    quality = quality || {};
    var level = String(quality.level || "");
    if (/^(ok|warn|error|off)$/.test(level)) return level;
    var status = String(quality.status || "unknown");
    if (status === "bad" || quality.bad) return "error";
    if (status === "warning") return "warn";
    if (status === "good") return "ok";
    return "off";
  }

  function formatPoseQualityMetricFallback(value, digits, unit) {
    var numeric = Number(value);
    if (!isFinite(numeric)) return "-";
    return numeric.toFixed(digits) + unit;
  }

  function renderPoseQualityFallback(state) {
    var quality = state.pose_quality || {};
    var metrics = quality.metrics || {};
    var card = byId("poseQualityCard");
    if (card) card.className = "device-card " + poseQualityLevelFallback(quality);
    var score = quality.score == null || !isFinite(Number(quality.score)) ? "-" : Number(quality.score).toFixed(2);
    setText("poseQualityStatus", quality.status || "unknown");
    setText("poseQualityScore", "score " + score);
    var reasons = Array.isArray(quality.reasons) ? quality.reasons : [];
    var badReasons = Array.isArray(quality.bad_reasons) ? quality.bad_reasons : [];
    var warningReasons = Array.isArray(quality.warning_reasons) ? quality.warning_reasons : [];
    var lines = [];
    lines.push("available: " + (quality.available ? "yes" : "no"));
    if (badReasons.length) lines.push("bad: " + badReasons.join(", "));
    if (warningReasons.length) lines.push("warning: " + warningReasons.join(", "));
    if (!badReasons.length && !warningReasons.length) {
      lines.push("reasons: " + (reasons.length ? reasons.join(", ") : "none"));
    }
    lines.push("source: " + (quality.source || "-"));
    lines.push(
      "collision: " + (metrics.collision_avoidance_status || "-") +
      " dist=" + formatPoseQualityMetricFallback(metrics.collision_distance_m, 3, "m") +
      " min=" + formatPoseQualityMetricFallback(metrics.collision_min_distance_m, 3, "m") +
      " margin=" + formatPoseQualityMetricFallback(metrics.collision_margin_m, 3, "m")
    );
    lines.push(
      "ik: jump=" + (metrics.ik_jump_gate_status || "-") +
      " branch=" + (metrics.ik_branch_candidate || "-") +
      " err=" + formatPoseQualityMetricFallback(metrics.ik_translation_error_m, 3, "m") +
      "/" + formatPoseQualityMetricFallback(metrics.ik_rotation_error_deg, 1, "deg")
    );
    lines.push(
      "joint: nearest_margin=" + formatPoseQualityMetricFallback(metrics.joint_nearest_margin_deg, 1, "deg") +
      " near_limit=" + (metrics.joint_near_limit ? "yes" : "no")
    );
    setText("poseQualityDetail", lines.join("\n"));
  }

  function statusLevelClass(value) {
    var level = String(value || "ok");
    return /^(ok|warn|error|waiting|off)$/.test(level) ? level : "ok";
  }

  function renderDeviceStatus(containerId, status, emptyText, titlePrefix) {
    var el = byId(containerId);
    if (!el) return;
    var entries = [];
    var key;
    status = status || {};
    for (key in status) {
      if (Object.prototype.hasOwnProperty.call(status, key)) entries.push([key, status[key] || {}]);
    }
    entries.sort(function (a, b) { return String(a[0]).localeCompare(String(b[0])); });
    if (!entries.length) {
      el.innerHTML = '<div class="device-card off"><div class="device-title"><span>未配置</span><span>off</span></div><div class="device-body mono">' +
        escapeHtml(emptyText) + '</div></div>';
      return;
    }
    var html = [];
    for (var i = 0; i < entries.length; i += 1) {
      var name = entries[i][0];
      var item = entries[i][1] || {};
      var lines = [];
      if (item.serial || item.serial_number) lines.push("serial: " + (item.serial_number || item.serial));
      if (item.frame_count != null) lines.push("frames: " + item.frame_count + " age=" + (item.last_frame_age_sec == null ? "-" : Number(item.last_frame_age_sec).toFixed(2) + " s"));
      if (item.depth_required || item.depth_enabled) {
        var depthWidth = item.depth_width == null ? "-" : String(item.depth_width);
        var depthHeight = item.depth_height == null ? "-" : String(item.depth_height);
        var depthFps = item.depth_fps == null ? "-" : String(item.depth_fps);
        var depthScale = item.depth_scale_m_per_unit == null ? "-" : Number(item.depth_scale_m_per_unit).toFixed(3);
        lines.push(
          "depth: " + (item.depth_status_label || "-") +
          " required=" + (item.depth_required === true ? "yes" : item.depth_required === false ? "no" : "-") +
          " present=" + (item.depth_frame_present === true ? "yes" : item.depth_frame_present === false ? "no" : "-") +
          " profile=" + depthWidth + "x" + depthHeight + "@" + depthFps +
          " scale=" + depthScale
        );
        lines.push(
          "depth meta: frame=" + (item.depth_frame_number == null ? "-" : item.depth_frame_number) +
          " ts=" + (item.depth_camera_timestamp_sec == null ? "-" : Number(item.depth_camera_timestamp_sec).toFixed(3)) +
          " sensor_ms=" + (item.depth_sensor_timestamp_ms == null ? "-" : Number(item.depth_sensor_timestamp_ms).toFixed(3)) +
          " repeat=" + (item.depth_signature_repeat_count == null ? "-" : item.depth_signature_repeat_count) +
          " unique=" + (item.depth_signature_unique_count == null ? "-" : item.depth_signature_unique_count) +
          " age=" + (item.depth_timestamp_age_sec == null ? "-" : Number(item.depth_timestamp_age_sec).toFixed(2) + " s")
        );
        if (item.depth_status_message) lines.push("depth status: " + item.depth_status_message);
      }
      if (item.device_count != null || item.ready_count != null) lines.push("ready: " + (item.ready_count == null ? "-" : item.ready_count) + "/" + (item.ready_total || item.device_count || "-"));
      if (item.status_message) lines.push("status: " + item.status_message);
      if (item.last_error) lines.push("error: " + item.last_error);
      if (item.telemetry_error) lines.push("error: " + item.telemetry_error);
      html.push(
        '<div class="device-card ' + statusLevelClass(item.status_level) + '">' +
        '<div class="device-title"><span>' + escapeHtml(titlePrefix ? titlePrefix + name : name) + '</span><span>' +
        escapeHtml(item.status_label || item.status_level || "-") + '</span></div>' +
        '<div class="device-body mono">' + escapeHtml(lines.join("\n") || "-") + '</div></div>'
      );
    }
    el.innerHTML = html.join("");
  }

  function renderCameraSelfCheckFallback(state) {
    var section = byId("cameraSelfCheckSection");
    var card = byId("cameraSelfCheckCard");
    var statusEl = byId("cameraSelfCheckStatus");
    var detailEl = byId("cameraSelfCheckDetail");
    if (!section || !card) return;
    var check = state.camera_self_check || {};
    var status = String(check.status || "idle");
    var enabled = !!check.enabled || status !== "idle";
    section.hidden = !enabled;
    if (!enabled) return;
    var level = check.status_level || (status === "failed" ? "error" : status === "passed" ? "ok" : "waiting");
    card.className = "device-card " + statusLevelClass(level);
    if (statusEl) statusEl.textContent = check.status_label || status;
    if (!detailEl) return;
    var lines = [
      "status: " + (check.message || "-"),
      "elapsed: " + (check.elapsed_sec == null ? "-" : Number(check.elapsed_sec).toFixed(2) + " s") +
        " / " + (check.duration_sec == null ? "-" : Number(check.duration_sec).toFixed(2) + " s"),
      "cameras: " + (check.passed_count == null ? "-" : check.passed_count) +
        "/" + (check.camera_count == null ? "-" : check.camera_count)
    ];
    if (check.failed_keys && check.failed_keys.length) lines.push("failed: " + check.failed_keys.join(", "));
    if (check.waiting_keys && check.waiting_keys.length) lines.push("waiting: " + check.waiting_keys.join(", "));
    if (check.warn_keys && check.warn_keys.length) lines.push("warn: " + check.warn_keys.join(", "));
    detailEl.textContent = lines.join("\n");
  }

  function renderState(state) {
    if (!state || window.__calibrationWebUiModernRefreshOk || window.__calibrationWebUiLegacyPollActive) return;
    fallbackLatestState = state || {};
    if (typeof window.dexTeleopRenderInputActivity === "function") window.dexTeleopRenderInputActivity(state);
    setText("stage", labelStage(state.stage || "Starting..."));
    setText("selectedSide", state.selected_side === "left" ? "左手" : state.selected_side === "right" ? "右手" : state.selected_side);
    setText("frameVizMode", state.frame_viz_mode);
    setText("virtualDeskYaw", state.virtual_desk_post_yaw_deg != null ? String(state.virtual_desk_post_yaw_deg) + " deg" : "-");
    setText("controlHint", state.control_hint || "-");
    var twoPedalAlert = state.two_pedal_alert || {};
    var twoPedalAlertEl = byId("twoPedalAlert");
    if (twoPedalAlertEl) {
      twoPedalAlertEl.hidden = !Boolean(twoPedalAlert.active);
      twoPedalAlertEl.className = "two-pedal-alert" + (String(twoPedalAlert.level || "") === "red" ? " red" : "");
      twoPedalAlertEl.textContent = twoPedalAlert.message || "";
    }
    setText("captureStage", labelStage(state.capture_stage));
    setText("captureSamples", captureSamplesText(state));
    setText("captureValid", captureValidText(state));
    setText("captureStatus", state.teleop_running ? "遥操中" : labelPhase(state.capture_phase || "idle"));
    renderCaptureMode(state);
    renderRuntimeHomeDiagnosticFallback(state);
    renderButtons(state);
    renderRawPoses(state);
    renderPoseQualityFallback(state);
    renderEpisodeStateFallback(state);
    if (typeof window.__dexteleopRenderProgressUi === "function") {
      window.__dexteleopRenderProgressUi(state);
    }
    renderCameraSelfCheckFallback(state);
    renderDeviceStatus("cameraCards", state.camera_status, "当前没有 RealSense camera_sources。", "");
    renderDeviceStatus("tactileCameraCards", state.tactile_camera_status, "当前没有 DMTac 视触觉状态。", "");
    renderMessages(state.messages || []);
    maybePromptTrackerSideConfirmation(state);
    setHealth("外部兼容脚本正在刷新 state.json；主内联脚本未接管，通常是浏览器策略、扩展或嵌入式页面拦截了 inline script。");
  }

  function fetchState() {
    if (window.__calibrationWebUiModernRefreshOk || window.__calibrationWebUiLegacyPollActive) return;
    try {
      var xhr = new XMLHttpRequest();
      xhr.open("GET", "/state.json?external_ts=" + new Date().getTime(), true);
      xhr.onreadystatechange = function () {
        if (xhr.readyState !== 4) return;
        if (xhr.status !== 200) {
          setHealth("外部兼容脚本请求 state.json 失败: HTTP " + xhr.status);
          return;
        }
        try {
          renderState(JSON.parse(xhr.responseText));
        } catch (error) {
          setHealth("外部兼容脚本解析 state.json 失败: " + error);
        }
      };
      xhr.send(null);
    } catch (error) {
      setHealth("外部兼容脚本请求 state.json 失败: " + error);
    }
  }

  function collectEpisodeStartPayloadFallback() {
    var ignoredCamera = Array.prototype.map.call(document.querySelectorAll('input[data-ignore-modality="camera"]:checked'), function (input) { return input.value; });
    var ignoredTactile = Array.prototype.map.call(document.querySelectorAll('input[data-ignore-modality="tactile"]:checked'), function (input) { return input.value; });
    var payload = {
      dataset_base_root: byId("datasetBaseRootInput") ? byId("datasetBaseRootInput").value : "",
      task_type: byId("taskTypeSelect") ? byId("taskTypeSelect").value : normalizeTaskTypeFallback(fallbackLatestState.collection_task_type),
      task_key: byId("taskSelect") ? byId("taskSelect").value : "",
      task_english_name: byId("taskEnglishNameInput") ? byId("taskEnglishNameInput").value : "",
      task_chinese_name: byId("taskChineseNameInput") ? byId("taskChineseNameInput").value : "",
      round_trip_phase: normalizeRoundTripPhaseFallback(fallbackLatestState.collection_round_trip_phase),
      ignored_camera_keys: ignoredCamera,
      ignored_tactile_keys: ignoredTactile
    };
    if (Boolean(fallbackLatestState.plus_curation_enabled) && byId("plusVariantSelect")) {
      payload.plus_variant_id = byId("plusVariantSelect").value || "";
    }
    return payload;
  }

  function postEvent(eventName, extraPayload) {
    try {
      var xhr = new XMLHttpRequest();
      var payload = {event: eventName};
      var key;
      if (extraPayload) {
        for (key in extraPayload) {
          if (Object.prototype.hasOwnProperty.call(extraPayload, key)) {
            payload[key] = extraPayload[key];
          }
        }
      }
      xhr.open("POST", "/event", true);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.onreadystatechange = function () {
        if (xhr.readyState !== 4) return;
        if (xhr.status >= 200 && xhr.status < 300) {
          setText("sendStatus", eventName === "episode_delete" ? "删除请求已接收，正在处理..." : "已发送: " + eventName);
        } else if (xhr.status >= 400) {
          setText("sendStatus", "事件未生效: HTTP " + xhr.status);
        }
      };
      xhr.send(JSON.stringify(payload));
    } catch (error) {
      setHealth("外部兼容脚本发送按钮事件失败: " + error);
    }
  }

  function chooseDatasetDirectoryFallback() {
    var input = byId("datasetBaseRootInput");
    var button = byId("datasetChooseRootButton");
    if (!input || !button || button.disabled) return;
    var xhr = new XMLHttpRequest();
    button.disabled = true;
    xhr.open("GET", "/choose-directory?initial=" + encodeURIComponent(input.value || ""), true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) return;
      button.disabled = false;
      if (xhr.status !== 200) {
        setText("sendStatus", "选择文件夹失败: HTTP " + xhr.status);
        return;
      }
      try {
        var result = JSON.parse(xhr.responseText);
        if (result.path) {
          input.value = result.path;
          setText("sendStatus", "已选择文件夹: " + result.path);
        } else {
          setText("sendStatus", result.message || "未选择文件夹");
        }
      } catch (error) {
        setText("sendStatus", "选择文件夹结果解析失败: " + error);
      }
    };
    xhr.send(null);
  }

  function isEditableTargetFallback(target) {
    var tagName = target && target.tagName ? String(target.tagName).toUpperCase() : "";
    return Boolean(target && (target.isContentEditable || tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT"));
  }

  function postKeyboardShortcutFallback(key) {
    var state = fallbackLatestState || {};
    var eventName = "";
    var payload = null;
    if (key === "a") {
      eventName = (
        fallbackActiveControls["episode_stop"]
        && (
          state.episode_recording
          || String(state.episode_event_status || "") === "preparing"
          || !fallbackActiveControls["episode_start"]
        )
      ) ? "episode_stop" : (fallbackActiveControls["episode_start"] ? "episode_start" : "");
      if (eventName === "episode_start") payload = collectEpisodeStartPayloadFallback();
    } else if (key === "b") {
      eventName = fallbackActiveControls["two_pedal_b"]
        ? "two_pedal_b"
        : (fallbackActiveControls["h"] ? "h" : (fallbackActiveControls["p"] ? "p" : ""));
    } else if (key === "c") {
      if (fallbackActiveControls["two_pedal_c"]) {
        eventName = "two_pedal_c";
      } else if (state.fix_arm_available && state.teleop_running && fallbackActiveControls["fix_arm_toggle"]) {
        eventName = "fix_arm_toggle";
      }
    } else if (key === "g") {
      if (state.init_arm_available && state.teleop_running && fallbackActiveControls["init_arm_toggle"]) {
        eventName = "init_arm_toggle";
      }
    } else {
      return false;
    }
    if (!eventName) {
      setText("sendStatus", "快捷键 " + key.toUpperCase() + " 当前不可用");
      return true;
    }
    if (!fallbackActiveControls[eventName]) {
      setText("sendStatus", "快捷键 " + key.toUpperCase() + " 当前不可用");
      return true;
    }
    postEvent(eventName, payload);
    setText("sendStatus", "快捷键 " + key.toUpperCase() + " 已发送");
    return true;
  }

  function bindKeyboard() {
    document.onkeydown = function (event) {
      event = event || window.event;
      if (event.repeat || isEditableTargetFallback(event.target || event.srcElement)) return;
      var key = String(event.key || "").toLowerCase();
      var code = String(event.code || "").toLowerCase();
      if (key !== "a" && key !== "b" && key !== "c" && key !== "g") {
        key = code === "keya" ? "a" : code === "keyb" ? "b" : code === "keyc" ? "c" : code === "keyg" ? "g" : key;
      }
      if (key !== "a" && key !== "b" && key !== "c" && key !== "g") return;
      if (event.preventDefault) event.preventDefault();
      event.returnValue = false;
      postKeyboardShortcutFallback(key);
    };
  }

  function bindButtons() {
    var buttons = document.querySelectorAll ? document.querySelectorAll("button[data-event]") : [];
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].onclick = function () {
        var eventName = this.getAttribute("data-event");
        if (
          eventName === "c" &&
          !(fallbackLatestState.fix_arm_available && fallbackLatestState.teleop_running)
        ) {
          setText("sendStatus", "C 仅用于运行中的 fix-arm");
          return;
        }
        var sendEventName = eventName === "c" ? "fix_arm_toggle" : eventName;
        var extraPayload = (
          eventName === "episode_start" || eventName === "task_select"
        ) ? collectEpisodeStartPayloadFallback() : null;
        if (eventName === "episode_delete") {
          var deleteInput = byId("episodeDeleteIndexInput");
          var deleteIndex = deleteInput ? String(deleteInput.value || "").trim() : "";
          if (!/^[0-9]+$/.test(deleteIndex)) {
            setText("sendStatus", "请输入非负整数 Episode 序号");
            return;
          }
          if (!window.confirm("确认删除 episode " + deleteIndex + " 并重编号后续数据？")) return;
          extraPayload = {episode_index: deleteIndex};
        }
        if (eventName === "plus_curation_save") {
          if (fallbackLatestState.plus_curation_episode_index == null) return;
          extraPayload = {
            episode_index: fallbackLatestState.plus_curation_episode_index,
            pending_token: fallbackLatestState.plus_curation_pending_token || "",
            disposition: byId("plusDispositionSelect") ? byId("plusDispositionSelect").value : "unknown",
            reason_code: byId("plusReasonSelect") ? byId("plusReasonSelect").value : "unknown"
          };
        }
        if (eventName === "init_arm_move_to_qpos" && !window.confirm(
          "确认精确归位所选机械臂？只有所选臂移动，另一臂保持为静态碰撞体。到位后需重采该侧 Home 才会恢复普通遥操。请确认周围无人且没有接触负载。"
        )) return;
        if (eventName === "init_arm_move_both_to_qpos" && !window.confirm(
          "确认按左臂 → 右臂顺序执行双臂精确归位？过程中不会修改 tracker Home；启动前执行时到位后停留在启动前 Hold，仍需另行点击“开始遥操”采集 runtime Home。任一步失败都会中止剩余动作并保持停止。请确认两臂周围无人且没有接触负载。"
        )) return;
        postEvent(sendEventName, extraPayload);
      };
    }
    var initSpeedApply = byId("initArmQposSpeedApplyButton");
    if (initSpeedApply) {
      initSpeedApply.onclick = function () {
        if (initSpeedApply.disabled) return;
        var minValue = Number(fallbackLatestState.init_arm_qpos_speed_scale_min == null ? 0.1 : fallbackLatestState.init_arm_qpos_speed_scale_min);
        var maxValue = Number(fallbackLatestState.init_arm_qpos_speed_scale_max == null ? 10.0 : fallbackLatestState.init_arm_qpos_speed_scale_max);
        var leftInput = byId("leftInitArmQposSpeedInput");
        var rightInput = byId("rightInitArmQposSpeedInput");
        var leftValue = Number(leftInput && leftInput.value);
        var rightValue = Number(rightInput && rightInput.value);
        if (!isFinite(leftValue) || !isFinite(rightValue) || leftValue < minValue || leftValue > maxValue || rightValue < minValue || rightValue > maxValue) {
          setText("sendStatus", "左右臂归位速度倍率必须在 " + minValue + " 到 " + maxValue + " 之间");
          return;
        }
        postEvent("init_arm_qpos_speed_set:" + JSON.stringify({left: leftValue, right: rightValue}));
      };
    }
    var plusVariant = byId("plusVariantSelect");
    if (plusVariant) {
      plusVariant.onchange = function () {
        fallbackPlusVariantDraft = plusVariant.value || "";
      };
    }
    var plusDisposition = byId("plusDispositionSelect");
    if (plusDisposition) {
      plusDisposition.onchange = function () {
        fallbackPlusReviewDisposition = plusDisposition.value || "unknown";
        if (fallbackPlusReviewDisposition !== "keep_recovery" && fallbackPlusReviewDisposition !== "exclude") {
          fallbackPlusReviewReason = "unknown";
          var plusReason = byId("plusReasonSelect");
          if (plusReason) plusReason.value = "unknown";
        }
        updatePlusReviewControlsFallback(fallbackLatestState);
      };
    }
    var plusReason = byId("plusReasonSelect");
    if (plusReason) {
      plusReason.onchange = function () {
        fallbackPlusReviewReason = plusReason.value || "unknown";
      };
    }
    var chooseButton = byId("datasetChooseRootButton");
    if (chooseButton) chooseButton.onclick = chooseDatasetDirectoryFallback;
  }

  if (window.__calibrationWebUiModernRefreshOk || window.__calibrationWebUiLegacyPollActive) {
    return;
  }
  window.__calibrationWebUiExternalFallbackActive = true;
  setHealth("外部兼容脚本已启动，正在接管 state.json 实时刷新...");
  bindButtons();
  bindKeyboard();
  fetchState();
  window.setInterval(fetchState, 500);
})();
