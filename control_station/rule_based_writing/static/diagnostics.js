(function installRuleWritingDiagnostics(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.RuleWritingDiagnostics = api;
}(typeof globalThis === "object" ? globalThis : window, () => {
  "use strict";

  const DOMAIN_LABELS = Object.freeze({
    gripper: "夹爪",
    camera: "摄像头",
    arm_force: "机械臂受力安全",
    arm_ik: "机械臂解 IK",
    arm_motion: "机械臂运动安全",
    dataset_storage: "数据采集空间",
    dataset_writer: "数据集 Writer",
    tactile: "触觉采集",
    input: "操作输入",
    session: "运行时安全状态",
    configuration: "采集配置",
    unknown: "其他运行问题",
  });

  const SEVERITY_LABELS = Object.freeze({
    critical: "急停 / 严重",
    error: "阻塞",
    warning: "警告",
    info: "信息",
  });

  const SEVERITY_ORDER = Object.freeze({critical: 0, error: 1, warning: 2, info: 3});

  const normalize = (value) => String(value == null ? "" : value).trim();
  const lower = (value) => normalize(value).toLowerCase();
  const containsAny = (value, terms) => terms.some((term) => value.includes(term));

  function compact(text, maxLength = 220) {
    const value = normalize(text).replace(/\s+/g, " ");
    return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}…`;
  }

  function reachabilityMode({key = "", message = "", waitingReason = "", reachabilityStatus = ""}) {
    const evidence = `${lower(key)} ${lower(message)}`;
    if (!containsAny(evidence, ["writing_reachability", "书写可达性", "writing reachability"])) {
      return null;
    }
    const waiting = lower(waitingReason);
    const status = lower(reachabilityStatus);
    if (
      status === "error"
      || waiting === "writing_reachability_error"
      || containsAny(evidence, [
        "未知结果", "后台预检失败", "下一候选准备失败", "检查预检实现",
        "worker crash", "brokenprocesspool",
      ])
    ) return "error";
    if (
      status === "pending"
      || status === "skipping_candidate"
      || waiting === "writing_reachability_preflight"
      || waiting === "writing_reachability_skip"
      || containsAny(evidence, ["正在后台准备", "正在重新检查", "尚未完成可达性检查"])
    ) return "progress";
    return "candidate_rejected";
  }

  function component(domain, evidence) {
    const value = lower(evidence);
    if (domain === "camera") {
      if (containsAny(value, ["left_wrist", "left wrist", "左腕", "左手腕"])) {
        return {id: "left_wrist_camera", label: "左腕相机"};
      }
      if (containsAny(value, ["right_wrist", "right wrist", "右腕", "右手腕"])) {
        return {id: "right_wrist_camera", label: "右腕相机"};
      }
      if (containsAny(value, ["front", "前置", "正面", "主视角"])) {
        return {id: "front_camera", label: "前置相机"};
      }
      return {id: "camera_system", label: "相机系统"};
    }
    if (domain === "gripper") {
      if (containsAny(value, ["hand.left", "left_hand", "left hand", "left_gripper", "左夹爪", "左侧夹爪"])) {
        return {id: "left_gripper", label: "左夹爪"};
      }
      if (containsAny(value, ["hand.right", "right_hand", "right hand", "right_gripper", "右夹爪", "右侧夹爪"])) {
        return {id: "right_gripper", label: "右夹爪"};
      }
      return {id: "gripper_system", label: "夹爪系统"};
    }
    if (domain === "arm_ik" && containsAny(value, ["writing_reachability", "书写可达性", "writing reachability"])) {
      return {id: "right_arm_writing_path", label: "右臂书写路径"};
    }
    if (["arm_force", "arm_ik", "arm_motion"].includes(domain)) {
      const left = containsAny(value, ["left rm65", "left_arm", "left arm", "左臂"]);
      const right = containsAny(value, ["right rm65", "right_arm", "right arm", "右臂"]);
      if (left && right) return {id: "both_arms", label: "双臂"};
      if (left) return {id: "left_arm", label: "左臂"};
      if (right) return {id: "right_arm", label: "右臂"};
      return {id: "both_arms", label: "双臂 / 未指明侧别"};
    }
    if (domain === "dataset_storage") return {id: "dataset_volume", label: "正式数据集磁盘"};
    if (domain === "dataset_writer") return {id: "dataset_writer", label: "正式数据集 Writer"};
    if (domain === "tactile") return {id: "tactile_capture", label: "触觉设备 / Bundle"};
    if (domain === "input") return {id: "operator_input", label: "踏板 / 操作输入"};
    if (domain === "session") return {id: "runtime_session", label: "Session / runtime Home"};
    if (domain === "configuration") return {id: "rule_configuration", label: "本轮规则配置"};
    return {id: "unspecified", label: "未识别部件"};
  }

  function domainFor(tag, key, message) {
    const tagText = lower(tag);
    const keyText = lower(key);
    const evidence = `${tagText} ${keyText} ${lower(message)}`;
    if (containsAny(evidence, [
      "no space left", "enospc", "disk full", "free space", "remaining space",
      "min_free", "minimum free", "空间不足", "磁盘已满", "剩余空间", "可用空间",
    ])) return "dataset_storage";
    if (containsAny(evidence, [
      "writing_reachability", "nosolutionfound", "ik_exception", "inverse kinematics",
      "solver_guard", "解ik", "解 ik", "书写可达性", "joint horizon",
    ]) || tagText === "ik") return "arm_ik";
    if (tagText === "positionforceguard" || containsAny(evidence, [
      "force_read_stale", "force reading is stale", "max_force", "max_torque",
      "受力急停", "力数据", "force guard", "force limit",
    ])) return "arm_force";
    if (tagText === "emergencystop" || tagText === "jointhardlimit" || containsAny(evidence, [
      "emergency stop", "hard joint limit", "机械臂急停", "关节硬限位",
      "tcp pose envelope", "tcp姿态包络",
    ])) return "arm_motion";
    if (containsAny(evidence, ["tactile", "dmtac camera", "触觉", "bundle root"])) return "tactile";
    if (tagText === "camera" || tagText === "cameraselfcheck" || containsAny(evidence, [
      "camera.", "camera/", " camera", "realsense", "left_wrist", "right_wrist",
      "相机", "摄像头",
    ])) return "camera";
    if (tagText === "dmtac" || containsAny(evidence, [
      "hand.left", "hand.right", "left_hand", "right_hand", "gripper", "夹爪",
    ])) return "gripper";
    if (tagText === "dataset" || containsAny(evidence, [
      "dataset writer", "writer_blocked", "writer failed", "episode open", "落盘",
      "数据集 writer", "正式数采",
    ])) return "dataset_writer";
    if (tagText === "footpad" || containsAny(evidence, ["foot_pad", "foot pad", "踏板", "输入设备"])) return "input";
    if (keyText === "rig.session" || containsAny(evidence, [
      "debug_only", "runtime home", "hardware arm admission", "session state",
      "重新安全初始化",
    ])) return "session";
    if (tagText === "configuration" || containsAny(evidence, ["configuration_error", "配置无效", "invalid option"])) {
      return "configuration";
    }
    return "unknown";
  }

  function severityFor(domain, source, tag, message, planningMode) {
    const evidence = `${lower(tag)} ${lower(message)}`;
    if (planningMode === "progress") return "warning";
    if (domain === "arm_force" && containsAny(evidence, ["tripped", "急停", "stale", "超限", "latched stop"])) {
      return "critical";
    }
    if (domain === "arm_motion" || containsAny(evidence, ["emergency stop", "急停", "hard limit", "硬限位"])) {
      return "critical";
    }
    if (source === "blocker") return "error";
    if (containsAny(evidence, [
      "failed", "failure", "error", "exception", "unavailable", "blocked", "rejected",
      "timeout", "timed out", "断联", "失败", "错误", "不可用", "阻塞", "超时", "不足",
    ])) return "error";
    if (containsAny(evidence, ["warning", "paused", "waiting", "retry", "警告", "暂停", "等待", "重试"])) {
      return "warning";
    }
    return "info";
  }

  function recoveryFor(domain, key, message, planningMode) {
    const evidence = `${lower(key)} ${lower(message)}`;
    if (lower(key) === "rule.right_hand_hold_owner") {
      return {
        id: "close_right_gripper",
        label: "完成任务链中的闭紧右夹爪",
        instruction: "这不是设备断联。执行任务链的“闭紧右夹爪”，并等待实测夹持反馈建立本轮 owner。",
      };
    }
    if (domain === "arm_force" && containsAny(evidence, ["stale", "timeout", "陈旧", "超时"])) {
      return {
        id: "force_telemetry_check",
        label: "检查对应机械臂的力数据新鲜度",
        instruction: "保持 Hold，检查所标明侧别的力读取延迟与阻塞；同时排除真实碰撞，确认恢复新鲜数据后重新安全初始化。",
      };
    }
    if (planningMode === "progress") {
      return {
        id: "wait_candidate_check",
        label: "等待自动候选检查",
        instruction: "系统正在命令惰性地检查或切换候选；不会启动 Episode。无需重连设备、重做 runtime Home 或手动跳过。",
      };
    }
    if (planningMode === "error") {
      return {
        id: "inspect_reachability_preflight",
        label: "检查可达性预检实现",
        instruction: "系统保持 fail-closed，当前候选不会启动 Episode。查看原始错误与 Debug 日志；不要把预检异常当作设备断联。",
      };
    }
    const recoveries = {
      gripper: {
        id: "device_retry",
        label: "重连对应夹爪",
        instruction: "保持 Hold，确认对应夹爪串口与供电后使用“重连失败设备”；恢复后按任务链提示继续。",
      },
      camera: {
        id: "device_retry",
        label: "检查并重连对应相机",
        instruction: "确认所标明视角的连接、权限与新鲜帧；保持机械臂静止后使用“重连失败设备”。",
      },
      arm_force: {
        id: "runtime_home",
        label: "检查受力后重新安全初始化",
        instruction: "保持 Hold，先排除真实碰撞、过载或力读取陈旧；人工确认安全后重新采集 runtime Home。",
      },
      arm_ik: {
        id: "adjust_candidate",
        label: "调整或跳过轨迹候选",
        instruction: "这不是设备断联。调整字形大小、方向或书写高度，或跳过当前候选；不要用设备重连掩盖 IK 失败。",
      },
      arm_motion: {
        id: "operator_safety_check",
        label: "保持 Hold 并检查机械臂",
        instruction: "禁止继续运动；检查机械臂与工作空间，确认触发原因并完成所需安全初始化。",
      },
      dataset_storage: {
        id: "free_space",
        label: "释放或更换数据盘空间",
        instruction: "检查正式数据集与触觉 Bundle 所在磁盘；释放空间或选择合规目录后再重试，禁止在已有数据集中混采模态。",
      },
      dataset_writer: {
        id: "writer_retry",
        label: "检查 Writer 后重试",
        instruction: "检查落盘路径、队列和后台 Writer 错误；问题解除后使用当前配置重试 Writer。",
      },
      tactile: {
        id: "device_retry",
        label: "检查触觉采集链路",
        instruction: "确认触觉模式与现有数据集一致，再检查对应设备与 Bundle 路径。",
      },
      input: {
        id: "operator_input_check",
        label: "检查踏板 / 输入服务",
        instruction: "检查踏板数量、虚拟键盘服务与权限；输入故障不应改变机械臂安全状态。",
      },
      session: {
        id: "runtime_home",
        label: "完成 runtime Home",
        instruction: "按界面要求完成硬件 admission 与 runtime Home；不要把 Session 状态问题当成设备断联。",
      },
      configuration: {
        id: "adjust_configuration",
        label: "修正并重新预验证配置",
        instruction: "修正当前规则或 Writer 配置并重新预验证；配置未通过前不会开始运动。",
      },
      unknown: {
        id: "inspect_evidence",
        label: "查看原始证据",
        instruction: "保持当前安全状态，查看原始错误与 Debug 日志；不要盲目重连或重复开始。",
      },
    };
    return recoveries[domain];
  }

  function classifyDiagnostic({
    tag = "",
    key = "",
    message = "",
    source = "message",
    waitingReason = "",
    reachabilityStatus = "",
  } = {}) {
    const rawMessage = normalize(message);
    const rawKey = normalize(key);
    const rawTag = normalize(tag);
    const evidence = [rawKey, rawTag, rawMessage].filter(Boolean).join(" · ");
    const domain = domainFor(rawTag, rawKey, rawMessage);
    const planningMode = reachabilityMode({
      key: rawKey,
      message: rawMessage,
      waitingReason,
      reachabilityStatus,
    });
    const target = component(domain, evidence);
    const severity = severityFor(domain, source, rawTag, rawMessage, planningMode);
    return {
      id: `${source}:${rawKey || rawTag || "unknown"}:${target.id}`,
      source,
      sourceKey: rawKey,
      sourceTag: rawTag,
      domain,
      domainLabel: DOMAIN_LABELS[domain],
      component: target.id,
      componentLabel: target.label,
      severity,
      severityLabel: SEVERITY_LABELS[severity],
      summary: compact(rawMessage || rawKey || rawTag || "未提供错误详情"),
      evidence: rawMessage || rawKey || rawTag || "未提供错误详情",
      planningMode,
      recovery: recoveryFor(domain, rawKey, rawMessage, planningMode),
    };
  }

  function buildActiveDiagnostics(state = {}) {
    const diagnostics = [];
    const blockers = state.startup_blockers && typeof state.startup_blockers === "object"
      ? state.startup_blockers
      : {};
    const rehomeAvailable = Boolean(state.can_runtime_home) && containsAny(
      lower(blockers["rig.session"]),
      ["runtime home", "runtime_home", "session state="],
    );
    Object.entries(blockers).forEach(([key, message]) => {
      const diagnostic = classifyDiagnostic({
        key,
        message,
        source: "blocker",
        waitingReason: state.waiting_reason,
        reachabilityStatus: state.writing_reachability_status,
      });
      if (rehomeAvailable && diagnostic.planningMode === "progress") {
        diagnostic.recovery = {
          id: "runtime_home_then_wait_candidate_check",
          label: "先重新安全初始化",
          instruction: "当前 Session 需要重采 runtime Home；确认机械臂安全后点击“重新安全初始化”。Home 完成后再等待候选可达性检查，未通过前不会启动 Episode。",
        };
      }
      diagnostics.push(diagnostic);
    });
    if (normalize(state.configuration_error)) {
      diagnostics.push(classifyDiagnostic({
        key: "configuration_error",
        tag: "Configuration",
        message: state.configuration_error,
        source: "blocker",
      }));
    }
    if (normalize(state.height_calibration_error)) {
      diagnostics.push(classifyDiagnostic({
        key: "height_calibration_error",
        tag: "Configuration",
        message: state.height_calibration_error,
        source: "blocker",
      }));
    }
    if (state.writer_blocked) {
      diagnostics.push(classifyDiagnostic({
        key: "dataset.writer",
        tag: "Dataset",
        message: state.detail || "Dataset writer is blocked.",
        source: "blocker",
      }));
    }
    if (
      diagnostics.length === 0
      && ["blocked", "aborted"].includes(lower(state.phase))
      && normalize(state.detail)
    ) {
      diagnostics.push(classifyDiagnostic({
        key: `runtime.phase.${lower(state.phase)}`,
        message: state.detail,
        source: "blocker",
      }));
    }
    const seen = new Set();
    return diagnostics
      .filter((item) => {
        const identity = `${item.domain}|${item.component}|${item.summary}`;
        if (seen.has(identity)) return false;
        seen.add(identity);
        return true;
      })
      .sort((left, right) => SEVERITY_ORDER[left.severity] - SEVERITY_ORDER[right.severity]);
  }

  return Object.freeze({
    buildActiveDiagnostics,
    classifyDiagnostic,
    domainLabels: DOMAIN_LABELS,
    severityLabels: SEVERITY_LABELS,
  });
}));
