"use strict";

// Human intervention is a declaration, never sensing. Only an explicitly
// bound probe app may also withdraw on true; declarations never renew leases.
(() => {
  const node = id => document.getElementById(id);
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  let latest = null;
  let pending = false;
  const rows = Array.from({length: 6}, (_, index) => {
    const row = document.createElement("tr");
    row.append(...Array.from({length: 8}, () => document.createElement("td")));
    row.cells[0].textContent = "J" + (index + 1);
    node("joint-rows").append(row);
    return row;
  });
  const fixed = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : "—";

  function render() {
    node("human-context-note").textContent = latest?.controller_mode === "automatic_calibration"
      ? "自动标定：有人施加外力会终止本批驱动，不会自动恢复；开始前清空运动空间，保持物理急停可用。没有自动接触检测。"
      : latest?.onset_capture?.probe
      ? "带力观察模式：有人施加外力会撤销附加助力；无人声明只允许随后手动准备，不自动恢复旧目标。接管前先撤力，物理急停优先。"
      : "拖动、托住或阻挡都算有外力；此选项只标记数据。接管前先撤销驱动力。";
    const journal = latest?.transparency_assessment;
    const available = journal?.available && journal.automatic;
    for (const [id, human] of [["human-no", false], ["human-yes", true]]) {
      node(id).disabled = pending || !available;
      node(id).setAttribute("aria-pressed", String(journal?.human_intervention === human));
    }
    const context = journal?.context;
    if (!latest) node("assessment-state").textContent = "页面连接中断；检查终端状态。";
    else if (!available) node("assessment-state").textContent = journal?.error || "等待自动记录服务…";
    else if (journal.human_intervention === null) node("assessment-state").textContent = "自动记录已开启。请点击上方选项，声明当前是否有人施加外力。";
    else {
      const drive = latest.operator_armed || latest.phase === "armed";
      const human = journal.human_intervention ? "有人施加外力" : "无人施加外力";
      const control = latest.controller_mode === "automatic_calibration"
        ? (latest.automatic_calibration.active || latest.automatic_calibration.start_requested
          ? "自动标定（含支撑轴阶段阻尼）" : "批次未驱动；请核对 owner 结束及基础参数回执")
        : latest.controller_mode === "staged_assist"
        ? (drive ? "允许附加驱动力" : "仅重力补偿 / 正在撤力")
        : directDriveLabel(latest);
      node("assessment-state").textContent = human + " · " + control + " · 自动记录第 " + (context?.context_id ?? 0) + " 段";
    }
    node("assessment-path").textContent = journal ? "记录日志：" + journal.journal_path : "";
    const commands = latest?.last_dispatch?.joint_commands;
    const dispatchFresh = Number.isFinite(latest?.dispatch_age_sec)
      && latest.dispatch_age_sec >= 0 && latest.dispatch_age_sec <= 0.25;
    const feedbackAge = latest?.feedback_age_sec;
    const complete = latest?.qpos_rad?.length === 6 && latest.qpos_rad.every(Number.isFinite);
    const feedbackFresh = complete && Number.isFinite(feedbackAge) && feedbackAge >= 0 && feedbackAge <= 0.25;
    node("joint-overview").classList.toggle("stale-feedback", !feedbackFresh);
    node("joint-feedback-status").textContent = !latest
      ? "连接中断：角度已清空，请检查终端。"
      : !complete ? "等待完整六轴反馈；角度尚不可用。"
      : feedbackFresh ? `六轴反馈更新中 · 反馈年龄 ${fixed(feedbackAge, 3)} s（不是控制周期保证）`
      : `反馈已过期或时间无效 · 年龄 ${fixed(feedbackAge, 3)} s；下表角度为最近值，不是实时测量。`;
    rows.forEach((row, i) => {
      const q = complete ? latest.qpos_rad[i] : null;
      const anchor = context?.start_qpos_rad?.[i];
      const delta = Number.isFinite(q) && Number.isFinite(anchor) ? q - anchor : null;
      const values = [Number.isFinite(q) ? q * 180 / Math.PI : null, q, latest?.qvel_fd_rad_s?.[i], delta,
        dispatchFresh ? commands?.gravity_nm?.[i] : null,
        dispatchFresh ? commands?.overlay_nm?.[i] : null,
        dispatchFresh ? commands?.total_nm?.[i] : null];
      values.forEach((value, j) => { row.cells[j + 1].textContent = fixed(value, j === 0 || j >= 4 ? 3 : 5); });
      row.classList.toggle("selected-joint", latest?.joint_name === "joint" + (i + 1));
    });
    node("joint-observation").textContent = dispatchFresh
      ? "位移以当前自动分段的起点为零；分析会按人工外力和驱动力状态区分每段。"
      : "尚无新鲜的六轴发送回执；力矩列暂不显示。";
  }

  async function declare(human) {
    if (pending) return;
    pending = true;
    render();
    try {
      const response = await fetch("/api/assessment", {method: "POST",
        headers: {"Content-Type": "application/json", "X-B601-Console-Token": csrf},
        body: JSON.stringify({action: "context", human_intervention: human})});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "HTTP " + response.status);
      node("assessment-message").textContent = "外力状态已保存：" + (human ? "有人施加外力" : "无人施加外力") + "。按实际情况随时切换。";
    } catch (error) {
      node("assessment-message").textContent = "外力状态未确认保存：" + error.message;
    } finally {
      pending = false;
      render();
    }
  }
  node("human-no").onclick = () => declare(false);
  node("human-yes").onclick = () => declare(true);
  document.addEventListener("b601-status", event => { latest = event.detail; render(); });
  render();
})();
