"""Status and opt-in typed acceptance surface for Control Station Sim-HIL.

The default server remains read-only.  An explicitly supplied acceptance
submitter enables a local POST endpoint that only queues typed application
intents; the HTTP thread never starts devices, steps simulation, changes
authority, or finalizes capture itself.
"""

from __future__ import annotations

import hmac
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .sim_hil_profile.status import SIM_HIL_STATUS_SCHEMA_V1


StatusProvider = Callable[[], dict[str, Any]]
LiveStatusProvider = Callable[[], dict[str, Any]]
IntentSubmitter = Callable[[Mapping[str, Any]], dict[str, object]]
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>DexTeleop · 仿真在环</title>
  <style>
    :root {
      --bg: #0a0f1c; --panel: #111827; --panel-2: #172033; --line: #26344d;
      --text: #e5edf8; --muted: #91a0b8; --blue: #73b7ff; --green: #69d39a;
      --amber: #f3bd62; --red: #ff7e8a; --shadow: 0 14px 36px rgba(0,0,0,.22);
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body { margin: 0; color: var(--text); background: radial-gradient(circle at 15% 0%, #182744, var(--bg) 40%); font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
    main { max-width: 1440px; margin: 0 auto; padding: 28px 24px 42px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 22px; }
    .kicker { color: var(--blue); font-size: 11px; letter-spacing: .16em; font-weight: 700; }
    h1 { margin: 5px 0 5px; font-size: clamp(26px, 4vw, 42px); letter-spacing: -.03em; }
    h2 { margin: 0; font-size: 15px; }
    p { margin: 0; color: var(--muted); }
    .badge { min-width: 130px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 14px; background: rgba(17,24,39,.82); box-shadow: var(--shadow); text-align: center; font-weight: 800; letter-spacing: .08em; }
    .badge.ready { color: var(--green); border-color: rgba(105,211,154,.45); }
    .badge.hold { color: var(--amber); border-color: rgba(243,189,98,.45); }
    .badge.error { color: var(--red); border-color: rgba(255,126,138,.5); }
    .error { display: none; margin: 0 0 18px; padding: 13px 15px; color: #ffd6da; border: 1px solid rgba(255,126,138,.5); border-radius: 12px; background: rgba(127,29,45,.3); }
    .error.visible { display: block; }
    .live-strip { display: grid; grid-template-columns: minmax(170px,.7fr) repeat(2,minmax(250px,1fr)) minmax(190px,.8fr); gap: 1px; margin-bottom: 16px; overflow: hidden; border: 1px solid var(--line); border-radius: 14px; background: var(--line); box-shadow: var(--shadow); }
    .live-cell { min-width: 0; padding: 11px 13px; background: rgba(17,24,39,.97); }
    .live-head { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; color: var(--muted); font-size: 11px; font-weight: 750; letter-spacing: .04em; text-transform: uppercase; }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
    .live-dot.ok { background: var(--green); box-shadow: 0 0 8px rgba(105,211,154,.55); }
    .live-dot.warn, .live-dot.stale { background: var(--amber); }
    .live-dot.invalid, .live-dot.no-data { background: var(--red); }
    .live-dot.inactive { background: #526077; }
    .live-primary { min-height: 20px; overflow: hidden; color: var(--text); font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .live-meta { overflow: hidden; color: var(--muted); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .live-error { overflow: hidden; color: var(--red); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .live-error.inactive { color: var(--muted); }
    .operation-panel { margin-bottom: 22px; padding: 17px; border: 1px solid rgba(115,183,255,.32); border-radius: 16px; background: linear-gradient(145deg, rgba(20,34,57,.97), rgba(17,24,39,.96)); box-shadow: var(--shadow); }
    .operation-panel[hidden] { display: none; }
    .operator-guidance { margin: 14px 0; padding: 16px; border: 1px solid rgba(115,183,255,.38); border-radius: 14px; background: rgba(11,19,33,.72); }
    .operator-guidance.ready { border-color: rgba(105,211,154,.48); }
    .operator-guidance.warning { border-color: rgba(243,189,98,.48); }
    .operator-guidance.error { border-color: rgba(255,126,138,.58); }
    .operator-guidance-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
    .operator-guidance h2 { margin-bottom: 4px; font-size: 18px; }
    .operator-guidance .controls { margin-top: 10px; }
    .operator-blockers { margin: 10px 0 0; padding-left: 20px; color: #ffd6da; }
    .operator-blockers[hidden], .operator-notices[hidden], .operator-error[hidden] { display: none; }
    .operator-notices { margin-top: 9px; color: var(--muted); font-size: 12px; }
    button.primary { border-color: rgba(115,183,255,.62); background: #17304f; font-weight: 750; }
    .readonly-hint { margin: 0 0 18px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 11px; color: var(--muted); background: rgba(17,24,39,.72); }
    .readonly-hint[hidden] { display: none; }
    .section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin: 22px 2px 10px; }
    .section-heading h2 { font-size: 16px; }
    .grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 14px; }
    .card { grid-column: span 4; min-width: 0; padding: 17px; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(145deg, rgba(23,32,51,.96), rgba(17,24,39,.94)); box-shadow: var(--shadow); }
    .card.wide { grid-column: span 8; }
    .card.full { grid-column: 1 / -1; }
    .card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 13px; }
    .label { color: var(--muted); font-size: 12px; }
    dl { display: grid; grid-template-columns: minmax(120px, .8fr) minmax(0, 1.4fr); gap: 9px 12px; margin: 0; }
    dt { color: var(--muted); overflow-wrap: anywhere; }
    dd { margin: 0; color: var(--text); overflow-wrap: anywhere; text-align: right; }
    dd.mono, pre, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .checks { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; margin: 0; padding: 0; list-style: none; }
    .check { display: flex; align-items: flex-start; gap: 9px; padding: 9px 10px; border-radius: 10px; background: rgba(10,15,28,.5); }
    .dot { flex: 0 0 auto; width: 9px; height: 9px; margin-top: 5px; border-radius: 50%; background: var(--red); }
    .dot.ok { background: var(--green); box-shadow: 0 0 10px rgba(105,211,154,.6); }
    .dot.idle { background: var(--muted); box-shadow: none; }
    .check-name { font-weight: 650; }
    .check-detail { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .source-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 13px; }
    .source { padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(10,15,28,.38); }
    .source h3 { margin: 0 0 9px; font-size: 13px; color: var(--blue); }
    .source h3.ok { color: var(--green); }
    .source h3.warn, .source h3.stale { color: var(--amber); }
    .source h3.invalid, .source h3.no-data { color: var(--red); }
    .source dl { grid-template-columns: 1fr 1fr; gap: 6px 8px; font-size: 12px; }
    .source dd { text-align: right; }
    .setup-panel { margin: 14px 0; padding: 14px; border: 1px solid var(--line); border-radius: 13px; background: rgba(10,15,28,.45); }
    .setup-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
    .setup-state { flex: 0 0 auto; padding: 4px 8px; border-radius: 999px; color: var(--amber); background: rgba(243,189,98,.12); font-size: 11px; font-weight: 750; }
    .setup-state.ready { color: var(--green); background: rgba(105,211,154,.12); }
    .calibration-guide { display: grid; grid-template-columns: minmax(150px,.45fr) minmax(0,1fr); gap: 14px; margin-top: 12px; }
    .calibration-guide ol { margin: 4px 0 0; padding-left: 20px; color: var(--muted); }
    .calibration-guide li + li { margin-top: 4px; }
    .direction { display: inline-block; margin-top: 7px; padding: 4px 8px; border: 1px solid rgba(115,183,255,.35); border-radius: 999px; color: var(--blue); font-size: 11px; font-weight: 750; }
    .setup-facts { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; margin-top: 12px; }
    .setup-fact { padding: 8px 10px; border-radius: 9px; background: rgba(17,24,39,.72); color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .setup-fact strong { display: block; margin-bottom: 2px; color: var(--text); }
    .acceptance-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .journey { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 1px; margin: 14px 0; overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: var(--line); }
    .journey-step { min-width: 0; padding: 10px 12px; color: var(--muted); background: rgba(10,15,28,.78); }
    .journey-step strong { display: block; margin-bottom: 2px; color: var(--text); font-size: 12px; }
    .journey-step.active { background: rgba(32,74,116,.42); box-shadow: inset 0 -3px 0 var(--blue); }
    .journey-step.done { background: rgba(30,82,60,.3); }
    .journey-step.done strong { color: var(--green); }
    .journey-index { display: inline-flex; width: 20px; height: 20px; align-items: center; justify-content: center; margin-right: 5px; border-radius: 50%; background: #23304a; font-size: 11px; }
    .source-control-overview { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 12px; padding: 11px 13px; border: 1px solid var(--line); border-radius: 11px; background: rgba(8,14,25,.7); }
    .source-control-overview strong { font-size: 16px; }
    .source-control-overview .subtle { text-align: right; }
    .source-module-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
    .source-module { display: flex; min-height: 210px; padding: 15px; flex-direction: column; border: 1px solid var(--line); border-radius: 13px; background: rgba(10,15,28,.45); transition: border-color .15s ease, background .15s ease; }
    .source-module.selected { border-color: rgba(105,211,154,.72); background: rgba(22,69,51,.32); box-shadow: inset 0 0 0 1px rgba(105,211,154,.16); }
    .source-module.blocked { border-color: rgba(243,189,98,.38); }
    .source-module-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .source-module h3 { margin: 0 0 5px; color: var(--blue); font-size: 16px; }
    .source-module.selected h3 { color: var(--green); }
    .source-module p { min-height: 40px; font-size: 12px; }
    .source-module-badge { flex: 0 0 auto; padding: 3px 8px; border-radius: 999px; color: var(--muted); background: rgba(91,105,128,.16); font-size: 11px; font-weight: 750; }
    .source-module-badge.ready { color: var(--blue); background: rgba(115,183,255,.12); }
    .source-module-badge.active { color: var(--green); background: rgba(105,211,154,.13); }
    .source-module-badge.blocked { color: var(--amber); background: rgba(243,189,98,.12); }
    .source-module-state { margin: 10px 0; color: var(--muted); font-size: 12px; }
    .source-module .controls { margin-top: auto; }
    .source-module button { width: 100%; min-height: 42px; font-weight: 750; }
    .source-module-action-reason { min-height: 34px; margin-top: 8px; color: var(--muted); font-size: 11px; }
    .source-runtime-meta { display: flex; justify-content: space-between; gap: 14px; margin-top: 10px; color: var(--muted); font-size: 11px; }
    .mode-card { padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(10,15,28,.45); }
    .mode-card h3 { margin: 0 0 5px; font-size: 13px; }
    .mode-card p { min-height: 36px; margin-bottom: 7px; font-size: 12px; }
    .mode-state { margin-bottom: 9px; color: var(--muted); font-size: 12px; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 12px; }
    button, input, select { border: 1px solid var(--line); border-radius: 9px; color: var(--text); background: #10192a; font: inherit; }
    button { padding: 8px 11px; cursor: pointer; }
    button:hover:not(:disabled) { border-color: var(--blue); }
    button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
    button:disabled { cursor: not-allowed; opacity: .4; }
    button.danger { border-color: rgba(255,126,138,.45); color: #ffd6da; }
    input { width: 76px; padding: 7px 8px; }
    input.profile-name { width: min(260px, 100%); }
    .intent-message { min-height: 42px; margin: 10px 0 14px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; color: var(--muted); background: rgba(10,15,28,.46); font-size: 12px; }
    .intent-message.pending { color: var(--blue); border-color: rgba(115,183,255,.42); }
    .intent-message.success { color: var(--green); border-color: rgba(105,211,154,.42); }
    .intent-message.error { display: block; color: #ffd6da; border-color: rgba(255,126,138,.5); background: rgba(127,29,45,.22); }
    .runtime-action-grid { display: grid; grid-template-columns: minmax(0,1.3fr) minmax(260px,.7fr); gap: 12px; margin-top: 12px; }
    .runtime-action-group { padding: 12px; border: 1px solid var(--line); border-radius: 11px; background: rgba(10,15,28,.42); }
    .runtime-action-group h3 { margin: 0 0 5px; font-size: 13px; }
    .control-profile-select { min-width: min(380px, 100%); padding: 8px 10px; }
    .collision-recovery { border-color: rgba(255,126,138,.58); background: rgba(98,26,40,.22); }
    .collision-recovery h3 { margin: 0 0 5px; color: #ffd6da; }
    .collision-recovery ol { margin: 10px 0 0; padding-left: 22px; color: var(--muted); font-size: 12px; line-height: 1.55; }
    .hold-action { border-color: rgba(243,189,98,.62); color: #ffe3aa; background: rgba(105,69,13,.34); font-weight: 800; }
    .lifecycle-panel { margin-top: 12px; }
    .lifecycle-panel > summary { color: var(--muted); }
    .demo-workflow { border-color: rgba(105,211,154,.38); background: linear-gradient(145deg, rgba(16,31,43,.96), rgba(12,20,34,.92)); }
    .demo-progress { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; margin: 14px 0; padding: 0; list-style: none; }
    .demo-step { min-height: 92px; padding: 11px; border: 1px solid var(--line); border-radius: 11px; background: rgba(7,12,22,.55); }
    .demo-step-index { display: inline-flex; width: 23px; height: 23px; align-items: center; justify-content: center; margin-right: 6px; border-radius: 50%; color: var(--muted); background: #19243a; font-size: 12px; font-weight: 800; }
    .demo-step strong { font-size: 13px; }
    .demo-step-detail { display: block; margin-top: 8px; color: var(--muted); font-size: 11px; line-height: 1.45; }
    .demo-step.active { border-color: rgba(87,184,255,.72); box-shadow: inset 0 0 0 1px rgba(87,184,255,.18); }
    .demo-step.active .demo-step-index { color: #06111d; background: var(--blue); }
    .demo-step.done { border-color: rgba(105,211,154,.48); }
    .demo-step.done .demo-step-index { color: #06140d; background: var(--green); }
    .demo-step.ready { border-color: rgba(246,196,83,.5); }
    .demo-step.blocked, .demo-step.waiting { opacity: .62; }
    .demo-step.fault { border-color: rgba(255,126,138,.62); }
    .demo-summary { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; margin: 12px 0; }
    .demo-fact { padding: 9px 10px; border: 1px solid var(--line); border-radius: 9px; background: rgba(7,12,22,.42); }
    .demo-fact strong { display: block; margin-bottom: 4px; color: var(--muted); font-size: 11px; }
    .demo-fact span { font-size: 12px; }
    .demo-status-line { display: flex; align-items: center; gap: 8px; min-height: 24px; margin-top: 10px; color: var(--muted); font-size: 12px; }
    .recording-dot { width: 9px; height: 9px; border-radius: 50%; background: #596274; }
    .recording-dot.active { background: var(--red); box-shadow: 0 0 0 4px rgba(255,126,138,.12); animation: pulse 1.4s ease-in-out infinite; }
    @keyframes pulse { 50% { opacity: .45; } }
    .demo-primary { min-width: 220px; padding: 11px 18px; font-weight: 800; }
    .demo-secondary { margin-left: auto; }
    .demo-recovery { margin-top: 12px; padding-top: 2px; border-top: 1px solid var(--line); }
    .demo-save-path { overflow-wrap: anywhere; color: var(--green); }
    .subtle { color: var(--muted); }
    .diagnostics-panel { margin-top: 18px; padding: 0; }
    .diagnostics-panel > summary { padding: 12px 14px; border: 1px solid var(--line); border-radius: 12px; background: rgba(17,24,39,.72); font-weight: 700; }
    .diagnostics-panel[open] > summary { margin-bottom: 12px; }
    details { margin-top: 14px; }
    summary { cursor: pointer; color: var(--blue); }
    pre { max-height: 420px; margin: 12px 0 0; padding: 14px; overflow: auto; border: 1px solid var(--line); border-radius: 12px; color: #c9d7ea; background: #080c15; white-space: pre-wrap; }
    @media (max-width: 1100px) { .live-strip { grid-template-columns: repeat(2,minmax(0,1fr)); } }
    @media (max-width: 980px) { .card, .card.wide { grid-column: span 6; } .acceptance-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 680px) { main { padding: 20px 14px 32px; } header { display: block; } .badge { margin-top: 15px; } .live-strip { grid-template-columns: 1fr; } .card, .card.wide { grid-column: 1 / -1; } .checks, .source-grid, .source-module-grid, .acceptance-grid, .calibration-guide, .setup-facts, .demo-progress, .demo-summary, .journey, .runtime-action-grid { grid-template-columns: 1fr; } .source-control-overview, .source-runtime-meta { align-items: flex-start; flex-direction: column; } .source-control-overview .subtle { text-align: left; } .demo-secondary { margin-left: 0; } }
  </style>
</head>
<body>
  <main>
    <header>
      <!-- Compatibility marker for clients that still search the old title. -->
      <!-- Sim-HIL 控制与 VIVE 标定 -->
      <div>
        <div class="kicker">控制台 / 仿真在环</div>
        <h1 id="page-title">仿真在环控制与 VIVE 标定</h1>
        <p id="page-summary">本页按准备、启动、运行和收尾四个阶段组织操作；PPO 与人类输入仅作为动作来源。</p>
      </div>
      <div id="badge" class="badge hold">加载中</div>
    </header>
    <div id="error" class="error"></div>
    <div id="readonly-hint" class="readonly-hint" hidden>当前是只读状态页；需要正式运行控制、VIVE 标定、Runtime Home、保持或恢复操作，请使用 <code>--control-gui</code>。</div>
    <section class="live-strip" aria-label="实时输入状态">
      <div class="live-cell">
        <div class="live-head"><span id="vive-receiver-dot" class="live-dot"></span>VIVE 接收器</div>
        <div id="vive-receiver-primary" class="live-primary">等待状态</div>
        <div id="vive-receiver-meta" class="live-meta">—</div>
      </div>
      <div class="live-cell">
        <div class="live-head"><span id="vive-left-dot" class="live-dot"></span><span id="vive-left-title">左 VIVE 来源</span></div>
        <div id="vive-left-primary" class="live-primary">—</div>
        <div id="vive-left-meta" class="live-meta">—</div>
        <div id="vive-left-error" class="live-error"></div>
      </div>
      <div class="live-cell">
        <div class="live-head"><span id="vive-right-dot" class="live-dot"></span><span id="vive-right-title">右 VIVE 来源</span></div>
        <div id="vive-right-primary" class="live-primary">—</div>
        <div id="vive-right-meta" class="live-meta">—</div>
        <div id="vive-right-error" class="live-error"></div>
      </div>
      <div class="live-cell">
        <div class="live-head"><span id="gripper-dot" class="live-dot"></span>夹爪输入</div>
        <div id="gripper-primary" class="live-primary">等待配置</div>
        <div id="gripper-meta" class="live-meta">—</div>
      </div>
    </section>
    <section id="acceptance-card" class="operation-panel" hidden>
      <div class="card-head"><h2 id="operation-title">操作区 · Sim-HIL 正式控制</h2><span id="acceptance-label" class="label">—</span></div>
      <p id="acceptance-description">正在读取所选控制方案。</p>
      <nav id="operator-journey" class="journey" aria-label="操作阶段">
        <div id="journey-prepare" class="journey-step active"><strong><span class="journey-index">1</span>准备输入</strong><span>设备与坐标</span></div>
        <div id="journey-source" class="journey-step"><strong><span class="journey-index">2</span>选择来源</strong><span>Policy / Human</span></div>
        <div id="journey-run" class="journey-step"><strong><span class="journey-index">3</span>运行控制</strong><span>切换、保持、恢复</span></div>
        <div id="journey-finish" class="journey-step"><strong><span class="journey-index">4</span>结束收尾</strong><span>封存或退出</span></div>
      </nav>
      <section id="control-profile-panel" class="setup-panel" hidden>
        <div class="setup-head">
          <div>
            <h2>控制工作流</h2>
            <p id="control-profile-description">直接接管、流式修正和兼容融合共用本页；每个 Runtime 只绑定其中一种完整工作流。</p>
          </div>
          <span id="control-profile-state" class="setup-state">准备中</span>
        </div>
        <div class="controls">
          <label>本次工作流
            <select id="control-profile-selector" class="control-profile-select"></select>
          </label>
          <button id="apply-control-profile" class="primary">选用工作流</button>
        </div>
        <div class="controls">
          <label>Sim 碰撞处理
            <select id="sim-collision-stop-selector" class="control-profile-select">
              <option value="enabled">安全保护触发后终止（代理间隙 / 禁碰接触）</option>
              <option value="disabled">关闭规划器碰撞约束（仅调试）</option>
            </select>
          </label>
          <button id="apply-sim-collision-stop">应用碰撞设置</button>
          <span id="sim-collision-stop-state" class="subtle">默认启用代理间隙与实际禁碰接触保护。</span>
        </div>
        <p id="control-profile-help" class="subtle">启动前可切换；Runtime 构造后需先结束并重建，避免跨 profile 隐式继承 authority、Home 或历史状态。</p>
        <p id="sim-collision-stop-help" class="subtle">该选择只影响 Sim pose planner；关闭时 MuJoCo 物理接触仍存在，但碰撞不会终止回合。</p>
      </section>
      <section id="operator-guidance" class="operator-guidance warning" hidden>
        <div class="operator-guidance-head">
          <div>
            <h2 id="operator-headline">正在检查运行状态</h2>
            <p id="operator-detail">—</p>
          </div>
          <span id="operator-phase" class="setup-state">—</span>
        </div>
        <ul id="operator-blockers" class="operator-blockers" hidden></ul>
        <div id="operator-notices" class="operator-notices" hidden></div>
        <div class="controls">
          <button id="operator-primary-action" class="primary" hidden>下一步</button>
        </div>
        <details id="operator-error" class="operator-error" hidden>
          <summary>查看原始错误</summary>
          <pre id="operator-raw-error">—</pre>
        </details>
      </section>
      <div id="intent-message" class="intent-message" role="status" aria-live="polite">尚未提交操作。</div>
      <div id="human-demo-workflow" class="setup-panel demo-workflow" hidden>
        <div class="setup-head">
          <div>
            <h2>Human Demo · 一条轨迹</h2>
            <p id="human-demo-workflow-message">系统会按“静态预览 → Human 检查 → 遥操记录 → 封存保存”推进。</p>
          </div>
          <span id="human-demo-workflow-state" class="setup-state">等待打开仿真</span>
        </div>
        <ol id="human-demo-progress" class="demo-progress" aria-label="Human demo 采集进度"></ol>
        <div class="demo-summary">
          <div class="demo-fact"><strong>MuJoCo</strong><span id="demo-preview-status">未打开</span></div>
          <div class="demo-fact"><strong>VIVE / 坐标</strong><span id="demo-human-status">等待检查</span></div>
          <div class="demo-fact"><strong>角度扳机</strong><span id="demo-trigger-status">启动记录时接入</span></div>
          <div class="demo-fact"><strong>记录 / 保存</strong><span id="demo-capture-status">尚未记录</span></div>
        </div>
        <div class="demo-status-line">
          <span id="demo-recording-dot" class="recording-dot"></span>
          <span id="demo-operation-status">预览阶段不会接入遥操、推进物理或创建 episode。</span>
        </div>
        <div id="demo-save-path" class="demo-save-path" hidden></div>
        <div class="controls">
          <button id="human-demo-primary-action" class="primary demo-primary">打开静态仿真</button>
          <button id="demo-swap-vive">交换左右 VIVE 来源</button>
          <button id="demo-hold" class="danger" hidden>立即保持输出</button>
          <button id="demo-rearm" hidden>检查后恢复遥操</button>
          <button id="demo-cancel-operation" class="danger" hidden>中断当前等待</button>
        </div>
        <details id="human-demo-recovery" class="demo-recovery">
          <summary>恢复与放弃本次采集</summary>
          <div class="controls">
            <button id="demo-capture-home">保持后重采 Runtime Home</button>
            <button id="demo-rebuild" class="danger">放弃当前回合并重新准备</button>
            <button id="demo-abandon" class="danger">放弃并退出</button>
          </div>
          <p class="subtle">未封存的记录不会作为已保存 demo；放弃操作会明确 abort 当前 episode。</p>
        </details>
      </div>
      <div id="exclusive-source-modules" class="setup-panel" hidden>
        <div class="setup-head">
          <div>
            <h2>控制来源</h2>
            <p>同一组来源卡同时负责启动与运行中切换；绿色卡片是当前真正下发的来源。</p>
          </div>
          <span id="exclusive-source-selection-state" class="setup-state">未启动</span>
        </div>
        <div class="source-control-overview">
          <strong id="exclusive-current-owner" aria-live="polite">当前控制：尚未启动</strong>
          <span id="exclusive-source-transition" class="subtle">请选择一个来源开始。</span>
        </div>
        <div class="source-module-grid">
          <section id="exclusive-policy-module" class="source-module">
            <div class="source-module-head"><h3>Policy 输入</h3><span id="exclusive-policy-badge" class="source-module-badge">检查中</span></div>
            <p>模型输出进入流式分配；Human 不参与当前动作。</p>
            <div id="exclusive-policy-module-state" class="source-module-state">正在检查策略。</div>
            <div class="controls"><button id="start-exclusive-policy-runtime" class="primary">以 Policy 启动</button></div>
            <div id="exclusive-policy-action-reason" class="source-module-action-reason">等待策略状态。</div>
          </section>
          <section id="exclusive-human-module" class="source-module">
            <div class="source-module-head"><h3>Human 输入</h3><span id="exclusive-human-badge" class="source-module-badge">检查中</span></div>
            <p>VIVE pose 与触发器直接成为当前来源；下方单独显示设备、坐标配置和 Home 状态。</p>
            <div id="exclusive-human-module-state" class="source-module-state">正在检查 Human 输入。</div>
            <div class="controls"><button id="start-exclusive-human-runtime" class="primary">以 Human 启动</button></div>
            <div id="exclusive-human-action-reason" class="source-module-action-reason">等待 Human 输入状态。</div>
          </section>
        </div>
        <div class="source-runtime-meta">
          <span id="exclusive-hitl-operation-status">来源启动与切换统一在上方两张卡片中操作。</span>
          <span id="exclusive-cycle-timing">控制周期：等待 Runtime。</span>
        </div>
      </div>
      <div id="vive-setup-panel" class="setup-panel">
        <div class="setup-head">
          <div>
            <h2 id="vive-setup-title">VIVE 输入与坐标配置</h2>
            <p id="vive-setup-message">正在检查输入。</p>
          </div>
          <span id="vive-setup-state" class="setup-state">等待</span>
        </div>
        <div class="setup-facts">
          <div class="setup-fact"><strong>实时输入</strong><span id="vive-input-ready">—</span></div>
          <div class="setup-fact"><strong>坐标配置</strong><span id="vive-profile-ready">—</span></div>
          <div class="setup-fact"><strong>MuJoCo</strong><span id="mujoco-construction-state">未构造</span></div>
          <div class="setup-fact"><strong>当前来源映射</strong><span id="vive-source-mapping">—</span></div>
        </div>
        <details id="vive-calibration-tools" class="lifecycle-panel">
          <summary>坐标标定与来源映射工具（需要修改时展开）</summary>
          <div id="vive-calibration-guide" class="calibration-guide">
            <div>
              <strong id="vive-calibration-stage">未开始标定</strong>
              <div id="vive-calibration-direction" class="direction">先确认来源</div>
              <div id="vive-calibration-progress" class="subtle">阶段：未开始</div>
            </div>
            <ol id="vive-calibration-steps"></ol>
          </div>
          <div class="controls">
            <button id="begin-vive-calibration">开始新标定</button>
            <button id="capture-vive-calibration">采集当前步骤</button>
            <button id="reset-vive-calibration">重做当前步骤</button>
            <button id="cancel-vive-calibration">取消本轮标定</button>
            <button id="swap-vive-sources">交换左右 VIVE 来源</button>
            <span id="vive-swap-status" class="subtle">只交换左右来源映射，不检查来源状态</span>
            <label>保存名称 <input id="vive-profile-name" class="profile-name" placeholder="留空自动命名"></label>
            <button id="save-vive-calibration">保存并选用</button>
          </div>
        </details>
      </div>
      <div id="control-modes" class="acceptance-grid"></div>
      <div id="streaming-control-controls" class="setup-panel" hidden>
        <div class="setup-head">
          <div>
            <h2 id="streaming-control-title">策略 / Human 流式控制</h2>
            <p id="streaming-control-help">当前启动配置决定可用的流式分配选项。</p>
          </div>
          <span id="streaming-control-status" class="setup-state">—</span>
        </div>
        <div id="allocation-controls" hidden>
          <div class="controls">
            <button id="start-allocation-runtime">启动动态分配运行会话</button>
            <button id="allocation-automatic">自动分配</button>
            <button id="allocation-base-only">仅策略基线</button>
            <button id="allocation-relative-overlay">人类相对修正</button>
            <button id="allocation-absolute-overlay-owner" hidden>人类绝对接管</button>
            <button id="allocation-rebase-and-clear">重新定基并清除修正</button>
          </div>
          <div id="allocation-mode-status" class="subtle">—</div>
          <div id="allocation-operation-status" class="subtle">等待 Runtime 启动。</div>
        </div>
      </div>
      <div id="fixed-alpha-controls" class="controls">
        <label>人类输入混合 α <input id="blend-alpha" type="number" min="0.01" max="0.99" step="0.05" value="0.50"></label>
        <button id="apply-blend-alpha">应用当前比例</button>
        <span id="active-blend-alpha" class="subtle">α 状态：尚未启动</span>
        <span class="subtle">α 是 Human target 权重，不是策略速度倍率；修改输入框后必须应用。</span>
      </div>
      <div id="blend-failure-controls" class="controls">
        <label>融合目标不可执行时
          <select id="blend-failure-fallback">
            <option value="hold">静止</option>
            <option value="human">切换到人类</option>
            <option value="policy">切换到策略</option>
          </select>
        </label>
        <label>连续失败切换延迟（秒）
          <input id="blend-failure-grace" type="number" min="0" step="0.05" value="0.00">
        </label>
        <button id="apply-blend-failure-fallback">应用回退源</button>
        <span id="blend-failure-fallback-status" class="subtle">静止始终立即生效；该窗口只延迟切换到人类或策略。</span>
      </div>
      <div id="human-timeout-controls" class="controls">
        <label>人类输入超时回退
          <select id="human-timeout-fallback">
            <option value="hold">静止</option>
            <option value="policy">切换到 PPO</option>
          </select>
        </label>
        <button id="apply-human-timeout-fallback">应用输入回退</button>
        <span id="human-timeout-fallback-status" class="subtle">默认静止；超过 5 秒无新的人类输出时触发。</span>
      </div>
      <div id="runtime-controls" class="setup-panel">
        <div class="setup-head">
          <div>
            <h2>运行控制与故障恢复</h2>
            <p id="runtime-control-help">启动正式 runtime 时会采集当前 VIVE 姿态作为 Runtime Home；保持后可原地重新采集，不会关闭 MuJoCo。</p>
          </div>
          <span id="runtime-control-state" class="setup-state">—</span>
        </div>
        <section id="collision-recovery" class="runtime-action-group collision-recovery" hidden>
          <h3>规划器安全间隙保护已触发，本回合已终止</h3>
          <p id="collision-recovery-detail">当前 episode 不允许恢复、切换来源或重采 Home。</p>
          <div class="controls">
            <button id="collision-reset-runtime" class="danger">Reset 仿真</button>
          </div>
          <div id="collision-recovery-status" class="subtle">Reset 会销毁当前 MuJoCo target，并返回启动准备页。</div>
        </section>
        <div id="runtime-primary-actions" class="runtime-action-grid">
          <section class="runtime-action-group">
            <h3>即时控制</h3>
            <p>保持会停止策略、混合与执行推进；恢复前会重新检查输入。</p>
            <div class="controls">
              <button id="hold-runtime" class="hold-action">立即保持输出</button>
              <button id="rearm-runtime" class="primary">检查后恢复运行</button>
              <button id="capture-runtime-home">保持后重采 Runtime Home</button>
            </div>
          </section>
          <section class="runtime-action-group">
            <h3>当前操作</h3>
            <p id="runtime-operation-summary">暂无正在处理的操作。</p>
            <div class="controls">
              <button id="cancel-operation" class="danger" hidden>中断当前等待</button>
            </div>
          </section>
        </div>
        <details id="runtime-lifecycle-actions" class="lifecycle-panel">
          <summary>结束、保存与重建（展开后操作）</summary>
          <div class="controls">
            <button id="finalize-capture" class="danger">结束回合、封存并自动保存</button>
            <button id="rebuild-runtime" class="danger">结束当前仿真并返回准备页</button>
            <button id="stop-runtime" class="danger">退出控制</button>
          </div>
        </details>
      </div>
    </section>
    <details id="operator-diagnostics" class="diagnostics-panel">
      <summary>高级诊断与回执信息（只读）</summary>
      <div class="section-heading"><h2>状态与执行信息</h2><span class="label">回执 / 来源链 / 诊断</span></div>
      <section class="grid">
      <article class="card">
        <div class="card-head"><h2>配置</h2><span id="sampled" class="label">—</span></div>
        <dl id="profile-rows"></dl>
      </article>
      <article class="card wide">
        <div class="card-head"><h2>就绪检查</h2><span id="readiness-label" class="label">—</span></div>
        <ul id="checks" class="checks"></ul>
      </article>
      <article class="card full">
        <div class="card-head"><h2>输入来源</h2><span class="label">不显示原始动作值</span></div>
        <!-- no raw action values (presentation label retained for old checks) -->
        <div class="source-grid">
          <div class="source"><h3>临时 PPO 策略</h3><dl id="policy-rows"></dl></div>
          <div class="source"><h3>被动 VIVE + 当前夹爪输入</h3><dl id="human-rows"></dl></div>
        </div>
      </article>
      <article class="card wide">
        <div class="card-head"><h2>动作混合</h2><span id="composition-label" class="label">—</span></div>
        <dl id="composition-rows"></dl>
      </article>
      <article class="card">
        <div class="card-head"><h2>执行</h2><span id="gate-label" class="label">—</span></div>
        <dl id="execution-rows"></dl>
      </article>
      <article id="capture-card" class="card full" hidden>
        <div class="card-head"><h2>采集 / ControlTrace</h2><span class="label">过程信息</span></div>
        <dl id="capture-rows"></dl>
      </article>
      <article class="card full">
        <details>
          <summary>查看完整状态 JSON（仅协议元数据与回执摘要）</summary>
          <pre id="raw-status">—</pre>
        </details>
      </article>
      </section>
    </details>
  </main>
  <script>
    const controlToken = __SIM_HIL_CONTROL_TOKEN_JSON__;
    const $ = (id) => document.getElementById(id);
    let latestStatus = null;
    let intentPending = false;
    let pendingIntentId = null;
    let pendingIntentKind = null;
    const clientId = (() => {
      let value = sessionStorage.getItem('simHilAcceptanceClientId');
      if (!value) {
        value = 'browser-' + crypto.randomUUID();
        sessionStorage.setItem('simHilAcceptanceClientId', value);
      }
      return value;
    })();
    function nextClientSequence() {
      const key = 'simHilAcceptanceClientSequence';
      const value = Number(sessionStorage.getItem(key) || '0') + 1;
      sessionStorage.setItem(key, String(value));
      return value;
    }
    const text = (value) => {
      if (value === null || value === undefined || value === '') return '—';
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    };
    // Protocol names remain English in the JSON contract; only the
    // presentation layer translates them.  This keeps receipts/debug output
    // stable while making the operator view consistently Chinese.
    const UI_LABELS = Object.freeze({
      'Readiness gate': '就绪检查',
      'lifecycle': '生命周期', 'profile': '配置', 'target': '执行目标',
      'real arm motion': '真机机械臂运动', 'owner': '执行所有者',
      'status schema': '状态协议', 'MuJoCo viewer': 'MuJoCo 窗口',
      'viewer frames': '窗口帧数', 'viewer error': '窗口错误',
      'compose skipped cycles': '跳过的混合周期',
      'last compose overrun': '最近混合超时', 'overrun handling': '超时处理',
      'cycle total': '最近完整周期', 'composition + execution': '混合与执行处理',
      'policy source': '策略来源处理',
      'restart required': '需要重启', 'first startup failure': '首次启动失败',
      'first control fault': '首次控制故障', 'Home profile': 'Home 配置',
      'binding': '绑定', 'VIVE coordinate': 'VIVE 坐标配置',
      'target handoff': '目标交接', 'target activation': '目标激活模式',
      'target capability': '目标能力', 'state': '状态', 'refresh': '刷新状态',
      'checkpoint': '检查点', 'queue': '队列长度', 'last emission': '最近输出',
      'emission hash': '输出哈希', 'payload kind': '载荷类型',
      'representation trace': '表示追踪', 'canonicalization': '规范化',
      'TTL age': 'TTL 年龄', 'inference duration': '推理耗时',
      'last error': '最近错误', 'endpoint': '设备端点', 'requested': '请求来源',
      'fallback': '回退原因', 'session': '会话', 'source started': '来源已启动',
      'age': '时延', 'rig absent': '无机械臂绑定', 'authoritative': '具备权威性',
      'hold': '保持状态', 'home re-calibration': '需要重新设置起始位姿', 'VIVE port': 'VIVE 端口',
      'VIVE source mapping': 'VIVE 来源映射', 'VIVE swaps': 'VIVE 交换次数',
      'no raw action values': '不显示原始动作值',
      'input devices': '输入设备', 'channels': '通道', 'started': '已启动',
      'calibration_ready': '标定就绪', 'deadman_ready': '操控使能就绪',
      'home_ready': 'Home 就绪', 'human_device_ready': '人类设备就绪',
      'policy_source_ready': '策略来源就绪', 'mujoco_state_ready': 'MuJoCo 状态就绪',
      'execution_target_ready': '执行目标就绪',
      'activation': '激活', 'activation hash': '激活哈希', 'cycle': '周期',
      'cycle hash': '周期哈希', 'support until': '支持截止', 'disposition': '处理决定',
      'proposal': '提案', 'proposal until': '提案截止', 'human input': '人类输入',
      'policy input': '策略输入', 'blocked': '阻塞原因', 'gate': '执行门',
      'gate reason': '执行门原因', 'gate epoch': '执行门轮次',
      'cycle execution': '周期执行', 'command': '命令', 'dispatch receipt': '下发回执',
      'target request': '目标请求', 'target submission': '目标提交',
      'target completion': '目标完成', 'component target': '组件目标',
      'target outcome': '目标结果', 'resolution reason': '解析失败原因',
      'resolver status': '解析器状态', 'resolution message': '解析说明',
      'resolution diagnostics': '解析诊断',
      'target admission': '目标准入', 'target activation': '目标激活',
      'target support until': '目标支持截止', 'execution difference': '执行差异',
      'difference reason': '差异原因', 'difference measurement': '差异测量',
      'difference reference': '差异参考', 'difference residual': '差异残差',
      'trace status': '记录状态', 'trace session': '记录会话', 'episode': '回合',
      'commit': '封存状态', 'storage root': '存储目录', 'message': '说明',
    });
    const UI_VALUES = Object.freeze({
      'LOADING': '加载中', 'READY': '就绪', 'HOLD': '保持',
      'RESTART REQUIRED': '需要重启', 'STARTING': '启动中', 'STOPPED': '已停止',
      'VIVE SETUP': 'VIVE 设置', 'STATUS ERROR': '状态错误',
      'true': '是', 'false': '否',
      'ALL READY': '全部就绪', 'receiving': '接收中',
      'packet_stale': '数据包过期', 'waiting_for_packet': '等待数据包',
      'stopped': '已停止', 'unavailable': '不可用', 'ok': '正常',
      'invalid': '无效', 'no-data': '无数据', 'stale': '过期', 'inactive': '未选用',
      'ready': '就绪', 'fault': '故障', 'running': '运行中', 'locked': '不可用',
      'available': '可开始', 'faulted': '故障待处理',
      'awaiting_confirmation': '等待处理', 'completed': '已完成',
      'passed': '已通过', 'failed': '失败', 'not_constructed': '未构造',
      'swapped': '已交换', 'accepted': '已接受', 'rejected': '已拒绝',
      'operation_requested': '已请求',
      'held': '已保持', 'rearmed': '已恢复运行',
      'runtime_rebuilt': '运行时已重建', 'saved': '已保存',
      'shutdown_requested': '已请求退出', 'complete': '已完成',
      'suspended_by_application_hold': '应用已保持，暂停刷新',
      'idle': '未开始', 'waiting': '等待', 'error': '错误',
      'not started': '未开始', 'policy_only': '仅策略', 'human_only': '仅人类',
      'proportional_blend': '比例混合', 'receiving': '接收中',
      'restart_required': '需要重建', 'mujoco_only': '仅 MuJoCo',
      'mujoco_not_constructed': 'MuJoCo 尚未构造',
      'left_wrist': '左腕来源', 'right_wrist': '右腕来源',
      'left_wrist_pose': '左腕位姿', 'right_wrist_pose': '右腕位姿',
      'right_angle_trigger': '右侧角度触发器', 'q6': 'Q6',
      'angle_trigger': '角度触发器', 'auto': '自动', 'none': '无',
      'passive_debug_only': '被动调试模式',
      'loaded_and_acknowledged': '已加载并确认', 'anchors_seeded': '锚点已设置',
      'no_emission': '尚无输出', 'emission_timestamp_invalid': '输出时间戳无效',
      'policy_chunk_ttl_expired': '策略动作已过期', 'finite': '数值有效',
      'rebase_required': '需要重新对齐', 'not_required': '无需检查',
      'mujoco_state_nonfinite': 'MuJoCo 状态无效',
      'fallback_selected': '回退源已选择',
      'hold_selected': '保持静止', 'latched': '已锁存切换',
    });
    const CONTROL_PROFILE_PRESENTATION = Object.freeze({
      fixed_alpha_compatibility: {
        label: '固定比例融合（兼容）',
        detail: '单帧 Policy / Human 目标按固定 α 融合，可运行中切换纯 Policy、纯 Human 或比例融合。',
      },
      exclusive_truncated_hitl: {
        label: '直接接管',
        detail: 'Policy 与 Human 二选一；接管和交还都经过 HOLD、Home / 策略重置与新目标接纳。',
      },
      policy_human_allocation_scheme_a: {
        label: '流式相对修正',
        detail: '策略基线持续运行，Human 以相对 residual 叠加；支持自动分配和重新定基。',
      },
      policy_human_allocation_scheme_b: {
        label: '流式完整分配',
        detail: '在相对 residual 基础上增加 Human 绝对接管与连续交接。',
      },
    });
    function uiLabel(value) {
      const key = String(value);
      return UI_LABELS[key] || key;
    }
    function uiToken(value) {
      const key = String(value);
      return UI_VALUES[key] || key;
    }
    function uiValue(value) {
      const rendered = text(value);
      if (rendered === '—') return rendered;
      if (UI_VALUES[rendered]) return UI_VALUES[rendered];
      const prefixes = [
        ['emission_stale:', '输入已过期：'],
        ['emission_in_future:', '输入时间超前：'],
        ['fresh:', '输入新鲜度：'],
        ['valid_until:', '有效截止：'],
        ['teleop_state:', '遥操状态：'],
        ['phase ', '阶段：'],
      ];
      for (const [prefix, translated] of prefixes) {
        if (rendered.startsWith(prefix)) return translated + rendered.slice(prefix.length);
      }
      return rendered;
    }
    const short = (value) => {
      const rendered = text(value);
      return rendered.length > 96 ? rendered.slice(0, 92) + '…' : rendered;
    };
    function rows(target, entries) {
      target.replaceChildren();
      for (const [name, value] of entries) {
        const dt = document.createElement('dt');
        const dd = document.createElement('dd');
        const rendered = uiValue(value);
        dt.textContent = uiLabel(name);
        dd.textContent = short(rendered);
        if (rendered !== '—') dd.title = rendered;
        if (typeof value === 'string' && (value.includes('sha') || value.length > 48)) dd.className = 'mono';
        target.append(dt, dd);
      }
    }
    function explainDisabled(button, reason) {
      button.title = button.disabled && reason ? String(reason) : '';
    }
    function setIntentMessage(message, tone = 'neutral') {
      const target = $('intent-message');
      target.textContent = String(message || '尚未提交操作。');
      target.className = `intent-message ${tone === 'neutral' ? '' : tone}`.trim();
    }
    function renderOperatorJourney(value) {
      const acceptance = value.acceptance || {};
      const humanDemonstration = acceptance.capture_profile
        === 'human_demonstration';
      const journey = $('operator-journey');
      journey.hidden = humanDemonstration;
      if (humanDemonstration) return;
      const setup = acceptance.setup || {};
      const runtimeConstructed = Boolean(
        (value.startup || {}).mujoco_constructed
        || setup.phase === 'runtime_constructed'
      );
      const anySourceReady = Boolean(
        setup.policy_runtime_launch_ready
        || setup.human_runtime_launch_ready
        || setup.runtime_launch_ready
      );
      let activeIndex = 0;
      if (value.episode_ended || acceptance.shutdown_requested) activeIndex = 3;
      else if (value.started || runtimeConstructed) activeIndex = 2;
      else if (anySourceReady) activeIndex = 1;
      for (const [index, id] of [
        'journey-prepare', 'journey-source', 'journey-run', 'journey-finish',
      ].entries()) {
        const item = $(id);
        item.className = `journey-step ${
          index < activeIndex ? 'done' : index === activeIndex ? 'active' : ''
        }`.trim();
      }
    }
    function renderChecks(readiness) {
      const target = $('checks');
      target.replaceChildren();
      const details = readiness.detail_by_check || {};
      const names = Object.keys(details);
      const required = new Set(readiness.required_checks || names);
      for (const name of names) {
        const item = document.createElement('li');
        item.className = 'check';
        const dot = document.createElement('span');
        const isRequired = required.has(name);
        const ok = isRequired && !(readiness.failed_checks || []).includes(name);
        dot.className = ok ? 'dot ok' : isRequired ? 'dot' : 'dot idle';
        const body = document.createElement('div');
        const title = document.createElement('div');
        const detail = document.createElement('div');
        title.className = 'check-name';
        detail.className = 'check-detail';
        title.textContent = uiLabel(name);
        const holdNote = latestStatus?.held && name === 'human_device_ready'
          ? '（应用保持期间暂停刷新）' : '';
        detail.textContent = isRequired
          ? uiValue(details[name] || (ok ? 'ready' : 'failed')) + holdNote
          : `当前混合无需检查 · ${uiValue(details[name] || '暂无证据')}`;
        body.append(title, detail);
        item.append(dot, body);
        target.append(item);
      }
      $('readiness-label').textContent = readiness.ready
        ? uiValue('ALL READY')
        : `${(readiness.failed_checks || []).length} 项检查未通过`;
    }
    const controlModeNames = {
      human_teleop: '纯遥操',
      ppo_only: '纯 PPO',
      proportional_blend: '比例融合',
    };
    const controlModeDescriptions = {
      human_teleop: '仅使用人类 pose / 末端 qpos；启动前必须接纳 Human 输入。',
      ppo_only: '仅使用策略动作；可在 VIVE 尚未就绪时独立启动。',
      proportional_blend: '按固定 α 融合人类与策略，不使用动态分配规则。',
    };
    const allocationModeNames = {
      automatic: '自动分配',
      base_only: '仅策略基线',
      relative_overlay: '人类相对修正',
      absolute_overlay_owner: '人类绝对接管',
    };
    const allocationSignalNames = {
      allocation_automatic: '切换到自动分配',
      allocation_base_only: '切换到仅策略基线',
      allocation_relative_overlay: '切换到人类相对修正',
      allocation_absolute_overlay_owner: '切换到人类绝对接管',
      residual_rebase_and_clear: '重新定基并清除修正',
    };
    const operatorPhaseNames = {
      setup_required: '需要准备', calibrating: '正在标定',
      ready_to_start: '可以启动', operation_pending: '操作中',
      ready_for_preview: '可以预览', sim_preview: '静态预览',
      running: '运行中', held: '保持', faulted: '故障',
      restart_required: '需要重建', episode_ended: '回合已结束',
      closing: '正在退出',
    };
    function renderOperatorGuidance(value) {
      const guidance = value.operator_guidance || {};
      const panel = $('operator-guidance');
      const humanDemonstration = value.acceptance?.capture_profile
        === 'human_demonstration';
      if (!guidance.phase || humanDemonstration) {
        panel.hidden = true;
        return;
      }
      panel.hidden = false;
      panel.className = `operator-guidance ${guidance.severity || 'warning'}`;
      $('operator-headline').textContent = text(guidance.headline);
      $('operator-detail').textContent = text(guidance.detail);
      $('operator-phase').textContent =
        operatorPhaseNames[guidance.phase] || uiValue(guidance.phase);
      $('operator-phase').className = `setup-state ${
        guidance.severity === 'ready' ? 'ready' : ''
      }`;

      const blockers = $('operator-blockers');
      blockers.replaceChildren();
      for (const blocker of guidance.blockers || []) {
        const item = document.createElement('li');
        item.textContent = text(blocker.message || blocker.code);
        blockers.append(item);
      }
      blockers.hidden = blockers.childElementCount === 0;
      const notices = guidance.notices || [];
      $('operator-notices').textContent = notices.join(' · ');
      $('operator-notices').hidden = notices.length === 0;

      const action = guidance.recommended_action || null;
      const actionButton = $('operator-primary-action');
      const sourceActionIsAlreadyVisible = Boolean(
        value.acceptance?.exclusive_hitl_enabled
        && (
          (action?.interaction === 'focus'
            && action.target_id === 'exclusive-source-modules'
          ) || (action?.interaction === 'intent'
            && ['start_runtime', 'request_human_takeover', 'request_policy_resume']
              .includes(action.intent_kind)
          )
        )
      );
      const runtimeActionIsAlreadyVisible = Boolean(
        value.started
        && action?.interaction === 'intent'
        && ['hold', 'rearm'].includes(action.intent_kind)
      );
      actionButton.hidden = !action || humanDemonstration
        || sourceActionIsAlreadyVisible || runtimeActionIsAlreadyVisible;
      actionButton.disabled = !action;
      actionButton.textContent = action ? text(action.label) : '下一步';
      const rawError = guidance.raw_error;
      const executionFault = guidance.execution_fault || null;
      $('operator-error').hidden = !rawError && !executionFault;
      $('operator-raw-error').textContent = rawError
        ? String(rawError)
        : executionFault ? JSON.stringify(executionFault, null, 2) : '—';
    }
    function formatDuration(seconds) {
      const value = Number(seconds);
      if (!Number.isFinite(value)) return '—';
      const total = Math.max(0, Math.floor(value));
      const minutes = Math.floor(total / 60);
      const remainder = total % 60;
      return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
    }
    function renderHumanDemoWorkflow(value, acceptanceBusy) {
      const acceptance = value.acceptance || {};
      const guidance = value.operator_guidance || {};
      const workflow = guidance.human_demo_workflow || {};
      const capture = workflow.capture || {};
      const recording = workflow.recording || {};
      const input = workflow.input || {};
      const steps = $('human-demo-progress');
      steps.replaceChildren();
      for (const [index, step] of (workflow.steps || []).entries()) {
        const item = document.createElement('li');
        item.className = `demo-step ${step.state || 'waiting'}`;
        const badge = document.createElement('span');
        badge.className = 'demo-step-index';
        badge.textContent = step.state === 'done' ? '✓' : String(index + 1);
        const title = document.createElement('strong');
        title.textContent = text(step.label);
        const detail = document.createElement('span');
        detail.className = 'demo-step-detail';
        detail.textContent = text(step.detail);
        item.append(badge, title, detail);
        steps.append(item);
      }
      $('human-demo-workflow-state').textContent = text(
        workflow.state_label || guidance.headline || '等待状态'
      );
      $('human-demo-workflow-state').className = `setup-state ${
        ['sim_preview', 'running', 'episode_ended'].includes(workflow.phase)
          ? 'ready' : ''
      }`;
      $('human-demo-workflow-message').textContent = text(
        guidance.detail || '按页面主操作继续。'
      );
      const previewStep = (workflow.steps || []).find(step => step.id === 'preview') || {};
      $('demo-preview-status').textContent = previewStep.state === 'done'
        ? (value.started ? '已进入控制' : '静态 viewer 已打开')
        : previewStep.state === 'active' ? '正在打开' : '尚未打开';
      $('demo-human-status').textContent = input.vive_ready
        ? (input.coordinate_profile_ready ? 'VIVE 与坐标已就绪' : 'VIVE 就绪，缺少坐标配置')
        : '等待有效 VIVE 输入';
      $('demo-trigger-status').textContent = input.angle_trigger === 'connected'
        ? '已接入并通过检查' : '开始记录时接入检查';
      $('demo-capture-status').textContent = capture.status === 'committed'
        ? '已封存并保存'
        : capture.status === 'recording'
        ? `记录中 ${formatDuration(recording.elapsed_sec)}`
        : capture.status === 'faulted' ? '记录故障，输出应保持' : '尚未开始记录';
      $('demo-recording-dot').className = `recording-dot ${
        recording.active && !recording.held ? 'active' : ''
      }`;
      $('demo-operation-status').textContent = workflow.operation_label
        || (recording.active
          ? `${recording.held ? '已保持 · ' : '正在记录 · '}${formatDuration(recording.elapsed_sec)}`
          : capture.status === 'committed'
          ? 'durable commit 已完成，可以安全退出本次进程。'
          : '预览阶段不会接入遥操、推进物理或创建 episode。');
      const savePath = $('demo-save-path');
      savePath.hidden = capture.status !== 'committed' || !capture.session_root;
      savePath.textContent = savePath.hidden
        ? '' : `已保存到：${capture.session_root}`;

      const primaryAction = workflow.primary_action || null;
      const primary = $('human-demo-primary-action');
      primary.textContent = workflow.active_intent_kind
        ? text(workflow.operation_label || '操作进行中…')
        : text(primaryAction?.label || '等待下一步');
      primary.disabled = acceptanceBusy || !primaryAction;
      explainDisabled(
        primary,
        workflow.active_intent_kind
          ? '当前操作正在由唯一 Station control thread 执行。'
          : !primaryAction ? '当前没有可执行的主操作。' : '',
      );

      const started = Boolean(value.started);
      const held = Boolean(value.held);
      const ended = Boolean(value.episode_ended);
      const setup = acceptance.setup || {};
      const calibration = setup.calibration || {};
      const demoSwap = $('demo-swap-vive');
      demoSwap.hidden = started || ended || capture.status === 'committed';
      demoSwap.disabled = acceptanceBusy
        || Boolean(calibration.workflow_active);
      const demoMapping = Object.entries(
        calibration.source_role_by_output_channel || {}
      ).map(([channel, role]) => `${uiToken(role)}→${uiToken(channel)}`).join(' · ');
      demoSwap.title = demoMapping
        ? `当前：${demoMapping}`
        : '只交换左右来源映射，不连接或识别设备。';
      $('demo-hold').hidden = !started || held || ended;
      $('demo-hold').disabled = acceptanceBusy;
      $('demo-rearm').hidden = !started || !held || ended
        || capture.status === 'faulted';
      $('demo-rearm').disabled = acceptanceBusy;
      $('demo-rearm').textContent = '检查后恢复遥操';
      $('demo-cancel-operation').hidden = !Boolean(
        acceptanceBusy && acceptance.operation_cancel_available
      );
      $('demo-capture-home').disabled = acceptanceBusy || !started || !held || ended;
      $('demo-rebuild').disabled = acceptanceBusy
        || !(value.startup || {}).mujoco_constructed || ended;
      $('demo-abandon').disabled = acceptanceBusy;
      $('human-demo-recovery').hidden = capture.status === 'committed';
      if (ended) $('human-demo-recovery').open = true;
    }
    function renderControlProfileSelector(
      acceptance,
      {acceptanceBusy, humanDemonstration, runtimeConstructed},
    ) {
      const panel = $('control-profile-panel');
      if (humanDemonstration) {
        panel.hidden = true;
        return;
      }
      const available = acceptance.available_control_profile_ids || [];
      const selectable = Boolean(
        acceptance.control_profile_selection_available
      );
      const collisionSelectable = Boolean(
        acceptance.sim_collision_stop_selection_available
      );
      panel.hidden = (!selectable || available.length === 0)
        && !collisionSelectable;
      if (panel.hidden) return;
      const selected = String(
        acceptance.selected_control_profile_id
        || 'fixed_alpha_compatibility'
      );
      const selector = $('control-profile-selector');
      const currentOptions = [...selector.options].map(option => option.value);
      if (JSON.stringify(currentOptions) !== JSON.stringify(available)) {
        selector.replaceChildren();
        for (const profileId of available) {
          const option = document.createElement('option');
          option.value = profileId;
          option.textContent = CONTROL_PROFILE_PRESENTATION[profileId]?.label
            || profileId;
          selector.append(option);
        }
      }
      if (document.activeElement !== selector) selector.value = selected;
      const selectedPresentation = CONTROL_PROFILE_PRESENTATION[
        selector.value || selected
      ] || {};
      $('control-profile-description').textContent = selectedPresentation.detail
        || '选择一个完整的 Sim-HIL 控制工作流。';
      const locked = Boolean(
        acceptance.control_profile_selection_locked || runtimeConstructed
      );
      selector.disabled = acceptanceBusy || locked || !selectable;
      const apply = $('apply-control-profile');
      apply.disabled = acceptanceBusy || locked || !selectable
        || selector.value === selected;
      $('control-profile-state').textContent = locked
        ? `本次已固定：${CONTROL_PROFILE_PRESENTATION[selected]?.label || selected}`
        : `当前选择：${CONTROL_PROFILE_PRESENTATION[selected]?.label || selected}`;
      $('control-profile-state').className = `setup-state ${locked ? 'ready' : ''}`;
      $('control-profile-help').textContent = locked
        ? 'Runtime 已绑定完整 lifecycle / control-cycle profile。若要更换，请先用“结束当前仿真并返回准备页”；系统不会在运行中隐式迁移 authority、Home 或历史状态。'
        : '启动前可在同一界面选择直接接管、流式修正或兼容融合；CLI --control-profile 只是初始默认值。';
      explainDisabled(
        apply,
        acceptanceBusy
          ? '当前有操作正在执行。'
          : locked
          ? '先结束并重建当前 Runtime。'
          : '当前工作流已经被选中。',
      );
      const collisionEnabled = Boolean(
        acceptance.selected_sim_collision_stop_enabled
      );
      const collisionSelector = $('sim-collision-stop-selector');
      const collisionSelectedValue = collisionEnabled ? 'enabled' : 'disabled';
      if (document.activeElement !== collisionSelector) {
        collisionSelector.value = collisionSelectedValue;
      }
      collisionSelector.disabled = acceptanceBusy || locked
        || !collisionSelectable;
      const collisionApply = $('apply-sim-collision-stop');
      collisionApply.disabled = acceptanceBusy || locked
        || !collisionSelectable
        || collisionSelector.value === collisionSelectedValue;
      $('sim-collision-stop-state').textContent = locked
        ? `本次已固定：${collisionEnabled ? '代理间隙或禁碰接触触发后终止' : '规划器碰撞约束已关闭'}`
        : `当前选择：${collisionEnabled ? '代理间隙或禁碰接触触发后终止' : '规划器碰撞约束已关闭'}`;
      $('sim-collision-stop-help').textContent = locked
        ? '该行为已经写入本次 Sim target 的规划契约；如需更改，请先 Reset 仿真。'
        : '启用时碰撞会终止当前 episode，只能 Reset；禁用时碰撞不会停止 Sim。';
      explainDisabled(
        collisionApply,
        acceptanceBusy
          ? '当前有操作正在执行。'
          : locked
          ? 'Runtime 已构造，必须先 Reset。'
          : '当前碰撞设置已经被选中。',
      );
    }
    function renderAcceptance(value) {
      const acceptance = value.acceptance;
      const card = $('acceptance-card');
      const captureCard = $('capture-card');
      $('readonly-hint').hidden = Boolean(controlToken);
      if (!acceptance || !controlToken) {
        card.hidden = true;
        captureCard.hidden = true;
        return;
      }
      card.hidden = false;
      captureCard.hidden = false;
      renderOperatorGuidance(value);
      const setup = acceptance.setup || {};
      const calibration = setup.calibration || {};
      const humanDemonstration = acceptance.capture_profile
        === 'human_demonstration';
      renderOperatorJourney(value);
      const simPreviewReady = Boolean(acceptance.sim_preview_ready);
      const executionTarget = value.execution_target;
      const runtimeConstructed = Boolean(
        (value.startup || {}).mujoco_constructed
        || setup.phase === 'runtime_constructed'
        || (executionTarget
          && executionTarget !== 'mujoco_not_constructed')
      );
      const policyRuntimeLaunchReady = Boolean(
        setup.policy_runtime_launch_ready
        ?? setup.runtime_launch_ready
        ?? runtimeConstructed
      );
      const humanRuntimeLaunchReady = Boolean(
        setup.human_runtime_launch_ready
        ?? setup.runtime_launch_ready
        ?? runtimeConstructed
      );
      const acceptanceBusy = intentPending
        || Number(acceptance.pending_intent_count || 0) > 0
        || Boolean(acceptance.active_intent_id);
      const busyReason = acceptanceBusy
        ? '当前已有操作正在执行；请等待完成或使用“中断当前操作”。'
        : '';
      renderControlProfileSelector(acceptance, {
        acceptanceBusy,
        humanDemonstration,
        runtimeConstructed,
      });
      const calibrationAvailable = Boolean(setup.calibration);
      const deferredHumanSetup = Boolean(
        runtimeConstructed
        && calibrationAvailable
        && setup.phase === 'policy_runtime_human_setup_available'
      );
      const workflowActive = Boolean(calibration.workflow_active);
      if (workflowActive) $('vive-calibration-tools').open = true;
      const lastResult = acceptance.last_result || {};
      const sourceConfiguration = value.source_configuration || {};
      const currentSourceMapping = (
        calibration.source_role_by_output_channel
        || sourceConfiguration.vive_source_role_by_output_channel
        || {}
      );
      $('vive-source-mapping').textContent = Object.entries(
        currentSourceMapping
      ).map(([channel, role]) => `${uiToken(role)} → ${uiToken(channel)}`).join(' · ') || '—';
      const calibrationStage = calibration.stage || 'idle';
      const calibrationPhase = calibration.phase || 'idle';
      const inputPreflight = calibration.input_preflight || {};
      const guide = calibration.stage_guide || {};
      const setupState = $('vive-setup-state');
      setupState.textContent = simPreviewReady
        ? (humanRuntimeLaunchReady ? '预览中 · Human 可接入' : '预览中 · 等待 Human')
        : deferredHumanSetup
        ? (humanRuntimeLaunchReady ? 'Human 可以接入' : 'PPO 运行中')
        : runtimeConstructed
        ? '运行中'
        : humanRuntimeLaunchReady ? 'Human 可以启动' : '需要设置';
      setupState.className = `setup-state ${
        (simPreviewReady
          || (deferredHumanSetup ? humanRuntimeLaunchReady : runtimeConstructed))
          ? 'ready' : ''
      }`;
      $('vive-setup-title').textContent = simPreviewReady
        ? '仿真预览中 · Human 输入尚未接入执行链路'
        : deferredHumanSetup
        ? 'PPO 已运行 · Human 输入按需接入'
        : runtimeConstructed
        ? 'VIVE 配置已交接给 Sim-HIL'
        : 'VIVE 输入与坐标配置';
      $('vive-setup-message').textContent = simPreviewReady
        ? (humanRuntimeLaunchReady
          ? '可继续调整 MuJoCo 视角；点击“开始遥操并记录”后才关闭设置接收器并接入 Human 源。'
          : uiValue(calibration.message || inputPreflight.message || '继续调整视角，同时等待 VIVE 与坐标配置就绪。'))
        : deferredHumanSetup
        ? (calibration.message || inputPreflight.message
          ? uiValue(calibration.message || inputPreflight.message)
          : '纯 PPO 不使用 VIVE；切换到遥操或混合模式时才检查并接入。')
        : runtimeConstructed
        ? '标定接收器已关闭；当前运行时持有唯一 VIVE 接收器。'
        : (calibration.message || inputPreflight.message
          ? uiValue(calibration.message || inputPreflight.message)
          : '确认实时输入和已保存坐标配置后再启动。');
      $('vive-calibration-stage').textContent =
        uiValue(calibration.stage_label || '按需标定');
      $('vive-calibration-direction').textContent =
        uiValue(guide.direction || '先确认来源');
      const sampleEntries = Object.entries(
        calibration.capture_samples || {}
      ).map(([name, count]) => `${uiValue(name)} ${count}`)
        .join(' · ');
      const timing = calibration.capture_prep_remaining_sec !== null
        && calibration.capture_prep_remaining_sec !== undefined
        ? `准备 ${Number(calibration.capture_prep_remaining_sec).toFixed(1)}s`
        : calibration.capture_elapsed_sec !== null
          && calibration.capture_elapsed_sec !== undefined
          ? `采集 ${Number(calibration.capture_elapsed_sec).toFixed(1)}s`
          : null;
      $('vive-calibration-progress').textContent = [
        `阶段：${uiValue(calibrationPhase)}`,
        sampleEntries || null,
        timing,
      ].filter(Boolean).join(' · ');
      const guideSteps = $('vive-calibration-steps');
      guideSteps.replaceChildren();
      (guide.steps || [
        '不需要重新标定时，可直接使用已保存的有效坐标配置。',
        '需要标定时，在这里启动并按当前方向完成每一步。',
      ]).forEach((description) => {
        const item = document.createElement('li');
        item.textContent = String(description);
        guideSteps.append(item);
      });
      $('vive-input-ready').textContent = setup.input_ready
        ? '已收到新鲜有效数据'
        : uiValue(inputPreflight.message || '未就绪');
      $('vive-profile-ready').textContent = setup.profile_ready
        ? text(setup.selected_profile_path)
        : setup.unsaved_calibration
        ? '标定完成但尚未保存'
        : '没有可用的已保存坐标配置';
      $('mujoco-construction-state').textContent = simPreviewReady
        ? '静态预览 · 控制与记录未启动'
        : deferredHumanSetup
        ? '已构造 · 尚未接入 Human 输入'
        : runtimeConstructed
        ? '已构造并接管 Human 输入'
        : '未构造（等待设置）';
      const canCapture = workflowActive
        && ['ready', 'waiting', 'error'].includes(calibrationPhase)
        && ['virtual_desk_x', 'virtual_desk_y', 'virtual_desk_z', 'wrist_offsets']
          .includes(calibrationStage);
      $('begin-vive-calibration').disabled = acceptanceBusy
        || !calibrationAvailable || workflowActive
        || !Boolean(inputPreflight.ready);
      explainDisabled(
        $('begin-vive-calibration'),
        busyReason || (!calibrationAvailable
          ? '当前配置不提供 VIVE 标定。'
          : workflowActive ? '标定已经开始。' : '先等待有效 VIVE 输入。'),
      );
      $('capture-vive-calibration').disabled = acceptanceBusy || !canCapture;
      $('reset-vive-calibration').disabled = acceptanceBusy
        || !workflowActive || !canCapture;
      $('cancel-vive-calibration').disabled = acceptanceBusy || !workflowActive;
      $('save-vive-calibration').disabled = acceptanceBusy
        || !calibrationAvailable || calibrationStage !== 'completed';
      $('vive-profile-name').disabled = acceptanceBusy
        || !calibrationAvailable || calibrationStage !== 'completed';
      const target = $('control-modes');
      target.replaceChildren();
      const runtimeStarted = Boolean(value.started);
      const collisionTerminal = Boolean(
        runtimeStarted
        && value.episode_ended
        && String(value.episode_end_reason || '').startsWith(
          'sim_collision_stop:'
        )
      );
      // The acceptance controller owns which typed start intent is legal.
      // Do not infer that capability from a runtime/bootstrap profile label:
      // those projections can change independently while the application is
      // being constructed or while a cached status is served.
      const exclusiveProfile = Boolean(
        acceptance.exclusive_hitl_enabled
      );
      if (exclusiveProfile && !runtimeConstructed) {
        $('vive-setup-title').textContent =
          'Human 输入模块 · VIVE、触发器与坐标配置';
      }
      // Exclusive takeover uses the allocation streaming backend internally,
      // but remains a restricted two-button application profile. Do not expose
      // the generic automatic/relative allocation controls in that UI.
      const allocationProfile = Boolean(
        acceptance.allocation_runtime_enabled
      ) && !exclusiveProfile;
      const availableControlModes = new Set(
        acceptance.available_control_modes || []
      );
      const selectedControlProfile = String(
        value.control_profile_id || 'fixed_alpha_compatibility'
      );
      const allocationSupportsAbsolute = selectedControlProfile.includes(
        'scheme_b'
      );
      $('page-summary').textContent = humanDemonstration
        ? '先调整静态仿真视角，再一次性开始 Human-only 遥操与记录；封存完成即自动保存。'
        : allocationProfile
        ? '动态分配以策略基线启动；VIVE 与夹爪只在首次进入 Human 模式时接入。'
        : exclusiveProfile
        ? '策略与 Human 通过两个显式触发器整段切换；不执行比例融合。'
        : '可选择纯 PPO、纯遥操或固定比例融合；纯 PPO 不要求 VIVE 先就绪。';
      $('page-title').textContent = humanDemonstration
        ? 'Human Demo 采集 · MuJoCo'
        : '仿真在环控制与 VIVE 标定';
      $('operation-title').textContent = humanDemonstration
        ? '本次轨迹'
        : '操作区 · Sim-HIL 正式控制';
      $('acceptance-description').textContent = humanDemonstration
        ? 'PPO、比例融合、动态分配和策略超时回退均被当前采集配置禁止。'
        : allocationProfile
        ? (allocationSupportsAbsolute
          ? '可在策略基线、自动分配、人类相对修正和人类绝对接管之间切换。'
          : '可在策略基线、自动分配和人类相对修正之间切换；当前方案不提供绝对接管。')
        : exclusiveProfile
        ? 'Human 接管前自动 HOLD/Home；交还策略时清空旧动作并从当前状态重新开始。'
        : '启动后可切换纯遥操、纯 PPO 和比例融合；切换到 Human 相关模式时才接纳 VIVE。';
      $('acceptance-description').hidden = humanDemonstration;
      target.hidden = allocationProfile || exclusiveProfile || humanDemonstration;
      $('human-demo-workflow').hidden = !humanDemonstration;
      $('runtime-controls').hidden = humanDemonstration;
      $('vive-setup-panel').hidden = Boolean(
        humanDemonstration
        && (runtimeStarted || (acceptance.capture || {}).status === 'committed')
      );
      if (humanDemonstration) {
        renderHumanDemoWorkflow(value, acceptanceBusy);
      }
      $('exclusive-source-modules').hidden = !exclusiveProfile;
      $('streaming-control-controls').hidden = !allocationProfile;
      $('fixed-alpha-controls').hidden = allocationProfile || exclusiveProfile
        || humanDemonstration;
      $('blend-failure-controls').hidden = allocationProfile || exclusiveProfile
        || humanDemonstration;
      $('human-timeout-controls').hidden = allocationProfile || exclusiveProfile
        || humanDemonstration;
      $('allocation-controls').hidden = !allocationProfile;
      $('streaming-control-help').textContent = allocationProfile
        ? '按当前 profile 在自动分配、策略基线、相对修正和可选绝对接管之间切换。'
        : '当前 profile 不使用流式分配。';
      $('streaming-control-title').textContent = '策略 / Human 流式控制';
      const reportedMixMode = ((value.composition || {}).mix || {}).mode;
      const blendFallback = value.blend_failure_fallback || {};
      const fallbackMode = {
        human: 'human_teleop',
        policy: 'ppo_only',
      }[blendFallback.latched_source] || null;
      const activeMode = fallbackMode || acceptance.active_control_mode || (
        runtimeStarted
          ? ({
              human_only: 'human_teleop',
              policy_only: 'ppo_only',
              proportional_blend: 'proportional_blend',
            }[reportedMixMode] || null)
          : null
      );
      const episodeEnded = Boolean(value.episode_ended);
      const rawActiveAlpha = acceptance.active_human_alpha;
      const activeAlpha = rawActiveAlpha === null || rawActiveAlpha === undefined
        ? Number.NaN : Number(rawActiveAlpha);
      const compositionAlphaByComponent = (((value.composition || {}).mix || {})
        .human_alpha_by_component || {});
      const compositionAlphaLabel = Object.entries(compositionAlphaByComponent)
        .map(([componentId, alpha]) => `${componentId}=${Number(alpha).toFixed(2)}`)
        .join(', ');
      $('active-blend-alpha').textContent = Number.isFinite(activeAlpha)
        ? `已接纳 α：${activeAlpha.toFixed(2)} · composition：${compositionAlphaLabel || '等待下一周期'}`
        : 'α 状态：尚未选择 fixed-alpha 模式';
      const requestedAlpha = Number($('blend-alpha').value);
      $('apply-blend-alpha').disabled = allocationProfile
        || !availableControlModes.has('proportional_blend')
        || acceptanceBusy
        || !runtimeStarted
        || Boolean(value.held)
        || episodeEnded
        || !humanRuntimeLaunchReady
        || !Number.isFinite(requestedAlpha)
        || requestedAlpha <= 0
        || requestedAlpha >= 1;
      explainDisabled(
        $('apply-blend-alpha'),
        busyReason || (!runtimeStarted
          ? '先启动 fixed-alpha 运行会话。'
          : value.held ? '先恢复运行。'
          : !humanRuntimeLaunchReady ? '等待 Human 输入与坐标配置就绪。'
          : '请输入 0 与 1 之间的比例。'),
      );
      if (!allocationProfile && !exclusiveProfile && !humanDemonstration) {
        for (const modeId of Object.keys(controlModeNames)) {
          if (!availableControlModes.has(modeId)) continue;
          const modeLaunchReady = modeId === 'ppo_only'
            ? policyRuntimeLaunchReady
            : humanRuntimeLaunchReady;
          const modeCard = document.createElement('div');
          modeCard.className = 'mode-card';
          const title = document.createElement('h3');
          title.textContent = controlModeNames[modeId];
          const description = document.createElement('p');
          description.textContent = controlModeDescriptions[modeId];
          const state = document.createElement('div');
          state.className = 'mode-state';
          state.textContent = activeMode === modeId
            ? '当前模式'
            : runtimeStarted ? '可随时切换' : '可作为启动模式';
          const button = document.createElement('button');
          button.textContent = !modeLaunchReady
            ? (modeId === 'ppo_only'
              ? '策略运行时尚未就绪'
              : '等待 VIVE / 坐标配置')
            : activeMode === modeId
            ? '当前模式'
            : runtimeStarted
            ? '切换到此模式'
            : '以此模式启动';
          button.disabled = acceptanceBusy
            || episodeEnded
            || activeMode === modeId
            || !modeLaunchReady;
          explainDisabled(
            button,
            busyReason || (episodeEnded
              ? '当前回合已经结束。'
              : activeMode === modeId
              ? '这已经是当前模式。'
              : 'Human 相关模式需要有效 VIVE 输入与已保存坐标配置。'),
          );
          button.addEventListener('click', () => {
            const alpha = modeId === 'proportional_blend'
              ? Number($('blend-alpha').value) : null;
            submitIntent(
              runtimeStarted ? 'set_control_mode' : 'start_runtime',
              modeId,
              alpha,
            );
          });
          modeCard.append(title, description, state, button);
          target.append(modeCard);
        }
      }
      if (exclusiveProfile) {
        const switchEvidence = acceptance.last_exclusive_hitl_switch || {};
        const cycleTiming = ((value.control_schedule || {}).last_cycle_timing || {});
        const policyStatus = value.policy || {};
        const humanStatus = value.human || {};
        const policyError = policyStatus.last_error
          || policyStatus.worker_error
          || null;
        const humanError = humanStatus.last_rejection_reason
          || ((humanStatus.last_step || {}).rejection_reason)
          || null;
        const held = Boolean(value.held);
        const policyActive = runtimeStarted && activeMode === 'ppo_only';
        const humanActive = runtimeStarted && activeMode === 'human_teleop';
        const sourceOperationBlocked = acceptanceBusy || episodeEnded
          || Boolean(value.held);
        $('exclusive-source-selection-state').textContent = !runtimeStarted
          ? '请选择启动来源'
          : held ? '当前 HOLD'
          : humanActive ? '当前 Human'
          : policyActive ? '当前 Policy' : '状态待确认';
        $('exclusive-source-selection-state').className =
          `setup-state ${runtimeStarted && !held ? 'ready' : ''}`;
        $('exclusive-policy-module').className =
          `source-module ${policyActive ? 'selected' : !policyRuntimeLaunchReady ? 'blocked' : ''}`;
        $('exclusive-human-module').className =
          `source-module ${humanActive ? 'selected' : !humanRuntimeLaunchReady ? 'blocked' : ''}`;
        $('exclusive-policy-module-state').textContent = policyError
          && !policyRuntimeLaunchReady
          ? `未就绪：${uiValue(policyError)}`
          : policyRuntimeLaunchReady
          ? (runtimeStarted
            ? (policyActive ? '当前动作由 Policy 下发' : '待命；Human 控制周期不会同步等待模型')
            : '已就绪，可作为启动来源')
          : '策略模型尚未就绪';
        $('exclusive-human-module-state').textContent = humanError
          && !humanRuntimeLaunchReady
          ? `最近输入异常：${uiValue(humanError)}`
          : humanRuntimeLaunchReady
          ? (runtimeStarted
            ? (humanActive ? '当前动作由 Human 下发' : '待命；切换时会先 HOLD 并采集 Runtime Home')
            : 'VIVE、坐标配置和触发器已就绪')
          : '等待 VIVE、坐标配置或触发器';
        for (const [badgeId, active, ready] of [
          ['exclusive-policy-badge', policyActive, policyRuntimeLaunchReady],
          ['exclusive-human-badge', humanActive, humanRuntimeLaunchReady],
        ]) {
          const badge = $(badgeId);
          badge.textContent = active ? '当前控制' : ready ? '可用' : '未就绪';
          badge.className = `source-module-badge ${
            active ? 'active' : ready ? 'ready' : 'blocked'
          }`;
        }
        const policyButton = $('start-exclusive-policy-runtime');
        const humanButton = $('start-exclusive-human-runtime');
        policyButton.hidden = false;
        humanButton.hidden = false;
        policyButton.setAttribute('aria-pressed', String(policyActive));
        humanButton.setAttribute('aria-pressed', String(humanActive));
        policyButton.textContent = acceptanceBusy
          ? '操作处理中…'
          : policyActive ? 'Policy 正在控制'
          : !policyRuntimeLaunchReady ? 'Policy 未就绪'
          : runtimeStarted && held ? '从 HOLD 恢复到 Policy'
          : runtimeStarted ? '切换到 Policy' : '以 Policy 启动';
        humanButton.textContent = acceptanceBusy
          ? '操作处理中…'
          : humanActive ? 'Human 正在控制'
          : !humanRuntimeLaunchReady ? 'Human 未就绪'
          : runtimeStarted && held ? '从 HOLD 恢复到 Human'
          : runtimeStarted ? '切换到 Human' : '以 Human 启动';
        policyButton.disabled = sourceOperationBlocked || policyActive
          || !policyRuntimeLaunchReady;
        humanButton.disabled = sourceOperationBlocked || humanActive
          || !humanRuntimeLaunchReady;
        const generalSourceReason = busyReason || (episodeEnded
          ? '当前回合已经结束，只能执行收尾。' : '');
        const policyActionReason = generalSourceReason || (policyActive
          ? 'Policy 已经是当前控制来源。'
          : !policyRuntimeLaunchReady ? '等待策略模型产生可接纳输出。'
          : held
          ? '保持不会先被解除；事务会重建 lower HOLD 证据、清空旧计划，并仅在新 Policy 目标被接纳后恢复。'
          : runtimeStarted
          ? '切换会先 HOLD、清空旧计划，再从当前状态恢复 Policy。'
          : '启动时 Human 不参与动作。');
        const humanActionReason = generalSourceReason || (humanActive
          ? 'Human 已经是当前控制来源。'
          : !humanRuntimeLaunchReady
          ? '需要新鲜 VIVE 输入、有效坐标配置和可用触发器。'
          : held
          ? '保持不会先被解除；切换会在 HOLD 中采集新 Home，并仅在 Human 新目标被接纳后恢复。'
          : runtimeStarted
          ? '切换会先 HOLD，以当前姿态采集 Runtime Home 后再接纳 Human。'
          : '启动前会采集 Runtime Home，通过后才进入 Human 控制。');
        $('exclusive-policy-action-reason').textContent = policyActionReason;
        $('exclusive-human-action-reason').textContent = humanActionReason;
        explainDisabled(
          policyButton,
          policyActionReason,
        );
        explainDisabled(
          humanButton,
          humanActionReason,
        );
        const ownerName = humanActive ? 'Human' : policyActive ? 'Policy' : '尚未启动';
        $('exclusive-current-owner').textContent = held && runtimeStarted
          ? `当前控制：HOLD（保留 ${ownerName} 选择）`
          : `当前控制：${ownerName}`;
        const switchDurationMs = Number(switchEvidence.completed_at)
          - Number(switchEvidence.requested_at);
        $('exclusive-source-transition').textContent = switchEvidence.status
          ? `最近切换：${switchEvidence.target_source === 'human' ? '转入 Human' : '交还 Policy'} · ${
              uiValue(switchEvidence.status)
            }${Number.isFinite(switchDurationMs) ? ` · ${(1000 * switchDurationMs).toFixed(0)} ms` : ''}`
          : runtimeStarted ? '尚未进行运行中来源切换。' : '任选一个已就绪来源启动。';
        $('exclusive-hitl-operation-status').textContent = switchEvidence.status
          ? '来源切换结果已同步到上方来源卡；完整回执位于高级诊断。'
          : '来源启动与切换统一在上方两张卡片中操作。';
        $('exclusive-cycle-timing').textContent = Number.isFinite(
          Number(cycleTiming.total_sec)
        )
          ? `最近控制周期：总计 ${(1000 * Number(cycleTiming.total_sec)).toFixed(1)} ms`
            + ` · Policy ${(1000 * Number(cycleTiming.policy_source_sec || 0)).toFixed(1)} ms`
            + ` · 组合/执行 ${(1000 * Number(cycleTiming.composition_and_execution_sec || 0)).toFixed(1)} ms`
          : '控制周期：等待首个正式周期。';
      }
      const allocationOperations = ((value.composition || {})
        .allocation_operations || {});
      const allocationMode = allocationOperations.allocation_override_mode
        || 'automatic';
      const effectiveAllocationModes = (((value.composition || {}).last_cycle
        || {}).composition_step || {}).allocation?.selected_mode_by_component || {};
      const effectiveAllocationModeSet = new Set(
        Object.values(effectiveAllocationModes)
      );
      const effectiveAllocationLabel = effectiveAllocationModeSet.size === 1
        ? (allocationModeNames[[...effectiveAllocationModeSet][0]]
          || [...effectiveAllocationModeSet][0])
        : effectiveAllocationModeSet.size > 1
        ? '按 component 分配'
        : '等待首个正式周期';
      const supportedAllocationSignals = new Set(
        allocationOperations.supported_signals || []
      );
      const allocationUnavailable = acceptanceBusy
        || !runtimeStarted || episodeEnded || Boolean(value.held);
      $('start-allocation-runtime').hidden = runtimeStarted;
      $('start-allocation-runtime').textContent = '以策略基线启动动态分配';
      $('start-allocation-runtime').disabled = acceptanceBusy
        || !allocationProfile || runtimeStarted || episodeEnded
        || !policyRuntimeLaunchReady;
      const guidanceBlockers = (value.operator_guidance?.blockers || [])
        .map(item => item.message || item.code).join('；');
      explainDisabled(
        $('start-allocation-runtime'),
        busyReason || (episodeEnded
          ? '当前回合已经结束。'
          : runtimeStarted ? '运行会话已经启动。'
          : guidanceBlockers || '启动条件尚未满足。'),
      );
      for (const [buttonId, modeId, signalName] of [
        ['allocation-automatic', 'automatic', 'allocation_automatic'],
        ['allocation-base-only', 'base_only', 'allocation_base_only'],
        ['allocation-relative-overlay', 'relative_overlay', 'allocation_relative_overlay'],
        ['allocation-absolute-overlay-owner', 'absolute_overlay_owner', 'allocation_absolute_overlay_owner'],
      ]) {
        const button = $(buttonId);
        if (modeId === 'absolute_overlay_owner') {
          button.hidden = !supportedAllocationSignals.has(signalName);
        }
        const modeNeedsHuman = modeId !== 'base_only';
        const waitingForHuman = modeNeedsHuman
          && deferredHumanSetup && !humanRuntimeLaunchReady;
        // Automatic may already be the profile default while Human is still
        // deferred.  Keep it clickable once Human becomes ready so that the
        // same intent performs the typed attachment before reasserting auto.
        const currentModeIsActionableAttachment = modeId === 'automatic'
          && deferredHumanSetup && humanRuntimeLaunchReady;
        button.disabled = allocationUnavailable || waitingForHuman
          || (allocationMode === modeId && !currentModeIsActionableAttachment);
        explainDisabled(
          button,
          busyReason || (!runtimeStarted
            ? '先启动动态分配运行会话。'
            : value.held ? '模式会在 HOLD 中接纳；随后仍需显式恢复运行。'
            : episodeEnded ? '当前回合已经结束。'
            : waitingForHuman
            ? '等待新鲜 VIVE 输入和有效坐标配置；策略基线可继续运行。'
            : '这已经是当前模式。'),
        );
      }
      const rebaseWaitingForHuman = deferredHumanSetup
        && !humanRuntimeLaunchReady;
      $('allocation-rebase-and-clear').disabled = allocationUnavailable
        || rebaseWaitingForHuman;
      explainDisabled(
        $('allocation-rebase-and-clear'),
        busyReason || (!runtimeStarted
          ? '先启动动态分配运行会话。'
          : value.held ? '将在 HOLD 中重新定基并清除修正；完成后仍需显式恢复。'
          : rebaseWaitingForHuman
          ? '等待 Human 输入与坐标配置就绪。'
          : '当前回合已经结束。'),
      );
      $('allocation-mode-status').textContent = allocationProfile
        ? (runtimeStarted
          ? `分配规则：${allocationModeNames[allocationMode] || allocationMode}`
            + ` · 本周期实际选择：${effectiveAllocationLabel}`
          : '尚未启动 · 可先使用策略基线')
        : '—';
      if (allocationProfile) {
        $('streaming-control-status').textContent = runtimeStarted
          ? `动态分配 · ${allocationModeNames[allocationMode] || allocationMode}`
          : '动态分配待启动';
      }
      $('allocation-operation-status').textContent = !runtimeStarted
        ? '等待 Runtime 启动。'
        : allocationOperations.last_operation_signal
        ? `最近操作：${allocationSignalNames[
            allocationOperations.last_operation_signal
          ] || allocationOperations.last_operation_signal} · revision ${text(
            allocationOperations.allocation_override_revision
          )}`
        : '尚未进行手动分配或 rebase。';
      const swapButton = $('swap-vive-sources');
      const humanDemoPreviewCanSwap = humanDemonstration
        && simPreviewReady && !runtimeStarted;
      const viveMappingFrozen = runtimeConstructed
        && !deferredHumanSetup && !humanDemoPreviewCanSwap;
      swapButton.disabled = acceptanceBusy
        || workflowActive
        || viveMappingFrozen;
      swapButton.title = '只交换左右来源映射，不检查来源状态。';
      $('vive-swap-status').textContent = viveMappingFrozen
        ? '运行时已冻结来源映射'
        : lastResult.intent_kind === 'swap_vive_sources'
          && lastResult.status === 'swapped'
        ? '已交换左右来源；当前映射已更新'
        : '只交换左右来源映射，不检查来源状态';
      const fallbackSelector = $('blend-failure-fallback');
      const blendFailureGraceInput = $('blend-failure-grace');
      if (document.activeElement !== fallbackSelector) {
        fallbackSelector.value = blendFallback.selected_source || 'hold';
      }
      if (document.activeElement !== blendFailureGraceInput) {
        blendFailureGraceInput.value = Number(
          blendFallback.grace_sec ?? 0
        ).toFixed(2);
      }
      fallbackSelector.disabled = acceptanceBusy || !runtimeStarted;
      blendFailureGraceInput.disabled = acceptanceBusy || !runtimeStarted;
      $('apply-blend-failure-fallback').disabled =
        acceptanceBusy || !runtimeStarted;
      const fallbackActivation = blendFallback.last_activation || {};
      $('blend-failure-fallback-status').textContent =
        blendFallback.latched_source
          ? `已因融合目标不可执行而锁存到${
              blendFallback.latched_source === 'human' ? '人类' : '策略'
            }；重新选择比例融合可解除。`
          : fallbackActivation.status === 'grace_active'
          ? `融合目标仍由下游立即保持；连续失败 ${Number(
              fallbackActivation.failure_elapsed_sec || 0
            ).toFixed(2)} / ${Number(blendFallback.grace_sec || 0).toFixed(2)} 秒后才切换来源。`
          : fallbackActivation.status === 'hold_selected'
          ? '最近一次融合目标不可执行，已保持静止。'
          : fallbackSelector.value === 'hold'
          ? '静止在首次不可执行时立即生效；切换延迟不适用于静止。'
          : `连续不可执行达到 ${Number(
              blendFallback.grace_sec || 0
            ).toFixed(2)} 秒后切换来源；每个失败周期仍立即保持。`;
      const humanTimeoutFallback = value.human_timeout_fallback || {};
      const humanTimeoutSelector = $('human-timeout-fallback');
      if (document.activeElement !== humanTimeoutSelector) {
        humanTimeoutSelector.value = humanTimeoutFallback.selected_source || 'hold';
      }
      humanTimeoutSelector.disabled = acceptanceBusy || !runtimeStarted;
      $('apply-human-timeout-fallback').disabled =
        acceptanceBusy || !runtimeStarted;
      const humanTimeoutLast = humanTimeoutFallback.last_activation || {};
      const timeoutSec = humanTimeoutFallback.timeout_sec || 5;
      $('human-timeout-fallback-status').textContent =
        humanTimeoutFallback.latched
          ? '人类输入已超时，已锁存到 PPO；恢复人类模式需人工切换。'
          : humanTimeoutLast.status === 'hold_selected'
          ? `人类输入超过 ${timeoutSec} 秒未更新，已保持静止。`
          : `超过 ${timeoutSec} 秒无新的人类输出时触发；默认静止。`;
      const collisionRecovery = $('collision-recovery');
      collisionRecovery.hidden = !collisionTerminal;
      if (collisionTerminal) {
        const executionFault = value.operator_guidance?.execution_fault || {};
        $('collision-recovery-detail').textContent = [
          executionFault.reason_code || executionFault.outcome_code,
          executionFault.message,
          executionFault.collision_summary,
        ].filter(Boolean).join(' · ')
          || '代理安全间隙保护已经终止当前 episode；不能原地恢复。';
        $('collision-reset-runtime').disabled = acceptanceBusy;
      }
      $('hold-runtime').disabled = acceptanceBusy || !value.started || value.held;
      $('hold-runtime').hidden = !value.started || Boolean(value.held)
        || Boolean(value.episode_ended);
      explainDisabled(
        $('hold-runtime'),
        busyReason || (!value.started ? '运行会话尚未启动。' : '输出已经保持。'),
      );
      $('rearm-runtime').disabled = acceptanceBusy || !value.started || !value.held
        || value.episode_ended;
      $('rearm-runtime').hidden = !value.started || !Boolean(value.held)
        || Boolean(value.episode_ended);
      $('rearm-runtime').textContent = '检查后恢复运行';
      explainDisabled(
        $('rearm-runtime'),
        busyReason || (!value.started
          ? '运行会话尚未启动。'
          : value.episode_ended ? '已结束的回合不能恢复。' : '当前正在运行。'),
      );
      $('capture-runtime-home').disabled = acceptanceBusy || !value.started
        || !value.held || value.episode_ended || deferredHumanSetup;
      $('capture-runtime-home').hidden = !value.started || !Boolean(value.held)
        || Boolean(value.episode_ended) || deferredHumanSetup;
      explainDisabled(
        $('capture-runtime-home'),
        busyReason || (!value.started
          ? '运行会话尚未启动。'
          : !value.held ? '必须先保持输出。'
          : deferredHumanSetup ? 'Human 输入尚未接入。' : '当前回合已经结束。'),
      );
      const runtimeHeld = Boolean(value.held);
      const runtimeAvailable = runtimeConstructed && !value.episode_ended;
      $('runtime-primary-actions').hidden = !value.started
        || Boolean(value.episode_ended);
      $('runtime-control-state').textContent = !runtimeConstructed
        ? '尚未构造'
        : collisionTerminal ? '代理安全间隙保护 · 必须 Reset'
        : runtimeHeld ? '已保持，可恢复或重建' : '运行中';
      $('runtime-control-state').className = `setup-state ${
        runtimeConstructed && !runtimeHeld ? 'ready' : ''
      }`;
      $('runtime-control-help').textContent = !runtimeConstructed
        ? '选择已就绪的控制来源后才会构造 Runtime；启动前不会下发动作。'
        : collisionTerminal
        ? '代理安全间隙保护已终止当前 episode；这不代表已发生物理接触。来源切换、重采 Home 和 rearm 均不可用，只能 Reset 仿真。'
        : runtimeHeld
        ? (deferredHumanSetup
          ? '当前为纯 PPO 保持状态；Human 输入接入后才可采集 Runtime Home。'
          : '当前已保持；VIVE 仍在低频被动刷新。可采集当前 Runtime Home，再恢复运行。')
        : (deferredHumanSetup
          ? '纯 PPO 正在运行；VIVE 与坐标配置不会阻塞策略执行。'
          : 'Runtime Home 已绑定；需要重采时先保持输出。完整重建会结束当前 MuJoCo 仿真。');
      $('rebuild-runtime').disabled = acceptanceBusy || !runtimeAvailable;
      $('rebuild-runtime').hidden = !runtimeConstructed
        || Boolean(value.episode_ended);
      $('rebuild-runtime').title =
        '销毁当前 MuJoCo 仿真并回到 VIVE 准备页；未封存记录会明确中止。';
      const operationCanCancel = Boolean(
        acceptance.operation_cancel_available || intentPending
      );
      $('cancel-operation').hidden = !operationCanCancel;
      $('cancel-operation').disabled = !operationCanCancel;
      $('runtime-operation-summary').textContent = acceptanceBusy
        ? `正在处理：${intentOperationLabel(
            acceptance.active_intent_kind || pendingIntentKind
          )}`
        : acceptance.last_error
        ? '最近操作失败；请按上方故障引导处理。'
        : '当前没有等待中的操作。';
      const capture = acceptance.capture || {};
      const captureWriter = capture.writer || {};
      $('finalize-capture').disabled = acceptanceBusy || !value.started
        || value.episode_ended || capture.status !== 'recording';
      $('finalize-capture').hidden = capture.status !== 'recording'
        || Boolean(value.episode_ended);
      explainDisabled(
        $('finalize-capture'),
        busyReason || (!value.started
          ? '运行会话尚未启动。'
          : value.episode_ended ? '当前回合已经结束。' : '记录链路不在 recording 状态。'),
      );
      $('stop-runtime').disabled = Boolean(acceptance.shutdown_requested);
      $('acceptance-label').textContent = humanDemonstration
        ? text(
            (value.operator_guidance?.human_demo_workflow || {}).state_label
            || '准备中'
          )
        : collisionTerminal
        ? '代理安全间隙保护 · 等待 Reset'
        : acceptance.last_error
        ? '故障'
        : allocationProfile && runtimeStarted
        ? `当前：${allocationModeNames[allocationMode] || allocationMode}`
        : allocationProfile
        ? '动态分配待启动'
        : activeMode
        ? `当前：${controlModeNames[activeMode] || uiValue(activeMode)}`
        : exclusiveProfile && humanRuntimeLaunchReady && policyRuntimeLaunchReady
        ? 'Policy / Human 均可启动'
        : exclusiveProfile && humanRuntimeLaunchReady
        ? 'Human 可以启动'
        : policyRuntimeLaunchReady
        ? '纯 PPO 可以启动'
        : '等待策略运行时';
      rows($('capture-rows'), [
        ['记录状态', capture.status], ['Trace 会话', capture.trace_session_id],
        ['回合', capture.episode_uuid], ['封存状态', capture.commit_status],
        ['写入故障', captureWriter.last_failure_code || captureWriter.fatal_error_type],
        ['已写入', captureWriter.write_offset_bytes],
        ['当前容量', captureWriter.allocated_bytes],
        ['存储目录', capture.session_root], ['说明', capture.message],
      ]);
      if (intentPending) {
        setIntentMessage(
          `正在处理：${intentOperationLabel(pendingIntentKind)}`,
          'pending',
        );
      } else if (
        acceptance.last_error
        || value.operator_guidance?.severity === 'error'
      ) {
        const guidance = value.operator_guidance || {};
        setIntentMessage(guidance.headline
          ? `${guidance.headline}；原始错误见上方或高级诊断。`
          : uiValue(acceptance.last_error), 'error');
      } else if (acceptance.last_result) {
        const result = acceptance.last_result;
        const resultValue = result.message || result.status || result.intent_kind;
        setIntentMessage(
          resultValue ? uiValue(resultValue) : '操作已完成',
          result.status === 'failed' ? 'error' : 'success',
        );
      } else {
        setIntentMessage('尚未提交操作。');
      }
    }
    const intentOperationLabels = {
      prepare_sim_preview: '打开静态仿真',
      select_control_profile: '选择控制工作流',
      select_sim_collision_stop: '选择 Sim 碰撞处理',
      start_runtime: '启动所选控制来源',
      request_human_takeover: '切换到 Human',
      request_policy_resume: '切换到 Policy',
      start_allocation_runtime: '启动动态分配',
      cancel_operation: '中断当前操作',
      finalize_capture: '封存并自动保存',
      hold: '保持输出',
      rearm: '恢复遥操',
      capture_runtime_home: '重采 Runtime Home',
      rebuild_runtime: '放弃当前回合并重新准备',
      stop_runtime: '退出控制',
      swap_vive_sources: '交换 VIVE 来源',
    };
    function intentOperationLabel(intentKind) {
      return intentOperationLabels[intentKind] || uiValue(intentKind || '操作');
    }
    async function waitForIntentSettlement(intentId, intentKind) {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        const value = await refresh();
        const result = value?.acceptance?.last_result || null;
        if (result?.intent_id === intentId) {
          if (intentKind === 'swap_vive_sources') await refreshVive();
          return {settled: true, result, value};
        }
        if (value?.acceptance?.shutdown_requested) {
          return {settled: true, result: {status: 'shutdown_requested'}, value};
        }
        await new Promise(resolve => window.setTimeout(resolve, 250));
      }
      return {settled: false, result: null, value: latestStatus};
    }
    async function submitIntent(
      intentKind,
      controlMode = null,
      humanAlpha = null,
      profileName = null,
      fallbackSource = null,
      allocationMode = null,
      blendFailureGraceSec = null,
      controlProfileId = null,
      simCollisionStopEnabled = null,
    ) {
      const outOfBand = ['cancel_operation', 'stop_runtime'].includes(intentKind);
      if (!latestStatus || (intentPending && !outOfBand)) return;
      const acceptance = latestStatus.acceptance || {};
      const payload = {
        client_id: clientId,
        client_sequence: nextClientSequence(),
        expected_revision: acceptance.revision,
        intent_kind: intentKind,
        reason: 'sim_hil_acceptance_gui',
      };
      if (controlMode !== null) payload.control_mode = controlMode;
      if (humanAlpha !== null) payload.human_alpha = humanAlpha;
      if (profileName !== null) payload.profile_name = profileName;
      if (fallbackSource !== null) payload.fallback_source = fallbackSource;
      if (allocationMode !== null) payload.allocation_mode = allocationMode;
      if (controlProfileId !== null) {
        payload.control_profile_id = controlProfileId;
      }
      if (simCollisionStopEnabled !== null) {
        payload.sim_collision_stop_enabled = simCollisionStopEnabled;
      }
      if (blendFailureGraceSec !== null) {
        payload.blend_failure_grace_sec = blendFailureGraceSec;
      }
      if (!outOfBand) {
        intentPending = true;
        pendingIntentKind = intentKind;
      }
      setIntentMessage(`正在提交：${intentOperationLabel(intentKind)}`, 'pending');
      let acceptedIntentId = null;
      try {
        const response = await fetch('/api/acceptance-intents', {
          method: 'POST',
          cache: 'no-store',
          headers: {
            'Content-Type': 'application/json',
            'X-Sim-HIL-Control-Token': controlToken,
          },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok || !result.accepted) {
          throw new Error(result.message || result.error || `HTTP ${response.status}`);
        }
        acceptedIntentId = result.intent_id;
        pendingIntentId = acceptedIntentId;
        setIntentMessage(
          `已提交：${intentOperationLabel(intentKind)}；等待完成…`,
          'pending',
        );
        if (outOfBand) {
          await refresh();
          return;
        }
        const settled = await waitForIntentSettlement(
          acceptedIntentId,
          intentKind,
        );
        if (!settled.settled) {
          setIntentMessage(
            `${intentOperationLabel(intentKind)}仍在执行；页面会继续显示服务端状态。`,
            'pending',
          );
        } else if (settled.result?.status === 'failed') {
          const guidance = settled.value?.operator_guidance || {};
          setIntentMessage(guidance.headline
            ? `${guidance.headline}；按主操作可重试或查看诊断。`
            : `操作失败：${uiValue(settled.result.error)}`, 'error');
        } else {
          setIntentMessage(
            `${intentOperationLabel(intentKind)}已完成。`,
            'success',
          );
        }
      } catch (error) {
        setIntentMessage(String(error), 'error');
      } finally {
        if (!outOfBand) {
          intentPending = false;
          pendingIntentId = null;
          pendingIntentKind = null;
        }
        if (!acceptedIntentId) await refresh();
      }
    }
    const lifecycleConfirmations = {
      rebuild_runtime: '这会结束当前 MuJoCo 仿真；未封存记录会被明确中止。确认继续吗？',
      finalize_capture: '这会结束当前回合，完成封存并自动保存记录；之后不能恢复本回合。确认继续吗？',
      stop_runtime: '这会保持输出并退出当前控制进程。确认继续吗？',
    };
    function confirmLifecycleIntent(intentKind) {
      const message = lifecycleConfirmations[intentKind];
      return !message || window.confirm(message);
    }
    function submitLifecycleIntent(intentKind) {
      if (!confirmLifecycleIntent(intentKind)) return;
      submitIntent(intentKind);
    }
    function executeOperatorAction(action) {
      if (!action) return;
      if (action.interaction === 'focus') {
        const target = $(action.target_id);
        if (!target) return;
        if (target.tagName === 'DETAILS') target.open = true;
        target.scrollIntoView({behavior: 'smooth', block: 'start'});
        return;
      }
      if (action.interaction !== 'intent') return;
      if (!confirmLifecycleIntent(action.intent_kind)) return;
      submitIntent(
        action.intent_kind,
        action.control_mode || null,
        action.human_alpha ?? null,
      );
    }
    $('operator-primary-action').addEventListener('click', () => {
      executeOperatorAction(
        latestStatus?.operator_guidance?.recommended_action
      );
    });
    $('human-demo-primary-action').addEventListener('click', () => {
      executeOperatorAction(
        latestStatus?.operator_guidance?.human_demo_workflow?.primary_action
      );
    });
    $('demo-swap-vive').addEventListener(
      'click',
      () => submitIntent('swap_vive_sources'),
    );
    $('demo-hold').addEventListener('click', () => submitIntent('hold'));
    $('demo-rearm').addEventListener('click', () => submitIntent('rearm'));
    $('demo-cancel-operation').addEventListener(
      'click',
      () => submitIntent('cancel_operation'),
    );
    $('demo-capture-home').addEventListener(
      'click',
      () => submitIntent('capture_runtime_home'),
    );
    $('demo-rebuild').addEventListener(
      'click',
      () => submitLifecycleIntent('rebuild_runtime'),
    );
    $('demo-abandon').addEventListener(
      'click',
      () => submitLifecycleIntent('stop_runtime'),
    );
    $('swap-vive-sources').addEventListener(
      'click',
      () => submitIntent('swap_vive_sources'),
    );
    $('begin-vive-calibration').addEventListener(
      'click',
      () => submitIntent('begin_vive_calibration'),
    );
    $('capture-vive-calibration').addEventListener(
      'click',
      () => submitIntent('capture_vive_calibration'),
    );
    $('reset-vive-calibration').addEventListener(
      'click',
      () => submitIntent('reset_vive_calibration_stage'),
    );
    $('cancel-vive-calibration').addEventListener(
      'click',
      () => submitIntent('cancel_vive_calibration'),
    );
    $('save-vive-calibration').addEventListener(
      'click',
      () => submitIntent(
        'save_vive_calibration_profile',
        null,
        null,
        $('vive-profile-name').value.trim(),
      ),
    );
    $('apply-blend-failure-fallback').addEventListener(
      'click',
      () => submitIntent(
        'set_blend_failure_fallback',
        null,
        null,
        null,
        $('blend-failure-fallback').value,
        null,
        Number($('blend-failure-grace').value),
      ),
    );
    $('apply-human-timeout-fallback').addEventListener(
      'click',
      () => submitIntent(
        'set_human_timeout_fallback',
        null,
        null,
        null,
        $('human-timeout-fallback').value,
      ),
    );
    $('apply-blend-alpha').addEventListener(
      'click',
      () => submitIntent(
        'set_control_mode',
        'proportional_blend',
        Number($('blend-alpha').value),
      ),
    );
    $('control-profile-selector').addEventListener('change', () => {
      const selector = $('control-profile-selector');
      const selected = String(
        latestStatus?.acceptance?.selected_control_profile_id || ''
      );
      const presentation = CONTROL_PROFILE_PRESENTATION[selector.value] || {};
      $('control-profile-description').textContent = presentation.detail
        || '选择一个完整的 Sim-HIL 控制工作流。';
      $('apply-control-profile').disabled = intentPending
        || Boolean(latestStatus?.acceptance?.control_profile_selection_locked)
        || selector.value === selected;
    });
    $('apply-control-profile').addEventListener(
      'click',
      () => submitIntent(
        'select_control_profile',
        null, null, null, null, null, null,
        $('control-profile-selector').value,
      ),
    );
    $('sim-collision-stop-selector').addEventListener('change', () => {
      const selector = $('sim-collision-stop-selector');
      const selected = Boolean(
        latestStatus?.acceptance?.selected_sim_collision_stop_enabled
      ) ? 'enabled' : 'disabled';
      $('apply-sim-collision-stop').disabled = intentPending
        || Boolean(
          latestStatus?.acceptance?.sim_collision_stop_selection_locked
        )
        || selector.value === selected;
    });
    $('apply-sim-collision-stop').addEventListener(
      'click',
      () => submitIntent(
        'select_sim_collision_stop',
        null, null, null, null, null, null, null,
        $('sim-collision-stop-selector').value === 'enabled',
      ),
    );
    function selectExclusiveSource(targetSource) {
      const runtimeStarted = Boolean(latestStatus?.started);
      if (!runtimeStarted) {
        submitIntent(
          'start_runtime',
          targetSource === 'human' ? 'human_teleop' : 'ppo_only',
        );
        return;
      }
      submitIntent(
        targetSource === 'human'
          ? 'request_human_takeover'
          : 'request_policy_resume',
      );
    }
    $('start-exclusive-policy-runtime').addEventListener(
      'click',
      () => selectExclusiveSource('policy'),
    );
    $('start-exclusive-human-runtime').addEventListener(
      'click',
      () => selectExclusiveSource('human'),
    );
    $('start-allocation-runtime').addEventListener(
      'click',
      () => submitIntent('start_allocation_runtime'),
    );
    for (const [buttonId, allocationMode] of [
      ['allocation-automatic', 'automatic'],
      ['allocation-base-only', 'base_only'],
      ['allocation-relative-overlay', 'relative_overlay'],
      ['allocation-absolute-overlay-owner', 'absolute_overlay_owner'],
    ]) {
      $(buttonId).addEventListener(
        'click',
        () => submitIntent(
          'set_allocation_mode', null, null, null, null, allocationMode
        ),
      );
    }
    $('allocation-rebase-and-clear').addEventListener(
      'click',
      () => submitIntent('request_residual_rebase_and_clear'),
    );
    $('hold-runtime').addEventListener('click', () => submitIntent('hold'));
    $('rearm-runtime').addEventListener('click', () => submitIntent('rearm'));
    $('capture-runtime-home').addEventListener(
      'click',
      () => submitIntent('capture_runtime_home'),
    );
    $('collision-reset-runtime').addEventListener(
      'click',
      () => submitLifecycleIntent('rebuild_runtime'),
    );
    $('rebuild-runtime').addEventListener(
      'click',
      () => submitLifecycleIntent('rebuild_runtime'),
    );
    $('cancel-operation').addEventListener(
      'click',
      () => submitIntent('cancel_operation'),
    );
    $('finalize-capture').addEventListener(
      'click',
      () => submitLifecycleIntent('finalize_capture'),
    );
    $('stop-runtime').addEventListener(
      'click',
      () => submitLifecycleIntent('stop_runtime'),
    );
    function renderViveReceiver(viveReceiver) {
      viveReceiver = viveReceiver || {};
      const receiverState = viveReceiver.status || 'unavailable';
      const receiverLevel = receiverState === 'receiving'
        ? 'ok' : receiverState === 'packet_stale'
        ? 'stale' : receiverState === 'waiting_for_packet'
        ? 'warn' : 'invalid';
      $('vive-receiver-dot').className = `live-dot ${receiverLevel}`;
      const mapping = Object.entries(
        viveReceiver.source_role_by_output_channel || {}
      ).map(([channel, role]) => `${uiToken(role)}→${uiToken(channel)}`).join(' · ') || '—';
      $('vive-receiver-primary').textContent = `${uiValue(receiverState)} · ${mapping}`;
      $('vive-receiver-meta').textContent = [
        `${text(viveReceiver.bind_host)}:${text(viveReceiver.port)}`,
        `数据包 ${text(viveReceiver.packet_count)}`,
        `已解析 ${text(viveReceiver.processed_packet_count)}`,
        `时延 ${text(viveReceiver.packet_age_sec)}s`,
        `映射 ${mapping}`,
        `错误 ${text(viveReceiver.recv_error_count)}`,
        `解析错误 ${text(viveReceiver.process_error_count)}`,
      ].join(' · ');
      const viveTrackers = viveReceiver.trackers || {};
      function vector(values, digits) {
        if (!Array.isArray(values)) return '—';
        return values.map(item => Number.isFinite(Number(item))
          ? Number(item).toFixed(digits) : '—').join(' ');
      }
      function renderViveTracker(side) {
        const tracker = viveTrackers[side] || {};
        const selectedChannels = tracker.selected_for_output_channels || [];
        const requiredSelectedChannels =
          tracker.selected_for_required_output_channels || selectedChannels;
        const selected = selectedChannels.map(uiToken).join(', ') || '未选用';
        const rawLevel = tracker.status_level || 'no-data';
        const inactive = requiredSelectedChannels.length === 0;
        const level = inactive ? 'inactive' : rawLevel;
        $(`vive-${side}-dot`).className = `live-dot ${level}`;
        $(`vive-${side}-title`).textContent = inactive
          ? `${side === 'left' ? '左' : '右'} VIVE 来源 · 本任务未使用`
          : `${side === 'left' ? '左' : '右'} VIVE 来源 · 输出 ${selected}`;
        $(`vive-${side}-primary`).textContent = inactive
          ? '保留用于来源诊断，不参与本次单右臂控制'
          : `位置 ${vector(tracker.xyz_m, 3)}  |  姿态 ${vector(tracker.rpy_deg, 1)}`;
        $(`vive-${side}-meta`).textContent = [
          inactive ? '未选用' : uiValue(tracker.status_label || '无数据'),
          uiToken(tracker.tracker_role || `${side}_wrist`),
          tracker.tracker_name || '—',
          `时延 ${text(tracker.age_sec)}s`,
          `序号 ${text(tracker.seq)}`,
          `→ ${selected}`,
        ].join(' · ');
        const errorNode = $(`vive-${side}-error`);
        errorNode.className = inactive ? 'live-error inactive' : 'live-error';
        errorNode.textContent = !inactive && tracker.sender_error
          ? `错误：${tracker.sender_error}` : '';
      }
      renderViveTracker('left');
      renderViveTracker('right');
    }
    function render(value) {
      latestStatus = value;
      $('error').className = 'error';
      const readiness = value.readiness || {};
      const gate = value.execution_gate || {};
      const startup = value.startup || {};
      const restartRequired = Boolean(startup.restart_required);
      const viveSetupOnly = startup.mujoco_constructed === false;
      const starting = Boolean(
        !value.started
        && value.acceptance
        && value.acceptance.active_intent_id
      );
      const ready = Boolean(value.started && readiness.ready && gate.status === 'ready');
      const badge = $('badge');
      const guidance = value.operator_guidance || {};
      if (guidance.phase) {
        badge.textContent = operatorPhaseNames[guidance.phase]
          || uiValue(guidance.phase);
        badge.className = `badge ${
          guidance.severity === 'error'
            ? 'error' : guidance.severity === 'ready' ? 'ready' : 'hold'
        }`;
      } else {
        badge.textContent = restartRequired
          ? uiValue('RESTART REQUIRED')
          : starting ? uiValue('STARTING')
          : value.started ? (ready ? uiValue('READY') : uiValue('HOLD'))
          : viveSetupOnly ? uiValue('VIVE SETUP') : uiValue('STOPPED');
        badge.className = `badge ${restartRequired ? 'error' : ready ? 'ready' : 'hold'}`;
      }
      $('sampled').textContent = `t=${text(value.sampled_at)}`;
      const binding = value.calibration || {};
      const contracts = value.contracts || {};
      const targetContract = contracts.target || {};
      const visualization = value.visualization || {};
      const controlSchedule = value.control_schedule || {};
      const lastOverrun = controlSchedule.last_overrun || {};
      const cycleTiming = controlSchedule.last_cycle_timing || {};
      rows($('profile-rows'), [
        ['lifecycle', value.lifecycle], ['profile', value.profile_id],
        ['target', value.execution_target], ['real arm motion', value.real_arm_motion],
        ['owner', value.single_execution_owner], ['status schema', value.schema],
        ['MuJoCo viewer', visualization.state],
        ['viewer frames', visualization.sync_count],
        ['viewer error', visualization.last_error],
        ['compose skipped cycles', controlSchedule.overrun_dropped_cycles],
        ['last compose overrun', lastOverrun.late_by_sec],
        ['overrun handling', lastOverrun.resolution],
        ['cycle total', cycleTiming.total_sec],
        ['composition + execution', cycleTiming.composition_and_execution_sec],
        ['policy source', cycleTiming.policy_source_sec],
        ['restart required', startup.restart_required],
        ['first startup failure', startup.first_failure],
        ['first control fault', value.first_control_fault],
        ['Home profile', binding.home_profile_path], ['binding', binding.mode],
        ['VIVE coordinate', binding.vive_coordinate_profile_path],
        ['target handoff', targetContract.handoff_enabled],
        ['target activation', targetContract.activation_mode],
        ['target capability', targetContract.component_target_capability_hash],
      ]);
      renderChecks(readiness);
      renderAcceptance(value);
      const policy = value.policy || {};
      const policyEmission = policy.last_emission || {};
      rows($('policy-rows'), [
        ['state', policy.started ? (policy.ready ? 'ready' : 'fault') : 'stopped'],
        ['refresh', policy.refresh_state],
        ['checkpoint', policy.checkpoint], ['queue', policy.queue_length],
        ['last emission', policyEmission.emission_id],
        ['emission hash', policyEmission.semantic_content_hash],
        ['payload kind', policyEmission.action_payload_kind],
        ['representation trace', policyEmission.representation_trace_id],
        ['canonicalization', policyEmission.canonicalization_receipt_hash],
        ['TTL age', policy.last_emission_age_sec],
        ['inference duration', policy.last_inference_duration_sec],
        ['last error', policy.last_error],
      ]);
      const human = value.human || {};
      const humanEmission = human.last_emission || {};
      const bindingStatus = human.binding || {};
      const safety = human.session_safety || {};
      const sourceConfig = value.source_configuration || {};
      const inputDevices = sourceConfig.input_device_bindings || [];
      const gripperDevice = inputDevices[0] || {};
      const gripper = value.gripper || {};
      const gripperHuman = gripper.human_target || {};
      const gripperExecution = gripper.execution || {};
      const humanChannels = human.channels || {};
      const angleChannel = humanChannels.right_angle_trigger || {};
      const q6Channel = humanChannels.right_hand_q6 || {};
      const gripperInputChannel = angleChannel.value !== undefined
        ? angleChannel : q6Channel;
      const liveGripperInput = gripperInputChannel.value;
      $('gripper-dot').className = `live-dot ${
        gripperInputChannel.valid === true ? 'ok'
          : liveGripperInput === undefined ? 'no-data' : 'invalid'
      }`;
      $('gripper-primary').textContent = gripperDevice.device_kind
        ? `${uiValue(gripperDevice.device_kind)} · 输入 ${text(liveGripperInput)}`
        : '等待配置';
      $('gripper-meta').textContent = [
        `人类目标 ${text(gripperHuman.normalized_close)}`,
        `执行目标 ${text(gripperExecution.accepted_close)}`,
        `仿真测量 ${text(gripperExecution.measured_close)}`,
        gripperInputChannel.read_error
          ? `输入错误 ${text(gripperInputChannel.read_error)}` : null,
        gripperInputChannel.reader_error_count !== undefined
          ? `累计读取错误 ${text(gripperInputChannel.reader_error_count)}` : null,
        gripperDevice.endpoint ? `端点 ${text(gripperDevice.endpoint)}` : null,
        gripperDevice.fallback_reason
          ? `回退 ${uiValue(gripperDevice.fallback_reason)}` : null,
      ].filter(Boolean).join(' · ');
      const viveRoleMapping =
        sourceConfig.vive_source_role_by_output_channel || {};
      rows($('human-rows'), [
        ['session', human.session_state], ['source started', human.source_started],
        ['observed sequence', human.last_observed_emission_sequence],
        ['source rejection', human.last_rejection_reason],
        ['last step valid', (human.last_step || {}).emission_valid],
        ['last step rejection', (human.last_step || {}).rejection_reason],
        ['last emission', humanEmission.emission_id],
        ['emission hash', humanEmission.semantic_content_hash],
        ['age', human.last_emission_age_sec], ['rig absent', bindingStatus.rig_absent],
        ['authoritative', bindingStatus.authoritative],
        ['hold', safety.motion_hold_mode || 'none'],
        ['home re-calibration', safety.home_recalibration_required],
        ['VIVE port', sourceConfig.vive_port],
        ['VIVE source mapping', Object.entries(viveRoleMapping).map(
          ([channel, role]) => `${uiToken(role)} → ${uiToken(channel)}`
        ).join(', ') || '—'],
        ['VIVE swaps', sourceConfig.vive_source_swap_count],
        ['input devices', inputDevices.map(
          item => `${item.component_id}:${uiValue(item.device_kind)}@${item.endpoint || '—'}`
        ).join(', ') || '—'],
        ['channels', Object.keys(human.channels || {}).map(uiToken).join(', ')],
      ]);
      renderViveReceiver(human.vive_receiver || {});
      const composition = value.composition || {};
      const activation = composition.activation || {};
      const last = composition.last_cycle || {};
      const step = last.composition_step || {};
      const disposition = step.disposition || {};
      const proposal = step.proposal || {};
      const humanSubmission = last.human_submission || {};
      const policySubmission = last.policy_submission || {};
      rows($('composition-rows'), [
        ['started', composition.started], ['activation', activation.activation_id],
        ['activation hash', activation.semantic_content_hash],
        ['cycle', last.cycle_index], ['cycle hash', last.semantic_content_hash],
        ['support until', (step.composition_evaluation || {}).composition_supported_until],
        ['disposition', disposition.downstream_request],
        ['proposal', proposal.proposal_id], ['proposal until', proposal.effective_until],
        ['human input', humanSubmission.accepted],
        ['human input error', humanSubmission.error],
        ['policy input', policySubmission.accepted],
        ['policy input error', policySubmission.error],
        ['composition error', step.error],
        ['blocked', step.blocked_reason],
        ['hold reason', disposition.reason || gate.reason],
      ]);
      const execution = last.execution || {};
      const resolutionFailure = execution.resolution_failure || {};
      const executionDifference = value.execution_difference || {};
      const latestExecutionDifference = executionDifference.latest || {};
      rows($('execution-rows'), [
        ['gate', gate.status], ['gate reason', gate.reason],
        ['gate epoch', gate.gate_epoch], ['cycle execution', execution.status],
        ['command', execution.command_id], ['dispatch receipt', execution.dispatch_receipt_id],
        ['target request', execution.target_execution_request_id],
        ['target submission', execution.target_submission_receipt_id],
        ['target completion', execution.target_completion_receipt_id],
        ['target outcome', execution.target_completion_outcome_code],
        ['resolution reason', resolutionFailure.reason_code],
        ['resolver status', resolutionFailure.resolver_status],
        ['resolution message', resolutionFailure.message],
        ['resolution diagnostics', JSON.stringify(
          resolutionFailure.diagnostics || {}
        )],
        ['component target', execution.component_execution_target_id],
        ['target admission', execution.component_target_admission_receipt_id],
        ['target activation', execution.component_target_activation_id],
        ['target support until', execution.target_supported_until],
        ['execution difference', latestExecutionDifference.status],
        ['difference reason', latestExecutionDifference.reason],
        ['difference measurement', latestExecutionDifference.measurement_id],
        ['difference reference', latestExecutionDifference.reference_frame_id],
        ['difference residual', JSON.stringify(
          latestExecutionDifference.component_difference || {}
        )],
        ['last error', value.last_error],
      ]);
      $('composition-label').textContent = last.cycle_index === undefined
        ? '等待' : `周期 ${last.cycle_index}`;
      $('gate-label').textContent = uiValue(gate.status || 'not started');
      $('raw-status').textContent = JSON.stringify(value, null, 2);
    }
    function renderError(error) {
      $('badge').textContent = uiValue('STATUS ERROR');
      $('badge').className = 'badge error';
      $('error').textContent = String(error);
      $('error').className = 'error visible';
    }
    let fullRefreshPending = false;
    let viveRefreshPending = false;
    async function refresh() {
      if (fullRefreshPending) return latestStatus;
      fullRefreshPending = true;
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        const payload = await response.text();
        let value;
        try { value = JSON.parse(payload); } catch (error) { throw new Error(`invalid status JSON: ${error}`); }
        if (!response.ok) throw new Error(value.status_error || `status HTTP ${response.status}`);
        if (value.status_error) throw new Error(value.status_error);
        render(value);
        return value;
      } catch (error) {
        renderError(error);
        return null;
      } finally {
        fullRefreshPending = false;
      }
    }
    async function refreshVive() {
      if (viveRefreshPending) return;
      viveRefreshPending = true;
      try {
        const response = await fetch('/api/vive-status', {cache: 'no-store'});
        const payload = await response.text();
        let value;
        try { value = JSON.parse(payload); } catch (error) { throw new Error(`invalid VIVE status JSON: ${error}`); }
        if (!response.ok) throw new Error(value.status_error || `VIVE status HTTP ${response.status}`);
        if (value.status_error) throw new Error(value.status_error);
        renderViveReceiver(value.vive_receiver || {});
      } catch (_error) {
        // The full status page remains the authoritative error surface. A
        // transient low-cost diagnostics failure must not repaint all cards.
      } finally {
        viveRefreshPending = false;
      }
    }
    refresh();
    refreshVive();
    setInterval(refresh, 1500);
    setInterval(refreshVive, 250);
  </script>
</body>
</html>
"""

class SimHILStatusServer:
    """Serve a read-only status page around an application status callable."""

    def __init__(
        self,
        status_provider: StatusProvider,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        live_status_provider: LiveStatusProvider | None = None,
        intent_submitter: IntentSubmitter | None = None,
        control_token: str | None = None,
        open_browser: bool = False,
    ) -> None:
        if not callable(status_provider):
            raise TypeError("status_provider must be callable.")
        self.status_provider = status_provider
        if live_status_provider is not None and not callable(live_status_provider):
            raise TypeError("live_status_provider must be callable or None.")
        self.live_status_provider = live_status_provider
        self.host = str(host).strip().lower()
        self.port = int(port)
        if intent_submitter is not None and not callable(intent_submitter):
            raise TypeError("intent_submitter must be callable or None.")
        token = None if control_token is None else str(control_token)
        if intent_submitter is None:
            if token is not None:
                raise ValueError(
                    "control_token requires an intent_submitter."
                )
        elif token is None or len(token) < 32 or any(
            character.isspace() for character in token
        ):
            raise ValueError(
                "Controlled Sim-HIL GUI requires a 32+ character token "
                "without whitespace."
            )
        self.intent_submitter = intent_submitter
        self.control_token = token
        if not isinstance(open_browser, bool):
            raise TypeError("open_browser must be a bool.")
        self.open_browser = open_browser
        if self.host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "Sim-HIL status GUI must use a loopback host: 127.0.0.1 or localhost."
            )
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535].")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._browser_thread: threading.Thread | None = None

    @property
    def started(self) -> bool:
        return self._server is not None

    @property
    def bound_address(self) -> tuple[str, int] | None:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            address = self.bound_address
            assert address is not None
            return address

        provider = self.status_provider
        live_status_provider = self.live_status_provider
        intent_submitter = self.intent_submitter
        control_token = self.control_token
        html = _HTML.replace(
            "__SIM_HIL_CONTROL_TOKEN_JSON__",
            json.dumps("" if control_token is None else control_token),
        )

        class Handler(BaseHTTPRequestHandler):
            server_version = "DexTeleopSimHILStatus/1"

            def _write(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src 'self'; "
                    "base-uri 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
                path = urlsplit(self.path).path
                if path == "/":
                    self._write(
                        200,
                        "text/html; charset=utf-8",
                        html.encode("utf-8"),
                    )
                    return
                if path == "/api/status":
                    try:
                        value = provider()
                        if not isinstance(value, dict):
                            raise TypeError("status provider must return a dict.")
                        payload = json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._write(200, "application/json; charset=utf-8", payload)
                    except Exception as error:  # noqa: BLE001 - status must stay observable.
                        payload = json.dumps(
                            {
                                "schema": SIM_HIL_STATUS_SCHEMA_V1,
                                "status_error": f"{type(error).__name__}:{error}",
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._write(500, "application/json; charset=utf-8", payload)
                    return
                if path == "/api/vive-status":
                    if live_status_provider is None:
                        self._write(
                            404,
                            "application/json; charset=utf-8",
                            b'{"status_error":"live status unavailable"}',
                        )
                        return
                    try:
                        value = live_status_provider()
                        if not isinstance(value, dict):
                            raise TypeError(
                                "live status provider must return a dict."
                            )
                        payload = json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._write(
                            200,
                            "application/json; charset=utf-8",
                            payload,
                        )
                    except Exception as error:  # noqa: BLE001 - keep diagnostics visible.
                        payload = json.dumps(
                            {
                                "schema": (
                                    "dexteleop.control_station.sim_hil_vive_status.v1"
                                ),
                                "status_error": f"{type(error).__name__}:{error}",
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._write(
                            500,
                            "application/json; charset=utf-8",
                            payload,
                        )
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found\n")

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                path = urlsplit(self.path).path
                if intent_submitter is None:
                    self.send_response(405)
                    self.send_header("Allow", "GET")
                    payload = b"read-only status server\n"
                    self.send_header(
                        "Content-Type",
                        "text/plain; charset=utf-8",
                    )
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("X-Frame-Options", "DENY")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if path != "/api/acceptance-intents":
                    self._write(
                        404,
                        "text/plain; charset=utf-8",
                        b"not found\n",
                    )
                    return
                supplied = self.headers.get(
                    "X-Sim-HIL-Control-Token",
                    "",
                )
                if not hmac.compare_digest(
                    supplied,
                    str(control_token),
                ):
                    self._write(
                        403,
                        "application/json; charset=utf-8",
                        b'{"accepted":false,"error":"forbidden"}',
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16384:
                        raise ValueError(
                            "intent body must contain 1..16384 bytes"
                        )
                    content_type = self.headers.get(
                        "Content-Type",
                        "",
                    ).partition(";")[0].strip().lower()
                    if content_type != "application/json":
                        raise ValueError(
                            "Content-Type must be application/json"
                        )
                    payload = json.loads(
                        self.rfile.read(length).decode("utf-8")
                    )
                    result = intent_submitter(payload)
                    accepted = bool(result.get("accepted", False))
                    encoded = json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ).encode("utf-8")
                    self._write(
                        202 if accepted else 409,
                        "application/json; charset=utf-8",
                        encoded,
                    )
                except Exception as error:  # noqa: BLE001 - typed boundary.
                    encoded = json.dumps(
                        {
                            "accepted": False,
                            "error": "invalid_acceptance_intent",
                            "message": (
                                f"{type(error).__name__}:{error}"
                            ),
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                    self._write(
                        400,
                        "application/json; charset=utf-8",
                        encoded,
                    )

            def log_message(self, _format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="control_station_sim_hil_status",
            daemon=True,
        )
        self._thread.start()
        address = self.bound_address
        assert address is not None
        if self.open_browser:
            # Browser startup is best-effort and must not delay the Station
            # control loop or make headless environments fail.
            self._browser_thread = threading.Thread(
                target=self._open_browser,
                args=(f"http://{address[0]}:{address[1]}/",),
                name="sim-hil-open-browser",
                daemon=True,
            )
            self._browser_thread.start()
        return address

    @staticmethod
    def _open_browser(url: str) -> None:
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            pass

    def close(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        self._browser_thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=1.0)


__all__ = ["SimHILStatusServer"]
