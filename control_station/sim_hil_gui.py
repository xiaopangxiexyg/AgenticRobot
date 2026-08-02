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


StatusProvider = Callable[[], dict[str, Any]]
LiveStatusProvider = Callable[[], dict[str, Any]]
IntentSubmitter = Callable[[Mapping[str, Any]], dict[str, object]]
SIM_HIL_STATUS_SCHEMA_V1 = "dexteleop.control_station.sim_hil_status.v1"
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
    .mode-card { padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(10,15,28,.45); }
    .mode-card h3 { margin: 0 0 5px; font-size: 13px; }
    .mode-state { margin-bottom: 9px; color: var(--muted); font-size: 12px; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 12px; }
    button, input { border: 1px solid var(--line); border-radius: 9px; color: var(--text); background: #10192a; font: inherit; }
    button { padding: 8px 11px; cursor: pointer; }
    button:hover:not(:disabled) { border-color: var(--blue); }
    button:disabled { cursor: not-allowed; opacity: .4; }
    button.danger { border-color: rgba(255,126,138,.45); color: #ffd6da; }
    input { width: 76px; padding: 7px 8px; }
    input.profile-name { width: min(260px, 100%); }
    .intent-message { min-height: 20px; margin-top: 9px; color: var(--muted); font-size: 12px; }
    .subtle { color: var(--muted); }
    details { margin-top: 14px; }
    summary { cursor: pointer; color: var(--blue); }
    pre { max-height: 420px; margin: 12px 0 0; padding: 14px; overflow: auto; border: 1px solid var(--line); border-radius: 12px; color: #c9d7ea; background: #080c15; white-space: pre-wrap; }
    @media (max-width: 1100px) { .live-strip { grid-template-columns: repeat(2,minmax(0,1fr)); } }
    @media (max-width: 980px) { .card, .card.wide { grid-column: span 6; } .acceptance-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 680px) { main { padding: 20px 14px 32px; } header { display: block; } .badge { margin-top: 15px; } .live-strip { grid-template-columns: 1fr; } .card, .card.wide { grid-column: 1 / -1; } .checks, .source-grid, .acceptance-grid, .calibration-guide, .setup-facts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <!-- Compatibility marker for clients that still search the old title. -->
      <!-- Sim-HIL 控制与 VIVE 标定 -->
      <div>
        <div class="kicker">控制台 / 仿真在环</div>
        <h1>仿真在环控制与 VIVE 标定</h1>
        <p>先在本页确认或标定 VIVE；只有实时输入与已保存坐标配置均就绪后才构造 MuJoCo。PPO 与人类输入仅作为动作来源。</p>
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
      <div class="card-head"><h2>操作区 · Sim-HIL 正式控制</h2><span id="acceptance-label" class="label">—</span></div>
      <p>启动一次运行会话后，可随时切换纯遥操、纯 PPO 和比例融合。VIVE 配置阶段独占接收器；运行时构造后由 Sim-HIL 持有唯一接收器。</p>
      <div id="vive-setup-panel" class="setup-panel">
        <div class="setup-head">
          <div>
            <h2 id="vive-setup-title">VIVE 输入与坐标配置</h2>
            <p id="vive-setup-message">正在检查输入。</p>
          </div>
          <span id="vive-setup-state" class="setup-state">等待</span>
        </div>
        <div id="vive-calibration-guide" class="calibration-guide">
          <div>
            <strong id="vive-calibration-stage">未开始标定</strong>
            <div id="vive-calibration-direction" class="direction">先确认来源</div>
            <div id="vive-calibration-progress" class="subtle">阶段：未开始</div>
          </div>
          <ol id="vive-calibration-steps"></ol>
        </div>
        <div class="setup-facts">
          <div class="setup-fact"><strong>实时输入</strong><span id="vive-input-ready">—</span></div>
          <div class="setup-fact"><strong>坐标配置</strong><span id="vive-profile-ready">—</span></div>
          <div class="setup-fact"><strong>MuJoCo</strong><span id="mujoco-construction-state">未构造</span></div>
          <div class="setup-fact"><strong>当前来源映射</strong><span id="vive-source-mapping">—</span></div>
        </div>
        <div class="controls">
          <button id="begin-vive-calibration">需要标定</button>
          <button id="capture-vive-calibration">采集当前步骤</button>
          <button id="reset-vive-calibration">重做当前步骤</button>
          <button id="cancel-vive-calibration">取消本轮标定</button>
          <button id="swap-vive-sources">交换左右 VIVE 来源</button>
          <span id="vive-swap-status" class="subtle">只交换左右来源映射，不检查来源状态</span>
          <label>保存名称 <input id="vive-profile-name" class="profile-name" placeholder="留空自动命名"></label>
          <button id="save-vive-calibration">保存并选用</button>
        </div>
      </div>
      <div id="control-modes" class="acceptance-grid"></div>
      <div class="controls">
        <label>人类输入混合 α <input id="blend-alpha" type="number" min="0.01" max="0.99" step="0.05" value="0.50"></label>
        <span class="subtle">α 仅用于比例融合；模式可按任意顺序反复切换。</span>
      </div>
      <div class="controls">
        <label>融合目标不可执行时
          <select id="blend-failure-fallback">
            <option value="hold">静止</option>
            <option value="human">切换到人类</option>
            <option value="policy">切换到策略</option>
          </select>
        </label>
        <button id="apply-blend-failure-fallback">应用回退源</button>
        <span id="blend-failure-fallback-status" class="subtle">默认静止；只在比例融合目标无法解析时触发。</span>
      </div>
      <div class="controls">
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
        <div class="controls">
          <button id="hold-runtime">保持输出</button>
          <button id="rearm-runtime">恢复运行</button>
          <button id="capture-runtime-home">采集当前 Runtime Home</button>
          <button id="rebuild-runtime">结束当前仿真并返回准备页</button>
          <button id="cancel-operation" class="danger">中断启动 / 等待</button>
          <button id="finalize-capture">结束回合并封存</button>
          <button id="stop-runtime" class="danger">退出控制</button>
        </div>
      </div>
      <div id="intent-message" class="intent-message"></div>
    </section>
    <div class="section-heading"><h2>状态与执行信息</h2><span class="label">只读 · 回执 / 来源链 / 诊断</span></div>
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
  </main>
  <script>
    const controlToken = __SIM_HIL_CONTROL_TOKEN_JSON__;
    const $ = (id) => document.getElementById(id);
    let latestStatus = null;
    let intentPending = false;
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
      const setup = acceptance.setup || {};
      const calibration = setup.calibration || {};
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
      const calibrationAvailable = Boolean(setup.calibration);
      const deferredHumanSetup = Boolean(
        runtimeConstructed
        && calibrationAvailable
        && setup.phase === 'policy_runtime_human_setup_available'
      );
      const workflowActive = Boolean(calibration.workflow_active);
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
      setupState.textContent = deferredHumanSetup
        ? (humanRuntimeLaunchReady ? 'Human 可以接入' : 'PPO 运行中')
        : runtimeConstructed
        ? '运行中'
        : humanRuntimeLaunchReady ? 'Human 可以启动' : '需要设置';
      setupState.className = `setup-state ${
        (deferredHumanSetup ? humanRuntimeLaunchReady : runtimeConstructed)
          ? 'ready' : ''
      }`;
      $('vive-setup-title').textContent = deferredHumanSetup
        ? 'PPO 已运行 · Human 输入按需接入'
        : runtimeConstructed
        ? 'VIVE 配置已交接给 Sim-HIL'
        : 'VIVE 输入与坐标配置';
      $('vive-setup-message').textContent = deferredHumanSetup
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
      $('mujoco-construction-state').textContent = deferredHumanSetup
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
      for (const modeId of Object.keys(controlModeNames)) {
        const modeLaunchReady = modeId === 'ppo_only'
          ? policyRuntimeLaunchReady
          : humanRuntimeLaunchReady;
        const modeCard = document.createElement('div');
        modeCard.className = 'mode-card';
        const title = document.createElement('h3');
        title.textContent = controlModeNames[modeId];
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
        button.addEventListener('click', () => {
          const alpha = modeId === 'proportional_blend'
            ? Number($('blend-alpha').value) : null;
          submitIntent(
            runtimeStarted ? 'set_control_mode' : 'start_runtime',
            modeId,
            alpha,
          );
        });
        modeCard.append(title, state, button);
        target.append(modeCard);
      }
      const swapButton = $('swap-vive-sources');
      swapButton.disabled = acceptanceBusy
        || workflowActive
        || (runtimeConstructed && !deferredHumanSetup);
      swapButton.title = '只交换左右来源映射，不检查来源状态。';
      $('vive-swap-status').textContent = runtimeConstructed
        && !deferredHumanSetup
        ? '运行时已冻结来源映射'
        : lastResult.intent_kind === 'swap_vive_sources'
          && lastResult.status === 'swapped'
        ? '已交换左右来源；当前映射已更新'
        : '只交换左右来源映射，不检查来源状态';
      const fallbackSelector = $('blend-failure-fallback');
      if (document.activeElement !== fallbackSelector) {
        fallbackSelector.value = blendFallback.selected_source || 'hold';
      }
      fallbackSelector.disabled = acceptanceBusy || !runtimeStarted;
      $('apply-blend-failure-fallback').disabled =
        acceptanceBusy || !runtimeStarted;
      const fallbackActivation = blendFallback.last_activation || {};
      $('blend-failure-fallback-status').textContent =
        blendFallback.latched_source
          ? `已因融合目标不可执行而锁存到${
              blendFallback.latched_source === 'human' ? '人类' : '策略'
            }；重新选择比例融合可解除。`
          : fallbackActivation.status === 'hold_selected'
          ? '最近一次融合目标不可执行，已保持静止。'
          : '只在比例融合目标无法解析时触发。';
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
      $('hold-runtime').disabled = acceptanceBusy || !value.started || value.held;
      $('rearm-runtime').disabled = acceptanceBusy || !value.started || !value.held
        || value.episode_ended;
      $('capture-runtime-home').disabled = acceptanceBusy || !value.started
        || !value.held || value.episode_ended || deferredHumanSetup;
      const runtimeHeld = Boolean(value.held);
      const runtimeAvailable = runtimeConstructed && !value.episode_ended;
      $('runtime-control-state').textContent = !runtimeConstructed
        ? '尚未构造'
        : runtimeHeld ? '已保持，可恢复或重建' : '运行中';
      $('runtime-control-state').className = `setup-state ${runtimeHeld ? '' : 'ready'}`;
      $('runtime-control-help').textContent = runtimeHeld
        ? (deferredHumanSetup
          ? '当前为纯 PPO 保持状态；Human 输入接入后才可采集 Runtime Home。'
          : '当前已保持；VIVE 仍在低频被动刷新。可采集当前 Runtime Home，再恢复运行。')
        : (deferredHumanSetup
          ? '纯 PPO 正在运行；VIVE 与坐标配置不会阻塞策略执行。'
          : 'Runtime Home 已绑定；需要重采时先保持输出。完整重建会结束当前 MuJoCo 仿真。');
      $('rebuild-runtime').disabled = acceptanceBusy || !runtimeAvailable
      $('rebuild-runtime').title =
        '销毁当前 MuJoCo 仿真并回到 VIVE 准备页；未封存记录会明确中止。';
      $('cancel-operation').disabled = !Boolean(
        acceptance.operation_cancel_available || intentPending
      );
      const capture = acceptance.capture || {};
      const captureWriter = capture.writer || {};
      $('finalize-capture').disabled = acceptanceBusy || !value.started
        || value.episode_ended || capture.status !== 'recording';
      $('stop-runtime').disabled = Boolean(acceptance.shutdown_requested);
      $('acceptance-label').textContent = acceptance.last_error
        ? '故障'
        : activeMode
        ? `当前：${controlModeNames[activeMode] || uiValue(activeMode)}`
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
      if (acceptance.last_error) {
        $('intent-message').textContent = uiValue(acceptance.last_error);
      } else if (acceptance.last_result) {
        const result = acceptance.last_result;
        const resultValue = result.message || result.status || result.intent_kind;
        $('intent-message').textContent = resultValue
          ? uiValue(resultValue) : '操作已完成';
      }
    }
    async function submitIntent(
      intentKind,
      controlMode = null,
      humanAlpha = null,
      profileName = null,
      fallbackSource = null,
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
      if (!outOfBand) intentPending = true;
      $('intent-message').textContent = '提交中…';
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
        $('intent-message').textContent = `已排队 ${result.intent_id}`;
        if (!outOfBand) intentPending = false;
        if (intentKind === 'swap_vive_sources') {
          let settled = false;
          for (let attempt = 0; attempt < 10; attempt += 1) {
            const value = await refresh();
            if (value?.acceptance?.last_result?.intent_id === acceptedIntentId) {
              settled = true;
              break;
            }
            await new Promise(resolve => window.setTimeout(resolve, 100));
          }
          await refreshVive();
          if (!settled) {
            $('intent-message').textContent =
              `已提交 ${acceptedIntentId}；等待 control thread 完成`;
          }
        } else {
          await refresh();
        }
      } catch (error) {
        $('intent-message').textContent = String(error);
      } finally {
        if (!outOfBand) intentPending = false;
        if (!acceptedIntentId) await refresh();
      }
    }
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
    $('hold-runtime').addEventListener('click', () => submitIntent('hold'));
    $('rearm-runtime').addEventListener('click', () => submitIntent('rearm'));
    $('capture-runtime-home').addEventListener(
      'click',
      () => submitIntent('capture_runtime_home'),
    );
    $('rebuild-runtime').addEventListener(
      'click',
      () => submitIntent('rebuild_runtime'),
    );
    $('cancel-operation').addEventListener(
      'click',
      () => submitIntent('cancel_operation'),
    );
    $('finalize-capture').addEventListener('click', () => submitIntent('finalize_capture'));
    $('stop-runtime').addEventListener('click', () => submitIntent('stop_runtime'));
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
        const selected = selectedChannels.map(uiToken).join(', ') || '未选用';
        const rawLevel = tracker.status_level || 'no-data';
        const inactive = selectedChannels.length === 0
          && ['invalid', 'no-data'].includes(rawLevel);
        const level = inactive ? 'inactive' : rawLevel;
        $(`vive-${side}-dot`).className = `live-dot ${level}`;
        $(`vive-${side}-title`).textContent =
          `${side === 'left' ? '左' : '右'} VIVE 来源 · 输出 ${selected}`;
        $(`vive-${side}-primary`).textContent =
          `位置 ${vector(tracker.xyz_m, 3)}  |  姿态 ${vector(tracker.rpy_deg, 1)}`;
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
        errorNode.textContent = tracker.sender_error ? `错误：${tracker.sender_error}` : '';
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
      badge.textContent = restartRequired
        ? uiValue('RESTART REQUIRED')
        : value.started ? (ready ? uiValue('READY') : uiValue('HOLD'))
        : viveSetupOnly ? uiValue('VIVE SETUP')
        : starting ? uiValue('STARTING') : uiValue('STOPPED');
      badge.className = `badge ${restartRequired ? 'error' : ready ? 'ready' : 'hold'}`;
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
        gripperDevice.endpoint ? `端点 ${text(gripperDevice.endpoint)}` : null,
        gripperDevice.fallback_reason
          ? `回退 ${uiValue(gripperDevice.fallback_reason)}` : null,
      ].filter(Boolean).join(' · ');
      const viveRoleMapping =
        sourceConfig.vive_source_role_by_output_channel || {};
      rows($('human-rows'), [
        ['session', human.session_state], ['source started', human.source_started],
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
