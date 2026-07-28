"""Small browser GUI for the Control Station VIVE calibration controller."""

from __future__ import annotations

import json
import math
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .vive_calibration import ViveCoordinateCalibrationController


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Control Station · VIVE 坐标标定</title>
<style id="vive-calibration-style" nonce="__CSP_NONCE__">
:root { color-scheme: dark; --bg:#0b1117; --panel:#121d28; --line:#294052; --line-soft:#1c3543; --text:#eaf1f7; --muted:#9bb0c0; --accent:#57d6bf; --warn:#ffca6a; --bad:#ff8c98; }
* { box-sizing:border-box; }
body { margin:0; min-height:100vh; background:radial-gradient(circle at top,#1a3040,var(--bg) 58%); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
header,main,footer { width:min(1280px,calc(100% - 32px)); margin:0 auto; }
header { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; padding:32px 0 22px; }
h1 { margin:0 0 8px; letter-spacing:-.03em; } h2 { margin:0 0 14px; color:var(--accent); font-size:1rem; letter-spacing:.08em; text-transform:uppercase; }
.eyebrow { margin:0 0 8px; color:var(--accent); font-size:.75rem; letter-spacing:.15em; } .subtitle,footer,.hint { color:var(--muted); }
.badge { border:1px solid var(--line); border-radius:999px; padding:8px 12px; color:var(--accent); font-size:.78rem; font-weight:700; white-space:nowrap; }
main { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; padding-bottom:24px; }
.panel { min-width:0; border:1px solid var(--line); border-radius:14px; background:color-mix(in srgb,var(--panel) 95%,transparent); padding:20px; box-shadow:0 14px 38px rgb(0 0 0 / 18%); }
.wide { grid-column:span 2; } .controls { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
label { display:grid; gap:6px; color:var(--muted); font-size:.82rem; } select,input,button { width:100%; min-height:40px; border:1px solid var(--line); border-radius:8px; background:#091016; color:var(--text); padding:8px 10px; font:inherit; }
button { cursor:pointer; color:var(--accent); font-weight:700; } button:hover { border-color:var(--accent); } button.warn { color:var(--warn); } button:disabled { cursor:not-allowed; opacity:.45; }
.button-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; } .button-row button { width:auto; min-width:130px; }
.stage { margin:6px 0; font-size:1.35rem; font-weight:800; } .message { min-height:44px; color:var(--muted); line-height:1.5; } .error { color:var(--bad); }
.metric-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; } .metric { border:1px solid var(--line); border-radius:9px; padding:12px; background:#0d161f; } .metric .label { color:var(--muted); font-size:.76rem; } .metric .value { margin-top:6px; font-weight:800; overflow-wrap:anywhere; }
pre { margin:0; padding:14px; border-radius:9px; background:#071016; overflow:auto; max-height:480px; color:#d4e7f2; font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }
table { width:100%; border-collapse:collapse; font-size:.83rem; } th,td { border-bottom:1px solid var(--line); padding:9px 6px; text-align:left; vertical-align:top; } th { color:var(--muted); font-weight:500; } .ok { color:var(--accent); } .bad { color:var(--bad); }
.notice { border-left:3px solid var(--warn); color:var(--warn); padding-left:12px; line-height:1.5; } footer { padding:0 0 32px; font-size:.82rem; }
@media (max-width:760px) { header,main,footer { width:min(100% - 20px,1280px); } header { padding-top:22px; } main { grid-template-columns:1fr; } .wide { grid-column:span 1; } .controls,.metric-grid { grid-template-columns:1fr; } .button-row button { flex:1 1 130px; } }

/* The coordinate-calibration page is a small dashboard rather than a log. */
body { background: radial-gradient(900px 520px at 8% -10%, #153848 0%, transparent 68%), radial-gradient(760px 460px at 100% 0%, #12302e 0%, transparent 65%), var(--bg); }
header, main, footer { width:min(1440px,calc(100% - 44px)); }
header { padding:38px 0 26px; }
h1 { font-size:clamp(1.8rem,3.2vw,2.65rem); letter-spacing:-.045em; line-height:1.05; }
h2 { color:var(--text); letter-spacing:-.015em; text-transform:none; }
.subtitle { max-width:700px; line-height:1.55; }
.header-status { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; padding-top:4px; }
.status-pill { display:inline-flex; align-items:center; gap:7px; min-height:30px; border:1px solid var(--line); border-radius:999px; padding:6px 10px; color:var(--muted); font-size:.72rem; font-weight:800; letter-spacing:.05em; white-space:nowrap; }
.status-pill.live { color:var(--accent); border-color:#246d60; background:rgba(26,132,105,.14); }
.status-pill.safe { color:#82b9ff; border-color:#315b86; background:rgba(59,109,166,.13); }
.status-pill.bad { color:var(--bad); border-color:#7d4050; background:rgba(145,47,69,.15); }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#64808d; box-shadow:0 0 0 4px rgba(100,128,141,.12); }
.status-pill.live .status-dot { background:var(--accent); box-shadow:0 0 0 4px rgba(83,223,192,.13); }
.status-pill.bad .status-dot { background:var(--bad); box-shadow:0 0 0 4px rgba(255,132,148,.12); }
main { grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px; }
.panel { grid-column:span 6; border-color:#1c3543; border-radius:16px; background:linear-gradient(145deg,rgba(17,39,53,.97),rgba(11,27,38,.97)); box-shadow:0 24px 64px rgba(0,0,0,.24); }
.panel:nth-of-type(1), .panel:nth-of-type(2) { grid-column:span 6; }
.panel:nth-of-type(3), .panel:nth-of-type(4) { grid-column:span 6; }
.panel.wide { grid-column:span 12; }
.panel > h2 { margin-bottom:6px; font-size:1.05rem; }
.panel > h2::before { display:inline-block; width:7px; height:7px; margin:0 8px 2px 0; border-radius:50%; background:var(--accent); content:""; }
.panel > .notice { margin:13px 0 15px; padding:11px 12px; border:1px solid #755d35; border-left:3px solid var(--warn); border-radius:8px; color:var(--muted); background:rgba(113,83,31,.12); font-size:.77rem; }
.controls { gap:13px; }
.controls.top-gap { margin-top:14px; }
label { gap:7px; font-size:.77rem; font-weight:700; }
label select, label input { min-height:43px; border-radius:9px; background:#09151e; }
button { min-height:42px; border-radius:10px; padding:9px 15px; color:var(--text); background:#102532; transition:border-color .16s ease,background .16s ease,transform .16s ease; }
button:hover:not(:disabled) { border-color:var(--accent); background:#153441; transform:translateY(-1px); }
button.primary { color:#05231c; border-color:var(--accent); background:var(--accent); }
button.primary:hover:not(:disabled) { background:#76efd2; border-color:#76efd2; }
button.warn { color:var(--warn); border-color:#795f32; background:rgba(109,77,24,.22); }
.button-row { margin-top:17px; gap:9px; }
.button-row button { min-width:140px; }
.step-rail { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:10px 0 15px; }
.step-rail span { padding:8px 7px; border:1px solid var(--line-soft); border-radius:8px; color:#64808d; background:rgba(7,19,27,.42); font-size:.7rem; font-weight:800; text-align:center; }
.step-rail span.active { color:var(--text); border-color:#348d79; background:rgba(40,136,111,.22); }
.step-rail span.done { color:var(--accent); border-color:#275c54; }
.stage-card { display:grid; grid-template-columns:54px minmax(0,1fr) auto; align-items:center; gap:13px; min-height:118px; padding:14px; border:1px solid var(--line); border-radius:12px; background:linear-gradient(115deg,rgba(7,21,30,.88),rgba(14,42,49,.52)); }
.stage-icon { display:grid; place-items:center; width:50px; height:50px; border:1px solid #367d6e; border-radius:14px; color:var(--accent); background:rgba(47,143,119,.17); font-size:1.2rem; font-weight:850; }
.stage-meta { display:grid; justify-items:end; gap:8px; min-width:95px; text-align:right; }
.stage-kicker { margin-bottom:4px; color:var(--accent); font-size:.68rem; font-weight:800; letter-spacing:.12em; }
.phase-pill { display:inline-flex; align-items:center; justify-content:center; border:1px solid #755d35; border-radius:999px; padding:6px 9px; color:var(--warn); background:rgba(117,93,53,.16); font-size:.68rem; font-weight:800; white-space:nowrap; }
.phase-pill.capturing,.phase-pill.completed { color:var(--accent); border-color:#2e806d; background:rgba(42,128,105,.16); }
.phase-pill.error { color:var(--bad); border-color:#7d4050; background:rgba(145,47,69,.15); }
.stage-progress { height:5px; margin:15px 0 9px; overflow:hidden; border-radius:99px; background:#07131b; }
.stage-progress.thin { height:4px; margin-top:8px; }
.stage-progress span { display:block; width:0; height:100%; border-radius:inherit; background:linear-gradient(90deg,#22b995,var(--accent)); transition:width .25s ease; }
.capture-row { display:flex; align-items:center; justify-content:space-between; gap:10px; color:var(--muted); font-size:.75rem; }
.capture-row strong { color:var(--text); }
.raw-toggle { margin-top:12px; color:#64808d; cursor:pointer; font-size:.72rem; }
.stage { margin:8px 0 4px; font-size:1.45rem; letter-spacing:-.03em; }
kbd { display:inline-block; min-width:20px; margin-left:3px; padding:2px 5px; border:1px solid #386071; border-bottom-width:2px; border-radius:5px; color:var(--text); background:#0a1821; font:700 .68rem ui-monospace,SFMono-Regular,Consolas,monospace; text-align:center; }
.message { min-height:46px; margin-bottom:15px; padding:11px 12px; border:1px solid var(--line-soft); border-radius:9px; color:var(--muted); background:rgba(7,19,27,.42); }
.metric-grid { grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; }
.metric-grid.live-metrics { margin-top:11px; }
.metric { border-color:var(--line-soft); border-radius:10px; background:rgba(7,19,27,.42); }
.metric .label { color:#64808d; font-size:.67rem; }
.metric .value { margin-top:5px; font-size:.82rem; }
pre { border:1px solid var(--line-soft); border-radius:10px; background:#09151e; white-space:pre-wrap; word-break:break-word; }
table { font-size:.76rem; }
th { color:#64808d; font-size:.68rem; letter-spacing:.07em; text-transform:uppercase; }
th,td { border-color:var(--line-soft); padding:10px 8px; }
.ok { color:var(--accent); }
.bad, .error { color:var(--bad); }
.hint, footer { color:#64808d; }
.live-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.channel-card { min-width:0; padding:14px; border:1px solid var(--line-soft); border-radius:12px; background:rgba(7,19,27,.52); }
.channel-card.live { border-color:#2d725f; background:rgba(30,93,77,.13); }
.channel-card.stale { border-color:#70404d; background:rgba(106,38,54,.12); }
.channel-card.inactive { opacity:.45; }
.channel-head { display:flex; align-items:center; gap:9px; }
.channel-dot { width:8px; height:8px; flex:0 0 auto; border-radius:50%; background:#64808d; }
.channel-card.live .channel-dot { background:var(--accent); box-shadow:0 0 0 4px rgba(83,223,192,.13); }
.channel-card.stale .channel-dot { background:var(--bad); }
.channel-title { min-width:0; flex:1; }
.channel-title strong,.channel-title span { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.channel-title strong { font-size:.83rem; }
.channel-title span { margin-top:3px; color:#64808d; font: .67rem ui-monospace,SFMono-Regular,Consolas,monospace; }
.channel-state { color:var(--muted); font-size:.68rem; font-weight:800; white-space:nowrap; }
.channel-card.live .channel-state { color:var(--accent); }
.channel-values { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; margin-top:15px; }
.channel-value { padding:8px 6px; border-radius:7px; text-align:center; background:rgba(1,9,14,.32); }
.channel-value span { display:block; color:#64808d; font-size:.64rem; }
.channel-value strong { display:block; margin-top:4px; color:var(--text); font: .75rem ui-monospace,SFMono-Regular,Consolas,monospace; }
.channel-footer { display:flex; justify-content:space-between; gap:8px; margin-top:11px; color:#64808d; font-size:.68rem; }
.profile-view { display:grid; grid-template-columns:minmax(250px,.8fr) minmax(0,1.2fr); gap:13px; }
.profile-block { min-width:0; padding:13px; border:1px solid var(--line-soft); border-radius:11px; background:rgba(7,19,27,.42); }
.profile-block h3 { margin:0 0 10px; color:var(--muted); font-size:.77rem; }
.matrix { width:100%; border-collapse:collapse; table-layout:fixed; }
.matrix td { padding:7px 4px; border:1px solid var(--line-soft); color:var(--text); font: .7rem ui-monospace,SFMono-Regular,Consolas,monospace; text-align:right; }
.offset-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
.offset-card { padding:10px; border:1px solid var(--line-soft); border-radius:8px; background:rgba(13,37,48,.42); }
.offset-card strong { display:block; font-size:.78rem; }
.offset-card span { display:block; margin-top:5px; color:var(--muted); font: .71rem ui-monospace,SFMono-Regular,Consolas,monospace; }
.empty-state { color:#64808d; font-size:.78rem; line-height:1.5; }
.profile-toolbar { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:13px; }
.profile-toolbar .path { min-width:0; overflow:hidden; color:#8da6b2; font: .7rem ui-monospace,SFMono-Regular,Consolas,monospace; text-overflow:ellipsis; white-space:nowrap; }
.profile-toolbar select { min-height:38px; max-width:360px; border-radius:8px; }
.raw-toggle { margin-top:11px; color:#64808d; cursor:pointer; font-size:.72rem; }
.toast { position:fixed; right:22px; bottom:22px; z-index:5; max-width:min(440px,calc(100% - 44px)); padding:12px 14px; border:1px solid #347e6d; border-radius:10px; color:var(--text); background:#102b32; box-shadow:0 24px 64px rgba(0,0,0,.24); font-size:.78rem; opacity:0; pointer-events:none; transform:translateY(8px); transition:opacity .18s ease,transform .18s ease; }
.toast.visible { opacity:1; transform:translateY(0); }
.toast.error { border-color:#7d4050; background:#321d27; }
@media (max-width:900px) { .panel,.panel:nth-of-type(1),.panel:nth-of-type(2),.panel:nth-of-type(3),.panel:nth-of-type(4) { grid-column:span 12; } .profile-view { grid-template-columns:1fr; } }
@media (max-width:700px) { header { flex-direction:column; } .header-status { justify-content:flex-start; } header,main,footer { width:min(100% - 24px,1440px); } .live-grid,.metric-grid,.offset-grid { grid-template-columns:1fr; } .profile-toolbar { align-items:stretch; flex-direction:column; } .profile-toolbar select { max-width:none; } .stage-card { grid-template-columns:44px minmax(0,1fr); } .stage-icon { width:42px; height:42px; border-radius:11px; } .stage-meta { grid-column:2; justify-items:start; margin-top:5px; text-align:left; } .step-rail span { padding-inline:3px; font-size:.64rem; } }
</style>
</head>
<body>
<header><div><p class="eyebrow">CONTROL STATION · VIVE COORDINATE PROFILE</p><h1>VIVE 坐标系标定</h1><p class="subtitle">完成虚拟桌面坐标系与 wrist tracker offset 标定。此页面只接收 VIVE pose，不创建机械臂、MuJoCo、Q6 或 dispatch owner。</p></div><div class="header-status"><div id="badge" class="status-pill"><span class="status-dot"></span><span id="connectionText">连接中</span></div><div class="status-pill safe"><span>◈</span><span>NO MOTION</span></div></div></header>
<div id="toast" class="toast" role="status" aria-live="polite"></div>
<main>
<section class="panel"><h2>标定配置</h2><div class="notice">只打开 VIVE 接收器。结果写入独立 Control Station profile 目录，不会读取或覆盖真机遥操 profiles。</div>
<div class="controls top-gap"><label>标定侧别<select id="sides"><option value="left">左手</option><option value="right">右手</option><option value="both">双手</option></select><span class="hint">单侧标定不会要求另一侧的 wrist offset。</span></label><label>坐标系 sweep tracker<select id="deskTracker"><option value="auto">自动：active wrist</option><option value="left">左 wrist</option><option value="right">右 wrist</option><option value="auxiliary">Auxiliary tracker</option></select><span class="hint">建议与本次标定侧别一致。</span></label></div>
<div class="button-row"><button id="begin" class="primary">开始 / 重新标定</button><button id="capture">采集当前阶段 <kbd>N</kbd></button><button id="reset" class="warn">重置当前阶段 <kbd>R</kbd></button></div>
<p class="hint">流程：X 后→前、Y 右→左、Z 低→高，最后保持 wrist tracker 中立姿态。快捷键仅在未聚焦输入框时生效。</p></section>
<section class="panel"><h2>当前阶段</h2><div id="stepRail" class="step-rail"><span data-step="virtual_desk_x">1 · X</span><span data-step="virtual_desk_y">2 · Y</span><span data-step="virtual_desk_z">3 · Z</span><span data-step="wrist_offsets">4 · Offset</span></div><div class="stage-card"><div id="stageIcon" class="stage-icon">1</div><div><div id="stageKicker" class="eyebrow">CURRENT STEP</div><div id="stage" class="stage">—</div><div id="message" class="message">—</div></div><div class="stage-meta"><span id="phase" class="phase-pill">—</span><span id="captureValidity" class="hint">等待 tracker</span></div></div><div class="stage-progress"><span id="stageProgress"></span></div><div class="capture-row"><span id="samples">尚未开始采集</span><span id="captureElapsed">—</span></div><div class="stage-progress thin"><span id="captureProgress"></span></div><div id="error" class="error" hidden></div></section>
<section class="panel"><h2>Live VIVE channels</h2><div id="channels" class="live-grid"><article id="channel-left" class="channel-card"><div class="channel-head"><span class="channel-dot"></span><div class="channel-title"><strong>左 wrist</strong><span>left_wrist_pose</span></div><span class="channel-state">等待</span></div><div class="channel-values"><div class="channel-value"><span>X</span><strong>—</strong></div><div class="channel-value"><span>Y</span><strong>—</strong></div><div class="channel-value"><span>Z</span><strong>—</strong></div></div><div class="channel-footer"><span>tracker: —</span><span>age: —</span></div></article><article id="channel-right" class="channel-card"><div class="channel-head"><span class="channel-dot"></span><div class="channel-title"><strong>右 wrist</strong><span>right_wrist_pose</span></div><span class="channel-state">等待</span></div><div class="channel-values"><div class="channel-value"><span>X</span><strong>—</strong></div><div class="channel-value"><span>Y</span><strong>—</strong></div><div class="channel-value"><span>Z</span><strong>—</strong></div></div><div class="channel-footer"><span>tracker: —</span><span>age: —</span></div></article><article id="channel-auxiliary" class="channel-card"><div class="channel-head"><span class="channel-dot"></span><div class="channel-title"><strong>Auxiliary</strong><span>auxiliary_tracker_pose</span></div><span class="channel-state">等待</span></div><div class="channel-values"><div class="channel-value"><span>X</span><strong>—</strong></div><div class="channel-value"><span>Y</span><strong>—</strong></div><div class="channel-value"><span>Z</span><strong>—</strong></div></div><div class="channel-footer"><span>tracker: —</span><span>age: —</span></div></article></div><div class="metric-grid live-metrics"><div class="metric"><div class="label">Receiver</div><div id="receiverMetric" class="value">—</div></div><div class="metric"><div class="label">Packets received</div><div id="packetMetric" class="value">—</div></div></div><details class="raw-toggle"><summary>查看 source diagnostics</summary><pre id="sourceDiagnostics">—</pre></details></section>
<section class="panel"><h2>标定结果查看</h2><div id="profile" class="profile-view"><div class="profile-block"><h3>World → VIVE retarget frame</h3><div id="worldMatrix" class="empty-state">完成三轴 sweep 后显示 4×4 transform。</div></div><div class="profile-block"><h3>Tracker offsets</h3><div id="offsetList" class="offset-grid"><div class="empty-state">完成 wrist offset capture 后显示。</div></div></div></div><details class="raw-toggle"><summary>查看完整 profile payload</summary><pre id="profileRaw">—</pre></details><div class="controls top-gap"><label>保存名称<input id="profileName" placeholder="例如 sim_right_vive_20260726"></label><label>加载已有 profile<select id="profileSelect"><option value="">选择 profile</option></select></label></div><div class="button-row"><button id="save" class="primary">保存独立 profile</button><button id="load">加载查看</button></div></section>
<section class="panel wide"><h2>独立 profile 目录</h2><p id="profileDir" class="hint">—</p><table><thead><tr><th>文件</th><th>侧别</th><th>保存时间</th><th>状态</th><th>错误</th><th>操作</th></tr></thead><tbody id="profileRows"></tbody></table></section>
</main><footer>Control Station VIVE calibration · coordinate/tracker profile only · real teleop profile path remains untouched.</footer>
<script id="vive-calibration-script" nonce="__CSP_NONCE__">
"use strict";
const byId = (id) => document.getElementById(id);
const stageOrder = ["virtual_desk_x", "virtual_desk_y", "virtual_desk_z", "wrist_offsets"];
const stageAliases = {VirtualDeskX:"virtual_desk_x", VirtualDeskY:"virtual_desk_y", VirtualDeskZ:"virtual_desk_z", WristOffset:"wrist_offsets"};
const phaseLabels = {idle:"未开始", ready:"等待采集", preparing:"准备中", capturing:"采集中", waiting:"等待 fresh sample", completed:"已完成", error:"需要处理"};
const channelSpecs = [["left_wrist_pose", "channel-left"], ["right_wrist_pose", "channel-right"], ["auxiliary_tracker_pose", "channel-auxiliary"]];
let latest = null;
let busy = false;
let toastTimer = null;

function display(value, fallback = "—") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}
function number(value, digits = 3) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
}
function clamp(value, low = 0, high = 1) {
  return Math.max(low, Math.min(high, Number(value) || 0));
}
function showToast(message, isError = false) {
  const toast = byId("toast");
  toast.textContent = display(message);
  toast.className = "toast visible" + (isError ? " error" : "");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { toast.className = "toast"; }, 3200);
}
function renderConnection(state, online = true) {
  const badge = byId("badge");
  if (!online) {
    badge.className = "status-pill bad";
    byId("connectionText").textContent = "GUI OFFLINE";
    return;
  }
  const active = Boolean(state.source_started);
  badge.className = "status-pill " + (active ? "live" : "");
  byId("connectionText").textContent = active ? "VIVE RECEIVER ACTIVE" : "VIEW ONLY";
}
function normalizedStage(state) {
  const raw = String(state.stage || "");
  return stageAliases[raw] || raw;
}
function renderSteps(state) {
  const stage = normalizedStage(state);
  const index = stageOrder.indexOf(stage);
  const completed = stage === "completed" || stage === "view";
  document.querySelectorAll("[data-step]").forEach((item) => {
    const itemIndex = stageOrder.indexOf(item.dataset.step);
    item.classList.toggle("done", completed || (index >= 0 && itemIndex < index));
    item.classList.toggle("active", !completed && item.dataset.step === stage);
  });
  byId("stageIcon").textContent = completed ? "✓" : stage === "error" ? "!" : String(Math.max(1, index + 1));
  byId("stageKicker").textContent = completed ? "PROFILE READY" : index >= 0 ? "STEP " + (index + 1) + " OF 4" : "CALIBRATION";
}
function captureEntries(state) {
  const entries = [];
  if (state.capture_left_samples !== null && state.capture_left_samples !== undefined) entries.push(["左手", state.capture_left_samples]);
  if (state.capture_right_samples !== null && state.capture_right_samples !== undefined) entries.push(["右手", state.capture_right_samples]);
  if (!entries.length) {
    Object.entries(state.capture_samples || {}).forEach(([key, value]) => {
      entries.push([key.replace("_wrist_pose", "").replace("_", " "), value]);
    });
  }
  return entries;
}
function renderCapture(state) {
  const entries = captureEntries(state);
  const minimum = state.capture_min_samples;
  byId("samples").textContent = entries.map((entry) => entry[0] + " " + display(entry[1]) + "/" + display(minimum)).join(" · ") || "尚未开始采集";
  const phase = String(state.phase || "");
  const elapsed = state.capture_elapsed_sec;
  const duration = state.capture_duration_sec;
  const prepRemaining = state.capture_prep_remaining_sec;
  let progress = 0;
  if (prepRemaining !== null && prepRemaining !== undefined) {
    progress = 1 - clamp(Number(prepRemaining) / Math.max(Number(state.capture_prep_total_sec) || 1, 0.001));
    byId("captureElapsed").textContent = "准备剩余 " + number(prepRemaining, 1) + "s";
  } else if (elapsed !== null && elapsed !== undefined && duration) {
    progress = clamp(Number(elapsed) / Number(duration));
    byId("captureElapsed").textContent = number(elapsed, 1) + "s / " + number(duration, 1) + "s";
  } else if (phase === "waiting") {
    progress = 1;
    byId("captureElapsed").textContent = "等待新样本";
  } else {
    byId("captureElapsed").textContent = phase === "completed" ? "已完成" : "—";
    const counts = entries.map((entry) => Number(entry[1]) || 0);
    const minCount = counts.length ? Math.min.apply(null, counts) : 0;
    progress = minimum ? clamp(minCount / Number(minimum)) : 0;
  }
  byId("captureProgress").style.width = (progress * 100) + "%";
  const validity = [];
  if (state.capture_left_valid !== null && state.capture_left_valid !== undefined) validity.push("左 " + (state.capture_left_valid ? "有效" : "无效"));
  if (state.capture_right_valid !== null && state.capture_right_valid !== undefined) validity.push("右 " + (state.capture_right_valid ? "有效" : "无效"));
  if (!validity.length && state.capture_valid !== null && state.capture_valid !== undefined) validity.push(state.capture_valid ? "输入有效" : "等待 tracker");
  byId("captureValidity").textContent = validity.join(" · ") || "等待 tracker";
  const stageIndex = stageOrder.indexOf(normalizedStage(state));
  const stageProgress = state.stage === "completed" || state.stage === "view" ? 1 : stageIndex < 0 ? 0 : (stageIndex + progress) / stageOrder.length;
  byId("stageProgress").style.width = (clamp(stageProgress) * 100) + "%";
}
function renderChannel(name, cardId, state) {
  const card = byId(cardId);
  const info = (state.live_channels || {})[name] || {};
  const age = info.age_sec;
  const live = Boolean(info.present && info.valid && (age === null || age === undefined || Number(age) <= 1.0));
  card.className = "channel-card " + (!info.present ? "inactive" : live ? "live" : "stale");
  card.querySelector(".channel-state").textContent = !info.present ? "未连接" : live ? "在线" : "过期 / 无效";
  const values = Array.isArray(info.xyz_m) ? info.xyz_m : [];
  card.querySelectorAll(".channel-value strong").forEach((item, index) => { item.textContent = number(values[index], 4); });
  const footer = card.querySelectorAll(".channel-footer span");
  footer[0].textContent = "tracker: " + display(info.tracker_name || info.tracker_role || info.frame_name);
  footer[1].textContent = "age: " + (age === null || age === undefined ? "—" : number(age, 3) + "s");
}
function renderLive(state) {
  channelSpecs.forEach((spec) => renderChannel(spec[0], spec[1], state));
  const stats = state.source_stats || {};
  byId("receiverMetric").textContent = state.source_started ? "connected" : "not started";
  byId("packetMetric").textContent = display(stats.packet_count);
  byId("sourceDiagnostics").textContent = JSON.stringify({
    packet_count: stats.packet_count,
    last_packet_received_at: stats.last_packet_received_at,
    dropped_stale_packet_count: stats.dropped_stale_packet_count,
    recv_error_count: stats.recv_error_count,
    channel_update_counts: stats.channel_update_counts,
  }, null, 2);
}
function renderMatrix(root, matrix) {
  root.replaceChildren();
  if (!Array.isArray(matrix) || matrix.length !== 4 || !matrix.every((row) => Array.isArray(row) && row.length === 4)) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "完成三轴 sweep 后显示 4×4 transform。";
    root.append(empty);
    return;
  }
  const table = document.createElement("table");
  table.className = "matrix";
  matrix.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = number(value, 4);
      tr.append(td);
    });
    table.append(tr);
  });
  root.append(table);
}
function renderProfile(state) {
  const profile = state.profile || {};
  renderMatrix(byId("worldMatrix"), profile.world_to_vive_retarget_pose);
  const offsets = profile.tracker_offsets || {};
  const root = byId("offsetList");
  root.replaceChildren();
  const names = Object.keys(offsets).sort();
  if (!names.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "完成 wrist offset capture 后显示。";
    root.append(empty);
  } else {
    names.forEach((name) => {
      const card = document.createElement("div");
      card.className = "offset-card";
      const title = document.createElement("strong");
      title.textContent = name;
      const value = document.createElement("span");
      const matrix = offsets[name];
      const translation = Array.isArray(matrix) && matrix.length >= 3 ? [matrix[0][3], matrix[1][3], matrix[2][3]] : null;
      value.textContent = translation ? "translation [" + translation.map((item) => number(item, 4)).join(", ") + "]" : "4×4 transform";
      card.append(title, value);
      root.append(card);
    });
  }
  byId("profileRaw").textContent = Object.keys(profile).length ? JSON.stringify(profile, null, 2) : "—";
}
function appendCell(row, value, className) {
  const cell = document.createElement("td");
  cell.textContent = display(value, "");
  if (className) cell.className = className;
  row.append(cell);
  return cell;
}
function renderLibrary(state) {
  byId("profileDir").textContent = display(state.profile_dir);
  const select = byId("profileSelect");
  const prior = select.value;
  select.replaceChildren(new Option("选择 profile", ""));
  (state.profiles || []).forEach((item) => {
    const option = new Option(item.name + " · " + ((item.active_sides || []).join(",") || "invalid"), item.name);
    option.disabled = !item.valid;
    select.append(option);
  });
  if ([...select.options].some((option) => option.value === prior)) select.value = prior;
  const rows = byId("profileRows");
  rows.replaceChildren();
  (state.profiles || []).forEach((item) => {
    const row = document.createElement("tr");
    appendCell(row, item.name);
    appendCell(row, (item.active_sides || []).join(",") || "—");
    appendCell(row, item.saved_at || "—");
    appendCell(row, item.valid ? "valid" : "invalid", item.valid ? "ok" : "bad");
    appendCell(row, item.error || "", item.error ? "bad" : "");
    const action = appendCell(row, "");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "查看";
    button.dataset.profile = item.name;
    button.disabled = !item.valid;
    action.append(button);
    rows.append(row);
  });
}
function render(state) {
  latest = state;
  renderConnection(state);
  const selectedSides = state.active_sides || [];
  byId("sides").value = selectedSides.length === 2 ? "both" : selectedSides[0] || "left";
  byId("deskTracker").value = state.desk_tracker || "auto";
  byId("stage").textContent = display(state.stage_label);
  byId("phase").textContent = phaseLabels[state.phase] || display(state.phase);
  byId("phase").className = "phase-pill " + (state.phase === "capturing" || state.phase === "completed" ? state.phase : state.phase === "error" ? "error" : "");
  byId("message").textContent = display(state.message);
  byId("error").textContent = state.error || "";
  byId("error").hidden = !state.error;
  renderSteps(state);
  renderCapture(state);
  renderLive(state);
  renderProfile(state);
  renderLibrary(state);
  const stage = normalizedStage(state);
  const canCapture = ["ready", "waiting", "error"].includes(state.phase) && stageOrder.includes(stage);
  byId("capture").disabled = busy || !canCapture;
  byId("reset").disabled = busy || !stageOrder.includes(stage);
  byId("save").disabled = busy || state.stage !== "completed";
  byId("load").disabled = busy || !byId("profileSelect").value;
  byId("begin").disabled = busy;
}
async function send(event, payload = {}) {
  if (busy) return;
  busy = true;
  if (latest) render(latest);
  try {
    const response = await fetch("/api/event", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(Object.assign({event:event}, payload)),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    showToast(event === "save_profile" ? "独立 profile 已保存。" : "操作已提交。");
    await refresh();
  } catch (error) {
    showToast(error.message || String(error), true);
    throw error;
  } finally {
    busy = false;
    if (latest) render(latest);
  }
}
async function refresh() {
  try {
    const response = await fetch("/api/state", {cache:"no-store"});
    if (!response.ok) throw new Error(response.statusText);
    render(await response.json());
  } catch (error) {
    renderConnection({}, false);
  }
}
function submit(event, payload) {
  send(event, payload).catch(() => {});
}
byId("sides").addEventListener("change", () => submit("set_active_sides", {active_sides:byId("sides").value}));
byId("deskTracker").addEventListener("change", () => submit("set_desk_tracker", {desk_tracker:byId("deskTracker").value}));
byId("begin").addEventListener("click", () => submit("begin_calibration"));
byId("capture").addEventListener("click", () => submit("capture"));
byId("reset").addEventListener("click", () => submit("reset_stage"));
byId("save").addEventListener("click", () => submit("save_profile", {profile_name:byId("profileName").value}));
byId("load").addEventListener("click", () => {
  const name = byId("profileSelect").value;
  if (name) submit("load_profile", {profile_name:name});
});
byId("profileSelect").addEventListener("change", () => {
  byId("load").disabled = !byId("profileSelect").value;
});
byId("profileRows").addEventListener("click", (event) => {
  const target = event.target.closest("button[data-profile]");
  if (target && !target.disabled) submit("load_profile", {profile_name:target.dataset.profile});
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input,select,textarea")) return;
  const key = event.key.toLowerCase();
  if (key === "n" && !byId("capture").disabled) { event.preventDefault(); submit("capture"); }
  if (key === "r" && !byId("reset").disabled) { event.preventDefault(); submit("reset_stage"); }
});
refresh();
window.setInterval(refresh, 800);
</script>
</body></html>"""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class _CalibrationHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, controller):
        self.controller = controller
        super().__init__(address, handler)


class _CalibrationHandler(BaseHTTPRequestHandler):
    server_version = "DexTeleopControlStationViveCalibration/1"

    @property
    def controller(self) -> ViveCoordinateCalibrationController:
        controller = self.server.controller
        if not isinstance(controller, ViveCoordinateCalibrationController):
            raise RuntimeError("Calibration GUI controller is invalid.")
        return controller

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        nonce: str | None = None,
    ) -> None:
        script_source = "'self'" if nonce is None else f"'self' 'nonce-{nonce}'"
        style_source = "'self'" if nonce is None else f"'self' 'nonce-{nonce}'"
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            f"script-src {script_source}; style-src {style_source}; "
            "connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, payload, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            nonce = secrets.token_urlsafe(18)
            page = _HTML.replace("__CSP_NONCE__", nonce).encode("utf-8")
            self._send(
                HTTPStatus.OK,
                page,
                "text/html; charset=utf-8",
                nonce=nonce,
            )
            return
        if path == "/api/state":
            self._send_json(HTTPStatus.OK, self.controller.snapshot())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/event":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 32 * 1024:
                raise ValueError("request body must be between 1 and 32768 bytes")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            event = payload.pop("event", "")
            if not isinstance(event, str) or not event.strip():
                raise ValueError("event must be a non-empty string")
            self.controller.dispatch(event, payload)
            self._send_json(HTTPStatus.OK, {"accepted": True, "state": self.controller.snapshot()})
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"accepted": False, "error": f"{type(exc).__name__}: {exc}"})


class ViveCoordinateCalibrationGui:
    """Local web UI server; all capture semantics remain in the controller."""

    def __init__(
        self,
        controller: ViveCoordinateCalibrationController,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = False,
    ) -> None:
        self.controller = controller
        self.host = str(host).strip().lower()
        if self.host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "VIVE calibration GUI must use a loopback host: "
                "127.0.0.1 or localhost."
            )
        self.port = int(port)
        self.open_browser = bool(open_browser)
        self._server: _CalibrationHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self.host}:{self.port}/"
        return f"http://{self.host}:{self._server.server_port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _CalibrationHttpServer(
            (self.host, self.port),
            _CalibrationHandler,
            controller=self.controller,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="control_station_vive_calibration_gui",
        )
        self._thread.start()
        if self.open_browser:
            webbrowser.open(self.url)

    def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)


__all__ = ["ViveCoordinateCalibrationGui"]
