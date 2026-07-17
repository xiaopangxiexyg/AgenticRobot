from __future__ import annotations

import html
import json
import math
import os
import queue
import re
import threading
from collections.abc import Mapping
from urllib.parse import parse_qs, urlparse
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
:root {
  color-scheme: light;
  --bg: #eef2f6;
  --panel: #ffffff;
  --panel-2: #f7f9fc;
  --text: #1d2735;
  --muted: #637083;
  --accent: #147a74;
  --accent-2: #a85b00;
  --good: #18794e;
  --warn: #a85b00;
  --danger: #bf2e2e;
  --border: #d7dee8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
  min-height: 100vh;
}
[hidden] { display: none !important; }
header {
  padding: 22px 24px 14px;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(8px);
}
h1 {
  margin: 0;
  font-size: 24px;
  letter-spacing: 0.02em;
}
.subtitle {
  margin-top: 6px;
  color: var(--muted);
  font-size: 13px;
}
main {
  display: grid;
  gap: 16px;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
  padding: 16px 24px 24px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 12px 30px rgba(29, 39, 53, 0.10);
  overflow: hidden;
}
.panel header {
  border: 0;
  padding: 16px 18px 0;
}
.panel h2 {
  margin: 0;
  font-size: 15px;
}
.panel-body {
  padding: 16px 18px 18px;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.metric {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}
.metric.full { grid-column: 1 / -1; }
.metric.selected-side {
  background: #e8f5f3;
  border-color: rgba(20, 122, 116, 0.42);
}
.label { color: var(--muted); font-size: 12px; }
.value { margin-top: 6px; font-size: 18px; font-weight: 700; }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 12px;
  border: 1px solid var(--border);
  color: var(--muted);
  margin-right: 8px;
  margin-bottom: 8px;
}
.pill strong { color: var(--text); font-weight: 600; }
.stage {
  font-size: 20px;
  font-weight: 800;
  letter-spacing: 0.02em;
  margin: 8px 0 4px;
}
.message-box {
  background: #0f1724;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: #edf2f7;
  padding: 12px;
  min-height: 180px;
  max-height: 380px;
  overflow: auto;
}
.message {
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.06);
  margin-bottom: 8px;
}
.message .tag { color: var(--accent); font-weight: 700; margin-right: 8px; }
.button-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
button {
  appearance: none;
  border: 1px solid var(--border);
  background: #e8f5f3;
  color: var(--text);
  border-radius: 8px;
  padding: 12px 10px;
  cursor: pointer;
  font: inherit;
  transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
}
button:hover { transform: translateY(-1px); border-color: rgba(20, 122, 116, 0.45); }
button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
  transform: none;
}
button.active {
  background: #cdeee8;
  border-color: rgba(20, 122, 116, 0.70);
  box-shadow: 0 0 0 2px rgba(20, 122, 116, 0.14);
  font-weight: 800;
}
button.secondary {
  background: #f3f6fa;
}
button.warn {
  background: #fff2dc;
}
button.danger {
  background: #ffe9e9;
}
button.recording-active {
  background: #dff5ed;
  border-color: #54b88d;
  box-shadow: 0 0 0 3px rgba(24, 121, 78, 0.16);
  color: #064d33;
  font-weight: 900;
}
button.stop-ready {
  background: #ffe9e9;
  border-color: #e38c8c;
  color: #8e1f1f;
  font-weight: 900;
}
.numeric-control {
  display: grid;
  grid-template-columns: auto minmax(110px, 1fr) auto auto;
  gap: 8px;
  align-items: center;
}
.numeric-control input {
  width: 100%;
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 11px 10px;
  font: inherit;
  background: #fff;
  color: var(--text);
}
.numeric-control button {
  min-width: 48px;
}
.numeric-value {
  color: var(--muted);
  font-size: 12px;
}
.form-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}
.form-grid label {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
}
.form-grid input,
.form-grid select {
  width: 100%;
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
  font: inherit;
  color: var(--text);
  background: #fff;
}
.path-control {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  align-items: center;
}
.path-control button {
  white-space: nowrap;
  padding: 10px 12px;
}
.round-trip-status {
  margin-top: 8px;
  min-height: 18px;
}
.section { margin-top: 14px; }
.hint {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
.shortcut-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 14px;
}
.shortcut-item {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel-2);
  padding: 8px 10px;
  color: var(--muted);
  font-size: 12px;
}
.shortcut-item kbd {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 24px;
  border-radius: 6px;
  border: 1px solid rgba(20, 122, 116, 0.30);
  background: #fff;
  color: var(--accent);
  font: 700 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.shortcut-item span {
  min-width: 0;
  line-height: 1.25;
}
.capture-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.capture-mode {
  border-radius: 8px;
  border: 1px solid var(--border);
  padding: 14px;
  margin-top: 14px;
  background: #edf1f6;
}
.capture-mode.ready {
  background: #f5f8fb;
  border-color: var(--border);
}
.capture-mode.preparing {
  background: #fff5e5;
  border-color: #dfa855;
  color: #643600;
}
.capture-mode.capturing {
  background: #dff5ed;
  border-color: #74cba9;
  color: #064d33;
}
.capture-mode.waiting {
  background: #ffe6e6;
  border-color: #e38c8c;
  color: #832020;
}
.capture-mode.teleop {
  background: #e4f5ef;
  border-color: #69b99f;
  color: #075238;
}
.runtime-home-diagnostic {
  margin-top: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  background: #f5f8fb;
}
.runtime-home-diagnostic.capturing {
  background: #e8f5f3;
  border-color: #74cba9;
  color: #064d33;
}
.runtime-home-diagnostic.error {
  background: #ffe9e9;
  border-color: #e38c8c;
  color: #832020;
}
.runtime-home-diagnostic-title {
  font-size: 15px;
  font-weight: 900;
}
.runtime-home-diagnostic-detail {
  margin-top: 7px;
  font-size: 13px;
  line-height: 1.5;
}
.capture-mode-label {
  font-size: 22px;
  font-weight: 900;
}
.capture-mode-detail {
  margin-top: 6px;
  color: inherit;
  font-size: 13px;
}
.capture-progress {
  height: 10px;
  margin-top: 12px;
  border-radius: 999px;
  background: rgba(29, 39, 53, 0.12);
  overflow: hidden;
}
.capture-progress > div {
  height: 100%;
  width: 0%;
  border-radius: inherit;
  background: var(--accent);
  transition: width 180ms ease;
}
.capture-mode.preparing .capture-progress > div { background: #c97800; }
.capture-mode.capturing .capture-progress > div { background: #0f8a5f; }
.capture-mode.waiting .capture-progress > div { background: #bf2e2e; }
.capture-mode.teleop .capture-progress > div { background: #0f8a5f; }
.script-health {
  margin-top: 10px;
  padding: 10px 12px;
  border: 1px solid #e0b15a;
  border-radius: 6px;
  background: #fff8e8;
  color: #6a4200;
  font-size: 13px;
  line-height: 1.45;
}
.episode-banner {
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #f5f8fb;
  padding: 14px 16px;
  margin-top: 14px;
}
.episode-banner.recording {
  background: #dff5ed;
  border-color: #54b88d;
  color: #064d33;
  box-shadow: 0 0 0 3px rgba(24, 121, 78, 0.10);
}
.episode-banner.preparing {
  background: #fff5e5;
  border-color: #d99525;
  color: #643600;
  box-shadow: 0 0 0 3px rgba(217, 149, 37, 0.16);
}
.episode-banner.paused,
.episode-banner.resume_pending {
  background: #eaf2ff;
  border-color: #6b95d8;
  color: #183f73;
  box-shadow: 0 0 0 3px rgba(55, 111, 189, 0.14);
}
.episode-banner.finalizing {
  background: #fff5e5;
  border-color: #dfa855;
  color: #643600;
}
.episode-banner.saved {
  background: #e8f5f3;
  border-color: rgba(20, 122, 116, 0.42);
  color: #075238;
}
.episode-banner.error {
  background: #ffe9e9;
  border-color: #e38c8c;
  color: #832020;
}
.episode-banner.unavailable {
  background: #eef0f3;
  border-color: #c8ced8;
  color: var(--muted);
}
.episode-banner-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  font-size: 24px;
  font-weight: 950;
  line-height: 1.15;
}
.episode-banner-state {
  white-space: nowrap;
}
.episode-banner-counter {
  font-size: 15px;
  font-weight: 900;
}
.episode-banner-detail {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.45;
}
.episode-banner-meta {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.episode-banner-chip {
  display: inline-flex;
  align-items: center;
  border-radius: 8px;
  border: 1px solid rgba(29, 39, 53, 0.14);
  background: rgba(255, 255, 255, 0.55);
  padding: 6px 9px;
  font-size: 12px;
  font-weight: 800;
}
.capture-status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.phase-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 8px;
  padding: 7px 10px;
  border: 1px solid var(--border);
  font-weight: 800;
  font-size: 13px;
}
.phase-chip.prep {
  background: #fff2dc;
  color: #7a4300;
  border-color: #efc784;
}
.phase-chip.capture {
  background: #e4f5ef;
  color: #0f6844;
  border-color: #9ed9c3;
}
.phase-chip.wait {
  background: #ffe9e9;
  color: #9d2323;
  border-color: #efadad;
}
.phase-chip.idle {
  background: #edf1f6;
  color: var(--muted);
}
.phase-chip.active {
  box-shadow: 0 0 0 2px rgba(20, 122, 116, 0.20);
}
.control-title {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  margin: 12px 0 8px;
  text-transform: uppercase;
}
.send-status {
  min-height: 18px;
  color: var(--muted);
  font-size: 12px;
}
.raw-pose-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.raw-pose-card {
  background: #0f1724;
  color: #edf2f7;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  min-height: 128px;
}
.raw-pose-card.warn {
  background: #38250a;
  border-color: #dfa855;
  color: #fff7e6;
}
.raw-pose-card.stale,
.raw-pose-card.invalid,
.raw-pose-card.no-data {
  background: #3a1515;
  border-color: #e38c8c;
  color: #ffecec;
  box-shadow: 0 0 0 2px rgba(191, 46, 46, 0.18);
}
.raw-pose-title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: #9fc7ff;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 8px;
}
.raw-pose-card.warn .raw-pose-title { color: #ffd28a; }
.raw-pose-card.stale .raw-pose-title,
.raw-pose-card.invalid .raw-pose-title,
.raw-pose-card.no-data .raw-pose-title { color: #ffb4b4; }
.raw-pose-body {
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
}
.ik-timing-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.ik-timing-grid .metric {
  min-height: 76px;
}
.metric.warn-metric {
  background: #fff2dc;
  border-color: #dfa855;
}
.metric.bad-metric {
  background: #ffe9e9;
  border-color: #e38c8c;
}
.ik-status-line {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
.ik-status-line.ik-warning,
.ik-warning-banner {
  background: #ffe9e9;
  border: 1px solid #e38c8c;
  color: #9d2323;
}
.ik-status-line.ik-warning {
  border-radius: 8px;
  padding: 8px 10px;
  font-weight: 800;
}
.ik-warning-banner {
  margin-top: 12px;
  border-radius: 8px;
  padding: 10px 12px;
}
.ik-warning-title {
  font-size: 13px;
  font-weight: 900;
}
.ik-warning-text {
  margin-top: 4px;
  font-size: 12px;
  line-height: 1.45;
}
.gripper-grid,
.device-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.gripper-card,
.device-card {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}
.gripper-card.connected,
.device-card.ok {
  background: #e4f5ef;
  border-color: #8fd1b8;
}
.device-card.waiting,
.device-card.warn {
  background: #fff3d6;
  border-color: #e0b35b;
}
.device-card.off {
  background: #eef0f3;
  border-color: #c8ced8;
}
.gripper-card.error,
.device-card.error {
  background: #ffe9e9;
  border-color: #e38c8c;
}
.gripper-title,
.device-title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
  font-weight: 800;
}
.gripper-body,
.device-body {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
.gripper-status {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
}
.shutdown-overlay {
  position: fixed;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 24px;
  background: rgba(15, 23, 36, 0.68);
}
.shutdown-card {
  width: min(460px, 100%);
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.24);
  padding: 22px;
}
.shutdown-card h2 {
  margin: 0;
  font-size: 20px;
}
.shutdown-card p {
  margin: 10px 0 0;
  color: var(--muted);
  line-height: 1.5;
}
@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  main { padding: 12px; }
  .status-grid, .capture-grid, .button-grid, .raw-pose-grid, .ik-timing-grid, .gripper-grid, .device-grid, .shortcut-strip { grid-template-columns: 1fr; }
  .numeric-control { grid-template-columns: auto minmax(92px, 1fr) auto; }
  .numeric-control button[data-event="movement_multiple_reset"] { grid-column: 1 / -1; }
  .episode-banner-title { flex-direction: column; align-items: flex-start; }
  .episode-banner-state { white-space: normal; }
}
</style>
</head>
<body>
<header>
  <h1>__PAGE_TITLE__</h1>
</header>
<main>
  <section class="panel">
    <header><h2>当前状态</h2></header>
    <div class="panel-body">
      <div id="stage" class="stage">__INITIAL_STAGE__</div>
      <div id="pills"></div>
      <div class="status-grid section">
        <div id="selectedSideMetric" class="metric"><div class="label">当前侧</div><div id="selectedSide" class="value">__INITIAL_SELECTED_SIDE__</div></div>
        <div class="metric"><div class="label">坐标显示</div><div id="frameVizMode" class="value">__INITIAL_FRAME_VIZ_MODE__</div></div>
        <div class="metric"><div class="label">虚拟桌面偏航</div><div id="virtualDeskYaw" class="value">__INITIAL_VIRTUAL_DESK_YAW__</div></div>
        <div class="metric full"><div class="label">当前提示</div><div id="controlHint" class="value mono">__INITIAL_CONTROL_HINT__</div></div>
      </div>
      <div id="captureMode" class="capture-mode __INITIAL_CAPTURE_PHASE_CLASS__">
        <div id="captureModeLabel" class="capture-mode-label">__INITIAL_CAPTURE_LABEL__</div>
        <div id="captureModeDetail" class="capture-mode-detail">__INITIAL_CAPTURE_DETAIL__</div>
        <div class="capture-progress"><div id="captureProgressBar" style="width: __INITIAL_CAPTURE_PROGRESS__%;"></div></div>
      </div>
      <div id="runtimeHomeDiagnostic" class="runtime-home-diagnostic __INITIAL_RUNTIME_HOME_DIAGNOSTIC_LEVEL__" __INITIAL_RUNTIME_HOME_DIAGNOSTIC_HIDDEN__>
        <div id="runtimeHomeDiagnosticTitle" class="runtime-home-diagnostic-title">__INITIAL_RUNTIME_HOME_DIAGNOSTIC_TITLE__</div>
        <div id="runtimeHomeDiagnosticDetail" class="runtime-home-diagnostic-detail mono">__INITIAL_RUNTIME_HOME_DIAGNOSTIC_DETAIL__</div>
      </div>
      <div id="episodeBanner" class="episode-banner unavailable">
        <div class="episode-banner-title">
          <span id="episodeBannerState" class="episode-banner-state">数采未启用</span>
          <span id="episodeBannerCounter" class="episode-banner-counter">-</span>
        </div>
        <div id="episodeBannerDetail" class="episode-banner-detail mono">等待状态更新</div>
        <div id="episodeBannerMeta" class="episode-banner-meta"></div>
      </div>
      <div id="scriptHealth" class="script-health">网页脚本尚未接管实时刷新；如果这里长期存在，请打开终端输出的 state_url/debug_url 检查后端状态，或硬刷新当前页面。</div>
      <div id="ikWarningBanner" class="ik-warning-banner" hidden>
        <div class="ik-warning-title">IK WARNING</div>
        <div id="ikWarningText" class="ik-warning-text mono">-</div>
      </div>
      <div id="poseQualitySection" class="section">
        <div class="label">Pose Quality</div>
        <div id="poseQualityCard" class="device-card off">
          <div class="device-title"><span id="poseQualityStatus">等待</span><span id="poseQualityScore">-</span></div>
          <div id="poseQualityDetail" class="device-body mono">等待遥操命令元数据</div>
        </div>
      </div>
      <div class="capture-grid section">
        <div class="metric"><div class="label">采集阶段</div><div id="captureStage" class="value">__INITIAL_CAPTURE_STAGE__</div></div>
        <div class="metric"><div class="label">样本</div><div id="captureSamples" class="value">__INITIAL_CAPTURE_SAMPLES__</div></div>
        <div class="metric"><div class="label">有效性</div><div id="captureValid" class="value">__INITIAL_CAPTURE_VALID__</div></div>
        <div class="metric full"><div class="label">采集状态</div><div id="captureStatus" class="value mono">__INITIAL_CAPTURE_STATUS__</div></div>
      </div>
      <div id="gripperStatusSection" class="section" hidden>
        <div class="label">Q6 夹爪设备</div>
        <div id="gripperCards" class="gripper-grid"></div>
      </div>
      <div id="cameraStatusSection" class="section">
        <div class="label">RealSense 相机</div>
        <div id="cameraSelfCheckSection" class="section" hidden>
          <div id="cameraSelfCheckCard" class="device-card off">
            <div class="device-title"><span>启动自检</span><span id="cameraSelfCheckStatus">未开始</span></div>
            <div id="cameraSelfCheckDetail" class="device-body mono">-</div>
          </div>
        </div>
        <div id="cameraCards" class="device-grid"></div>
      </div>
      <div id="tactileCameraStatusSection" class="section">
        <div class="label">视触觉相机</div>
        <div id="tactileCameraCards" class="device-grid"></div>
      </div>
      <div id="ikTimingSection" class="section" hidden>
        <div class="label">IK 耗时诊断</div>
        <div class="ik-timing-grid">
          <div id="ikTimingTotalMetric" class="metric"><div class="label">总耗时 / 预算</div><div id="ikTimingTotal" class="value">-</div></div>
          <div class="metric"><div class="label">求解细分</div><div id="ikTimingSolve" class="value">-</div></div>
          <div class="metric"><div class="label">检查细分</div><div id="ikTimingCollision" class="value">-</div></div>
          <div class="metric"><div class="label">调用次数</div><div id="ikTimingCalls" class="value">-</div></div>
        </div>
        <div id="ikTimingStatus" class="ik-status-line mono">-</div>
      </div>
      <div class="section">
        <div class="label">左右原始位姿（VIVE world）</div>
        <div class="raw-pose-grid">
          <div id="leftRawCard" class="raw-pose-card">
            <div class="raw-pose-title"><span>左 wrist</span><span id="leftRawStatus">__INITIAL_LEFT_RAW_STATUS__</span></div>
            <pre id="leftRawPose" class="raw-pose-body mono">__INITIAL_LEFT_RAW_POSE__</pre>
          </div>
          <div id="rightRawCard" class="raw-pose-card">
            <div class="raw-pose-title"><span>右 wrist</span><span id="rightRawStatus">__INITIAL_RIGHT_RAW_STATUS__</span></div>
            <pre id="rightRawPose" class="raw-pose-body mono">__INITIAL_RIGHT_RAW_POSE__</pre>
          </div>
        </div>
      </div>
      <div class="section">
        <div class="label">最新日志</div>
        <div id="messages" class="message-box mono"></div>
      </div>
    </div>
  </section>
  <section class="panel">
    <header><h2>控制</h2></header>
    <div class="panel-body">
      <div class="shortcut-strip" aria-label="快捷键">
        <div class="shortcut-item"><kbd>A</kbd><span>开始 / 停止数采</span></div>
        <div class="shortcut-item"><kbd>B</kbd><span>开始遥操 / Hold</span></div>
        <div class="shortcut-item"><kbd>C</kbd><span id="shortcutCLabel">固定臂不可用</span></div>
        <div class="shortcut-item"><kbd>G</kbd><span id="shortcutGLabel">归位不可用</span></div>
      </div>
      <div id="sendStatus" class="send-status"></div>
      <div class="control-title">标定 / 坐标系</div>
      <div class="button-grid">
        <button id="viveCalibrationCalibrateButton" data-event="calibrate_vive_coordinate_profile">标定 VIVE 坐标系</button>
        <button id="viveCalibrationSaveButton" class="secondary" data-event="save_vive_calibration_profile">保存当前坐标系</button>
        <button id="viveCalibrationLoadButton" class="secondary" data-event="load_vive_calibration_profile">加载坐标系</button>
      </div>
      <div id="viveCalibrationStatus" class="gripper-status mono">-</div>
      <div class="control-title">移动倍率</div>
      <div class="numeric-control">
        <button class="secondary" data-event="movement_multiple_down">-</button>
        <input id="movementMultipleInput" type="number" min="0.05" max="10" step="0.1" value="1.0">
        <button class="secondary" data-event="movement_multiple_up">+</button>
        <button class="secondary" data-event="movement_multiple_reset">重置</button>
      </div>
      <div id="movementMultipleStatus" class="numeric-value mono">movement_multiple=1.000</div>
      <div id="gripperForceSection" hidden>
        <div class="control-title">左右夹爪力度（仅非数采）</div>
        <div class="form-grid">
          <label>左夹爪力度
            <input id="leftGripperForceInput" type="number" min="0" max="1" step="0.01" value="0.20">
          </label>
          <label>右夹爪力度
            <input id="rightGripperForceInput" type="number" min="0" max="1" step="0.01" value="0.20">
          </label>
        </div>
        <button id="gripperForceApplyButton" class="secondary">应用左右力度</button>
        <div id="gripperForceStatus" class="numeric-value mono">仅在非数采阶段可设置；开始数采前会恢复任务配置。</div>
      </div>
      <div class="control-title">流程</div>
      <div class="button-grid">
        <button class="danger" data-event="q">退出 / 停臂</button>
        <button class="warn" data-event="r">重置</button>
        <button data-event="p">开始遥操</button>
        <button data-event="c">固定臂不可用</button>
        <button id="ikDebugLogButton" class="secondary" data-event="d">开始 IK 记录</button>
        <button class="secondary" data-event="n">下一步</button>
        <button class="secondary" data-event="s">保存</button>
        <button class="secondary" data-event="h">Hold / 冻结</button>
      </div>
      <div id="fixArmSection" hidden>
        <div class="control-title">固定臂</div>
        <div class="form-grid">
          <label>固定侧
            <select id="fixArmSideSelect">
              <option value="left">左臂</option>
              <option value="right">右臂</option>
            </select>
          </label>
        </div>
        <div class="button-grid">
          <button id="fixArmToggleButton" data-event="fix_arm_toggle">固定左臂</button>
        </div>
        <div id="fixArmStatus" class="numeric-value mono">-</div>
      </div>
      <div id="initArmSection" hidden>
        <div class="control-title">归位</div>
        <div class="form-grid">
          <label>归位侧
            <select id="initArmSideSelect">
              <option value="left">左臂</option>
              <option value="right">右臂</option>
            </select>
          </label>
        </div>
        <div class="button-grid">
          <button id="initArmToggleButton" data-event="init_arm_toggle">左臂归位</button>
        </div>
        <div id="initArmStatus" class="numeric-value mono">-</div>
      </div>
      <div id="footPadSpecialArmHint" class="gripper-status mono" hidden>
        Foot-pad 模式：踏板 1/2 固定左/右臂；踏板 3/4 归位左/右臂。再次踩同一踏板重采该侧 Home。
      </div>
      <div class="control-title">数采</div>
      <div class="form-grid">
        <label>保存根目录
          <div class="path-control">
            <input id="datasetBaseRootInput" type="text" autocomplete="off">
            <button id="datasetChooseRootButton" class="secondary" type="button">选择文件夹</button>
          </div>
        </label>
        <label>任务类型
          <select id="taskTypeSelect">
            <option value="general">一般类</option>
            <option value="round_trip">来回类</option>
          </select>
        </label>
        <label>Task
          <select id="taskSelect"></select>
        </label>
        <label>英文名
          <input id="taskEnglishNameInput" type="text" autocomplete="off">
        </label>
        <label>中文名
          <input id="taskChineseNameInput" type="text" autocomplete="off">
        </label>
      </div>
      <div class="button-grid">
        <button id="episodeStartButton" data-event="episode_start">开始数采</button>
        <button id="episodeStopButton" class="secondary" data-event="episode_stop">停止数采</button>
        <button id="roundTripToggleButton" class="secondary" data-event="round_trip_toggle">切换 in/out</button>
      </div>
      <div id="roundTripStatus" class="numeric-value mono round-trip-status">-</div>
      <div id="episodeStatus" class="gripper-status mono">-</div>
      <div id="episodeWarningBanner" class="ik-warning-banner" hidden>
        <div class="ik-warning-title">DATA WARNING</div>
        <div id="episodeWarningText" class="ik-warning-text mono">-</div>
      </div>
      <div class="control-title">可视化窗口</div>
      <div class="button-grid">
        <button id="cameraVisButton" class="secondary" data-event="toggle_camera_vis">打开图像窗口</button>
        <button id="tactileVisButton" class="secondary" data-event="toggle_tactile_vis">打开触觉窗口</button>
      </div>
      <div id="visualizationStatus" class="gripper-status mono">-</div>
      <div class="control-title">左右 / 视图</div>
      <div class="button-grid">
        <button data-event="1">左手</button>
        <button data-event="2">右手</button>
        <button data-event="t">切换左右</button>
        <button data-event="w">交换 Tracker</button>
        <button data-event="v">切换坐标</button>
        <button data-event="f">翻转 XY</button>
        <button data-event="x">清零微调</button>
      </div>
      <div class="control-title">姿态微调</div>
      <div class="button-grid">
        <button data-event="j">Yaw -</button>
        <button data-event="l">Yaw +</button>
        <button data-event="i">Pitch +</button>
        <button data-event="k">Pitch -</button>
        <button data-event="u">Roll -</button>
        <button data-event="o">Roll +</button>
      </div>
      <div id="poseSnapshotSection" hidden>
        <div class="control-title">Pose 抽帧</div>
        <div class="button-grid">
          <button id="poseSnapshotButton" class="secondary" data-event="record_pose_snapshot">录制当前 Pose</button>
        </div>
        <div id="poseSnapshotStatus" class="gripper-status mono">-</div>
      </div>
    </div>
  </section>
</main>
<div id="shutdownOverlay" class="shutdown-overlay" hidden>
  <div class="shutdown-card">
    <h2>服务已退出</h2>
    <p id="shutdownMessage">这个网页已经和后端断开连接，可以关闭。</p>
  </div>
</div>
<script>
(function () {
  var legacyTimer = null;
  var legacyLatestState = {};
  var legacyActiveControls = {};

  function byId(id) {
    return document.getElementById(id);
  }

  function markScriptRunning() {
    var el = byId("scriptHealth");
    if (el) el.hidden = true;
  }

  function setText(id, value) {
    var el = byId(id);
    if (el) el.textContent = value == null || value === "" ? "-" : String(value);
  }

  function labelStage(value) {
    var labels = {
      "Starting...": "启动中...",
      "RM65 真机遥操准备": "RM65 真机遥操准备",
      "Home Capture": "起点采集",
      "HomeCapture": "起点采集",
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

  function normalizeTaskTypeLegacy(value) {
    var text = String(value || "").trim().toLowerCase().replace(/[-\\s]+/g, "_");
    return /^(roundtrip|round_trip|back_and_forth|in_out|inout|往返|来回)$/.test(text) ? "round_trip" : "general";
  }

  function normalizeRoundTripPhaseLegacy(value) {
    return String(value || "").trim().toLowerCase() === "out" ? "out" : "in";
  }

  function setCaptureClass(phase) {
    var el = byId("captureMode");
    if (!el) return;
    el.className = "capture-mode " + (phase || "idle");
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

  function renderRuntimeHomeDiagnosticLegacy(state) {
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

  function legacyNumberText(value, digits) {
    var numeric = Number(value);
    if (!isFinite(numeric)) return value == null ? "nan" : String(value);
    return numeric.toFixed(digits);
  }

  function legacyVectorText(values, digits) {
    if (!values || !values.length) return "-";
    var parts = [];
    for (var i = 0; i < values.length; i += 1) {
      parts.push(legacyNumberText(values[i], digits));
    }
    return parts.join(", ");
  }

  function legacyRawPoseText(pose) {
    if (pose && pose.enabled === false) {
      return {status: "未启用", text: "当前 logical side 未启用"};
    }
    if (!pose) {
      return {status: "无数据", text: "未收到 channel"};
    }
    var status = pose.status_label || (pose.valid ? "有效" : "无效");
    var tracker = pose.tracker_name || "-";
    var role = pose.tracker_role || "-";
    var age = pose.age_sec == null ? "-" : legacyNumberText(pose.age_sec, 2) + " s";
    var seq = pose.seq == null ? "-" : String(pose.seq);
    return {
      status: status,
      text: [
        "tracker: " + tracker,
        "role:    " + role,
        "seq:     " + seq,
        "age:     " + age,
        "xyz(m):  " + legacyVectorText(pose.xyz_m, 4),
        "rpy(deg):" + legacyVectorText(pose.rpy_deg, 1)
      ].join("\\n")
    };
  }

  function setLegacyRawPoseClass(id, pose) {
    var el = byId(id);
    if (!el) return;
    var level = pose && pose.status_level ? String(pose.status_level) : "ok";
    el.className = "raw-pose-card " + level;
  }

  function renderRawPosesLegacy(state) {
    var poses = state.raw_tracker_poses || {};
    var left = legacyRawPoseText(poses.left);
    var right = legacyRawPoseText(poses.right);
    setText("leftRawStatus", left.status);
    setText("rightRawStatus", right.status);
    setText("leftRawPose", left.text);
    setText("rightRawPose", right.text);
    setLegacyRawPoseClass("leftRawCard", poses.left);
    setLegacyRawPoseClass("rightRawCard", poses.right);
  }

  function poseQualityLevelLegacy(quality) {
    quality = quality || {};
    var level = String(quality.level || "");
    if (/^(ok|warn|error|off)$/.test(level)) return level;
    var status = String(quality.status || "unknown");
    if (status === "bad" || quality.bad) return "error";
    if (status === "warning") return "warn";
    if (status === "good") return "ok";
    return "off";
  }

  function formatPoseQualityMetricLegacy(value, digits, unit) {
    var numeric = Number(value);
    if (!isFinite(numeric)) return "-";
    return numeric.toFixed(digits) + unit;
  }

  function renderPoseQualityLegacy(state) {
    var quality = state.pose_quality || {};
    var metrics = quality.metrics || {};
    var card = byId("poseQualityCard");
    if (card) card.className = "device-card " + poseQualityLevelLegacy(quality);
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
      " dist=" + formatPoseQualityMetricLegacy(metrics.collision_distance_m, 3, "m") +
      " min=" + formatPoseQualityMetricLegacy(metrics.collision_min_distance_m, 3, "m") +
      " margin=" + formatPoseQualityMetricLegacy(metrics.collision_margin_m, 3, "m")
    );
    lines.push(
      "ik: jump=" + (metrics.ik_jump_gate_status || "-") +
      " branch=" + (metrics.ik_branch_candidate || "-") +
      " err=" + formatPoseQualityMetricLegacy(metrics.ik_translation_error_m, 3, "m") +
      "/" + formatPoseQualityMetricLegacy(metrics.ik_rotation_error_deg, 1, "deg")
    );
    lines.push(
      "joint: nearest_margin=" + formatPoseQualityMetricLegacy(metrics.joint_nearest_margin_deg, 1, "deg") +
      " near_limit=" + (metrics.joint_near_limit ? "yes" : "no")
    );
    setText("poseQualityDetail", lines.join("\\n"));
  }

  function renderIkDebugLogButtonLegacy(state) {
    var button = byId("ikDebugLogButton");
    if (!button) return;
    var logging = Boolean(state.debug_ik_logging);
    button.textContent = logging ? "停止 IK 记录" : "开始 IK 记录";
    if (button.className.indexOf("active") < 0 && logging) {
      button.className += " active";
    } else if (!logging) {
      var classes = button.className.split(" ");
      var kept = [];
      for (var i = 0; i < classes.length; i += 1) {
        if (classes[i] && classes[i] !== "active") kept.push(classes[i]);
      }
      button.className = kept.join(" ");
    }
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
      } else if (duration > 0) {
        progress = elapsed / duration * 100;
      }
      detail = "正在采集" + sampleLabel + "，请保持双腕稳定";
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
    var i;
    for (i = 0; i < controls.length; i += 1) active[String(controls[i])] = true;
    legacyActiveControls = active;
    var shortcutC = byId("shortcutCLabel");
    if (shortcutC) {
      if (state.fix_arm_available && state.teleop_running) {
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
      if (state.init_arm_available && state.teleop_running) {
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
            : initLabel + "归位";
      }
    }
  }

  function renderEpisodeStateLegacy(state) {
    var available = Boolean(state.episode_recording_available);
    var recording = Boolean(state.episode_recording);
    var preparing = String(state.episode_event_status || "") === "preparing";
    var finalizing = String(state.episode_event_status || "") === "finalizing";
    var startButton = byId("episodeStartButton");
    var stopButton = byId("episodeStopButton");
    var statusEl = byId("episodeStatus");
    var warning = state.episode_frequency_warning || {};
    var warningBanner = byId("episodeWarningBanner");
    var warningText = byId("episodeWarningText");
    var status = String(state.episode_event_status || "");
    var paused = status === "paused" || status === "resume_pending";
    var statusLabels = {
      unavailable: "数采不可用",
      idle: "未开始数采",
      preparing: "3 秒后开始数采",
      recording: "正在数采",
      paused: "数采暂停",
      resume_pending: "等待恢复写入",
      finalizing: "停止中 / 后台保存",
      saved: "已保存",
      discarded: "已停止 / 无样本",
      error: "保存失败"
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
      startButton.textContent = paused ? "数采暂停中" : (recording ? "数采进行中" : (preparing ? "准备中..." : "开始数采"));
      startButton.disabled = !legacyActiveControls.episode_start || recording || preparing || finalizing;
    }
    if (stopButton) {
      stopButton.disabled = !legacyActiveControls.episode_stop || (!recording && !preparing);
    }
    var locked = recording || preparing || finalizing;
    var fields = ["datasetBaseRootInput", "datasetChooseRootButton", "taskTypeSelect", "taskSelect", "taskEnglishNameInput", "taskChineseNameInput"];
    for (var i = 0; i < fields.length; i += 1) {
      var field = byId(fields[i]);
      if (field) field.disabled = locked;
    }
    var roundTripButton = byId("roundTripToggleButton");
    if (roundTripButton) {
      roundTripButton.disabled = locked || normalizeTaskTypeLegacy(state.collection_task_type) !== "round_trip" || !legacyActiveControls.round_trip_toggle;
    }
    var roundTripStatus = byId("roundTripStatus");
    if (roundTripStatus) {
      roundTripStatus.textContent = normalizeTaskTypeLegacy(state.collection_task_type) === "round_trip"
        ? "来回类: 当前将采 _" + normalizeRoundTripPhaseLegacy(state.collection_round_trip_phase)
        : "一般类任务：每次采集使用同一个 task 标签。";
    }
    if (statusEl) {
      var lines = [
        available ? "episode 录制可用" : "episode 录制不可用",
        paused ? "状态: 暂停中" : (recording ? "状态: 录制中" : (preparing ? "状态: 倒计时准备中" : "状态: 未录制"))
      ];
      if (state.episode_dir) lines.push("路径: " + state.episode_dir);
      if (state.dataset_root) lines.push("保存根目录: " + state.dataset_root);
      if (state.episode_frame_count != null) lines.push("帧数: " + state.episode_frame_count);
      statusEl.textContent = lines.join("\\n");
    }
    if (warningBanner) warningBanner.hidden = !Boolean(warning.active);
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
      recording
        ? "正在写入正式数据集；结束前请确认画面、触觉和机械臂状态正常。"
        : available
          ? "点击“开始数采”或按 A 开始；再次按 A 停止。"
          : "当前启动参数禁用了正式数据集和 debug episode 录制。"
    );
    var path = state.episode_dir || state.dataset_root || "";
    setText("episodeBannerDetail", detail + (path ? "\\n路径: " + path : ""));
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
      var tag = String(messages[i].tag || "LOG").replace(/&/g, "&amp;").replace(/</g, "&lt;");
      var text = String(messages[i].message || "").replace(/&/g, "&amp;").replace(/</g, "&lt;");
      html.push('<div class="message"><span class="tag">[' + tag + ']</span>' + text + '</div>');
    }
    el.innerHTML = html.join("");
  }

  function renderState(state) {
    if (!state || window.__calibrationWebUiModernRefreshOk) return;
    legacyLatestState = state || {};
    setText("stage", labelStage(state.stage || "Starting..."));
    setText("selectedSide", state.selected_side === "left" ? "左手" : state.selected_side === "right" ? "右手" : state.selected_side);
    setText("frameVizMode", state.frame_viz_mode);
    setText("virtualDeskYaw", state.virtual_desk_post_yaw_deg != null ? String(state.virtual_desk_post_yaw_deg) + " deg" : "-");
    setText("controlHint", state.control_hint || "-");
    setText("captureStage", labelStage(state.capture_stage));
    setText("captureSamples", captureSamplesText(state));
    setText("captureValid", captureValidText(state));
    setText("captureStatus", state.teleop_running ? "遥操中" : labelPhase(state.capture_phase || "idle"));
    renderCaptureMode(state);
    renderRuntimeHomeDiagnosticLegacy(state);
    renderButtons(state);
    renderRawPosesLegacy(state);
    renderPoseQualityLegacy(state);
    renderEpisodeStateLegacy(state);
    renderIkDebugLogButtonLegacy(state);
    renderMessages(state.messages || []);
  }

  function fetchState() {
    if (window.__calibrationWebUiModernRefreshOk) {
      return;
    }
    try {
      var xhr = new XMLHttpRequest();
      xhr.open("GET", "/state.json?legacy_ts=" + new Date().getTime(), true);
      xhr.onreadystatechange = function () {
        if (xhr.readyState !== 4) return;
        if (xhr.status !== 200) {
          setText("captureModeDetail", "legacy 轮询 state.json 失败: HTTP " + xhr.status);
          return;
        }
        try {
          renderState(JSON.parse(xhr.responseText));
        } catch (error) {
          setText("captureModeDetail", "legacy 解析 state.json 失败: " + error);
        }
      };
      xhr.send(null);
    } catch (error) {
      setText("captureModeDetail", "legacy 请求 state.json 失败: " + error);
    }
  }

  function collectEpisodeStartPayloadLegacy() {
    return {
      dataset_base_root: byId("datasetBaseRootInput") ? byId("datasetBaseRootInput").value : "",
      task_type: byId("taskTypeSelect") ? byId("taskTypeSelect").value : normalizeTaskTypeLegacy(legacyLatestState.collection_task_type),
      task_key: byId("taskSelect") ? byId("taskSelect").value : "",
      task_english_name: byId("taskEnglishNameInput") ? byId("taskEnglishNameInput").value : "",
      task_chinese_name: byId("taskChineseNameInput") ? byId("taskChineseNameInput").value : "",
      round_trip_phase: normalizeRoundTripPhaseLegacy(legacyLatestState.collection_round_trip_phase)
    };
  }

  function postEvent(eventName, extraPayload) {
    if (window.__calibrationWebUiModernRefreshOk) return;
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
        if (xhr.readyState === 4 && xhr.status >= 400) {
          setText("sendStatus", "事件未生效: HTTP " + xhr.status);
        }
      };
      xhr.send(JSON.stringify(payload));
      if (eventName === "d") {
        var button = byId("ikDebugLogButton");
        if (button) button.textContent = "IK 记录切换中...";
      }
    } catch (error) {
      setText("captureModeDetail", "legacy 发送按钮事件失败: " + error);
    }
  }

  function chooseDatasetDirectoryLegacy() {
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

  function isEditableTargetLegacy(target) {
    var tagName = target && target.tagName ? String(target.tagName).toUpperCase() : "";
    return Boolean(target && (target.isContentEditable || tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT"));
  }

  function postKeyboardShortcutLegacy(key) {
    var state = legacyLatestState || {};
    var eventName = "";
    var payload = null;
    if (key === "a") {
      eventName = (state.episode_recording || String(state.episode_event_status || "") === "preparing") ? "episode_stop" : "episode_start";
      if (eventName === "episode_start") payload = collectEpisodeStartPayloadLegacy();
    } else if (key === "b") {
      eventName = state.teleop_running ? "h" : "p";
    } else if (key === "c") {
      if (state.fix_arm_available && state.teleop_running && legacyActiveControls["fix_arm_toggle"]) {
        eventName = "fix_arm_toggle";
      }
    } else if (key === "g") {
      if (state.init_arm_available && state.teleop_running && legacyActiveControls["init_arm_toggle"]) {
        eventName = "init_arm_toggle";
      }
    } else {
      return false;
    }
    if (!eventName) {
      setText("sendStatus", "快捷键 " + key.toUpperCase() + " 当前不可用");
      return true;
    }
    if (!legacyActiveControls[eventName]) {
      setText("sendStatus", "快捷键 " + key.toUpperCase() + " 当前不可用");
      return true;
    }
    postEvent(eventName, payload);
    setText("sendStatus", "快捷键 " + key.toUpperCase() + " 已发送");
    return true;
  }

  function bindKeyboard() {
    document.onkeydown = function (event) {
      if (window.__calibrationWebUiModernRefreshOk) return;
      event = event || window.event;
      if (event.repeat || isEditableTargetLegacy(event.target || event.srcElement)) return;
      var key = String(event.key || "").toLowerCase();
      if (key !== "a" && key !== "b" && key !== "c" && key !== "g") return;
      if (event.preventDefault) event.preventDefault();
      event.returnValue = false;
      postKeyboardShortcutLegacy(key);
    };
  }

  function bindButtons() {
    var buttons = document.querySelectorAll ? document.querySelectorAll("button[data-event]") : [];
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].onclick = function () {
        var eventName = this.getAttribute("data-event");
        if (
          eventName === "c" &&
          !(legacyLatestState.fix_arm_available && legacyLatestState.teleop_running)
        ) {
          setText("sendStatus", "C 仅用于运行中的 fix-arm");
          return;
        }
        var sendEventName = eventName === "c" ? "fix_arm_toggle" : eventName;
        postEvent(sendEventName, eventName === "episode_start" ? collectEpisodeStartPayloadLegacy() : null);
      };
    }
    var chooseButton = byId("datasetChooseRootButton");
    if (chooseButton) chooseButton.onclick = chooseDatasetDirectoryLegacy;
  }

  window.__calibrationWebUiLegacyPollActive = true;
  markScriptRunning();
  bindButtons();
  bindKeyboard();
  fetchState();
  legacyTimer = window.setInterval(fetchState, 500);
})();
</script>
<script>
window.__calibrationWebUiModernScriptLoaded = true;
const buttons = document.querySelectorAll("button[data-event]");
const stageEl = document.getElementById("stage");
const pillsEl = document.getElementById("pills");
const messagesEl = document.getElementById("messages");
const selectedSideMetricEl = document.getElementById("selectedSideMetric");
const selectedSideEl = document.getElementById("selectedSide");
const frameVizModeEl = document.getElementById("frameVizMode");
const virtualDeskYawEl = document.getElementById("virtualDeskYaw");
const controlHintEl = document.getElementById("controlHint");
const captureStageEl = document.getElementById("captureStage");
const captureSamplesEl = document.getElementById("captureSamples");
const captureValidEl = document.getElementById("captureValid");
const captureStatusEl = document.getElementById("captureStatus");
const sendStatusEl = document.getElementById("sendStatus");
const shortcutCLabelEl = document.getElementById("shortcutCLabel");
const shortcutGLabelEl = document.getElementById("shortcutGLabel");
const viveCalibrationCalibrateButtonEl = document.getElementById("viveCalibrationCalibrateButton");
const viveCalibrationSaveButtonEl = document.getElementById("viveCalibrationSaveButton");
const viveCalibrationLoadButtonEl = document.getElementById("viveCalibrationLoadButton");
const viveCalibrationStatusEl = document.getElementById("viveCalibrationStatus");
const movementMultipleInputEl = document.getElementById("movementMultipleInput");
const movementMultipleStatusEl = document.getElementById("movementMultipleStatus");
const gripperForceSectionEl = document.getElementById("gripperForceSection");
const leftGripperForceInputEl = document.getElementById("leftGripperForceInput");
const rightGripperForceInputEl = document.getElementById("rightGripperForceInput");
const gripperForceApplyButtonEl = document.getElementById("gripperForceApplyButton");
const gripperForceStatusEl = document.getElementById("gripperForceStatus");
const fixArmSectionEl = document.getElementById("fixArmSection");
const fixArmSideSelectEl = document.getElementById("fixArmSideSelect");
const fixArmToggleButtonEl = document.getElementById("fixArmToggleButton");
const fixArmStatusEl = document.getElementById("fixArmStatus");
const initArmSectionEl = document.getElementById("initArmSection");
const initArmSideSelectEl = document.getElementById("initArmSideSelect");
const initArmToggleButtonEl = document.getElementById("initArmToggleButton");
const initArmStatusEl = document.getElementById("initArmStatus");
const footPadSpecialArmHintEl = document.getElementById("footPadSpecialArmHint");
const leftRawCardEl = document.getElementById("leftRawCard");
const rightRawCardEl = document.getElementById("rightRawCard");
const leftRawPoseEl = document.getElementById("leftRawPose");
const rightRawPoseEl = document.getElementById("rightRawPose");
const leftRawStatusEl = document.getElementById("leftRawStatus");
const rightRawStatusEl = document.getElementById("rightRawStatus");
const captureModeEl = document.getElementById("captureMode");
const captureModeLabelEl = document.getElementById("captureModeLabel");
const captureModeDetailEl = document.getElementById("captureModeDetail");
const captureProgressBarEl = document.getElementById("captureProgressBar");
const runtimeHomeDiagnosticEl = document.getElementById("runtimeHomeDiagnostic");
const runtimeHomeDiagnosticTitleEl = document.getElementById("runtimeHomeDiagnosticTitle");
const runtimeHomeDiagnosticDetailEl = document.getElementById("runtimeHomeDiagnosticDetail");
const ikWarningBannerEl = document.getElementById("ikWarningBanner");
const ikWarningTextEl = document.getElementById("ikWarningText");
const poseQualityCardEl = document.getElementById("poseQualityCard");
const poseQualityStatusEl = document.getElementById("poseQualityStatus");
const poseQualityScoreEl = document.getElementById("poseQualityScore");
const poseQualityDetailEl = document.getElementById("poseQualityDetail");
const ikTimingSectionEl = document.getElementById("ikTimingSection");
const ikTimingTotalMetricEl = document.getElementById("ikTimingTotalMetric");
const ikTimingTotalEl = document.getElementById("ikTimingTotal");
const ikTimingSolveEl = document.getElementById("ikTimingSolve");
const ikTimingCollisionEl = document.getElementById("ikTimingCollision");
const ikTimingCallsEl = document.getElementById("ikTimingCalls");
const ikTimingStatusEl = document.getElementById("ikTimingStatus");
const ikDebugLogButtonEl = document.getElementById("ikDebugLogButton");
const episodeStartButtonEl = document.getElementById("episodeStartButton");
const episodeStopButtonEl = document.getElementById("episodeStopButton");
const episodeBannerEl = document.getElementById("episodeBanner");
const episodeBannerStateEl = document.getElementById("episodeBannerState");
const episodeBannerCounterEl = document.getElementById("episodeBannerCounter");
const episodeBannerDetailEl = document.getElementById("episodeBannerDetail");
const episodeBannerMetaEl = document.getElementById("episodeBannerMeta");
const episodeStatusEl = document.getElementById("episodeStatus");
const datasetBaseRootInputEl = document.getElementById("datasetBaseRootInput");
const datasetChooseRootButtonEl = document.getElementById("datasetChooseRootButton");
const taskTypeSelectEl = document.getElementById("taskTypeSelect");
const taskSelectEl = document.getElementById("taskSelect");
const taskEnglishNameInputEl = document.getElementById("taskEnglishNameInput");
const taskChineseNameInputEl = document.getElementById("taskChineseNameInput");
const roundTripToggleButtonEl = document.getElementById("roundTripToggleButton");
const roundTripStatusEl = document.getElementById("roundTripStatus");
const episodeWarningBannerEl = document.getElementById("episodeWarningBanner");
const episodeWarningTextEl = document.getElementById("episodeWarningText");
const cameraVisButtonEl = document.getElementById("cameraVisButton");
const tactileVisButtonEl = document.getElementById("tactileVisButton");
const visualizationStatusEl = document.getElementById("visualizationStatus");
const poseSnapshotSectionEl = document.getElementById("poseSnapshotSection");
const poseSnapshotButtonEl = document.getElementById("poseSnapshotButton");
const poseSnapshotStatusEl = document.getElementById("poseSnapshotStatus");
const gripperStatusSectionEl = document.getElementById("gripperStatusSection");
const gripperCardsEl = document.getElementById("gripperCards");
const cameraStatusSectionEl = document.getElementById("cameraStatusSection");
const cameraSelfCheckSectionEl = document.getElementById("cameraSelfCheckSection");
const cameraSelfCheckCardEl = document.getElementById("cameraSelfCheckCard");
const cameraSelfCheckStatusEl = document.getElementById("cameraSelfCheckStatus");
const cameraSelfCheckDetailEl = document.getElementById("cameraSelfCheckDetail");
const cameraCardsEl = document.getElementById("cameraCards");
const tactileCameraStatusSectionEl = document.getElementById("tactileCameraStatusSection");
const tactileCameraCardsEl = document.getElementById("tactileCameraCards");
const shutdownOverlayEl = document.getElementById("shutdownOverlay");
const shutdownMessageEl = document.getElementById("shutdownMessage");
const scriptHealthEl = document.getElementById("scriptHealth");
let activeControls = new Set();
let refreshTimer = null;
let shutdownVisible = false;
let latestIkDebugLogging = false;
let ikDebugLogTogglePending = false;
let ikDebugLogToggleExpected = null;
let ikDebugLogToggleRequestedAt = 0;
let stateRefreshFailureCount = 0;
let latestState = {};
let taskTypeOverride = null;
let roundTripPhaseOverride = null;

const stageLabels = {
  "Starting...": "启动中...",
  "Link Check": "连接检查",
  "Virtual Desk Calibration": "虚拟桌面标定",
  "VirtualDeskX": "虚拟桌面 X 轴",
  "VirtualDeskY": "虚拟桌面 Y 轴",
  "VirtualDeskZ": "虚拟桌面 Z 轴",
  "Wrist Offset Capture": "手腕外参采集",
  "WristOffset": "手腕外参采集",
  "Wrist/Home Capture": "手腕外参采集",
  "WristHome": "手腕外参采集",
  "Wrist Coarse": "手腕/起点采集",
  "WristCoarse": "手腕/起点采集",
  "Tracker Frame Viz": "坐标系检查",
  "Home Capture": "起点采集",
  "HomeCapture": "起点采集",
  "Validate": "验证",
  "Save": "保存",
  "Saved": "已保存",
  "teleop": "遥操",
};
const sideLabels = {left: "左手", right: "右手"};
const frameVizLabels = {calibrated: "校准后", raw: "原始", both: "全部"};
const phaseLabels = {
  idle: "空闲",
  ready: "准备就绪",
  preparing: "准备倒计时",
  capturing: "正在标定",
  waiting: "等待样本",
  teleop: "遥操中",
};
const taskTypeLabels = {general: "一般类", round_trip: "来回类"};
const roundTripPhaseLabels = {in: "_in", out: "_out"};

function labelFor(mapping, value) {
  return mapping[value] || value || "-";
}

function normalizeTaskType(value) {
  const text = String(value || "").trim().toLowerCase().replace(/[-\\s]+/g, "_");
  if (["roundtrip", "round_trip", "back_and_forth", "in_out", "inout", "往返", "来回"].includes(text)) {
    return "round_trip";
  }
  return "general";
}

function normalizeRoundTripPhase(value) {
  return String(value || "").trim().toLowerCase() === "out" ? "out" : "in";
}

function showShutdown(message) {
  if (shutdownVisible) return;
  shutdownVisible = true;
  shutdownMessageEl.textContent = message || "这个网页已经和后端断开连接，可以关闭。";
  shutdownOverlayEl.hidden = false;
  buttons.forEach((button) => { button.disabled = true; });
  activeControls = new Set();
  if (refreshTimer != null) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}

function showRefreshWarning(error) {
  stateRefreshFailureCount += 1;
  window.__calibrationWebUiModernRefreshOk = false;
  if (scriptHealthEl) {
    scriptHealthEl.hidden = false;
    scriptHealthEl.textContent =
      `state.json 刷新失败 ${stateRefreshFailureCount} 次，正在继续重试；请检查浏览器代理是否绕过 127.0.0.1/localhost。${String(error || "")}`;
  }
}

function clearRefreshWarning() {
  stateRefreshFailureCount = 0;
  if (scriptHealthEl) {
    scriptHealthEl.hidden = true;
  }
}

async function sendEvent(eventName, extraPayload) {
  const payload = {event: eventName};
  if (extraPayload && typeof extraPayload === "object") {
    Object.assign(payload, extraPayload);
  }
  const response = await fetch("/event", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error("事件未生效: " + response.status);
  sendStatusEl.textContent = "已发送: " + eventName;
  if (eventName === "d") {
    ikDebugLogTogglePending = true;
    ikDebugLogToggleExpected = !latestIkDebugLogging;
    ikDebugLogToggleRequestedAt = Date.now();
    updateIkDebugLogButton({
      debug_ik_logging: latestIkDebugLogging,
      ik_debug_log_path: "",
    });
  }
  if (eventName === "q") {
    showShutdown("退出/停臂指令已发送，后端会先停止机械臂再关闭。若浏览器没有自动关闭，可以直接关闭此页。");
    setTimeout(() => {
      try {
        window.close();
      } catch (error) {
        // Some browsers only allow closing tabs opened by script.
      }
    }, 600);
  }
}

function updateIkDebugLogButton(state) {
  if (!ikDebugLogButtonEl) return;
  const logging = Boolean(state.debug_ik_logging);
  if (
    ikDebugLogTogglePending &&
    ikDebugLogToggleExpected !== null &&
    logging === Boolean(ikDebugLogToggleExpected)
  ) {
    ikDebugLogTogglePending = false;
  }
  if (ikDebugLogTogglePending && Date.now() - ikDebugLogToggleRequestedAt > 5000) {
    ikDebugLogTogglePending = false;
  }
  latestIkDebugLogging = logging;
  if (ikDebugLogTogglePending) {
    ikDebugLogButtonEl.textContent = "IK 记录切换中...";
  } else {
    ikDebugLogButtonEl.textContent = logging ? "停止 IK 记录" : "开始 IK 记录";
  }
  ikDebugLogButtonEl.disabled = !activeControls.has("d") || ikDebugLogTogglePending;
  ikDebugLogButtonEl.classList.toggle("active", logging);
}

function collectEpisodeStartPayload() {
  const selectedTaskType = taskTypeSelectEl
    ? taskTypeSelectEl.value
    : normalizeTaskType(taskTypeOverride || latestState.collection_task_type);
  return {
    dataset_base_root: datasetBaseRootInputEl ? datasetBaseRootInputEl.value : "",
    task_type: selectedTaskType,
    task_key: taskSelectEl ? taskSelectEl.value : "",
    task_english_name: taskEnglishNameInputEl ? taskEnglishNameInputEl.value : "",
    task_chinese_name: taskChineseNameInputEl ? taskChineseNameInputEl.value : "",
    round_trip_phase: normalizeRoundTripPhase(roundTripPhaseOverride || latestState.collection_round_trip_phase),
  };
}

async function chooseDatasetDirectory() {
  if (!datasetBaseRootInputEl || !datasetChooseRootButtonEl || datasetChooseRootButtonEl.disabled) return;
  const originalText = datasetChooseRootButtonEl.textContent;
  datasetChooseRootButtonEl.disabled = true;
  datasetChooseRootButtonEl.textContent = "选择中...";
  try {
    const initial = encodeURIComponent(datasetBaseRootInputEl.value || latestState.dataset_base_root || "");
    const response = await fetch("/choose-directory?initial=" + initial, {cache: "no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    const result = await response.json();
    if (result && result.path) {
      datasetBaseRootInputEl.value = result.path;
      sendStatusEl.textContent = "已选择文件夹: " + result.path;
    } else {
      sendStatusEl.textContent = (result && result.message) || "未选择文件夹";
    }
  } catch (error) {
    sendStatusEl.textContent = "选择文件夹失败: " + String(error);
  } finally {
    datasetChooseRootButtonEl.textContent = originalText || "选择文件夹";
    const latestEpisodeStatus = String(latestState.episode_event_status || "");
    datasetChooseRootButtonEl.disabled = Boolean(latestState.episode_recording) || latestEpisodeStatus === "preparing" || latestEpisodeStatus === "finalizing";
  }
}

function isEditableTarget(target) {
  if (!target) return false;
  const tagName = String(target.tagName || "").toUpperCase();
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(tagName);
}

function buttonForEvent(eventName) {
  return Array.from(buttons).find((button) => button.dataset.event === eventName) || null;
}

function canSendEvent(eventName) {
  const button = buttonForEvent(eventName);
  if (button) return !button.disabled;
  return activeControls.has(eventName);
}

async function sendGuiShortcut(key) {
  const state = latestState || {};
  const shortcutStatus = String(state.episode_event_status || "");
  let eventName = "";
  let payload = null;
  if (key === "a") {
    eventName = (state.episode_recording || shortcutStatus === "preparing") ? "episode_stop" : "episode_start";
    if (eventName === "episode_start") payload = collectEpisodeStartPayload();
  } else if (key === "b") {
    eventName = state.teleop_running ? "h" : "p";
  } else if (key === "c") {
    if (state.fix_arm_available && state.teleop_running && canSendEvent("fix_arm_toggle")) {
      eventName = "fix_arm_toggle";
    }
  } else if (key === "g") {
    if (state.init_arm_available && state.teleop_running && canSendEvent("init_arm_toggle")) {
      eventName = "init_arm_toggle";
    }
  } else {
    return false;
  }
  if (!eventName) {
    const label = key.toUpperCase();
    sendStatusEl.textContent = `快捷键 ${label} 当前不可用`;
    return true;
  }
  if (!canSendEvent(eventName)) {
    const label = key.toUpperCase();
    sendStatusEl.textContent = `快捷键 ${label} 当前不可用`;
    return true;
  }
  await sendEvent(eventName, payload);
  sendStatusEl.textContent = `快捷键 ${key.toUpperCase()} 已发送`;
  return true;
}

function clampNumber(value, minValue, maxValue) {
  if (!Number.isFinite(value)) return minValue;
  return Math.max(minValue, Math.min(maxValue, value));
}

async function sendMovementMultipleSet() {
  if (!movementMultipleInputEl || movementMultipleInputEl.disabled) return;
  const minValue = Number(movementMultipleInputEl.min || 0.05);
  const maxValue = Number(movementMultipleInputEl.max || 10);
  const value = clampNumber(Number(movementMultipleInputEl.value), minValue, maxValue);
  movementMultipleInputEl.value = value.toFixed(3);
  try {
    await sendEvent("movement_multiple_set:" + value.toFixed(6));
  } catch (error) {
    sendStatusEl.textContent = String(error);
  }
}

function gripperForceValue(input, side) {
  const value = Number(input.value);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${sideLabel(side)}夹爪力度必须在 0 到 1 之间`);
  }
  return value;
}

async function sendGripperForceSet() {
  if (!gripperForceApplyButtonEl || gripperForceApplyButtonEl.disabled) return;
  const payload = {
    left: leftGripperForceInputEl && !leftGripperForceInputEl.disabled
      ? gripperForceValue(leftGripperForceInputEl, "left")
      : null,
    right: rightGripperForceInputEl && !rightGripperForceInputEl.disabled
      ? gripperForceValue(rightGripperForceInputEl, "right")
      : null,
  };
  try {
    await sendEvent("gripper_force_set:" + JSON.stringify(payload));
  } catch (error) {
    sendStatusEl.textContent = String(error);
  }
}

buttons.forEach((button) => {
  button.addEventListener("click", async () => {
    if (button.disabled) return;
    try {
      const eventName = button.dataset.event;
      if (eventName === "round_trip_toggle") {
        const selectedType = normalizeTaskType(
          taskTypeSelectEl ? taskTypeSelectEl.value : taskTypeOverride || latestState.collection_task_type
        );
        const backendRoundTripReady = (
          normalizeTaskType(latestState.collection_task_type) === "round_trip" &&
          !taskTypeOverride
        );
        if (selectedType === "round_trip" && !backendRoundTripReady) {
          const current = normalizeRoundTripPhase(roundTripPhaseOverride || latestState.collection_round_trip_phase);
          roundTripPhaseOverride = current === "in" ? "out" : "in";
          renderTaskForm(Object.assign({}, latestState, {
            collection_task_type: "round_trip",
            collection_round_trip_phase: roundTripPhaseOverride,
            collection_round_trip_next_phase: current,
          }), false);
          sendStatusEl.textContent = `已切换待采方向: ${roundTripPhaseLabels[roundTripPhaseOverride] || "_" + roundTripPhaseOverride}`;
          return;
        }
      }
      if (
        eventName === "c" &&
        !(latestState.fix_arm_available && latestState.teleop_running)
      ) {
        sendStatusEl.textContent = "C 仅用于运行中的 fix-arm";
        return;
      }
      const sendEventName = eventName === "c" ? "fix_arm_toggle" : eventName;
      if (eventName === "episode_start") {
        await sendEvent(sendEventName, collectEpisodeStartPayload());
      } else {
        await sendEvent(sendEventName);
      }
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
  });
});

if (movementMultipleInputEl) {
  movementMultipleInputEl.addEventListener("change", sendMovementMultipleSet);
  movementMultipleInputEl.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    await sendMovementMultipleSet();
  });
}

if (gripperForceApplyButtonEl) {
  gripperForceApplyButtonEl.addEventListener("click", async () => {
    try {
      await sendGripperForceSet();
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
  });
}

for (const input of [leftGripperForceInputEl, rightGripperForceInputEl]) {
  if (!input) continue;
  input.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    try {
      await sendGripperForceSet();
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
  });
}

if (datasetChooseRootButtonEl) {
  datasetChooseRootButtonEl.addEventListener("click", async () => {
    await chooseDatasetDirectory();
  });
}

if (taskTypeSelectEl) {
  taskTypeSelectEl.addEventListener("change", () => {
    const nextType = normalizeTaskType(taskTypeSelectEl.value);
    taskTypeOverride = nextType;
    roundTripPhaseOverride = nextType === "round_trip" ? "in" : null;
    const draftState = Object.assign({}, latestState, {
      collection_task_type: nextType,
      collection_task_key: "",
      collection_task_id: "",
      collection_round_trip_phase: roundTripPhaseOverride || "in",
    });
    renderTaskForm(draftState, false);
  });
}

if (taskSelectEl) {
  taskSelectEl.addEventListener("change", async () => {
    const selectedKey = taskSelectEl.value;
    const stateRegistry = window.__latestTaskRegistry || [];
    const item = stateRegistry.find((entry) => (
      (entry.key || entry.gripper_task || "") === selectedKey
    ));
    if (!item) return;
    if (taskTypeSelectEl) {
      taskTypeOverride = normalizeTaskType(item.task_type);
      roundTripPhaseOverride = taskTypeOverride === "round_trip" ? "in" : null;
      taskTypeSelectEl.value = taskTypeOverride;
    }
    if (taskEnglishNameInputEl) taskEnglishNameInputEl.value = item.english_name || item.key || "";
    if (taskChineseNameInputEl) taskChineseNameInputEl.value = item.chinese_name || "";
    try {
      await sendEvent("task_select", collectEpisodeStartPayload());
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
  });
}

if (fixArmSideSelectEl) {
  fixArmSideSelectEl.addEventListener("change", async () => {
    if (fixArmSideSelectEl.disabled) return;
    try {
      await sendEvent("fix_arm_select:" + fixArmSideSelectEl.value);
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
  });
}

if (initArmSideSelectEl) {
  initArmSideSelectEl.addEventListener("change", async () => {
    if (initArmSideSelectEl.disabled) return;
    try {
      await sendEvent("init_arm_select:" + initArmSideSelectEl.value);
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
  });
}

document.addEventListener("keydown", async (event) => {
  if (event.repeat) return;
  if (isEditableTarget(event.target)) return;
  const key = event.key.toLowerCase();
  if (["a", "b", "c", "g"].includes(key)) {
    event.preventDefault();
    try {
      await sendGuiShortcut(key);
    } catch (error) {
      sendStatusEl.textContent = String(error);
    }
    return;
  }
  if (!activeControls.has(key)) return;
  event.preventDefault();
  try {
    await sendEvent(key);
  } catch (error) {
    sendStatusEl.textContent = String(error);
  }
});

function updateButtons(state) {
  activeControls = new Set(state.active_controls || []);
  const episodeStatus = String(state.episode_event_status || "");
  const episodePreparing = episodeStatus === "preparing";
  const episodeFinalizing = episodeStatus === "finalizing";
  buttons.forEach((button) => {
    const eventName = button.dataset.event;
    button.disabled = !activeControls.has(eventName);
    if (
      state.capturing &&
      eventName === "c" &&
      !(state.fix_arm_available && state.teleop_running)
    ) {
      button.disabled = true;
    }
    if ((state.episode_recording || episodePreparing || episodeFinalizing) && eventName === "episode_start") button.disabled = true;
    if (!state.episode_recording && !episodePreparing && eventName === "episode_stop") button.disabled = true;
    if ((state.episode_recording || episodePreparing) && eventName === "record_pose_snapshot") button.disabled = true;
    const isSelectedSideButton =
      (eventName === "1" && state.selected_side === "left") ||
      (eventName === "2" && state.selected_side === "right");
    button.classList.toggle("active", isSelectedSideButton);
  });
  updateIkDebugLogButton(state);
  if (viveCalibrationSaveButtonEl) {
    viveCalibrationSaveButtonEl.classList.toggle("active", Boolean(state.vive_calibration_profile_ready));
  }
  if (viveCalibrationLoadButtonEl) {
    viveCalibrationLoadButtonEl.classList.toggle("active", Boolean(state.vive_calibration_profile_loaded));
  }
  if (viveCalibrationCalibrateButtonEl) {
    viveCalibrationCalibrateButtonEl.classList.toggle("active", Boolean(!state.teleop_running && String(state.capture_stage || "").startsWith("VirtualDesk")));
  }
  if (movementMultipleInputEl) {
    movementMultipleInputEl.disabled = !activeControls.has("movement_multiple_set");
  }
  const captureButton = Array.from(buttons).find((button) => button.dataset.event === "c");
  if (captureButton) {
    captureButton.hidden = String(state.special_arm_control_mode || "manual") === "foot-pad";
    if (state.fix_arm_available && state.teleop_running) {
      const side = state.fix_arm_fixed_side || state.fix_arm_selected_side || "left";
      const label = side === "right" ? "右臂" : "左臂";
      captureButton.textContent = state.fix_arm_recalibrating
        ? `重采 ${label} Home...`
        : state.fix_arm_active
          ? `重采 ${label} Home`
          : `固定 ${label}`;
    } else if (state.teleop_running) {
      captureButton.textContent = "无 fix-arm 时不可用";
    } else {
      captureButton.textContent = "固定臂不可用";
    }
  }
}

function updateShortcutHints(state) {
  const footPadMode = String(state.special_arm_control_mode || "manual") === "foot-pad";
  if (shortcutCLabelEl) {
    if (footPadMode && state.fix_arm_available) {
      shortcutCLabelEl.textContent = "踏板 1/2：固定 / 重采左右臂 Home";
    } else if (state.fix_arm_available && state.teleop_running) {
      const side = state.fix_arm_fixed_side || state.fix_arm_selected_side || "left";
      const label = side === "right" ? "右臂" : "左臂";
      shortcutCLabelEl.textContent = state.fix_arm_recalibrating
        ? `重采 ${label} Home`
        : state.fix_arm_active
          ? `重采 ${label} Home`
          : `固定 ${label}`;
    } else if (state.teleop_running) {
      shortcutCLabelEl.textContent = "无 fix-arm 时不可用";
    } else {
      shortcutCLabelEl.textContent = "固定臂不可用";
    }
  }
  if (shortcutGLabelEl) {
    if (footPadMode && state.init_arm_available) {
      shortcutGLabelEl.textContent = "踏板 3/4：归位 / 重采左右臂 Home";
    } else if (state.init_arm_available && state.teleop_running) {
      const side = state.init_arm_active_side || state.init_arm_selected_side || "left";
      const label = side === "right" ? "右臂" : "左臂";
      shortcutGLabelEl.textContent = state.init_arm_recalibrating
        ? `重采 ${label} Home`
        : state.init_arm_active
          ? `重采 ${label} Home`
          : `${label}归位`;
    } else {
      shortcutGLabelEl.textContent = "归位不可用";
    }
  }
}

function renderMovementMultiple(state) {
  if (!movementMultipleInputEl || !movementMultipleStatusEl) return;
  const value = Number(state.movement_multiple == null ? 1.0 : state.movement_multiple);
  const minValue = Number(state.movement_multiple_min == null ? 0.05 : state.movement_multiple_min);
  const maxValue = Number(state.movement_multiple_max == null ? 10.0 : state.movement_multiple_max);
  const stepValue = Number(state.movement_multiple_step == null ? 0.1 : state.movement_multiple_step);
  movementMultipleInputEl.min = String(minValue);
  movementMultipleInputEl.max = String(maxValue);
  movementMultipleInputEl.step = String(stepValue);
  if (document.activeElement !== movementMultipleInputEl && Number.isFinite(value)) {
    movementMultipleInputEl.value = value.toFixed(3);
  }
  movementMultipleStatusEl.textContent = `movement_multiple=${Number.isFinite(value) ? value.toFixed(3) : "-"}`;
}

function renderGripperForce(state) {
  const available = Boolean(state.gripper_control_available);
  if (gripperForceSectionEl) gripperForceSectionEl.hidden = !available;
  if (!available) return;

  const canSet = activeControls.has("gripper_force_set");
  const gripperStatus = state.gripper_status || {};
  const inputs = {
    left: leftGripperForceInputEl,
    right: rightGripperForceInputEl,
  };
  let configuredCount = 0;
  for (const side of ["left", "right"]) {
    const input = inputs[side];
    if (!input) continue;
    const item = gripperStatus[side] || {};
    const configured = Boolean(item.configured);
    if (configured) configuredCount += 1;
    input.disabled = !canSet || !configured;
    const torque = Number(item.torque);
    if (configured && document.activeElement !== input && Number.isFinite(torque)) {
      input.value = torque.toFixed(3);
    }
  }
  if (gripperForceApplyButtonEl) {
    gripperForceApplyButtonEl.disabled = !canSet || configuredCount === 0;
  }
  if (gripperForceStatusEl) {
    gripperForceStatusEl.textContent = !canSet
      ? "数采录制、倒计时或保存中已锁定；当前 episode 按任务配置运行。"
      : configuredCount === 0
        ? "未检测到可设置的 DM-Tac 夹爪。"
        : "范围 0–1；应用后立即更新已连接夹爪，开始数采前会恢复任务配置。";
  }
}

function renderFixArm(state) {
  const available = Boolean(state.fix_arm_available);
  if (fixArmSectionEl) fixArmSectionEl.hidden = !available;
  const footPadMode = String(state.special_arm_control_mode || "manual") === "foot-pad";
  if (footPadSpecialArmHintEl) footPadSpecialArmHintEl.hidden = !footPadMode;
  if (!available) return;
  const selected = state.fix_arm_selected_side || "left";
  const fixed = state.fix_arm_fixed_side || "";
  const recalibrating = Boolean(state.fix_arm_recalibrating);
  const side = fixed || selected;
  const label = side === "right" ? "右臂" : "左臂";
  if (fixArmSideSelectEl) {
    if (fixArmSideSelectEl.parentElement && fixArmSideSelectEl.parentElement.parentElement) {
      fixArmSideSelectEl.parentElement.parentElement.hidden = footPadMode;
    }
    fixArmSideSelectEl.disabled = !activeControls.has("fix_arm_select") || Boolean(fixed);
    if (document.activeElement !== fixArmSideSelectEl) {
      fixArmSideSelectEl.value = selected;
    }
  }
  if (fixArmToggleButtonEl) {
    if (fixArmToggleButtonEl.parentElement) fixArmToggleButtonEl.parentElement.hidden = footPadMode;
    fixArmToggleButtonEl.disabled = !activeControls.has("fix_arm_toggle");
    fixArmToggleButtonEl.textContent = recalibrating
      ? `重采 ${label} Home...`
      : fixed
        ? `重采 ${label} Home`
        : `固定 ${label}`;
  }
  if (!fixArmStatusEl) return;
  const lines = [];
  if (recalibrating) {
    const count = state.fix_arm_sample_count == null ? "-" : String(state.fix_arm_sample_count);
    const total = state.fix_arm_min_samples == null ? "-" : String(state.fix_arm_min_samples);
    lines.push(`${label}: 重采 home ${count}/${total}`);
  } else if (fixed) {
    lines.push(`${label}: 固定中`);
  } else {
    lines.push(`${label}: 待固定`);
  }
  if (state.fix_arm_status) lines.push(String(state.fix_arm_status));
  fixArmStatusEl.textContent = lines.join("\\n");
}

function renderInitArm(state) {
  const available = Boolean(state.init_arm_available);
  if (initArmSectionEl) initArmSectionEl.hidden = !available;
  const footPadMode = String(state.special_arm_control_mode || "manual") === "foot-pad";
  if (!available) return;
  const selected = state.init_arm_selected_side || "left";
  const activeSide = state.init_arm_active_side || "";
  const recalibrating = Boolean(state.init_arm_recalibrating);
  const side = activeSide || selected;
  const label = side === "right" ? "右臂" : "左臂";
  if (initArmSideSelectEl) {
    if (initArmSideSelectEl.parentElement && initArmSideSelectEl.parentElement.parentElement) {
      initArmSideSelectEl.parentElement.parentElement.hidden = footPadMode;
    }
    initArmSideSelectEl.disabled = !activeControls.has("init_arm_select") || Boolean(activeSide);
    if (document.activeElement !== initArmSideSelectEl) {
      initArmSideSelectEl.value = selected;
    }
  }
  if (initArmToggleButtonEl) {
    if (initArmToggleButtonEl.parentElement) initArmToggleButtonEl.parentElement.hidden = footPadMode;
    initArmToggleButtonEl.disabled = !activeControls.has("init_arm_toggle");
    initArmToggleButtonEl.textContent = recalibrating
      ? `重采 ${label} Home...`
      : activeSide
        ? `重采 ${label} Home`
        : `${label}归位`;
  }
  if (!initArmStatusEl) return;
  const lines = [];
  if (recalibrating) {
    const count = state.init_arm_sample_count == null ? "-" : String(state.init_arm_sample_count);
    const total = state.init_arm_min_samples == null ? "-" : String(state.init_arm_min_samples);
    lines.push(`${label}: 重采 home ${count}/${total}`);
  } else if (activeSide) {
    lines.push(`${label}: 归位中`);
  } else {
    lines.push(`${label}: 待归位`);
  }
  if (activeSide && state.init_arm_arrival_label) {
    lines.push(`实际反馈: ${String(state.init_arm_arrival_label)}`);
  }
  if (state.init_arm_status) lines.push(String(state.init_arm_status));
  initArmStatusEl.textContent = lines.join("\\n");
}

function formatCapture(state) {
  const left = state.capture_left_samples;
  const right = state.capture_right_samples;
  if (left != null || right != null) {
    return `左 ${left || 0}/${state.capture_min_samples || 0}  右 ${right || 0}/${state.capture_min_samples || 0}`;
  }
  if (state.capture_samples != null) {
    return `${state.capture_samples}/${state.capture_min_samples || 0}`;
  }
  return "-";
}

function formatValid(state) {
  if (state.capture_left_valid != null || state.capture_right_valid != null) {
    return `左 ${state.capture_left_valid ? "有效" : "缺失"} / 右 ${state.capture_right_valid ? "有效" : "缺失"}`;
  }
  if (state.capture_valid != null) return state.capture_valid ? "有效" : "缺失";
  return "-";
}

function formatCaptureStatus(state) {
  const parts = ['<div class="capture-status-row">'];
  if (state.teleop_running) {
    parts.push('<span class="phase-chip capture active">遥操中</span>');
    parts.push("</div>");
    return parts.join("");
  }
  const phase = state.capture_phase || "idle";
  const prepTotal = state.capture_prep_total_sec;
  const captureTotal = state.capture_duration_sec;
  if (prepTotal != null) {
    const prepText = phase === "preparing" && state.capture_prep_remaining_sec != null
      ? `准备 ${Number(state.capture_prep_remaining_sec).toFixed(1)}s`
      : `准备 ${Number(prepTotal).toFixed(1)}s`;
    parts.push(`<span class="phase-chip prep ${phase === "preparing" ? "active" : ""}">${prepText}</span>`);
  }
  if (captureTotal != null) {
    const captureText = phase === "capturing" && state.capture_elapsed_sec != null
      ? `采集 ${Number(state.capture_elapsed_sec).toFixed(1)}/${Number(captureTotal).toFixed(1)}s`
      : `采集 ${Number(captureTotal).toFixed(1)}s`;
    parts.push(`<span class="phase-chip capture ${phase === "capturing" ? "active" : ""}">${captureText}</span>`);
  }
  if (phase === "waiting" || state.capture_waiting_for_samples) {
    parts.push('<span class="phase-chip wait active">等待样本</span>');
  } else if (!state.capture_stage) {
    parts.push('<span class="phase-chip idle">空闲</span>');
  }
  parts.push("</div>");
  return parts.join("");
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function renderCaptureMode(state) {
  const rawPhase = state.teleop_running ? "teleop" : (state.capture_phase || "idle");
  const phase = (state.teleop_running || state.capture_stage) ? rawPhase : "idle";
  const prepTotal = Number(state.capture_prep_total_sec || 0);
  const prepRemaining = Number(state.capture_prep_remaining_sec || 0);
  const duration = Number(state.capture_duration_sec || 0);
  const elapsed = Number(state.capture_elapsed_sec || 0);
  let progress = 0;
  let detail = "等待流程进入采集阶段";
  const captureStage = String(state.capture_stage || "");
  const sampleLabel = captureStage === "runtime_home" ? "Home 样本" : (captureStage === "force_control_warmup" ? "力控预热" : "标定样本");
  if (phase === "ready") {
    detail = "按“开始采集”后会先进入准备倒计时";
  } else if (phase === "preparing") {
    progress = prepTotal > 0 ? (1 - prepRemaining / prepTotal) * 100 : 100;
    detail = captureStage === "force_control_warmup"
      ? `正在预热 RM65 力控，预计剩余 ${prepRemaining.toFixed(1)}s`
      : `倒计时 ${prepRemaining.toFixed(1)}s，当前还未写入${sampleLabel}`;
  } else if (phase === "capturing") {
    const minSamples = Number(state.capture_min_samples || 0);
    const leftSamples = state.capture_left_samples == null ? null : Number(state.capture_left_samples);
    const rightSamples = state.capture_right_samples == null ? null : Number(state.capture_right_samples);
    const singleSamples = state.capture_samples == null ? null : Number(state.capture_samples);
    if (minSamples > 0 && (leftSamples != null || rightSamples != null)) {
      const sampleProgress = Math.min(leftSamples || 0, rightSamples || 0);
      progress = (sampleProgress / minSamples) * 100;
      detail = `正在采集 ${sampleLabel}，左 ${leftSamples || 0}/${minSamples}，右 ${rightSamples || 0}/${minSamples}`;
    } else if (minSamples > 0 && singleSamples != null) {
      progress = (singleSamples / minSamples) * 100;
      detail = `正在采集 ${sampleLabel}，${singleSamples}/${minSamples}`;
    } else {
      progress = duration > 0 ? (elapsed / duration) * 100 : 0;
      detail = `正在写入样本，已采集 ${elapsed.toFixed(1)} / ${duration.toFixed(1)}s`;
    }
  } else if (phase === "waiting") {
    progress = 100;
    detail = "计时窗口已结束，仍在等待足够的新样本";
  } else if (phase === "teleop") {
    progress = 100;
    detail = "遥操已开始；网页退出会先请求机械臂停臂再关闭程序";
  }
  captureModeEl.className = `capture-mode ${phase}`;
  captureModeLabelEl.textContent = labelFor(phaseLabels, phase);
  captureModeDetailEl.textContent = detail;
  captureProgressBarEl.style.width = `${clampPercent(progress)}%`;
}

function renderRuntimeHomeDiagnostic(state) {
  if (!runtimeHomeDiagnosticEl) return;
  const diagnostic = state.runtime_home_diagnostic || {};
  const visible = Boolean(diagnostic.visible);
  runtimeHomeDiagnosticEl.hidden = !visible;
  if (!visible) {
    runtimeHomeDiagnosticEl.className = "runtime-home-diagnostic";
    if (runtimeHomeDiagnosticTitleEl) runtimeHomeDiagnosticTitleEl.textContent = "";
    if (runtimeHomeDiagnosticDetailEl) runtimeHomeDiagnosticDetailEl.textContent = "";
    return;
  }
  const level = diagnostic.level === "error" ? "error" : "capturing";
  runtimeHomeDiagnosticEl.className = `runtime-home-diagnostic ${level}`;
  if (runtimeHomeDiagnosticTitleEl) {
    runtimeHomeDiagnosticTitleEl.textContent = diagnostic.title || "Runtime Home 状态";
  }
  if (runtimeHomeDiagnosticDetailEl) {
    runtimeHomeDiagnosticDetailEl.textContent = diagnostic.detail || "-";
  }
}

function formatPills(state) {
  const items = [];
  if (state.ui_mode) items.push(["界面", state.ui_mode]);
  if (state.host) items.push(["VIVE", `${state.host}:${state.port}`]);
  if (state.load_profile) items.push(["配置", state.load_profile]);
  if (state.show_tracker_frame_viz) items.push(["坐标检查", "开启"]);
  if (state.home_ready) items.push(["起点", "就绪"]);
  if (state.vive_calibration_profile_ready) items.push(["VIVE 坐标系", "已标定"]);
  if (state.episode_recording) items.push(["数采", "录制中"]);
  if (state.camera_visualization_active) items.push(["图像窗口", "开启"]);
  if (state.tactile_visualization_active) items.push(["触觉窗口", "开启"]);
  return items.map(([label, value]) => `<span class="pill"><strong>${label}</strong>${value}</span>`).join("");
}

function renderViveCalibrationProfile(state) {
  const lines = [];
  lines.push(state.vive_calibration_profile_ready ? "状态: 已有左右 VIVE 坐标系" : "状态: 未确认左右 VIVE 坐标系");
  if (state.vive_calibration_profile_loaded) lines.push("profile: 已应用");
  if (state.vive_calibration_profile_path) lines.push(`路径: ${state.vive_calibration_profile_path}`);
  if (state.vive_calibration_profile_status) lines.push(String(state.vive_calibration_profile_status));
  viveCalibrationStatusEl.textContent = lines.join("\\n");
}

function formatVector(values, unit) {
  if (!values || !Array.isArray(values)) return "-";
  return values.map((value) => value == null ? "nan" : Number(value).toFixed(unit === "m" ? 4 : 1)).join(", ");
}

function formatRawPose(pose) {
  if (pose && pose.enabled === false) return {status: "未启用", text: "当前 logical side 未启用"};
  if (!pose) return {status: "无数据", text: "未收到 channel"};
  const status = pose.status_label || (pose.valid ? "有效" : "无效");
  const tracker = pose.tracker_name || "-";
  const role = pose.tracker_role || "-";
  const age = pose.age_sec == null ? "-" : Number(pose.age_sec).toFixed(2) + " s";
  const seq = pose.seq == null ? "-" : String(pose.seq);
  const xyz = formatVector(pose.xyz_m, "m");
  const rpy = formatVector(pose.rpy_deg, "deg");
  return {
    status,
    text: [
      `tracker: ${tracker}`,
      `role:    ${role}`,
      `seq:     ${seq}`,
      `age:     ${age}`,
      `xyz(m):  ${xyz}`,
      `rpy(deg):${rpy}`,
    ].join("\\n"),
  };
}

function setRawPoseCardClass(el, pose) {
  if (!el) return;
  const level = pose && pose.status_level ? String(pose.status_level) : "ok";
  el.className = `raw-pose-card ${level}`;
}

function renderRawPoses(state) {
  const poses = state.raw_tracker_poses || {};
  const left = formatRawPose(poses.left);
  const right = formatRawPose(poses.right);
  leftRawStatusEl.textContent = left.status;
  rightRawStatusEl.textContent = right.status;
  leftRawPoseEl.textContent = left.text;
  rightRawPoseEl.textContent = right.text;
  setRawPoseCardClass(leftRawCardEl, poses.left);
  setRawPoseCardClass(rightRawCardEl, poses.right);
}

function formatMs(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(1) + " ms";
}

function formatScalar(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(3);
}

function formatAge(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(2) + " s";
}

function formatCount(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return String(Number(value));
}

function formatFlag(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "-";
}

function formatQ6Values(values) {
  if (!Array.isArray(values) || values.length === 0) return "-";
  return "[" + values.map((value) => formatScalar(value)).join(", ") + "]";
}

const gripperCommandLabels = {
  control_disabled: "Q6 控制未启用",
  no_observation: "无 observation",
  no_backend: "无夹爪 backend",
  no_retargeter: "无 Q6 retargeter",
  retarget_exception: "retarget 异常",
  no_retarget_command: "未生成命令",
  missing_effector_command: "缺少 effector 命令",
  retargeted: "已生成命令",
  send_returned_false: "本帧未发送",
  send_exception: "发送异常",
  sent: "已发送",
};

function labelGripperCommandStatus(status) {
  return gripperCommandLabels[status] || status || "-";
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sideLabel(side) {
  return sideLabels[side] || side;
}

function formatTactileReady(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "-";
  const entries = Object.entries(value);
  if (entries.length === 0) return "-";
  const readyCount = entries.filter(([, ready]) => Boolean(ready)).length;
  return `${readyCount}/${entries.length}`;
}

function formatTactileStatus(item) {
  if (!item.connect_tactile) return "disabled";
  const mode = item.tactile_process_read ? "process" : "inline";
  const worker = item.tactile_process_read ? formatFlag(item.tactile_worker_alive) : "-";
  const readMs = item.tactile_read_elapsed_s == null
    ? "-"
    : formatMs(Number(item.tactile_read_elapsed_s) * 1000.0);
  return [
    `ready=${formatTactileReady(item.tactile_ready)}`,
    `ver=${formatCount(item.tactile_version)}`,
    `mode=${mode}`,
    `worker=${worker}`,
    `read=${readMs}`,
    `skip=${formatFlag(item.tactile_read_skipped)}`,
  ].join(" ");
}

function statusLevelClass(value) {
  const level = String(value || "ok");
  return ["ok", "warn", "error", "waiting", "off"].includes(level) ? level : "ok";
}

function formatProfile(item) {
  const width = item.width == null ? "-" : String(item.width);
  const height = item.height == null ? "-" : String(item.height);
  const fps = item.fps == null ? "-" : String(item.fps);
  return `${width}x${height}@${fps} ${item.color_space || ""}`.trim();
}

function formatBrief(value, maxItems = 4) {
  if (value == null) return "-";
  if (Array.isArray(value)) {
    if (value.length === 0) return "-";
    const items = value.slice(0, maxItems).map((item) => {
      if (Array.isArray(item)) return "[" + item.join(",") + "]";
      if (item && typeof item === "object") return JSON.stringify(item);
      return String(item);
    });
    if (value.length > maxItems) items.push("...");
    return items.join(", ");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length === 0) return "-";
    const items = entries.slice(0, maxItems).map(([key, item]) => {
      const rendered = Array.isArray(item)
        ? "[" + item.join(",") + "]"
        : (item && typeof item === "object" ? JSON.stringify(item) : String(item));
      return `${key}:${rendered}`;
    });
    if (entries.length > maxItems) items.push("...");
    return items.join(" ");
  }
  return String(value);
}

function renderCameraSelfCheck(state) {
  if (!cameraSelfCheckSectionEl || !cameraSelfCheckCardEl) return;
  const check = state.camera_self_check || {};
  const status = String(check.status || "idle");
  const enabled = Boolean(check.enabled) || status !== "idle";
  cameraSelfCheckSectionEl.hidden = !enabled;
  if (!enabled) return;
  const level = statusLevelClass(check.status_level || (status === "failed" ? "error" : status === "passed" ? "ok" : "waiting"));
  cameraSelfCheckCardEl.className = `device-card ${level}`;
  if (cameraSelfCheckStatusEl) {
    cameraSelfCheckStatusEl.textContent = check.status_label || status;
  }
  if (!cameraSelfCheckDetailEl) return;
  const lines = [
    `status: ${check.message || "-"}`,
    `elapsed: ${formatAge(check.elapsed_sec)} / ${formatAge(check.duration_sec)}`,
    `cameras: ${formatCount(check.passed_count)}/${formatCount(check.camera_count)}`,
  ];
  if (Array.isArray(check.failed_keys) && check.failed_keys.length) {
    lines.push(`failed: ${check.failed_keys.join(", ")}`);
  }
  if (Array.isArray(check.waiting_keys) && check.waiting_keys.length) {
    lines.push(`waiting: ${check.waiting_keys.join(", ")}`);
  }
  if (Array.isArray(check.warn_keys) && check.warn_keys.length) {
    lines.push(`warn: ${check.warn_keys.join(", ")}`);
  }
  cameraSelfCheckDetailEl.textContent = lines.join("\\n");
}

function renderCameraStatus(state) {
  if (!cameraStatusSectionEl || !cameraCardsEl) return;
  const cameraStatus = state.camera_status || {};
  const entries = Object.entries(cameraStatus).sort(([left], [right]) => left.localeCompare(right));
  cameraStatusSectionEl.hidden = false;
  if (entries.length === 0) {
    cameraCardsEl.innerHTML = `
      <div class="device-card off">
        <div class="device-title"><span>未配置</span><span>off</span></div>
        <div class="device-body mono">当前没有 RealSense camera_sources。</div>
      </div>
    `;
    return;
  }
  cameraCardsEl.innerHTML = entries.map(([key, item]) => {
    const level = statusLevelClass(item.status_level);
    const serial = item.serial_number || item.serial || "-";
    const device = item.device_name || item.product_line || item.name || "-";
    const preview = item.preview_selected
      ? (item.preview_active ? "active" : "selected")
      : "not selected";
    const lines = [
      `serial: ${serial}`,
      `device: ${device}`,
      `profile: ${formatProfile(item)}`,
      `frames: ${formatCount(item.frame_count)} age=${formatAge(item.last_frame_age_sec)}`,
      `timeout: ${formatCount(item.timeout_count)} consecutive=${formatCount(item.consecutive_timeout_count)}`,
      `restart: ${formatCount(item.restart_count)} preview=${preview}`,
    ];
    if (item.depth_required || item.depth_enabled) {
      const depthWidth = item.depth_width == null ? "-" : String(item.depth_width);
      const depthHeight = item.depth_height == null ? "-" : String(item.depth_height);
      const depthFps = item.depth_fps == null ? "-" : String(item.depth_fps);
      lines.push(
        `depth: ${item.depth_status_label || "-"} required=${formatFlag(item.depth_required)} present=${formatFlag(item.depth_frame_present)} profile=${depthWidth}x${depthHeight}@${depthFps} scale=${formatScalar(item.depth_scale_m_per_unit)}`
      );
      lines.push(
        `depth meta: frame=${formatCount(item.depth_frame_number)} ts=${formatScalar(item.depth_camera_timestamp_sec)} sensor_ms=${formatScalar(item.depth_sensor_timestamp_ms)} repeat=${formatCount(item.depth_signature_repeat_count)} unique=${formatCount(item.depth_signature_unique_count)} age=${formatAge(item.depth_timestamp_age_sec)}`
      );
      if (item.depth_status_message) lines.push(`depth status: ${item.depth_status_message}`);
    }
    if (item.status_message) lines.push(`status: ${item.status_message}`);
    if (item.last_runtime_error && item.last_runtime_error !== item.last_error) {
      lines.push(`runtime: ${item.last_runtime_error}`);
    }
    if (item.last_error) lines.push(`error: ${item.last_error}`);
    return `
      <div class="device-card ${level}">
        <div class="device-title">
          <span>${escapeHtml(key)}</span>
          <span>${escapeHtml(item.status_label || level)}</span>
        </div>
        <div class="device-body mono">${escapeHtml(lines.join("\\n"))}</div>
      </div>
    `;
  }).join("");
}

function renderTactileCameraStatus(state) {
  if (!tactileCameraStatusSectionEl || !tactileCameraCardsEl) return;
  const tactileStatus = state.tactile_camera_status || {};
  const entries = Object.entries(tactileStatus).sort(([left], [right]) => left.localeCompare(right));
  tactileCameraStatusSectionEl.hidden = false;
  if (entries.length === 0) {
    tactileCameraCardsEl.innerHTML = `
      <div class="device-card off">
        <div class="device-title"><span>未配置</span><span>off</span></div>
        <div class="device-body mono">当前没有 DMTac 视触觉状态。</div>
      </div>
    `;
    return;
  }
  tactileCameraCardsEl.innerHTML = entries.map(([side, item]) => {
    const level = statusLevelClass(item.status_level);
    const mode = item.tactile_process_read ? "process" : "inline";
    const readyTotal = item.ready_total || item.device_count || "-";
    const lines = [
      `connected/tactile: ${formatFlag(item.is_connected)} / ${formatFlag(item.connect_tactile)}`,
      `devices: ${formatBrief(item.tactile_devices)}`,
      `ready: ${formatCount(item.ready_count)}/${readyTotal}`,
      `frame ids: ${formatBrief(item.tactile_frame_ids)}`,
      `age/version: ${formatAge(item.tactile_age_sec)} / ${formatCount(item.tactile_version)}`,
      `mode: ${mode} worker=${formatFlag(item.tactile_worker_alive)} receiver=${formatFlag(item.tactile_worker_receiver_alive)}`,
      `read/gate: ${formatMs(item.tactile_read_elapsed_s == null ? null : Number(item.tactile_read_elapsed_s) * 1000.0)} / ${formatMs(item.tactile_gate_wait_s == null ? null : Number(item.tactile_gate_wait_s) * 1000.0)}`,
      `hz/max_fps: ${formatScalar(item.tactile_read_hz)} / ${formatScalar(item.tactile_max_fps)} skip=${formatFlag(item.tactile_read_skipped)}`,
      `shapes: ${formatBrief(item.tactile_shapes)}`,
    ];
    if (item.status_message) lines.push(`status: ${item.status_message}`);
    if (item.telemetry_error) lines.push(`error: ${item.telemetry_error}`);
    return `
      <div class="device-card ${level}">
        <div class="device-title">
          <span>${escapeHtml(sideLabel(side))}视触觉</span>
          <span>${escapeHtml(item.status_label || level)}</span>
        </div>
        <div class="device-body mono">${escapeHtml(lines.join("\\n"))}</div>
      </div>
    `;
  }).join("");
}

function renderGripper(state) {
  const available = Boolean(state.gripper_control_available);
  gripperStatusSectionEl.hidden = !available;
  if (!available) return;

  const gripperStatus = state.gripper_status || {};
  const cards = [];
  for (const side of ["left", "right"]) {
    const item = gripperStatus[side] || {};
    if (!item.configured) continue;
    const connected = Boolean(item.is_connected);
    const hasError = Boolean(item.telemetry_error);
    const className = `gripper-card ${connected ? "connected" : ""} ${hasError ? "error" : ""}`;
    const tactileDevices = Array.isArray(item.tactile_devices) && item.tactile_devices.length > 0
      ? item.tactile_devices.join(", ")
      : "-";
    const command = item.q6_command || {};
    const inputName = item.input_source === "angle_trigger" ? "angle trigger" : "Q6";
    const q6Status = item.q6_control_available
      ? `${item.q6_valid ? "有效" : "无效"} age=${formatAge(item.q6_age_sec)} src ${item.q6_source_dim == null ? "scalar" : `dim${item.q6_source_dim}`}=${formatScalar(item.q6_source_value)} target=${formatScalar(item.q6_target_preview)}`
      : "未配置";
    const commandStatus = labelGripperCommandStatus(command.status);
    const commandTarget = command.target != null ? command.target : item.q6_filtered_target;
    const q6SerialText = [
      `port=${item.q6_port || "-"}`,
      `err=${formatCount(item.q6_reader_error_count)}`,
      `recover=${formatCount(item.q6_reader_recovery_count)}`,
      `stale=${formatCount(item.q6_stale_frames_dropped)}`,
    ].join(" ");
    const lines = [
      `gripper port: ${item.port || "-"}`,
      `${inputName}: ${q6Status}`,
      `${inputName} values: ${formatQ6Values(item.q6_values)}`,
      `${inputName} input: ${q6SerialText}`,
      `cmd: ${commandStatus} sent=${command.sent ? "yes" : "no"} target=${formatScalar(commandTarget)}`,
      `target/measured: ${formatScalar(item.commanded)} / ${formatScalar(item.measured)}`,
      `state: ${formatScalar(item.state)}  max: ${formatScalar(item.max_action)}`,
      `speed/force: ${formatScalar(item.speed)} / ${formatScalar(item.torque)}`,
      `tactile: ${tactileDevices}`,
      `tactile status: ${formatTactileStatus(item)}`,
    ];
    if (item.q6_read_error) lines.push(`${inputName} error: ${item.q6_read_error}`);
    if (command.error) lines.push(`cmd error: ${command.error}`);
    if (item.telemetry_error) lines.push(`telemetry error: ${item.telemetry_error}`);
    cards.push(`
      <div class="${className}">
        <div class="gripper-title">
          <span>${sideLabel(side)}夹爪</span>
          <span>${connected ? "已连接" : "未连接"}</span>
        </div>
        <div class="gripper-body mono">${escapeHtml(lines.join("\\n"))}</div>
      </div>
    `);
  }
  gripperCardsEl.innerHTML = cards.length > 0
    ? cards.join("")
    : '<div class="gripper-card"><div class="gripper-title"><span>未配置</span><span>-</span></div></div>';
}

function renderEpisode(state) {
  const available = Boolean(state.episode_recording_available);
  const recording = Boolean(state.episode_recording);
  const status = String(state.episode_event_status || "");
  const preparing = status === "preparing";
  const finalizing = status === "finalizing";
  const paused = status === "paused" || status === "resume_pending";
  if (episodeStartButtonEl) {
    episodeStartButtonEl.textContent = paused ? "数采暂停中" : (recording ? "数采进行中" : (preparing ? "准备中..." : "开始数采"));
    episodeStartButtonEl.classList.toggle("active", recording || preparing);
    episodeStartButtonEl.classList.toggle("recording-active", recording || preparing);
  }
  if (episodeStopButtonEl) {
    episodeStopButtonEl.classList.toggle("active", recording || preparing);
    episodeStopButtonEl.classList.toggle("stop-ready", recording || preparing);
  }
  renderEpisodeBanner(state);
  renderTaskForm(state, recording || preparing || finalizing);
  const lines = [
    available ? "episode 录制可用" : "episode 录制不可用",
    paused ? "状态: 暂停中" : (recording ? "状态: 录制中" : (preparing ? "状态: 倒计时准备中" : "状态: 未录制")),
  ];
  if (state.episode_dir) lines.push(`路径: ${state.episode_dir}`);
  if (state.dataset_root) lines.push(`保存根目录: ${state.dataset_root}`);
  if (state.collection_task_name || state.collection_task_id) {
    lines.push(`task: ${state.collection_task_name || "-"} (${state.collection_task_id || "-"})`);
  }
  if (normalizeTaskType(state.collection_task_type) === "round_trip") {
    lines.push(`来回方向: ${roundTripPhaseLabels[normalizeRoundTripPhase(state.collection_round_trip_phase)] || "-"}`);
  }
  if (state.collection_task_name_zh) lines.push(`中文名: ${state.collection_task_name_zh}`);
  if (state.episode_frame_count != null) lines.push(`帧数: ${state.episode_frame_count}`);
  if (state.episode_sample_hz != null) {
    const sampleHz = Number(state.episode_sample_hz);
    const targetHz = state.episode_target_hz == null ? null : Number(state.episode_target_hz);
    const targetText = Number.isFinite(targetHz) ? ` / 目标 ${targetHz.toFixed(1)} Hz` : "";
    lines.push(`采集频率: ${Number.isFinite(sampleHz) ? sampleHz.toFixed(1) : "-"} Hz${targetText}`);
  }
  const tactileWriteHz = state.episode_tactile_write_hz || {};
  const tactileSides = ["left", "right"].filter((side) => tactileWriteHz[side] != null);
  if (tactileSides.length > 0) {
    const sideText = tactileSides.map((side) => {
      const label = side === "left" ? "L" : (side === "right" ? "R" : side);
      const hz = Number(tactileWriteHz[side]);
      return `${label} ${Number.isFinite(hz) ? hz.toFixed(1) : "-"} Hz`;
    }).join(" / ");
    const meanHz = Number(state.episode_tactile_write_hz_mean);
    const meanText = Number.isFinite(meanHz) && tactileSides.length > 1 ? ` / 均值 ${meanHz.toFixed(1)} Hz` : "";
    lines.push(`触觉写入: ${sideText}${meanText}`);
  }
  episodeStatusEl.textContent = lines.join("\\n");
  const frequencyWarning = state.episode_frequency_warning || {};
  const warningActive = Boolean(frequencyWarning.active);
  if (episodeWarningBannerEl) {
    episodeWarningBannerEl.hidden = !warningActive;
  }
  if (episodeWarningTextEl) {
    episodeWarningTextEl.textContent = frequencyWarning.message || "采集频率低于目标值";
  }
}

function renderEpisodeBanner(state) {
  if (!episodeBannerEl || !episodeBannerStateEl || !episodeBannerCounterEl || !episodeBannerDetailEl || !episodeBannerMetaEl) {
    return;
  }
  const available = Boolean(state.episode_recording_available);
  const recording = Boolean(state.episode_recording);
  let status = String(state.episode_event_status || "");
  let paused = status === "paused" || status === "resume_pending";
  if (!available) {
    status = "unavailable";
  } else if (recording && !paused) {
    status = "recording";
  } else if (!status) {
    status = "idle";
  }
  paused = status === "paused" || status === "resume_pending";
  const statusLabels = {
    unavailable: "数采不可用",
    idle: "未开始数采",
    preparing: "3 秒后开始数采",
    recording: "● 正在数采",
    paused: "Ⅱ 数采暂停",
    resume_pending: "等待恢复写入",
    finalizing: "停止中 / 后台保存",
    saved: "已保存",
    discarded: "已停止 / 无样本",
    error: "保存失败",
  };
  const className = ["episode-banner"];
  if (["preparing", "recording", "paused", "resume_pending", "finalizing", "saved", "discarded", "error", "unavailable"].includes(status)) {
    className.push(status === "discarded" ? "finalizing" : status);
  }
  episodeBannerEl.className = className.join(" ");
  episodeBannerStateEl.textContent = statusLabels[status] || status || "未开始数采";

  const frameCount = state.episode_frame_count;
  const frameText = frameCount == null ? "-" : `${frameCount} 帧`;
  const sampleHz = state.episode_sample_hz == null ? null : Number(state.episode_sample_hz);
  const targetHz = state.episode_target_hz == null ? null : Number(state.episode_target_hz);
  const hzText = Number.isFinite(sampleHz)
    ? `${sampleHz.toFixed(1)} Hz${Number.isFinite(targetHz) ? " / " + targetHz.toFixed(1) + " Hz" : ""}`
    : "";
  episodeBannerCounterEl.textContent = hzText ? `${frameText} · ${hzText}` : frameText;

  const message = state.episode_event_message || "";
  const detailFallback = paused
    ? "当前 episode 保持打开；恢复遥操且输入稳定后会继续写入。"
    : recording
    ? "正在写入正式数据集；结束前请确认画面、触觉和机械臂状态正常。"
    : available
      ? "点击“开始数采”或按 A 开始；再次按 A 停止。"
      : "当前启动参数禁用了正式数据集和 debug episode 录制。";
  const path = state.episode_dir || state.dataset_root || "";
  episodeBannerDetailEl.textContent = [
    message || detailFallback,
    path ? `路径: ${path}` : "",
  ].filter(Boolean).join("\\n");

  const chips = [];
  if (state.collection_task_name || state.collection_task_id) {
    chips.push(["Task", state.collection_task_name || state.collection_task_id]);
  }
  if (normalizeTaskType(state.collection_task_type) === "round_trip") {
    chips.push(["方向", roundTripPhaseLabels[normalizeRoundTripPhase(state.collection_round_trip_phase)] || "-"]);
  }
  if (state.collection_task_name_zh) {
    chips.push(["中文", state.collection_task_name_zh]);
  }
  if (state.episode_last_episode_index != null) {
    chips.push(["Episode", `#${state.episode_last_episode_index}`]);
  }
  const tactileWriteHz = state.episode_tactile_write_hz || {};
  const tactileSides = ["left", "right"].filter((side) => tactileWriteHz[side] != null);
  if (tactileSides.length > 0) {
    chips.push([
      "触觉",
      tactileSides.map((side) => `${side === "left" ? "L" : "R"} ${Number(tactileWriteHz[side]).toFixed(1)}Hz`).join(" / "),
    ]);
  }
  const warning = state.episode_frequency_warning || {};
  if (warning.active) {
    chips.push(["频率", "低于目标"]);
  }
  episodeBannerMetaEl.innerHTML = chips.map(([label, value]) => (
    `<span class="episode-banner-chip">${escapeHtml(label)}: ${escapeHtml(value)}</span>`
  )).join("");
}

function renderTaskForm(state, recording) {
  window.__latestTaskRegistry = Array.isArray(state.task_registry) ? state.task_registry : [];
  const preparing = String(state.episode_event_status || "") === "preparing";
  const finalizing = String(state.episode_event_status || "") === "finalizing";
  const locked = Boolean(recording || preparing || finalizing);
  const selectedType = normalizeTaskType(
    taskTypeOverride
      ? taskTypeOverride
      : taskTypeSelectEl && document.activeElement === taskTypeSelectEl
      ? taskTypeSelectEl.value
      : state.collection_task_type
  );
  if (datasetBaseRootInputEl) {
    datasetBaseRootInputEl.disabled = locked;
    if (document.activeElement !== datasetBaseRootInputEl) {
      datasetBaseRootInputEl.value = state.dataset_base_root || "";
    }
  }
  if (datasetChooseRootButtonEl) {
    datasetChooseRootButtonEl.disabled = locked;
  }
  if (taskTypeSelectEl) {
    taskTypeSelectEl.disabled = locked;
    if (document.activeElement !== taskTypeSelectEl) {
      taskTypeSelectEl.value = selectedType;
    }
  }
  const registry = Array.isArray(state.task_registry) ? state.task_registry : [];
  if (taskSelectEl) {
    taskSelectEl.disabled = locked || !activeControls.has("task_select");
    const current = state.collection_task_key || state.collection_task_id || state.gripper_task || "";
    const existing = Array.from(taskSelectEl.options).map((option) => option.value).join("\\n");
    const filteredRegistry = registry.filter((item) => normalizeTaskType(item.task_type) === selectedType);
    const desiredItems = [{key: "", label: "自定义"}].concat(filteredRegistry.map((item) => ({
      key: item.key || item.gripper_task || "",
      label: `${item.chinese_name || item.english_name || item.key || ""} / ${item.english_name || item.key || ""}`,
    })));
      const existingLabels = Array.from(taskSelectEl.options).map((option) => option.textContent || "").join("\\n");
      const desiredValues = desiredItems.map((item) => item.key).join("\\n");
      const desiredLabels = desiredItems.map((item) => item.label).join("\\n");
    if (existing !== desiredValues || existingLabels !== desiredLabels) {
      taskSelectEl.innerHTML = desiredItems.map((item) => (
        `<option value="${escapeHtml(item.key)}">${escapeHtml(item.label)}</option>`
      )).join("");
    }
    if (document.activeElement !== taskSelectEl) {
      taskSelectEl.value = current;
    }
  }
  if (taskEnglishNameInputEl) {
    taskEnglishNameInputEl.disabled = locked;
    if (document.activeElement !== taskEnglishNameInputEl) {
      taskEnglishNameInputEl.value = selectedType === "round_trip"
        ? (state.collection_round_trip_base_name || state.collection_task_name || "").replace(/_(in|out)$/i, "")
        : state.collection_task_name || "";
    }
  }
  if (taskChineseNameInputEl) {
    taskChineseNameInputEl.disabled = locked;
    if (document.activeElement !== taskChineseNameInputEl) {
      taskChineseNameInputEl.value = state.collection_task_name_zh || "";
    }
  }
  const roundTripActive = selectedType === "round_trip";
  const phase = normalizeRoundTripPhase(roundTripPhaseOverride || state.collection_round_trip_phase);
  const nextPhase = roundTripPhaseOverride
    ? (phase === "in" ? "out" : "in")
    : normalizeRoundTripPhase(state.collection_round_trip_next_phase || (phase === "in" ? "out" : "in"));
  if (roundTripToggleButtonEl) {
    roundTripToggleButtonEl.disabled = locked || !roundTripActive || !activeControls.has("round_trip_toggle");
    roundTripToggleButtonEl.classList.toggle("active", roundTripActive);
    roundTripToggleButtonEl.textContent = roundTripActive
      ? `切换到 ${roundTripPhaseLabels[nextPhase] || "_" + nextPhase}`
      : "切换 in/out";
  }
  if (roundTripStatusEl) {
    if (roundTripActive) {
      const baseName = state.collection_round_trip_base_name || (state.collection_task_name || "").replace(/_(in|out)$/i, "");
      const baseId = state.collection_round_trip_base_id || (state.collection_task_id || "").replace(/_(in|out)$/i, "");
      roundTripStatusEl.textContent = [
        `来回类: 当前将采 ${roundTripPhaseLabels[phase] || "_" + phase}`,
        `task=${state.collection_task_name || baseName || "-"}`,
        `id=${state.collection_task_id || baseId || "-"}`,
      ].join("  ");
    } else {
      roundTripStatusEl.textContent = "一般类任务：每次采集使用同一个 task 标签。";
    }
  }
}

function renderVisualization(state) {
  const cameraAvailable = Boolean(state.camera_visualization_available);
  const cameraActive = Boolean(state.camera_visualization_active);
  const tactileAvailable = Boolean(state.tactile_visualization_available);
  const tactileActive = Boolean(state.tactile_visualization_active);
  if (cameraVisButtonEl) {
    cameraVisButtonEl.textContent = cameraActive ? "关闭图像窗口" : "打开图像窗口";
    cameraVisButtonEl.classList.toggle("active", cameraActive);
  }
  if (tactileVisButtonEl) {
    tactileVisButtonEl.textContent = tactileActive ? "关闭触觉窗口" : "打开触觉窗口";
    tactileVisButtonEl.classList.toggle("active", tactileActive);
  }
  const lines = [
    `图像: ${cameraAvailable ? (cameraActive ? "已打开" : "可打开") : "不可用"}`,
    `触觉: ${tactileAvailable ? (tactileActive ? "已打开" : "可打开") : "不可用"}`,
  ];
  if (state.visualization_status) lines.push(String(state.visualization_status));
  visualizationStatusEl.textContent = lines.join("\\n");
}

function renderPoseSnapshot(state) {
  const available = Boolean(state.pose_snapshot_available);
  const recording = Boolean(state.episode_recording);
  const preparing = String(state.episode_event_status || "") === "preparing";
  const finalizing = String(state.episode_event_status || "") === "finalizing";
  if (poseSnapshotSectionEl) {
    poseSnapshotSectionEl.hidden = !available;
  }
  if (poseSnapshotButtonEl) {
    poseSnapshotButtonEl.disabled = !available || recording || preparing || finalizing || !activeControls.has("record_pose_snapshot");
  }
  if (!poseSnapshotStatusEl) return;
  if (!available) {
    poseSnapshotStatusEl.textContent = "-";
    return;
  }
  const lines = [
    recording || preparing || finalizing ? "状态: 与正式数采互斥，当前不可抽帧" : "状态: 可抽帧",
    `数量: ${state.pose_snapshot_count == null ? 0 : state.pose_snapshot_count}`,
  ];
  if (state.pose_snapshot_dir) lines.push(`目录: ${state.pose_snapshot_dir}`);
  if (state.pose_snapshot_last_path) lines.push(`最近: ${state.pose_snapshot_last_path}`);
  if (state.pose_snapshot_status) lines.push(String(state.pose_snapshot_status));
  poseSnapshotStatusEl.textContent = lines.join("\\n");
}

const poseQualityLabels = {
  good: "good",
  warning: "warning",
  bad: "bad",
  unknown: "unknown",
};

function poseQualityCardLevel(quality) {
  const level = String(quality.level || "");
  if (["ok", "warn", "error", "off"].includes(level)) return level;
  const status = String(quality.status || "unknown");
  if (status === "bad" || quality.bad) return "error";
  if (status === "warning") return "warn";
  if (status === "good") return "ok";
  return "off";
}

function formatPoseQualityMetric(value, digits, unit) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(digits) + unit;
}

function renderPoseQuality(state) {
  const quality = state.pose_quality || {};
  const metrics = quality.metrics || {};
  const status = String(quality.status || "unknown");
  const score = quality.score == null || !Number.isFinite(Number(quality.score))
    ? "-"
    : Number(quality.score).toFixed(2);
  const reasons = Array.isArray(quality.reasons) ? quality.reasons : [];
  const badReasons = Array.isArray(quality.bad_reasons) ? quality.bad_reasons : [];
  const warningReasons = Array.isArray(quality.warning_reasons) ? quality.warning_reasons : [];
  if (poseQualityCardEl) {
    poseQualityCardEl.className = `device-card ${poseQualityCardLevel(quality)}`;
  }
  if (poseQualityStatusEl) {
    poseQualityStatusEl.textContent = poseQualityLabels[status] || status;
  }
  if (poseQualityScoreEl) {
    poseQualityScoreEl.textContent = `score ${score}`;
  }
  if (!poseQualityDetailEl) return;
  const lines = [];
  lines.push(`available: ${quality.available ? "yes" : "no"}`);
  if (badReasons.length) lines.push(`bad: ${badReasons.join(", ")}`);
  if (warningReasons.length) lines.push(`warning: ${warningReasons.join(", ")}`);
  if (!badReasons.length && !warningReasons.length) {
    lines.push(`reasons: ${reasons.length ? reasons.join(", ") : "none"}`);
  }
  lines.push(`source: ${quality.source || "-"}`);
  lines.push(
    `collision: ${metrics.collision_avoidance_status || "-"} ` +
    `dist=${formatPoseQualityMetric(metrics.collision_distance_m, 3, "m")} ` +
    `min=${formatPoseQualityMetric(metrics.collision_min_distance_m, 3, "m")} ` +
    `margin=${formatPoseQualityMetric(metrics.collision_margin_m, 3, "m")}`
  );
  lines.push(
    `ik: jump=${metrics.ik_jump_gate_status || "-"} branch=${metrics.ik_branch_candidate || "-"} ` +
    `err=${formatPoseQualityMetric(metrics.ik_translation_error_m, 3, "m")}/` +
    `${formatPoseQualityMetric(metrics.ik_rotation_error_deg, 1, "deg")}`
  );
  lines.push(
    `joint: nearest_margin=${formatPoseQualityMetric(metrics.joint_nearest_margin_deg, 1, "deg")} ` +
    `near_limit=${metrics.joint_near_limit ? "yes" : "no"}`
  );
  if (metrics.execution_step_limited || metrics.ik_time_budget_exceeded) {
    lines.push(
      `runtime: step_limited=${metrics.execution_step_limited ? "yes" : "no"} ` +
      `time_budget_exceeded=${metrics.ik_time_budget_exceeded ? "yes" : "no"}`
    );
  }
  poseQualityDetailEl.textContent = lines.join("\\n");
}

function renderIkWarning(state) {
  const warning = state.ik_warning || {};
  const active = Boolean(warning.active);
  ikWarningBannerEl.hidden = !active;
  if (!active) {
    ikWarningTextEl.textContent = "-";
    return;
  }
  ikWarningTextEl.textContent = warning.message || "IK warning active";
}

function renderIkTiming(state) {
  const enabled = Boolean(state.debug_ik_timing);
  ikTimingSectionEl.hidden = !enabled;
  if (!enabled) return;
  const timing = state.ik_timing || {};
  const available = Boolean(timing.available);
  const warning = state.ik_warning || timing.ik_warning || {};
  const warningActive = Boolean(warning.active);
  ikTimingStatusEl.classList.toggle("ik-warning", warningActive);
  ikTimingTotalMetricEl.classList.toggle("bad-metric", Boolean(timing.time_budget_exceeded));
  ikTimingTotalMetricEl.classList.toggle("warn-metric", !available);
  const warningText = warningActive ? `IK WARNING ${warning.message || ""}`.trim() : "";
  if (!available) {
    ikTimingTotalEl.textContent = "-";
    ikTimingSolveEl.textContent = "-";
    ikTimingCollisionEl.textContent = "-";
    ikTimingCallsEl.textContent = "-";
    ikTimingStatusEl.textContent = warningText || timing.message || "暂无 IK timing 元数据；若使用 --no-collision，双臂 IK 不在 retargeter 中执行。";
    return;
  }
  const budget = timing.time_budget_ms == null || Number(timing.time_budget_ms) <= 0
    ? "无预算"
    : formatMs(timing.time_budget_ms);
  ikTimingTotalEl.textContent = `${formatMs(timing.frame_total_ms)} / ${budget}`;
  const hasProviderBreakdown = timing.provider_elapsed_ms != null || timing.non_provider_ms != null;
  ikTimingSolveEl.textContent = hasProviderBreakdown
    ? `provider ${formatMs(timing.provider_elapsed_ms)} / 其他 ${formatMs(timing.non_provider_ms)}`
    : `${formatMs(timing.solve_total_ms)} / ${formatMs(timing.solve_qp_ms)}`;
  const hasCheckBreakdown = timing.fk_ms != null || timing.singularity_ms != null || timing.collision_ms != null;
  ikTimingCollisionEl.textContent = hasCheckBreakdown
    ? `FK ${formatMs(timing.fk_ms)} / 奇异 ${formatMs(timing.singularity_ms)} / 碰撞 ${formatMs(timing.collision_ms)}`
    : formatMs(timing.min_distance_total_ms);
  const providerCalls = Number(timing.provider_call_count || 0);
  const candidateEvals = Number(timing.candidate_eval_count || 0);
  ikTimingCallsEl.textContent = providerCalls > 0 || candidateEvals > 0
    ? `provider ${providerCalls} / candidate ${candidateEvals}`
    : `solve ${timing.solve_calls || 0} / dist ${timing.min_distance_calls || 0}`;
  const budgetText = timing.time_budget_exceeded ? "超过预算" : "预算内";
  ikTimingStatusEl.textContent = [
    warningText,
    `source=${timing.source || "-"}`,
    `status=${timing.status || "-"}`,
    `jump=${timing.jump_gate_status || "-"}`,
    `branch=${timing.branch_candidate || "-"}`,
    state.debug_ik_logging ? `记录=${state.ik_debug_log_path || "teleop_records/ik_debug"}` : "",
    budgetText,
    timing.message || "",
  ].filter(Boolean).join("  ");
}

function renderMessages(messages) {
  const wasNearBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 24;
  const previousScrollTop = messagesEl.scrollTop;
  if (!messages || messages.length === 0) {
    messagesEl.innerHTML = '<div class="message">等待更新...</div>';
    if (wasNearBottom) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } else {
      messagesEl.scrollTop = previousScrollTop;
    }
    return;
  }
  messagesEl.innerHTML = messages.map((message) => {
    const tag = message.tag || "LOG";
    const text = message.message || "";
    return `<div class="message"><span class="tag">[${tag}]</span>${text}</div>`;
  }).join("");
  if (wasNearBottom) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else {
    messagesEl.scrollTop = previousScrollTop;
  }
}

async function refresh() {
  try {
    const response = await fetch("/state.json?ts=" + Date.now(), {cache: "no-store"});
    if (!response.ok) throw new Error("state fetch failed: " + response.status);
    const state = await response.json();
    latestState = state || {};
    if (taskTypeOverride && normalizeTaskType(state.collection_task_type) === taskTypeOverride) {
      taskTypeOverride = null;
    }
    if (
      roundTripPhaseOverride &&
      normalizeTaskType(state.collection_task_type) === "round_trip" &&
      normalizeRoundTripPhase(state.collection_round_trip_phase) === roundTripPhaseOverride
    ) {
      roundTripPhaseOverride = null;
    }
    stageEl.textContent = labelFor(stageLabels, state.stage || "Starting...");
    pillsEl.innerHTML = formatPills(state);
    selectedSideEl.textContent = labelFor(sideLabels, state.selected_side);
    selectedSideMetricEl.classList.toggle("selected-side", Boolean(state.selected_side));
    frameVizModeEl.textContent = labelFor(frameVizLabels, state.frame_viz_mode);
    virtualDeskYawEl.textContent = state.virtual_desk_post_yaw_deg !== undefined ? String(state.virtual_desk_post_yaw_deg) + " deg" : "-";
    controlHintEl.textContent = state.control_hint || "-";
    captureStageEl.textContent = labelFor(stageLabels, state.capture_stage);
    captureSamplesEl.textContent = formatCapture(state);
    captureValidEl.textContent = formatValid(state);
    captureStatusEl.innerHTML = formatCaptureStatus(state);
    renderCaptureMode(state);
    renderRuntimeHomeDiagnostic(state);
    updateButtons(state);
    updateShortcutHints(state);
    renderViveCalibrationProfile(state);
    renderMovementMultiple(state);
    renderGripperForce(state);
    renderFixArm(state);
    renderInitArm(state);
    renderRawPoses(state);
    renderEpisode(state);
    renderVisualization(state);
    renderPoseSnapshot(state);
    renderGripper(state);
    renderCameraSelfCheck(state);
    renderCameraStatus(state);
    renderTactileCameraStatus(state);
    renderIkWarning(state);
    renderPoseQuality(state);
    renderIkTiming(state);
    renderMessages(state.messages || []);
    window.__calibrationWebUiModernRefreshOk = true;
    clearRefreshWarning();
    if (state.shutdown) {
      showShutdown(state.shutdown_message || "服务已退出，这个网页已经失效。");
    }
  } catch (error) {
    showRefreshWarning(error);
  }
}

refresh();
refreshTimer = setInterval(refresh, 500);
</script>
<script src="/calibration-web-ui-fallback.js"></script>
</body>
</html>
"""


_FALLBACK_SCRIPT = r"""
(function () {
  if (window.__calibrationWebUiExternalFallbackLoaded) return;
  window.__calibrationWebUiExternalFallbackLoaded = true;
  var fallbackLatestState = {};
  var fallbackActiveControls = {};

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
    var i;
    for (i = 0; i < controls.length; i += 1) active[String(controls[i])] = true;
    fallbackActiveControls = active;
    var shortcutC = byId("shortcutCLabel");
    if (shortcutC) {
      if (state.fix_arm_available && state.teleop_running) {
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
      if (state.init_arm_available && state.teleop_running) {
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
            : initLabel + "归位";
      }
    }
  }

  function renderEpisodeStateFallback(state) {
    var available = Boolean(state.episode_recording_available);
    var recording = Boolean(state.episode_recording);
    var preparing = String(state.episode_event_status || "") === "preparing";
    var finalizing = String(state.episode_event_status || "") === "finalizing";
    var startButton = byId("episodeStartButton");
    var stopButton = byId("episodeStopButton");
    var statusEl = byId("episodeStatus");
    var warning = state.episode_frequency_warning || {};
    var warningBanner = byId("episodeWarningBanner");
    var warningText = byId("episodeWarningText");
    var status = String(state.episode_event_status || "");
    var paused = status === "paused" || status === "resume_pending";
    var statusLabels = {
      unavailable: "数采不可用",
      idle: "未开始数采",
      preparing: "3 秒后开始数采",
      recording: "正在数采",
      paused: "数采暂停",
      resume_pending: "等待恢复写入",
      finalizing: "停止中 / 后台保存",
      saved: "已保存",
      discarded: "已停止 / 无样本",
      error: "保存失败"
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
      startButton.textContent = paused ? "数采暂停中" : (recording ? "数采进行中" : (preparing ? "准备中..." : "开始数采"));
      startButton.disabled = !fallbackActiveControls.episode_start || recording || preparing || finalizing;
    }
    if (stopButton) {
      stopButton.disabled = !fallbackActiveControls.episode_stop || (!recording && !preparing);
    }
    var locked = recording || preparing || finalizing;
    var fields = ["datasetBaseRootInput", "datasetChooseRootButton", "taskTypeSelect", "taskSelect", "taskEnglishNameInput", "taskChineseNameInput"];
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
        paused ? "状态: 暂停中" : (recording ? "状态: 录制中" : (preparing ? "状态: 倒计时准备中" : "状态: 未录制"))
      ];
      if (state.episode_dir) lines.push("路径: " + state.episode_dir);
      if (state.dataset_root) lines.push("保存根目录: " + state.dataset_root);
      if (state.episode_frame_count != null) lines.push("帧数: " + state.episode_frame_count);
      statusEl.textContent = lines.join("\n");
    }
    if (warningBanner) warningBanner.hidden = !Boolean(warning.active);
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
    setText("stage", labelStage(state.stage || "Starting..."));
    setText("selectedSide", state.selected_side === "left" ? "左手" : state.selected_side === "right" ? "右手" : state.selected_side);
    setText("frameVizMode", state.frame_viz_mode);
    setText("virtualDeskYaw", state.virtual_desk_post_yaw_deg != null ? String(state.virtual_desk_post_yaw_deg) + " deg" : "-");
    setText("controlHint", state.control_hint || "-");
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
    renderCameraSelfCheckFallback(state);
    renderDeviceStatus("cameraCards", state.camera_status, "当前没有 RealSense camera_sources。", "");
    renderDeviceStatus("tactileCameraCards", state.tactile_camera_status, "当前没有 DMTac 视触觉状态。", "");
    renderMessages(state.messages || []);
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
    return {
      dataset_base_root: byId("datasetBaseRootInput") ? byId("datasetBaseRootInput").value : "",
      task_type: byId("taskTypeSelect") ? byId("taskTypeSelect").value : normalizeTaskTypeFallback(fallbackLatestState.collection_task_type),
      task_key: byId("taskSelect") ? byId("taskSelect").value : "",
      task_english_name: byId("taskEnglishNameInput") ? byId("taskEnglishNameInput").value : "",
      task_chinese_name: byId("taskChineseNameInput") ? byId("taskChineseNameInput").value : "",
      round_trip_phase: normalizeRoundTripPhaseFallback(fallbackLatestState.collection_round_trip_phase)
    };
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
        if (xhr.readyState === 4 && xhr.status >= 400) {
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
      eventName = (state.episode_recording || String(state.episode_event_status || "") === "preparing") ? "episode_stop" : "episode_start";
      if (eventName === "episode_start") payload = collectEpisodeStartPayloadFallback();
    } else if (key === "b") {
      eventName = state.teleop_running ? "h" : "p";
    } else if (key === "c") {
      if (state.fix_arm_available && state.teleop_running && fallbackActiveControls["fix_arm_toggle"]) {
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
        postEvent(sendEventName, eventName === "episode_start" ? collectEpisodeStartPayloadFallback() : null);
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
"""


_PAGE_VERSION = "real_teleop_gui_init_arm_v1"


def _html_text(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return html.escape(str(value), quote=False)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe_value(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_value(item())
        except Exception:
            pass
    return str(value)


def _json_response_payload(value: Any) -> bytes:
    safe_value = _json_safe_value(value)
    return json.dumps(safe_value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _encode_http_event_payload(payload: Mapping[str, Any]) -> str:
    """Normalize events received from the browser before queueing them.

    Task selection sends its task metadata as sibling JSON fields.  The
    runtime event consumer expects that metadata in the event suffix, just as
    it does for episode start, so preserve the same wire format for both
    browser events.
    """
    event = str(payload.get("event", "")).strip()
    if event in {"episode_start", "task_select"}:
        details = {
            key: str(payload.get(key, "") or "").strip()
            for key in (
                "dataset_base_root",
                "task_type",
                "task_key",
                "task_english_name",
                "task_chinese_name",
                "round_trip_phase",
            )
        }
        event = event + ":" + json.dumps(
            details,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return event


def _choose_directory_payload(initial_dir: str = "") -> dict[str, Any]:
    initial_text = os.path.expandvars(str(initial_dir or "")).strip()
    initial_path = Path(initial_text).expanduser() if initial_text else Path.home()
    if not initial_path.exists():
        initial_path = initial_path.parent if initial_path.parent.exists() else Path.home()
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(
            parent=root,
            title="选择数据集保存根目录",
            initialdir=str(initial_path),
            mustexist=False,
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001 - GUI folder dialog is optional.
        return {
            "ok": False,
            "path": "",
            "message": f"无法打开文件夹选择器，请手动填写路径: {type(exc).__name__}: {exc}",
        }
    if not selected:
        return {"ok": False, "path": "", "message": "已取消文件夹选择。"}
    return {"ok": True, "path": str(Path(selected).expanduser()), "message": ""}


def _label_stage(value: Any) -> str:
    labels = {
        "Starting...": "启动中...",
        "RM65 真机遥操准备": "RM65 真机遥操准备",
        "Home Capture": "起点采集",
        "HomeCapture": "起点采集",
        "runtime_home": "runtime home",
        "teleop": "遥操",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def _label_phase(value: Any) -> str:
    labels = {
        "idle": "空闲",
        "ready": "准备就绪",
        "preparing": "准备倒计时",
        "capturing": "正在采集",
        "waiting": "等待样本",
        "teleop": "遥操中",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def _label_side(value: Any) -> str:
    labels = {"left": "左手", "right": "右手"}
    text = str(value or "")
    return labels.get(text, text or "-")


def _label_frame_viz(value: Any) -> str:
    labels = {"calibrated": "校准后", "raw": "原始", "both": "全部"}
    text = str(value or "")
    return labels.get(text, text or "-")


_RUNTIME_HOME_LIMIT_FAILURE_RE = re.compile(
    r"\b(?P<arm>left_arm|right_arm)\s+local\s+qpos\s+joint\s+"
    r"(?P<joint>\d+)\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))rad\s+"
    r"exceeds\s+absolute\s+limits\s+\[\s*"
    r"(?P<lower>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<upper>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]rad",
    re.IGNORECASE,
)


def _runtime_home_diagnostic_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Derive a display-only runtime-home status from the already published state."""

    hidden = {"visible": False, "level": "", "title": "", "detail": ""}
    control_hint = str(snapshot.get("control_hint") or "").strip()
    if "runtime home 标定失败" in control_hint:
        failure_line = next(
            (line.strip() for line in control_hint.splitlines() if line.strip()),
            control_hint,
        )
        limit_match = _RUNTIME_HOME_LIMIT_FAILURE_RE.search(control_hint)
        if limit_match is not None:
            groups = limit_match.groupdict()
            arm_label = "左臂" if groups["arm"] == "left_arm" else "右臂"
            return {
                "visible": True,
                "level": "error",
                "title": f"Runtime Home 标定失败：{arm_label}关节 {groups['joint']} 到达绝对限位",
                "detail": "\n".join(
                    (
                        f"当前位置：{groups['value']} rad",
                        f"允许范围：[{groups['lower']}, {groups['upper']}] rad",
                        "本次 Runtime Home 未完成，界面保持在重新标定等待状态。",
                        "请按现场安全流程检查关节姿态/限位后，点击“开始遥操”重新采集 Runtime Home。",
                        f"后端错误：{failure_line}",
                    )
                ),
            }
        return {
            "visible": True,
            "level": "error",
            "title": "Runtime Home 标定失败",
            "detail": "\n".join(
                (
                    failure_line or "后端未提供失败详情。",
                    "本次 Runtime Home 未完成，界面保持在重新标定等待状态。",
                    "请检查 VIVE tracker、机械臂状态和限位后，重新发起标定。",
                )
            ),
        }

    if (
        str(snapshot.get("capture_stage") or "") != "runtime_home"
        or str(snapshot.get("capture_phase") or "") != "capturing"
    ):
        return hidden

    min_samples = int(snapshot.get("capture_min_samples") or 0)
    left_samples = snapshot.get("capture_left_samples")
    right_samples = snapshot.get("capture_right_samples")
    samples = snapshot.get("capture_samples")
    progress_parts: list[str] = []
    if left_samples is not None:
        progress_parts.append(f"左臂 {int(left_samples or 0)}/{min_samples}")
    if right_samples is not None:
        progress_parts.append(f"右臂 {int(right_samples or 0)}/{min_samples}")
    if not progress_parts and samples is not None:
        progress_parts.append(f"{int(samples or 0)}/{min_samples}")
    validity_parts: list[str] = []
    if snapshot.get("capture_left_valid") is not None:
        validity_parts.append(
            f"左臂 {'有效' if bool(snapshot.get('capture_left_valid')) else '等待有效 tracker'}"
        )
    if snapshot.get("capture_right_valid") is not None:
        validity_parts.append(
            f"右臂 {'有效' if bool(snapshot.get('capture_right_valid')) else '等待有效 tracker'}"
        )
    if not validity_parts and snapshot.get("capture_valid") is not None:
        validity_parts.append(
            "tracker 有效" if bool(snapshot.get("capture_valid")) else "等待有效 tracker"
        )
    elapsed = snapshot.get("capture_elapsed_sec")
    timeout = snapshot.get("capture_duration_sec")
    timing = ""
    if elapsed is not None and timeout is not None:
        try:
            timing = f"采样时间：{float(elapsed):.1f}/{float(timeout):.1f}s"
        except (TypeError, ValueError):
            timing = ""
    detail_parts = [
        f"样本进度：{'；'.join(progress_parts) if progress_parts else '正在等待第一帧'}",
    ]
    if validity_parts:
        detail_parts.append(f"输入状态：{'；'.join(validity_parts)}")
    if timing:
        detail_parts.append(timing)
    if bool(snapshot.get("capture_waiting_for_samples")):
        detail_parts.append("正在等待新的有效 VIVE tracker 样本；无效帧不会计入进度。")
    else:
        detail_parts.append("请保持当前 VIVE 与机械臂 Home 姿态稳定，直到左右样本均采满。")
    return {
        "visible": True,
        "level": "capturing",
        "title": "Runtime Home 正在采样",
        "detail": "\n".join(detail_parts),
    }


def _initial_capture_phase(snapshot: dict[str, Any]) -> str:
    if bool(snapshot.get("teleop_running")):
        return "teleop"
    if snapshot.get("capture_stage"):
        return str(snapshot.get("capture_phase") or "idle")
    return "idle"


def _initial_capture_progress_and_detail(snapshot: dict[str, Any]) -> tuple[float, str]:
    phase = _initial_capture_phase(snapshot)
    capture_stage = str(snapshot.get("capture_stage") or "")
    sample_label = (
        "Home 样本"
        if capture_stage == "runtime_home"
        else "力控预热"
        if capture_stage == "force_control_warmup"
        else "标定样本"
    )
    if phase == "ready":
        return 0.0, "按“开始采集”后会先进入准备倒计时"
    if phase == "preparing":
        total = float(snapshot.get("capture_prep_total_sec") or 0.0)
        remaining = float(snapshot.get("capture_prep_remaining_sec") or 0.0)
        progress = (1.0 - remaining / total) * 100.0 if total > 0.0 else 100.0
        if capture_stage == "force_control_warmup":
            return progress, f"正在预热 RM65 力控，预计剩余 {remaining:.1f}s"
        return progress, f"倒计时 {remaining:.1f}s，当前还未写入{sample_label}"
    if phase == "capturing":
        min_samples = int(snapshot.get("capture_min_samples") or 0)
        left_samples = snapshot.get("capture_left_samples")
        right_samples = snapshot.get("capture_right_samples")
        single_samples = snapshot.get("capture_samples")
        if min_samples > 0 and (left_samples is not None or right_samples is not None):
            left = int(left_samples or 0)
            right = int(right_samples or 0)
            progress = min(left, right) / float(min_samples) * 100.0
            return progress, f"正在采集 {sample_label}，左 {left}/{min_samples}，右 {right}/{min_samples}"
        if min_samples > 0 and single_samples is not None:
            samples = int(single_samples or 0)
            progress = samples / float(min_samples) * 100.0
            return progress, f"正在采集 {sample_label}，{samples}/{min_samples}"
        duration = float(snapshot.get("capture_duration_sec") or 0.0)
        elapsed = float(snapshot.get("capture_elapsed_sec") or 0.0)
        progress = elapsed / duration * 100.0 if duration > 0.0 else 0.0
        return progress, f"正在写入样本，已采集 {elapsed:.1f} / {duration:.1f}s"
    if phase == "waiting":
        return 100.0, "等待 VIVE/Q6/相机样本进入采集窗口"
    if phase == "teleop":
        detail = "真机遥操中，Q6 夹爪控制已启用" if bool(snapshot.get("gripper_control_enabled")) else "真机遥操中"
        return 100.0, detail
    if snapshot.get("capture_stage"):
        return 0.0, "等待流程进入采集阶段"
    return 0.0, "等待后端发布实时状态"


def _initial_capture_samples(snapshot: dict[str, Any]) -> str:
    left = snapshot.get("capture_left_samples")
    right = snapshot.get("capture_right_samples")
    min_samples = int(snapshot.get("capture_min_samples") or 0)
    if left is not None or right is not None:
        return f"左 {int(left or 0)}/{min_samples}  右 {int(right or 0)}/{min_samples}"
    samples = snapshot.get("capture_samples")
    if samples is not None:
        return f"{int(samples or 0)}/{min_samples}"
    return "-"


def _initial_capture_valid(snapshot: dict[str, Any]) -> str:
    left = snapshot.get("capture_left_valid")
    right = snapshot.get("capture_right_valid")
    if left is not None or right is not None:
        return f"左 {'有效' if bool(left) else '缺失'} / 右 {'有效' if bool(right) else '缺失'}"
    valid = snapshot.get("capture_valid")
    if valid is not None:
        return "有效" if bool(valid) else "缺失"
    return "-"


def _format_initial_vector(values: Any, *, digits: int) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return "-"
    parts: list[str] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            parts.append("nan" if value is None else str(value))
        else:
            parts.append(f"{numeric:.{digits}f}")
    return ", ".join(parts)


def _initial_raw_pose_text(pose: Any) -> tuple[str, str]:
    if isinstance(pose, dict) and pose.get("enabled") is False:
        return "未启用", "当前 logical side 未启用"
    if not isinstance(pose, dict) or not pose:
        return "无数据", "未收到 channel"
    status = str(pose.get("status_label") or ("有效" if bool(pose.get("valid")) else "无效"))
    age = pose.get("age_sec")
    try:
        age_text = "-" if age is None else f"{float(age):.2f} s"
    except (TypeError, ValueError):
        age_text = str(age)
    text = "\n".join(
        (
            f"tracker: {pose.get('tracker_name') or '-'}",
            f"role:    {pose.get('tracker_role') or '-'}",
            f"seq:     {'-' if pose.get('seq') is None else pose.get('seq')}",
            f"age:     {age_text}",
            f"xyz(m):  {_format_initial_vector(pose.get('xyz_m'), digits=4)}",
            f"rpy(deg):{_format_initial_vector(pose.get('rpy_deg'), digits=1)}",
        )
    )
    return status, text


def _render_initial_html(snapshot: dict[str, Any]) -> str:
    phase = _initial_capture_phase(snapshot)
    allowed_phase = phase if phase in {"idle", "ready", "preparing", "capturing", "waiting", "teleop"} else "idle"
    progress, capture_detail = _initial_capture_progress_and_detail(snapshot)
    progress = max(0.0, min(100.0, progress))
    raw_poses = snapshot.get("raw_tracker_poses") if isinstance(snapshot.get("raw_tracker_poses"), dict) else {}
    left_raw_status, left_raw_pose = _initial_raw_pose_text(raw_poses.get("left") if isinstance(raw_poses, dict) else None)
    right_raw_status, right_raw_pose = _initial_raw_pose_text(raw_poses.get("right") if isinstance(raw_poses, dict) else None)
    yaw = snapshot.get("virtual_desk_post_yaw_deg")
    yaw_text = "-" if yaw is None else f"{yaw} deg"
    runtime_home_diagnostic = snapshot.get("runtime_home_diagnostic")
    if not isinstance(runtime_home_diagnostic, dict):
        runtime_home_diagnostic = _runtime_home_diagnostic_payload(snapshot)
    diagnostic_visible = bool(runtime_home_diagnostic.get("visible"))
    diagnostic_level = str(runtime_home_diagnostic.get("level") or "")
    if diagnostic_level not in {"capturing", "error"}:
        diagnostic_level = ""
    replacements = {
        "__PAGE_TITLE__": (
            "DexTeleop UMI 数采"
            if str(snapshot.get("ui_mode")) == "umi_collection"
            else "VIVE -> RM65 标定"
        ),
        "__INITIAL_STAGE__": _label_stage(snapshot.get("stage")),
        "__INITIAL_SELECTED_SIDE__": _label_side(snapshot.get("selected_side")),
        "__INITIAL_FRAME_VIZ_MODE__": _label_frame_viz(snapshot.get("frame_viz_mode")),
        "__INITIAL_VIRTUAL_DESK_YAW__": yaw_text,
        "__INITIAL_CONTROL_HINT__": str(snapshot.get("control_hint") or "-"),
        "__INITIAL_CAPTURE_PHASE_CLASS__": allowed_phase,
        "__INITIAL_CAPTURE_LABEL__": _label_phase(phase),
        "__INITIAL_CAPTURE_DETAIL__": capture_detail,
        "__INITIAL_CAPTURE_PROGRESS__": f"{progress:.1f}",
        "__INITIAL_CAPTURE_STAGE__": _label_stage(snapshot.get("capture_stage")),
        "__INITIAL_CAPTURE_SAMPLES__": _initial_capture_samples(snapshot),
        "__INITIAL_CAPTURE_VALID__": _initial_capture_valid(snapshot),
        "__INITIAL_CAPTURE_STATUS__": "遥操中" if bool(snapshot.get("teleop_running")) else _label_phase(phase),
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_LEVEL__": diagnostic_level,
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_HIDDEN__": "" if diagnostic_visible else "hidden",
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_TITLE__": str(runtime_home_diagnostic.get("title") or ""),
        "__INITIAL_RUNTIME_HOME_DIAGNOSTIC_DETAIL__": str(runtime_home_diagnostic.get("detail") or ""),
        "__INITIAL_LEFT_RAW_STATUS__": left_raw_status,
        "__INITIAL_LEFT_RAW_POSE__": left_raw_pose,
        "__INITIAL_RIGHT_RAW_STATUS__": right_raw_status,
        "__INITIAL_RIGHT_RAW_POSE__": right_raw_pose,
    }
    page = _HTML_PAGE
    for token, value in replacements.items():
        escaped = _html_text(value)
        page = page.replace(token, escaped)
    return page


@dataclass
class CalibrationWebUiState:
    stage: str = "Starting..."
    selected_side: str = "right"
    frame_viz_mode: str = "calibrated"
    virtual_desk_post_yaw_deg: float = 180.0
    renderer_ready: bool = False
    ui_mode: str = "web"
    host: str = ""
    port: int = 0
    load_profile: str = ""
    show_tracker_frame_viz: bool = False
    home_ready: bool = False
    teleop_running: bool = False
    raw_tracker_poses: dict[str, Any] = field(default_factory=dict)
    debug_ik_timing: bool = False
    ik_timing: dict[str, Any] = field(default_factory=dict)
    ik_warning: dict[str, Any] = field(default_factory=dict)
    pose_quality: dict[str, Any] = field(default_factory=dict)
    debug_ik_logging: bool = False
    ik_debug_log_path: str = ""
    gripper_control_available: bool = False
    gripper_q6_available: bool = False
    gripper_control_enabled: bool = False
    gripper_status: dict[str, Any] = field(default_factory=dict)
    camera_status: dict[str, Any] = field(default_factory=dict)
    camera_self_check: dict[str, Any] = field(default_factory=dict)
    tactile_camera_status: dict[str, Any] = field(default_factory=dict)
    episode_recording_available: bool = False
    episode_recording: bool = False
    episode_dir: str = ""
    episode_frame_count: int | None = None
    episode_sample_hz: float | None = None
    episode_target_hz: float | None = None
    episode_tactile_write_hz: dict[str, Any] = field(default_factory=dict)
    episode_tactile_write_hz_mean: float | None = None
    episode_tactile_frame_counts: dict[str, Any] = field(default_factory=dict)
    episode_frequency_warning: dict[str, Any] = field(default_factory=dict)
    episode_event_status: str = "unavailable"
    episode_event_message: str = ""
    episode_last_episode_index: int | None = None
    episode_last_frame_count: int | None = None
    dataset_base_root: str = ""
    dataset_root: str = ""
    dataset_dir_name: str = ""
    collection_task_id: str = ""
    collection_task_key: str = ""
    collection_task_name: str = ""
    collection_task_name_zh: str = ""
    collection_task_description: str = ""
    collection_task_type: str = "general"
    collection_round_trip_phase: str = "in"
    collection_round_trip_next_phase: str = "out"
    collection_round_trip_base_name: str = ""
    collection_round_trip_base_id: str = ""
    collection_round_trip_active: bool = False
    gripper_task: str = ""
    task_registry: list[dict[str, Any]] = field(default_factory=list)
    camera_visualization_available: bool = False
    camera_visualization_active: bool = False
    tactile_visualization_available: bool = False
    tactile_visualization_active: bool = False
    visualization_status: str = ""
    vive_calibration_profile_path: str = ""
    vive_calibration_profile_status: str = ""
    vive_calibration_profile_ready: bool = False
    vive_calibration_profile_loaded: bool = False
    movement_multiple: float = 1.0
    movement_multiple_min: float = 0.05
    movement_multiple_max: float = 10.0
    movement_multiple_step: float = 0.1
    fix_arm_available: bool = False
    fix_arm_selected_side: str = ""
    fix_arm_fixed_side: str = ""
    fix_arm_active: bool = False
    fix_arm_recalibrating: bool = False
    fix_arm_status: str = ""
    fix_arm_sample_count: int | None = None
    fix_arm_min_samples: int | None = None
    special_arm_control_mode: str = "manual"
    init_arm_available: bool = False
    init_arm_selected_side: str = ""
    init_arm_active_side: str = ""
    init_arm_active: bool = False
    init_arm_recalibrating: bool = False
    init_arm_status: str = ""
    init_arm_sample_count: int | None = None
    init_arm_min_samples: int | None = None
    init_arm_profile_path: str = ""
    init_arm_arrival_status: str = "inactive"
    init_arm_arrival_label: str = ""
    init_arm_arrival_reason: str = ""
    init_arm_target_position_error_m: float | None = None
    init_arm_target_rotation_error_deg: float | None = None
    init_arm_stability_dwell_sec: float | None = None
    pose_snapshot_available: bool = False
    pose_snapshot_dir: str = ""
    pose_snapshot_count: int = 0
    pose_snapshot_last_path: str = ""
    pose_snapshot_status: str = ""
    active_controls: list[str] = field(default_factory=lambda: ["q"])
    control_hint: str = ""
    capture_stage: str = ""
    capture_phase: str = "idle"
    capturing: bool = False
    capture_channel: str = ""
    capture_samples: int | None = None
    capture_left_samples: int | None = None
    capture_right_samples: int | None = None
    capture_min_samples: int | None = None
    capture_valid: bool | None = None
    capture_left_valid: bool | None = None
    capture_right_valid: bool | None = None
    capture_elapsed_sec: float | None = None
    capture_duration_sec: float | None = None
    capture_prep_remaining_sec: float | None = None
    capture_prep_total_sec: float | None = None
    capture_waiting_for_samples: bool = False
    shutdown: bool = False
    shutdown_message: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)


class CalibrationWebUi:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._open_browser = bool(open_browser)
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._events: queue.Queue[str] = queue.Queue()
        self._state_lock = threading.Lock()
        self._state = CalibrationWebUiState()

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}/"
        address = self._server.server_address
        host = address[0]
        port = address[1]
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return

        ui = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/state.json"):
                    payload = _json_response_payload(ui.snapshot())
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path.startswith("/choose-directory"):
                    parsed = urlparse(self.path)
                    params = parse_qs(parsed.query)
                    initial = params.get("initial", [""])[0]
                    payload = _json_response_payload(_choose_directory_payload(initial))
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path.startswith("/calibration-web-ui-fallback.js"):
                    payload = _FALLBACK_SCRIPT.encode("utf-8")
                    self._send(200, "application/javascript; charset=utf-8", payload)
                    return
                if self.path.startswith("/debug.json"):
                    snapshot = ui.snapshot()
                    payload = _json_response_payload(
                        {
                            "page_version": _PAGE_VERSION,
                            "html_contains_legacy_poll": "__calibrationWebUiLegacyPollActive" in _HTML_PAGE,
                            "html_contains_modern_poll": "__calibrationWebUiModernScriptLoaded" in _HTML_PAGE,
                            "html_contains_external_fallback": "calibration-web-ui-fallback.js" in _HTML_PAGE,
                            "fallback_script_path": "/calibration-web-ui-fallback.js",
                            "json_sanitized": True,
                            "snapshot": snapshot,
                        }
                    )
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if self.path in {"/", "/index.html"}:
                    payload = _render_initial_html(ui.snapshot()).encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", payload)
                    return
                self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/event":
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    event = _encode_http_event_payload(payload)
                except Exception:
                    self._send(400, "text/plain; charset=utf-8", b"invalid json")
                    return
                if not event:
                    self._send(400, "text/plain; charset=utf-8", b"missing event")
                    return
                if not ui.push_event(event):
                    self._send(409, "text/plain; charset=utf-8", b"inactive event")
                    return
                self._send(204, "text/plain; charset=utf-8", b"")

            def log_message(self, fmt: str, *args: Any) -> None:
                del fmt, args

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        if self._open_browser:
            try:
                webbrowser.open(self.url, new=1, autoraise=True)
            except Exception:
                pass

    def close(self) -> None:
        if self._server is None:
            return
        self.mark_shutdown("服务已退出，这个网页已经失效。")
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._server_thread = None

    def push_event(self, event: str) -> bool:
        event_name = str(event)
        event_control = event_name.split(":", 1)[0]
        if ":" in event_name and event_control not in {
            "movement_multiple_set",
            "gripper_force_set",
            "episode_start",
            "task_select",
            "fix_arm_select",
            "init_arm_select",
        }:
            return False
        with self._state_lock:
            active_controls = set(self._state.active_controls)
            accepted = event_name in active_controls or event_control in active_controls
            if (
                event_name in {"c", "fix_arm_toggle", "g", "init_arm_toggle"}
                or event_control in {"fix_arm_select", "init_arm_select"}
            ):
                status = "accepted" if accepted else "rejected"
                controls = ",".join(sorted(str(item) for item in active_controls))
                self._state.messages.append(
                    {
                        "tag": "WebGUIEvent",
                        "message": (
                            f"{status}: {event_name}; active_controls=[{controls}]"
                        ),
                    }
                )
                if len(self._state.messages) > 48:
                    self._state.messages = self._state.messages[-48:]
        if not accepted:
            return False
        self._events.put(event_name)
        return True

    def consume_events(self) -> list[str]:
        events: list[str] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def publish(self, **updates: Any) -> None:
        with self._state_lock:
            for key, value in updates.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

    def append_message(self, tag: str, message: str) -> None:
        with self._state_lock:
            self._state.messages.append({"tag": str(tag), "message": str(message)})
            if len(self._state.messages) > 48:
                self._state.messages = self._state.messages[-48:]

    def mark_shutdown(self, message: str = "") -> None:
        with self._state_lock:
            self._state.shutdown = True
            self._state.shutdown_message = str(message)
            self._state.active_controls = []

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            snapshot = {
                "stage": self._state.stage,
                "selected_side": self._state.selected_side,
                "frame_viz_mode": self._state.frame_viz_mode,
                "virtual_desk_post_yaw_deg": self._state.virtual_desk_post_yaw_deg,
                "renderer_ready": self._state.renderer_ready,
                "ui_mode": self._state.ui_mode,
                "host": self._state.host,
                "port": self._state.port,
                "load_profile": self._state.load_profile,
                "show_tracker_frame_viz": self._state.show_tracker_frame_viz,
                "home_ready": self._state.home_ready,
                "teleop_running": self._state.teleop_running,
                "raw_tracker_poses": dict(self._state.raw_tracker_poses),
                "debug_ik_timing": self._state.debug_ik_timing,
                "ik_timing": dict(self._state.ik_timing),
                "ik_warning": dict(self._state.ik_warning),
                "pose_quality": dict(self._state.pose_quality),
                "debug_ik_logging": self._state.debug_ik_logging,
                "ik_debug_log_path": self._state.ik_debug_log_path,
                "gripper_control_available": self._state.gripper_control_available,
                "gripper_q6_available": self._state.gripper_q6_available,
                "gripper_control_enabled": self._state.gripper_control_enabled,
                "gripper_status": dict(self._state.gripper_status),
                "camera_status": dict(self._state.camera_status),
                "camera_self_check": dict(self._state.camera_self_check),
                "tactile_camera_status": dict(self._state.tactile_camera_status),
                "episode_recording_available": self._state.episode_recording_available,
                "episode_recording": self._state.episode_recording,
                "episode_dir": self._state.episode_dir,
                "episode_frame_count": self._state.episode_frame_count,
                "episode_sample_hz": self._state.episode_sample_hz,
                "episode_target_hz": self._state.episode_target_hz,
                "episode_tactile_write_hz": dict(self._state.episode_tactile_write_hz),
                "episode_tactile_write_hz_mean": self._state.episode_tactile_write_hz_mean,
                "episode_tactile_frame_counts": dict(self._state.episode_tactile_frame_counts),
                "episode_frequency_warning": dict(self._state.episode_frequency_warning),
                "episode_event_status": self._state.episode_event_status,
                "episode_event_message": self._state.episode_event_message,
                "episode_last_episode_index": self._state.episode_last_episode_index,
                "episode_last_frame_count": self._state.episode_last_frame_count,
                "dataset_base_root": self._state.dataset_base_root,
                "dataset_root": self._state.dataset_root,
                "dataset_dir_name": self._state.dataset_dir_name,
                "collection_task_id": self._state.collection_task_id,
                "collection_task_key": self._state.collection_task_key,
                "collection_task_name": self._state.collection_task_name,
                "collection_task_name_zh": self._state.collection_task_name_zh,
                "collection_task_description": self._state.collection_task_description,
                "collection_task_type": self._state.collection_task_type,
                "collection_round_trip_phase": self._state.collection_round_trip_phase,
                "collection_round_trip_next_phase": self._state.collection_round_trip_next_phase,
                "collection_round_trip_base_name": self._state.collection_round_trip_base_name,
                "collection_round_trip_base_id": self._state.collection_round_trip_base_id,
                "collection_round_trip_active": self._state.collection_round_trip_active,
                "gripper_task": self._state.gripper_task,
                "task_registry": [dict(item) for item in self._state.task_registry],
                "camera_visualization_available": self._state.camera_visualization_available,
                "camera_visualization_active": self._state.camera_visualization_active,
                "tactile_visualization_available": self._state.tactile_visualization_available,
                "tactile_visualization_active": self._state.tactile_visualization_active,
                "visualization_status": self._state.visualization_status,
                "vive_calibration_profile_path": self._state.vive_calibration_profile_path,
                "vive_calibration_profile_status": self._state.vive_calibration_profile_status,
                "vive_calibration_profile_ready": self._state.vive_calibration_profile_ready,
                "vive_calibration_profile_loaded": self._state.vive_calibration_profile_loaded,
                "movement_multiple": self._state.movement_multiple,
                "movement_multiple_min": self._state.movement_multiple_min,
                "movement_multiple_max": self._state.movement_multiple_max,
                "movement_multiple_step": self._state.movement_multiple_step,
                "fix_arm_available": self._state.fix_arm_available,
                "fix_arm_selected_side": self._state.fix_arm_selected_side,
                "fix_arm_fixed_side": self._state.fix_arm_fixed_side,
                "fix_arm_active": self._state.fix_arm_active,
                "fix_arm_recalibrating": self._state.fix_arm_recalibrating,
                "fix_arm_status": self._state.fix_arm_status,
                "fix_arm_sample_count": self._state.fix_arm_sample_count,
                "fix_arm_min_samples": self._state.fix_arm_min_samples,
                "special_arm_control_mode": self._state.special_arm_control_mode,
                "init_arm_available": self._state.init_arm_available,
                "init_arm_selected_side": self._state.init_arm_selected_side,
                "init_arm_active_side": self._state.init_arm_active_side,
                "init_arm_active": self._state.init_arm_active,
                "init_arm_recalibrating": self._state.init_arm_recalibrating,
                "init_arm_status": self._state.init_arm_status,
                "init_arm_sample_count": self._state.init_arm_sample_count,
                "init_arm_min_samples": self._state.init_arm_min_samples,
                "init_arm_profile_path": self._state.init_arm_profile_path,
                "init_arm_arrival_status": self._state.init_arm_arrival_status,
                "init_arm_arrival_label": self._state.init_arm_arrival_label,
                "init_arm_arrival_reason": self._state.init_arm_arrival_reason,
                "init_arm_target_position_error_m": self._state.init_arm_target_position_error_m,
                "init_arm_target_rotation_error_deg": self._state.init_arm_target_rotation_error_deg,
                "init_arm_stability_dwell_sec": self._state.init_arm_stability_dwell_sec,
                "pose_snapshot_available": self._state.pose_snapshot_available,
                "pose_snapshot_dir": self._state.pose_snapshot_dir,
                "pose_snapshot_count": self._state.pose_snapshot_count,
                "pose_snapshot_last_path": self._state.pose_snapshot_last_path,
                "pose_snapshot_status": self._state.pose_snapshot_status,
                "active_controls": list(self._state.active_controls),
                "control_hint": self._state.control_hint,
                "capture_stage": self._state.capture_stage,
                "capture_phase": self._state.capture_phase,
                "capturing": self._state.capturing,
                "capture_channel": self._state.capture_channel,
                "capture_samples": self._state.capture_samples,
                "capture_left_samples": self._state.capture_left_samples,
                "capture_right_samples": self._state.capture_right_samples,
                "capture_min_samples": self._state.capture_min_samples,
                "capture_valid": self._state.capture_valid,
                "capture_left_valid": self._state.capture_left_valid,
                "capture_right_valid": self._state.capture_right_valid,
                "capture_elapsed_sec": self._state.capture_elapsed_sec,
                "capture_duration_sec": self._state.capture_duration_sec,
                "capture_prep_remaining_sec": self._state.capture_prep_remaining_sec,
                "capture_prep_total_sec": self._state.capture_prep_total_sec,
                "capture_waiting_for_samples": self._state.capture_waiting_for_samples,
                "shutdown": self._state.shutdown,
                "shutdown_message": self._state.shutdown_message,
                "messages": list(self._state.messages),
            }
        snapshot["runtime_home_diagnostic"] = _runtime_home_diagnostic_payload(snapshot)
        return snapshot


class WebUiKeyInput:
    def __init__(self, ui: CalibrationWebUi) -> None:
        self.ui = ui
        self.enabled = True
        self.should_quit = False
        self._events: list[str] = []

    def start(self) -> None:
        self.ui.start()

    def close(self) -> None:
        return

    def poll(self) -> None:
        events = self.ui.consume_events()
        if not events:
            return
        self._events.extend(events)
        if "q" in events:
            self.should_quit = True

    def consume_events(self) -> list[str]:
        events = list(self._events)
        self._events.clear()
        return events
