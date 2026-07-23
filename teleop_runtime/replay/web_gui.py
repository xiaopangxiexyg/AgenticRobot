from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from teleop_runtime.action.replay_shadow_frontend import ReplayGuiShadowProfile
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
)

from ..configs.dmtac_hardware import (
    add_dmtac_assignment_arguments,
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
)
from ..http_security import (
    CONTROL_TOKEN_HEADER,
    DEFAULT_MAX_CONTROL_BODY_BYTES,
    HttpRequestSecurityError,
    WebControlSecurity,
    generate_session_token,
    validate_bind_host,
)
from ..defaults import DEFAULT_TELEOP_POSITION_REFERENCE_HZ
from .arm_init.qpos_execute import normalize_qpos_init_max_angvel
from .executor import ReplayExecutionSettings
from .hardware import (
    DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
    END_EFFECTOR_BACKEND_DMTAC_GRIPPER,
    RM65_ARM_CONTROL_MODE_POSITION,
    RM65_ARM_CONTROL_MODES,
    ReplayHardwareConfig,
    normalize_end_effector_backend,
)
from .session import ReplayGuiSession


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DexTeleop Replay</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f5f2;
      --surface: #ffffff;
      --surface-2: #eef1ef;
      --line: #c9d1cf;
      --ink: #172026;
      --muted: #5d6871;
      --accent: #0b6f68;
      --accent-2: #2457a6;
      --danger: #9b3d34;
      --warn: #9a6a18;
      --shadow: 0 1px 2px rgba(20, 28, 34, 0.08);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 0 12px;
      cursor: pointer;
      box-shadow: var(--shadow);
    }
    button:hover { border-color: var(--accent); }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button.danger {
      color: var(--danger);
    }
    button:disabled,
    button.primary:disabled,
    button.danger:disabled {
      background: #e8ecea;
      border-color: #d4dbd8;
      color: #87918f;
      cursor: not-allowed;
      box-shadow: none;
    }
    button:disabled:hover { border-color: #d4dbd8; }
    input, select {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      min-height: 34px;
      padding: 0 10px;
      color: var(--ink);
      width: 100%;
    }
    input:disabled, select:disabled {
      background: #edf0ee;
      color: #87918f;
      cursor: not-allowed;
    }
    input[type="range"] {
      padding: 0;
      accent-color: var(--accent);
    }
    .app {
      min-height: 100%;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    .brand {
      font-weight: 700;
      white-space: nowrap;
    }
    .pathbox {
      display: grid;
      grid-template-columns: auto minmax(280px, 1fr) auto;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .status {
      color: var(--muted);
      min-width: 190px;
      text-align: right;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 360px;
    }
    .status.notice { color: var(--accent-2); }
    .status.error { color: var(--danger); }
    .workspace {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 340px;
      min-height: 0;
    }
    .sidebar, .inspector {
      border-right: 1px solid var(--line);
      background: #fafbf9;
      min-height: 0;
      overflow: auto;
    }
    .inspector {
      border-right: 0;
      border-left: 1px solid var(--line);
    }
    .section {
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .section-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .episodes {
      display: grid;
      gap: 6px;
    }
    .episode-row {
      display: grid;
      grid-template-columns: 92px 1fr auto;
      gap: 6px;
      align-items: center;
      padding: 8px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      cursor: pointer;
    }
    .episode-row.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .episode-row .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .main {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .transport {
      padding: 10px 12px;
      background: #fbfcfb;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: auto minmax(180px, 1fr) 220px auto;
      gap: 12px;
      align-items: center;
    }
    .button-group {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }
    .form-row {
      display: grid;
      grid-template-columns: minmax(76px, 0.55fr) minmax(0, 1fr);
      gap: 8px;
      align-items: center;
    }
    .check-row {
      display: flex;
      gap: 8px;
      align-items: center;
      min-height: 28px;
      color: var(--ink);
    }
    .check-row input {
      width: auto;
      min-height: 0;
      accent-color: var(--accent);
    }
    .checks {
      display: grid;
      gap: 6px;
      margin-top: 10px;
    }
    .check {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      padding: 7px 8px;
      font-size: 12px;
    }
    .check.error { border-color: #d5a5a0; color: var(--danger); }
    .check.warning { border-color: #d9c08a; color: var(--warn); }
    .check.info { color: var(--muted); }
    .readout {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: flex-end;
      color: var(--muted);
      white-space: nowrap;
    }
    .media {
      padding: 12px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .media-group {
      display: grid;
      gap: 8px;
    }
    .media-title {
      font-weight: 700;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }
    .media-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 10px;
    }
    .viewer {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
      box-shadow: var(--shadow);
    }
    .viewer-title {
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      background: var(--surface-2);
      border-bottom: 1px solid var(--line);
      font-weight: 600;
      white-space: nowrap;
    }
    .viewer img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #111820;
    }
    .kv {
      display: grid;
      grid-template-columns: minmax(90px, 0.8fr) minmax(0, 1.2fr);
      gap: 4px 8px;
      align-items: start;
      font-size: 13px;
    }
    .kv .k { color: var(--muted); }
    .kv .v {
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .messages {
      display: grid;
      gap: 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #27313a;
      max-height: 300px;
      overflow: auto;
      align-content: start;
    }
    .message {
      padding: 4px 0;
      border-bottom: 1px solid rgba(201, 209, 207, 0.55);
      overflow-wrap: anywhere;
    }
    .message.error { color: var(--danger); }
    .empty {
      color: var(--muted);
      padding: 10px 0;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      background: #e8eeec;
      color: #26323a;
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.warn { background: #f3ead8; color: var(--warn); }
    .pill.live { background: #dfefec; color: var(--accent); }
    @media (max-width: 1120px) {
      .workspace { grid-template-columns: 280px minmax(0, 1fr); }
      .inspector { display: none; }
      .transport { grid-template-columns: 1fr; }
      .readout { justify-content: flex-start; }
    }
    @media (max-width: 760px) {
      .topbar { grid-template-columns: 1fr; }
      .pathbox { grid-template-columns: 1fr; }
      .status { text-align: left; }
      .workspace { grid-template-columns: 1fr; }
      .sidebar { max-height: 260px; border-right: 0; border-bottom: 1px solid var(--line); }
      .media-grid { grid-template-columns: minmax(0, 1fr); }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">DexTeleop Replay</div>
      <div class="pathbox">
        <span class="label">Dataset</span>
        <input id="datasetRoot" autocomplete="off" spellcheck="false">
        <button id="loadBtn" class="primary">Load</button>
      </div>
      <span id="playState" class="pill">idle</span>
      <div id="status" class="status">ready</div>
    </header>
    <div class="workspace">
      <aside class="sidebar">
        <div class="section">
          <div class="section-title">
            <span>Episodes</span>
            <span id="episodeCount" class="pill">0</span>
          </div>
          <div id="episodes" class="episodes"></div>
        </div>
      </aside>
      <main class="main">
        <div class="transport">
          <div class="button-group">
            <button id="firstBtn">|&lt;</button>
            <button id="prevBtn">&lt;</button>
            <button id="playBtn" class="primary">Play</button>
            <button id="pauseBtn">Pause</button>
            <button id="nextBtn">&gt;</button>
            <button id="lastBtn">&gt;|</button>
          </div>
          <input id="timeline" type="range" min="0" max="0" value="0">
          <div>
            <div class="label">Speed</div>
            <input id="speed" type="range" min="0.1" max="4" step="0.1" value="1">
          </div>
          <div class="readout">
            <span id="frameText">0 / 0</span>
            <span id="timeText">0.000 s</span>
            <span id="speedText">1.0x</span>
          </div>
        </div>
        <div class="media">
          <section class="media-group">
            <div class="media-title">Cameras <span id="cameraCount" class="pill">0</span></div>
            <div id="cameraGrid" class="media-grid"></div>
          </section>
          <section class="media-group">
            <div class="media-title">Tactile <span id="tactileCount" class="pill">0</span></div>
            <div id="tactileGrid" class="media-grid"></div>
          </section>
        </div>
      </main>
      <aside class="inspector">
        <div class="section">
          <div class="section-title">Episode</div>
          <div id="episodeInfo" class="kv"></div>
        </div>
        <div class="section">
          <div class="section-title">
            <span>Hardware Replay</span>
            <span id="execState" class="pill">configured</span>
          </div>
          <div class="form-grid">
            <div class="form-row">
              <span class="label">Active</span>
              <select id="activeSides">
                <option value="right">right</option>
                <option value="left">left</option>
                <option value="both">both</option>
              </select>
            </div>
            <div class="form-row">
              <span class="label">Left IP</span>
              <input id="leftArmIp" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-row">
              <span class="label">Right IP</span>
              <input id="rightArmIp" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-row">
              <span class="label">Left TCP</span>
              <input id="leftToolProfile" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-row">
              <span class="label">Right TCP</span>
              <input id="rightToolProfile" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-row">
              <span class="label">Mode</span>
              <select id="armControlMode">
                <option value="position">position</option>
                <option value="impedance">impedance</option>
              </select>
            </div>
            <div class="form-row">
              <span class="label">Command Hz</span>
              <input id="armControlHz" type="number" min="1" step="1" value="150">
            </div>
            <div class="form-row">
              <span class="label">Reference Hz</span>
              <input id="positionReferenceHz" type="number" min="1" step="1" value="__DEFAULT_POSITION_REFERENCE_HZ__">
            </div>
            <div class="form-row">
              <span class="label">Collision Scene</span>
              <input id="positionCollisionScene" value="dual_arm_hand_proxy_open" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-row">
              <span class="label">Collision Min m</span>
              <input id="positionCollisionMinDistance" type="number" min="0" step="0.001" value="0.03">
            </div>
            <div class="form-row">
              <span class="label">Collision Eps m</span>
              <input id="positionCollisionEpsilon" type="number" min="0" step="0.0001" value="0.0001">
            </div>
            <div class="form-row">
              <span class="label">Reference Max Vel</span>
              <input id="positionCollisionMaxVelocity" value="0.35,0.35,0.35,0.5,0.5,0.5" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-row">
              <span class="label">Reference Max Step</span>
              <input id="positionCollisionMaxStep" autocomplete="off" spellcheck="false" placeholder="velocity / Hz">
            </div>
            <div class="form-row">
              <span class="label">Tare Sec</span>
              <input id="forceTareSec" type="number" min="0.1" step="0.1" value="3">
            </div>
            <div class="form-row">
              <span class="label">QPOS Init Vel</span>
              <input id="qposInitArmMaxAngvel" autocomplete="off" spellcheck="false" placeholder="profile">
            </div>
            <label class="check-row">
              <input id="armCanfdFollow" type="checkbox" checked>
              <span>CANFD Follow</span>
            </label>
            <label class="check-row">
              <input id="positionCollisionRequired" type="checkbox" checked>
              <span>Reference Collision Required</span>
            </label>
            <label class="check-row">
              <input id="positionCollisionDebug" type="checkbox">
              <span>Collision Raw Debug</span>
            </label>
            <label class="check-row">
              <input id="sendActions" type="checkbox">
              <span>Arm Output</span>
            </label>
            <label class="check-row">
              <input id="connectHardware" type="checkbox" checked>
              <span>Connect Hardware</span>
            </label>
            <label class="check-row">
              <input id="strictToolProfile" type="checkbox" checked>
              <span>Strict Tool</span>
            </label>
          </div>
          <div class="button-group">
            <button id="configureHardwareBtn">Configure</button>
            <button id="connectHardwareBtn">Connect</button>
            <button id="preflightHardwareBtn" class="primary">Preflight</button>
            <button id="moveStartBtn">Move Start</button>
            <button id="startHardwareBtn" class="primary">Start</button>
            <button id="abortHardwareBtn" class="danger">Abort</button>
          </div>
          <div id="execInfo" class="kv" style="margin-top:10px"></div>
          <div id="preflightChecks" class="checks"></div>
        </div>
        <div class="section">
          <div class="section-title">Frame</div>
          <div id="rowInfo" class="kv"></div>
        </div>
        <div class="section">
          <div class="section-title">Log</div>
          <div id="messages" class="messages"></div>
        </div>
      </aside>
    </div>
  </div>
  <script>
    const els = {
      datasetRoot: document.getElementById("datasetRoot"),
      loadBtn: document.getElementById("loadBtn"),
      episodes: document.getElementById("episodes"),
      episodeCount: document.getElementById("episodeCount"),
      playState: document.getElementById("playState"),
      status: document.getElementById("status"),
      firstBtn: document.getElementById("firstBtn"),
      prevBtn: document.getElementById("prevBtn"),
      playBtn: document.getElementById("playBtn"),
      pauseBtn: document.getElementById("pauseBtn"),
      nextBtn: document.getElementById("nextBtn"),
      lastBtn: document.getElementById("lastBtn"),
      timeline: document.getElementById("timeline"),
      speed: document.getElementById("speed"),
      frameText: document.getElementById("frameText"),
      timeText: document.getElementById("timeText"),
      speedText: document.getElementById("speedText"),
      cameraCount: document.getElementById("cameraCount"),
      tactileCount: document.getElementById("tactileCount"),
      cameraGrid: document.getElementById("cameraGrid"),
      tactileGrid: document.getElementById("tactileGrid"),
      episodeInfo: document.getElementById("episodeInfo"),
      execState: document.getElementById("execState"),
      activeSides: document.getElementById("activeSides"),
      leftArmIp: document.getElementById("leftArmIp"),
      rightArmIp: document.getElementById("rightArmIp"),
      leftToolProfile: document.getElementById("leftToolProfile"),
      rightToolProfile: document.getElementById("rightToolProfile"),
      armControlMode: document.getElementById("armControlMode"),
      armControlHz: document.getElementById("armControlHz"),
      positionReferenceHz: document.getElementById("positionReferenceHz"),
      positionCollisionScene: document.getElementById("positionCollisionScene"),
      positionCollisionMinDistance: document.getElementById("positionCollisionMinDistance"),
      positionCollisionEpsilon: document.getElementById("positionCollisionEpsilon"),
      positionCollisionMaxVelocity: document.getElementById("positionCollisionMaxVelocity"),
      positionCollisionMaxStep: document.getElementById("positionCollisionMaxStep"),
      positionCollisionRequired: document.getElementById("positionCollisionRequired"),
      positionCollisionDebug: document.getElementById("positionCollisionDebug"),
      forceTareSec: document.getElementById("forceTareSec"),
      qposInitArmMaxAngvel: document.getElementById("qposInitArmMaxAngvel"),
      armCanfdFollow: document.getElementById("armCanfdFollow"),
      sendActions: document.getElementById("sendActions"),
      connectHardware: document.getElementById("connectHardware"),
      strictToolProfile: document.getElementById("strictToolProfile"),
      configureHardwareBtn: document.getElementById("configureHardwareBtn"),
      connectHardwareBtn: document.getElementById("connectHardwareBtn"),
      preflightHardwareBtn: document.getElementById("preflightHardwareBtn"),
      moveStartBtn: document.getElementById("moveStartBtn"),
      startHardwareBtn: document.getElementById("startHardwareBtn"),
      abortHardwareBtn: document.getElementById("abortHardwareBtn"),
      execInfo: document.getElementById("execInfo"),
      preflightChecks: document.getElementById("preflightChecks"),
      rowInfo: document.getElementById("rowInfo"),
      messages: document.getElementById("messages"),
    };
    const controlToken = __REPLAY_CONTROL_TOKEN_JSON__;
    let latest = null;
    let dragging = false;
    let notice = {text: "", kind: "notice", until: 0};
    let clientMessages = [];
    let lastPollError = "";

    function text(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }

    async function api(path, payload) {
      const response = await fetch(path, {
        method: payload ? "POST" : "GET",
        headers: payload ? {
          "Content-Type": "application/json",
          "X-DexTeleop-Control-Token": controlToken,
        } : {},
        body: payload ? JSON.stringify(payload) : undefined,
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    async function control(payload) {
      try {
        const data = await api("/api/control", payload);
        render(data.state || data);
        const state = data.state || data;
        const execution = state.execution || {};
        const message = execution.last_error || execution.last_message || newestServerMessage(state);
        if (message) setNotice(message, execution.last_error ? "error" : "notice", execution.last_error ? 30000 : 15000);
        return true;
      } catch (err) {
        addClientMessage(err.message, "error");
        return false;
      }
    }

    function newestServerMessage(state) {
      const messages = state && Array.isArray(state.messages) ? state.messages : [];
      return messages.length ? messages[messages.length - 1] : "";
    }

    function setNotice(message, kind = "notice", durationMs = 15000) {
      const textValue = text(message).trim();
      if (!textValue) return;
      notice = {
        text: textValue,
        kind: kind === "error" ? "error" : "notice",
        until: Date.now() + Math.max(1000, Number(durationMs || 15000)),
      };
      renderStatus(latest);
    }

    function addClientMessage(message, kind = "notice") {
      const stamp = new Date().toLocaleTimeString();
      clientMessages.push({text: `[${stamp}] ${message}`, kind: kind === "error" ? "error" : "notice"});
      clientMessages = clientMessages.slice(-80);
      setNotice(message, kind, kind === "error" ? 30000 : 15000);
      renderMessages(latest);
    }

    function kv(container, payload) {
      container.innerHTML = "";
      const entries = Object.entries(payload || {});
      if (!entries.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "none";
        container.appendChild(empty);
        return;
      }
      for (const [key, value] of entries) {
        const k = document.createElement("div");
        const v = document.createElement("div");
        k.className = "k";
        v.className = "v";
        k.textContent = key;
        v.textContent = text(value);
        container.appendChild(k);
        container.appendChild(v);
      }
    }

    function renderEpisodes(state) {
      const current = state.episode ? state.episode.episode_id : "";
      els.episodes.innerHTML = "";
      els.episodeCount.textContent = String((state.episodes || []).length);
      if (!state.episodes || !state.episodes.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "none";
        els.episodes.appendChild(empty);
        return;
      }
      for (const ep of state.episodes) {
        const row = document.createElement("div");
        row.className = "episode-row" + (ep.episode_id === current ? " active" : "");
        row.onclick = () => control({type: "select_episode", episode: ep.episode_id});
        const id = document.createElement("div");
        const meta = document.createElement("div");
        const badge = document.createElement("span");
        id.textContent = ep.episode_id || ep.episode_index;
        meta.className = "meta";
        meta.textContent = `${ep.frame_count || 0} frames`;
        badge.className = "pill" + (ep.exists ? "" : " warn");
        badge.textContent = ep.exists ? "ok" : "missing";
        row.appendChild(id);
        row.appendChild(meta);
        row.appendChild(badge);
        els.episodes.appendChild(row);
      }
    }

    function renderMediaGrid(container, keys, kind, frameIndex) {
      const existing = new Set([...container.querySelectorAll(".viewer")].map(el => el.dataset.key));
      const desired = new Set(keys || []);
      for (const node of [...container.querySelectorAll(".viewer")]) {
        if (!desired.has(node.dataset.key)) node.remove();
      }
      for (const key of desired) {
        let viewer = container.querySelector(`.viewer[data-key="${CSS.escape(key)}"]`);
        if (!viewer) {
          viewer = document.createElement("div");
          viewer.className = "viewer";
          viewer.dataset.key = key;
          const title = document.createElement("div");
          title.className = "viewer-title";
          title.textContent = key;
          const img = document.createElement("img");
          img.alt = key;
          viewer.appendChild(title);
          viewer.appendChild(img);
          container.appendChild(viewer);
        }
        const img = viewer.querySelector("img");
        const route = kind === "camera" ? "/media/camera/" : "/media/tactile/";
        img.src = `${route}${encodeURIComponent(key)}.jpg?frame=${frameIndex}&t=${Date.now()}`;
      }
      if (!desired.size) {
        container.innerHTML = '<div class="empty">none</div>';
      } else {
        for (const empty of [...container.querySelectorAll(".empty")]) empty.remove();
      }
    }

    function fieldValue(el, value) {
      if (document.activeElement !== el) el.value = value == null ? "" : String(value);
    }

    function activeSidesValue(sides) {
      const list = Array.isArray(sides) ? sides : [];
      if (list.includes("left") && list.includes("right")) return "both";
      if (list.includes("left")) return "left";
      return "right";
    }

    function defaultHzForMode(mode) {
      return mode === "impedance" ? 40 : 150;
    }

    function hardwarePayload() {
      const mode = els.armControlMode.value || "position";
      const hzText = els.armControlHz.value.trim();
      const referenceHzText = els.positionReferenceHz.value.trim();
      const tareText = els.forceTareSec.value.trim();
      return {
        type: "configure_hardware",
        active_sides: els.activeSides.value,
        left_arm_ip: els.leftArmIp.value.trim(),
        right_arm_ip: els.rightArmIp.value.trim(),
        left_tool_profile: els.leftToolProfile.value.trim(),
        right_tool_profile: els.rightToolProfile.value.trim(),
        arm_control_mode: mode,
        arm_control_hz: Number(hzText || defaultHzForMode(mode)),
        position_reference_hz: Number(referenceHzText || __DEFAULT_POSITION_REFERENCE_HZ__),
        position_collision_required: els.positionCollisionRequired.checked,
        position_collision_scene_profile: els.positionCollisionScene.value.trim(),
        position_collision_min_distance_m: Number(
          els.positionCollisionMinDistance.value.trim() || 0.03
        ),
        position_collision_distance_epsilon_m: Number(
          els.positionCollisionEpsilon.value.trim() || 0.0001
        ),
        position_collision_max_joint_velocity_rad_s:
          els.positionCollisionMaxVelocity.value.trim(),
        position_collision_max_joint_step_rad:
          els.positionCollisionMaxStep.value.trim(),
        position_collision_debug: els.positionCollisionDebug.checked,
        arm_canfd_follow: els.armCanfdFollow.checked,
        force_tare_sec: Number(tareText || 3),
        qpos_init_max_angvel_rad_s: els.qposInitArmMaxAngvel.value.trim(),
        send_actions: els.sendActions.checked,
        connect_hardware: els.connectHardware.checked,
        strict_tool_profile: els.strictToolProfile.checked,
      };
    }

    function hardwarePayloadFor(type) {
      return {...hardwarePayload(), type};
    }

    function renderChecks(execution) {
      els.preflightChecks.innerHTML = "";
      const checks = execution && execution.preflight ? (execution.preflight.checks || []) : [];
      if (!checks.length) return;
      for (const check of checks) {
        const div = document.createElement("div");
        div.className = `check ${check.ok ? (check.level || "info") : "error"}`;
        div.textContent = `${check.ok ? "OK" : "BLOCK"} ${check.name}: ${check.message || ""}`;
        els.preflightChecks.appendChild(div);
      }
    }

    function renderExecution(state) {
      const execution = state.execution || {};
      const config = execution.hardware_config || {};
      const settings = execution.settings || {};
      const ips = config.arm_ip_by_side || {};
      const profiles = config.tool_profile_by_side || {};
      fieldValue(els.activeSides, activeSidesValue(config.active_sides || execution.active_sides || ["left", "right"]));
      fieldValue(els.leftArmIp, ips.left || "");
      fieldValue(els.rightArmIp, ips.right || "");
      fieldValue(els.leftToolProfile, profiles.left || "");
      fieldValue(els.rightToolProfile, profiles.right || "");
      fieldValue(els.armControlMode, config.arm_control_mode || "position");
      fieldValue(els.armControlHz, config.arm_control_hz || defaultHzForMode(config.arm_control_mode || "position"));
      fieldValue(
        els.positionReferenceHz,
        config.position_reference_hz || __DEFAULT_POSITION_REFERENCE_HZ__
      );
      fieldValue(
        els.positionCollisionScene,
        settings.position_collision_scene_profile || "dual_arm_hand_proxy_open"
      );
      fieldValue(
        els.positionCollisionMinDistance,
        settings.position_collision_min_distance_m == null
          ? 0.03
          : settings.position_collision_min_distance_m
      );
      fieldValue(
        els.positionCollisionEpsilon,
        settings.position_collision_distance_epsilon_m == null
          ? 0.0001
          : settings.position_collision_distance_epsilon_m
      );
      fieldValue(
        els.positionCollisionMaxVelocity,
        Array.isArray(settings.position_collision_max_joint_velocity_rad_s)
          ? settings.position_collision_max_joint_velocity_rad_s.join(",")
          : "0.35,0.35,0.35,0.5,0.5,0.5"
      );
      fieldValue(
        els.positionCollisionMaxStep,
        Array.isArray(settings.position_collision_max_joint_step_rad)
          ? settings.position_collision_max_joint_step_rad.join(",")
          : ""
      );
      fieldValue(els.forceTareSec, config.force_tare_sec || 3);
      fieldValue(
        els.qposInitArmMaxAngvel,
        Array.isArray(settings.qpos_init_max_angvel_rad_s)
          ? settings.qpos_init_max_angvel_rad_s.join(",")
          : ""
      );
      if (document.activeElement !== els.armCanfdFollow) els.armCanfdFollow.checked = config.arm_canfd_follow !== false;
      if (document.activeElement !== els.positionCollisionRequired) {
        els.positionCollisionRequired.checked = settings.position_collision_required !== false;
      }
      if (document.activeElement !== els.positionCollisionDebug) {
        els.positionCollisionDebug.checked = !!settings.position_collision_debug;
      }
      if (document.activeElement !== els.sendActions) els.sendActions.checked = !!settings.send_actions_enabled;
      if (document.activeElement !== els.connectHardware) els.connectHardware.checked = config.connect_hardware !== false;
      if (document.activeElement !== els.strictToolProfile) els.strictToolProfile.checked = config.strict_tool_profile !== false;
      const stateText = execution.state || "configured";
      els.execState.textContent = stateText;
      els.execState.className = "pill" + (stateText === "replaying" || stateText === "moving_to_start" ? " live" : "");
      kv(els.execInfo, {
        connected: execution.connected,
        output: settings.send_actions_enabled ? "enabled" : "disabled",
        end_effectors: settings.end_effectors_enabled ? "enabled" : "disabled",
        end_effector_backends: JSON.stringify(config.end_effector_backend_by_side || {}),
        qpos_init_vel: Array.isArray(settings.qpos_init_max_angvel_rad_s)
          ? settings.qpos_init_max_angvel_rad_s.join(",")
          : (settings.qpos_init_joint_velocity_profile || ""),
        frame: execution.frame_index || 0,
        group: execution.group ? execution.group.source_episode_index : "",
        hz: execution.group ? Number(execution.group.action_hz || 0).toFixed(3) : "",
        message: execution.last_message || "",
        error: execution.last_error || "",
      });
      renderChecks(execution);
    }

    function isExecutionBusy(execution) {
      const stateText = execution && execution.state ? execution.state : "";
      return !!(execution && execution.worker_alive) || ["connecting", "moving_to_start", "replaying"].includes(stateText);
    }

    function setDisabled(elements, disabled) {
      for (const element of elements) element.disabled = !!disabled;
    }

    function updateControls(state) {
      const execution = state.execution || {};
      const execState = execution.state || "configured";
      const settings = execution.settings || {};
      const hasEpisode = !!state.episode;
      const hasFrames = Number(state.frame_count || 0) > 0;
      const busy = isExecutionBusy(execution);
      const connected = !!execution.connected;
      const outputEnabled = !!settings.send_actions_enabled;
      const canMoveStart = hasEpisode && connected && outputEnabled && execState === "preflight_ok" && !busy;
      const canStart = hasEpisode && connected && outputEnabled && execState === "start_ready" && !busy;
      const canAbort = busy || execState === "start_ready";

      setDisabled([els.firstBtn, els.prevBtn, els.nextBtn, els.lastBtn, els.timeline], !hasFrames);
      els.playBtn.disabled = !hasFrames || !!state.playing;
      els.pauseBtn.disabled = !hasFrames || !state.playing;
      els.loadBtn.disabled = busy;
      els.configureHardwareBtn.disabled = busy;
      els.connectHardwareBtn.disabled = busy || connected;
      els.preflightHardwareBtn.disabled = !hasEpisode || busy;
      els.moveStartBtn.disabled = !canMoveStart;
      els.startHardwareBtn.disabled = !canStart;
      els.abortHardwareBtn.disabled = !canAbort;
      els.sendActions.disabled = busy;
      setDisabled(
        [
          els.activeSides,
          els.leftArmIp,
          els.rightArmIp,
          els.leftToolProfile,
          els.rightToolProfile,
          els.armControlMode,
          els.armControlHz,
          els.positionReferenceHz,
          els.positionCollisionScene,
          els.positionCollisionMinDistance,
          els.positionCollisionEpsilon,
          els.positionCollisionMaxVelocity,
          els.positionCollisionMaxStep,
          els.positionCollisionRequired,
          els.positionCollisionDebug,
          els.forceTareSec,
          els.qposInitArmMaxAngvel,
          els.armCanfdFollow,
          els.connectHardware,
          els.strictToolProfile,
        ],
        busy,
      );
    }

    function renderStatus(state) {
      const execution = state && state.execution ? state.execution : {};
      const now = Date.now();
      const noticeActive = notice.text && now < notice.until;
      if (noticeActive) {
        els.status.textContent = notice.text;
        els.status.title = notice.text;
        els.status.className = `status ${notice.kind}`;
        return;
      }
      if (execution.last_error) {
        els.status.textContent = execution.last_error;
        els.status.title = execution.last_error;
        els.status.className = "status error";
        return;
      }
      const fallback = state && state.episode ? state.episode.episode_id : "ready";
      els.status.textContent = fallback;
      els.status.title = fallback;
      els.status.className = "status";
    }

    function renderMessages(state) {
      const server = state && Array.isArray(state.messages) ? state.messages.map(message => ({text: message, kind: "notice"})) : [];
      const combined = server.concat(clientMessages).slice(-60).reverse();
      els.messages.innerHTML = "";
      for (const message of combined) {
        const div = document.createElement("div");
        div.className = "message" + (message.kind === "error" ? " error" : "");
        div.textContent = message.text;
        els.messages.appendChild(div);
      }
    }

    function render(state) {
      latest = state;
      if (!state) return;
      if (state.dataset_root && document.activeElement !== els.datasetRoot) {
        els.datasetRoot.value = state.dataset_root;
      }
      const count = state.frame_count || 0;
      const frame = Math.max(0, state.frame_index || 0);
      els.timeline.max = String(Math.max(0, count - 1));
      if (!dragging) els.timeline.value = String(frame);
      els.frameText.textContent = `${count ? frame + 1 : 0} / ${count}`;
      els.timeText.textContent = `${Number(state.frame_time_sec || 0).toFixed(3)} s`;
      els.speed.value = String(state.speed || 1);
      els.speedText.textContent = `${Number(state.speed || 1).toFixed(1)}x`;
      els.playState.textContent = state.playing ? "playing" : "paused";
      els.playState.className = "pill" + (state.playing ? " live" : "");
      renderStatus(state);
      renderEpisodes(state);
      els.cameraCount.textContent = String((state.camera_keys || []).length);
      els.tactileCount.textContent = String((state.tactile_sides || []).length);
      renderMediaGrid(els.cameraGrid, state.camera_keys || [], "camera", frame);
      renderMediaGrid(els.tactileGrid, state.tactile_sides || [], "tactile", frame);
      if (state.episode) {
        kv(els.episodeInfo, {
          id: state.episode.episode_id,
          frames: state.episode.frame_count,
          duration: Number(state.episode.duration_sec || 0).toFixed(3),
          hz: Number(state.episode.nominal_hz || 0).toFixed(3),
          timing: state.episode.timing_source,
          parquet: state.episode.parquet_path,
          cameras: state.episode.camera_keys,
          tactile: state.episode.tactile_sides,
        });
      } else {
        kv(els.episodeInfo, {});
      }
      renderExecution(state);
      kv(els.rowInfo, state.row || {});
      renderMessages(state);
      updateControls(state);
    }

    async function poll() {
      try {
        const data = await api("/api/state");
        lastPollError = "";
        render(data.state || data);
      } catch (err) {
        if (err.message !== lastPollError) {
          lastPollError = err.message;
          addClientMessage(err.message, "error");
        } else {
          setNotice(err.message, "error", 30000);
        }
      } finally {
        const execState = latest && latest.execution ? latest.execution.state : "";
        const busy = execState === "replaying" || execState === "moving_to_start";
        setTimeout(poll, latest && (latest.playing || busy) ? 180 : 650);
      }
    }

    els.loadBtn.onclick = () => control({type: "load_dataset", dataset_root: els.datasetRoot.value});
    els.firstBtn.onclick = () => control({type: "seek", frame: 0});
    els.prevBtn.onclick = () => control({type: "step", delta: -1});
    els.playBtn.onclick = () => control({type: "play"});
    els.pauseBtn.onclick = () => control({type: "pause"});
    els.nextBtn.onclick = () => control({type: "step", delta: 1});
    els.lastBtn.onclick = () => control({type: "seek", frame: Math.max(0, (latest ? latest.frame_count : 1) - 1)});
    els.timeline.addEventListener("input", () => {
      dragging = true;
      els.frameText.textContent = `${Number(els.timeline.value) + 1} / ${latest ? latest.frame_count : 0}`;
    });
    els.timeline.addEventListener("change", () => {
      dragging = false;
      control({type: "seek", frame: Number(els.timeline.value)});
    });
    els.speed.addEventListener("change", () => control({type: "speed", speed: Number(els.speed.value)}));
    els.sendActions.addEventListener("change", () => {
      control({type: "set_send_actions", enabled: els.sendActions.checked});
    });
    els.armControlMode.addEventListener("change", () => {
      const mode = els.armControlMode.value || "position";
      els.armControlHz.value = String(defaultHzForMode(mode));
      if (mode === "position") {
        els.positionReferenceHz.value = "__DEFAULT_POSITION_REFERENCE_HZ__";
      }
      els.armCanfdFollow.checked = mode === "position";
    });
    els.configureHardwareBtn.onclick = () => control(hardwarePayload());
    els.connectHardwareBtn.onclick = async () => {
      if (await control(hardwarePayload())) {
        await control({type: "connect_hardware"});
      }
    };
    els.preflightHardwareBtn.onclick = () => control(hardwarePayloadFor("preflight_hardware"));
    els.moveStartBtn.onclick = () => control({type: "move_to_start"});
    els.startHardwareBtn.onclick = () => control({type: "start_hardware_replay"});
    els.abortHardwareBtn.onclick = () => control({type: "abort_hardware_replay"});
    poll();
  </script>
</body>
</html>
"""


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_float_csv(text: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    value = str(text or "").strip()
    if not value:
        return tuple(default)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    parsed = tuple(float(part) for part in parts)
    if len(parsed) != len(default):
        raise ValueError(f"Expected {len(default)} comma-separated values, got {len(parsed)}.")
    return parsed


def _parse_one_or_six_float_csv(text: str | None, *, optional: bool = False) -> tuple[float, ...] | None:
    value = str(text or "").strip()
    if not value:
        if optional:
            return None
        raise ValueError("Expected one or six comma-separated values.")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    parsed = tuple(float(part) for part in parts)
    if len(parsed) == 1:
        parsed = parsed * 6
    if len(parsed) != 6:
        raise ValueError(f"Expected one or six comma-separated values, got {len(parsed)}.")
    return parsed


def _parse_unit_interval_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


def _resolve_replay_dmtac_fields(
    args: argparse.Namespace,
    *,
    active_sides: Sequence[str],
    end_effector_backend_by_side: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Resolve replay DMTAC serial settings with the shared assignment precedence."""
    dmtac_sides = tuple(
        str(side)
        for side in active_sides
        if normalize_end_effector_backend(
            end_effector_backend_by_side.get(str(side))
        )
        == END_EFFECTOR_BACKEND_DMTAC_GRIPPER
    )
    fields: dict[str, dict[str, Any]] = {
        "port": {},
        "id": {},
        "baudrate": {},
        "timeout": {},
    }
    if not bool(args.end_effectors_enabled) or not dmtac_sides:
        return fields

    if bool(getattr(args, "dmtac_default_config", True)):
        try:
            assignment = load_dmtac_hardware_assignment(
                str(getattr(args, "dmtac_config", "latest") or "latest"),
                validate_current_usb=bool(
                    getattr(args, "dmtac_validate_current_usb", True)
                ),
                allow_missing=bool(getattr(args, "dmtac_allow_missing", False)),
                assume_yes_for_extra=bool(
                    getattr(args, "dmtac_assume_yes_for_extra", True)
                ),
            )
            configs = dmtac_backend_configs_from_assignment(
                assignment,
                active_sides=dmtac_sides,
                speed=float(args.gripper_speed),
                torque=float(args.gripper_torque),
                max_action=float(args.gripper_max_action),
            )
            for side, config in configs.items():
                if not bool(config.connect_hardware):
                    continue
                fields["port"][side] = str(config.gripper_port)
                fields["id"][side] = int(config.gripper_id)
                fields["baudrate"][side] = int(config.gripper_baudrate)
                fields["timeout"][side] = float(config.gripper_timeout)
        except Exception as exc:  # noqa: BLE001
            if bool(getattr(args, "dmtac_config_required", False)):
                raise RuntimeError(
                    "Failed to load DMTAC hardware assignment "
                    f"{getattr(args, 'dmtac_config', 'latest')!r}: {exc}"
                ) from exc
            print(
                f"[ReplayGUI] WARNING: could not load DMTAC hardware assignment: {exc}",
                flush=True,
            )

    for side in dmtac_sides:
        port = str(getattr(args, f"{side}_gripper_port", "") or "").strip()
        if not port:
            continue
        fields["port"][side] = port
        fields["id"][side] = int(getattr(args, f"{side}_gripper_id", 1) or 1)
        fields["baudrate"][side] = int(
            getattr(args, f"{side}_gripper_baudrate", 115200) or 115200
        )
        fields["timeout"][side] = float(
            getattr(args, f"{side}_gripper_timeout", 0.10) or 0.10
        )

    missing = [side for side in dmtac_sides if side not in fields["port"]]
    if missing and bool(getattr(args, "dmtac_config_required", False)):
        raise RuntimeError(
            "DMTAC hardware assignment does not define replay grippers for active "
            f"sides: {missing}"
        )
    return fields


def _should_log_http_message(access_log: bool, args: tuple[Any, ...]) -> bool:
    if access_log:
        return True
    if len(args) >= 2:
        try:
            status = int(args[1])
        except (TypeError, ValueError):
            return True
        return status >= 400
    return True


_HARDWARE_OUTPUT_ACTIONS = frozenset({"move_to_start", "start_hardware_replay"})


def _truthy_control_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _payload_requests_hardware_output(payload: Mapping[str, Any]) -> bool:
    action = str(payload.get("type") or payload.get("action") or "").strip()
    if action in _HARDWARE_OUTPUT_ACTIONS:
        return True
    if action == "set_send_actions":
        value = payload.get("enabled") if "enabled" in payload else payload.get("send_actions")
        return _truthy_control_value(value)
    if "send_actions" in payload:
        return _truthy_control_value(payload.get("send_actions"))
    if "sendActions" in payload:
        return _truthy_control_value(payload.get("sendActions"))
    return False


def _render_index_html(control_token: str) -> bytes:
    token_json = (
        json.dumps(str(control_token), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return (
        INDEX_HTML.replace("__REPLAY_CONTROL_TOKEN_JSON__", token_json)
        .replace(
            "__DEFAULT_POSITION_REFERENCE_HZ__",
            f"{DEFAULT_TELEOP_POSITION_REFERENCE_HZ:g}",
        )
        .encode("utf-8")
    )


def make_handler(
    session: ReplayGuiSession,
    *,
    security: WebControlSecurity,
    hardware_output_armed: bool = False,
    max_concurrent_control_requests: int = 4,
    access_log: bool = False,
) -> type[BaseHTTPRequestHandler]:
    if int(max_concurrent_control_requests) <= 0:
        raise ValueError("max_concurrent_control_requests must be positive")
    control_slots = threading.BoundedSemaphore(int(max_concurrent_control_requests))
    index_html = _render_index_html(security.token)

    class ReplayRequestHandler(BaseHTTPRequestHandler):
        server_version = "DexTeleopReplay/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            if not _should_log_http_message(bool(access_log), args):
                return
            sys.stderr.write("[ReplayGUI] " + format % args + "\n")

        def _send(
            self,
            status: int,
            body: bytes,
            *,
            content_type: str,
            cache: bool = False,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Any) -> None:
            self._send(status, _json_bytes(payload), content_type="application/json; charset=utf-8")

        def _send_error_json(self, status: int, exc: BaseException) -> None:
            self._send_json(status, {"ok": False, "error": str(exc)})

        def _server_address(self) -> tuple[str, int]:
            address = self.server.server_address
            return str(address[0]), int(address[1])

        def _validate_host(self) -> None:
            server_host, server_port = self._server_address()
            security.validate_host(
                self.headers,
                server_host=server_host,
                server_port=server_port,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                self._validate_host()
                if path == "/":
                    self._send(200, index_html, content_type="text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._send_json(200, {"ok": True, "state": session.state()})
                    return
                if path == "/api/position-collision-debug":
                    query = parse_qs(parsed.query, keep_blank_values=False)
                    raw_limit = query.get("limit", ["100"])[-1]
                    try:
                        limit = int(raw_limit)
                    except (TypeError, ValueError) as exc:
                        raise HttpRequestSecurityError(
                            400,
                            "position collision debug limit must be an integer",
                        ) from exc
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "position_collision_debug": (
                                session.position_collision_debug_snapshot(
                                    limit=limit
                                )
                            ),
                        },
                    )
                    return
                if path.startswith("/media/camera/") and path.endswith(".jpg"):
                    camera_key = unquote(path[len("/media/camera/") : -len(".jpg")])
                    self._send(200, session.camera_jpeg(camera_key), content_type="image/jpeg")
                    return
                if path.startswith("/media/tactile/") and path.endswith(".jpg"):
                    side = unquote(path[len("/media/tactile/") : -len(".jpg")])
                    self._send(200, session.tactile_jpeg(side), content_type="image/jpeg")
                    return
                self._send_json(404, {"ok": False, "error": f"not found: {path}"})
            except HttpRequestSecurityError as exc:
                self._send_error_json(exc.status, exc)
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(500, exc)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path != "/api/control":
                self._send_json(404, {"ok": False, "error": f"not found: {path}"})
                return
            try:
                server_host, server_port = self._server_address()
                length = security.validate_control_request(
                    self.headers,
                    server_host=server_host,
                    server_port=server_port,
                )
                body = self.rfile.read(length)
                if len(body) != length:
                    raise HttpRequestSecurityError(400, "incomplete control request body")
                payload = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise HttpRequestSecurityError(400, "control request body must be a JSON object")
                if _payload_requests_hardware_output(payload) and not bool(hardware_output_armed):
                    raise HttpRequestSecurityError(
                        403,
                        "hardware output is not armed by the local replay process",
                    )
                if not control_slots.acquire(blocking=False):
                    raise HttpRequestSecurityError(429, "too many concurrent control requests")
                try:
                    state = session.dispatch(payload)
                finally:
                    control_slots.release()
                self._send_json(200, {"ok": True, "state": state})
            except HttpRequestSecurityError as exc:
                self.close_connection = True
                self._send_error_json(exc.status, exc)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self.close_connection = True
                self._send_error_json(400, exc)
            except PermissionError as exc:
                self.close_connection = True
                self._send_error_json(403, exc)
            except RuntimeError as exc:
                self.close_connection = True
                self._send_error_json(409, exc)
            except Exception as exc:  # noqa: BLE001
                self.close_connection = True
                self._send_error_json(500, exc)

    return ReplayRequestHandler


def create_server(
    session: ReplayGuiSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    control_token: str | None = None,
    hardware_output_armed: bool = False,
    unsafe_allow_remote_web: bool = False,
    allowed_web_hosts: Sequence[str] = (),
    max_control_body_bytes: int = DEFAULT_MAX_CONTROL_BODY_BYTES,
    max_concurrent_control_requests: int = 4,
    access_log: bool = False,
) -> ThreadingHTTPServer:
    validate_bind_host(
        str(host),
        unsafe_allow_remote=bool(unsafe_allow_remote_web),
        allowed_hosts=tuple(str(value) for value in allowed_web_hosts),
    )
    security = WebControlSecurity(
        token=str(control_token or generate_session_token()),
        bind_host=str(host),
        allowed_hosts=tuple(str(value) for value in allowed_web_hosts),
        max_control_body_bytes=int(max_control_body_bytes),
    )
    server = ThreadingHTTPServer(
        (str(host), int(port)),
        make_handler(
            session,
            security=security,
            hardware_output_armed=bool(hardware_output_armed),
            max_concurrent_control_requests=int(max_concurrent_control_requests),
            access_log=bool(access_log),
        ),
    )
    server.daemon_threads = True
    server.replay_session = session  # type: ignore[attr-defined]
    server.replay_control_token = security.token  # type: ignore[attr-defined]
    server.replay_hardware_output_armed = bool(hardware_output_armed)  # type: ignore[attr-defined]
    return server


def run_server(
    *,
    dataset_root: str | Path | None = None,
    episode: str | int | None = None,
    hardware_config: ReplayHardwareConfig | None = None,
    execution_settings: ReplayExecutionSettings | None = None,
    send_actions: bool = False,
    end_effectors_enabled: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    start_playing: bool = False,
    hardware_output_armed: bool = False,
    unsafe_allow_remote_web: bool = False,
    allowed_web_hosts: Sequence[str] = (),
    access_log: bool = False,
    replay_action_shadow: bool = False,
    replay_action_shadow_profile: ReplayGuiShadowProfile | None = None,
) -> ThreadingHTTPServer:
    local_hardware_output_armed = bool(hardware_output_armed or send_actions)
    session = ReplayGuiSession(
        hardware_config=hardware_config,
        execution_settings=execution_settings,
        send_actions=bool(send_actions),
        end_effectors_enabled=bool(end_effectors_enabled),
        replay_action_shadow=bool(replay_action_shadow),
        replay_action_shadow_profile=replay_action_shadow_profile,
    )
    if dataset_root is not None and str(dataset_root):
        session.load_dataset(dataset_root, episode=episode)
    if start_playing:
        session.play()
    server = create_server(
        session,
        host=str(host),
        port=int(port),
        hardware_output_armed=local_hardware_output_armed,
        unsafe_allow_remote_web=bool(unsafe_allow_remote_web),
        allowed_web_hosts=allowed_web_hosts,
        access_log=bool(access_log),
    )
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"[ReplayGUI] serving {url}", flush=True)
    control_token = server.replay_control_token  # type: ignore[attr-defined]
    print(
        f"[ReplayGUI] control token ({CONTROL_TOKEN_HEADER}): {control_token}",
        flush=True,
    )
    if not local_hardware_output_armed:
        print("[ReplayGUI] hardware output is locally disarmed", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        session.close()
        server.server_close()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the DexTeleop dataset replay GUI.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root containing meta/episodes.jsonl and data/chunk-* parquet files.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episode index, episode id, or parquet path to select at startup.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port. Use 0 for any free port.")
    parser.add_argument(
        "--unsafe-allow-remote-web",
        action="store_true",
        help=(
            "Allow binding the replay control server beyond loopback. Unsafe: token-authenticated "
            "control traffic is plain HTTP. Wildcard binds also require --web-allowed-host."
        ),
    )
    parser.add_argument(
        "--web-allowed-host",
        action="append",
        default=[],
        help="Additional exact Host name or IP accepted by the replay server; may be repeated.",
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the GUI in the default browser after the server starts.",
    )
    parser.add_argument(
        "--start-playing",
        action="store_true",
        help="Start playback immediately after loading the selected episode.",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Log every HTTP request. By default only 4xx/5xx replay GUI requests are logged.",
    )
    parser.add_argument(
        "--replay-action-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the default-off passive replay ActionProposal/Arbiter shadow. "
            "It consumes the existing source-frame tap and never owns dispatch."
        ),
    )
    parser.add_argument(
        "--replay-action-shadow-trace-root",
        type=Path,
        default=None,
        help=(
            "Optional debug WAL root for passive replay shadow RuntimeTraceSession "
            "records; this is not a formal dataset episode."
        ),
    )
    parser.add_argument(
        "--replay-action-shadow-queue-size",
        type=int,
        default=4096,
        help="Bounded nonblocking passive replay emission queue capacity.",
    )
    parser.add_argument(
        "--replay-action-shadow-trace-capacity",
        type=int,
        default=65536,
        help="Maximum in-memory passive replay ControlTrace facts.",
    )
    parser.add_argument(
        "--replay-action-shadow-wal-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Maximum debug WAL size when a shadow trace root is configured.",
    )
    parser.add_argument(
        "--active-sides",
        choices=("left", "right", "both"),
        default="both",
        help=(
            "Arm side(s) used for hardware trajectory replay. Defaults to both because "
            "position collision planning is a coupled dual-arm safety contract."
        ),
    )
    parser.add_argument("--left-arm-ip", default="", help="RM65 left arm IP for hardware replay.")
    parser.add_argument("--right-arm-ip", default="", help="RM65 right arm IP for hardware replay.")
    parser.add_argument("--left-tool-profile", default="", help="Left arm tool profile used for TCP checks.")
    parser.add_argument("--right-tool-profile", default="", help="Right arm tool profile used for TCP checks.")
    parser.add_argument(
        "--arm-control-mode",
        choices=RM65_ARM_CONTROL_MODES,
        default=RM65_ARM_CONTROL_MODE_POSITION,
        help="RM65 replay execution mode. Position defaults to CANFD follow at 120Hz; impedance defaults to 40Hz.",
    )
    parser.add_argument(
        "--arm-control-hz",
        type=float,
        default=None,
        help="RM65 replay command/update rate. Defaults to 120Hz for position and 40Hz for impedance.",
    )
    parser.add_argument(
        "--position-reference-hz",
        type=float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help="Position replay pose-reference/IK rate between source actions and the command stream.",
    )
    parser.add_argument(
        "--position-control-profile",
        choices=RM65_POSITION_CONTROL_PROFILE_CHOICES,
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help=(
            "Position action-execution profile. Formal replay does not repeat "
            "collection-side source/qref shaping; legacy preserves the previous adapter."
        ),
    )
    parser.add_argument(
        "--position-workspace-min",
        type=float,
        nargs=3,
        default=(-0.60, -0.80, 0.00),
        metavar=("X", "Y", "Z"),
        help="Minimum xyz workspace for each 15Hz replay pose keyframe.",
    )
    parser.add_argument(
        "--position-workspace-max",
        type=float,
        nargs=3,
        default=(0.60, 0.80, 0.90),
        metavar=("X", "Y", "Z"),
        help="Maximum xyz workspace for each 15Hz replay pose keyframe.",
    )
    parser.add_argument(
        "--position-max-xyz-step-m",
        type=float,
        default=0.10,
        help="Maximum translation norm between consecutive safe 15Hz pose keyframes.",
    )
    parser.add_argument(
        "--position-max-rot-step-deg",
        type=float,
        default=15.0,
        help="Maximum shortest-path rotation between consecutive safe 15Hz pose keyframes.",
    )
    parser.add_argument(
        "--position-max-joint-step-rad",
        type=float,
        default=float(math.radians(6.0)),
        help="Maximum per-joint delta between consecutive safe 15Hz qpos keyframes.",
    )
    parser.add_argument(
        "--position-collision-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require coupled dual-arm collision IK for each high-rate pose reference. "
            "Enabled by default; disabling is allowed only for a non-motion legacy/debug dry run."
        ),
    )
    parser.add_argument(
        "--position-collision-scene-profile",
        default="dual_arm_hand_proxy_open",
        help="Dual-arm collision scene used by the high-rate pose-to-qpos planner.",
    )
    parser.add_argument(
        "--position-collision-min-distance-m",
        type=float,
        default=0.03,
        help="Minimum accepted collision distance at each limited high-rate qpos endpoint.",
    )
    parser.add_argument(
        "--position-collision-distance-epsilon-m",
        type=float,
        default=1e-4,
        help="Numerical tolerance applied to the high-rate endpoint collision threshold.",
    )
    parser.add_argument(
        "--position-collision-max-joint-velocity-rad-s",
        default=",".join(
            str(value)
            for value in ReplayExecutionSettings().position_collision_max_joint_velocity_rad_s
        ),
        help=(
            "One or six per-joint safety velocity limits used at the high-rate IK stage. "
            "Defaults to the rm65_position_2x profile."
        ),
    )
    parser.add_argument(
        "--position-collision-max-joint-step-rad",
        default="",
        help=(
            "Optional one or six per-reference joint-step limits at the high-rate IK stage; "
            "empty uses velocity/reference_hz only."
        ),
    )
    parser.add_argument(
        "--position-collision-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach raw IK qpos/pose arrays and complete planner diagnostics to each command.",
    )
    parser.add_argument(
        "--arm-canfd-follow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send RM65 replay CANFD commands with SDK follow=True. Defaults on only for position mode.",
    )
    parser.add_argument("--force-tare-sec", type=float, default=3.0, help="Tare duration for impedance replay.")
    parser.add_argument(
        "--strict-tool-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require confirmed TCP/payload/impedance profile for impedance replay.",
    )
    parser.add_argument(
        "--replay-arm-max-angvel",
        default="3,3,3,4,4,4",
        help="Comma-separated 6D joint velocity limits while replay is active.",
    )
    parser.add_argument(
        "--qpos-init-arm-max-angvel",
        default="",
        help=(
            "Optional first-frame QPOS init joint velocity limit in rad/s. "
            "Accepts one value or six comma-separated values; empty keeps the configured velocity profile."
        ),
    )
    parser.add_argument(
        "--singularity-qpos-init-slowdown",
        type=_parse_unit_interval_float,
        default=DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
        help=(
            "Scale joint steps when replay move-to-start QPOS init crosses RM65 singularity. "
            "Must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help=(
            "Locally arm and enable robot arm command output at startup. Defaults to disabled; "
            "this process-side flag cannot be supplied through the web API."
        ),
    )
    parser.add_argument(
        "--arm-web-hardware-output",
        action="store_true",
        help=(
            "Locally authorize this web session to enable hardware output later, while leaving "
            "send_actions disabled at startup."
        ),
    )
    parser.add_argument(
        "--connect-hardware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Connect to real RM65 hardware. Use --no-connect-hardware for dry-run checks.",
    )
    parser.add_argument(
        "--end-effectors-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay active-side end-effector targets together with arm actions.",
    )
    parser.add_argument(
        "--hand-backend",
        choices=(
            "none",
            "inspire_tac",
            "inspire-tac",
            "dmtac_gripper",
            "dmtac-gripper",
        ),
        default="none",
        help="Default active-side end-effector backend for hardware replay.",
    )
    parser.add_argument("--left-hand-backend", default="", help="Optional left end-effector backend override.")
    parser.add_argument("--right-hand-backend", default="", help="Optional right end-effector backend override.")
    parser.add_argument("--left-hand-ip", default="", help="Left Inspire hand IP for end-effector replay.")
    parser.add_argument("--right-hand-ip", default="", help="Right Inspire hand IP for end-effector replay.")
    parser.add_argument("--left-hand-port", type=int, default=6000, help="Left Inspire hand port.")
    parser.add_argument("--right-hand-port", type=int, default=6000, help="Right Inspire hand port.")
    add_dmtac_assignment_arguments(parser, assignment_includes_tactile=False)
    parser.add_argument("--gripper-speed", type=float, default=1.0, help="DMTAC gripper speed for replay.")
    parser.add_argument("--gripper-torque", type=float, default=1.0, help="DMTAC gripper torque/force for replay.")
    parser.add_argument("--gripper-max-action", type=float, default=1.0, help="DMTAC max normalized close command.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    active_sides = ("left", "right") if args.active_sides == "both" else (str(args.active_sides),)
    end_effector_backend_by_side = {
        "left": str(args.left_hand_backend or args.hand_backend or "none"),
        "right": str(args.right_hand_backend or args.hand_backend or "none"),
    }
    dmtac_fields = _resolve_replay_dmtac_fields(
        args,
        active_sides=active_sides,
        end_effector_backend_by_side=end_effector_backend_by_side,
    )
    hardware_config = ReplayHardwareConfig(
        active_sides=active_sides,
        arm_ip_by_side={
            "left": str(args.left_arm_ip or ""),
            "right": str(args.right_arm_ip or ""),
        },
        tool_profile_by_side={
            "left": str(args.left_tool_profile or ""),
            "right": str(args.right_tool_profile or ""),
        },
        end_effectors_enabled=bool(args.end_effectors_enabled),
        end_effector_backend_by_side=end_effector_backend_by_side,
        hand_ip_by_side={
            "left": str(args.left_hand_ip or ""),
            "right": str(args.right_hand_ip or ""),
        },
        hand_port_by_side={
            "left": int(args.left_hand_port),
            "right": int(args.right_hand_port),
        },
        gripper_port_by_side=dict(dmtac_fields["port"]),
        gripper_id_by_side=dict(dmtac_fields["id"]),
        gripper_baudrate_by_side=dict(dmtac_fields["baudrate"]),
        gripper_timeout_by_side=dict(dmtac_fields["timeout"]),
        gripper_speed=float(args.gripper_speed),
        gripper_torque=float(args.gripper_torque),
        gripper_max_action=float(args.gripper_max_action),
        arm_control_mode=str(args.arm_control_mode),
        arm_control_hz=args.arm_control_hz,
        position_reference_hz=float(args.position_reference_hz),
        arm_canfd_follow=args.arm_canfd_follow,
        force_tare_sec=float(args.force_tare_sec),
        strict_tool_profile=bool(args.strict_tool_profile),
        replay_arm_max_angvel=_parse_float_csv(
            args.replay_arm_max_angvel,
            default=(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
        ),
        singularity_qpos_init_slowdown=float(args.singularity_qpos_init_slowdown),
        connect_hardware=bool(args.connect_hardware),
    )
    execution_settings = ReplayExecutionSettings(
        end_effectors_enabled=bool(args.end_effectors_enabled),
        qpos_init_max_angvel_rad_s=normalize_qpos_init_max_angvel(args.qpos_init_arm_max_angvel),
        position_workspace_min=tuple(float(value) for value in args.position_workspace_min),
        position_workspace_max=tuple(float(value) for value in args.position_workspace_max),
        position_max_xyz_step_m=float(args.position_max_xyz_step_m),
        position_max_rot_step_deg=float(args.position_max_rot_step_deg),
        position_max_joint_step_rad=float(args.position_max_joint_step_rad),
        position_control_profile_name=str(args.position_control_profile),
        position_collision_required=bool(args.position_collision_required),
        position_collision_scene_profile=str(args.position_collision_scene_profile),
        position_collision_min_distance_m=float(
            args.position_collision_min_distance_m
        ),
        position_collision_distance_epsilon_m=float(
            args.position_collision_distance_epsilon_m
        ),
        position_collision_max_joint_velocity_rad_s=tuple(
            _parse_one_or_six_float_csv(
                args.position_collision_max_joint_velocity_rad_s
            )
            or ()
        ),
        position_collision_max_joint_step_rad=_parse_one_or_six_float_csv(
            args.position_collision_max_joint_step_rad,
            optional=True,
        ),
        position_collision_debug=bool(args.position_collision_debug),
    )
    run_server(
        dataset_root=args.dataset_root,
        episode=args.episode,
        hardware_config=hardware_config,
        execution_settings=execution_settings,
        send_actions=bool(args.send_actions),
        end_effectors_enabled=bool(args.end_effectors_enabled),
        host=args.host,
        port=args.port,
        open_browser=bool(args.open_browser),
        start_playing=bool(args.start_playing),
        hardware_output_armed=bool(args.arm_web_hardware_output),
        unsafe_allow_remote_web=bool(args.unsafe_allow_remote_web),
        allowed_web_hosts=tuple(args.web_allowed_host),
        access_log=bool(args.access_log),
        replay_action_shadow=bool(args.replay_action_shadow),
        replay_action_shadow_profile=(
            ReplayGuiShadowProfile(
                queue_capacity=int(args.replay_action_shadow_queue_size),
                trace_capacity=int(args.replay_action_shadow_trace_capacity),
                trace_root=args.replay_action_shadow_trace_root,
                wal_max_bytes=int(args.replay_action_shadow_wal_max_bytes),
            )
            if args.replay_action_shadow
            else None
        ),
    )


if __name__ == "__main__":
    main()
