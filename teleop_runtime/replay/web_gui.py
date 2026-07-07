from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .arm_init.qpos_execute import normalize_qpos_init_max_angvel
from .executor import ReplayExecutionSettings
from .hardware import DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN, ReplayHardwareConfig
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
              <span class="label">Hz</span>
              <input id="armControlHz" type="number" min="1" step="1" value="50">
            </div>
            <div class="form-row">
              <span class="label">QPOS Init Vel</span>
              <input id="qposInitArmMaxAngvel" autocomplete="off" spellcheck="false" placeholder="profile">
            </div>
            <label class="check-row">
              <input id="sendActions" type="checkbox">
              <span>Arm Output</span>
            </label>
            <label class="check-row">
              <input id="connectHardware" type="checkbox" checked>
              <span>Connect Hardware</span>
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
      armControlHz: document.getElementById("armControlHz"),
      qposInitArmMaxAngvel: document.getElementById("qposInitArmMaxAngvel"),
      sendActions: document.getElementById("sendActions"),
      connectHardware: document.getElementById("connectHardware"),
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
        headers: payload ? {"Content-Type": "application/json"} : {},
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

    function hardwarePayload() {
      return {
        type: "configure_hardware",
        active_sides: els.activeSides.value,
        left_arm_ip: els.leftArmIp.value.trim(),
        right_arm_ip: els.rightArmIp.value.trim(),
        left_tool_profile: els.leftToolProfile.value.trim(),
        right_tool_profile: els.rightToolProfile.value.trim(),
        arm_control_hz: Number(els.armControlHz.value || 50),
        qpos_init_max_angvel_rad_s: els.qposInitArmMaxAngvel.value.trim(),
        send_actions: els.sendActions.checked,
        connect_hardware: els.connectHardware.checked,
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
      fieldValue(els.activeSides, activeSidesValue(config.active_sides || execution.active_sides || ["right"]));
      fieldValue(els.leftArmIp, ips.left || "");
      fieldValue(els.rightArmIp, ips.right || "");
      fieldValue(els.leftToolProfile, profiles.left || "");
      fieldValue(els.rightToolProfile, profiles.right || "");
      fieldValue(els.armControlHz, config.arm_control_hz || 50);
      fieldValue(
        els.qposInitArmMaxAngvel,
        Array.isArray(settings.qpos_init_max_angvel_rad_s)
          ? settings.qpos_init_max_angvel_rad_s.join(",")
          : ""
      );
      if (document.activeElement !== els.sendActions) els.sendActions.checked = !!settings.send_actions_enabled;
      if (document.activeElement !== els.connectHardware) els.connectHardware.checked = config.connect_hardware !== false;
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
          els.armControlHz,
          els.qposInitArmMaxAngvel,
          els.connectHardware,
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


def _parse_unit_interval_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


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


def make_handler(
    session: ReplayGuiSession,
    *,
    access_log: bool = False,
) -> type[BaseHTTPRequestHandler]:
    class ReplayRequestHandler(BaseHTTPRequestHandler):
        server_version = "DexTeleopReplay/1.0"

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
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Any) -> None:
            self._send(status, _json_bytes(payload), content_type="application/json; charset=utf-8")

        def _send_error_json(self, status: int, exc: BaseException) -> None:
            self._send_json(status, {"ok": False, "error": str(exc)})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/":
                    self._send(200, INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._send_json(200, {"ok": True, "state": session.state()})
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
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(500, exc)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(min(length, 1_000_000))
                payload = json.loads(body.decode("utf-8") or "{}")
                if path == "/api/control":
                    self._send_json(200, {"ok": True, "state": session.dispatch(payload)})
                    return
                self._send_json(404, {"ok": False, "error": f"not found: {path}"})
            except Exception as exc:  # noqa: BLE001
                self._send_error_json(400, exc)

    return ReplayRequestHandler


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
    access_log: bool = False,
) -> ThreadingHTTPServer:
    session = ReplayGuiSession(
        hardware_config=hardware_config,
        execution_settings=execution_settings,
        send_actions=bool(send_actions),
        end_effectors_enabled=bool(end_effectors_enabled),
    )
    if dataset_root is not None and str(dataset_root):
        session.load_dataset(dataset_root, episode=episode)
    if start_playing:
        session.play()
    server = ThreadingHTTPServer((str(host), int(port)), make_handler(session, access_log=bool(access_log)))
    server.replay_session = session  # type: ignore[attr-defined]
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"[ReplayGUI] serving {url}", flush=True)
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
        "--active-sides",
        choices=("left", "right", "both"),
        default="right",
        help="Arm side(s) used for hardware trajectory replay.",
    )
    parser.add_argument("--left-arm-ip", default="", help="RM65 left arm IP for hardware replay.")
    parser.add_argument("--right-arm-ip", default="", help="RM65 right arm IP for hardware replay.")
    parser.add_argument("--left-tool-profile", default="", help="Left arm tool profile used for TCP checks.")
    parser.add_argument("--right-tool-profile", default="", help="Right arm tool profile used for TCP checks.")
    parser.add_argument("--arm-control-hz", type=float, default=50.0, help="RM65 replay command/update rate.")
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
        help="Enable robot arm command output at startup. Defaults to disabled.",
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
        choices=("none", "inspire_tac", "dmtac_gripper"),
        default="none",
        help="Default active-side end-effector backend for hardware replay.",
    )
    parser.add_argument("--left-hand-backend", default="", help="Optional left end-effector backend override.")
    parser.add_argument("--right-hand-backend", default="", help="Optional right end-effector backend override.")
    parser.add_argument("--left-hand-ip", default="", help="Left Inspire hand IP for end-effector replay.")
    parser.add_argument("--right-hand-ip", default="", help="Right Inspire hand IP for end-effector replay.")
    parser.add_argument("--left-hand-port", type=int, default=6000, help="Left Inspire hand port.")
    parser.add_argument("--right-hand-port", type=int, default=6000, help="Right Inspire hand port.")
    parser.add_argument("--left-gripper-port", default="", help="Left DMTAC gripper serial port.")
    parser.add_argument("--right-gripper-port", default="", help="Right DMTAC gripper serial port.")
    parser.add_argument("--left-gripper-id", type=int, default=1, help="Left DMTAC gripper Modbus id.")
    parser.add_argument("--right-gripper-id", type=int, default=1, help="Right DMTAC gripper Modbus id.")
    parser.add_argument("--gripper-speed", type=float, default=1.0, help="DMTAC gripper speed for replay.")
    parser.add_argument("--gripper-torque", type=float, default=1.0, help="DMTAC gripper torque/force for replay.")
    parser.add_argument("--gripper-max-action", type=float, default=1.0, help="DMTAC max normalized close command.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    active_sides = ("left", "right") if args.active_sides == "both" else (str(args.active_sides),)
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
        end_effector_backend_by_side={
            "left": str(args.left_hand_backend or args.hand_backend or "none"),
            "right": str(args.right_hand_backend or args.hand_backend or "none"),
        },
        hand_ip_by_side={
            "left": str(args.left_hand_ip or ""),
            "right": str(args.right_hand_ip or ""),
        },
        hand_port_by_side={
            "left": int(args.left_hand_port),
            "right": int(args.right_hand_port),
        },
        gripper_port_by_side={
            "left": str(args.left_gripper_port or ""),
            "right": str(args.right_gripper_port or ""),
        },
        gripper_id_by_side={
            "left": int(args.left_gripper_id),
            "right": int(args.right_gripper_id),
        },
        gripper_speed=float(args.gripper_speed),
        gripper_torque=float(args.gripper_torque),
        gripper_max_action=float(args.gripper_max_action),
        arm_control_hz=float(args.arm_control_hz),
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
        access_log=bool(args.access_log),
    )


if __name__ == "__main__":
    main()
