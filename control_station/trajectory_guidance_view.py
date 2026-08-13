"""Presentation fragment for the standalone continuous-guidance experiment."""

from __future__ import annotations

from .trajectory_guidance_webgl import TRAJECTORY_GUIDANCE_WEBGL_JS


TRAJECTORY_GUIDANCE_VIEW_CSS = r"""
    .trajectory-panel { margin-bottom:16px; padding:15px; border:1px solid var(--line); border-radius:14px; background:rgba(17,24,39,.9); box-shadow:var(--shadow); }
    .trajectory-scene-wrap { position:relative; }
    .trajectory-scene { display:block; width:100%; height:420px; border:1px solid #33435f; border-radius:10px; background:#09101f; cursor:grab; touch-action:none; }
    .trajectory-scene:active { cursor:grabbing; }
    .trajectory-axis-labels { position:absolute; inset:0 0 auto 0; height:420px; pointer-events:none; overflow:hidden; }
    .trajectory-axis-label { position:absolute; z-index:2; transform:translate(-50%,-50%); padding:2px 5px; border-radius:5px; background:rgba(5,12,24,.82); color:var(--axis-color); font:600 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:nowrap; }
    .trajectory-axis-label.x { --axis-color:#f5525c; } .trajectory-axis-label.y { --axis-color:#4ddb7a; } .trajectory-axis-label.z { --axis-color:#4d8fff; }
    .trajectory-controls { display:grid; gap:10px; margin-top:12px; }
    .trajectory-control { display:grid; grid-template-columns:minmax(160px,1fr) minmax(180px,1.3fr) 74px; gap:10px; align-items:center; color:var(--muted); font-size:12px; }
    .trajectory-control input,.trajectory-control select { width:100%; }
    .trajectory-control select { min-height:34px; padding:4px 8px; border:1px solid var(--line); border-radius:7px; background:#0d1a2d; color:var(--text); }
    .trajectory-clear { justify-self:start; min-height:34px; }
    .trajectory-metrics { display:grid; gap:5px; margin-top:10px; color:var(--muted); font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
    .trajectory-legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; color:var(--muted); font-size:11px; }
    .trajectory-legend span::before { content:''; display:inline-block; width:18px; height:3px; margin-right:5px; vertical-align:middle; background:var(--swatch); }
    .trajectory-legend .trajectory-nominal { --swatch:#94b0d6; }
    .trajectory-legend .trajectory-corrected { --swatch:#5cf096; }
    .trajectory-formula { margin-top:10px; padding:9px 10px; border:1px solid #2d415f; border-radius:9px; background:#0b1424; color:#b8c6da; font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
    @media (max-width:760px) { .trajectory-scene,.trajectory-axis-labels { height:330px; } .trajectory-control { grid-template-columns:1fr; } }
"""


TRAJECTORY_GUIDANCE_VIEW_HTML = r"""
    <section id="trajectory-guidance-panel" class="trajectory-panel" aria-label="持续引导轨迹实验投影">
      <div class="card-head">
        <div>
          <h2>持续修正轨迹（实验投影）</h2>
          <p>灰蓝原始轨迹和绿色修正轨迹都以 30 Hz 连续实时运行；每帧绿色轨迹读取左侧最新 Home 偏转。</p>
        </div>
        <span id="trajectory-guidance-status" class="label">等待输入</span>
      </div>
      <div class="trajectory-scene-wrap">
        <canvas id="trajectory-guidance-scene" class="trajectory-scene" role="img" aria-label="以米为单位的三维原始与修正轨迹"></canvas>
        <div id="trajectory-axis-labels" class="trajectory-axis-labels" aria-hidden="true">
          <span class="trajectory-axis-label x" data-axis-direction="positive_x" data-axis-name="+X">+X</span>
          <span class="trajectory-axis-label x" data-axis-direction="negative_x" data-axis-name="−X">−X</span>
          <span class="trajectory-axis-label y" data-axis-direction="positive_y" data-axis-name="+Y">+Y</span>
          <span class="trajectory-axis-label y" data-axis-direction="negative_y" data-axis-name="−Y">−Y</span>
          <span class="trajectory-axis-label z" data-axis-direction="positive_z" data-axis-name="+Z">+Z</span>
          <span class="trajectory-axis-label z" data-axis-direction="negative_z" data-axis-name="−Z">−Z</span>
        </div>
      </div>
      <div class="trajectory-legend">
        <span class="trajectory-nominal">原始轨迹 [m]</span>
        <span class="trajectory-corrected">修正轨迹 [m]</span>
        <span id="trajectory-axis-contract">网格间距 0.05 m · 轨迹刷新 30 Hz · 等待坐标语义</span>
      </div>
      <div class="trajectory-controls">
        <label class="trajectory-control" for="trajectory-guidance-mode">
          <span>修正模式</span>
          <select id="trajectory-guidance-mode">
            <option value="velocity_joystick" selected>持续 joystick</option>
            <option value="position_offset">固定位置偏移</option>
          </select>
          <output id="trajectory-guidance-mode-value">joystick</output>
        </label>
        <label class="trajectory-control" for="trajectory-guidance-gain">
          <span>满量程修正速度 [m/s]</span>
          <input id="trajectory-guidance-gain" type="range" min="0" max="0.20" step="0.005" value="0.08">
          <output id="trajectory-guidance-gain-value">0.080 m/s</output>
        </label>
        <label class="trajectory-control" for="trajectory-guidance-deadzone">
          <span>joystick Home 死区 [m]</span>
          <input id="trajectory-guidance-deadzone" type="range" min="0" max="0.030" step="0.001" value="0.005">
          <output id="trajectory-guidance-deadzone-value">0.005 m</output>
        </label>
        <label class="trajectory-control" for="trajectory-position-gain">
          <span>位置偏移比例 [m/m]</span>
          <input id="trajectory-position-gain" type="range" min="0" max="2.0" step="0.05" value="1.0">
          <output id="trajectory-position-gain-value">1.00 m/m</output>
        </label>
        <button id="trajectory-guidance-clear" class="trajectory-clear" type="button">清除累计修正</button>
      </div>
      <div id="trajectory-guidance-metrics" class="trajectory-metrics"></div>
      <div id="trajectory-guidance-formula" class="trajectory-formula"></div>
      <div class="deflection-legend">这是 demo-only、无控制权的浏览器投影；不写回 composition，不是 executable Reference，也不做 sim/robot 坐标变换。当前仅评估平移持续引导，左侧旋转偏转仍只显示。</div>
    </section>
"""


TRAJECTORY_GUIDANCE_VIEW_JS = TRAJECTORY_GUIDANCE_WEBGL_JS + r"""
    let trajectoryGuidanceScene = null;
    let trajectoryGuidanceLastSampleToken = null;
    let trajectoryGuidanceLastMetricText = null;
    let trajectoryGuidanceMetricTimer = null;

    const formatVector = (values, digits, unit) => values.map(value =>
      `${Number(value).toFixed(digits)} ${unit}`
    ).join(', ');

    function initializeTrajectoryGuidanceView() {
      const canvas = $('trajectory-guidance-scene');
      try {
        trajectoryGuidanceScene = new TrajectoryGuidanceScene(
          canvas, $('trajectory-axis-labels')
        );
      } catch (error) {
        $('trajectory-guidance-status').textContent = `WebGL 不可用：${error}`;
        return;
      }
      const gain = $('trajectory-guidance-gain');
      const deadzone = $('trajectory-guidance-deadzone');
      const positionGain = $('trajectory-position-gain');
      const mode = $('trajectory-guidance-mode');
      const applyGain = () => {
        const value = Number(gain.value);
        trajectoryGuidanceScene.setMaxGuidanceSpeed(value);
        $('trajectory-guidance-gain-value').textContent = `${value.toFixed(3)} m/s`;
        renderTrajectoryGuidanceMetrics();
      };
      const applyDeadzone = () => {
        const value = Number(deadzone.value);
        trajectoryGuidanceScene.setDeadzoneMeters(value);
        $('trajectory-guidance-deadzone-value').textContent =
          `${value.toFixed(3)} m`;
        renderTrajectoryGuidanceMetrics();
      };
      const applyPositionGain = () => {
        const value = Number(positionGain.value);
        trajectoryGuidanceScene.setPositionGain(value);
        $('trajectory-position-gain-value').textContent =
          `${value.toFixed(2)} m/m`;
        renderTrajectoryGuidanceMetrics();
      };
      const applyMode = () => {
        trajectoryGuidanceScene.setMode(mode.value);
        const joystick = mode.value === 'velocity_joystick';
        gain.disabled = !joystick;
        deadzone.disabled = !joystick;
        positionGain.disabled = joystick;
        $('trajectory-guidance-clear').disabled = !joystick;
        $('trajectory-guidance-mode-value').textContent = joystick
          ? 'joystick' : 'position';
        $('trajectory-guidance-status').textContent = `模式 ${mode.value}`;
        $('trajectory-guidance-formula').textContent = joystick
          ? '严格 SI / joystick：u = deadzone_clamp(Δp / 0.12 m)；v = u × gain [m/s]；offset(t+dt) = offset(t) + v × dt；p修正(t,τ) = p原始(t,τ) + offset(t) + τ × v。回 Home 后 v=0 并保留 offset；清除或 rehome 才清零。'
          : '严格 SI / position：Δp修正 [m] = gain [m/m] × Δp人手 [m]；p修正(t,τ) = p原始(t,τ) + Δp修正。该模式不积分；回 Home 后偏移归零。';
        trajectoryGuidanceLastMetricText = null;
        renderTrajectoryGuidanceMetrics();
      };
      gain.addEventListener('input', applyGain);
      deadzone.addEventListener('input', applyDeadzone);
      positionGain.addEventListener('input', applyPositionGain);
      mode.addEventListener('change', applyMode);
      $('trajectory-guidance-clear').addEventListener('click', () => {
        trajectoryGuidanceScene.clearIntegratedOffset();
        trajectoryGuidanceLastMetricText = null;
        renderTrajectoryGuidanceMetrics();
      });
      applyGain();
      applyDeadzone();
      applyPositionGain();
      applyMode();
      trajectoryGuidanceMetricTimer = window.setInterval(
        renderTrajectoryGuidanceMetrics, 100
      );
    }

    function renderTrajectoryGuidanceMetrics() {
      const target = $('trajectory-guidance-metrics');
      if (!trajectoryGuidanceScene) return;
      const correction = trajectoryGuidanceScene.correctionState();
      const velocity = correction.futureVelocityMps;
      const terminalShift = correction.offsetM.map(
        (value, index) => value
          + velocity[index] * trajectoryGuidanceScene.horizonSec
      );
      const lines = [
        `模式 = ${trajectoryGuidanceScene.mode}`,
        `当前累计/位置偏移 XYZ = [${formatVector(correction.offsetM, 3, 'm')}]`,
        `当前修正速度 XYZ = [${formatVector(velocity, 3, 'm/s')}]`,
        `2.0 s 末端位移差 XYZ = [${formatVector(terminalShift, 3, 'm')}]`,
        `|v修正| = ${Math.hypot(...velocity).toFixed(3)} m/s`,
      ];
      const metricText = lines.join('\n');
      if (metricText === trajectoryGuidanceLastMetricText) return;
      trajectoryGuidanceLastMetricText = metricText;
      target.replaceChildren();
      for (const line of lines) {
        const row = document.createElement('div');
        row.textContent = line;
        target.append(row);
      }
    }

    function updateTrajectoryGuidance(deflection, sourceSampleToken = null) {
      if (!trajectoryGuidanceScene) return;
      const components = Array.isArray(deflection && deflection.components)
        ? deflection.components : [];
      const component = components.find(item =>
        Array.isArray(item.translation_xyz_m)
        && item.translation_xyz_m.length === 3
        && item.translation_xyz_m.every(value => Number.isFinite(Number(value)))
      );
      if (!component) {
        if (trajectoryGuidanceLastSampleToken !== null) {
          trajectoryGuidanceScene.setDeflectionMeters([0, 0, 0]);
          $('trajectory-guidance-status').textContent =
            trajectoryGuidanceScene.mode === 'velocity_joystick'
              ? '无新鲜输入 · 停止累积并保持现有 offset'
              : '无新鲜输入 · 位置偏移归零';
        }
        trajectoryGuidanceLastSampleToken = null;
        renderTrajectoryGuidanceMetrics();
        return;
      }
      const axisSemantics = component.translation_axis_semantics || {};
      trajectoryGuidanceScene.setAxisSemantics(axisSemantics);
      const calibratedAxes = axisSemantics.positive_x === 'forward'
        && axisSemantics.positive_y === 'left'
        && axisSemantics.positive_z === 'up';
      const physicallyLabelledAxes = [
        'positive_x', 'negative_x', 'positive_y',
        'negative_y', 'positive_z', 'negative_z',
      ].every(key => Boolean(axisSemantics[key]));
      $('trajectory-axis-contract').textContent = calibratedAxes
        ? '网格间距 0.05 m · 轨迹刷新 30 Hz · +X 前 / +Y 左 / +Z 上'
        : physicallyLabelledAxes
          ? '网格间距 0.05 m · 轨迹刷新 30 Hz · 按 profile 实际物理方向标注（非 canonical）'
        : '网格间距 0.05 m · 轨迹刷新 30 Hz · 原始帧仅标记 ±X/±Y/±Z';
      const token = sourceSampleToken === null || sourceSampleToken === undefined
        ? JSON.stringify(component.translation_xyz_m) : String(sourceSampleToken);
      if (trajectoryGuidanceLastSampleToken !== token) {
        trajectoryGuidanceScene.setDeflectionMeters(component.translation_xyz_m);
        trajectoryGuidanceLastSampleToken = token;
        $('trajectory-guidance-status').textContent =
          `${text(component.component_id)} · ${trajectoryGuidanceScene.mode}`;
        renderTrajectoryGuidanceMetrics();
      }
    }

    window.addEventListener('beforeunload', () => {
      if (trajectoryGuidanceMetricTimer !== null) {
        window.clearInterval(trajectoryGuidanceMetricTimer);
      }
      if (trajectoryGuidanceScene) trajectoryGuidanceScene.dispose();
    });
    initializeTrajectoryGuidanceView();
"""


__all__ = [
    "TRAJECTORY_GUIDANCE_VIEW_CSS",
    "TRAJECTORY_GUIDANCE_VIEW_HTML",
    "TRAJECTORY_GUIDANCE_VIEW_JS",
]
