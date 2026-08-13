"""Reusable presentation fragment for a read-only pose-deflection view."""

from __future__ import annotations

from .relative_deflection_webgl import RELATIVE_DEFLECTION_WEBGL_JS


RELATIVE_DEFLECTION_VIEW_CSS = r"""
    .deflection-panel { margin-bottom: 16px; padding: 15px; border: 1px solid var(--line); border-radius: 14px; background: rgba(17,24,39,.9); box-shadow: var(--shadow); }
    .deflection-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }
    .deflection-empty { padding: 18px; border: 1px dashed var(--line); border-radius: 12px; color: var(--muted); text-align: center; }
    .deflection-card { display: grid; grid-template-columns: minmax(300px,1.35fr) minmax(220px,1fr); gap: 14px; align-items: center; padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(10,15,28,.48); }
    .deflection-scene-wrap { position:relative; min-width:0; }
    .deflection-scene { display: block; width: 100%; height: 330px; border: 1px solid #33435f; border-radius: 10px; background: #09101f; cursor: grab; touch-action: none; }
    .deflection-scene:active { cursor: grabbing; }
    .deflection-scene-help { margin-top: 6px; color: var(--muted); font-size: 11px; text-align: center; }
    .deflection-axis-labels { position:absolute; inset:0 0 auto 0; height:330px; pointer-events:none; overflow:hidden; }
    .deflection-axis-label { position:absolute; z-index:2; transform:translate(-50%,-50%); padding:2px 5px; border:1px solid color-mix(in srgb,var(--axis-color) 55%,transparent); border-radius:5px; background:rgba(5,12,24,.82); color:var(--axis-color); font:600 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:nowrap; }
    .deflection-axis-label.x { --axis-color:#f5525c; } .deflection-axis-label.y { --axis-color:#4ddb7a; } .deflection-axis-label.z { --axis-color:#4d8fff; }
    .deflection-webgl-error { display: grid; place-items: center; min-height: 250px; padding: 16px; border: 1px dashed #8d5260; border-radius: 10px; color: #ff9daf; text-align: center; }
    .deflection-card h3 { margin: 0 0 8px; color: var(--blue); font-size: 14px; }
    .deflection-values { display: grid; gap: 6px; color: var(--muted); font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .deflection-values strong { color: var(--text); }
    .deflection-legend { margin-top: 7px; color: var(--muted); font-size: 11px; }
    @media (max-width: 760px) { .deflection-card { grid-template-columns: 1fr; } .deflection-scene,.deflection-axis-labels { height:290px; } }
"""


RELATIVE_DEFLECTION_VIEW_HTML = r"""
    <section id="human-deflection-panel" class="deflection-panel" aria-label="人类相对 Home 偏转">
      <div class="card-head">
        <div>
          <h2>输入相对 Home 偏转</h2>
          <p>灰球为当前显示来源的 Home；绿球在三维空间中显示实时偏转，球上的 RGB 坐标轴显示局部旋转。</p>
        </div>
        <span id="human-deflection-status" class="label">等待输入</span>
      </div>
      <div id="human-deflection-views" class="deflection-grid">
        <div class="deflection-empty">等待有效 pose。</div>
      </div>
      <div class="deflection-legend">真正的三维视图：固定坐标轴端点按 profile 的实际物理方向标注；canonical profile 为 +X 前、+Y 左、+Z 上，显式 post-yaw profile 会显示翻转后的有符号轴，raw debug 模式只标记 ±X/±Y/±Z。球上的 RGB 短轴显示局部旋转；蓝线连接 Home 与偏转球。拖动旋转视角，滚轮缩放。显示量程为 120 mm，超量程只缩放显示、不修改原始数值。该视图只读，不参与控制。</div>
    </section>
"""


RELATIVE_DEFLECTION_VIEW_JS = RELATIVE_DEFLECTION_WEBGL_JS + r"""
    const relativeDeflectionEntries = new Map();
    const disposeRelativeDeflectionEntries = () => {
      for (const entry of relativeDeflectionEntries.values()) entry.scene.dispose();
      relativeDeflectionEntries.clear();
    };
    const numericVector = (value) => Array.isArray(value)
      ? value.map(item => Number(item)) : [];
    const formattedDeflectionVector = (values, scale, digits) => values.map(item =>
      Number.isFinite(item) ? (item * scale).toFixed(digits) : '—'
    ).join(', ');
    const makeRelativeDeflectionEntry = (componentId) => {
      const card = document.createElement('article');
      card.className = 'deflection-card';
      const sceneWrap = document.createElement('div');
      sceneWrap.className = 'deflection-scene-wrap';
      const canvas = document.createElement('canvas');
      canvas.className = 'deflection-scene';
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `${componentId} 3D Home deflection`);
      const help = document.createElement('div');
      help.className = 'deflection-scene-help';
      help.textContent = '拖动：旋转视角 · 滚轮：缩放';
      const axisLabels = document.createElement('div');
      axisLabels.className = 'deflection-axis-labels';
      const axisDefinitions = [
        ['positive_x', '+X', 'x'], ['negative_x', '−X', 'x'],
        ['positive_y', '+Y', 'y'], ['negative_y', '−Y', 'y'],
        ['positive_z', '+Z', 'z'], ['negative_z', '−Z', 'z'],
      ];
      for (const [direction, name, axisClass] of axisDefinitions) {
        const label = document.createElement('span');
        label.className = `deflection-axis-label ${axisClass}`;
        label.dataset.axisDirection = direction;
        label.dataset.axisName = name;
        label.textContent = name;
        axisLabels.append(label);
      }
      sceneWrap.append(canvas, axisLabels, help);
      const details = document.createElement('div');
      const title = document.createElement('h3');
      title.textContent = componentId;
      const values = document.createElement('div');
      values.className = 'deflection-values';
      details.append(title, values);
      card.append(sceneWrap, details);
      try {
        return {
          card,
          values,
          scene: new PoseDeflectionScene(canvas, axisLabels),
          lastSampleToken: null,
        };
      } catch (error) {
        sceneWrap.replaceChildren();
        const failure = document.createElement('div');
        failure.className = 'deflection-webgl-error';
        failure.textContent = `3D WebGL 不可用：${error}`;
        sceneWrap.append(failure);
        return {
          card,
          values,
          scene: {setComponent: () => {}, dispose: () => {}},
          lastSampleToken: null,
        };
      }
    };

    function renderRelativeDeflection(deflection, sourceSampleToken = null) {
      deflection = deflection || {};
      const target = $('human-deflection-views');
      const components = Array.isArray(deflection.components)
        ? deflection.components : [];
      const age = Number(deflection.age_sec);
      $('human-deflection-status').textContent = [
        uiValue(deflection.status || 'waiting'),
        Number.isFinite(age) ? `输入年龄 ${age.toFixed(3)}s` : null,
        deflection.anchor_request_generation !== null
          && deflection.anchor_request_generation !== undefined
          ? `Home #${deflection.anchor_request_generation}` : null,
      ].filter(Boolean).join(' · ');
      if (!components.length) {
        disposeRelativeDeflectionEntries();
        target.replaceChildren();
        const empty = document.createElement('div');
        empty.className = 'deflection-empty';
        empty.textContent = deflection.message || '等待有效 pose。';
        target.append(empty);
        return;
      }
      for (const empty of target.querySelectorAll('.deflection-empty')) empty.remove();
      const activeIds = new Set();
      let componentOrder = 0;
      for (const component of components) {
        const translation = numericVector(component.translation_xyz_m);
        const rotvec = numericVector(component.rotation_rotvec_rad);
        const componentId = String(component.component_id || '').trim();
        if (!componentId || translation.length !== 3 || rotvec.length !== 3
            || !translation.every(Number.isFinite) || !rotvec.every(Number.isFinite)) continue;
        activeIds.add(componentId);
        let entry = relativeDeflectionEntries.get(componentId);
        if (!entry) {
          entry = makeRelativeDeflectionEntry(componentId);
          relativeDeflectionEntries.set(componentId, entry);
        }
        const sampleToken = sourceSampleToken === null
          || sourceSampleToken === undefined
          ? JSON.stringify([translation, rotvec, deflection.anchor_request_generation])
          : `${sourceSampleToken}:${deflection.anchor_request_generation}`;
        if (entry.lastSampleToken !== sampleToken) {
          entry.scene.setComponent(component);
          entry.lastSampleToken = sampleToken;
        }
        const angle = Number(component.rotation_angle_rad) || 0;
        const distanceMm = Number(component.translation_distance_m) * 1000;
        const angleDeg = angle * 180 / Math.PI;
        const clipped = Number(component.translation_distance_m) > 0.15
          ? '（已等比缩放到显示边界）' : '';
        if (entry.values.dataset.sampleToken !== sampleToken) {
          entry.values.replaceChildren();
          for (const line of [
            `Δp [mm] = ${formattedDeflectionVector(translation, 1000, 1)}`,
            `|Δp| = ${Number.isFinite(distanceMm) ? distanceMm.toFixed(1) : '—'} mm ${clipped}`,
            `旋转向量 [deg] = ${formattedDeflectionVector(rotvec, 180 / Math.PI, 1)}`,
            `旋转角 = ${Number.isFinite(angleDeg) ? angleDeg.toFixed(1) : '—'}°`,
            `平移坐标系 = ${text(component.translation_reference_frame)}`,
            `旋转局部系 = ${text(component.rotation_controlled_frame)}`,
          ]) {
            const row = document.createElement('div');
            row.textContent = line;
            entry.values.append(row);
          }
          entry.values.dataset.sampleToken = sampleToken;
        }
        const currentAtIndex = target.children[componentOrder] || null;
        if (currentAtIndex !== entry.card) {
          target.insertBefore(entry.card, currentAtIndex);
        }
        componentOrder += 1;
      }
      for (const [componentId, entry] of relativeDeflectionEntries) {
        if (!activeIds.has(componentId)) {
          entry.scene.dispose();
          entry.card.remove();
          relativeDeflectionEntries.delete(componentId);
        }
      }
      if (!relativeDeflectionEntries.size) {
        target.replaceChildren();
        const empty = document.createElement('div');
        empty.className = 'deflection-empty';
        empty.textContent = 'pose 数据格式无效。';
        target.append(empty);
      }
    }
"""


__all__ = [
    "RELATIVE_DEFLECTION_VIEW_CSS",
    "RELATIVE_DEFLECTION_VIEW_HTML",
    "RELATIVE_DEFLECTION_VIEW_JS",
]
