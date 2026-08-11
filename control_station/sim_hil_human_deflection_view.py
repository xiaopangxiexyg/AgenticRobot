"""Presentation fragment for the read-only Sim-HIL Human deflection view."""

from __future__ import annotations


HUMAN_DEFLECTION_VIEW_CSS = r"""
    .deflection-panel { margin-bottom: 16px; padding: 15px; border: 1px solid var(--line); border-radius: 14px; background: rgba(17,24,39,.9); box-shadow: var(--shadow); }
    .deflection-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }
    .deflection-empty { padding: 18px; border: 1px dashed var(--line); border-radius: 12px; color: var(--muted); text-align: center; }
    .deflection-card { display: grid; grid-template-columns: 190px minmax(0,1fr); gap: 14px; align-items: center; padding: 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(10,15,28,.48); }
    .deflection-card svg { width: 190px; height: 190px; }
    .deflection-card h3 { margin: 0 0 8px; color: var(--blue); font-size: 14px; }
    .deflection-values { display: grid; gap: 6px; color: var(--muted); font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .deflection-values strong { color: var(--text); }
    .deflection-legend { margin-top: 7px; color: var(--muted); font-size: 11px; }
    @media (max-width: 680px) { .deflection-card { grid-template-columns: 1fr; } .deflection-card svg { margin: 0 auto; } }
"""


HUMAN_DEFLECTION_VIEW_HTML = r"""
    <section id="human-deflection-panel" class="deflection-panel" aria-label="人类相对 Home 偏转">
      <div class="card-head">
        <div>
          <h2>人类相对 Home 偏转</h2>
          <p>中心为当前已接纳 residual anchor；蓝线表示参考坐标系中的平移，橙线表示末端局部旋转轴。</p>
        </div>
        <span id="human-deflection-status" class="label">等待 relative 输入</span>
      </div>
      <div id="human-deflection-views" class="deflection-grid">
        <div class="deflection-empty">启动 relative 模式后显示已接纳的偏转。</div>
      </div>
      <div class="deflection-legend">圆盘显示 XY 投影，固定量程 ±120 mm；Z、三维距离和旋转轴角在右侧显示。该视图只读，不参与控制。</div>
    </section>
"""


HUMAN_DEFLECTION_VIEW_JS = r"""
    function renderHumanDeflection(deflection) {
      deflection = deflection || {};
      const target = $('human-deflection-views');
      target.replaceChildren();
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
        const empty = document.createElement('div');
        empty.className = 'deflection-empty';
        empty.textContent = deflection.message || '等待已接纳的 relative pose。';
        target.append(empty);
        return;
      }
      const svgNamespace = 'http://www.w3.org/2000/svg';
      const svgNode = (name, attributes = {}) => {
        const node = document.createElementNS(svgNamespace, name);
        for (const [key, value] of Object.entries(attributes)) {
          node.setAttribute(key, String(value));
        }
        return node;
      };
      const numericVector = (value) => Array.isArray(value)
        ? value.map(item => Number(item)) : [];
      const formatted = (values, scale, digits) => values.map(item =>
        Number.isFinite(item) ? (item * scale).toFixed(digits) : '—'
      ).join(', ');
      for (const component of components) {
        const translation = numericVector(component.translation_xyz_m);
        const rotvec = numericVector(component.rotation_rotvec_rad);
        if (translation.length !== 3 || rotvec.length !== 3) continue;
        const card = document.createElement('article');
        card.className = 'deflection-card';
        const svg = svgNode('svg', {
          viewBox: '0 0 200 200',
          role: 'img',
          'aria-label': `${component.component_id} Human Home deflection`,
        });
        svg.append(
          svgNode('circle', {cx: 100, cy: 100, r: 82, fill: '#0b1322', stroke: '#33435f', 'stroke-width': 2}),
          svgNode('line', {x1: 18, y1: 100, x2: 182, y2: 100, stroke: '#26344d', 'stroke-width': 1}),
          svgNode('line', {x1: 100, y1: 18, x2: 100, y2: 182, stroke: '#26344d', 'stroke-width': 1}),
          svgNode('circle', {cx: 100, cy: 100, r: 4, fill: '#e5edf8'}),
        );
        const displayRangeM = 0.12;
        const xyNorm = Math.hypot(translation[0], translation[1]);
        const clampScale = xyNorm > displayRangeM
          ? displayRangeM / xyNorm : 1;
        const pointX = 100 + translation[0] * clampScale / displayRangeM * 78;
        const pointY = 100 - translation[1] * clampScale / displayRangeM * 78;
        svg.append(
          svgNode('line', {x1: 100, y1: 100, x2: pointX, y2: pointY, stroke: '#73b7ff', 'stroke-width': 4, 'stroke-linecap': 'round'}),
          svgNode('circle', {cx: pointX, cy: pointY, r: 9, fill: '#69d39a', stroke: '#d9ffec', 'stroke-width': 2}),
        );
        const angle = Number(component.rotation_angle_rad) || 0;
        const rotationXY = Math.hypot(rotvec[0], rotvec[1]);
        if (rotationXY > 1e-9) {
          const rotationLength = 16 + Math.min(28, angle / (Math.PI / 4) * 28);
          svg.append(svgNode('line', {
            x1: pointX,
            y1: pointY,
            x2: pointX + rotvec[0] / rotationXY * rotationLength,
            y2: pointY - rotvec[1] / rotationXY * rotationLength,
            stroke: '#f3bd62',
            'stroke-width': 4,
            'stroke-linecap': 'round',
          }));
        }
        const rotationFraction = Math.min(1, Math.abs(angle) / Math.PI);
        svg.append(svgNode('circle', {
          cx: pointX,
          cy: pointY,
          r: 14,
          fill: 'none',
          stroke: '#f3bd62',
          'stroke-width': 3,
          'stroke-dasharray': `${rotationFraction * 88} 88`,
          transform: `rotate(-90 ${pointX} ${pointY})`,
        }));
        const details = document.createElement('div');
        const title = document.createElement('h3');
        title.textContent = component.component_id;
        const values = document.createElement('div');
        values.className = 'deflection-values';
        const distanceMm = Number(component.translation_distance_m) * 1000;
        const angleDeg = angle * 180 / Math.PI;
        const clipped = xyNorm > displayRangeM ? '（XY 已裁到显示边界）' : '';
        for (const line of [
          `Δp [mm] = ${formatted(translation, 1000, 1)}`,
          `|Δp| = ${Number.isFinite(distanceMm) ? distanceMm.toFixed(1) : '—'} mm ${clipped}`,
          `旋转向量 [deg] = ${formatted(rotvec, 180 / Math.PI, 1)}`,
          `旋转角 = ${Number.isFinite(angleDeg) ? angleDeg.toFixed(1) : '—'}°`,
          `平移坐标系 = ${text(component.translation_reference_frame)}`,
          `旋转局部系 = ${text(component.rotation_controlled_frame)}`,
        ]) {
          const row = document.createElement('div');
          row.textContent = line;
          values.append(row);
        }
        details.append(title, values);
        card.append(svg, details);
        target.append(card);
      }
      if (!target.childElementCount) {
        const empty = document.createElement('div');
        empty.className = 'deflection-empty';
        empty.textContent = 'relative pose 数据格式无效。';
        target.append(empty);
      }
    }
"""


__all__ = [
    "HUMAN_DEFLECTION_VIEW_CSS",
    "HUMAN_DEFLECTION_VIEW_HTML",
    "HUMAN_DEFLECTION_VIEW_JS",
]
