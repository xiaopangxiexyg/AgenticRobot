// WebGL axis renderer extracted from quest_pose/web/app.js.
(function exposeQuestDevVisualizationRenderer(global) {
  'use strict';

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || 'WebGL shader compile failed');
    }
    return shader;
  }

  function pushVertex(positions, colors, point, color) {
    positions.push(point[0], point[1], point[2]);
    colors.push(color[0], color[1], color[2]);
  }

  function pushTriangle(positions, colors, a, b, c, color) {
    pushVertex(positions, colors, a, color);
    pushVertex(positions, colors, b, color);
    pushVertex(positions, colors, c, color);
  }

  function axisPoint(axis, t, radius, theta) {
    const c = Math.cos(theta) * radius;
    const s = Math.sin(theta) * radius;
    if (axis === 0) return [t, c, s];
    if (axis === 1) return [c, t, s];
    return [c, s, t];
  }

  function createArrowAxesGeometry(options) {
    const positions = [];
    const colors = [];
    const axisColors = [
      [1.0, 0.06, 0.04],
      [0.05, 0.90, 0.20],
      [0.12, 0.38, 1.0],
    ];
    const segments = options.segments || 12;
    const shaftEnd = options.length - options.headLength;
    for (let axis = 0; axis < 3; axis += 1) {
      for (let i = 0; i < segments; i += 1) {
        const a0 = (i / segments) * Math.PI * 2;
        const a1 = ((i + 1) / segments) * Math.PI * 2;
        const s0 = axisPoint(axis, 0, options.shaftRadius, a0);
        const s1 = axisPoint(axis, 0, options.shaftRadius, a1);
        const e0 = axisPoint(axis, shaftEnd, options.shaftRadius, a0);
        const e1 = axisPoint(axis, shaftEnd, options.shaftRadius, a1);
        pushTriangle(positions, colors, s0, e0, e1, axisColors[axis]);
        pushTriangle(positions, colors, s0, e1, s1, axisColors[axis]);
        const b0 = axisPoint(axis, shaftEnd, options.headRadius, a0);
        const b1 = axisPoint(axis, shaftEnd, options.headRadius, a1);
        const tip = axisPoint(axis, options.length, 0, 0);
        pushTriangle(positions, colors, b0, tip, b1, axisColors[axis]);
      }
    }
    return { positions, colors };
  }

  function createBuffer(gl, values) {
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.STATIC_DRAW);
    return buffer;
  }

  function create(gl, options = {}) {
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, `
      attribute vec3 a_position;
      attribute vec3 a_color;
      uniform mat4 u_mvp;
      varying vec3 v_color;
      void main() {
        gl_Position = u_mvp * vec4(a_position, 1.0);
        v_color = a_color;
      }
    `);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec3 v_color;
      void main() { gl_FragColor = vec4(v_color, 1.0); }
    `);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || 'WebGL program link failed');
    }

    const length = Number(options.axisLength || 0.18);
    const geometry = createArrowAxesGeometry({
      length,
      shaftRadius: length * 0.037,
      headRadius: length * 0.095,
      headLength: length * 0.226,
      segments: 12,
    });
    return {
      program,
      aPosition: gl.getAttribLocation(program, 'a_position'),
      aColor: gl.getAttribLocation(program, 'a_color'),
      uMvp: gl.getUniformLocation(program, 'u_mvp'),
      positionBuffer: createBuffer(gl, geometry.positions),
      colorBuffer: createBuffer(gl, geometry.colors),
      count: geometry.positions.length / 3,
    };
  }

  function drawAxes(gl, renderer, mvp) {
    gl.useProgram(renderer.program);
    gl.uniformMatrix4fv(renderer.uMvp, false, mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.positionBuffer);
    gl.vertexAttribPointer(renderer.aPosition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(renderer.aPosition);
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.colorBuffer);
    gl.vertexAttribPointer(renderer.aColor, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(renderer.aColor);
    gl.drawArrays(gl.TRIANGLES, 0, renderer.count);
  }

  global.QuestDevVisualizationRenderer = { create, drawAxes };
}(window));
