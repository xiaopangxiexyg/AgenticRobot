"""Self-contained WebGL renderer used by the pose-deflection GUI fragment."""

from __future__ import annotations


RELATIVE_DEFLECTION_WEBGL_JS = r"""
    const deflectionMat4Multiply = (a, b) => {
      const out = new Float32Array(16);
      for (let column = 0; column < 4; column += 1) {
        for (let row = 0; row < 4; row += 1) {
          let value = 0;
          for (let index = 0; index < 4; index += 1) {
            value += a[index * 4 + row] * b[column * 4 + index];
          }
          out[column * 4 + row] = value;
        }
      }
      return out;
    };
    const deflectionPerspective = (fov, aspect, near, far) => {
      const f = 1 / Math.tan(fov / 2);
      const out = new Float32Array(16);
      out[0] = f / aspect;
      out[5] = f;
      out[10] = (far + near) / (near - far);
      out[11] = -1;
      out[14] = 2 * far * near / (near - far);
      return out;
    };
    const deflectionNormalize = (value) => {
      const length = Math.hypot(value[0], value[1], value[2]) || 1;
      return value.map(item => item / length);
    };
    const deflectionCross = (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const deflectionLookAt = (eye, center, up) => {
      const z = deflectionNormalize([
        eye[0] - center[0], eye[1] - center[1], eye[2] - center[2],
      ]);
      const x = deflectionNormalize(deflectionCross(up, z));
      const y = deflectionCross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -x[0] * eye[0] - x[1] * eye[1] - x[2] * eye[2],
        -y[0] * eye[0] - y[1] * eye[1] - y[2] * eye[2],
        -z[0] * eye[0] - z[1] * eye[1] - z[2] * eye[2],
        1,
      ]);
    };
    const deflectionProjectPoint = (point, matrix, width, height) => {
      const vector = [point[0], point[1], point[2], 1];
      const clip = [0, 0, 0, 0];
      for (let row = 0; row < 4; row += 1) {
        for (let column = 0; column < 4; column += 1) {
          clip[row] += matrix[column * 4 + row] * vector[column];
        }
      }
      if (clip[3] <= 1e-6) return null;
      const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
      if (ndc[2] < -1.2 || ndc[2] > 1.2) return null;
      return [
        (ndc[0] * 0.5 + 0.5) * width,
        (0.5 - ndc[1] * 0.5) * height,
      ];
    };
    const deflectionModel = (position, scale) => new Float32Array([
      scale, 0, 0, 0,
      0, scale, 0, 0,
      0, 0, scale, 0,
      position[0], position[1], position[2], 1,
    ]);
    const deflectionRotateVector = (vector, rotvec) => {
      const angle = Math.hypot(rotvec[0], rotvec[1], rotvec[2]);
      if (angle < 1e-9) return [...vector];
      const axis = rotvec.map(item => item / angle);
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
      const cross = deflectionCross(axis, vector);
      return vector.map((item, index) =>
        item * cosine + cross[index] * sine + axis[index] * dot * (1 - cosine)
      );
    };

    class PoseDeflectionScene {
      constructor(canvas, axisLabelLayer = null) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl', {antialias: true, alpha: false});
        if (!this.gl) throw new Error('WebGL unavailable');
        this.yaw = -0.78;
        this.pitch = 0.55;
        this.distance = 3.5;
        this.point = [0, 0, 0];
        this.rotvec = [0, 0, 0];
        this.axisLabelLayer = axisLabelLayer;
        this.axisLabelElements = axisLabelLayer === null
          ? [] : [...axisLabelLayer.querySelectorAll('[data-axis-direction]')];
        this.drag = null;
        this._buildPrograms();
        this._buildSphere();
        this._bindInteraction();
        this.resizeObserver = new ResizeObserver(() => this.draw());
        this.resizeObserver.observe(canvas);
      }

      _shader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          throw new Error(gl.getShaderInfoLog(shader) || 'shader compile failed');
        }
        return shader;
      }

      _program(vertexSource, fragmentSource) {
        const gl = this.gl;
        const program = gl.createProgram();
        gl.attachShader(program, this._shader(gl.VERTEX_SHADER, vertexSource));
        gl.attachShader(program, this._shader(gl.FRAGMENT_SHADER, fragmentSource));
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
          throw new Error(gl.getProgramInfoLog(program) || 'program link failed');
        }
        return program;
      }

      _buildPrograms() {
        const gl = this.gl;
        this.sphereProgram = this._program(
          'attribute vec3 aPosition; attribute vec3 aNormal; uniform mat4 uViewProjection; uniform mat4 uModel; varying vec3 vNormal; void main(){ vNormal=aNormal; gl_Position=uViewProjection*uModel*vec4(aPosition,1.0); }',
          'precision mediump float; uniform vec4 uColor; varying vec3 vNormal; void main(){ vec3 n=normalize(vNormal); float light=0.28+0.72*max(dot(n,normalize(vec3(0.4,-0.3,0.85))),0.0); gl_FragColor=vec4(uColor.rgb*light,uColor.a); }',
        );
        this.lineProgram = this._program(
          'attribute vec3 aPosition; uniform mat4 uViewProjection; void main(){ gl_Position=uViewProjection*vec4(aPosition,1.0); }',
          'precision mediump float; uniform vec4 uColor; void main(){ gl_FragColor=uColor; }',
        );
        this.spherePosition = gl.getAttribLocation(this.sphereProgram, 'aPosition');
        this.sphereNormal = gl.getAttribLocation(this.sphereProgram, 'aNormal');
        this.sphereViewProjection = gl.getUniformLocation(this.sphereProgram, 'uViewProjection');
        this.sphereModel = gl.getUniformLocation(this.sphereProgram, 'uModel');
        this.sphereColor = gl.getUniformLocation(this.sphereProgram, 'uColor');
        this.linePosition = gl.getAttribLocation(this.lineProgram, 'aPosition');
        this.lineViewProjection = gl.getUniformLocation(this.lineProgram, 'uViewProjection');
        this.lineColor = gl.getUniformLocation(this.lineProgram, 'uColor');
        this.lineBuffer = gl.createBuffer();
      }

      _buildSphere() {
        const gl = this.gl;
        const positions = [];
        const indices = [];
        const latitudeCount = 18;
        const longitudeCount = 24;
        for (let latitude = 0; latitude <= latitudeCount; latitude += 1) {
          const theta = latitude / latitudeCount * Math.PI;
          for (let longitude = 0; longitude <= longitudeCount; longitude += 1) {
            const phi = longitude / longitudeCount * Math.PI * 2;
            positions.push(
              Math.sin(theta) * Math.cos(phi),
              Math.sin(theta) * Math.sin(phi),
              Math.cos(theta),
            );
          }
        }
        for (let latitude = 0; latitude < latitudeCount; latitude += 1) {
          for (let longitude = 0; longitude < longitudeCount; longitude += 1) {
            const first = latitude * (longitudeCount + 1) + longitude;
            const second = first + longitudeCount + 1;
            indices.push(first, second, first + 1, second, second + 1, first + 1);
          }
        }
        this.sphereVertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sphereVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        this.sphereIndexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.sphereIndexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
        this.sphereIndexCount = indices.length;
      }

      _bindInteraction() {
        this.onPointerDown = (event) => {
          this.drag = [event.clientX, event.clientY, this.yaw, this.pitch];
          this.canvas.setPointerCapture(event.pointerId);
        };
        this.onPointerMove = (event) => {
          if (!this.drag) return;
          this.yaw = this.drag[2] + (event.clientX - this.drag[0]) * 0.008;
          this.pitch = Math.max(-1.25, Math.min(1.25,
            this.drag[3] + (event.clientY - this.drag[1]) * 0.008));
          this.draw();
        };
        this.onPointerUp = () => { this.drag = null; };
        this.onWheel = (event) => {
          event.preventDefault();
          this.distance = Math.max(2.2, Math.min(6.0,
            this.distance * Math.exp(event.deltaY * 0.001)));
          this.draw();
        };
        this.canvas.addEventListener('pointerdown', this.onPointerDown);
        this.canvas.addEventListener('pointermove', this.onPointerMove);
        this.canvas.addEventListener('pointerup', this.onPointerUp);
        this.canvas.addEventListener('pointercancel', this.onPointerUp);
        this.canvas.addEventListener('wheel', this.onWheel, {passive: false});
      }

      setComponent(component) {
        const translation = component.translation_xyz_m.map(Number);
        const length = Math.hypot(...translation);
        const displayRange = 0.12;
        const scale = length > displayRange * 1.25
          ? displayRange * 1.25 / length : 1;
        this.point = translation.map(item => item * scale / displayRange);
        this.rotvec = component.rotation_rotvec_rad.map(Number);
        const semantics = component.translation_axis_semantics || {};
        const localized = {
          forward: '前', backward: '后', left: '左', right: '右',
          up: '上', down: '下',
        };
        for (const element of this.axisLabelElements) {
          const direction = element.dataset.axisDirection;
          const semantic = semantics[direction];
          const axis = element.dataset.axisName || '';
          element.textContent = semantic && localized[semantic]
            ? `${localized[semantic]} ${axis}` : axis;
        }
        this.draw();
      }

      _updateAxisLabels(viewProjection) {
        if (!this.axisLabelLayer) return;
        const endpoints = {
          positive_x: [1.35, 0, 0], negative_x: [-1.35, 0, 0],
          positive_y: [0, 1.35, 0], negative_y: [0, -1.35, 0],
          positive_z: [0, 0, 1.35], negative_z: [0, 0, -1.35],
        };
        for (const element of this.axisLabelElements) {
          const point = endpoints[element.dataset.axisDirection];
          const projected = point && deflectionProjectPoint(
            point,
            viewProjection,
            this.canvas.clientWidth,
            this.canvas.clientHeight,
          );
          element.hidden = !projected;
          if (!projected) continue;
          element.style.left = `${projected[0]}px`;
          element.style.top = `${projected[1]}px`;
        }
      }

      _viewProjection() {
        const width = Math.max(1, this.canvas.clientWidth);
        const height = Math.max(1, this.canvas.clientHeight);
        const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
        const targetWidth = Math.round(width * pixelRatio);
        const targetHeight = Math.round(height * pixelRatio);
        if (this.canvas.width !== targetWidth || this.canvas.height !== targetHeight) {
          this.canvas.width = targetWidth;
          this.canvas.height = targetHeight;
        }
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        const eye = [
          this.distance * Math.cos(this.pitch) * Math.cos(this.yaw),
          this.distance * Math.cos(this.pitch) * Math.sin(this.yaw),
          this.distance * Math.sin(this.pitch),
        ];
        return deflectionMat4Multiply(
          deflectionPerspective(Math.PI / 4, width / height, 0.05, 50),
          deflectionLookAt(eye, [0, 0, 0], [0, 0, 1]),
        );
      }

      _drawLines(viewProjection, vertices, color, width = 1) {
        const gl = this.gl;
        gl.useProgram(this.lineProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(this.linePosition);
        gl.vertexAttribPointer(this.linePosition, 3, gl.FLOAT, false, 0, 0);
        gl.uniformMatrix4fv(this.lineViewProjection, false, viewProjection);
        gl.uniform4fv(this.lineColor, color);
        gl.lineWidth(width);
        gl.drawArrays(gl.LINES, 0, vertices.length / 3);
      }

      _drawSphere(viewProjection, position, radius, color) {
        const gl = this.gl;
        gl.useProgram(this.sphereProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sphereVertexBuffer);
        gl.enableVertexAttribArray(this.spherePosition);
        gl.vertexAttribPointer(this.spherePosition, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.sphereNormal);
        gl.vertexAttribPointer(this.sphereNormal, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.sphereIndexBuffer);
        gl.uniformMatrix4fv(this.sphereViewProjection, false, viewProjection);
        gl.uniformMatrix4fv(this.sphereModel, false, deflectionModel(position, radius));
        gl.uniform4fv(this.sphereColor, color);
        gl.drawElements(gl.TRIANGLES, this.sphereIndexCount, gl.UNSIGNED_SHORT, 0);
      }

      draw() {
        const gl = this.gl;
        const viewProjection = this._viewProjection();
        gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0.035, 0.065, 0.12, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        const grid = [];
        for (let value = -1; value <= 1.001; value += 0.25) {
          grid.push(-1, value, 0, 1, value, 0, value, -1, 0, value, 1, 0);
        }
        this._drawLines(viewProjection, grid, [0.18, 0.24, 0.34, 1]);
        this._drawLines(viewProjection, [-1.35, 0, 0, 1.35, 0, 0], [0.95, 0.32, 0.36, 1], 2);
        this._drawLines(viewProjection, [0, -1.35, 0, 0, 1.35, 0], [0.30, 0.86, 0.48, 1], 2);
        this._drawLines(viewProjection, [0, 0, -1.35, 0, 0, 1.35], [0.30, 0.56, 1.0, 1], 2);
        this._drawLines(viewProjection, [0, 0, 0, ...this.point], [0.45, 0.72, 1.0, 1], 3);
        this._drawSphere(viewProjection, [0, 0, 0], 0.075, [0.78, 0.84, 0.92, 1]);
        this._drawSphere(viewProjection, this.point, 0.13, [0.35, 0.9, 0.58, 1]);
        const localAxisLength = 0.32;
        const localAxes = [
          [[1, 0, 0], [0.95, 0.32, 0.36, 1]],
          [[0, 1, 0], [0.30, 0.86, 0.48, 1]],
          [[0, 0, 1], [0.30, 0.56, 1.0, 1]],
        ];
        for (const [axis, color] of localAxes) {
          const direction = deflectionRotateVector(axis, this.rotvec);
          const endpoint = this.point.map(
            (item, index) => item + direction[index] * localAxisLength
          );
          this._drawLines(
            viewProjection,
            [...this.point, ...endpoint],
            color,
            3,
          );
        }
        this._updateAxisLabels(viewProjection);
      }

      dispose() {
        this.resizeObserver.disconnect();
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        this.canvas.removeEventListener('pointermove', this.onPointerMove);
        this.canvas.removeEventListener('pointerup', this.onPointerUp);
        this.canvas.removeEventListener('pointercancel', this.onPointerUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
      }
    }
"""


__all__ = ["RELATIVE_DEFLECTION_WEBGL_JS"]
