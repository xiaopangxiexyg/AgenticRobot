"""WebGL renderer for the standalone continuous-guidance trajectory demo.

The renderer is deliberately presentation-only.  Its input is the already
projected VIVE Home deflection and its output never leaves the browser.
"""

from __future__ import annotations


TRAJECTORY_GUIDANCE_WEBGL_JS = r"""
    const trajectorySignedAxisVectors = {
      positive_x: [1, 0, 0], negative_x: [-1, 0, 0],
      positive_y: [0, 1, 0], negative_y: [0, -1, 0],
      positive_z: [0, 0, 1], negative_z: [0, 0, -1],
    };
    const trajectorySemanticBasis = (semantics = {}) => {
      const vectorFor = (meaning, fallback) => {
        for (const [direction, semantic] of Object.entries(semantics)) {
          if (semantic === meaning && trajectorySignedAxisVectors[direction]) {
            return [...trajectorySignedAxisVectors[direction]];
          }
        }
        return [...fallback];
      };
      return {
        forward: vectorFor('forward', [1, 0, 0]),
        left: vectorFor('left', [0, 1, 0]),
        up: vectorFor('up', [0, 0, 1]),
      };
    };
    const trajectoryMapPhysicalVector = (basis, vector) => [0, 1, 2].map(
      index => basis.forward[index] * vector[0]
        + basis.left[index] * vector[1]
        + basis.up[index] * vector[2]
    );

    class TrajectoryGuidanceScene {
      constructor(canvas, axisLabelLayer = null) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl', {antialias: true, alpha: false});
        if (!this.gl) throw new Error('WebGL unavailable');
        // Match the deflection-ball viewer's initial camera orientation so
        // identical signed axes do not appear rotated between the two panels.
        this.yaw = -0.78;
        this.pitch = 0.55;
        this.distance = 0.86;
        this.drag = null;
        this.deflectionM = [0, 0, 0];
        this.fullScaleM = 0.12;
        this.deadzoneM = 0.005;
        this.maxGuidanceSpeedMps = 0.08;
        this.positionGain = 1.0;
        this.mode = 'velocity_joystick';
        this.integratedOffsetM = [0, 0, 0];
        this.lastIntegrationAt = null;
        this.horizonSec = 2.0;
        this.bouncePeriodSec = 2.4;
        this.semanticBasis = trajectorySemanticBasis();
        this.axisSemanticsToken = null;
        this.axisLabelLayer = axisLabelLayer;
        this.axisLabelElements = axisLabelLayer === null
          ? [] : [...axisLabelLayer.querySelectorAll('[data-axis-direction]')];
        this.disposed = false;
        this.animationFrame = null;
        this.lastFrameAt = Number.NEGATIVE_INFINITY;
        this.framePeriodMs = 1000 / 30;
        this._buildPrograms();
        this._buildGeometry();
        this._bindInteraction();
        this.resizeObserver = new ResizeObserver(
          () => this.draw(performance.now())
        );
        this.resizeObserver.observe(canvas);
        this.animationFrame = window.requestAnimationFrame(
          timestamp => this._animate(timestamp)
        );
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
        const fragmentSource = 'precision mediump float; uniform vec4 uColor; uniform float uRoundPoint; void main(){ if(uRoundPoint>0.5 && distance(gl_PointCoord,vec2(0.5))>0.5) discard; gl_FragColor=uColor; }';
        this.staticProgram = this._program(
          'attribute vec3 aPosition; uniform mat4 uViewProjection; uniform float uPointSize; void main(){ gl_Position=uViewProjection*vec4(aPosition,1.0); gl_PointSize=uPointSize; }',
          fragmentSource,
        );
        this.positionLocation = gl.getAttribLocation(
          this.staticProgram, 'aPosition'
        );
        this.viewProjectionLocation = gl.getUniformLocation(
          this.staticProgram, 'uViewProjection'
        );
        this.colorLocation = gl.getUniformLocation(this.staticProgram, 'uColor');
        this.pointSizeLocation = gl.getUniformLocation(
          this.staticProgram, 'uPointSize'
        );
        this.roundPointLocation = gl.getUniformLocation(
          this.staticProgram, 'uRoundPoint'
        );
        this.trajectoryProgram = this._program(
          'attribute float aTau; uniform mat4 uViewProjection; uniform float uPhaseSec; uniform float uBouncePeriodSec; uniform vec3 uForwardAxis; uniform vec3 uLeftAxis; uniform vec3 uUpAxis; uniform vec3 uCorrectionOffset; uniform vec3 uGuidanceVelocity; uniform float uApplyCorrection; uniform float uPointSize; void main(){ float bouncePhase=mod(uPhaseSec+aTau,uBouncePeriodSec)/uBouncePeriodSec; vec3 physicalNominal=vec3(0.02+0.16*aTau,0.035*sin(3.14159265*aTau),0.025+0.44*bouncePhase*(1.0-bouncePhase)); vec3 nominal=uForwardAxis*physicalNominal.x+uLeftAxis*physicalNominal.y+uUpAxis*physicalNominal.z; vec3 correction=uCorrectionOffset+aTau*uGuidanceVelocity; vec3 position=nominal+uApplyCorrection*correction; gl_Position=uViewProjection*vec4(position,1.0); gl_PointSize=uPointSize; }',
          fragmentSource,
        );
        this.tauLocation = gl.getAttribLocation(this.trajectoryProgram, 'aTau');
        this.trajectoryViewProjection = gl.getUniformLocation(
          this.trajectoryProgram, 'uViewProjection'
        );
        this.trajectoryPhase = gl.getUniformLocation(
          this.trajectoryProgram, 'uPhaseSec'
        );
        this.trajectoryBouncePeriod = gl.getUniformLocation(
          this.trajectoryProgram, 'uBouncePeriodSec'
        );
        this.trajectoryVelocity = gl.getUniformLocation(
          this.trajectoryProgram, 'uGuidanceVelocity'
        );
        this.trajectoryForwardAxis = gl.getUniformLocation(
          this.trajectoryProgram, 'uForwardAxis'
        );
        this.trajectoryLeftAxis = gl.getUniformLocation(
          this.trajectoryProgram, 'uLeftAxis'
        );
        this.trajectoryUpAxis = gl.getUniformLocation(
          this.trajectoryProgram, 'uUpAxis'
        );
        this.trajectoryCorrectionOffset = gl.getUniformLocation(
          this.trajectoryProgram, 'uCorrectionOffset'
        );
        this.trajectoryApplyCorrection = gl.getUniformLocation(
          this.trajectoryProgram, 'uApplyCorrection'
        );
        this.trajectoryColor = gl.getUniformLocation(
          this.trajectoryProgram, 'uColor'
        );
        this.trajectoryPointSize = gl.getUniformLocation(
          this.trajectoryProgram, 'uPointSize'
        );
        this.trajectoryRoundPoint = gl.getUniformLocation(
          this.trajectoryProgram, 'uRoundPoint'
        );
      }

      _staticBuffer(values) {
        const gl = this.gl;
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.STATIC_DRAW);
        return buffer;
      }

      _buildGeometry() {
        const gl = this.gl;
        this.gridBuffer = gl.createBuffer();
        this._updateGridGeometry();
        this.axisBuffer = this._staticBuffer([
          -0.08,0,0, 0.44,0,0,
          0,-0.24,0, 0,0.24,0,
          0,0,-0.10, 0,0,0.28,
        ]);
        this.trajectoryPointCount = 65;
        const tauValues = [];
        for (let index = 0; index < this.trajectoryPointCount; index += 1) {
          tauValues.push(
            this.horizonSec * index / (this.trajectoryPointCount - 1)
          );
        }
        this.tauBuffer = this._staticBuffer(tauValues);
      }

      _updateGridGeometry() {
        const grid = [];
        for (let x = 0; x <= 0.4001; x += 0.05) {
          grid.push(
            ...trajectoryMapPhysicalVector(this.semanticBasis, [x, -0.2, 0]),
            ...trajectoryMapPhysicalVector(this.semanticBasis, [x, 0.2, 0]),
          );
        }
        for (let y = -0.2; y <= 0.2001; y += 0.05) {
          grid.push(
            ...trajectoryMapPhysicalVector(this.semanticBasis, [0, y, 0]),
            ...trajectoryMapPhysicalVector(this.semanticBasis, [0.4, y, 0]),
          );
        }
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffer);
        // Rebuilt only when the profile axis semantics changes, never per
        // animation frame.
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(grid), gl.STATIC_DRAW);
        this.gridVertexCount = grid.length / 3;
      }

      _bindInteraction() {
        this.onPointerDown = (event) => {
          this.drag = [event.clientX, event.clientY, this.yaw, this.pitch];
          this.canvas.setPointerCapture(event.pointerId);
        };
        this.onPointerMove = (event) => {
          if (!this.drag) return;
          this.yaw = this.drag[2] + (event.clientX - this.drag[0]) * 0.006;
          this.pitch = Math.max(-1.15, Math.min(1.2,
            this.drag[3] + (event.clientY - this.drag[1]) * 0.006));
        };
        this.onPointerUp = () => { this.drag = null; };
        this.onWheel = (event) => {
          event.preventDefault();
          this.distance = Math.max(0.48, Math.min(1.6,
            this.distance * Math.exp(event.deltaY * 0.001)));
        };
        this.canvas.addEventListener('pointerdown', this.onPointerDown);
        this.canvas.addEventListener('pointermove', this.onPointerMove);
        this.canvas.addEventListener('pointerup', this.onPointerUp);
        this.canvas.addEventListener('pointercancel', this.onPointerUp);
        this.canvas.addEventListener('wheel', this.onWheel, {passive: false});
      }

      setDeflectionMeters(value) {
        if (!Array.isArray(value) || value.length !== 3
            || !value.every(item => Number.isFinite(Number(item)))) {
          this.deflectionM = [0, 0, 0];
          return;
        }
        const next = value.map(Number);
        if (next.every((item, index) => item === this.deflectionM[index])) return;
        this.deflectionM = next;
      }

      setAxisSemantics(semantics = {}) {
        const semanticsToken = JSON.stringify(semantics);
        if (semanticsToken === this.axisSemanticsToken) return;
        this.axisSemanticsToken = semanticsToken;
        this.semanticBasis = trajectorySemanticBasis(semantics);
        this._updateGridGeometry();
        const localized = {
          forward: '前', backward: '后', left: '左', right: '右',
          up: '上', down: '下',
        };
        for (const element of this.axisLabelElements) {
          const semantic = semantics[element.dataset.axisDirection];
          const axis = element.dataset.axisName || '';
          element.textContent = semantic && localized[semantic]
            ? `${localized[semantic]} ${axis}` : axis;
        }
      }

      setMaxGuidanceSpeed(value) {
        const speed = Number(value);
        if (!Number.isFinite(speed) || speed < 0 || speed === this.maxGuidanceSpeedMps) {
          return;
        }
        this.maxGuidanceSpeedMps = speed;
      }

      setDeadzoneMeters(value) {
        const deadzone = Number(value);
        if (Number.isFinite(deadzone) && deadzone >= 0
            && deadzone < this.fullScaleM) {
          this.deadzoneM = deadzone;
        }
      }

      setPositionGain(value) {
        const gain = Number(value);
        if (Number.isFinite(gain) && gain >= 0) this.positionGain = gain;
      }

      setMode(value) {
        if (!['velocity_joystick', 'position_offset'].includes(value)) {
          throw new Error(`Unsupported trajectory guidance mode: ${value}`);
        }
        this.mode = value;
        this.lastIntegrationAt = null;
      }

      clearIntegratedOffset() {
        this.integratedOffsetM = [0, 0, 0];
        this.lastIntegrationAt = null;
      }

      guidanceVelocityMps() {
        const magnitudeM = Math.hypot(...this.deflectionM);
        if (magnitudeM <= this.deadzoneM) return [0, 0, 0];
        const activeScale = Math.min(
          1,
          (magnitudeM - this.deadzoneM) / (this.fullScaleM - this.deadzoneM),
        );
        return this.deflectionM.map(
          item => item / magnitudeM * activeScale * this.maxGuidanceSpeedMps
        );
      }

      correctionState() {
        if (this.mode === 'position_offset') {
          return {
            offsetM: this.deflectionM.map(item => item * this.positionGain),
            futureVelocityMps: [0, 0, 0],
          };
        }
        return {
          offsetM: [...this.integratedOffsetM],
          futureVelocityMps: this.guidanceVelocityMps(),
        };
      }

      _integrateJoystick(timestamp) {
        if (this.lastIntegrationAt === null) {
          this.lastIntegrationAt = timestamp;
          return;
        }
        const dtSec = Math.min(
          0.05,
          Math.max(0, (timestamp - this.lastIntegrationAt) / 1000),
        );
        this.lastIntegrationAt = timestamp;
        if (this.mode !== 'velocity_joystick' || dtSec <= 0) return;
        const velocity = this.guidanceVelocityMps();
        this.integratedOffsetM = this.integratedOffsetM.map(
          (item, index) => item + velocity[index] * dtSec
        );
      }

      _viewProjection() {
        const width = Math.max(1, this.canvas.clientWidth);
        const height = Math.max(1, this.canvas.clientHeight);
        const pixelRatio = Math.min(1.5, window.devicePixelRatio || 1);
        const targetWidth = Math.round(width * pixelRatio);
        const targetHeight = Math.round(height * pixelRatio);
        if (this.canvas.width !== targetWidth || this.canvas.height !== targetHeight) {
          this.canvas.width = targetWidth;
          this.canvas.height = targetHeight;
        }
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        const correction = this.correctionState();
        const terminalCorrection = correction.offsetM.map(
          (item, index) => item
            + this.horizonSec * correction.futureVelocityMps[index]
        );
        const nominalCenter = trajectoryMapPhysicalVector(
          this.semanticBasis, [0.18, 0, 0.09]
        );
        const center = nominalCenter.map(
          (item, index) => item + terminalCorrection[index] * 0.5
        );
        const correctionReach = Math.hypot(...terminalCorrection);
        const cameraDistance = Math.max(
          this.distance,
          0.86 + correctionReach * 1.4,
        );
        const eye = [
          center[0] + cameraDistance * Math.cos(this.pitch) * Math.cos(this.yaw),
          center[1] + cameraDistance * Math.cos(this.pitch) * Math.sin(this.yaw),
          center[2] + cameraDistance * Math.sin(this.pitch),
        ];
        return deflectionMat4Multiply(
          deflectionPerspective(Math.PI / 4, width / height, 0.01, 10),
          deflectionLookAt(eye, center, [0, 0, 1]),
        );
      }

      _drawStatic(
        viewProjection, buffer, first, count, color, mode, pointSize = 1
      ) {
        const gl = this.gl;
        gl.useProgram(this.staticProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(this.positionLocation);
        gl.vertexAttribPointer(this.positionLocation, 3, gl.FLOAT, false, 0, 0);
        gl.uniformMatrix4fv(this.viewProjectionLocation, false, viewProjection);
        gl.uniform4fv(this.colorLocation, color);
        gl.uniform1f(this.pointSizeLocation, pointSize);
        gl.uniform1f(this.roundPointLocation, mode === gl.POINTS ? 1 : 0);
        gl.drawArrays(mode, first, count);
      }

      _drawTrajectory(
        viewProjection, phaseSec, correctionOffset, guidanceVelocity,
        applyCorrection, color, mode,
        first = 0, count = this.trajectoryPointCount, pointSize = 1
      ) {
        const gl = this.gl;
        gl.useProgram(this.trajectoryProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.tauBuffer);
        gl.enableVertexAttribArray(this.tauLocation);
        gl.vertexAttribPointer(this.tauLocation, 1, gl.FLOAT, false, 0, 0);
        gl.uniformMatrix4fv(
          this.trajectoryViewProjection, false, viewProjection
        );
        gl.uniform1f(this.trajectoryPhase, phaseSec);
        gl.uniform1f(this.trajectoryBouncePeriod, this.bouncePeriodSec);
        gl.uniform3fv(this.trajectoryForwardAxis, this.semanticBasis.forward);
        gl.uniform3fv(this.trajectoryLeftAxis, this.semanticBasis.left);
        gl.uniform3fv(this.trajectoryUpAxis, this.semanticBasis.up);
        gl.uniform3fv(this.trajectoryCorrectionOffset, correctionOffset);
        gl.uniform3fv(this.trajectoryVelocity, guidanceVelocity);
        gl.uniform1f(this.trajectoryApplyCorrection, applyCorrection ? 1 : 0);
        gl.uniform4fv(this.trajectoryColor, color);
        gl.uniform1f(this.trajectoryPointSize, pointSize);
        gl.uniform1f(this.trajectoryRoundPoint, mode === gl.POINTS ? 1 : 0);
        gl.drawArrays(mode, first, count);
      }

      draw(nowMs) {
        const gl = this.gl;
        const viewProjection = this._viewProjection();
        gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0.035, 0.065, 0.12, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        this._drawStatic(
          viewProjection,
          this.gridBuffer,
          0,
          this.gridVertexCount,
          [0.18,0.24,0.34,1],
          gl.LINES,
        );
        this._drawStatic(
          viewProjection, this.axisBuffer, 0, 2, [0.95,0.32,0.36,1], gl.LINES
        );
        this._drawStatic(
          viewProjection, this.axisBuffer, 2, 2, [0.30,0.86,0.48,1], gl.LINES
        );
        this._drawStatic(
          viewProjection, this.axisBuffer, 4, 2, [0.30,0.56,1,1], gl.LINES
        );
        const phaseSec = (nowMs / 1000) % this.bouncePeriodSec;
        const correction = this.correctionState();
        const zero = [0, 0, 0];
        this._drawTrajectory(
          viewProjection, phaseSec, zero, zero, false,
          [0.58,0.69,0.84,1], gl.LINE_STRIP
        );
        this._drawTrajectory(
          viewProjection, phaseSec, correction.offsetM,
          correction.futureVelocityMps, true,
          [0.36,0.94,0.59,1], gl.LINE_STRIP
        );
        this._drawTrajectory(
          viewProjection, phaseSec, zero, zero, false,
          [0.75,0.84,0.96,1], gl.POINTS,
          0, 1, 10
        );
        this._drawTrajectory(
          viewProjection, phaseSec, correction.offsetM,
          correction.futureVelocityMps, true,
          [0.47,1.0,0.67,1], gl.POINTS,
          this.trajectoryPointCount - 1, 1, 12
        );
        this._updateAxisLabels(viewProjection);
      }

      _updateAxisLabels(viewProjection) {
        if (!this.axisLabelLayer) return;
        const endpoints = {
          positive_x: [0.44, 0, 0], negative_x: [-0.08, 0, 0],
          positive_y: [0, 0.24, 0], negative_y: [0, -0.24, 0],
          positive_z: [0, 0, 0.28], negative_z: [0, 0, -0.10],
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

      _animate(timestamp) {
        if (this.disposed) return;
        this._integrateJoystick(timestamp);
        if (timestamp - this.lastFrameAt >= this.framePeriodMs) {
          this.lastFrameAt = timestamp;
          this.draw(timestamp);
        }
        this.animationFrame = window.requestAnimationFrame(
          nextTimestamp => this._animate(nextTimestamp)
        );
      }

      dispose() {
        this.disposed = true;
        if (this.animationFrame !== null) {
          window.cancelAnimationFrame(this.animationFrame);
        }
        this.resizeObserver.disconnect();
        this.canvas.removeEventListener('pointerdown', this.onPointerDown);
        this.canvas.removeEventListener('pointermove', this.onPointerMove);
        this.canvas.removeEventListener('pointerup', this.onPointerUp);
        this.canvas.removeEventListener('pointercancel', this.onPointerUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
      }
    }
"""


__all__ = ["TRAJECTORY_GUIDANCE_WEBGL_JS"]
