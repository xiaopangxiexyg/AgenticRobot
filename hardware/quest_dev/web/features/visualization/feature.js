(function registerVisualizationFeature(global) {
  'use strict';

  const statusRoot = document.getElementById('feature-status-grid');
  const statusCard = document.createElement('div');
  statusCard.className = 'status-item';
  statusCard.innerHTML = '<span class="label">Visualization</span><span class="bad">disabled</span>';
  statusRoot.appendChild(statusCard);
  const status = statusCard.querySelector('span:last-child');
  const captureStatusCard = document.createElement('div');
  captureStatusCard.className = 'status-item';
  captureStatusCard.innerHTML = '<span class="label">Quest capture</span><span class="bad">waiting for terminal</span>';
  statusRoot.appendChild(captureStatusCard);
  const captureStatus = captureStatusCard.querySelector('span:last-child');

  let enabled = false;
  let config = {};
  let renderer = null;
  let latestRobotMessage = null;
  let latestCaptureStatus = null;
  let latestCalibrationPreview = null;
  let referenceMarkerMatrix = null;
  let referenceGeneration = null;
  let coreProtocolVersion = 2;

  function setStatus(text, ok) {
    status.textContent = text;
    status.className = ok ? 'ok' : 'bad';
  }

  function questReferenceEpochId(context) {
    if (
      typeof context.xrSessionId !== 'string'
      || typeof context.referenceSpaceType !== 'string'
      || !Number.isInteger(context.referenceSpaceGeneration)
    ) {
      return null;
    }
    return (
      `${context.xrSessionId}:${context.referenceSpaceType}:`
      + `${context.referenceSpaceGeneration}`
    );
  }

  function hasFiniteMatrix(value) {
    return (
      Array.isArray(value)
      && value.length === 16
      && value.every((item) => Number.isFinite(Number(item)))
    );
  }

  function questEpochMatches(context, sessionId, spaceType, generation) {
    const epochId = questReferenceEpochId(context);
    return (
      epochId !== null
      && sessionId === context.xrSessionId
      && spaceType === context.referenceSpaceType
      && generation === context.referenceSpaceGeneration
    );
  }

  function setCaptureStatus(text, ok) {
    captureStatus.textContent = text;
    captureStatus.className = ok ? 'ok' : 'bad';
  }

  function captureStatusText(message) {
    const capture = message.capture || {};
    const candidates = Array.isArray(capture.candidates) ? capture.candidates : [];
    const candidateText = candidates.length === 0
      ? 'no controller candidate'
      : candidates.map((candidate) => (
        `${candidate.handedness || 'none'}:${String(candidate.candidate_hash || '').slice(0, 8)}`
      )).join(', ');
    const controls = 'r=start/finish, c=seal+solve';
    const terminal = message.terminal_message ? ` | ${message.terminal_message}` : '';
    return `${controls} | ${capture.state || 'waiting'} | candidates=${candidateText}${terminal}`;
  }

  function renderableRobotRecord(record, context) {
    if (
      !record
      || record.fresh !== true
      || record.alignment_ready !== true
      || !hasFiniteMatrix(record.matrix_quest_from_tcp)
    ) {
      return false;
    }
    const binding = record.frame_binding;
    const epochId = questReferenceEpochId(context);
    if (!binding || !epochId) return false;
    return (
      binding.quest_session_id === context.xrSessionId
      && binding.reference_space_type === context.referenceSpaceType
      && binding.reference_space_generation === context.referenceSpaceGeneration
      && binding.quest_reference_epoch_id === epochId
    );
  }

  global.QuestDevFeatures.register('visualization', {
    configure(featureConfig, context) {
      enabled = context.enabled;
      config = featureConfig;
      coreProtocolVersion = context.protocolVersion;
      setStatus(enabled ? 'enabled; XR not started' : 'disabled', enabled);
    },

    onXRReady(context) {
      if (!enabled) return;
      if (renderer) global.QuestDevVisualizationRenderer.destroy(renderer);
      renderer = global.QuestDevVisualizationRenderer.create(context.gl, {
        axisLength: Number(config.axis_length_m || 0.18),
        markerSizeM: Number(config.reference_marker_size_m || 0.10),
      });
      setStatus('XR renderer ready', true);
    },

    onXRFrame(context) {
      if (!renderer || config.show_reference_marker === false) return;
      if (referenceGeneration !== context.referenceSpaceGeneration) {
        referenceMarkerMatrix = null;
        referenceGeneration = context.referenceSpaceGeneration;
      }
      if (!referenceMarkerMatrix && context.viewerPose) {
        const offset = global.QuestDevXRMath.mat4Translation(
          0,
          -Number(config.reference_marker_down_from_viewer_m || 0.15),
          -Number(config.reference_marker_forward_m || 0.8),
        );
        referenceMarkerMatrix = global.QuestDevXRMath.mat4Multiply(
          context.viewerPose.transform.matrix,
          offset,
        );
        setStatus(`reference marker generation ${referenceGeneration}`, true);
      }
    },

    handleServerMessage(context) {
      const message = context.message;
      if (message.feature === 'visualization' && message.type === 'capture_status') {
        if (
          message.protocol_version !== coreProtocolVersion
          || message.feature_protocol_version !== 2
        ) {
          setCaptureStatus('capture protocol mismatch', false);
          return;
        }
        latestCaptureStatus = message;
        const capture = message.capture || {};
        setCaptureStatus(
          captureStatusText(message),
          capture.state !== 'terminal' && capture.state !== 'closed',
        );
        return;
      }
      if (message.feature === 'visualization' && message.type === 'calibration_preview') {
        if (
          message.protocol_version !== coreProtocolVersion
          || message.feature_protocol_version !== 2
          || message.preview !== true
          || message.accepted !== false
          || message.alignment_status !== 'diagnostic_preview'
          || !hasFiniteMatrix(message.matrix_quest_from_robot_base)
        ) {
          latestCalibrationPreview = null;
          setCaptureStatus('invalid calibration preview', false);
          return;
        }
        latestCalibrationPreview = message;
        setCaptureStatus(
          `diagnostic preview only | ${message.robot_frame || 'robot base'} -> ${message.tcp_frame || 'TCP'}`,
          true,
        );
        return;
      }
      if (message.feature !== 'visualization' || message.type !== 'robot_tcp') return;
      if (
        message.protocol_version !== coreProtocolVersion
        || message.feature_protocol_version !== 2
      ) {
        latestRobotMessage = null;
        setStatus('robot feedback protocol mismatch', false);
        return;
      }
      latestRobotMessage = message;
      const freshCount = Object.values(message.robots || {})
        .filter((item) => item && item.fresh === true)
        .length;
      const boundCount = Object.values(message.robots || {})
        .filter((item) => item && item.alignment_ready === true)
        .length;
      setStatus(
        `robot feedback ${freshCount} fresh; ${boundCount} Quest-bound`,
        boundCount > 0,
      );
    },

    renderView(context) {
      if (!renderer) return;
      if (config.show_reference_marker !== false && referenceMarkerMatrix) {
        const markerMvp = global.QuestDevXRMath.mat4Multiply(
          context.viewProjection,
          referenceMarkerMatrix,
        );
        global.QuestDevVisualizationRenderer.drawReferenceMarker(
          context.gl,
          renderer,
          markerMvp,
        );
      }
      if (config.show_controller_axes !== false) {
        for (const controller of context.controllerPoses) {
          const mvp = global.QuestDevXRMath.mat4Multiply(
            context.viewProjection,
            controller.pose.transform.matrix,
          );
          global.QuestDevVisualizationRenderer.drawAxes(context.gl, renderer, mvp);
        }
      }
      const preview = latestCalibrationPreview;
      const previewEpochMatches = preview && questEpochMatches(
        context,
        preview.quest_session_id,
        preview.reference_space_type,
        preview.reference_space_generation,
      );
      const previewRecord = (
        previewEpochMatches
        && latestRobotMessage
        && latestRobotMessage.robots
        && latestRobotMessage.robots[preview.robot]
      );
      if (previewEpochMatches && preview) {
        const baseMatrix = new Float32Array(preview.matrix_quest_from_robot_base);
        const baseMvp = global.QuestDevXRMath.mat4Multiply(
          context.viewProjection,
          baseMatrix,
        );
        if (
          config.show_robot_base_axes !== false
          && typeof global.QuestDevVisualizationRenderer.drawRobotBaseAxes === 'function'
        ) {
          global.QuestDevVisualizationRenderer.drawRobotBaseAxes(
            context.gl,
            renderer,
            baseMvp,
          );
        }
        if (
          previewRecord
          && previewRecord.fresh === true
          && hasFiniteMatrix(previewRecord.matrix_robot_from_tcp)
        ) {
          const tcpMatrix = global.QuestDevXRMath.mat4Multiply(
            baseMatrix,
            new Float32Array(previewRecord.matrix_robot_from_tcp),
          );
          const tcpMvp = global.QuestDevXRMath.mat4Multiply(
            context.viewProjection,
            tcpMatrix,
          );
          if (config.show_robot_tcp_axes !== false) {
            global.QuestDevVisualizationRenderer.drawAxes(
              context.gl,
              renderer,
              tcpMvp,
            );
          }
          if (
            config.show_robot_tcp_marker !== false
            && typeof global.QuestDevVisualizationRenderer.drawTcpMarker === 'function'
          ) {
            global.QuestDevVisualizationRenderer.drawTcpMarker(
              context.gl,
              renderer,
              tcpMvp,
            );
          }
        }
      }
      if (config.show_robot_tcp_axes !== false && latestRobotMessage) {
        for (const record of Object.values(latestRobotMessage.robots || {})) {
          if (!renderableRobotRecord(record, context)) continue;
          const tcpMatrix = new Float32Array(record.matrix_quest_from_tcp);
          const mvp = global.QuestDevXRMath.mat4Multiply(
            context.viewProjection,
            tcpMatrix,
          );
          global.QuestDevVisualizationRenderer.drawAxes(
            context.gl,
            renderer,
            mvp,
          );
        }
      }
    },

    onReferenceSpaceReset() {
      latestRobotMessage = null;
      latestCalibrationPreview = null;
      referenceMarkerMatrix = null;
      referenceGeneration = null;
      setStatus('alignment invalidated by reference-space reset', false);
    },

    onSessionEnd() {
      if (renderer) global.QuestDevVisualizationRenderer.destroy(renderer);
      renderer = null;
      latestRobotMessage = null;
      latestCalibrationPreview = null;
      referenceMarkerMatrix = null;
      referenceGeneration = null;
      setStatus(enabled ? 'enabled; XR ended' : 'disabled', false);
    },
  });
}(window));
