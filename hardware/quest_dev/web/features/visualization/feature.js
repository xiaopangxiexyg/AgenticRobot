(function registerVisualizationFeature(global) {
  'use strict';

  const statusRoot = document.getElementById('feature-status-grid');
  const statusCard = document.createElement('div');
  statusCard.className = 'status-item';
  statusCard.innerHTML = '<span class="label">Visualization</span><span class="bad">disabled</span>';
  statusRoot.appendChild(statusCard);
  const status = statusCard.querySelector('span:last-child');

  let enabled = false;
  let config = {};
  let renderer = null;
  let latestRobotMessage = null;
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
      referenceMarkerMatrix = null;
      referenceGeneration = null;
      setStatus('alignment invalidated by reference-space reset', false);
    },

    onSessionEnd() {
      if (renderer) global.QuestDevVisualizationRenderer.destroy(renderer);
      renderer = null;
      latestRobotMessage = null;
      referenceMarkerMatrix = null;
      referenceGeneration = null;
      setStatus(enabled ? 'enabled; XR ended' : 'disabled', false);
    },
  });
}(window));
