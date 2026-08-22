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

  function setStatus(text, ok) {
    status.textContent = text;
    status.className = ok ? 'ok' : 'bad';
  }

  global.QuestDevFeatures.register('visualization', {
    configure(featureConfig, context) {
      enabled = context.enabled;
      config = featureConfig;
      setStatus(enabled ? 'enabled; XR not started' : 'disabled', enabled);
    },

    onXRReady(context) {
      if (!enabled) return;
      renderer = global.QuestDevVisualizationRenderer.create(context.gl, {
        axisLength: Number(config.axis_length_m || 0.18),
      });
      setStatus('XR renderer ready', true);
    },

    handleServerMessage(context) {
      const message = context.message;
      if (message.feature !== 'visualization' && message.type !== 'robot_tcp') return;
      if (message.type === 'robot_tcp') {
        latestRobotMessage = message;
        const freshCount = Object.values(message.robots || {}).filter((item) => item.fresh).length;
        setStatus(`robot feedback ${freshCount} fresh; alignment pending`, freshCount > 0);
      }
    },

    renderView(context) {
      if (!renderer || config.show_controller_axes === false) return;
      for (const pose of context.controllerPoses) {
        const mvp = global.QuestDevXRMath.mat4Multiply(context.viewProjection, pose.transform.matrix);
        global.QuestDevVisualizationRenderer.drawAxes(context.gl, renderer, mvp);
      }
      // Robot TCP rendering intentionally waits for a validated
      // T_quest_from_robot alignment owned by this feature.
      void latestRobotMessage;
    },

    onReferenceSpaceReset() {
      latestRobotMessage = null;
      setStatus('alignment invalidated by reference-space reset', false);
    },

    onSessionEnd() {
      renderer = null;
      setStatus(enabled ? 'enabled; XR ended' : 'disabled', false);
    },
  });
}(window));
