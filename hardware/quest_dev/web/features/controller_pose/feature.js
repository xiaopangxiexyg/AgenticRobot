(function registerControllerPoseFeature(global) {
  'use strict';

  const statusRoot = document.getElementById('feature-status-grid');
  const statusCard = document.createElement('div');
  statusCard.className = 'status-item';
  statusCard.innerHTML = '<span class="label">Controller pose</span><span class="bad">disabled</span>';
  statusRoot.appendChild(statusCard);
  const status = statusCard.querySelector('span:last-child');

  let enabled = false;
  let publishPeriodMs = 50;
  let lastPublishedAtMs = -Infinity;
  let sampleSequence = 0;

  function setStatus(text, ok) {
    status.textContent = text;
    status.className = ok ? 'ok' : 'bad';
  }

  global.QuestDevFeatures.register('controller_pose', {
    configure(config, context) {
      enabled = context.enabled;
      const rateHz = Math.max(1, Math.min(30, Number(config.publish_rate_hz || 20)));
      publishPeriodMs = 1000 / rateHz;
      setStatus(enabled ? `ready; ${rateHz} Hz max` : 'disabled', enabled);
    },

    onXRFrame(context) {
      if (!enabled || !context.xrSessionId) return;
      if (context.time - lastPublishedAtMs < publishPeriodMs) return;
      lastPublishedAtMs = context.time;
      sampleSequence += 1;
      context.send({
        type: 'controller_pose_sample',
        feature: 'controller_pose',
        protocol_version: context.protocolVersion,
        feature_protocol_version: 1,
        client_instance_id: context.clientInstanceId,
        xr_session_id: context.xrSessionId,
        reference_space: {
          type: context.referenceSpaceType,
          generation: context.referenceSpaceGeneration,
        },
        sample_sequence: sampleSequence,
        xr_frame_time_ms: context.time,
        controllers: context.controllerPoses.map((controller) => ({
          source_session_id: controller.sourceSessionId,
          handedness: controller.handedness,
          target_ray_mode: controller.targetRayMode,
          profiles: controller.profiles,
          emulated_position: controller.emulatedPosition,
          matrix_reference_from_grip: controller.matrixReferenceFromGrip,
        })),
      });
      setStatus(
        `sequence ${sampleSequence}; ${context.controllerPoses.length} controller(s)`,
        true,
      );
    },

    onReferenceSpaceReset() {
      lastPublishedAtMs = -Infinity;
      setStatus('old pose invalidated by reference-space reset', false);
    },

    onSessionEnd() {
      lastPublishedAtMs = -Infinity;
      sampleSequence = 0;
      setStatus(enabled ? 'ready; XR ended' : 'disabled', false);
    },
  });
}(window));
