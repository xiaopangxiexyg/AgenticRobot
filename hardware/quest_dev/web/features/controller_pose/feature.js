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
  let lastDiagnosticSignature = null;

  function diagnosticSummary(context) {
    const diagnostics = Array.isArray(context.controllerDiagnostics)
      ? context.controllerDiagnostics
      : [];
    const counts = {
      tracked: 0,
      emulated: 0,
      untracked: 0,
    };
    const reasons = new Map();
    for (const item of diagnostics) {
      const state = item && item.trackingState;
      if (Object.prototype.hasOwnProperty.call(counts, state)) counts[state] += 1;
      const reason = item && item.reason;
      if (reason && reason !== 'tracked') {
        const label = item && item.errorName
          ? `${reason}[${item.errorName}]`
          : reason;
        reasons.set(label, (reasons.get(label) || 0) + 1);
      }
    }
    if (context.viewerPoseAvailable === false) {
      const viewerReason = context.viewerPoseDiagnostic
        && context.viewerPoseDiagnostic.reason;
      const viewerLabel = context.viewerPoseDiagnostic
        && context.viewerPoseDiagnostic.errorName
        ? `${viewerReason || 'viewer_pose_unavailable'}[${context.viewerPoseDiagnostic.errorName}]`
        : viewerReason || 'viewer_pose_unavailable';
      reasons.set(
        viewerLabel,
        1,
      );
    }
    const reasonText = Array.from(reasons.entries())
      .map(([reason, count]) => `${reason}=${count}`)
      .join(',');
    const sourceText = diagnostics.length === 0
      ? 'no-input-sources'
      : reasonText || 'ok';
    return {
      counts,
      reasonText,
      sourceText,
      signature: JSON.stringify({
        diagnostics: diagnostics.map((item) => ({
          sourceSessionId: item && item.sourceSessionId,
          trackingState: item && item.trackingState,
          reason: item && item.reason,
          errorName: item && item.errorName,
        })),
        viewerPoseAvailable: context.viewerPoseAvailable,
        viewerPoseReason: context.viewerPoseDiagnostic
          && context.viewerPoseDiagnostic.reason,
        viewerPoseErrorName: context.viewerPoseDiagnostic
          && context.viewerPoseDiagnostic.errorName,
      }),
    };
  }

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
      const diagnostics = diagnosticSummary(context);
      const trackingText = (
        `tracked=${diagnostics.counts.tracked}; `
        + `emulated=${diagnostics.counts.emulated}; `
        + `untracked=${diagnostics.counts.untracked}`
      );
      const statusText = diagnostics.reasonText
        ? `${trackingText}; ${diagnostics.reasonText}`
        : `${trackingText}; ${diagnostics.sourceText}`;
      if (
        diagnostics.signature !== lastDiagnosticSignature
        && typeof context.logEvent === 'function'
      ) {
        context.logEvent(`[Quest] controller tracking: ${statusText}`);
        lastDiagnosticSignature = diagnostics.signature;
      }
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
        `sequence ${sampleSequence}; ${context.controllerPoses.length} pose(s); ${statusText}`,
        diagnostics.counts.tracked > 0 && context.viewerPoseAvailable !== false,
      );
    },

    onReferenceSpaceReset() {
      lastPublishedAtMs = -Infinity;
      lastDiagnosticSignature = null;
      setStatus('old pose invalidated by reference-space reset', false);
    },

    onSessionEnd() {
      lastPublishedAtMs = -Infinity;
      sampleSequence = 0;
      lastDiagnosticSignature = null;
      setStatus(enabled ? 'ready; XR ended' : 'disabled', false);
    },
  });
}(window));
