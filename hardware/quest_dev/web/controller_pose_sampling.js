(function exposeQuestDevControllerPoseSampling(global) {
  'use strict';

  function sourceMetadata(source, sourceSessionId) {
    return {
      sourceSessionId: String(sourceSessionId || ''),
      handedness: String((source && source.handedness) || 'none'),
      targetRayMode: String((source && source.targetRayMode) || 'tracked-pointer'),
    };
  }

  function diagnostic(source, sourceSessionId, trackingState, reason, error) {
    const result = {
      ...sourceMetadata(source, sourceSessionId),
      trackingState,
      reason,
    };
    if (error && typeof error.name === 'string' && error.name) {
      result.errorName = error.name.slice(0, 80);
    }
    return result;
  }

  function unavailable(source, sourceSessionId, reason, error) {
    return {
      sample: null,
      diagnostic: diagnostic(
        source,
        sourceSessionId,
        'untracked',
        reason,
        error,
      ),
    };
  }

  function sampleControllerPose(frame, referenceSpace, source, sourceSessionId) {
    if (!source || !source.gripSpace) {
      return unavailable(source, sourceSessionId, 'missing_grip_space');
    }

    let pose;
    try {
      pose = frame.getPose(source.gripSpace, referenceSpace);
    } catch (error) {
      return unavailable(source, sourceSessionId, 'pose_error', error);
    }
    if (!pose) {
      // WebXR explicitly returns null when the grip cannot be located in the
      // requested reference space. This is tracking loss, not a terminal
      // capture fault, and must be observable instead of silently disappearing.
      return unavailable(source, sourceSessionId, 'pose_unavailable');
    }
    const transform = pose.transform;
    const matrix = transform && transform.matrix;
    if (!matrix || typeof matrix.length !== 'number' || matrix.length !== 16) {
      return unavailable(source, sourceSessionId, 'pose_transform_unavailable');
    }
    const matrixReferenceFromGrip = Array.from(matrix);
    if (!matrixReferenceFromGrip.every((value) => Number.isFinite(Number(value)))) {
      return unavailable(source, sourceSessionId, 'pose_matrix_invalid');
    }

    const emulatedPosition = pose.emulatedPosition === true;
    return {
      sample: {
        pose,
        sourceSessionId: String(sourceSessionId || ''),
        handedness: source.handedness || 'none',
        targetRayMode: source.targetRayMode || 'tracked-pointer',
        profiles: Array.from(source.profiles || []),
        emulatedPosition,
        matrixReferenceFromGrip,
      },
      diagnostic: diagnostic(
        source,
        sourceSessionId,
        emulatedPosition ? 'emulated' : 'tracked',
        emulatedPosition ? 'emulated_position' : 'tracked',
      ),
    };
  }

  global.QuestDevControllerPoseSampling = Object.freeze({
    sampleControllerPose,
  });
}(window));
