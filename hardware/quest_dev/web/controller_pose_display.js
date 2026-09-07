(function exposeQuestDevControllerPoseDisplay(global) {
  'use strict';

  function displayPose(sample, trackingState, displayOnly, trackingReason) {
    return {
      ...sample,
      displayOnly,
      trackingState,
      trackingReason,
    };
  }

  function createTracker(options = {}) {
    const maximumHoldMs = Number(options.maximumHoldMs === undefined
      ? 250
      : options.maximumHoldMs);
    if (!Number.isFinite(maximumHoldMs) || maximumHoldMs < 0) {
      throw new TypeError('maximumHoldMs must be finite and non-negative');
    }
    const lastTrackedBySource = new Map();

    return {
      reset() {
        lastTrackedBySource.clear();
      },

      observe(sourceSessionId, sample, diagnostic, frameTimeMs = 0) {
        const sourceId = String(sourceSessionId || '').trim();
        if (!sourceId) throw new TypeError('controller source session ID is required');
        const nowMs = Number(frameTimeMs);
        if (!Number.isFinite(nowMs) || nowMs < 0) {
          throw new TypeError('frameTimeMs must be finite and non-negative');
        }

        if (sample && sample.emulatedPosition === false) {
          lastTrackedBySource.set(sourceId, { sample, trackedAtMs: nowMs });
          return displayPose(sample, 'tracked', false, 'tracked');
        }

        const cached = lastTrackedBySource.get(sourceId);
        if (cached && nowMs - cached.trackedAtMs <= maximumHoldMs) {
          const lastTracked = cached.sample;
          // Keep the last measured transform visible while WebXR temporarily
          // reports an emulated pose or no grip pose.  This object is supplied
          // only to the renderer; the raw sample remains separate for capture.
          const metadata = sample || lastTracked;
          return displayPose(
            {
              ...lastTracked,
              sourceSessionId: sourceId,
              handedness: metadata.handedness,
              targetRayMode: metadata.targetRayMode,
              profiles: metadata.profiles,
            },
            sample ? 'emulated' : 'untracked',
            true,
            diagnostic && diagnostic.reason
              ? diagnostic.reason
              : sample
                ? 'emulated_position'
                : 'pose_unavailable',
          );
        }

        // A system-rendered controller model may remain visible even when the
        // WebXR grip pose is unavailable. Do not let this project's axes keep
        // presenting an old transform as if it were current tracking.
        if (cached) lastTrackedBySource.delete(sourceId);

        // There is no safe measured transform yet.  An available emulated pose
        // can still be rendered as a diagnostic, but it is always marked as
        // display-only and is never sent through the capture feature.
        return sample
          ? displayPose(sample, 'emulated', true, 'emulated_position')
          : null;
      },

      prune(activeSourceIds) {
        if (!activeSourceIds || typeof activeSourceIds.has !== 'function') {
          throw new TypeError('activeSourceIds must be a Set-like value');
        }
        for (const sourceId of lastTrackedBySource.keys()) {
          if (!activeSourceIds.has(sourceId)) lastTrackedBySource.delete(sourceId);
        }
      },
    };
  }

  global.QuestDevControllerPoseDisplay = Object.freeze({ createTracker });
}(window));
