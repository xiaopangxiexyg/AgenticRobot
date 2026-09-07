(function exposeQuestDevControllerPoseDisplay(global) {
  'use strict';

  function displayPose(sample, trackingState, displayOnly) {
    return {
      ...sample,
      displayOnly,
      trackingState,
    };
  }

  function createTracker() {
    const lastTrackedBySource = new Map();

    return {
      reset() {
        lastTrackedBySource.clear();
      },

      observe(sourceSessionId, sample) {
        const sourceId = String(sourceSessionId || '').trim();
        if (!sourceId) throw new TypeError('controller source session ID is required');

        if (sample && sample.emulatedPosition === false) {
          lastTrackedBySource.set(sourceId, sample);
          return displayPose(sample, 'tracked', false);
        }

        const lastTracked = lastTrackedBySource.get(sourceId);
        if (lastTracked) {
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
          );
        }

        // There is no safe measured transform yet.  An available emulated pose
        // can still be rendered as a diagnostic, but it is always marked as
        // display-only and is never sent through the capture feature.
        return sample ? displayPose(sample, 'emulated', true) : null;
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
