(function exposeQuestDevControllerSourceIdentity(global) {
  'use strict';

  function sourceFingerprint(source) {
    if (!source || typeof source !== 'object') {
      throw new TypeError('controller source must be an object');
    }
    const handedness = String(source.handedness || 'none');
    const targetRayMode = String(source.targetRayMode || 'tracked-pointer');
    // WebXR profiles are diagnostic capability metadata, not a stable source
    // identity. Quest Browser may replace the generic profile with a concrete
    // controller profile after tracking starts. Including profiles here would
    // mint a new input-N ID for the same physical controller and split one
    // selected source into multiple passive-capture candidates.
    return [handedness, targetRayMode].join('\u001e');
  }

  function createTracker() {
    const idsByKey = new Map();
    let nextId = 1;
    return {
      reset() {
        idsByKey.clear();
        nextId = 1;
      },

      fingerprint: sourceFingerprint,

      idFor(source, occurrence = 0) {
        if (!Number.isInteger(occurrence) || occurrence < 0) {
          throw new TypeError('controller source occurrence must be non-negative');
        }
        const key = `${sourceFingerprint(source)}\u001e${occurrence}`;
        let value = idsByKey.get(key);
        if (!value) {
          value = `input-${nextId}`;
          nextId += 1;
          idsByKey.set(key, value);
        }
        return value;
      },
    };
  }

  global.QuestDevControllerSourceIdentity = Object.freeze({
    createTracker,
    sourceFingerprint,
  });
}(window));
