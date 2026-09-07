'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function loadTracker() {
  const context = { window: {} };
  const filename = path.join(
    __dirname,
    '..',
    'web',
    'controller_pose_display.js',
  );
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context, { filename });
  return context.window.QuestDevControllerPoseDisplay.createTracker();
}

function sample({ x = 1, emulatedPosition = false } = {}) {
  return {
    sourceSessionId: 'input-1',
    handedness: 'left',
    targetRayMode: 'tracked-pointer',
    profiles: ['oculus-touch-controller'],
    emulatedPosition,
    pose: { transform: { matrix: [x] } },
  };
}

test('display tracker keeps the last measured pose separate from raw loss', () => {
  const tracker = loadTracker();
  const tracked = tracker.observe('input-1', sample({ x: 1 }));
  assert.equal(tracked.displayOnly, false);
  assert.equal(tracked.trackingState, 'tracked');

  const missing = tracker.observe('input-1', null);
  assert.equal(missing.displayOnly, true);
  assert.equal(missing.trackingState, 'untracked');
  assert.strictEqual(missing.pose, tracked.pose);

  const emulated = tracker.observe(
    'input-1',
    sample({ x: 9, emulatedPosition: true }),
  );
  assert.equal(emulated.displayOnly, true);
  assert.equal(emulated.trackingState, 'emulated');
  assert.strictEqual(emulated.pose, tracked.pose);
});

test('display tracker does not invent a pose before first real tracking', () => {
  const tracker = loadTracker();
  assert.equal(tracker.observe('input-1', null), null);
  const emulated = tracker.observe(
    'input-1',
    sample({ emulatedPosition: true }),
  );
  assert.equal(emulated.displayOnly, true);
  assert.equal(emulated.trackingState, 'emulated');
  assert.equal(tracker.observe('input-2', null), null);
});

test('display tracker drops cached poses when an input source leaves XR', () => {
  const tracker = loadTracker();
  tracker.observe('input-1', sample());
  tracker.prune(new Set(['input-2']));
  assert.equal(tracker.observe('input-1', null), null);
  tracker.reset();
  assert.equal(tracker.observe('input-2', null), null);
});
