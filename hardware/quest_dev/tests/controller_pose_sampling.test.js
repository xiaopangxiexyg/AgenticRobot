'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function loadSampling() {
  const context = { window: {} };
  const filename = path.join(
    __dirname,
    '..',
    'web',
    'controller_pose_sampling.js',
  );
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context, { filename });
  return context.window.QuestDevControllerPoseSampling;
}

function source(overrides = {}) {
  return {
    gripSpace: {},
    handedness: 'left',
    targetRayMode: 'tracked-pointer',
    profiles: ['oculus-touch-controller'],
    ...overrides,
  };
}

function pose(overrides = {}) {
  return {
    emulatedPosition: false,
    transform: { matrix: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0.1, 0.2, 0.3, 1,
    ] },
    ...overrides,
  };
}

test('getPose null is reported as pose_unavailable and never becomes a raw sample', () => {
  const sampling = loadSampling();
  const result = sampling.sampleControllerPose(
    { getPose: () => null },
    {},
    source(),
    'input-1',
  );
  assert.equal(result.sample, null);
  assert.equal(result.diagnostic.sourceSessionId, 'input-1');
  assert.equal(result.diagnostic.handedness, 'left');
  assert.equal(result.diagnostic.targetRayMode, 'tracked-pointer');
  assert.equal(result.diagnostic.trackingState, 'untracked');
  assert.equal(result.diagnostic.reason, 'pose_unavailable');
});

test('missing gripSpace and getPose exceptions remain distinct diagnostics', () => {
  const sampling = loadSampling();
  const missingGrip = sampling.sampleControllerPose(
    { getPose: () => pose() },
    {},
    source({ gripSpace: null }),
    'input-1',
  );
  assert.equal(missingGrip.diagnostic.reason, 'missing_grip_space');

  const error = new Error('reference space unavailable');
  error.name = 'InvalidStateError';
  const thrown = sampling.sampleControllerPose(
    { getPose: () => { throw error; } },
    {},
    source(),
    'input-1',
  );
  assert.equal(thrown.diagnostic.reason, 'pose_error');
  assert.equal(thrown.diagnostic.errorName, 'InvalidStateError');
});

test('real and emulated poses are returned with separate tracking states', () => {
  const sampling = loadSampling();
  const tracked = sampling.sampleControllerPose(
    { getPose: () => pose() },
    {},
    source(),
    'input-1',
  );
  assert.ok(tracked.sample);
  assert.equal(tracked.sample.emulatedPosition, false);
  assert.equal(tracked.diagnostic.trackingState, 'tracked');
  assert.equal(tracked.diagnostic.reason, 'tracked');

  const emulated = sampling.sampleControllerPose(
    { getPose: () => pose({ emulatedPosition: true }) },
    {},
    source(),
    'input-1',
  );
  assert.ok(emulated.sample);
  assert.equal(emulated.sample.emulatedPosition, true);
  assert.equal(emulated.diagnostic.trackingState, 'emulated');
  assert.equal(emulated.diagnostic.reason, 'emulated_position');
});
