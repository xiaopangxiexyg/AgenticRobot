'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function loadFeature() {
  let implementation = null;
  const status = { className: '', textContent: '' };
  const statusCard = {
    className: '',
    innerHTML: '',
    querySelector: () => status,
  };
  const context = {
    document: {
      createElement: () => statusCard,
      getElementById: () => ({ appendChild: () => {} }),
    },
    window: {
      QuestDevFeatures: {
        register: (name, feature) => {
          assert.equal(name, 'controller_pose');
          implementation = feature;
        },
      },
    },
  };
  const filename = path.join(
    __dirname,
    '..',
    'web',
    'features',
    'controller_pose',
    'feature.js',
  );
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context, { filename });
  implementation.configure({ publish_rate_hz: 20 }, {
    enabled: true,
    protocolVersion: 2,
  });
  return { implementation, status };
}

function frameContext(overrides = {}) {
  const sent = [];
  const logs = [];
  return {
    time: 0,
    xrSessionId: 'xr-session-1',
    protocolVersion: 2,
    clientInstanceId: 'browser-1',
    referenceSpaceType: 'local-floor',
    referenceSpaceGeneration: 0,
    controllerPoses: [],
    controllerDiagnostics: [],
    viewerPoseAvailable: true,
    send: (payload) => sent.push(payload),
    logEvent: (message) => logs.push(message),
    sent,
    logs,
    ...overrides,
  };
}

test('pose_unavailable is visible while raw capture receives an empty controller list', () => {
  const { implementation, status } = loadFeature();
  const context = frameContext({
    controllerDiagnostics: [{
      sourceSessionId: 'input-1',
      handedness: 'left',
      targetRayMode: 'tracked-pointer',
      trackingState: 'untracked',
      reason: 'pose_unavailable',
    }],
  });
  implementation.onXRFrame(context);

  assert.equal(context.sent.length, 1);
  assert.deepEqual(context.sent[0].controllers, []);
  assert.equal(context.sent[0].controller_diagnostics.length, 1);
  assert.equal(
    context.sent[0].controller_diagnostics[0].source_session_id,
    'input-1',
  );
  assert.equal(
    context.sent[0].controller_diagnostics[0].tracking_state,
    'untracked',
  );
  assert.equal(
    context.sent[0].controller_diagnostics[0].reason,
    'pose_unavailable',
  );
  assert.match(status.textContent, /pose_unavailable=1/);
  assert.equal(status.className, 'bad');
  assert.equal(context.logs.length, 1);
  assert.match(context.logs[0], /pose_unavailable=1/);
});

test('viewer pose loss is diagnosed independently and does not remove a real grip sample', () => {
  const { implementation, status } = loadFeature();
  const context = frameContext({
    controllerPoses: [{
      sourceSessionId: 'input-1',
      handedness: 'left',
      targetRayMode: 'tracked-pointer',
      profiles: [],
      emulatedPosition: false,
      matrixReferenceFromGrip: Array(16).fill(0),
    }],
    controllerDiagnostics: [{
      sourceSessionId: 'input-1',
      handedness: 'left',
      targetRayMode: 'tracked-pointer',
      trackingState: 'tracked',
      reason: 'tracked',
    }],
    viewerPoseAvailable: false,
    viewerPoseDiagnostic: { reason: 'viewer_pose_unavailable' },
  });
  implementation.onXRFrame(context);

  assert.equal(context.sent.length, 1);
  assert.equal(context.sent[0].controllers.length, 1);
  assert.equal(context.sent[0].controller_diagnostics[0].tracking_state, 'tracked');
  assert.match(status.textContent, /viewer_pose_unavailable=1/);
  assert.equal(status.className, 'bad');
});
