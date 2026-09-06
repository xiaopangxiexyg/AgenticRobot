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
  const draws = [];
  const renderer = {
    create: () => ({}),
    destroy: () => {},
    drawAxes: (...args) => draws.push(args),
    drawReferenceMarker: () => {},
    drawRobotBaseAxes: (...args) => draws.push(args),
    drawTcpMarker: (...args) => draws.push(args),
  };
  const xrMath = {
    mat4Multiply: (_a, b) => b,
    mat4Translation: () => new Float32Array(16),
  };
  const context = {
    document: {
      createElement: () => statusCard,
      getElementById: () => ({ appendChild: () => {} }),
    },
    window: {
      QuestDevFeatures: {
        register: (name, feature) => {
          assert.equal(name, 'visualization');
          implementation = feature;
        },
      },
      QuestDevVisualizationRenderer: renderer,
      QuestDevXRMath: xrMath,
    },
  };
  const filename = path.join(
    __dirname,
    '..',
    'web',
    'features',
    'visualization',
    'feature.js',
  );
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context, { filename });
  implementation.configure({}, { enabled: true, protocolVersion: 2 });
  return { implementation, status, draws };
}

function identityMatrix() {
  return Array.from({ length: 16 }, (_, index) => (index % 5 === 0 ? 1 : 0));
}

function matchingContext() {
  return {
    gl: {},
    viewProjection: identityMatrix(),
    controllerPoses: [],
    xrSessionId: 'xr-session-1',
    referenceSpaceType: 'local-floor',
    referenceSpaceGeneration: 0,
  };
}

function robotMessage(overrides = {}) {
  return {
    type: 'robot_tcp',
    feature: 'visualization',
    protocol_version: 2,
    feature_protocol_version: 2,
    robots: {
      ar5_1: {
        robot: 'ar5_1',
        fresh: true,
        alignment_ready: true,
        matrix_quest_from_tcp: identityMatrix(),
        frame_binding: {
          binding_id: 'binding-1',
          acceptance_receipt_sha256: 'a'.repeat(64),
          quest_session_id: 'xr-session-1',
          reference_space_type: 'local-floor',
          reference_space_generation: 0,
          quest_reference_epoch_id: 'xr-session-1:local-floor:0',
          reference_frame_id: 'quest-local-floor',
        },
      },
    },
    ...overrides,
  };
}

test('visualization accepts only its versioned robot_tcp message', () => {
  const { implementation, status } = loadFeature();
  const readyText = status.textContent;

  implementation.handleServerMessage({
    message: {
      type: 'robot_tcp',
      feature: 'other_feature',
      protocol_version: 2,
      feature_protocol_version: 2,
      robots: {},
    },
  });
  assert.equal(status.textContent, readyText);

  implementation.handleServerMessage({
    message: {
      type: 'robot_tcp',
      feature: 'visualization',
      protocol_version: 2,
      feature_protocol_version: 1,
      robots: {},
    },
  });
  assert.equal(status.textContent, 'robot feedback protocol mismatch');

  implementation.handleServerMessage({
    message: robotMessage({ robots: {} }),
  });
  assert.equal(status.textContent, 'robot feedback 0 fresh; 0 Quest-bound');
});

test('visualization draws TCP axes only for the current accepted Quest epoch', () => {
  const { implementation, draws } = loadFeature();
  implementation.onXRReady({ gl: {} });
  implementation.handleServerMessage({ message: robotMessage() });

  const context = matchingContext();
  implementation.renderView(context);
  assert.equal(draws.length, 1);

  implementation.handleServerMessage({
    message: robotMessage({
      robots: {
        ar5_1: {
          ...robotMessage().robots.ar5_1,
          frame_binding: {
            ...robotMessage().robots.ar5_1.frame_binding,
            reference_space_generation: 1,
            quest_reference_epoch_id: 'xr-session-1:local-floor:1',
          },
        },
      },
    }),
  });
  implementation.renderView(context);
  assert.equal(draws.length, 1);

  implementation.onReferenceSpaceReset();
  implementation.renderView(context);
  assert.equal(draws.length, 1);
});

test('visualization draws diagnostic robot base and raw-feedback TCP preview', () => {
  const { implementation, draws, status } = loadFeature();
  implementation.onXRReady({ gl: {} });
  implementation.handleServerMessage({
    message: {
      type: 'calibration_preview',
      feature: 'visualization',
      protocol_version: 2,
      feature_protocol_version: 2,
      preview: true,
      accepted: false,
      alignment_status: 'diagnostic_preview',
      robot: 'rm65_left',
      robot_frame: 'left_arm_base',
      tcp_frame: 'tool_tcp',
      quest_session_id: 'xr-session-1',
      reference_space_type: 'local-floor',
      reference_space_generation: 0,
      matrix_quest_from_robot_base: identityMatrix(),
    },
  });
  implementation.handleServerMessage({
    message: {
      type: 'robot_tcp',
      feature: 'visualization',
      protocol_version: 2,
      feature_protocol_version: 2,
      robots: {
        rm65_left: {
          robot: 'rm65_left',
          fresh: true,
          alignment_ready: false,
          matrix_robot_from_tcp: identityMatrix(),
        },
      },
    },
  });
  implementation.renderView(matchingContext());
  assert.equal(draws.length, 3);
  assert.match(status.textContent, /robot feedback 1 fresh/);
});
