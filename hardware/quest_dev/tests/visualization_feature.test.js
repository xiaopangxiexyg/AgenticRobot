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
          assert.equal(name, 'visualization');
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
    'visualization',
    'feature.js',
  );
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context, { filename });
  implementation.configure({}, { enabled: true, protocolVersion: 2 });
  return { implementation, status };
}

test('visualization accepts only its versioned robot_tcp message', () => {
  const { implementation, status } = loadFeature();
  const readyText = status.textContent;

  implementation.handleServerMessage({
    message: {
      type: 'robot_tcp',
      feature: 'other_feature',
      protocol_version: 2,
      feature_protocol_version: 1,
      robots: {},
    },
  });
  assert.equal(status.textContent, readyText);

  implementation.handleServerMessage({
    message: {
      type: 'robot_tcp',
      feature: 'visualization',
      protocol_version: 2,
      feature_protocol_version: 99,
      robots: {},
    },
  });
  assert.equal(status.textContent, 'robot feedback protocol mismatch');

  implementation.handleServerMessage({
    message: {
      type: 'robot_tcp',
      feature: 'visualization',
      protocol_version: 2,
      feature_protocol_version: 1,
      robots: {},
    },
  });
  assert.equal(status.textContent, 'robot feedback 0 fresh; alignment pending');
});
