'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function loadIdentity() {
  const context = { window: {} };
  const filename = path.join(
    __dirname,
    '..',
    'web',
    'controller_source_identity.js',
  );
  vm.runInNewContext(fs.readFileSync(filename, 'utf8'), context, { filename });
  return context.window.QuestDevControllerSourceIdentity;
}

function source(overrides = {}) {
  return {
    handedness: 'left',
    targetRayMode: 'tracked-pointer',
    profiles: ['oculus-touch-controller', 'generic-trigger'],
    ...overrides,
  };
}

test('controller identity survives XRInputSource wrapper replacement', () => {
  const tracker = loadIdentity().createTracker();
  const first = tracker.idFor(source(), 0);
  const replacement = tracker.idFor(source(), 0);
  assert.equal(first, 'input-1');
  assert.equal(replacement, first);

  const reorderedProfiles = tracker.idFor(
    source({ profiles: ['generic-trigger', 'oculus-touch-controller'] }),
    0,
  );
  assert.equal(reorderedProfiles, first);

  const right = tracker.idFor(source({ handedness: 'right' }), 0);
  const secondLeft = tracker.idFor(source(), 1);
  assert.equal(right, 'input-2');
  assert.equal(secondLeft, 'input-3');
  assert.notEqual(secondLeft, first);

  tracker.reset();
  assert.equal(tracker.idFor(source(), 0), 'input-1');
});
